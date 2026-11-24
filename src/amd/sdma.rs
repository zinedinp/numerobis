//! Copies on the SDMA engine, two ways to reach it:
//!
//! - [`Route::Kfd`]: a user-mode SDMA queue of this process (KFD
//!   `CREATE_QUEUE` of type SDMA): a ring of packets, read and write
//!   pointers in memory, a doorbell. Nothing goes through the kernel per copy.
//! - [`Route::Drm`]: the kernel driver's own SDMA ring, through
//!   `DRM_AMDGPU_CS` on the render node (as Mesa submits): the packets go in
//!   an indirect buffer, the kernel schedules it. One syscall per
//!   submission, and the kernel arbitrates with every other user of the
//!   engine. This works because KFD turns the render node's VM into the
//!   process's compute VM, so the buffers' addresses hold for it too.
//!
//! Both build the same packets (SDMA 4 to 7, `navi10_sdma_pkt_open.h` and its
//! siblings) and finish every submission with an `ATOMIC` decrement of an
//! HSA signal (`amd_signal_t`): the host waits on it, and the compute queue's
//! barrier packets wait on it. Waiting the other way, a copy starts with a
//! `POLL_REGMEM` on a compute signal.

use std::sync::atomic::{Ordering, fence};
use std::time::{Duration, Instant};

use super::kfd::{Bo, Gpu, Kind, call};
use super::queue::{DOORBELL_WINDOW, destroy};
use super::signals::{SLOTS, Signals, view};
use super::sys;
use crate::runtime::error::{Error, Result};
use crate::runtime::os::{Fd, Map, RW};

/// SDMA packet encodings.
pub mod packet {
    pub const OP_NOP: u32 = 0;
    pub const OP_COPY: u32 = 1;
    pub const OP_POLL_REGMEM: u32 = 8;
    pub const OP_ATOMIC: u32 = 10;
    pub const OP_TIMESTAMP: u32 = 13;
    pub const SUBOP_COPY_LINEAR: u32 = 0;
    /// `TIMESTAMP` sub-op 2: the GPU's global timestamp counter.
    pub const SUBOP_TIMESTAMP_GET_GLOBAL: u32 = 2;
    /// `SDMA_ATOMIC_ADD64`: the TC opcode of a 64-bit add.
    pub const ATOMIC_ADD64: u32 = 47;
    /// `POLL_REGMEM` function 3: until `(*addr & mask) == value`.
    pub const FUNC_EQUAL: u32 = 3;
    /// The largest linear copy on SDMA 4 to 6: `count` is `bytes - 1` in
    /// 22 bits (amdgpu's `copy_max_bytes`).
    pub const MAX_COPY: u64 = 1 << 22;

    fn lo(v: u64) -> u32 {
        v as u32
    }
    fn hi(v: u64) -> u32 {
        (v >> 32) as u32
    }

    /// `COPY_LINEAR` of `bytes` (1 ..= [`MAX_COPY`]) from `src` to `dst`.
    #[must_use]
    pub fn copy(dst: u64, src: u64, bytes: u64) -> [u32; 7] {
        debug_assert!((1..=MAX_COPY).contains(&bytes));
        [
            OP_COPY | (SUBOP_COPY_LINEAR << 8),
            (bytes - 1) as u32,
            0, // no endian swap
            lo(src),
            hi(src),
            lo(dst),
            hi(dst),
        ]
    }

    /// `POLL_REGMEM` on memory: waits until the dword at `addr` is
    /// `value`, polling every 16 clocks, for ever.
    #[must_use]
    pub fn wait_equal(addr: u64, value: u32) -> [u32; 6] {
        [
            OP_POLL_REGMEM | (FUNC_EQUAL << 28) | (1 << 31),
            lo(addr),
            hi(addr),
            value,
            u32::MAX,             // mask
            0x10 | (0xfff << 16), // interval, retries: 0xfff is unlimited
        ]
    }

    /// `ATOMIC` add of `v` (two's complement) to the 64-bit value at `addr`.
    #[must_use]
    pub fn add64(addr: u64, v: i64) -> [u32; 8] {
        let v = v as u64;
        [
            OP_ATOMIC | (ATOMIC_ADD64 << 25),
            lo(addr),
            hi(addr),
            lo(v),
            hi(v),
            0,
            0,
            0,
        ]
    }

    /// `TIMESTAMP` (`GET_GLOBAL`): the 64-bit global timestamp to `addr`
    /// (8-byte aligned).
    #[must_use]
    pub fn timestamp(addr: u64) -> [u32; 3] {
        debug_assert!(addr.is_multiple_of(8));
        [
            OP_TIMESTAMP | (SUBOP_TIMESTAMP_GET_GLOBAL << 8),
            lo(addr),
            hi(addr),
        ]
    }

    /// One-dword `NOP`.
    pub const NOP: u32 = OP_NOP;
}

/// The SDMA generation of a `gfx_target_version`: 4 on gfx9 and CDNA, 5 on
/// RDNA 1 and 2, 6 on RDNA 3, 7 on RDNA 4. `None` before gfx9 (SDMA 2 and
/// 3 keep 32-bit ring pointers, which this queue doesn't do).
#[must_use]
pub fn version(gfx_target_version: u32) -> Option<u32> {
    Some(match gfx_target_version / 10000 {
        9 => 4,
        10 => 5,
        11 => 6,
        12 => 7,
        _ => return None,
    })
}

/// The largest linear copy on SDMA `version`. SDMA 7 gets a little less
/// than 4 MiB, a 32-byte multiple: its limit isn't documented, and smaller
/// copies cost nothing but packets.
#[must_use]
pub fn max_copy(version: u32) -> u64 {
    if version >= 7 {
        packet::MAX_COPY - 256
    } else {
        packet::MAX_COPY
    }
}

/// Timestamps a submission writes: before its copies, after them.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct Stamps {
    pub start: Option<u64>,
    pub end: Option<u64>,
}

/// Linear copies per submission; a larger copy takes several.
const COPIES_PER_SUBMIT: usize = 256;
/// Dwords one submission can take: a wait, two timestamps, the copies, the
/// signal, padding.
const SUBMIT_DWORDS: usize = 6 + 3 + 7 * COPIES_PER_SUBMIT + 3 + 8 + 8;
/// Indirect buffer bytes per signal slot ([`Route::Drm`]).
const IB_SLOT: u64 = (SUBMIT_DWORDS as u64 * 4).next_multiple_of(4096);
/// Bytes of the user-mode ring ([`Route::Kfd`]), a power of two.
const RING_BYTES: u64 = 64 << 10;

/// How the SDMA engine is reached.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Route {
    /// A user-mode SDMA queue of this process (KFD).
    Kfd,
    /// The kernel driver's SDMA ring (`DRM_AMDGPU_CS`).
    Drm,
}

/// One copy: `bytes` from GPU address `src` to `dst`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Copy {
    pub dst: u64,
    pub src: u64,
    pub bytes: u64,
}

/// The packets of one submission: an optional wait on a compute signal's
/// value address, the start timestamp, the copies split to `max` bytes, the
/// end timestamp, the decrement of `signal_value`.
#[must_use]
pub fn build(
    wait: Option<u64>,
    copies: &[Copy],
    max: u64,
    stamps: Stamps,
    signal_value: u64,
) -> Vec<u32> {
    let mut d = Vec::with_capacity(SUBMIT_DWORDS);
    if let Some(addr) = wait {
        d.extend(packet::wait_equal(addr, 0));
    }
    if let Some(at) = stamps.start {
        d.extend(packet::timestamp(at));
    }
    for c in copies {
        let mut done = 0;
        while done < c.bytes {
            let n = (c.bytes - done).min(max);
            d.extend(packet::copy(c.dst + done, c.src + done, n));
            done += n;
        }
    }
    if let Some(at) = stamps.end {
        d.extend(packet::timestamp(at));
    }
    d.extend(packet::add64(signal_value, -1));
    d
}

/// Splits `copies` so that no submission holds more than
/// [`COPIES_PER_SUBMIT`] copy packets of `max` bytes.
fn groups(copies: &[Copy], max: u64) -> Vec<Vec<Copy>> {
    let mut out = vec![Vec::new()];
    let mut packets = 0;
    for c in copies {
        let mut done = 0;
        while done < c.bytes {
            if packets == COPIES_PER_SUBMIT {
                out.push(Vec::new());
                packets = 0;
            }
            let room = (COPIES_PER_SUBMIT - packets) as u64 * max;
            let n = (c.bytes - done).min(room);
            out.last_mut().expect("one group").push(Copy {
                dst: c.dst + done,
                src: c.src + done,
                bytes: n,
            });
            packets += n.div_ceil(max) as usize;
            done += n;
        }
    }
    out.retain(|g| !g.is_empty());
    out
}

/// A user-mode SDMA queue.
struct Ring {
    id: u32,
    /// The packets.
    buffer: Bo,
    /// The write pointer at 0, the read pointer at 8, both in bytes and
    /// counting up for ever (SDMA 4 and later).
    pointers: Bo,
    doorbell: Map,
    doorbell_at: usize,
    wptr: u64,
}

impl Ring {
    fn new(gpu: &Gpu) -> Result<Ring> {
        let ring = gpu.alloc(RING_BYTES, Kind::Gtt)?;
        let pointers = gpu.alloc(4096, Kind::Gtt)?;
        view(&pointers).write_bytes(0, &[0; 16]);
        let mut c = sys::CreateQueue {
            ring_base_address: ring.va,
            write_pointer_address: pointers.va,
            read_pointer_address: pointers.va + 8,
            ring_size: RING_BYTES as u32,
            gpu_id: gpu.node.gpu_id,
            queue_type: sys::QUEUE_TYPE_SDMA,
            queue_percentage: sys::MAX_QUEUE_PERCENTAGE,
            queue_priority: sys::NORMAL_QUEUE_PRIORITY,
            ..Default::default()
        };
        // SAFETY: matches AMDKFD_IOC_CREATE_QUEUE; ring and pointers stay
        // allocated for the queue's life.
        if let Err(e) = unsafe {
            call(
                gpu.kfd,
                sys::CREATE_QUEUE,
                &raw mut c,
                "CREATE_QUEUE (SDMA)",
            )
        } {
            gpu.free(ring);
            gpu.free(pointers);
            return Err(e);
        }
        let window = c.doorbell_offset & !(DOORBELL_WINDOW as u64 - 1);
        let doorbell = match Map::new(0, DOORBELL_WINDOW, RW, libc::MAP_SHARED, gpu.kfd.0, window) {
            Ok(m) => m,
            Err(e) => {
                destroy(gpu, c.queue_id);
                gpu.free(ring);
                gpu.free(pointers);
                return Err(e);
            }
        };
        Ok(Ring {
            id: c.queue_id,
            buffer: ring,
            pointers,
            doorbell,
            doorbell_at: usize::try_from(c.doorbell_offset - window).expect("in the window"),
            wptr: 0,
        })
    }

    fn rptr(&self) -> u64 {
        view(&self.pointers).read64(8)
    }

    /// Hands the engine everything written so far.
    fn ring_doorbell(&self) {
        view(&self.pointers).write64(0, self.wptr);
        fence(Ordering::SeqCst);
        self.doorbell.write64(self.doorbell_at, self.wptr);
    }

    /// Appends `dwords` (one submission) and rings. A submission that would
    /// cross the end of the ring starts at its beginning, the rest padded
    /// with `NOP`s.
    fn submit(&mut self, dwords: &[u32]) -> Result<()> {
        let bytes = dwords.len() as u64 * 4;
        let at = self.wptr % RING_BYTES;
        let pad = if at + bytes > RING_BYTES {
            RING_BYTES - at
        } else {
            0
        };
        // Room: what the engine hasn't read stays untouched.
        let start = Instant::now();
        while self.wptr + pad + bytes - self.rptr() > RING_BYTES - 8 {
            if start.elapsed() > Duration::from_secs(10) {
                return Err(Error::Timeout(Duration::from_secs(10)));
            }
            std::thread::sleep(Duration::from_micros(20));
        }
        let r = view(&self.buffer);
        for i in 0..pad / 4 {
            r.write32((at + i * 4) as usize, packet::NOP);
        }
        self.wptr += pad;
        let at = (self.wptr % RING_BYTES) as usize;
        for (i, d) in dwords.iter().enumerate() {
            r.write32(at + i * 4, *d);
        }
        self.wptr += bytes;
        self.ring_doorbell();
        Ok(())
    }

    fn close(self, gpu: &Gpu) {
        destroy(gpu, self.id);
        drop(self.doorbell);
        gpu.free(self.buffer);
        gpu.free(self.pointers);
    }
}

/// The driver's SDMA ring through `DRM_AMDGPU_CS`.
struct Cs {
    drm: &'static Fd,
    ctx: u32,
    /// One indirect buffer per signal slot, reused with it.
    ibs: Bo,
}

impl Cs {
    fn new(gpu: &Gpu) -> Result<Cs> {
        let drm = gpu.drm();
        let mut c = sys::drm::Ctx {
            op: sys::drm::CTX_OP_ALLOC_CTX,
            ..Default::default()
        };
        // SAFETY: matches DRM_IOCTL_AMDGPU_CTX.
        unsafe { call(drm, sys::drm::IOCTL_CTX, &raw mut c, "DRM_AMDGPU_CTX") }?;
        let ibs = match gpu.alloc(IB_SLOT * SLOTS, Kind::Gtt) {
            Ok(b) => b,
            Err(e) => {
                Self::free_ctx(drm, c.op);
                return Err(e);
            }
        };
        // The allocated id comes back in the place of `op`.
        Ok(Cs {
            drm,
            ctx: c.op,
            ibs,
        })
    }

    fn free_ctx(drm: &Fd, ctx: u32) {
        let mut c = sys::drm::Ctx {
            op: sys::drm::CTX_OP_FREE_CTX,
            ctx_id: ctx,
            ..Default::default()
        };
        // SAFETY: matches DRM_IOCTL_AMDGPU_CTX.
        let _ = unsafe { call(drm, sys::drm::IOCTL_CTX, &raw mut c, "DRM_AMDGPU_CTX") };
    }

    /// Submits `dwords` as the indirect buffer of fence `f`'s slot.
    fn submit(&mut self, f: u64, dwords: &[u32]) -> Result<()> {
        let at = Signals::slot(f) as u64 * IB_SLOT;
        let ib = view(&self.ibs);
        for (i, d) in dwords.iter().enumerate() {
            ib.write32(at as usize + i * 4, *d);
        }
        // SDMA fetches indirect buffers in 8-dword lines.
        let len = dwords.len().next_multiple_of(8);
        for i in dwords.len()..len {
            ib.write32(at as usize + i * 4, packet::NOP);
        }
        fence(Ordering::SeqCst);
        let mut chunk_ib = sys::drm::CsChunkIb {
            va_start: self.ibs.va + at,
            ib_bytes: (len * 4) as u32,
            ip_type: sys::drm::HW_IP_DMA,
            ..Default::default()
        };
        let mut chunk = sys::drm::CsChunk {
            chunk_id: sys::drm::CHUNK_ID_IB,
            length_dw: (size_of::<sys::drm::CsChunkIb>() / 4) as u32,
            chunk_data: (&raw mut chunk_ib) as u64,
        };
        let chunks = [(&raw mut chunk) as u64];
        let mut cs = sys::drm::Cs {
            ctx_id: self.ctx,
            num_chunks: 1,
            chunks: chunks.as_ptr() as u64,
            ..Default::default()
        };
        // SAFETY: matches DRM_IOCTL_AMDGPU_CS; the chunk array and the chunk
        // live across the call, the indirect buffer until its fence.
        unsafe { call(self.drm, sys::drm::IOCTL_CS, &raw mut cs, "DRM_AMDGPU_CS") }
    }

    fn close(self, gpu: &Gpu) {
        Self::free_ctx(self.drm, self.ctx);
        gpu.free(self.ibs);
    }
}

enum Way {
    Ring(Ring),
    Cs(Cs),
}

/// The SDMA engine of one GPU, reached by [`Route`].
pub struct Sdma {
    pub route: Route,
    /// The engine's generation ([`version`]).
    pub version: u32,
    max_copy: u64,
    pub(crate) signals: Signals,
    way: Way,
    /// Copies of several submissions, as (first, last) fence: the first
    /// one's slot holds the start timestamp.
    spans: std::collections::VecDeque<(u64, u64)>,
}

impl Sdma {
    /// # Errors
    ///
    /// [`Error::Unsupported`] for a GPU without SDMA engines; errors of KFD
    /// or DRM pass through.
    pub fn new(gpu: &Gpu, route: Route) -> Result<Sdma> {
        let version = version(gpu.node.gfx_target_version).ok_or_else(|| {
            Error::Unsupported(format!("SDMA of {} (before gfx9)", gpu.node.gfx()))
        })?;
        if gpu.node.sdma_engines == 0 {
            return Err(Error::Unsupported(format!(
                "{} has no SDMA engine",
                gpu.node.gfx()
            )));
        }
        let signals = Signals::new(gpu)?;
        let way = match route {
            Route::Kfd => Ring::new(gpu).map(Way::Ring),
            Route::Drm => Cs::new(gpu).map(Way::Cs),
        };
        match way {
            Ok(way) => Ok(Sdma {
                route,
                version,
                max_copy: max_copy(version),
                signals,
                way,
                spans: std::collections::VecDeque::new(),
            }),
            Err(e) => {
                signals.free(gpu);
                Err(e)
            }
        }
    }

    #[must_use]
    pub fn last(&self) -> u64 {
        self.signals.last
    }

    pub fn done(&mut self, f: u64) -> bool {
        self.signals.done(f)
    }

    /// # Errors
    ///
    /// [`Error::Timeout`] when fence `f` hasn't signalled after `timeout`.
    pub fn wait(&mut self, f: u64, timeout: Duration) -> Result<()> {
        let route = self.route;
        let rptr = match &self.way {
            Way::Ring(r) => Some((r.rptr(), r.wptr)),
            Way::Cs(_) => None,
        };
        self.signals.wait(f, timeout, |s| {
            format!(
                "SDMA ({route:?}) fence {f} (last {}): signal {}{}",
                s.last,
                s.value(f),
                rptr.map_or_else(String::new, |(r, w)| format!(", read {r} of {w} bytes"))
            )
        })
    }

    /// Queues `copies` after the compute signal value at `wait` (if any) is
    /// 0; returns the fence of the last submission. `before_arm` runs before
    /// each fence is armed, with the fence that slot's consumer is (0: none),
    /// so the caller can wait for it on the compute queue.
    ///
    /// # Errors
    ///
    /// As [`wait`](Self::wait) for a ring or slot that stays busy, and the
    /// errors of `DRM_AMDGPU_CS`.
    pub fn copy(
        &mut self,
        wait: Option<u64>,
        copies: &[Copy],
        mut before_arm: impl FnMut(u64) -> Result<()>,
    ) -> Result<u64> {
        let mut last = self.signals.last;
        let mut first = None;
        let mut wait = wait;
        let groups = groups(copies, self.max_copy);
        let n = groups.len();
        for (i, group) in groups.into_iter().enumerate() {
            before_arm(self.signals.next_consumer())?;
            let route = self.route;
            let (f, signal) = self
                .signals
                .arm(|s| format!("SDMA ({route:?}): slot of fence {} busy", s.last + 1))?;
            let first = *first.get_or_insert(f);
            // The start in the first submission's slot, the end in the last's.
            let stamps = Stamps {
                start: (i == 0).then_some(signal + sys::signal::START_TS as u64),
                end: (i + 1 == n).then_some(signal + sys::signal::END_TS as u64),
            };
            let dwords = build(
                wait.take(),
                &group,
                self.max_copy,
                stamps,
                signal + sys::signal::VALUE as u64,
            );
            match &mut self.way {
                Way::Ring(r) => r.submit(&dwords)?,
                Way::Cs(c) => c.submit(f, &dwords)?,
            }
            last = f;
            if i + 1 == n && first != f {
                self.spans.push_back((first, f));
            }
        }
        // Spans whose first slot has been armed again tell nothing any more.
        while self
            .spans
            .front()
            .is_some_and(|&(first, _)| first + SLOTS <= self.signals.last)
        {
            self.spans.pop_front();
        }
        Ok(last)
    }

    /// Times fences `first..=last` (several [`copy`](Self::copy) calls) as
    /// one: [`timestamps`](Self::timestamps) of `last` then starts at
    /// `first`'s start.
    pub fn join(&mut self, first: u64, last: u64) {
        if first != last {
            self.spans.retain(|&(_, l)| l != last);
            self.spans.push_back((first, last));
        }
    }

    /// Global timestamps (start, end) of the copy that fence `f` ends, once
    /// it signalled and while the slots still hold them.
    pub fn timestamps(&mut self, f: u64) -> Option<(u64, u64)> {
        if !self.done(f) {
            return None;
        }
        let first = self
            .spans
            .iter()
            .find(|&&(_, last)| last == f)
            .map_or(f, |&(first, _)| first);
        if first + SLOTS <= self.signals.last {
            return None;
        }
        let (a, _) = self.signals.raw_timestamps(first);
        let (_, b) = self.signals.raw_timestamps(f);
        (a != 0 && b >= a).then_some((a, b))
    }

    pub fn close(mut self, gpu: &Gpu) {
        let last = self.signals.last;
        let _ = self.wait(last, Duration::from_secs(5));
        match self.way {
            Way::Ring(r) => r.close(gpu),
            Way::Cs(c) => c.close(gpu),
        }
        self.signals.free(gpu);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn generations_map_to_their_sdma() {
        assert_eq!(version(90012), Some(4));
        assert_eq!(version(90402), Some(4));
        assert_eq!(version(100_305), Some(5));
        assert_eq!(version(110_501), Some(6));
        assert_eq!(version(120_001), Some(7));
        assert_eq!(version(80003), None);
        assert_eq!(max_copy(5), 1 << 22);
        assert!(max_copy(7) < 1 << 22 && max_copy(7).is_multiple_of(32));
    }

    #[test]
    fn packets_encode_as_the_engine_reads_them() {
        let c = packet::copy(0x1_2345_6000, 0xabcd_0000, 4096);
        assert_eq!(c, [0x1, 4095, 0, 0xabcd_0000, 0, 0x2345_6000, 0x1]);
        let w = packet::wait_equal(0x7f00_0000_0008, 0);
        assert_eq!(w[0], 0x8 | (3 << 28) | (1 << 31));
        assert_eq!((w[1], w[2], w[3], w[4]), (8, 0x7f00, 0, u32::MAX));
        assert_eq!(w[5], 0x0fff_0010);
        assert_eq!(packet::timestamp(0x1_0000_0028), [0xd | (2 << 8), 0x28, 1]);
        let a = packet::add64(0x10, -1);
        assert_eq!(a[0], 0xa | (47 << 25));
        assert_eq!((a[3], a[4]), (u32::MAX, u32::MAX));
    }

    #[test]
    fn large_copies_split_into_packets_and_submissions() {
        let big = Copy {
            dst: 0x1000_0000,
            src: 0x8000_0000,
            bytes: 3 * packet::MAX_COPY + 5,
        };
        let max = packet::MAX_COPY;
        let d = build(Some(0x40), &[big], max, Stamps::default(), 0x80);
        // wait (6) + four copies (7 each) + signal (8).
        assert_eq!(d.len(), 6 + 4 * 7 + 8);
        assert_eq!(
            d[6 + 3 * 7 + 1],
            4,
            "the last copy is the remaining 5 bytes"
        );
        assert_eq!(d[6 + 3 * 7 + 5], 0x1000_0000 + 3 * (1 << 22));
        // With timestamps around the copies.
        let stamps = Stamps {
            start: Some(0x100),
            end: Some(0x108),
        };
        let t = build(None, &[big], max, stamps, 0x80);
        assert_eq!(t.len(), 3 + 4 * 7 + 3 + 8);
        assert_eq!((t[0] & 0xff, t[3] & 0xff, t[31] & 0xff), (13, 1, 13));
        assert_eq!(t[32], 0x108);
        // A copy too large for one submission becomes several.
        let huge = Copy {
            bytes: (COPIES_PER_SUBMIT as u64 + 1) * packet::MAX_COPY,
            ..big
        };
        let g = groups(&[huge], max);
        assert_eq!(g.len(), 2);
        assert_eq!(g[1][0].bytes, packet::MAX_COPY);
        assert_eq!(
            g[1][0].dst,
            big.dst + COPIES_PER_SUBMIT as u64 * packet::MAX_COPY
        );
        let both = Stamps {
            start: Some(0),
            end: Some(8),
        };
        assert!(build(Some(0), &g[0], max, both, 0).len() <= SUBMIT_DWORDS);
        assert_eq!(groups(&[], max), Vec::<Vec<Copy>>::new());
        // SDMA 7's smaller copies split differently.
        let seven = groups(&[huge], max_copy(7));
        assert_eq!(seven.len(), 2);
        let packets: u64 = seven[0].iter().map(|c| c.bytes.div_ceil(max_copy(7))).sum();
        assert!(packets <= COPIES_PER_SUBMIT as u64);
    }
}
