//! One AQL compute queue on KFD: the packet ring, `amd_queue_t` with the
//! dispatch ids, EOP and context save buffers, the doorbell, and a pool of
//! completion signals ([`Signals`]) with one kernarg slot each.
//!
//! Fences count packets from 1: fence `n` is packet `n - 1` and owns signal
//! and kernarg slot `n % SLOTS`. Every packet carries the barrier bit, so
//! packets retire in order.

use std::sync::atomic::{Ordering, fence};
use std::time::Duration;

use super::kfd::{Bo, Gpu, Kind, call};
use super::signals::{SLOTS, Signals, view};
use super::sys::{self, DispatchPacket, queue as q};
use crate::runtime::error::{Error, Result};
use crate::runtime::os::{Map, RW};

/// Packets in the ring (a power of two).
const RING: u64 = 1024;
/// Bytes of kernarg per slot (the shared kernels take under 400).
pub const KERNARG_SLOT: u64 = 1024;
/// Doorbells come in a two-page window.
pub(crate) const DOORBELL_WINDOW: usize = 0x2000;
/// `dep_signal` entries of a barrier-AND packet.
pub const BARRIER_DEPS: usize = 5;

/// What a dispatch needs from its kernel.
#[derive(Debug, Clone, Copy)]
pub struct Launch<'a> {
    /// GPU address of the kernel descriptor.
    pub descriptor: u64,
    pub kernargs: &'a [u8],
    /// Work items, a multiple of `group`.
    pub grid: u32,
    pub group: u16,
    /// LDS bytes.
    pub lds: u32,
}

pub struct Queue {
    id: u32,
    ring: Bo,
    /// `amd_queue_t`: the dispatch ids the CP reads and writes.
    header: Bo,
    pub(crate) signals: Signals,
    kernargs: Bo,
    /// Kept for the queue's life, which the CP uses.
    eop: Bo,
    cwsr: Bo,
    doorbell: Map,
    doorbell_at: usize,
}

impl Queue {
    /// Creates the queue on `gpu`.
    ///
    /// # Errors
    ///
    /// [`Error::OutOfMemory`] when the buffers don't fit, [`Error::Os`]
    /// when KFD refuses the queue or the doorbell mapping.
    pub fn new(gpu: &Gpu) -> Result<Queue> {
        let sizes = gpu.node.queue_sizes()?;
        let ring = gpu.alloc(RING * 64, Kind::Gtt)?;
        let amd_queue = gpu.alloc(4096, Kind::Gtt)?;
        let signals = Signals::new(gpu)?;
        let kernargs = gpu.alloc(SLOTS * KERNARG_SLOT, Kind::Gtt)?;
        let eop = gpu.alloc(u64::from(sizes.eop), Kind::Gtt)?;
        let cwsr = gpu.alloc(sizes.cwsr_buffer(), Kind::Gtt)?;
        // In zeroed system memory, as `ROCr` keeps them: with uncleared device
        // memory there (KFD doesn't clear VRAM) the 680M's CP never takes a
        // packet. The CP must find no saved state.
        for bo in [&eop, &cwsr] {
            view(bo).write_bytes(0, &vec![0; bo.size as usize]);
        }

        // Every slot invalid until published: the CP stops there.
        let r = view(&ring);
        for slot in 0..RING as usize {
            r.write32(slot * 64, u32::from(sys::PACKET_INVALID));
        }
        let a = view(&amd_queue);
        a.write32(q::TYPE, 1);
        a.write32(q::FEATURES, 1);
        a.write64(q::BASE_ADDRESS, ring.va);
        a.write32(q::SIZE_PACKETS, RING as u32);
        a.write32(q::QUEUE_PROPERTIES, q::IS_PTR64 | q::ENABLE_PROFILING);
        a.write32(
            q::READ_DISPATCH_ID_FIELD_BASE_BYTE_OFFSET,
            q::READ_DISPATCH_ID as u32,
        );

        let mut c = sys::CreateQueue {
            ring_base_address: ring.va,
            write_pointer_address: amd_queue.va + q::WRITE_DISPATCH_ID as u64,
            read_pointer_address: amd_queue.va + q::READ_DISPATCH_ID as u64,
            ring_size: u32::try_from(ring.size).expect("small ring"),
            gpu_id: gpu.node.gpu_id,
            queue_type: sys::QUEUE_TYPE_COMPUTE_AQL,
            queue_percentage: sys::MAX_QUEUE_PERCENTAGE,
            queue_priority: sys::NORMAL_QUEUE_PRIORITY,
            eop_buffer_address: eop.va,
            eop_buffer_size: u64::from(sizes.eop),
            ctx_save_restore_address: cwsr.va,
            ctx_save_restore_size: sizes.cwsr,
            ctl_stack_size: sizes.ctl_stack,
            ..Default::default()
        };
        // SAFETY: matches AMDKFD_IOC_CREATE_QUEUE; every buffer it names
        // stays allocated for the queue's life.
        unsafe { call(gpu.kfd, sys::CREATE_QUEUE, &raw mut c, "CREATE_QUEUE") }?;
        let window = c.doorbell_offset & !(DOORBELL_WINDOW as u64 - 1);
        let doorbell = match Map::new(0, DOORBELL_WINDOW, RW, libc::MAP_SHARED, gpu.kfd.0, window) {
            Ok(m) => m,
            Err(e) => {
                destroy(gpu, c.queue_id);
                return Err(e);
            }
        };
        Ok(Queue {
            id: c.queue_id,
            ring,
            header: amd_queue,
            signals,
            kernargs,
            eop,
            cwsr,
            doorbell,
            doorbell_at: usize::try_from(c.doorbell_offset - window).expect("in the window"),
        })
    }

    /// Fences handed out so far; the next packet gets `last() + 1`.
    #[must_use]
    pub fn last(&self) -> u64 {
        self.signals.last
    }

    /// Whether fence `f` has signalled. Fence 0 (nothing) always has.
    pub fn done(&mut self, f: u64) -> bool {
        self.signals.done(f)
    }

    /// Blocks until fence `f` has signalled.
    ///
    /// # Errors
    ///
    /// [`Error::Timeout`] when it hasn't after `timeout`.
    pub fn wait(&mut self, f: u64, timeout: Duration) -> Result<()> {
        let (ring, header) = (view(&self.ring), view(&self.header));
        self.signals.wait(f, timeout, |s| state(s, ring, header, f))
    }

    /// Queues a barrier packet that runs nothing until every signal in
    /// `deps` (at most [`BARRIER_DEPS`] `amd_signal_t` addresses) is 0;
    /// returns its fence. The backend checks with one without deps that the
    /// queue runs at all.
    ///
    /// # Errors
    ///
    /// [`Error::InvalidArgument`] for more than five deps, else as
    /// [`dispatch`](Self::dispatch).
    pub fn barrier(&mut self, deps: &[u64]) -> Result<u64> {
        if deps.len() > BARRIER_DEPS {
            return Err(Error::InvalidArgument(format!(
                "{} signals for one barrier, at most {BARRIER_DEPS}",
                deps.len()
            )));
        }
        let (f, signal) = self.claim()?;
        // header, reserved, dep_signal[5] at 8, reserved, completion_signal at 56.
        let mut p = [0u8; 64];
        for (i, d) in deps.iter().enumerate() {
            p[8 + i * 8..16 + i * 8].copy_from_slice(&d.to_le_bytes());
        }
        p[56..64].copy_from_slice(&signal.to_le_bytes());
        let header = sys::PACKET_BARRIER_AND
            | sys::HEADER_BARRIER
            | sys::HEADER_ACQUIRE_SYSTEM
            | sys::HEADER_RELEASE_SYSTEM;
        self.publish(f, &p, u32::from(header));
        Ok(f)
    }

    /// The next fence, once a ring slot is free; its signal armed. Returns
    /// the fence and the signal's address.
    fn claim(&mut self) -> Result<(u64, u64)> {
        let next = self.signals.last + 1;
        self.wait(next.saturating_sub(RING), Duration::from_secs(10))?;
        let (ring, header) = (view(&self.ring), view(&self.header));
        self.signals
            .arm(|s| state(s, ring, header, next.saturating_sub(SLOTS)))
    }

    /// Writes packet `f - 1` (its first word `first` last), and rings.
    fn publish(&mut self, f: u64, packet: &[u8; 64], first: u32) {
        let index = f - 1;
        let r = view(&self.ring);
        let base = ((index % RING) * 64) as usize;
        r.write_bytes(base + 4, &packet[4..]);
        // The header last: the CP takes the packet once it is valid.
        fence(Ordering::Release);
        r.write32(base, first);
        view(&self.header).write64(q::WRITE_DISPATCH_ID, index + 1);
        // Write-combined host writes and the dispatch id before the doorbell.
        fence(Ordering::SeqCst);
        self.doorbell.write64(self.doorbell_at, index);
    }

    /// Queues one dispatch; returns its fence.
    ///
    /// # Errors
    ///
    /// [`Error::InvalidArgument`] for kernargs larger than a slot,
    /// [`Error::Timeout`] when the ring or the slot stays busy for 10 s.
    pub fn dispatch(&mut self, l: &Launch<'_>) -> Result<u64> {
        if l.kernargs.len() as u64 > KERNARG_SLOT {
            return Err(Error::InvalidArgument(format!(
                "{} bytes of kernel arguments, at most {KERNARG_SLOT}",
                l.kernargs.len()
            )));
        }
        let (f, signal) = self.claim()?;
        let karg = Signals::slot(f) as u64 * KERNARG_SLOT;
        view(&self.kernargs).write_bytes(karg as usize, l.kernargs);

        let p = DispatchPacket {
            header: 0,
            setup: 1,
            workgroup_size_x: l.group,
            workgroup_size_y: 1,
            workgroup_size_z: 1,
            grid_size_x: l.grid,
            grid_size_y: 1,
            grid_size_z: 1,
            group_segment_size: l.lds,
            kernel_object: l.descriptor,
            kernarg_address: self.kernargs.va + karg,
            completion_signal: signal,
            ..Default::default()
        };
        // SAFETY: DispatchPacket is plain old data of 64 bytes.
        let bytes: [u8; 64] = unsafe { core::mem::transmute(p) };
        let first = u32::from(sys::DISPATCH_HEADER) | (u32::from(p.setup) << 16);
        self.publish(f, &bytes, first);
        Ok(f)
    }

    /// CP timestamps of fence `f` (start, end), once signalled and while its
    /// slot holds it.
    pub fn timestamps(&mut self, f: u64) -> Option<(u64, u64)> {
        self.signals.timestamps(f)
    }

    /// Waits for everything and destroys the queue on `gpu`, then frees its
    /// buffers.
    pub fn close(mut self, gpu: &Gpu) {
        let _ = self.wait(self.last(), Duration::from_secs(5));
        destroy(gpu, self.id);
        drop(self.doorbell);
        self.signals.free(gpu);
        for bo in [self.ring, self.header, self.kernargs, self.eop, self.cwsr] {
            gpu.free(bo);
        }
    }
}

/// What the CP has done about fence `f`, for the trace of a timeout.
fn state(
    s: &Signals,
    ring: crate::runtime::os::View,
    header: crate::runtime::os::View,
    f: u64,
) -> String {
    let packet = (f.saturating_sub(1) % RING) as usize * 64;
    format!(
        "AQL fence {f} (last {}): read dispatch id {}, write dispatch id {}, signal {}, header {:#06x}",
        s.last,
        header.read64(q::READ_DISPATCH_ID),
        header.read64(q::WRITE_DISPATCH_ID),
        s.value(f),
        ring.read32(packet) & 0xffff
    )
}

pub(crate) fn destroy(gpu: &Gpu, id: u32) {
    let mut d = sys::DestroyQueue {
        queue_id: id,
        pad: 0,
    };
    // SAFETY: matches AMDKFD_IOC_DESTROY_QUEUE.
    let _ = unsafe { call(gpu.kfd, sys::DESTROY_QUEUE, &raw mut d, "DESTROY_QUEUE") };
}
