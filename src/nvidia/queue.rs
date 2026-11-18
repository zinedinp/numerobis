//! GPFIFO channels and the device timeline.
//!
//! One channel group with two channels on the GR runlist: compute
//! (subchannel 1) and copy (subchannel 4). Every submission waits for the
//! previous one on a 64-bit timeline semaphore in system memory and releases
//! the next value, so all work is ordered and a [`Fence`](crate::runtime::Fence)
//! is just a timeline value. Each submission also writes start/end GPU
//! timestamps for `elapsed`.
//!
//! Push buffers, QMDs and constant banks live in a system-memory arena that is
//! recycled once the GPU has passed every submission in it.

use std::sync::atomic::{Ordering, fence};
use std::time::{Duration, Instant};

use super::gpu::{Alloc, Gpu, Mem};
use super::sys::*;
use crate::runtime::error::{Error, Result};
use crate::runtime::os::View;

const ENTRIES: u32 = 0x8000;
/// Second channel's offset in the GPFIFO area.
const COPY_OFFSET: u64 = 0x10_0000;
const ARENA: u64 = 16 << 20;
/// Signal page layout.
const SEM_TIMELINE: usize = 0;
const SEM_SCRATCH: usize = 64;
const TS_BASE: usize = 4096;
const TS_SLOTS: u64 = 1024;
const SIGNALS: u64 = TS_BASE as u64 + TS_SLOTS * 32;

/// Which channel a submission goes to.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Engine {
    Compute,
    Copy,
}

struct Channel {
    ring: View,
    controls: View,
    put: u32,
    token: u32,
}

/// Commands of one submission plus its side data in the arena.
pub struct Push<'a> {
    words: Vec<u32>,
    q: &'a mut Queues,
}

impl Push<'_> {
    /// Appends `method(subc, mthd, data.len())` and `data`.
    pub fn method(&mut self, subc: u32, mthd: u32, data: &[u32]) {
        self.words.push(method(subc, mthd, data.len() as u32));
        self.words.extend_from_slice(data);
    }

    /// Host semaphore operation on `addr` (`SEM_ADDR_LO..SEM_EXECUTE`).
    pub fn semaphore(&mut self, addr: u64, payload: u64, exec: u32) {
        self.method(
            0,
            NVC56F_SEM_ADDR_LO,
            &[
                addr as u32,
                (addr >> 32) as u32,
                payload as u32,
                (payload >> 32) as u32,
                exec | SEM_EXECUTE_PAYLOAD_SIZE_64BIT,
            ],
        );
    }

    /// Waits for the engine to drain (a release with WFI to a scratch slot).
    pub fn barrier(&mut self) {
        let a = self.q.signals.va + SEM_SCRATCH as u64;
        self.semaphore(a, 0, SEM_EXECUTE_RELEASE | SEM_EXECUTE_RELEASE_WFI);
    }

    /// Copies `bytes` into the arena (aligned to `align`) and returns its GPU address.
    pub fn data(&mut self, bytes: &[u8], align: u64) -> Result<u64> {
        let va = self.q.arena_alloc(bytes.len() as u64, align)?;
        self.q
            .arena
            .cpu()
            .write_bytes((va - self.q.arena.va) as usize, bytes);
        Ok(va)
    }
}

pub struct Queues {
    _group: NvHandle,
    _area: Alloc,
    /// Error notifier of both channels (released with the RM client).
    _notifier: Alloc,
    compute: Channel,
    copy: Channel,
    signals: Alloc,
    arena: Alloc,
    head: u64,
    /// Timeline value of the last submission.
    pub last: u64,
    doorbell: View,
}

fn data64(v: u64) -> [u32; 2] {
    [(v >> 32) as u32, v as u32]
}

impl Queues {
    pub fn new(gpu: &Gpu, compute_class: u32, copy_class: u32) -> Result<Queues> {
        let mut gp = NV_CHANNEL_GROUP_ALLOCATION_PARAMETERS {
            engineType: NV2080_ENGINE_TYPE_GRAPHICS,
            ..Default::default()
        };
        let group = rm_alloc(
            &gpu.ctl,
            gpu.root,
            gpu.device,
            KEPLER_CHANNEL_GROUP_A,
            Some(&mut gp),
        )?;
        let area = gpu.alloc(2 << 20, Mem::VramMapped)?;
        let mut cp = NV_CTXSHARE_ALLOCATION_PARAMETERS {
            hVASpace: gpu.vaspace,
            flags: NV_CTXSHARE_ALLOCATION_FLAGS_SUBCONTEXT_ASYNC,
            ..Default::default()
        };
        let ctxshare = rm_alloc(
            &gpu.ctl,
            gpu.root,
            group,
            FERMI_CONTEXT_SHARE_A,
            Some(&mut cp),
        )?;
        let notifier = gpu.alloc(64 << 10, Mem::Vram)?;
        let channel = |offset: u64| -> Result<Channel> {
            let mut p = NV_CHANNEL_ALLOC_PARAMS {
                hObjectError: notifier.handle(),
                hObjectBuffer: area.handle(),
                gpFifoOffset: area.va + offset,
                gpFifoEntries: ENTRIES,
                hContextShare: ctxshare,
                ..Default::default()
            };
            p.hUserdMemory[0] = area.handle();
            p.userdOffset[0] = offset + ENTRIES as u64 * 8;
            let ch = rm_alloc(
                &gpu.ctl,
                gpu.root,
                group,
                AMPERE_CHANNEL_GPFIFO_A,
                Some(&mut p),
            )?;
            rm_alloc::<()>(&gpu.ctl, gpu.root, ch, compute_class, None)?;
            rm_alloc::<()>(&gpu.ctl, gpu.root, ch, copy_class, None)?;
            let mut t = NVC36F_CTRL_CMD_GPFIFO_GET_WORK_SUBMIT_TOKEN_PARAMS {
                workSubmitToken: u32::MAX,
            };
            rm_control(
                &gpu.ctl,
                gpu.root,
                ch,
                NVC36F_CTRL_CMD_GPFIFO_GET_WORK_SUBMIT_TOKEN,
                &mut t,
            )?;
            let base = gpu.reserve_va(64 << 20)?;
            gpu.register_channel(ch, base, 64 << 20)?;
            let cpu = area.cpu();
            Ok(Channel {
                ring: View {
                    ptr: cpu.ptr.wrapping_add(offset as usize),
                    len: ENTRIES as usize * 8,
                },
                controls: View {
                    ptr: cpu.ptr.wrapping_add((offset + ENTRIES as u64 * 8) as usize),
                    len: 512,
                },
                put: 0,
                token: t.workSubmitToken,
            })
        };
        let compute = channel(0)?;
        let copy = channel(COPY_OFFSET)?;
        let mut sp = NVA06C_CTRL_GPFIFO_SCHEDULE_PARAMS {
            bEnable: 1,
            ..Default::default()
        };
        rm_control(
            &gpu.ctl,
            gpu.root,
            group,
            NVA06C_CTRL_CMD_GPFIFO_SCHEDULE,
            &mut sp,
        )?;

        let signals = gpu.alloc(SIGNALS, Mem::Sys)?;
        let arena = gpu.alloc(ARENA, Mem::Sys)?;
        let doorbell = View {
            ptr: gpu.usermode.ptr,
            len: gpu.usermode.len,
        };
        Ok(Queues {
            _group: group,
            _area: area,
            _notifier: notifier,
            compute,
            copy,
            signals,
            arena,
            head: 0,
            last: 0,
            doorbell,
        })
    }

    /// Current timeline value (last completed submission).
    pub fn completed(&self) -> u64 {
        self.signals.cpu().read64(SEM_TIMELINE)
    }

    /// Spins (then sleeps) until the timeline reaches `value`.
    pub fn wait(&self, value: u64, timeout: Duration) -> Result<()> {
        let t0 = Instant::now();
        let mut spins = 0u32;
        while self.completed() < value {
            if t0.elapsed() > timeout {
                return Err(Error::Timeout(timeout));
            }
            spins += 1;
            if spins < 1000 {
                core::hint::spin_loop();
            } else if spins < 2000 {
                std::thread::yield_now();
            } else {
                std::thread::sleep(Duration::from_micros(50));
            }
        }
        Ok(())
    }

    /// GPU start/end timestamps (ns) of submission `value`, if still recorded.
    pub fn timestamps(&self, value: u64) -> Option<(u64, u64)> {
        let v = self.signals.cpu();
        let o = TS_BASE + (value % TS_SLOTS) as usize * 32;
        (v.read64(o) == value && v.read64(o + 16) == value)
            .then(|| (v.read64(o + 8), v.read64(o + 24)))
    }

    fn arena_alloc(&mut self, size: u64, align: u64) -> Result<u64> {
        let mut at = self.head.next_multiple_of(align);
        if at + size > self.arena.size {
            if size > self.arena.size {
                return Err(Error::InvalidArgument("submission too large".into()));
            }
            // Recycle: everything submitted so far must be done.
            self.wait(self.last, Duration::from_secs(10))?;
            at = 0;
        }
        self.head = at + size;
        Ok(self.arena.va + at)
    }

    /// Builds and submits one ordered unit of work on `engine`; returns its timeline value.
    pub fn submit(
        &mut self,
        engine: Engine,
        body: impl FnOnce(&mut Push) -> Result<()>,
    ) -> Result<u64> {
        let value = self.last + 1;
        let sem = self.signals.va + SEM_TIMELINE as u64;
        let ts = self.signals.va + TS_BASE as u64 + (value % TS_SLOTS) * 32;
        let mut p = Push {
            words: Vec::with_capacity(64),
            q: self,
        };
        if value > 1 {
            p.semaphore(sem, value - 1, SEM_EXECUTE_ACQ_CIRC_GEQ);
        }
        p.semaphore(
            ts,
            value,
            SEM_EXECUTE_RELEASE | SEM_EXECUTE_RELEASE_TIMESTAMP,
        );
        body(&mut p)?;
        p.semaphore(
            ts + 16,
            value,
            SEM_EXECUTE_RELEASE | SEM_EXECUTE_RELEASE_WFI | SEM_EXECUTE_RELEASE_TIMESTAMP,
        );
        p.semaphore(
            sem,
            value,
            SEM_EXECUTE_RELEASE | SEM_EXECUTE_RELEASE_WFI | SEM_EXECUTE_RELEASE_TIMESTAMP,
        );
        p.method(0, NVC56F_NON_STALL_INTERRUPT, &[0]);
        let words = core::mem::take(&mut p.words);
        let bytes: Vec<u8> = words.iter().flat_map(|w| w.to_le_bytes()).collect();
        let cmd = p.data(&bytes, 256)?;
        self.kick(engine, cmd, words.len() as u64)?;
        self.last = value;
        Ok(value)
    }

    fn kick(&mut self, engine: Engine, cmd: u64, len_words: u64) -> Result<()> {
        let doorbell = self.doorbell;
        let ch = match engine {
            Engine::Compute => &mut self.compute,
            Engine::Copy => &mut self.copy,
        };
        // Ring full? (never in practice: the arena recycles first)
        let t0 = Instant::now();
        while (ch.put + 1) % ENTRIES == ch.controls.read32(GPFIFO_CONTROL_GPGET) % ENTRIES {
            if t0.elapsed() > Duration::from_secs(10) {
                return Err(Error::Timeout(Duration::from_secs(10)));
            }
            std::thread::yield_now();
        }
        let entry = ((cmd >> 2) << 2) | (len_words << 42) | (1 << 41);
        let slot = (ch.put % ENTRIES) as usize * 8;
        ch.ring.write32(slot, entry as u32);
        ch.ring.write32(slot + 4, (entry >> 32) as u32);
        ch.put = (ch.put + 1) % ENTRIES;
        fence(Ordering::SeqCst);
        ch.controls.write32(GPFIFO_CONTROL_GPPUT, ch.put);
        fence(Ordering::SeqCst);
        doorbell.write32(USERMODE_NOTIFY_CHANNEL_PENDING, ch.token);
        Ok(())
    }

    /// Binds the classes and sets up shader windows and local memory.
    pub fn init(
        &mut self,
        compute_class: u32,
        copy_class: u32,
        local: (u64, u64),
        windows: (u64, u64),
    ) -> Result<()> {
        let (local_va, per_tpc) = local;
        let (shared_window, local_window) = windows;
        self.submit(Engine::Compute, |p| {
            p.method(SUBC_COMPUTE, NVC7C0_SET_OBJECT, &[compute_class]);
            p.method(
                SUBC_COMPUTE,
                NVC7C0_SET_SHADER_LOCAL_MEMORY_WINDOW_A,
                &data64(local_window),
            );
            p.method(
                SUBC_COMPUTE,
                NVC7C0_SET_SHADER_SHARED_MEMORY_WINDOW_A,
                &data64(shared_window),
            );
            p.method(
                SUBC_COMPUTE,
                NVC7C0_SET_SHADER_LOCAL_MEMORY_A,
                &data64(local_va),
            );
            let [hi, lo] = data64(per_tpc);
            p.method(
                SUBC_COMPUTE,
                NVC7C0_SET_SHADER_LOCAL_MEMORY_NON_THROTTLED_A,
                &[hi, lo, 0xff],
            );
            Ok(())
        })?;
        self.submit(Engine::Copy, |p| {
            p.method(SUBC_COPY, NVC7C0_SET_OBJECT, &[copy_class]);
            Ok(())
        })?;
        Ok(())
    }

    /// Copy-engine transfer of `len` bytes.
    pub fn copy(p: &mut Push, dst: u64, src: u64, len: u64) {
        let mut off = 0;
        while off < len {
            let n = (len - off).min(1 << 30);
            let [sh, sl] = data64(src + off);
            let [dh, dl] = data64(dst + off);
            p.method(SUBC_COPY, NVC7B5_OFFSET_IN_UPPER, &[sh, sl, dh, dl]);
            p.method(SUBC_COPY, NVC7B5_LINE_LENGTH_IN, &[n as u32]);
            p.method(SUBC_COPY, NVC7B5_LAUNCH_DMA, &[LAUNCH_DMA_COPY]);
            off += n;
        }
    }
}
