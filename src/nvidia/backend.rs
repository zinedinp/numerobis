//! `Backend` over the raw RM path: VRAM buffers, copy-engine staging,
//! QMD launches of the shared Emulated kernels (`kernels/bin/nvidia`).

use std::collections::HashMap;
use std::sync::{Arc, Mutex, MutexGuard, Weak};
use std::time::Duration;

use super::Tier as NvTier;
use super::gpu::{Alloc, Gpu, Mem, cards};
use super::program::{Kernel, Program};
use super::qmd::{self, Qmd};
use super::queue::{Engine as Ch, Queues};
use super::sys::{
    ADA_COMPUTE_A, AMPERE_COMPUTE_B, AMPERE_DMA_COPY_B, ARCH_AD100, ARCH_GA100,
    NV2080_CTRL_MC_GET_ARCH_INFO_PARAMS, NVC7C0_INVALIDATE_SHADER_CACHES_NO_WFI,
    NVC7C0_SEND_PCAS_A, NVC7C0_SEND_SIGNALING_PCAS2_B, PCAS_ACTION_INVALIDATE_SCHEDULE,
    SUBC_COMPUTE, TESTED_DRIVERS,
};
use crate::engine::{Access, DeviceClass, Engine, FeatureSet};
use crate::runtime::Device;
use crate::runtime::abi::{Launch, lower};
use crate::runtime::backend::{Arg, Backend, BufId, Fence, KernelId, KernelKey, OpSig};
use crate::runtime::error::{Error, Result};
use crate::runtime::os::Fd;
use crate::tiers::Tier;

static SM80: &[u8] = include_bytes!("../../kernels/bin/nvidia/sm_80-emulated.cubin");
static SM89: &[u8] = include_bytes!("../../kernels/bin/nvidia/sm_89-emulated.cubin");

/// Generic-address windows for shared and local memory (must not overlap allocations).
const SHARED_WINDOW: u64 = 0xfe00_0000;
const LOCAL_WINDOW: u64 = 0xff00_0000;
/// Local memory per thread reserved at setup.
const SLM_PER_THREAD: u64 = 0x200;
const BLOCK: u32 = 128;
const AMPERE_COMPUTE_A: u32 = 0xc6c0;

/// CTAs for `items` work items.
fn grid(items: u32) -> u32 {
    items.div_ceil(BLOCK).max(1)
}

struct Loaded {
    image: Alloc,
    program: Program,
}

struct State {
    q: Queues,
    next: u64,
    bufs: HashMap<u64, Alloc>,
    kernels: Vec<OpSig>,
    /// Staging buffers to free once their fence has passed.
    retire: Vec<(u64, Alloc)>,
    downloads: HashMap<u64, (Alloc, usize)>,
    code: Loaded,
    _local: Alloc,
}

/// One NVIDIA GPU on the raw path.
pub struct NvBackend {
    features: FeatureSet,
    sass: u32,
    gpu: Gpu,
    state: Mutex<State>,
}

/// Compute class, kernel image and tier for an RM architecture.
fn select(arch: &NV2080_CTRL_MC_GET_ARCH_INFO_PARAMS) -> Option<(u32, &'static [u8], NvTier)> {
    match arch.architecture {
        ARCH_GA100 if arch.implementation == 0 => Some((AMPERE_COMPUTE_A, SM80, NvTier::Sm80)),
        ARCH_GA100 => Some((AMPERE_COMPUTE_B, SM80, NvTier::Sm80)),
        ARCH_AD100 => Some((ADA_COMPUTE_A, SM89, NvTier::Sm89)),
        _ => None,
    }
}

fn driver_version() -> Option<String> {
    std::fs::read_to_string("/sys/module/nvidia/version")
        .ok()
        .map(|s| s.trim().to_string())
}

/// Tries to open every GPU the driver lists; errors are kept for diagnostics
/// (`examples/probe.rs`). Unknown driver versions are refused unless
/// `NUMEROBIS_NV_ANY_DRIVER=1` (the RM ABI changes between releases).
#[must_use]
pub fn probe() -> Vec<Result<Device>> {
    let Some(version) = driver_version() else {
        return Vec::new();
    };
    if !TESTED_DRIVERS.contains(&version.as_str())
        && std::env::var_os("NUMEROBIS_NV_ANY_DRIVER").is_none()
    {
        return vec![Err(Error::DriverTooOld {
            have: version,
            need: TESTED_DRIVERS.join(" or "),
        })];
    }
    let list = match Fd::open("/dev/nvidiactl").and_then(|ctl| cards(&ctl)) {
        Ok(l) => l,
        Err(e) => return vec![Err(e)],
    };
    list.into_iter()
        .map(|(minor, gpu_id, _)| {
            // One backend per GPU per process: opening costs RM objects, UVM
            // registration, channels and a kernel upload, and the GPU may also
            // drive the display.
            static OPEN: Mutex<Vec<(u32, Weak<NvBackend>)>> = Mutex::new(Vec::new());
            let mut open = OPEN
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            open.retain(|(_, w)| w.strong_count() > 0);
            if let Some(b) = open
                .iter()
                .find(|(m, _)| *m == minor)
                .and_then(|(_, w)| w.upgrade())
            {
                return Ok(Device::from_backend(b));
            }
            let b = Arc::new(NvBackend::open(minor, gpu_id, &version)?);
            open.push((minor, Arc::downgrade(&b)));
            Ok(Device::from_backend(b))
        })
        .collect()
}

/// Every GPU that opens on the raw path.
pub fn open_all() -> Vec<Device> {
    probe().into_iter().filter_map(Result::ok).collect()
}

impl NvBackend {
    /// Opens GPU `gpu_id` behind `/dev/nvidia<minor>` (driver `version`) and loads its Emulated
    /// kernels.
    ///
    /// # Errors
    ///
    /// [`Error::Os`] when a device node or RM call fails,
    /// [`Error::Unsupported`] for an architecture the raw path has no
    /// kernels for, [`Error::BadBinary`] for kernels that do not load or
    /// need more stack than reserved; errors while setting up the queues pass through.
    pub fn open(minor: u32, gpu_id: u32, version: &str) -> Result<NvBackend> {
        let gpu = Gpu::open(Fd::open("/dev/nvidiactl")?, minor, gpu_id)?;
        let (compute_class, image, tier) = select(&gpu.arch).ok_or_else(|| {
            Error::Unsupported(format!(
                "NVIDIA architecture {:#x} on the raw path",
                gpu.arch.architecture
            ))
        })?;
        let program = Program::parse(image)?;
        if u64::from(program.stack) > SLM_PER_THREAD {
            return Err(Error::BadBinary(format!(
                "kernels need {} bytes of stack",
                program.stack
            )));
        }
        let mut q = Queues::new(&gpu, compute_class, AMPERE_DMA_COPY_B)?;

        let gr = gpu.gr;
        let per_tpc = ((SLM_PER_THREAD * 32).next_multiple_of(0x200)
            * u64::from(gr.max_warps_per_sm.max(1))
            * u64::from(gr.sm_per_tpc.max(1)))
        .next_multiple_of(0x8000);
        let local_size = (per_tpc * u64::from(gr.tpc_per_gpc.max(1)) * u64::from(gr.gpcs.max(1)))
            .next_multiple_of(0x20000);
        let local = gpu.alloc(local_size, Mem::Vram)?;
        q.init(
            compute_class,
            AMPERE_DMA_COPY_B,
            (local.va, per_tpc),
            (SHARED_WINDOW, LOCAL_WINDOW),
        )?;

        // Kernel image: VRAM, uploaded once through staging.
        let code = gpu.alloc(image.len() as u64, Mem::Vram)?;
        let staging = gpu.alloc(image.len() as u64, Mem::Sys)?;
        staging.cpu().write_bytes(0, image);
        let v = q.submit(Ch::Copy, |p| {
            Queues::copy(p, code.va, staging.va, image.len() as u64);
            Ok(())
        })?;
        q.wait(v, Duration::from_secs(10))?;
        gpu.free(staging);

        let sv = gr.sm_version;
        let sass = (((sv >> 8) & 0xf) << 4) | (sv & 0xf);
        let features = FeatureSet {
            name: gpu.name.clone(),
            class: DeviceClass::Gpu,
            driver: format!("nvidia {version} (raw RM)"),
            engines: vec![(Engine::Matrix, Access::Raw), (Engine::Copy, Access::Raw)],
            tiers: vec![Tier::Emulated, Tier::Nvidia(tier)],
        };
        Ok(NvBackend {
            features,
            sass,
            state: Mutex::new(State {
                q,
                next: 0,
                bufs: HashMap::new(),
                kernels: Vec::new(),
                retire: Vec::new(),
                downloads: HashMap::new(),
                code: Loaded {
                    image: code,
                    program,
                },
                _local: local,
            }),
            gpu,
        })
    }

    fn lock(&self) -> MutexGuard<'_, State> {
        self.state
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
    }

    /// Frees staging buffers whose work has finished.
    fn reap(&self, s: &mut State) {
        let done = s.q.completed();
        let (old, keep): (Vec<_>, Vec<_>) = core::mem::take(&mut s.retire)
            .into_iter()
            .partition(|(f, _)| *f <= done);
        s.retire = keep;
        for (_, a) in old {
            self.gpu.free(a);
        }
    }

    fn qmd(&self, k: &Kernel, image: u64, cbuf0: u64, items: u32) -> Qmd {
        let mut q = Qmd::default();
        let prog = image + k.text.0;
        let shared = (0x400 + k.shared).next_multiple_of(128);
        // SM config in 4 KiB units + 1, smallest of 32/64/100 KiB that fits.
        let cfg = [32u32, 64, 100]
            .into_iter()
            .map(|c| c * 1024)
            .find(|&c| c >= shared)
            .unwrap_or(100 * 1024)
            / 4096
            + 1;
        q.set(qmd::QMD_GROUP_ID, 0x3f);
        q.set(qmd::SM_GLOBAL_CACHING_ENABLE, 1);
        q.set(qmd::INVALIDATE_TEXTURE_HEADER_CACHE, 1);
        q.set(qmd::INVALIDATE_TEXTURE_SAMPLER_CACHE, 1);
        q.set(qmd::INVALIDATE_TEXTURE_DATA_CACHE, 1);
        q.set(qmd::INVALIDATE_SHADER_DATA_CACHE, 1);
        q.set(qmd::API_VISIBLE_CALL_LIMIT, 1);
        q.set(qmd::SAMPLER_INDEX, 1);
        q.set(
            qmd::CWD_MEMBAR_TYPE,
            u64::from(qmd::CWD_MEMBAR_L1_SYSMEMBAR),
        );
        q.set(qmd::QMD_MAJOR_VERSION, 3);
        q.set(qmd::QMD_VERSION, 0);
        q.set(qmd::SHARED_MEMORY_SIZE, u64::from(shared));
        q.set(qmd::MIN_SM_CONFIG_SHARED_MEM_SIZE, u64::from(cfg));
        q.set(qmd::TARGET_SM_CONFIG_SHARED_MEM_SIZE, u64::from(cfg));
        q.set(qmd::MAX_SM_CONFIG_SHARED_MEM_SIZE, 0x1a);
        q.set(qmd::REGISTER_COUNT_V, u64::from(k.registers));
        q.set(qmd::BARRIER_COUNT, 1);
        q.set(qmd::SHADER_LOCAL_MEMORY_HIGH_SIZE, SLM_PER_THREAD);
        q.set(qmd::PROGRAM_ADDRESS_LOWER, prog & 0xffff_ffff);
        q.set(qmd::PROGRAM_ADDRESS_UPPER, prog >> 32);
        q.set(
            qmd::PROGRAM_PREFETCH_ADDR_LOWER_SHIFTED,
            (prog >> 8) & 0xffff_ffff,
        );
        q.set(qmd::PROGRAM_PREFETCH_ADDR_UPPER_SHIFTED, prog >> 40);
        q.set(qmd::PROGRAM_PREFETCH_SIZE, (k.text.1 >> 8).min(0x1ff));
        q.set(qmd::SASS_VERSION, u64::from(self.sass));
        q.set(qmd::CTA_RASTER_WIDTH, u64::from(grid(items)));
        q.set(qmd::CTA_RASTER_HEIGHT, 1);
        q.set(qmd::CTA_RASTER_DEPTH, 1);
        q.set(qmd::CTA_THREAD_DIMENSION0, u64::from(BLOCK));
        q.set(qmd::CTA_THREAD_DIMENSION1, 1);
        q.set(qmd::CTA_THREAD_DIMENSION2, 1);
        q.constant_buffer(0, cbuf0, k.cbuf0_size);
        if let Some((off, size)) = k.cbuf2 {
            q.constant_buffer(2, image + off, size);
        }
        q
    }

    /// Constant bank 0: driver area (launch dimensions, windows, stack top)
    /// plus the kernel parameters.
    fn cbuf0(k: &Kernel, l: &Launch, bufs: &HashMap<u64, Alloc>) -> Result<Vec<u8>> {
        if l.args.len() != k.param_offsets.len() {
            return Err(Error::InvalidArgument(format!(
                "{}: {} args for {} parameters",
                k.name,
                l.args.len(),
                k.param_offsets.len()
            )));
        }
        let mut c = vec![0u8; k.cbuf0_size as usize];
        // ntid.{x,y,z} at 0x0 and nctaid.{x,y,z} at 0xc: SASS reads them from here.
        for (i, v) in [BLOCK, 1, 1, grid(l.items), 1, 1].into_iter().enumerate() {
            c[i * 4..i * 4 + 4].copy_from_slice(&v.to_le_bytes());
        }
        c[0x18..0x20].copy_from_slice(&SHARED_WINDOW.to_le_bytes());
        c[0x20..0x28].copy_from_slice(&LOCAL_WINDOW.to_le_bytes());
        c[0x28..0x30].copy_from_slice(&0x00ff_fdc0_u64.to_le_bytes());
        for (a, &off) in l.args.iter().zip(&k.param_offsets) {
            let at = (k.params.0 + off) as usize;
            match *a {
                Arg::Buf(b) => {
                    let va = bufs
                        .get(&b.0)
                        .ok_or_else(|| Error::InvalidArgument("unknown buffer".into()))?
                        .va;
                    c[at..at + 8].copy_from_slice(&va.to_le_bytes());
                }
                Arg::U32(v) => c[at..at + 4].copy_from_slice(&v.to_le_bytes()),
                Arg::I32(v) => c[at..at + 4].copy_from_slice(&v.to_le_bytes()),
                Arg::F32(v) => c[at..at + 4].copy_from_slice(&v.to_bits().to_le_bytes()),
                #[allow(unreachable_patterns)]
                _ => return Err(Error::InvalidArgument("unsupported argument".into())),
            }
        }
        Ok(c)
    }
}

impl Backend for NvBackend {
    fn features(&self) -> &FeatureSet {
        &self.features
    }

    fn alloc(&self, bytes: usize) -> Result<BufId> {
        let a = self.gpu.alloc(bytes as u64, Mem::Vram)?;
        let mut s = self.lock();
        s.next += 1;
        let id = s.next;
        s.bufs.insert(id, a);
        Ok(BufId(id))
    }

    fn free(&self, id: BufId) {
        let a = self.lock().bufs.remove(&id.0);
        if let Some(a) = a {
            self.gpu.free(a);
        }
    }

    fn upload(&self, id: BufId, offset: usize, src: &[u8]) -> Result<Fence> {
        let mut s = self.lock();
        self.reap(&mut s);
        let dst = s
            .bufs
            .get(&id.0)
            .ok_or_else(|| Error::InvalidArgument("unknown buffer".into()))?;
        if offset + src.len() > dst.size as usize {
            return Err(Error::InvalidArgument("upload out of bounds".into()));
        }
        let dst = dst.va + offset as u64;
        if src.is_empty() {
            return Ok(Fence(s.q.last));
        }
        let staging = self.gpu.alloc(src.len() as u64, Mem::Sys)?;
        staging.cpu().write_bytes(0, src);
        let v = s.q.submit(Ch::Copy, |p| {
            Queues::copy(p, dst, staging.va, src.len() as u64);
            Ok(())
        })?;
        s.retire.push((v, staging));
        Ok(Fence(v))
    }

    fn download(&self, id: BufId, offset: usize, len: usize) -> Result<Fence> {
        let mut s = self.lock();
        self.reap(&mut s);
        let src = s
            .bufs
            .get(&id.0)
            .ok_or_else(|| Error::InvalidArgument("unknown buffer".into()))?;
        if offset + len > src.size as usize {
            return Err(Error::InvalidArgument("download out of bounds".into()));
        }
        let src = src.va + offset as u64;
        let staging = self.gpu.alloc(len as u64, Mem::Sys)?;
        let v = s.q.submit(Ch::Copy, |p| {
            if len > 0 {
                Queues::copy(p, staging.va, src, len as u64);
            }
            Ok(())
        })?;
        s.downloads.insert(v, (staging, len));
        Ok(Fence(v))
    }

    fn take_download(&self, fence: Fence, dst: &mut [u8]) -> Result<()> {
        let (staging, len) = self
            .lock()
            .downloads
            .remove(&fence.0)
            .ok_or_else(|| Error::InvalidArgument("unknown download".into()))?;
        if dst.len() != len {
            return Err(Error::InvalidArgument("download size mismatch".into()));
        }
        staging.cpu().read_bytes(0, dst);
        self.gpu.free(staging);
        Ok(())
    }

    fn supports(&self, key: &KernelKey) -> bool {
        // Only the format-generic Emulated kernels exist on this path so far.
        key.tier == Tier::Emulated && crate::runtime::abi::supported(key.sig)
    }

    fn load_kernel(&self, key: &KernelKey) -> Result<KernelId> {
        if !self.supports(key) {
            return Err(Error::Unsupported(format!(
                "{} on {}",
                key.name,
                key.tier.name()
            )));
        }
        let mut s = self.lock();
        s.kernels.push(key.sig);
        Ok(KernelId(s.kernels.len() as u64 - 1))
    }

    fn dispatch(&self, kernel: KernelId, args: &[Arg]) -> Result<Fence> {
        let mut s = self.lock();
        self.reap(&mut s);
        let sig = *s
            .kernels
            .get(kernel.0 as usize)
            .ok_or_else(|| Error::InvalidArgument("unknown kernel".into()))?;
        let launches = lower(sig, args)?;
        // Build every launch's data first (needs `bufs`), then push.
        let mut prepared = Vec::with_capacity(launches.len());
        for l in &launches {
            let k = s
                .code
                .program
                .kernel(l.symbol)
                .ok_or_else(|| Error::BadBinary(format!("missing kernel {}", l.symbol)))?
                .clone();
            let c = Self::cbuf0(&k, l, &s.bufs)?;
            prepared.push((k, c, l.items));
        }
        let image = s.code.image.va;
        let v = s.q.submit(Ch::Compute, |p| {
            for (i, (k, c, items)) in prepared.iter().enumerate() {
                if *items == 0 {
                    continue;
                }
                if i > 0 {
                    p.barrier();
                }
                let cb = p.data(c, 256)?;
                let qmd = self.qmd(k, image, cb, *items);
                let qa = p.data(&qmd.to_bytes(), 256)?;
                p.method(
                    SUBC_COMPUTE,
                    NVC7C0_INVALIDATE_SHADER_CACHES_NO_WFI,
                    &[(1 << 12) | (1 << 4) | 1],
                );
                p.method(SUBC_COMPUTE, NVC7C0_SEND_PCAS_A, &[(qa >> 8) as u32]);
                p.method(
                    SUBC_COMPUTE,
                    NVC7C0_SEND_SIGNALING_PCAS2_B,
                    &[PCAS_ACTION_INVALIDATE_SCHEDULE],
                );
            }
            Ok(())
        })?;
        Ok(Fence(v))
    }

    fn wait(&self, fence: Fence, timeout: Duration) -> Result<()> {
        // Poll without holding the lock for long.
        let r = {
            let s = self.lock();
            s.q.wait(fence.0, timeout)
        };
        let mut s = self.lock();
        self.reap(&mut s);
        r
    }

    fn elapsed(&self, fence: Fence) -> Option<Duration> {
        let s = self.lock();
        s.q.timestamps(fence.0)
            .map(|(a, b)| Duration::from_nanos(b.saturating_sub(a)))
    }
}

impl Drop for NvBackend {
    fn drop(&mut self) {
        let s = self.state.get_mut().unwrap_or_else(|e| e.into_inner());
        let _ = s.q.wait(s.q.last, Duration::from_secs(5));
    }
}
