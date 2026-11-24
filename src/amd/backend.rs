//! `Backend` over KFD: buffers the CPU and GPU share an address for, one
//! AQL compute queue for the shared Emulated kernels from `kernels/bin/amd`,
//! and copies by one of three [`CopyEngine`]s.
//!
//! With an SDMA engine, every buffer remembers its last compute and its
//! last copy fence. A copy waits on the GPU for the compute work that last
//! touched its buffer (`POLL_REGMEM` on the AQL signal), a dispatch for the
//! copies into or out of its buffers (an AQL barrier on the SDMA signal):
//! uploads, kernels and downloads queue up without the host blocking, and
//! copies overlap kernels on other buffers.

use std::collections::HashMap;
use std::sync::{Arc, Mutex, MutexGuard, Weak};
use std::time::Duration;

use super::Tier as AmdTier;
use super::kfd::{self, Bo, Gpu, Kind, Node};
use super::program::{Program, Word};
use super::queue::{Launch, Queue};
use super::sdma::{self, Route, Sdma};
use crate::engine::{Access, DeviceClass, Engine, FeatureSet};
use crate::pci;
use crate::platform::{Status, Support};
use crate::runtime::Device;
use crate::runtime::abi::lower;
use crate::runtime::backend::{Arg, Backend, BufId, Fence, KernelId, KernelKey, OpSig};
use crate::runtime::error::{Error, Result};
use crate::tiers::Tier;

/// Run on an AMD Radeon 680M (gfx1035, RDNA 2 APU, SDMA 5.2, Linux 7.3,
/// KFD 1.23), with every [`CopyEngine`]; the other targets are built and
/// load-checked only.
pub const SUPPORT: Support = Support::new(Status::Tested, Status::Unavailable);

static RDNA2: &[u8] = include_bytes!("../../kernels/bin/amd/gfx10-3-generic-emulated.hsaco");
static GFX1100: &[u8] = include_bytes!("../../kernels/bin/amd/gfx1100-emulated.hsaco");
static GFX1200: &[u8] = include_bytes!("../../kernels/bin/amd/gfx1200-emulated.hsaco");
static GFX908: &[u8] = include_bytes!("../../kernels/bin/amd/gfx908-emulated.hsaco");
static GFX90A: &[u8] = include_bytes!("../../kernels/bin/amd/gfx90a-emulated.hsaco");
static GFX942: &[u8] = include_bytes!("../../kernels/bin/amd/gfx942-emulated.hsaco");
static GFX950: &[u8] = include_bytes!("../../kernels/bin/amd/gfx950-emulated.hsaco");
static GFX9: &[u8] = include_bytes!("../../kernels/bin/amd/gfx9-generic-emulated.hsaco");
static GFX9_4: &[u8] = include_bytes!("../../kernels/bin/amd/gfx9-4-generic-emulated.hsaco");
static RDNA1: &[u8] = include_bytes!("../../kernels/bin/amd/gfx10-1-generic-emulated.hsaco");
static GFX11: &[u8] = include_bytes!("../../kernels/bin/amd/gfx11-generic-emulated.hsaco");
static GFX12: &[u8] = include_bytes!("../../kernels/bin/amd/gfx12-generic-emulated.hsaco");

/// Work-group size of every launch (the shared kernels allow 256).
const GROUP: u16 = 256;
/// Fences of copies done by the host: done when made.
const HOST: u64 = 1 << 63;
/// Fences of the SDMA engine, apart from the compute queue's.
const SDMA: u64 = 1 << 62;
/// How long the queues get to show they run when the backend opens.
const SELF_TEST: Duration = Duration::from_secs(2);
/// Staging kept for reuse, at most this many bytes.
const POOL_BYTES: u64 = 256 << 20;
/// Uploads and downloads go through staging chunks of this many bytes.
const CHUNK: usize = 32 << 20;
/// Upload chunks in flight before the host waits for the oldest: the pool's
/// worth.
const IN_FLIGHT: usize = (POOL_BYTES / CHUNK as u64) as usize;

/// What a staging buffer for `len` bytes is allocated as, so that it fits
/// later copies of about that size: a power of two up to 64 MiB, a
/// multiple of 64 MiB above (a power of two would waste up to half).
fn staging_size(len: u64) -> u64 {
    const STEP: u64 = 64 << 20;
    if len <= STEP {
        len.next_power_of_two().max(4096)
    } else {
        len.next_multiple_of(STEP)
    }
}

/// Staging buffers kept for the next copies: allocating one costs four
/// ioctls and a page fault per page on first touch.
#[derive(Default)]
struct Pool {
    free: Vec<Bo>,
    bytes: u64,
}

impl Pool {
    /// A kept buffer of at least `len` bytes (at most twice that), or a new one.
    fn take(&mut self, gpu: &Gpu, len: u64) -> Result<Bo> {
        let fit = self
            .free
            .iter()
            .enumerate()
            .filter(|(_, b)| b.size >= len && b.size <= len.max(4096) * 2)
            .min_by_key(|(_, b)| b.size)
            .map(|(i, _)| i);
        if let Some(i) = fit {
            let b = self.free.swap_remove(i);
            self.bytes -= b.size;
            return Ok(b);
        }
        gpu.alloc(staging_size(len), Kind::Staging)
    }

    /// Keeps `bo` for reuse, or frees it when the pool is full.
    fn give(&mut self, gpu: &Gpu, bo: Bo) {
        if self.bytes + bo.size > POOL_BYTES {
            gpu.free(bo);
        } else {
            self.bytes += bo.size;
            self.free.push(bo);
        }
    }
}

/// How uploads and downloads move data.
///
/// The OS-provided way to the copy engine is the driver's ring
/// ([`DriverSdma`](Self::DriverSdma)); other systems offer the same only
/// through their graphics APIs (D3D12 copy queues, Metal blit encoders,
/// Vulkan transfer queues), Linux alone has KFD's user-mode queues.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum CopyEngine {
    /// A user-mode SDMA queue of this process (KFD): no syscall per copy.
    Sdma,
    /// The `amdgpu` driver's own SDMA ring (`DRM_AMDGPU_CS` on the render
    /// node): the kernel schedules every submission. Buffers the kernel
    /// evicts while a copy runs are not waited for (KFD's eviction stops
    /// only KFD queues), so this is for when the user-mode queue can't be
    /// had.
    DriverSdma,
    /// The host copies through the shared mapping after the compute queue
    /// drained. Fine on an APU, slow for VRAM behind a small BAR.
    Cpu,
}

impl CopyEngine {
    /// `NUMEROBIS_AMD_COPY`: `sdma`, `drm` or `cpu`.
    #[must_use]
    pub fn from_env() -> Option<CopyEngine> {
        Self::parse(&std::env::var("NUMEROBIS_AMD_COPY").ok()?)
    }

    #[must_use]
    pub fn parse(s: &str) -> Option<CopyEngine> {
        Some(match s.trim() {
            "sdma" => CopyEngine::Sdma,
            "drm" => CopyEngine::DriverSdma,
            "cpu" => CopyEngine::Cpu,
            _ => return None,
        })
    }

    #[must_use]
    pub fn name(self) -> &'static str {
        match self {
            CopyEngine::Sdma => "SDMA user queue",
            CopyEngine::DriverSdma => "SDMA through the amdgpu driver",
            CopyEngine::Cpu => "CPU",
        }
    }
}

/// The code object and matrix tier for a `gfx_target_version`
/// (`major * 10000 + minor * 100 + stepping`, the last two in hex digits:
/// gfx90c is 90012); `None` where no binary is built for it. A target's
/// own binary first, else its family's generic one.
#[must_use]
pub fn select(gfx_target_version: u32) -> Option<(&'static [u8], Option<AmdTier>)> {
    Some(match gfx_target_version {
        // Vega: gfx900, 902, 904, 906, 909, 90c; no matrix engine.
        90000 | 90002 | 90004 | 90006 | 90009 | 90012 => (GFX9, None),
        90008 => (GFX908, Some(AmdTier::Gfx908)),
        90010 => (GFX90A, Some(AmdTier::Gfx90a)),
        90400 | 90401 => (GFX9_4, Some(AmdTier::Gfx942)),
        90402 => (GFX942, Some(AmdTier::Gfx942)),
        90500 => (GFX950, Some(AmdTier::Gfx950)),
        100_100..=100_103 => (RDNA1, None),
        100_300..=100_306 => (RDNA2, None),
        110_000 => (GFX1100, Some(AmdTier::Gfx11)),
        110_001..=110_003 | 110_500..=110_503 => (GFX11, Some(AmdTier::Gfx11)),
        120_000 => (GFX1200, Some(AmdTier::Gfx12)),
        120_001 => (GFX12, Some(AmdTier::Gfx12)),
        _ => return None,
    })
}

struct Buf {
    /// Allocated and imported alike: KFD frees an import's handle only.
    bo: Bo,
    /// Last compute fence that used it, last copy fence that did.
    compute: u64,
    sdma: u64,
}

enum Staged {
    /// Copied by the host already.
    Host(Vec<u8>),
    /// In staging buffers, chunk after chunk, once the last copy's fence
    /// signals.
    Device {
        pieces: Vec<(Bo, usize)>,
        len: usize,
        fence: u64,
    },
}

struct State {
    q: Queue,
    sdma: Option<Sdma>,
    next: u64,
    bufs: HashMap<u64, Buf>,
    kernels: Vec<OpSig>,
    downloads: HashMap<u64, Staged>,
    /// Upload staging, back to the pool once its copy fence has passed.
    retire: Vec<(u64, Bo)>,
    pool: Pool,
}

/// Before the compute queue arms its next signal: the copy that waits on
/// that slot's previous fence must have run.
fn compute_slot_free(q: &Queue, sdma: Option<&mut Sdma>) -> Result<()> {
    let c = q.signals.next_consumer();
    match sdma {
        Some(e) if c != 0 => e.wait(c, Duration::from_secs(10)),
        _ => Ok(()),
    }
}

/// Queues `copies` on the SDMA engine after compute fence `after`.
fn sdma_copy(q: &mut Queue, e: &mut Sdma, after: u64, copies: &[sdma::Copy]) -> Result<u64> {
    let wait = (!q.done(after)).then(|| q.signals.value_address(after));
    let first = e.last() + 1;
    let f = e.copy(wait, copies, |c| {
        if c == 0 {
            Ok(())
        } else {
            q.wait(c, Duration::from_secs(10))
        }
    })?;
    if wait.is_some() {
        q.signals.add_consumer(after, first);
    }
    Ok(f)
}

/// One AMD GPU on the raw KFD path.
pub struct AmdBackend {
    features: FeatureSet,
    copies: CopyEngine,
    program: Program,
    code: Option<Bo>,
    /// GPU timestamp ticks per nanosecond.
    tick_ns: Option<f64>,
    state: Mutex<Option<State>>,
    gpu: Gpu,
}

/// Opens every GPU node KFD lists, copying with `NUMEROBIS_AMD_COPY` if
/// set, else with the SDMA user queue where it runs and the CPU where not.
/// Errors are kept for diagnostics (`examples/probe.rs`).
#[must_use]
pub fn probe() -> Vec<Result<Device>> {
    open_nodes(CopyEngine::from_env())
}

/// As [`probe`], copying with `engine` and nothing else.
#[must_use]
pub fn probe_with(engine: CopyEngine) -> Vec<Result<Device>> {
    open_nodes(Some(engine))
}

fn open_nodes(engine: Option<CopyEngine>) -> Vec<Result<Device>> {
    let Ok(nodes) = kfd::nodes() else {
        return Vec::new();
    };
    nodes
        .into_iter()
        .map(|node| {
            // One backend per GPU and copy engine per process: it owns its
            // queues and its upload of the code.
            type Open = Vec<((u32, Option<CopyEngine>), Weak<AmdBackend>)>;
            static OPEN: Mutex<Open> = Mutex::new(Vec::new());
            let mut open = OPEN
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            open.retain(|(_, w)| w.strong_count() > 0);
            let key = (node.gpu_id, engine);
            if let Some(b) = open
                .iter()
                .find(|(k, _)| *k == key)
                .and_then(|(_, w)| w.upgrade())
            {
                return Ok(Device::from_backend(b));
            }
            let b = Arc::new(match engine {
                Some(e) => AmdBackend::open_with(node, e)?,
                None => AmdBackend::open(node)?,
            });
            open.push((key, Arc::downgrade(&b)));
            Ok(Device::from_backend(b))
        })
        .collect()
}

/// Every AMD GPU that opens on the raw path.
pub fn open_all() -> Vec<Device> {
    probe().into_iter().filter_map(Result::ok).collect()
}

fn name(node: &Node) -> String {
    let gpu = pci::Gpu::from_pci(node.vendor_id, node.device_id);
    match gpu.architecture.name() {
        Some(arch) => format!("AMD {arch} {} ({})", node.gfx(), node.name),
        None => format!("AMD {} ({})", node.gfx(), node.name),
    }
}

/// Opens the SDMA engine by `route` and checks it copies.
fn open_sdma(gpu: &Gpu, route: Route) -> Result<Sdma> {
    let mut e = Sdma::new(gpu, route)?;
    let probe = gpu.alloc(4096, Kind::Gtt)?;
    let cpu = probe.cpu().expect("GTT is CPU-mapped");
    cpu.write_bytes(0, &[0x5a; 64]);
    cpu.write_bytes(64, &[0; 64]);
    let copied = e
        .copy(
            None,
            &[sdma::Copy {
                dst: probe.va + 64,
                src: probe.va,
                bytes: 64,
            }],
            |_| Ok(()),
        )
        .and_then(|f| e.wait(f, SELF_TEST));
    let mut back = [0u8; 64];
    cpu.read_bytes(64, &mut back);
    gpu.free(probe);
    match copied {
        Ok(()) if back == [0x5a; 64] => Ok(e),
        r => {
            e.close(gpu);
            Err(match r {
                Ok(()) => Error::Unsupported(format!("SDMA ({route:?}) copied wrong bytes")),
                Err(Error::Timeout(_)) => {
                    Error::Unsupported(format!("SDMA ({route:?}) does not run"))
                }
                Err(e) => e,
            })
        }
    }
}

impl AmdBackend {
    /// Opens the GPU of topology node `node`, copying with the SDMA user
    /// queue, or with the CPU when that doesn't open (the reason goes into
    /// the driver string).
    ///
    /// # Errors
    ///
    /// As [`open_with`](Self::open_with).
    pub fn open(node: Node) -> Result<AmdBackend> {
        Self::open_inner(node, None)
    }

    /// Opens the GPU of topology node `node`, copying with `engine`.
    ///
    /// # Errors
    ///
    /// [`Error::Unsupported`] for a GPU without a built code object or
    /// queues that don't run, [`Error::BadBinary`] for kernels that need
    /// scratch; errors of KFD and DRM pass through.
    pub fn open_with(node: Node, engine: CopyEngine) -> Result<AmdBackend> {
        Self::open_inner(node, Some(engine))
    }

    fn open_inner(node: Node, engine: Option<CopyEngine>) -> Result<AmdBackend> {
        let (image, tier) = select(node.gfx_target_version)
            .ok_or_else(|| Error::Unsupported(format!("{} on the raw AMD path", node.gfx())))?;
        let program = Program::parse(image)?;
        if let Some(k) = program
            .kernels
            .iter()
            .find(|k| k.private_segment > 0 || k.dynamic_stack)
        {
            return Err(Error::BadBinary(format!(
                "{} needs scratch memory, which the raw queue has none of",
                k.name
            )));
        }
        let display = name(&node);
        let gpu = Gpu::open(node)?;
        let code = gpu.alloc(program.image.len() as u64, Kind::Code)?;
        // Code memory is CPU-mapped (`Kind::Code`).
        if let Some(cpu) = code.cpu() {
            cpu.write_bytes(0, &program.image);
        }
        let mut q = match Queue::new(&gpu) {
            Ok(q) => q,
            Err(e) => {
                gpu.free(code);
                return Err(e);
            }
        };
        // A packet that runs nothing: does the queue run at all?
        let ran = q.barrier(&[]).and_then(|f| q.wait(f, SELF_TEST));
        if let Err(e) = ran {
            q.close(&gpu);
            gpu.free(code);
            return Err(match e {
                Error::Timeout(_) => Error::Unsupported(
                    "the AQL queue does not run (a barrier packet never completed)".into(),
                ),
                e => e,
            });
        }

        let route = |e| match e {
            CopyEngine::Sdma => Some(Route::Kfd),
            CopyEngine::DriverSdma => Some(Route::Drm),
            CopyEngine::Cpu => None,
        };
        let (copies, sdma, fallback) = match engine {
            Some(wanted) => match route(wanted).map(|r| open_sdma(&gpu, r)) {
                None => (wanted, None, String::new()),
                Some(Ok(s)) => (wanted, Some(s), String::new()),
                Some(Err(err)) => {
                    q.close(&gpu);
                    gpu.free(code);
                    return Err(err);
                }
            },
            None => match open_sdma(&gpu, Route::Kfd) {
                Ok(s) => (CopyEngine::Sdma, Some(s), String::new()),
                Err(err) => (CopyEngine::Cpu, None, format!(", SDMA: {err}")),
            },
        };

        let (major, minor) = gpu.version;
        let tick_ns = gpu.timestamp_frequency().map(|hz| hz / 1e9);
        let mut engines = Vec::new();
        if tier.is_some() {
            engines.push((Engine::Matrix, Access::Raw));
        }
        engines.push((Engine::Vector, Access::Raw));
        if sdma.is_some() {
            engines.push((Engine::Copy, Access::Raw));
        }
        Ok(AmdBackend {
            features: FeatureSet {
                name: display,
                class: DeviceClass::Gpu,
                driver: format!(
                    "amdgpu KFD {major}.{minor} (raw AQL, copies: {}{fallback})",
                    copies.name()
                ),
                engines,
                tiers: std::iter::once(Tier::Emulated)
                    .chain(tier.map(Tier::Amd))
                    .collect(),
            },
            copies,
            program,
            code: Some(code),
            tick_ns,
            state: Mutex::new(Some(State {
                q,
                sdma,
                next: 0,
                bufs: HashMap::new(),
                kernels: Vec::new(),
                downloads: HashMap::new(),
                retire: Vec::new(),
                pool: Pool::default(),
            })),
            gpu,
        })
    }

    /// How this backend copies.
    #[must_use]
    pub fn copy_engine(&self) -> CopyEngine {
        self.copies
    }

    fn lock(&self) -> MutexGuard<'_, Option<State>> {
        self.state
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
    }

    fn with<T>(&self, f: impl FnOnce(&mut State) -> Result<T>) -> Result<T> {
        let mut g = self.lock();
        f(g.as_mut().ok_or(Error::DeviceLost)?)
    }

    fn insert(s: &mut State, bo: Bo) -> BufId {
        s.next += 1;
        s.bufs.insert(
            s.next,
            Buf {
                bo,
                compute: 0,
                sdma: 0,
            },
        );
        BufId(s.next)
    }

    /// The buffer and a checked `offset..offset + len` in it.
    fn range(s: &State, id: BufId, offset: usize, len: usize) -> Result<&Buf> {
        let b = s
            .bufs
            .get(&id.0)
            .ok_or_else(|| Error::InvalidArgument("unknown buffer".into()))?;
        if offset
            .checked_add(len)
            .is_none_or(|end| end as u64 > b.bo.size)
        {
            return Err(Error::InvalidArgument("copy out of bounds".into()));
        }
        Ok(b)
    }

    /// The CPU view of `id` over `offset..offset + len`, after the compute
    /// queue has drained ([`CopyEngine::Cpu`]).
    fn host_range(
        s: &mut State,
        id: BufId,
        offset: usize,
        len: usize,
    ) -> Result<crate::runtime::os::View> {
        let last = s.q.last();
        s.q.wait(last, Duration::from_secs(60))?;
        Self::range(s, id, offset, len)?
            .bo
            .cpu()
            .ok_or_else(|| Error::Unsupported("buffer without a CPU mapping".into()))
    }

    /// Returns upload staging whose copies have run to the pool.
    fn reap(&self, s: &mut State) {
        let Some(e) = s.sdma.as_mut() else { return };
        let (done, keep): (Vec<_>, Vec<_>) = std::mem::take(&mut s.retire)
            .into_iter()
            .partition(|(f, _)| e.done(*f));
        s.retire = keep;
        for (_, bo) in done {
            s.pool.give(&self.gpu, bo);
        }
    }

    /// Waits until fewer than [`IN_FLIGHT`] upload chunks are in flight.
    fn make_room(&self, s: &mut State) -> Result<()> {
        while s.retire.len() >= IN_FLIGHT {
            let oldest = s.retire.iter().map(|(f, _)| *f).min().unwrap_or(0);
            if let Some(e) = s.sdma.as_mut() {
                e.wait(oldest, Duration::from_secs(60))?;
            }
            self.reap(s);
        }
        Ok(())
    }

    fn give_back(gpu: &Gpu, s: &mut State, pieces: Vec<(Bo, usize)>) {
        for (bo, _) in pieces {
            s.pool.give(gpu, bo);
        }
    }

    fn host_fence(s: &mut State) -> Fence {
        s.next += 1;
        Fence(HOST | s.next)
    }
}

impl Backend for AmdBackend {
    fn features(&self) -> &FeatureSet {
        &self.features
    }

    fn alloc(&self, bytes: usize) -> Result<BufId> {
        let size = bytes as u64;
        // Device memory the CPU can map, else (with SDMA, which reaches all
        // of it: a discrete card with a small BAR) device memory it can't,
        // else system memory.
        let sdma = self.copies != CopyEngine::Cpu;
        let mut bo = self.gpu.alloc(size, Kind::Vram);
        if sdma && matches!(bo, Err(Error::OutOfMemory { .. })) {
            bo = self.gpu.alloc(size, Kind::VramPrivate);
        }
        let bo = match bo {
            Err(Error::OutOfMemory { .. }) => self.gpu.alloc(size, Kind::Gtt)?,
            r => r?,
        };
        self.with(|s| Ok(Self::insert(s, bo)))
    }

    fn free(&self, id: BufId) {
        let b = self.lock().as_mut().and_then(|s| s.bufs.remove(&id.0));
        if let Some(b) = b {
            self.gpu.free(b.bo);
        }
    }

    fn upload(&self, id: BufId, offset: usize, src: &[u8]) -> Result<Fence> {
        self.with(|s| {
            self.reap(s);
            if s.sdma.is_none() {
                let v = Self::host_range(s, id, offset, src.len())?;
                v.write_bytes(offset, src);
                return Ok(Self::host_fence(s));
            }
            let (dst, after) = {
                let b = Self::range(s, id, offset, src.len())?;
                (b.bo.va + offset as u64, b.compute)
            };
            if src.is_empty() {
                return Ok(Self::host_fence(s));
            }
            // In chunks through kept staging: the host fills one while the
            // engine copies the ones before.
            let mut span = None;
            for (i, piece) in src.chunks(CHUNK).enumerate() {
                self.make_room(s)?;
                let staging = s.pool.take(&self.gpu, piece.len() as u64)?;
                staging
                    .cpu()
                    .expect("staging is CPU-mapped")
                    .write_bytes(0, piece);
                let e = s.sdma.as_mut().expect("checked above");
                let copy = sdma::Copy {
                    dst: dst + (i * CHUNK) as u64,
                    src: staging.va,
                    bytes: piece.len() as u64,
                };
                match sdma_copy(&mut s.q, e, after, &[copy]) {
                    Ok(f) => {
                        s.retire.push((f, staging));
                        span = Some((span.map_or(f, |(first, _)| first), f));
                    }
                    Err(err) => {
                        s.pool.give(&self.gpu, staging);
                        return Err(err);
                    }
                }
            }
            let (first, last) = span.expect("a non-empty upload has a chunk");
            s.sdma.as_mut().expect("checked above").join(first, last);
            if let Some(b) = s.bufs.get_mut(&id.0) {
                b.sdma = last;
            }
            Ok(Fence(SDMA | last))
        })
    }

    fn download(&self, id: BufId, offset: usize, len: usize) -> Result<Fence> {
        self.with(|s| {
            self.reap(s);
            if s.sdma.is_none() || len == 0 {
                let mut out = vec![0; len];
                if len > 0 {
                    Self::host_range(s, id, offset, len)?.read_bytes(offset, &mut out);
                } else {
                    Self::range(s, id, offset, 0)?;
                }
                let f = Self::host_fence(s);
                s.downloads.insert(f.0, Staged::Host(out));
                return Ok(f);
            }
            let (src, after) = {
                let b = Self::range(s, id, offset, len)?;
                (b.bo.va + offset as u64, b.compute)
            };
            let mut pieces: Vec<(Bo, usize)> = Vec::new();
            let mut span = None;
            let mut done = 0;
            while done < len {
                let n = (len - done).min(CHUNK);
                let staging = match s.pool.take(&self.gpu, n as u64) {
                    Ok(b) => b,
                    Err(err) => {
                        Self::give_back(&self.gpu, s, pieces);
                        return Err(err);
                    }
                };
                let e = s.sdma.as_mut().expect("checked above");
                let copy = sdma::Copy {
                    dst: staging.va,
                    src: src + done as u64,
                    bytes: n as u64,
                };
                match sdma_copy(&mut s.q, e, after, &[copy]) {
                    Ok(f) => span = Some((span.map_or(f, |(first, _)| first), f)),
                    Err(err) => {
                        pieces.push((staging, n));
                        Self::give_back(&self.gpu, s, pieces);
                        return Err(err);
                    }
                }
                pieces.push((staging, n));
                done += n;
            }
            let (first, last) = span.expect("a non-empty download has a chunk");
            s.sdma.as_mut().expect("checked above").join(first, last);
            if let Some(b) = s.bufs.get_mut(&id.0) {
                b.sdma = last;
            }
            s.downloads.insert(
                SDMA | last,
                Staged::Device {
                    pieces,
                    len,
                    fence: last,
                },
            );
            Ok(Fence(SDMA | last))
        })
    }

    fn take_download(&self, fence: Fence, dst: &mut [u8]) -> Result<()> {
        let staged = self
            .with(|s| Ok(s.downloads.remove(&fence.0)))?
            .ok_or_else(|| Error::InvalidArgument("unknown download".into()))?;
        match staged {
            Staged::Host(out) => {
                if out.len() != dst.len() {
                    return Err(Error::InvalidArgument("download size mismatch".into()));
                }
                dst.copy_from_slice(&out);
                Ok(())
            }
            Staged::Device { pieces, len, fence } => {
                let r = self.with(|s| {
                    s.sdma
                        .as_mut()
                        .ok_or(Error::DeviceLost)?
                        .wait(fence, Duration::from_secs(60))
                });
                let r = r.and_then(|()| {
                    if len != dst.len() {
                        return Err(Error::InvalidArgument("download size mismatch".into()));
                    }
                    let mut at = 0;
                    for (bo, n) in &pieces {
                        let cpu = bo.cpu().expect("staging is CPU-mapped");
                        cpu.read_bytes(0, &mut dst[at..at + n]);
                        at += n;
                    }
                    Ok(())
                });
                let _ = self.with(|s| {
                    Self::give_back(&self.gpu, s, pieces);
                    Ok(())
                });
                r
            }
        }
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
        self.with(|s| {
            s.kernels.push(key.sig);
            Ok(KernelId(s.kernels.len() as u64 - 1))
        })
    }

    fn dispatch(&self, kernel: KernelId, args: &[Arg]) -> Result<Fence> {
        let code = self.code.as_ref().ok_or(Error::DeviceLost)?.va;
        self.with(|s| {
            self.reap(s);
            let sig = *s
                .kernels
                .get(usize::try_from(kernel.0).unwrap_or(usize::MAX))
                .ok_or_else(|| Error::InvalidArgument("unknown kernel".into()))?;
            let launches = lower(sig, args)?;
            // Arguments are all checked before anything is queued.
            let mut prepared = Vec::with_capacity(launches.len());
            let mut copied = 0;
            for l in launches.iter().filter(|l| l.items > 0) {
                let k = self
                    .program
                    .kernel(l.symbol)
                    .ok_or_else(|| Error::BadBinary(format!("missing kernel {}", l.symbol)))?;
                let group = GROUP.min(u16::try_from(k.max_workgroup).unwrap_or(GROUP));
                let grid = l
                    .items
                    .checked_next_multiple_of(u32::from(group))
                    .ok_or_else(|| Error::InvalidArgument("grid too large".into()))?;
                let mut words = Vec::with_capacity(l.args.len());
                for a in &l.args {
                    words.push(match *a {
                        Arg::Buf(b) => {
                            let b = s
                                .bufs
                                .get(&b.0)
                                .ok_or_else(|| Error::InvalidArgument("unknown buffer".into()))?;
                            copied = copied.max(b.sdma);
                            Word::Ptr(b.bo.va)
                        }
                        Arg::U32(v) => Word::B32(v),
                        Arg::I32(v) => Word::B32(v.cast_unsigned()),
                        Arg::F32(v) => Word::B32(v.to_bits()),
                    });
                }
                let kernargs = k.kernargs(&words, grid, group)?;
                prepared.push((code + k.descriptor, kernargs, grid, group, k.group_segment));
            }
            let State { q, sdma, .. } = s;
            let mut last = q.last();
            // The copies into or out of these buffers first, on the GPU.
            if let Some(e) = sdma.as_mut()
                && !e.done(copied)
            {
                compute_slot_free(q, Some(e))?;
                let address = e.signals.address(copied);
                last = q.barrier(&[address])?;
                e.signals.add_consumer(copied, last);
            }
            for (descriptor, kernargs, grid, group, lds) in &prepared {
                compute_slot_free(q, sdma.as_mut())?;
                last = q.dispatch(&Launch {
                    descriptor: *descriptor,
                    kernargs,
                    grid: *grid,
                    group: *group,
                    lds: *lds,
                })?;
            }
            for a in args {
                if let Arg::Buf(b) = a
                    && let Some(b) = s.bufs.get_mut(&b.0)
                {
                    b.compute = last;
                }
            }
            Ok(Fence(last))
        })
    }

    fn wait(&self, fence: Fence, timeout: Duration) -> Result<()> {
        if fence.0 & HOST != 0 {
            return Ok(());
        }
        self.with(|s| {
            let r = if fence.0 & SDMA != 0 {
                s.sdma
                    .as_mut()
                    .ok_or_else(|| Error::InvalidArgument("not a fence of this device".into()))?
                    .wait(fence.0 & !SDMA, timeout)
            } else {
                s.q.wait(fence.0, timeout)
            };
            self.reap(s);
            r
        })
    }

    fn elapsed(&self, fence: Fence) -> Option<Duration> {
        if fence.0 & HOST != 0 {
            return None;
        }
        let per_ns = self.tick_ns?;
        // CP and SDMA timestamps count the same global clock.
        let (a, b) = self
            .with(|s| {
                Ok(if fence.0 & SDMA != 0 {
                    s.sdma.as_mut().and_then(|e| e.timestamps(fence.0 & !SDMA))
                } else {
                    s.q.timestamps(fence.0)
                })
            })
            .ok()??;
        Some(Duration::from_nanos(((b - a) as f64 / per_ns) as u64))
    }

    fn export_dmabuf(&self, id: BufId) -> Result<std::os::fd::OwnedFd> {
        self.with(|s| {
            let b = s
                .bufs
                .get(&id.0)
                .ok_or_else(|| Error::InvalidArgument("unknown buffer".into()))?;
            self.gpu.export(&b.bo)
        })
    }

    fn import_dmabuf(&self, fd: std::os::fd::BorrowedFd<'_>, bytes: usize) -> Result<BufId> {
        use std::os::fd::AsRawFd;
        let size = crate::dmabuf::size(fd)?;
        if size < bytes {
            return Err(Error::InvalidArgument(format!(
                "dma-buf of {size} bytes for {bytes}"
            )));
        }
        let bo = self.gpu.import(fd.as_raw_fd(), size as u64)?;
        self.with(|s| Ok(Self::insert(s, bo)))
    }
}

impl Drop for AmdBackend {
    fn drop(&mut self) {
        let state = self
            .state
            .get_mut()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .take();
        if let Some(s) = state {
            if let Some(e) = s.sdma {
                e.close(&self.gpu);
            }
            s.q.close(&self.gpu);
            for (_, b) in s.bufs {
                self.gpu.free(b.bo);
            }
            for (_, bo) in s.retire {
                self.gpu.free(bo);
            }
            for (_, d) in s.downloads {
                if let Staged::Device { pieces, .. } = d {
                    for (bo, _) in pieces {
                        self.gpu.free(bo);
                    }
                }
            }
            for bo in s.pool.free {
                self.gpu.free(bo);
            }
        }
        if let Some(code) = self.code.take() {
            self.gpu.free(code);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_built_target_parses_and_needs_no_scratch() {
        for v in [
            90000, 90012, 90008, 90010, 90400, 90402, 90500, 100_100, 100_300, 100_305, 110_000,
            110_003, 110_501, 120_000, 120_001,
        ] {
            let (image, _) = select(v).unwrap();
            let p = Program::parse(image).unwrap();
            assert!(
                p.kernels
                    .iter()
                    .all(|k| k.private_segment == 0 && !k.dynamic_stack),
                "{v}"
            );
            for s in crate::runtime::abi::SYMBOLS {
                assert!(p.kernel(s).is_some(), "{v}: {s}");
            }
        }
        // Each family's waves: 64 on GCN and CDNA, 32 on RDNA.
        let wave = |v| Program::parse(select(v).unwrap().0).unwrap().kernels[0].wavefront;
        assert_eq!(
            (wave(90012), wave(90402), wave(100_100), wave(110_501)),
            (64, 64, 32, 32)
        );
        // gfx8 and older, and what no family covers, are refused.
        assert!(select(80003).is_none());
        assert!(select(100_200).is_none());
    }

    #[test]
    fn staging_sizes_fit_again_without_wasting_half() {
        assert_eq!(staging_size(1), 4096);
        assert_eq!(staging_size(48 << 20), 64 << 20);
        assert_eq!(staging_size((64 << 20) + 1), 128 << 20);
        // 1.25 GiB: 1.25 GiB, not 2.
        assert_eq!(staging_size(5 << 28), 5 << 28);
    }

    #[test]
    fn copy_engines_parse_from_the_environment_value() {
        assert_eq!(CopyEngine::parse("sdma"), Some(CopyEngine::Sdma));
        assert_eq!(CopyEngine::parse("drm\n"), Some(CopyEngine::DriverSdma));
        assert_eq!(CopyEngine::parse("cpu"), Some(CopyEngine::Cpu));
        assert_eq!(CopyEngine::parse("blit"), None);
    }
}
