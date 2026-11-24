//! `Backend` over KFD: buffers the CPU and GPU share an address for, one
//! AQL compute queue, and the shared Emulated kernels from
//! `kernels/bin/amd`.
//!
//! Buffers live in CPU-visible device memory (falling back to system memory
//! when that is full, as on a discrete card without resizable BAR), so
//! uploads and downloads are CPU copies after the queue has drained; there
//! is no SDMA queue yet.

use std::collections::HashMap;
use std::sync::{Arc, Mutex, MutexGuard, Weak};
use std::time::Duration;

use super::Tier as AmdTier;
use super::kfd::{self, Bo, Gpu, Kind, Node};
use super::program::{Program, Word};
use super::queue::{Launch, Queue};
use crate::engine::{Access, DeviceClass, Engine, FeatureSet};
use crate::pci;
use crate::platform::{Status, Support};
use crate::runtime::Device;
use crate::runtime::abi::lower;
use crate::runtime::backend::{Arg, Backend, BufId, Fence, KernelId, KernelKey, OpSig};
use crate::runtime::error::{Error, Result};
use crate::tiers::Tier;

/// Run on an AMD Radeon 680M (gfx1035, RDNA 2 APU, Linux 7.3, KFD 1.18 and
/// later); the other targets are built and load-checked only.
pub const SUPPORT: Support = Support::new(Status::Tested, Status::Unavailable);

static RDNA2: &[u8] = include_bytes!("../../kernels/bin/amd/gfx10-3-generic-emulated.hsaco");
static GFX1100: &[u8] = include_bytes!("../../kernels/bin/amd/gfx1100-emulated.hsaco");
static GFX1200: &[u8] = include_bytes!("../../kernels/bin/amd/gfx1200-emulated.hsaco");
static GFX908: &[u8] = include_bytes!("../../kernels/bin/amd/gfx908-emulated.hsaco");
static GFX90A: &[u8] = include_bytes!("../../kernels/bin/amd/gfx90a-emulated.hsaco");
static GFX942: &[u8] = include_bytes!("../../kernels/bin/amd/gfx942-emulated.hsaco");
static GFX950: &[u8] = include_bytes!("../../kernels/bin/amd/gfx950-emulated.hsaco");

/// Work-group size of every launch (the shared kernels allow 256).
const GROUP: u16 = 256;
/// Download fences live apart from queue fences: they are done when made.
const DOWNLOAD: u64 = 1 << 63;

/// The code object and matrix tier for a `gfx_target_version`; `None`
/// where no binary is built for it.
#[must_use]
pub fn select(gfx_target_version: u32) -> Option<(&'static [u8], Option<AmdTier>)> {
    Some(match gfx_target_version {
        100_300..=100_306 => (RDNA2, None),
        110_000 => (GFX1100, Some(AmdTier::Gfx11)),
        120_000 | 120_001 => (GFX1200, Some(AmdTier::Gfx12)),
        90008 => (GFX908, Some(AmdTier::Gfx908)),
        90010 => (GFX90A, Some(AmdTier::Gfx90a)),
        90402 => (GFX942, Some(AmdTier::Gfx942)),
        90500 => (GFX950, Some(AmdTier::Gfx950)),
        _ => return None,
    })
}

struct State {
    q: Queue,
    next: u64,
    /// Allocated and imported alike: KFD frees an import's handle only.
    bufs: HashMap<u64, Bo>,
    kernels: Vec<OpSig>,
    downloads: HashMap<u64, Vec<u8>>,
}

/// One AMD GPU on the raw KFD path.
pub struct AmdBackend {
    features: FeatureSet,
    program: Program,
    code: Option<Bo>,
    /// GPU timestamp ticks per nanosecond.
    tick_ns: Option<f64>,
    state: Mutex<Option<State>>,
    gpu: Gpu,
}

/// Opens every GPU node KFD lists; errors are kept for diagnostics
/// (`examples/probe.rs`).
#[must_use]
pub fn probe() -> Vec<Result<Device>> {
    let Ok(nodes) = kfd::nodes() else {
        return Vec::new();
    };
    nodes
        .into_iter()
        .map(|node| {
            // One backend per GPU per process: it owns the process's queue
            // and code upload on that GPU.
            static OPEN: Mutex<Vec<(u32, Weak<AmdBackend>)>> = Mutex::new(Vec::new());
            let mut open = OPEN
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            open.retain(|(_, w)| w.strong_count() > 0);
            if let Some(b) = open
                .iter()
                .find(|(id, _)| *id == node.gpu_id)
                .and_then(|(_, w)| w.upgrade())
            {
                return Ok(Device::from_backend(b));
            }
            let id = node.gpu_id;
            let b = Arc::new(AmdBackend::open(node)?);
            open.push((id, Arc::downgrade(&b)));
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

impl AmdBackend {
    /// Opens the GPU of topology node `node` and uploads its kernels.
    ///
    /// # Errors
    ///
    /// [`Error::Unsupported`] for a GPU without a built code object,
    /// [`Error::BadBinary`] for kernels that need scratch, errors of KFD
    /// pass through.
    pub fn open(node: Node) -> Result<AmdBackend> {
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
        let features = FeatureSet {
            name: name(&node),
            class: DeviceClass::Gpu,
            driver: String::new(),
            engines: if tier.is_some() {
                vec![(Engine::Matrix, Access::Raw), (Engine::Vector, Access::Raw)]
            } else {
                vec![(Engine::Vector, Access::Raw)]
            },
            tiers: std::iter::once(Tier::Emulated)
                .chain(tier.map(Tier::Amd))
                .collect(),
        };
        let gpu = Gpu::open(node)?;
        let code = gpu.alloc(program.image.len() as u64, Kind::Code)?;
        // Code memory is CPU-mapped (`Kind::Code`).
        if let Some(cpu) = code.cpu() {
            cpu.write_bytes(0, &program.image);
        }
        let q = match Queue::new(&gpu) {
            Ok(q) => q,
            Err(e) => {
                gpu.free(code);
                return Err(e);
            }
        };
        // A packet that runs nothing: does the queue run at all?
        let mut q = q;
        let ran = q.barrier().and_then(|f| q.wait(f, Duration::from_secs(2)));
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
        let (major, minor) = gpu.version;
        let tick_ns = gpu.timestamp_frequency().map(|hz| hz / 1e9);
        Ok(AmdBackend {
            features: FeatureSet {
                driver: format!("amdgpu KFD {major}.{minor} (raw AQL)"),
                ..features
            },
            program,
            code: Some(code),
            tick_ns,
            state: Mutex::new(Some(State {
                q,
                next: 0,
                bufs: HashMap::new(),
                kernels: Vec::new(),
                downloads: HashMap::new(),
            })),
            gpu,
        })
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
        s.bufs.insert(s.next, bo);
        BufId(s.next)
    }

    /// The CPU view of `id` over `offset..offset + len`, after the queue
    /// has drained.
    fn host_range(
        s: &mut State,
        id: BufId,
        offset: usize,
        len: usize,
    ) -> Result<crate::runtime::os::View> {
        let last = s.q.last;
        s.q.wait(last, Duration::from_secs(60))?;
        let b = s
            .bufs
            .get(&id.0)
            .ok_or_else(|| Error::InvalidArgument("unknown buffer".into()))?;
        if offset
            .checked_add(len)
            .is_none_or(|end| end as u64 > b.size)
        {
            return Err(Error::InvalidArgument("copy out of bounds".into()));
        }
        b.cpu()
            .ok_or_else(|| Error::Unsupported("buffer without a CPU mapping".into()))
    }
}

impl Backend for AmdBackend {
    fn features(&self) -> &FeatureSet {
        &self.features
    }

    fn alloc(&self, bytes: usize) -> Result<BufId> {
        let size = bytes as u64;
        let bo = match self.gpu.alloc(size, Kind::Vram) {
            Err(Error::OutOfMemory { .. }) => self.gpu.alloc(size, Kind::Gtt)?,
            r => r?,
        };
        self.with(|s| Ok(Self::insert(s, bo)))
    }

    fn free(&self, id: BufId) {
        let b = self.lock().as_mut().and_then(|s| s.bufs.remove(&id.0));
        if let Some(b) = b {
            self.gpu.free(b);
        }
    }

    fn upload(&self, id: BufId, offset: usize, src: &[u8]) -> Result<Fence> {
        self.with(|s| {
            let v = Self::host_range(s, id, offset, src.len())?;
            v.write_bytes(offset, src);
            Ok(Fence(s.q.last))
        })
    }

    fn download(&self, id: BufId, offset: usize, len: usize) -> Result<Fence> {
        self.with(|s| {
            let v = Self::host_range(s, id, offset, len)?;
            let mut out = vec![0; len];
            v.read_bytes(offset, &mut out);
            s.next += 1;
            let f = DOWNLOAD | s.next;
            s.downloads.insert(f, out);
            Ok(Fence(f))
        })
    }

    fn take_download(&self, fence: Fence, dst: &mut [u8]) -> Result<()> {
        let staged = self
            .with(|s| Ok(s.downloads.remove(&fence.0)))?
            .ok_or_else(|| Error::InvalidArgument("unknown download".into()))?;
        if staged.len() != dst.len() {
            return Err(Error::InvalidArgument("download size mismatch".into()));
        }
        dst.copy_from_slice(&staged);
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
        self.with(|s| {
            s.kernels.push(key.sig);
            Ok(KernelId(s.kernels.len() as u64 - 1))
        })
    }

    fn dispatch(&self, kernel: KernelId, args: &[Arg]) -> Result<Fence> {
        let code = self.code.as_ref().ok_or(Error::DeviceLost)?.va;
        self.with(|s| {
            let sig = *s
                .kernels
                .get(usize::try_from(kernel.0).unwrap_or(usize::MAX))
                .ok_or_else(|| Error::InvalidArgument("unknown kernel".into()))?;
            let launches = lower(sig, args)?;
            // Arguments are all checked before anything is queued.
            let mut prepared = Vec::with_capacity(launches.len());
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
                        Arg::Buf(b) => Word::Ptr(
                            s.bufs
                                .get(&b.0)
                                .ok_or_else(|| Error::InvalidArgument("unknown buffer".into()))?
                                .va,
                        ),
                        Arg::U32(v) => Word::B32(v),
                        Arg::I32(v) => Word::B32(v.cast_unsigned()),
                        Arg::F32(v) => Word::B32(v.to_bits()),
                    });
                }
                let kernargs = k.kernargs(&words, grid, group)?;
                prepared.push((code + k.descriptor, kernargs, grid, group, k.group_segment));
            }
            let mut last = s.q.last;
            for (descriptor, kernargs, grid, group, lds) in &prepared {
                last = s.q.dispatch(&Launch {
                    descriptor: *descriptor,
                    kernargs,
                    grid: *grid,
                    group: *group,
                    lds: *lds,
                })?;
            }
            Ok(Fence(last))
        })
    }

    fn wait(&self, fence: Fence, timeout: Duration) -> Result<()> {
        if fence.0 & DOWNLOAD != 0 {
            return Ok(());
        }
        self.with(|s| s.q.wait(fence.0, timeout))
    }

    fn elapsed(&self, fence: Fence) -> Option<Duration> {
        let per_ns = self.tick_ns?;
        let (a, b) = self.with(|s| Ok(s.q.timestamps(fence.0))).ok()??;
        Some(Duration::from_nanos(((b - a) as f64 / per_ns) as u64))
    }

    fn export_dmabuf(&self, id: BufId) -> Result<std::os::fd::OwnedFd> {
        self.with(|s| {
            let b = s
                .bufs
                .get(&id.0)
                .ok_or_else(|| Error::InvalidArgument("unknown buffer".into()))?;
            self.gpu.export(b)
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
            s.q.close(&self.gpu);
            for (_, b) in s.bufs {
                self.gpu.free(b);
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
            100_300, 100_305, 110_000, 120_000, 90008, 90010, 90402, 90500,
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
        // gfx1101..gfx1103 differ from gfx1100 in register files: no guess.
        assert!(select(110_003).is_none());
        assert!(select(100_100).is_none());
    }
}
