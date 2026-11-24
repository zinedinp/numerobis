//! The KFD side of one AMD GPU: its topology node, the process VM on its
//! render node, and buffer objects that the CPU and the GPU see at the same
//! virtual address.

use std::collections::HashMap;
use std::fs;
use std::sync::{Mutex, OnceLock};

use super::sys::{self, mem};
use crate::runtime::error::{Error, Result};
use crate::runtime::os::{Fd, Map, RW, View};

const TOPOLOGY: &str = "/sys/class/kfd/kfd/topology/nodes";

/// `ioctl` on a KFD or DRM descriptor; with `NUMEROBIS_AMD_TRACE` set, a
/// refusal is printed with the call's name (the error itself carries only
/// the errno).
///
/// # Safety
///
/// As [`Fd::ioctl`].
pub(crate) unsafe fn call<T>(fd: &Fd, req: u64, arg: *mut T, what: &str) -> Result<()> {
    // SAFETY: forwarded to the caller.
    let r = unsafe { fd.ioctl(req, arg) };
    if let Err(e) = &r {
        trace(what, e);
    }
    r
}

/// Prints `what` failed with `e` when `NUMEROBIS_AMD_TRACE` is set.
pub(crate) fn trace(what: &str, e: &Error) {
    if std::env::var_os("NUMEROBIS_AMD_TRACE").is_some() {
        eprintln!("numerobis amd: {what}: {e}");
    }
}
const PAGE: u64 = 4096;

/// `key value` lines of a topology `properties` file.
#[must_use]
pub fn parse_properties(text: &str) -> HashMap<&str, u64> {
    text.lines()
        .filter_map(|l| {
            let (k, v) = l.split_once(' ')?;
            Some((k, v.trim().parse().ok()?))
        })
        .collect()
}

/// A GPU node of the KFD topology.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Node {
    /// Topology node index.
    pub index: u32,
    /// What every ioctl names the GPU by.
    pub gpu_id: u32,
    /// The ASIC's code name (`yellow_carp`).
    pub name: String,
    /// `major * 10000 + minor * 100 + stepping`: 100305 is gfx1035.
    pub gfx_target_version: u32,
    pub vendor_id: u32,
    pub device_id: u32,
    pub drm_render_minor: u32,
    /// PCI `bus << 8 | devfn`.
    pub location_id: u32,
    pub domain: u32,
    pub simd_count: u32,
    pub simd_per_cu: u32,
    pub max_waves_per_simd: u32,
    pub array_count: u32,
    pub simd_arrays_per_engine: u32,
    pub lds_size_in_kb: u32,
    pub wave_front_size: u32,
    pub num_xcc: u32,
    /// SDMA engines (`num_sdma_engines`, without the XGMI ones).
    pub sdma_engines: u32,
    /// Context save area and control stack sizes, as the kernel checks them
    /// at queue creation (Linux 6.12 and later publish them; else 0).
    pub cwsr_size: u32,
    pub ctl_stack_size: u32,
}

impl Node {
    /// From a `properties` file; `None` for a CPU node or a GPU without
    /// the fields a queue needs.
    #[must_use]
    pub fn from_properties(index: u32, gpu_id: u32, name: &str, text: &str) -> Option<Node> {
        let p = parse_properties(text);
        let get = |k: &str| p.get(k).and_then(|&v| u32::try_from(v).ok());
        let node = Node {
            index,
            gpu_id,
            name: name.trim().to_string(),
            gfx_target_version: get("gfx_target_version")?,
            vendor_id: get("vendor_id")?,
            device_id: get("device_id")?,
            drm_render_minor: get("drm_render_minor")?,
            location_id: get("location_id").unwrap_or(0),
            domain: get("domain").unwrap_or(0),
            simd_count: get("simd_count")?,
            simd_per_cu: get("simd_per_cu")?,
            max_waves_per_simd: get("max_waves_per_simd").unwrap_or(0),
            array_count: get("array_count").unwrap_or(0),
            simd_arrays_per_engine: get("simd_arrays_per_engine").unwrap_or(1),
            lds_size_in_kb: get("lds_size_in_kb").unwrap_or(64),
            wave_front_size: get("wave_front_size").unwrap_or(64),
            num_xcc: get("num_xcc").unwrap_or(1).max(1),
            sdma_engines: get("num_sdma_engines").unwrap_or(0),
            cwsr_size: get("cwsr_size").unwrap_or(0),
            ctl_stack_size: get("ctl_stack_size").unwrap_or(0),
        };
        (gpu_id != 0 && node.simd_count > 0 && node.simd_per_cu > 0).then_some(node)
    }

    /// The ISA name: `gfx1035`, `gfx90a`, `gfx942`.
    #[must_use]
    pub fn gfx(&self) -> String {
        let v = self.gfx_target_version;
        format!("gfx{}{:x}{:x}", v / 10000, (v / 100) % 100, v % 100)
    }

    /// Compute units per XCC.
    #[must_use]
    pub fn cus(&self) -> u32 {
        self.simd_count / self.simd_per_cu / self.num_xcc
    }

    /// The queue buffer sizes, as the kernel computes them
    /// (`kfd_queue_ctx_save_restore_size`); checked against the published
    /// sizes, so a kernel that counts otherwise is noticed.
    ///
    /// # Errors
    ///
    /// [`Error::Unsupported`] for a GPU before gfx8, or when the kernel
    /// publishes sizes this formula doesn't give.
    pub fn queue_sizes(&self) -> Result<QueueSizes> {
        let gfxv = self.gfx_target_version;
        if gfxv < 80001 {
            return Err(Error::Unsupported(format!(
                "{}: no AQL queues before gfx8",
                self.gfx()
            )));
        }
        let cu = self.cus();
        let waves = if gfxv < 100_100 {
            (cu * 40).min(self.array_count / self.simd_arrays_per_engine.max(1) * 512)
        } else {
            cu * 32
        };
        let vgpr = match gfxv {
            90402 | 90010 | 90008 | 90500 => 0x8_0000,
            110_000 | 110_001 | 120_000 | 120_001 => 0x6_0000,
            _ => 0x4_0000,
        };
        let lds = if gfxv == 90500 {
            self.lds_size_in_kb << 10
        } else {
            0x1_0000
        };
        let wg_data = align(u64::from(cu) * u64::from(vgpr + 0x4000 + lds + 0x1000));
        let per_wave = if gfxv >= 100_100 { 12 } else { 8 };
        let mut ctl = align(40 + u64::from(waves) * per_wave + 8);
        if gfxv / 10000 == 10 {
            ctl = ctl.min(0x7000);
        }
        let sizes = QueueSizes {
            ctl_stack: u32::try_from(ctl).map_err(|_| Error::Unsupported("ctl stack".into()))?,
            cwsr: u32::try_from(ctl + wg_data).map_err(|_| Error::Unsupported("cwsr".into()))?,
            debug: (waves * 32).next_multiple_of(64),
            eop: if gfxv == 80002 { 0x8000 } else { 4096 },
            xcc: self.num_xcc,
        };
        let published = (self.cwsr_size, self.ctl_stack_size);
        if published != (0, 0) && published != (sizes.cwsr, sizes.ctl_stack) {
            return Err(Error::Unsupported(format!(
                "{}: the kernel counts the context save area as {} + {} bytes, numerobis as {} + {}",
                self.gfx(),
                self.cwsr_size,
                self.ctl_stack_size,
                sizes.cwsr,
                sizes.ctl_stack
            )));
        }
        Ok(sizes)
    }
}

fn align(n: u64) -> u64 {
    n.next_multiple_of(PAGE)
}

/// Buffer sizes for one compute queue.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct QueueSizes {
    pub ctl_stack: u32,
    /// Control stack plus wave state, per XCC.
    pub cwsr: u32,
    /// Debugger bytes after the save area, per XCC.
    pub debug: u32,
    pub eop: u32,
    pub xcc: u32,
}

impl QueueSizes {
    /// The context save buffer: save area and debugger bytes, per XCC.
    #[must_use]
    pub fn cwsr_buffer(&self) -> u64 {
        align((u64::from(self.cwsr) + u64::from(self.debug)) * u64::from(self.xcc))
    }
}

/// Every GPU node of the KFD topology.
///
/// # Errors
///
/// [`Error::Unsupported`] without KFD (`/sys/class/kfd`).
pub fn nodes() -> Result<Vec<Node>> {
    let dir = fs::read_dir(TOPOLOGY).map_err(|_| Error::Unsupported("no KFD topology".into()))?;
    let mut out = Vec::new();
    for e in dir.flatten() {
        let Some(index) = e.file_name().to_str().and_then(|s| s.parse().ok()) else {
            continue;
        };
        let read = |f: &str| fs::read_to_string(e.path().join(f)).ok();
        let gpu_id = read("gpu_id")
            .and_then(|s| s.trim().parse().ok())
            .unwrap_or(0);
        let (Some(name), Some(props)) = (read("name"), read("properties")) else {
            continue;
        };
        if let Some(n) = Node::from_properties(index, gpu_id, &name, &props) {
            out.push(n);
        }
    }
    out.sort_by_key(|n| n.index);
    Ok(out)
}

/// Where a buffer lives.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Kind {
    /// Device memory the CPU can map (on an APU the carve-out).
    Vram,
    /// Device memory without a CPU mapping.
    VramPrivate,
    /// System memory, uncached and coherent: rings, signals, kernargs,
    /// context save area. Executable, as `ROCr` allocates the AQL ring: without
    /// it the CP never fetches a packet.
    Gtt,
    /// System memory, cached and coherent: the host side of SDMA copies,
    /// which the CPU fills and reads at full speed.
    Staging,
    /// [`Kind::Vram`], executable: the code object.
    Code,
}

impl Kind {
    fn flags(self) -> u32 {
        match self {
            Kind::Vram => mem::VRAM | mem::WRITABLE | mem::PUBLIC,
            Kind::VramPrivate => mem::VRAM | mem::WRITABLE,
            Kind::Gtt => mem::GTT | mem::WRITABLE | mem::COHERENT | mem::UNCACHED | mem::EXECUTABLE,
            Kind::Staging => mem::GTT | mem::WRITABLE | mem::COHERENT,
            Kind::Code => mem::VRAM | mem::WRITABLE | mem::PUBLIC | mem::EXECUTABLE,
        }
    }

    fn cpu(self) -> bool {
        self != Kind::VramPrivate
    }
}

/// A buffer object: GPU and CPU address are the same, [`Bo::va`].
#[derive(Debug)]
pub struct Bo {
    pub handle: u64,
    pub va: u64,
    pub size: u64,
    /// The reservation, and the CPU mapping over it when there is one.
    map: Map,
    cpu: bool,
}

impl Bo {
    /// The CPU mapping; `None` for [`Kind::VramPrivate`].
    #[must_use]
    pub(crate) fn cpu(&self) -> Option<View> {
        self.cpu.then_some(*self.map)
    }
}

/// `/dev/kfd`, opened once: KFD keeps one process per address space, however
/// often it is opened.
fn kfd_fd() -> Result<&'static Fd> {
    static KFD: OnceLock<Fd> = OnceLock::new();
    if let Some(fd) = KFD.get() {
        return Ok(fd);
    }
    let fd = Fd::open("/dev/kfd")?;
    Ok(KFD.get_or_init(|| fd))
}

/// The render node whose VM the process uses on `node`'s GPU. KFD binds the
/// GPU VM to the first render-node file it is handed and refuses any other
/// (`EBUSY`), so that file stays open for the process, and a GPU opened
/// again later reuses it.
fn vm(kfd: &Fd, node: &Node) -> Result<&'static Fd> {
    static VMS: Mutex<Vec<(u32, &'static Fd)>> = Mutex::new(Vec::new());
    let mut vms = VMS
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    if let Some(&(_, fd)) = vms.iter().find(|(id, _)| *id == node.gpu_id) {
        return Ok(fd);
    }
    let drm = Fd::open(&format!("/dev/dri/renderD{}", node.drm_render_minor))?;
    let mut acquire = sys::AcquireVm {
        drm_fd: u32::try_from(drm.0).map_err(|_| Error::Os(libc::EBADF))?,
        gpu_id: node.gpu_id,
    };
    // SAFETY: matches AMDKFD_IOC_ACQUIRE_VM; the render node stays open.
    unsafe { call(kfd, sys::ACQUIRE_VM, &raw mut acquire, "ACQUIRE_VM") }?;
    let drm: &'static Fd = Box::leak(Box::new(drm));
    vms.push((node.gpu_id, drm));
    Ok(drm)
}

/// `/dev/kfd` and the render node of one GPU, with the process VM acquired.
pub struct Gpu {
    pub node: Node,
    pub version: (u32, u32),
    pub(crate) kfd: &'static Fd,
    drm: &'static Fd,
    gpuvm: (u64, u64),
}

impl Gpu {
    /// Opens the GPU of topology node `node`.
    ///
    /// # Errors
    ///
    /// [`Error::Os`] when `/dev/kfd` or the render node don't open or the
    /// VM can't be acquired, [`Error::DriverTooOld`] for KFD before 1.14.
    pub fn open(node: Node) -> Result<Gpu> {
        let kfd = kfd_fd()?;
        let mut v = sys::GetVersion::default();
        // SAFETY: matches AMDKFD_IOC_GET_VERSION.
        unsafe { call(kfd, sys::GET_VERSION, &raw mut v, "GET_VERSION") }?;
        if v.major != 1 || v.minor < 14 {
            return Err(Error::DriverTooOld {
                have: format!("KFD {}.{}", v.major, v.minor),
                need: "KFD 1.14 (Linux 6.6)".into(),
            });
        }
        let drm = vm(kfd, &node)?;
        let gpuvm = Self::gpuvm(kfd, node.gpu_id)?;
        Ok(Gpu {
            node,
            version: (v.major, v.minor),
            kfd,
            drm,
            gpuvm,
        })
    }

    /// The GPU VM aperture of `gpu_id` in this process.
    fn gpuvm(kfd: &Fd, gpu_id: u32) -> Result<(u64, u64)> {
        // First the count (KFD refuses a buffer of another size), then the entries.
        let mut args = sys::GetAperturesNew::default();
        // SAFETY: a zero count only reads the number of devices.
        unsafe {
            call(
                kfd,
                sys::GET_PROCESS_APERTURES_NEW,
                &raw mut args,
                "GET_PROCESS_APERTURES_NEW",
            )
        }?;
        let mut ap = vec![sys::DeviceApertures::default(); args.num_of_nodes as usize];
        let mut args = sys::GetAperturesNew {
            apertures_ptr: ap.as_mut_ptr() as u64,
            num_of_nodes: ap.len() as u32,
            pad: 0,
        };
        // SAFETY: `ap` holds `num_of_nodes` entries.
        unsafe {
            call(
                kfd,
                sys::GET_PROCESS_APERTURES_NEW,
                &raw mut args,
                "GET_PROCESS_APERTURES_NEW",
            )
        }?;
        ap.iter()
            .take(args.num_of_nodes as usize)
            .find(|a| a.gpu_id == gpu_id)
            .map(|a| (a.gpuvm_base, a.gpuvm_limit))
            .ok_or_else(|| Error::Unsupported(format!("no aperture for GPU {gpu_id}")))
    }

    /// Reserves address space for `size` bytes inside the GPU VM aperture.
    fn reserve(&self, size: u64) -> Result<Map> {
        let len = usize::try_from(size).map_err(|_| Error::OutOfMemory {
            requested: usize::MAX,
        })?;
        let map = Map::new(
            0,
            len,
            libc::PROT_NONE,
            libc::MAP_PRIVATE | libc::MAP_ANONYMOUS | libc::MAP_NORESERVE,
            -1,
            0,
        )?;
        let va = map.addr() as u64;
        if va < self.gpuvm.0 || va + size > self.gpuvm.1 {
            return Err(Error::Unsupported(format!(
                "address {va:#x} outside the GPU VM {:#x}..{:#x}",
                self.gpuvm.0, self.gpuvm.1
            )));
        }
        Ok(map)
    }

    fn map_to_gpu(&self, handle: u64) -> Result<()> {
        let ids = [self.node.gpu_id];
        let mut m = sys::MapMemory {
            handle,
            device_ids_array_ptr: ids.as_ptr() as u64,
            n_devices: 1,
            n_success: 0,
        };
        // SAFETY: `ids` holds one gpu id.
        unsafe {
            call(
                self.kfd,
                sys::MAP_MEMORY_TO_GPU,
                &raw mut m,
                "MAP_MEMORY_TO_GPU",
            )
        }
    }

    fn release(&self, handle: u64) {
        let ids = [self.node.gpu_id];
        let mut m = sys::MapMemory {
            handle,
            device_ids_array_ptr: ids.as_ptr() as u64,
            n_devices: 1,
            n_success: 0,
        };
        // SAFETY: as in map_to_gpu; failures leave the handle to the process exit.
        let _ = unsafe {
            call(
                self.kfd,
                sys::UNMAP_MEMORY_FROM_GPU,
                &raw mut m,
                "UNMAP_MEMORY_FROM_GPU",
            )
        };
        let mut f = sys::FreeMemory { handle };
        // SAFETY: matches AMDKFD_IOC_FREE_MEMORY_OF_GPU.
        let _ = unsafe {
            call(
                self.kfd,
                sys::FREE_MEMORY_OF_GPU,
                &raw mut f,
                "FREE_MEMORY_OF_GPU",
            )
        };
    }

    /// Allocates `size` bytes (rounded up to pages) of `kind`, mapped for
    /// the GPU and, unless [`Kind::VramPrivate`], for the CPU at the same
    /// address.
    ///
    /// # Errors
    ///
    /// [`Error::OutOfMemory`] when KFD has no room, [`Error::Os`] for other
    /// refusals.
    pub fn alloc(&self, size: u64, kind: Kind) -> Result<Bo> {
        let size = align(size.max(1));
        let map = self.reserve(size)?;
        let mut a = sys::AllocMemory {
            va_addr: map.addr() as u64,
            size,
            gpu_id: self.node.gpu_id,
            flags: kind.flags(),
            ..Default::default()
        };
        // SAFETY: matches AMDKFD_IOC_ALLOC_MEMORY_OF_GPU.
        match unsafe {
            call(
                self.kfd,
                sys::ALLOC_MEMORY_OF_GPU,
                &raw mut a,
                "ALLOC_MEMORY_OF_GPU",
            )
        } {
            Err(Error::Os(libc::ENOMEM)) => {
                return Err(Error::OutOfMemory {
                    requested: usize::try_from(size).unwrap_or(usize::MAX),
                });
            }
            r => r?,
        }
        let bo = |map| Bo {
            handle: a.handle,
            va: a.va_addr,
            size,
            map,
            cpu: kind.cpu(),
        };
        let mut map = map;
        if kind.cpu() {
            match Map::new(
                map.addr(),
                map.len,
                RW,
                libc::MAP_SHARED | libc::MAP_FIXED,
                self.drm.0,
                a.mmap_offset,
            ) {
                // The fixed mapping replaced the reservation's pages: one
                // range, unmapped once by the new map.
                Ok(cpu) => {
                    map.forget();
                    map = cpu;
                }
                Err(e) => {
                    trace(
                        &format!(
                            "mmap of {kind:?} at {:#x} (offset {:#x})",
                            a.va_addr, a.mmap_offset
                        ),
                        &e,
                    );
                    self.release(a.handle);
                    return Err(e);
                }
            }
        }
        if let Err(e) = self.map_to_gpu(a.handle) {
            self.release(a.handle);
            return Err(e);
        }
        Ok(bo(map))
    }

    /// Unmaps and frees `bo`. The GPU must be done with it.
    pub fn free(&self, bo: Bo) {
        self.release(bo.handle);
        drop(bo);
    }

    /// Imports a dma-buf of `size` bytes, mapped for the GPU and, through
    /// the dma-buf, for the CPU at the same address.
    ///
    /// # Errors
    ///
    /// [`Error::Os`] when KFD refuses the dma-buf (not importable on this
    /// GPU) or it can't be mapped.
    pub fn import(&self, fd: i32, size: u64) -> Result<Bo> {
        let size = align(size.max(1));
        let map = self.reserve(size)?;
        let mut a = sys::ImportDmabuf {
            va_addr: map.addr() as u64,
            gpu_id: self.node.gpu_id,
            dmabuf_fd: u32::try_from(fd).map_err(|_| Error::Os(libc::EBADF))?,
            ..Default::default()
        };
        // SAFETY: matches AMDKFD_IOC_IMPORT_DMABUF.
        unsafe { call(self.kfd, sys::IMPORT_DMABUF, &raw mut a, "IMPORT_DMABUF") }?;
        if let Err(e) = self.map_to_gpu(a.handle) {
            self.release(a.handle);
            return Err(e);
        }
        let (map, cpu) = match Map::new(
            map.addr(),
            map.len,
            RW,
            libc::MAP_SHARED | libc::MAP_FIXED,
            fd,
            0,
        ) {
            Ok(cpu) => {
                map.forget();
                (cpu, true)
            }
            // GPU-only: uploads and downloads then refuse.
            Err(_) => (map, false),
        };
        Ok(Bo {
            handle: a.handle,
            va: a.va_addr,
            size,
            map,
            cpu,
        })
    }

    /// `bo` as a dma-buf.
    ///
    /// # Errors
    ///
    /// [`Error::Os`] when KFD refuses the export.
    pub fn export(&self, bo: &Bo) -> Result<std::os::fd::OwnedFd> {
        use std::os::fd::FromRawFd;
        let mut e = sys::ExportDmabuf {
            handle: bo.handle,
            flags: (libc::O_CLOEXEC | libc::O_RDWR) as u32,
            dmabuf_fd: 0,
        };
        // SAFETY: matches AMDKFD_IOC_EXPORT_DMABUF.
        unsafe { call(self.kfd, sys::EXPORT_DMABUF, &raw mut e, "EXPORT_DMABUF") }?;
        // SAFETY: KFD returned a new descriptor we own.
        Ok(unsafe { std::os::fd::OwnedFd::from_raw_fd(e.dmabuf_fd as i32) })
    }

    /// The render node whose VM KFD uses (for `DRM_AMDGPU_CS`).
    pub(crate) fn drm(&self) -> &'static Fd {
        self.drm
    }

    /// GPU timestamp ticks per second, from two clock-counter reads.
    #[must_use]
    pub fn timestamp_frequency(&self) -> Option<f64> {
        let read = || {
            let mut c = sys::ClockCounters {
                gpu_id: self.node.gpu_id,
                ..Default::default()
            };
            // SAFETY: matches AMDKFD_IOC_GET_CLOCK_COUNTERS.
            unsafe {
                call(
                    self.kfd,
                    sys::GET_CLOCK_COUNTERS,
                    &raw mut c,
                    "GET_CLOCK_COUNTERS",
                )
            }
            .ok()?;
            Some(c)
        };
        let a = read()?;
        std::thread::sleep(std::time::Duration::from_millis(20));
        let b = read()?;
        let seconds = (b.system_clock_counter.checked_sub(a.system_clock_counter)?) as f64
            / a.system_clock_freq.max(1) as f64;
        let ticks = b.gpu_clock_counter.checked_sub(a.gpu_clock_counter)? as f64;
        (seconds > 0.0 && ticks > 0.0).then(|| ticks / seconds)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The Radeon 680M's node as Linux 7.3 publishes it.
    const REMBRANDT: &str = "cpu_cores_count 0\nsimd_count 24\nmem_banks_count 1\n\
        max_waves_per_simd 16\nlds_size_in_kb 64\nwave_front_size 32\narray_count 2\n\
        simd_arrays_per_engine 2\ncu_per_simd_array 6\nsimd_per_cu 2\n\
        gfx_target_version 100305\nvendor_id 4098\ndevice_id 5761\nlocation_id 1024\n\
        domain 0\ndrm_render_minor 128\nnum_xcc 1\ncwsr_size 4186112\nctl_stack_size 8192\n";

    #[test]
    fn a_gpu_node_reads_with_its_queue_sizes() {
        let n = Node::from_properties(1, 43337, "yellow_carp\n", REMBRANDT).unwrap();
        assert_eq!(
            (n.gfx(), n.cus(), n.name.as_str()),
            ("gfx1035".into(), 12, "yellow_carp")
        );
        let q = n.queue_sizes().unwrap();
        assert_eq!(
            (q.cwsr, q.ctl_stack, q.debug, q.eop),
            (4_186_112, 8192, 12288, 4096)
        );
        assert_eq!(q.cwsr_buffer(), 4_198_400);
        // A kernel that counts otherwise is refused, not guessed at.
        let other = Node {
            cwsr_size: 4_190_208,
            ..n.clone()
        };
        assert!(other.queue_sizes().is_err());
        // Before 6.12 the sizes aren't published: the formula alone.
        let old = Node {
            cwsr_size: 0,
            ctl_stack_size: 0,
            ..n
        };
        assert_eq!(old.queue_sizes().unwrap(), q);
    }

    #[test]
    fn cpu_nodes_and_other_isas_are_named() {
        let cpu = "cpu_cores_count 16\nsimd_count 0\ngfx_target_version 0\nvendor_id 0\n\
            device_id 0\ndrm_render_minor 0\nsimd_per_cu 0\n";
        assert!(Node::from_properties(0, 0, "", cpu).is_none());
        let mi = Node {
            gfx_target_version: 90010,
            ..Node::from_properties(1, 1, "", REMBRANDT).unwrap()
        };
        assert_eq!(mi.gfx(), "gfx90a");
        let navi31 = Node {
            gfx_target_version: 110_000,
            ..mi
        };
        assert_eq!(navi31.gfx(), "gfx1100");
    }
}
