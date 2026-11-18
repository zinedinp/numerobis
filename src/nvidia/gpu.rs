//! One opened GPU: RM client/device/subdevice, the UVM-managed GPU virtual
//! address space and memory allocations (VRAM and pinned system memory).
//!
//! GPU virtual addresses equal CPU virtual addresses: numerobis reserves one
//! CPU range (`PROT_NONE`) and places every allocation inside it. System
//! memory is mapped there for both sides, VRAM only for the GPU (the GPFIFO
//! area additionally through BAR1). UVM maps the RM memory handles at those
//! addresses (`UVM_CREATE_EXTERNAL_RANGE` + `UVM_MAP_EXTERNAL_ALLOCATION`).

use std::collections::BTreeMap;
use std::sync::Mutex;

use super::sys::*;
use crate::runtime::error::{Error, Result};
use crate::runtime::os::{Fd, Map, RW, View};

/// Candidate places for the address reservation: below 1 TiB, because
/// Ampere/Ada channel resources must sit under `max_channel_va` (2^40), and
/// well clear of the shared/local memory windows at `0xfe000000`/`0xff000000`.
const VA_BASES: [usize; 4] = [
    0x40_0000_0000,
    0x60_0000_0000,
    0x80_0000_0000,
    0xa0_0000_0000,
];
const VA_SIZE: usize = 64 << 30;
const VA_ALIGN: u64 = 2 << 20;

/// `/dev/nvidia-uvm` plus the secondary fd that keeps the mm alive.
struct Uvm {
    fd: Fd,
    _mm: Fd,
}

/// First-fit allocator over the reserved range, 2 MiB granules.
struct VaSpace {
    free: BTreeMap<u64, u64>,
}

impl VaSpace {
    fn alloc(&mut self, size: u64) -> Option<u64> {
        let size = size.next_multiple_of(VA_ALIGN);
        let (&start, &len) = self.free.iter().find(|&(_, &len)| len >= size)?;
        self.free.remove(&start);
        if len > size {
            self.free.insert(start + size, len - size);
        }
        Some(start)
    }

    fn release(&mut self, start: u64, size: u64) {
        let mut start = start;
        let mut size = size.next_multiple_of(VA_ALIGN);
        // Merge with neighbours.
        if let Some((&s, &l)) = self.free.range(..start).next_back()
            && s + l == start
        {
            self.free.remove(&s);
            start = s;
            size += l;
        }
        if let Some(&l) = self.free.get(&(start + size)) {
            self.free.remove(&(start + size));
            size += l;
        }
        self.free.insert(start, size);
    }
}

/// Memory kinds.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Mem {
    /// Device memory, GPU-only.
    Vram,
    /// Device memory, physically contiguous, also mapped for the CPU through BAR1.
    VramMapped,
    /// Pinned host memory, mapped for both sides at the same address.
    Sys,
}

/// One allocation (freed through [`Gpu::free`]).
#[derive(Debug)]
pub struct Alloc {
    pub va: u64,
    pub size: u64,
    pub kind: Mem,
    handle: NvHandle,
}

impl Alloc {
    pub fn handle(&self) -> NvHandle {
        self.handle
    }

    /// CPU view of a [`Mem::Sys`] / [`Mem::VramMapped`] allocation.
    pub fn cpu(&self) -> View {
        assert_ne!(self.kind, Mem::Vram, "VRAM allocation has no CPU mapping");
        View {
            ptr: self.va as *mut u8,
            len: self.size as usize,
        }
    }
}

/// GR engine geometry (for local memory sizing).
#[derive(Debug, Clone, Copy, Default)]
pub struct GrInfo {
    pub sm_version: u32,
    pub max_warps_per_sm: u32,
    pub gpcs: u32,
    pub tpc_per_gpc: u32,
    pub sm_per_tpc: u32,
}

/// An opened GPU.
pub struct Gpu {
    pub ctl: Fd,
    uvm: Uvm,
    /// `/dev/nvidiaN`, registered with `ctl`.
    dev_fd: Fd,
    pub minor: u32,
    pub root: NvHandle,
    pub device: NvHandle,
    pub vaspace: NvHandle,
    pub uuid: [u8; 16],
    pub name: String,
    pub arch: NV2080_CTRL_MC_GET_ARCH_INFO_PARAMS,
    pub gr: GrInfo,
    /// Usermode page (doorbell).
    pub usermode: Map,
    _reservation: Map,
    va: Mutex<VaSpace>,
    /// Client-chosen handles (OS descriptors take `hObjectNew` as input).
    next_handle: std::sync::atomic::AtomicU32,
}

// SAFETY: the raw pointers inside `Map` refer to process-wide mappings; all
// mutable state is behind `Mutex`.
unsafe impl Send for Gpu {}
unsafe impl Sync for Gpu {}

fn open_dev(minor: u32, ctl: &Fd) -> Result<Fd> {
    let fd = Fd::open(&format!("/dev/nvidia{minor}"))?;
    let mut r = nv_ioctl_register_fd_t { ctl_fd: ctl.0 };
    // SAFETY: NV_ESC_REGISTER_FD takes nv_ioctl_register_fd_t.
    unsafe { rm_escape(&fd, NV_ESC_REGISTER_FD, &mut r)? };
    Ok(fd)
}

/// Maps an RM object for the CPU (`NV_ESC_RM_MAP_MEMORY` on a fresh
/// `/dev/nvidiaN` fd, then `mmap`), optionally at the fixed address `at`.
#[allow(clippy::too_many_arguments)]
fn map_object(
    ctl: &Fd,
    minor: u32,
    root: NvHandle,
    device: NvHandle,
    handle: NvHandle,
    len: u64,
    flags: u32,
    at: Option<u64>,
) -> Result<Map> {
    let fd = open_dev(minor, ctl)?;
    let mut p = nv_ioctl_nvos33_parameters_with_fd {
        params: NVOS33_PARAMETERS {
            hClient: root,
            hDevice: device,
            hMemory: handle,
            length: len,
            flags,
            ..Default::default()
        },
        fd: fd.0,
        _pad: 0,
    };
    // SAFETY: NV_ESC_RM_MAP_MEMORY takes nv_ioctl_nvos33_parameters_with_fd.
    unsafe { rm_escape(ctl, NV_ESC_RM_MAP_MEMORY, &mut p)? };
    status("map memory", p.params.status)?;
    let fixed = if at.is_some() { libc::MAP_FIXED } else { 0 };
    // The fd may be closed after mmap; the mapping keeps the context alive.
    Map::new(
        at.unwrap_or(0) as usize,
        len as usize,
        RW,
        libc::MAP_SHARED | fixed,
        fd.0,
        0,
    )
}

/// GPUs known to the driver: `(minor, gpu_id, pci device id)`.
pub fn cards(ctl: &Fd) -> Result<Vec<(u32, u32, u16)>> {
    let mut info = [nv_ioctl_card_info_t::default(); 32];
    // SAFETY: NV_ESC_CARD_INFO takes an array of nv_ioctl_card_info_t; the
    // request size encodes the array length.
    unsafe {
        ctl.ioctl(
            crate::runtime::os::iowr::<[nv_ioctl_card_info_t; 32]>(
                NV_IOCTL_MAGIC,
                NV_ESC_CARD_INFO,
            ),
            &mut info,
        )?
    };
    Ok(info
        .iter()
        .filter(|c| c.valid != 0)
        .map(|c| (c.minor_number, c.gpu_id, c.pci_info.device_id))
        .collect())
}

impl Gpu {
    /// Opens the GPU with RM id `gpu_id` and `/dev/nvidia<minor>`.
    pub fn open(ctl: Fd, minor: u32, gpu_id: u32) -> Result<Gpu> {
        let root = rm_alloc::<()>(&ctl, 0, 0, NV01_ROOT_CLIENT, None)?;

        let uvm_fd = Fd::open("/dev/nvidia-uvm")?;
        let mut p = UVM_INITIALIZE_PARAMS::default();
        // SAFETY: UVM_INITIALIZE params.
        unsafe { uvm(&uvm_fd, UVM_INITIALIZE, &mut p, |p| p.rmStatus)? };
        let mm = Fd::open("/dev/nvidia-uvm")?;
        let mut p = UVM_MM_INITIALIZE_PARAMS {
            uvmFd: uvm_fd.0,
            rmStatus: 0,
        };
        // SAFETY: UVM_MM_INITIALIZE params. NV_WARN_NOTHING_TO_DO is fine (platform without it).
        let _ = unsafe {
            uvm(&mm, UVM_MM_INITIALIZE, &mut p, |p| {
                if p.rmStatus == NV_WARN_NOTHING_TO_DO {
                    NV_OK
                } else {
                    p.rmStatus
                }
            })
        };

        let dev_fd = open_dev(minor, &ctl)?;

        let mut id = NV0000_CTRL_GPU_GET_ID_INFO_V2_PARAMS {
            gpuId: gpu_id,
            ..Default::default()
        };
        rm_control(
            &ctl,
            root,
            root,
            NV0000_CTRL_CMD_GPU_GET_ID_INFO_V2,
            &mut id,
        )?;

        let mut dp = NV0080_ALLOC_PARAMETERS {
            deviceId: id.deviceInstance,
            hClientShare: root,
            vaMode: NV_DEVICE_ALLOCATION_VAMODE_MULTIPLE_VASPACES,
            ..Default::default()
        };
        let device = rm_alloc(&ctl, root, root, NV01_DEVICE_0, Some(&mut dp))?;
        let mut sp = NV2080_ALLOC_PARAMETERS::default();
        let subdevice = rm_alloc(&ctl, root, device, NV20_SUBDEVICE_0, Some(&mut sp))?;

        let mut arch = NV2080_CTRL_MC_GET_ARCH_INFO_PARAMS::default();
        rm_control(
            &ctl,
            root,
            subdevice,
            NV2080_CTRL_CMD_MC_GET_ARCH_INFO,
            &mut arch,
        )?;
        let mut np = NV2080_CTRL_GPU_GET_NAME_STRING_PARAMS {
            gpuNameStringFlags: 0,
            ascii: [0; 64],
        };
        rm_control(
            &ctl,
            root,
            subdevice,
            NV2080_CTRL_CMD_GPU_GET_NAME_STRING,
            &mut np,
        )?;
        let name = String::from_utf8_lossy(
            &np.ascii[..np.ascii.iter().position(|&c| c == 0).unwrap_or(64)],
        )
        .trim()
        .to_string();

        let mut list = [
            NV2080_CTRL_GR_INFO {
                index: GR_INFO_SM_VERSION,
                data: 0,
            },
            NV2080_CTRL_GR_INFO {
                index: GR_INFO_MAX_WARPS_PER_SM,
                data: 0,
            },
            NV2080_CTRL_GR_INFO {
                index: GR_INFO_LITTER_NUM_GPCS,
                data: 0,
            },
            NV2080_CTRL_GR_INFO {
                index: GR_INFO_LITTER_NUM_TPC_PER_GPC,
                data: 0,
            },
            NV2080_CTRL_GR_INFO {
                index: GR_INFO_LITTER_NUM_SM_PER_TPC,
                data: 0,
            },
        ];
        let mut gp = NV2080_CTRL_GR_GET_INFO_PARAMS {
            grInfoListSize: list.len() as u32,
            grInfoList: list.as_mut_ptr() as u64,
            ..Default::default()
        };
        rm_control(&ctl, root, subdevice, NV2080_CTRL_CMD_GR_GET_INFO, &mut gp)?;
        let gr = GrInfo {
            sm_version: list[0].data,
            max_warps_per_sm: list[1].data,
            gpcs: list[2].data,
            tpc_per_gpc: list[3].data,
            sm_per_tpc: list[4].data,
        };

        let usermode_obj = rm_alloc::<()>(&ctl, root, subdevice, TURING_USERMODE_A, None)?;

        let mut vp = NV_VASPACE_ALLOCATION_PARAMETERS {
            vaBase: 0x1000,
            vaSize: 0x1_ffff_fb00_0000,
            flags: NV_VASPACE_ALLOCATION_FLAGS_ENABLE_PAGE_FAULTING
                | NV_VASPACE_ALLOCATION_FLAGS_IS_EXTERNALLY_OWNED,
            ..Default::default()
        };
        let vaspace = rm_alloc(&ctl, root, device, FERMI_VASPACE_A, Some(&mut vp))?;

        let mut gid = NV2080_CTRL_GPU_GET_GID_INFO_PARAMS {
            index: 0,
            flags: NV2080_GPU_CMD_GPU_GET_GID_FLAGS_FORMAT_BINARY,
            length: 16,
            data: [0; 256],
        };
        rm_control(
            &ctl,
            root,
            subdevice,
            NV2080_CTRL_CMD_GPU_GET_GID_INFO,
            &mut gid,
        )?;
        let uuid: [u8; 16] = gid.data[..16].try_into().unwrap();

        let mut rg = UVM_REGISTER_GPU_PARAMS {
            gpu_uuid: uuid,
            rmCtrlFd: -1,
            ..Default::default()
        };
        // SAFETY: UVM_REGISTER_GPU params.
        unsafe { uvm(&uvm_fd, UVM_REGISTER_GPU, &mut rg, |p| p.rmStatus)? };
        let mut rv = UVM_REGISTER_GPU_VASPACE_PARAMS {
            gpuUuid: uuid,
            rmCtrlFd: ctl.0,
            hClient: root,
            hVaSpace: vaspace,
            rmStatus: 0,
        };
        // SAFETY: UVM_REGISTER_GPU_VASPACE params.
        unsafe { uvm(&uvm_fd, UVM_REGISTER_GPU_VASPACE, &mut rv, |p| p.rmStatus)? };

        let flags = libc::MAP_PRIVATE | libc::MAP_ANONYMOUS | libc::MAP_NORESERVE;
        let reservation = VA_BASES
            .iter()
            .find_map(|&b| {
                Map::new(
                    b,
                    VA_SIZE,
                    libc::PROT_NONE,
                    flags | libc::MAP_FIXED_NOREPLACE,
                    -1,
                    0,
                )
                .ok()
            })
            .ok_or(Error::OutOfMemory { requested: VA_SIZE })?;
        let start = (reservation.addr() as u64).next_multiple_of(VA_ALIGN);
        let end = reservation.addr() as u64 + reservation.len as u64;
        let mut free = BTreeMap::new();
        free.insert(start, (end - start) / VA_ALIGN * VA_ALIGN);

        let usermode = map_object(
            &ctl,
            minor,
            root,
            device,
            usermode_obj,
            0x10000,
            NVOS33_FLAGS_ACCESS_WRITE_ONLY,
            None,
        )?;
        Ok(Gpu {
            ctl,
            uvm: Uvm {
                fd: uvm_fd,
                _mm: mm,
            },
            dev_fd,
            minor,
            root,
            device,
            vaspace,
            uuid,
            name,
            arch,
            gr,
            usermode,
            _reservation: reservation,
            va: Mutex::new(VaSpace { free }),
            next_handle: std::sync::atomic::AtomicU32::new(0xcf00_0000),
        })
    }

    fn map_object(&self, handle: NvHandle, len: u64, flags: u32, at: Option<u64>) -> Result<Map> {
        map_object(
            &self.ctl,
            self.minor,
            self.root,
            self.device,
            handle,
            len,
            flags,
            at,
        )
    }

    fn uvm_map(&self, va: u64, size: u64, handle: NvHandle) -> Result<()> {
        let mut r = UVM_CREATE_EXTERNAL_RANGE_PARAMS {
            base: va,
            length: size,
            ..Default::default()
        };
        // SAFETY: UVM_CREATE_EXTERNAL_RANGE params.
        unsafe {
            uvm(&self.uvm.fd, UVM_CREATE_EXTERNAL_RANGE, &mut r, |p| {
                p.rmStatus
            })?
        };
        let mut m = Box::new(UVM_MAP_EXTERNAL_ALLOCATION_PARAMS {
            base: va,
            length: size,
            offset: 0,
            perGpuAttributes: [UvmGpuMappingAttributes::default(); UVM_MAX_GPUS],
            gpuAttributesCount: 1,
            rmCtrlFd: self.ctl.0,
            hClient: self.root,
            hMemory: handle,
            rmStatus: 0,
        });
        m.perGpuAttributes[0] = UvmGpuMappingAttributes {
            gpuUuid: self.uuid,
            gpuMappingType: UVM_GPU_MAPPING_READ_WRITE_ATOMIC,
            ..Default::default()
        };
        // SAFETY: UVM_MAP_EXTERNAL_ALLOCATION params.
        let r = unsafe {
            uvm(&self.uvm.fd, UVM_MAP_EXTERNAL_ALLOCATION, &mut *m, |p| {
                p.rmStatus
            })
        };
        if r.is_err() {
            self.uvm_free(va);
        }
        r
    }

    fn uvm_free(&self, va: u64) {
        let mut f = UVM_FREE_PARAMS {
            base: va,
            ..Default::default()
        };
        // SAFETY: UVM_FREE params.
        let _ = unsafe { uvm(&self.uvm.fd, UVM_FREE, &mut f, |p| p.rmStatus) };
    }

    fn va_alloc(&self, size: u64) -> Result<u64> {
        self.va
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .alloc(size)
            .ok_or(Error::OutOfMemory {
                requested: size as usize,
            })
    }

    fn va_release(&self, va: u64, size: u64, cpu_mapped: bool) {
        if cpu_mapped {
            // Put the PROT_NONE reservation back.
            let flags =
                libc::MAP_PRIVATE | libc::MAP_ANONYMOUS | libc::MAP_NORESERVE | libc::MAP_FIXED;
            if let Ok(m) = Map::new(va as usize, size as usize, libc::PROT_NONE, flags, -1, 0) {
                m.forget();
            }
        }
        self.va
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .release(va, size);
    }

    /// Allocates `bytes` of `kind` memory, mapped for the GPU.
    pub fn alloc(&self, bytes: u64, kind: Mem) -> Result<Alloc> {
        let bytes = bytes.max(1);
        let size = match kind {
            Mem::Sys => bytes.next_multiple_of(4096),
            _ if bytes > (1 << 20) => bytes.next_multiple_of(2 << 20),
            _ => bytes.next_multiple_of(64 << 10),
        };
        let va = self.va_alloc(size)?;
        let made = self.alloc_at(va, size, kind);
        if made.is_err() {
            self.va_release(va, size, kind != Mem::Vram);
        }
        made
    }

    fn alloc_at(&self, va: u64, size: u64, kind: Mem) -> Result<Alloc> {
        let handle = match kind {
            Mem::Sys => {
                let m = Map::new(
                    va as usize,
                    size as usize,
                    RW,
                    libc::MAP_SHARED | libc::MAP_ANONYMOUS | libc::MAP_FIXED,
                    -1,
                    0,
                )?;
                m.forget();
                let handle = self
                    .next_handle
                    .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                let mut p = nv_ioctl_nvos02_parameters_with_fd {
                    params: NVOS02_PARAMETERS {
                        hRoot: self.root,
                        hObjectParent: self.device,
                        hObjectNew: handle,
                        hClass: NV01_MEMORY_SYSTEM_OS_DESCRIPTOR,
                        flags: NVOS02_FLAGS_PHYSICALITY_NONCONTIGUOUS
                            | NVOS02_FLAGS_COHERENCY_CACHED
                            | NVOS02_FLAGS_MAPPING_NO_MAP,
                        pMemory: va,
                        limit: size - 1,
                        ..Default::default()
                    },
                    fd: -1,
                    _pad: 0,
                };
                // SAFETY: NV_ESC_RM_ALLOC_MEMORY takes nv_ioctl_nvos02_parameters_with_fd.
                unsafe { rm_escape(&self.dev_fd, NV_ESC_RM_ALLOC_MEMORY, &mut p)? };
                status("alloc sysmem", p.params.status)?;
                handle
            }
            Mem::Vram | Mem::VramMapped => {
                let huge = size >= (2 << 20) && size.is_multiple_of(2 << 20);
                let page = if huge { 2 << 20 } else { 64 << 10 };
                let phys = if kind == Mem::VramMapped {
                    NVOS32_ATTR_PHYSICALITY_CONTIGUOUS
                } else {
                    NVOS32_ATTR_PHYSICALITY_ALLOW_NONCONTIGUOUS
                };
                let mut p = NV_MEMORY_ALLOCATION_PARAMS {
                    owner: self.root,
                    r#type: NVOS32_TYPE_IMAGE,
                    flags: NVOS32_ALLOC_FLAGS_MAP_NOT_REQUIRED
                        | NVOS32_ALLOC_FLAGS_MEMORY_HANDLE_PROVIDED
                        | NVOS32_ALLOC_FLAGS_ALIGNMENT_FORCE,
                    attr: phys
                        | if huge {
                            NVOS32_ATTR_PAGE_SIZE_HUGE
                        } else {
                            NVOS32_ATTR_PAGE_SIZE_BIG
                        },
                    attr2: NVOS32_ATTR2_ZBC_PREFER_NO_ZBC
                        | NVOS32_ATTR2_GPU_CACHEABLE_YES
                        | if huge {
                            NVOS32_ATTR2_PAGE_SIZE_HUGE_2MB
                        } else {
                            0
                        },
                    format: 6,
                    size,
                    alignment: page,
                    limit: size - 1,
                    ..Default::default()
                };
                let h = rm_alloc(
                    &self.ctl,
                    self.root,
                    self.device,
                    NV1_MEMORY_USER,
                    Some(&mut p),
                )
                .map_err(|e| match e {
                    Error::OutOfMemory { .. } => Error::OutOfMemory {
                        requested: size as usize,
                    },
                    e => e,
                })?;
                if kind == Mem::VramMapped
                    && let Err(e) = self
                        .map_object(h, size, NVOS33_FLAGS_FIFO_AREA, Some(va))
                        .map(Map::forget)
                {
                    let _ = rm_free(&self.ctl, self.root, self.device, h);
                    return Err(e);
                }
                h
            }
        };
        if let Err(e) = self.uvm_map(va, size, handle) {
            let e = match e {
                Error::Unsupported(m) => {
                    Error::Unsupported(format!("{m} (mapping {kind:?}, {size} bytes)"))
                }
                e => e,
            };
            let _ = rm_free(&self.ctl, self.root, self.device, handle);
            return Err(e);
        }
        Ok(Alloc {
            va,
            size,
            kind,
            handle,
        })
    }

    /// Releases an allocation. The caller guarantees the GPU is done with it.
    pub fn free(&self, a: Alloc) {
        self.uvm_free(a.va);
        let _ = rm_free(&self.ctl, self.root, self.device, a.handle);
        self.va_release(a.va, a.size, a.kind != Mem::Vram);
    }

    /// Reserves `size` bytes of GPU VA without memory (UVM channel registration).
    pub fn reserve_va(&self, size: u64) -> Result<u64> {
        self.va_alloc(size)
    }

    pub fn register_channel(&self, channel: NvHandle, base: u64, length: u64) -> Result<()> {
        let mut p = UVM_REGISTER_CHANNEL_PARAMS {
            gpuUuid: self.uuid,
            rmCtrlFd: self.ctl.0,
            hClient: self.root,
            hChannel: channel,
            base,
            length,
            ..Default::default()
        };
        // SAFETY: UVM_REGISTER_CHANNEL params.
        unsafe { uvm(&self.uvm.fd, UVM_REGISTER_CHANNEL, &mut p, |p| p.rmStatus) }
    }
}

impl Drop for Gpu {
    fn drop(&mut self) {
        // Freeing the client frees every RM object below it; UVM state goes
        // with its fds, the reservation with `reservation`.
        let _ = rm_free(&self.ctl, self.root, 0, self.root);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn va_first_fit_and_merge() {
        let mut v = VaSpace {
            free: BTreeMap::from([(0, 10 * VA_ALIGN)]),
        };
        let a = v.alloc(1).unwrap();
        let b = v.alloc(3 * VA_ALIGN).unwrap();
        let c = v.alloc(VA_ALIGN).unwrap();
        assert_eq!((a, b, c), (0, VA_ALIGN, 4 * VA_ALIGN));
        v.release(b, 3 * VA_ALIGN);
        assert_eq!(v.alloc(2 * VA_ALIGN), Some(VA_ALIGN));
        v.release(VA_ALIGN, 2 * VA_ALIGN);
        v.release(a, 1);
        v.release(c, VA_ALIGN);
        assert_eq!(v.free, BTreeMap::from([(0, 10 * VA_ALIGN)]));
        assert_eq!(v.alloc(11 * VA_ALIGN), None);
    }
}
