//! KFD UAPI (`linux/kfd_ioctl.h`, interface 1.x) and the HSA structures the
//! CP firmware reads in AQL mode (`hsa.h`, `amd_hsa_queue.h`,
//! `amd_hsa_signal.h`). Sizes are asserted against the C layouts.

#![allow(
    dead_code,
    reason = "the UAPI is declared whole, the backend uses a part"
)]

use crate::runtime::os::{IOC_READ, IOC_WRITE, ioc};

const K: u8 = b'K';

const fn ior<T>(nr: u8) -> u64 {
    ioc(IOC_READ, K, nr, size_of::<T>())
}
const fn iow<T>(nr: u8) -> u64 {
    ioc(IOC_WRITE, K, nr, size_of::<T>())
}
const fn iowr<T>(nr: u8) -> u64 {
    ioc(IOC_READ | IOC_WRITE, K, nr, size_of::<T>())
}

#[repr(C)]
#[derive(Debug, Default, Clone, Copy)]
pub struct GetVersion {
    pub major: u32,
    pub minor: u32,
}

#[repr(C)]
#[derive(Debug, Default, Clone, Copy)]
pub struct CreateQueue {
    pub ring_base_address: u64,
    pub write_pointer_address: u64,
    pub read_pointer_address: u64,
    pub doorbell_offset: u64,
    pub ring_size: u32,
    pub gpu_id: u32,
    pub queue_type: u32,
    pub queue_percentage: u32,
    pub queue_priority: u32,
    pub queue_id: u32,
    pub eop_buffer_address: u64,
    pub eop_buffer_size: u64,
    pub ctx_save_restore_address: u64,
    pub ctx_save_restore_size: u32,
    pub ctl_stack_size: u32,
    pub sdma_engine_id: u32,
    pub metadata_ring_size: u32,
}

#[repr(C)]
#[derive(Debug, Default, Clone, Copy)]
pub struct DestroyQueue {
    pub queue_id: u32,
    pub pad: u32,
}

#[repr(C)]
#[derive(Debug, Default, Clone, Copy)]
pub struct ClockCounters {
    pub gpu_clock_counter: u64,
    pub cpu_clock_counter: u64,
    pub system_clock_counter: u64,
    pub system_clock_freq: u64,
    pub gpu_id: u32,
    pub pad: u32,
}

#[repr(C)]
#[derive(Debug, Default, Clone, Copy)]
pub struct DeviceApertures {
    pub lds_base: u64,
    pub lds_limit: u64,
    pub scratch_base: u64,
    pub scratch_limit: u64,
    pub gpuvm_base: u64,
    pub gpuvm_limit: u64,
    pub gpu_id: u32,
    pub pad: u32,
}

#[repr(C)]
#[derive(Debug, Default, Clone, Copy)]
pub struct GetAperturesNew {
    pub apertures_ptr: u64,
    pub num_of_nodes: u32,
    pub pad: u32,
}

#[repr(C)]
#[derive(Debug, Default, Clone, Copy)]
pub struct AcquireVm {
    pub drm_fd: u32,
    pub gpu_id: u32,
}

#[repr(C)]
#[derive(Debug, Default, Clone, Copy)]
pub struct AllocMemory {
    pub va_addr: u64,
    pub size: u64,
    pub handle: u64,
    pub mmap_offset: u64,
    pub gpu_id: u32,
    pub flags: u32,
}

#[repr(C)]
#[derive(Debug, Default, Clone, Copy)]
pub struct FreeMemory {
    pub handle: u64,
}

/// `kfd_ioctl_map_memory_to_gpu_args`, also for unmapping.
#[repr(C)]
#[derive(Debug, Default, Clone, Copy)]
pub struct MapMemory {
    pub handle: u64,
    pub device_ids_array_ptr: u64,
    pub n_devices: u32,
    pub n_success: u32,
}

#[repr(C)]
#[derive(Debug, Default, Clone, Copy)]
pub struct ImportDmabuf {
    pub va_addr: u64,
    pub handle: u64,
    pub gpu_id: u32,
    pub dmabuf_fd: u32,
}

#[repr(C)]
#[derive(Debug, Default, Clone, Copy)]
pub struct ExportDmabuf {
    pub handle: u64,
    pub flags: u32,
    pub dmabuf_fd: u32,
}

#[repr(C)]
#[derive(Debug, Default, Clone, Copy)]
pub struct AvailableMemory {
    pub available: u64,
    pub gpu_id: u32,
    pub pad: u32,
}

const _: () = {
    assert!(size_of::<GetVersion>() == 8);
    assert!(size_of::<CreateQueue>() == 96);
    assert!(size_of::<DestroyQueue>() == 8);
    assert!(size_of::<ClockCounters>() == 40);
    assert!(size_of::<DeviceApertures>() == 56);
    assert!(size_of::<GetAperturesNew>() == 16);
    assert!(size_of::<AcquireVm>() == 8);
    assert!(size_of::<AllocMemory>() == 40);
    assert!(size_of::<FreeMemory>() == 8);
    assert!(size_of::<MapMemory>() == 24);
    assert!(size_of::<ImportDmabuf>() == 24);
    assert!(size_of::<ExportDmabuf>() == 16);
    assert!(size_of::<AvailableMemory>() == 16);
};

pub const GET_VERSION: u64 = ior::<GetVersion>(0x01);
pub const CREATE_QUEUE: u64 = iowr::<CreateQueue>(0x02);
pub const DESTROY_QUEUE: u64 = iowr::<DestroyQueue>(0x03);
pub const GET_CLOCK_COUNTERS: u64 = iowr::<ClockCounters>(0x05);
pub const GET_PROCESS_APERTURES_NEW: u64 = iowr::<GetAperturesNew>(0x14);
pub const ACQUIRE_VM: u64 = iow::<AcquireVm>(0x15);
pub const ALLOC_MEMORY_OF_GPU: u64 = iowr::<AllocMemory>(0x16);
pub const FREE_MEMORY_OF_GPU: u64 = iow::<FreeMemory>(0x17);
pub const MAP_MEMORY_TO_GPU: u64 = iowr::<MapMemory>(0x18);
pub const UNMAP_MEMORY_FROM_GPU: u64 = iowr::<MapMemory>(0x19);
pub const IMPORT_DMABUF: u64 = iowr::<ImportDmabuf>(0x1D);
pub const AVAILABLE_MEMORY: u64 = iowr::<AvailableMemory>(0x23);
pub const EXPORT_DMABUF: u64 = iowr::<ExportDmabuf>(0x24);

pub const QUEUE_TYPE_SDMA: u32 = 1;
pub const QUEUE_TYPE_COMPUTE_AQL: u32 = 2;
pub const MAX_QUEUE_PERCENTAGE: u32 = 100;
/// The middle of KFD's 0..=15, what `ROCr` asks for a normal queue.
pub const NORMAL_QUEUE_PRIORITY: u32 = 7;

pub mod mem {
    pub const VRAM: u32 = 1 << 0;
    pub const GTT: u32 = 1 << 1;
    pub const WRITABLE: u32 = 1 << 31;
    pub const EXECUTABLE: u32 = 1 << 30;
    pub const PUBLIC: u32 = 1 << 29;
    pub const NO_SUBSTITUTE: u32 = 1 << 28;
    pub const COHERENT: u32 = 1 << 26;
    pub const UNCACHED: u32 = 1 << 25;
}

/// `hsa_kernel_dispatch_packet_t`.
#[repr(C)]
#[derive(Debug, Default, Clone, Copy)]
pub struct DispatchPacket {
    pub header: u16,
    pub setup: u16,
    pub workgroup_size_x: u16,
    pub workgroup_size_y: u16,
    pub workgroup_size_z: u16,
    pub reserved0: u16,
    pub grid_size_x: u32,
    pub grid_size_y: u32,
    pub grid_size_z: u32,
    pub private_segment_size: u32,
    pub group_segment_size: u32,
    pub kernel_object: u64,
    pub kernarg_address: u64,
    pub reserved2: u64,
    pub completion_signal: u64,
}

const _: () = assert!(size_of::<DispatchPacket>() == 64);

/// `HSA_PACKET_TYPE_KERNEL_DISPATCH`.
pub const PACKET_KERNEL_DISPATCH: u16 = 2;
/// `HSA_PACKET_TYPE_BARRIER_AND`: waits for up to five signals, runs nothing.
pub const PACKET_BARRIER_AND: u16 = 3;
/// `HSA_PACKET_TYPE_INVALID`: what a slot holds until it is published.
pub const PACKET_INVALID: u16 = 1;
pub const HEADER_BARRIER: u16 = 1 << 8;
/// Acquire (bits 9..11) and release (bits 11..13) fence scope `SYSTEM`.
pub const HEADER_ACQUIRE_SYSTEM: u16 = 2 << 9;
pub const HEADER_RELEASE_SYSTEM: u16 = 2 << 11;
/// A kernel dispatch that waits for the packets before it and makes its
/// writes visible to the host when done.
pub const DISPATCH_HEADER: u16 =
    PACKET_KERNEL_DISPATCH | HEADER_BARRIER | HEADER_ACQUIRE_SYSTEM | HEADER_RELEASE_SYSTEM;

/// `amd_queue_t` offsets: 256 bytes, its own page. The CP reads and writes
/// the dispatch ids; everything else stays zero (no scratch, no traps).
pub mod queue {
    pub const SIZE: usize = 256;
    /// `hsa_queue_t::type` (`HSA_QUEUE_TYPE_SINGLE` = 1).
    pub const TYPE: usize = 0;
    /// `hsa_queue_t::features` (`HSA_QUEUE_FEATURE_KERNEL_DISPATCH` = 1).
    pub const FEATURES: usize = 4;
    pub const BASE_ADDRESS: usize = 8;
    pub const SIZE_PACKETS: usize = 24;
    pub const WRITE_DISPATCH_ID: usize = 56;
    pub const READ_DISPATCH_ID: usize = 128;
    /// Where `read_dispatch_id` sits from the queue's base, for the CP.
    pub const READ_DISPATCH_ID_FIELD_BASE_BYTE_OFFSET: usize = 136;
    pub const QUEUE_PROPERTIES: usize = 180;
    /// `AMD_QUEUE_PROPERTIES_IS_PTR64`.
    pub const IS_PTR64: u32 = 1 << 1;
    /// `AMD_QUEUE_PROPERTIES_ENABLE_PROFILING`: the CP writes start and end
    /// timestamps into the completion signal.
    pub const ENABLE_PROFILING: u32 = 1 << 3;
}

/// `amd_signal_t` offsets: 64 bytes.
pub mod signal {
    pub const SIZE: usize = 64;
    /// `AMD_SIGNAL_KIND_USER` = 1.
    pub const KIND: usize = 0;
    /// Decremented by the CP when the packet completes.
    pub const VALUE: usize = 8;
    pub const START_TS: usize = 32;
    pub const END_TS: usize = 40;
}

/// The amdgpu DRM UAPI (`drm/amdgpu_drm.h`) for submitting to the driver's
/// own rings.
pub mod drm {
    use crate::runtime::os::{IOC_READ, IOC_WRITE, ioc};

    /// `DRM_COMMAND_BASE`.
    const BASE: u8 = 0x40;

    const fn iowr<T>(nr: u8) -> u64 {
        ioc(IOC_READ | IOC_WRITE, b'd', BASE + nr, size_of::<T>())
    }

    /// `union drm_amdgpu_ctx`: `drm_amdgpu_ctx_in`, and on the way out
    /// `alloc.ctx_id` in the place of `op`.
    #[repr(C)]
    #[derive(Debug, Default, Clone, Copy)]
    #[allow(clippy::struct_field_names, reason = "the UAPI's field names")]
    pub struct Ctx {
        pub op: u32,
        pub flags: u32,
        pub ctx_id: u32,
        pub priority: i32,
    }

    /// `union drm_amdgpu_cs`: `drm_amdgpu_cs_in`, and on the way out the
    /// submission's sequence handle in its first 8 bytes.
    #[repr(C)]
    #[derive(Debug, Default, Clone, Copy)]
    pub struct Cs {
        pub ctx_id: u32,
        pub bo_list_handle: u32,
        pub num_chunks: u32,
        pub flags: u32,
        pub chunks: u64,
    }

    #[repr(C)]
    #[derive(Debug, Default, Clone, Copy)]
    pub struct CsChunk {
        pub chunk_id: u32,
        pub length_dw: u32,
        pub chunk_data: u64,
    }

    #[repr(C)]
    #[derive(Debug, Default, Clone, Copy)]
    pub struct CsChunkIb {
        pub pad: u32,
        pub flags: u32,
        pub va_start: u64,
        pub ib_bytes: u32,
        pub ip_type: u32,
        pub ip_instance: u32,
        pub ring: u32,
    }

    const _: () = {
        assert!(size_of::<Ctx>() == 16);
        assert!(size_of::<Cs>() == 24);
        assert!(size_of::<CsChunk>() == 16);
        assert!(size_of::<CsChunkIb>() == 32);
    };

    pub const IOCTL_CTX: u64 = iowr::<Ctx>(0x02);
    pub const IOCTL_CS: u64 = iowr::<Cs>(0x04);
    pub const CTX_OP_ALLOC_CTX: u32 = 1;
    pub const CTX_OP_FREE_CTX: u32 = 2;
    pub const HW_IP_DMA: u32 = 2;
    pub const CHUNK_ID_IB: u32 = 1;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ioctl_numbers_match_the_header() {
        // From `linux/kfd_ioctl.h` (1.23) through the C preprocessor.
        assert_eq!(GET_VERSION, 0x8008_4b01);
        assert_eq!(CREATE_QUEUE, 0xc060_4b02);
        assert_eq!(DESTROY_QUEUE, 0xc008_4b03);
        assert_eq!(GET_CLOCK_COUNTERS, 0xc028_4b05);
        assert_eq!(GET_PROCESS_APERTURES_NEW, 0xc010_4b14);
        assert_eq!(ACQUIRE_VM, 0x4008_4b15);
        assert_eq!(ALLOC_MEMORY_OF_GPU, 0xc028_4b16);
        assert_eq!(FREE_MEMORY_OF_GPU, 0x4008_4b17);
        assert_eq!(MAP_MEMORY_TO_GPU, 0xc018_4b18);
        assert_eq!(UNMAP_MEMORY_FROM_GPU, 0xc018_4b19);
        assert_eq!(IMPORT_DMABUF, 0xc018_4b1d);
        assert_eq!(AVAILABLE_MEMORY, 0xc010_4b23);
        assert_eq!(EXPORT_DMABUF, 0xc010_4b24);
        // `drm/amdgpu_drm.h`.
        assert_eq!(drm::IOCTL_CTX, 0xc010_6442);
        assert_eq!(drm::IOCTL_CS, 0xc018_6444);
    }
}
