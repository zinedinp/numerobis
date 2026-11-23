//! UAPI of the NVIDIA open kernel modules (RM escapes on `/dev/nvidiactl` and
//! `/dev/nvidiaN`, UVM on `/dev/nvidia-uvm`) plus the class/method numbers
//! numerobis uses.
//!
//! Layouts follow `open-gpu-kernel-modules` 620.70.06 (`nvos.h`, `nv-ioctl.h`,
//! `uvm_ioctl.h`, `alloc_channel.h`, `ctrl*.h`, `cl*.h`); every struct has a
//! size assert taken from a C `sizeof` dump of those headers. The QMD layout
//! (`qmd.rs`) comes from NVIDIA's `open-gpu-doc` (`clc6c0qmd.h`).
//!
//! The RM ABI is not stable across driver versions:
//! [`TESTED_DRIVERS`] lists the versions these layouts were checked against.

#![allow(non_camel_case_types, non_snake_case, dead_code)]

use crate::runtime::error::{Error, Result};
use crate::runtime::os::{Fd, iowr};

/// Driver versions whose headers these layouts were checked against.
pub const TESTED_DRIVERS: &[&str] = &["620.70.06"];

pub type NvHandle = u32;

macro_rules! sized {
    ($($t:ty = $n:expr),* $(,)?) => {$(const _: () = assert!(size_of::<$t>() == $n);)*};
}

// ---- RM escapes (nv_escape.h, nv-ioctl-numbers.h) ----

pub const NV_IOCTL_MAGIC: u8 = b'F';
pub const NV_ESC_CARD_INFO: u8 = 200;
pub const NV_ESC_REGISTER_FD: u8 = 201;
pub const NV_ESC_RM_ALLOC_MEMORY: u8 = 0x27;
pub const NV_ESC_RM_FREE: u8 = 0x29;
pub const NV_ESC_RM_CONTROL: u8 = 0x2A;
pub const NV_ESC_RM_ALLOC: u8 = 0x2B;
pub const NV_ESC_RM_MAP_MEMORY: u8 = 0x4E;

#[repr(C)]
#[derive(Debug, Default, Clone, Copy)]
pub struct NVOS00_PARAMETERS {
    pub hRoot: NvHandle,
    pub hObjectParent: NvHandle,
    pub hObjectOld: NvHandle,
    pub status: u32,
}

#[repr(C)]
#[derive(Debug, Default, Clone, Copy)]
pub struct NVOS02_PARAMETERS {
    pub hRoot: NvHandle,
    pub hObjectParent: NvHandle,
    pub hObjectNew: NvHandle,
    pub hClass: u32,
    pub flags: u32,
    pub _pad: u32,
    pub pMemory: u64,
    pub limit: u64,
    pub status: u32,
    pub _pad2: u32,
}

#[repr(C)]
#[derive(Debug, Default, Clone, Copy)]
pub struct NVOS21_PARAMETERS {
    pub hRoot: NvHandle,
    pub hObjectParent: NvHandle,
    pub hObjectNew: NvHandle,
    pub hClass: u32,
    pub pAllocParms: u64,
    pub paramsSize: u32,
    pub status: u32,
}

#[repr(C)]
#[derive(Debug, Default, Clone, Copy)]
pub struct NVOS33_PARAMETERS {
    pub hClient: NvHandle,
    pub hDevice: NvHandle,
    pub hMemory: NvHandle,
    pub _pad: u32,
    pub offset: u64,
    pub length: u64,
    pub pLinearAddress: u64,
    pub status: u32,
    pub flags: u32,
}

#[repr(C)]
#[derive(Debug, Default, Clone, Copy)]
pub struct NVOS54_PARAMETERS {
    pub hClient: NvHandle,
    pub hObject: NvHandle,
    pub cmd: u32,
    pub flags: u32,
    pub params: u64,
    pub paramsSize: u32,
    pub status: u32,
}

#[repr(C)]
#[derive(Debug, Default, Clone, Copy)]
pub struct nv_ioctl_nvos02_parameters_with_fd {
    pub params: NVOS02_PARAMETERS,
    pub fd: i32,
    pub _pad: u32,
}

#[repr(C)]
#[derive(Debug, Default, Clone, Copy)]
pub struct nv_ioctl_nvos33_parameters_with_fd {
    pub params: NVOS33_PARAMETERS,
    pub fd: i32,
    pub _pad: u32,
}

#[repr(C)]
#[derive(Debug, Default, Clone, Copy)]
pub struct nv_pci_info_t {
    pub domain: u32,
    pub bus: u8,
    pub slot: u8,
    pub function: u8,
    pub _pad: u8,
    pub vendor_id: u16,
    pub device_id: u16,
}

#[repr(C)]
#[derive(Debug, Default, Clone, Copy)]
pub struct nv_ioctl_card_info_t {
    pub valid: u8,
    pub _pad: [u8; 3],
    pub pci_info: nv_pci_info_t,
    pub gpu_id: u32,
    pub interrupt_line: u16,
    pub _pad2: u16,
    pub reg_address: u64,
    pub reg_size: u64,
    pub fb_address: u64,
    pub fb_size: u64,
    pub minor_number: u32,
    pub dev_name: [u8; 10],
    pub _pad3: [u8; 2],
}

#[repr(C)]
#[derive(Debug, Default, Clone, Copy)]
pub struct nv_ioctl_register_fd_t {
    pub ctl_fd: i32,
}

sized!(
    NVOS00_PARAMETERS = 16,
    NVOS02_PARAMETERS = 48,
    NVOS21_PARAMETERS = 32,
    NVOS33_PARAMETERS = 48,
    NVOS54_PARAMETERS = 32,
    nv_ioctl_nvos02_parameters_with_fd = 56,
    nv_ioctl_nvos33_parameters_with_fd = 56,
    nv_ioctl_card_info_t = 72,
);
const _: () = assert!(core::mem::offset_of!(nv_ioctl_card_info_t, gpu_id) == 16);
const _: () = assert!(core::mem::offset_of!(nv_ioctl_card_info_t, minor_number) == 56);

// ---- Classes ----

pub const NV01_ROOT_CLIENT: u32 = 0x41;
pub const NV01_DEVICE_0: u32 = 0x80;
pub const NV20_SUBDEVICE_0: u32 = 0x2080;
pub const NV01_MEMORY_SYSTEM_OS_DESCRIPTOR: u32 = 0x71;
pub const NV1_MEMORY_USER: u32 = 0x40;
pub const FERMI_CONTEXT_SHARE_A: u32 = 0x9067;
pub const FERMI_VASPACE_A: u32 = 0x90f1;
pub const KEPLER_CHANNEL_GROUP_A: u32 = 0xa06c;
pub const TURING_USERMODE_A: u32 = 0xc461;
pub const AMPERE_CHANNEL_GPFIFO_A: u32 = 0xc56f;
pub const AMPERE_DMA_COPY_B: u32 = 0xc7b5;
pub const AMPERE_COMPUTE_B: u32 = 0xc7c0;
pub const ADA_COMPUTE_A: u32 = 0xc9c0;

// ---- Class allocation parameters ----

#[repr(C)]
#[derive(Debug, Default, Clone, Copy)]
pub struct NV0080_ALLOC_PARAMETERS {
    pub deviceId: u32,
    pub hClientShare: NvHandle,
    pub hTargetClient: NvHandle,
    pub hTargetDevice: NvHandle,
    pub flags: u32,
    pub _pad: u32,
    pub vaSpaceSize: u64,
    pub vaStartInternal: u64,
    pub vaLimitInternal: u64,
    pub vaMode: u32,
    pub _pad2: u32,
}
pub const NV_DEVICE_ALLOCATION_VAMODE_MULTIPLE_VASPACES: u32 = 2;

#[repr(C)]
#[derive(Debug, Default, Clone, Copy)]
pub struct NV2080_ALLOC_PARAMETERS {
    pub subDeviceId: u32,
}

#[repr(C)]
#[derive(Debug, Default, Clone, Copy)]
pub struct NV_VASPACE_ALLOCATION_PARAMETERS {
    pub index: u32,
    pub flags: u32,
    pub vaSize: u64,
    pub vaStartInternal: u64,
    pub vaLimitInternal: u64,
    pub bigPageSize: u32,
    pub _pad: u32,
    pub vaBase: u64,
    pub pasid: u32,
    pub _pad2: u32,
}
pub const NV_VASPACE_ALLOCATION_FLAGS_IS_EXTERNALLY_OWNED: u32 = 1 << 3;
pub const NV_VASPACE_ALLOCATION_FLAGS_ENABLE_PAGE_FAULTING: u32 = 1 << 6;

#[repr(C)]
#[derive(Debug, Default, Clone, Copy)]
pub struct NV_MEMORY_ALLOCATION_PARAMS {
    pub owner: u32,
    pub r#type: u32,
    pub flags: u32,
    pub width: u32,
    pub height: u32,
    pub pitch: i32,
    pub attr: u32,
    pub attr2: u32,
    pub format: u32,
    pub comprCovg: u32,
    pub zcullCovg: u32,
    pub _pad: u32,
    pub rangeLo: u64,
    pub rangeHi: u64,
    pub size: u64,
    pub alignment: u64,
    pub offset: u64,
    pub limit: u64,
    pub address: u64,
    pub ctagOffset: u32,
    pub hVASpace: NvHandle,
    pub internalflags: u32,
    pub tag: u32,
    pub numaNode: i32,
    pub _pad2: u32,
}
pub const NVOS32_TYPE_IMAGE: u32 = 0;
pub const NVOS32_ALLOC_FLAGS_ALIGNMENT_FORCE: u32 = 0x100;
pub const NVOS32_ALLOC_FLAGS_MEMORY_HANDLE_PROVIDED: u32 = 0x4000;
pub const NVOS32_ALLOC_FLAGS_MAP_NOT_REQUIRED: u32 = 0x8000;
/// `attr` 24:23.
pub const NVOS32_ATTR_PAGE_SIZE_BIG: u32 = 2 << 23;
pub const NVOS32_ATTR_PAGE_SIZE_HUGE: u32 = 3 << 23;
/// `attr` 28:27.
pub const NVOS32_ATTR_PHYSICALITY_CONTIGUOUS: u32 = 2 << 27;
pub const NVOS32_ATTR_PHYSICALITY_ALLOW_NONCONTIGUOUS: u32 = 3 << 27;
/// `attr2` 1:0, 3:2, 21:20.
pub const NVOS32_ATTR2_ZBC_PREFER_NO_ZBC: u32 = 1;
pub const NVOS32_ATTR2_GPU_CACHEABLE_YES: u32 = 1 << 2;
pub const NVOS32_ATTR2_PAGE_SIZE_HUGE_2MB: u32 = 1 << 20;
/// `NVOS02` flags: physicality 7:4, coherency 15:12, mapping 31:30.
pub const NVOS02_FLAGS_PHYSICALITY_NONCONTIGUOUS: u32 = 1 << 4;
pub const NVOS02_FLAGS_COHERENCY_CACHED: u32 = 1 << 12;
pub const NVOS02_FLAGS_MAPPING_NO_MAP: u32 = 1 << 30;
/// `NVOS33` flags for the BAR1 mapping of the GPFIFO/USERD area: mapping
/// REFLECTED (16:15), `MAP_FIXED` (18), `RESERVE_ON_UNMAP` (19), caching 25:23 = 2.
pub const NVOS33_FLAGS_FIFO_AREA: u32 = (2 << 15) | (1 << 18) | (1 << 19) | (2 << 23);
/// `NVOS33_FLAGS_ACCESS_WRITE_ONLY` (usermode doorbell page).
pub const NVOS33_FLAGS_ACCESS_WRITE_ONLY: u32 = 2;

#[repr(C)]
#[derive(Debug, Default, Clone, Copy)]
pub struct NV_CHANNEL_GROUP_ALLOCATION_PARAMETERS {
    pub hObjectError: NvHandle,
    pub hObjectEccError: NvHandle,
    pub hVASpace: NvHandle,
    pub engineType: u32,
    pub bIsCallingContextVgpuPlugin: u8,
    pub _pad: [u8; 3],
    pub reserved: u32,
    pub reserved2: NvHandle,
    pub internalFlags: u32,
}
pub const NV2080_ENGINE_TYPE_GRAPHICS: u32 = 1;

#[repr(C)]
#[derive(Debug, Default, Clone, Copy)]
pub struct NV_CTXSHARE_ALLOCATION_PARAMETERS {
    pub hVASpace: NvHandle,
    pub flags: u32,
    pub subctxId: u32,
    pub reserved: NvHandle,
}
pub const NV_CTXSHARE_ALLOCATION_FLAGS_SUBCONTEXT_ASYNC: u32 = 1;

#[repr(C)]
#[derive(Debug, Default, Clone, Copy)]
pub struct NV_MEMORY_DESC_PARAMS {
    pub base: u64,
    pub size: u64,
    pub addressSpace: u32,
    pub cacheAttrib: u32,
}

#[repr(C)]
#[derive(Debug, Default, Clone, Copy)]
pub struct NV_CHANNEL_ALLOC_PARAMS {
    pub hObjectError: NvHandle,
    pub hObjectBuffer: NvHandle,
    pub gpFifoOffset: u64,
    pub gpFifoEntries: u32,
    pub flags: u32,
    pub hContextShare: NvHandle,
    pub hVASpace: NvHandle,
    pub hHandleVASpace: NvHandle,
    pub hUserdMemory: [NvHandle; 8],
    pub _pad: u32,
    pub userdOffset: [u64; 8],
    pub engineType: u32,
    pub cid: u32,
    pub subDeviceId: u32,
    pub hObjectEccError: NvHandle,
    pub instanceMem: NV_MEMORY_DESC_PARAMS,
    pub userdMem: NV_MEMORY_DESC_PARAMS,
    pub ramfcMem: NV_MEMORY_DESC_PARAMS,
    pub mthdbufMem: NV_MEMORY_DESC_PARAMS,
    pub hPhysChannelGroup: NvHandle,
    pub internalFlags: u32,
    pub errorNotifierMem: NV_MEMORY_DESC_PARAMS,
    pub eccErrorNotifierMem: NV_MEMORY_DESC_PARAMS,
    pub ProcessID: u32,
    pub SubProcessID: u32,
    pub encryptIv: [u32; 3],
    pub decryptIv: [u32; 3],
    pub hmacNonce: [u32; 8],
    pub tpcConfigID: u32,
    pub _pad2: u32,
}

sized!(
    NV0080_ALLOC_PARAMETERS = 56,
    NV2080_ALLOC_PARAMETERS = 4,
    NV_VASPACE_ALLOCATION_PARAMETERS = 56,
    NV_MEMORY_ALLOCATION_PARAMS = 128,
    NV_CHANNEL_GROUP_ALLOCATION_PARAMETERS = 32,
    NV_CTXSHARE_ALLOCATION_PARAMETERS = 16,
    NV_CHANNEL_ALLOC_PARAMS = 376,
);
const _: () = assert!(core::mem::offset_of!(NV_VASPACE_ALLOCATION_PARAMETERS, vaBase) == 40);
const _: () = assert!(core::mem::offset_of!(NV_MEMORY_ALLOCATION_PARAMS, numaNode) == 120);
const _: () = assert!(core::mem::offset_of!(NV_CHANNEL_ALLOC_PARAMS, userdOffset) == 72);
const _: () = assert!(core::mem::offset_of!(NV_CHANNEL_ALLOC_PARAMS, engineType) == 136);
const _: () = assert!(core::mem::offset_of!(NV_CHANNEL_ALLOC_PARAMS, hPhysChannelGroup) == 248);
const _: () = assert!(core::mem::offset_of!(NV_CHANNEL_ALLOC_PARAMS, tpcConfigID) == 368);

// ---- Controls ----

pub const NV0000_CTRL_CMD_GPU_GET_ID_INFO_V2: u32 = 0x205;
pub const NV2080_CTRL_CMD_GPU_GET_NAME_STRING: u32 = 0x2080_0110;
pub const NV2080_CTRL_CMD_GPU_GET_GID_INFO: u32 = 0x2080_014a;
pub const NV2080_CTRL_CMD_GR_GET_INFO: u32 = 0x2080_1201;
pub const NV2080_CTRL_CMD_MC_GET_ARCH_INFO: u32 = 0x2080_1701;
pub const NVA06C_CTRL_CMD_GPFIFO_SCHEDULE: u32 = 0xa06c_0101;
pub const NVC36F_CTRL_CMD_GPFIFO_GET_WORK_SUBMIT_TOKEN: u32 = 0xc36f_0108;

#[repr(C)]
#[derive(Debug, Default, Clone, Copy)]
pub struct NV0000_CTRL_GPU_GET_ID_INFO_V2_PARAMS {
    pub gpuId: u32,
    pub gpuFlags: u32,
    pub deviceInstance: u32,
    pub subDeviceInstance: u32,
    pub sliStatus: u32,
    pub boardId: u32,
    pub gpuInstance: u32,
    pub numaId: i32,
}

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct NV2080_CTRL_GPU_GET_GID_INFO_PARAMS {
    pub index: u32,
    pub flags: u32,
    pub length: u32,
    pub data: [u8; 256],
}
pub const NV2080_GPU_CMD_GPU_GET_GID_FLAGS_FORMAT_BINARY: u32 = 2;

#[repr(C)]
#[derive(Debug, Default, Clone, Copy)]
pub struct NV2080_CTRL_MC_GET_ARCH_INFO_PARAMS {
    pub architecture: u32,
    pub implementation: u32,
    pub revision: u32,
    pub subRevision: u8,
    pub _pad: [u8; 3],
}

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct NV2080_CTRL_GPU_GET_NAME_STRING_PARAMS {
    pub gpuNameStringFlags: u32,
    pub ascii: [u8; 64],
}

#[repr(C)]
#[derive(Debug, Default, Clone, Copy)]
pub struct NV2080_CTRL_GR_INFO {
    pub index: u32,
    pub data: u32,
}

#[repr(C)]
#[derive(Debug, Default, Clone, Copy)]
pub struct NV2080_CTRL_GR_GET_INFO_PARAMS {
    pub grInfoListSize: u32,
    pub _pad: u32,
    pub grInfoList: u64,
    /// `NV2080_CTRL_GR_ROUTE_INFO { u32 flags; u64 route; }`.
    pub grRouteInfo: [u64; 2],
}
pub const GR_INFO_SM_VERSION: u32 = 12;
pub const GR_INFO_MAX_WARPS_PER_SM: u32 = 13;
pub const GR_INFO_LITTER_NUM_GPCS: u32 = 20;
pub const GR_INFO_LITTER_NUM_TPC_PER_GPC: u32 = 23;
pub const GR_INFO_LITTER_NUM_SM_PER_TPC: u32 = 32;

#[repr(C)]
#[derive(Debug, Default, Clone, Copy)]
pub struct NVA06C_CTRL_GPFIFO_SCHEDULE_PARAMS {
    pub bEnable: u8,
    pub bSkipSubmit: u8,
    pub bSkipEnable: u8,
}

#[repr(C)]
#[derive(Debug, Default, Clone, Copy)]
pub struct NVC36F_CTRL_CMD_GPFIFO_GET_WORK_SUBMIT_TOKEN_PARAMS {
    pub workSubmitToken: u32,
}

sized!(
    NV0000_CTRL_GPU_GET_ID_INFO_V2_PARAMS = 32,
    NV2080_CTRL_GPU_GET_GID_INFO_PARAMS = 268,
    NV2080_CTRL_MC_GET_ARCH_INFO_PARAMS = 16,
    NV2080_CTRL_GPU_GET_NAME_STRING_PARAMS = 68,
    NV2080_CTRL_GR_INFO = 8,
    NV2080_CTRL_GR_GET_INFO_PARAMS = 32,
    NVA06C_CTRL_GPFIFO_SCHEDULE_PARAMS = 3,
    NVC36F_CTRL_CMD_GPFIFO_GET_WORK_SUBMIT_TOKEN_PARAMS = 4,
);

/// `NV2080_CTRL_MC_GET_ARCH_INFO` architecture values (`NV2080_CTRL_MC_ARCH_INFO_ARCHITECTURE_*`).
pub const ARCH_GA100: u32 = 0x170;
pub const ARCH_AD100: u32 = 0x190;
pub const ARCH_GH100: u32 = 0x180;
pub const ARCH_GB100: u32 = 0x1a0;
pub const ARCH_GB200: u32 = 0x1b0;

// ---- GPFIFO control area (clc56f.h `Nvc56fControl`) ----

pub const GPFIFO_CONTROL_GPGET: usize = 0x88;
pub const GPFIFO_CONTROL_GPPUT: usize = 0x8c;
/// Doorbell register in the usermode (`TURING_USERMODE_A`) page.
pub const USERMODE_NOTIFY_CHANNEL_PENDING: usize = 0x90;

// ---- Methods ----

/// Host (channel) methods, subchannel 0 (`clc56f.h`).
pub const NVC56F_SEM_ADDR_LO: u32 = 0x5c;
pub const NVC56F_NON_STALL_INTERRUPT: u32 = 0x20;
pub const SEM_EXECUTE_RELEASE: u32 = 1;
pub const SEM_EXECUTE_ACQ_CIRC_GEQ: u32 = 3;
pub const SEM_EXECUTE_RELEASE_WFI: u32 = 1 << 20;
pub const SEM_EXECUTE_PAYLOAD_SIZE_64BIT: u32 = 1 << 24;
pub const SEM_EXECUTE_RELEASE_TIMESTAMP: u32 = 1 << 25;

/// Compute methods (`clc7c0.h`).
pub const NVC7C0_SET_OBJECT: u32 = 0x0;
pub const NVC7C0_SET_SHADER_SHARED_MEMORY_WINDOW_A: u32 = 0x2a0;
pub const NVC7C0_SEND_PCAS_A: u32 = 0x2b4;
pub const NVC7C0_SEND_SIGNALING_PCAS2_B: u32 = 0x2c0;
pub const NVC7C0_SET_SHADER_LOCAL_MEMORY_NON_THROTTLED_A: u32 = 0x2e4;
pub const NVC7C0_SET_SHADER_LOCAL_MEMORY_A: u32 = 0x790;
pub const NVC7C0_SET_SHADER_LOCAL_MEMORY_WINDOW_A: u32 = 0x7b0;
pub const NVC7C0_INVALIDATE_SHADER_CACHES_NO_WFI: u32 = 0x1698;
/// `SEND_SIGNALING_PCAS2_B`: INVALIDATE | SCHEDULE.
pub const PCAS_ACTION_INVALIDATE_SCHEDULE: u32 = 9;

/// Copy engine methods (`clc7b5.h`).
pub const NVC7B5_OFFSET_IN_UPPER: u32 = 0x400;
pub const NVC7B5_LINE_LENGTH_IN: u32 = 0x418;
pub const NVC7B5_LAUNCH_DMA: u32 = 0x300;
/// `NON_PIPELINED` transfer, FLUSH, pitch source and destination.
pub const LAUNCH_DMA_COPY: u32 = 2 | (1 << 2) | (1 << 7) | (1 << 8);

/// Push-buffer header for `count` incrementing method data words (`SEC_OP` 1 = `INC_METHOD`).
pub const fn method(subc: u32, mthd: u32, count: u32) -> u32 {
    (1 << 29) | (count << 16) | (subc << 13) | (mthd >> 2)
}

/// Subchannel bindings.
pub const SUBC_COMPUTE: u32 = 1;
pub const SUBC_COPY: u32 = 4;

// ---- UVM (uvm_ioctl.h, uvm_linux_ioctl.h): raw command numbers ----

pub const UVM_INITIALIZE: u64 = 0x3000_0001;
pub const UVM_REGISTER_GPU_VASPACE: u64 = 25;
pub const UVM_REGISTER_CHANNEL: u64 = 27;
pub const UVM_MAP_EXTERNAL_ALLOCATION: u64 = 33;
pub const UVM_FREE: u64 = 34;
pub const UVM_REGISTER_GPU: u64 = 37;
pub const UVM_CREATE_EXTERNAL_RANGE: u64 = 73;
pub const UVM_MM_INITIALIZE: u64 = 75;

#[repr(C)]
#[derive(Debug, Default, Clone, Copy)]
pub struct UVM_INITIALIZE_PARAMS {
    pub flags: u64,
    pub rmStatus: u32,
    pub _pad: u32,
}

#[repr(C)]
#[derive(Debug, Default, Clone, Copy)]
pub struct UVM_MM_INITIALIZE_PARAMS {
    pub uvmFd: i32,
    pub rmStatus: u32,
}

#[repr(C)]
#[derive(Debug, Default, Clone, Copy)]
pub struct UVM_REGISTER_GPU_PARAMS {
    pub gpu_uuid: [u8; 16],
    pub numaEnabled: u8,
    pub _pad: [u8; 3],
    pub numaNodeId: i32,
    pub rmCtrlFd: i32,
    pub hClient: NvHandle,
    pub hSmcPartRef: NvHandle,
    pub rmStatus: u32,
}

#[repr(C)]
#[derive(Debug, Default, Clone, Copy)]
pub struct UVM_REGISTER_GPU_VASPACE_PARAMS {
    pub gpuUuid: [u8; 16],
    pub rmCtrlFd: i32,
    pub hClient: NvHandle,
    pub hVaSpace: NvHandle,
    pub rmStatus: u32,
}

#[repr(C)]
#[derive(Debug, Default, Clone, Copy)]
pub struct UVM_REGISTER_CHANNEL_PARAMS {
    pub gpuUuid: [u8; 16],
    pub rmCtrlFd: i32,
    pub hClient: NvHandle,
    pub hChannel: NvHandle,
    pub _pad: u32,
    pub base: u64,
    pub length: u64,
    pub rmStatus: u32,
    pub _pad2: u32,
}

#[repr(C)]
#[derive(Debug, Default, Clone, Copy)]
pub struct UVM_CREATE_EXTERNAL_RANGE_PARAMS {
    pub base: u64,
    pub length: u64,
    pub rmStatus: u32,
    pub _pad: u32,
}

#[repr(C)]
#[derive(Debug, Default, Clone, Copy)]
pub struct UvmGpuMappingAttributes {
    pub gpuUuid: [u8; 16],
    pub gpuMappingType: u32,
    pub gpuCachingType: u32,
    pub gpuFormatType: u32,
    pub gpuElementBits: u32,
    pub gpuCompressionType: u32,
}
/// `UvmGpuMappingTypeReadWriteAtomic`.
pub const UVM_GPU_MAPPING_READ_WRITE_ATOMIC: u32 = 1;

pub const UVM_MAX_GPUS: usize = 256;

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct UVM_MAP_EXTERNAL_ALLOCATION_PARAMS {
    pub base: u64,
    pub length: u64,
    pub offset: u64,
    pub perGpuAttributes: [UvmGpuMappingAttributes; UVM_MAX_GPUS],
    pub gpuAttributesCount: u64,
    pub rmCtrlFd: i32,
    pub hClient: NvHandle,
    pub hMemory: NvHandle,
    pub rmStatus: u32,
}

#[repr(C)]
#[derive(Debug, Default, Clone, Copy)]
pub struct UVM_FREE_PARAMS {
    pub base: u64,
    pub rmStatus: u32,
    pub _pad: u32,
}

sized!(
    UVM_INITIALIZE_PARAMS = 16,
    UVM_MM_INITIALIZE_PARAMS = 8,
    UVM_REGISTER_GPU_PARAMS = 40,
    UVM_REGISTER_GPU_VASPACE_PARAMS = 32,
    UVM_REGISTER_CHANNEL_PARAMS = 56,
    UVM_CREATE_EXTERNAL_RANGE_PARAMS = 24,
    UvmGpuMappingAttributes = 36,
    UVM_MAP_EXTERNAL_ALLOCATION_PARAMS = 9264,
    UVM_FREE_PARAMS = 16,
);
const _: () =
    assert!(core::mem::offset_of!(UVM_MAP_EXTERNAL_ALLOCATION_PARAMS, gpuAttributesCount) == 9240);
const _: () = assert!(core::mem::offset_of!(UVM_REGISTER_GPU_PARAMS, rmCtrlFd) == 24);
const _: () = assert!(core::mem::offset_of!(UVM_REGISTER_CHANNEL_PARAMS, base) == 32);

// ---- Status codes (nvstatuscodes.h) ----

pub const NV_OK: u32 = 0;
pub const NV_WARN_NOTHING_TO_DO: u32 = 0x10006;
pub const NV_ERR_INSUFFICIENT_RESOURCES: u32 = 0x1a;
pub const NV_ERR_INVALID_ARGUMENT: u32 = 0x1f;
pub const NV_ERR_NO_MEMORY: u32 = 0x51;
pub const NV_ERR_NOT_SUPPORTED: u32 = 0x56;

/// RM/UVM status -> [`Error`].
pub fn status(what: &str, st: u32) -> Result<()> {
    match st {
        NV_OK => Ok(()),
        NV_ERR_NO_MEMORY | NV_ERR_INSUFFICIENT_RESOURCES => {
            Err(Error::OutOfMemory { requested: 0 })
        }
        NV_ERR_NOT_SUPPORTED => Err(Error::Unsupported(format!("{what}: NV_ERR_NOT_SUPPORTED"))),
        _ => Err(Error::Unsupported(format!("{what}: NV status {st:#x}"))),
    }
}

// ---- Call helpers ----

/// RM escape on `fd` (`_IOWR('F', nr, T)`), then checks `status` via `st`.
///
/// # Safety
/// `T` must be the parameter struct of escape `nr`.
pub unsafe fn rm_escape<T>(fd: &Fd, nr: u8, p: &mut T) -> Result<()> {
    let req = match nr {
        NV_ESC_RM_ALLOC_MEMORY
        | NV_ESC_RM_FREE
        | NV_ESC_RM_CONTROL
        | NV_ESC_RM_ALLOC
        | NV_ESC_RM_MAP_MEMORY
        | NV_ESC_CARD_INFO
        | NV_ESC_REGISTER_FD => iowr::<T>(NV_IOCTL_MAGIC, nr),
        _ => return Err(Error::InvalidArgument(format!("unknown RM escape {nr:#x}"))),
    };
    // SAFETY: forwarded to the caller.
    unsafe { fd.ioctl(req, p) }
}

/// `NV_ESC_RM_ALLOC` with class parameters `params` (or none).
pub fn rm_alloc<P>(
    ctl: &Fd,
    client: NvHandle,
    parent: NvHandle,
    class: u32,
    params: Option<&mut P>,
) -> Result<NvHandle> {
    let (ptr, size) = match params {
        Some(p) => (std::ptr::from_mut::<P>(p) as u64, size_of::<P>() as u32),
        None => (0, 0),
    };
    let mut a = NVOS21_PARAMETERS {
        hRoot: client,
        hObjectParent: parent,
        hClass: class,
        pAllocParms: ptr,
        paramsSize: size,
        ..Default::default()
    };
    // SAFETY: NVOS21 for NV_ESC_RM_ALLOC; params outlive the call.
    unsafe { rm_escape(ctl, NV_ESC_RM_ALLOC, &mut a)? };
    status(&format!("alloc class {class:#x}"), a.status)?;
    Ok(a.hObjectNew)
}

/// `NV_ESC_RM_CONTROL`.
pub fn rm_control<P>(
    ctl: &Fd,
    client: NvHandle,
    object: NvHandle,
    cmd: u32,
    params: &mut P,
) -> Result<()> {
    let mut a = NVOS54_PARAMETERS {
        hClient: client,
        hObject: object,
        cmd,
        params: std::ptr::from_mut::<P>(params) as u64,
        paramsSize: size_of::<P>() as u32,
        ..Default::default()
    };
    // SAFETY: NVOS54 for NV_ESC_RM_CONTROL; params outlive the call.
    unsafe { rm_escape(ctl, NV_ESC_RM_CONTROL, &mut a)? };
    status(&format!("control {cmd:#x}"), a.status)
}

/// `NV_ESC_RM_FREE`.
pub fn rm_free(ctl: &Fd, client: NvHandle, parent: NvHandle, object: NvHandle) -> Result<()> {
    let mut a = NVOS00_PARAMETERS {
        hRoot: client,
        hObjectParent: parent,
        hObjectOld: object,
        status: 0,
    };
    // SAFETY: NVOS00 for NV_ESC_RM_FREE.
    unsafe { rm_escape(ctl, NV_ESC_RM_FREE, &mut a)? };
    status("free", a.status)
}

/// UVM ioctl: raw command number, parameter struct ends with `rmStatus` at `status_off`.
///
/// # Safety
/// `T` must be the parameter struct of `cmd`.
pub unsafe fn uvm<T>(fd: &Fd, cmd: u64, p: &mut T, rm_status: impl Fn(&T) -> u32) -> Result<()> {
    // SAFETY: forwarded to the caller.
    unsafe { fd.ioctl(cmd, p)? };
    status(&format!("uvm {cmd}"), rm_status(p))
}
