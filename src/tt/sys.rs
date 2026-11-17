//! tt-kmd UAPI (`/dev/tenstorrent/N`, `ioctl.h` of tt-kmd 2.x). Every struct
//! carries a size assert from a C `sizeof` dump of the header.

#![allow(non_camel_case_types, dead_code)]

use crate::runtime::error::Result;
use crate::runtime::os::{Fd, ioc};

pub const MAGIC: u8 = 0xFA;
pub const GET_DEVICE_INFO: u8 = 0;
pub const GET_DRIVER_INFO: u8 = 5;
pub const ALLOCATE_TLB: u8 = 11;
pub const FREE_TLB: u8 = 12;
pub const CONFIGURE_TLB: u8 = 13;
pub const NOC_READ: u8 = 18;
pub const NOC_WRITE: u8 = 19;

#[repr(C)]
#[derive(Debug, Default, Clone, Copy)]
pub struct get_device_info {
    pub in_output_size_bytes: u32,
    pub output_size_bytes: u32,
    pub vendor_id: u16,
    pub device_id: u16,
    pub subsystem_vendor_id: u16,
    pub subsystem_id: u16,
    pub bus_dev_fn: u16,
    pub max_dma_buf_size_log2: u16,
    pub pci_domain: u16,
    pub reserved: u16,
}

#[repr(C)]
#[derive(Debug, Default, Clone, Copy)]
pub struct get_driver_info {
    pub in_output_size_bytes: u32,
    pub output_size_bytes: u32,
    pub driver_version: u32,
    pub major: u8,
    pub minor: u8,
    pub patch: u8,
    pub reserved: u8,
}

#[repr(C)]
#[derive(Debug, Default, Clone, Copy)]
pub struct allocate_tlb {
    pub size: u64,
    pub reserved: u64,
    pub id: u32,
    pub reserved0: u32,
    pub mmap_offset_uc: u64,
    pub mmap_offset_wc: u64,
    pub reserved1: u64,
}

#[repr(C)]
#[derive(Debug, Default, Clone, Copy)]
pub struct free_tlb {
    pub id: u32,
}

#[repr(C)]
#[derive(Debug, Default, Clone, Copy)]
pub struct noc_tlb_config {
    pub addr: u64,
    pub x_end: u16,
    pub y_end: u16,
    pub x_start: u16,
    pub y_start: u16,
    pub noc: u8,
    pub mcast: u8,
    pub ordering: u8,
    pub linked: u8,
    pub static_vc: u8,
    pub reserved0: [u8; 3],
    pub reserved1: [u32; 2],
}

/// `ordering`: 0 relaxed, 1 strict (in order, non-posted), 2 posted.
pub const ORDER_STRICT: u8 = 1;
pub const ORDER_POSTED: u8 = 2;

#[repr(C)]
#[derive(Debug, Default, Clone, Copy)]
pub struct configure_tlb {
    pub id: u32,
    pub reserved: u32,
    pub config: noc_tlb_config,
    pub out_reserved: u64,
}

#[repr(C)]
#[derive(Debug, Default, Clone, Copy)]
pub struct noc_io {
    pub argsz: u32,
    pub flags: u32,
    pub x: u16,
    pub y: u16,
    pub noc: u8,
    pub width: u8,
    pub reserved0: [u8; 2],
    pub addr: u64,
    pub value: u64,
}

const _: () = {
    assert!(size_of::<get_device_info>() == 24);
    assert!(size_of::<get_driver_info>() == 16);
    assert!(size_of::<allocate_tlb>() == 48);
    assert!(size_of::<free_tlb>() == 4);
    assert!(size_of::<noc_tlb_config>() == 32);
    assert!(core::mem::offset_of!(noc_tlb_config, static_vc) == 20);
    assert!(size_of::<configure_tlb>() == 48);
    assert!(size_of::<noc_io>() == 32);
};

/// `_IO(0xFA, nr)`: tt-kmd encodes no size; structs start with their own sizes.
pub fn call<T>(fd: &Fd, nr: u8, arg: &mut T) -> Result<()> {
    // SAFETY: every tt-kmd ioctl takes a pointer to its in/out struct.
    unsafe { fd.ioctl(ioc(0, MAGIC, nr, 0), arg) }
}
