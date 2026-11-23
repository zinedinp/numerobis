//! Buffers, memory layouts and residency.

use super::{Area, Op, Spec, core, core10, ext};
use crate::vulkan::ApiVersion as V;
use crate::vulkan::features::Chain;

const STORAGE_CLASS: &[&std::ffi::CStr] = &[c"VK_KHR_storage_buffer_storage_class"];

pub const SPECS: &[Spec] = &[
    Spec {
        op: Op::RobustBufferAccess,
        area: Area::Memory,
        name: "robust_buffer_access",
        doc: "out-of-bounds buffer reads give zeros instead of faulting",
        routes: &[core10(&["robustBufferAccess"])],
    },
    Spec {
        op: Op::Storage16Bit,
        area: Area::Memory,
        name: "storage_16bit",
        doc: "16-bit types in storage buffers",
        routes: &[
            core(V::V1_2, Chain::Vulkan11, &["storageBuffer16BitAccess"]),
            core(V::V1_1, Chain::Storage16Bit, &["storageBuffer16BitAccess"]),
            ext(
                c"VK_KHR_16bit_storage",
                STORAGE_CLASS,
                Some(Chain::Storage16Bit),
                &["storageBuffer16BitAccess"],
            ),
        ],
    },
    Spec {
        op: Op::Storage8Bit,
        area: Area::Memory,
        name: "storage_8bit",
        doc: "8-bit types in storage buffers",
        routes: &[
            core(V::V1_2, Chain::Vulkan12, &["storageBuffer8BitAccess"]),
            ext(
                c"VK_KHR_8bit_storage",
                STORAGE_CLASS,
                Some(Chain::Storage8Bit),
                &["storageBuffer8BitAccess"],
            ),
        ],
    },
    Spec {
        op: Op::ScalarBlockLayout,
        area: Area::Memory,
        name: "scalar_block_layout",
        doc: "C-like (scalar) buffer layouts, no std430 padding",
        routes: &[
            core(V::V1_2, Chain::Vulkan12, &["scalarBlockLayout"]),
            ext(
                c"VK_EXT_scalar_block_layout",
                &[],
                Some(Chain::ScalarBlockLayout),
                &["scalarBlockLayout"],
            ),
        ],
    },
    Spec {
        op: Op::BufferDeviceAddress,
        area: Area::Memory,
        name: "buffer_device_address",
        doc: "64-bit GPU pointers to buffers in shaders",
        routes: &[
            core(V::V1_2, Chain::Vulkan12, &["bufferDeviceAddress"]),
            ext(
                c"VK_KHR_buffer_device_address",
                &[c"VK_KHR_device_group"],
                Some(Chain::BufferDeviceAddress),
                &["bufferDeviceAddress"],
            ),
        ],
    },
    Spec {
        op: Op::MemoryPriority,
        area: Area::Memory,
        name: "memory_priority",
        doc: "allocation priorities for the driver's eviction",
        routes: &[ext(
            c"VK_EXT_memory_priority",
            &[],
            Some(Chain::MemoryPriority),
            &["memoryPriority"],
        )],
    },
    Spec {
        op: Op::PageableDeviceLocalMemory,
        area: Area::Memory,
        name: "pageable_device_local_memory",
        doc: "VRAM can spill to system memory instead of failing allocations",
        routes: &[ext(
            c"VK_EXT_pageable_device_local_memory",
            &[c"VK_EXT_memory_priority"],
            Some(Chain::PageableDeviceLocalMemory),
            &["pageableDeviceLocalMemory"],
        )],
    },
];
