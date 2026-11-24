//! Buffers, memory layouts and residency.

use super::{Area, Spec, Support, core, core10, ext};
use crate::vulkan::ApiVersion as V;
use crate::vulkan::features::Chain;

const AREA: Area = Area::Memory;

pub const ROBUST_BUFFER_ACCESS: Spec = Spec {
    area: AREA,
    support: Support::WINDOWS_TESTED,
    needs: &[],
    name: "robust_buffer_access",
    doc: "out-of-bounds buffer reads give zeros instead of faulting",
    routes: &[core10(&["robustBufferAccess"])],
};

pub const STORAGE_16BIT: Spec = Spec {
    area: AREA,
    support: Support::WINDOWS_TESTED,
    needs: &[],
    name: "storage_16bit",
    doc: "16-bit types in storage buffers",
    routes: &[
        core(V::V1_2, Chain::Vulkan11, &["storageBuffer16BitAccess"]),
        core(V::V1_1, Chain::Storage16Bit, &["storageBuffer16BitAccess"]),
        ext(
            c"VK_KHR_16bit_storage",
            Some(Chain::Storage16Bit),
            &["storageBuffer16BitAccess"],
        ),
    ],
};

pub const STORAGE_8BIT: Spec = Spec {
    area: AREA,
    support: Support::WINDOWS_TESTED,
    needs: &[],
    name: "storage_8bit",
    doc: "8-bit types in storage buffers",
    routes: &[
        core(V::V1_2, Chain::Vulkan12, &["storageBuffer8BitAccess"]),
        ext(
            c"VK_KHR_8bit_storage",
            Some(Chain::Storage8Bit),
            &["storageBuffer8BitAccess"],
        ),
    ],
};

pub const SCALAR_BLOCK_LAYOUT: Spec = Spec {
    area: AREA,
    support: Support::WINDOWS_TESTED,
    needs: &[],
    name: "scalar_block_layout",
    doc: "C-like (scalar) buffer layouts, no std430 padding",
    routes: &[
        core(V::V1_2, Chain::Vulkan12, &["scalarBlockLayout"]),
        ext(
            c"VK_EXT_scalar_block_layout",
            Some(Chain::ScalarBlockLayout),
            &["scalarBlockLayout"],
        ),
    ],
};

pub const BUFFER_DEVICE_ADDRESS: Spec = Spec {
    area: AREA,
    support: Support::WINDOWS_TESTED,
    needs: &[],
    name: "buffer_device_address",
    doc: "64-bit GPU pointers to buffers in shaders",
    routes: &[
        core(V::V1_2, Chain::Vulkan12, &["bufferDeviceAddress"]),
        ext(
            c"VK_KHR_buffer_device_address",
            Some(Chain::BufferDeviceAddress),
            &["bufferDeviceAddress"],
        ),
    ],
};

pub const MEMORY_PRIORITY: Spec = Spec {
    area: AREA,
    support: Support::WINDOWS_TESTED,
    needs: &[],
    name: "memory_priority",
    doc: "allocation priorities for the driver's eviction",
    routes: &[ext(
        c"VK_EXT_memory_priority",
        Some(Chain::MemoryPriority),
        &["memoryPriority"],
    )],
};

pub const PAGEABLE_DEVICE_LOCAL_MEMORY: Spec = Spec {
    area: AREA,
    support: Support::WINDOWS_TESTED,
    needs: &[],
    name: "pageable_device_local_memory",
    doc: "VRAM can spill to system memory instead of failing allocations",
    routes: &[ext(
        c"VK_EXT_pageable_device_local_memory",
        Some(Chain::PageableDeviceLocalMemory),
        &["pageableDeviceLocalMemory"],
    )],
};

pub const MEMORY_BUDGET: Spec = Spec {
    area: AREA,
    support: Support::WINDOWS_TESTED,
    needs: &[],
    name: "memory_budget",
    doc: "per-heap budget and usage (VkPhysicalDeviceMemoryBudgetPropertiesEXT)",
    routes: &[ext(c"VK_EXT_memory_budget", None, &[])],
};

pub const HOST_IMAGE_COPY: Spec = Spec {
    area: AREA,
    support: Support::WINDOWS_TESTED,
    needs: &[],
    name: "host_image_copy",
    doc: "image uploads and downloads from the CPU, no staging buffer or queue",
    routes: &[
        core(V::V1_4, Chain::Vulkan14, &["hostImageCopy"]),
        ext(
            c"VK_EXT_host_image_copy",
            Some(Chain::HostImageCopy),
            &["hostImageCopy"],
        ),
    ],
};

pub const SPARSE_RESIDENCY: Spec = Spec {
    area: AREA,
    support: Support::WINDOWS_TESTED,
    needs: &[],
    name: "sparse_residency",
    doc: "sparse buffers and 2D images, partly resident and aliased (Direct3D's tiled resources)",
    routes: &[core10(&[
        "sparseBinding",
        "sparseResidencyBuffer",
        "sparseResidencyImage2D",
        "sparseResidencyAliased",
    ])],
};
