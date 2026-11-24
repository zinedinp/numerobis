//! Descriptor binding.

use super::{Area, Capability, Spec, Support, core, ext};
use crate::vulkan::ApiVersion as V;
use crate::vulkan::features::Chain;

const TEXTURES: &[&str] = &[
    "runtimeDescriptorArray",
    "descriptorBindingPartiallyBound",
    "shaderSampledImageArrayNonUniformIndexing",
];

const BUFFERS: &[&str] = &[
    "runtimeDescriptorArray",
    "descriptorBindingPartiallyBound",
    "shaderStorageBufferArrayNonUniformIndexing",
];

const INDEXING: &std::ffi::CStr = c"VK_EXT_descriptor_indexing";

const AREA: Area = Area::Descriptor;

pub const BINDLESS_TEXTURES: Spec = Spec {
    area: AREA,
    support: Support::TESTED,
    needs: &[],
    name: "bindless_textures",
    doc: "unsized, partly bound image arrays indexed per fragment",
    routes: &[
        core(V::V1_2, Chain::Vulkan12, TEXTURES),
        ext(INDEXING, Some(Chain::DescriptorIndexing), TEXTURES),
    ],
};

pub const BINDLESS_BUFFERS: Spec = Spec {
    area: AREA,
    support: Support::TESTED,
    needs: &[],
    name: "bindless_buffers",
    doc: "unsized, partly bound storage buffer arrays indexed per invocation",
    routes: &[
        core(V::V1_2, Chain::Vulkan12, BUFFERS),
        ext(INDEXING, Some(Chain::DescriptorIndexing), BUFFERS),
    ],
};

pub const PUSH_DESCRIPTOR: Spec = Spec {
    area: AREA,
    support: Support::TESTED,
    needs: &[],
    name: "push_descriptor",
    doc: "descriptors written into the command buffer, no sets",
    routes: &[
        core(V::V1_4, Chain::Vulkan14, &["pushDescriptor"]),
        ext(c"VK_KHR_push_descriptor", None, &[]),
    ],
};

const NULL: &[&str] = &["nullDescriptor"];

pub const NULL_DESCRIPTOR: Spec = Spec {
    area: AREA,
    support: Support::TESTED,
    needs: &[],
    name: "null_descriptor",
    doc: "VK_NULL_HANDLE descriptors that read zeros (holes in bindless arrays)",
    routes: &[
        ext(c"VK_KHR_robustness2", Some(Chain::Robustness2), NULL),
        ext(c"VK_EXT_robustness2", Some(Chain::Robustness2), NULL),
    ],
};

pub const DESCRIPTOR_BUFFER: Spec = Spec {
    area: AREA,
    support: Support::TESTED,
    needs: &[Capability::BufferDeviceAddress],
    name: "descriptor_buffer",
    doc: "descriptors written into buffers by the host (deprecated by VK_EXT_descriptor_heap)",
    routes: &[ext(
        c"VK_EXT_descriptor_buffer",
        Some(Chain::DescriptorBuffer),
        &["descriptorBuffer"],
    )],
};
