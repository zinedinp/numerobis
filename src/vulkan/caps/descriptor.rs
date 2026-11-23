//! Descriptor binding.

use super::{Area, Spec, core, ext};
use crate::vulkan::ApiVersion as V;
use crate::vulkan::features::{self, Chain};

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
const INDEXING_DEPS: &[&std::ffi::CStr] = &[c"VK_KHR_maintenance3"];

const AREA: Area = Area::Descriptor;

pub const BINDLESS_TEXTURES: Spec = Spec {
    area: AREA,
    name: "bindless_textures",
    doc: "unsized, partly bound image arrays indexed per fragment",
    routes: &[
        core(V::V1_2, Chain::Vulkan12, TEXTURES),
        ext(
            INDEXING,
            INDEXING_DEPS,
            Some(Chain::DescriptorIndexing),
            TEXTURES,
        ),
    ],
};

pub const BINDLESS_BUFFERS: Spec = Spec {
    area: AREA,
    name: "bindless_buffers",
    doc: "unsized, partly bound storage buffer arrays indexed per invocation",
    routes: &[
        core(V::V1_2, Chain::Vulkan12, BUFFERS),
        ext(
            INDEXING,
            INDEXING_DEPS,
            Some(Chain::DescriptorIndexing),
            BUFFERS,
        ),
    ],
};

pub const PUSH_DESCRIPTOR: Spec = Spec {
    area: AREA,
    name: "push_descriptor",
    doc: "descriptors written into the command buffer, no sets",
    routes: &[
        core(V::V1_4, Chain::Vulkan14, &["pushDescriptor"]),
        ext(
            c"VK_KHR_push_descriptor",
            &[features::GET_PHYSICAL_DEVICE_PROPERTIES2],
            None,
            &[],
        ),
    ],
};
