//! Descriptor binding.

use super::{Area, Op, Spec, core, ext};
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
const INDEXING_DEPS: &[&std::ffi::CStr] = &[c"VK_KHR_maintenance3"];

pub const SPECS: &[Spec] = &[
    Spec {
        op: Op::BindlessTextures,
        area: Area::Descriptor,
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
    },
    Spec {
        op: Op::BindlessBuffers,
        area: Area::Descriptor,
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
    },
    Spec {
        op: Op::PushDescriptor,
        area: Area::Descriptor,
        name: "push_descriptor",
        doc: "descriptors written into the command buffer, no sets",
        routes: &[
            core(V::V1_4, Chain::Vulkan14, &["pushDescriptor"]),
            ext(c"VK_KHR_push_descriptor", &[], None, &[]),
        ],
    },
];
