//! Render passes, sampling and presentation.

use super::{Area, Spec, core, core10, ext};
use crate::vulkan::ApiVersion as V;
use crate::vulkan::features::{self, Chain};

const LATEST_READY: &[&str] = &["presentModeFifoLatestReady"];

const AREA: Area = Area::Render;

pub const DYNAMIC_RENDERING: Spec = Spec {
    area: AREA,
    name: "dynamic_rendering",
    doc: "vkCmdBeginRendering, no render pass or framebuffer objects",
    routes: &[
        core(V::V1_3, Chain::Vulkan13, &["dynamicRendering"]),
        ext(
            c"VK_KHR_dynamic_rendering",
            &[
                c"VK_KHR_depth_stencil_resolve",
                c"VK_KHR_create_renderpass2",
            ],
            Some(Chain::DynamicRendering),
            &["dynamicRendering"],
        ),
    ],
};

pub const SAMPLER_FILTER_MINMAX: Spec = Spec {
    area: AREA,
    name: "sampler_filter_minmax",
    doc: "min/max reduction samplers (one-pass depth pyramids)",
    routes: &[
        core(V::V1_2, Chain::Vulkan12, &["samplerFilterMinmax"]),
        ext(
            c"VK_EXT_sampler_filter_minmax",
            &[features::GET_PHYSICAL_DEVICE_PROPERTIES2],
            None,
            &[],
        ),
    ],
};

pub const SAMPLER_ANISOTROPY: Spec = Spec {
    area: AREA,
    name: "sampler_anisotropy",
    doc: "anisotropic texture filtering",
    routes: &[core10(&["samplerAnisotropy"])],
};

pub const FIFO_LATEST_READY: Spec = Spec {
    area: AREA,
    name: "fifo_latest_ready",
    doc: "FIFO_LATEST_READY present mode (needs VK_KHR_swapchain, which the host enables)",
    routes: &[
        ext(
            c"VK_KHR_present_mode_fifo_latest_ready",
            &[],
            Some(Chain::FifoLatestReady),
            LATEST_READY,
        ),
        ext(
            c"VK_EXT_present_mode_fifo_latest_ready",
            &[],
            Some(Chain::FifoLatestReady),
            LATEST_READY,
        ),
    ],
};
