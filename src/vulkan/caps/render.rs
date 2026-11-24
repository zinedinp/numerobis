//! Render passes, sampling and presentation.

use super::{Area, Spec, Support, core, core10, ext};
use crate::vulkan::ApiVersion as V;
use crate::vulkan::features::Chain;

const AREA: Area = Area::Render;

pub const DYNAMIC_RENDERING: Spec = Spec {
    area: AREA,
    support: Support::WINDOWS_TESTED,
    needs: &[],
    name: "dynamic_rendering",
    doc: "vkCmdBeginRendering, no render pass or framebuffer objects",
    routes: &[
        core(V::V1_3, Chain::Vulkan13, &["dynamicRendering"]),
        ext(
            c"VK_KHR_dynamic_rendering",
            Some(Chain::DynamicRendering),
            &["dynamicRendering"],
        ),
    ],
};

pub const SAMPLER_FILTER_MINMAX: Spec = Spec {
    area: AREA,
    support: Support::WINDOWS_TESTED,
    needs: &[],
    name: "sampler_filter_minmax",
    doc: "min/max reduction samplers (one-pass depth pyramids)",
    routes: &[
        core(V::V1_2, Chain::Vulkan12, &["samplerFilterMinmax"]),
        ext(c"VK_EXT_sampler_filter_minmax", None, &[]),
    ],
};

pub const SAMPLER_ANISOTROPY: Spec = Spec {
    area: AREA,
    support: Support::WINDOWS_TESTED,
    needs: &[],
    name: "sampler_anisotropy",
    doc: "anisotropic texture filtering",
    routes: &[core10(&["samplerAnisotropy"])],
};

pub const TEXTURE_COMPRESSION_BC: Spec = Spec {
    area: AREA,
    support: Support::WINDOWS_TESTED,
    needs: &[],
    name: "texture_compression_bc",
    doc: "BC1-BC7 textures (desktop GPUs)",
    routes: &[core10(&["textureCompressionBC"])],
};

pub const TEXTURE_COMPRESSION_ETC2: Spec = Spec {
    area: AREA,
    support: Support::WINDOWS_TESTED,
    needs: &[],
    name: "texture_compression_etc2",
    doc: "ETC2 and EAC textures (mobile GPUs)",
    routes: &[core10(&["textureCompressionETC2"])],
};

pub const TEXTURE_COMPRESSION_ASTC_LDR: Spec = Spec {
    area: AREA,
    support: Support::UNTESTED,
    needs: &[],
    name: "texture_compression_astc_ldr",
    doc: "ASTC LDR textures (mobile GPUs, Apple)",
    routes: &[core10(&["textureCompressionASTC_LDR"])],
};

const ASTC_HDR: &[&str] = &["textureCompressionASTC_HDR"];

pub const TEXTURE_COMPRESSION_ASTC_HDR: Spec = Spec {
    area: AREA,
    support: Support::UNTESTED,
    needs: &[],
    name: "texture_compression_astc_hdr",
    doc: "ASTC HDR textures",
    routes: &[
        core(V::V1_3, Chain::Vulkan13, ASTC_HDR),
        ext(
            c"VK_EXT_texture_compression_astc_hdr",
            Some(Chain::TextureCompressionAstcHdr),
            ASTC_HDR,
        ),
    ],
};

pub const FILL_MODE_NON_SOLID: Spec = Spec {
    area: AREA,
    support: Support::WINDOWS_TESTED,
    needs: &[],
    name: "fill_mode_non_solid",
    doc: "wireframe and point polygon modes (debug views)",
    routes: &[core10(&["fillModeNonSolid"])],
};

pub const DEPTH_CLAMP: Spec = Spec {
    area: AREA,
    support: Support::WINDOWS_TESTED,
    needs: &[],
    name: "depth_clamp",
    doc: "clamp depth instead of clipping (shadow pancaking)",
    routes: &[core10(&["depthClamp"])],
};

pub const WIDE_LINES: Spec = Spec {
    area: AREA,
    support: Support::WINDOWS_TESTED,
    needs: &[],
    name: "wide_lines",
    doc: "line widths other than 1.0",
    routes: &[core10(&["wideLines"])],
};
