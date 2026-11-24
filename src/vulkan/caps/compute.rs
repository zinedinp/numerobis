//! Compute and matrix engines: shader types and cooperative matrices, what
//! numerobis's comfort Vulkan backend builds on.

use super::{Area, Capability, Spec, Support, core, ext};
use crate::vulkan::ApiVersion as V;
use crate::vulkan::features::Chain;

const AREA: Area = Area::Compute;

const MODEL: &[&str] = &["vulkanMemoryModel"];

pub const VULKAN_MEMORY_MODEL: Spec = Spec {
    area: AREA,
    support: Support::WINDOWS_TESTED,
    needs: &[],
    name: "vulkan_memory_model",
    doc: "the Vulkan memory model (acquire/release, availability and visibility)",
    routes: &[
        core(V::V1_2, Chain::Vulkan12, MODEL),
        ext(
            c"VK_KHR_vulkan_memory_model",
            Some(Chain::VulkanMemoryModel),
            MODEL,
        ),
    ],
};

pub const COOPERATIVE_MATRIX: Spec = Spec {
    area: AREA,
    support: Support::WINDOWS_TESTED,
    needs: &[],
    name: "cooperative_matrix",
    doc: "subgroup matrix multiply-add (tensor cores, XMX, WMMA); shapes and types from \
          vkGetPhysicalDeviceCooperativeMatrixPropertiesKHR",
    routes: &[ext(
        c"VK_KHR_cooperative_matrix",
        Some(Chain::CooperativeMatrix),
        &["cooperativeMatrix"],
    )],
};

pub const BFLOAT16: Spec = Spec {
    area: AREA,
    support: Support::UNTESTED,
    needs: &[],
    name: "bfloat16",
    doc: "bfloat16 as a shader type",
    routes: &[ext(
        c"VK_KHR_shader_bfloat16",
        Some(Chain::ShaderBfloat16),
        &["shaderBFloat16Type"],
    )],
};

pub const BFLOAT16_COOPERATIVE_MATRIX: Spec = Spec {
    area: AREA,
    support: Support::UNTESTED,
    needs: &[Capability::CooperativeMatrix],
    name: "bfloat16_cooperative_matrix",
    doc: "bfloat16 cooperative matrices",
    routes: &[ext(
        c"VK_KHR_shader_bfloat16",
        Some(Chain::ShaderBfloat16),
        &["shaderBFloat16Type", "shaderBFloat16CooperativeMatrix"],
    )],
};

pub const FLOAT8: Spec = Spec {
    area: AREA,
    support: Support::UNTESTED,
    needs: &[],
    name: "float8",
    doc: "E4M3 and E5M2 as shader types",
    routes: &[ext(
        c"VK_EXT_shader_float8",
        Some(Chain::ShaderFloat8),
        &["shaderFloat8"],
    )],
};

pub const FLOAT8_COOPERATIVE_MATRIX: Spec = Spec {
    area: AREA,
    support: Support::UNTESTED,
    needs: &[Capability::CooperativeMatrix],
    name: "float8_cooperative_matrix",
    doc: "E4M3 and E5M2 cooperative matrices",
    routes: &[ext(
        c"VK_EXT_shader_float8",
        Some(Chain::ShaderFloat8),
        &["shaderFloat8", "shaderFloat8CooperativeMatrix"],
    )],
};
