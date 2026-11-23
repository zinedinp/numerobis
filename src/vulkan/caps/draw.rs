//! Draw submission.

use super::{Area, Spec, core, core10, ext};
use crate::vulkan::ApiVersion as V;
use crate::vulkan::features::Chain;

const AREA: Area = Area::Draw;

pub const MULTI_DRAW_INDIRECT: Spec = Spec {
    area: AREA,
    name: "multi_draw_indirect",
    doc: "many indirect draws per call, each with its own firstInstance",
    routes: &[core10(&["multiDrawIndirect", "drawIndirectFirstInstance"])],
};

pub const SHADER_DRAW_PARAMETERS: Spec = Spec {
    area: AREA,
    name: "shader_draw_parameters",
    doc: "gl_DrawID, gl_BaseInstance and gl_BaseVertex in shaders",
    routes: &[
        core(V::V1_2, Chain::Vulkan11, &["shaderDrawParameters"]),
        core(
            V::V1_1,
            Chain::ShaderDrawParameters,
            &["shaderDrawParameters"],
        ),
        ext(c"VK_KHR_shader_draw_parameters", &[], None, &[]),
    ],
};

pub const DRAW_INDIRECT_COUNT: Spec = Spec {
    area: AREA,
    name: "draw_indirect_count",
    doc: "draw counts the GPU writes (vkCmdDrawIndexedIndirectCount)",
    routes: &[
        core(V::V1_2, Chain::Vulkan12, &["drawIndirectCount"]),
        ext(c"VK_KHR_draw_indirect_count", &[], None, &[]),
    ],
};

pub const INDEX_TYPE_UINT8: Spec = Spec {
    area: AREA,
    name: "index_type_uint8",
    doc: "8-bit index buffers",
    routes: &[
        core(V::V1_4, Chain::Vulkan14, &["indexTypeUint8"]),
        ext(
            c"VK_KHR_index_type_uint8",
            &[],
            Some(Chain::IndexTypeUint8),
            &["indexTypeUint8"],
        ),
        ext(
            c"VK_EXT_index_type_uint8",
            &[],
            Some(Chain::IndexTypeUint8),
            &["indexTypeUint8"],
        ),
    ],
};

pub const MESH_SHADER: Spec = Spec {
    area: AREA,
    name: "mesh_shader",
    doc: "mesh shaders (VK_EXT_mesh_shader); task shaders are a separate bit",
    routes: &[ext(
        c"VK_EXT_mesh_shader",
        &[c"VK_KHR_spirv_1_4", c"VK_KHR_shader_float_controls"],
        Some(Chain::MeshShader),
        &["meshShader"],
    )],
};
