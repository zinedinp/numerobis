//! Draw submission.

use super::{Area, Spec, Support, core, core10, ext};
use crate::vulkan::ApiVersion as V;
use crate::vulkan::features::Chain;

const AREA: Area = Area::Draw;

pub const MULTI_DRAW_INDIRECT: Spec = Spec {
    area: AREA,
    support: Support::WINDOWS_TESTED,
    needs: &[],
    name: "multi_draw_indirect",
    doc: "many indirect draws per call, each with its own firstInstance",
    routes: &[core10(&["multiDrawIndirect", "drawIndirectFirstInstance"])],
};

pub const SHADER_DRAW_PARAMETERS: Spec = Spec {
    area: AREA,
    support: Support::WINDOWS_TESTED,
    needs: &[],
    name: "shader_draw_parameters",
    doc: "gl_DrawID, gl_BaseInstance and gl_BaseVertex in shaders",
    routes: &[
        core(V::V1_2, Chain::Vulkan11, &["shaderDrawParameters"]),
        core(
            V::V1_1,
            Chain::ShaderDrawParameters,
            &["shaderDrawParameters"],
        ),
        ext(c"VK_KHR_shader_draw_parameters", None, &[]),
    ],
};

pub const DRAW_INDIRECT_COUNT: Spec = Spec {
    area: AREA,
    support: Support::WINDOWS_TESTED,
    needs: &[],
    name: "draw_indirect_count",
    doc: "draw counts the GPU writes (vkCmdDrawIndexedIndirectCount)",
    routes: &[
        core(V::V1_2, Chain::Vulkan12, &["drawIndirectCount"]),
        ext(c"VK_KHR_draw_indirect_count", None, &[]),
    ],
};

pub const INDEX_TYPE_UINT8: Spec = Spec {
    area: AREA,
    support: Support::WINDOWS_TESTED,
    needs: &[],
    name: "index_type_uint8",
    doc: "8-bit index buffers",
    routes: &[
        core(V::V1_4, Chain::Vulkan14, &["indexTypeUint8"]),
        ext(
            c"VK_KHR_index_type_uint8",
            Some(Chain::IndexTypeUint8),
            &["indexTypeUint8"],
        ),
        ext(
            c"VK_EXT_index_type_uint8",
            Some(Chain::IndexTypeUint8),
            &["indexTypeUint8"],
        ),
    ],
};

const DIVISOR: &[&str] = &["vertexAttributeInstanceRateDivisor"];

pub const VERTEX_ATTRIBUTE_DIVISOR: Spec = Spec {
    area: AREA,
    support: Support::WINDOWS_TESTED,
    needs: &[],
    name: "vertex_attribute_divisor",
    doc: "instanced vertex attributes that advance every N instances",
    routes: &[
        core(V::V1_4, Chain::Vulkan14, DIVISOR),
        ext(
            c"VK_KHR_vertex_attribute_divisor",
            Some(Chain::VertexAttributeDivisor),
            DIVISOR,
        ),
        ext(
            c"VK_EXT_vertex_attribute_divisor",
            Some(Chain::VertexAttributeDivisor),
            DIVISOR,
        ),
    ],
};
