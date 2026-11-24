//! Pipeline creation and dynamic state.

use super::{Area, Capability, Spec, Support, core, ext, version};
use crate::vulkan::ApiVersion as V;
use crate::vulkan::features::Chain;

const AREA: Area = Area::Pipeline;

pub const EXTENDED_DYNAMIC_STATE: Spec = Spec {
    area: AREA,
    support: Support::TESTED,
    needs: &[],
    name: "extended_dynamic_state",
    doc: "cull mode, front face, topology, depth state ... set per draw",
    routes: &[
        version(V::V1_3),
        ext(
            c"VK_EXT_extended_dynamic_state",
            Some(Chain::ExtendedDynamicState),
            &["extendedDynamicState"],
        ),
    ],
};

pub const EXTENDED_DYNAMIC_STATE2: Spec = Spec {
    area: AREA,
    support: Support::TESTED,
    needs: &[],
    name: "extended_dynamic_state2",
    doc: "rasterizer discard, depth bias and primitive restart set per draw (logic op and \
          patch control points are separate bits)",
    routes: &[
        version(V::V1_3),
        ext(
            c"VK_EXT_extended_dynamic_state2",
            Some(Chain::ExtendedDynamicState2),
            &["extendedDynamicState2"],
        ),
    ],
};

pub const GRAPHICS_PIPELINE_LIBRARY: Spec = Spec {
    area: AREA,
    support: Support::TESTED,
    needs: &[],
    name: "graphics_pipeline_library",
    doc: "pipelines linked from precompiled parts (fewer compile stalls)",
    routes: &[ext(
        c"VK_EXT_graphics_pipeline_library",
        Some(Chain::GraphicsPipelineLibrary),
        &["graphicsPipelineLibrary"],
    )],
};

pub const SHADER_OBJECT: Spec = Spec {
    area: AREA,
    support: Support::LINUX_TESTED,
    needs: &[Capability::DynamicRendering],
    name: "shader_object",
    doc: "shaders bound without pipelines, all state dynamic",
    routes: &[ext(
        c"VK_EXT_shader_object",
        Some(Chain::ShaderObject),
        &["shaderObject"],
    )],
};

pub const MAINTENANCE4: Spec = Spec {
    area: AREA,
    support: Support::TESTED,
    needs: &[],
    name: "maintenance4",
    doc: "maintenance4: LocalSizeId, memory requirements without objects, ...",
    routes: &[
        core(V::V1_3, Chain::Vulkan13, &["maintenance4"]),
        ext(
            c"VK_KHR_maintenance4",
            Some(Chain::Maintenance4),
            &["maintenance4"],
        ),
    ],
};

pub const MAINTENANCE5: Spec = Spec {
    area: AREA,
    support: Support::TESTED,
    needs: &[],
    name: "maintenance5",
    doc: "maintenance5: pipelines from shader module infos, vkCmdBindIndexBuffer2, ...",
    routes: &[
        core(V::V1_4, Chain::Vulkan14, &["maintenance5"]),
        ext(
            c"VK_KHR_maintenance5",
            Some(Chain::Maintenance5),
            &["maintenance5"],
        ),
    ],
};

pub const MAINTENANCE6: Spec = Spec {
    area: AREA,
    support: Support::TESTED,
    needs: &[],
    name: "maintenance6",
    doc: "maintenance6: vkCmdBindDescriptorSets2, null index buffers, ...",
    routes: &[
        core(V::V1_4, Chain::Vulkan14, &["maintenance6"]),
        ext(
            c"VK_KHR_maintenance6",
            Some(Chain::Maintenance6),
            &["maintenance6"],
        ),
    ],
};
