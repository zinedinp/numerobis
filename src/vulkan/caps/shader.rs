//! Shader arithmetic and subgroups.

use super::{Area, Route, Spec, core, core10, ext};
use crate::vulkan::ApiVersion as V;
use crate::vulkan::features::Chain;
use crate::vulkan::subgroup;

const fn in_compute(operations: u32) -> Route {
    Route::Subgroup {
        since: V::V1_1,
        operations,
        stages: subgroup::stage::COMPUTE,
    }
}

const FLOAT16_INT8: &std::ffi::CStr = c"VK_KHR_shader_float16_int8";

const AREA: Area = Area::Shader;

pub const INT16: Spec = Spec {
    area: AREA,
    name: "int16",
    doc: "16-bit integers in shader arithmetic",
    routes: &[core10(&["shaderInt16"])],
};

pub const INT64: Spec = Spec {
    area: AREA,
    name: "int64",
    doc: "64-bit integers in shader arithmetic",
    routes: &[core10(&["shaderInt64"])],
};

pub const FLOAT16: Spec = Spec {
    area: AREA,
    name: "float16",
    doc: "16-bit floats in shader arithmetic",
    routes: &[
        core(V::V1_2, Chain::Vulkan12, &["shaderFloat16"]),
        ext(
            FLOAT16_INT8,
            &[],
            Some(Chain::Float16Int8),
            &["shaderFloat16"],
        ),
    ],
};

pub const INT8: Spec = Spec {
    area: AREA,
    name: "int8",
    doc: "8-bit integers in shader arithmetic",
    routes: &[
        core(V::V1_2, Chain::Vulkan12, &["shaderInt8"]),
        ext(FLOAT16_INT8, &[], Some(Chain::Float16Int8), &["shaderInt8"]),
    ],
};

pub const SUBGROUP_VOTE: Spec = Spec {
    area: AREA,
    name: "subgroup_vote",
    doc: "subgroupAll/Any in compute shaders",
    routes: &[in_compute(subgroup::op::BASIC | subgroup::op::VOTE)],
};

pub const SUBGROUP_ARITHMETIC: Spec = Spec {
    area: AREA,
    name: "subgroup_arithmetic",
    doc: "subgroupAdd/Min/Max and the scans in compute shaders",
    routes: &[in_compute(subgroup::op::BASIC | subgroup::op::ARITHMETIC)],
};

pub const SUBGROUP_BALLOT: Spec = Spec {
    area: AREA,
    name: "subgroup_ballot",
    doc: "subgroupBallot and broadcasts in compute shaders",
    routes: &[in_compute(subgroup::op::BASIC | subgroup::op::BALLOT)],
};

pub const SUBGROUP_SHUFFLE: Spec = Spec {
    area: AREA,
    name: "subgroup_shuffle",
    doc: "subgroupShuffle in compute shaders",
    routes: &[in_compute(subgroup::op::BASIC | subgroup::op::SHUFFLE)],
};

pub const SUBGROUP_SIZE_CONTROL: Spec = Spec {
    area: AREA,
    name: "subgroup_size_control",
    doc: "a pipeline can pick its subgroup size",
    routes: &[
        core(V::V1_3, Chain::Vulkan13, &["subgroupSizeControl"]),
        ext(
            c"VK_EXT_subgroup_size_control",
            &[],
            Some(Chain::SubgroupSizeControl),
            &["subgroupSizeControl"],
        ),
    ],
};

pub const DEMOTE_TO_HELPER_INVOCATION: Spec = Spec {
    area: AREA,
    name: "demote_to_helper_invocation",
    doc: "discard that keeps derivatives valid for the quad",
    routes: &[
        core(
            V::V1_3,
            Chain::Vulkan13,
            &["shaderDemoteToHelperInvocation"],
        ),
        ext(
            c"VK_EXT_shader_demote_to_helper_invocation",
            &[],
            Some(Chain::DemoteToHelperInvocation),
            &["shaderDemoteToHelperInvocation"],
        ),
    ],
};

pub const INTEGER_DOT_PRODUCT: Spec = Spec {
    area: AREA,
    name: "integer_dot_product",
    doc: "packed 8-bit integer dot products",
    routes: &[
        core(V::V1_3, Chain::Vulkan13, &["shaderIntegerDotProduct"]),
        ext(
            c"VK_KHR_shader_integer_dot_product",
            &[],
            Some(Chain::IntegerDotProduct),
            &["shaderIntegerDotProduct"],
        ),
    ],
};
