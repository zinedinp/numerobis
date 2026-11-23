//! Shader arithmetic and subgroups.

use super::{Area, Op, Route, Spec, core, core10, ext};
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

pub const SPECS: &[Spec] = &[
    Spec {
        op: Op::Int16,
        area: Area::Shader,
        name: "int16",
        doc: "16-bit integers in shader arithmetic",
        routes: &[core10(&["shaderInt16"])],
    },
    Spec {
        op: Op::Int64,
        area: Area::Shader,
        name: "int64",
        doc: "64-bit integers in shader arithmetic",
        routes: &[core10(&["shaderInt64"])],
    },
    Spec {
        op: Op::Float16,
        area: Area::Shader,
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
    },
    Spec {
        op: Op::Int8,
        area: Area::Shader,
        name: "int8",
        doc: "8-bit integers in shader arithmetic",
        routes: &[
            core(V::V1_2, Chain::Vulkan12, &["shaderInt8"]),
            ext(FLOAT16_INT8, &[], Some(Chain::Float16Int8), &["shaderInt8"]),
        ],
    },
    Spec {
        op: Op::SubgroupVote,
        area: Area::Shader,
        name: "subgroup_vote",
        doc: "subgroupAll/Any in compute shaders",
        routes: &[in_compute(subgroup::op::BASIC | subgroup::op::VOTE)],
    },
    Spec {
        op: Op::SubgroupArithmetic,
        area: Area::Shader,
        name: "subgroup_arithmetic",
        doc: "subgroupAdd/Min/Max and the scans in compute shaders",
        routes: &[in_compute(subgroup::op::BASIC | subgroup::op::ARITHMETIC)],
    },
    Spec {
        op: Op::SubgroupBallot,
        area: Area::Shader,
        name: "subgroup_ballot",
        doc: "subgroupBallot and broadcasts in compute shaders",
        routes: &[in_compute(subgroup::op::BASIC | subgroup::op::BALLOT)],
    },
    Spec {
        op: Op::SubgroupShuffle,
        area: Area::Shader,
        name: "subgroup_shuffle",
        doc: "subgroupShuffle in compute shaders",
        routes: &[in_compute(subgroup::op::BASIC | subgroup::op::SHUFFLE)],
    },
    Spec {
        op: Op::SubgroupSizeControl,
        area: Area::Shader,
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
    },
    Spec {
        op: Op::DemoteToHelperInvocation,
        area: Area::Shader,
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
    },
    Spec {
        op: Op::IntegerDotProduct,
        area: Area::Shader,
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
    },
];
