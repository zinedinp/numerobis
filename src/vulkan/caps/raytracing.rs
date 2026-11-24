//! Hardware raytracing: acceleration structures, raytracing pipelines and
//! inline ray queries, each its own extension and its own block of
//! hardware.

use super::{Area, Capability, Spec, Support, ext};
use crate::vulkan::features::Chain;

const AREA: Area = Area::RayTracing;

pub const ACCELERATION_STRUCTURE: Spec = Spec {
    area: AREA,
    support: Support::TESTED,
    needs: &[],
    name: "acceleration_structure",
    doc: "bottom- and top-level acceleration structures built on the GPU",
    routes: &[ext(
        c"VK_KHR_acceleration_structure",
        Some(Chain::AccelerationStructure),
        &["accelerationStructure"],
    )],
};

pub const RAY_TRACING_PIPELINE: Spec = Spec {
    area: AREA,
    support: Support::TESTED,
    needs: &[Capability::AccelerationStructure],
    name: "ray_tracing_pipeline",
    doc: "raytracing pipelines and vkCmdTraceRaysKHR (raygen, hit, miss shaders)",
    routes: &[ext(
        c"VK_KHR_ray_tracing_pipeline",
        Some(Chain::RayTracingPipeline),
        &["rayTracingPipeline"],
    )],
};

pub const RAY_TRAVERSAL_PRIMITIVE_CULLING: Spec = Spec {
    area: AREA,
    support: Support::TESTED,
    needs: &[Capability::RayTracingPipeline],
    name: "ray_traversal_primitive_culling",
    doc: "skipping triangles or AABBs during traversal (what DXR 1.1 needs beside ray queries)",
    routes: &[ext(
        c"VK_KHR_ray_tracing_pipeline",
        Some(Chain::RayTracingPipeline),
        &["rayTraversalPrimitiveCulling"],
    )],
};

pub const RAY_QUERY: Spec = Spec {
    area: AREA,
    support: Support::TESTED,
    needs: &[Capability::AccelerationStructure],
    name: "ray_query",
    doc: "inline ray queries from any shader stage",
    routes: &[ext(
        c"VK_KHR_ray_query",
        Some(Chain::RayQuery),
        &["rayQuery"],
    )],
};

pub const RAY_TRACING_MAINTENANCE1: Spec = Spec {
    area: AREA,
    support: Support::TESTED,
    needs: &[Capability::AccelerationStructure],
    name: "ray_tracing_maintenance1",
    doc: "acceleration structure size queries, vkCmdTraceRaysIndirect2KHR",
    routes: &[ext(
        c"VK_KHR_ray_tracing_maintenance1",
        Some(Chain::RayTracingMaintenance1),
        &["rayTracingMaintenance1"],
    )],
};

pub const OPACITY_MICROMAP: Spec = Spec {
    area: AREA,
    support: Support::UNTESTED,
    needs: &[Capability::AccelerationStructure],
    name: "opacity_micromap",
    doc: "opacity micromaps: alpha-tested geometry without any-hit shaders",
    routes: &[
        ext(
            c"VK_KHR_opacity_micromap",
            Some(Chain::OpacityMicromap),
            &["micromap"],
        ),
        ext(
            c"VK_EXT_opacity_micromap",
            Some(Chain::OpacityMicromapExt),
            &["micromap"],
        ),
    ],
};

pub const RAY_TRACING_INVOCATION_REORDER: Spec = Spec {
    area: AREA,
    support: Support::WINDOWS_TESTED,
    needs: &[Capability::RayTracingPipeline],
    name: "ray_tracing_invocation_reorder",
    doc: "shader execution reordering of hit shaders",
    routes: &[ext(
        c"VK_EXT_ray_tracing_invocation_reorder",
        Some(Chain::RayTracingInvocationReorder),
        &["rayTracingInvocationReorder"],
    )],
};
