//! Mesh and task shaders, which replace the vertex pipeline.

use super::{Area, Capability, Spec, Support, ext};
use crate::vulkan::features::Chain;

const AREA: Area = Area::Mesh;

pub const MESH_SHADER: Spec = Spec {
    area: AREA,
    support: Support::WINDOWS_TESTED,
    needs: &[],
    name: "mesh_shader",
    doc: "mesh shaders (VK_EXT_mesh_shader); task shaders are a separate bit",
    routes: &[ext(
        c"VK_EXT_mesh_shader",
        Some(Chain::MeshShader),
        &["meshShader"],
    )],
};

pub const TASK_SHADER: Spec = Spec {
    area: AREA,
    support: Support::WINDOWS_TESTED,
    needs: &[Capability::MeshShader],
    name: "task_shader",
    doc: "task shaders in front of mesh shaders (Direct3D's amplification shaders)",
    routes: &[ext(
        c"VK_EXT_mesh_shader",
        Some(Chain::MeshShader),
        &["taskShader"],
    )],
};
