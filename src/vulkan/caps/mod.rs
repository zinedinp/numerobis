//! The single capabilities a renderer asks for, one module per area, each
//! with its waterfall of [`Route`]s: the newest core version first, then the
//! extensions that offered it before.

use std::ffi::CStr;

use super::ApiVersion;
use super::features::Chain;

pub mod descriptor;
pub mod draw;
pub mod memory;
pub mod render;
pub mod shader;
pub mod sync;

/// One way to get an [`Op`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Route {
    /// Core from `since`: the `fields` of `chain` must all be on.
    Core {
        since: ApiVersion,
        chain: Chain,
        fields: &'static [&'static str],
    },
    /// Extension `name` (plus `deps`, which count as there once core). With
    /// a `chain`, its `fields` must be on too; without one, being offered is
    /// support.
    Extension {
        name: &'static CStr,
        deps: &'static [&'static CStr],
        chain: Option<Chain>,
        fields: &'static [&'static str],
    },
    /// Core from `since`: `VkPhysicalDeviceSubgroupProperties` must list all
    /// of `operations` for all of `stages`.
    Subgroup {
        since: ApiVersion,
        operations: u32,
        stages: u32,
    },
}

/// The area an op belongs to (its module).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Area {
    Draw,
    Memory,
    Shader,
    Descriptor,
    Sync,
    Render,
}

impl Area {
    #[must_use]
    pub const fn name(self) -> &'static str {
        match self {
            Self::Draw => "draw",
            Self::Memory => "memory",
            Self::Shader => "shader",
            Self::Descriptor => "descriptor",
            Self::Sync => "sync",
            Self::Render => "render",
        }
    }
}

/// What an op is and how to get it.
#[derive(Debug, Clone, Copy)]
pub struct Spec {
    pub op: Op,
    pub area: Area,
    /// Short snake-case name for logs (`draw_indirect_count`).
    pub name: &'static str,
    /// What it unlocks, one line.
    pub doc: &'static str,
    /// Tried in order; see [`Plan`](super::Plan) for the rules.
    pub routes: &'static [Route],
}

/// A capability a renderer can turn on.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum Op {
    // Draw.
    MultiDrawIndirect,
    ShaderDrawParameters,
    DrawIndirectCount,
    IndexTypeUint8,
    MeshShader,
    // Memory.
    RobustBufferAccess,
    Storage16Bit,
    Storage8Bit,
    ScalarBlockLayout,
    BufferDeviceAddress,
    MemoryPriority,
    PageableDeviceLocalMemory,
    // Shader.
    Int16,
    Int64,
    Float16,
    Int8,
    SubgroupVote,
    SubgroupArithmetic,
    SubgroupBallot,
    SubgroupShuffle,
    SubgroupSizeControl,
    DemoteToHelperInvocation,
    IntegerDotProduct,
    // Descriptor.
    BindlessTextures,
    BindlessBuffers,
    PushDescriptor,
    // Sync.
    TimelineSemaphore,
    Synchronization2,
    HostQueryReset,
    PipelineStatistics,
    // Render.
    DynamicRendering,
    SamplerFilterMinmax,
    SamplerAnisotropy,
    FifoLatestReady,
}

impl Op {
    /// Every op, grouped by area.
    pub const ALL: &[Op] = &[
        Op::MultiDrawIndirect,
        Op::ShaderDrawParameters,
        Op::DrawIndirectCount,
        Op::IndexTypeUint8,
        Op::MeshShader,
        Op::RobustBufferAccess,
        Op::Storage16Bit,
        Op::Storage8Bit,
        Op::ScalarBlockLayout,
        Op::BufferDeviceAddress,
        Op::MemoryPriority,
        Op::PageableDeviceLocalMemory,
        Op::Int16,
        Op::Int64,
        Op::Float16,
        Op::Int8,
        Op::SubgroupVote,
        Op::SubgroupArithmetic,
        Op::SubgroupBallot,
        Op::SubgroupShuffle,
        Op::SubgroupSizeControl,
        Op::DemoteToHelperInvocation,
        Op::IntegerDotProduct,
        Op::BindlessTextures,
        Op::BindlessBuffers,
        Op::PushDescriptor,
        Op::TimelineSemaphore,
        Op::Synchronization2,
        Op::HostQueryReset,
        Op::PipelineStatistics,
        Op::DynamicRendering,
        Op::SamplerFilterMinmax,
        Op::SamplerAnisotropy,
        Op::FifoLatestReady,
    ];

    /// Its name, doc and routes.
    #[must_use]
    pub fn spec(self) -> &'static Spec {
        [
            draw::SPECS,
            memory::SPECS,
            shader::SPECS,
            descriptor::SPECS,
            sync::SPECS,
            render::SPECS,
        ]
        .into_iter()
        .flatten()
        .find(|spec| spec.op == self)
        .expect("every op has a spec")
    }

    #[must_use]
    pub fn name(self) -> &'static str {
        self.spec().name
    }
}

/// A core-1.0 route: `fields` of `VkPhysicalDeviceFeatures`.
pub(crate) const fn core10(fields: &'static [&'static str]) -> Route {
    Route::Core {
        since: ApiVersion::V1_0,
        chain: Chain::Core10,
        fields,
    }
}

/// A core route of `chain` from `since`.
pub(crate) const fn core(
    since: ApiVersion,
    chain: Chain,
    fields: &'static [&'static str],
) -> Route {
    Route::Core {
        since,
        chain,
        fields,
    }
}

/// An extension route.
pub(crate) const fn ext(
    name: &'static CStr,
    deps: &'static [&'static CStr],
    chain: Option<Chain>,
    fields: &'static [&'static str],
) -> Route {
    Route::Extension {
        name,
        deps,
        chain,
        fields,
    }
}
