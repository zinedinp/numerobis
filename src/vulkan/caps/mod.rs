//! The capabilities a renderer asks for, one module per area, each with its
//! waterfall of [`Route`]s: the newest core version first, then the
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

/// One way to get a [`Capability`].
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

/// The area a capability belongs to (its module).
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

/// What a capability is and how to get it.
#[derive(Debug, Clone, Copy)]
pub struct Spec {
    pub area: Area,
    /// Short snake-case name for logs (`draw_indirect_count`).
    pub name: &'static str,
    /// What it unlocks, one line.
    pub doc: &'static str,
    /// Tried in order; see [`Plan`](super::Plan) for the rules.
    pub routes: &'static [Route],
}

/// One list makes the enum, [`Capability::ALL`] and [`Capability::spec`], so
/// they can't drift apart.
macro_rules! capabilities {
    ($($cap:ident => $module:ident::$spec:ident,)*) => {
        /// A capability a renderer can turn on.
        #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
        #[non_exhaustive]
        pub enum Capability {
            $(#[doc = concat!("See [`", stringify!($module), "::", stringify!($spec), "`].")] $cap,)*
        }

        impl Capability {
            /// Every capability, grouped by area.
            pub const ALL: &[Capability] = &[$(Capability::$cap,)*];

            /// Its name, doc and routes.
            #[must_use]
            pub const fn spec(self) -> &'static Spec {
                match self {
                    $(Capability::$cap => &$module::$spec,)*
                }
            }
        }
    };
}

capabilities! {
    // Draw.
    MultiDrawIndirect => draw::MULTI_DRAW_INDIRECT,
    ShaderDrawParameters => draw::SHADER_DRAW_PARAMETERS,
    DrawIndirectCount => draw::DRAW_INDIRECT_COUNT,
    IndexTypeUint8 => draw::INDEX_TYPE_UINT8,
    MeshShader => draw::MESH_SHADER,
    // Memory.
    RobustBufferAccess => memory::ROBUST_BUFFER_ACCESS,
    Storage16Bit => memory::STORAGE_16BIT,
    Storage8Bit => memory::STORAGE_8BIT,
    ScalarBlockLayout => memory::SCALAR_BLOCK_LAYOUT,
    BufferDeviceAddress => memory::BUFFER_DEVICE_ADDRESS,
    MemoryPriority => memory::MEMORY_PRIORITY,
    PageableDeviceLocalMemory => memory::PAGEABLE_DEVICE_LOCAL_MEMORY,
    // Shader.
    Int16 => shader::INT16,
    Int64 => shader::INT64,
    Float16 => shader::FLOAT16,
    Int8 => shader::INT8,
    SubgroupVote => shader::SUBGROUP_VOTE,
    SubgroupArithmetic => shader::SUBGROUP_ARITHMETIC,
    SubgroupBallot => shader::SUBGROUP_BALLOT,
    SubgroupShuffle => shader::SUBGROUP_SHUFFLE,
    SubgroupSizeControl => shader::SUBGROUP_SIZE_CONTROL,
    DemoteToHelperInvocation => shader::DEMOTE_TO_HELPER_INVOCATION,
    IntegerDotProduct => shader::INTEGER_DOT_PRODUCT,
    // Descriptor.
    BindlessTextures => descriptor::BINDLESS_TEXTURES,
    BindlessBuffers => descriptor::BINDLESS_BUFFERS,
    PushDescriptor => descriptor::PUSH_DESCRIPTOR,
    // Sync.
    TimelineSemaphore => sync::TIMELINE_SEMAPHORE,
    Synchronization2 => sync::SYNCHRONIZATION2,
    HostQueryReset => sync::HOST_QUERY_RESET,
    PipelineStatistics => sync::PIPELINE_STATISTICS,
    // Render.
    DynamicRendering => render::DYNAMIC_RENDERING,
    SamplerFilterMinmax => render::SAMPLER_FILTER_MINMAX,
    SamplerAnisotropy => render::SAMPLER_ANISOTROPY,
    FifoLatestReady => render::FIFO_LATEST_READY,
}

impl Capability {
    #[must_use]
    pub const fn name(self) -> &'static str {
        self.spec().name
    }
}

/// A route reads at least one bit, and only bits its struct has: checked
/// while the tables compile.
const fn check(chain: Chain, fields: &[&str]) {
    assert!(
        !fields.is_empty(),
        "a route with a struct reads at least one bit"
    );
    let mut i = 0;
    while i < fields.len() {
        assert!(
            chain.index(fields[i]).is_some(),
            "a route field is not a member of its struct"
        );
        i += 1;
    }
}

/// A core-1.0 route: `fields` of `VkPhysicalDeviceFeatures`.
pub(crate) const fn core10(fields: &'static [&'static str]) -> Route {
    core(ApiVersion::V1_0, Chain::Core10, fields)
}

/// A core route of `chain` from `since`.
pub(crate) const fn core(
    since: ApiVersion,
    chain: Chain,
    fields: &'static [&'static str],
) -> Route {
    check(chain, fields);
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
    match chain {
        Some(chain) => check(chain, fields),
        None => assert!(
            fields.is_empty(),
            "an extension without a struct reads no bits"
        ),
    }
    Route::Extension {
        name,
        deps,
        chain,
        fields,
    }
}
