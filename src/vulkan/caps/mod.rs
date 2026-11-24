//! The capabilities a renderer asks for, one module per area, each with its
//! waterfall of [`Route`]s: the newest core version first, then the
//! extensions that offered it before.

use std::ffi::CStr;

use super::ApiVersion;
use super::features::Chain;
pub use crate::platform::{Os, Status, Support};

pub mod compute;
pub mod descriptor;
pub mod draw;
pub mod interop;
pub mod memory;
pub mod mesh;
pub mod pipeline;
pub mod present;
pub mod raytracing;
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
    /// Extension `name`, with its dependencies from `vk.xml`
    /// ([`features::depends`](super::features::depends)). With a `chain`, its
    /// `fields` must be on too; without one, being offered is support.
    Extension {
        name: &'static CStr,
        chain: Option<Chain>,
        fields: &'static [&'static str],
    },
    /// Core from `since` without a feature bit: the version is support
    /// (`vkCmdSetCullMode` in 1.3, which had `extendedDynamicState`).
    Version { since: ApiVersion },
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
    Mesh,
    RayTracing,
    Memory,
    Shader,
    Compute,
    Descriptor,
    Pipeline,
    Sync,
    Render,
    Present,
    Interop,
}

impl Area {
    #[must_use]
    pub const fn name(self) -> &'static str {
        match self {
            Self::Draw => "draw",
            Self::Mesh => "mesh",
            Self::RayTracing => "ray tracing",
            Self::Memory => "memory",
            Self::Shader => "shader",
            Self::Compute => "compute",
            Self::Descriptor => "descriptor",
            Self::Pipeline => "pipeline",
            Self::Sync => "sync",
            Self::Render => "render",
            Self::Present => "present",
            Self::Interop => "interop",
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
    /// Per OS: whether it exists there (`Unavailable` skips it), and whether
    /// it was granted and enabled on a real driver there (`Tested`).
    pub support: Support,
    /// Capabilities whose bits have to be on too: the probe asks for them as
    /// well, and without them this one is not granted.
    pub needs: &'static [Capability],
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
    VertexAttributeDivisor => draw::VERTEX_ATTRIBUTE_DIVISOR,
    // Mesh.
    MeshShader => mesh::MESH_SHADER,
    TaskShader => mesh::TASK_SHADER,
    // Ray tracing.
    AccelerationStructure => raytracing::ACCELERATION_STRUCTURE,
    RayTracingPipeline => raytracing::RAY_TRACING_PIPELINE,
    RayTraversalPrimitiveCulling => raytracing::RAY_TRAVERSAL_PRIMITIVE_CULLING,
    RayQuery => raytracing::RAY_QUERY,
    RayTracingMaintenance1 => raytracing::RAY_TRACING_MAINTENANCE1,
    OpacityMicromap => raytracing::OPACITY_MICROMAP,
    RayTracingInvocationReorder => raytracing::RAY_TRACING_INVOCATION_REORDER,
    // Memory.
    RobustBufferAccess => memory::ROBUST_BUFFER_ACCESS,
    Storage16Bit => memory::STORAGE_16BIT,
    Storage8Bit => memory::STORAGE_8BIT,
    ScalarBlockLayout => memory::SCALAR_BLOCK_LAYOUT,
    BufferDeviceAddress => memory::BUFFER_DEVICE_ADDRESS,
    MemoryPriority => memory::MEMORY_PRIORITY,
    PageableDeviceLocalMemory => memory::PAGEABLE_DEVICE_LOCAL_MEMORY,
    MemoryBudget => memory::MEMORY_BUDGET,
    HostImageCopy => memory::HOST_IMAGE_COPY,
    SparseResidency => memory::SPARSE_RESIDENCY,
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
    GeometryShader => shader::GEOMETRY_SHADER,
    TessellationShader => shader::TESSELLATION_SHADER,
    Float64 => shader::FLOAT64,
    AtomicInt64 => shader::ATOMIC_INT64,
    ImageAtomicInt64 => shader::IMAGE_ATOMIC_INT64,
    FragmentShaderBarycentric => shader::FRAGMENT_SHADER_BARYCENTRIC,
    // Compute.
    VulkanMemoryModel => compute::VULKAN_MEMORY_MODEL,
    CooperativeMatrix => compute::COOPERATIVE_MATRIX,
    Bfloat16 => compute::BFLOAT16,
    Bfloat16CooperativeMatrix => compute::BFLOAT16_COOPERATIVE_MATRIX,
    Float8 => compute::FLOAT8,
    Float8CooperativeMatrix => compute::FLOAT8_COOPERATIVE_MATRIX,
    // Descriptor.
    BindlessTextures => descriptor::BINDLESS_TEXTURES,
    BindlessBuffers => descriptor::BINDLESS_BUFFERS,
    PushDescriptor => descriptor::PUSH_DESCRIPTOR,
    NullDescriptor => descriptor::NULL_DESCRIPTOR,
    DescriptorBuffer => descriptor::DESCRIPTOR_BUFFER,
    // Pipeline.
    ExtendedDynamicState => pipeline::EXTENDED_DYNAMIC_STATE,
    ExtendedDynamicState2 => pipeline::EXTENDED_DYNAMIC_STATE2,
    GraphicsPipelineLibrary => pipeline::GRAPHICS_PIPELINE_LIBRARY,
    ShaderObject => pipeline::SHADER_OBJECT,
    Maintenance4 => pipeline::MAINTENANCE4,
    Maintenance5 => pipeline::MAINTENANCE5,
    Maintenance6 => pipeline::MAINTENANCE6,
    // Sync.
    TimelineSemaphore => sync::TIMELINE_SEMAPHORE,
    Synchronization2 => sync::SYNCHRONIZATION2,
    HostQueryReset => sync::HOST_QUERY_RESET,
    PipelineStatistics => sync::PIPELINE_STATISTICS,
    CalibratedTimestamps => sync::CALIBRATED_TIMESTAMPS,
    GlobalPriority => sync::GLOBAL_PRIORITY,
    GlobalPriorityQuery => sync::GLOBAL_PRIORITY_QUERY,
    // Render.
    DynamicRendering => render::DYNAMIC_RENDERING,
    SamplerFilterMinmax => render::SAMPLER_FILTER_MINMAX,
    SamplerAnisotropy => render::SAMPLER_ANISOTROPY,
    TextureCompressionBc => render::TEXTURE_COMPRESSION_BC,
    TextureCompressionEtc2 => render::TEXTURE_COMPRESSION_ETC2,
    TextureCompressionAstcLdr => render::TEXTURE_COMPRESSION_ASTC_LDR,
    TextureCompressionAstcHdr => render::TEXTURE_COMPRESSION_ASTC_HDR,
    FillModeNonSolid => render::FILL_MODE_NON_SOLID,
    DepthClamp => render::DEPTH_CLAMP,
    WideLines => render::WIDE_LINES,
    DepthBounds => render::DEPTH_BOUNDS,
    LogicOp => render::LOGIC_OP,
    Multiview => render::MULTIVIEW,
    FragmentShadingRate => render::FRAGMENT_SHADING_RATE,
    FragmentShadingRateAttachment => render::FRAGMENT_SHADING_RATE_ATTACHMENT,
    FragmentShaderInterlock => render::FRAGMENT_SHADER_INTERLOCK,
    ConservativeRasterization => render::CONSERVATIVE_RASTERIZATION,
    // Present.
    FifoLatestReady => present::FIFO_LATEST_READY,
    PresentId => present::PRESENT_ID,
    PresentWait => present::PRESENT_WAIT,
    PresentId2 => present::PRESENT_ID2,
    PresentWait2 => present::PRESENT_WAIT2,
    // Interop.
    ExternalMemoryFd => interop::EXTERNAL_MEMORY_FD,
    DmaBuf => interop::DMA_BUF,
    DrmFormatModifier => interop::DRM_FORMAT_MODIFIER,
    ExternalSemaphoreFd => interop::EXTERNAL_SEMAPHORE_FD,
    ExternalMemoryWin32 => interop::EXTERNAL_MEMORY_WIN32,
    ExternalSemaphoreWin32 => interop::EXTERNAL_SEMAPHORE_WIN32,
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

/// A core route without a feature bit.
pub(crate) const fn version(since: ApiVersion) -> Route {
    Route::Version { since }
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
        chain,
        fields,
    }
}
