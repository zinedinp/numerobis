//! The Direct3D 10 to 12 capabilities, one area per kind of hardware
//! feature. Raytracing, mesh shaders, variable rate shading, sampler
//! feedback and work graphs each have their own: each is a block of
//! hardware a GPU either has or not.

use super::super::generated::{D3d11Feature, D3d12Feature, value as v};
use super::super::level::{FeatureLevel, ShaderModel};
use super::super::{Api, same};
pub use crate::platform::Support;

/// The area a capability belongs to.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Area {
    /// Stages and shader arithmetic: geometry, tessellation, compute,
    /// doubles, 16-bit, 64-bit.
    Shader,
    /// Wave (subgroup) intrinsics and wave matrices.
    Wave,
    /// DXR: acceleration structures, raytracing pipelines, inline ray queries.
    Raytracing,
    /// Mesh and amplification shaders.
    Mesh,
    /// Variable rate shading.
    ShadingRate,
    /// Sampler feedback (texture streaming, texture-space shading).
    SamplerFeedback,
    /// Work graphs: GPU-driven work creation.
    WorkGraphs,
    /// Resource binding tiers, root signatures, bindless.
    Binding,
    /// Tiled resources, typed UAV loads, ROVs, format casting.
    Resource,
    /// Conservative rasterization, depth bounds, sample positions, view
    /// instancing, logic ops.
    Raster,
    /// Heaps, upload heaps on resizable BAR, unified memory.
    Memory,
    /// Barriers, render passes, indirect execution, queues, threading.
    Pipeline,
}

impl Area {
    #[must_use]
    pub const fn name(self) -> &'static str {
        match self {
            Self::Shader => "shader",
            Self::Wave => "wave",
            Self::Raytracing => "raytracing",
            Self::Mesh => "mesh",
            Self::ShadingRate => "shading rate",
            Self::SamplerFeedback => "sampler feedback",
            Self::WorkGraphs => "work graphs",
            Self::Binding => "binding",
            Self::Resource => "resource",
            Self::Raster => "raster",
            Self::Memory => "memory",
            Self::Pipeline => "pipeline",
        }
    }
}

/// How a `CheckFeatureSupport` member is read.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Test {
    /// At least this (a `BOOL` is `AtLeast(1)`, a tier its value).
    AtLeast(u32),
    /// Every bit of this set.
    Bits(u32),
}

impl Test {
    #[must_use]
    pub const fn passes(self, have: u32) -> bool {
        match self {
            Self::AtLeast(min) => have >= min,
            Self::Bits(mask) => have & mask == mask,
        }
    }
}

/// One way to get a [`Capability`]; a capability's routes are
/// alternatives, the first that passes grants it.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Route {
    /// Every device of a modern version at `level` or above has it.
    Level(FeatureLevel),
    /// Every device of this version has it.
    Api(Api),
    /// Direct3D 12: member `field` (index into
    /// [`D3d12Feature::fields`]) of `feature` passes `test`.
    D3d12 {
        feature: D3d12Feature,
        field: usize,
        test: Test,
    },
    /// Direct3D 11: the same with `ID3D11Device::CheckFeatureSupport`.
    D3d11 {
        feature: D3d11Feature,
        field: usize,
        test: Test,
    },
}

impl Route {
    /// Whether this route can apply on `api` at all.
    #[must_use]
    pub const fn on(self, api: Api) -> bool {
        match self {
            Self::Level(_) => !matches!(api, Api::D3D8 | Api::D3D9),
            Self::Api(a) => a as u8 == api as u8,
            Self::D3d12 { .. } => matches!(api, Api::D3D12),
            Self::D3d11 { .. } => matches!(api, Api::D3D11),
        }
    }

    /// The member name it reads, for logs.
    #[must_use]
    pub const fn member(self) -> Option<&'static str> {
        match self {
            Self::D3d12 { feature, field, .. } => Some(feature.fields()[field]),
            Self::D3d11 { feature, field, .. } => Some(feature.fields()[field]),
            Self::Level(_) | Self::Api(_) => None,
        }
    }
}

impl D3d12Feature {
    /// The index of member `name`; `const`, so the tables are checked while
    /// compiling.
    #[must_use]
    pub const fn index(self, name: &str) -> Option<usize> {
        index_in(self.fields(), name)
    }
}

impl D3d11Feature {
    /// The index of member `name`.
    #[must_use]
    pub const fn index(self, name: &str) -> Option<usize> {
        index_in(self.fields(), name)
    }
}

const fn index_in(fields: &[&str], name: &str) -> Option<usize> {
    let mut i = 0;
    while i < fields.len() {
        if same(fields[i], name) {
            return Some(i);
        }
        i += 1;
    }
    None
}

const fn d12(feature: D3d12Feature, field: &str, test: Test) -> Route {
    match feature.index(field) {
        Some(field) => Route::D3d12 {
            feature,
            field,
            test,
        },
        None => panic!("not a member of the D3D12 struct"),
    }
}

const fn d11(feature: D3d11Feature, field: &str, test: Test) -> Route {
    match feature.index(field) {
        Some(field) => Route::D3d11 {
            feature,
            field,
            test,
        },
        None => panic!("not a member of the D3D11 struct"),
    }
}

/// A `BOOL` member.
const ON: Test = Test::AtLeast(1);

/// What a capability is and how to get it.
#[derive(Debug, Clone, Copy)]
pub struct Spec {
    pub area: Area,
    pub name: &'static str,
    pub doc: &'static str,
    pub routes: &'static [Route],
    /// Direct3D 12: the HLSL shader model that exposes it; granted only
    /// when the device reports at least this one.
    pub shader_model: Option<ShaderModel>,
    pub needs: &'static [Capability],
    pub support: Support,
}

macro_rules! capabilities {
    ($($cap:ident => $spec:ident,)*) => {
        /// A Direct3D 10 to 12 capability.
        #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
        #[non_exhaustive]
        pub enum Capability {
            $(#[doc = concat!("See [`", stringify!($spec), "`].")] $cap,)*
        }

        impl Capability {
            /// Every capability, grouped by area.
            pub const ALL: &[Capability] = &[$(Capability::$cap,)*];

            #[must_use]
            pub const fn spec(self) -> &'static Spec {
                match self {
                    $(Capability::$cap => &$spec,)*
                }
            }
        }
    };
}

capabilities! {
    // Shader.
    GeometryShader => GEOMETRY_SHADER,
    StreamOutput => STREAM_OUTPUT,
    Tessellation => TESSELLATION,
    ComputeShader => COMPUTE_SHADER,
    Doubles => DOUBLES,
    MinPrecision16 => MIN_PRECISION_16,
    Native16Bit => NATIVE_16BIT,
    Int64 => INT64,
    Barycentrics => BARYCENTRICS,
    StencilRefFromShader => STENCIL_REF_FROM_SHADER,
    ViewportIndexAnyStage => VIEWPORT_INDEX_ANY_STAGE,
    AtomicInt64Typed => ATOMIC_INT64_TYPED,
    AtomicInt64GroupShared => ATOMIC_INT64_GROUP_SHARED,
    AtomicInt64DescriptorHeap => ATOMIC_INT64_DESCRIPTOR_HEAP,
    AdvancedTextureOps => ADVANCED_TEXTURE_OPS,
    SampleCmpGradient => SAMPLE_CMP_GRADIENT,
    // Wave.
    WaveOps => WAVE_OPS,
    WaveMma => WAVE_MMA,
    // Raytracing.
    Raytracing => RAYTRACING,
    Raytracing11 => RAYTRACING_1_1,
    Raytracing12 => RAYTRACING_1_2,
    // Mesh.
    MeshShader => MESH_SHADER,
    MeshShaderPipelineStats => MESH_SHADER_PIPELINE_STATS,
    MeshShaderFullRangeRtArrayIndex => MESH_SHADER_FULL_RANGE_RT_ARRAY_INDEX,
    MeshShaderDerivatives => MESH_SHADER_DERIVATIVES,
    MeshShaderPerPrimitiveShadingRate => MESH_SHADER_PER_PRIMITIVE_SHADING_RATE,
    // Shading rate.
    VariableRateShading => VARIABLE_RATE_SHADING,
    VariableRateShading2 => VARIABLE_RATE_SHADING_2,
    AdditionalShadingRates => ADDITIONAL_SHADING_RATES,
    ShadingRateSumCombiner => SHADING_RATE_SUM_COMBINER,
    // Sampler feedback.
    SamplerFeedback => SAMPLER_FEEDBACK,
    SamplerFeedback10 => SAMPLER_FEEDBACK_1_0,
    // Work graphs.
    WorkGraphs => WORK_GRAPHS,
    // Binding.
    BindingTier2 => BINDING_TIER_2,
    BindingTier3 => BINDING_TIER_3,
    DynamicResources => DYNAMIC_RESOURCES,
    RootSignature11 => ROOT_SIGNATURE_1_1,
    // Resource.
    TiledResources => TILED_RESOURCES,
    TiledResources2 => TILED_RESOURCES_2,
    TiledResources3 => TILED_RESOURCES_3,
    TiledResources4 => TILED_RESOURCES_4,
    TypedUavLoads => TYPED_UAV_LOADS,
    Rovs => ROVS,
    StandardSwizzle => STANDARD_SWIZZLE,
    CastingFullyTyped => CASTING_FULLY_TYPED,
    RelaxedFormatCasting => RELAXED_FORMAT_CASTING,
    UnalignedBlockTextures => UNALIGNED_BLOCK_TEXTURES,
    WriteableMsaaTextures => WRITEABLE_MSAA_TEXTURES,
    // Raster.
    ConservativeRasterization => CONSERVATIVE_RASTERIZATION,
    ConservativeRasterization3 => CONSERVATIVE_RASTERIZATION_3,
    DepthBounds => DEPTH_BOUNDS,
    ProgrammableSamplePositions => PROGRAMMABLE_SAMPLE_POSITIONS,
    ViewInstancing => VIEW_INSTANCING,
    LogicOps => LOGIC_OPS,
    TriangleFans => TRIANGLE_FANS,
    DynamicDepthBias => DYNAMIC_DEPTH_BIAS,
    IndependentFrontBackStencil => INDEPENDENT_FRONT_BACK_STENCIL,
    // Memory.
    ResourceHeapTier2 => RESOURCE_HEAP_TIER_2,
    GpuUploadHeap => GPU_UPLOAD_HEAP,
    Uma => UMA,
    CacheCoherentUma => CACHE_COHERENT_UMA,
    MapOnDefaultBuffers => MAP_ON_DEFAULT_BUFFERS,
    ExistingHeaps => EXISTING_HEAPS,
    // Pipeline.
    EnhancedBarriers => ENHANCED_BARRIERS,
    RenderPasses => RENDER_PASSES,
    ExecuteIndirect11 => EXECUTE_INDIRECT_1_1,
    CopyQueueTimestamps => COPY_QUEUE_TIMESTAMPS,
    BackgroundProcessing => BACKGROUND_PROCESSING,
    DriverCommandLists => DRIVER_COMMAND_LISTS,
    DriverConcurrentCreates => DRIVER_CONCURRENT_CREATES,
}

impl Capability {
    #[must_use]
    pub const fn name(self) -> &'static str {
        self.spec().name
    }
}

use Capability as C;
use D3d11Feature as F11;
use D3d12Feature as F12;
use FeatureLevel as L;

const T: Support = Support::WINDOWS_TESTED;
const U: Support = Support::UNTESTED;

#[allow(clippy::unnecessary_wraps, reason = "fills `Spec::shader_model`")]
const fn sm(major: u8, minor: u8) -> Option<ShaderModel> {
    Some(ShaderModel::new(major, minor))
}

macro_rules! spec {
    ($area:ident, $name:literal, $doc:literal, $support:expr, $routes:expr) => {
        spec!($area, $name, $doc, $support, $routes, None, &[])
    };
    ($area:ident, $name:literal, $doc:literal, $support:expr, $routes:expr, $sm:expr) => {
        spec!($area, $name, $doc, $support, $routes, $sm, &[])
    };
    ($area:ident, $name:literal, $doc:literal, $support:expr, $routes:expr, $sm:expr, $needs:expr) => {
        Spec {
            area: Area::$area,
            name: $name,
            doc: $doc,
            support: $support,
            routes: $routes,
            shader_model: $sm,
            needs: $needs,
        }
    };
}

// Shader.
pub const GEOMETRY_SHADER: Spec = spec!(
    Shader,
    "geometry_shader",
    "geometry shaders (Direct3D 10: absent from 8 and 9 altogether)",
    T,
    &[Route::Level(L::L10_0)]
);
pub const STREAM_OUTPUT: Spec = spec!(
    Shader,
    "stream_output",
    "writing vertices back to buffers from the geometry stage",
    T,
    &[Route::Level(L::L10_0)]
);
pub const TESSELLATION: Spec = spec!(
    Shader,
    "tessellation",
    "hull and domain shaders",
    T,
    &[Route::Level(L::L11_0)]
);
pub const COMPUTE_SHADER: Spec = spec!(
    Shader,
    "compute_shader",
    "compute shaders: cs_5_0 from 11_0, cs_4_x with raw and structured buffers on some 10_x devices",
    T,
    &[
        Route::Level(L::L11_0),
        d11(
            F11::D3d10XHardwareOptions,
            "ComputeShaders_Plus_RawAndStructuredBuffers_Via_Shader_4_x",
            ON
        ),
    ]
);
pub const DOUBLES: Spec = spec!(
    Shader,
    "doubles",
    "double-precision shader arithmetic",
    T,
    &[
        d12(F12::Options, "DoublePrecisionFloatShaderOps", ON),
        d11(F11::Doubles, "DoublePrecisionFloatShaderOps", ON),
    ]
);
pub const MIN_PRECISION_16: Spec = spec!(
    Shader,
    "min_precision_16",
    "min16float / min16int may run at 16 bits",
    T,
    &[
        d12(
            F12::Options,
            "MinPrecisionSupport",
            Test::Bits(v::D3D12_SHADER_MIN_PRECISION_SUPPORT_16_BIT)
        ),
        d11(
            F11::ShaderMinPrecisionSupport,
            "AllOtherShaderStagesMinPrecision",
            Test::Bits(v::D3D12_SHADER_MIN_PRECISION_SUPPORT_16_BIT)
        ),
    ]
);
pub const NATIVE_16BIT: Spec = spec!(
    Shader,
    "native_16bit",
    "float16_t / int16_t as real 16-bit types (shader model 6.2)",
    T,
    &[d12(F12::Options4, "Native16BitShaderOpsSupported", ON)],
    sm(6, 2)
);
pub const INT64: Spec = spec!(
    Shader,
    "int64",
    "64-bit integer shader arithmetic",
    T,
    &[d12(F12::Options1, "Int64ShaderOps", ON)],
    sm(6, 0)
);
pub const BARYCENTRICS: Spec = spec!(
    Shader,
    "barycentrics",
    "SV_Barycentrics in pixel shaders",
    T,
    &[d12(F12::Options3, "BarycentricsSupported", ON)],
    sm(6, 1)
);
pub const STENCIL_REF_FROM_SHADER: Spec = spec!(
    Shader,
    "stencil_ref_from_shader",
    "SV_StencilRef written by the pixel shader",
    T,
    &[
        d12(F12::Options, "PSSpecifiedStencilRefSupported", ON),
        d11(F11::Options2, "PSSpecifiedStencilRefSupported", ON),
    ]
);
pub const VIEWPORT_INDEX_ANY_STAGE: Spec = spec!(
    Shader,
    "viewport_index_any_stage",
    "SV_ViewportArrayIndex / SV_RenderTargetArrayIndex from any stage before the rasterizer",
    T,
    &[
        d12(
            F12::Options,
            "VPAndRTArrayIndexFromAnyShaderFeedingRasterizerSupportedWithoutGSEmulation",
            ON
        ),
        d11(
            F11::Options3,
            "VPAndRTArrayIndexFromAnyShaderFeedingRasterizer",
            ON
        ),
    ]
);
pub const ATOMIC_INT64_TYPED: Spec = spec!(
    Shader,
    "atomic_int64_typed",
    "64-bit atomics on typed resources",
    T,
    &[d12(
        F12::Options9,
        "AtomicInt64OnTypedResourceSupported",
        ON
    )],
    sm(6, 6)
);
pub const ATOMIC_INT64_GROUP_SHARED: Spec = spec!(
    Shader,
    "atomic_int64_group_shared",
    "64-bit atomics on groupshared memory",
    U,
    &[d12(F12::Options9, "AtomicInt64OnGroupSharedSupported", ON)],
    sm(6, 6)
);
pub const ATOMIC_INT64_DESCRIPTOR_HEAP: Spec = spec!(
    Shader,
    "atomic_int64_descriptor_heap",
    "64-bit atomics on resources from the descriptor heap",
    T,
    &[d12(
        F12::Options11,
        "AtomicInt64OnDescriptorHeapResourceSupported",
        ON
    )],
    sm(6, 6)
);
pub const ADVANCED_TEXTURE_OPS: Spec = spec!(
    Shader,
    "advanced_texture_ops",
    "shader model 6.7 texture operations (integer sampling, SampleCmpLevel, raw gather)",
    T,
    &[d12(F12::Options14, "AdvancedTextureOpsSupported", ON)],
    sm(6, 7)
);
pub const SAMPLE_CMP_GRADIENT: Spec = spec!(
    Shader,
    "sample_cmp_gradient",
    "SampleCmpGrad and SampleCmpBias (shader model 6.8)",
    U,
    &[d12(F12::Options21, "SampleCmpGradientAndBiasSupported", ON)],
    sm(6, 8)
);

// Wave.
pub const WAVE_OPS: Spec = spec!(
    Wave,
    "wave_ops",
    "wave intrinsics: votes, ballots, prefix sums, lane reads (shader model 6.0)",
    T,
    &[d12(F12::Options1, "WaveOps", ON)],
    sm(6, 0)
);
pub const WAVE_MMA: Spec = spec!(
    Wave,
    "wave_mma",
    "wave matrix multiply-accumulate (preview)",
    U,
    &[d12(
        F12::Options9,
        "WaveMMATier",
        Test::AtLeast(v::D3D12_WAVE_MMA_TIER_1_0)
    )]
);

// Raytracing.
pub const RAYTRACING: Spec = spec!(
    Raytracing,
    "raytracing",
    "DXR 1.0: acceleration structures, DispatchRays and raytracing pipelines",
    T,
    &[d12(
        F12::Options5,
        "RaytracingTier",
        Test::AtLeast(v::D3D12_RAYTRACING_TIER_1_0)
    )],
    sm(6, 3)
);
pub const RAYTRACING_1_1: Spec = spec!(
    Raytracing,
    "raytracing_1_1",
    "DXR 1.1: inline ray queries in any shader, indirect DispatchRays, AddToStateObject",
    T,
    &[d12(
        F12::Options5,
        "RaytracingTier",
        Test::AtLeast(v::D3D12_RAYTRACING_TIER_1_1)
    )],
    sm(6, 5),
    &[C::Raytracing]
);
pub const RAYTRACING_1_2: Spec = spec!(
    Raytracing,
    "raytracing_1_2",
    "DXR 1.2: opacity micromaps and shader execution reordering",
    U,
    &[d12(
        F12::Options5,
        "RaytracingTier",
        Test::AtLeast(v::D3D12_RAYTRACING_TIER_1_2)
    )],
    sm(6, 9),
    &[C::Raytracing11]
);

// Mesh.
pub const MESH_SHADER: Spec = spec!(
    Mesh,
    "mesh_shader",
    "mesh and amplification shaders in place of the vertex pipeline",
    T,
    &[d12(
        F12::Options7,
        "MeshShaderTier",
        Test::AtLeast(v::D3D12_MESH_SHADER_TIER_1)
    )],
    sm(6, 5)
);
pub const MESH_SHADER_PIPELINE_STATS: Spec = spec!(
    Mesh,
    "mesh_shader_pipeline_stats",
    "mesh and amplification invocations in pipeline statistics",
    T,
    &[d12(F12::Options9, "MeshShaderPipelineStatsSupported", ON)],
    None,
    &[C::MeshShader]
);
pub const MESH_SHADER_FULL_RANGE_RT_ARRAY_INDEX: Spec = spec!(
    Mesh,
    "mesh_shader_full_range_rt_array_index",
    "any render target array index from a mesh shader",
    T,
    &[d12(
        F12::Options9,
        "MeshShaderSupportsFullRangeRenderTargetArrayIndex",
        ON
    )],
    None,
    &[C::MeshShader]
);
pub const MESH_SHADER_DERIVATIVES: Spec = spec!(
    Mesh,
    "mesh_shader_derivatives",
    "derivatives and implicit LOD sampling in mesh and amplification shaders",
    U,
    &[d12(
        F12::Options9,
        "DerivativesInMeshAndAmplificationShadersSupported",
        ON
    )],
    sm(6, 6),
    &[C::MeshShader]
);
pub const MESH_SHADER_PER_PRIMITIVE_SHADING_RATE: Spec = spec!(
    Mesh,
    "mesh_shader_per_primitive_shading_rate",
    "SV_ShadingRate per primitive from a mesh shader",
    T,
    &[d12(
        F12::Options10,
        "MeshShaderPerPrimitiveShadingRateSupported",
        ON
    )],
    None,
    &[C::MeshShader, C::VariableRateShading2]
);

// Shading rate.
pub const VARIABLE_RATE_SHADING: Spec = spec!(
    ShadingRate,
    "variable_rate_shading",
    "tier 1: a shading rate per draw",
    T,
    &[d12(
        F12::Options6,
        "VariableShadingRateTier",
        Test::AtLeast(v::D3D12_VARIABLE_SHADING_RATE_TIER_1)
    )]
);
pub const VARIABLE_RATE_SHADING_2: Spec = spec!(
    ShadingRate,
    "variable_rate_shading_2",
    "tier 2: per primitive and from a screen-space rate image, with combiners",
    T,
    &[d12(
        F12::Options6,
        "VariableShadingRateTier",
        Test::AtLeast(v::D3D12_VARIABLE_SHADING_RATE_TIER_2)
    )],
    None,
    &[C::VariableRateShading]
);
pub const ADDITIONAL_SHADING_RATES: Spec = spec!(
    ShadingRate,
    "additional_shading_rates",
    "the 2x4, 4x2 and 4x4 rates",
    T,
    &[d12(F12::Options6, "AdditionalShadingRatesSupported", ON)],
    None,
    &[C::VariableRateShading]
);
pub const SHADING_RATE_SUM_COMBINER: Spec = spec!(
    ShadingRate,
    "shading_rate_sum_combiner",
    "the SUM combiner for shading rates",
    U,
    &[d12(
        F12::Options10,
        "VariableRateShadingSumCombinerSupported",
        ON
    )],
    None,
    &[C::VariableRateShading2]
);

// Sampler feedback.
pub const SAMPLER_FEEDBACK: Spec = spec!(
    SamplerFeedback,
    "sampler_feedback",
    "tier 0.9: which texels and mips sampling touched (streaming, texture-space shading)",
    T,
    &[d12(
        F12::Options7,
        "SamplerFeedbackTier",
        Test::AtLeast(v::D3D12_SAMPLER_FEEDBACK_TIER_0_9)
    )],
    sm(6, 5)
);
pub const SAMPLER_FEEDBACK_1_0: Spec = spec!(
    SamplerFeedback,
    "sampler_feedback_1_0",
    "tier 1.0: any sampler and addressing mode",
    T,
    &[d12(
        F12::Options7,
        "SamplerFeedbackTier",
        Test::AtLeast(v::D3D12_SAMPLER_FEEDBACK_TIER_1_0)
    )],
    None,
    &[C::SamplerFeedback]
);

// Work graphs.
pub const WORK_GRAPHS: Spec = spec!(
    WorkGraphs,
    "work_graphs",
    "work graphs: nodes that launch other nodes on the GPU (shader model 6.8)",
    T,
    &[d12(
        F12::Options21,
        "WorkGraphsTier",
        Test::AtLeast(v::D3D12_WORK_GRAPHS_TIER_1_0)
    )],
    sm(6, 8)
);

// Binding.
pub const BINDING_TIER_2: Spec = spec!(
    Binding,
    "binding_tier_2",
    "resource binding tier 2: full SRV heaps, bounded UAVs and CBVs",
    T,
    &[d12(
        F12::Options,
        "ResourceBindingTier",
        Test::AtLeast(v::D3D12_RESOURCE_BINDING_TIER_2)
    )]
);
pub const BINDING_TIER_3: Spec = spec!(
    Binding,
    "binding_tier_3",
    "resource binding tier 3: everything bindless, unbound descriptors allowed",
    T,
    &[d12(
        F12::Options,
        "ResourceBindingTier",
        Test::AtLeast(v::D3D12_RESOURCE_BINDING_TIER_3)
    )],
    None,
    &[C::BindingTier2]
);
pub const DYNAMIC_RESOURCES: Spec = spec!(
    Binding,
    "dynamic_resources",
    "ResourceDescriptorHeap / SamplerDescriptorHeap indexed straight from HLSL (shader model 6.6)",
    T,
    &[d12(
        F12::Options,
        "ResourceBindingTier",
        Test::AtLeast(v::D3D12_RESOURCE_BINDING_TIER_3)
    )],
    sm(6, 6),
    &[C::BindingTier3]
);
pub const ROOT_SIGNATURE_1_1: Spec = spec!(
    Binding,
    "root_signature_1_1",
    "root signature 1.1: static and volatile descriptor flags",
    T,
    &[d12(
        F12::RootSignature,
        "HighestVersion",
        Test::AtLeast(v::D3D_ROOT_SIGNATURE_VERSION_1_1)
    )]
);

// Resource.
pub const TILED_RESOURCES: Spec = spec!(
    Resource,
    "tiled_resources",
    "tier 1: reserved (sparse) resources mapped per 64 KiB tile",
    T,
    &[
        d12(
            F12::Options,
            "TiledResourcesTier",
            Test::AtLeast(v::D3D12_TILED_RESOURCES_TIER_1)
        ),
        d11(
            F11::Options1,
            "TiledResourcesTier",
            Test::AtLeast(v::D3D11_TILED_RESOURCES_TIER_1)
        ),
    ]
);
pub const TILED_RESOURCES_2: Spec = spec!(
    Resource,
    "tiled_resources_2",
    "tier 2: unmapped reads return zero, residency feedback in shaders",
    T,
    &[
        d12(
            F12::Options,
            "TiledResourcesTier",
            Test::AtLeast(v::D3D12_TILED_RESOURCES_TIER_2)
        ),
        d11(
            F11::Options1,
            "TiledResourcesTier",
            Test::AtLeast(v::D3D11_TILED_RESOURCES_TIER_2)
        ),
    ],
    None,
    &[C::TiledResources]
);
pub const TILED_RESOURCES_3: Spec = spec!(
    Resource,
    "tiled_resources_3",
    "tier 3: tiled 3D textures",
    T,
    &[
        d12(
            F12::Options,
            "TiledResourcesTier",
            Test::AtLeast(v::D3D12_TILED_RESOURCES_TIER_3)
        ),
        d11(
            F11::Options2,
            "TiledResourcesTier",
            Test::AtLeast(v::D3D11_TILED_RESOURCES_TIER_3)
        ),
    ],
    None,
    &[C::TiledResources2]
);
pub const TILED_RESOURCES_4: Spec = spec!(
    Resource,
    "tiled_resources_4",
    "tier 4: packed mips of 64 KiB textures tiled too",
    T,
    &[d12(
        F12::Options,
        "TiledResourcesTier",
        Test::AtLeast(v::D3D12_TILED_RESOURCES_TIER_4)
    )],
    None,
    &[C::TiledResources3]
);
pub const TYPED_UAV_LOADS: Spec = spec!(
    Resource,
    "typed_uav_loads",
    "typed UAV loads of the additional formats (R16G16B16A16, R8G8B8A8, ...)",
    T,
    &[
        d12(F12::Options, "TypedUAVLoadAdditionalFormats", ON),
        d11(F11::Options2, "TypedUAVLoadAdditionalFormats", ON),
    ]
);
pub const ROVS: Spec = spec!(
    Resource,
    "rovs",
    "rasterizer-ordered views (ordered pixel shader access)",
    T,
    &[
        d12(F12::Options, "ROVsSupported", ON),
        d11(F11::Options2, "ROVsSupported", ON),
    ]
);
pub const STANDARD_SWIZZLE: Spec = spec!(
    Resource,
    "standard_swizzle",
    "the standard 64 KiB texture swizzle, shareable between CPU and GPU",
    U,
    &[
        d12(F12::Options, "StandardSwizzle64KBSupported", ON),
        d11(F11::Options2, "StandardSwizzle", ON),
    ]
);
pub const CASTING_FULLY_TYPED: Spec = spec!(
    Resource,
    "casting_fully_typed",
    "views of another format in the same family on fully typed resources",
    T,
    &[d12(F12::Options3, "CastingFullyTypedFormatSupported", ON)]
);
pub const RELAXED_FORMAT_CASTING: Spec = spec!(
    Resource,
    "relaxed_format_casting",
    "castable formats listed at resource creation",
    T,
    &[d12(F12::Options12, "RelaxedFormatCastingSupported", ON)]
);
pub const UNALIGNED_BLOCK_TEXTURES: Spec = spec!(
    Resource,
    "unaligned_block_textures",
    "block-compressed textures whose size is not a multiple of the block",
    T,
    &[d12(F12::Options8, "UnalignedBlockTexturesSupported", ON)]
);
pub const WRITEABLE_MSAA_TEXTURES: Spec = spec!(
    Resource,
    "writeable_msaa_textures",
    "UAVs on multisampled textures",
    U,
    &[d12(F12::Options14, "WriteableMSAATexturesSupported", ON)]
);

// Raster.
pub const CONSERVATIVE_RASTERIZATION: Spec = spec!(
    Raster,
    "conservative_rasterization",
    "tier 1: every pixel a primitive touches is shaded",
    T,
    &[
        d12(
            F12::Options,
            "ConservativeRasterizationTier",
            Test::AtLeast(v::D3D12_CONSERVATIVE_RASTERIZATION_TIER_1)
        ),
        d11(
            F11::Options2,
            "ConservativeRasterizationTier",
            Test::AtLeast(v::D3D11_CONSERVATIVE_RASTERIZATION_TIER_1)
        ),
    ]
);
pub const CONSERVATIVE_RASTERIZATION_3: Spec = spec!(
    Raster,
    "conservative_rasterization_3",
    "tier 3: SV_InnerCoverage, degenerate triangles not culled",
    T,
    &[
        d12(
            F12::Options,
            "ConservativeRasterizationTier",
            Test::AtLeast(v::D3D12_CONSERVATIVE_RASTERIZATION_TIER_3)
        ),
        d11(
            F11::Options2,
            "ConservativeRasterizationTier",
            Test::AtLeast(v::D3D11_CONSERVATIVE_RASTERIZATION_TIER_3)
        ),
    ],
    None,
    &[C::ConservativeRasterization]
);
pub const DEPTH_BOUNDS: Spec = spec!(
    Raster,
    "depth_bounds",
    "the depth bounds test",
    T,
    &[d12(F12::Options2, "DepthBoundsTestSupported", ON)]
);
pub const PROGRAMMABLE_SAMPLE_POSITIONS: Spec = spec!(
    Raster,
    "programmable_sample_positions",
    "custom MSAA sample positions",
    T,
    &[d12(
        F12::Options2,
        "ProgrammableSamplePositionsTier",
        Test::AtLeast(v::D3D12_PROGRAMMABLE_SAMPLE_POSITIONS_TIER_1)
    )]
);
pub const VIEW_INSTANCING: Spec = spec!(
    Raster,
    "view_instancing",
    "one draw to several views (stereo, cube maps)",
    T,
    &[d12(
        F12::Options3,
        "ViewInstancingTier",
        Test::AtLeast(v::D3D12_VIEW_INSTANCING_TIER_1)
    )],
    sm(6, 1)
);
pub const LOGIC_OPS: Spec = spec!(
    Raster,
    "logic_ops",
    "logic operations in the output merger",
    T,
    &[
        d12(F12::Options, "OutputMergerLogicOp", ON),
        d11(F11::Options, "OutputMergerLogicOp", ON),
    ]
);
pub const TRIANGLE_FANS: Spec = spec!(
    Raster,
    "triangle_fans",
    "triangle fan topology",
    T,
    &[d12(F12::Options15, "TriangleFanSupported", ON)]
);
pub const DYNAMIC_DEPTH_BIAS: Spec = spec!(
    Raster,
    "dynamic_depth_bias",
    "depth bias set on the command list",
    T,
    &[d12(F12::Options16, "DynamicDepthBiasSupported", ON)]
);
pub const INDEPENDENT_FRONT_BACK_STENCIL: Spec = spec!(
    Raster,
    "independent_front_back_stencil",
    "stencil reference and masks apart for front and back faces",
    T,
    &[d12(
        F12::Options14,
        "IndependentFrontAndBackStencilRefMaskSupported",
        ON
    )]
);

// Memory.
pub const RESOURCE_HEAP_TIER_2: Spec = spec!(
    Memory,
    "resource_heap_tier_2",
    "buffers, textures and render targets in one heap",
    T,
    &[d12(
        F12::Options,
        "ResourceHeapTier",
        Test::AtLeast(v::D3D12_RESOURCE_HEAP_TIER_2)
    )]
);
pub const GPU_UPLOAD_HEAP: Spec = spec!(
    Memory,
    "gpu_upload_heap",
    "D3D12_HEAP_TYPE_GPU_UPLOAD: VRAM the CPU writes directly, which needs resizable BAR (pci::bar)",
    T,
    &[d12(F12::Options16, "GPUUploadHeapSupported", ON)]
);
pub const UMA: Spec = spec!(
    Memory,
    "uma",
    "unified memory: the GPU uses system memory (integrated GPUs)",
    U,
    &[
        d12(F12::Architecture1, "UMA", ON),
        d11(F11::Options2, "UnifiedMemoryArchitecture", ON),
    ]
);
pub const CACHE_COHERENT_UMA: Spec = spec!(
    Memory,
    "cache_coherent_uma",
    "unified memory with CPU and GPU caches coherent",
    U,
    &[d12(F12::Architecture1, "CacheCoherentUMA", ON)],
    None,
    &[C::Uma]
);
pub const MAP_ON_DEFAULT_BUFFERS: Spec = spec!(
    Memory,
    "map_on_default_buffers",
    "Map on D3D11_USAGE_DEFAULT buffers",
    T,
    &[d11(F11::Options1, "MapOnDefaultBuffers", ON)]
);
pub const EXISTING_HEAPS: Spec = spec!(
    Memory,
    "existing_heaps",
    "OpenExistingHeapFromAddress: host memory as a heap",
    T,
    &[d12(F12::ExistingHeaps, "Supported", ON)]
);

// Pipeline.
pub const ENHANCED_BARRIERS: Spec = spec!(
    Pipeline,
    "enhanced_barriers",
    "barriers with sync, access and layout apart (Vulkan-style)",
    T,
    &[d12(F12::Options12, "EnhancedBarriersSupported", ON)]
);
pub const RENDER_PASSES: Spec = spec!(
    Pipeline,
    "render_passes",
    "render passes the driver implements (tier 1 and up; tier 0 is emulated)",
    T,
    &[d12(
        F12::Options5,
        "RenderPassesTier",
        Test::AtLeast(v::D3D12_RENDER_PASS_TIER_1)
    )]
);
pub const EXECUTE_INDIRECT_1_1: Spec = spec!(
    Pipeline,
    "execute_indirect_1_1",
    "ExecuteIndirect tier 1.1: incrementing constants",
    U,
    &[d12(
        F12::Options21,
        "ExecuteIndirectTier",
        Test::AtLeast(v::D3D12_EXECUTE_INDIRECT_TIER_1_1)
    )]
);
pub const COPY_QUEUE_TIMESTAMPS: Spec = spec!(
    Pipeline,
    "copy_queue_timestamps",
    "timestamp queries on copy queues",
    T,
    &[d12(F12::Options3, "CopyQueueTimestampQueriesSupported", ON)]
);
pub const BACKGROUND_PROCESSING: Spec = spec!(
    Pipeline,
    "background_processing",
    "the driver may compile and optimise in the background (SetBackgroundProcessingMode)",
    T,
    &[d12(F12::Options6, "BackgroundProcessingSupported", ON)]
);
pub const DRIVER_COMMAND_LISTS: Spec = spec!(
    Pipeline,
    "driver_command_lists",
    "command lists recorded by the driver, not emulated (Direct3D 11 deferred contexts)",
    T,
    &[
        Route::Api(Api::D3D12),
        d11(F11::Threading, "DriverCommandLists", ON),
    ]
);
pub const DRIVER_CONCURRENT_CREATES: Spec = spec!(
    Pipeline,
    "driver_concurrent_creates",
    "resources created from several threads at once",
    T,
    &[
        Route::Api(Api::D3D12),
        d11(F11::Threading, "DriverConcurrentCreates", ON),
    ]
);
