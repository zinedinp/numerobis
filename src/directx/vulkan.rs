//! Direct3D next to Vulkan (features `directx` and `vulkan`): values that
//! mean the same converted in `const fn`s, and which Vulkan capabilities a
//! translation layer needs to offer a Direct3D one.
//!
//! The capability and level tables follow what DXVK (`d3d11_features.cpp`,
//! `dxvk_device_info.cpp`) and vkd3d-proton (`libs/vkd3d/device.c`) check
//! before they report a feature; they hold the [`vulkan::Capability`]s
//! numerobis knows, so a few conditions (format features, limits,
//! properties) are left out and named in the docs.
//!
//! ```
//! use numerobis::directx::modern::{Capability as Dx, QueuePriority};
//! use numerobis::directx::vulkan::{format, needs};
//! use numerobis::vulkan::{Capability as Vk, GlobalPriority};
//!
//! assert_eq!(QueuePriority::from_vulkan(GlobalPriority::High), QueuePriority::High);
//! assert_eq!(format::to_vk(format::dxgi::BC7_UNORM), Some(format::vk::BC7_UNORM_BLOCK));
//! assert_eq!(needs(Dx::MeshShader), Some(&[Vk::MeshShader, Vk::TaskShader][..]));
//! ```

use super::level::FeatureLevel;
use super::modern::{Capability as Dx, QueuePriority};
use crate::vulkan::{self, Capability as Vk, GlobalPriority};

impl QueuePriority {
    /// Vulkan's global priority as a Direct3D 12 queue priority. Direct3D
    /// has nothing below normal, so low becomes normal.
    #[must_use]
    pub const fn from_vulkan(p: GlobalPriority) -> Self {
        match p {
            GlobalPriority::Low | GlobalPriority::Medium => Self::Normal,
            GlobalPriority::High => Self::High,
            GlobalPriority::Realtime => Self::GlobalRealtime,
        }
    }

    /// The other way: normal is Vulkan's default, medium.
    #[must_use]
    pub const fn to_vulkan(self) -> GlobalPriority {
        match self {
            Self::Normal => GlobalPriority::Medium,
            Self::High => GlobalPriority::High,
            Self::GlobalRealtime => GlobalPriority::Realtime,
        }
    }
}

/// `DXGI_FORMAT` and `VkFormat`, for the formats both sides share, as
/// DXVK maps them (`src/dxgi/dxgi_format.cpp`). Depth formats map to the
/// Vulkan depth format; `R16_FLOAT` / `R32_FLOAT` to the colour one.
pub mod format {
    /// `DXGI_FORMAT` values.
    #[allow(missing_docs)]
    pub mod dxgi {
        pub const R32G32B32A32_FLOAT: u32 = 2;
        pub const R16G16B16A16_FLOAT: u32 = 10;
        pub const D32_FLOAT_S8X24_UINT: u32 = 20;
        pub const R10G10B10A2_UNORM: u32 = 24;
        pub const R11G11B10_FLOAT: u32 = 26;
        pub const R8G8B8A8_UNORM: u32 = 28;
        pub const R8G8B8A8_UNORM_SRGB: u32 = 29;
        pub const R16G16_FLOAT: u32 = 34;
        pub const D32_FLOAT: u32 = 40;
        pub const R32_FLOAT: u32 = 41;
        pub const R32_UINT: u32 = 42;
        pub const D24_UNORM_S8_UINT: u32 = 45;
        pub const R8G8_UNORM: u32 = 49;
        pub const R16_FLOAT: u32 = 54;
        pub const D16_UNORM: u32 = 55;
        pub const R8_UNORM: u32 = 61;
        pub const BC1_UNORM: u32 = 71;
        pub const BC1_UNORM_SRGB: u32 = 72;
        pub const BC2_UNORM: u32 = 74;
        pub const BC3_UNORM: u32 = 77;
        pub const BC3_UNORM_SRGB: u32 = 78;
        pub const BC4_UNORM: u32 = 80;
        pub const BC5_UNORM: u32 = 83;
        pub const B8G8R8A8_UNORM: u32 = 87;
        pub const B8G8R8A8_UNORM_SRGB: u32 = 91;
        pub const BC6H_UF16: u32 = 95;
        pub const BC6H_SF16: u32 = 96;
        pub const BC7_UNORM: u32 = 98;
        pub const BC7_UNORM_SRGB: u32 = 99;
    }

    /// `VkFormat` values.
    #[allow(missing_docs)]
    pub mod vk {
        pub const R8_UNORM: u32 = 9;
        pub const R8G8_UNORM: u32 = 16;
        pub const R8G8B8A8_UNORM: u32 = 37;
        pub const R8G8B8A8_SRGB: u32 = 43;
        pub const B8G8R8A8_UNORM: u32 = 44;
        pub const B8G8R8A8_SRGB: u32 = 50;
        pub const A2B10G10R10_UNORM_PACK32: u32 = 64;
        pub const R16_SFLOAT: u32 = 76;
        pub const R16G16_SFLOAT: u32 = 83;
        pub const R16G16B16A16_SFLOAT: u32 = 97;
        pub const R32_UINT: u32 = 98;
        pub const R32_SFLOAT: u32 = 100;
        pub const R32G32B32A32_SFLOAT: u32 = 109;
        pub const B10G11R11_UFLOAT_PACK32: u32 = 122;
        pub const D16_UNORM: u32 = 124;
        pub const D32_SFLOAT: u32 = 126;
        pub const D24_UNORM_S8_UINT: u32 = 129;
        pub const D32_SFLOAT_S8_UINT: u32 = 130;
        pub const BC1_RGBA_UNORM_BLOCK: u32 = 133;
        pub const BC1_RGBA_SRGB_BLOCK: u32 = 134;
        pub const BC2_UNORM_BLOCK: u32 = 135;
        pub const BC3_UNORM_BLOCK: u32 = 137;
        pub const BC3_SRGB_BLOCK: u32 = 138;
        pub const BC4_UNORM_BLOCK: u32 = 139;
        pub const BC5_UNORM_BLOCK: u32 = 141;
        pub const BC6H_UFLOAT_BLOCK: u32 = 143;
        pub const BC6H_SFLOAT_BLOCK: u32 = 144;
        pub const BC7_UNORM_BLOCK: u32 = 145;
        pub const BC7_SRGB_BLOCK: u32 = 146;
    }

    /// `(DXGI_FORMAT, VkFormat)`.
    pub const TABLE: &[(u32, u32)] = &[
        (dxgi::R32G32B32A32_FLOAT, vk::R32G32B32A32_SFLOAT),
        (dxgi::R16G16B16A16_FLOAT, vk::R16G16B16A16_SFLOAT),
        (dxgi::D32_FLOAT_S8X24_UINT, vk::D32_SFLOAT_S8_UINT),
        (dxgi::R10G10B10A2_UNORM, vk::A2B10G10R10_UNORM_PACK32),
        (dxgi::R11G11B10_FLOAT, vk::B10G11R11_UFLOAT_PACK32),
        (dxgi::R8G8B8A8_UNORM, vk::R8G8B8A8_UNORM),
        (dxgi::R8G8B8A8_UNORM_SRGB, vk::R8G8B8A8_SRGB),
        (dxgi::R16G16_FLOAT, vk::R16G16_SFLOAT),
        (dxgi::D32_FLOAT, vk::D32_SFLOAT),
        (dxgi::R32_FLOAT, vk::R32_SFLOAT),
        (dxgi::R32_UINT, vk::R32_UINT),
        (dxgi::D24_UNORM_S8_UINT, vk::D24_UNORM_S8_UINT),
        (dxgi::R8G8_UNORM, vk::R8G8_UNORM),
        (dxgi::R16_FLOAT, vk::R16_SFLOAT),
        (dxgi::D16_UNORM, vk::D16_UNORM),
        (dxgi::R8_UNORM, vk::R8_UNORM),
        (dxgi::BC1_UNORM, vk::BC1_RGBA_UNORM_BLOCK),
        (dxgi::BC1_UNORM_SRGB, vk::BC1_RGBA_SRGB_BLOCK),
        (dxgi::BC2_UNORM, vk::BC2_UNORM_BLOCK),
        (dxgi::BC3_UNORM, vk::BC3_UNORM_BLOCK),
        (dxgi::BC3_UNORM_SRGB, vk::BC3_SRGB_BLOCK),
        (dxgi::BC4_UNORM, vk::BC4_UNORM_BLOCK),
        (dxgi::BC5_UNORM, vk::BC5_UNORM_BLOCK),
        (dxgi::B8G8R8A8_UNORM, vk::B8G8R8A8_UNORM),
        (dxgi::B8G8R8A8_UNORM_SRGB, vk::B8G8R8A8_SRGB),
        (dxgi::BC6H_UF16, vk::BC6H_UFLOAT_BLOCK),
        (dxgi::BC6H_SF16, vk::BC6H_SFLOAT_BLOCK),
        (dxgi::BC7_UNORM, vk::BC7_UNORM_BLOCK),
        (dxgi::BC7_UNORM_SRGB, vk::BC7_SRGB_BLOCK),
    ];

    /// The `VkFormat` for a `DXGI_FORMAT`.
    #[must_use]
    pub const fn to_vk(dxgi: u32) -> Option<u32> {
        let mut i = 0;
        while i < TABLE.len() {
            if TABLE[i].0 == dxgi {
                return Some(TABLE[i].1);
            }
            i += 1;
        }
        None
    }

    /// The `DXGI_FORMAT` for a `VkFormat`.
    #[must_use]
    pub const fn from_vk(vk: u32) -> Option<u32> {
        let mut i = 0;
        while i < TABLE.len() {
            if TABLE[i].1 == vk {
                return Some(TABLE[i].0);
            }
            i += 1;
        }
        None
    }

    // One to one, checked while compiling: each side appears once.
    const _: () = {
        let mut i = 0;
        while i < TABLE.len() {
            let mut j = i + 1;
            while j < TABLE.len() {
                assert!(
                    TABLE[i].0 != TABLE[j].0 && TABLE[i].1 != TABLE[j].1,
                    "a format twice"
                );
                j += 1;
            }
            i += 1;
        }
    };
}

/// The Vulkan capabilities a translation layer needs before it reports
/// `cap`. `Some(&[])`: it always does (part of every device it accepts);
/// `None`: it hangs on something the Vulkan tables don't describe (format
/// features, limits, memory topology) or the layers don't offer it.
///
/// Left out, where `Some`: vkd3d-proton also checks properties for
/// raytracing (handle sizes, vertex formats), variable rate shading (tile
/// sizes), 16-bit ops (denorm behaviour) and the shader model.
#[must_use]
pub const fn needs(cap: Dx) -> Option<&'static [Vk]> {
    Some(match cap {
        Dx::GeometryShader => &[Vk::GeometryShader],
        Dx::Tessellation => &[Vk::TessellationShader],
        Dx::ComputeShader
        | Dx::RootSignature11
        | Dx::EnhancedBarriers
        | Dx::DriverCommandLists
        | Dx::DriverConcurrentCreates => &[],
        Dx::Doubles => &[Vk::Float64],
        Dx::MinPrecision16 => &[Vk::Float16, Vk::Int16],
        Dx::Native16Bit => &[Vk::Float16, Vk::Int16, Vk::Storage16Bit],
        Dx::Int64 => &[Vk::Int64],
        Dx::Barycentrics => &[Vk::FragmentShaderBarycentric],
        Dx::AtomicInt64Typed | Dx::AtomicInt64DescriptorHeap => &[Vk::ImageAtomicInt64],
        Dx::AtomicInt64GroupShared => &[Vk::AtomicInt64],
        Dx::WaveOps => &[
            Vk::SubgroupVote,
            Vk::SubgroupArithmetic,
            Vk::SubgroupBallot,
            Vk::SubgroupShuffle,
        ],
        Dx::Raytracing => &[Vk::AccelerationStructure, Vk::RayTracingPipeline],
        Dx::Raytracing11 => &[
            Vk::AccelerationStructure,
            Vk::RayTracingPipeline,
            Vk::RayQuery,
            Vk::RayTraversalPrimitiveCulling,
        ],
        Dx::Raytracing12 => &[
            Vk::AccelerationStructure,
            Vk::RayTracingPipeline,
            Vk::RayQuery,
            Vk::RayTraversalPrimitiveCulling,
            Vk::OpacityMicromap,
        ],
        Dx::MeshShader => &[Vk::MeshShader, Vk::TaskShader],
        Dx::VariableRateShading => &[Vk::FragmentShadingRate],
        Dx::VariableRateShading2 => &[Vk::FragmentShadingRate, Vk::FragmentShadingRateAttachment],
        Dx::SamplerFeedback => &[Vk::Int64, Vk::ImageAtomicInt64],
        Dx::BindingTier2 | Dx::BindingTier3 | Dx::DynamicResources => {
            &[Vk::BindlessTextures, Vk::BindlessBuffers]
        }
        Dx::TiledResources => &[Vk::SparseResidency],
        Dx::Rovs => &[Vk::FragmentShaderInterlock],
        Dx::ConservativeRasterization => &[Vk::ConservativeRasterization],
        Dx::DepthBounds => &[Vk::DepthBounds],
        Dx::LogicOps => &[Vk::LogicOp],
        Dx::ViewInstancing => &[Vk::Multiview],
        _ => return None,
    })
}

/// Whether a Vulkan device with `plan` gives what a translation layer
/// needs for `cap` ([`needs`]); `None` where the tables can't tell.
#[must_use]
pub fn translatable(plan: &vulkan::Plan, cap: Dx) -> Option<bool> {
    needs(cap).map(|n| plan.lacking(n).is_empty())
}

/// What DXVK (Direct3D 10 and 11) and vkd3d-proton (12) check for each
/// feature level on top of the one below. Below `11_0` there is no check: a
/// device DXVK accepts at all is `10_1`. Left out: DXVK's `11_0` also needs
/// `drawIndirectFirstInstance` and `fragmentStoresAndAtomics`, `11_1`
/// `vertexPipelineStoresAndAtomics`, `12_0` the typed UAV load formats and
/// tiled resources tier 2; `12_2` (vkd3d-proton) also shader model 6.5 and
/// tiled resources tier 3.
#[must_use]
pub const fn level_needs(level: FeatureLevel) -> &'static [Vk] {
    match level {
        FeatureLevel::L11_0 => &[Vk::MultiDrawIndirect, Vk::TessellationShader],
        FeatureLevel::L11_1 => &[Vk::LogicOp],
        FeatureLevel::L12_0 => &[
            Vk::SparseResidency,
            Vk::BindlessTextures,
            Vk::BindlessBuffers,
        ],
        FeatureLevel::L12_1 => &[Vk::ConservativeRasterization, Vk::FragmentShaderInterlock],
        FeatureLevel::L12_2 => &[
            Vk::RayQuery,
            Vk::RayTraversalPrimitiveCulling,
            Vk::MeshShader,
            Vk::TaskShader,
            Vk::FragmentShadingRateAttachment,
            Vk::Int64,
            Vk::ImageAtomicInt64,
            Vk::DepthBounds,
        ],
        _ => &[],
    }
}

/// The highest feature level the checks of [`level_needs`] allow on a
/// Vulkan device with `plan`, from `10_1` (DXVK's floor) up to `12_2`.
#[must_use]
pub fn max_level(plan: &vulkan::Plan) -> FeatureLevel {
    let mut best = FeatureLevel::L10_1;
    for level in [
        FeatureLevel::L11_0,
        FeatureLevel::L11_1,
        FeatureLevel::L12_0,
        FeatureLevel::L12_1,
        FeatureLevel::L12_2,
    ] {
        if !plan.lacking(level_needs(level)).is_empty() {
            break;
        }
        best = level;
    }
    best
}
