//! Feature levels, shader models, and numerobis's own levels for the
//! versions before feature levels existed.

use std::fmt;

use super::generated::value;
use super::legacy::{Caps9, bits};

/// `D3D_FEATURE_LEVEL`, as Direct3D defines it (10.1 and 11 introduced
/// them; 12 uses them unchanged). Ordered like the raw values.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum FeatureLevel {
    /// Compute only, no shader model guarantees (MCDM devices: NPUs).
    L1_0Generic,
    /// Compute only, a subset of `11_0` compute (MCDM: NPUs, compute GPUs).
    L1_0Core,
    /// Direct3D 9-class hardware through Direct3D 10.1+ (`ps_2_0`).
    L9_1,
    L9_2,
    /// `ps_2_x`, four render targets, instancing.
    L9_3,
    /// Shader model 4.0, geometry shaders, stream output.
    L10_0,
    L10_1,
    /// Shader model 5.0, tessellation, compute shaders.
    L11_0,
    L11_1,
    /// Shader model 5.1 (D3D12), tiled resources tier 2, binding tier 2,
    /// typed UAV loads.
    L12_0,
    /// Rasterizer-ordered views, conservative rasterization.
    L12_1,
    /// DirectX 12 Ultimate: shader model 6.5, raytracing 1.1, mesh shaders,
    /// variable rate shading tier 2, sampler feedback.
    L12_2,
}

impl FeatureLevel {
    /// Oldest first.
    pub const ALL: [Self; 12] = [
        Self::L1_0Generic,
        Self::L1_0Core,
        Self::L9_1,
        Self::L9_2,
        Self::L9_3,
        Self::L10_0,
        Self::L10_1,
        Self::L11_0,
        Self::L11_1,
        Self::L12_0,
        Self::L12_1,
        Self::L12_2,
    ];

    /// The `D3D_FEATURE_LEVEL` value.
    #[must_use]
    pub const fn raw(self) -> u32 {
        match self {
            Self::L1_0Generic => value::D3D_FEATURE_LEVEL_1_0_GENERIC,
            Self::L1_0Core => value::D3D_FEATURE_LEVEL_1_0_CORE,
            Self::L9_1 => value::D3D_FEATURE_LEVEL_9_1,
            Self::L9_2 => value::D3D_FEATURE_LEVEL_9_2,
            Self::L9_3 => value::D3D_FEATURE_LEVEL_9_3,
            Self::L10_0 => value::D3D_FEATURE_LEVEL_10_0,
            Self::L10_1 => value::D3D_FEATURE_LEVEL_10_1,
            Self::L11_0 => value::D3D_FEATURE_LEVEL_11_0,
            Self::L11_1 => value::D3D_FEATURE_LEVEL_11_1,
            Self::L12_0 => value::D3D_FEATURE_LEVEL_12_0,
            Self::L12_1 => value::D3D_FEATURE_LEVEL_12_1,
            Self::L12_2 => value::D3D_FEATURE_LEVEL_12_2,
        }
    }

    #[must_use]
    pub const fn from_raw(raw: u32) -> Option<Self> {
        let mut i = 0;
        while i < Self::ALL.len() {
            if Self::ALL[i].raw() == raw {
                return Some(Self::ALL[i]);
            }
            i += 1;
        }
        None
    }

    /// `12_2`, `1_0_core`.
    #[must_use]
    pub const fn name(self) -> &'static str {
        match self {
            Self::L1_0Generic => "1_0_generic",
            Self::L1_0Core => "1_0_core",
            Self::L9_1 => "9_1",
            Self::L9_2 => "9_2",
            Self::L9_3 => "9_3",
            Self::L10_0 => "10_0",
            Self::L10_1 => "10_1",
            Self::L11_0 => "11_0",
            Self::L11_1 => "11_1",
            Self::L12_0 => "12_0",
            Self::L12_1 => "12_1",
            Self::L12_2 => "12_2",
        }
    }

    /// Compute-only levels (no graphics pipeline): MCDM devices such as
    /// NPUs, which `D3D12CreateDevice` reaches through DXCore.
    #[must_use]
    pub const fn is_compute_only(self) -> bool {
        matches!(self, Self::L1_0Generic | Self::L1_0Core)
    }

    /// The shader model every device at this level has: `vs_4_0_level_9_x`
    /// on `9_x`, 4.0 / 4.1 on `10_x`, 5.0 on `11_x`, 5.1 on `12_0` and `12_1` (as
    /// Direct3D 12 sees them), 6.5 on `12_2`. `None` on the compute-only
    /// levels.
    #[must_use]
    pub const fn shader_model(self) -> Option<ShaderModel> {
        Some(match self {
            Self::L1_0Generic | Self::L1_0Core => return None,
            Self::L9_1 | Self::L9_2 | Self::L9_3 | Self::L10_0 => ShaderModel::new(4, 0),
            Self::L10_1 => ShaderModel::new(4, 1),
            Self::L11_0 | Self::L11_1 => ShaderModel::new(5, 0),
            Self::L12_0 | Self::L12_1 => ShaderModel::new(5, 1),
            Self::L12_2 => ShaderModel::new(6, 5),
        })
    }

    /// The levels from `newest` down to `oldest`, as the raw array
    /// `D3D11CreateDevice` takes.
    #[must_use]
    pub fn down_from(newest: Self, oldest: Self) -> Vec<u32> {
        Self::ALL
            .iter()
            .rev()
            .filter(|&&l| l <= newest && l >= oldest && !l.is_compute_only())
            .map(|l| l.raw())
            .collect()
    }
}

impl fmt::Display for FeatureLevel {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.name())
    }
}

/// A shader model or version: HLSL's `6_5`, `D3D_SHADER_MODEL` (0x65), or a
/// `D3DCAPS` `VertexShaderVersion` / `PixelShaderVersion` (`ps_1_4`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct ShaderModel {
    pub major: u8,
    pub minor: u8,
}

impl ShaderModel {
    /// The newest `D3D_SHADER_MODEL` the tables know.
    pub const NEWEST: Self = Self::new(6, 9);

    #[must_use]
    pub const fn new(major: u8, minor: u8) -> Self {
        Self { major, minor }
    }

    /// `D3D_SHADER_MODEL` (`0x65` for 6.5).
    #[must_use]
    pub const fn raw(self) -> u32 {
        ((self.major as u32) << 4) | self.minor as u32
    }

    #[must_use]
    pub const fn from_raw(raw: u32) -> Self {
        #[allow(clippy::cast_possible_truncation, reason = "a nibble each")]
        Self::new(((raw >> 4) & 0xF) as u8, (raw & 0xF) as u8)
    }

    /// The `D3D_SHADER_MODEL`s Direct3D 12 knows, newest first: what to put
    /// in `HighestShaderModel` until `CheckFeatureSupport` accepts one (a
    /// runtime fails a model newer than itself).
    #[must_use]
    pub fn d3d12_query_order() -> Vec<u32> {
        let mut out = vec![value::D3D_SHADER_MODEL_5_1];
        out.extend((0..=9).map(|minor| Self::new(6, minor).raw()));
        out.reverse();
        out
    }

    /// From a `D3DCAPS` `VertexShaderVersion` (`0xFFFE0000 | major << 8 |
    /// minor`); `None` without vertex shaders.
    #[must_use]
    pub const fn from_vs(caps: u32) -> Option<Self> {
        Self::from_caps(caps, 0xFFFE)
    }

    /// From a `D3DCAPS` `PixelShaderVersion` (`0xFFFF0000 | ...`).
    #[must_use]
    pub const fn from_ps(caps: u32) -> Option<Self> {
        Self::from_caps(caps, 0xFFFF)
    }

    const fn from_caps(caps: u32, kind: u32) -> Option<Self> {
        #[allow(clippy::cast_possible_truncation, reason = "a byte each")]
        let v = Self::new((caps >> 8) as u8, caps as u8);
        if caps >> 16 != kind || v.major == 0 {
            None
        } else {
            Some(v)
        }
    }

    /// As `D3DCAPS` writes a vertex shader version.
    #[must_use]
    pub const fn vs_caps(self) -> u32 {
        0xFFFE_0000 | ((self.major as u32) << 8) | self.minor as u32
    }

    /// As `D3DCAPS` writes a pixel shader version.
    #[must_use]
    pub const fn ps_caps(self) -> u32 {
        0xFFFF_0000 | ((self.major as u32) << 8) | self.minor as u32
    }
}

impl fmt::Display for ShaderModel {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}.{}", self.major, self.minor)
    }
}

/// numerobis's levels for Direct3D 8 and 9, which had none: what the
/// shader versions in `D3DCAPS` say about the hardware generation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum LegacyLevel {
    /// No programmable shaders: the fixed-function pipeline only (Direct3D
    /// 7-class hardware; hardware T&L is a capability of its own).
    FixedFunction,
    /// `vs_1_1`, `ps_1_1` to `ps_1_3` (Direct3D 8.0 hardware).
    Sm1,
    /// `ps_1_4` (Direct3D 8.1).
    Sm1_4,
    /// `vs_2_0`, `ps_2_0` (Direct3D 9).
    Sm2,
    /// `ps_2_x` with at least 512 instruction slots (`ps_2_a` / `ps_2_b`).
    Sm2x,
    /// `vs_3_0`, `ps_3_0` (Direct3D 9.0c).
    Sm3,
}

impl LegacyLevel {
    /// From the shader versions `caps` reports (the lower of the two
    /// counts; a device with a vertex shader but no pixel shader is still
    /// fixed function where it draws).
    #[must_use]
    pub fn of(caps: &Caps9) -> Self {
        let vs = ShaderModel::from_vs(caps.vertex_shader_version);
        let ps = ShaderModel::from_ps(caps.pixel_shader_version);
        let (Some(vs), Some(ps)) = (vs, ps) else {
            return Self::FixedFunction;
        };
        let low = vs.min(ps);
        if low >= ShaderModel::new(3, 0) {
            Self::Sm3
        } else if low >= ShaderModel::new(2, 0) {
            if ps >= ShaderModel::new(2, 0) && caps.ps20_caps_num_instruction_slots >= 512 {
                Self::Sm2x
            } else {
                Self::Sm2
            }
        } else if ps >= ShaderModel::new(1, 4) {
            Self::Sm1_4
        } else {
            Self::Sm1
        }
    }

    #[must_use]
    pub const fn name(self) -> &'static str {
        match self {
            Self::FixedFunction => "fixed function",
            Self::Sm1 => "shader model 1",
            Self::Sm1_4 => "shader model 1.4",
            Self::Sm2 => "shader model 2",
            Self::Sm2x => "shader model 2.x",
            Self::Sm3 => "shader model 3",
        }
    }

    /// The `9_x` feature level a Direct3D 9 device with these caps matches,
    /// from the caps Microsoft's feature level table lists (texture sizes,
    /// anisotropy, primitive counts, separate alpha blend, mirror once,
    /// render targets). An approximation: `d3d10level9` decides from the
    /// driver's DDI caps, which can differ. `None` below shader model 2.
    #[must_use]
    pub fn level9(caps: &Caps9) -> Option<FeatureLevel> {
        if Self::of(caps) < Self::Sm2 {
            return None;
        }
        let l9_1 = caps.max_texture_width >= 2048
            && caps.max_texture_height >= 2048
            && caps.max_volume_extent >= 256
            && caps.max_anisotropy >= 2
            && caps.max_primitive_count >= 65535
            && caps.max_vertex_index >= 65534;
        if !l9_1 {
            return None;
        }
        let l9_2 = caps.max_anisotropy >= 16
            && caps.max_primitive_count >= 1_048_575
            && caps.max_vertex_index >= 1_048_575
            && caps.primitive_misc_caps & bits::D3DPMISCCAPS_SEPARATEALPHABLEND != 0
            && caps.texture_address_caps & bits::D3DPTADDRESSCAPS_MIRRORONCE != 0;
        if !l9_2 {
            return Some(FeatureLevel::L9_1);
        }
        let l9_3 = caps.max_texture_width >= 4096
            && caps.max_texture_height >= 4096
            && caps.num_simultaneous_rts >= 4
            && Self::of(caps) >= Self::Sm2x;
        Some(if l9_3 {
            FeatureLevel::L9_3
        } else {
            FeatureLevel::L9_2
        })
    }
}

impl fmt::Display for LegacyLevel {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.name())
    }
}
