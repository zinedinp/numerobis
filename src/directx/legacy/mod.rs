//! Direct3D 8 and 9: one caps struct per device, read with
//! `IDirect3D8::GetDeviceCaps` / `IDirect3D9::GetDeviceCaps`, no device
//! needed.
//!
//! `D3DCAPS8` is the first [`CAPS8_WORDS`] words of `D3DCAPS9`, so one
//! [`Caps9`] holds either: hand Direct3D 8 the same buffer and the rest
//! stays zero. What only Direct3D 9 has (`DevCaps2`, `NumSimultaneousRTs`,
//! the 2.0 shader caps, ...) is then simply not granted.
//!
//! Direct3D 8 on a 64-bit process: Microsoft ships `d3d8.dll` for 32-bit
//! only. The ways there, evaluated:
//!
//! - **d3d8to9** (crosire, BSD-2-Clause): a 32-bit `d3d8.dll` that turns
//!   each call into Direct3D 9; its x86-only build and 32-bit shader
//!   handles keep it out of 64-bit processes. Its caps conversion is a
//!   prefix copy with a handful of changes.
//! - **DXVK's d3d8** (zlib): the same idea on DXVK's Direct3D 9, 64-bit, on
//!   Vulkan. Its `ConvertCaps8` also clears the bits Direct3D 8 doesn't
//!   define.
//!
//! [`Caps9::as_d3d8`] is that conversion (DXVK's bit list plus d3d8to9's
//! constant clamp), done on the host's Direct3D 9 caps: what a game would
//! see through either wrapper, without loading one.
//!
//! [`plan`] reads the caps into [`Capability`]s, grouped like
//! [`modern`](super::modern) but by what the hardware generation had:
//! fixed function, shaders, textures, output merger, vertices.

use std::fmt;

pub use super::generated::{CAPS8_WORDS, CAPS9_MEMBERS, CAPS9_WORDS, Caps9, caps9 as bits};
use super::level::{FeatureLevel, LegacyLevel, ShaderModel};
use super::{Api, Era, same};
use crate::platform::{Os, Support};

/// The `D3DCAPS8` bits Direct3D 9 dropped, which the wrappers set.
pub mod bits8 {
    /// `D3DCAPS2_CANRENDERWINDOWED`.
    pub const D3DCAPS2_CANRENDERWINDOWED: u32 = 0x0008_0000;
    /// `D3DPRASTERCAPS_ZBIAS` (Direct3D 9 has `DEPTHBIAS` instead).
    pub const D3DPRASTERCAPS_ZBIAS: u32 = 0x0000_4000;
}

/// `Direct3DCreate8`'s `SDKVersion` (`D3D_SDK_VERSION` of the DirectX 8.1
/// SDK). Both wrappers ignore it.
pub const D3D8_SDK_VERSION: u32 = 220;
/// `D3DDEVTYPE_HAL`, the device type to ask caps for.
pub const D3DDEVTYPE_HAL: u32 = 1;

const _: () = assert!(size_of::<Caps9>() == 4 * CAPS9_WORDS);
const _: () = assert!(CAPS9_WORDS == 76 && CAPS8_WORDS == 53);

impl Caps9 {
    /// The caps as their 4-byte words, in declaration order.
    #[must_use]
    pub fn words(&self) -> &[u32; CAPS9_WORDS] {
        // SAFETY: `Caps9` is `repr(C)` with `CAPS9_WORDS` `u32`s, nothing else.
        unsafe { &*(&raw const *self).cast::<[u32; CAPS9_WORDS]>() }
    }

    #[must_use]
    pub fn from_words(words: [u32; CAPS9_WORDS]) -> Self {
        // SAFETY: as above; every bit pattern is a valid `u32`.
        unsafe { std::mem::transmute::<[u32; CAPS9_WORDS], Self>(words) }
    }

    /// `VertexShaderVersion`; `None` without hardware vertex shaders.
    #[must_use]
    pub const fn vertex_shader(&self) -> Option<ShaderModel> {
        ShaderModel::from_vs(self.vertex_shader_version)
    }

    /// `PixelShaderVersion`.
    #[must_use]
    pub const fn pixel_shader(&self) -> Option<ShaderModel> {
        ShaderModel::from_ps(self.pixel_shader_version)
    }

    /// `MaxPointSize`.
    #[must_use]
    pub const fn max_point_size(&self) -> f32 {
        f32::from_bits(self.max_point_size)
    }

    /// The `D3DCAPS8` a Direct3D 8 wrapper on these Direct3D 9 caps reports:
    /// the prefix, shader versions capped at `vs_1_1` / `ps_1_4`, at most
    /// 256 vertex shader constants, the Direct3D 9-only bits cleared and
    /// `CANRENDERWINDOWED` / `ZBIAS` set. The Direct3D 9-only words are
    /// zero.
    ///
    /// The bits are DXVK's list plus three Direct3D 9 / 9Ex bits both
    /// wrappers pass through (`CANSHARERESOURCE`, `POSTBLENDSRGBCONVERT`,
    /// `NOPROJECTEDBUMPENV`), which Microsoft's own 32-bit Direct3D 8
    /// runtime didn't report on the B580. Measured there against that
    /// runtime: 48 of 53 words equal and every [`Capability`] the same;
    /// it differs in `Caps3` (it keeps `LINEAR_TO_SRGB_PRESENTATION`), one
    /// `DevCaps` bit Direct3D 9 doesn't define, and 8192 instead of 16384
    /// for the texture size limits.
    #[must_use]
    pub fn as_d3d8(&self) -> Self {
        use bits as b;
        let mut words = [0u32; CAPS9_WORDS];
        words[..CAPS8_WORDS].copy_from_slice(&self.words()[..CAPS8_WORDS]);
        let mut c = Self::from_words(words);
        c.caps2 &= !(b::D3DCAPS2_CANAUTOGENMIPMAP | b::D3DCAPS2_CANSHARERESOURCE);
        c.caps2 |= bits8::D3DCAPS2_CANRENDERWINDOWED;
        c.caps3 &= !(b::D3DCAPS3_LINEAR_TO_SRGB_PRESENTATION
            | b::D3DCAPS3_COPY_TO_VIDMEM
            | b::D3DCAPS3_COPY_TO_SYSTEMMEM);
        c.primitive_misc_caps &= !(b::D3DPMISCCAPS_INDEPENDENTWRITEMASKS
            | b::D3DPMISCCAPS_PERSTAGECONSTANT
            | b::D3DPMISCCAPS_FOGANDSPECULARALPHA
            | b::D3DPMISCCAPS_SEPARATEALPHABLEND
            | b::D3DPMISCCAPS_MRTINDEPENDENTBITDEPTHS
            | b::D3DPMISCCAPS_MRTPOSTPIXELSHADERBLENDING
            | b::D3DPMISCCAPS_FOGVERTEXCLAMPED
            | b::D3DPMISCCAPS_POSTBLENDSRGBCONVERT);
        c.texture_caps &= !b::D3DPTEXTURECAPS_NOPROJECTEDBUMPENV;
        c.raster_caps &= !(b::D3DPRASTERCAPS_SCISSORTEST
            | b::D3DPRASTERCAPS_SLOPESCALEDEPTHBIAS
            | b::D3DPRASTERCAPS_DEPTHBIAS
            | b::D3DPRASTERCAPS_MULTISAMPLE_TOGGLE);
        c.raster_caps |= bits8::D3DPRASTERCAPS_ZBIAS;
        c.src_blend_caps &= !b::D3DPBLENDCAPS_BLENDFACTOR;
        c.dest_blend_caps &= !b::D3DPBLENDCAPS_BLENDFACTOR;
        c.line_caps &= !b::D3DLINECAPS_ANTIALIAS;
        c.stencil_caps &= !b::D3DSTENCILCAPS_TWOSIDED;
        c.vertex_processing_caps &= !b::D3DVTXPCAPS_TEXGEN_SPHEREMAP;
        if let Some(vs) = self.vertex_shader() {
            c.vertex_shader_version = vs.min(ShaderModel::new(1, 1)).vs_caps();
        }
        if let Some(ps) = self.pixel_shader() {
            c.pixel_shader_version = ps.min(ShaderModel::new(1, 4)).ps_caps();
        }
        c.max_vertex_shader_const = c.max_vertex_shader_const.min(256);
        c
    }
}

/// The index of `D3DCAPS9` member `name` (flattened: `PS20Caps.NumTemps`).
/// `const`, so the tables below are checked while compiling.
///
/// # Panics
///
/// When there is no such member (while compiling, in a table).
#[must_use]
pub const fn word(name: &str) -> usize {
    let mut i = 0;
    while i < CAPS9_WORDS {
        if same(CAPS9_MEMBERS[i], name) {
            return i;
        }
        i += 1;
    }
    panic!("not a D3DCAPS9 member");
}

/// The area a legacy capability belongs to.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Area {
    /// Transform and lighting, vertex blending, fog, texture stages: what
    /// the fixed-function pipeline does in hardware.
    FixedFunction,
    Shader,
    Texture,
    /// Blending, stencil, render targets, rasterizer state.
    Output,
    Vertex,
}

impl Area {
    #[must_use]
    pub const fn name(self) -> &'static str {
        match self {
            Self::FixedFunction => "fixed function",
            Self::Shader => "shader",
            Self::Texture => "texture",
            Self::Output => "output",
            Self::Vertex => "vertex",
        }
    }
}

/// How a capability is read from the caps.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Check {
    /// Every bit of `mask` set in word `word`.
    Bits { word: usize, mask: u32 },
    /// Every bit of `mask` clear (`D3DPTEXTURECAPS_POW2` means a limit).
    Clear { word: usize, mask: u32 },
    /// All of `set` set, or all of `clear` clear: a limit bit that only
    /// matters when another is set (`NONPOW2CONDITIONAL` with `POW2`).
    SetOrClear { word: usize, set: u32, clear: u32 },
    /// Word `word` at least `min`.
    AtLeast { word: usize, min: u32 },
    /// The `float` in word `word` at least `min`.
    FloatAtLeast { word: usize, min: f32 },
    /// `VertexShaderVersion` at least this.
    Vs(ShaderModel),
    /// `PixelShaderVersion` at least this.
    Ps(ShaderModel),
}

const fn bits(name: &str, mask: u32) -> Check {
    Check::Bits {
        word: word(name),
        mask,
    }
}

const fn clear(name: &str, mask: u32) -> Check {
    Check::Clear {
        word: word(name),
        mask,
    }
}

const fn at_least(name: &str, min: u32) -> Check {
    Check::AtLeast {
        word: word(name),
        min,
    }
}

impl Check {
    /// The caps word it reads, if one.
    #[must_use]
    pub const fn word(self) -> Option<usize> {
        match self {
            Self::Bits { word, .. }
            | Self::Clear { word, .. }
            | Self::SetOrClear { word, .. }
            | Self::AtLeast { word, .. }
            | Self::FloatAtLeast { word, .. } => Some(word),
            Self::Vs(_) => Some(word("VertexShaderVersion")),
            Self::Ps(_) => Some(word("PixelShaderVersion")),
        }
    }

    #[must_use]
    pub fn passes(self, caps: &Caps9) -> bool {
        let w = caps.words();
        match self {
            Self::Bits { word, mask } => w[word] & mask == mask,
            Self::Clear { word, mask } => w[word] & mask == 0,
            Self::SetOrClear { word, set, clear } => w[word] & set == set || w[word] & clear == 0,
            Self::AtLeast { word, min } => w[word] >= min,
            Self::FloatAtLeast { word, min } => f32::from_bits(w[word]) >= min,
            Self::Vs(v) => caps.vertex_shader().is_some_and(|have| have >= v),
            Self::Ps(v) => caps.pixel_shader().is_some_and(|have| have >= v),
        }
    }
}

/// What a legacy capability is and how to read it.
#[derive(Debug, Clone, Copy)]
pub struct Spec {
    pub area: Area,
    pub name: &'static str,
    pub doc: &'static str,
    /// The first version that has it (bits Direct3D 8 doesn't define, words
    /// past its caps).
    pub since: Api,
    pub check: Check,
    pub needs: &'static [Capability],
    pub support: Support,
}

macro_rules! capabilities {
    ($($cap:ident => $spec:ident,)*) => {
        /// A Direct3D 8 / 9 capability.
        #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
        #[non_exhaustive]
        pub enum Capability {
            $(#[doc = concat!("See [`caps::", stringify!($spec), "`].")] $cap,)*
        }

        impl Capability {
            /// Every capability, grouped by area.
            pub const ALL: &[Capability] = &[$(Capability::$cap,)*];

            #[must_use]
            pub const fn spec(self) -> &'static Spec {
                match self {
                    $(Capability::$cap => &caps::$spec,)*
                }
            }
        }
    };
}

capabilities! {
    HardwareTnl => HARDWARE_TNL,
    PureDevice => PURE_DEVICE,
    VertexBlending => VERTEX_BLENDING,
    IndexedVertexBlending => INDEXED_VERTEX_BLENDING,
    HardwareLights => HARDWARE_LIGHTS,
    UserClipPlanes => USER_CLIP_PLANES,
    Multitexture => MULTITEXTURE,
    EightTextureStages => EIGHT_TEXTURE_STAGES,
    TableFog => TABLE_FOG,
    PointSprites => POINT_SPRITES,
    NPatches => N_PATCHES,
    VertexShader11 => VERTEX_SHADER_1_1,
    PixelShader11 => PIXEL_SHADER_1_1,
    PixelShader14 => PIXEL_SHADER_1_4,
    VertexShader20 => VERTEX_SHADER_2_0,
    PixelShader20 => PIXEL_SHADER_2_0,
    PixelShader2x => PIXEL_SHADER_2_X,
    VertexShader30 => VERTEX_SHADER_3_0,
    PixelShader30 => PIXEL_SHADER_3_0,
    VertexTextureFetch => VERTEX_TEXTURE_FETCH,
    CubeMaps => CUBE_MAPS,
    VolumeTextures => VOLUME_TEXTURES,
    Mipmaps => MIPMAPS,
    NonPow2Conditional => NON_POW2_CONDITIONAL,
    NonPow2 => NON_POW2,
    AnisotropicFilter => ANISOTROPIC_FILTER,
    Anisotropy16 => ANISOTROPY_16,
    LargeTextures => LARGE_TEXTURES,
    BorderAddress => BORDER_ADDRESS,
    MirrorOnce => MIRROR_ONCE,
    AutoMipmaps => AUTO_MIPMAPS,
    DynamicTextures => DYNAMIC_TEXTURES,
    BlendOp => BLEND_OP,
    ColorWriteEnable => COLOR_WRITE_ENABLE,
    SeparateAlphaBlend => SEPARATE_ALPHA_BLEND,
    TwoSidedStencil => TWO_SIDED_STENCIL,
    MultipleRenderTargets => MULTIPLE_RENDER_TARGETS,
    MrtIndependentBitDepths => MRT_INDEPENDENT_BIT_DEPTHS,
    ScissorTest => SCISSOR_TEST,
    DepthBias => DEPTH_BIAS,
    SlopeScaleDepthBias => SLOPE_SCALE_DEPTH_BIAS,
    Index32 => INDEX_32,
    EightStreams => EIGHT_STREAMS,
    StreamOffset => STREAM_OFFSET,
    Float16Vertices => FLOAT16_VERTICES,
    Ubyte4Vertices => UBYTE4_VERTICES,
    Instancing => INSTANCING,
}

impl Capability {
    #[must_use]
    pub const fn name(self) -> &'static str {
        self.spec().name
    }
}

/// The capability tables, one `const` per capability.
pub mod caps {
    use super::bits as b;
    use super::{
        Api, Area, Capability, Check, ShaderModel, Spec, Support, at_least, bits, clear, word,
    };

    const D8: Api = Api::D3D8;
    const D9: Api = Api::D3D9;
    const T: Support = Support::WINDOWS_TESTED;
    /// Not reported by the hardware it ran on.
    const U: Support = Support::UNTESTED;

    macro_rules! spec {
        ($area:ident, $name:literal, $doc:literal, $since:expr, $check:expr) => {
            spec!($area, $name, $doc, $since, $check, &[])
        };
        ($area:ident, $name:literal, $doc:literal, $since:expr, $check:expr, $needs:expr) => {
            Spec {
                area: Area::$area,
                name: $name,
                doc: $doc,
                since: $since,
                check: $check,
                needs: $needs,
                support: T,
            }
        };
    }

    // Fixed function.
    pub const HARDWARE_TNL: Spec = spec!(
        FixedFunction,
        "hardware_tnl",
        "transform and lighting on the GPU (D3DCREATE_HARDWARE_VERTEXPROCESSING)",
        D8,
        bits("DevCaps", b::D3DDEVCAPS_HWTRANSFORMANDLIGHT)
    );
    pub const PURE_DEVICE: Spec = spec!(
        FixedFunction,
        "pure_device",
        "D3DCREATE_PUREDEVICE: no state shadowing in the runtime",
        D8,
        bits("DevCaps", b::D3DDEVCAPS_PUREDEVICE),
        &[Capability::HardwareTnl]
    );
    pub const VERTEX_BLENDING: Spec = spec!(
        FixedFunction,
        "vertex_blending",
        "skinning with at least two blend matrices",
        D8,
        at_least("MaxVertexBlendMatrices", 2)
    );
    pub const INDEXED_VERTEX_BLENDING: Spec = Spec {
        support: U,
        ..spec!(
            FixedFunction,
            "indexed_vertex_blending",
            "skinning with per-vertex matrix indices",
            D8,
            at_least("MaxVertexBlendMatrixIndex", 1)
        )
    };
    pub const HARDWARE_LIGHTS: Spec = spec!(
        FixedFunction,
        "hardware_lights",
        "eight active fixed-function lights",
        D8,
        at_least("MaxActiveLights", 8)
    );
    pub const USER_CLIP_PLANES: Spec = spec!(
        FixedFunction,
        "user_clip_planes",
        "six user clip planes",
        D8,
        at_least("MaxUserClipPlanes", 6)
    );
    pub const MULTITEXTURE: Spec = spec!(
        FixedFunction,
        "multitexture",
        "two textures in one pass",
        D8,
        at_least("MaxSimultaneousTextures", 2)
    );
    pub const EIGHT_TEXTURE_STAGES: Spec = spec!(
        FixedFunction,
        "eight_texture_stages",
        "eight texture blend stages",
        D8,
        at_least("MaxTextureBlendStages", 8)
    );
    pub const TABLE_FOG: Spec = spec!(
        FixedFunction,
        "table_fog",
        "per-pixel (table) fog",
        D8,
        bits("RasterCaps", b::D3DPRASTERCAPS_FOGTABLE)
    );
    pub const POINT_SPRITES: Spec = spec!(
        FixedFunction,
        "point_sprites",
        "point sprites larger than one pixel",
        D8,
        Check::FloatAtLeast {
            word: word("MaxPointSize"),
            min: 2.0
        }
    );
    pub const N_PATCHES: Spec = Spec {
        support: U,
        ..spec!(
            FixedFunction,
            "n_patches",
            "N-patch tessellation (TruForm)",
            D8,
            bits("DevCaps", b::D3DDEVCAPS_NPATCHES)
        )
    };

    // Shader.
    pub const VERTEX_SHADER_1_1: Spec = spec!(
        Shader,
        "vertex_shader_1_1",
        "vs_1_1 in hardware",
        D8,
        Check::Vs(ShaderModel::new(1, 1))
    );
    pub const PIXEL_SHADER_1_1: Spec = spec!(
        Shader,
        "pixel_shader_1_1",
        "ps_1_1",
        D8,
        Check::Ps(ShaderModel::new(1, 1))
    );
    pub const PIXEL_SHADER_1_4: Spec = spec!(
        Shader,
        "pixel_shader_1_4",
        "ps_1_4 (Direct3D 8.1)",
        D8,
        Check::Ps(ShaderModel::new(1, 4))
    );
    pub const VERTEX_SHADER_2_0: Spec = spec!(
        Shader,
        "vertex_shader_2_0",
        "vs_2_0",
        D9,
        Check::Vs(ShaderModel::new(2, 0))
    );
    pub const PIXEL_SHADER_2_0: Spec = spec!(
        Shader,
        "pixel_shader_2_0",
        "ps_2_0",
        D9,
        Check::Ps(ShaderModel::new(2, 0))
    );
    pub const PIXEL_SHADER_2_X: Spec = spec!(
        Shader,
        "pixel_shader_2_x",
        "ps_2_x with 512 instruction slots (ps_2_a / ps_2_b)",
        D9,
        at_least("PS20Caps.NumInstructionSlots", 512),
        &[Capability::PixelShader20]
    );
    pub const VERTEX_SHADER_3_0: Spec = spec!(
        Shader,
        "vertex_shader_3_0",
        "vs_3_0",
        D9,
        Check::Vs(ShaderModel::new(3, 0))
    );
    pub const PIXEL_SHADER_3_0: Spec = spec!(
        Shader,
        "pixel_shader_3_0",
        "ps_3_0",
        D9,
        Check::Ps(ShaderModel::new(3, 0))
    );
    pub const VERTEX_TEXTURE_FETCH: Spec = spec!(
        Shader,
        "vertex_texture_fetch",
        "texture reads in vs_3_0",
        D9,
        bits("VertexTextureFilterCaps", b::D3DPTFILTERCAPS_MINFPOINT),
        &[Capability::VertexShader30]
    );

    // Texture.
    pub const CUBE_MAPS: Spec = spec!(
        Texture,
        "cube_maps",
        "cube textures",
        D8,
        bits("TextureCaps", b::D3DPTEXTURECAPS_CUBEMAP)
    );
    pub const VOLUME_TEXTURES: Spec = spec!(
        Texture,
        "volume_textures",
        "3D textures",
        D8,
        bits("TextureCaps", b::D3DPTEXTURECAPS_VOLUMEMAP)
    );
    pub const MIPMAPS: Spec = spec!(
        Texture,
        "mipmaps",
        "mipmapped textures",
        D8,
        bits("TextureCaps", b::D3DPTEXTURECAPS_MIPMAP)
    );
    pub const NON_POW2_CONDITIONAL: Spec = spec!(
        Texture,
        "non_pow2_conditional",
        "non-power-of-two 2D textures, at least without mips or wrapping",
        D8,
        Check::SetOrClear {
            word: word("TextureCaps"),
            set: b::D3DPTEXTURECAPS_NONPOW2CONDITIONAL,
            clear: b::D3DPTEXTURECAPS_POW2,
        }
    );
    pub const NON_POW2: Spec = spec!(
        Texture,
        "non_pow2",
        "non-power-of-two textures without limits (POW2 clear)",
        D8,
        clear("TextureCaps", b::D3DPTEXTURECAPS_POW2)
    );
    pub const ANISOTROPIC_FILTER: Spec = spec!(
        Texture,
        "anisotropic_filter",
        "anisotropic minification",
        D8,
        bits("TextureFilterCaps", b::D3DPTFILTERCAPS_MINFANISOTROPIC)
    );
    pub const ANISOTROPY_16: Spec = spec!(
        Texture,
        "anisotropy_16",
        "16x anisotropy",
        D8,
        at_least("MaxAnisotropy", 16),
        &[Capability::AnisotropicFilter]
    );
    pub const LARGE_TEXTURES: Spec = spec!(
        Texture,
        "large_textures",
        "4096x4096 textures",
        D8,
        at_least("MaxTextureWidth", 4096)
    );
    pub const BORDER_ADDRESS: Spec = spec!(
        Texture,
        "border_address",
        "border colour addressing",
        D8,
        bits("TextureAddressCaps", b::D3DPTADDRESSCAPS_BORDER)
    );
    pub const MIRROR_ONCE: Spec = spec!(
        Texture,
        "mirror_once",
        "mirror-once addressing",
        D8,
        bits("TextureAddressCaps", b::D3DPTADDRESSCAPS_MIRRORONCE)
    );
    pub const AUTO_MIPMAPS: Spec = spec!(
        Texture,
        "auto_mipmaps",
        "mipmaps generated by the driver (D3DUSAGE_AUTOGENMIPMAP)",
        D9,
        bits("Caps2", b::D3DCAPS2_CANAUTOGENMIPMAP)
    );
    pub const DYNAMIC_TEXTURES: Spec = spec!(
        Texture,
        "dynamic_textures",
        "D3DUSAGE_DYNAMIC textures",
        D9,
        bits("Caps2", b::D3DCAPS2_DYNAMICTEXTURES)
    );

    // Output.
    pub const BLEND_OP: Spec = spec!(
        Output,
        "blend_op",
        "blend operations other than add",
        D8,
        bits("PrimitiveMiscCaps", b::D3DPMISCCAPS_BLENDOP)
    );
    pub const COLOR_WRITE_ENABLE: Spec = spec!(
        Output,
        "color_write_enable",
        "per-channel colour write masks",
        D8,
        bits("PrimitiveMiscCaps", b::D3DPMISCCAPS_COLORWRITEENABLE)
    );
    pub const SEPARATE_ALPHA_BLEND: Spec = spec!(
        Output,
        "separate_alpha_blend",
        "a blend for alpha apart from colour",
        D9,
        bits("PrimitiveMiscCaps", b::D3DPMISCCAPS_SEPARATEALPHABLEND)
    );
    pub const TWO_SIDED_STENCIL: Spec = spec!(
        Output,
        "two_sided_stencil",
        "front and back stencil in one pass",
        D9,
        bits("StencilCaps", b::D3DSTENCILCAPS_TWOSIDED)
    );
    pub const MULTIPLE_RENDER_TARGETS: Spec = spec!(
        Output,
        "multiple_render_targets",
        "four simultaneous render targets",
        D9,
        at_least("NumSimultaneousRTs", 4)
    );
    pub const MRT_INDEPENDENT_BIT_DEPTHS: Spec = spec!(
        Output,
        "mrt_independent_bit_depths",
        "render targets of different bit depths together",
        D9,
        bits("PrimitiveMiscCaps", b::D3DPMISCCAPS_MRTINDEPENDENTBITDEPTHS),
        &[Capability::MultipleRenderTargets]
    );
    pub const SCISSOR_TEST: Spec = spec!(
        Output,
        "scissor_test",
        "scissor rectangles",
        D9,
        bits("RasterCaps", b::D3DPRASTERCAPS_SCISSORTEST)
    );
    pub const DEPTH_BIAS: Spec = spec!(
        Output,
        "depth_bias",
        "D3DRS_DEPTHBIAS (Direct3D 8 had ZBIAS)",
        D9,
        bits("RasterCaps", b::D3DPRASTERCAPS_DEPTHBIAS)
    );
    pub const SLOPE_SCALE_DEPTH_BIAS: Spec = spec!(
        Output,
        "slope_scale_depth_bias",
        "slope-scaled depth bias",
        D9,
        bits("RasterCaps", b::D3DPRASTERCAPS_SLOPESCALEDEPTHBIAS)
    );

    // Vertex.
    pub const INDEX_32: Spec = spec!(
        Vertex,
        "index_32",
        "32-bit indices",
        D8,
        at_least("MaxVertexIndex", 0x1_0000)
    );
    pub const EIGHT_STREAMS: Spec = spec!(
        Vertex,
        "eight_streams",
        "eight vertex streams",
        D8,
        at_least("MaxStreams", 8)
    );
    pub const STREAM_OFFSET: Spec = spec!(
        Vertex,
        "stream_offset",
        "an offset in SetStreamSource",
        D9,
        bits("DevCaps2", b::D3DDEVCAPS2_STREAMOFFSET)
    );
    pub const FLOAT16_VERTICES: Spec = spec!(
        Vertex,
        "float16_vertices",
        "half-float vertex elements",
        D9,
        bits("DeclTypes", b::D3DDTCAPS_FLOAT16_2 | b::D3DDTCAPS_FLOAT16_4)
    );
    pub const UBYTE4_VERTICES: Spec = spec!(
        Vertex,
        "ubyte4_vertices",
        "UBYTE4 vertex elements",
        D9,
        bits("DeclTypes", b::D3DDTCAPS_UBYTE4)
    );
    pub const INSTANCING: Spec = spec!(
        Vertex,
        "instancing",
        "geometry instancing (SetStreamSourceFreq), which needs vs_3_0 hardware",
        D9,
        Check::Vs(ShaderModel::new(3, 0))
    );
}

// What Direct3D 8 can't read is Direct3D 9's.
const _: () = {
    let mut i = 0;
    while i < Capability::ALL.len() {
        let spec = Capability::ALL[i].spec();
        if let Some(w) = spec.check.word() {
            assert!(
                w < CAPS8_WORDS || matches!(spec.since, Api::D3D9),
                "a capability past D3DCAPS8 is Direct3D 9's"
            );
        }
        i += 1;
    }
};

/// Why a legacy capability was not granted.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Why {
    /// It came with a later version.
    Since(Api),
    /// The caps don't report it: this member fails the check.
    Lacks(&'static str),
    /// A capability it needs is missing.
    Needs(Capability),
    /// It doesn't exist on this OS.
    Os(Os),
}

/// One capability's verdict.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Outcome {
    pub cap: Capability,
    /// `None` when granted.
    pub why: Option<Why>,
}

impl Outcome {
    #[must_use]
    pub const fn granted(&self) -> bool {
        self.why.is_none()
    }
}

/// What a Direct3D 8 or 9 device can do.
#[derive(Debug, Clone, PartialEq)]
pub struct Plan {
    pub api: Api,
    pub caps: Caps9,
    pub level: LegacyLevel,
    /// The nearest `9_x` feature level ([`LegacyLevel::level9`]).
    pub level9: Option<FeatureLevel>,
    outcomes: Vec<Outcome>,
}

/// Reads `caps` (from `GetDeviceCaps` of `api`) into a plan for `wanted`
/// and what they need, on the OS this was built for.
///
/// # Panics
///
/// When `api` is not Direct3D 8 or 9.
#[must_use]
pub fn plan(api: Api, caps: &Caps9, wanted: &[Capability]) -> Plan {
    plan_on(api, caps, wanted, Os::HOST)
}

/// [`plan`] as on `os`.
///
/// # Panics
///
/// When `api` is not Direct3D 8 or 9.
#[must_use]
pub fn plan_on(api: Api, caps: &Caps9, wanted: &[Capability], os: Os) -> Plan {
    assert!(api.era() == Era::Legacy, "{api} is not Direct3D 8 or 9");
    let mut caps = *caps;
    if api == Api::D3D8 {
        // A real D3D8 runtime writes only its prefix; make sure of it.
        let mut words = *caps.words();
        words[CAPS8_WORDS..].fill(0);
        caps = Caps9::from_words(words);
    }
    let mut order: Vec<Capability> = Vec::new();
    for &cap in wanted {
        push_with_needs(cap, &mut order);
    }
    let mut outcomes: Vec<Outcome> = Vec::with_capacity(order.len());
    for &cap in &order {
        let spec = cap.spec();
        let why = if !spec.support.available_on(os) {
            Some(Why::Os(os))
        } else if spec.since > api {
            Some(Why::Since(spec.since))
        } else if !spec.check.passes(&caps) {
            Some(Why::Lacks(
                spec.check.word().map_or("?", |w| CAPS9_MEMBERS[w]),
            ))
        } else {
            spec.needs
                .iter()
                .find(|n| !outcomes.iter().any(|o| o.cap == **n && o.granted()))
                .map(|&n| Why::Needs(n))
        };
        outcomes.push(Outcome { cap, why });
    }
    // Present the wanted ones in the order asked, needs after.
    Plan {
        api,
        level: LegacyLevel::of(&caps),
        level9: if api == Api::D3D9 {
            LegacyLevel::level9(&caps)
        } else {
            None
        },
        caps,
        outcomes,
    }
}

/// Needs first, so a capability's needs have their verdict before it.
fn push_with_needs(cap: Capability, order: &mut Vec<Capability>) {
    if order.contains(&cap) {
        return;
    }
    for &n in cap.spec().needs {
        push_with_needs(n, order);
    }
    order.push(cap);
}

impl Plan {
    #[must_use]
    pub fn has(&self, cap: Capability) -> bool {
        self.outcome(cap).is_some_and(Outcome::granted)
    }

    #[must_use]
    pub fn lacking(&self, required: &[Capability]) -> Vec<Capability> {
        required.iter().copied().filter(|&c| !self.has(c)).collect()
    }

    #[must_use]
    pub fn outcome(&self, cap: Capability) -> Option<&Outcome> {
        self.outcomes.iter().find(|o| o.cap == cap)
    }

    /// Every verdict, needs before what needs them.
    #[must_use]
    pub fn outcomes(&self) -> &[Outcome] {
        &self.outcomes
    }
}

impl fmt::Display for Why {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Since(api) => write!(f, "since {api}"),
            Self::Lacks(member) => write!(f, "not in {member}"),
            Self::Needs(cap) => write!(f, "needs {}", cap.name()),
            Self::Os(os) => write!(f, "not on {os}"),
        }
    }
}

impl fmt::Display for Plan {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let shader =
            |v: Option<ShaderModel>| v.map_or_else(|| "none".to_string(), |v| v.to_string());
        write!(
            f,
            "{}: {}, vs {}, ps {}",
            self.api,
            self.level,
            shader(self.caps.vertex_shader()),
            shader(self.caps.pixel_shader())
        )?;
        if let Some(l) = self.level9 {
            write!(f, ", like feature level {l}")?;
        }
        writeln!(f)?;
        let granted = self.outcomes.iter().filter(|o| o.granted()).count();
        writeln!(f, "  {granted} of {} granted", self.outcomes.len())?;
        for o in &self.outcomes {
            let spec = o.cap.spec();
            match o.why {
                None => writeln!(f, "  + {:<12} {}", spec.area.name(), spec.name)?,
                Some(why) => writeln!(f, "  - {:<12} {} ({why})", spec.area.name(), spec.name)?,
            }
        }
        Ok(())
    }
}
