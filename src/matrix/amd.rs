//! AMD: Matrix Cores on CDNA (`v_mfma_*`, per wave of 64) and AI
//! Accelerators on RDNA 3 and 4 (`v_wmma_*`, per wave of 32), from the
//! CDNA and RDNA ISA guides.
//!
//! `MFMA` comes in two tiles per type (32x32 and 16x16); the larger is
//! listed first. CDNA 3 reads FP8 as FNUZ (`E4M3Fnuz`, `E5M2Fnuz`) and
//! tf32 as `xf32`; CDNA 4 switches to OCP FP8, adds FP6/FP4 and MX block
//! scales (`v_mfma_scale_*`) and drops `xf32`. RDNA 2 and older (and GCN
//! before CDNA) have only dot-product instructions on the vector units.
//! Sparse forms (`SMFMAC`, `SWMMAC`) and `f64` are left out.

use super::{Mma, Scope, mmas};
use crate::amd::Tier;
use crate::pci::AmdArch;

const WAVE64: Scope = Scope::Subgroup { lanes: 64 };
const WAVE32: Scope = Scope::Subgroup { lanes: 32 };

/// A CDNA Matrix Core or RDNA AI Accelerator generation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum MatrixCore {
    /// MI100, `gfx908`.
    Cdna1,
    /// MI200, `gfx90a`.
    Cdna2,
    /// MI300, `gfx942`: FP8 (FNUZ), `xf32`.
    Cdna3,
    /// MI350, `gfx950`: OCP FP8, FP6, FP4, MX.
    Cdna4,
    /// RX 7000 and the RDNA 3.5 APUs, `gfx11`.
    Rdna3,
    /// RX 9000, `gfx12`: FP8.
    Rdna4,
}

const CDNA1: &[Mma] = mmas![
    F16, F16 -> F32, 32 32 8, WAVE64, "v_mfma_f32_32x32x8f16";
    F16, F16 -> F32, 16 16 16, WAVE64, "v_mfma_f32_16x16x16f16";
    BF16, BF16 -> F32, 32 32 4, WAVE64, "v_mfma_f32_32x32x4bf16";
    BF16, BF16 -> F32, 16 16 8, WAVE64, "v_mfma_f32_16x16x8bf16";
    I8, I8 -> I32, 32 32 8, WAVE64, "v_mfma_i32_32x32x8i8";
    I8, I8 -> I32, 16 16 16, WAVE64, "v_mfma_i32_16x16x16i8";
    F32, F32 -> F32, 32 32 2, WAVE64, "v_mfma_f32_32x32x2f32";
    F32, F32 -> F32, 16 16 4, WAVE64, "v_mfma_f32_16x16x4f32";
];

const CDNA2: &[Mma] = mmas![
    F16, F16 -> F32, 32 32 8, WAVE64, "v_mfma_f32_32x32x8f16";
    F16, F16 -> F32, 16 16 16, WAVE64, "v_mfma_f32_16x16x16f16";
    BF16, BF16 -> F32, 32 32 8, WAVE64, "v_mfma_f32_32x32x8bf16_1k";
    BF16, BF16 -> F32, 16 16 16, WAVE64, "v_mfma_f32_16x16x16bf16_1k";
    I8, I8 -> I32, 32 32 8, WAVE64, "v_mfma_i32_32x32x8i8";
    I8, I8 -> I32, 16 16 16, WAVE64, "v_mfma_i32_16x16x16i8";
    F32, F32 -> F32, 32 32 2, WAVE64, "v_mfma_f32_32x32x2f32";
    F32, F32 -> F32, 16 16 4, WAVE64, "v_mfma_f32_16x16x4f32";
];

const CDNA3: &[Mma] = mmas![
    F16, F16 -> F32, 32 32 8, WAVE64, "v_mfma_f32_32x32x8_f16";
    F16, F16 -> F32, 16 16 16, WAVE64, "v_mfma_f32_16x16x16_f16";
    BF16, BF16 -> F32, 32 32 8, WAVE64, "v_mfma_f32_32x32x8_bf16";
    BF16, BF16 -> F32, 16 16 16, WAVE64, "v_mfma_f32_16x16x16_bf16";
    I8, I8 -> I32, 32 32 16, WAVE64, "v_mfma_i32_32x32x16_i8";
    I8, I8 -> I32, 16 16 32, WAVE64, "v_mfma_i32_16x16x32_i8";
    TF32, TF32 -> F32, 32 32 4, WAVE64, "v_mfma_f32_32x32x4_xf32";
    TF32, TF32 -> F32, 16 16 8, WAVE64, "v_mfma_f32_16x16x8_xf32";
    E4M3FNUZ, E4M3FNUZ -> F32, 32 32 16, WAVE64, "v_mfma_f32_32x32x16_fp8_fp8";
    E4M3FNUZ, E4M3FNUZ -> F32, 16 16 32, WAVE64, "v_mfma_f32_16x16x32_fp8_fp8";
    E5M2FNUZ, E5M2FNUZ -> F32, 32 32 16, WAVE64, "v_mfma_f32_32x32x16_bf8_bf8";
    E4M3FNUZ, E5M2FNUZ -> F32, 32 32 16, WAVE64, "v_mfma_f32_32x32x16_fp8_bf8";
    E5M2FNUZ, E4M3FNUZ -> F32, 32 32 16, WAVE64, "v_mfma_f32_32x32x16_bf8_fp8";
    F32, F32 -> F32, 32 32 2, WAVE64, "v_mfma_f32_32x32x2_f32";
];

const CDNA4: &[Mma] = mmas![
    F16, F16 -> F32, 32 32 16, WAVE64, "v_mfma_f32_32x32x16_f16";
    F16, F16 -> F32, 16 16 32, WAVE64, "v_mfma_f32_16x16x32_f16";
    BF16, BF16 -> F32, 32 32 16, WAVE64, "v_mfma_f32_32x32x16_bf16";
    BF16, BF16 -> F32, 16 16 32, WAVE64, "v_mfma_f32_16x16x32_bf16";
    I8, I8 -> I32, 32 32 32, WAVE64, "v_mfma_i32_32x32x32_i8";
    I8, I8 -> I32, 16 16 64, WAVE64, "v_mfma_i32_16x16x64_i8";
    E4M3, E4M3 -> F32, 32 32 64, WAVE64, "v_mfma_f32_32x32x64_f8f6f4";
    E5M2, E5M2 -> F32, 32 32 64, WAVE64, "v_mfma_f32_32x32x64_f8f6f4";
    E2M3, E2M3 -> F32, 32 32 64, WAVE64, "v_mfma_f32_32x32x64_f8f6f4";
    E3M2, E3M2 -> F32, 32 32 64, WAVE64, "v_mfma_f32_32x32x64_f8f6f4";
    E2M1, E2M1 -> F32, 32 32 64, WAVE64, "v_mfma_f32_32x32x64_f8f6f4";
    MXFP8E4M3, MXFP8E4M3 -> F32, 32 32 64, WAVE64, "v_mfma_scale_f32_32x32x64_f8f6f4";
    MXFP8E5M2, MXFP8E5M2 -> F32, 32 32 64, WAVE64, "v_mfma_scale_f32_32x32x64_f8f6f4";
    MXFP6E2M3, MXFP6E2M3 -> F32, 32 32 64, WAVE64, "v_mfma_scale_f32_32x32x64_f8f6f4";
    MXFP6E3M2, MXFP6E3M2 -> F32, 32 32 64, WAVE64, "v_mfma_scale_f32_32x32x64_f8f6f4";
    MXFP4, MXFP4 -> F32, 32 32 64, WAVE64, "v_mfma_scale_f32_32x32x64_f8f6f4";
    F32, F32 -> F32, 32 32 2, WAVE64, "v_mfma_f32_32x32x2_f32";
];

const RDNA3: &[Mma] = mmas![
    F16, F16 -> F32, 16 16 16, WAVE32, "v_wmma_f32_16x16x16_f16";
    F16, F16 -> F16, 16 16 16, WAVE32, "v_wmma_f16_16x16x16_f16";
    BF16, BF16 -> F32, 16 16 16, WAVE32, "v_wmma_f32_16x16x16_bf16";
    BF16, BF16 -> BF16, 16 16 16, WAVE32, "v_wmma_bf16_16x16x16_bf16";
    I8, I8 -> I32, 16 16 16, WAVE32, "v_wmma_i32_16x16x16_iu8";
    U8, U8 -> I32, 16 16 16, WAVE32, "v_wmma_i32_16x16x16_iu8";
    I8, U8 -> I32, 16 16 16, WAVE32, "v_wmma_i32_16x16x16_iu8";
    U8, I8 -> I32, 16 16 16, WAVE32, "v_wmma_i32_16x16x16_iu8";
    I4, I4 -> I32, 16 16 16, WAVE32, "v_wmma_i32_16x16x16_iu4";
    U4, U4 -> I32, 16 16 16, WAVE32, "v_wmma_i32_16x16x16_iu4";
];

const RDNA4_EXTRA: &[Mma] = mmas![
    E4M3, E4M3 -> F32, 16 16 16, WAVE32, "v_wmma_f32_16x16x16_fp8_fp8";
    E5M2, E5M2 -> F32, 16 16 16, WAVE32, "v_wmma_f32_16x16x16_bf8_bf8";
    E4M3, E5M2 -> F32, 16 16 16, WAVE32, "v_wmma_f32_16x16x16_fp8_bf8";
    E5M2, E4M3 -> F32, 16 16 16, WAVE32, "v_wmma_f32_16x16x16_bf8_fp8";
    I4, I4 -> I32, 16 16 32, WAVE32, "v_wmma_i32_16x16x32_iu4";
    U4, U4 -> I32, 16 16 32, WAVE32, "v_wmma_i32_16x16x32_iu4";
];

impl MatrixCore {
    #[must_use]
    pub const fn name(self) -> &'static str {
        match self {
            Self::Cdna1 => "CDNA 1",
            Self::Cdna2 => "CDNA 2",
            Self::Cdna3 => "CDNA 3",
            Self::Cdna4 => "CDNA 4",
            Self::Rdna3 => "RDNA 3",
            Self::Rdna4 => "RDNA 4",
        }
    }

    /// AMD's name for the unit.
    #[must_use]
    pub const fn unit(self) -> &'static str {
        match self {
            Self::Cdna1 | Self::Cdna2 | Self::Cdna3 | Self::Cdna4 => "Matrix Core",
            Self::Rdna3 | Self::Rdna4 => "AI Accelerator",
        }
    }

    #[must_use]
    pub const fn families(self) -> &'static [&'static [Mma]] {
        match self {
            Self::Cdna1 => &[CDNA1],
            Self::Cdna2 => &[CDNA2],
            Self::Cdna3 => &[CDNA3],
            Self::Cdna4 => &[CDNA4],
            Self::Rdna3 => &[RDNA3],
            // The 16x16x32 int4 form first: twice the K of RDNA 3's.
            Self::Rdna4 => &[RDNA4_EXTRA, RDNA3],
        }
    }

    #[must_use]
    pub const fn tier(self) -> crate::Tier {
        crate::Tier::Amd(match self {
            Self::Cdna1 => Tier::Gfx908,
            Self::Cdna2 => Tier::Gfx90a,
            Self::Cdna3 => Tier::Gfx942,
            Self::Cdna4 => Tier::Gfx950,
            Self::Rdna3 => Tier::Gfx11,
            Self::Rdna4 => Tier::Gfx12,
        })
    }

    #[must_use]
    pub const fn of(tier: Tier) -> Self {
        match tier {
            Tier::Gfx908 => Self::Cdna1,
            Tier::Gfx90a => Self::Cdna2,
            Tier::Gfx942 => Self::Cdna3,
            Tier::Gfx950 => Self::Cdna4,
            Tier::Gfx11 => Self::Rdna3,
            Tier::Gfx12 => Self::Rdna4,
        }
    }

    /// CDNA is one architecture in [`crate::pci`]; the device id tells the
    /// generation (MI350 is not in the id tables yet).
    #[must_use]
    pub const fn from_pci(arch: AmdArch, device: u32) -> Option<Self> {
        Some(match arch {
            AmdArch::Gcn | AmdArch::Rdna1 | AmdArch::Rdna2 => return None,
            AmdArch::Rdna3 | AmdArch::Rdna3_5 => Self::Rdna3,
            AmdArch::Rdna4 => Self::Rdna4,
            AmdArch::Cdna => match device {
                0x7388..=0x738F => Self::Cdna1,
                0x7408..=0x740F => Self::Cdna2,
                0x74A0..=0x74BF => Self::Cdna3,
                _ => return None,
            },
        })
    }
}
