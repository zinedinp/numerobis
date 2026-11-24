//! NVIDIA Tensor Cores, by PTX instruction family.
//!
//! - `mma.sync` (Volta on): per warp of 32, the tiles in registers. The
//!   raw backend's next tier (`nb_nv.h`) and still the only one on GeForce
//!   Blackwell (`sm_120`), which extends it with FP6/FP4 and block scales.
//! - `wgmma.mma_async` (Hopper, `sm_90a`): per warpgroup of 128, B from
//!   shared memory; N from 8 to 256 in steps of 8 (the table lists 256).
//! - `tcgen05.mma` (data-center Blackwell, `sm_100a`): one thread issues it
//!   for its thread block, the accumulator lives in tensor memory; M 64 or
//!   128 (256 over a thread block pair, `cta_group::2`), N up to 256.
//!
//! Turing's GTX 16 chips (TU116, TU117) have no Tensor Cores; Pascal and
//! older none at all. `f64` (`DMMA`) and the 1-bit `b1` forms are left out.

use super::{Mma, Scope, mmas};
use crate::nvidia::Tier;
use crate::pci::NvidiaArch;

const WARP: Scope = Scope::Subgroup { lanes: 32 };
const WG: Scope = Scope::Warpgroup;
const CTA: Scope = Scope::Cta { pair: false };

/// A Tensor Core generation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum TensorCore {
    /// `sm_70`: first generation.
    Volta,
    /// `sm_75`: integer, `m16n8k8`.
    Turing,
    /// `sm_80`, `sm_86`, `sm_87`: bf16, tf32, `m16n8k16`.
    Ampere,
    /// `sm_89`: FP8.
    Ada,
    /// `sm_90`: `wgmma`.
    Hopper,
    /// `sm_100`, `sm_103`: `tcgen05`, FP6/FP4, MX and NVFP4.
    Blackwell,
    /// `sm_120`: GeForce RTX 50, `mma.sync` with FP6/FP4 and block scales.
    BlackwellGeForce,
}

const VOLTA: &[Mma] = mmas![
    F16, F16 -> F32, 8 8 4, WARP, "mma.sync.aligned.m8n8k4";
    F16, F16 -> F16, 8 8 4, WARP, "mma.sync.aligned.m8n8k4";
];

const TURING: &[Mma] = mmas![
    F16, F16 -> F32, 16 8 8, WARP, "mma.sync.aligned.m16n8k8";
    F16, F16 -> F16, 16 8 8, WARP, "mma.sync.aligned.m16n8k8";
    I8, I8 -> I32, 8 8 16, WARP, "mma.sync.aligned.m8n8k16.s8.s8";
    U8, U8 -> I32, 8 8 16, WARP, "mma.sync.aligned.m8n8k16.u8.u8";
    I8, U8 -> I32, 8 8 16, WARP, "mma.sync.aligned.m8n8k16.s8.u8";
    U8, I8 -> I32, 8 8 16, WARP, "mma.sync.aligned.m8n8k16.u8.s8";
    I4, I4 -> I32, 8 8 32, WARP, "mma.sync.aligned.m8n8k32.s4.s4";
    U4, U4 -> I32, 8 8 32, WARP, "mma.sync.aligned.m8n8k32.u4.u4";
];

const AMPERE: &[Mma] = mmas![
    F16, F16 -> F32, 16 8 16, WARP, "mma.sync.aligned.m16n8k16.f32.f16";
    F16, F16 -> F16, 16 8 16, WARP, "mma.sync.aligned.m16n8k16.f16.f16";
    BF16, BF16 -> F32, 16 8 16, WARP, "mma.sync.aligned.m16n8k16.f32.bf16";
    TF32, TF32 -> F32, 16 8 8, WARP, "mma.sync.aligned.m16n8k8.f32.tf32";
    I8, I8 -> I32, 16 8 32, WARP, "mma.sync.aligned.m16n8k32.s8.s8";
    U8, U8 -> I32, 16 8 32, WARP, "mma.sync.aligned.m16n8k32.u8.u8";
    I8, U8 -> I32, 16 8 32, WARP, "mma.sync.aligned.m16n8k32.s8.u8";
    U8, I8 -> I32, 16 8 32, WARP, "mma.sync.aligned.m16n8k32.u8.s8";
    I4, I4 -> I32, 16 8 64, WARP, "mma.sync.aligned.m16n8k64.s4.s4";
    U4, U4 -> I32, 16 8 64, WARP, "mma.sync.aligned.m16n8k64.u4.u4";
];

const ADA_FP8: &[Mma] = mmas![
    E4M3, E4M3 -> F32, 16 8 32, WARP, "mma.sync.aligned.m16n8k32.e4m3.e4m3";
    E5M2, E5M2 -> F32, 16 8 32, WARP, "mma.sync.aligned.m16n8k32.e5m2.e5m2";
    E4M3, E5M2 -> F32, 16 8 32, WARP, "mma.sync.aligned.m16n8k32.e4m3.e5m2";
    E5M2, E4M3 -> F32, 16 8 32, WARP, "mma.sync.aligned.m16n8k32.e5m2.e4m3";
];

const HOPPER: &[Mma] = mmas![
    F16, F16 -> F32, 64 256 16, WG, "wgmma.mma_async.m64nNk16.f32.f16";
    F16, F16 -> F16, 64 256 16, WG, "wgmma.mma_async.m64nNk16.f16.f16";
    BF16, BF16 -> F32, 64 256 16, WG, "wgmma.mma_async.m64nNk16.f32.bf16";
    TF32, TF32 -> F32, 64 256 8, WG, "wgmma.mma_async.m64nNk8.f32.tf32";
    E4M3, E4M3 -> F32, 64 256 32, WG, "wgmma.mma_async.m64nNk32.f32.e4m3.e4m3";
    E5M2, E5M2 -> F32, 64 256 32, WG, "wgmma.mma_async.m64nNk32.f32.e5m2.e5m2";
    E4M3, E5M2 -> F32, 64 256 32, WG, "wgmma.mma_async.m64nNk32.f32.e4m3.e5m2";
    E5M2, E4M3 -> F32, 64 256 32, WG, "wgmma.mma_async.m64nNk32.f32.e5m2.e4m3";
    I8, I8 -> I32, 64 256 32, WG, "wgmma.mma_async.m64nNk32.s32.s8.s8";
    U8, U8 -> I32, 64 256 32, WG, "wgmma.mma_async.m64nNk32.s32.u8.u8";
    I8, U8 -> I32, 64 256 32, WG, "wgmma.mma_async.m64nNk32.s32.s8.u8";
    U8, I8 -> I32, 64 256 32, WG, "wgmma.mma_async.m64nNk32.s32.u8.s8";
];

const TCGEN05: &[Mma] = mmas![
    F16, F16 -> F32, 128 256 16, CTA, "tcgen05.mma.cta_group::1.kind::f16";
    F16, F16 -> F16, 128 256 16, CTA, "tcgen05.mma.cta_group::1.kind::f16";
    BF16, BF16 -> F32, 128 256 16, CTA, "tcgen05.mma.cta_group::1.kind::f16";
    TF32, TF32 -> F32, 128 256 8, CTA, "tcgen05.mma.cta_group::1.kind::tf32";
    E4M3, E4M3 -> F32, 128 256 32, CTA, "tcgen05.mma.cta_group::1.kind::f8f6f4";
    E5M2, E5M2 -> F32, 128 256 32, CTA, "tcgen05.mma.cta_group::1.kind::f8f6f4";
    E2M3, E2M3 -> F32, 128 256 32, CTA, "tcgen05.mma.cta_group::1.kind::f8f6f4";
    E3M2, E3M2 -> F32, 128 256 32, CTA, "tcgen05.mma.cta_group::1.kind::f8f6f4";
    E2M1, E2M1 -> F32, 128 256 32, CTA, "tcgen05.mma.cta_group::1.kind::f8f6f4";
    I8, I8 -> I32, 128 256 32, CTA, "tcgen05.mma.cta_group::1.kind::i8";
    U8, U8 -> I32, 128 256 32, CTA, "tcgen05.mma.cta_group::1.kind::i8";
    MXFP8E4M3, MXFP8E4M3 -> F32, 128 256 32, CTA, "tcgen05.mma.cta_group::1.kind::mxf8f6f4.block_scale";
    MXFP8E5M2, MXFP8E5M2 -> F32, 128 256 32, CTA, "tcgen05.mma.cta_group::1.kind::mxf8f6f4.block_scale";
    MXFP6E2M3, MXFP6E2M3 -> F32, 128 256 32, CTA, "tcgen05.mma.cta_group::1.kind::mxf8f6f4.block_scale";
    MXFP6E3M2, MXFP6E3M2 -> F32, 128 256 32, CTA, "tcgen05.mma.cta_group::1.kind::mxf8f6f4.block_scale";
    MXFP4, MXFP4 -> F32, 128 256 64, CTA, "tcgen05.mma.cta_group::1.kind::mxf4.block_scale";
    NVFP4, NVFP4 -> F32, 128 256 64, CTA, "tcgen05.mma.cta_group::1.kind::mxf4nvf4.block_scale.scale_vec::4X";
];

/// `sm_120a`: FP6/FP4 and block scales on `mma.sync`.
const SM120: &[Mma] = mmas![
    E2M3, E2M3 -> F32, 16 8 32, WARP, "mma.sync.aligned.m16n8k32.kind::f8f6f4.e2m3.e2m3";
    E3M2, E3M2 -> F32, 16 8 32, WARP, "mma.sync.aligned.m16n8k32.kind::f8f6f4.e3m2.e3m2";
    E2M1, E2M1 -> F32, 16 8 32, WARP, "mma.sync.aligned.m16n8k32.kind::f8f6f4.e2m1.e2m1";
    MXFP8E4M3, MXFP8E4M3 -> F32, 16 8 32, WARP, "mma.sync.aligned.m16n8k32.kind::mxf8f6f4.block_scale";
    MXFP8E5M2, MXFP8E5M2 -> F32, 16 8 32, WARP, "mma.sync.aligned.m16n8k32.kind::mxf8f6f4.block_scale";
    MXFP6E2M3, MXFP6E2M3 -> F32, 16 8 32, WARP, "mma.sync.aligned.m16n8k32.kind::mxf8f6f4.block_scale";
    MXFP6E3M2, MXFP6E3M2 -> F32, 16 8 32, WARP, "mma.sync.aligned.m16n8k32.kind::mxf8f6f4.block_scale";
    MXFP4, MXFP4 -> F32, 16 8 64, WARP, "mma.sync.aligned.m16n8k64.kind::mxf4.block_scale";
    NVFP4, NVFP4 -> F32, 16 8 64, WARP, "mma.sync.aligned.m16n8k64.kind::mxf4nvf4.block_scale.scale_vec::4X";
];

impl TensorCore {
    #[must_use]
    pub const fn name(self) -> &'static str {
        match self {
            Self::Volta => "Volta",
            Self::Turing => "Turing",
            Self::Ampere => "Ampere",
            Self::Ada => "Ada",
            Self::Hopper => "Hopper",
            Self::Blackwell => "Blackwell",
            Self::BlackwellGeForce => "Blackwell (GeForce)",
        }
    }

    /// Newest family first: `wgmma` before `mma.sync` on Hopper.
    #[must_use]
    pub const fn families(self) -> &'static [&'static [Mma]] {
        match self {
            Self::Volta => &[VOLTA],
            Self::Turing => &[TURING],
            Self::Ampere => &[AMPERE],
            Self::Ada => &[AMPERE, ADA_FP8],
            Self::Hopper => &[HOPPER, AMPERE, ADA_FP8],
            Self::Blackwell => &[TCGEN05, AMPERE, ADA_FP8],
            Self::BlackwellGeForce => &[AMPERE, ADA_FP8, SM120],
        }
    }

    /// Volta and Turing have no tier: the raw backend starts at `sm_80`.
    #[must_use]
    pub const fn tier(self) -> Option<crate::Tier> {
        Some(crate::Tier::Nvidia(match self {
            Self::Volta | Self::Turing => return None,
            Self::Ampere => Tier::Sm80,
            Self::Ada => Tier::Sm89,
            Self::Hopper => Tier::Sm90,
            Self::Blackwell => Tier::Sm100,
            Self::BlackwellGeForce => Tier::Sm120,
        }))
    }

    #[must_use]
    pub const fn of(tier: Tier) -> Self {
        match tier {
            Tier::Sm80 => Self::Ampere,
            Tier::Sm89 => Self::Ada,
            Tier::Sm90 => Self::Hopper,
            Tier::Sm100 => Self::Blackwell,
            Tier::Sm120 => Self::BlackwellGeForce,
        }
    }

    /// From the architecture and, where one architecture holds chips with and
    /// without Tensor Cores (Turing) or two Blackwells, the device id.
    #[must_use]
    pub const fn from_pci(arch: NvidiaArch, device: u32) -> Option<Self> {
        Some(match arch {
            NvidiaArch::Maxwell | NvidiaArch::Pascal => return None,
            NvidiaArch::Volta => Self::Volta,
            // TU102/TU104/TU106; TU117 (0x1F8x up) and TU116 (0x21xx) have none.
            NvidiaArch::Turing => match device {
                0x1E00..=0x1F7F => Self::Turing,
                _ => return None,
            },
            NvidiaArch::Ampere => Self::Ampere,
            NvidiaArch::Ada => Self::Ada,
            NvidiaArch::Hopper => Self::Hopper,
            // GB100/GB102 in 0x29xx; GB20x (GeForce, RTX PRO) from 0x2B00.
            NvidiaArch::Blackwell => match device {
                0x2900..=0x29FF => Self::Blackwell,
                _ => Self::BlackwellGeForce,
            },
        })
    }
}
