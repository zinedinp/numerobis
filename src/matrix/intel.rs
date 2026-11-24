//! Intel XMX (Xe Matrix eXtensions): the systolic arrays behind `dpas`.
//!
//! `dpas` works per subgroup: systolic depth 8 (8 dwords of A per lane
//! along K), a repeat count of up to 8 rows (M) and the execution size as N:
//! 8 lanes on Xe-HPG, 16 on Xe-HPC and Xe2. K in elements is 8 dwords of
//! the input: 16 for f16/bf16, 32 for 8-bit, 64 for 4-bit, 8 for tf32. The
//! signedness of each 8- and 4-bit source is its own (`s8 x u8` is one
//! instruction).
//!
//! Xe-LP (Tiger Lake to Raptor Lake, DG1) and Xe-LPG (Meteor Lake) have no
//! XMX, only `dp4a` on the vector units. Arrow Lake-H's Xe-LPG+ adds `dpas`
//! but shares its PCI range with Xe-LPG in [`crate::pci`], so it is not told
//! apart (never a guess).
//!
//! Xe2 checked on the Arc B580 through Vulkan: the driver's cooperative
//! matrices are `f16 x f16 -> f32 | f16` (8x16x16) and `s8 x s8`, `u8 x u8
//! -> 32-bit` (8x16x32), every one an instruction of this table.

use super::{Mma, Scope, mmas};
use crate::intel::Tier;
use crate::pci::IntelArch;

const SIMD8: Scope = Scope::Subgroup { lanes: 8 };
const SIMD16: Scope = Scope::Subgroup { lanes: 16 };

/// An XMX generation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Xmx {
    /// Arc A-series (Alchemist, DG2): SIMD8.
    XeHpg,
    /// Data Center GPU Max (Ponte Vecchio): SIMD16, tf32.
    XeHpc,
    /// Arc B-series (Battlemage) and Lunar Lake: SIMD16.
    Xe2,
}

const XE_HPG: &[Mma] = mmas![
    F16, F16 -> F32, 8 8 16, SIMD8, "dpas.8x8 hf";
    F16, F16 -> F16, 8 8 16, SIMD8, "dpas.8x8 hf";
    BF16, BF16 -> F32, 8 8 16, SIMD8, "dpas.8x8 bf";
    I8, I8 -> I32, 8 8 32, SIMD8, "dpas.8x8 s8";
    U8, U8 -> I32, 8 8 32, SIMD8, "dpas.8x8 u8";
    I8, U8 -> I32, 8 8 32, SIMD8, "dpas.8x8 s8 u8";
    U8, I8 -> I32, 8 8 32, SIMD8, "dpas.8x8 u8 s8";
    I4, I4 -> I32, 8 8 64, SIMD8, "dpas.8x8 s4";
    U4, U4 -> I32, 8 8 64, SIMD8, "dpas.8x8 u4";
];

/// Xe-HPC and Xe2: the same instructions at execution size 16.
const SIMD16_COMMON: &[Mma] = mmas![
    F16, F16 -> F32, 8 16 16, SIMD16, "dpas.8x8 hf";
    F16, F16 -> F16, 8 16 16, SIMD16, "dpas.8x8 hf";
    BF16, BF16 -> F32, 8 16 16, SIMD16, "dpas.8x8 bf";
    I8, I8 -> I32, 8 16 32, SIMD16, "dpas.8x8 s8";
    U8, U8 -> I32, 8 16 32, SIMD16, "dpas.8x8 u8";
    I8, U8 -> I32, 8 16 32, SIMD16, "dpas.8x8 s8 u8";
    U8, I8 -> I32, 8 16 32, SIMD16, "dpas.8x8 u8 s8";
    I4, I4 -> I32, 8 16 64, SIMD16, "dpas.8x8 s4";
    U4, U4 -> I32, 8 16 64, SIMD16, "dpas.8x8 u4";
];

/// tf32 is documented for Xe-HPC; for Xe2 it is not confirmed (the B580's
/// driver does not report it), so it stays out of that table.
const XE_HPC_TF32: &[Mma] = mmas![
    TF32, TF32 -> F32, 8 16 8, SIMD16, "dpas.8x8 tf32";
];

impl Xmx {
    #[must_use]
    pub const fn name(self) -> &'static str {
        match self {
            Self::XeHpg => "Xe-HPG",
            Self::XeHpc => "Xe-HPC",
            Self::Xe2 => "Xe2",
        }
    }

    #[must_use]
    pub const fn families(self) -> &'static [&'static [Mma]] {
        match self {
            Self::XeHpg => &[XE_HPG],
            Self::XeHpc => &[SIMD16_COMMON, XE_HPC_TF32],
            Self::Xe2 => &[SIMD16_COMMON],
        }
    }

    #[must_use]
    pub const fn tier(self) -> Option<crate::Tier> {
        Some(crate::Tier::Intel(match self {
            Self::XeHpg => Tier::XeHpg,
            Self::XeHpc => Tier::XeHpc,
            Self::Xe2 => Tier::Xe2,
        }))
    }

    /// `None` for the DP4A-only [`Tier::XeLp`].
    #[must_use]
    pub const fn of(tier: Tier) -> Option<Self> {
        match tier {
            Tier::XeLp => None,
            Tier::XeHpg => Some(Self::XeHpg),
            Tier::XeHpc => Some(Self::XeHpc),
            Tier::Xe2 => Some(Self::Xe2),
        }
    }

    #[must_use]
    pub const fn from_arch(arch: IntelArch) -> Option<Self> {
        match arch {
            IntelArch::XeHpg => Some(Self::XeHpg),
            IntelArch::XeHpc => Some(Self::XeHpc),
            IntelArch::Xe2Lpg | IntelArch::Xe2Hpg => Some(Self::Xe2),
            IntelArch::Gen9 | IntelArch::Gen11 | IntelArch::XeLp | IntelArch::XeLpg => None,
        }
    }
}
