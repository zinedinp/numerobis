//! Tenstorrent Tensix: the FPU of each Tensix core.
//!
//! Not a subgroup instruction but a core's pipeline: the unpackers move
//! 32x32 tiles (four 16x16 faces) from L1 into the source registers, the
//! FPU's `MVMUL` multiplies 8x16 by 16x16 into the destination register,
//! the packer writes the result back. The unit kernels name is the tile
//! (`matmul_tiles`: 32x32x32), listed here as the shape.
//!
//! Inputs are what the unpackers read: `BF16` (`Float16_b`), `F16`,
//! `Float32` (held as tf32 in the source registers), the shared-exponent
//! `BFP8_b`/`BFP4_b`/`BFP2_b` (`Bfp<M7|M3|M1>`) and `Int8`. Accumulation is
//! `F32` (`fp32_dest_acc_en`) or `BF16`. Math fidelity (`LoFi` to `HiFi4`)
//! chooses how many mantissa passes the FPU makes; `HiFi4` keeps every bit
//! of its inputs.

use super::{Mma, Scope, mmas};
use crate::pci::TtArch;
use crate::tt::Tier;

const CORE: Scope = Scope::Core;

/// A Tensix generation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Tensix {
    Wormhole,
    Blackhole,
}

const TENSIX: &[Mma] = mmas![
    BF16, BF16 -> F32, 32 32 32, CORE, "matmul_tiles (MVMUL)";
    BF16, BF16 -> BF16, 32 32 32, CORE, "matmul_tiles (MVMUL)";
    F16, F16 -> F32, 32 32 32, CORE, "matmul_tiles (MVMUL)";
    TF32, TF32 -> F32, 32 32 32, CORE, "matmul_tiles (MVMUL)";
    BFP8, BFP8 -> F32, 32 32 32, CORE, "matmul_tiles (MVMUL)";
    BFP8, BFP8 -> BF16, 32 32 32, CORE, "matmul_tiles (MVMUL)";
    BFP4, BFP4 -> F32, 32 32 32, CORE, "matmul_tiles (MVMUL)";
    BFP2, BFP2 -> F32, 32 32 32, CORE, "matmul_tiles (MVMUL)";
    I8, I8 -> I32, 32 32 32, CORE, "matmul_tiles (MVMUL, Int8)";
];

impl Tensix {
    #[must_use]
    pub const fn name(self) -> &'static str {
        match self {
            Self::Wormhole => "Wormhole",
            Self::Blackhole => "Blackhole",
        }
    }

    #[must_use]
    pub const fn families(self) -> &'static [&'static [Mma]] {
        match self {
            Self::Wormhole | Self::Blackhole => &[TENSIX],
        }
    }

    #[must_use]
    pub const fn tier(self) -> crate::Tier {
        crate::Tier::Tt(match self {
            Self::Wormhole => Tier::Wormhole,
            Self::Blackhole => Tier::Blackhole,
        })
    }

    #[must_use]
    pub const fn of(tier: Tier) -> Self {
        match tier {
            Tier::Wormhole => Self::Wormhole,
            Tier::Blackhole => Self::Blackhole,
        }
    }

    /// `None` for Grayskull: tt-kmd still knows it, the software stack no
    /// longer supports it.
    #[must_use]
    pub const fn from_arch(arch: TtArch) -> Option<Self> {
        match arch {
            TtArch::Grayskull => None,
            TtArch::Wormhole => Some(Self::Wormhole),
            TtArch::Blackhole => Some(Self::Blackhole),
        }
    }
}
