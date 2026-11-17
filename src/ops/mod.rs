//! Stage-1 ops (plan B.1). Every op takes a [`Choice`] of tier: [`Choice::Up`]
//! is `auto_up` (best tier that has the kernel), [`Choice::Down`] forces one
//! tier (`auto_down`, for test matrices and benchmarks).

mod convert;
mod gemm;
mod reduce;

pub use convert::convert;
pub use gemm::{GemmShape, gemm};
pub use reduce::absmax_block;

use crate::backend::{KernelKey, OpSig};
use crate::error::{Error, Result};
use crate::stream::Stream;
use crate::tiers::Tier;

/// Tier selection for one op call.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Choice {
    /// Highest tier of the device that has this kernel.
    Up,
    /// Exactly this tier, or [`Error::Unsupported`].
    Down(Tier),
}

pub(crate) fn key(s: &Stream, choice: Choice, sig: OpSig, batched: bool) -> Result<KernelKey> {
    let name = sig.name(batched);
    let tier = match choice {
        Choice::Up => crate::auto_up::choose(s.device(), sig, batched)?,
        Choice::Down(t) => t,
    };
    let key = KernelKey { name, sig, tier };
    if s.device().backend().supports(&key) {
        Ok(key)
    } else {
        Err(Error::Unsupported(format!("{} on tier {}", key.name, tier.name())))
    }
}

pub(crate) fn len_u32(n: usize) -> Result<u32> {
    u32::try_from(n).map_err(|_| crate::error::invalid("length exceeds u32"))
}
