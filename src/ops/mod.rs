//! Stage-1 ops, one module per calculation kind:
//!
//! | Module | Ops | Kernel names |
//! |---|---|---|
//! | [`convert`](mod@convert) | float <-> float | `convert_<from>_<to>` |
//! | [`quantize`](mod@quantize) | F32 <-> block formats | `quantize_<to>`, `dequantize_<from>` |
//! | [`int`] | I8/U8 <-> packed I4/U4/I2/I1 | `pack_<to>`, `unpack_<from>` |
//! | [`gemm`](mod@gemm) | matrix multiply | `gemm[_strided]_<a>_<b>_<acc>` |
//! | [`reduce`] | per-block absmax | `absmax_block_<src>` |
//!
//! The element traits ([`FloatElem`](crate::elem::FloatElem),
//! [`Quantize`](crate::elem::Quantize), [`PackedInt`](crate::elem::PackedInt), ...)
//! rule out meaningless pairs at compile time:
//!
//! ```compile_fail
//! # use numerobis::{elem::*, *};
//! // Block formats go through `quantize`, not `convert`.
//! fn f(s: &mut Stream, x: &DeviceBuffer<F32>, y: &mut DeviceBuffer<Mx<E2M1>>) {
//!     let _ = ops::convert(s, Choice::Up, x, y, Overflow::Saturate);
//! }
//! ```
//! ```compile_fail
//! # use numerobis::{elem::*, *};
//! // GGUF is decode-only.
//! fn f(s: &mut Stream, x: &DeviceBuffer<F32>, y: &mut DeviceBuffer<Q8_0>) {
//!     let _ = ops::quantize(s, Choice::Up, x, y);
//! }
//! ```
//! ```compile_fail
//! # use numerobis::{elem::*, *};
//! // I4 packs from I8, not U8.
//! fn f(s: &mut Stream, x: &DeviceBuffer<U8>, y: &mut DeviceBuffer<I4>) {
//!     let _ = ops::pack(s, Choice::Up, x, y);
//! }
//! ```
//!
//! Every op takes a [`Choice`] of tier: [`Choice::Up`] is `auto_up` (best tier
//! that has the kernel), [`Choice::Down`] forces one tier (`auto_down`, for
//! test matrices and benchmarks).

pub mod convert;
pub mod gemm;
pub mod int;
pub mod quantize;
pub mod reduce;

pub use convert::convert;
pub use gemm::{GemmShape, gemm};
pub use int::{pack, unpack};
pub use quantize::{dequantize, quantize};
pub use reduce::absmax_block;

use crate::elem::Elem;
use crate::runtime::backend::{Arg, KernelKey, OpSig};
use crate::runtime::error::{Error, Result, invalid};
use crate::runtime::{DeviceBuffer, Pending, Stream};
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
    u32::try_from(n).map_err(|_| invalid("length exceeds u32"))
}

/// Shared launcher for element-wise `src -> dst` ops: `[src, dst, len, extra...]`.
pub(crate) fn unary<F: Elem, T: Elem>(
    s: &mut Stream,
    choice: Choice,
    sig: OpSig,
    src: &DeviceBuffer<F>,
    dst: &mut DeviceBuffer<T>,
    extra: Option<Arg>,
) -> Result<Pending<()>> {
    if src.len() != dst.len() {
        return Err(invalid(format!("{}: {} elements into {}", sig.name(false), src.len(), dst.len())));
    }
    let k = key(s, choice, sig, false)?;
    let mut args = vec![Arg::Buf(src.raw.id), Arg::Buf(dst.raw.id), Arg::U32(len_u32(src.len())?)];
    args.extend(extra);
    s.launch(&k, &[&src.raw, &dst.raw], &args)
}
