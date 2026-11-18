//! Stage-1 ops, one module per calculation kind:
//!
//! | Module | Ops | Kernel names |
//! |---|---|---|
//! | [`convert`](mod@convert) | float <-> float | `convert_<from>_<to>` |
//! | [`quantize`](mod@quantize) | F32 <-> block formats | `quantize_<to>`, `dequantize_<from>` |
//! | [`int`] | I8/U8 <-> packed I4/U4/I2/I1 | `pack_<to>`, `unpack_<from>` |
//! | [`gemm`](mod@gemm) | matrix multiply | `gemm[_strided]_<a>_<b>_<acc>` |
//! | [`reduce`] | per-block absmax | `absmax_block_<src>` |
//! | [`act`](mod@act) | activations (exact `exp`/`tanh`/`gelu`) | `act_<kind>_<src>_<dst>` |
//! | [`conv`] | direct 2-D convolution, optional bias + fused activation | `conv2d[_bias]_<x>_<w>_<y>` |
//! | [`resize`](mod@resize) | nearest / bilinear image resize | `resize_<mode>_<src>_<dst>` |
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

pub mod act;
pub mod conv;
pub mod convert;
pub mod gemm;
pub mod int;
pub mod quantize;
pub mod reduce;
pub mod resize;

pub use convert::convert;
pub use gemm::{GemmShape, gemm};
pub use int::{pack, unpack};
pub use quantize::{dequantize, quantize};
pub use act::{Act, act};
pub use conv::{Conv2dShape, conv2d};
pub use reduce::absmax_block;
pub use resize::resize;
pub use crate::runtime::backend::ResizeMode;

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
    let exact = s.exactness() == crate::Exactness::Bit;
    let tier = match choice {
        Choice::Up => crate::auto_up::choose_with(s.device(), sig, batched, exact)?,
        Choice::Down(t) => {
            if exact && !sig.exact_on(t) {
                return Err(Error::Unsupported(format!(
                    "{name} on tier {} is not bit-exact; this stream is Exactness::Bit (game state)",
                    t.name()
                )));
            }
            t
        }
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
    unary_args(s, choice, sig, src, dst, extra.as_slice())
}

/// [`unary`] with any number of trailing scalar arguments.
pub(crate) fn unary_args<F: Elem, T: Elem>(
    s: &mut Stream,
    choice: Choice,
    sig: OpSig,
    src: &DeviceBuffer<F>,
    dst: &mut DeviceBuffer<T>,
    extra: &[Arg],
) -> Result<Pending<()>> {
    if src.len() != dst.len() {
        return Err(invalid(format!("{}: {} elements into {}", sig.name(false), src.len(), dst.len())));
    }
    let k = key(s, choice, sig, false)?;
    let mut args = vec![Arg::Buf(src.raw.id), Arg::Buf(dst.raw.id), Arg::U32(len_u32(src.len())?)];
    args.extend_from_slice(extra);
    s.launch(&k, &[&src.raw], &[&dst.raw], &args)
}
