//! Element-wise activations.

use super::Choice;
use crate::elem::FloatElem;
use crate::runtime::backend::{ActKind, Arg, OpSig};
use crate::runtime::error::{Result, invalid};
use crate::runtime::{DeviceBuffer, Pending, Stream};

/// An activation function with its parameters.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Act {
    Relu,
    /// `x >= 0 ? x : alpha*x`.
    LeakyRelu(f32),
    Sigmoid,
    Tanh,
    /// `x * sigmoid(x)`.
    Silu,
    /// Tanh form of GELU.
    Gelu,
    /// `min(max(x, lo), hi)`, NaN passes through.
    Clamp(f32, f32),
    Exp,
}

impl Act {
    pub fn kind(self) -> ActKind {
        match self {
            Act::Relu => ActKind::Relu,
            Act::LeakyRelu(_) => ActKind::LeakyRelu,
            Act::Sigmoid => ActKind::Sigmoid,
            Act::Tanh => ActKind::Tanh,
            Act::Silu => ActKind::Silu,
            Act::Gelu => ActKind::Gelu,
            Act::Clamp(..) => ActKind::Clamp,
            Act::Exp => ActKind::Exp,
        }
    }

    fn params(self) -> (f32, f32) {
        match self {
            Act::LeakyRelu(a) => (a, 0.0),
            Act::Clamp(lo, hi) => (lo, hi),
            _ => (0.0, 0.0),
        }
    }
}

/// `act_<kind>_<src>_<dst>`: `dst[i] = act(src[i])`, computed in F32.
///
/// On [`Tier::Emulated`](crate::Tier::Emulated) the transcendental functions
/// come from `kernels/include/nb_math.h` (only `+ - * /` and `fma` in a fixed
/// order), so every device returns the same bits: usable for game state.
/// `dst` may be a different float format (e.g. F32 in, BF16 out).
pub fn act<S: FloatElem, D: FloatElem>(
    s: &mut Stream,
    choice: Choice,
    f: Act,
    src: &DeviceBuffer<S>,
    dst: &mut DeviceBuffer<D>,
) -> Result<Pending<()>> {
    if let Act::Clamp(lo, hi) = f
        && lo
            .partial_cmp(&hi)
            .is_none_or(|o| o == core::cmp::Ordering::Greater)
    {
        return Err(invalid(format!("clamp: lo {lo} > hi {hi}")));
    }
    let (p0, p1) = f.params();
    let sig = OpSig::Act {
        src: S::FORMAT,
        dst: D::FORMAT,
        act: f.kind(),
    };
    super::unary_args(s, choice, sig, src, dst, &[Arg::F32(p0), Arg::F32(p1)])
}
