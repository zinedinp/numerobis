//! Float <-> float conversion.

use miraculix::formats::Overflow;

use super::{Choice, unary};
use crate::elem::FloatElem;
use crate::runtime::backend::{Arg, OpSig};
use crate::runtime::{DeviceBuffer, Pending, Result, Stream};

/// `convert_<from>_<to>`, element for element, bit-exact with the miraculix
/// codec of the same pair. Stage 1 has `F32 -> any` and `any -> F32`.
///
/// `overflow` chooses between saturating and producing Inf/NaN for values
/// above the target's largest finite value (formats with neither always saturate).
pub fn convert<F: FloatElem, T: FloatElem>(
    s: &mut Stream,
    choice: Choice,
    src: &DeviceBuffer<F>,
    dst: &mut DeviceBuffer<T>,
    overflow: Overflow,
) -> Result<Pending<()>> {
    let ov = match overflow {
        Overflow::Saturate => 0,
        _ => 1,
    };
    unary(s, choice, OpSig::Convert { from: F::FORMAT, to: T::FORMAT }, src, dst, Some(Arg::U32(ov)))
}
