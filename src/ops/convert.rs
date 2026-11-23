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
///
/// # Errors
///
/// [`Error::InvalidArgument`](crate::Error::InvalidArgument) for lengths that differ.
/// [`Error::Unsupported`](crate::Error::Unsupported) when the [`Choice`] finds no
/// kernel for these formats (on an [`Exactness::Bit`](crate::Exactness::Bit) stream: none that is
/// bit-exact); [`Error::InvalidArgument`](crate::Error::InvalidArgument) also for a buffer of
/// another device or an approximate input on such a stream. Errors of the device pass through.
pub fn convert<F: FloatElem, T: FloatElem>(
    s: &mut Stream,
    choice: Choice,
    src: &DeviceBuffer<F>,
    dst: &mut DeviceBuffer<T>,
    overflow: Overflow,
) -> Result<Pending<()>> {
    let ov = match overflow {
        Overflow::Saturate => 0,
        Overflow::Special => 1,
    };
    unary(
        s,
        choice,
        OpSig::Convert {
            from: F::FORMAT,
            to: T::FORMAT,
        },
        src,
        dst,
        Some(Arg::U32(ov)),
    )
}
