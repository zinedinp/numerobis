//! Packed sub-byte integers.

use super::{Choice, unary};
use crate::elem::PackedInt;
use crate::runtime::backend::OpSig;
use crate::runtime::{DeviceBuffer, Pending, Result, Stream};

/// `pack_<to>`: `I8`/`U8` -> packed, saturating (miraculix `pack_int`/`pack_uint`).
///
/// # Errors
///
/// [`Error::InvalidArgument`](crate::Error::InvalidArgument) for lengths that differ.
/// [`Error::Unsupported`](crate::Error::Unsupported) when the [`Choice`] finds no
/// kernel for these formats (on an [`Exactness::Bit`](crate::Exactness::Bit) stream: none that is
/// bit-exact); [`Error::InvalidArgument`](crate::Error::InvalidArgument) also for a buffer of
/// another device or an approximate input on such a stream. Errors of the device pass through.
pub fn pack<P: PackedInt>(
    s: &mut Stream,
    choice: Choice,
    src: &DeviceBuffer<P::Wide>,
    dst: &mut DeviceBuffer<P>,
) -> Result<Pending<()>> {
    unary(s, choice, OpSig::Pack { to: P::FORMAT }, src, dst, None)
}

/// `unpack_<from>`: packed -> `I8` (sign-extended) or `U8`.
///
/// # Errors
///
/// [`Error::InvalidArgument`](crate::Error::InvalidArgument) for lengths that differ.
/// [`Error::Unsupported`](crate::Error::Unsupported) when the [`Choice`] finds no
/// kernel for these formats (on an [`Exactness::Bit`](crate::Exactness::Bit) stream: none that is
/// bit-exact); [`Error::InvalidArgument`](crate::Error::InvalidArgument) also for a buffer of
/// another device or an approximate input on such a stream. Errors of the device pass through.
pub fn unpack<P: PackedInt>(
    s: &mut Stream,
    choice: Choice,
    src: &DeviceBuffer<P>,
    dst: &mut DeviceBuffer<P::Wide>,
) -> Result<Pending<()>> {
    unary(s, choice, OpSig::Unpack { from: P::FORMAT }, src, dst, None)
}
