//! Packed sub-byte integers.

use super::{Choice, unary};
use crate::elem::PackedInt;
use crate::runtime::backend::OpSig;
use crate::runtime::{DeviceBuffer, Pending, Result, Stream};

/// `pack_<to>`: `I8`/`U8` -> packed, saturating (miraculix `pack_int`/`pack_uint`).
pub fn pack<P: PackedInt>(
    s: &mut Stream,
    choice: Choice,
    src: &DeviceBuffer<P::Wide>,
    dst: &mut DeviceBuffer<P>,
) -> Result<Pending<()>> {
    unary(s, choice, OpSig::Pack { to: P::FORMAT }, src, dst, None)
}

/// `unpack_<from>`: packed -> `I8` (sign-extended) or `U8`.
pub fn unpack<P: PackedInt>(
    s: &mut Stream,
    choice: Choice,
    src: &DeviceBuffer<P>,
    dst: &mut DeviceBuffer<P::Wide>,
) -> Result<Pending<()>> {
    unary(s, choice, OpSig::Unpack { from: P::FORMAT }, src, dst, None)
}
