//! Block formats: MX, NVFP4, `BFP_B` (both ways) and GGUF (decode only).

use super::{Choice, unary};
use crate::elem::{BlockElem, F32, Quantize};
use crate::runtime::backend::OpSig;
use crate::runtime::{DeviceBuffer, Pending, Result, Stream};

/// `quantize_<to>`: `F32` -> block format, bit-exact with miraculix
/// (`mx_quantize`, `nvfp4_quantize` with `nvfp4_tensor_scale`, `bfp_encode_rows`).
/// Elements saturate.
///
/// # Errors
///
/// [`Error::InvalidArgument`](crate::Error::InvalidArgument) for lengths that differ.
/// [`Error::Unsupported`](crate::Error::Unsupported) when the [`Choice`] finds no
/// kernel for these formats (on an [`Exactness::Bit`](crate::Exactness::Bit) stream: none that is
/// bit-exact); [`Error::InvalidArgument`](crate::Error::InvalidArgument) also for a buffer of
/// another device or an approximate input on such a stream. Errors of the device pass through.
pub fn quantize<B: Quantize>(
    s: &mut Stream,
    choice: Choice,
    src: &DeviceBuffer<F32>,
    dst: &mut DeviceBuffer<B>,
) -> Result<Pending<()>> {
    unary(s, choice, OpSig::Quantize { to: B::FORMAT }, src, dst, None)
}

/// `dequantize_<from>`: block format -> `F32`, bit-exact with miraculix.
///
/// # Errors
///
/// [`Error::InvalidArgument`](crate::Error::InvalidArgument) for lengths that differ.
/// [`Error::Unsupported`](crate::Error::Unsupported) when the [`Choice`] finds no
/// kernel for these formats (on an [`Exactness::Bit`](crate::Exactness::Bit) stream: none that is
/// bit-exact); [`Error::InvalidArgument`](crate::Error::InvalidArgument) also for a buffer of
/// another device or an approximate input on such a stream. Errors of the device pass through.
pub fn dequantize<B: BlockElem>(
    s: &mut Stream,
    choice: Choice,
    src: &DeviceBuffer<B>,
    dst: &mut DeviceBuffer<F32>,
) -> Result<Pending<()>> {
    unary(
        s,
        choice,
        OpSig::Dequantize { from: B::FORMAT },
        src,
        dst,
        None,
    )
}
