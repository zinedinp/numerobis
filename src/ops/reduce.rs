//! Reductions (stage 1: only the per-block absmax for MX/NVFP4 scales).

use super::{Choice, key, len_u32};
use crate::elem::{Elem, F32};
use crate::runtime::DeviceBuffer;
use crate::runtime::backend::{Arg, OpSig};
use crate::runtime::error::{Result, invalid};
use crate::runtime::{Pending, Stream};

/// `absmax_block_<src>`: `dst[i] = max |src[i·block .. (i+1)·block]|`, the scale
/// input for MX / NVFP4 quantization. NaN inputs are ignored. The only
/// reduction in stage 1.
///
/// # Errors
///
/// [`Error::InvalidArgument`](crate::Error::InvalidArgument) when `block` is 0 or does not divide
/// `src`, `dst` does not hold one value per block, or a length is above `u32::MAX`.
/// [`Error::Unsupported`](crate::Error::Unsupported) when the [`Choice`] finds no
/// kernel for these formats (on an [`Exactness::Bit`](crate::Exactness::Bit) stream: none that is
/// bit-exact); [`Error::InvalidArgument`](crate::Error::InvalidArgument) also for a buffer of
/// another device or an approximate input on such a stream. Errors of the device pass through.
pub fn absmax_block<S: Elem>(
    s: &mut Stream,
    choice: Choice,
    src: &DeviceBuffer<S>,
    block: usize,
    dst: &mut DeviceBuffer<F32>,
) -> Result<Pending<()>> {
    if block == 0 || !src.len().is_multiple_of(block) || dst.len() != src.len() / block {
        return Err(invalid(format!(
            "absmax_block: {} elements, block {block}, {} outputs",
            src.len(),
            dst.len()
        )));
    }
    let k = key(s, choice, OpSig::AbsmaxBlock { src: S::FORMAT }, false)?;
    let args = [
        Arg::Buf(src.raw.id),
        Arg::Buf(dst.raw.id),
        Arg::U32(len_u32(src.len())?),
        Arg::U32(len_u32(block)?),
    ];
    s.launch(&k, &[&src.raw], &[&dst.raw], &args)
}
