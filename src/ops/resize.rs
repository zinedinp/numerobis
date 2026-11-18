//! Image resize.

use super::{Choice, key, len_u32};
use crate::elem::FloatElem;
use crate::runtime::backend::{Arg, OpSig, ResizeMode};
use crate::runtime::error::{Result, invalid};
use crate::runtime::{DeviceBuffer, Pending, Stream};

/// `resize_<mode>_<src>_<dst>`: `planes` row-major `h x w` planes (e.g. N*C of
/// an NCHW tensor, or the channels of a planar image) to `ho x wo`, half-pixel
/// centres (`align_corners = false`). Nearest is exact on every tier;
/// bilinear is exact on [`Tier::Emulated`](crate::Tier::Emulated).
#[allow(clippy::too_many_arguments)]
pub fn resize<S: FloatElem, D: FloatElem>(
    s: &mut Stream,
    choice: Choice,
    mode: ResizeMode,
    planes: usize,
    (h, w): (usize, usize),
    (ho, wo): (usize, usize),
    src: &DeviceBuffer<S>,
    dst: &mut DeviceBuffer<D>,
) -> Result<Pending<()>> {
    if h == 0 || w == 0 || ho == 0 || wo == 0 {
        return Err(invalid("resize: empty plane"));
    }
    if src.len() != planes * h * w || dst.len() != planes * ho * wo {
        return Err(invalid(format!(
            "resize: {planes} planes {h}x{w} -> {ho}x{wo} need {} -> {} elements, got {} -> {}",
            planes * h * w,
            planes * ho * wo,
            src.len(),
            dst.len()
        )));
    }
    // (2*o+1)*h must fit the kernel's 32-bit index arithmetic.
    len_u32((2 * ho + 1) * h)?;
    len_u32((2 * wo + 1) * w)?;
    len_u32(planes * (h * w).max(ho * wo))?;
    let k = key(
        s,
        choice,
        OpSig::Resize {
            src: S::FORMAT,
            dst: D::FORMAT,
            mode,
        },
        false,
    )?;
    let args = [
        Arg::Buf(src.raw.id),
        Arg::Buf(dst.raw.id),
        Arg::U32(len_u32(planes)?),
        Arg::U32(len_u32(h)?),
        Arg::U32(len_u32(w)?),
        Arg::U32(len_u32(ho)?),
        Arg::U32(len_u32(wo)?),
    ];
    s.launch(&k, &[&src.raw], &[&dst.raw], &args)
}
