use super::{Mem, get, put, u};
use crate::elem::{Format, as_bytes};
use crate::mock::codec;
use crate::runtime::backend::Arg;

/// NaN never wins the `>` comparison, so NaN inputs are ignored.
pub(super) fn absmax_block(mem: &mut Mem, src: Format, a: &[Arg]) -> Option<()> {
    let (len, block) = (u(a, 2)?, u(a, 3)?);
    let x = codec::decode_f32(src, &get(mem, a, 0)?, len)?;
    let m: Vec<f32> = x
        .chunks_exact(block)
        .map(|c| {
            c.iter()
                .fold(0.0f32, |m, v| if v.abs() > m { v.abs() } else { m })
        })
        .collect();
    put(mem, a, 1, as_bytes(&m).to_vec())
}
