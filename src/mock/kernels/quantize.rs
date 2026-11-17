use super::{Mem, get, put, u};
use crate::elem::{Format, as_bytes};
use crate::mock::codec::block;
use crate::runtime::backend::Arg;

pub(super) fn encodable(to: Format) -> bool {
    matches!(to, Format::Mx(_) | Format::Nvfp4 | Format::Bfp(_))
}

pub(super) fn quantize(mem: &mut Mem, to: Format, a: &[Arg]) -> Option<()> {
    let len = u(a, 2)?;
    let x: Vec<f32> = crate::elem::from_bytes(&get(mem, a, 0)?);
    put(mem, a, 1, block::encode(to, &x[..len])?)
}

pub(super) fn dequantize(mem: &mut Mem, from: Format, a: &[Arg]) -> Option<()> {
    let len = u(a, 2)?;
    let y = block::decode(from, &get(mem, a, 0)?, len)?;
    put(mem, a, 1, as_bytes(&y).to_vec())
}
