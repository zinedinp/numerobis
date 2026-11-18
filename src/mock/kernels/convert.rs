use miraculix::formats::Overflow;

use super::{Mem, get, put, u};
use crate::elem::Format;
use crate::mock::codec::float;
use crate::runtime::backend::Arg;

/// Stage 1: `F32 -> any float` and `any float -> F32`.
pub(super) fn run(mem: &mut Mem, from: Format, to: Format, a: &[Arg]) -> Option<()> {
    let len = u(a, 2)?;
    let ov = if u(a, 3)? == 0 {
        Overflow::Saturate
    } else {
        Overflow::Special
    };
    let x = float::decode(from, &get(mem, a, 0)?, len)?;
    put(mem, a, 1, float::encode(to, &x, ov)?)
}
