use super::{Mem, get, put, u};
use crate::elem::Format;
use crate::mock::codec::int;
use crate::runtime::backend::Arg;

/// The byte type a packed format widens to.
fn wide(f: Format) -> Format {
    match f {
        Format::Int { signed: true, .. } => Format::I8,
        _ => Format::U8,
    }
}

pub(super) fn pack(mem: &mut Mem, to: Format, a: &[Arg]) -> Option<()> {
    let len = u(a, 2)?;
    let v = int::decode(wide(to), &get(mem, a, 0)?, len)?;
    put(mem, a, 1, int::encode(to, &v)?)
}

pub(super) fn unpack(mem: &mut Mem, from: Format, a: &[Arg]) -> Option<()> {
    let len = u(a, 2)?;
    let v = int::decode(from, &get(mem, a, 0)?, len)?;
    put(mem, a, 1, int::encode(wide(from), &v)?)
}
