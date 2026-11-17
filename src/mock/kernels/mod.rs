//! Host implementations of the Emulated semantics, one module per op kind
//! (mirrors [`crate::ops`] and `kernels/src/*.cl`).

mod convert;
mod gemm;
mod int;
mod quantize;
mod reduce;

use std::collections::HashMap;

use crate::runtime::backend::{Arg, OpSig};
use crate::runtime::error::{Error, Result};

/// Mock device memory.
pub(super) type Mem = HashMap<u64, Vec<u8>>;

pub(super) fn supported(sig: OpSig) -> bool {
    match sig {
        OpSig::Convert { from, to } => convert::supported(from, to),
        OpSig::Quantize { to } => quantize::encodable(to),
        OpSig::Dequantize { .. } | OpSig::Pack { .. } | OpSig::Unpack { .. } => true,
        OpSig::Gemm { a, b, acc } => gemm::supported(a, b, acc),
        OpSig::AbsmaxBlock { src } => !src.is_integer(),
        #[allow(unreachable_patterns)]
        _ => false,
    }
}

pub(super) fn run(mem: &mut Mem, sig: OpSig, args: &[Arg]) -> Result<()> {
    let r = match sig {
        OpSig::Convert { from, to } => convert::run(mem, from, to, args),
        OpSig::Quantize { to } => quantize::quantize(mem, to, args),
        OpSig::Dequantize { from } => quantize::dequantize(mem, from, args),
        OpSig::Pack { to } => int::pack(mem, to, args),
        OpSig::Unpack { from } => int::unpack(mem, from, args),
        OpSig::Gemm { a, b, acc } => gemm::run(mem, a, b, acc, args),
        OpSig::AbsmaxBlock { src } => reduce::absmax_block(mem, src, args),
        #[allow(unreachable_patterns)]
        _ => None,
    };
    r.ok_or_else(|| Error::Unsupported(format!("mock: {sig:?}")))
}

// Argument accessors; `None` turns into `Unsupported` in `run`.
fn buf(a: &[Arg], i: usize) -> Option<u64> {
    match a.get(i)? {
        Arg::Buf(b) => Some(b.0),
        _ => None,
    }
}
fn u(a: &[Arg], i: usize) -> Option<usize> {
    match a.get(i)? {
        Arg::U32(v) => Some(*v as usize),
        _ => None,
    }
}
fn get(mem: &Mem, a: &[Arg], i: usize) -> Option<Vec<u8>> {
    mem.get(&buf(a, i)?).cloned()
}
fn put(mem: &mut Mem, a: &[Arg], i: usize, v: Vec<u8>) -> Option<()> {
    *mem.get_mut(&buf(a, i)?)? = v;
    Some(())
}
