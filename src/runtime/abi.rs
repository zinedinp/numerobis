//! Kernel ABI of the shared Emulated kernels (`kernels/src/*.cl`).
//!
//! The kernels are format-generic: a format travels as its
//! [`Format::desc`](crate::elem::Format::desc) argument, so one binary per
//! target covers every `<op>_<types>` name. [`lower`] turns an [`OpSig`] plus
//! the op arguments (as documented on [`OpSig`]) into the kernel launches a
//! SIMT backend has to issue, in order, on one queue.
//!
//! Kernel arguments: buffers are device pointers, everything else 32 bits.
//! `items` is the number of work items (1-D); backends may round the grid
//! up, the kernels guard against the excess.

use crate::elem::Format;
use crate::runtime::backend::{Arg, OpSig};
use crate::runtime::error::{Error, Result};

/// One kernel launch.
#[derive(Debug, Clone, PartialEq)]
pub struct Launch {
    /// Kernel symbol in the target binary (`nb_convert`, `nb_gemm_f`, ...).
    pub symbol: &'static str,
    pub args: Vec<Arg>,
    pub items: u32,
}

/// Every kernel symbol a target binary must export.
pub const SYMBOLS: [&str; 8] = [
    "nb_convert",
    "nb_int_convert",
    "nb_nvfp4_tensor_scale",
    "nb_quantize",
    "nb_dequantize",
    "nb_gemm_f",
    "nb_gemm_i",
    "nb_absmax_block",
];

/// Which signatures the shared Emulated kernels implement (every SIMT
/// backend's `supports` for [`Tier::Emulated`](crate::Tier::Emulated); the mock
/// oracle covers the same set).
pub fn supported(sig: OpSig) -> bool {
    use miraculix::formats::MiniFloat;
    match sig {
        OpSig::Convert { from, to } => from == Format::F32 || to == Format::F32,
        OpSig::Quantize { to } => matches!(to, Format::Mx(_) | Format::Nvfp4 | Format::Bfp(_)),
        OpSig::Dequantize { .. } | OpSig::Pack { .. } | OpSig::Unpack { .. } => true,
        OpSig::Gemm { a, b, acc } => match acc {
            Format::I32 => a.is_integer() && b.is_integer(),
            Format::F32 | Format::Float(MiniFloat::F16) => !a.is_integer() && !b.is_integer(),
            _ => false,
        },
        OpSig::AbsmaxBlock { src } => !src.is_integer(),
        #[allow(unreachable_patterns)]
        _ => false,
    }
}

fn bad(sig: OpSig) -> Error {
    Error::InvalidArgument(format!("bad arguments for {}", sig.name(false)))
}

fn u(args: &[Arg], i: usize) -> Option<u32> {
    match args.get(i)? {
        Arg::U32(v) => Some(*v),
        _ => None,
    }
}

fn items(n: usize) -> Option<u32> {
    u32::try_from(n).ok()
}

/// The byte format a packed integer widens to.
fn wide(f: Format) -> Format {
    match f {
        Format::Int { signed: true, .. } => Format::I8,
        _ => Format::U8,
    }
}

/// Lowers one op to its kernel launches.
pub fn lower(sig: OpSig, args: &[Arg]) -> Result<Vec<Launch>> {
    lower_opt(sig, args).ok_or_else(|| bad(sig))
}

fn launch(symbol: &'static str, args: Vec<Arg>, n: usize) -> Option<Launch> {
    Some(Launch { symbol, args, items: items(n)? })
}

fn lower_opt(sig: OpSig, a: &[Arg]) -> Option<Vec<Launch>> {
    // Element count (arg 2) for every op but GEMM, whose arg 2 is C.
    let len = u(a, 2).map(|v| v as usize);
    let (src, dst) = (*a.first()?, *a.get(1)?);
    let u32a = |v: usize| Some(Arg::U32(u32::try_from(v).ok()?));
    Some(match sig {
        OpSig::Convert { from, to } => {
            let len = len?;
            let group = from.block().max(to.block());
            let sat = Arg::U32((u(a, 3)? == 0) as u32);
            let args = vec![src, dst, Arg::U32(len as u32), sat, Arg::U32(from.desc()), Arg::U32(to.desc()), u32a(group)?];
            vec![launch("nb_convert", args, len / group)?]
        }
        OpSig::Pack { to } | OpSig::Unpack { from: to } => {
            let len = len?;
            let (from, to) = if matches!(sig, OpSig::Pack { .. }) { (wide(to), to) } else { (to, wide(to)) };
            let group = from.block().max(to.block());
            let args = vec![src, dst, Arg::U32(len as u32), Arg::U32(from.desc()), Arg::U32(to.desc()), u32a(group)?];
            vec![launch("nb_int_convert", args, len / group)?]
        }
        OpSig::Quantize { to } => {
            let len = len?;
            let soff = to.scale_offset(len);
            let mut v = Vec::new();
            if to == Format::Nvfp4 {
                let ts_off = soff + len / to.block();
                v.push(launch("nb_nvfp4_tensor_scale", vec![src, dst, Arg::U32(len as u32), u32a(ts_off)?], 1)?);
            }
            let args = vec![src, dst, Arg::U32(len as u32), Arg::U32(to.desc()), u32a(soff)?];
            v.push(launch("nb_quantize", args, len / to.block())?);
            v
        }
        OpSig::Dequantize { from } => {
            let len = len?;
            let args = vec![src, dst, Arg::U32(len as u32), Arg::U32(from.desc()), u32a(from.scale_offset(len))?];
            vec![launch("nb_dequantize", args, len / from.block())?]
        }
        OpSig::Gemm { a: fa, b: fb, acc } => {
            let (m, n, k, batch) = (u(a, 3)? as usize, u(a, 4)? as usize, u(a, 5)? as usize, u(a, 6)? as usize);
            let mut args = a.get(..9)?.to_vec();
            args.extend([
                Arg::U32(fa.desc()),
                Arg::U32(fb.desc()),
                Arg::U32(acc.desc()),
                u32a(fa.scale_offset(batch * m * k))?,
                u32a(fb.scale_offset(batch * n * k))?,
            ]);
            let sym = if acc == Format::I32 { "nb_gemm_i" } else { "nb_gemm_f" };
            vec![launch(sym, args, batch * m * n)?]
        }
        OpSig::AbsmaxBlock { src: f } => {
            let len = len?;
            let block = u(a, 3)? as usize;
            let args = vec![src, dst, Arg::U32(len as u32), Arg::U32(block as u32), Arg::U32(f.desc()), u32a(f.scale_offset(len))?];
            vec![launch("nb_absmax_block", args, len / block.max(1))?]
        }
        #[allow(unreachable_patterns)]
        _ => return None,
    })
}
