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
pub const SYMBOLS: [&str; 11] = [
    "nb_convert",
    "nb_int_convert",
    "nb_nvfp4_tensor_scale",
    "nb_quantize",
    "nb_dequantize",
    "nb_gemm_f",
    "nb_gemm_i",
    "nb_absmax_block",
    "nb_act",
    "nb_conv2d",
    "nb_resize",
];

/// Element-wise float formats the image ops read and write (one element per
/// byte or more, so work items never share a byte).
#[must_use]
pub fn elementwise_float(f: Format) -> bool {
    matches!(f, Format::F32 | Format::Tf32)
        || (matches!(f, Format::Float(_)) && f.block() == 1 && f.storage_bytes(1).is_some())
}

/// Which signatures the shared Emulated kernels implement (every SIMT
/// backend's `supports` for [`Tier::Emulated`](crate::Tier::Emulated); the mock
/// oracle covers the same set).
#[must_use]
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
        OpSig::Act { src, dst, .. } | OpSig::Resize { src, dst, .. } => {
            elementwise_float(src) && elementwise_float(dst)
        }
        OpSig::Conv2d { x, w, y, .. } => {
            elementwise_float(x) && elementwise_float(w) && elementwise_float(y)
        }
        #[allow(unreachable_patterns)]
        _ => false,
    }
}

fn bad(sig: OpSig) -> Error {
    Error::InvalidArgument(format!("bad arguments for {}", sig.name(false)))
}

/// The op arguments, read by position as documented on [`OpSig`].
#[derive(Clone, Copy)]
struct Args<'a>(&'a [Arg]);

impl Args<'_> {
    fn u32(self, i: usize) -> Option<u32> {
        match self.0.get(i)? {
            Arg::U32(v) => Some(*v),
            _ => None,
        }
    }

    fn usize(self, i: usize) -> Option<usize> {
        self.u32(i).map(|v| v as usize)
    }

    fn get(self, i: usize) -> Option<Arg> {
        self.0.get(i).copied()
    }

    /// The first `n` arguments, passed through unchanged.
    fn head(self, n: usize) -> Option<Vec<Arg>> {
        Some(self.0.get(..n)?.to_vec())
    }

    /// `[src, dst, len]`, the start of every element-wise op.
    fn unary(self) -> Option<(Arg, Arg, u32)> {
        Some((self.get(0)?, self.get(1)?, self.u32(2)?))
    }
}

fn u32_arg(v: usize) -> Option<Arg> {
    Some(Arg::U32(u32::try_from(v).ok()?))
}

fn items(n: usize) -> Option<u32> {
    u32::try_from(n).ok()
}

/// `a * b * c` without wrapping.
fn product(a: usize, b: usize, c: usize) -> Option<usize> {
    a.checked_mul(b)?.checked_mul(c)
}

/// The byte format a packed integer widens to.
fn wide(f: Format) -> Format {
    match f {
        Format::Int { signed: true, .. } => Format::I8,
        _ => Format::U8,
    }
}

/// Lowers one op to its kernel launches.
///
/// # Errors
///
/// [`Error::InvalidArgument`] when `args` don't match `sig` (count
/// or kinds), or a launch would exceed `u32::MAX` work items or offsets.
pub fn lower(sig: OpSig, args: &[Arg]) -> Result<Vec<Launch>> {
    lower_opt(sig, args).ok_or_else(|| bad(sig))
}

fn launch(symbol: &'static str, args: Vec<Arg>, n: usize) -> Option<Launch> {
    Some(Launch {
        symbol,
        args,
        items: items(n)?,
    })
}

fn lower_opt(sig: OpSig, a: &[Arg]) -> Option<Vec<Launch>> {
    let a = Args(a);
    match sig {
        OpSig::Convert { from, to } => convert(a, from, to),
        OpSig::Pack { to } => int_convert(a, wide(to), to),
        OpSig::Unpack { from } => int_convert(a, from, wide(from)),
        OpSig::Quantize { to } => quantize(a, to),
        OpSig::Dequantize { from } => dequantize(a, from),
        OpSig::Gemm { a: fa, b: fb, acc } => gemm(a, fa, fb, acc),
        OpSig::AbsmaxBlock { src } => absmax_block(a, src),
        OpSig::Act { src, dst, act } => {
            let (s, d, len) = a.unary()?;
            let args = vec![
                s,
                d,
                Arg::U32(len),
                Arg::U32(act.code()),
                a.get(3)?,
                a.get(4)?,
                Arg::U32(src.desc()),
                Arg::U32(dst.desc()),
            ];
            Some(vec![launch("nb_act", args, len as usize)?])
        }
        OpSig::Conv2d { x, w, y, bias } => {
            let outputs = a.usize(5)?;
            let mut args = a.head(5)?;
            args.extend([
                Arg::U32(x.desc()),
                Arg::U32(w.desc()),
                Arg::U32(y.desc()),
                Arg::U32(u32::from(bias)),
            ]);
            Some(vec![launch("nb_conv2d", args, outputs)?])
        }
        OpSig::Resize { src, dst, mode } => {
            let outputs = product(a.usize(2)?, a.usize(5)?, a.usize(6)?)?;
            let mut args = a.head(7)?;
            args.extend([
                Arg::U32(mode.code()),
                Arg::U32(src.desc()),
                Arg::U32(dst.desc()),
            ]);
            Some(vec![launch("nb_resize", args, outputs)?])
        }
        #[allow(unreachable_patterns)]
        _ => None,
    }
}

/// `nb_convert`: one work item per group of elements that share bytes.
fn convert(a: Args<'_>, from: Format, to: Format) -> Option<Vec<Launch>> {
    let (src, dst, len) = a.unary()?;
    let group = from.block().max(to.block());
    let saturate = Arg::U32(u32::from(a.u32(3)? == 0));
    let args = vec![
        src,
        dst,
        Arg::U32(len),
        saturate,
        Arg::U32(from.desc()),
        Arg::U32(to.desc()),
        u32_arg(group)?,
    ];
    Some(vec![launch("nb_convert", args, len as usize / group)?])
}

/// `nb_int_convert`: packing and unpacking sub-byte integers.
fn int_convert(a: Args<'_>, from: Format, to: Format) -> Option<Vec<Launch>> {
    let (src, dst, len) = a.unary()?;
    let group = from.block().max(to.block());
    let args = vec![
        src,
        dst,
        Arg::U32(len),
        Arg::U32(from.desc()),
        Arg::U32(to.desc()),
        u32_arg(group)?,
    ];
    Some(vec![launch("nb_int_convert", args, len as usize / group)?])
}

/// `nb_quantize`, after `nb_nvfp4_tensor_scale` for NVFP4's tensor scale.
fn quantize(a: Args<'_>, to: Format) -> Option<Vec<Launch>> {
    let (src, dst, len) = a.unary()?;
    let n = len as usize;
    let scales = to.scale_offset(n);
    let mut launches = Vec::new();
    if to == Format::Nvfp4 {
        let tensor_scale = scales + n / to.block();
        launches.push(launch(
            "nb_nvfp4_tensor_scale",
            vec![src, dst, Arg::U32(len), u32_arg(tensor_scale)?],
            1,
        )?);
    }
    let args = vec![
        src,
        dst,
        Arg::U32(len),
        Arg::U32(to.desc()),
        u32_arg(scales)?,
    ];
    launches.push(launch("nb_quantize", args, n / to.block())?);
    Some(launches)
}

/// `nb_dequantize`: one work item per block.
fn dequantize(a: Args<'_>, from: Format) -> Option<Vec<Launch>> {
    let (src, dst, len) = a.unary()?;
    let n = len as usize;
    let args = vec![
        src,
        dst,
        Arg::U32(len),
        Arg::U32(from.desc()),
        u32_arg(from.scale_offset(n))?,
    ];
    Some(vec![launch("nb_dequantize", args, n / from.block())?])
}

/// `nb_gemm_f` / `nb_gemm_i`: one work item per output element.
fn gemm(a: Args<'_>, fa: Format, fb: Format, acc: Format) -> Option<Vec<Launch>> {
    let (m, n, k, batch) = (a.usize(3)?, a.usize(4)?, a.usize(5)?, a.usize(6)?);
    let mut args = a.head(9)?;
    args.extend([
        Arg::U32(fa.desc()),
        Arg::U32(fb.desc()),
        Arg::U32(acc.desc()),
        u32_arg(fa.scale_offset(product(batch, m, k)?))?,
        u32_arg(fb.scale_offset(product(batch, n, k)?))?,
    ]);
    let symbol = if acc == Format::I32 {
        "nb_gemm_i"
    } else {
        "nb_gemm_f"
    };
    Some(vec![launch(symbol, args, product(batch, m, n)?)?])
}

/// `nb_absmax_block`: one work item per block.
fn absmax_block(a: Args<'_>, f: Format) -> Option<Vec<Launch>> {
    let (src, dst, len) = a.unary()?;
    let block = a.u32(3)?;
    let n = len as usize;
    let args = vec![
        src,
        dst,
        Arg::U32(len),
        Arg::U32(block),
        Arg::U32(f.desc()),
        u32_arg(f.scale_offset(n))?,
    ];
    Some(vec![launch(
        "nb_absmax_block",
        args,
        n / (block as usize).max(1),
    )?])
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::runtime::BufId;

    #[test]
    fn oversized_shapes_are_bad_arguments_not_overflows() {
        let sig = OpSig::Gemm {
            a: Format::F32,
            b: Format::F32,
            acc: Format::F32,
        };
        let buf = |i| Arg::Buf(BufId(i));
        let big = Arg::U32(u32::MAX);
        let args = [
            buf(0),
            buf(1),
            buf(2),
            big,
            big,
            big,
            big,
            Arg::F32(1.0),
            Arg::F32(0.0),
        ];
        assert!(matches!(lower(sig, &args), Err(Error::InvalidArgument(_))));
        // A shape that fits lowers to one launch per output element.
        let small = [
            buf(0),
            buf(1),
            buf(2),
            Arg::U32(2),
            Arg::U32(3),
            Arg::U32(4),
            Arg::U32(5),
        ];
        let args = [&small[..], &[Arg::F32(1.0), Arg::F32(0.0)]].concat();
        let launches = lower(sig, &args).expect("valid gemm");
        assert_eq!(launches.len(), 1);
        assert_eq!(
            (launches[0].symbol, launches[0].items),
            ("nb_gemm_f", 5 * 2 * 3)
        );
    }
}
