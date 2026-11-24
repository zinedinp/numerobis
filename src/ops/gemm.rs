//! Matrix multiply.

#![allow(
    clippy::many_single_char_names,
    reason = "M, N, K are named as in the math"
)]

use super::{Choice, key, len_u32};
use crate::elem::{Acc, Elem};
use crate::runtime::DeviceBuffer;
use crate::runtime::backend::{Arg, OpSig};
use crate::runtime::error::{Result, invalid};
use crate::runtime::{Pending, Stream};

/// GEMM problem size. `batch > 1` selects the strided-batched kernel; batches
/// are stored back to back.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GemmShape {
    pub m: usize,
    pub n: usize,
    pub k: usize,
    pub batch: usize,
}

impl GemmShape {
    #[must_use]
    pub fn new(m: usize, n: usize, k: usize) -> Self {
        GemmShape { m, n, k, batch: 1 }
    }

    #[must_use]
    pub fn batched(self, batch: usize) -> Self {
        GemmShape { batch, ..self }
    }
}

/// `gemm_<a>_<b>_<acc>`: `C = alpha*A*B^T + beta*C`.
///
/// Layout: A is `MxK`, **B is `NxK`** (both K-contiguous, as block-scaled
/// formats need the blocks along K), C is `MxN`, all row-major. For block
/// formats `K` must be a multiple of the block size. With `beta == 0`, C is
/// not read (NaNs in C do not propagate).
///
/// Integer inputs need an `I32` accumulator (`alpha`/`beta` are `i32`, the
/// arithmetic wraps). Floating inputs use `F32` or `F16` accumulators.
///
/// Which matrix instruction takes these formats on each engine (XMX, Tensor
/// Cores, Matrix Cores, Tensix), natively or after an exact widening:
/// [`matrix::Engine::route`](crate::matrix::Engine::route).
///
/// # Errors
///
/// [`Error::InvalidArgument`](crate::Error::InvalidArgument) for `batch` 0, buffer lengths that
/// don't match `shape`, a `K` that is not a multiple of the block size, an accumulator that doesn't
/// fit the inputs, or a dimension above `u32::MAX`.
/// [`Error::Unsupported`](crate::Error::Unsupported) when the [`Choice`] finds no
/// kernel for these formats (on an [`Exactness::Bit`](crate::Exactness::Bit) stream: none that is
/// bit-exact); [`Error::InvalidArgument`](crate::Error::InvalidArgument) also for a buffer of
/// another device or an approximate input on such a stream. Errors of the device pass through.
#[allow(clippy::too_many_arguments)]
pub fn gemm<A: Elem, B: Elem, C: Acc>(
    s: &mut Stream,
    choice: Choice,
    shape: GemmShape,
    alpha: C::Scalar,
    a: &DeviceBuffer<A>,
    b: &DeviceBuffer<B>,
    beta: C::Scalar,
    c: &mut DeviceBuffer<C>,
) -> Result<Pending<()>> {
    let GemmShape { m, n, k, batch } = shape;
    if batch == 0 {
        return Err(invalid("gemm: batch must be at least 1"));
    }
    if a.len() != batch * m * k || b.len() != batch * n * k || c.len() != batch * m * n {
        return Err(invalid(format!(
            "gemm {batch}x({m}x{k})*({n}x{k})^T: got A {}, B {}, C {} elements",
            a.len(),
            b.len(),
            c.len()
        )));
    }
    for (f, name) in [(A::FORMAT, "A"), (B::FORMAT, "B")] {
        if k % f.block() != 0 {
            return Err(invalid(format!(
                "gemm: K = {k} is not a multiple of the {name} block {}",
                f.block()
            )));
        }
    }
    if A::FORMAT.is_integer() != B::FORMAT.is_integer()
        || A::FORMAT.is_integer() != C::FORMAT.is_integer()
    {
        return Err(invalid(
            "gemm: integer inputs need an I32 accumulator, float inputs a float one",
        ));
    }
    let sig = OpSig::Gemm {
        a: A::FORMAT,
        b: B::FORMAT,
        acc: C::FORMAT,
    };
    let kk = key(s, choice, sig, batch > 1)?;
    let args = [
        Arg::Buf(a.raw.id),
        Arg::Buf(b.raw.id),
        Arg::Buf(c.raw.id),
        Arg::U32(len_u32(m)?),
        Arg::U32(len_u32(n)?),
        Arg::U32(len_u32(k)?),
        Arg::U32(len_u32(batch)?),
        alpha.into(),
        beta.into(),
    ];
    // C is an input too when beta != 0.
    let beta_zero = matches!(args[8], Arg::F32(v) if v == 0.0) || args[8] == Arg::I32(0);
    if beta_zero {
        s.launch(&kk, &[&a.raw, &b.raw], &[&c.raw], &args)
    } else {
        s.launch(&kk, &[&a.raw, &b.raw, &c.raw], &[&c.raw], &args)
    }
}
