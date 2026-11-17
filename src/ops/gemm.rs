use super::{Choice, key, len_u32};
use crate::backend::{Arg, OpSig};
use crate::buffer::DeviceBuffer;
use crate::elem::{Acc, Elem};
use crate::error::{Result, invalid};
use crate::stream::{Pending, Stream};

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
    pub fn new(m: usize, n: usize, k: usize) -> Self {
        GemmShape { m, n, k, batch: 1 }
    }

    pub fn batched(self, batch: usize) -> Self {
        GemmShape { batch, ..self }
    }
}

/// `gemm_<a>_<b>_<acc>`: `C = alpha·A·Bᵀ + beta·C`.
///
/// Layout: A is `M×K`, **B is `N×K`** (both K-contiguous, as block-scaled
/// formats need the blocks along K), C is `M×N`, all row-major. For block
/// formats `K` must be a multiple of the block size. With `beta == 0`, C is
/// not read (NaNs in C do not propagate).
///
/// Integer inputs need an `I32` accumulator (`alpha`/`beta` are `i32`, the
/// arithmetic wraps). Floating inputs use `F32` or `F16` accumulators.
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
            "gemm {batch}×({m}×{k})·({n}×{k})ᵀ: got A {}, B {}, C {} elements",
            a.len(),
            b.len(),
            c.len()
        )));
    }
    for (f, name) in [(A::FORMAT, "A"), (B::FORMAT, "B")] {
        if k % f.block() != 0 {
            return Err(invalid(format!("gemm: K = {k} is not a multiple of the {name} block {}", f.block())));
        }
    }
    if A::FORMAT.is_integer() != B::FORMAT.is_integer() || A::FORMAT.is_integer() != C::FORMAT.is_integer() {
        return Err(invalid("gemm: integer inputs need an I32 accumulator, float inputs a float one"));
    }
    let sig = OpSig::Gemm { a: A::FORMAT, b: B::FORMAT, acc: C::FORMAT };
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
    s.launch(&kk, &[&a.raw, &b.raw, &c.raw], &args)
}
