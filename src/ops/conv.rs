//! Direct 2-D convolution (no im2col buffer).

use super::{Choice, key, len_u32};
use crate::elem::{F32, FloatElem, U8, as_bytes};
use crate::ops::act::Act;
use crate::runtime::backend::{Arg, OpSig};
use crate::runtime::error::{Result, invalid};
use crate::runtime::{DeviceBuffer, Pending, Stream};

/// Words of the parameter buffer (`CV_*` in `kernels/src/image.cl`).
pub const PARAM_WORDS: usize = 17;

/// Index into the parameter buffer.
#[derive(Debug, Clone, Copy)]
#[repr(usize)]
pub enum P {
    N,
    C,
    H,
    W,
    K,
    R,
    S,
    P,
    Q,
    Sh,
    Sw,
    Ph,
    Pw,
    Dh,
    Dw,
    G,
    Act,
}

/// Geometry of one convolution. Input `N x C x H x W`, weights
/// `K x (C/groups) x R x S`, output `N x K x P x Q`, all row-major (NCHW).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Conv2dShape {
    pub n: usize,
    pub c: usize,
    pub h: usize,
    pub w: usize,
    pub k: usize,
    pub r: usize,
    pub s: usize,
    pub stride: (usize, usize),
    pub pad: (usize, usize),
    pub dilation: (usize, usize),
    pub groups: usize,
}

impl Conv2dShape {
    /// Stride 1, no padding, no dilation, one group.
    pub fn new(n: usize, c: usize, h: usize, w: usize, k: usize, r: usize, s: usize) -> Self {
        Conv2dShape {
            n,
            c,
            h,
            w,
            k,
            r,
            s,
            stride: (1, 1),
            pad: (0, 0),
            dilation: (1, 1),
            groups: 1,
        }
    }
    pub fn stride(self, sh: usize, sw: usize) -> Self {
        Conv2dShape {
            stride: (sh, sw),
            ..self
        }
    }
    pub fn pad(self, ph: usize, pw: usize) -> Self {
        Conv2dShape {
            pad: (ph, pw),
            ..self
        }
    }
    pub fn dilation(self, dh: usize, dw: usize) -> Self {
        Conv2dShape {
            dilation: (dh, dw),
            ..self
        }
    }
    pub fn groups(self, g: usize) -> Self {
        Conv2dShape { groups: g, ..self }
    }

    /// Output height and width.
    pub fn out_hw(&self) -> Option<(usize, usize)> {
        let ext = |len: usize, pad: usize, k: usize, d: usize, st: usize| {
            let span = d.checked_mul(k.checked_sub(1)?)?.checked_add(1)?;
            let padded = len.checked_add(2 * pad)?;
            if st == 0 || padded < span {
                None
            } else {
                Some((padded - span) / st + 1)
            }
        };
        Some((
            ext(self.h, self.pad.0, self.r, self.dilation.0, self.stride.0)?,
            ext(self.w, self.pad.1, self.s, self.dilation.1, self.stride.1)?,
        ))
    }

    pub fn input_len(&self) -> usize {
        self.n * self.c * self.h * self.w
    }
    pub fn weight_len(&self) -> usize {
        self.k * (self.c / self.groups.max(1)) * self.r * self.s
    }
    pub fn output_len(&self) -> Option<usize> {
        let (p, q) = self.out_hw()?;
        Some(self.n * self.k * p * q)
    }
}

/// `conv2d[_bias]_<x>_<w>_<y>`: direct convolution, F32 accumulation in the
/// fixed order (c, r, s), optional F32 bias per output channel and an optional
/// fused activation (`Relu`, `Sigmoid`, `Tanh`, `Silu`, `Gelu`, `Exp`; the
/// parametrised ones are rejected here, run [`act`](crate::ops::act) after).
///
/// The parameter block is uploaded once per call; inside a recorded
/// [`CommandList`](crate::CommandList) it is recorded with the launch.
#[allow(clippy::too_many_arguments)]
pub fn conv2d<X: FloatElem, W: FloatElem, Y: FloatElem>(
    s: &mut Stream,
    choice: Choice,
    shape: Conv2dShape,
    x: &DeviceBuffer<X>,
    w: &DeviceBuffer<W>,
    bias: Option<&DeviceBuffer<F32>>,
    fused: Option<Act>,
    y: &mut DeviceBuffer<Y>,
) -> Result<Pending<()>> {
    let sh = shape;
    let g = sh.groups;
    if g == 0 || !sh.c.is_multiple_of(g) || !sh.k.is_multiple_of(g) || sh.r == 0 || sh.s == 0 {
        return Err(invalid(format!(
            "conv2d: C {} and K {} must be multiples of groups {g}",
            sh.c, sh.k
        )));
    }
    let (p, q) = sh
        .out_hw()
        .ok_or_else(|| invalid("conv2d: kernel larger than the padded input or stride 0"))?;
    let out = sh.n * sh.k * p * q;
    if x.len() != sh.input_len() || w.len() != sh.weight_len() || y.len() != out {
        return Err(invalid(format!(
            "conv2d: want x {}, w {}, y {} elements; got {}, {}, {}",
            sh.input_len(),
            sh.weight_len(),
            out,
            x.len(),
            w.len(),
            y.len()
        )));
    }
    if let Some(b) = bias
        && b.len() != sh.k
    {
        return Err(invalid(format!(
            "conv2d: bias has {} elements, K is {}",
            b.len(),
            sh.k
        )));
    }
    let act = match fused {
        None => 0,
        Some(a @ (Act::LeakyRelu(_) | Act::Clamp(..))) => {
            return Err(invalid(format!(
                "conv2d: {a:?} cannot be fused (it has parameters)"
            )));
        }
        Some(a) => a.kind().code() + 1,
    };
    let words = [
        sh.n,
        sh.c,
        sh.h,
        sh.w,
        sh.k,
        sh.r,
        sh.s,
        p,
        q,
        sh.stride.0,
        sh.stride.1,
        sh.pad.0,
        sh.pad.1,
        sh.dilation.0,
        sh.dilation.1,
        g,
    ];
    let mut prm = [0u32; PARAM_WORDS];
    for (d, v) in prm.iter_mut().zip(words) {
        *d = len_u32(v)?;
    }
    prm[P::Act as usize] = act;
    // Index arithmetic in the kernel is 32-bit.
    len_u32(sh.input_len().max(out).max(sh.weight_len()))?;

    let sig = OpSig::Conv2d {
        x: X::FORMAT,
        w: W::FORMAT,
        y: Y::FORMAT,
        bias: bias.is_some(),
    };
    let k = key(s, choice, sig, false)?;
    let params = s.upload_packed::<U8>(PARAM_WORDS * 4, as_bytes(&prm))?;
    let braw = bias.map_or(&x.raw, |b| &b.raw);
    let args = [
        Arg::Buf(x.raw.id),
        Arg::Buf(w.raw.id),
        Arg::Buf(braw.id),
        Arg::Buf(y.raw.id),
        Arg::Buf(params.raw.id),
        Arg::U32(len_u32(out)?),
    ];
    let mut reads = vec![&x.raw, &w.raw, &params.raw];
    if let Some(b) = bias {
        reads.push(&b.raw);
    }
    s.launch(&k, &reads, &[&y.raw], &args)
}
