//! Bit-exact host mirror of `kernels/include/nb_math.h` and `nb_act`: the
//! same operations in the same order, constants as bit patterns, `mul_add`
//! for every `nb_fma`.

use miraculix::formats::Overflow;

use super::{Mem, get, put, u};
use crate::elem::Format;
use crate::mock::codec;
use crate::runtime::backend::{ActKind, Arg};

const fn b(u: u32) -> f32 {
    f32::from_bits(u)
}

fn pow2(k: i32) -> f32 {
    if k >= -126 {
        b(((k + 127) as u32) << 23)
    } else {
        b(1u32 << (k + 149))
    }
}

pub(crate) fn exp(x: f32) -> f32 {
    if x.is_nan() {
        return x;
    }
    if x > b(0x42B1_7218) {
        return f32::INFINITY;
    }
    if x < b(0xC2CF_F1B5) {
        return 0.0;
    }
    let t = x * b(0x3FB8_AA3B);
    let k = (if t >= 0.0 { t + 0.5 } else { t - 0.5 }) as i32;
    let kf = k as f32;
    let mut r = (-kf).mul_add(b(0x3F31_7200), x);
    r = (-kf).mul_add(b(0x35BF_BE8E), r);
    let mut p = b(0x3950_0D01);
    for c in [
        b(0x3AB6_0B61),
        b(0x3C08_8889),
        b(0x3D2A_AAAB),
        b(0x3E2A_AAAB),
        0.5,
        1.0,
        1.0,
    ] {
        p = p.mul_add(r, c);
    }
    if k > 127 {
        p * pow2(127) * pow2(k - 127)
    } else if k < -126 {
        p * pow2(k + 64) * pow2(-64)
    } else {
        p * pow2(k)
    }
}

/// Canonical quiet NaN for every NaN result (`nb_canon`).
pub(crate) fn canon(x: f32) -> f32 {
    if x.is_nan() {
        f32::from_bits(0x7FC0_0000)
    } else {
        x
    }
}

pub(crate) fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + exp(-x))
}

pub(crate) fn tanh(x: f32) -> f32 {
    if x.is_nan() {
        return x;
    }
    let a = x.abs();
    if a < b(0x3980_0000) {
        return x;
    }
    let t = if a > 9.0 {
        1.0
    } else {
        1.0 - 2.0 / (exp(2.0 * a) + 1.0)
    };
    if x < 0.0 { -t } else { t }
}

pub(crate) fn gelu(x: f32) -> f32 {
    let x3 = x * x * x;
    let u = b(0x3F4C_422A) * (x + b(0x3D37_2713) * x3);
    0.5 * x * (1.0 + tanh(u))
}

pub(crate) fn activate(act: ActKind, x: f32, p0: f32, p1: f32) -> f32 {
    match act {
        ActKind::Relu => {
            if x.is_nan() || x > 0.0 {
                x
            } else {
                0.0
            }
        }
        ActKind::LeakyRelu => {
            if x >= 0.0 {
                x
            } else {
                x * p0
            }
        }
        ActKind::Sigmoid => sigmoid(x),
        ActKind::Tanh => tanh(x),
        ActKind::Silu => x * sigmoid(x),
        ActKind::Gelu => gelu(x),
        ActKind::Clamp => {
            if x.is_nan() {
                x
            } else if x < p0 {
                p0
            } else if x > p1 {
                p1
            } else {
                x
            }
        }
        ActKind::Exp => exp(x),
    }
}

fn f(a: &[Arg], i: usize) -> Option<f32> {
    match a.get(i)? {
        Arg::F32(v) => Some(*v),
        _ => None,
    }
}

pub(super) fn run(mem: &mut Mem, src: Format, dst: Format, act: ActKind, a: &[Arg]) -> Option<()> {
    let len = u(a, 2)?;
    let (p0, p1) = (f(a, 3)?, f(a, 4)?);
    let x = codec::decode_f32(src, &get(mem, a, 0)?, len)?;
    let y: Vec<f32> = x.iter().map(|&v| canon(activate(act, v, p0, p1))).collect();
    put(mem, a, 1, codec::encode_f32(dst, &y, Overflow::Special)?)
}
