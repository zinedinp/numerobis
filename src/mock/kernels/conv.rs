//! Direct convolution, mirror of `nb_conv2d` (loop order c, r, s; padding
//! taps skipped; optional fused activation).

use miraculix::formats::Overflow;

use super::{Mem, get, put};
use crate::elem::Format;
use crate::mock::codec;
use crate::ops::conv::{P, PARAM_WORDS};
use crate::runtime::backend::{ActKind, Arg};

pub(super) fn run(
    mem: &mut Mem,
    fx: Format,
    fw: Format,
    fy: Format,
    bias: bool,
    a: &[Arg],
) -> Option<()> {
    let pb = get(mem, a, 4)?;
    if pb.len() < PARAM_WORDS * 4 {
        return None;
    }
    let p = |i: P| u32::from_le_bytes(pb[i as usize * 4..][..4].try_into().unwrap()) as usize;
    let (n, c, h, w, k, r, s) = (
        p(P::N),
        p(P::C),
        p(P::H),
        p(P::W),
        p(P::K),
        p(P::R),
        p(P::S),
    );
    let (pp, q, g) = (p(P::P), p(P::Q), p(P::G));
    let (sh, sw, ph, pw, dh, dw) = (p(P::Sh), p(P::Sw), p(P::Ph), p(P::Pw), p(P::Dh), p(P::Dw));
    let cg = c / g;
    let xv = codec::decode_f32(fx, &get(mem, a, 0)?, n * c * h * w)?;
    let wv = codec::decode_f32(fw, &get(mem, a, 1)?, k * cg * r * s)?;
    let bv = if bias {
        codec::decode_f32(Format::F32, &get(mem, a, 2)?, k)?
    } else {
        Vec::new()
    };
    let act = match p(P::Act) {
        0 => None,
        code => Some(act_from_code(code as u32 - 1)?),
    };
    let mut y = vec![0f32; n * k * pp * q];
    for (gi, out) in y.iter_mut().enumerate() {
        let (qq, p_, kk, nn) = (
            gi % q,
            (gi / q) % pp,
            (gi / (pp * q)) % k,
            gi / (k * pp * q),
        );
        let grp = kk / (k / g);
        let mut acc = if bias { bv[kk] } else { 0.0 };
        for cc in 0..cg {
            let ci = grp * cg + cc;
            for rr in 0..r {
                let ih = (p_ * sh + rr * dh) as i64 - ph as i64;
                if ih < 0 || ih >= h as i64 {
                    continue;
                }
                for ss in 0..s {
                    let iw = (qq * sw + ss * dw) as i64 - pw as i64;
                    if iw < 0 || iw >= w as i64 {
                        continue;
                    }
                    let x = xv[((nn * c + ci) * h + ih as usize) * w + iw as usize];
                    let wt = wv[((kk * cg + cc) * r + rr) * s + ss];
                    acc = x.mul_add(wt, acc);
                }
            }
        }
        if let Some(act) = act {
            acc = super::act::activate(act, acc, 0.0, 0.0);
        }
        *out = super::act::canon(acc);
    }
    put(mem, a, 3, codec::encode_f32(fy, &y, Overflow::Special)?)
}

fn act_from_code(c: u32) -> Option<ActKind> {
    use ActKind::{Clamp, Exp, Gelu, LeakyRelu, Relu, Sigmoid, Silu, Tanh};
    [Relu, LeakyRelu, Sigmoid, Tanh, Silu, Gelu, Clamp, Exp]
        .into_iter()
        .find(|a| a.code() == c)
}
