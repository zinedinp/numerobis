//! `ops::convert` against the miraculix MiniFloat codec, bit-exact.

mod common;

use common::*;
use miraculix::formats::{MiniFloat, round_to_tf32};
use numerobis::elem::*;
use numerobis::{Choice, Overflow, ops};

macro_rules! fp8_case {
    ($dev:expr, $t:ty, $f:expr) => {{
        let mut s = $dev.stream();
        let codes: Vec<u8> = (0..=255).collect();
        let b = s.upload::<$t>(&codes).unwrap();
        for (c, g) in codes.iter().zip(float_to_host(&mut s, &b)) {
            assert!(same(g, $f.decode(*c as u32)), "{:?} decode {c:#x}", $f);
        }
        let x = edge_values(&mut Rng(7));
        let xb = s.upload::<F32>(&x).unwrap();
        for ov in [Overflow::Saturate, Overflow::Special] {
            let mut y = s.alloc::<$t>(x.len()).unwrap();
            ops::convert(&mut s, Choice::Up, &xb, &mut y, ov)
                .unwrap()
                .wait()
                .unwrap();
            for (v, g) in x.iter().zip(download(&mut s, &y)) {
                assert_eq!(g as u32, $f.encode(*v, ov), "{:?} encode {v} {ov:?}", $f);
            }
        }
    }};
}

#[test]
fn fp8_every_code_and_encode() {
    for dev in devices() {
        fp8_case!(dev, E4M3Fn, MiniFloat::E4M3Fn);
        fp8_case!(dev, E4M3Fnuz, MiniFloat::E4M3Fnuz);
        fp8_case!(dev, E5M2, MiniFloat::E5M2);
        fp8_case!(dev, E5M2Fnuz, MiniFloat::E5M2Fnuz);
    }
}

#[test]
fn half_and_tf32() {
    for dev in devices() {
        let mut s = dev.stream();
        let x = edge_values(&mut Rng(11));
        let xb = s.upload::<F32>(&x).unwrap();
        let mut want = vec![0u16; x.len()];

        MiniFloat::F16.encode_u16_slice(&x, &mut want, Overflow::Special);
        let mut y = s.alloc::<F16>(x.len()).unwrap();
        ops::convert(&mut s, Choice::Up, &xb, &mut y, Overflow::Special)
            .unwrap()
            .wait()
            .unwrap();
        assert_eq!(download(&mut s, &y), want);

        MiniFloat::Bf16.encode_u16_slice(&x, &mut want, Overflow::Special);
        let mut y = s.alloc::<Bf16>(x.len()).unwrap();
        ops::convert(&mut s, Choice::Up, &xb, &mut y, Overflow::Special)
            .unwrap()
            .wait()
            .unwrap();
        assert_eq!(download(&mut s, &y), want);

        let mut t = s.alloc::<Tf32>(x.len()).unwrap();
        ops::convert(&mut s, Choice::Up, &xb, &mut t, Overflow::Special)
            .unwrap()
            .wait()
            .unwrap();
        for (v, g) in x.iter().zip(download(&mut s, &t)) {
            assert!(same(g, round_to_tf32(*v)));
        }
    }
}

#[test]
fn packed_fp4_fp6() {
    for dev in devices() {
        let mut s = dev.stream();
        let x = Rng(3).vec(64, 10.0);
        let xb = s.upload::<F32>(&x).unwrap();
        let mut y = s.alloc::<E2M1>(64).unwrap();
        ops::convert(&mut s, Choice::Up, &xb, &mut y, Overflow::Saturate)
            .unwrap()
            .wait()
            .unwrap();
        let b = bytes(&mut s, &y);
        assert_eq!(b.len(), 32);
        for (i, v) in x.iter().enumerate() {
            let code = (b[i / 2] >> (4 * (i % 2))) & 0xF;
            assert_eq!(code as u32, MiniFloat::E2M1.encode(*v, Overflow::Saturate));
        }
        let mut y6 = s.alloc::<E3M2>(64).unwrap();
        ops::convert(&mut s, Choice::Up, &xb, &mut y6, Overflow::Saturate)
            .unwrap()
            .wait()
            .unwrap();
        let f = MiniFloat::E3M2;
        for (v, g) in x.iter().zip(float_to_host(&mut s, &y6)) {
            assert!(same(g, f.decode(f.encode(*v, Overflow::Saturate))));
        }
    }
}

#[test]
fn minifloat_to_minifloat_is_not_stage_1() {
    for dev in devices() {
        let mut s = dev.stream();
        let a = s.alloc::<F16>(4).unwrap();
        let mut b = s.alloc::<Bf16>(4).unwrap();
        let r = ops::convert(&mut s, Choice::Up, &a, &mut b, Overflow::Saturate);
        assert!(matches!(r, Err(numerobis::Error::Unsupported(_))));
    }
}
