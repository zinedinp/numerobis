//! `ops::quantize` / `ops::dequantize` against miraculix, bit-exact.

#![allow(
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap,
    clippy::cast_precision_loss,
    clippy::cast_sign_loss,
    clippy::many_single_char_names,
    reason = "test data is built with bit-level casts; shapes are named as in the math"
)]

mod common;

use common::*;
use miraculix::formats::bfp::{BfpElem, bfp_encode_rows};
use miraculix::formats::gguf::{Q8_0_BYTES, dequantize_q8_0};
use miraculix::formats::mx::{
    MxElem, mx_dequantize, mx_quantize, nvfp4_quantize, nvfp4_tensor_scale,
};
use numerobis::elem::*;
use numerobis::{Choice, ops};

#[test]
fn mxfp4() {
    for dev in devices() {
        let mut s = dev.stream();
        let x = Rng(5).vec(128, 30.0);
        let xb = s.upload::<F32>(&x).unwrap();
        let mut q = s.alloc::<Mx<E2M1>>(128).unwrap();
        ops::quantize(&mut s, Choice::Up, &xb, &mut q)
            .unwrap()
            .wait()
            .unwrap();
        let got = bytes(&mut s, &q);
        let (mut sc, mut d) = (vec![0u8; 4], vec![0u8; 64]);
        mx_quantize(MxElem::Fp4, &x, &mut sc, &mut d);
        assert_eq!(got, [d, sc].concat());
        let mut want = vec![0.0; 128];
        mx_dequantize(MxElem::Fp4, &got[64..], &got[..64], &mut want);
        assert_eq!(block_to_host(&mut s, &q), want);
    }
}

#[test]
fn nvfp4() {
    for dev in devices() {
        let mut s = dev.stream();
        let x = Rng(6).vec(128, 30.0);
        let xb = s.upload::<F32>(&x).unwrap();
        let mut q = s.alloc::<Nvfp4>(128).unwrap();
        ops::quantize(&mut s, Choice::Up, &xb, &mut q)
            .unwrap()
            .wait()
            .unwrap();
        let ts = nvfp4_tensor_scale(&x);
        let (mut sc, mut d) = (vec![0u8; 8], vec![0u8; 64]);
        nvfp4_quantize(&x, ts, &mut sc, &mut d);
        assert_eq!(
            bytes(&mut s, &q),
            [d, sc, ts.to_le_bytes().to_vec()].concat()
        );
    }
}

#[test]
fn bfp8_b() {
    for dev in devices() {
        let mut s = dev.stream();
        let x = Rng(8).vec(128, 30.0);
        let xb = s.upload::<F32>(&x).unwrap();
        let mut q = s.alloc::<Bfp<M7>>(128).unwrap();
        ops::quantize(&mut s, Choice::Up, &xb, &mut q)
            .unwrap()
            .wait()
            .unwrap();
        let (mut ex, mut d) = (vec![0u8; 8], vec![0u8; 128]);
        bfp_encode_rows(BfpElem::M7, &x, &mut ex, &mut d);
        assert_eq!(bytes(&mut s, &q), [d, ex].concat());
    }
}

#[test]
fn gguf_q8_0_decode() {
    for dev in devices() {
        let mut s = dev.stream();
        let mut rng = Rng(9);
        let mut blk = Vec::new();
        for _ in 0..2 {
            blk.extend_from_slice(&0x2E66u16.to_le_bytes()); // f16 scale
            blk.extend((0..32).map(|_| rng.next() as u8));
        }
        assert_eq!(blk.len(), 2 * Q8_0_BYTES);
        let q = s.upload_packed::<Q8_0>(64, &blk).unwrap();
        let mut want = vec![0.0; 64];
        dequantize_q8_0(&blk, &mut want);
        assert_eq!(block_to_host(&mut s, &q), want);
    }
}
