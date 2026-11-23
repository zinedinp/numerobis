//! `ops::pack` / `ops::unpack` against miraculix, bit-exact.

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
use miraculix::formats::int::{pack_int, pack_uint};
use numerobis::elem::*;
use numerobis::{Choice, ops};

#[test]
fn i4_roundtrip_saturates() {
    for dev in devices() {
        let mut s = dev.stream();
        let v: Vec<i8> = (-10..10).collect();
        let b = s.upload::<I8>(&v).unwrap();
        let mut p = s.alloc::<I4>(20).unwrap();
        ops::pack(&mut s, Choice::Up, &b, &mut p)
            .unwrap()
            .wait()
            .unwrap();
        let mut want = vec![0u8; 10];
        pack_int::<4>(&v, &mut want);
        assert_eq!(bytes(&mut s, &p), want);
        let mut back = s.alloc::<I8>(20).unwrap();
        ops::unpack(&mut s, Choice::Up, &p, &mut back)
            .unwrap()
            .wait()
            .unwrap();
        assert_eq!(
            download(&mut s, &back),
            v.iter().map(|&x| x.clamp(-8, 7)).collect::<Vec<_>>()
        );
    }
}

#[test]
fn u4_and_i1() {
    for dev in devices() {
        let mut s = dev.stream();
        let v: Vec<u8> = (0..24).map(|i| (i * 3) as u8).collect();
        let b = s.upload::<U8>(&v).unwrap();
        let mut p = s.alloc::<U4>(24).unwrap();
        ops::pack(&mut s, Choice::Up, &b, &mut p)
            .unwrap()
            .wait()
            .unwrap();
        let mut want = vec![0u8; 12];
        pack_uint::<4>(&v, &mut want);
        assert_eq!(bytes(&mut s, &p), want);
        let mut p1 = s.alloc::<I1>(24).unwrap();
        ops::pack(&mut s, Choice::Up, &b, &mut p1)
            .unwrap()
            .wait()
            .unwrap();
        let mut want = vec![0u8; 3];
        pack_uint::<1>(&v, &mut want);
        assert_eq!(bytes(&mut s, &p1), want);
    }
}
