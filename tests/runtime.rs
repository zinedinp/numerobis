//! Runtime behaviour: argument checks, support errors, buffer lifetime, reduce.

mod common;

use common::*;
use numerobis::elem::*;
use numerobis::{Choice, Device, Error, GemmShape, Overflow, Tier, nvidia, ops};

#[test]
fn absmax_block_matches_host() {
    for dev in devices() {
        let mut s = dev.stream();
        let mut x = Rng(2).vec(96, 5.0);
        x[3] = f32::NAN;
        let xb = s.upload::<F32>(&x).unwrap();
        let mut m = s.alloc::<F32>(3).unwrap();
        ops::absmax_block(&mut s, Choice::Up, &xb, 32, &mut m).unwrap().wait().unwrap();
        for (g, c) in download(&mut s, &m).iter().zip(x.chunks(32)) {
            assert_eq!(*g, c.iter().filter(|v| !v.is_nan()).fold(0.0f32, |a, v| a.max(v.abs())));
        }
    }
}

#[test]
fn argument_and_support_errors() {
    for dev in devices() {
        let mut s = dev.stream();
        assert!(matches!(s.alloc::<Mx<E2M1>>(33), Err(Error::InvalidArgument(_))));
        assert!(matches!(s.upload_packed::<E2M1>(4, &[0u8; 3]), Err(Error::InvalidArgument(_))));
        let a = s.upload::<F32>(&[1.0; 4]).unwrap();
        let mut c = s.alloc::<F32>(3).unwrap();
        assert!(matches!(
            ops::gemm(&mut s, Choice::Up, GemmShape::new(2, 2, 1), 1.0, &a, &a, 0.0, &mut c),
            Err(Error::InvalidArgument(_))
        ));
        // A tier the device does not have.
        let x = s.upload::<F32>(&[0.0; 32]).unwrap();
        let mut y = s.alloc::<F16>(32).unwrap();
        let r = ops::convert(&mut s, Choice::Down(Tier::Nvidia(nvidia::Tier::Sm80)), &x, &mut y, Overflow::Saturate);
        assert!(matches!(r, Err(Error::Unsupported(_))));
        // Buffers of another device are rejected.
        let other = Device::mock();
        let mut s2 = other.stream();
        assert!(matches!(ops::convert(&mut s2, Choice::Up, &x, &mut y, Overflow::Saturate), Err(Error::InvalidArgument(_))));
    }
}

#[test]
fn buffers_outlive_pending_work() {
    for dev in devices() {
        let mut s = dev.stream();
        let x = s.upload::<F32>(&[1.0, 2.0]).unwrap();
        let mut y = s.alloc::<Bf16>(2).unwrap();
        let p = ops::convert(&mut s, Choice::Up, &x, &mut y, Overflow::Saturate).unwrap();
        let d = s.download(&y).unwrap();
        drop(x);
        drop(y);
        p.wait().unwrap();
        assert_eq!(d.wait().unwrap().to_vec(), [0x3F80, 0x4000]);
        s.synchronize().unwrap();
    }
}
