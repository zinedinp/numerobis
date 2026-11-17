//! The real Emulated kernels (`kernels/src`: host build, plus every hardware
//! device of the enabled backend features) must match the mock oracle bit for
//! bit on random inputs, for every format and op. F32 outputs compare NaN ==
//! NaN (FP16 decode NaN payloads are host-dependent in miraculix).

mod common;

use common::*;
use numerobis::elem::*;
use numerobis::{Choice, Device, DeviceBuffer, GemmShape, Overflow, Stream, ops};

/// Runs `f` on both devices and compares the downloaded results.
fn agree<E: Elem>(what: &str, f: impl Fn(&mut Stream) -> DeviceBuffer<E>) {
    let run = |dev: Device| {
        let mut s = dev.stream();
        let out = f(&mut s);
        bytes(&mut s, &out)
    };
    let want = run(Device::mock());
    for dev in targets() {
        let on = &dev.features().name;
        let got = run(dev.clone());
        assert_eq!(want.len(), got.len(), "{what} on {on}");
        if E::FORMAT == F32::FORMAT {
            for (i, (w, g)) in want.chunks(4).zip(got.chunks(4)).enumerate() {
                let (w, g) = (f32::from_le_bytes(w.try_into().unwrap()), f32::from_le_bytes(g.try_into().unwrap()));
                assert!(same(w, g), "{what} on {on}: element {i}: mock {w} ({:#x}), kernel {g} ({:#x})", w.to_bits(), g.to_bits());
            }
        } else {
            for (i, (w, g)) in want.iter().zip(&got).enumerate() {
                assert_eq!(w, g, "{what} on {on}: byte {i}");
            }
        }
    }
}

/// Random f32 bit patterns (subnormals, huge, inf), NaNs made canonical,
/// plus a band of ordinary values.
fn wild(n: usize, seed: u64) -> Vec<f32> {
    let mut r = Rng(seed);
    (0..n)
        .map(|i| match i % 3 {
            0 => {
                let v = f32::from_bits(r.next() as u32);
                if v.is_nan() { f32::NAN.copysign(v) } else { v }
            }
            1 => r.f() * 500.0,
            _ => r.f() * 1e-3,
        })
        .collect()
}

fn random_bytes(n: usize, seed: u64) -> Vec<u8> {
    let mut r = Rng(seed);
    (0..n).map(|_| r.next() as u8).collect()
}

macro_rules! convert_both_ways {
    ($($t:ty),*) => {$(
        for ov in [Overflow::Saturate, Overflow::Special] {
            agree(&format!("f32 -> {} {ov:?}", <$t>::NAME), |s| {
                let x = s.upload::<F32>(&wild(4096, 1)).unwrap();
                let mut y = s.alloc::<$t>(4096).unwrap();
                ops::convert(s, Choice::Up, &x, &mut y, ov).unwrap().wait().unwrap();
                y
            });
        }
        agree(&format!("{} -> f32", <$t>::NAME), |s| {
            let n = 4096;
            let raw = random_bytes(<$t>::FORMAT.storage_bytes(n).unwrap(), 2);
            let x = s.upload_packed::<$t>(n, &raw).unwrap();
            let mut y = s.alloc::<F32>(n).unwrap();
            ops::convert(s, Choice::Up, &x, &mut y, Overflow::Saturate).unwrap().wait().unwrap();
            y
        });
    )*};
}

#[test]
fn convert_matches_oracle() {
    convert_both_ways!(F32, Tf32, F16, Bf16, E4M3Fn, E4M3Fnuz, E5M2, E5M2Fnuz, E2M1, E2M3, E3M2);
}

macro_rules! quantize_both_ways {
    ($($t:ty),*) => {$(
        agree(&format!("quantize {}", <$t>::NAME), |s| {
            let x = s.upload::<F32>(&wild(1024, 3)).unwrap();
            let mut y = s.alloc::<$t>(1024).unwrap();
            ops::quantize(s, Choice::Up, &x, &mut y).unwrap().wait().unwrap();
            y
        });
        agree(&format!("dequantize {}", <$t>::NAME), |s| {
            let raw = random_bytes(<$t>::FORMAT.storage_bytes(1024).unwrap(), 4);
            let x = s.upload_packed::<$t>(1024, &raw).unwrap();
            let mut y = s.alloc::<F32>(1024).unwrap();
            ops::dequantize(s, Choice::Up, &x, &mut y).unwrap().wait().unwrap();
            y
        });
    )*};
}

#[test]
fn block_formats_match_oracle() {
    quantize_both_ways!(Mx<E2M1>, Mx<E2M3>, Mx<E3M2>, Mx<E4M3Fn>, Mx<E5M2>, Mx<I8>, Nvfp4, Bfp<M7>, Bfp<M3>, Bfp<M1>);
    for (name, n) in [("q8_0", 1024usize), ("q6_k", 1024)] {
        agree(&format!("dequantize {name}"), |s| {
            let mut y = s.alloc::<F32>(n).unwrap();
            if name == "q8_0" {
                let x = s.upload_packed::<Q8_0>(n, &random_bytes(Q8_0::FORMAT.storage_bytes(n).unwrap(), 5)).unwrap();
                ops::dequantize(s, Choice::Up, &x, &mut y).unwrap().wait().unwrap();
            } else {
                let x = s.upload_packed::<Q6K>(n, &random_bytes(Q6K::FORMAT.storage_bytes(n).unwrap(), 6)).unwrap();
                ops::dequantize(s, Choice::Up, &x, &mut y).unwrap().wait().unwrap();
            }
            y
        });
    }
}

#[test]
fn int_pack_unpack_match_oracle() {
    macro_rules! case {
        ($p:ty, $w:ty) => {
            agree(&format!("pack {}", <$p>::NAME), |s| {
                let x = s.upload_packed::<$w>(512, &random_bytes(512, 7)).unwrap();
                let mut y = s.alloc::<$p>(512).unwrap();
                ops::pack(s, Choice::Up, &x, &mut y).unwrap().wait().unwrap();
                y
            });
            agree(&format!("unpack {}", <$p>::NAME), |s| {
                let raw = random_bytes(<$p>::FORMAT.storage_bytes(512).unwrap(), 8);
                let x = s.upload_packed::<$p>(512, &raw).unwrap();
                let mut y = s.alloc::<$w>(512).unwrap();
                ops::unpack(s, Choice::Up, &x, &mut y).unwrap().wait().unwrap();
                y
            });
        };
    }
    case!(I4, I8);
    case!(U4, U8);
    case!(I2, I8);
    case!(I1, U8);
}

/// Float GEMM inputs from F32 through the matching producer op.
trait Make: Elem + Sized {
    fn make(s: &mut Stream, x: &[f32]) -> DeviceBuffer<Self>;
}
macro_rules! make_float { ($($t:ty),*) => {$(impl Make for $t {
    fn make(s: &mut Stream, x: &[f32]) -> DeviceBuffer<Self> {
        let xb = s.upload::<F32>(x).unwrap();
        let mut y = s.alloc(x.len()).unwrap();
        ops::convert(s, Choice::Up, &xb, &mut y, Overflow::Saturate).unwrap().wait().unwrap();
        y
    }
})*}}
macro_rules! make_block { ($($t:ty),*) => {$(impl Make for $t {
    fn make(s: &mut Stream, x: &[f32]) -> DeviceBuffer<Self> {
        let xb = s.upload::<F32>(x).unwrap();
        let mut y = s.alloc(x.len()).unwrap();
        ops::quantize(s, Choice::Up, &xb, &mut y).unwrap().wait().unwrap();
        y
    }
})*}}
make_float!(F32, Tf32, F16, Bf16, E4M3Fn, E5M2, E2M3);
make_block!(Mx<E2M1>, Mx<I8>, Nvfp4, Bfp<M3>);

fn gemm_case<A: Make, C: Acc<Scalar = f32> + FloatElem>(shape: GemmShape) {
    let GemmShape { m, n, k, batch } = shape;
    agree(&format!("gemm {} -> {}", A::NAME, C::NAME), |s| {
        let mut r = Rng(9);
        let a = A::make(s, &r.vec(batch * m * k, 3.0));
        let b = A::make(s, &r.vec(batch * n * k, 3.0));
        let c0 = r.vec(batch * m * n, 1.0);
        let c0 = s.upload::<F32>(&c0).unwrap();
        let mut c = s.alloc::<C>(batch * m * n).unwrap();
        ops::convert(s, Choice::Up, &c0, &mut c, Overflow::Saturate).unwrap().wait().unwrap();
        ops::gemm(s, Choice::Up, shape, 0.75, &a, &b, -1.25, &mut c).unwrap().wait().unwrap();
        c
    });
}

#[test]
fn gemm_matches_oracle_bit_exact() {
    let sh = GemmShape::new(5, 3, 64).batched(2);
    gemm_case::<F32, F32>(sh);
    gemm_case::<Tf32, F32>(sh);
    gemm_case::<F16, F32>(sh);
    gemm_case::<Bf16, F16>(sh);
    gemm_case::<E4M3Fn, F32>(sh);
    gemm_case::<E5M2, F16>(sh);
    gemm_case::<E2M3, F32>(sh);
    gemm_case::<Mx<E2M1>, F32>(sh);
    gemm_case::<Mx<I8>, F32>(sh);
    gemm_case::<Nvfp4, F16>(sh);
    gemm_case::<Bfp<M3>, F32>(sh);
}

#[test]
fn absmax_matches_oracle() {
    agree("absmax_block f32", |s| {
        let x = s.upload::<F32>(&wild(1024, 10)).unwrap();
        let mut y = s.alloc::<F32>(32).unwrap();
        ops::absmax_block(s, Choice::Up, &x, 32, &mut y).unwrap().wait().unwrap();
        y
    });
    agree("absmax_block mxfp4", |s| {
        let raw = random_bytes(<Mx<E2M1>>::FORMAT.storage_bytes(1024).unwrap(), 11);
        let x = s.upload_packed::<Mx<E2M1>>(1024, &raw).unwrap();
        let mut y = s.alloc::<F32>(64).unwrap();
        ops::absmax_block(s, Choice::Up, &x, 16, &mut y).unwrap().wait().unwrap();
        y
    });
}
