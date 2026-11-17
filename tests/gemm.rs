//! `ops::gemm`: float formats against an f64 reference with tolerance, integer
//! formats exactly. Every tier `auto_down` lists is run.

mod common;

use common::*;
use numerobis::elem::*;
use numerobis::runtime::OpSig;
use numerobis::{Choice, DeviceBuffer, GemmShape, Overflow, Stream, auto_down, ops};

/// `|got - ref| <= tol * sum|a*b|` per element; B is NxK.
fn check(got: &[f32], a: &[f32], b: &[f32], (m, n, k): (usize, usize, usize), tol: f64) {
    for i in 0..m {
        for j in 0..n {
            let (mut r, mut mag) = (0f64, 0f64);
            for p in 0..k {
                let t = a[i * k + p] as f64 * b[j * k + p] as f64;
                r += t;
                mag += t.abs();
            }
            let g = got[i * n + j] as f64;
            assert!((g - r).abs() <= tol * mag.max(1e-30), "C[{i},{j}] = {g}, want {r} (+-{})", tol * mag);
        }
    }
}

/// How a test input of format `A` is produced from `F32` and read back.
trait Input: Elem + Sized {
    fn make(s: &mut Stream, x: &DeviceBuffer<F32>) -> DeviceBuffer<Self>;
    fn host(s: &mut Stream, b: &DeviceBuffer<Self>) -> Vec<f32>;
}
macro_rules! float_input { ($($t:ty),*) => {$(
    impl Input for $t {
        fn make(s: &mut Stream, x: &DeviceBuffer<F32>) -> DeviceBuffer<Self> {
            let mut y = s.alloc(x.len()).unwrap();
            ops::convert(s, Choice::Up, x, &mut y, Overflow::Saturate).unwrap().wait().unwrap();
            y
        }
        fn host(s: &mut Stream, b: &DeviceBuffer<Self>) -> Vec<f32> { float_to_host(s, b) }
    }
)*}}
macro_rules! block_input { ($($t:ty),*) => {$(
    impl Input for $t {
        fn make(s: &mut Stream, x: &DeviceBuffer<F32>) -> DeviceBuffer<Self> {
            let mut y = s.alloc(x.len()).unwrap();
            ops::quantize(s, Choice::Up, x, &mut y).unwrap().wait().unwrap();
            y
        }
        fn host(s: &mut Stream, b: &DeviceBuffer<Self>) -> Vec<f32> { block_to_host(s, b) }
    }
)*}}
float_input!(F32, Tf32, F16, Bf16, E4M3Fn, E5M2Fnuz, E2M1);
block_input!(Mx<E2M1>, Mx<E4M3Fn>, Nvfp4, Bfp<M3>);

fn case<A: Input>(s: &mut Stream, m: usize, n: usize, k: usize) {
    let eps = 64.0 * f32::EPSILON as f64;
    let mut rng = Rng(42 + k as u64);
    let (ah, bh) = (rng.vec(m * k, 4.0), rng.vec(n * k, 4.0));
    let (af, bf) = (s.upload::<F32>(&ah).unwrap(), s.upload::<F32>(&bh).unwrap());
    let (a, b) = (A::make(s, &af), A::make(s, &bf));
    // The reference sees exactly the quantized inputs.
    let (aq, bq) = (A::host(s, &a), A::host(s, &b));
    let mut c = s.alloc::<F32>(m * n).unwrap();
    let sig = OpSig::Gemm { a: A::FORMAT, b: A::FORMAT, acc: F32::FORMAT };
    let tiers = auto_down::tiers(s.device(), sig, false);
    assert!(!tiers.is_empty(), "{}", A::NAME);
    for tier in tiers {
        ops::gemm(s, Choice::Down(tier), GemmShape::new(m, n, k), 1.0, &a, &b, 0.0, &mut c).unwrap().wait().unwrap();
        check(&download(s, &c), &aq, &bq, (m, n, k), eps);
    }
    // F16 accumulator output: one more rounding (2^-11 relative).
    let mut c16 = s.alloc::<F16>(m * n).unwrap();
    ops::gemm(s, Choice::Up, GemmShape::new(m, n, k), 1.0, &a, &b, 0.0, &mut c16).unwrap().wait().unwrap();
    check(&float_to_host(s, &c16), &aq, &bq, (m, n, k), eps + 1.0 / 2048.0);
}

#[test]
fn float_formats_vs_f64() {
    for dev in devices() {
        let mut s = dev.stream();
        case::<F32>(&mut s, 5, 7, 64);
        case::<Tf32>(&mut s, 5, 7, 64);
        case::<F16>(&mut s, 3, 4, 96);
        case::<Bf16>(&mut s, 3, 4, 96);
        case::<E4M3Fn>(&mut s, 4, 4, 64);
        case::<E5M2Fnuz>(&mut s, 4, 4, 64);
        case::<E2M1>(&mut s, 4, 4, 64);
        case::<Mx<E2M1>>(&mut s, 4, 4, 64);
        case::<Mx<E4M3Fn>>(&mut s, 4, 4, 64);
        case::<Nvfp4>(&mut s, 4, 4, 64);
        case::<Bfp<M3>>(&mut s, 4, 4, 64);
    }
}

#[test]
fn int_exact_alpha_beta_batched() {
    for dev in devices() {
        let mut s = dev.stream();
        let (m, n, k, batch) = (3, 5, 16, 2);
        let mut rng = Rng(1);
        let a: Vec<i8> = (0..batch * m * k).map(|_| (rng.next() % 15) as i8 - 7).collect();
        let b: Vec<i8> = (0..batch * n * k).map(|_| (rng.next() % 15) as i8 - 7).collect();
        let c0: Vec<i32> = (0..batch * m * n).map(|i| i as i32 - 10).collect();
        let (ad, bd) = (s.upload::<I8>(&a).unwrap(), s.upload::<I8>(&b).unwrap());
        let mut c = s.upload::<I32>(&c0).unwrap();
        let sh = GemmShape::new(m, n, k).batched(batch);
        ops::gemm(&mut s, Choice::Up, sh, 3, &ad, &bd, -2, &mut c).unwrap().wait().unwrap();
        let mut want = vec![0i32; batch * m * n];
        for z in 0..batch {
            for i in 0..m {
                for j in 0..n {
                    let dot: i32 = (0..k).map(|p| a[z * m * k + i * k + p] as i32 * b[z * n * k + j * k + p] as i32).sum();
                    want[z * m * n + i * n + j] = 3 * dot - 2 * c0[z * m * n + i * n + j];
                }
            }
        }
        assert_eq!(download(&mut s, &c), want);

        // Same through packed INT4 (all values fit in [-8, 7]).
        let mut a4 = s.alloc::<I4>(a.len()).unwrap();
        let mut b4 = s.alloc::<I4>(b.len()).unwrap();
        ops::pack(&mut s, Choice::Up, &ad, &mut a4).unwrap().wait().unwrap();
        ops::pack(&mut s, Choice::Up, &bd, &mut b4).unwrap().wait().unwrap();
        let mut c = s.upload::<I32>(&c0).unwrap();
        ops::gemm(&mut s, Choice::Up, sh, 3, &a4, &b4, -2, &mut c).unwrap().wait().unwrap();
        assert_eq!(download(&mut s, &c), want);
    }
}

#[test]
fn beta_zero_ignores_nan_c() {
    for dev in devices() {
        let mut s = dev.stream();
        let a = s.upload::<F32>(&[2.0]).unwrap();
        let mut c = s.upload::<F32>(&[f32::NAN]).unwrap();
        ops::gemm(&mut s, Choice::Up, GemmShape::new(1, 1, 1), 1.5, &a, &a, 0.0, &mut c).unwrap().wait().unwrap();
        assert_eq!(download(&mut s, &c), [6.0]);
        ops::gemm(&mut s, Choice::Up, GemmShape::new(1, 1, 1), 1.0, &a, &a, 0.5, &mut c).unwrap().wait().unwrap();
        assert_eq!(download(&mut s, &c), [7.0]);
    }
}
