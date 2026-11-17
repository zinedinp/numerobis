//! Oracle tests (plan B.5): every op on the mock device against the miraculix
//! codecs (bit-exact) or an f64 reference (GEMM, with tolerance).

use miraculix::formats::bfp::{BfpElem, bfp_encode_rows};
use miraculix::formats::gguf::{Q8_0_BYTES, dequantize_q8_0};
use miraculix::formats::int::pack_int;
use miraculix::formats::mx::{MxElem, mx_quantize, nvfp4_quantize, nvfp4_tensor_scale};
use miraculix::formats::{MiniFloat, round_to_tf32};
use numerobis::backend::OpSig;
use numerobis::elem::*;
use numerobis::{Choice, Device, Error, GemmShape, Overflow, Stream, Tier, auto_down, ops};

struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }
    fn f(&mut self) -> f32 {
        (self.next() >> 40) as f32 / (1u64 << 24) as f32 * 2.0 - 1.0
    }
    fn vec(&mut self, n: usize, scale: f32) -> Vec<f32> {
        (0..n).map(|_| self.f() * scale).collect()
    }
}

fn same(a: f32, b: f32) -> bool {
    a.to_bits() == b.to_bits() || (a.is_nan() && b.is_nan())
}

fn to_f32<E: Elem>(s: &mut Stream, b: &numerobis::DeviceBuffer<E>) -> Vec<f32> {
    let mut o = s.alloc::<F32>(b.len()).unwrap();
    ops::convert(s, Choice::Up, b, &mut o, Overflow::Saturate).unwrap().wait().unwrap();
    s.download(&o).unwrap().wait().unwrap().to_vec()
}

fn edge_values(rng: &mut Rng) -> Vec<f32> {
    let mut v = vec![0.0, -0.0, 1.0, -1.0, 0.5, 448.0, 449.0, 480.0, 1e6, -1e6, 6.0, 7.0, 1e-9, f32::INFINITY, f32::NEG_INFINITY, f32::NAN];
    v.extend(rng.vec(240, 600.0));
    v.extend(rng.vec(256, 0.01));
    v
}

macro_rules! fp8_case {
    ($t:ty, $f:expr) => {{
        let dev = Device::mock();
        let mut s = dev.stream();
        let codes: Vec<u8> = (0..=255).collect();
        let b = s.upload::<$t>(&codes).unwrap();
        let got = to_f32(&mut s, &b);
        for (c, g) in codes.iter().zip(&got) {
            assert!(same(*g, $f.decode(*c as u32)), "{:?} decode {c:#x}", $f);
        }
        let mut rng = Rng(7);
        let x = edge_values(&mut rng);
        let xb = s.upload::<F32>(&x).unwrap();
        for ov in [Overflow::Saturate, Overflow::Special] {
            let mut y = s.alloc::<$t>(x.len()).unwrap();
            ops::convert(&mut s, Choice::Up, &xb, &mut y, ov).unwrap().wait().unwrap();
            let got = s.download(&y).unwrap().wait().unwrap().to_vec();
            for (v, g) in x.iter().zip(got) {
                assert_eq!(g as u32, $f.encode(*v, ov), "{:?} encode {v} {ov:?}", $f);
            }
        }
    }};
}

#[test]
fn convert_fp8_bit_exact() {
    fp8_case!(E4M3Fn, MiniFloat::E4M3Fn);
    fp8_case!(E4M3Fnuz, MiniFloat::E4M3Fnuz);
    fp8_case!(E5M2, MiniFloat::E5M2);
    fp8_case!(E5M2Fnuz, MiniFloat::E5M2Fnuz);
}

#[test]
fn convert_half_and_tf32_bit_exact() {
    let dev = Device::mock();
    let mut s = dev.stream();
    let mut rng = Rng(11);
    let x = edge_values(&mut rng);
    let xb = s.upload::<F32>(&x).unwrap();
    for (f, is_bf) in [(MiniFloat::F16, false), (MiniFloat::Bf16, true)] {
        let mut y16 = vec![0u16; x.len()];
        f.encode_u16_slice(&x, &mut y16, Overflow::Special);
        let got = if is_bf {
            let mut y = s.alloc::<Bf16>(x.len()).unwrap();
            ops::convert(&mut s, Choice::Up, &xb, &mut y, Overflow::Special).unwrap().wait().unwrap();
            s.download(&y).unwrap().wait().unwrap().to_vec()
        } else {
            let mut y = s.alloc::<F16>(x.len()).unwrap();
            ops::convert(&mut s, Choice::Up, &xb, &mut y, Overflow::Special).unwrap().wait().unwrap();
            s.download(&y).unwrap().wait().unwrap().to_vec()
        };
        assert_eq!(got, y16, "{f:?}");
    }
    let mut t = s.alloc::<Tf32>(x.len()).unwrap();
    ops::convert(&mut s, Choice::Up, &xb, &mut t, Overflow::Special).unwrap().wait().unwrap();
    let got = s.download(&t).unwrap().wait().unwrap().to_vec();
    for (v, g) in x.iter().zip(got) {
        assert!(same(g, round_to_tf32(*v)));
    }
}

#[test]
fn convert_packed_fp4_fp6() {
    let dev = Device::mock();
    let mut s = dev.stream();
    let mut rng = Rng(3);
    let x = rng.vec(64, 10.0);
    let xb = s.upload::<F32>(&x).unwrap();
    let mut y = s.alloc::<E2M1>(64).unwrap();
    ops::convert(&mut s, Choice::Up, &xb, &mut y, Overflow::Saturate).unwrap().wait().unwrap();
    let bytes = s.download(&y).unwrap().wait().unwrap().into_bytes();
    assert_eq!(bytes.len(), 32);
    for (i, v) in x.iter().enumerate() {
        let code = (bytes[i / 2] >> (4 * (i % 2))) & 0xF;
        assert_eq!(code as u32, MiniFloat::E2M1.encode(*v, Overflow::Saturate));
    }
    // FP6 roundtrip: decode(encode(x)) through the device equals the scalar codec.
    let mut y6 = s.alloc::<E3M2>(64).unwrap();
    ops::convert(&mut s, Choice::Up, &xb, &mut y6, Overflow::Saturate).unwrap().wait().unwrap();
    let back = to_f32(&mut s, &y6);
    for (v, g) in x.iter().zip(back) {
        let f = MiniFloat::E3M2;
        assert!(same(g, f.decode(f.encode(*v, Overflow::Saturate))));
    }
}

#[test]
fn block_quantizers_bit_exact() {
    let dev = Device::mock();
    let mut s = dev.stream();
    let mut rng = Rng(5);
    let x = rng.vec(128, 30.0);
    let xb = s.upload::<F32>(&x).unwrap();

    // MXFP4
    let mut q = s.alloc::<Mx<E2M1>>(128).unwrap();
    ops::convert(&mut s, Choice::Up, &xb, &mut q, Overflow::Saturate).unwrap().wait().unwrap();
    let got = s.download(&q).unwrap().wait().unwrap().into_bytes();
    let (mut sc, mut d) = (vec![0u8; 4], vec![0u8; 64]);
    mx_quantize(MxElem::Fp4, &x, &mut sc, &mut d);
    assert_eq!(got, [d, sc].concat());
    let back = to_f32(&mut s, &q);
    let mut want = vec![0.0; 128];
    miraculix::formats::mx::mx_dequantize(MxElem::Fp4, &got[64..], &got[..64], &mut want);
    assert_eq!(back, want);

    // NVFP4
    let mut q = s.alloc::<Nvfp4>(128).unwrap();
    ops::convert(&mut s, Choice::Up, &xb, &mut q, Overflow::Saturate).unwrap().wait().unwrap();
    let got = s.download(&q).unwrap().wait().unwrap().into_bytes();
    let ts = nvfp4_tensor_scale(&x);
    let (mut sc, mut d) = (vec![0u8; 8], vec![0u8; 64]);
    nvfp4_quantize(&x, ts, &mut sc, &mut d);
    assert_eq!(got, [d, sc, ts.to_le_bytes().to_vec()].concat());

    // BFP8_b
    let mut q = s.alloc::<Bfp<M7>>(128).unwrap();
    ops::convert(&mut s, Choice::Up, &xb, &mut q, Overflow::Saturate).unwrap().wait().unwrap();
    let got = s.download(&q).unwrap().wait().unwrap().into_bytes();
    let (mut ex, mut d) = (vec![0u8; 8], vec![0u8; 128]);
    bfp_encode_rows(BfpElem::M7, &x, &mut ex, &mut d);
    assert_eq!(got, [d, ex].concat());
}

#[test]
fn int_pack_unpack_and_gguf() {
    let dev = Device::mock();
    let mut s = dev.stream();
    let v: Vec<i8> = (-10..10).collect();
    let b = s.upload::<I8>(&v).unwrap();
    let mut p = s.alloc::<I4>(20).unwrap();
    ops::convert(&mut s, Choice::Up, &b, &mut p, Overflow::Saturate).unwrap().wait().unwrap();
    let got = s.download(&p).unwrap().wait().unwrap().into_bytes();
    let mut want = vec![0u8; 10];
    pack_int::<4>(&v, &mut want);
    assert_eq!(got, want);
    let mut back = s.alloc::<I8>(20).unwrap();
    ops::convert(&mut s, Choice::Up, &p, &mut back, Overflow::Saturate).unwrap().wait().unwrap();
    let back = s.download(&back).unwrap().wait().unwrap().to_vec();
    assert_eq!(back, v.iter().map(|&x| x.clamp(-8, 7)).collect::<Vec<_>>());

    // Q8_0: two blocks of f16 scale + 32 i8.
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
    assert_eq!(to_f32(&mut s, &q), want);
}

/// f64 reference GEMM on the host-decoded inputs; checks
/// `|got - ref| <= tol · Σ|a||b|` per element.
fn check_gemm(got: &[f32], a: &[f32], b: &[f32], m: usize, n: usize, k: usize, tol: f64) {
    for i in 0..m {
        for j in 0..n {
            let (mut r, mut mag) = (0f64, 0f64);
            for p in 0..k {
                let t = a[i * k + p] as f64 * b[j * k + p] as f64;
                r += t;
                mag += t.abs();
            }
            let g = got[i * n + j] as f64;
            assert!((g - r).abs() <= tol * mag.max(1e-30), "C[{i},{j}] = {g}, want {r} (±{})", tol * mag);
        }
    }
}

fn gemm_case<A: Elem>(s: &mut Stream, m: usize, n: usize, k: usize, tol_acc: f64) {
    let mut rng = Rng(42 + k as u64);
    let (ah, bh) = (rng.vec(m * k, 4.0), rng.vec(n * k, 4.0));
    let (af, bf) = (s.upload::<F32>(&ah).unwrap(), s.upload::<F32>(&bh).unwrap());
    let mut a = s.alloc::<A>(m * k).unwrap();
    let mut b = s.alloc::<A>(n * k).unwrap();
    ops::convert(s, Choice::Up, &af, &mut a, Overflow::Saturate).unwrap().wait().unwrap();
    ops::convert(s, Choice::Up, &bf, &mut b, Overflow::Saturate).unwrap().wait().unwrap();
    // Reference sees exactly the quantized inputs.
    let (aq, bq) = (to_f32(s, &a), to_f32(s, &b));
    let mut c = s.alloc::<F32>(m * n).unwrap();
    for tier in auto_down::tiers(s.device(), OpSig::Gemm { a: A::FORMAT, b: A::FORMAT, acc: F32::FORMAT }, false) {
        ops::gemm(s, Choice::Down(tier), GemmShape::new(m, n, k), 1.0, &a, &b, 0.0, &mut c).unwrap().wait().unwrap();
        let got = s.download(&c).unwrap().wait().unwrap().to_vec();
        check_gemm(&got, &aq, &bq, m, n, k, tol_acc);
    }
    // F16 accumulator output: one more rounding (2^-11 relative).
    let mut c16 = s.alloc::<F16>(m * n).unwrap();
    ops::gemm(s, Choice::Up, GemmShape::new(m, n, k), 1.0, &a, &b, 0.0, &mut c16).unwrap().wait().unwrap();
    check_gemm(&to_f32(s, &c16), &aq, &bq, m, n, k, tol_acc + 1.0 / 2048.0);
}

#[test]
fn gemm_float_formats_vs_f64() {
    let dev = Device::mock();
    let mut s = dev.stream();
    let eps = 64.0 * f32::EPSILON as f64;
    gemm_case::<F32>(&mut s, 5, 7, 64, eps);
    gemm_case::<Tf32>(&mut s, 5, 7, 64, eps);
    gemm_case::<F16>(&mut s, 3, 4, 96, eps);
    gemm_case::<Bf16>(&mut s, 3, 4, 96, eps);
    gemm_case::<E4M3Fn>(&mut s, 4, 4, 64, eps);
    gemm_case::<E5M2Fnuz>(&mut s, 4, 4, 64, eps);
    gemm_case::<E2M1>(&mut s, 4, 4, 64, eps);
    gemm_case::<Mx<E2M1>>(&mut s, 4, 4, 64, eps);
    gemm_case::<Mx<E4M3Fn>>(&mut s, 4, 4, 64, eps);
    gemm_case::<Nvfp4>(&mut s, 4, 4, 64, eps);
    gemm_case::<Bfp<M3>>(&mut s, 4, 4, 64, eps);
}

#[test]
fn gemm_int_exact_alpha_beta_batched() {
    let dev = Device::mock();
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
    let got = s.download(&c).unwrap().wait().unwrap().to_vec();
    let mut want = vec![0i32; batch * m * n];
    for z in 0..batch {
        for i in 0..m {
            for j in 0..n {
                let dot: i32 = (0..k).map(|p| a[z * m * k + i * k + p] as i32 * b[z * n * k + j * k + p] as i32).sum();
                want[z * m * n + i * n + j] = 3 * dot - 2 * c0[z * m * n + i * n + j];
            }
        }
    }
    assert_eq!(got, want);

    // Same through packed INT4.
    let mut a4 = s.alloc::<I4>(a.len()).unwrap();
    let mut b4 = s.alloc::<I4>(b.len()).unwrap();
    ops::convert(&mut s, Choice::Up, &ad, &mut a4, Overflow::Saturate).unwrap().wait().unwrap();
    ops::convert(&mut s, Choice::Up, &bd, &mut b4, Overflow::Saturate).unwrap().wait().unwrap();
    let mut c = s.upload::<I32>(&c0).unwrap();
    ops::gemm(&mut s, Choice::Up, sh, 3, &a4, &b4, -2, &mut c).unwrap().wait().unwrap();
    assert_eq!(s.download(&c).unwrap().wait().unwrap().to_vec(), want);
}

#[test]
fn gemm_beta_zero_ignores_nan_c() {
    let dev = Device::mock();
    let mut s = dev.stream();
    let a = s.upload::<F32>(&[2.0]).unwrap();
    let mut c = s.upload::<F32>(&[f32::NAN]).unwrap();
    ops::gemm(&mut s, Choice::Up, GemmShape::new(1, 1, 1), 1.5, &a, &a, 0.0, &mut c).unwrap().wait().unwrap();
    assert_eq!(s.download(&c).unwrap().wait().unwrap().to_vec(), [6.0]);
    ops::gemm(&mut s, Choice::Up, GemmShape::new(1, 1, 1), 1.0, &a, &a, 0.5, &mut c).unwrap().wait().unwrap();
    assert_eq!(s.download(&c).unwrap().wait().unwrap().to_vec(), [7.0]);
}

#[test]
fn absmax_block_matches_host() {
    let dev = Device::mock();
    let mut s = dev.stream();
    let mut x = Rng(2).vec(96, 5.0);
    x[3] = f32::NAN;
    let xb = s.upload::<F32>(&x).unwrap();
    let mut m = s.alloc::<F32>(3).unwrap();
    ops::absmax_block(&mut s, Choice::Up, &xb, 32, &mut m).unwrap().wait().unwrap();
    let got = s.download(&m).unwrap().wait().unwrap().to_vec();
    for (g, c) in got.iter().zip(x.chunks(32)) {
        assert_eq!(*g, c.iter().filter(|v| !v.is_nan()).fold(0.0f32, |a, v| a.max(v.abs())));
    }
}

#[test]
fn argument_and_support_errors() {
    let dev = Device::mock();
    let mut s = dev.stream();
    assert!(matches!(s.alloc::<Mx<E2M1>>(33), Err(Error::InvalidArgument(_))));
    assert!(matches!(s.upload_packed::<E2M1>(4, &[0u8; 3]), Err(Error::InvalidArgument(_))));
    let a = s.upload::<F32>(&[1.0; 4]).unwrap();
    let mut c = s.alloc::<F32>(3).unwrap();
    assert!(matches!(
        ops::gemm(&mut s, Choice::Up, GemmShape::new(2, 2, 1), 1.0, &a, &a, 0.0, &mut c),
        Err(Error::InvalidArgument(_))
    ));
    // GGUF has no encoder.
    let mut q = s.alloc::<Q8_0>(32).unwrap();
    let x = s.upload::<F32>(&[0.0; 32]).unwrap();
    assert!(matches!(ops::convert(&mut s, Choice::Up, &x, &mut q, Overflow::Saturate), Err(Error::Unsupported(_))));
    // Tier the device does not have.
    let mut y = s.alloc::<F16>(32).unwrap();
    let r = ops::convert(&mut s, Choice::Down(Tier::Nvidia(numerobis::tiers::Nvidia::Sm80)), &x, &mut y, Overflow::Saturate);
    assert!(matches!(r, Err(Error::Unsupported(_))));
    // Buffers of another device are rejected.
    let other = Device::mock();
    let mut s2 = other.stream();
    assert!(matches!(ops::convert(&mut s2, Choice::Up, &x, &mut y, Overflow::Saturate), Err(Error::InvalidArgument(_))));
}

#[test]
fn buffers_outlive_pending_work() {
    let dev = Device::mock();
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
