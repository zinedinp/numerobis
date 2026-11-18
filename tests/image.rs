//! Ops beyond GEMM (activations, direct convolution, resize): the real
//! Emulated kernels must match the mock oracle bit for bit, and the oracle
//! must be close to the textbook functions.

mod common;

use common::*;
use numerobis::elem::*;
use numerobis::ops::{self, Act, Conv2dShape, ResizeMode};
use numerobis::{Choice, Device, DeviceBuffer, Stream};

fn agree<E: Elem>(what: &str, f: impl Fn(&mut Stream) -> DeviceBuffer<E>) {
    let run = |dev: Device| {
        let mut s = dev.stream();
        let out = f(&mut s);
        bytes(&mut s, &out)
    };
    let want = run(Device::mock());
    for dev in targets() {
        let got = run(dev.clone());
        assert_eq!(want.len(), got.len(), "{what} on {}", dev.features().name);
        if E::FORMAT == F32::FORMAT {
            for (i, (w, g)) in want.chunks(4).zip(got.chunks(4)).enumerate() {
                let (w, g) = (
                    f32::from_le_bytes(w.try_into().unwrap()),
                    f32::from_le_bytes(g.try_into().unwrap()),
                );
                assert!(
                    same(w, g),
                    "{what} on {}: element {i}: mock {w:e} ({:#x}), kernel {g:e} ({:#x})",
                    dev.features().name,
                    w.to_bits(),
                    g.to_bits()
                );
            }
        } else {
            assert_eq!(want, got, "{what} on {}", dev.features().name);
        }
    }
}

fn inputs() -> Vec<f32> {
    let mut r = Rng(7);
    let mut v = edge_values(&mut r);
    v.extend([
        88.7,
        88.73,
        -103.9,
        -104.0,
        1e-5,
        -1e-5,
        2.0f32.powi(-13),
        9.0,
        9.01,
        -9.01,
        20.0,
        -20.0,
    ]);
    v.extend(r.vec(200, 12.0));
    v.truncate(v.len() / 8 * 8);
    v
}

const ACTS: [Act; 9] = [
    Act::Relu,
    Act::LeakyRelu(0.1),
    Act::Sigmoid,
    Act::Tanh,
    Act::Silu,
    Act::Gelu,
    Act::Clamp(-1.5, 2.5),
    Act::Exp,
    Act::Clamp(0.0, 0.0),
];

#[test]
fn activations_bit_exact() {
    let x = inputs();
    for a in ACTS {
        agree::<F32>(&format!("{a:?} f32"), |s| {
            let i = s.upload::<F32>(&x).unwrap();
            let mut o = s.alloc::<F32>(x.len()).unwrap();
            ops::act(s, Choice::Up, a, &i, &mut o)
                .unwrap()
                .wait()
                .unwrap();
            o
        });
        agree::<Bf16>(&format!("{a:?} f32->bf16"), |s| {
            let i = s.upload::<F32>(&x).unwrap();
            let mut o = s.alloc::<Bf16>(x.len()).unwrap();
            ops::act(s, Choice::Up, a, &i, &mut o)
                .unwrap()
                .wait()
                .unwrap();
            o
        });
    }
}

#[test]
fn activations_accuracy() {
    let x: Vec<f32> = (0..4001).map(|i| -20.0 + i as f32 * 0.01).collect();
    let dev = Device::mock();
    let mut s = dev.stream();
    let i = s.upload::<F32>(&x).unwrap();
    let mut o = s.alloc::<F32>(x.len()).unwrap();
    let check = |name: &str, got: &[f32], f: &dyn Fn(f64) -> f64, tol: f64| {
        let worst = x
            .iter()
            .zip(got)
            .map(|(&x, &g)| {
                let w = f(x as f64);
                ((g as f64 - w).abs() / w.abs().max(1e-30)).min((g as f64 - w).abs())
            })
            .fold(0.0f64, f64::max);
        assert!(worst < tol, "{name}: worst error {worst:e}");
    };
    for (a, f, tol) in [
        (Act::Exp, &(|x: f64| x.exp()) as &dyn Fn(f64) -> f64, 3e-7),
        (Act::Sigmoid, &|x: f64| 1.0 / (1.0 + (-x).exp()), 4e-7),
        (Act::Tanh, &|x: f64| x.tanh(), 5e-7),
        (Act::Silu, &|x: f64| x / (1.0 + (-x).exp()), 6e-7),
        (
            Act::Gelu,
            &|x: f64| {
                0.5 * x
                    * (1.0
                        + ((2.0 / std::f64::consts::PI).sqrt() * (x + 0.044715 * x * x * x)).tanh())
            },
            8e-7,
        ),
    ] {
        ops::act(&mut s, Choice::Up, a, &i, &mut o)
            .unwrap()
            .wait()
            .unwrap();
        check(&format!("{a:?}"), &download(&mut s, &o), f, tol);
    }
}

fn conv_case(shape: Conv2dShape, bias: bool, fused: Option<Act>, seed: u64) {
    let mut r = Rng(seed);
    let x = r.vec(shape.input_len(), 2.0);
    let w = r.vec(shape.weight_len(), 0.5);
    let b = r.vec(shape.k, 1.0);
    let out = shape.output_len().unwrap();
    agree::<F32>(&format!("{shape:?} bias {bias} {fused:?}"), |s| {
        let (xi, wi, bi) = (
            s.upload::<F32>(&x).unwrap(),
            s.upload::<F32>(&w).unwrap(),
            s.upload::<F32>(&b).unwrap(),
        );
        let mut y = s.alloc::<F32>(out).unwrap();
        ops::conv2d(
            s,
            Choice::Up,
            shape,
            &xi,
            &wi,
            bias.then_some(&bi),
            fused,
            &mut y,
        )
        .unwrap()
        .wait()
        .unwrap();
        y
    });
}

#[test]
fn conv2d_bit_exact() {
    conv_case(Conv2dShape::new(1, 3, 9, 11, 4, 3, 3), false, None, 1);
    conv_case(
        Conv2dShape::new(2, 4, 8, 8, 6, 3, 3).pad(1, 1).stride(2, 2),
        true,
        Some(Act::Silu),
        2,
    );
    conv_case(
        Conv2dShape::new(1, 6, 7, 7, 6, 3, 3)
            .pad(2, 2)
            .dilation(2, 2)
            .groups(6),
        true,
        Some(Act::Relu),
        3,
    );
    conv_case(
        Conv2dShape::new(1, 8, 5, 6, 4, 1, 1).groups(2),
        false,
        Some(Act::Gelu),
        4,
    );
}

#[test]
fn conv2d_matches_reference_math() {
    // 1x1 input channel, 3x3 box filter over a ramp with zero padding.
    let shape = Conv2dShape::new(1, 1, 3, 3, 1, 3, 3).pad(1, 1);
    let dev = Device::mock();
    let mut s = dev.stream();
    let x = s
        .upload::<F32>(&[1., 2., 3., 4., 5., 6., 7., 8., 9.])
        .unwrap();
    let w = s.upload::<F32>(&[1.0; 9]).unwrap();
    let mut y = s.alloc::<F32>(9).unwrap();
    ops::conv2d(&mut s, Choice::Up, shape, &x, &w, None, None, &mut y)
        .unwrap()
        .wait()
        .unwrap();
    assert_eq!(
        download(&mut s, &y),
        [12., 21., 16., 27., 45., 33., 24., 39., 28.]
    );
    // Wrong lengths and unfusable activations are rejected before submission.
    assert!(
        ops::conv2d(
            &mut s,
            Choice::Up,
            shape,
            &x,
            &w,
            None,
            Some(Act::LeakyRelu(0.1)),
            &mut y
        )
        .is_err()
    );
    let mut small = s.alloc::<F32>(4).unwrap();
    assert!(ops::conv2d(&mut s, Choice::Up, shape, &x, &w, None, None, &mut small).is_err());
}

#[test]
fn resize_bit_exact() {
    let mut r = Rng(11);
    for (planes, (h, w), (ho, wo)) in [
        (3, (5, 7), (10, 14)),
        (2, (16, 9), (5, 4)),
        (1, (1, 1), (3, 3)),
        (4, (8, 8), (8, 8)),
    ] {
        let x = r.vec(planes * h * w, 100.0);
        for mode in [ResizeMode::Nearest, ResizeMode::Bilinear] {
            agree::<F32>(&format!("{mode:?} {h}x{w}->{ho}x{wo}"), |s| {
                let i = s.upload::<F32>(&x).unwrap();
                let mut o = s.alloc::<F32>(planes * ho * wo).unwrap();
                ops::resize(s, Choice::Up, mode, planes, (h, w), (ho, wo), &i, &mut o)
                    .unwrap()
                    .wait()
                    .unwrap();
                o
            });
            agree::<F16>(&format!("{mode:?} f16 {h}x{w}->{ho}x{wo}"), |s| {
                let i = s.upload::<F32>(&x).unwrap();
                let mut o = s.alloc::<F16>(planes * ho * wo).unwrap();
                ops::resize(s, Choice::Up, mode, planes, (h, w), (ho, wo), &i, &mut o)
                    .unwrap()
                    .wait()
                    .unwrap();
                o
            });
        }
    }
}

#[test]
fn resize_values() {
    let dev = Device::mock();
    let mut s = dev.stream();
    let x = s.upload::<F32>(&[0.0, 10.0, 20.0, 30.0]).unwrap(); // 2x2
    let mut o = s.alloc::<F32>(16).unwrap();
    ops::resize(
        &mut s,
        Choice::Up,
        ResizeMode::Nearest,
        1,
        (2, 2),
        (4, 4),
        &x,
        &mut o,
    )
    .unwrap()
    .wait()
    .unwrap();
    assert_eq!(
        download(&mut s, &o),
        [
            0., 0., 10., 10., 0., 0., 10., 10., 20., 20., 30., 30., 20., 20., 30., 30.
        ]
    );
    ops::resize(
        &mut s,
        Choice::Up,
        ResizeMode::Bilinear,
        1,
        (2, 2),
        (4, 4),
        &x,
        &mut o,
    )
    .unwrap()
    .wait()
    .unwrap();
    // Row 0: x = 0, 2.5, 7.5, 10 (half-pixel centres, edge clamped).
    assert_eq!(&download(&mut s, &o)[..4], &[0.0, 2.5, 7.5, 10.0]);
}
