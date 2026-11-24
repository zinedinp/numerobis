use miraculix::formats::{MiniFloat, Overflow};

use super::*;
use crate::pci::{Gpu, TtArch};

#[test]
fn every_engine_has_instructions_that_fit_their_formats() {
    for engine in Engine::ALL {
        assert!(engine.mmas().next().is_some(), "{engine}");
        for mma in engine.mmas() {
            for x in [mma.a, mma.b] {
                assert!(
                    (mma.shape.k as usize).is_multiple_of(x.block()),
                    "{engine}: {mma}: K is not whole blocks of {}",
                    x.name()
                );
            }
            assert_eq!(
                mma.a.is_integer(),
                mma.acc.is_integer(),
                "{engine}: {mma}: an integer input needs the I32 accumulator"
            );
            assert!(mma.shape.macs() > 0);
        }
    }
}

#[test]
fn tiers_and_engines_point_at_each_other() {
    for engine in Engine::ALL {
        if let Some(tier) = engine.tier() {
            assert_eq!(Engine::of(tier), Some(engine), "{tier:?}");
            assert_eq!(tier.matrix(), Some(engine));
        }
    }
    assert_eq!(Engine::of(Tier::Emulated), None);
    assert_eq!(Engine::of(Tier::Intel(crate::intel::Tier::XeLp)), None);
    let ladder = [
        Tier::Emulated,
        Tier::Nvidia(crate::nvidia::Tier::Sm80),
        Tier::Nvidia(crate::nvidia::Tier::Sm89),
    ];
    assert_eq!(
        engines(&ladder),
        [
            Engine::TensorCore(nvidia::TensorCore::Ampere),
            Engine::TensorCore(nvidia::TensorCore::Ada)
        ]
    );
}

#[test]
fn known_cards_name_their_engine() {
    use amd::MatrixCore as M;
    use nvidia::TensorCore as T;
    let engine = |vendor, device| Engine::from_gpu(Gpu::from_pci(vendor, device));
    assert_eq!(engine(0x8086, 0xE20B), Some(Engine::Xmx(intel::Xmx::Xe2))); // Arc B580
    assert_eq!(engine(0x8086, 0x56A0), Some(Engine::Xmx(intel::Xmx::XeHpg))); // Arc A770
    assert_eq!(engine(0x8086, 0x0BD5), Some(Engine::Xmx(intel::Xmx::XeHpc))); // Max 1550
    assert_eq!(engine(0x8086, 0xA780), None); // UHD 770: dp4a only
    assert_eq!(engine(0x10DE, 0x1D81), Some(Engine::TensorCore(T::Volta))); // Titan V
    assert_eq!(engine(0x10DE, 0x1E04), Some(Engine::TensorCore(T::Turing))); // RTX 2080 Ti
    assert_eq!(engine(0x10DE, 0x1F82), None); // GTX 1650: TU117
    assert_eq!(engine(0x10DE, 0x2184), None); // GTX 1660: TU116
    assert_eq!(engine(0x10DE, 0x2206), Some(Engine::TensorCore(T::Ampere))); // RTX 3080
    assert_eq!(engine(0x10DE, 0x2684), Some(Engine::TensorCore(T::Ada))); // RTX 4090
    assert_eq!(engine(0x10DE, 0x2330), Some(Engine::TensorCore(T::Hopper))); // H100
    assert_eq!(
        engine(0x10DE, 0x2901),
        Some(Engine::TensorCore(T::Blackwell))
    ); // B200
    assert_eq!(
        engine(0x10DE, 0x2B85),
        Some(Engine::TensorCore(T::BlackwellGeForce))
    ); // RTX 5090
    assert_eq!(engine(0x1002, 0x73BF), None); // RX 6800 XT: dot products only
    // None from the tables: known to have no engine, or not known at all.
    let absent = |vendor, device| Engine::known_absent(Gpu::from_pci(vendor, device));
    assert!(absent(0x1002, 0x73BF) && absent(0x1002, 0x1681)); // RX 6800 XT, Radeon 680M
    assert!(absent(0x8086, 0xA780) && absent(0x10DE, 0x1F82));
    assert!(!absent(0x1002, 0x744C)); // RX 7900 XTX has one
    assert!(!absent(0x1002, 0x0001) && !absent(0x1234, 0x5678)); // not in the tables
    assert!(!absent(0x1002, 0x7380)); // a CDNA id the tables don't place
    assert_eq!(engine(0x1002, 0x744C), Some(Engine::MatrixCore(M::Rdna3))); // RX 7900 XTX
    assert_eq!(engine(0x1002, 0x7550), Some(Engine::MatrixCore(M::Rdna4))); // RX 9070 XT
    assert_eq!(engine(0x1002, 0x738C), Some(Engine::MatrixCore(M::Cdna1))); // MI100
    assert_eq!(engine(0x1002, 0x740C), Some(Engine::MatrixCore(M::Cdna2))); // MI250X
    assert_eq!(engine(0x1002, 0x74A1), Some(Engine::MatrixCore(M::Cdna3))); // MI300X
    assert_eq!(
        engine(0x1E52, 0xB140),
        Some(Engine::Tensix(tt::Tensix::Blackhole))
    );
    assert_eq!(
        engine(0x1E52, 0x401E),
        Some(Engine::Tensix(tt::Tensix::Wormhole))
    );
    assert_eq!(tt::Tensix::from_arch(TtArch::Grayskull), None);
    assert_eq!(
        Engine::MatrixCore(M::Rdna3).to_string(),
        "RDNA 3 AI Accelerator"
    );
    assert_eq!(Engine::MatrixCore(M::Cdna3).unit(), "Matrix Core");
}

#[test]
fn routes_prefer_native_then_the_narrowest_widening() {
    use f::*;
    let xe2 = Engine::Xmx(intel::Xmx::Xe2);
    let ada = Engine::TensorCore(nvidia::TensorCore::Ada);
    let native = |e: Engine, a, b, acc| matches!(e.route(a, b, acc), Route::Native(_));

    assert!(native(xe2, F16, F16, F32));
    assert!(native(ada, E4M3, E4M3, F32));
    assert!(native(ada, E4M3, E5M2, F32));
    // Xe2 has no FP8: widened to F16 on both sides.
    match xe2.route(E4M3, E5M2, F32) {
        Route::Widened { a, b, acc, mma } => {
            assert_eq!((a, b, acc), (F16, F16, F32));
            assert_eq!(mma.shape, Shape::new(8, 16, 16));
        }
        other => panic!("{other:?}"),
    }
    // Only the side that needs it is widened.
    match ada.route(E2M1, E4M3, F32) {
        Route::Widened { a, b, .. } => assert_eq!((a, b), (E4M3, E4M3)),
        other => panic!("{other:?}"),
    }
    match xe2.route(I4, I8, I32) {
        Route::Widened { a, b, .. } => assert_eq!((a, b), (I8, I8)),
        other => panic!("{other:?}"),
    }
    // A wider accumulator when the narrow one has no instruction.
    let cdna3 = Engine::MatrixCore(amd::MatrixCore::Cdna3);
    match cdna3.route(F16, F16, F16) {
        Route::Widened { acc, .. } => assert_eq!(acc, F32),
        other => panic!("{other:?}"),
    }
    // No F32 instruction on XMX: F32 is not narrowed (that would round).
    assert_eq!(xe2.route(F32, F32, F32), Route::Vector);
    assert!(native(cdna3, F32, F32, F32));
    // Block formats: native where the hardware reads the scales, else none.
    let blackwell = Engine::TensorCore(nvidia::TensorCore::Blackwell);
    assert!(native(blackwell, NVFP4, NVFP4, F32));
    assert!(native(blackwell, MXFP4, MXFP4, F32));
    assert_eq!(xe2.route(NVFP4, NVFP4, F32), Route::Vector);
    let tensix = Engine::Tensix(tt::Tensix::Blackhole);
    assert!(native(tensix, BFP8, BFP8, F32));
    assert_eq!(ada.route(BFP8, BFP8, F32), Route::Vector);
    // Integers stay integers.
    assert_eq!(xe2.route(I8, I8, F32), Route::Vector);
}

/// Every finite code of the source decodes to a value the target encodes
/// and decodes back unchanged: the widenings lose nothing.
#[test]
fn widening_keeps_every_value() {
    let floats = [
        MiniFloat::E2M1,
        MiniFloat::E2M3,
        MiniFloat::E3M2,
        MiniFloat::E4M3Fn,
        MiniFloat::E5M2,
        MiniFloat::E4M3Fnuz,
        MiniFloat::E5M2Fnuz,
        MiniFloat::F16,
        MiniFloat::Bf16,
    ];
    let mut checked = 0;
    for from in floats {
        for &to in widen(Format::Float(from)) {
            for code in 0..(1u32 << from.bits()) {
                let v = from.decode(code);
                if v.is_nan() {
                    continue;
                }
                let back = match to {
                    Format::Float(t) => t.decode(t.encode(v, Overflow::Special)),
                    Format::Tf32 => {
                        assert_eq!(v.to_bits() & 0x1FFF, 0, "{from:?} {code:#x} is not tf32");
                        v
                    }
                    Format::F32 => v,
                    other => panic!("{from:?} widens to {other:?}"),
                };
                assert_eq!(
                    (back.to_bits(), back),
                    (v.to_bits(), v),
                    "{from:?} {code:#x} into {}",
                    to.name()
                );
                checked += 1;
            }
        }
    }
    assert!(checked > 100_000);
    // Integers: every value of the narrow range fits the wide one.
    let range = |x: Format| match x {
        Format::I8 => (-128, 127),
        Format::U8 => (0, 255),
        Format::Int { bits, signed: true } => (-(1 << (bits - 1)), (1 << (bits - 1)) - 1),
        Format::Int {
            bits,
            signed: false,
        } => (0, (1 << bits) - 1),
        other => panic!("{other:?}"),
    };
    for bits in [2, 4] {
        for signed in [true, false] {
            let x = Format::Int { bits, signed };
            let (lo, hi) = range(x);
            for &to in widen(x) {
                let (tlo, thi) = range(to);
                assert!(tlo <= lo && hi <= thi, "{} into {}", x.name(), to.name());
            }
        }
    }
}

#[cfg(feature = "vulkan")]
mod vulkan {
    use super::super::vulkan::{Properties, compare, component};
    use super::*;

    /// `vkGetPhysicalDeviceCooperativeMatrixPropertiesKHR` on the Arc B580,
    /// driver 101.9175 (Windows), in the driver's order.
    const B580: [(u32, u32, u32, i32, i32); 4] = [
        (8, 16, 16, component::FLOAT16, component::FLOAT32),
        (8, 16, 32, component::UINT8, component::UINT32),
        (8, 16, 32, component::SINT8, component::SINT32),
        (8, 16, 16, component::FLOAT16, component::FLOAT16),
    ];

    #[test]
    fn the_b580s_cooperative_matrices_are_xe2_instructions() {
        let reported: Vec<Mma> = B580
            .iter()
            .map(|&(m, n, k, x, acc)| Properties::new(m, n, k, x, acc).mma(16).expect("known"))
            .collect();
        let xe2 = Engine::Xmx(intel::Xmx::Xe2);
        let c = compare(xe2, &reported);
        assert!(c.unexplained.is_empty(), "{c}");
        assert_eq!(c.matched.len(), 4);
        for (r, t) in &c.matched {
            assert_eq!(r.shape, t.shape, "one dpas per cooperative matrix");
        }
        let hidden: Vec<String> = c
            .hidden
            .iter()
            .map(|m| format!("{}x{}", m.a.name(), m.b.name()))
            .collect();
        assert_eq!(hidden, ["bf16xbf16", "i8xu8", "u8xi8", "i4xi4", "u4xu4"]);
        // On Xe-HPG's SIMD8 tiles they would be two instructions each.
        let hpg = compare(Engine::Xmx(intel::Xmx::XeHpg), &reported);
        assert!(hpg.matched.iter().all(|(r, t)| t.shape.n * 2 == r.shape.n));
    }

    #[test]
    fn driver_tiles_may_span_several_instructions() {
        // NVIDIA reports 16x16x16 for f16; mma.sync is m16n8k16.
        let tile = Properties::new(16, 16, 16, component::FLOAT16, component::FLOAT32)
            .mma(32)
            .expect("known");
        let c = compare(Engine::TensorCore(nvidia::TensorCore::Ampere), &[tile]);
        assert_eq!(c.matched[0].1.shape, Shape::new(16, 8, 16));
    }

    #[test]
    fn component_types_round_trip() {
        for engine in Engine::ALL {
            for m in engine.mmas() {
                if let Some(c) = super::super::vulkan::component(m.a) {
                    assert_eq!(super::super::vulkan::format(c), Some(m.a));
                }
            }
        }
        let mut odd = Properties::new(8, 16, 16, component::FLOAT16, component::FLOAT32);
        odd.result = component::FLOAT16;
        assert_eq!(odd.mma(16), None, "C and Result differ");
        odd = Properties::new(8, 16, 16, component::FLOAT64, component::FLOAT64);
        assert_eq!(odd.mma(16), None, "no f64 format");
    }
}
