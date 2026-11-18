use std::sync::Arc;

use numerobis::elem::{Elem, F16, F32};
use numerobis::mock::MockBackend;
use numerobis::runtime::OpSig;
use numerobis::{Device, DeviceClass, Placement, Tier, auto_down, auto_up};
use numerobis::{intel, npu};

fn dev(class: DeviceClass, tiers: Vec<Tier>, kernels: Vec<Tier>) -> Device {
    Device::from_backend(Arc::new(MockBackend::with_tiers(class, tiers, kernels)))
}

#[test]
fn auto_up_picks_best_tier_with_a_kernel() {
    let ladder = vec![
        Tier::Emulated,
        Tier::Intel(intel::Tier::XeLp),
        Tier::Intel(intel::Tier::XeHpg),
        Tier::Intel(intel::Tier::Xe2),
    ];
    // Xe2 reported but has no kernel yet: falls back to XeHpg.
    let d = dev(
        DeviceClass::Gpu,
        ladder.clone(),
        vec![Tier::Emulated, Tier::Intel(intel::Tier::XeHpg)],
    );
    let sig = OpSig::Gemm {
        a: F16::FORMAT,
        b: F16::FORMAT,
        acc: F32::FORMAT,
    };
    assert_eq!(
        auto_up::choose(&d, sig, false).unwrap(),
        Tier::Intel(intel::Tier::XeHpg)
    );
    assert_eq!(
        auto_down::tiers(&d, sig, false),
        [Tier::Emulated, Tier::Intel(intel::Tier::XeHpg)]
    );
    assert_eq!(sig.name(false), "gemm_f16_f16_f32");
    assert_eq!(sig.name(true), "gemm_strided_f16_f16_f32");
}

#[test]
fn placement_low_power_prefers_npu() {
    let gpu = dev(
        DeviceClass::Gpu,
        vec![Tier::Emulated, Tier::Intel(intel::Tier::Xe2)],
        vec![Tier::Emulated],
    );
    let npu = dev(
        DeviceClass::Npu,
        vec![Tier::Emulated, Tier::IntelNpu(npu::ivpu::Tier::Npu4)],
        vec![Tier::Emulated],
    );
    let all = [gpu, npu];
    assert_eq!(
        Placement::LowPower.pick(&all).unwrap().features().class,
        DeviceClass::Npu
    );
    assert_eq!(
        Placement::Fastest.pick(&all).unwrap().features().class,
        DeviceClass::Gpu
    );
    assert_eq!(
        Placement::Device(1).pick(&all).unwrap().features().class,
        DeviceClass::Npu
    );
    assert!(Placement::Device(2).pick(&all).is_none());
}

#[test]
fn open_reports_no_device_only_without_hardware() {
    let r = Device::open(Placement::Fastest);
    if Device::all().is_empty() {
        assert!(matches!(r, Err(numerobis::Error::NoDevice)));
    } else {
        assert_eq!(r.unwrap().features().class, DeviceClass::Gpu);
    }
}
