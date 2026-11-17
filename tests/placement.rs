use std::sync::Arc;

use numerobis::backend::OpSig;
use numerobis::elem::{Elem, F16, F32};
use numerobis::mock::MockBackend;
use numerobis::tiers::{Intel, IntelNpu};
use numerobis::{Device, DeviceClass, Placement, Tier, auto_down, auto_up};

fn dev(class: DeviceClass, tiers: Vec<Tier>, kernels: Vec<Tier>) -> Device {
    Device::from_backend(Arc::new(MockBackend::with_tiers(class, tiers, kernels)))
}

#[test]
fn auto_up_picks_best_tier_with_a_kernel() {
    let ladder = vec![Tier::Emulated, Tier::Intel(Intel::XeLp), Tier::Intel(Intel::XeHpg), Tier::Intel(Intel::Xe2)];
    // Xe2 reported but has no kernel yet: falls back to XeHpg.
    let d = dev(DeviceClass::Gpu, ladder.clone(), vec![Tier::Emulated, Tier::Intel(Intel::XeHpg)]);
    let sig = OpSig::Gemm { a: F16::FORMAT, b: F16::FORMAT, acc: F32::FORMAT };
    assert_eq!(auto_up::choose(&d, sig, false).unwrap(), Tier::Intel(Intel::XeHpg));
    assert_eq!(auto_down::tiers(&d, sig, false), [Tier::Emulated, Tier::Intel(Intel::XeHpg)]);
    assert_eq!(sig.name(false), "gemm_f16_f16_f32");
    assert_eq!(sig.name(true), "gemm_strided_f16_f16_f32");
}

#[test]
fn placement_low_power_prefers_npu() {
    let gpu = dev(DeviceClass::Gpu, vec![Tier::Emulated, Tier::Intel(Intel::Xe2)], vec![Tier::Emulated]);
    let npu = dev(DeviceClass::Npu, vec![Tier::Emulated, Tier::IntelNpu(IntelNpu::Npu4)], vec![Tier::Emulated]);
    let all = [gpu, npu];
    assert_eq!(Placement::LowPower.pick(&all).unwrap().features().class, DeviceClass::Npu);
    assert_eq!(Placement::Fastest.pick(&all).unwrap().features().class, DeviceClass::Gpu);
    assert_eq!(Placement::Device(1).pick(&all).unwrap().features().class, DeviceClass::Npu);
    assert!(Placement::Device(2).pick(&all).is_none());
}

#[test]
fn open_without_backends_reports_no_device() {
    assert!(matches!(Device::open(Placement::Fastest), Err(numerobis::Error::NoDevice)));
}
