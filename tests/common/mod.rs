#![allow(dead_code)]

use numerobis::elem::*;
use numerobis::{Choice, DeviceBuffer, Overflow, Stream, ops};

pub struct Rng(pub u64);
impl Rng {
    pub fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }
    pub fn f(&mut self) -> f32 {
        (self.next() >> 40) as f32 / (1u64 << 24) as f32 * 2.0 - 1.0
    }
    pub fn vec(&mut self, n: usize, scale: f32) -> Vec<f32> {
        (0..n).map(|_| self.f() * scale).collect()
    }
}

/// Bit-identical, or both NaN.
pub fn same(a: f32, b: f32) -> bool {
    a.to_bits() == b.to_bits() || (a.is_nan() && b.is_nan())
}

pub fn edge_values(rng: &mut Rng) -> Vec<f32> {
    let mut v = vec![
        0.0, -0.0, 1.0, -1.0, 0.5, 448.0, 449.0, 480.0, 1e6, -1e6, 6.0, 7.0, 1e-9,
        f32::INFINITY, f32::NEG_INFINITY, f32::NAN,
    ];
    v.extend(rng.vec(240, 600.0));
    v.extend(rng.vec(256, 0.01));
    v
}

pub fn download<E: HostRepr>(s: &mut Stream, b: &DeviceBuffer<E>) -> Vec<E::Host> {
    s.download(b).unwrap().wait().unwrap().to_vec()
}

pub fn bytes<E: Elem>(s: &mut Stream, b: &DeviceBuffer<E>) -> Vec<u8> {
    s.download(b).unwrap().wait().unwrap().into_bytes()
}

/// Float format -> host `f32` through the device.
pub fn float_to_host<E: FloatElem>(s: &mut Stream, b: &DeviceBuffer<E>) -> Vec<f32> {
    let mut o = s.alloc::<F32>(b.len()).unwrap();
    ops::convert(s, Choice::Up, b, &mut o, Overflow::Saturate).unwrap().wait().unwrap();
    download(s, &o)
}

/// Block format -> host `f32` through the device.
pub fn block_to_host<E: BlockElem>(s: &mut Stream, b: &DeviceBuffer<E>) -> Vec<f32> {
    let mut o = s.alloc::<F32>(b.len()).unwrap();
    ops::dequantize(s, Choice::Up, b, &mut o).unwrap().wait().unwrap();
    download(s, &o)
}

pub mod hostcl;

/// Every device the suite runs on: the mock oracle, the real Emulated
/// kernels compiled for the host, and every raw hardware device of the
/// enabled backend features (`--features nvidia`, ...).
pub fn devices() -> Vec<numerobis::Device> {
    let mut v = vec![numerobis::Device::mock()];
    v.extend(targets());
    v
}

/// Devices that run real kernel code (host build, the Tenstorrent RV32
/// build under qemu when `qemu-riscv32` is installed, and hardware),
/// compared against the mock oracle in `tests/emulated.rs`.
pub fn targets() -> Vec<numerobis::Device> {
    let mut v = vec![hostcl::HostCl::device()];
    if hostcl::have_qemu_rv32() {
        v.push(hostcl::HostCl::rv32());
    }
    v.extend(numerobis::Device::all());
    v
}
