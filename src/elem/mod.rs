//! Element types, split by kind like `miraculix::formats`:
//! [`float`] (F32, TF32, F16/BF16, FP8, FP6, FP4), [`int`] (I32, I8, U8 and
//! packed I4/U4/I2/I1) and [`block`] (MX, NVFP4, BFP_B, GGUF).
//!
//! Each marker is an uninhabited type naming a device storage format. Bit
//! widths, bias, NaN/Inf kind and block sizes all come from
//! [`miraculix::formats`]; nothing is redefined here.
//!
//! # Device layout
//! - Byte-aligned formats (`F32`, `F16`, FP8, `I8`, ...): one element after another, little-endian.
//! - Sub-byte formats (`E2M1`, `E2M3`, `E3M2`, `I4`, `U4`, `I2`, `I1`): packed LSB-first,
//!   element 0 in the low bits of byte 0 (same as miraculix).
//! - Block formats (`Mx<E>`, `Nvfp4`, `Bfp<M>`): all element data first, then one scale
//!   byte per block; `Nvfp4` ends with the per-tensor scale as little-endian `f32`.
//! - `Q8_0`, `Q6K`: native GGUF blocks (scale interleaved).

use miraculix::formats::MiniFloat;
use miraculix::formats::bfp::{BFP_ROW, BfpElem};
use miraculix::formats::gguf::{Q6_K_BLOCK, Q6_K_BYTES, Q8_0_BLOCK, Q8_0_BYTES};
use miraculix::formats::int::packed_len;
use miraculix::formats::mx::{MX_BLOCK, MxElem, NVFP4_BLOCK};

macro_rules! elem {
    ($(#[$m:meta])* $t:ident = $fmt:expr $(, host $h:ty)?) => {
        $(#[$m])*
        #[derive(Debug, Clone, Copy, PartialEq, Eq)]
        pub enum $t {}
        impl $crate::elem::sealed::Sealed for $t {}
        impl $crate::elem::Elem for $t { const FORMAT: $crate::elem::Format = $fmt; }
        $(impl $crate::elem::HostRepr for $t { type Host = $h; })?
    };
}

pub mod block;
pub mod float;
mod header;
pub mod int;

pub use block::{Bfp, BfpKind, BlockElem, M1, M3, M7, Mx, MxKind, Nvfp4, Q6K, Q8_0, Quantize};
pub use float::{
    Bf16, E2M1, E2M3, E3M2, E4M3Fn, E4M3Fnuz, E5M2, E5M2Fnuz, F16, F32, FloatElem, Tf32,
};
pub use header::formats_header;
pub use int::{I1, I2, I4, I8, I32, IntElem, PackedInt, U4, U8};

/// Runtime description of a storage format.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum Format {
    F32,
    /// `f32` storage holding TF32-rounded values.
    Tf32,
    I32,
    I8,
    U8,
    /// A [`MiniFloat`] code: 16/8 bits byte-aligned, 6/4 bits packed.
    Float(MiniFloat),
    /// Packed integer of 1, 2 or 4 bits.
    Int {
        bits: u8,
        signed: bool,
    },
    Mx(MxElem),
    Nvfp4,
    Bfp(BfpElem),
    Q8_0,
    Q6K,
}

impl Format {
    /// Lower-case suffix used in kernel names (`gemm_<a>_<b>_<acc>`).
    pub const fn name(self) -> &'static str {
        use MiniFloat as M;
        match self {
            Format::F32 => "f32",
            Format::Tf32 => "tf32",
            Format::I32 => "i32",
            Format::I8 => "i8",
            Format::U8 => "u8",
            Format::Float(M::E4M3Fn) => "e4m3fn",
            Format::Float(M::E4M3Fnuz) => "e4m3fnuz",
            Format::Float(M::E5M2) => "e5m2",
            Format::Float(M::E5M2Fnuz) => "e5m2fnuz",
            Format::Float(M::E2M1) => "e2m1",
            Format::Float(M::E2M3) => "e2m3",
            Format::Float(M::E3M2) => "e3m2",
            Format::Float(M::F16) => "f16",
            Format::Float(M::Bf16) => "bf16",
            Format::Float(_) => "minifloat",
            Format::Int {
                bits: 4,
                signed: true,
            } => "i4",
            Format::Int {
                bits: 4,
                signed: false,
            } => "u4",
            Format::Int {
                bits: 2,
                signed: true,
            } => "i2",
            Format::Int {
                bits: 2,
                signed: false,
            } => "u2",
            Format::Int { bits: 1, .. } => "i1",
            Format::Int { .. } => "int",
            Format::Mx(MxElem::Fp4) => "mxfp4",
            Format::Mx(MxElem::Fp6E2M3) => "mxfp6e2m3",
            Format::Mx(MxElem::Fp6E3M2) => "mxfp6e3m2",
            Format::Mx(MxElem::Fp8E4M3) => "mxfp8e4m3",
            Format::Mx(MxElem::Fp8E5M2) => "mxfp8e5m2",
            Format::Mx(MxElem::Int8) => "mxint8",
            Format::Mx(_) => "mx",
            Format::Nvfp4 => "nvfp4",
            Format::Bfp(BfpElem::M7) => "bfp8b",
            Format::Bfp(BfpElem::M3) => "bfp4b",
            Format::Bfp(BfpElem::M1) => "bfp2b",
            Format::Q8_0 => "q8_0",
            Format::Q6K => "q6_k",
        }
    }

    /// Elements per block (1 for byte-aligned element-wise formats). Buffer lengths must be a multiple.
    pub const fn block(self) -> usize {
        match self {
            Format::Mx(_) => MX_BLOCK,
            Format::Nvfp4 => NVFP4_BLOCK,
            Format::Bfp(_) => BFP_ROW,
            Format::Q8_0 => Q8_0_BLOCK,
            Format::Q6K => Q6_K_BLOCK,
            // 6-bit codes pack 4 per 3 bytes, 4-bit 2 per byte, ...
            Format::Float(f) => 8 / gcd(f.bits() as usize, 8),
            Format::Int { bits, .. } => 8 / bits as usize,
            _ => 1,
        }
    }

    /// Block-scaled or GGUF format (needs a quantizer, not a plain convert).
    pub const fn is_block_scaled(self) -> bool {
        matches!(
            self,
            Format::Mx(_) | Format::Nvfp4 | Format::Bfp(_) | Format::Q8_0 | Format::Q6K
        )
    }

    /// Integer formats (usable with an `I32` accumulator).
    pub const fn is_integer(self) -> bool {
        matches!(
            self,
            Format::I32 | Format::I8 | Format::U8 | Format::Int { .. }
        )
    }

    /// Device bytes for `len` elements, or `None` if `len` is not a multiple of [`block`](Self::block).
    pub const fn storage_bytes(self, len: usize) -> Option<usize> {
        if !len.is_multiple_of(self.block()) {
            return None;
        }
        let blocks = len / self.block();
        Some(match self {
            Format::F32 | Format::Tf32 | Format::I32 => len * 4,
            Format::I8 | Format::U8 => len,
            Format::Float(f) => len * f.bits() as usize / 8,
            Format::Int { bits: 4, .. } => packed_len::<4>(len),
            Format::Int { bits: 2, .. } => packed_len::<2>(len),
            Format::Int { .. } => packed_len::<1>(len),
            Format::Mx(e) => blocks * (e.block_bytes() + 1),
            Format::Nvfp4 => blocks * (NVFP4_BLOCK / 2 + 1) + 4,
            Format::Bfp(e) => blocks * (e.row_bytes() + 1),
            Format::Q8_0 => blocks * Q8_0_BYTES,
            Format::Q6K => blocks * Q6_K_BYTES,
        })
    }

    /// Kernel-ABI format descriptor, passed to the format-generic Emulated
    /// kernels (decoded by the `NB_*` macros in `kernels/include/nb_codec.h`):
    ///
    /// | bits | field |
    /// |---|---|
    /// | 0..4 | kind: 0 F32, 1 TF32, 2 I32, 3 I8, 4 U8, 5 Float, 6 Int, 7 Mx, 8 NVFP4, 9 BFP, 10 Q8_0, 11 Q6_K |
    /// | Float, Mx: 4..8, 8..12, 12..20, 20..22 | exponent bits, mantissa bits, bias, [`Special`](miraculix::formats::Special) |
    /// | Mx: 22 | MXINT8 element |
    /// | Int: 4..8, 8 | width, signed |
    /// | Bfp: 4..8 | mantissa bits |
    pub fn desc(self) -> u32 {
        use miraculix::formats::Special;
        fn fl(f: MiniFloat) -> u32 {
            let sp = match f.special() {
                Special::Ieee => 0,
                Special::Fn => 1,
                Special::Fnuz => 2,
                Special::Finite => 3,
            };
            (f.exp_bits() << 4) | (f.man_bits() << 8) | ((f.bias() as u32) << 12) | (sp << 20)
        }
        match self {
            Format::F32 => 0,
            Format::Tf32 => 1,
            Format::I32 => 2,
            Format::I8 => 3,
            Format::U8 => 4,
            Format::Float(f) => 5 | fl(f),
            Format::Int { bits, signed } => 6 | ((bits as u32) << 4) | ((signed as u32) << 8),
            Format::Mx(e) => 7 | e.float().map_or(1 << 22, fl),
            Format::Nvfp4 => 8,
            Format::Bfp(e) => 9 | (e.mantissa_bits() << 4),
            Format::Q8_0 => 10,
            Format::Q6K => 11,
        }
    }

    /// Byte offset of the scale section of a block format (= size of the element data).
    pub const fn scale_offset(self, len: usize) -> usize {
        let blocks = len / self.block();
        match self {
            Format::Mx(e) => blocks * e.block_bytes(),
            Format::Nvfp4 => blocks * NVFP4_BLOCK / 2,
            Format::Bfp(e) => blocks * e.row_bytes(),
            _ => match self.storage_bytes(len) {
                Some(n) => n,
                None => 0,
            },
        }
    }
}

const fn gcd(a: usize, b: usize) -> usize {
    if b == 0 { a } else { gcd(b, a % b) }
}

pub(crate) mod sealed {
    pub trait Sealed {}
}

/// A device element type. Sealed: the set of formats is fixed by miraculix.
pub trait Elem: sealed::Sealed + 'static {
    const FORMAT: Format;
    /// Kernel-name suffix, same as `FORMAT.name()`.
    const NAME: &'static str = Self::FORMAT.name();
}

/// Plain-old-data host types that may be reinterpreted as bytes.
///
/// # Safety
/// Implementors must have no padding and accept every bit pattern.
pub unsafe trait Pod: Copy + Send + Sync + 'static {}
unsafe impl Pod for u8 {}
unsafe impl Pod for i8 {}
unsafe impl Pod for u16 {}
unsafe impl Pod for i32 {}
unsafe impl Pod for u32 {}
unsafe impl Pod for f32 {}

/// Byte-aligned element-wise formats with a natural host type for
/// [`Stream::upload`](crate::Stream::upload). Other formats go through
/// [`Stream::upload_packed`](crate::Stream::upload_packed) with bytes produced by
/// the miraculix codecs.
pub trait HostRepr: Elem {
    type Host: Pod;
}

/// GEMM accumulators (`acc in {F32, F16, I32}`).
pub trait Acc: Elem {
    /// Type of `alpha`/`beta`.
    type Scalar: Copy + Into<crate::runtime::Arg>;
}
impl Acc for F32 {
    type Scalar = f32;
}
impl Acc for F16 {
    type Scalar = f32;
}
impl Acc for I32 {
    type Scalar = i32;
}

pub(crate) fn as_bytes<T: Pod>(s: &[T]) -> &[u8] {
    // SAFETY: `Pod` guarantees no padding; u8 has alignment 1.
    unsafe { core::slice::from_raw_parts(s.as_ptr().cast(), core::mem::size_of_val(s)) }
}

pub(crate) fn from_bytes<T: Pod>(b: &[u8]) -> Vec<T> {
    let n = b.len() / core::mem::size_of::<T>();
    (0..n)
        // SAFETY: in bounds; `Pod` accepts every bit pattern; unaligned read.
        .map(|i| unsafe { b.as_ptr().cast::<T>().add(i).read_unaligned() })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn storage_sizes() {
        assert_eq!(F32::FORMAT.storage_bytes(3), Some(12));
        assert_eq!(E2M1::FORMAT.storage_bytes(4), Some(2));
        assert_eq!(E2M1::FORMAT.storage_bytes(3), None);
        assert_eq!(E2M3::FORMAT.storage_bytes(4), Some(3));
        assert_eq!(E2M3::FORMAT.storage_bytes(2), None);
        assert_eq!(I4::FORMAT.storage_bytes(8), Some(4));
        assert_eq!(<Mx<E2M1>>::FORMAT.storage_bytes(64), Some(2 * 16 + 2));
        assert_eq!(<Mx<E2M1>>::FORMAT.storage_bytes(33), None);
        assert_eq!(Nvfp4::FORMAT.storage_bytes(32), Some(2 * 8 + 2 + 4));
        assert_eq!(<Bfp<M7>>::FORMAT.storage_bytes(32), Some(2 * 17));
        assert_eq!(<Mx<E4M3Fn>>::NAME, "mxfp8e4m3");
        assert_eq!(Bf16::NAME, "bf16");
    }
}
