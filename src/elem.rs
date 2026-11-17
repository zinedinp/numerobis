//! Element types (plan A.5).
//!
//! Each marker is an uninhabited type naming a device storage format. Bit
//! widths, bias, NaN/Inf kind and block sizes all come from
//! [`miraculix::formats`]; nothing is redefined here.
//!
//! # Device layout
//! - Byte-aligned formats (`F32`, `F16`, FP8, `I8`, …): one element after another, little-endian.
//! - Sub-byte formats (`E2M1`, `E2M3`, `E3M2`, `I4`, `U4`, `I2`, `I1`): packed LSB-first,
//!   element 0 in the low bits of byte 0 (same as miraculix).
//! - Block formats (`Mx<E>`, `Nvfp4`, `Bfp<M>`): all element data first, then one scale
//!   byte per block; `Nvfp4` ends with the per-tensor scale as little-endian `f32`.
//! - `Q8_0`, `Q6K`: native GGUF blocks (scale interleaved).

use miraculix::formats::bfp::{BFP_ROW, BfpElem};
use miraculix::formats::gguf::{Q6_K_BLOCK, Q6_K_BYTES, Q8_0_BLOCK, Q8_0_BYTES};
use miraculix::formats::int::packed_len;
use miraculix::formats::mx::{MX_BLOCK, MxElem, NVFP4_BLOCK};
use miraculix::formats::{MiniFloat, Special};

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
    Int { bits: u8, signed: bool },
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
            Format::Int { bits: 4, signed: true } => "i4",
            Format::Int { bits: 4, signed: false } => "u4",
            Format::Int { bits: 2, signed: true } => "i2",
            Format::Int { bits: 2, signed: false } => "u2",
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

    /// Elements per block (1 for element-wise formats). Buffer lengths must be a multiple.
    pub const fn block(self) -> usize {
        match self {
            Format::Mx(_) => MX_BLOCK,
            Format::Nvfp4 => NVFP4_BLOCK,
            Format::Bfp(_) => BFP_ROW,
            Format::Q8_0 => Q8_0_BLOCK,
            Format::Q6K => Q6_K_BLOCK,
            // 6-bit codes pack 4 per 3 bytes, 4-bit 2 per byte, …
            Format::Float(f) => 8 / gcd(f.bits() as usize, 8),
            Format::Int { bits, .. } => 8 / bits as usize,
            _ => 1,
        }
    }

    /// Is this a block-scaled or GGUF format (needs a quantizer, not a plain convert)?
    pub const fn is_block_scaled(self) -> bool {
        matches!(self, Format::Mx(_) | Format::Nvfp4 | Format::Bfp(_) | Format::Q8_0 | Format::Q6K)
    }

    /// Integer formats (usable with an `I32` accumulator).
    pub const fn is_integer(self) -> bool {
        matches!(self, Format::I32 | Format::I8 | Format::U8 | Format::Int { .. })
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

mod sealed {
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

/// Byte-aligned element-wise formats that have a natural host type for
/// [`Stream::upload`](crate::Stream::upload). Other formats go through
/// [`Stream::upload_packed`](crate::Stream::upload_packed) with bytes produced by
/// the miraculix codecs.
pub trait HostRepr: Elem {
    type Host: Pod;
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

macro_rules! elem {
    ($(#[$m:meta])* $t:ident = $fmt:expr $(, host $h:ty)?) => {
        $(#[$m])*
        #[derive(Debug, Clone, Copy, PartialEq, Eq)]
        pub enum $t {}
        impl sealed::Sealed for $t {}
        impl Elem for $t { const FORMAT: Format = $fmt; }
        $(impl HostRepr for $t { type Host = $h; })?
    };
}

elem!(/// IEEE binary32.
    F32 = Format::F32, host f32);
elem!(/// TF32 values in `f32` storage (low 13 mantissa bits zero).
    Tf32 = Format::Tf32, host f32);
elem!(I32 = Format::I32, host i32);
elem!(I8 = Format::I8, host i8);
elem!(U8 = Format::U8, host u8);
elem!(F16 = Format::Float(MiniFloat::F16), host u16);
elem!(Bf16 = Format::Float(MiniFloat::Bf16), host u16);
elem!(E4M3Fn = Format::Float(MiniFloat::E4M3Fn), host u8);
elem!(E4M3Fnuz = Format::Float(MiniFloat::E4M3Fnuz), host u8);
elem!(E5M2 = Format::Float(MiniFloat::E5M2), host u8);
elem!(E5M2Fnuz = Format::Float(MiniFloat::E5M2Fnuz), host u8);
elem!(/// FP4, packed two per byte.
    E2M1 = Format::Float(MiniFloat::E2M1));
elem!(/// FP6, packed four per three bytes.
    E2M3 = Format::Float(MiniFloat::E2M3));
elem!(/// FP6, packed four per three bytes.
    E3M2 = Format::Float(MiniFloat::E3M2));
elem!(I4 = Format::Int { bits: 4, signed: true });
elem!(U4 = Format::Int { bits: 4, signed: false });
elem!(I2 = Format::Int { bits: 2, signed: true });
elem!(/// INT1 (one bit, `0`/`1`; `±1` semantics are op-specific).
    I1 = Format::Int { bits: 1, signed: false });
elem!(/// NVFP4: E2M1 in 16-element blocks with E4M3 scales and an `f32` tensor scale.
    Nvfp4 = Format::Nvfp4);
elem!(/// GGUF `Q8_0` (decode only).
    Q8_0 = Format::Q8_0);
elem!(/// GGUF `Q6_K` (decode only).
    Q6K = Format::Q6K);

/// Element markers that can sit inside an OCP MX block.
pub trait MxKind: Elem {
    const MX: MxElem;
}
impl MxKind for E2M1 { const MX: MxElem = MxElem::Fp4; }
impl MxKind for E2M3 { const MX: MxElem = MxElem::Fp6E2M3; }
impl MxKind for E3M2 { const MX: MxElem = MxElem::Fp6E3M2; }
impl MxKind for E4M3Fn { const MX: MxElem = MxElem::Fp8E4M3; }
impl MxKind for E5M2 { const MX: MxElem = MxElem::Fp8E5M2; }
impl MxKind for I8 { const MX: MxElem = MxElem::Int8; }

/// OCP MX block of 32 `E` elements with one E8M0 scale (`Mx<E2M1>` = MXFP4).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Mx<E: MxKind> {
    #[doc(hidden)]
    _Never(core::convert::Infallible, core::marker::PhantomData<E>),
}
impl<E: MxKind> sealed::Sealed for Mx<E> {}
impl<E: MxKind> Elem for Mx<E> {
    const FORMAT: Format = Format::Mx(E::MX);
}

/// Mantissa width marker for Tenstorrent `BFP_B`.
pub trait BfpKind: sealed::Sealed + 'static {
    const BFP: BfpElem;
}
macro_rules! bfp_kind {
    ($($(#[$m:meta])* $t:ident = $v:expr;)*) => {$(
        $(#[$m])*
        #[derive(Debug, Clone, Copy, PartialEq, Eq)]
        pub enum $t {}
        impl sealed::Sealed for $t {}
        impl BfpKind for $t { const BFP: BfpElem = $v; }
    )*};
}
bfp_kind! {
    /// `Bfp8_b`.
    M7 = BfpElem::M7;
    /// `Bfp4_b`.
    M3 = BfpElem::M3;
    /// `Bfp2_b`.
    M1 = BfpElem::M1;
}

/// Tenstorrent `BFP_B`: 16-value rows with one shared exponent byte.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Bfp<M: BfpKind> {
    #[doc(hidden)]
    _Never(core::convert::Infallible, core::marker::PhantomData<M>),
}
impl<M: BfpKind> sealed::Sealed for Bfp<M> {}
impl<M: BfpKind> Elem for Bfp<M> {
    const FORMAT: Format = Format::Bfp(M::BFP);
}

/// Accumulator types for GEMM (`acc ∈ {F32, F16, I32}`).
pub trait Acc: Elem {
    /// Type of `alpha`/`beta`.
    type Scalar: Copy + Into<crate::backend::Arg>;
}
impl Acc for F32 { type Scalar = f32; }
impl Acc for F16 { type Scalar = f32; }
impl Acc for I32 { type Scalar = i32; }

const ALL_MINIFLOATS: [MiniFloat; 9] = [
    MiniFloat::E4M3Fn,
    MiniFloat::E4M3Fnuz,
    MiniFloat::E5M2,
    MiniFloat::E5M2Fnuz,
    MiniFloat::E2M1,
    MiniFloat::E2M3,
    MiniFloat::E3M2,
    MiniFloat::F16,
    MiniFloat::Bf16,
];

/// Generates `kernels/include/nb_formats.h`: the kernel-side constants for every
/// format, taken from miraculix so each definition exists exactly once.
pub fn formats_header() -> String {
    use core::fmt::Write;
    let mut s = String::from(
        "/* Generated by numerobis::elem::formats_header() from miraculix::formats. Do not edit. */\n\
         #ifndef NB_FORMATS_H\n#define NB_FORMATS_H\n\n\
         #define NB_SPECIAL_IEEE 0\n#define NB_SPECIAL_FN 1\n#define NB_SPECIAL_FNUZ 2\n#define NB_SPECIAL_FINITE 3\n\n",
    );
    for f in ALL_MINIFLOATS {
        let n = Format::Float(f).name().to_ascii_uppercase();
        let sp = match f.special() {
            Special::Ieee => "IEEE",
            Special::Fn => "FN",
            Special::Fnuz => "FNUZ",
            Special::Finite => "FINITE",
        };
        let _ = writeln!(s, "#define NB_{n}_BITS {}", f.bits());
        let _ = writeln!(s, "#define NB_{n}_EXP_BITS {}", f.exp_bits());
        let _ = writeln!(s, "#define NB_{n}_MAN_BITS {}", f.man_bits());
        let _ = writeln!(s, "#define NB_{n}_BIAS {}", f.bias());
        let _ = writeln!(s, "#define NB_{n}_EMAX {}", f.emax());
        let _ = writeln!(s, "#define NB_{n}_MAX_CODE 0x{:X}u", f.max_code());
        let _ = writeln!(s, "#define NB_{n}_SPECIAL NB_SPECIAL_{sp}\n");
    }
    let _ = writeln!(s, "#define NB_MX_BLOCK {MX_BLOCK}");
    let _ = writeln!(s, "#define NB_NVFP4_BLOCK {NVFP4_BLOCK}");
    let _ = writeln!(s, "#define NB_BFP_ROW {BFP_ROW}");
    let _ = writeln!(s, "#define NB_Q8_0_BLOCK {Q8_0_BLOCK}");
    let _ = writeln!(s, "#define NB_Q8_0_BYTES {Q8_0_BYTES}");
    let _ = writeln!(s, "#define NB_Q6_K_BLOCK {Q6_K_BLOCK}");
    let _ = writeln!(s, "#define NB_Q6_K_BYTES {Q6_K_BYTES}");
    s.push_str("\n#endif /* NB_FORMATS_H */\n");
    s
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

    /// `kernels/include/nb_formats.h` must match miraculix. Regenerate with
    /// `NUMEROBIS_BLESS=1 cargo test`.
    #[test]
    fn nb_formats_header_is_current() {
        let path = concat!(env!("CARGO_MANIFEST_DIR"), "/kernels/include/nb_formats.h");
        let want = formats_header();
        if std::env::var_os("NUMEROBIS_BLESS").is_some() {
            std::fs::write(path, &want).unwrap();
        }
        let have = std::fs::read_to_string(path).unwrap_or_default();
        assert!(have == want, "{path} is stale; rerun with NUMEROBIS_BLESS=1");
    }
}
