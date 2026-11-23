//! Block-scaled element types: OCP MX, NVFP4, Tenstorrent `BFP_B`, GGUF.

use miraculix::formats::bfp::BfpElem;
use miraculix::formats::mx::MxElem;

use super::float::{E2M1, E2M3, E3M2, E4M3Fn, E5M2};
use super::int::I8;
use super::{Elem, Format, sealed};

/// Block formats; [`ops::dequantize`](crate::ops::dequantize) turns them into `F32`.
pub trait BlockElem: Elem {}

/// Block formats with an encoder; [`ops::quantize`](fn@crate::ops::quantize) produces them
/// from `F32`. GGUF is decode-only, as in miraculix.
pub trait Quantize: BlockElem {}

elem!(
    /// NVFP4: E2M1 in 16-element blocks with E4M3 scales and an `f32` tensor scale.
    Nvfp4 = Format::Nvfp4
);
elem!(
    /// GGUF `Q8_0` (decode only).
    Q8_0 = Format::Q8_0
);
elem!(
    /// GGUF `Q6_K` (decode only).
    Q6K = Format::Q6K
);
impl BlockElem for Nvfp4 {}
impl Quantize for Nvfp4 {}
impl BlockElem for Q8_0 {}
impl BlockElem for Q6K {}

/// Element markers that can sit inside an OCP MX block.
pub trait MxKind: Elem {
    const MX: MxElem;
}
impl MxKind for E2M1 {
    const MX: MxElem = MxElem::Fp4;
}
impl MxKind for E2M3 {
    const MX: MxElem = MxElem::Fp6E2M3;
}
impl MxKind for E3M2 {
    const MX: MxElem = MxElem::Fp6E3M2;
}
impl MxKind for E4M3Fn {
    const MX: MxElem = MxElem::Fp8E4M3;
}
impl MxKind for E5M2 {
    const MX: MxElem = MxElem::Fp8E5M2;
}
impl MxKind for I8 {
    const MX: MxElem = MxElem::Int8;
}

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
impl<E: MxKind> BlockElem for Mx<E> {}
impl<E: MxKind> Quantize for Mx<E> {}

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
impl<M: BfpKind> BlockElem for Bfp<M> {}
impl<M: BfpKind> Quantize for Bfp<M> {}
