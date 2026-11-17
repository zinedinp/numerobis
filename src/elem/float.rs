//! Floating-point element types: F32, TF32 and every [`MiniFloat`].

use miraculix::formats::MiniFloat;

use super::{Elem, Format};

/// Element-wise floating formats; [`ops::convert`](crate::ops::convert) maps between them.
pub trait FloatElem: Elem {}

macro_rules! float {
    ($($(#[$m:meta])* $t:ident = $fmt:expr $(, host $h:ty)?;)*) => {$(
        elem!($(#[$m])* $t = $fmt $(, host $h)?);
        impl FloatElem for $t {}
    )*};
}

float! {
    /// IEEE binary32.
    F32 = Format::F32, host f32;
    /// TF32 values in `f32` storage (low 13 mantissa bits zero).
    Tf32 = Format::Tf32, host f32;
    /// IEEE binary16.
    F16 = Format::Float(MiniFloat::F16), host u16;
    /// bfloat16.
    Bf16 = Format::Float(MiniFloat::Bf16), host u16;
    /// FP8 E4M3, OCP FN.
    E4M3Fn = Format::Float(MiniFloat::E4M3Fn), host u8;
    /// FP8 E4M3 FNUZ (AMD gfx942).
    E4M3Fnuz = Format::Float(MiniFloat::E4M3Fnuz), host u8;
    /// FP8 E5M2.
    E5M2 = Format::Float(MiniFloat::E5M2), host u8;
    /// FP8 E5M2 FNUZ (AMD gfx942).
    E5M2Fnuz = Format::Float(MiniFloat::E5M2Fnuz), host u8;
    /// FP4, packed two per byte.
    E2M1 = Format::Float(MiniFloat::E2M1);
    /// FP6, packed four per three bytes.
    E2M3 = Format::Float(MiniFloat::E2M3);
    /// FP6, packed four per three bytes.
    E3M2 = Format::Float(MiniFloat::E3M2);
}
