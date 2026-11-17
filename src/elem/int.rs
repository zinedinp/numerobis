//! Integer element types: I32, I8, U8 and packed I4/U4/I2/I1.

use super::{Elem, Format, HostRepr};

/// Integer formats (GEMM with an `I32` accumulator).
pub trait IntElem: Elem {}

/// Packed sub-byte integers. [`Wide`](Self::Wide) is the byte type that
/// [`ops::pack`](crate::ops::pack) reads and [`ops::unpack`](crate::ops::unpack) writes.
pub trait PackedInt: IntElem {
    type Wide: HostRepr + IntElem;
}

macro_rules! int {
    ($($(#[$m:meta])* $t:ident = $fmt:expr $(, host $h:ty)? $(, wide $w:ty)?;)*) => {$(
        elem!($(#[$m])* $t = $fmt $(, host $h)?);
        impl IntElem for $t {}
        $(impl PackedInt for $t { type Wide = $w; })?
    )*};
}

int! {
    I32 = Format::I32, host i32;
    I8 = Format::I8, host i8;
    U8 = Format::U8, host u8;
    I4 = Format::Int { bits: 4, signed: true }, wide I8;
    U4 = Format::Int { bits: 4, signed: false }, wide U8;
    I2 = Format::Int { bits: 2, signed: true }, wide I8;
    /// INT1 (one bit, `0`/`1`; `+-1` semantics are op-specific).
    I1 = Format::Int { bits: 1, signed: false }, wide U8;
}
