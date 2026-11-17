/// AMD matrix ladders: CDNA (`Gfx908..Gfx950`) and RDNA (`Gfx11 < Gfx12`). Derived `Ord` follows the ladder.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Tier {
    Gfx908,
    Gfx90a,
    /// FP8 FNUZ.
    Gfx942,
    /// FP8 OCP, MXFP4/6.
    Gfx950,
    /// RDNA3 WMMA.
    Gfx11,
    /// RDNA4 WMMA FP8.
    Gfx12,
}
