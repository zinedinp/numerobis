/// NVIDIA matrix ladder. `Sm120` is a side branch next to `Sm100`. Derived `Ord` follows the ladder.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Tier {
    /// `mma.sync`.
    Sm80,
    /// FP8.
    Sm89,
    /// `wgmma` (comfort first).
    Sm90,
    /// `tcgen05`, MX, NVFP4 (comfort first).
    Sm100,
    Sm120,
}
