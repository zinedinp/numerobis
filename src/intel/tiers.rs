/// Intel GPU matrix ladder. Derived `Ord` follows the ladder.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Tier {
    /// DP4A only.
    XeLp,
    /// XMX F16/BF16/I8, SIMD8.
    XeHpg,
    /// XMX SIMD16.
    XeHpc,
    /// XMX SIMD16, FP8 when the device query reports it.
    Xe2,
}
