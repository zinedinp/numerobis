/// XDNA (AIE) ladder. Derived `Ord` follows the ladder.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Tier {
    /// I8, BF16.
    Aie2,
    /// BFP16.
    Aie2p,
}
