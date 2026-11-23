/// Tenstorrent ladder. Derived `Ord` follows the ladder.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Tier {
    /// `BFP8/4/2_b`, BF16.
    Wormhole,
    Blackhole,
}
