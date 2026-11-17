/// Intel NPU ladder. Derived `Ord` follows the ladder.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Tier {
    /// Meteor Lake.
    Npu3,
    /// Lunar / Arrow Lake.
    Npu4,
    /// Panther Lake.
    Npu5,
}
