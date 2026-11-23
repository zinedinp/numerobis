/// `VideoCore` ladder (Emulated matrix only). Derived `Ord` follows the ladder.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Tier {
    /// Pi 4 / CM4.
    V3d42,
    /// Pi 5 / CM5.
    V3d71,
}
