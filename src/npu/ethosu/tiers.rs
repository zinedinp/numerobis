/// Ethos-U ladder. Derived `Ord` follows the ladder.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Tier {
    U55,
    U65,
    U85,
}
