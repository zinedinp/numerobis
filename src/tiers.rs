//! The crate-wide [`Tier`]: [`Tier::Emulated`] plus one variant per vendor
//! module, each wrapping that vendor's linear ladder (`intel::Tier`,
//! `amd::Tier`, ...), like the per-arch `tiers.rs` of miraculix.
//!
//! A device's [`FeatureSet::tiers`](crate::FeatureSet) lists the tiers it
//! supports in ascending order; `auto_up` walks it from the top.

use crate::{amd, intel, npu, nvidia, tt, v3d};

/// A matrix tier.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum Tier {
    /// Integer bit-logic decode + F32 vector FMA. Runs on every backend.
    Emulated,
    Intel(intel::Tier),
    Amd(amd::Tier),
    Nvidia(nvidia::Tier),
    Tt(tt::Tier),
    V3d(v3d::Tier),
    IntelNpu(npu::ivpu::Tier),
    Xdna(npu::xdna::Tier),
    Rknpu(npu::rknpu::Tier),
    EthosU(npu::ethosu::Tier),
    Zhouyi(npu::zhouyi::Tier),
}

impl Tier {
    /// Short name used in kernel binary paths (`kernels/bin/<vendor>/<tier>/...`).
    #[must_use]
    pub fn name(self) -> String {
        let (v, t) = match self {
            Tier::Emulated => return "emulated".into(),
            Tier::Intel(t) => ("intel", format!("{t:?}")),
            Tier::Amd(t) => ("amd", format!("{t:?}")),
            Tier::Nvidia(t) => ("nvidia", format!("{t:?}")),
            Tier::Tt(t) => ("tt", format!("{t:?}")),
            Tier::V3d(t) => ("v3d", format!("{t:?}")),
            Tier::IntelNpu(t) => ("ivpu", format!("{t:?}")),
            Tier::Xdna(t) => ("xdna", format!("{t:?}")),
            Tier::Rknpu(t) => ("rknpu", format!("{t:?}")),
            Tier::EthosU(t) => ("ethosu", format!("{t:?}")),
            Tier::Zhouyi(t) => ("zhouyi", format!("{t:?}")),
        };
        format!("{v}-{}", t.to_ascii_lowercase())
    }
}
