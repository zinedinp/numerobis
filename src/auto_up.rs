//! `auto_up`: run every op on the best tier the device has a kernel for.

use crate::runtime::backend::{KernelKey, OpSig};
use crate::runtime::Device;
use crate::runtime::error::{Error, Result};
use crate::tiers::Tier;

/// Highest tier in the device's ladder that supports `sig`.
pub fn choose(dev: &Device, sig: OpSig, batched: bool) -> Result<Tier> {
    let name = sig.name(batched);
    dev.features()
        .tiers
        .iter()
        .rev()
        .copied()
        .find(|&tier| dev.backend().supports(&KernelKey { name: name.clone(), sig, tier }))
        .ok_or_else(|| Error::Unsupported(format!("{name} on {}", dev.features().name)))
}
