//! `auto_down`: enumerate every tier that can run an op, to force each one in
//! turn (test matrices, benchmarks). Pass the result to
//! [`Choice::Down`](crate::Choice::Down).

use crate::runtime::Device;
use crate::runtime::backend::{KernelKey, OpSig};
use crate::tiers::Tier;

/// All tiers of the device that support `sig`, ascending.
pub fn tiers(dev: &Device, sig: OpSig, batched: bool) -> Vec<Tier> {
    let name = sig.name(batched);
    dev.features()
        .tiers
        .iter()
        .copied()
        .filter(|&tier| {
            dev.backend().supports(&KernelKey {
                name: name.clone(),
                sig,
                tier,
            })
        })
        .collect()
}
