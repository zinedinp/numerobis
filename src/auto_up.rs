//! `auto_up`: run every op on the best tier the device has a kernel for.

use crate::runtime::Device;
use crate::runtime::backend::{KernelKey, OpSig};
use crate::runtime::error::{Error, Result};
use crate::tiers::Tier;

/// Highest tier in the device's ladder that supports `sig`.
///
/// # Errors
///
/// [`Error::Unsupported`] when no tier of the device has a kernel for
/// `sig`.
pub fn choose(dev: &Device, sig: OpSig, batched: bool) -> Result<Tier> {
    choose_with(dev, sig, batched, false)
}

/// [`choose`], optionally only among tiers on which `sig` is bit-exact
/// ([`OpSig::exact_on`]): the rule for [`Exactness::Bit`](crate::Exactness) streams.
///
/// # Errors
///
/// [`Error::Unsupported`] when no tier (with `bit_exact`: no bit-exact
/// tier) has a kernel for `sig`.
pub fn choose_with(dev: &Device, sig: OpSig, batched: bool, bit_exact: bool) -> Result<Tier> {
    let name = sig.name(batched);
    dev.features()
        .tiers
        .iter()
        .rev()
        .copied()
        .filter(|&tier| !bit_exact || sig.exact_on(tier))
        .find(|&tier| {
            dev.backend().supports(&KernelKey {
                name: name.clone(),
                sig,
                tier,
            })
        })
        .ok_or_else(|| {
            let how = if bit_exact {
                " (bit-exact tiers only)"
            } else {
                ""
            };
            Error::Unsupported(format!("{name} on {}{how}", dev.features().name))
        })
}
