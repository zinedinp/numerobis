//! The process's plans, made once at startup and kept, like miraculix's
//! `detect_features`. numerobis makes no Vulkan calls, so the host fills it:
//! probe every physical device once, [`publish`] (or [`get_or_init`]), and
//! ask [`plans`] / [`best`] anywhere later without probing again.
//!
//! What a plan holds doesn't change while the program runs (feature bits,
//! extensions, device identity, heap sizes), except the memory budget: that
//! one is read again through a [`BudgetCache`](super::memory::BudgetCache).
//!
//! ```
//! use numerobis::vulkan::{ApiVersion, Capability as Cap, Offer, Probe, cache};
//!
//! // At startup, one plan per vkEnumeratePhysicalDevices entry, in order.
//! let plans = cache::get_or_init(|| {
//!     let offer = Offer::new(ApiVersion::V1_3, ApiVersion::V1_3, []);
//!     let mut probe = Probe::new(offer, &[Cap::DynamicRendering]);
//!     let _ = probe.features2(); // vkGetPhysicalDeviceFeatures2 ...
//!     vec![probe.plan()]
//! });
//! assert_eq!(plans.len(), 1);
//! // Anywhere later: the same plans, no probe.
//! assert!(std::ptr::eq(plans, cache::plans()));
//! let _best: Option<(usize, _)> = cache::best(&[]);
//! ```

use std::sync::OnceLock;

use super::{Capability, Plan, rank};

static PLANS: OnceLock<Vec<Plan>> = OnceLock::new();

/// Keeps `plans` for the process, one per physical device in the host's
/// order. The first call wins; a later one gets its plans back.
///
/// # Errors
///
/// The given plans, when plans were kept already.
pub fn publish(plans: Vec<Plan>) -> Result<&'static [Plan], Vec<Plan>> {
    PLANS.set(plans)?;
    Ok(plans_or_empty())
}

/// The kept plans, or `init`'s, kept now. `init` runs at most once per
/// process (racing threads wait for it).
pub fn get_or_init(init: impl FnOnce() -> Vec<Plan>) -> &'static [Plan] {
    PLANS.get_or_init(init)
}

/// The kept plans; empty before [`publish`] or [`get_or_init`].
#[must_use]
pub fn plans() -> &'static [Plan] {
    plans_or_empty()
}

fn plans_or_empty() -> &'static [Plan] {
    PLANS.get().map_or(&[], Vec::as_slice)
}

/// The best kept plan that grants all of `required`, with its index (the
/// physical device it was made for): [`rank`] over the cache.
#[must_use]
pub fn best(required: &[Capability]) -> Option<(usize, &'static Plan)> {
    let plans = plans();
    rank(plans, required).first().map(|&i| (i, &plans[i]))
}
