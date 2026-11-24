//! The process's Direct3D plans, made once at startup and kept, as
//! `vulkan::cache` (feature `vulkan`)
//! does for Vulkan: probe every adapter once, [`publish`] (or
//! [`get_or_init`]), and ask [`plans`] / [`best`] anywhere later. What
//! changes while the program runs, the video memory budget, goes through a
//! [`BudgetCache`](super::BudgetCache).

use std::sync::OnceLock;

use super::{Capability, Plan, rank};

static PLANS: OnceLock<Vec<Plan>> = OnceLock::new();

/// Keeps `plans` for the process, one per adapter in the host's order. The
/// first call wins.
///
/// # Errors
///
/// The given plans, when plans were kept already.
pub fn publish(plans: Vec<Plan>) -> Result<&'static [Plan], Vec<Plan>> {
    PLANS.set(plans)?;
    Ok(self::plans())
}

/// The kept plans, or `init`'s, kept now; `init` runs at most once.
pub fn get_or_init(init: impl FnOnce() -> Vec<Plan>) -> &'static [Plan] {
    PLANS.get_or_init(init)
}

/// The kept plans; empty before [`publish`] or [`get_or_init`].
#[must_use]
pub fn plans() -> &'static [Plan] {
    PLANS.get().map_or(&[], Vec::as_slice)
}

/// The best kept plan granting all of `required`, with its index.
#[must_use]
pub fn best(required: &[Capability]) -> Option<(usize, &'static Plan)> {
    let plans = plans();
    rank(plans, required).first().map(|&i| (i, &plans[i]))
}
