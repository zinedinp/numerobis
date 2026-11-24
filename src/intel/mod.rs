//! Intel GPUs via the `xe` DRM driver.
//!
//! Raw: device query, VM/BO/bind, exec queue, `COMPUTE_WALKER`; kernels from
//! the shared `.cl` source through `ocloc`, XMX through `nb_xmx.h`.
//!
//! [`xess`]: XeSS 2, the upscaler that runs its network on XMX, as data for
//! a host that calls the library (Windows).

pub mod tiers;
pub mod xess;

pub use tiers::Tier;
