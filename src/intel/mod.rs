//! Intel GPUs via the `xe` DRM driver (plan B.3 step 3).
//!
//! Raw: device query, VM/BO/bind, exec queue, `COMPUTE_WALKER`; kernels from
//! the shared `.cl` source through `ocloc`, XMX through `nb_xmx.h`.

pub mod tiers;

pub use tiers::Tier;
