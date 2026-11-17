//! AMD GPUs via KFD (plan B.3 step 5).
//!
//! Raw: AQL queue, doorbell, signals; HSACO from the shared `.cl` source,
//! MFMA/WMMA through `nb_amd.h`.

pub mod tiers;

pub use tiers::Tier;
