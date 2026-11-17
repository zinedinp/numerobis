//! AMD GPUs via KFD.
//!
//! Raw: AQL queue, doorbell, signals; HSACO from the shared `.cl` source,
//! MFMA/WMMA through `nb_amd.h`.

pub mod tiers;

pub use tiers::Tier;
