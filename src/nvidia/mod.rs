//! NVIDIA GPUs via the open kernel modules (plan B.3 step 6).
//!
//! Raw up to `Sm89` (RM, GPFIFO, QMD, `mma.sync` via `nb_nv.h`); `wgmma` and
//! `tcgen05` go through the `cuda` comfort backend first.

pub mod tiers;

pub use tiers::Tier;
