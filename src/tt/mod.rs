//! Tenstorrent via tt-kmd (plan B.3 step 9).
//!
//! Raw: TLB/NoC, Tensix kernel loading; own GEMM kernels, BFP_B native.

pub mod tiers;

pub use tiers::Tier;
