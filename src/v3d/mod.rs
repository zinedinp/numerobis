//! Broadcom VideoCore (Raspberry Pi 4/5) via the `v3d` DRM driver (plan B.2c, B.3 step 8).
//!
//! No matrix engine: the Emulated tier only, with kernels from a Rust QPU
//! assembler (`qpu.rs`) submitted through `SUBMIT_CSD`.

pub mod tiers;

pub use tiers::Tier;
