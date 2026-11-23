//! NPUs: `SoC` matrix accelerators next to CPU and iGPU.
//!
//! They run precompiled command streams, not SIMT kernels, and only report
//! `Engine::Matrix` (plus `Copy`). All Linux NPU drivers live in the `accel`
//! subsystem and share one buffer/submit/fence mechanism ([`accel`]); each NPU
//! adds only its UAPI and command-stream builder.

pub mod accel;
pub mod ethosu;
pub mod ivpu;
pub mod rknpu;
pub mod xdna;
pub mod zhouyi;
