//! NVIDIA GPUs via the open kernel modules (plan B.3 step 6).
//!
//! Raw path (feature `nvidia`): RM objects on `/dev/nvidiactl`, GPU virtual
//! memory through `/dev/nvidia-uvm`, a GPFIFO channel pair (compute + copy)
//! with QMD launches, no CUDA user space. The shared Emulated kernels run
//! from `kernels/bin/nvidia` (sm_80 cubins on Ampere, sm_89 on Ada).
//! `mma.sync` (`nb_nv.h`) is the next tier; `wgmma`/`tcgen05` (Sm90+) go
//! through the `cuda` comfort backend first, so Hopper/Blackwell are not
//! opened raw.
//!
//! | Module | Contents |
//! |---|---|
//! | `sys` | RM/UVM ioctl structs (size-asserted), classes, methods |
//! | `qmd` | QMD v03_00 field layout |
//! | `program` | cubin parsing (entry points, constant banks, parameters) |
//! | `gpu` | RM client/device, VA reservation, VRAM / sysmem allocations |
//! | `queue` | GPFIFO channels, timeline semaphore, push buffers |
//! | `backend` | the [`Backend`](crate::runtime::Backend) implementation |

pub mod program;
pub mod qmd;
pub mod tiers;

#[cfg(all(feature = "nvidia", target_os = "linux"))]
pub mod backend;
#[cfg(all(feature = "nvidia", target_os = "linux"))]
mod gpu;
#[cfg(all(feature = "nvidia", target_os = "linux"))]
mod queue;
#[cfg(all(feature = "nvidia", target_os = "linux"))]
pub(crate) mod sys;

pub use tiers::Tier;
