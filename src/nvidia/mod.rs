//! NVIDIA GPUs via the open kernel modules.
//!
//! Raw path (feature `nvidia`): RM objects on `/dev/nvidiactl`, GPU virtual
//! memory through `/dev/nvidia-uvm`, a GPFIFO channel pair (compute + copy)
//! with QMD launches, no CUDA user space. The shared Emulated kernels run
//! from `kernels/bin/nvidia` (`sm_80` cubins on Ampere, `sm_89` on Ada).
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

#![allow(
    clippy::cast_possible_truncation,
    reason = "64-bit Linux only (gated below): u64 -> usize is lossless; \
              narrower casts fill hardware-sized fields (QMD, pushbuffer, RM params)"
)]

pub mod program;
pub mod qmd;
pub mod tiers;

// The raw backend casts u64 device sizes and addresses to usize, so it is
// built only for 64-bit targets: on a 32-bit one the feature compiles to
// nothing and `Device::all` skips it.
#[cfg(all(feature = "nvidia", target_os = "linux", target_pointer_width = "64"))]
pub mod backend;
#[cfg(all(feature = "nvidia", target_os = "linux", target_pointer_width = "64"))]
mod gpu;
#[cfg(all(feature = "nvidia", target_os = "linux", target_pointer_width = "64"))]
mod queue;
#[cfg(all(feature = "nvidia", target_os = "linux", target_pointer_width = "64"))]
pub(crate) mod sys;

pub use tiers::Tier;
