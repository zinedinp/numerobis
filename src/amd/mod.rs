//! AMD GPUs via KFD.
//!
//! Raw path (feature `amd`, Linux): `/dev/kfd` with the process VM on the
//! GPU's render node, buffers at one address for CPU and GPU, one AQL
//! compute queue (ring, `amd_queue_t`, doorbell, completion signals), and
//! the shared Emulated kernels from `kernels/bin/amd` (`gfx10-3-generic` for
//! all of RDNA 2, `gfx1100`, `gfx1200`, CDNA `gfx908` to `gfx950`). No `ROCm`
//! user space. MFMA/WMMA tiers (`nb_amd.h`) are the next step.
//!
//! | Module | Contents |
//! |---|---|
//! | `program` | HSACO parsing: image, kernel descriptors, kernarg layout (code object v5) |
//! | `sys` | KFD ioctl structs (size-asserted), AQL packet, `amd_queue_t` and `amd_signal_t` offsets |
//! | `kfd` | topology nodes, queue buffer sizes, VM, buffer objects |
//! | `queue` | the AQL queue |
//! | `backend` | the [`Backend`](crate::runtime::Backend) implementation |

#![allow(
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap,
    clippy::cast_sign_loss,
    clippy::cast_precision_loss,
    reason = "64-bit Linux only (gated below): u64 <-> usize is lossless; \
              narrower casts fill hardware-sized fields (AQL packet, KFD args)"
)]

pub mod program;
pub mod tiers;

// The raw backend casts u64 device sizes and addresses to usize, so it is
// built only for 64-bit targets: on a 32-bit one the feature compiles to
// nothing and `Device::all` skips it.
#[cfg(all(feature = "amd", target_os = "linux", target_pointer_width = "64"))]
pub mod backend;
#[cfg(all(feature = "amd", target_os = "linux", target_pointer_width = "64"))]
pub mod kfd;
#[cfg(all(feature = "amd", target_os = "linux", target_pointer_width = "64"))]
mod queue;
#[cfg(all(feature = "amd", target_os = "linux", target_pointer_width = "64"))]
pub(crate) mod sys;

pub use tiers::Tier;
