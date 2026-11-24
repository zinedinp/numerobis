//! Tenstorrent via tt-kmd.
//!
//! Raw path (feature `tt`, Blackhole): TLB windows into the `NoC`, DRAM device
//! buffers, the shared Emulated kernels on a Tensix BRISC core
//! (`kernels/tt`, RV32IM with soft-float). Next: work spread over all
//! Tensix tiles, then the FPU tier with `BFP_B` formats.
//!
//! | Module | Contents |
//! |---|---|
//! | `sys` | tt-kmd ioctl structs (size-asserted) |
//! | `card` | device node, TLB windows, NoC word/bulk access |
//! | `backend` | the [`Backend`](crate::runtime::Backend) implementation |

#![allow(
    clippy::cast_possible_truncation,
    clippy::many_single_char_names,
    reason = "64-bit Linux only (gated below): u64 -> usize is lossless; \
              L1 addresses and mailbox words are 32-bit; x, y are NoC coordinates"
)]

pub mod tiers;

// The raw backend casts u64 device sizes and addresses to usize, so it is
// built only for 64-bit targets: on a 32-bit one the feature compiles to
// nothing and `Device::all` skips it.
#[cfg(all(feature = "tt", target_os = "linux", target_pointer_width = "64"))]
pub mod backend;
#[cfg(all(feature = "tt", target_os = "linux", target_pointer_width = "64"))]
pub mod card;
#[cfg(all(feature = "tt", target_os = "linux", target_pointer_width = "64"))]
pub(crate) mod sys;

pub use tiers::Tier;
