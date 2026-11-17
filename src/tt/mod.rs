//! Tenstorrent via tt-kmd.
//!
//! Raw path (feature `tt`, Blackhole): TLB windows into the NoC, DRAM device
//! buffers, the shared Emulated kernels on a Tensix BRISC core
//! (`kernels/tt`, RV32IM with soft-float). Next: work spread over all
//! Tensix tiles, then the FPU tier with BFP_B formats.
//!
//! | Module | Contents |
//! |---|---|
//! | `sys` | tt-kmd ioctl structs (size-asserted) |
//! | `card` | device node, TLB windows, NoC word/bulk access |
//! | `backend` | the [`Backend`](crate::runtime::Backend) implementation |

pub mod tiers;

#[cfg(all(feature = "tt", target_os = "linux"))]
pub mod backend;
#[cfg(all(feature = "tt", target_os = "linux"))]
pub mod card;
#[cfg(all(feature = "tt", target_os = "linux"))]
pub(crate) mod sys;

pub use tiers::Tier;
