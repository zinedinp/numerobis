//! # numerobis
//!
//! Raw accelerator computing on the matrix engines of GPUs, NPUs and
//! accelerator cards, with the number formats of [`miraculix::formats`].
//!
//! Data lives on the device. The only ways in and out are
//! [`Stream::upload`] / [`Stream::upload_packed`] and [`Stream::download`].
//!
//! ```
//! use numerobis::{Choice, Device, GemmShape, Overflow, elem::*, ops};
//!
//! let dev = Device::mock();
//! let mut s = dev.stream();
//! let a = s.upload::<F32>(&[1.0, 2.0, 3.0, 4.0])?; // 2×2
//! let b = s.upload::<F32>(&[1.0, 0.0, 0.0, 1.0])?; // 2×2, stored N×K
//! let mut a8 = s.alloc::<E4M3Fn>(4)?;
//! let mut b8 = s.alloc::<E4M3Fn>(4)?;
//! ops::convert(&mut s, Choice::Up, &a, &mut a8, Overflow::Saturate)?.wait()?;
//! ops::convert(&mut s, Choice::Up, &b, &mut b8, Overflow::Saturate)?.wait()?;
//! let mut c = s.alloc::<F32>(4)?;
//! ops::gemm(&mut s, Choice::Up, GemmShape::new(2, 2, 2), 1.0, &a8, &b8, 0.0, &mut c)?.wait()?;
//! assert_eq!(s.download(&c)?.wait()?.to_vec(), [1.0, 2.0, 3.0, 4.0]);
//! # Ok::<(), numerobis::Error>(())
//! ```

#![deny(unsafe_op_in_unsafe_fn)]

pub mod auto_down;
pub mod auto_up;
pub mod backend;
mod buffer;
mod device;
pub mod elem;
pub mod elf;
pub mod engine;
mod error;
#[cfg(feature = "mock")]
pub mod mock;
pub mod ops;
mod stream;
pub mod tiers;

pub use buffer::{DeviceBuffer, HostCopy};
pub use device::{Device, Placement};
pub use engine::{Access, DeviceClass, Engine, FeatureSet};
pub use error::{Error, Result};
pub use miraculix::formats::Overflow;
pub use ops::{Choice, GemmShape};
pub use stream::{DEFAULT_TIMEOUT, Pending, Stream};
pub use tiers::Tier;
