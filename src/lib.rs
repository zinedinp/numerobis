//! # numerobis
//!
//! Raw accelerator computing on the matrix engines of GPUs, NPUs and
//! accelerator cards, with the number formats of [`miraculix::formats`].
//!
//! Data lives on the device. The only ways in and out are
//! [`Stream::upload`] / [`Stream::upload_packed`] and [`Stream::download`].
//!
//! ## Layout (like miraculix: one module per target, shared API on top)
//!
//! | Module | Contents |
//! |---|---|
//! | [`elem`] | element types, split into `float`, `int`, `block` |
//! | [`ops`] | the API, one module per calculation kind (`convert`, `quantize`, `int`, `gemm`, `reduce`, `act`, `conv`, `resize`) |
#![cfg_attr(
    target_os = "linux",
    doc = "| [`dmabuf`] | `HostDmabuf`, PRIME helpers: zero-copy sharing with Vulkan/KMS/V4L2 |",
    doc = "| [`sync`] | `SyncFile` / `Timeline`: fences shared with Vulkan semaphores |"
)]
#![cfg_attr(
    not(target_os = "linux"),
    doc = "| `dmabuf` | Linux only: `HostDmabuf`, PRIME helpers, zero-copy sharing with Vulkan/KMS/V4L2 |",
    doc = "| `sync` | Linux only: `SyncFile` / `Timeline`, fences shared with Vulkan semaphores |"
)]
//! | [`runtime`] | `Backend` trait, `Device`, `Stream`, `DeviceBuffer`, `CommandList`, `Exactness`, dma-buf, errors |
//! | [`auto_up`], [`auto_down`], [`tiers`] | tier selection over the vendor ladders |
//! | [`intel`], [`amd`], [`nvidia`], [`tt`], [`v3d`] | one module per GPU / card vendor |
//! | [`npu`] | `accel` subsystem base plus one module per NPU |
//! | [`loader`] | kernel binary containers (ELF) |
//! | [`pci`] | which GPU a PCI vendor/device id is, and its driver version: every OS, no backend |
//! | [`vulkan`] | Vulkan versions, feature bits per version, and a waterfall picking each capability's best route: no bindings, the host calls Vulkan |
//! | `mock` | host oracle backend (feature `mock`) |
//!
//! ```
//! use numerobis::{Choice, Device, GemmShape, Overflow, elem::*, ops};
//!
//! let dev = Device::mock();
//! let mut s = dev.stream();
//! let a = s.upload::<F32>(&[1.0, 2.0, 3.0, 4.0])?; // 2x2
//! let b = s.upload::<F32>(&[1.0, 0.0, 0.0, 1.0])?; // 2x2, stored NxK
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

#[macro_use]
pub mod elem;

pub mod auto_down;
pub mod auto_up;
pub mod engine;
pub mod loader;
#[cfg(feature = "mock")]
pub mod mock;
pub mod ops;
pub mod pci;
pub mod runtime;
pub mod tiers;
pub mod vulkan;

// Vendors.
pub mod amd;
pub mod intel;
pub mod npu;
pub mod nvidia;
pub mod tt;
pub mod v3d;

pub use engine::{Access, DeviceClass, Engine, FeatureSet};
pub use miraculix::formats::Overflow;
pub use ops::{Choice, GemmShape};
pub use runtime::{
    Bind, CommandList, DEFAULT_TIMEOUT, Device, DeviceBuffer, Error, Exactness, HostCopy, Pending,
    Placement, Result, Stream,
};
#[cfg(target_os = "linux")]
pub use runtime::{dmabuf, sync};
pub use tiers::Tier;
