//! Vendor-neutral runtime: the `Backend` trait every target implements, and
//! the API on top of it (devices, streams, buffers, errors).

pub mod abi;
pub mod backend;
pub(crate) mod buffer;
pub(crate) mod device;
#[cfg(target_os = "linux")]
pub(crate) mod dmabuf;
pub(crate) mod error;
#[cfg(target_os = "linux")]
#[allow(dead_code)]
pub(crate) mod os;
pub(crate) mod record;
pub(crate) mod stream;
#[cfg(target_os = "linux")]
pub mod sync;

pub use backend::{ActKind, Arg, Backend, BufId, Fence, KernelId, KernelKey, OpSig, ResizeMode};
pub use buffer::{DeviceBuffer, HostCopy};
pub use device::{Device, Placement};
pub use error::{Error, Result};
pub use record::{Bind, CommandList};
pub use stream::{DEFAULT_TIMEOUT, Exactness, Pending, Stream};
