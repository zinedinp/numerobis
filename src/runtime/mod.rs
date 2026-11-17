//! Vendor-neutral runtime: the `Backend` trait every target implements, and
//! the API on top of it (devices, streams, buffers, errors).

pub mod abi;
pub mod backend;
pub(crate) mod buffer;
pub(crate) mod device;
pub(crate) mod error;
pub(crate) mod stream;

pub use backend::{Arg, Backend, BufId, Fence, KernelId, KernelKey, OpSig};
pub use buffer::{DeviceBuffer, HostCopy};
pub use device::{Device, Placement};
pub use error::{Error, Result};
pub use stream::{DEFAULT_TIMEOUT, Pending, Stream};
