//! One error type for every backend (plan C.3).

use core::fmt;
use std::time::Duration;

/// Result alias.
pub type Result<T, E = Error> = core::result::Result<T, E>;

/// Errors from devices, streams and ops.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum Error {
    /// The op, format or tier is not available on this device.
    Unsupported(String),
    /// Device memory exhausted.
    OutOfMemory { requested: usize },
    /// A wait did not finish in time. The work may still complete later.
    Timeout(Duration),
    /// GPU reset or NPU firmware crash; all buffers of the device are invalid.
    DeviceLost,
    /// Required firmware is missing; `package` names what to install.
    MissingFirmware { package: String },
    /// The kernel driver is older than the backend needs.
    DriverTooOld { have: String, need: String },
    /// No device matched the placement request.
    NoDevice,
    /// Argument check failed before anything was submitted.
    InvalidArgument(String),
    /// Malformed kernel binary.
    BadBinary(String),
    /// A raw syscall/ioctl failed with this errno.
    Os(i32),
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Error::Unsupported(s) => write!(f, "unsupported: {s}"),
            Error::OutOfMemory { requested } => write!(f, "out of device memory ({requested} bytes requested)"),
            Error::Timeout(d) => write!(f, "timed out after {d:?}"),
            Error::DeviceLost => f.write_str("device lost"),
            Error::MissingFirmware { package } => write!(f, "missing firmware (install `{package}`)"),
            Error::DriverTooOld { have, need } => write!(f, "driver too old: have {have}, need {need}"),
            Error::NoDevice => f.write_str("no matching device"),
            Error::InvalidArgument(s) => write!(f, "invalid argument: {s}"),
            Error::BadBinary(s) => write!(f, "bad kernel binary: {s}"),
            Error::Os(e) => write!(f, "os error {e}"),
        }
    }
}

impl std::error::Error for Error {}

pub(crate) fn invalid(msg: impl Into<String>) -> Error {
    Error::InvalidArgument(msg.into())
}
