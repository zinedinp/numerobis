//! Device buffers.

use std::marker::PhantomData;
#[cfg(target_os = "linux")]
use std::os::fd::OwnedFd;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use crate::elem::{Elem, Format, HostRepr, from_bytes};
use crate::runtime::backend::{Backend, BufId};

/// Owner of one backend allocation. Freed when the last `Arc` drops; streams
/// keep an `Arc` for every buffer used by in-flight work, so a buffer is never
/// freed under a running dispatch.
pub(crate) struct RawBuffer {
    pub(crate) backend: Arc<dyn Backend>,
    pub(crate) id: BufId,
    pub(crate) bytes: usize,
    /// Holds results of a tier that is not bit-exact (see [`Exactness`](crate::Exactness)).
    pub(crate) approx: AtomicBool,
}

impl RawBuffer {
    pub(crate) fn new(backend: Arc<dyn Backend>, id: BufId, bytes: usize) -> RawBuffer {
        RawBuffer {
            backend,
            id,
            bytes,
            approx: AtomicBool::new(false),
        }
    }
    pub(crate) fn is_approx(&self) -> bool {
        self.approx.load(Ordering::Acquire)
    }
    pub(crate) fn set_approx(&self, v: bool) {
        self.approx.store(v, Ordering::Release);
    }
}

impl Drop for RawBuffer {
    fn drop(&mut self) {
        self.backend.free(self.id);
    }
}

/// `len` elements of `E` in device memory.
///
/// There is no way to see the contents from the host except
/// [`Stream::download`](crate::Stream::download):
///
/// ```compile_fail
/// # use numerobis::{elem::F32, DeviceBuffer};
/// fn peek(b: &DeviceBuffer<F32>) -> f32 { b[0] }
/// ```
/// ```compile_fail
/// # use numerobis::{elem::F32, DeviceBuffer};
/// fn peek(b: &DeviceBuffer<F32>) -> &[f32] { b.as_slice() }
/// ```
/// ```compile_fail
/// # use numerobis::{elem::F32, DeviceBuffer};
/// fn peek(b: &DeviceBuffer<F32>) -> &[f32] { &*b }
/// ```
/// ```compile_fail
/// # use numerobis::{elem::F32, DeviceBuffer};
/// fn make(v: Vec<f32>) -> DeviceBuffer<F32> { v.into() }
/// ```
pub struct DeviceBuffer<E: Elem> {
    pub(crate) raw: Arc<RawBuffer>,
    len: usize,
    _e: PhantomData<fn() -> E>,
}

impl<E: Elem> DeviceBuffer<E> {
    pub(crate) fn new(raw: Arc<RawBuffer>, len: usize) -> Self {
        DeviceBuffer {
            raw,
            len,
            _e: PhantomData,
        }
    }

    /// Number of elements.
    pub fn len(&self) -> usize {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Device bytes, including block scales.
    pub fn bytes(&self) -> usize {
        self.raw.bytes
    }

    pub fn format(&self) -> Format {
        E::FORMAT
    }

    /// `false` once an op on a tier that is not bit-exact wrote it (directly
    /// or through an input). Such a buffer is fine for display, but an
    /// [`Exactness::Bit`](crate::Exactness) stream refuses to read it.
    /// [`Stream::upload`](crate::Stream::upload) makes it exact again.
    pub fn is_exact(&self) -> bool {
        !self.raw.is_approx()
    }

    /// Exports the buffer as a dma-buf (zero-copy sharing with Vulkan, KMS,
    /// V4L2, other accelerators). Synchronise with [`Pending::signal_timeline`](crate::Pending::signal_timeline)
    /// or a sync file before the consumer reads it. Linux only.
    #[cfg(target_os = "linux")]
    pub fn export_dmabuf(&self) -> crate::Result<OwnedFd> {
        self.raw.backend.export_dmabuf(self.raw.id)
    }
}

impl<E: Elem> core::fmt::Debug for DeviceBuffer<E> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(
            f,
            "DeviceBuffer<{}>(len {}, {} bytes)",
            E::NAME,
            self.len,
            self.raw.bytes
        )
    }
}

/// Host copy of a device buffer, produced only by [`Stream::download`](crate::Stream::download).
/// It is not a [`DeviceBuffer`], so it cannot be passed to an op:
///
/// ```compile_fail
/// # use numerobis::{elem::F32, HostCopy, DeviceBuffer, Stream, ops, Choice, Overflow};
/// fn f(s: &mut Stream, h: &HostCopy<F32>, d: &mut DeviceBuffer<F32>) {
///     let _ = ops::convert(s, Choice::Up, h, d, Overflow::Saturate);
/// }
/// ```
pub struct HostCopy<E: Elem> {
    bytes: Vec<u8>,
    len: usize,
    _e: PhantomData<fn() -> E>,
}

impl<E: Elem> HostCopy<E> {
    pub(crate) fn new(bytes: Vec<u8>, len: usize) -> Self {
        HostCopy {
            bytes,
            len,
            _e: PhantomData,
        }
    }

    pub fn len(&self) -> usize {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Raw device layout (see [`elem`](crate::elem)), for the miraculix codecs.
    pub fn as_bytes(&self) -> &[u8] {
        &self.bytes
    }

    pub fn into_bytes(self) -> Vec<u8> {
        self.bytes
    }
}

impl<E: HostRepr> HostCopy<E> {
    pub fn to_vec(&self) -> Vec<E::Host> {
        from_bytes(&self.bytes)
    }
}
