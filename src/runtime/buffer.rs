//! Device buffers (plan A.4, C.1).

use std::marker::PhantomData;
use std::sync::Arc;

use crate::backend::{Backend, BufId};
use crate::elem::{Elem, Format, HostRepr, from_bytes};

/// Owner of one backend allocation. Freed when the last `Arc` drops; streams
/// keep an `Arc` for every buffer used by in-flight work, so a buffer is never
/// freed under a running dispatch.
pub(crate) struct RawBuffer {
    pub(crate) backend: Arc<dyn Backend>,
    pub(crate) id: BufId,
    pub(crate) bytes: usize,
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
        DeviceBuffer { raw, len, _e: PhantomData }
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
}

impl<E: Elem> core::fmt::Debug for DeviceBuffer<E> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "DeviceBuffer<{}>(len {}, {} bytes)", E::NAME, self.len, self.raw.bytes)
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
        HostCopy { bytes, len, _e: PhantomData }
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
