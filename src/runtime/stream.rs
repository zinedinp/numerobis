//! Streams, pending results and the host ↔ device boundary (plan A.4, C.1, C.2).

use std::cell::Cell;
use std::marker::PhantomData;
use std::sync::Arc;
use std::time::Duration;

use crate::runtime::backend::{Arg, Backend, Fence, KernelKey};
use crate::runtime::buffer::{DeviceBuffer, HostCopy, RawBuffer};
use crate::runtime::device::Device;
use crate::elem::{Elem, HostRepr, as_bytes};
use crate::runtime::error::{Result, invalid};

/// Default timeout for waits that do not name one.
pub const DEFAULT_TIMEOUT: Duration = Duration::from_secs(30);

/// Work still running on the device. Must be waited on or dropped
/// deliberately; dropping does not cancel the work (the stream keeps the
/// buffers alive until it retires).
#[must_use = "device work is asynchronous; call .wait() to get the result"]
pub struct Pending<T> {
    backend: Arc<dyn Backend>,
    fence: Fence,
    timeout: Duration,
    finish: Finish<T>,
}

/// Runs on the host after the fence (e.g. copies staged download bytes out).
type Finish<T> = Box<dyn FnOnce(&dyn Backend) -> Result<T> + Send>;

impl<T> Pending<T> {
    /// Waits with the stream's default timeout.
    pub fn wait(self) -> Result<T> {
        let t = self.timeout;
        self.wait_timeout(t)
    }

    pub fn wait_timeout(self, timeout: Duration) -> Result<T> {
        self.backend.wait(self.fence, timeout)?;
        (self.finish)(&*self.backend)
    }

    /// Waits and also returns the device time of the work (plan C.11).
    pub fn wait_elapsed(self) -> Result<(T, Option<Duration>)> {
        self.backend.wait(self.fence, self.timeout)?;
        let el = self.backend.elapsed(self.fence);
        Ok(((self.finish)(&*self.backend)?, el))
    }

    /// Device time, once the work has finished; `None` before or if the backend has no timestamps.
    pub fn elapsed(&self) -> Option<Duration> {
        self.backend.elapsed(self.fence)
    }
}

struct InFlight {
    fence: Fence,
    _keep: Vec<Arc<RawBuffer>>,
}

/// An ordered queue of work on one device. `Send` but not `Sync`: one thread
/// drives a stream; use several streams for concurrency.
pub struct Stream {
    dev: Device,
    inflight: Vec<InFlight>,
    timeout: Duration,
    _not_sync: PhantomData<Cell<()>>,
}

const RETIRE_AT: usize = 64;

impl Stream {
    pub(crate) fn new(dev: Device) -> Stream {
        Stream { dev, inflight: Vec::new(), timeout: DEFAULT_TIMEOUT, _not_sync: PhantomData }
    }

    pub fn device(&self) -> &Device {
        &self.dev
    }

    pub fn set_timeout(&mut self, timeout: Duration) {
        self.timeout = timeout;
    }

    fn backend(&self) -> &Arc<dyn Backend> {
        &self.dev.inner.backend
    }

    fn track(&mut self, fence: Fence, keep: Vec<Arc<RawBuffer>>) {
        self.inflight.push(InFlight { fence, _keep: keep });
        if self.inflight.len() >= RETIRE_AT {
            self.retire_finished();
        }
    }

    /// Drops buffer references of work that has already finished.
    fn retire_finished(&mut self) {
        let b = self.dev.inner.backend.clone();
        self.inflight.retain(|f| b.wait(f.fence, Duration::ZERO).is_err());
    }

    /// Waits for all submitted work.
    pub fn synchronize(&mut self) -> Result<()> {
        let b = self.backend().clone();
        while let Some(f) = self.inflight.first() {
            b.wait(f.fence, self.timeout)?;
            self.inflight.remove(0);
        }
        Ok(())
    }

    /// Allocates `len` elements. The contents are unspecified.
    pub fn alloc<E: Elem>(&mut self, len: usize) -> Result<DeviceBuffer<E>> {
        let bytes = E::FORMAT
            .storage_bytes(len)
            .ok_or_else(|| invalid(format!("{} needs a length that is a multiple of {}", E::NAME, E::FORMAT.block())))?;
        let id = self.backend().alloc(bytes.max(1))?;
        let raw = Arc::new(RawBuffer { backend: self.backend().clone(), id, bytes });
        Ok(DeviceBuffer::new(raw, len))
    }

    /// Host → device for byte-aligned formats.
    pub fn upload<E: HostRepr>(&mut self, data: &[E::Host]) -> Result<DeviceBuffer<E>> {
        self.upload_packed(data.len(), as_bytes(data))
    }

    /// Host → device from bytes already in device layout (packed sub-byte
    /// formats, block formats; produce them with the miraculix codecs).
    pub fn upload_packed<E: Elem>(&mut self, len: usize, bytes: &[u8]) -> Result<DeviceBuffer<E>> {
        let buf = self.alloc::<E>(len)?;
        if bytes.len() != buf.bytes() {
            return Err(invalid(format!("{} × {} needs {} bytes, got {}", len, E::NAME, buf.bytes(), bytes.len())));
        }
        let fence = self.backend().upload(buf.raw.id, 0, bytes)?;
        self.track(fence, vec![buf.raw.clone()]);
        Ok(buf)
    }

    /// Device → host.
    pub fn download<E: Elem>(&mut self, buf: &DeviceBuffer<E>) -> Result<Pending<HostCopy<E>>> {
        self.check_same_device(&buf.raw)?;
        let (bytes, len) = (buf.bytes(), buf.len());
        let fence = self.backend().download(buf.raw.id, 0, bytes)?;
        self.track(fence, vec![buf.raw.clone()]);
        Ok(Pending {
            backend: self.backend().clone(),
            fence,
            timeout: self.timeout,
            finish: Box::new(move |b| {
                let mut v = vec![0u8; bytes];
                b.take_download(fence, &mut v)?;
                Ok(HostCopy::new(v, len))
            }),
        })
    }

    fn check_same_device(&self, raw: &RawBuffer) -> Result<()> {
        if Arc::ptr_eq(&raw.backend, self.backend()) {
            Ok(())
        } else {
            Err(invalid("buffer belongs to a different device (use share_with)"))
        }
    }

    /// Loads (cached) and dispatches a kernel. `bufs` are the buffers the
    /// args reference; they stay alive until the work retires.
    pub(crate) fn launch(&mut self, key: &KernelKey, bufs: &[&Arc<RawBuffer>], args: &[Arg]) -> Result<Pending<()>> {
        for b in bufs {
            self.check_same_device(b)?;
        }
        let k = self.dev.kernel(key)?;
        let fence = self.backend().dispatch(k, args)?;
        self.track(fence, bufs.iter().map(|b| Arc::clone(b)).collect());
        Ok(Pending { backend: self.backend().clone(), fence, timeout: self.timeout, finish: Box::new(|_| Ok(())) })
    }
}

impl Drop for Stream {
    fn drop(&mut self) {
        if self.synchronize().is_err() {
            // Work did not finish (timeout / device lost): never free memory a
            // running dispatch may still touch. Leak instead.
            for f in self.inflight.drain(..) {
                core::mem::forget(f);
            }
        }
    }
}

#[allow(dead_code)]
fn _assert_send() {
    fn send<T: Send>() {}
    send::<Stream>();
    send::<Pending<HostCopy<crate::elem::F32>>>();
}
