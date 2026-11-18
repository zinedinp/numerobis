//! Streams, pending results and the host <-> device boundary.

use std::cell::Cell;
use std::marker::PhantomData;
use std::sync::Arc;
use std::time::Duration;

use crate::elem::{Elem, HostRepr, as_bytes};
use crate::runtime::backend::{Arg, Backend, Fence, KernelKey};
use crate::runtime::buffer::{DeviceBuffer, HostCopy, RawBuffer};
use crate::runtime::device::Device;
use crate::runtime::error::{Error, Result, invalid};
use crate::runtime::record::{Cmd, CommandList, Recorder};

/// Default timeout for waits that do not name one.
pub const DEFAULT_TIMEOUT: Duration = Duration::from_secs(30);

/// What a stream may trade for speed.
///
/// Tiers that are not bit-exact (XMX/MFMA/Tensix-FPU GEMM, native `exp`) are
/// fine for anything that is only displayed: rendering, upscaling, denoising,
/// effects. They are never fine for state that several machines or frames
/// must agree on: client prediction, physics, server reconciliation, replays.
/// Such work runs on an [`Exactness::Bit`] stream, which
/// - picks only tiers on which the op is bit-identical to the oracle
///   ([`OpSig::exact_on`](crate::runtime::OpSig::exact_on)), and
/// - refuses inputs that an approximate op has written
///   ([`DeviceBuffer::is_exact`]), so display results cannot leak into state.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Exactness {
    /// Bit-identical results on every device and tier (default).
    #[default]
    Bit,
    /// Any tier; results may differ in the last bits between devices.
    Approx,
}

/// Work still running on the device. Must be waited on or dropped
/// deliberately; dropping does not cancel the work (the stream keeps the
/// buffers alive until it retires).
#[must_use = "device work is asynchronous; call .wait() to get the result"]
pub struct Pending<T> {
    backend: Arc<dyn Backend>,
    /// `None`: nothing was submitted (recorded launch, already complete).
    fence: Option<Fence>,
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
        if let Some(f) = self.fence {
            self.backend.wait(f, timeout)?;
        }
        (self.finish)(&*self.backend)
    }

    /// Waits and also returns the device time of the work.
    pub fn wait_elapsed(self) -> Result<(T, Option<Duration>)> {
        let el = match self.fence {
            Some(f) => {
                self.backend.wait(f, self.timeout)?;
                self.backend.elapsed(f)
            }
            None => None,
        };
        Ok(((self.finish)(&*self.backend)?, el))
    }

    /// Device time, once the work has finished; `None` before or if the backend has no timestamps.
    pub fn elapsed(&self) -> Option<Duration> {
        self.backend.elapsed(self.fence?)
    }

    /// Signals `timeline` at `value` once this work is done, without blocking:
    /// the hand-off to a Vulkan timeline semaphore (see [`crate::sync`]).
    #[cfg(target_os = "linux")]
    pub fn signal_timeline(&self, timeline: &crate::sync::Timeline, value: u64) -> Result<()> {
        crate::runtime::sync::signal_after(self.backend.clone(), self.fence, timeline, value)
    }

    /// A kernel `sync_file` for this work, on backends whose driver has
    /// dma-fences; others report `Unsupported` (use
    /// [`signal_timeline`](Self::signal_timeline), which works everywhere).
    #[cfg(target_os = "linux")]
    pub fn sync_file(&self) -> Result<crate::sync::SyncFile> {
        match self.fence {
            Some(f) => self
                .backend
                .export_sync_file(f)
                .map(crate::sync::SyncFile::from_fd)
                .ok_or_else(|| {
                    Error::Unsupported(format!(
                        "{} has no kernel dma-fences; use Pending::signal_timeline",
                        self.backend.features().name
                    ))
                }),
            None => {
                // Nothing in flight: an already-signalled sync_file.
                let tl = crate::sync::Timeline::new()?;
                tl.signal(1)?;
                tl.sync_file(1)
            }
        }
    }
}

pub(crate) fn ready(backend: Arc<dyn Backend>, timeout: Duration) -> Pending<()> {
    Pending {
        backend,
        fence: None,
        timeout,
        finish: Box::new(|_| Ok(())),
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
    exactness: Exactness,
    recorder: Option<Recorder>,
    #[cfg(target_os = "linux")]
    deps: Vec<crate::runtime::sync::Dep>,
    _not_sync: PhantomData<Cell<()>>,
}

const RETIRE_AT: usize = 64;

impl Stream {
    pub(crate) fn new(dev: Device) -> Stream {
        Stream {
            dev,
            inflight: Vec::new(),
            timeout: DEFAULT_TIMEOUT,
            exactness: Exactness::default(),
            recorder: None,
            #[cfg(target_os = "linux")]
            deps: Vec::new(),
            _not_sync: PhantomData,
        }
    }

    pub fn device(&self) -> &Device {
        &self.dev
    }

    pub fn set_timeout(&mut self, timeout: Duration) {
        self.timeout = timeout;
    }

    pub fn exactness(&self) -> Exactness {
        self.exactness
    }

    /// See [`Exactness`]. Streams start as [`Exactness::Bit`].
    pub fn set_exactness(&mut self, e: Exactness) {
        self.exactness = e;
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
        self.inflight
            .retain(|f| b.wait(f.fence, Duration::ZERO).is_err());
    }

    /// Resolves host-side dependencies before the next submission.
    fn before_submit(&mut self) -> Result<()> {
        #[cfg(target_os = "linux")]
        while let Some(d) = self.deps.first() {
            d.wait(self.timeout)?;
            self.deps.remove(0);
        }
        Ok(())
    }

    /// The next submission of this stream waits for `sf` (e.g. the
    /// `SYNC_FD` of a Vulkan semaphore after the render pass that wrote an
    /// imported dma-buf).
    #[cfg(target_os = "linux")]
    pub fn wait_sync_file(&mut self, sf: crate::sync::SyncFile) {
        self.deps.push(crate::runtime::sync::Dep::SyncFile(sf));
    }

    /// The next submission of this stream waits until `timeline` reaches `value`.
    #[cfg(target_os = "linux")]
    pub fn wait_timeline(&mut self, timeline: &crate::sync::Timeline, value: u64) {
        self.deps
            .push(crate::runtime::sync::Dep::Timeline(timeline.clone(), value));
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
        let bytes = E::FORMAT.storage_bytes(len).ok_or_else(|| {
            invalid(format!(
                "{} needs a length that is a multiple of {}",
                E::NAME,
                E::FORMAT.block()
            ))
        })?;
        let id = self.backend().alloc(bytes.max(1))?;
        let raw = Arc::new(RawBuffer::new(self.backend().clone(), id, bytes));
        Ok(DeviceBuffer::new(raw, len))
    }

    /// Imports a dma-buf (from Vulkan, KMS, V4L2, another accelerator) as a
    /// buffer of `len` elements, without a copy. The contents count as exact.
    #[cfg(target_os = "linux")]
    pub fn import_dmabuf<E: Elem>(
        &mut self,
        fd: std::os::fd::BorrowedFd<'_>,
        len: usize,
    ) -> Result<DeviceBuffer<E>> {
        let bytes = E::FORMAT.storage_bytes(len).ok_or_else(|| {
            invalid(format!(
                "{} needs a length that is a multiple of {}",
                E::NAME,
                E::FORMAT.block()
            ))
        })?;
        let have = crate::runtime::dmabuf::size(fd)?;
        if have < bytes {
            return Err(invalid(format!(
                "dma-buf has {have} bytes, {len} × {} needs {bytes}",
                E::NAME
            )));
        }
        let id = self.backend().import_dmabuf(fd, bytes)?;
        let raw = Arc::new(RawBuffer::new(self.backend().clone(), id, bytes));
        Ok(DeviceBuffer::new(raw, len))
    }

    /// Host -> device for byte-aligned formats.
    pub fn upload<E: HostRepr>(&mut self, data: &[E::Host]) -> Result<DeviceBuffer<E>> {
        self.upload_packed(data.len(), as_bytes(data))
    }

    /// Host -> device from bytes already in device layout (packed sub-byte
    /// formats, block formats; produce them with the miraculix codecs).
    pub fn upload_packed<E: Elem>(&mut self, len: usize, bytes: &[u8]) -> Result<DeviceBuffer<E>> {
        let buf = self.alloc::<E>(len)?;
        self.write_packed(&buf, bytes)?;
        Ok(buf)
    }

    /// Overwrites an existing buffer from the host (per-frame inputs of a
    /// [`CommandList`]). The buffer counts as exact afterwards.
    pub fn write<E: HostRepr>(&mut self, buf: &DeviceBuffer<E>, data: &[E::Host]) -> Result<()> {
        self.write_packed(buf, as_bytes(data))
    }

    /// [`write`](Self::write) from bytes in device layout.
    pub fn write_packed<E: Elem>(&mut self, buf: &DeviceBuffer<E>, bytes: &[u8]) -> Result<()> {
        self.check_same_device(&buf.raw)?;
        if bytes.len() != buf.bytes() {
            return Err(invalid(format!(
                "{} × {} needs {} bytes, got {}",
                buf.len(),
                E::NAME,
                buf.bytes(),
                bytes.len()
            )));
        }
        self.before_submit()?;
        let fence = self.backend().upload(buf.raw.id, 0, bytes)?;
        buf.raw.set_approx(false);
        self.track(fence, vec![buf.raw.clone()]);
        Ok(())
    }

    /// Device -> host.
    pub fn download<E: Elem>(&mut self, buf: &DeviceBuffer<E>) -> Result<Pending<HostCopy<E>>> {
        self.check_same_device(&buf.raw)?;
        if self.recorder.is_some() {
            return Err(invalid("download while recording a command list"));
        }
        self.before_submit()?;
        let (bytes, len) = (buf.bytes(), buf.len());
        let fence = self.backend().download(buf.raw.id, 0, bytes)?;
        self.track(fence, vec![buf.raw.clone()]);
        Ok(Pending {
            backend: self.backend().clone(),
            fence: Some(fence),
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
            Err(invalid(
                "buffer belongs to a different device (use share_with)",
            ))
        }
    }

    fn check_exact_inputs(&self, name: &str, reads: &[&Arc<RawBuffer>]) -> Result<()> {
        if self.exactness == Exactness::Bit && reads.iter().any(|b| b.is_approx()) {
            return Err(invalid(format!(
                "{name}: an input holds approximate (display-only) results; an Exactness::Bit stream cannot read it"
            )));
        }
        Ok(())
    }

    /// Loads (cached) and dispatches a kernel. `reads` / `writes` are the
    /// buffers the args reference; they stay alive until the work retires,
    /// and `writes` inherit the exactness of the tier and of `reads`.
    pub(crate) fn launch(
        &mut self,
        key: &KernelKey,
        reads: &[&Arc<RawBuffer>],
        writes: &[&Arc<RawBuffer>],
        args: &[Arg],
    ) -> Result<Pending<()>> {
        for b in reads.iter().chain(writes) {
            self.check_same_device(b)?;
        }
        let k = self.dev.kernel(key)?;
        let approx_op = !key.sig.exact_on(key.tier);
        if let Some(r) = self.recorder.as_mut() {
            r.push(Cmd::new(k, args, reads, writes, approx_op));
            return Ok(ready(self.backend().clone(), self.timeout));
        }
        self.check_exact_inputs(&key.name, reads)?;
        self.before_submit()?;
        let approx = approx_op || reads.iter().any(|b| b.is_approx());
        let fence = self.backend().dispatch(k, args)?;
        for w in writes {
            w.set_approx(approx);
        }
        let keep = reads.iter().chain(writes).map(|b| Arc::clone(b)).collect();
        self.track(fence, keep);
        Ok(Pending {
            backend: self.backend().clone(),
            fence: Some(fence),
            timeout: self.timeout,
            finish: Box::new(|_| Ok(())),
        })
    }

    // ---- reusable command lists ----

    /// Starts recording: ops called on this stream are translated (kernel
    /// lookup, lowering, argument checks) but not run. Uploads still run.
    pub fn begin_recording(&mut self) -> Result<()> {
        if self.recorder.is_some() {
            return Err(invalid("already recording"));
        }
        self.recorder = Some(Recorder::default());
        Ok(())
    }

    /// Ends recording and returns the command list.
    pub fn end_recording(&mut self) -> Result<CommandList> {
        let r = self
            .recorder
            .take()
            .ok_or_else(|| invalid("not recording"))?;
        Ok(r.finish(self.dev.clone()))
    }

    /// `begin_recording`, `f`, `end_recording`.
    pub fn record(&mut self, f: impl FnOnce(&mut Stream) -> Result<()>) -> Result<CommandList> {
        self.begin_recording()?;
        let r = f(self);
        let cl = self.end_recording()?;
        r.map(|()| cl)
    }

    /// Runs a recorded command list, with buffers swapped per
    /// [`Bind`](crate::Bind) (same element type and length): translate once,
    /// then per frame only exchange buffers.
    pub fn replay(&mut self, cl: &CommandList, binds: &[crate::Bind<'_>]) -> Result<Pending<()>> {
        if !Arc::ptr_eq(&cl.dev.inner.backend, self.backend()) {
            return Err(invalid("command list was recorded on a different device"));
        }
        if self.recorder.is_some() {
            return Err(invalid("replay while recording"));
        }
        let plan = cl.plan(binds, self.exactness)?;
        self.before_submit()?;
        let cmds: Vec<_> = plan.cmds.iter().map(|(k, a)| (*k, a.as_slice())).collect();
        let fence = self.backend().dispatch_batch(&cmds)?;
        for (b, approx) in &plan.final_approx {
            b.set_approx(*approx);
        }
        self.track(fence, plan.keep);
        Ok(Pending {
            backend: self.backend().clone(),
            fence: Some(fence),
            timeout: self.timeout,
            finish: Box::new(|_| Ok(())),
        })
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

#[allow(dead_code)]
fn _unused(_: Error) {}
