//! Cross-API synchronisation: kernel `sync_file`s and DRM timeline syncobjs,
//! the two objects Vulkan semaphores import and export on Linux.
//!
//! | numerobis | Vulkan |
//! |---|---|
//! | [`SyncFile`] | `VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_SYNC_FD_BIT` (binary semaphore / fence payload) |
//! | [`Timeline`] | timeline `VkSemaphore` imported/exported as `OPAQUE_FD` on drivers whose semaphores are DRM syncobjs (Mesa: RADV, ANV, NVK, Turnip, V3DV; NVIDIA ≥ 555 for explicit sync) |
//!
//! Direction device → Vulkan: [`Pending::signal_timeline`](crate::Pending::signal_timeline)
//! signals a point once the work is done. Backends whose kernel driver has
//! dma-fences return a real `sync_file` from [`Pending::sync_file`](crate::Pending::sync_file);
//! for the others (tt-kmd, NVIDIA RM, mock) a helper thread waits for the
//! device fence and signals the syncobj point from the host. Vulkan timeline
//! semaphores allow waiting before the signal is submitted, so this works
//! without a kernel fence.
//!
//! Direction Vulkan → device: [`Stream::wait_sync_file`](crate::Stream::wait_sync_file) and
//! [`Stream::wait_timeline`](crate::Stream::wait_timeline) make the next
//! submission of the stream wait for the point.

use std::os::fd::{AsFd, AsRawFd, BorrowedFd, FromRawFd, OwnedFd, RawFd};
use std::sync::mpsc::{Sender, channel};
use std::sync::{Arc, Mutex, OnceLock};
use std::time::{Duration, Instant};

use crate::runtime::backend::{Backend, Fence};
use crate::runtime::error::{Error, Result, invalid};
use crate::runtime::os::{Fd, errno, iowr};

/// A kernel `sync_file` (one dma-fence). Signalled once the fence is.
#[derive(Debug)]
pub struct SyncFile(OwnedFd);

impl SyncFile {
    /// Takes ownership of a `sync_file` descriptor (e.g. from `vkGetSemaphoreFdKHR` with `SYNC_FD`).
    #[must_use]
    pub fn from_fd(fd: OwnedFd) -> SyncFile {
        SyncFile(fd)
    }
    #[must_use]
    pub fn into_fd(self) -> OwnedFd {
        self.0
    }

    /// Non-blocking check.
    ///
    /// # Errors
    ///
    /// [`Error::Os`] when the descriptor cannot be polled.
    pub fn is_signaled(&self) -> Result<bool> {
        poll_in(self.0.as_raw_fd(), Duration::ZERO)
    }

    /// Blocks until signalled.
    ///
    /// # Errors
    ///
    /// [`Error::Timeout`] when it has not signalled in time,
    /// [`Error::Os`] when the descriptor cannot be polled.
    pub fn wait(&self, timeout: Duration) -> Result<()> {
        if poll_in(self.0.as_raw_fd(), timeout)? {
            Ok(())
        } else {
            Err(Error::Timeout(timeout))
        }
    }
}

impl AsFd for SyncFile {
    fn as_fd(&self) -> BorrowedFd<'_> {
        self.0.as_fd()
    }
}

fn poll_in(fd: RawFd, timeout: Duration) -> Result<bool> {
    let deadline = Instant::now() + timeout;
    loop {
        let left = deadline.saturating_duration_since(Instant::now());
        // Rounded up, so a short wait never becomes a busy poll(0).
        let ceil = left.as_millis() + u128::from(!left.subsec_nanos().is_multiple_of(1_000_000));
        let ms = i32::try_from(ceil).unwrap_or(i32::MAX);
        let mut p = libc::pollfd {
            fd,
            events: libc::POLLIN,
            revents: 0,
        };
        // SAFETY: one valid pollfd.
        let r = unsafe { libc::poll(&raw mut p, 1, ms) };
        if r > 0 {
            return Ok(
                p.revents & libc::POLLIN != 0 || p.revents & (libc::POLLERR | libc::POLLHUP) == 0
            );
        }
        if r == 0 {
            return Ok(false);
        }
        if errno() != libc::EINTR {
            return Err(Error::Os(errno()));
        }
    }
}

// ---- DRM syncobj UAPI (include/uapi/drm/drm.h) ----

#[repr(C)]
#[derive(Default)]
struct GetCap {
    capability: u64,
    value: u64,
}
#[repr(C)]
#[derive(Default)]
struct SyncobjCreate {
    handle: u32,
    flags: u32,
}
#[repr(C)]
#[derive(Default)]
struct SyncobjDestroy {
    handle: u32,
    pad: u32,
}
#[repr(C)]
#[derive(Default)]
struct SyncobjHandle {
    handle: u32,
    flags: u32,
    fd: i32,
    pad: u32,
    point: u64,
}
#[repr(C)]
#[derive(Default)]
struct SyncobjTransfer {
    src_handle: u32,
    dst_handle: u32,
    src_point: u64,
    dst_point: u64,
    flags: u32,
    pad: u32,
}
#[repr(C)]
#[derive(Default)]
struct SyncobjTimelineWait {
    handles: u64,
    points: u64,
    timeout_nsec: i64,
    count_handles: u32,
    flags: u32,
    first_signaled: u32,
    pad: u32,
    deadline_nsec: u64,
}
#[repr(C)]
#[derive(Default)]
struct SyncobjTimelineArray {
    handles: u64,
    points: u64,
    count_handles: u32,
    flags: u32,
}
const _: () = {
    assert!(size_of::<SyncobjHandle>() == 24);
    assert!(size_of::<SyncobjTransfer>() == 32);
    assert!(size_of::<SyncobjTimelineWait>() == 48);
    assert!(size_of::<SyncobjTimelineArray>() == 24);
};

const DRM: u8 = b'd';
const IOCTL_GET_CAP: u64 = iowr::<GetCap>(DRM, 0x0C);
const IOCTL_SYNCOBJ_CREATE: u64 = iowr::<SyncobjCreate>(DRM, 0xBF);
const IOCTL_SYNCOBJ_DESTROY: u64 = iowr::<SyncobjDestroy>(DRM, 0xC0);
const IOCTL_SYNCOBJ_HANDLE_TO_FD: u64 = iowr::<SyncobjHandle>(DRM, 0xC1);
const IOCTL_SYNCOBJ_FD_TO_HANDLE: u64 = iowr::<SyncobjHandle>(DRM, 0xC2);
const IOCTL_SYNCOBJ_TIMELINE_WAIT: u64 = iowr::<SyncobjTimelineWait>(DRM, 0xCA);
const IOCTL_SYNCOBJ_QUERY: u64 = iowr::<SyncobjTimelineArray>(DRM, 0xCB);
const IOCTL_SYNCOBJ_TRANSFER: u64 = iowr::<SyncobjTransfer>(DRM, 0xCC);
const IOCTL_SYNCOBJ_TIMELINE_SIGNAL: u64 = iowr::<SyncobjTimelineArray>(DRM, 0xCD);
const CAP_SYNCOBJ_TIMELINE: u64 = 0x14;
const EXPORT_SYNC_FILE: u32 = 1;
const IMPORT_SYNC_FILE: u32 = 1;
const WAIT_ALL: u32 = 1;
const WAIT_FOR_SUBMIT: u32 = 2;

/// The DRM node that holds syncobjs: `NUMEROBIS_SYNCOBJ_NODE`, else the first
/// render node with timeline support. Syncobjs are DRM-core objects; the
/// GPU behind the node does no work for them.
fn drm_node() -> Result<Arc<Fd>> {
    static NODE: OnceLock<std::result::Result<Arc<Fd>, Error>> = OnceLock::new();
    NODE.get_or_init(|| {
        let mut paths: Vec<String> = std::env::var("NUMEROBIS_SYNCOBJ_NODE")
            .into_iter()
            .collect();
        paths.extend((128..192).map(|i| format!("/dev/dri/renderD{i}")));
        for p in paths {
            let Ok(fd) = Fd::open(&p) else { continue };
            let mut cap = GetCap {
                capability: CAP_SYNCOBJ_TIMELINE,
                value: 0,
            };
            // SAFETY: GetCap matches DRM_IOCTL_GET_CAP.
            if unsafe { fd.ioctl(IOCTL_GET_CAP, &raw mut cap) }.is_ok() && cap.value != 0 {
                return Ok(Arc::new(fd));
            }
        }
        Err(Error::Unsupported(
            "no DRM render node with timeline syncobjs (set NUMEROBIS_SYNCOBJ_NODE)".into(),
        ))
    })
    .clone()
}

fn monotonic_ns_after(d: Duration) -> i64 {
    let mut ts = libc::timespec {
        tv_sec: 0,
        tv_nsec: 0,
    };
    // SAFETY: valid out pointer.
    unsafe { libc::clock_gettime(libc::CLOCK_MONOTONIC, &raw mut ts) };
    let now = i128::from(ts.tv_sec) * 1_000_000_000 + i128::from(ts.tv_nsec);
    let after = now.saturating_add(i128::try_from(d.as_nanos()).unwrap_or(i128::MAX));
    i64::try_from(after).unwrap_or(i64::MAX)
}

/// A DRM timeline syncobj: a 64-bit counter that only grows, the kernel
/// object behind Vulkan timeline semaphores. Cheap to clone.
#[derive(Clone)]
pub struct Timeline {
    inner: Arc<TimelineInner>,
}

pub(crate) struct TimelineInner {
    node: Arc<Fd>,
    handle: u32,
}

impl Drop for TimelineInner {
    fn drop(&mut self) {
        let mut d = SyncobjDestroy {
            handle: self.handle,
            pad: 0,
        };
        // SAFETY: we own the handle.
        let _ = unsafe { self.node.ioctl(IOCTL_SYNCOBJ_DESTROY, &raw mut d) };
    }
}

impl core::fmt::Debug for Timeline {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(
            f,
            "Timeline(handle {}, value {:?})",
            self.inner.handle,
            self.value().ok()
        )
    }
}

impl Timeline {
    /// A new timeline at 0.
    ///
    /// # Errors
    ///
    /// [`Error::Unsupported`] without a DRM render node that has
    /// timeline syncobjs, [`Error::Os`] when the kernel refuses the syncobj.
    pub fn new() -> Result<Timeline> {
        let node = drm_node()?;
        let mut c = SyncobjCreate::default();
        // SAFETY: matches DRM_IOCTL_SYNCOBJ_CREATE.
        unsafe { node.ioctl(IOCTL_SYNCOBJ_CREATE, &raw mut c) }?;
        Ok(Timeline {
            inner: Arc::new(TimelineInner {
                node,
                handle: c.handle,
            }),
        })
    }

    /// Imports a syncobj fd (from `vkGetSemaphoreFdKHR` with `OPAQUE_FD` on a
    /// syncobj-based driver, or [`export_fd`](Self::export_fd) of another process).
    ///
    /// # Errors
    ///
    /// [`Error::Unsupported`] without a DRM render node that has
    /// timeline syncobjs, [`Error::Os`] when `fd` is not a syncobj.
    pub fn import_fd(fd: BorrowedFd<'_>) -> Result<Timeline> {
        let node = drm_node()?;
        let mut h = SyncobjHandle {
            fd: fd.as_raw_fd(),
            ..Default::default()
        };
        // SAFETY: matches DRM_IOCTL_SYNCOBJ_FD_TO_HANDLE.
        unsafe { node.ioctl(IOCTL_SYNCOBJ_FD_TO_HANDLE, &raw mut h) }?;
        Ok(Timeline {
            inner: Arc::new(TimelineInner {
                node,
                handle: h.handle,
            }),
        })
    }

    /// A syncobj fd for `vkImportSemaphoreFdKHR` (`OPAQUE_FD`, timeline semaphore).
    ///
    /// # Errors
    ///
    /// [`Error::Os`] when the kernel refuses the export.
    pub fn export_fd(&self) -> Result<OwnedFd> {
        let mut h = SyncobjHandle {
            handle: self.inner.handle,
            fd: -1,
            ..Default::default()
        };
        // SAFETY: matches DRM_IOCTL_SYNCOBJ_HANDLE_TO_FD.
        unsafe {
            self.inner
                .node
                .ioctl(IOCTL_SYNCOBJ_HANDLE_TO_FD, &raw mut h)
        }?;
        // SAFETY: the kernel returned a new descriptor we own.
        Ok(unsafe { OwnedFd::from_raw_fd(h.fd) })
    }

    /// Signals `value` from the host (must be above the current value).
    ///
    /// # Errors
    ///
    /// [`Error::Os`] when the kernel refuses the signal.
    pub fn signal(&self, value: u64) -> Result<()> {
        let (h, mut p) = (self.inner.handle, value);
        let mut a = SyncobjTimelineArray {
            handles: &raw const h as u64,
            points: &raw mut p as u64,
            count_handles: 1,
            flags: 0,
        };
        // SAFETY: arrays of one, alive for the call.
        unsafe {
            self.inner
                .node
                .ioctl(IOCTL_SYNCOBJ_TIMELINE_SIGNAL, &raw mut a)
        }
    }

    /// Highest signalled value.
    ///
    /// # Errors
    ///
    /// [`Error::Os`] when the query fails.
    pub fn value(&self) -> Result<u64> {
        let (h, mut p) = (self.inner.handle, 0u64);
        let mut a = SyncobjTimelineArray {
            handles: &raw const h as u64,
            points: &raw mut p as u64,
            count_handles: 1,
            flags: 0,
        };
        // SAFETY: arrays of one, alive for the call.
        unsafe { self.inner.node.ioctl(IOCTL_SYNCOBJ_QUERY, &raw mut a) }?;
        Ok(p)
    }

    /// Blocks until `value` is signalled (also when its signal is submitted later).
    ///
    /// # Errors
    ///
    /// [`Error::Timeout`] when `value` is not signalled in time,
    /// [`Error::Os`] for other kernel errors.
    pub fn wait(&self, value: u64, timeout: Duration) -> Result<()> {
        let (h, p) = (self.inner.handle, value);
        let mut w = SyncobjTimelineWait {
            handles: &raw const h as u64,
            points: &raw const p as u64,
            timeout_nsec: monotonic_ns_after(timeout),
            count_handles: 1,
            flags: WAIT_ALL | WAIT_FOR_SUBMIT,
            ..Default::default()
        };
        // SAFETY: arrays of one, alive for the call.
        match unsafe {
            self.inner
                .node
                .ioctl(IOCTL_SYNCOBJ_TIMELINE_WAIT, &raw mut w)
        } {
            Err(Error::Os(libc::ETIME)) => Err(Error::Timeout(timeout)),
            r => r,
        }
    }

    /// The fence of point `value` as a `sync_file` (for `SYNC_FD` consumers).
    /// The point must already have a fence: signalled, or submitted by a
    /// driver; otherwise the kernel reports `ENOENT`/`EINVAL`.
    ///
    /// # Errors
    ///
    /// [`Error::Os`] when point `value` has no fence yet (`ENOENT`/`EINVAL`) or
    /// the export fails.
    pub fn sync_file(&self, value: u64) -> Result<SyncFile> {
        let tmp = Binary::new(&self.inner.node)?;
        let mut t = SyncobjTransfer {
            src_handle: self.inner.handle,
            dst_handle: tmp.0,
            src_point: value,
            ..Default::default()
        };
        // SAFETY: matches DRM_IOCTL_SYNCOBJ_TRANSFER.
        unsafe { self.inner.node.ioctl(IOCTL_SYNCOBJ_TRANSFER, &raw mut t) }?;
        let mut h = SyncobjHandle {
            handle: tmp.0,
            flags: EXPORT_SYNC_FILE,
            fd: -1,
            ..Default::default()
        };
        // SAFETY: matches DRM_IOCTL_SYNCOBJ_HANDLE_TO_FD.
        unsafe {
            self.inner
                .node
                .ioctl(IOCTL_SYNCOBJ_HANDLE_TO_FD, &raw mut h)
        }?;
        // SAFETY: new descriptor we own.
        Ok(SyncFile(unsafe { OwnedFd::from_raw_fd(h.fd) }))
    }

    /// Makes point `value` signal when `sf` does (Vulkan `SYNC_FD` → timeline).
    ///
    /// # Errors
    ///
    /// [`Error::Os`] when the kernel refuses the import.
    pub fn import_sync_file(&self, value: u64, sf: &SyncFile) -> Result<()> {
        let tmp = Binary::new(&self.inner.node)?;
        let mut h = SyncobjHandle {
            handle: tmp.0,
            flags: IMPORT_SYNC_FILE,
            fd: sf.0.as_raw_fd(),
            ..Default::default()
        };
        // SAFETY: matches DRM_IOCTL_SYNCOBJ_FD_TO_HANDLE.
        unsafe {
            self.inner
                .node
                .ioctl(IOCTL_SYNCOBJ_FD_TO_HANDLE, &raw mut h)
        }?;
        let mut t = SyncobjTransfer {
            src_handle: tmp.0,
            dst_handle: self.inner.handle,
            dst_point: value,
            ..Default::default()
        };
        // SAFETY: matches DRM_IOCTL_SYNCOBJ_TRANSFER.
        unsafe { self.inner.node.ioctl(IOCTL_SYNCOBJ_TRANSFER, &raw mut t) }
    }
}

/// A temporary binary syncobj.
struct Binary<'a>(u32, &'a Fd);

impl<'a> Binary<'a> {
    fn new(node: &'a Fd) -> Result<Binary<'a>> {
        let mut c = SyncobjCreate::default();
        // SAFETY: matches DRM_IOCTL_SYNCOBJ_CREATE.
        unsafe { node.ioctl(IOCTL_SYNCOBJ_CREATE, &raw mut c) }?;
        Ok(Binary(c.handle, node))
    }
}

impl Drop for Binary<'_> {
    fn drop(&mut self) {
        let mut d = SyncobjDestroy {
            handle: self.0,
            pad: 0,
        };
        // SAFETY: we own the handle.
        let _ = unsafe { self.1.ioctl(IOCTL_SYNCOBJ_DESTROY, &raw mut d) };
    }
}

// ---- fence bridge: device fence -> timeline point, on one helper thread ----

struct Job {
    backend: Arc<dyn Backend>,
    fence: Fence,
    timeline: Timeline,
    value: u64,
}

fn bridge() -> &'static Mutex<Sender<Job>> {
    static TX: OnceLock<Mutex<Sender<Job>>> = OnceLock::new();
    TX.get_or_init(|| {
        let (tx, rx) = channel::<Job>();
        std::thread::Builder::new()
            .name("numerobis-fence-bridge".into())
            .spawn(move || {
                for j in rx {
                    // Keep waiting through timeouts; give up (and signal, so no
                    // consumer deadlocks) only if the device is lost.
                    while let Err(Error::Timeout(_)) =
                        j.backend.wait(j.fence, Duration::from_secs(1))
                    {}
                    let _ = j.timeline.signal(j.value);
                }
            })
            .expect("spawn fence bridge");
        Mutex::new(tx)
    })
}

/// Signals `tl` at `value` once `fence` of `backend` has completed.
pub(crate) fn signal_after(
    backend: Arc<dyn Backend>,
    fence: Option<Fence>,
    tl: &Timeline,
    value: u64,
) -> Result<()> {
    if tl.value()? >= value {
        return Err(invalid(format!(
            "timeline is already at {} >= {value}",
            tl.value()?
        )));
    }
    match fence {
        None => tl.signal(value),
        Some(fence) if backend.wait(fence, Duration::ZERO).is_ok() => tl.signal(value),
        Some(fence) => bridge()
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .send(Job {
                backend,
                fence,
                timeline: tl.clone(),
                value,
            })
            .map_err(|_| Error::Unsupported("fence bridge thread is gone".into())),
    }
}

/// Host-side dependency of a stream (resolved before its next submission).
pub(crate) enum Dep {
    SyncFile(SyncFile),
    Timeline(Timeline, u64),
}

impl Dep {
    pub(crate) fn wait(&self, timeout: Duration) -> Result<()> {
        match self {
            Dep::SyncFile(s) => s.wait(timeout),
            Dep::Timeline(t, v) => t.wait(*v, timeout),
        }
    }
}
