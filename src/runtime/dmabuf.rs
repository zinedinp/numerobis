//! dma-buf: zero-copy buffers shared between devices, processes and APIs
//! (Vulkan, KMS, V4L2, other accelerators).
//!
//! - [`HostDmabuf`]: host memory as a real dma-buf (sealed memfd turned into
//!   a dma-buf by `/dev/udmabuf`; a plain memfd where that is not accessible),
//!   or the CPU mapping of a dma-buf someone else exported. Every access is
//!   bracketed with `DMA_BUF_IOCTL_SYNC`, so caches stay coherent on non-coherent
//!   SoCs. Hand [`HostDmabuf::as_fd`] to [`Stream::import_dmabuf`](crate::Stream::import_dmabuf)
//!   to use it on a device without a copy.
//! - [`prime_export`] / [`prime_import`]: GEM handle <-> dma-buf on a DRM device
//!   (`DRM_IOCTL_PRIME_*`), the building blocks for the `export_dmabuf` /
//!   `import_dmabuf` hooks of DRM backends (xe, amdgpu, v3d, accel), in-tree or
//!   through [`Device::from_backend`](crate::Device::from_backend). On SoCs (Pi,
//!   Rockchip, iGPUs) that is the path between camera, display, Vulkan and NPU.
//!
//! ```
//! use std::os::fd::AsFd;
//! use numerobis::{Choice, Device, dmabuf::HostDmabuf, elem::F32, ops::{self, Act}};
//!
//! // A frame in shareable host memory ...
//! let frame = HostDmabuf::new(4 * 4)?;
//! let px: Vec<u8> = [-1.0f32, 2.0, -3.0, 4.0].iter().flat_map(|v| v.to_le_bytes()).collect();
//! frame.write(0, &px)?;
//! // ... used by a device in place, the result written back into the same pages.
//! let dev = Device::mock();
//! let mut s = dev.stream();
//! let x = s.import_dmabuf::<F32>(frame.as_fd(), 4)?;
//! let mut tmp = s.alloc::<F32>(4)?;
//! ops::act(&mut s, Choice::Up, Act::Relu, &x, &mut tmp)?.wait()?;
//! ops::act(&mut s, Choice::Up, Act::Relu, &tmp, &mut { x })?.wait()?;
//! let mut out = [0u8; 16];
//! frame.read(0, &mut out)?;
//! assert_eq!(out[0..4], 0.0f32.to_le_bytes());
//! assert_eq!(out[4..8], 2.0f32.to_le_bytes());
//! # Ok::<(), numerobis::Error>(())
//! ```

use std::os::fd::{AsFd, AsRawFd, BorrowedFd, FromRawFd, OwnedFd, RawFd};

use crate::runtime::error::{Error, Result, invalid};
use crate::runtime::os::{Fd, IOC_WRITE, Map, RW, errno, ioc, iowr};

#[repr(C)]
struct DmaBufSync {
    flags: u64,
}
const DMA_BUF_IOCTL_SYNC: u64 = ioc(IOC_WRITE, b'b', 0, size_of::<DmaBufSync>());
const SYNC_READ: u64 = 1;
const SYNC_WRITE: u64 = 2;
const SYNC_END: u64 = 4;

#[repr(C)]
struct UdmabufCreate {
    memfd: u32,
    flags: u32,
    offset: u64,
    size: u64,
}
const UDMABUF_CREATE: u64 = ioc(IOC_WRITE, b'u', 0x42, size_of::<UdmabufCreate>());
const UDMABUF_FLAGS_CLOEXEC: u32 = 1;

#[repr(C)]
#[derive(Default)]
struct PrimeHandle {
    handle: u32,
    flags: u32,
    fd: i32,
}
const PRIME_HANDLE_TO_FD: u64 = iowr::<PrimeHandle>(b'd', 0x2D);
const PRIME_FD_TO_HANDLE: u64 = iowr::<PrimeHandle>(b'd', 0x2E);
const _: () = assert!(size_of::<PrimeHandle>() == 12);

/// `ioctl` on a borrowed descriptor, `EINTR`/`EAGAIN` retried.
///
/// # Safety
/// `arg` must be what the driver expects for `req`.
unsafe fn ioctl<T>(fd: RawFd, req: u64, arg: *mut T) -> Result<i32> {
    loop {
        // SAFETY: forwarded to the caller.
        let r = unsafe { libc::ioctl(fd, req as _, arg) };
        if r >= 0 {
            return Ok(r);
        }
        match errno() {
            libc::EINTR | libc::EAGAIN => continue,
            e => return Err(Error::Os(e)),
        }
    }
}

/// GEM `handle` of the DRM device `drm` -> dma-buf (read-write, close-on-exec).
pub fn prime_export(drm: BorrowedFd<'_>, handle: u32) -> Result<OwnedFd> {
    let mut p = PrimeHandle {
        handle,
        flags: (libc::O_CLOEXEC | libc::O_RDWR) as u32,
        fd: -1,
    };
    // SAFETY: PrimeHandle matches DRM_IOCTL_PRIME_HANDLE_TO_FD.
    unsafe { ioctl(drm.as_raw_fd(), PRIME_HANDLE_TO_FD, &mut p) }?;
    // SAFETY: the kernel returned a new descriptor we own.
    Ok(unsafe { OwnedFd::from_raw_fd(p.fd) })
}

/// dma-buf -> GEM handle on the DRM device `drm` (close it with the driver's
/// `GEM_CLOSE` when done).
pub fn prime_import(drm: BorrowedFd<'_>, dmabuf: BorrowedFd<'_>) -> Result<u32> {
    let mut p = PrimeHandle {
        fd: dmabuf.as_raw_fd(),
        ..Default::default()
    };
    // SAFETY: PrimeHandle matches DRM_IOCTL_PRIME_FD_TO_HANDLE.
    unsafe { ioctl(drm.as_raw_fd(), PRIME_FD_TO_HANDLE, &mut p) }?;
    Ok(p.handle)
}

/// Size in bytes of a dma-buf (or memfd).
pub fn size(fd: BorrowedFd<'_>) -> Result<usize> {
    // SAFETY: lseek on a borrowed descriptor; dma-bufs report their size via SEEK_END.
    let end = unsafe { libc::lseek(fd.as_raw_fd(), 0, libc::SEEK_END) };
    if end < 0 {
        return Err(Error::Os(errno()));
    }
    // SAFETY: as above.
    unsafe { libc::lseek(fd.as_raw_fd(), 0, libc::SEEK_SET) };
    Ok(end as usize)
}

/// Host memory or a foreign dma-buf, mapped for the CPU.
pub struct HostDmabuf {
    fd: OwnedFd,
    map: Map,
    len: usize,
    dmabuf: bool,
}

impl core::fmt::Debug for HostDmabuf {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        let kind = if self.dmabuf { "dma-buf" } else { "memfd" };
        write!(
            f,
            "HostDmabuf({} bytes, {kind}, fd {})",
            self.len,
            self.fd.as_raw_fd()
        )
    }
}

// SAFETY: plain shared memory behind an owned fd; accesses go through &self
// methods that copy, and concurrent writers are the sharing parties' contract
// (synchronise with `crate::sync`).
unsafe impl Send for HostDmabuf {}
unsafe impl Sync for HostDmabuf {}

impl HostDmabuf {
    /// `len` bytes of zeroed host memory (rounded up to a page in the fd).
    /// A real dma-buf when `/dev/udmabuf` is accessible (see [`is_dmabuf`](Self::is_dmabuf)).
    pub fn new(len: usize) -> Result<HostDmabuf> {
        let (fd, dmabuf) = alloc(len)?;
        Self::with_fd(fd, len, dmabuf)
    }

    /// Maps the first `len` bytes of a dma-buf another driver or process exported.
    pub fn map(fd: BorrowedFd<'_>, len: usize) -> Result<HostDmabuf> {
        let have = size(fd)?;
        if have < len {
            return Err(invalid(format!(
                "dma-buf has {have} bytes, {len} requested"
            )));
        }
        let fd = fd
            .try_clone_to_owned()
            .map_err(|e| Error::Os(e.raw_os_error().unwrap_or(0)))?;
        // A kernel dma-buf answers the sync ioctl; a memfd says ENOTTY.
        let mut sy = DmaBufSync { flags: SYNC_READ };
        // SAFETY: DmaBufSync matches DMA_BUF_IOCTL_SYNC.
        let dmabuf = unsafe { ioctl(fd.as_raw_fd(), DMA_BUF_IOCTL_SYNC, &mut sy) }.is_ok();
        if dmabuf {
            sy.flags = SYNC_READ | SYNC_END;
            // SAFETY: as above, closes the bracket just opened.
            let _ = unsafe { ioctl(fd.as_raw_fd(), DMA_BUF_IOCTL_SYNC, &mut sy) };
        }
        Self::with_fd(fd, len, dmabuf)
    }

    fn with_fd(fd: OwnedFd, len: usize, dmabuf: bool) -> Result<HostDmabuf> {
        let map = Map::new(0, len.max(1), RW, libc::MAP_SHARED, fd.as_raw_fd(), 0)?;
        Ok(HostDmabuf {
            fd,
            map,
            len,
            dmabuf,
        })
    }

    pub fn len(&self) -> usize {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// `true`: a kernel dma-buf, importable by drivers. `false`: a memfd
    /// (shareable with processes and the mock backend, not with drivers).
    pub fn is_dmabuf(&self) -> bool {
        self.dmabuf
    }

    /// A new descriptor for the same memory (to pass to another process or API).
    pub fn try_clone_fd(&self) -> Result<OwnedFd> {
        self.fd
            .try_clone()
            .map_err(|e| Error::Os(e.raw_os_error().unwrap_or(0)))
    }

    fn range(&self, off: usize, n: usize) -> Result<()> {
        if off.checked_add(n).is_some_and(|end| end <= self.len) {
            Ok(())
        } else {
            Err(invalid(format!(
                "bytes {off}..{} outside a {}-byte dma-buf",
                off.saturating_add(n),
                self.len
            )))
        }
    }

    fn sync(&self, flags: u64) {
        let mut s = DmaBufSync { flags };
        // SAFETY: DmaBufSync matches DMA_BUF_IOCTL_SYNC; memfds answer ENOTTY, which is fine.
        let _ = unsafe { libc::ioctl(self.fd.as_raw_fd(), DMA_BUF_IOCTL_SYNC as _, &mut s) };
    }

    /// Copies `dst.len()` bytes from offset `off` out (CPU read bracket).
    pub fn read(&self, off: usize, dst: &mut [u8]) -> Result<()> {
        self.range(off, dst.len())?;
        self.sync(SYNC_READ);
        self.map.read_bytes(off, dst);
        self.sync(SYNC_READ | SYNC_END);
        Ok(())
    }

    /// Copies `src` in at offset `off` (CPU write bracket).
    pub fn write(&self, off: usize, src: &[u8]) -> Result<()> {
        self.range(off, src.len())?;
        self.sync(SYNC_WRITE);
        self.map.write_bytes(off, src);
        self.sync(SYNC_WRITE | SYNC_END);
        Ok(())
    }
}

impl AsFd for HostDmabuf {
    fn as_fd(&self) -> BorrowedFd<'_> {
        self.fd.as_fd()
    }
}

/// Sealed memfd of `len` bytes (page-rounded) -> udmabuf, else the memfd.
fn alloc(len: usize) -> Result<(OwnedFd, bool)> {
    let size = len.max(1).next_multiple_of(4096);
    // SAFETY: plain syscall with a static name.
    let mfd = unsafe {
        libc::memfd_create(
            c"numerobis".as_ptr(),
            libc::MFD_CLOEXEC | libc::MFD_ALLOW_SEALING,
        )
    };
    if mfd < 0 {
        return Err(Error::Os(errno()));
    }
    // SAFETY: we own mfd.
    let memfd = unsafe { OwnedFd::from_raw_fd(mfd) };
    // SAFETY: valid descriptor.
    if unsafe { libc::ftruncate(mfd, size as libc::off_t) } != 0 {
        return Err(Error::Os(errno()));
    }
    // udmabuf requires F_SEAL_SHRINK.
    // SAFETY: valid descriptor.
    unsafe { libc::fcntl(mfd, libc::F_ADD_SEALS, libc::F_SEAL_SHRINK) };
    if let Ok(dev) = Fd::open("/dev/udmabuf") {
        let mut c = UdmabufCreate {
            memfd: mfd as u32,
            flags: UDMABUF_FLAGS_CLOEXEC,
            offset: 0,
            size: size as u64,
        };
        // udmabuf returns the new descriptor as the ioctl result.
        // SAFETY: UdmabufCreate matches UDMABUF_CREATE.
        if let Ok(r) = unsafe { ioctl(dev.0, UDMABUF_CREATE, &mut c) } {
            // SAFETY: new descriptor we own.
            return Ok((unsafe { OwnedFd::from_raw_fd(r) }, true));
        }
    }
    Ok((memfd, false))
}
