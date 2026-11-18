//! dma-buf helpers shared by backends: host memory exported as a real dma-buf
//! (memfd + `/dev/udmabuf`), CPU mappings of foreign dma-bufs with the
//! `DMA_BUF_IOCTL_SYNC` bracket, and size queries.
//!
//! DRM backends (xe, amdgpu, v3d, accel) export and import through PRIME
//! (`DRM_IOCTL_PRIME_HANDLE_TO_FD` / `FD_TO_HANDLE`, [`prime_export`] /
//! [`prime_import`]); on SoCs (Pi, Rockchip, iGPUs) that is the zero-copy path
//! between camera, display, Vulkan and the NPU.

use std::os::fd::{AsRawFd, BorrowedFd, FromRawFd, OwnedFd};

use crate::runtime::error::{Error, Result};
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
pub(crate) struct PrimeHandle {
    pub handle: u32,
    pub flags: u32,
    pub fd: i32,
}
const PRIME_HANDLE_TO_FD: u64 = iowr::<PrimeHandle>(b'd', 0x2D);
const PRIME_FD_TO_HANDLE: u64 = iowr::<PrimeHandle>(b'd', 0x2E);

/// GEM handle -> dma-buf (DRM PRIME), read-write, close-on-exec.
#[allow(dead_code)]
pub(crate) fn prime_export(drm: &Fd, handle: u32) -> Result<OwnedFd> {
    let mut p = PrimeHandle {
        handle,
        flags: (libc::O_CLOEXEC | libc::O_RDWR) as u32,
        fd: -1,
    };
    // SAFETY: matches DRM_IOCTL_PRIME_HANDLE_TO_FD.
    unsafe { drm.ioctl(PRIME_HANDLE_TO_FD, &mut p) }?;
    // SAFETY: new descriptor we own.
    Ok(unsafe { OwnedFd::from_raw_fd(p.fd) })
}

/// dma-buf -> GEM handle of this DRM device (DRM PRIME).
#[allow(dead_code)]
pub(crate) fn prime_import(drm: &Fd, fd: BorrowedFd<'_>) -> Result<u32> {
    let mut p = PrimeHandle {
        fd: fd.as_raw_fd(),
        ..Default::default()
    };
    // SAFETY: matches DRM_IOCTL_PRIME_FD_TO_HANDLE.
    unsafe { drm.ioctl(PRIME_FD_TO_HANDLE, &mut p) }?;
    Ok(p.handle)
}

/// Size of a dma-buf (or memfd) in bytes.
pub(crate) fn fd_size(fd: BorrowedFd<'_>) -> Result<usize> {
    // SAFETY: lseek on a borrowed descriptor; dma-bufs support SEEK_END for the size.
    let end = unsafe { libc::lseek(fd.as_raw_fd(), 0, libc::SEEK_END) };
    if end < 0 {
        return Err(Error::Os(errno()));
    }
    // SAFETY: as above.
    unsafe { libc::lseek(fd.as_raw_fd(), 0, libc::SEEK_SET) };
    Ok(end as usize)
}

/// Host memory as a shareable fd: a sealed memfd turned into a dma-buf by
/// `/dev/udmabuf` when that is accessible, else the memfd itself (still
/// mmap-able and passable between processes, but not importable by drivers).
pub(crate) fn host_dmabuf(bytes: usize) -> Result<(OwnedFd, bool)> {
    let page = 4096usize;
    let size = bytes.max(1).next_multiple_of(page);
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
        // udmabuf returns the new fd as the ioctl result, so call libc directly.
        // SAFETY: matches UDMABUF_CREATE.
        let r = unsafe { libc::ioctl(dev.0, UDMABUF_CREATE as _, &mut c) };
        if r >= 0 {
            // SAFETY: new descriptor we own.
            return Ok((unsafe { OwnedFd::from_raw_fd(r) }, true));
        }
    }
    Ok((memfd, false))
}

/// A CPU mapping of a dma-buf or memfd. Accesses are bracketed with
/// `DMA_BUF_IOCTL_SYNC` (ignored for memfds).
pub(crate) struct HostMapping {
    fd: OwnedFd,
    map: Map,
}

// SAFETY: plain shared memory; the owner serialises access.
unsafe impl Send for HostMapping {}
unsafe impl Sync for HostMapping {}

impl HostMapping {
    pub(crate) fn new(fd: BorrowedFd<'_>, len: usize) -> Result<HostMapping> {
        let fd = fd
            .try_clone_to_owned()
            .map_err(|e| Error::Os(e.raw_os_error().unwrap_or(0)))?;
        let map = Map::new(0, len.max(1), RW, libc::MAP_SHARED, fd.as_raw_fd(), 0)?;
        Ok(HostMapping { fd, map })
    }

    fn sync(&self, flags: u64) {
        let mut s = DmaBufSync { flags };
        // SAFETY: matches DMA_BUF_IOCTL_SYNC; ENOTTY on memfds is fine.
        let _ = unsafe { libc::ioctl(self.fd.as_raw_fd(), DMA_BUF_IOCTL_SYNC as _, &mut s) };
    }

    pub(crate) fn read(&self, dst: &mut [u8]) {
        self.sync(SYNC_READ);
        self.map.read_bytes(0, dst);
        self.sync(SYNC_READ | SYNC_END);
    }

    pub(crate) fn write(&self, src: &[u8]) {
        self.sync(SYNC_WRITE);
        self.map.write_bytes(0, src);
        self.sync(SYNC_WRITE | SYNC_END);
    }

    pub(crate) fn fd(&self) -> BorrowedFd<'_> {
        use std::os::fd::AsFd;
        self.fd.as_fd()
    }
}
