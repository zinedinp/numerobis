//! Thin libc layer shared by the raw backends: owned file descriptors,
//! ioctl with errno mapping, RAII memory mappings.

use std::ffi::CString;

use crate::runtime::error::{Error, Result};

/// Linux `_IOC` request number.
pub(crate) const fn ioc(dir: u32, ty: u8, nr: u8, size: usize) -> u64 {
    ((dir as u64) << 30) | ((size as u64) << 16) | ((ty as u64) << 8) | nr as u64
}
pub(crate) const IOC_WRITE: u32 = 1;
pub(crate) const IOC_READ: u32 = 2;

/// `_IOWR(ty, nr, T)`.
pub(crate) const fn iowr<T>(ty: u8, nr: u8) -> u64 {
    ioc(IOC_READ | IOC_WRITE, ty, nr, size_of::<T>())
}

pub(crate) fn errno() -> i32 {
    std::io::Error::last_os_error().raw_os_error().unwrap_or(0)
}

/// Owned file descriptor, closed on drop.
#[derive(Debug)]
pub(crate) struct Fd(pub(crate) i32);

impl Fd {
    pub(crate) fn open(path: &str) -> Result<Fd> {
        let c = CString::new(path).map_err(|_| Error::InvalidArgument(path.into()))?;
        // SAFETY: valid NUL-terminated path.
        let fd = unsafe { libc::open(c.as_ptr(), libc::O_RDWR | libc::O_CLOEXEC) };
        if fd < 0 {
            Err(Error::Os(errno()))
        } else {
            Ok(Fd(fd))
        }
    }

    /// `ioctl(fd, req, arg)`; `EINTR`/`EAGAIN` are retried.
    ///
    /// # Safety
    /// `arg` must be what the driver expects for `req` (size encoded in `req`
    /// or documented by the driver), valid for reads and writes.
    pub(crate) unsafe fn ioctl<T>(&self, req: u64, arg: *mut T) -> Result<()> {
        loop {
            // SAFETY: forwarded to the caller.
            let r = unsafe { libc::ioctl(self.0, req as _, arg) };
            if r >= 0 {
                return Ok(());
            }
            match errno() {
                libc::EINTR | libc::EAGAIN => {}
                e => return Err(Error::Os(e)),
            }
        }
    }
}

impl Drop for Fd {
    fn drop(&mut self) {
        // SAFETY: we own the descriptor.
        unsafe { libc::close(self.0) };
    }
}

/// A non-owning view of mapped memory with volatile accessors.
#[derive(Debug, Clone, Copy)]
pub(crate) struct View {
    pub(crate) ptr: *mut u8,
    pub(crate) len: usize,
}

// SAFETY: the view is plain memory; synchronisation is the owner's job.
unsafe impl Send for View {}
unsafe impl Sync for View {}

impl View {
    pub(crate) fn addr(&self) -> usize {
        self.ptr as usize
    }

    /// Volatile 32-bit read at byte offset `off`.
    pub(crate) fn read32(&self, off: usize) -> u32 {
        assert!(off + 4 <= self.len);
        debug_assert!(off.is_multiple_of(4), "unaligned 32-bit access at {off:#x}");
        // SAFETY: in bounds; the mapping is page aligned and `off` a multiple of 4.
        #[allow(clippy::cast_ptr_alignment, reason = "alignment checked above")]
        unsafe {
            self.ptr.add(off).cast::<u32>().read_volatile()
        }
    }

    /// Volatile 32-bit write at byte offset `off`.
    pub(crate) fn write32(&self, off: usize, v: u32) {
        assert!(off + 4 <= self.len);
        debug_assert!(off.is_multiple_of(4), "unaligned 32-bit access at {off:#x}");
        // SAFETY: in bounds; the mapping is page aligned and `off` a multiple of 4.
        #[allow(clippy::cast_ptr_alignment, reason = "alignment checked above")]
        unsafe {
            self.ptr.add(off).cast::<u32>().write_volatile(v);
        }
    }

    /// Volatile 64-bit read at byte offset `off`.
    pub(crate) fn read64(&self, off: usize) -> u64 {
        assert!(off + 8 <= self.len);
        debug_assert!(off.is_multiple_of(8), "unaligned 64-bit access at {off:#x}");
        // SAFETY: in bounds; the mapping is page aligned and `off` a multiple of 8.
        #[allow(clippy::cast_ptr_alignment, reason = "alignment checked above")]
        unsafe {
            self.ptr.add(off).cast::<u64>().read_volatile()
        }
    }

    /// Volatile 64-bit write at byte offset `off`.
    pub(crate) fn write64(&self, off: usize, v: u64) {
        assert!(off + 8 <= self.len);
        debug_assert!(off.is_multiple_of(8), "unaligned 64-bit access at {off:#x}");
        // SAFETY: in bounds; the mapping is page aligned and `off` a multiple of 8.
        #[allow(clippy::cast_ptr_alignment, reason = "alignment checked above")]
        unsafe {
            self.ptr.add(off).cast::<u64>().write_volatile(v);
        }
    }

    pub(crate) fn write_bytes(&self, off: usize, src: &[u8]) {
        assert!(off + src.len() <= self.len);
        // SAFETY: in bounds; the device does not access this range concurrently
        // (callers order host writes before submission).
        unsafe { core::ptr::copy_nonoverlapping(src.as_ptr(), self.ptr.add(off), src.len()) }
    }

    pub(crate) fn read_bytes(&self, off: usize, dst: &mut [u8]) {
        assert!(off + dst.len() <= self.len);
        // SAFETY: as in write_bytes, after the producing work has signalled.
        unsafe { core::ptr::copy_nonoverlapping(self.ptr.add(off), dst.as_mut_ptr(), dst.len()) }
    }
}

/// A memory mapping, unmapped on drop.
#[derive(Debug)]
pub(crate) struct Map(View);

impl core::ops::Deref for Map {
    type Target = View;
    fn deref(&self) -> &View {
        &self.0
    }
}

impl Map {
    /// `mmap(addr, len, prot, flags, fd, off)`. `addr` is a hint unless `flags` has `MAP_FIXED`.
    pub(crate) fn new(
        addr: usize,
        len: usize,
        prot: i32,
        flags: i32,
        fd: i32,
        off: u64,
    ) -> Result<Map> {
        let off = libc::off_t::try_from(off).map_err(|_| Error::Os(libc::EINVAL))?;
        // SAFETY: plain mmap; MAP_FIXED callers own the target range.
        let p = unsafe { libc::mmap(addr as *mut _, len, prot, flags, fd, off) };
        if p == libc::MAP_FAILED {
            Err(Error::Os(errno()))
        } else {
            Ok(Map(View { ptr: p.cast(), len }))
        }
    }

    /// Leaks the mapping (the range stays mapped, e.g. when `MAP_FIXED` over a reservation).
    pub(crate) fn forget(self) {
        core::mem::forget(self);
    }
}

impl Drop for Map {
    fn drop(&mut self) {
        // SAFETY: we own the mapping.
        unsafe { libc::munmap(self.0.ptr.cast(), self.0.len) };
    }
}

pub(crate) const RW: i32 = libc::PROT_READ | libc::PROT_WRITE;
