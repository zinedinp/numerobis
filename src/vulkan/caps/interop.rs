//! Sharing memory and semaphores with other APIs and processes. The fd and
//! dma-buf routes exist on Linux, the Win32 handle routes on Windows; on the
//! other OS the capability is skipped.

use super::{Area, Spec, Support, ext};

const AREA: Area = Area::Interop;

pub const EXTERNAL_MEMORY_FD: Spec = Spec {
    area: AREA,
    support: Support::LINUX,
    needs: &[],
    name: "external_memory_fd",
    doc: "memory exported and imported as POSIX fds",
    routes: &[ext(c"VK_KHR_external_memory_fd", None, &[])],
};

pub const DMA_BUF: Spec = Spec {
    area: AREA,
    support: Support::LINUX,
    needs: &[],
    name: "dma_buf",
    doc: "dma-buf import and export: zero-copy with numerobis::dmabuf, KMS and V4L2",
    routes: &[ext(c"VK_EXT_external_memory_dma_buf", None, &[])],
};

pub const DRM_FORMAT_MODIFIER: Spec = Spec {
    area: AREA,
    support: Support::LINUX,
    needs: &[],
    name: "drm_format_modifier",
    doc: "images with DRM format modifiers (tiled dma-bufs)",
    routes: &[ext(c"VK_EXT_image_drm_format_modifier", None, &[])],
};

pub const EXTERNAL_SEMAPHORE_FD: Spec = Spec {
    area: AREA,
    support: Support::LINUX,
    needs: &[],
    name: "external_semaphore_fd",
    doc: "semaphores as fds, SYNC_FD included (numerobis::sync)",
    routes: &[ext(c"VK_KHR_external_semaphore_fd", None, &[])],
};

pub const EXTERNAL_MEMORY_WIN32: Spec = Spec {
    area: AREA,
    support: Support::WINDOWS_ONLY_TESTED,
    needs: &[],
    name: "external_memory_win32",
    doc: "memory exported and imported as NT handles (D3D12, CUDA)",
    routes: &[ext(c"VK_KHR_external_memory_win32", None, &[])],
};

pub const EXTERNAL_SEMAPHORE_WIN32: Spec = Spec {
    area: AREA,
    support: Support::WINDOWS_ONLY_TESTED,
    needs: &[],
    name: "external_semaphore_win32",
    doc: "semaphores as NT handles (D3D12 fences)",
    routes: &[ext(c"VK_KHR_external_semaphore_win32", None, &[])],
};
