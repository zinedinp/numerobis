//! Presentation. Every capability here needs `VK_KHR_swapchain`, which the
//! host enables with its surface.

use super::{Area, Capability, Spec, Support, ext};
use crate::vulkan::features::Chain;

const AREA: Area = Area::Present;

const LATEST_READY: &[&str] = &["presentModeFifoLatestReady"];

pub const FIFO_LATEST_READY: Spec = Spec {
    area: AREA,
    support: Support::UNTESTED,
    needs: &[],
    name: "fifo_latest_ready",
    doc: "FIFO_LATEST_READY present mode",
    routes: &[
        ext(
            c"VK_KHR_present_mode_fifo_latest_ready",
            Some(Chain::FifoLatestReady),
            LATEST_READY,
        ),
        ext(
            c"VK_EXT_present_mode_fifo_latest_ready",
            Some(Chain::FifoLatestReady),
            LATEST_READY,
        ),
    ],
};

pub const PRESENT_ID: Spec = Spec {
    area: AREA,
    support: Support::LINUX_TESTED,
    needs: &[],
    name: "present_id",
    doc: "an id per present (VkPresentIdKHR)",
    routes: &[ext(
        c"VK_KHR_present_id",
        Some(Chain::PresentId),
        &["presentId"],
    )],
};

pub const PRESENT_WAIT: Spec = Spec {
    area: AREA,
    support: Support::LINUX_TESTED,
    needs: &[Capability::PresentId],
    name: "present_wait",
    doc: "vkWaitForPresentKHR: wait until a present id reached the screen",
    routes: &[ext(
        c"VK_KHR_present_wait",
        Some(Chain::PresentWait),
        &["presentWait"],
    )],
};

pub const PRESENT_ID2: Spec = Spec {
    area: AREA,
    support: Support::LINUX_TESTED,
    needs: &[],
    name: "present_id2",
    doc: "present ids per surface (needs the instance extensions \
          VK_KHR_get_surface_capabilities2 and VK_KHR_surface)",
    routes: &[ext(
        c"VK_KHR_present_id2",
        Some(Chain::PresentId2),
        &["presentId2"],
    )],
};

pub const PRESENT_WAIT2: Spec = Spec {
    area: AREA,
    support: Support::LINUX_TESTED,
    needs: &[Capability::PresentId2],
    name: "present_wait2",
    doc: "vkWaitForPresent2KHR, per surface",
    routes: &[ext(
        c"VK_KHR_present_wait2",
        Some(Chain::PresentWait2),
        &["presentWait2"],
    )],
};

pub const HDR_METADATA: Spec = Spec {
    area: AREA,
    support: Support::WINDOWS_TESTED,
    needs: &[],
    name: "hdr_metadata",
    doc: "vkSetHdrMetadataEXT: the mastering display and content light \
          levels of an HDR swapchain (color_space::HdrMetadata)",
    routes: &[ext(c"VK_EXT_hdr_metadata", None, &[])],
};

pub const SWAPCHAIN_COLOR_SPACE: Spec = Spec {
    area: AREA,
    support: Support::WINDOWS_TESTED,
    needs: &[],
    name: "swapchain_color_space",
    doc: "surface formats beyond sRGB non-linear: HDR10, HLG, scRGB, \
          Display P3 (an instance extension: Instance::with_color_spaces)",
    routes: &[ext(c"VK_EXT_swapchain_colorspace", None, &[])],
};
