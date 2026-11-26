//! Image formats a renderer picks at startup: the depth buffer's from what
//! `vkGetPhysicalDeviceFormatProperties` reports, and the `VkFormat`
//! values [`swapchain`](super::swapchain) chooses among.
//!
//! ```
//! use numerobis::vulkan::format::{self, Depth, feature};
//!
//! // The host's vkGetPhysicalDeviceFormatProperties(physical, format)
//! // .optimalTilingFeatures; this device has no 32-bit depth.
//! let optimal = |f| match f {
//!     format::D24_UNORM_S8_UINT | format::D16_UNORM => feature::DEPTH_STENCIL_ATTACHMENT,
//!     _ => 0,
//! };
//! let depth = format::depth(Depth::default(), optimal);
//! assert_eq!(depth, Some(format::D24_UNORM_S8_UINT));
//! assert_eq!(format::aspect_mask(format::D24_UNORM_S8_UINT), format::aspect::DEPTH | format::aspect::STENCIL);
//! ```

pub const UNDEFINED: u32 = 0;
pub const R8_SRGB: u32 = 15;
pub const R8G8_SRGB: u32 = 22;
pub const R8G8B8_SRGB: u32 = 29;
pub const B8G8R8_SRGB: u32 = 36;
pub const R8G8B8A8_UNORM: u32 = 37;
pub const R8G8B8A8_SRGB: u32 = 43;
pub const B8G8R8A8_UNORM: u32 = 44;
pub const B8G8R8A8_SRGB: u32 = 50;
pub const A8B8G8R8_UNORM_PACK32: u32 = 51;
pub const A8B8G8R8_SRGB_PACK32: u32 = 57;
pub const A2R10G10B10_UNORM_PACK32: u32 = 58;
pub const A2B10G10R10_UNORM_PACK32: u32 = 64;
pub const R16G16B16A16_SFLOAT: u32 = 97;
pub const D16_UNORM: u32 = 124;
pub const X8_D24_UNORM_PACK32: u32 = 125;
pub const D32_SFLOAT: u32 = 126;
pub const S8_UINT: u32 = 127;
pub const D16_UNORM_S8_UINT: u32 = 128;
pub const D24_UNORM_S8_UINT: u32 = 129;
pub const D32_SFLOAT_S8_UINT: u32 = 130;

/// `VkFormatFeatureFlagBits` a renderer asks of an attachment.
pub mod feature {
    pub const SAMPLED_IMAGE: u32 = 0x1;
    pub const COLOR_ATTACHMENT: u32 = 0x80;
    pub const DEPTH_STENCIL_ATTACHMENT: u32 = 0x200;
    pub const SAMPLED_IMAGE_FILTER_LINEAR: u32 = 0x1000;
}

/// `VkImageAspectFlagBits`.
pub mod aspect {
    pub const COLOR: u32 = 0x1;
    pub const DEPTH: u32 = 0x2;
    pub const STENCIL: u32 = 0x4;
}

/// Whether `format` is one of the 8-bit `_SRGB` formats, which encode on
/// write and decode on read.
#[must_use]
pub const fn is_srgb(format: u32) -> bool {
    matches!(
        format,
        R8_SRGB
            | R8G8_SRGB
            | R8G8B8_SRGB
            | B8G8R8_SRGB
            | R8G8B8A8_SRGB
            | B8G8R8A8_SRGB
            | A8B8G8R8_SRGB_PACK32
    )
}

#[must_use]
pub const fn has_depth(format: u32) -> bool {
    matches!(
        format,
        D16_UNORM
            | X8_D24_UNORM_PACK32
            | D32_SFLOAT
            | D16_UNORM_S8_UINT
            | D24_UNORM_S8_UINT
            | D32_SFLOAT_S8_UINT
    )
}

#[must_use]
pub const fn has_stencil(format: u32) -> bool {
    matches!(
        format,
        S8_UINT | D16_UNORM_S8_UINT | D24_UNORM_S8_UINT | D32_SFLOAT_S8_UINT
    )
}

/// The `aspectMask` of a view of all of `format`: depth and stencil as the
/// format has them, else colour.
#[must_use]
pub const fn aspect_mask(format: u32) -> u32 {
    let depth = if has_depth(format) { aspect::DEPTH } else { 0 };
    let stencil = if has_stencil(format) {
        aspect::STENCIL
    } else {
        0
    };
    if depth | stencil == 0 {
        aspect::COLOR
    } else {
        depth | stencil
    }
}

/// What the depth buffer needs.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Hash)]
pub struct Depth {
    /// A stencil buffer beside it.
    pub stencil: bool,
    /// Shaders sample it (a depth pyramid, soft particles, SSAO).
    pub sampled: bool,
}

impl Depth {
    /// The candidates, in order. Without stencil the most precise first
    /// (32-bit float, best with reversed Z), then the packed 24-bit ones,
    /// 16-bit last; a stencil format only where it has the depth bits.
    /// The spec makes `D16_UNORM` and one of `X8_D24_UNORM_PACK32` and
    /// `D32_SFLOAT` work everywhere, and one of `D24_UNORM_S8_UINT` and
    /// `D32_SFLOAT_S8_UINT`.
    #[must_use]
    pub const fn candidates(self) -> &'static [u32] {
        if self.stencil {
            &[D32_SFLOAT_S8_UINT, D24_UNORM_S8_UINT, D16_UNORM_S8_UINT]
        } else {
            &[
                D32_SFLOAT,
                D32_SFLOAT_S8_UINT,
                X8_D24_UNORM_PACK32,
                D24_UNORM_S8_UINT,
                D16_UNORM,
            ]
        }
    }

    /// The `optimalTilingFeatures` a candidate must have.
    #[must_use]
    pub const fn features(self) -> u32 {
        if self.sampled {
            feature::DEPTH_STENCIL_ATTACHMENT | feature::SAMPLED_IMAGE
        } else {
            feature::DEPTH_STENCIL_ATTACHMENT
        }
    }
}

/// The first of `wish`'s [`candidates`](Depth::candidates) whose
/// `optimalTilingFeatures` (`optimal(format)`, the host's
/// `vkGetPhysicalDeviceFormatProperties`) has what it needs.
pub fn depth(wish: Depth, mut optimal: impl FnMut(u32) -> u32) -> Option<u32> {
    let needs = wish.features();
    wish.candidates()
        .iter()
        .copied()
        .find(|&f| optimal(f) & needs == needs)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn depth_takes_the_first_candidate_the_device_has() {
        let all = |_| feature::DEPTH_STENCIL_ATTACHMENT | feature::SAMPLED_IMAGE;
        assert_eq!(depth(Depth::default(), all), Some(D32_SFLOAT));
        let stencil = Depth {
            stencil: true,
            sampled: false,
        };
        assert_eq!(depth(stencil, all), Some(D32_SFLOAT_S8_UINT));
        // Attachment only where sampling is needed: the next one.
        let sampled = Depth {
            stencil: false,
            sampled: true,
        };
        let d32_unsampled = |f| match f {
            D32_SFLOAT => feature::DEPTH_STENCIL_ATTACHMENT,
            _ => feature::DEPTH_STENCIL_ATTACHMENT | feature::SAMPLED_IMAGE,
        };
        assert_eq!(depth(sampled, d32_unsampled), Some(D32_SFLOAT_S8_UINT));
        assert_eq!(depth(sampled, |_| 0), None);
        let mut asked = Vec::new();
        let _ = depth(Depth::default(), |f| {
            asked.push(f);
            0
        });
        assert_eq!(asked, Depth::default().candidates());
    }

    #[test]
    fn aspects_follow_the_format() {
        assert_eq!(aspect_mask(D32_SFLOAT), aspect::DEPTH);
        assert_eq!(aspect_mask(S8_UINT), aspect::STENCIL);
        assert_eq!(
            aspect_mask(D32_SFLOAT_S8_UINT),
            aspect::DEPTH | aspect::STENCIL
        );
        assert_eq!(aspect_mask(B8G8R8A8_UNORM), aspect::COLOR);
        assert!(is_srgb(B8G8R8A8_SRGB) && !is_srgb(B8G8R8A8_UNORM));
    }
}
