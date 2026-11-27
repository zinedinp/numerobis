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

/// One block of an aspect as a buffer holds it in a buffer-image copy:
/// `width` x `height` texels in `bytes` (1 x 1 for uncompressed formats).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Block {
    pub bytes: u32,
    pub width: u32,
    pub height: u32,
}

impl Block {
    const fn texel(bytes: u32) -> Self {
        Self {
            bytes,
            width: 1,
            height: 1,
        }
    }

    const fn of(bytes: u32, width: u32, height: u32) -> Self {
        Self {
            bytes,
            width,
            height,
        }
    }
}

/// One block of the single aspect `aspect` (an [`aspect`] bit) of
/// `format`, for the bounds of a buffer-image copy: depth and stencil are
/// copied one aspect at a time, each tightly packed (`D24` in 4 bytes,
/// stencil in 1). Covers the core formats (1 to 184: uncompressed, BC,
/// ETC2, EAC, ASTC); `None` for the others (multi-planar, `_422`, the
/// extensions' packed ones) and for an aspect the format lacks.
#[must_use]
pub const fn block(format: u32, aspect: u32) -> Option<Block> {
    // Depth and stencil bytes per texel; 0 where the format has none.
    let (depth, stencil) = match format {
        D16_UNORM => (2, 0),
        X8_D24_UNORM_PACK32 | D32_SFLOAT => (4, 0),
        S8_UINT => (0, 1),
        D16_UNORM_S8_UINT => (2, 1),
        D24_UNORM_S8_UINT | D32_SFLOAT_S8_UINT => (4, 1),
        _ => (0, 0),
    };
    if depth | stencil != 0 {
        return match aspect {
            aspect::DEPTH if depth != 0 => Some(Block::texel(depth)),
            aspect::STENCIL if stencil != 0 => Some(Block::texel(stencil)),
            _ => None,
        };
    }
    if aspect != aspect::COLOR {
        return None;
    }
    // VkFormat's core values run in groups of one layout; the comments
    // name the first and last of each.
    match format {
        // R4G4_UNORM_PACK8; R8_UNORM ..= R8_SRGB.
        1 | 9..=15 => Some(Block::texel(1)),
        // R4G4B4A4_UNORM_PACK16 ..= A1R5G5B5_UNORM_PACK16; R8G8_UNORM
        // ..= R8G8_SRGB; R16_UNORM ..= R16_SFLOAT.
        2..=8 | 16..=22 | 70..=76 => Some(Block::texel(2)),
        // R8G8B8_UNORM ..= B8G8R8_SRGB.
        23..=36 => Some(Block::texel(3)),
        // R8G8B8A8_UNORM ..= A2B10G10R10_SINT_PACK32; R16G16_UNORM ..=
        // R16G16_SFLOAT; R32_UINT ..= R32_SFLOAT; B10G11R11_UFLOAT_PACK32,
        // E5B9G9R9_UFLOAT_PACK32.
        37..=69 | 77..=83 | 98..=100 | 122 | 123 => Some(Block::texel(4)),
        // R16G16B16_UNORM ..= R16G16B16_SFLOAT.
        84..=90 => Some(Block::texel(6)),
        // R16G16B16A16_UNORM ..= R16G16B16A16_SFLOAT; R32G32_UINT ..=
        // R32G32_SFLOAT; R64_UINT ..= R64_SFLOAT.
        91..=97 | 101..=103 | 110..=112 => Some(Block::texel(8)),
        // R32G32B32_UINT ..= R32G32B32_SFLOAT.
        104..=106 => Some(Block::texel(12)),
        // R32G32B32A32_UINT ..= R32G32B32A32_SFLOAT; R64G64_UINT ..=
        // R64G64_SFLOAT.
        107..=109 | 113..=115 => Some(Block::texel(16)),
        // R64G64B64_UINT ..= R64G64B64_SFLOAT.
        116..=118 => Some(Block::texel(24)),
        // R64G64B64A64_UINT ..= R64G64B64A64_SFLOAT.
        119..=121 => Some(Block::texel(32)),
        // BC1_RGB_UNORM_BLOCK ..= BC1_RGBA_SRGB_BLOCK; BC4_UNORM_BLOCK,
        // BC4_SNORM_BLOCK; ETC2_R8G8B8_UNORM_BLOCK ..=
        // ETC2_R8G8B8A1_SRGB_BLOCK; EAC_R11_UNORM_BLOCK, EAC_R11_SNORM_BLOCK.
        131..=134 | 139 | 140 | 147..=150 | 153 | 154 => Some(Block::of(8, 4, 4)),
        // BC2_UNORM_BLOCK ..= BC3_SRGB_BLOCK; BC5_UNORM_BLOCK ..=
        // BC7_SRGB_BLOCK; ETC2_R8G8B8A8_UNORM_BLOCK,
        // ETC2_R8G8B8A8_SRGB_BLOCK; EAC_R11G11_UNORM_BLOCK,
        // EAC_R11G11_SNORM_BLOCK.
        135..=138 | 141..=146 | 151 | 152 | 155 | 156 => Some(Block::of(16, 4, 4)),
        // ASTC_4x4_UNORM_BLOCK ..= ASTC_12x12_SRGB_BLOCK: 16 bytes, an
        // UNORM and an SRGB format per footprint.
        157..=184 => {
            const FOOTPRINTS: [(u32, u32); 14] = [
                (4, 4),
                (5, 4),
                (5, 5),
                (6, 5),
                (6, 6),
                (8, 5),
                (8, 6),
                (8, 8),
                (10, 5),
                (10, 6),
                (10, 8),
                (10, 10),
                (12, 10),
                (12, 12),
            ];
            let (width, height) = FOOTPRINTS[((format - 157) / 2) as usize];
            Some(Block::of(16, width, height))
        }
        _ => None,
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

    #[test]
    fn blocks_by_group() {
        let texel = |bytes| Some(Block::texel(bytes));
        assert_eq!(block(R8G8B8A8_UNORM, aspect::COLOR), texel(4));
        assert_eq!(block(R16G16B16A16_SFLOAT, aspect::COLOR), texel(8));
        assert_eq!(block(A2B10G10R10_UNORM_PACK32, aspect::COLOR), texel(4));
        // R4G4_UNORM_PACK8, R5G6B5_UNORM_PACK16, R8G8B8_SRGB, R32G32B32_SFLOAT,
        // R64G64B64A64_SFLOAT.
        assert_eq!(block(1, aspect::COLOR), texel(1));
        assert_eq!(block(4, aspect::COLOR), texel(2));
        assert_eq!(block(29, aspect::COLOR), texel(3));
        assert_eq!(block(106, aspect::COLOR), texel(12));
        assert_eq!(block(121, aspect::COLOR), texel(32));
        // BC1_RGBA_SRGB, BC7_SRGB, ETC2_R8G8B8A8_SRGB, EAC_R11_SNORM.
        assert_eq!(block(134, aspect::COLOR), Some(Block::of(8, 4, 4)));
        assert_eq!(block(146, aspect::COLOR), Some(Block::of(16, 4, 4)));
        assert_eq!(block(152, aspect::COLOR), Some(Block::of(16, 4, 4)));
        assert_eq!(block(154, aspect::COLOR), Some(Block::of(8, 4, 4)));
        // ASTC_4x4_UNORM, ASTC_8x5_SRGB, ASTC_12x12_SRGB.
        assert_eq!(block(157, aspect::COLOR), Some(Block::of(16, 4, 4)));
        assert_eq!(block(168, aspect::COLOR), Some(Block::of(16, 8, 5)));
        assert_eq!(block(184, aspect::COLOR), Some(Block::of(16, 12, 12)));
        assert_eq!(block(UNDEFINED, aspect::COLOR), None);
        // G8_B8_R8_3PLANE_420_UNORM.
        assert_eq!(block(1_000_156_002, aspect::COLOR), None);
        assert_eq!(block(R8G8B8A8_UNORM, aspect::DEPTH), None);
    }

    #[test]
    fn depth_and_stencil_blocks_by_aspect() {
        let texel = |bytes| Some(Block::texel(bytes));
        assert_eq!(block(D24_UNORM_S8_UINT, aspect::DEPTH), texel(4));
        assert_eq!(block(D24_UNORM_S8_UINT, aspect::STENCIL), texel(1));
        assert_eq!(block(D16_UNORM_S8_UINT, aspect::DEPTH), texel(2));
        assert_eq!(block(S8_UINT, aspect::STENCIL), texel(1));
        assert_eq!(block(D32_SFLOAT, aspect::STENCIL), None);
        assert_eq!(block(D24_UNORM_S8_UINT, aspect::COLOR), None);
        // Both aspects at once is no copy.
        assert_eq!(
            block(D24_UNORM_S8_UINT, aspect::DEPTH | aspect::STENCIL),
            None
        );
    }
}
