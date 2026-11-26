//! What a DXGI swap chain's pixels mean (`DXGI_COLOR_SPACE_TYPE`, for
//! `IDXGISwapChain3::SetColorSpace1`) as a [`keinbonus`] colour
//! description, and HDR10 metadata in DXGI's coding
//! (`DXGI_HDR_METADATA_HDR10`, for `IDXGISwapChain4::SetHDRMetaData`).
//! With the `vulkan` feature too, `directx::vulkan::color_space` converts
//! to Vulkan's colour spaces.
//!
//! ```
//! use numerobis::directx::modern::color::{self, HdrMetadataHdr10, space};
//! use numerobis::keinbonus::cicp::{Cicp, MatrixCoefficients};
//! use numerobis::keinbonus::hdr::formats::Hdr10;
//! use numerobis::keinbonus::hdr::{ContentLight, MasteringDisplay};
//!
//! // HDR10's signal, as RGB (no matrix) in full range.
//! let rgb = Cicp { matrix: MatrixCoefficients::Identity, full_range: true, ..Cicp::HDR10 };
//! assert_eq!(color::cicp(space::RGB_FULL_G2084_NONE_P2020), Some(rgb));
//! let stream = Hdr10 {
//!     mastering: MasteringDisplay { max_luminance: 1000.0, min_luminance: 0.001, ..Default::default() },
//!     content: ContentLight { max_cll: 2000, max_fall: 500 },
//! };
//! let m = HdrMetadataHdr10::from(&stream);
//! assert_eq!((m.max_mastering_luminance, m.min_mastering_luminance), (1000, 10));
//! ```

use keinbonus::cicp::{
    Cicp, ColourPrimaries as P, MatrixCoefficients, TransferCharacteristics as T,
};
use keinbonus::hdr::formats::Hdr10;

/// The RGB `DXGI_COLOR_SPACE_TYPE` values: what a swap chain takes. The
/// YCbCr ones (video surfaces) aren't listed yet.
pub mod space {
    /// sRGB: "intended to be implemented with sRGB gamma (linear segment +
    /// 2.4 power)". The default.
    pub const RGB_FULL_G22_NONE_P709: u32 = 0;
    /// scRGB: linear BT.709, float formats, 1.0 at 80 cd/m².
    pub const RGB_FULL_G10_NONE_P709: u32 = 1;
    /// BT.709 RGB, limited range.
    pub const RGB_STUDIO_G22_NONE_P709: u32 = 2;
    /// BT.2020 RGB, limited range, "gamma 2.2".
    pub const RGB_STUDIO_G22_NONE_P2020: u32 = 3;
    /// HDR10 RGB: BT.2020, PQ, full range.
    pub const RGB_FULL_G2084_NONE_P2020: u32 = 12;
    /// BT.2020, PQ, limited range.
    pub const RGB_STUDIO_G2084_NONE_P2020: u32 = 14;
    /// BT.2020 RGB, full range, "gamma 2.2".
    pub const RGB_FULL_G22_NONE_P2020: u32 = 17;
    /// BT.709 RGB, limited range, gamma 2.4.
    pub const RGB_STUDIO_G24_NONE_P709: u32 = 20;
    /// BT.2020 RGB, limited range, gamma 2.4.
    pub const RGB_STUDIO_G24_NONE_P2020: u32 = 21;
    pub const CUSTOM: u32 = 0xFFFF_FFFF;
}

/// `DXGI_HDR_METADATA_TYPE`.
pub mod metadata_type {
    pub const NONE: u32 = 0;
    pub const HDR10: u32 = 1;
    pub const HDR10PLUS: u32 = 2;
}

const fn rgb(primaries: P, transfer: T, full_range: bool) -> Cicp {
    Cicp {
        primaries,
        transfer,
        matrix: MatrixCoefficients::Identity,
        full_range,
    }
}

/// The H.273 description of a DXGI colour space where DXGI's text makes it
/// unambiguous: sRGB, scRGB (linear BT.709), BT.709 studio RGB, and PQ in
/// BT.2020 both ranges. `None` for the "gamma 2.2" BT.2020 ones (DXGI
/// says only "commonly used for sRGB and BT.709") and the gamma 2.4 ones
/// (a display gamma, not a signal's curve: BT.1886 showing the BT.709
/// curve comes closest), and for the rest.
#[must_use]
pub const fn cicp(space: u32) -> Option<Cicp> {
    Some(match space {
        space::RGB_FULL_G22_NONE_P709 => rgb(P::Bt709, T::Srgb, true),
        space::RGB_FULL_G10_NONE_P709 => rgb(P::Bt709, T::Linear, true),
        space::RGB_STUDIO_G22_NONE_P709 => rgb(P::Bt709, T::Bt709, false),
        space::RGB_FULL_G2084_NONE_P2020 => rgb(P::Bt2020, T::Pq, true),
        space::RGB_STUDIO_G2084_NONE_P2020 => rgb(P::Bt2020, T::Pq, false),
        _ => return None,
    })
}

/// `DXGI_HDR_METADATA_HDR10`: chromaticities normalized to 50 000, the
/// mastering peak in whole cd/m², its black in 0.0001 cd/m², the content
/// light levels in cd/m².
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Hash)]
pub struct HdrMetadataHdr10 {
    /// x, y.
    pub red_primary: [u16; 2],
    pub green_primary: [u16; 2],
    pub blue_primary: [u16; 2],
    pub white_point: [u16; 2],
    pub max_mastering_luminance: u32,
    pub min_mastering_luminance: u32,
    pub max_content_light_level: u16,
    pub max_frame_average_light_level: u16,
}

const _: () = assert!(size_of::<HdrMetadataHdr10>() == 28);

impl From<&Hdr10> for HdrMetadataHdr10 {
    /// The chromaticities and the black level code as HEVC's SEI does
    /// (keinbonus's [`hevc`](keinbonus::hdr::MasteringDisplay::hevc)); the
    /// peak rounds to whole cd/m².
    #[allow(
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss,
        reason = "clamped to the field"
    )]
    fn from(m: &Hdr10) -> Self {
        let h = m.mastering.hevc();
        // HEVC lists green, blue, red.
        let xy = |i: usize| [h.display_primaries_x[i], h.display_primaries_y[i]];
        Self {
            red_primary: xy(2),
            green_primary: xy(0),
            blue_primary: xy(1),
            white_point: [h.white_point_x, h.white_point_y],
            max_mastering_luminance: m
                .mastering
                .max_luminance
                .round()
                .clamp(0.0, f64::from(u32::MAX)) as u32,
            min_mastering_luminance: h.min_display_mastering_luminance,
            max_content_light_level: m.content.max_cll,
            max_frame_average_light_level: m.content.max_fall,
        }
    }
}

#[cfg(test)]
mod tests {
    use keinbonus::hdr::{ContentLight, MasteringDisplay};

    use super::*;

    #[test]
    fn swap_chain_spaces_describe_as_rgb() {
        assert_eq!(cicp(space::RGB_FULL_G22_NONE_P709), Some(Cicp::SRGB));
        let hdr10 = cicp(space::RGB_STUDIO_G2084_NONE_P2020).unwrap();
        assert_eq!((hdr10.transfer, hdr10.full_range), (T::Pq, false));
        assert_eq!(cicp(space::RGB_FULL_G22_NONE_P2020), None);
        assert_eq!(cicp(space::RGB_STUDIO_G24_NONE_P709), None);
        assert_eq!(cicp(space::CUSTOM), None);
    }

    #[test]
    fn hdr10_metadata_codes_as_dxgi_documents_it() {
        // Microsoft's example: a DCI-P3 D65 mastering display of
        // 1000 / 0.001 cd/m², content up to 2000, frames up to 500.
        let m = Hdr10 {
            mastering: MasteringDisplay {
                primaries: P::Smpte432.chromaticities().unwrap(),
                max_luminance: 1000.0,
                min_luminance: 0.001,
            },
            content: ContentLight {
                max_cll: 2000,
                max_fall: 500,
            },
        };
        let d = HdrMetadataHdr10::from(&m);
        assert_eq!(d.red_primary, [34_000, 16_000]);
        assert_eq!(d.green_primary, [13_250, 34_500]);
        assert_eq!(d.blue_primary, [7_500, 3_000]);
        assert_eq!(d.white_point, [15_635, 16_450]);
        assert_eq!(
            (d.max_mastering_luminance, d.min_mastering_luminance),
            (1000, 10)
        );
        assert_eq!(
            (d.max_content_light_level, d.max_frame_average_light_level),
            (2000, 500)
        );
    }
}
