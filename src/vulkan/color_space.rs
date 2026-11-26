//! What a swapchain's pixels mean: `VkColorSpaceKHR` as a [`keinbonus`]
//! colour description, and the HDR metadata a swapchain is given
//! (`VkHdrMetadataEXT`). Which colour space and format to take is
//! [`swapchain::output_format`](super::swapchain::output_format).
//!
//! Every colour space but sRGB non-linear needs the instance extension
//! `VK_EXT_swapchain_colorspace`
//! ([`Instance::with_color_spaces`](super::Instance::with_color_spaces),
//! [`Capability::SwapchainColorSpace`](super::Capability::SwapchainColorSpace));
//! the metadata needs `VK_EXT_hdr_metadata`
//! ([`Capability::HdrMetadata`](super::Capability::HdrMetadata)).
//!
//! ```
//! use numerobis::keinbonus::cicp::Cicp;
//! use numerobis::keinbonus::hdr::formats::Hdr10;
//! use numerobis::keinbonus::hdr::{ContentLight, MasteringDisplay};
//! use numerobis::vulkan::color_space::{ColorSpace, HdrMetadata};
//!
//! // HDR10 video presented as it is: the swapchain's colour space for its
//! // signal (the matrix and range are the decoder's to undo).
//! let space = ColorSpace::for_signal(&Cicp::HDR10).unwrap();
//! assert_eq!(space, ColorSpace::Hdr10St2084);
//! assert_eq!(space.raw(), 1_000_104_008);
//! // vkSetHdrMetadataEXT with the stream's ST 2086 and CTA-861.3 values.
//! let stream = Hdr10 {
//!     mastering: MasteringDisplay { max_luminance: 1000.0, min_luminance: 0.005, ..Default::default() },
//!     content: ContentLight { max_cll: 1200, max_fall: 400 },
//! };
//! let metadata = HdrMetadata::from(&stream);
//! assert_eq!(metadata.max_content_light_level, 1200.0);
//! ```

use std::ffi::{CStr, c_void};
use std::fmt;

use keinbonus::cicp::{
    Cicp, ColourPrimaries as P, MatrixCoefficients, TransferCharacteristics as T,
};
use keinbonus::hdr::formats::Hdr10;
use keinbonus::primaries::{self, Chromaticities, Xy};

use super::features::s_type;

/// The instance extension behind every colour space but sRGB non-linear.
pub const SWAPCHAIN_COLORSPACE: &CStr = c"VK_EXT_swapchain_colorspace";

/// `VkColorSpaceKHR`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ColorSpace {
    /// sRGB: BT.709 primaries, the sRGB curve. Every surface has it.
    SrgbNonlinear,
    /// Display P3 (P3 primaries, D65) with the sRGB curve. (Vulkan's
    /// Display-P3 OETF has slightly different constants; "for most uses
    /// the sRGB OETF is equivalent", the spec says.)
    DisplayP3Nonlinear,
    /// scRGB, linear: BT.709 primaries, values below 0 and above 1 allowed
    /// (a float format), 1.0 at 80 cd/m² ([`reference_white`](Self::reference_white)).
    ExtendedSrgbLinear,
    /// Display P3, linear.
    DisplayP3Linear,
    /// DCI-P3 as cinema encodes it: CIE XYZ (not RGB in P3 primaries),
    /// each component `(X / 52.37 cd/m²)^(1/2.6)`, which is SMPTE ST 428-1.
    DciP3Nonlinear,
    /// BT.709 primaries, linear.
    Bt709Linear,
    /// BT.709 primaries and the BT.709 curve.
    Bt709Nonlinear,
    /// BT.2020 primaries, linear.
    Bt2020Linear,
    /// HDR10: BT.2020 primaries, PQ (SMPTE ST 2084).
    Hdr10St2084,
    /// Legacy: PQ followed by a proprietary Dolby OOTF with no way to pass
    /// its metadata. Kept for old drivers; no description here.
    DolbyVision,
    /// BT.2020 primaries, HLG (BT.2100).
    Hdr10Hlg,
    /// Adobe RGB (1998), linear. Its primaries have no H.273 code point.
    AdobeRgbLinear,
    /// Adobe RGB (1998), `V = L^(256/563)`. Neither has a code point.
    AdobeRgbNonlinear,
    /// The values reach the display as they are.
    PassThrough,
    /// scRGB, non-linear: the sRGB curve mirrored below 0, values beyond
    /// 0..1 allowed, 1.0 at 80 cd/m².
    ExtendedSrgbNonlinear,
    /// The display's native colour space (AMD `FreeSync` 2), with
    /// `VK_AMD_display_native_hdr`. What it is depends on the display.
    DisplayNativeAmd,
}

/// Adobe RGB (1998) primaries, D65.
pub const ADOBE_RGB: Chromaticities = Chromaticities {
    red: primaries::xy(0.64, 0.33),
    green: primaries::xy(0.21, 0.71),
    blue: primaries::xy(0.15, 0.06),
    white: primaries::D65,
};

/// An RGB description: no matrix, full range, as swapchain images are.
const fn rgb(primaries: P, transfer: T) -> Cicp {
    Cicp {
        primaries,
        transfer,
        matrix: MatrixCoefficients::Identity,
        full_range: true,
    }
}

impl ColorSpace {
    pub const ALL: [Self; 16] = [
        Self::SrgbNonlinear,
        Self::DisplayP3Nonlinear,
        Self::ExtendedSrgbLinear,
        Self::DisplayP3Linear,
        Self::DciP3Nonlinear,
        Self::Bt709Linear,
        Self::Bt709Nonlinear,
        Self::Bt2020Linear,
        Self::Hdr10St2084,
        Self::DolbyVision,
        Self::Hdr10Hlg,
        Self::AdobeRgbLinear,
        Self::AdobeRgbNonlinear,
        Self::PassThrough,
        Self::ExtendedSrgbNonlinear,
        Self::DisplayNativeAmd,
    ];

    #[must_use]
    pub const fn raw(self) -> u32 {
        match self {
            Self::SrgbNonlinear => 0,
            Self::DisplayP3Nonlinear => 1_000_104_001,
            Self::ExtendedSrgbLinear => 1_000_104_002,
            Self::DisplayP3Linear => 1_000_104_003,
            Self::DciP3Nonlinear => 1_000_104_004,
            Self::Bt709Linear => 1_000_104_005,
            Self::Bt709Nonlinear => 1_000_104_006,
            Self::Bt2020Linear => 1_000_104_007,
            Self::Hdr10St2084 => 1_000_104_008,
            Self::DolbyVision => 1_000_104_009,
            Self::Hdr10Hlg => 1_000_104_010,
            Self::AdobeRgbLinear => 1_000_104_011,
            Self::AdobeRgbNonlinear => 1_000_104_012,
            Self::PassThrough => 1_000_104_013,
            Self::ExtendedSrgbNonlinear => 1_000_104_014,
            Self::DisplayNativeAmd => 1_000_213_000,
        }
    }

    /// `None` for a value these tables don't know. The legacy aliases
    /// (`VK_COLORSPACE_SRGB_NONLINEAR_KHR`, `VK_COLOR_SPACE_DCI_P3_LINEAR_EXT`)
    /// share their values with what they alias.
    #[must_use]
    pub const fn from_raw(raw: u32) -> Option<Self> {
        let mut i = 0;
        while i < Self::ALL.len() {
            if Self::ALL[i].raw() == raw {
                return Some(Self::ALL[i]);
            }
            i += 1;
        }
        None
    }

    #[must_use]
    pub const fn name(self) -> &'static str {
        match self {
            Self::SrgbNonlinear => "srgb_nonlinear",
            Self::DisplayP3Nonlinear => "display_p3_nonlinear",
            Self::ExtendedSrgbLinear => "extended_srgb_linear",
            Self::DisplayP3Linear => "display_p3_linear",
            Self::DciP3Nonlinear => "dci_p3_nonlinear",
            Self::Bt709Linear => "bt709_linear",
            Self::Bt709Nonlinear => "bt709_nonlinear",
            Self::Bt2020Linear => "bt2020_linear",
            Self::Hdr10St2084 => "hdr10_st2084",
            Self::DolbyVision => "dolbyvision",
            Self::Hdr10Hlg => "hdr10_hlg",
            Self::AdobeRgbLinear => "adobergb_linear",
            Self::AdobeRgbNonlinear => "adobergb_nonlinear",
            Self::PassThrough => "pass_through",
            Self::ExtendedSrgbNonlinear => "extended_srgb_nonlinear",
            Self::DisplayNativeAmd => "display_native_amd",
        }
    }

    /// The extension that defines it: `VK_KHR_surface` for sRGB
    /// non-linear, `VK_AMD_display_native_hdr` (a device extension) for the
    /// AMD one, [`SWAPCHAIN_COLORSPACE`] (an instance extension) for the
    /// rest.
    #[must_use]
    pub const fn extension(self) -> &'static CStr {
        match self {
            Self::SrgbNonlinear => c"VK_KHR_surface",
            Self::DisplayNativeAmd => c"VK_AMD_display_native_hdr",
            _ => SWAPCHAIN_COLORSPACE,
        }
    }

    /// The H.273 description of the pixels (RGB, full range), where H.273
    /// has the code points: not for Adobe RGB, Dolby Vision, pass-through
    /// and the display's native space. The scRGB spaces describe as their
    /// curve and primaries; that values beyond 0..1 count is
    /// [`extended`](Self::extended), and keinbonus's curves take them
    /// (linear as it is, sRGB mirrored below 0, as scRGB defines it).
    #[must_use]
    pub const fn cicp(self) -> Option<Cicp> {
        Some(match self {
            Self::SrgbNonlinear | Self::ExtendedSrgbNonlinear => rgb(P::Bt709, T::Srgb),
            Self::DisplayP3Nonlinear => rgb(P::Smpte432, T::Srgb),
            Self::ExtendedSrgbLinear | Self::Bt709Linear => rgb(P::Bt709, T::Linear),
            Self::DisplayP3Linear => rgb(P::Smpte432, T::Linear),
            Self::DciP3Nonlinear => rgb(P::Smpte428, T::Smpte428),
            Self::Bt709Nonlinear => rgb(P::Bt709, T::Bt709),
            Self::Bt2020Linear => rgb(P::Bt2020, T::Linear),
            Self::Hdr10St2084 => rgb(P::Bt2020, T::Pq),
            Self::Hdr10Hlg => rgb(P::Bt2020, T::Hlg),
            Self::DolbyVision
            | Self::AdobeRgbLinear
            | Self::AdobeRgbNonlinear
            | Self::PassThrough
            | Self::DisplayNativeAmd => return None,
        })
    }

    /// The primaries and white point: those of [`cicp`](Self::cicp), and
    /// [`ADOBE_RGB`]. For DCI-P3 non-linear they are XYZ's (white E),
    /// as the components are X, Y and Z.
    #[must_use]
    pub fn chromaticities(self) -> Option<Chromaticities> {
        match self {
            Self::AdobeRgbLinear | Self::AdobeRgbNonlinear => Some(ADOBE_RGB),
            _ => self.cicp()?.primaries.chromaticities().ok(),
        }
    }

    /// The colour space a renderer presents a signal of `cicp` in without
    /// converting it, its matrix and range undone; `None` where Vulkan has
    /// none (BT.2020 with the SDR curve, BT.601 primaries, ...). The BT.709
    /// curve under any of its names (BT.601, BT.2020 10- and 12-bit) is
    /// one curve. Never one of the extended spaces: values beyond 0..1 come
    /// from rendering, not from a signal.
    #[must_use]
    pub const fn for_signal(cicp: &Cicp) -> Option<Self> {
        Some(match (cicp.primaries, cicp.transfer) {
            (P::Bt709, T::Srgb) => Self::SrgbNonlinear,
            (P::Bt709, T::Bt709 | T::Bt601 | T::Bt2020_10 | T::Bt2020_12) => Self::Bt709Nonlinear,
            (P::Bt709, T::Linear) => Self::Bt709Linear,
            (P::Smpte432, T::Srgb) => Self::DisplayP3Nonlinear,
            (P::Smpte432, T::Linear) => Self::DisplayP3Linear,
            (P::Smpte428, T::Smpte428) => Self::DciP3Nonlinear,
            (P::Bt2020, T::Linear) => Self::Bt2020Linear,
            (P::Bt2020, T::Pq) => Self::Hdr10St2084,
            (P::Bt2020, T::Hlg) => Self::Hdr10Hlg,
            _ => return None,
        })
    }

    /// Values below 0 and above 1 mean colours outside the primaries and
    /// light above reference white: the scRGB spaces, with a float format.
    #[must_use]
    pub const fn extended(self) -> bool {
        matches!(self, Self::ExtendedSrgbLinear | Self::ExtendedSrgbNonlinear)
    }

    /// Whether it carries light above SDR white: PQ, HLG, Dolby Vision and
    /// scRGB. The display's native space is the display's to say.
    #[must_use]
    pub const fn is_hdr(self) -> bool {
        matches!(
            self,
            Self::Hdr10St2084
                | Self::Hdr10Hlg
                | Self::DolbyVision
                | Self::ExtendedSrgbLinear
                | Self::ExtendedSrgbNonlinear
        )
    }

    /// The light 1.0 stands for, in cd/m², where the colour space fixes it:
    /// 80 for scRGB (IEC 61966-2-2's reference display, and what Windows
    /// composes scRGB with). Use it as keinbonus's
    /// [`Luminance::sdr_white`](keinbonus::hdr::Luminance::sdr_white) for
    /// this space. PQ is absolute and HLG relative to the display's peak;
    /// for the SDR spaces it is the host's SDR white (BT.2408: 203).
    #[must_use]
    pub const fn reference_white(self) -> Option<f64> {
        if self.extended() { Some(80.0) } else { None }
    }
}

impl fmt::Display for ColorSpace {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.name())
    }
}

/// `VkXYColorEXT`.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct XyColor {
    pub x: f32,
    pub y: f32,
}

impl From<Xy> for XyColor {
    #[allow(clippy::cast_possible_truncation, reason = "f32 is the field's type")]
    fn from(c: Xy) -> Self {
        Self {
            x: c.x as f32,
            y: c.y as f32,
        }
    }
}

/// `VkHdrMetadataEXT`, for `vkSetHdrMetadataEXT`: SMPTE ST 2086 (the
/// mastering display, chromaticities and cd/m²) and CTA-861.3 (content
/// light levels, cd/m², 0 unknown).
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct HdrMetadata {
    pub s_type: i32,
    pub p_next: *const c_void,
    pub display_primary_red: XyColor,
    pub display_primary_green: XyColor,
    pub display_primary_blue: XyColor,
    pub white_point: XyColor,
    pub max_luminance: f32,
    pub min_luminance: f32,
    pub max_content_light_level: f32,
    pub max_frame_average_light_level: f32,
}

const _: () = assert!(size_of::<HdrMetadata>() == super::HEADER + 48);

impl Default for HdrMetadata {
    fn default() -> Self {
        Self {
            s_type: s_type::HDR_METADATA,
            p_next: std::ptr::null(),
            display_primary_red: XyColor::default(),
            display_primary_green: XyColor::default(),
            display_primary_blue: XyColor::default(),
            white_point: XyColor::default(),
            max_luminance: 0.0,
            min_luminance: 0.0,
            max_content_light_level: 0.0,
            max_frame_average_light_level: 0.0,
        }
    }
}

impl From<&Hdr10> for HdrMetadata {
    #[allow(clippy::cast_possible_truncation, reason = "f32 is the field's type")]
    fn from(m: &Hdr10) -> Self {
        let p = &m.mastering.primaries;
        Self {
            display_primary_red: p.red.into(),
            display_primary_green: p.green.into(),
            display_primary_blue: p.blue.into(),
            white_point: p.white.into(),
            max_luminance: m.mastering.max_luminance as f32,
            min_luminance: m.mastering.min_luminance as f32,
            max_content_light_level: f32::from(m.content.max_cll),
            max_frame_average_light_level: f32::from(m.content.max_fall),
            ..Self::default()
        }
    }
}

#[cfg(test)]
mod tests {
    use keinbonus::hdr::{ContentLight, MasteringDisplay};

    use super::*;

    #[test]
    fn raw_values_round_trip() {
        for c in ColorSpace::ALL {
            assert_eq!(ColorSpace::from_raw(c.raw()), Some(c));
        }
        assert_eq!(ColorSpace::from_raw(1_000_104_015), None);
        assert_eq!(ColorSpace::SrgbNonlinear.extension(), c"VK_KHR_surface");
        assert_eq!(ColorSpace::Hdr10Hlg.extension(), SWAPCHAIN_COLORSPACE);
    }

    #[test]
    fn descriptions_match_the_vulkan_table() {
        // Vulkan's "Color Spaces and Attributes" table, red, green, blue,
        // white.
        let table = [
            (
                ColorSpace::SrgbNonlinear,
                [
                    (0.640, 0.330),
                    (0.300, 0.600),
                    (0.150, 0.060),
                    (0.3127, 0.3290),
                ],
            ),
            (
                ColorSpace::DisplayP3Nonlinear,
                [
                    (0.680, 0.320),
                    (0.265, 0.690),
                    (0.150, 0.060),
                    (0.3127, 0.3290),
                ],
            ),
            (
                ColorSpace::Hdr10St2084,
                [
                    (0.708, 0.292),
                    (0.170, 0.797),
                    (0.131, 0.046),
                    (0.3127, 0.3290),
                ],
            ),
            (
                ColorSpace::AdobeRgbNonlinear,
                [
                    (0.640, 0.330),
                    (0.210, 0.710),
                    (0.150, 0.060),
                    (0.3127, 0.3290),
                ],
            ),
            (
                ColorSpace::DciP3Nonlinear,
                [(1.0, 0.0), (0.0, 1.0), (0.0, 0.0), (1.0 / 3.0, 1.0 / 3.0)],
            ),
        ];
        for (space, rgbw) in table {
            let c = space.chromaticities().unwrap();
            for (got, want) in [c.red, c.green, c.blue, c.white].iter().zip(rgbw) {
                assert!(
                    (got.x - want.0).abs() < 1e-4 && (got.y - want.1).abs() < 1e-4,
                    "{space}: {got:?} against {want:?}"
                );
            }
        }
        assert_eq!(ColorSpace::PassThrough.chromaticities(), None);
    }

    #[test]
    fn signals_find_their_colour_space() {
        for c in ColorSpace::ALL {
            let Some(cicp) = c.cicp() else { continue };
            let back = ColorSpace::for_signal(&cicp).unwrap();
            let twin = match c {
                ColorSpace::ExtendedSrgbLinear => ColorSpace::Bt709Linear,
                ColorSpace::ExtendedSrgbNonlinear => ColorSpace::SrgbNonlinear,
                c => c,
            };
            assert_eq!(back, twin);
        }
        assert_eq!(
            ColorSpace::for_signal(&Cicp::BT709),
            Some(ColorSpace::Bt709Nonlinear)
        );
        assert_eq!(
            ColorSpace::for_signal(&Cicp::HLG),
            Some(ColorSpace::Hdr10Hlg)
        );
        assert_eq!(
            ColorSpace::for_signal(&Cicp::DISPLAY_P3),
            Some(ColorSpace::DisplayP3Nonlinear)
        );
        assert_eq!(
            ColorSpace::for_signal(&Cicp::BT2020),
            None,
            "no BT.2020 SDR space"
        );
        assert_eq!(ColorSpace::for_signal(&Cicp::BT601_525), None);
        assert!(ColorSpace::ExtendedSrgbLinear.is_hdr() && !ColorSpace::Bt2020Linear.is_hdr());
        assert_eq!(
            ColorSpace::ExtendedSrgbNonlinear.reference_white(),
            Some(80.0)
        );
        assert_eq!(ColorSpace::Hdr10St2084.reference_white(), None);
    }

    #[test]
    fn hdr10_metadata_fills_the_vulkan_struct() {
        let m = Hdr10 {
            mastering: MasteringDisplay {
                primaries: P::Smpte432.chromaticities().unwrap(),
                max_luminance: 4000.0,
                min_luminance: 0.0050,
            },
            content: ContentLight {
                max_cll: 1000,
                max_fall: 180,
            },
        };
        let v = HdrMetadata::from(&m);
        assert_eq!(v.s_type, 1_000_105_000);
        assert!(v.p_next.is_null());
        assert_eq!(v.display_primary_green, XyColor { x: 0.265, y: 0.690 });
        assert_eq!(
            v.white_point,
            XyColor {
                x: 0.3127,
                y: 0.3290
            }
        );
        assert_eq!((v.max_luminance, v.min_luminance), (4000.0, 0.005));
        assert_eq!(
            (v.max_content_light_level, v.max_frame_average_light_level),
            (1000.0, 180.0)
        );
    }
}
