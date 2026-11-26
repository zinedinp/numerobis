//! Swapchain choices and when to make a new swapchain: present mode,
//! surface format (SDR by [`surface_format`], HDR and wide gamut by
//! [`output_format`]), composite alpha, image count and extent from what the
//! surface reports, and the policy after `vkQueuePresentKHR` and
//! `vkAcquireNextImageKHR`. The capabilities behind the present modes are
//! in [`caps::present`](super::caps::present); what a colour space means is
//! in [`color_space`](super::color_space).
//!
//! ```
//! use numerobis::vulkan::format;
//! use numerobis::vulkan::swapchain::{
//!     self, Encoding, Extent, Pacing, PresentMode, SurfaceCapabilities, SurfaceFormat, result,
//! };
//!
//! // vkGetPhysicalDeviceSurfaceCapabilitiesKHR (the same layout, so the
//! // host may also read it with `SurfaceCapabilities::from_raw`).
//! let caps = SurfaceCapabilities {
//!     min_image_count: 2,
//!     max_image_count: 8,
//!     current_extent: Extent::new(1920, 1080),
//!     supported_composite_alpha: swapchain::alpha::OPAQUE,
//!     ..SurfaceCapabilities::default()
//! };
//! // vkGetPhysicalDeviceSurfaceFormatsKHR and ...PresentModesKHR.
//! let formats = [
//!     SurfaceFormat::new(format::B8G8R8A8_SRGB),
//!     SurfaceFormat::new(format::B8G8R8A8_UNORM),
//! ];
//! let modes = [PresentMode::Fifo.raw(), PresentMode::Mailbox.raw()];
//!
//! let extent = caps.extent(Extent::new(1280, 720)).expect("not minimized");
//! let format = swapchain::surface_format(&formats, Encoding::Srgb).unwrap();
//! let mode = swapchain::present_mode(&modes, Pacing::LowLatency, None, false);
//! assert_eq!(extent, Extent::new(1920, 1080), "the surface's size wins");
//! assert_eq!(format.format, format::B8G8R8A8_SRGB);
//! assert_eq!(mode, PresentMode::Mailbox);
//! assert_eq!(caps.image_count(3), 3);
//! // ...vkCreateSwapchainKHR, and later after vkQueuePresentKHR:
//! let again = swapchain::recreate(result::SUBOPTIMAL, extent, || caps.current());
//! assert_eq!(again, Ok(false), "suboptimal at the same size: keep it");
//! ```

use std::ffi::c_void;
use std::fmt;

use super::color_space::ColorSpace;
use super::format;

/// The `VkResult`s that concern the swapchain.
pub mod result {
    pub const SUCCESS: i32 = 0;
    pub const NOT_READY: i32 = 1;
    pub const TIMEOUT: i32 = 2;
    /// `VK_SUBOPTIMAL_KHR`: presented (or acquired), but the swapchain no
    /// longer matches the surface exactly.
    pub const SUBOPTIMAL: i32 = 1_000_001_003;
    /// `VK_ERROR_OUT_OF_DATE_KHR`: the swapchain can't be presented to.
    pub const OUT_OF_DATE: i32 = -1_000_001_004;
    /// `VK_ERROR_SURFACE_LOST_KHR`: the surface is gone; a new swapchain
    /// needs a new surface.
    pub const SURFACE_LOST: i32 = -1_000_000_000;
}

/// `VkCompositeAlphaFlagBitsKHR`.
pub mod alpha {
    pub const OPAQUE: u32 = 0x1;
    pub const PRE_MULTIPLIED: u32 = 0x2;
    pub const POST_MULTIPLIED: u32 = 0x4;
    pub const INHERIT: u32 = 0x8;
}

/// `VK_COLOR_SPACE_SRGB_NONLINEAR_KHR`, the colour space every surface has.
pub const SRGB_NONLINEAR: u32 = ColorSpace::SrgbNonlinear.raw();

/// `VkExtent2D`.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Hash)]
pub struct Extent {
    pub width: u32,
    pub height: u32,
}

impl Extent {
    #[must_use]
    pub const fn new(width: u32, height: u32) -> Self {
        Self { width, height }
    }

    /// No pixels (a minimized window on Windows).
    #[must_use]
    pub const fn is_empty(self) -> bool {
        self.width == 0 || self.height == 0
    }
}

impl fmt::Display for Extent {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}x{}", self.width, self.height)
    }
}

/// `VkSurfaceCapabilitiesKHR`, field for field.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct SurfaceCapabilities {
    pub min_image_count: u32,
    /// 0: no limit.
    pub max_image_count: u32,
    /// `u32::MAX` in both: the swapchain sets the size (Wayland).
    pub current_extent: Extent,
    pub min_image_extent: Extent,
    pub max_image_extent: Extent,
    pub max_image_array_layers: u32,
    pub supported_transforms: u32,
    pub current_transform: u32,
    pub supported_composite_alpha: u32,
    pub supported_usage_flags: u32,
}

const _: () = assert!(size_of::<SurfaceCapabilities>() == 52);

impl SurfaceCapabilities {
    /// From a `VkSurfaceCapabilitiesKHR` the host read.
    ///
    /// # Safety
    ///
    /// `raw` points at a filled `VkSurfaceCapabilitiesKHR`.
    #[must_use]
    pub unsafe fn from_raw(raw: *const c_void) -> Self {
        // SAFETY: the caller's promise; the same layout, every field a u32.
        unsafe { raw.cast::<Self>().read_unaligned() }
    }

    /// The surface's size, `None` where the swapchain sets it.
    #[must_use]
    pub const fn current(&self) -> Option<Extent> {
        if self.current_extent.width == u32::MAX {
            None
        } else {
            Some(self.current_extent)
        }
    }

    /// The swapchain's `imageExtent`: the surface's size, or where the
    /// swapchain sets it, `window` within the surface's limits. `None`
    /// while that has no pixels (minimized): keep the old swapchain until
    /// the window has a size again.
    #[must_use]
    pub fn extent(&self, window: Extent) -> Option<Extent> {
        let (min, max) = (self.min_image_extent, self.max_image_extent);
        let extent = self.current().unwrap_or(Extent {
            width: window.width.clamp(min.width, max.width.max(min.width)),
            height: window.height.clamp(min.height, max.height.max(min.height)),
        });
        (!extent.is_empty()).then_some(extent)
    }

    /// `minImageCount` for the swapchain: one more than the surface's
    /// minimum, so acquiring doesn't wait for the compositor to give an
    /// image back, at least `at_least` (one per frame in flight plus the
    /// one on screen), within the surface's maximum.
    #[must_use]
    pub const fn image_count(&self, at_least: u32) -> u32 {
        let mut count = self.min_image_count.saturating_add(1);
        if count < at_least {
            count = at_least;
        }
        if self.max_image_count > 0 && count > self.max_image_count {
            count = self.max_image_count;
        }
        count
    }

    /// Opaque when the surface offers it, else the first mode it does: the
    /// spec wants one of `supportedCompositeAlpha`.
    #[must_use]
    pub const fn composite_alpha(&self) -> u32 {
        composite_alpha(self.supported_composite_alpha)
    }

    /// Whether swapchain images can have all of `usage`
    /// (`VkImageUsageFlags`, such as `TRANSFER_SRC` for screenshots).
    #[must_use]
    pub const fn supports_usage(&self, usage: u32) -> bool {
        self.supported_usage_flags & usage == usage
    }
}

/// [`SurfaceCapabilities::composite_alpha`] from the bits alone.
#[must_use]
pub const fn composite_alpha(supported: u32) -> u32 {
    let order = [
        alpha::OPAQUE,
        alpha::INHERIT,
        alpha::PRE_MULTIPLIED,
        alpha::POST_MULTIPLIED,
    ];
    let mut i = 0;
    while i < order.len() {
        if supported & order[i] != 0 {
            return order[i];
        }
        i += 1;
    }
    alpha::OPAQUE
}

/// After `vkQueuePresentKHR` or `vkAcquireNextImageKHR` returned `result`:
/// whether to make a new swapchain, or the error.
///
/// - `OUT_OF_DATE`: always.
/// - `SUBOPTIMAL`: only when the surface's size now (`surface`, called only
///   then: [`SurfaceCapabilities::current`]) has pixels and differs from
///   `extent`. Mesa on X11 reports it on every present, so that alone
///   doesn't recreate; where the swapchain sets the size there is nothing
///   to compare, so neither does it.
/// - Other successes (`NOT_READY`, `TIMEOUT` from acquiring): no.
///
/// # Errors
///
/// Any other error (`SURFACE_LOST`, `DEVICE_LOST`, out of memory), as it
/// came: a new swapchain doesn't help.
pub fn recreate(
    result: i32,
    extent: Extent,
    surface: impl FnOnce() -> Option<Extent>,
) -> Result<bool, i32> {
    match result {
        result::SUBOPTIMAL => Ok(surface().is_some_and(|now| !now.is_empty() && now != extent)),
        result::OUT_OF_DATE => Ok(true),
        e if e < 0 => Err(e),
        _ => Ok(false),
    }
}

/// `VkPresentModeKHR`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum PresentMode {
    /// No wait for vertical blank: tears.
    Immediate,
    /// The newest frame at vertical blank, older ones dropped; no wait.
    Mailbox,
    /// A queue shown one per vertical blank; offered everywhere.
    Fifo,
    /// FIFO, but a late frame is shown at once (and may tear).
    FifoRelaxed,
    /// FIFO, but at vertical blank the newest ready frame is shown and the
    /// older ones in the queue dropped (`VK_KHR_present_mode_fifo_latest_ready`).
    FifoLatestReady,
}

impl PresentMode {
    pub const ALL: [Self; 5] = [
        Self::Immediate,
        Self::Mailbox,
        Self::Fifo,
        Self::FifoRelaxed,
        Self::FifoLatestReady,
    ];

    #[must_use]
    pub const fn raw(self) -> u32 {
        match self {
            Self::Immediate => 0,
            Self::Mailbox => 1,
            Self::Fifo => 2,
            Self::FifoRelaxed => 3,
            Self::FifoLatestReady => 1_000_361_000,
        }
    }

    #[must_use]
    pub const fn from_raw(raw: u32) -> Option<Self> {
        match raw {
            0 => Some(Self::Immediate),
            1 => Some(Self::Mailbox),
            2 => Some(Self::Fifo),
            3 => Some(Self::FifoRelaxed),
            1_000_361_000 => Some(Self::FifoLatestReady),
            _ => None,
        }
    }

    #[must_use]
    pub const fn name(self) -> &'static str {
        match self {
            Self::Immediate => "immediate",
            Self::Mailbox => "mailbox",
            Self::Fifo => "fifo",
            Self::FifoRelaxed => "fifo_relaxed",
            Self::FifoLatestReady => "fifo_latest_ready",
        }
    }
}

impl fmt::Display for PresentMode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.name())
    }
}

impl std::str::FromStr for PresentMode {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, String> {
        Self::ALL
            .into_iter()
            .find(|m| m.name().eq_ignore_ascii_case(s.trim()))
            .ok_or_else(|| {
                format!("{s:?}: not immediate, mailbox, fifo, fifo_relaxed or fifo_latest_ready")
            })
    }
}

/// How frames meet the display.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Pacing {
    /// Tear-free, one frame per refresh, the renderer waits for the queue:
    /// FIFO.
    Vsync,
    /// Tear-free, the newest frame at each refresh: FIFO latest ready, else
    /// mailbox, else FIFO.
    LowLatency,
    /// As fast as it renders, tearing allowed: immediate, else mailbox,
    /// else FIFO.
    Uncapped,
}

impl Pacing {
    /// The waterfall; FIFO, last, is offered everywhere.
    #[must_use]
    pub const fn order(self) -> &'static [PresentMode] {
        use PresentMode as M;
        match self {
            Self::Vsync => &[M::Fifo],
            Self::LowLatency => &[M::FifoLatestReady, M::Mailbox, M::Fifo],
            Self::Uncapped => &[M::Immediate, M::Mailbox, M::Fifo],
        }
    }
}

/// The present mode: `requested` when the surface offers it, else the first
/// of `pacing`'s [`order`](Pacing::order) that it does. `offered` is what
/// `vkGetPhysicalDeviceSurfacePresentModesKHR` listed; FIFO counts as
/// offered whatever it says. FIFO latest ready also needs `latest_ready`:
/// the plan has [`FifoLatestReady`](super::Capability::FifoLatestReady) and
/// the device was created with it. A `requested` mode not taken is the
/// host's to report (`chosen != requested`).
#[must_use]
pub fn present_mode(
    offered: &[u32],
    pacing: Pacing,
    requested: Option<PresentMode>,
    latest_ready: bool,
) -> PresentMode {
    let usable = |m: PresentMode| {
        m == PresentMode::Fifo
            || (offered.contains(&m.raw()) && (m != PresentMode::FifoLatestReady || latest_ready))
    };
    requested
        .filter(|&m| usable(m))
        .or_else(|| pacing.order().iter().copied().find(|&m| usable(m)))
        .unwrap_or(PresentMode::Fifo)
}

/// `VkSurfaceFormatKHR`.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct SurfaceFormat {
    pub format: u32,
    /// `VkColorSpaceKHR`, raw: a driver may list values newer than these
    /// tables ([`space`](Self::space)).
    pub color_space: u32,
}

impl SurfaceFormat {
    /// `format` in sRGB non-linear, the colour space every surface has.
    #[must_use]
    pub const fn new(format: u32) -> Self {
        Self::with(format, ColorSpace::SrgbNonlinear)
    }

    #[must_use]
    pub const fn with(format: u32, space: ColorSpace) -> Self {
        Self {
            format,
            color_space: space.raw(),
        }
    }

    /// The colour space, `None` for one these tables don't know.
    #[must_use]
    pub const fn space(&self) -> Option<ColorSpace> {
        ColorSpace::from_raw(self.color_space)
    }
}

/// What the shaders write to the swapchain image.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Encoding {
    /// Linear colour: an `_SRGB` format encodes it on write.
    Srgb,
    /// Colour already sRGB-encoded (as an OpenGL default framebuffer
    /// stores it): a `_UNORM` format, which an `_SRGB` one would encode a
    /// second time.
    Unorm,
}

/// The swapchain's format from what `vkGetPhysicalDeviceSurfaceFormatsKHR`
/// listed: 8-bit BGRA, then RGBA, then ABGR with `encoding`, in sRGB
/// non-linear; then any other format with that encoding, in that colour
/// space first; then the first one listed. A lone `UNDEFINED` (Vulkan 1.0
/// drivers: any format will do) gives BGRA. `None` for an empty list.
#[must_use]
pub fn surface_format(offered: &[SurfaceFormat], encoding: Encoding) -> Option<SurfaceFormat> {
    let srgb = encoding == Encoding::Srgb;
    let preferred: [u32; 3] = if srgb {
        [
            format::B8G8R8A8_SRGB,
            format::R8G8B8A8_SRGB,
            format::A8B8G8R8_SRGB_PACK32,
        ]
    } else {
        [
            format::B8G8R8A8_UNORM,
            format::R8G8B8A8_UNORM,
            format::A8B8G8R8_UNORM_PACK32,
        ]
    };
    if let [only] = offered
        && only.format == format::UNDEFINED
    {
        return Some(SurfaceFormat {
            format: preferred[0],
            color_space: only.color_space,
        });
    }
    let nonlinear = |f: &&SurfaceFormat| f.color_space == SRGB_NONLINEAR;
    let encoded =
        |f: &&SurfaceFormat| f.format != format::UNDEFINED && format::is_srgb(f.format) == srgb;
    preferred
        .iter()
        .find_map(|&wanted| {
            offered
                .iter()
                .filter(nonlinear)
                .find(|f| f.format == wanted)
        })
        .or_else(|| offered.iter().filter(nonlinear).find(encoded))
        .or_else(|| offered.iter().find(encoded))
        .or_else(|| offered.iter().find(|f| f.format != format::UNDEFINED))
        .copied()
}

/// What the renderer writes to the swapchain: a colour space and the
/// formats it takes, best first. [`output_format`] picks among them.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Output {
    /// sRGB non-linear: 8-bit with the encoding, then (for
    /// [`Encoding::Unorm`]) 10:10:10:2.
    Sdr(Encoding),
    /// HDR10: the shaders write PQ (BT.2100) in BT.2020. 10:10:10:2, then
    /// 16-bit float (still PQ values in 0..1). Give the swapchain
    /// [`HdrMetadata`](super::color_space::HdrMetadata) when the plan has
    /// [`Capability::HdrMetadata`](super::Capability::HdrMetadata).
    Hdr10,
    /// BT.2100 HLG in BT.2020: the shaders write the HLG signal; the
    /// display applies the OOTF for its peak. 10:10:10:2, then 16-bit
    /// float.
    Hlg,
    /// scRGB: linear BT.709 in 16-bit float, 1.0 at 80 cd/m²; above 1 is
    /// brighter than that, below 0 outside BT.709. Windows composes in it.
    ScRgb,
    /// Display P3 with the sRGB curve: for [`Encoding::Unorm`] 10:10:10:2
    /// first, then 8-bit; for [`Encoding::Srgb`] the 8-bit `_SRGB` formats
    /// (there are no 10-bit ones).
    DisplayP3(Encoding),
}

impl Output {
    #[must_use]
    pub const fn color_space(self) -> ColorSpace {
        match self {
            Self::Sdr(_) => ColorSpace::SrgbNonlinear,
            Self::Hdr10 => ColorSpace::Hdr10St2084,
            Self::Hlg => ColorSpace::Hdr10Hlg,
            Self::ScRgb => ColorSpace::ExtendedSrgbLinear,
            Self::DisplayP3(_) => ColorSpace::DisplayP3Nonlinear,
        }
    }

    /// The formats it takes, in order. No `_SRGB` format for PQ, HLG or
    /// scRGB: those would apply the sRGB curve on top.
    #[must_use]
    pub const fn formats(self) -> &'static [u32] {
        use format::{
            A2B10G10R10_UNORM_PACK32, A2R10G10B10_UNORM_PACK32, A8B8G8R8_SRGB_PACK32,
            A8B8G8R8_UNORM_PACK32, B8G8R8A8_SRGB, B8G8R8A8_UNORM, R8G8B8A8_SRGB, R8G8B8A8_UNORM,
            R16G16B16A16_SFLOAT,
        };
        const SRGB8: &[u32] = &[B8G8R8A8_SRGB, R8G8B8A8_SRGB, A8B8G8R8_SRGB_PACK32];
        const UNORM: &[u32] = &[
            B8G8R8A8_UNORM,
            R8G8B8A8_UNORM,
            A8B8G8R8_UNORM_PACK32,
            A2B10G10R10_UNORM_PACK32,
            A2R10G10B10_UNORM_PACK32,
        ];
        const WIDE: &[u32] = &[
            A2B10G10R10_UNORM_PACK32,
            A2R10G10B10_UNORM_PACK32,
            B8G8R8A8_UNORM,
            R8G8B8A8_UNORM,
            A8B8G8R8_UNORM_PACK32,
        ];
        const HDR: &[u32] = &[
            A2B10G10R10_UNORM_PACK32,
            A2R10G10B10_UNORM_PACK32,
            R16G16B16A16_SFLOAT,
        ];
        match self {
            Self::Sdr(Encoding::Srgb) | Self::DisplayP3(Encoding::Srgb) => SRGB8,
            Self::Sdr(Encoding::Unorm) => UNORM,
            Self::DisplayP3(Encoding::Unorm) => WIDE,
            Self::Hdr10 | Self::Hlg => HDR,
            Self::ScRgb => &[R16G16B16A16_SFLOAT],
        }
    }
}

/// The first of `wishes` the surface offers (`offered`, from
/// `vkGetPhysicalDeviceSurfaceFormatsKHR`), with its format: per wish the
/// first of its [`formats`](Output::formats) listed in its colour space.
/// A lone `UNDEFINED` (any format) takes the first wish in the colour space
/// it names. `None` when none is offered: then [`surface_format`], which
/// always finds one. Without `VK_EXT_swapchain_colorspace` enabled the
/// surface lists sRGB non-linear only, so only [`Output::Sdr`] can match.
///
/// ```
/// use numerobis::vulkan::color_space::ColorSpace;
/// use numerobis::vulkan::format;
/// use numerobis::vulkan::swapchain::{self, Encoding, Output, SurfaceFormat};
///
/// let offered = [
///     SurfaceFormat::new(format::B8G8R8A8_UNORM),
///     SurfaceFormat::with(format::A2B10G10R10_UNORM_PACK32, ColorSpace::Hdr10St2084),
///     SurfaceFormat::with(format::R16G16B16A16_SFLOAT, ColorSpace::ExtendedSrgbLinear),
/// ];
/// let wishes = [Output::Hdr10, Output::ScRgb, Output::Sdr(Encoding::Unorm)];
/// let (output, chosen) = swapchain::output_format(&offered, &wishes).unwrap();
/// assert_eq!(output, Output::Hdr10);
/// assert_eq!(chosen, offered[1]);
/// ```
#[must_use]
pub fn output_format(
    offered: &[SurfaceFormat],
    wishes: &[Output],
) -> Option<(Output, SurfaceFormat)> {
    if let [only] = offered
        && only.format == format::UNDEFINED
    {
        return wishes
            .iter()
            .find(|w| w.color_space().raw() == only.color_space)
            .map(|&w| (w, SurfaceFormat::with(w.formats()[0], w.color_space())));
    }
    wishes.iter().find_map(|&w| {
        let space = w.color_space().raw();
        w.formats().iter().find_map(|&f| {
            offered
                .iter()
                .find(|o| o.format == f && o.color_space == space)
                .map(|&o| (w, o))
        })
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    const EXTENT: Extent = Extent::new(854, 480);

    #[test]
    fn suboptimal_recreates_only_when_the_surface_resized() {
        let grown = Extent::new(1280, 720);
        let again = |r, now| recreate(r, EXTENT, || now);
        assert_eq!(again(result::SUCCESS, Some(grown)), Ok(false));
        assert_eq!(again(result::SUBOPTIMAL, Some(EXTENT)), Ok(false));
        assert_eq!(again(result::SUBOPTIMAL, None), Ok(false));
        assert_eq!(
            again(result::SUBOPTIMAL, Some(Extent::default())),
            Ok(false)
        );
        assert_eq!(again(result::SUBOPTIMAL, Some(grown)), Ok(true));
        assert_eq!(again(result::OUT_OF_DATE, Some(EXTENT)), Ok(true));
        assert_eq!(again(result::TIMEOUT, None), Ok(false));
        assert_eq!(again(result::SURFACE_LOST, None), Err(result::SURFACE_LOST));
        let device_lost = -4;
        assert_eq!(again(device_lost, None), Err(device_lost));
        // The surface is asked only after a suboptimal present.
        let _ = recreate(result::SUCCESS, EXTENT, || unreachable!());
    }

    fn surface() -> SurfaceCapabilities {
        SurfaceCapabilities {
            min_image_count: 2,
            max_image_count: 3,
            current_extent: Extent::new(u32::MAX, u32::MAX),
            min_image_extent: Extent::new(1, 1),
            max_image_extent: Extent::new(4096, 4096),
            supported_composite_alpha: alpha::PRE_MULTIPLIED | alpha::INHERIT,
            supported_usage_flags: 0x10 | 0x1,
            ..SurfaceCapabilities::default()
        }
    }

    #[test]
    fn the_swapchain_sets_the_size_where_the_surface_has_none() {
        let wayland = surface();
        assert_eq!(wayland.current(), None);
        assert_eq!(wayland.extent(EXTENT), Some(EXTENT));
        assert_eq!(
            wayland.extent(Extent::new(9000, 0)),
            Some(Extent::new(4096, 1))
        );
        let windows = SurfaceCapabilities {
            current_extent: Extent::new(0, 0),
            min_image_extent: Extent::new(0, 0),
            max_image_extent: Extent::new(0, 0),
            ..surface()
        };
        assert_eq!(windows.extent(EXTENT), None, "minimized");
        let x11 = SurfaceCapabilities {
            current_extent: EXTENT,
            ..surface()
        };
        assert_eq!(x11.extent(Extent::new(1, 1)), Some(EXTENT));
    }

    #[test]
    fn image_count_alpha_and_usage_stay_within_the_surface() {
        let s = surface();
        assert_eq!(s.image_count(3), 3);
        assert_eq!(s.image_count(5), 3, "capped by max");
        assert_eq!(s.image_count(0), 3, "min + 1");
        let unlimited = SurfaceCapabilities {
            max_image_count: 0,
            ..s
        };
        assert_eq!(unlimited.image_count(5), 5);
        assert_eq!(s.composite_alpha(), alpha::INHERIT);
        assert_eq!(
            composite_alpha(alpha::OPAQUE | alpha::INHERIT),
            alpha::OPAQUE
        );
        assert_eq!(composite_alpha(0), alpha::OPAQUE);
        assert!(s.supports_usage(0x10) && !s.supports_usage(0x10 | 0x2));
    }

    #[test]
    fn the_raw_struct_reads_field_for_field() {
        let raw: [u32; 13] = [2, 8, 640, 480, 1, 1, 8192, 8192, 1, 1, 1, 1, 0x11];
        // SAFETY: 13 u32s, the layout of VkSurfaceCapabilitiesKHR.
        let s = unsafe { SurfaceCapabilities::from_raw(raw.as_ptr().cast()) };
        assert_eq!(s.current(), Some(Extent::new(640, 480)));
        assert_eq!(s.max_image_extent, Extent::new(8192, 8192));
        assert_eq!(s.supported_usage_flags, 0x11);
    }

    #[test]
    fn present_modes_fall_back_to_fifo() {
        use PresentMode as M;
        let all: Vec<u32> = M::ALL.iter().map(|m| m.raw()).collect();
        assert_eq!(present_mode(&all, Pacing::Vsync, None, true), M::Fifo);
        assert_eq!(
            present_mode(&all, Pacing::LowLatency, None, true),
            M::FifoLatestReady
        );
        assert_eq!(
            present_mode(&all, Pacing::LowLatency, None, false),
            M::Mailbox
        );
        assert_eq!(
            present_mode(&all, Pacing::Uncapped, None, true),
            M::Immediate
        );
        let fifo_only = [M::Fifo.raw()];
        assert_eq!(
            present_mode(&fifo_only, Pacing::Uncapped, None, true),
            M::Fifo
        );
        assert_eq!(present_mode(&[], Pacing::LowLatency, None, true), M::Fifo);
        // A request wins when offered, else the pacing decides.
        assert_eq!(
            present_mode(&all, Pacing::Vsync, Some(M::FifoRelaxed), false),
            M::FifoRelaxed
        );
        assert_eq!(
            present_mode(&fifo_only, Pacing::Uncapped, Some(M::Mailbox), false),
            M::Fifo
        );
        assert_eq!(
            present_mode(&all, Pacing::Vsync, Some(M::FifoLatestReady), false),
            M::Fifo,
            "offered but not enabled"
        );
        for m in M::ALL {
            assert_eq!(M::from_raw(m.raw()), Some(m));
            assert_eq!(m.name().parse(), Ok(m));
        }
        assert!("vsync".parse::<M>().is_err());
    }

    #[test]
    fn surface_formats_follow_the_encoding() {
        use format::*;
        let f = SurfaceFormat::new;
        let hdr = |format| SurfaceFormat::with(format, ColorSpace::ExtendedSrgbLinear);
        let usual = [f(B8G8R8A8_SRGB), f(B8G8R8A8_UNORM)];
        assert_eq!(
            surface_format(&usual, Encoding::Unorm),
            Some(f(B8G8R8A8_UNORM))
        );
        assert_eq!(
            surface_format(&usual, Encoding::Srgb),
            Some(f(B8G8R8A8_SRGB))
        );
        // RGBA when there is no BGRA; then any other format of the encoding.
        let rgba = [
            f(A2B10G10R10_UNORM_PACK32),
            f(R8G8B8A8_SRGB),
            f(R8G8B8A8_UNORM),
        ];
        assert_eq!(
            surface_format(&rgba, Encoding::Unorm),
            Some(f(R8G8B8A8_UNORM))
        );
        let ten_bit = [f(B8G8R8A8_SRGB), f(A2B10G10R10_UNORM_PACK32)];
        assert_eq!(
            surface_format(&ten_bit, Encoding::Unorm),
            Some(f(A2B10G10R10_UNORM_PACK32))
        );
        // Another colour space before an sRGB format that would encode twice.
        let odd = [f(B8G8R8A8_SRGB), hdr(R16G16B16A16_SFLOAT)];
        assert_eq!(
            surface_format(&odd, Encoding::Unorm),
            Some(hdr(R16G16B16A16_SFLOAT))
        );
        // Nothing of the encoding at all: the first.
        assert_eq!(
            surface_format(&[f(B8G8R8A8_SRGB)], Encoding::Unorm),
            Some(f(B8G8R8A8_SRGB))
        );
        assert_eq!(
            surface_format(&[f(UNDEFINED)], Encoding::Srgb),
            Some(f(B8G8R8A8_SRGB))
        );
        assert_eq!(surface_format(&[], Encoding::Srgb), None);
    }

    #[test]
    fn outputs_take_their_colour_space_and_formats() {
        use format::*;
        let w = SurfaceFormat::with;
        let sdr = [f(B8G8R8A8_SRGB), f(B8G8R8A8_UNORM)];
        let all = [
            sdr[0],
            sdr[1],
            w(A2R10G10B10_UNORM_PACK32, ColorSpace::Hdr10St2084),
            w(R16G16B16A16_SFLOAT, ColorSpace::Hdr10St2084),
            w(R16G16B16A16_SFLOAT, ColorSpace::ExtendedSrgbLinear),
            w(A2B10G10R10_UNORM_PACK32, ColorSpace::DisplayP3Nonlinear),
            w(B8G8R8A8_SRGB, ColorSpace::DisplayP3Nonlinear),
        ];
        // The 10-bit one before float, in its colour space only.
        assert_eq!(
            output_format(&all, &[Output::Hdr10]),
            Some((Output::Hdr10, all[2]))
        );
        assert_eq!(output_format(&all, &[Output::Hlg]), None);
        assert_eq!(
            output_format(&all, &[Output::Hlg, Output::ScRgb]),
            Some((Output::ScRgb, all[4]))
        );
        let p3 = Output::DisplayP3(Encoding::Unorm);
        assert_eq!(output_format(&all, &[p3]), Some((p3, all[5])));
        let p3 = Output::DisplayP3(Encoding::Srgb);
        assert_eq!(output_format(&all, &[p3]), Some((p3, all[6])));
        // Without the extension only SDR matches.
        let sdr_srgb = Output::Sdr(Encoding::Srgb);
        assert_eq!(
            output_format(&sdr, &[Output::Hdr10, sdr_srgb]),
            Some((sdr_srgb, sdr[0]))
        );
        assert_eq!(output_format(&sdr, &[Output::Hdr10]), None);
        // No sRGB-encoding format for PQ.
        let srgb_pq = [w(B8G8R8A8_SRGB, ColorSpace::Hdr10St2084)];
        assert_eq!(output_format(&srgb_pq, &[Output::Hdr10]), None);
        // Any format, in the colour space the lone entry names.
        let any_hdr = [w(UNDEFINED, ColorSpace::Hdr10St2084)];
        assert_eq!(
            output_format(&any_hdr, &[Output::ScRgb, Output::Hdr10]),
            Some((
                Output::Hdr10,
                w(A2B10G10R10_UNORM_PACK32, ColorSpace::Hdr10St2084)
            ))
        );
        assert_eq!(all[2].space(), Some(ColorSpace::Hdr10St2084));
        let unknown = SurfaceFormat {
            format: B8G8R8A8_UNORM,
            color_space: 7,
        };
        assert_eq!(unknown.space(), None);
    }

    fn f(format: u32) -> SurfaceFormat {
        SurfaceFormat::new(format)
    }
}
