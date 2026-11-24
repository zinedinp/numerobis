//! XeSS 2: Intel's AI upscaler (super resolution), frame generation and
//! low latency, as data plus logic for a host that calls the library.
//!
//! The super resolution network runs on a matrix engine where there is one:
//! on Intel GPUs with XMX, the driver's own implementation ([`Path::Xmx`]);
//! elsewhere (other vendors, Intel without XMX) the SDK's smaller network on
//! DP4a dot products, which needs shader model 6.4 ([`Path::Dp4a`]). Which
//! one a GPU gets follows from [`crate::matrix`]: [`Path::of`].
//!
//! The Intel driver ships XeSS 2 itself, next to its user-mode driver in
//! the driver store ([`Component::driver_dll`]); games ship the SDK's
//! `libxess.dll` ([`Component::sdk_dll`]), which on Intel hands the work to
//! the driver's. The structs here are the SDK's (`xess.h`, MIT), laid out to
//! be passed to it. Windows only: XeSS has no Linux library.
//!
//! Checked on the Arc B580 (driver 101.9175, `examples/xess_probe.rs`): the
//! driver's XeSS-SR 2.0.2, XeSS-FG 1.3.3 and XeLL 1.3.2; a context and
//! `xessD3D12Init` (3.9 s the first time, under 60 ms from the pipeline
//! cache); 3DMark's `libxess.dll` 1.3.1 loading the driver's `igxess2.dll`;
//! every input resolution as [`input_resolution`] computes it, current and
//! legacy factors. `xessGetIntelXeFXVersion` answered 0.0.0 there through
//! both libraries, before and after init: it is no sign of the path.

use std::fmt;

use crate::matrix::Engine;
use crate::pci::{Gpu, Vendor};
use crate::platform::Support;

/// Where XeSS was run.
pub const SUPPORT: Support = Support::WINDOWS_ONLY_TESTED;

/// `xess_version_t` (and the identical `xefg_swapchain_version_t`,
/// `xell_version_t`).
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Default)]
pub struct Version {
    pub major: u16,
    pub minor: u16,
    pub patch: u16,
    pub reserved: u16,
}

impl Version {
    /// `0.0.0`: what `xessGetIntelXeFXVersion` returns off Intel (and on the
    /// B580 too).
    #[must_use]
    pub const fn is_zero(self) -> bool {
        self.major == 0 && self.minor == 0 && self.patch == 0
    }
}

impl fmt::Display for Version {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}.{}.{}", self.major, self.minor, self.patch)
    }
}

/// `xess_2d_t`: a resolution.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub struct Size {
    pub x: u32,
    pub y: u32,
}

impl Size {
    #[must_use]
    pub const fn new(x: u32, y: u32) -> Self {
        Self { x, y }
    }
}

impl fmt::Display for Size {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}x{}", self.x, self.y)
    }
}

/// `xess_properties_t`: what XeSS needs for an output resolution.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub struct Properties {
    pub required_descriptor_count: u32,
    pub temp_buffer_heap_size: u64,
    pub temp_texture_heap_size: u64,
}

const _: () = assert!(size_of::<Version>() == 8 && size_of::<Size>() == 8);
// 24 bytes where u64 is 8-aligned, 20 on 32-bit x86 System V (no padding
// after the count).
const _: () = assert!(match align_of::<u64>() {
    8 => size_of::<Properties>() == 24,
    4 => size_of::<Properties>() == 20,
    _ => false,
});

/// `xess_quality_settings_t`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Quality {
    UltraPerformance,
    Performance,
    Balanced,
    Quality,
    UltraQuality,
    UltraQualityPlus,
    /// Native resolution, anti-aliasing only.
    Aa,
}

impl Quality {
    pub const ALL: [Self; 7] = [
        Self::UltraPerformance,
        Self::Performance,
        Self::Balanced,
        Self::Quality,
        Self::UltraQuality,
        Self::UltraQualityPlus,
        Self::Aa,
    ];

    /// `XESS_QUALITY_SETTING_*`.
    #[must_use]
    pub const fn raw(self) -> i32 {
        match self {
            Self::UltraPerformance => 100,
            Self::Performance => 101,
            Self::Balanced => 102,
            Self::Quality => 103,
            Self::UltraQuality => 104,
            Self::UltraQualityPlus => 105,
            Self::Aa => 106,
        }
    }

    #[must_use]
    pub const fn from_raw(raw: i32) -> Option<Self> {
        Some(match raw {
            100 => Self::UltraPerformance,
            101 => Self::Performance,
            102 => Self::Balanced,
            103 => Self::Quality,
            104 => Self::UltraQuality,
            105 => Self::UltraQualityPlus,
            106 => Self::Aa,
            _ => return None,
        })
    }

    #[must_use]
    pub const fn name(self) -> &'static str {
        match self {
            Self::UltraPerformance => "ultra performance",
            Self::Performance => "performance",
            Self::Balanced => "balanced",
            Self::Quality => "quality",
            Self::UltraQuality => "ultra quality",
            Self::UltraQualityPlus => "ultra quality plus",
            Self::Aa => "native AA",
        }
    }

    /// Output over input per axis in tenths, XeSS 1.3 and later (30 is 3.0x).
    #[must_use]
    pub const fn scale_tenths(self) -> u32 {
        match self {
            Self::UltraPerformance => 30,
            Self::Performance => 23,
            Self::Balanced => 20,
            Self::Quality => 17,
            Self::UltraQuality => 15,
            Self::UltraQualityPlus => 13,
            Self::Aa => 10,
        }
    }

    /// The factors before XeSS 1.3 (`xessForceLegacyScaleFactors`). The
    /// header lists none for ultra quality plus; XeSS 2.0.2 gives it 1.3x,
    /// the same as legacy ultra quality.
    #[must_use]
    pub const fn legacy_scale_tenths(self) -> u32 {
        match self {
            Self::UltraPerformance => 30,
            Self::Performance => 20,
            Self::Balanced => 17,
            Self::Quality => 15,
            Self::UltraQuality | Self::UltraQualityPlus => 13,
            Self::Aa => 10,
        }
    }

    /// [`scale_tenths`](Self::scale_tenths) as a factor.
    #[must_use]
    pub fn scale(self) -> f64 {
        f64::from(self.scale_tenths()) / 10.0
    }
}

impl fmt::Display for Quality {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.name())
    }
}

/// Each axis divided by the factor, rounded up (as XeSS does: 1920 at 1.7x
/// is 1130, not 1129), in integers so that exact quotients stay exact.
const fn divide(output: Size, tenths: u32) -> Size {
    const fn axis(v: u32, tenths: u32) -> u32 {
        // At least 1.0x: the quotient is at most `v`.
        #[allow(clippy::cast_possible_truncation, reason = "at most v")]
        let q = (v as u64 * 10).div_ceil(tenths as u64) as u32;
        q
    }
    Size::new(axis(output.x, tenths), axis(output.y, tenths))
}

/// The input resolution XeSS asks for at `output` and `quality`
/// (`xessGetOptimalInputResolution`'s optimal one, and
/// `xessGetInputResolution`'s); the lower end of its range.
#[must_use]
pub const fn input_resolution(output: Size, quality: Quality) -> Size {
    divide(output, quality.scale_tenths())
}

/// [`input_resolution`] with the legacy factors.
#[must_use]
pub const fn legacy_input_resolution(output: Size, quality: Quality) -> Size {
    divide(output, quality.legacy_scale_tenths())
}

/// `xess_d3d12_init_params_t`, for `xessD3D12Init`: the heaps and the
/// pipeline library left null (XeSS allocates its own).
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct D3d12InitParams {
    pub output_resolution: Size,
    /// [`Quality::raw`].
    pub quality_setting: i32,
    /// `xess_init_flags_t` ([`init_flags`]).
    pub init_flags: u32,
    pub creation_node_mask: u32,
    pub visible_node_mask: u32,
    /// `ID3D12Heap*`.
    pub temp_buffer_heap: *mut std::ffi::c_void,
    pub buffer_heap_offset: u64,
    /// `ID3D12Heap*`.
    pub temp_texture_heap: *mut std::ffi::c_void,
    pub texture_heap_offset: u64,
    /// `ID3D12PipelineLibrary*`.
    pub pipeline_library: *mut std::ffi::c_void,
}

impl D3d12InitParams {
    #[must_use]
    pub const fn new(output: Size, quality: Quality) -> Self {
        Self {
            output_resolution: output,
            quality_setting: quality.raw(),
            init_flags: init_flags::NONE,
            creation_node_mask: 0,
            visible_node_mask: 0,
            temp_buffer_heap: std::ptr::null_mut(),
            buffer_heap_offset: 0,
            temp_texture_heap: std::ptr::null_mut(),
            texture_heap_offset: 0,
            pipeline_library: std::ptr::null_mut(),
        }
    }
}

const _: () = assert!(
    size_of::<D3d12InitParams>() == 24 + 3 * size_of::<*const u8>() + 16
        || size_of::<*const u8>() == 4
);

/// `xess_init_flags_t`.
pub mod init_flags {
    pub const NONE: u32 = 0;
    pub const HIGH_RES_MV: u32 = 1 << 0;
    pub const INVERTED_DEPTH: u32 = 1 << 1;
    pub const EXPOSURE_SCALE_TEXTURE: u32 = 1 << 2;
    pub const RESPONSIVE_PIXEL_MASK: u32 = 1 << 3;
    pub const USE_NDC_VELOCITY: u32 = 1 << 4;
    pub const EXTERNAL_DESCRIPTOR_HEAP: u32 = 1 << 5;
    pub const LDR_INPUT_COLOR: u32 = 1 << 6;
    pub const JITTERED_MV: u32 = 1 << 7;
    pub const ENABLE_AUTOEXPOSURE: u32 = 1 << 8;
}

/// Which network XeSS super resolution runs.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Path {
    /// On the XMX engines: the Intel driver's implementation.
    Xmx,
    /// On DP4a dot products (shader model 6.4): other vendors, and Intel
    /// GPUs without XMX.
    Dp4a,
}

impl Path {
    /// The path XeSS takes on `gpu`: XMX on an Intel GPU whose matrix
    /// engine is XMX, DP4a everywhere else.
    #[must_use]
    pub const fn of(gpu: Gpu) -> Self {
        match (gpu.vendor, Engine::from_gpu(gpu)) {
            (Vendor::Intel, Some(Engine::Xmx(_))) => Self::Xmx,
            _ => Self::Dp4a,
        }
    }
}

impl fmt::Display for Path {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::Xmx => "XMX (driver implementation)",
            Self::Dp4a => "DP4a",
        })
    }
}

/// The parts of XeSS 2.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Component {
    /// XeSS-SR, the upscaler (`xess*`).
    SuperResolution,
    /// XeSS-FG, frame generation on a proxy swap chain (`xefgSwapChain*`).
    FrameGeneration,
    /// XeLL, latency reduction (`xell*`); frame generation needs it.
    LowLatency,
}

impl Component {
    pub const ALL: [Self; 3] = [
        Self::SuperResolution,
        Self::FrameGeneration,
        Self::LowLatency,
    ];

    /// The library in the Intel driver's folder.
    #[must_use]
    pub const fn driver_dll(self) -> &'static str {
        match self {
            Self::SuperResolution => "igxess2.dll",
            Self::FrameGeneration => "igxess_fg.dll",
            Self::LowLatency => "igxell.dll",
        }
    }

    /// The library the SDK ships for applications.
    #[must_use]
    pub const fn sdk_dll(self) -> &'static str {
        match self {
            Self::SuperResolution => "libxess.dll",
            Self::FrameGeneration => "libxess_fg.dll",
            Self::LowLatency => "libxell.dll",
        }
    }

    /// Its `GetVersion` entry point, `fn(*mut Version) -> i32`.
    #[must_use]
    pub const fn version_entry(self) -> &'static std::ffi::CStr {
        match self {
            Self::SuperResolution => c"xessGetVersion",
            Self::FrameGeneration => c"xefgSwapChainGetVersion",
            Self::LowLatency => c"xellGetVersion",
        }
    }

    #[must_use]
    pub const fn name(self) -> &'static str {
        match self {
            Self::SuperResolution => "XeSS-SR",
            Self::FrameGeneration => "XeSS-FG",
            Self::LowLatency => "XeLL",
        }
    }
}

impl fmt::Display for Component {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.name())
    }
}

/// The super resolution entry points a host needs for D3D12, all `extern
/// "C"`.
pub const ENTRY_POINTS: [&str; 12] = [
    "xessGetVersion",
    "xessD3D12CreateContext",
    "xessD3D12Init",
    "xessD3D12Execute",
    "xessDestroyContext",
    "xessIsOptimalDriver",
    "xessGetIntelXeFXVersion",
    "xessGetOptimalInputResolution",
    "xessGetInputResolution",
    "xessGetProperties",
    "xessGetJitterScale",
    "xessForceLegacyScaleFactors",
];

/// `xess_result_t` values. XeFG and XeLL share 0 and the errors down to
/// -4; beyond that their codes are their own.
pub mod result {
    pub const WARNING_NONEXISTING_FOLDER: i32 = 1;
    pub const WARNING_OLD_DRIVER: i32 = 2;
    pub const SUCCESS: i32 = 0;
    pub const ERROR_UNSUPPORTED_DEVICE: i32 = -1;
    pub const ERROR_UNSUPPORTED_DRIVER: i32 = -2;
    pub const ERROR_UNINITIALIZED: i32 = -3;
    pub const ERROR_INVALID_ARGUMENT: i32 = -4;
    pub const ERROR_DEVICE_OUT_OF_MEMORY: i32 = -5;
    pub const ERROR_DEVICE: i32 = -6;
    pub const ERROR_NOT_IMPLEMENTED: i32 = -7;
    pub const ERROR_INVALID_CONTEXT: i32 = -8;
    pub const ERROR_OPERATION_IN_PROGRESS: i32 = -9;
    pub const ERROR_UNSUPPORTED: i32 = -10;
    pub const ERROR_CANT_LOAD_LIBRARY: i32 = -11;
    pub const ERROR_WRONG_CALL_ORDER: i32 = -12;
    pub const ERROR_UNKNOWN: i32 = -1000;

    /// A short name for logs (`"success"`, `"unsupported driver"`).
    #[must_use]
    pub const fn name(r: i32) -> &'static str {
        match r {
            WARNING_NONEXISTING_FOLDER => "warning: dump folder missing",
            WARNING_OLD_DRIVER => "warning: old driver",
            SUCCESS => "success",
            ERROR_UNSUPPORTED_DEVICE => "unsupported device",
            ERROR_UNSUPPORTED_DRIVER => "unsupported driver",
            ERROR_UNINITIALIZED => "uninitialized",
            ERROR_INVALID_ARGUMENT => "invalid argument",
            ERROR_DEVICE_OUT_OF_MEMORY => "out of device memory",
            ERROR_DEVICE => "device error",
            ERROR_NOT_IMPLEMENTED => "not implemented",
            ERROR_INVALID_CONTEXT => "invalid context",
            ERROR_OPERATION_IN_PROGRESS => "in progress",
            ERROR_UNSUPPORTED => "unsupported",
            ERROR_CANT_LOAD_LIBRARY => "cannot load library",
            ERROR_WRONG_CALL_ORDER => "wrong call order",
            _ => "unknown error",
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn presets_round_trip_and_scale_down() {
        for q in Quality::ALL {
            assert_eq!(Quality::from_raw(q.raw()), Some(q));
            assert!(q.scale() >= 1.0 && q.legacy_scale_tenths() >= 10);
        }
        // What XeSS 2.0.2 answers on the B580 (xess_probe).
        let measured = [
            (1920, 1080, Quality::Performance, 835, 470),
            (1920, 1080, Quality::Quality, 1130, 636),
            (1920, 1080, Quality::UltraQualityPlus, 1477, 831),
            (2560, 1440, Quality::UltraPerformance, 854, 480),
            (2560, 1440, Quality::Performance, 1114, 627),
            (2560, 1440, Quality::UltraQuality, 1707, 960),
            (3840, 2160, Quality::Performance, 1670, 940),
            (3840, 2160, Quality::Quality, 2259, 1271),
            (3840, 2160, Quality::UltraQualityPlus, 2954, 1662),
            (3840, 2160, Quality::Aa, 3840, 2160),
        ];
        for (x, y, q, ix, iy) in measured {
            assert_eq!(
                input_resolution(Size::new(x, y), q),
                Size::new(ix, iy),
                "{q}"
            );
        }
        // Exact quotients stay exact (1300 / 1.3 in f64 is above 1000).
        assert_eq!(
            input_resolution(Size::new(1300, 1300), Quality::UltraQualityPlus),
            Size::new(1000, 1000)
        );
        let legacy = Quality::ALL.map(|q| legacy_input_resolution(Size::new(3840, 2160), q));
        assert_eq!(
            legacy.map(|s| s.to_string()),
            [
                "1280x720",
                "1920x1080",
                "2259x1271",
                "2560x1440",
                "2954x1662",
                "2954x1662",
                "3840x2160"
            ]
        );
    }

    #[test]
    fn the_matrix_engine_picks_the_network() {
        assert_eq!(Path::of(Gpu::from_pci(0x8086, 0xE20B)), Path::Xmx); // B580
        assert_eq!(Path::of(Gpu::from_pci(0x8086, 0xA780)), Path::Dp4a); // UHD 770
        assert_eq!(Path::of(Gpu::from_pci(0x10DE, 0x2684)), Path::Dp4a); // RTX 4090
    }
}
