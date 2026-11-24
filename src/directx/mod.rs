//! Direct3D 8 to 12: which version a device speaks, its feature level, and
//! per capability where it comes from (feature `directx`).
//!
//! Like `vulkan` (feature `vulkan`) this
//! is data plus logic without bindings: the host makes the COM calls (with
//! windows-rs, winapi or by hand) and hands over the structs they fill. The
//! tables come from the Windows SDK headers through
//! `scripts/gen-directx.py` ([`generated`]).
//!
//! Two eras, kept apart because nothing carries over between them:
//!
//! - [`legacy`]: **Direct3D 8 and 9**. The fixed-function pipeline next to
//!   shader models 1 to 3, and one `D3DCAPS8` / `D3DCAPS9` of bits per
//!   device. Direct3D had no feature levels then, so [`LegacyLevel`] is
//!   numerobis's own, with the nearest `9_x` [`FeatureLevel`] where one fits.
//! - [`modern`]: **Direct3D 10, 11 and 12** on DXGI and WDDM. No fixed
//!   function, geometry shaders from 10, [`FeatureLevel`]s as Direct3D
//!   defines them (10.1 and 11 introduced them), optional features through
//!   `CheckFeatureSupport`. Raytracing, mesh shaders, variable rate
//!   shading, sampler feedback and work graphs are areas of their own.
//!
//! Older than 8 is not covered: DirectDraw (and Direct3D 7 with it, which
//! lives on DirectDraw surfaces) is a separate surface and palette API, too
//! much work for a version nobody targets any more. Direct3D 7-class
//! hardware still shows up through the D3D8/9 caps as
//! [`LegacyLevel::FixedFunction`]; use Direct3D 8 or 9 for it.
//! TODO(ddraw): a DirectDraw / Direct3D 7 route, if a user ever needs one.
//!
//! Not built yet:
//! - TODO(directstorage): DirectStorage, reads from NVMe straight into
//!   D3D12 resources with GPU decompression (GDeflate). It is a
//!   redistributable (`dstorage.dll`, `dstoragecore.dll`) with its own
//!   factory and queues, not part of the OS or of `CheckFeatureSupport`, so
//!   it needs its own probe.
//! - TODO(directsr): DirectSR, the D3D12 super resolution API
//!   (`ID3D12DSRDeviceFactory`, Agility SDK), which picks an upscaler (DLSS,
//!   FSR, XeSS or the built-in one) per device.
//!
//! Which version: pin one, or detect the newest that works ([`Select`]).
//! Which implementation answers ([`Runtime`]): Microsoft's, the Xbox GDK,
//! or one of the community layers on Vulkan (DXVK, vkd3d-proton), which is
//! how Direct3D runs on Linux (Wine, Proton).
//!
//! With feature `vulkan` too, `directx::vulkan` converts between the two:
//! queue priority, formats, and which Vulkan capabilities a translation
//! layer needs for a Direct3D one.
//!
//! ```
//! use numerobis::directx::{Api, FeatureLevel, Select};
//! use numerobis::directx::modern::{Capability as Cap, Probe};
//!
//! // Detect: the newest version first, down to Direct3D 11.
//! let select: Select = "auto-11".parse().unwrap();
//! assert_eq!(select.order(), [Api::D3D12, Api::D3D11]);
//!
//! // After D3D12CreateDevice: hand every CheckFeatureSupport to the probe.
//! let mut probe = Probe::new(Api::D3D12, FeatureLevel::L12_1, &[Cap::Raytracing, Cap::MeshShader, Cap::GeometryShader]);
//! probe.fill(|_feature, _data, _size| {
//!     // unsafe { device.CheckFeatureSupport(feature, data, size) }.is_ok()
//!     false
//! });
//! let plan = probe.plan();
//! assert!(!plan.has(Cap::Raytracing)); // nothing was reported
//! assert!(plan.has(Cap::GeometryShader)); // every `10_0` device has it
//! ```

use std::fmt;
use std::str::FromStr;

pub mod generated;
pub mod legacy;
mod level;
pub mod modern;
pub mod runtime;
#[cfg(feature = "vulkan")]
pub mod vulkan;

pub use crate::platform::{Os, Status, Support};
pub use level::{FeatureLevel, LegacyLevel, ShaderModel};
pub use runtime::Runtime;

/// A Direct3D version.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Api {
    D3D8,
    D3D9,
    D3D10,
    D3D11,
    D3D12,
}

/// The two architectures, which share nothing but the name.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Era {
    /// Direct3D 8 and 9: fixed function and shader models 1 to 3, caps bits.
    Legacy,
    /// Direct3D 10 to 12: DXGI, WDDM, feature levels, geometry shaders, no
    /// fixed function.
    Modern,
}

impl Api {
    /// Oldest first.
    pub const ALL: [Self; 5] = [
        Self::D3D8,
        Self::D3D9,
        Self::D3D10,
        Self::D3D11,
        Self::D3D12,
    ];
    /// The newest version the tables know.
    pub const NEWEST: Self = Self::D3D12;

    #[must_use]
    pub const fn name(self) -> &'static str {
        match self {
            Self::D3D8 => "Direct3D 8",
            Self::D3D9 => "Direct3D 9",
            Self::D3D10 => "Direct3D 10",
            Self::D3D11 => "Direct3D 11",
            Self::D3D12 => "Direct3D 12",
        }
    }

    #[must_use]
    pub const fn era(self) -> Era {
        match self {
            Self::D3D8 | Self::D3D9 => Era::Legacy,
            Self::D3D10 | Self::D3D11 | Self::D3D12 => Era::Modern,
        }
    }

    /// The DLL Windows ships it in. Direct3D 10 is created through
    /// `d3d10_1.dll` (`D3D10CreateDevice1`), which has feature levels.
    #[must_use]
    pub const fn dll(self) -> &'static str {
        match self {
            Self::D3D8 => "d3d8.dll",
            Self::D3D9 => "d3d9.dll",
            Self::D3D10 => "d3d10_1.dll",
            Self::D3D11 => "d3d11.dll",
            Self::D3D12 => "d3d12.dll",
        }
    }

    /// The function that creates it.
    #[must_use]
    pub const fn entry(self) -> &'static str {
        match self {
            Self::D3D8 => "Direct3DCreate8",
            Self::D3D9 => "Direct3DCreate9",
            Self::D3D10 => "D3D10CreateDevice1",
            Self::D3D11 => "D3D11CreateDevice",
            Self::D3D12 => "D3D12CreateDevice",
        }
    }

    /// The `SDKVersion` argument of [`entry`](Self::entry): 220 for
    /// `Direct3DCreate8`, `D3D_SDK_VERSION` (32) for `Direct3DCreate9`,
    /// `D3D10_1_SDK_VERSION` (0x20), `D3D11_SDK_VERSION` (7). Direct3D 12
    /// takes none.
    #[must_use]
    pub const fn sdk_version(self) -> Option<u32> {
        match self {
            Self::D3D8 => Some(220),
            Self::D3D9 | Self::D3D10 => Some(0x20),
            Self::D3D11 => Some(7),
            Self::D3D12 => None,
        }
    }

    /// Whether Microsoft ships a 64-bit runtime. `d3d8.dll` exists only in
    /// `SysWOW64`: a 64-bit process gets Direct3D 8 only through a wrapper
    /// on Direct3D 9 (DXVK's d3d8, or [`legacy::Caps9::as_d3d8`] for the
    /// caps).
    #[must_use]
    pub const fn has_64bit_runtime(self) -> bool {
        !matches!(self, Self::D3D8)
    }

    /// The feature levels a device of this version can be created at, newest
    /// first: what to pass to `D3D11CreateDevice` (`pFeatureLevels`) or try
    /// in turn with `D3D10CreateDevice1` / `D3D12CreateDevice`. Empty for
    /// the legacy versions, which have none.
    #[must_use]
    pub const fn levels(self) -> &'static [FeatureLevel] {
        use FeatureLevel as L;
        match self {
            Self::D3D8 | Self::D3D9 => &[],
            Self::D3D10 => &[L::L10_1, L::L10_0, L::L9_3, L::L9_2, L::L9_1],
            Self::D3D11 => &[
                L::L12_1,
                L::L12_0,
                L::L11_1,
                L::L11_0,
                L::L10_1,
                L::L10_0,
                L::L9_3,
                L::L9_2,
                L::L9_1,
            ],
            Self::D3D12 => &[L::L12_2, L::L12_1, L::L12_0, L::L11_1, L::L11_0],
        }
    }
}

impl fmt::Display for Api {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.name())
    }
}

/// A string that names no [`Api`] or [`Select`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ParseError(pub String);

impl fmt::Display for ParseError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{:?}: expected 8, 9, 10, 11, 12 (or d3d12, dx12), auto, or auto-<oldest>",
            self.0
        )
    }
}

impl std::error::Error for ParseError {}

impl FromStr for Api {
    type Err = ParseError;

    /// `12`, `d3d12`, `dx12`, `Direct3D 12`, any case.
    fn from_str(s: &str) -> Result<Self, ParseError> {
        let lower = s.trim().to_ascii_lowercase();
        let digits = lower
            .trim_start_matches("direct3d")
            .trim_start_matches("d3d")
            .trim_start_matches("dx")
            .trim();
        Ok(match digits {
            "8" => Self::D3D8,
            "9" => Self::D3D9,
            "10" => Self::D3D10,
            "11" => Self::D3D11,
            "12" => Self::D3D12,
            _ => return Err(ParseError(s.into())),
        })
    }
}

/// Which version to use: pinned, or the newest that the host manages to
/// create, from `newest` down to `oldest`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Select {
    Pin(Api),
    Detect { newest: Api, oldest: Api },
}

impl Select {
    /// Every version, Direct3D 12 first.
    pub const DETECT: Self = Self::Detect {
        newest: Api::D3D12,
        oldest: Api::D3D8,
    };

    /// The versions to try, in order.
    #[must_use]
    pub fn order(self) -> Vec<Api> {
        match self {
            Self::Pin(api) => vec![api],
            Self::Detect { newest, oldest } => Api::ALL
                .iter()
                .rev()
                .copied()
                .filter(|&a| a <= newest && a >= oldest)
                .collect(),
        }
    }

    /// The first version `create` succeeds with, and what it made. Errors of
    /// the versions tried before go to `create` only.
    pub fn first<T>(self, mut create: impl FnMut(Api) -> Option<T>) -> Option<(Api, T)> {
        self.order()
            .into_iter()
            .find_map(|api| create(api).map(|made| (api, made)))
    }
}

impl FromStr for Select {
    type Err = ParseError;

    /// A version pins it (`11`, `d3d9`); `auto` detects all, `auto-11` down
    /// to that one.
    fn from_str(s: &str) -> Result<Self, ParseError> {
        let lower = s.trim().to_ascii_lowercase();
        if lower == "auto" || lower == "detect" {
            return Ok(Self::DETECT);
        }
        if let Some(oldest) = lower
            .strip_prefix("auto-")
            .or_else(|| lower.strip_prefix("detect-"))
        {
            return Ok(Self::Detect {
                newest: Api::NEWEST,
                oldest: oldest.parse().map_err(|_| ParseError(s.into()))?,
            });
        }
        s.parse().map(Self::Pin)
    }
}

impl fmt::Display for Select {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Pin(api) => write!(f, "{api} (pinned)"),
            Self::Detect { newest, oldest } => write!(f, "newest of {newest} down to {oldest}"),
        }
    }
}

/// A COM interface id, laid out as Windows' `GUID`.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Guid {
    pub data1: u32,
    pub data2: u16,
    pub data3: u16,
    pub data4: [u8; 8],
}

impl Guid {
    /// From `"xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx"`, any case. `const`, so
    /// a table of ids is converted while compiling and a typo fails the
    /// build.
    ///
    /// # Panics
    ///
    /// On anything else (at compile time in a `const`).
    #[must_use]
    pub const fn parse(s: &str) -> Self {
        let b = s.as_bytes();
        assert!(
            b.len() == 36 && b[8] == b'-' && b[13] == b'-' && b[18] == b'-' && b[23] == b'-',
            "a GUID is 8-4-4-4-12 hex digits"
        );
        let mut data4 = [0u8; 8];
        let mut i = 0;
        while i < 8 {
            let at = if i < 2 { 19 + 2 * i } else { 24 + 2 * (i - 2) };
            #[allow(clippy::cast_possible_truncation, reason = "two hex digits")]
            {
                data4[i] = hex(b, at, 2) as u8;
            }
            i += 1;
        }
        #[allow(clippy::cast_possible_truncation, reason = "four hex digits")]
        Self {
            data1: hex(b, 0, 8),
            data2: hex(b, 9, 4) as u16,
            data3: hex(b, 14, 4) as u16,
            data4,
        }
    }
}

const fn hex(text: &[u8], at: usize, len: usize) -> u32 {
    let mut value = 0u32;
    let mut i = 0;
    while i < len {
        let digit = match text[at + i] {
            c @ b'0'..=b'9' => c - b'0',
            c @ b'a'..=b'f' => c - b'a' + 10,
            c @ b'A'..=b'F' => c - b'A' + 10,
            _ => panic!("not a hex digit"),
        };
        value = (value << 4) | digit as u32;
        i += 1;
    }
    value
}

impl fmt::Display for Guid {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let d = self.data4;
        write!(
            f,
            "{:08x}-{:04x}-{:04x}-{:02x}{:02x}-{:02x}{:02x}{:02x}{:02x}{:02x}{:02x}",
            self.data1, self.data2, self.data3, d[0], d[1], d[2], d[3], d[4], d[5], d[6], d[7]
        )
    }
}

/// `const` string equality, for the route tables.
pub(crate) const fn same(a: &str, b: &str) -> bool {
    let (a, b) = (a.as_bytes(), b.as_bytes());
    if a.len() != b.len() {
        return false;
    }
    let mut i = 0;
    while i < a.len() {
        if a[i] != b[i] {
            return false;
        }
        i += 1;
    }
    true
}

#[cfg(test)]
mod tests;
