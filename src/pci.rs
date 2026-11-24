//! Which GPU a pair of PCI ids names, on every OS and before any backend
//! opens the device: the ids are what Vulkan (`vendorID`, `deviceID`), DXGI
//! and sysfs all report. For a graphics host that picks vendor paths at run
//! time, as miraculix's feature sets do for CPUs.
//!
//! The architecture tables hold the ids known to this crate; an id outside
//! them is [`Architecture::Unknown`], never a guess.

use std::fmt;
use std::str::FromStr;

pub mod bar;

/// PCI vendor ids.
pub mod vendor_id {
    pub const AMD: u32 = 0x1002;
    pub const NVIDIA: u32 = 0x10DE;
    pub const INTEL: u32 = 0x8086;
    pub const ARM: u32 = 0x13B5;
    pub const QUALCOMM: u32 = 0x5143;
    pub const IMGTEC: u32 = 0x1010;
    pub const APPLE: u32 = 0x106B;
    /// Software rasterisers (WARP, the D3D12 mapping layer's own ids).
    pub const MICROSOFT: u32 = 0x1414;
    /// Mesa's software drivers (llvmpipe, lavapipe) report this.
    pub const MESA: u32 = 0x10005;
}

/// A GPU vendor.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Vendor {
    Amd,
    Nvidia,
    Intel,
    Arm,
    Qualcomm,
    ImgTec,
    Apple,
    Microsoft,
    Mesa,
    Other(u32),
}

impl Vendor {
    #[must_use]
    pub const fn from_pci(id: u32) -> Self {
        match id {
            vendor_id::AMD => Self::Amd,
            vendor_id::NVIDIA => Self::Nvidia,
            vendor_id::INTEL => Self::Intel,
            vendor_id::ARM => Self::Arm,
            vendor_id::QUALCOMM => Self::Qualcomm,
            vendor_id::IMGTEC => Self::ImgTec,
            vendor_id::APPLE => Self::Apple,
            vendor_id::MICROSOFT => Self::Microsoft,
            vendor_id::MESA => Self::Mesa,
            other => Self::Other(other),
        }
    }

    /// The PCI vendor id.
    #[must_use]
    pub const fn id(self) -> u32 {
        match self {
            Self::Amd => vendor_id::AMD,
            Self::Nvidia => vendor_id::NVIDIA,
            Self::Intel => vendor_id::INTEL,
            Self::Arm => vendor_id::ARM,
            Self::Qualcomm => vendor_id::QUALCOMM,
            Self::ImgTec => vendor_id::IMGTEC,
            Self::Apple => vendor_id::APPLE,
            Self::Microsoft => vendor_id::MICROSOFT,
            Self::Mesa => vendor_id::MESA,
            Self::Other(id) => id,
        }
    }

    /// The name drivers and tools show (`"NVIDIA"`, `"AMD"`, ...); `None` for
    /// an unknown id.
    #[must_use]
    pub const fn name(self) -> Option<&'static str> {
        Some(match self {
            Self::Amd => "AMD",
            Self::Nvidia => "NVIDIA",
            Self::Intel => "Intel",
            Self::Arm => "ARM",
            Self::Qualcomm => "Qualcomm",
            Self::ImgTec => "ImgTec",
            Self::Apple => "Apple",
            Self::Microsoft => "Microsoft",
            Self::Mesa => "Mesa",
            Self::Other(_) => return None,
        })
    }
}

/// Intel graphics architectures.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum IntelArch {
    /// Gen9 / Gen9.5: Skylake to Comet Lake.
    Gen9,
    /// Gen11: Ice Lake.
    Gen11,
    /// Xe-LP: Tiger Lake, Rocket Lake, Alder Lake, Raptor Lake, DG1.
    XeLp,
    /// Xe-LPG: Meteor Lake, Arrow Lake.
    XeLpg,
    /// Xe-HPG: Arc A-series (Alchemist, DG2).
    XeHpg,
    /// Xe2-LPG: Lunar Lake.
    Xe2Lpg,
    /// Xe2-HPG: Arc B-series (Battlemage).
    Xe2Hpg,
}

/// NVIDIA architectures.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum NvidiaArch {
    Maxwell,
    Pascal,
    Volta,
    Turing,
    Ampere,
    Ada,
    Hopper,
    Blackwell,
}

/// AMD architectures.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum AmdArch {
    /// GCN up to Vega (incl. the Vega APUs).
    Gcn,
    Rdna1,
    Rdna2,
    Rdna3,
    /// RDNA 3.5: Strix Point, Strix Halo.
    Rdna3_5,
    Rdna4,
    /// CDNA (Instinct): no display engine.
    Cdna,
}

/// A GPU's architecture, as far as the tables know it.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Architecture {
    Intel(IntelArch),
    Nvidia(NvidiaArch),
    Amd(AmdArch),
    Unknown,
}

impl Architecture {
    /// The architecture of PCI device `device` of vendor `vendor`.
    #[must_use]
    pub const fn from_pci(vendor: Vendor, device: u32) -> Self {
        match vendor {
            Vendor::Intel => match intel(device) {
                Some(arch) => Self::Intel(arch),
                None => Self::Unknown,
            },
            Vendor::Nvidia => match nvidia(device) {
                Some(arch) => Self::Nvidia(arch),
                None => Self::Unknown,
            },
            Vendor::Amd => match amd(device) {
                Some(arch) => Self::Amd(arch),
                None => Self::Unknown,
            },
            _ => Self::Unknown,
        }
    }

    /// A short name: `"Xe2-HPG"`, `"Ada"`, `"RDNA 3"`.
    #[must_use]
    pub const fn name(self) -> Option<&'static str> {
        Some(match self {
            Self::Intel(arch) => match arch {
                IntelArch::Gen9 => "Gen9",
                IntelArch::Gen11 => "Gen11",
                IntelArch::XeLp => "Xe-LP",
                IntelArch::XeLpg => "Xe-LPG",
                IntelArch::XeHpg => "Xe-HPG",
                IntelArch::Xe2Lpg => "Xe2-LPG",
                IntelArch::Xe2Hpg => "Xe2-HPG",
            },
            Self::Nvidia(arch) => match arch {
                NvidiaArch::Maxwell => "Maxwell",
                NvidiaArch::Pascal => "Pascal",
                NvidiaArch::Volta => "Volta",
                NvidiaArch::Turing => "Turing",
                NvidiaArch::Ampere => "Ampere",
                NvidiaArch::Ada => "Ada",
                NvidiaArch::Hopper => "Hopper",
                NvidiaArch::Blackwell => "Blackwell",
            },
            Self::Amd(arch) => match arch {
                AmdArch::Gcn => "GCN",
                AmdArch::Rdna1 => "RDNA 1",
                AmdArch::Rdna2 => "RDNA 2",
                AmdArch::Rdna3 => "RDNA 3",
                AmdArch::Rdna3_5 => "RDNA 3.5",
                AmdArch::Rdna4 => "RDNA 4",
                AmdArch::Cdna => "CDNA",
            },
            Self::Unknown => return None,
        })
    }
}

/// Intel device ids by generation.
const fn intel(device: u32) -> Option<IntelArch> {
    Some(match device {
        0x1900..=0x193F | 0x5900..=0x593F | 0x3E90..=0x3EFF | 0x9B00..=0x9BFF | 0x3184 | 0x3185 => {
            IntelArch::Gen9
        }
        0x8A50..=0x8A5F => IntelArch::Gen11,
        0x9A40..=0x9AFF | 0x4C80..=0x4C9F | 0x4680..=0x46FF | 0xA720..=0xA7FF | 0x4905..=0x4908 => {
            IntelArch::XeLp
        }
        0x7D00..=0x7DFF => IntelArch::XeLpg,
        0x5690..=0x56CF => IntelArch::XeHpg,
        0x6420 | 0x64A0 | 0x64B0 => IntelArch::Xe2Lpg,
        0xE200..=0xE21F => IntelArch::Xe2Hpg,
        _ => return None,
    })
}

/// NVIDIA device ids by generation (their chips number upward).
const fn nvidia(device: u32) -> Option<NvidiaArch> {
    Some(match device {
        0x1340..=0x13FF | 0x1400..=0x14FF | 0x1600..=0x17FF => NvidiaArch::Maxwell,
        0x1500..=0x15FF | 0x1B00..=0x1D7F => NvidiaArch::Pascal,
        0x1D80..=0x1DFF => NvidiaArch::Volta,
        0x1E00..=0x1FFF | 0x2180..=0x21FF => NvidiaArch::Turing,
        0x2080..=0x20FF | 0x2200..=0x22FF | 0x2400..=0x25FF => NvidiaArch::Ampere,
        0x2300..=0x23FF => NvidiaArch::Hopper,
        0x2600..=0x28FF => NvidiaArch::Ada,
        0x2900..=0x29FF | 0x2B00..=0x2FFF => NvidiaArch::Blackwell,
        _ => return None,
    })
}

/// AMD device ids by generation.
const fn amd(device: u32) -> Option<AmdArch> {
    Some(match device {
        // Polaris, Vega, Vega 20 and the Vega APUs (Raven, Picasso, Renoir, Cezanne).
        0x67C0..=0x67FF
        | 0x6980..=0x699F
        | 0x6860..=0x687F
        | 0x69A0..=0x69AF
        | 0x66A0..=0x66AF
        | 0x15D8
        | 0x15DD
        | 0x1636
        | 0x1638 => AmdArch::Gcn,
        0x7310..=0x731F | 0x7340..=0x734F | 0x7360..=0x736F => AmdArch::Rdna1,
        // Navi 21-24, and the APUs Van Gogh, Rembrandt, Raphael, Mendocino.
        0x73A0..=0x73FF | 0x7420..=0x743F | 0x163F | 0x1681 | 0x164E | 0x1506 => AmdArch::Rdna2,
        // Navi 31-33, Phoenix.
        0x7440..=0x749F | 0x15BF | 0x15C8 => AmdArch::Rdna3,
        0x150E | 0x1586 => AmdArch::Rdna3_5,
        0x7550..=0x759F => AmdArch::Rdna4,
        // MI100, MI200, MI300.
        0x7388..=0x738F | 0x7408..=0x740F | 0x74A0..=0x74BF => AmdArch::Cdna,
        _ => return None,
    })
}

/// A GPU known by its PCI ids.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Gpu {
    pub vendor: Vendor,
    pub device: u32,
    pub architecture: Architecture,
}

impl Gpu {
    #[must_use]
    pub const fn from_pci(vendor: u32, device: u32) -> Self {
        let vendor = Vendor::from_pci(vendor);
        Self {
            vendor,
            device,
            architecture: Architecture::from_pci(vendor, device),
        }
    }
}

/// The OS a driver version was read on: one vendor packs it differently per OS.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DriverOs {
    Windows,
    Other,
}

impl DriverOs {
    /// The OS this was built for.
    #[must_use]
    pub const fn host() -> Self {
        if cfg!(windows) {
            Self::Windows
        } else {
            Self::Other
        }
    }
}

/// A driver version as its vendor counts: up to four numbers, compared
/// left to right (compare versions of one vendor only).
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct DriverVersion {
    parts: [u32; 4],
    len: usize,
}

impl DriverVersion {
    /// From its numbers, most significant first; more than four are cut.
    #[must_use]
    pub const fn new(parts: &[u32]) -> Self {
        let mut out = [0; 4];
        let mut i = 0;
        while i < parts.len() && i < 4 {
            out[i] = parts[i];
            i += 1;
        }
        Self { parts: out, len: i }
    }

    #[must_use]
    pub fn parts(&self) -> &[u32] {
        &self.parts[..self.len]
    }
}

impl fmt::Display for DriverVersion {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        for (i, part) in self.parts().iter().enumerate() {
            if i > 0 {
                f.write_str(".")?;
            }
            write!(f, "{part}")?;
        }
        Ok(())
    }
}

/// `VkPhysicalDeviceProperties::driverVersion` as its vendor writes it out:
/// NVIDIA packs 10.8.8.6 bits, Intel's Windows driver 18.14 bits, everyone
/// else (Mesa among them) the Vulkan version layout.
#[must_use]
pub fn driver_version(vendor: Vendor, raw: u32, os: DriverOs) -> DriverVersion {
    match (vendor, os) {
        (Vendor::Nvidia, _) => {
            DriverVersion::new(&[raw >> 22, (raw >> 14) & 0xFF, (raw >> 6) & 0xFF, raw & 0x3F])
        }
        (Vendor::Intel, DriverOs::Windows) => DriverVersion::new(&[raw >> 14, raw & 0x3FFF]),
        _ => DriverVersion::new(&[raw >> 22, (raw >> 12) & 0x3FF, raw & 0xFFF]),
    }
}

/// A PCI function's address: `domain:bus:device.function`, as Linux names it
/// (`0000:03:00.0`) and `VK_EXT_pci_bus_info` reports it.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct PciAddress {
    pub domain: u32,
    pub bus: u8,
    pub device: u8,
    pub function: u8,
}

impl fmt::Display for PciAddress {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{:04x}:{:02x}:{:02x}.{:x}",
            self.domain, self.bus, self.device, self.function
        )
    }
}

impl FromStr for PciAddress {
    type Err = ();

    /// `0000:03:00.0`; the domain may be left out (`03:00.0`).
    fn from_str(s: &str) -> Result<Self, ()> {
        let (rest, function) = s.rsplit_once('.').ok_or(())?;
        let mut parts = rest.rsplit(':');
        let device = parts.next().ok_or(())?;
        let bus = parts.next().ok_or(())?;
        let domain = parts.next().unwrap_or("0");
        if parts.next().is_some() {
            return Err(());
        }
        let hex8 = |v: &str| u8::from_str_radix(v, 16).map_err(|_| ());
        Ok(Self {
            domain: u32::from_str_radix(domain, 16).map_err(|_| ())?,
            bus: hex8(bus)?,
            device: hex8(device)?,
            function: hex8(function)?,
        })
    }
}

/// `bytes` with a binary unit, for logs (`11.83 GiB`, `256 MiB`).
#[must_use]
pub fn bytes(bytes: u64) -> String {
    const UNITS: [&str; 5] = ["B", "KiB", "MiB", "GiB", "TiB"];
    let mut unit = 0;
    let mut whole = bytes;
    while whole >= 1024 && unit + 1 < UNITS.len() {
        whole /= 1024;
        unit += 1;
    }
    let exact = bytes.is_multiple_of(1 << (10 * unit));
    if exact {
        format!("{whole} {}", UNITS[unit])
    } else {
        #[allow(clippy::cast_precision_loss, reason = "two decimals for a log")]
        let v = bytes as f64 / (1u64 << (10 * unit)) as f64;
        format!("{v:.2} {}", UNITS[unit])
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn known_cards_are_named_by_their_ids() {
        let arch = |vendor, device| Gpu::from_pci(vendor, device).architecture;
        assert_eq!(arch(0x8086, 0xE20B), Architecture::Intel(IntelArch::Xe2Hpg)); // Arc B580
        assert_eq!(arch(0x8086, 0x56A0), Architecture::Intel(IntelArch::XeHpg)); // Arc A770
        assert_eq!(arch(0x8086, 0xA780), Architecture::Intel(IntelArch::XeLp)); // UHD 770
        assert_eq!(arch(0x10DE, 0x2684), Architecture::Nvidia(NvidiaArch::Ada)); // RTX 4090
        assert_eq!(
            arch(0x10DE, 0x2206),
            Architecture::Nvidia(NvidiaArch::Ampere)
        ); // RTX 3080
        assert_eq!(
            arch(0x10DE, 0x2B85),
            Architecture::Nvidia(NvidiaArch::Blackwell)
        ); // RTX 5090
        assert_eq!(arch(0x1002, 0x744C), Architecture::Amd(AmdArch::Rdna3)); // RX 7900 XTX
        assert_eq!(arch(0x1002, 0x73BF), Architecture::Amd(AmdArch::Rdna2)); // RX 6800 XT
        assert_eq!(arch(0x1002, 0x7550), Architecture::Amd(AmdArch::Rdna4)); // RX 9070 XT
    }

    #[test]
    fn unknown_ids_stay_unknown() {
        assert_eq!(
            Gpu::from_pci(0x8086, 0x0001).architecture,
            Architecture::Unknown
        );
        let other = Gpu::from_pci(0x1234, 0x5678);
        assert_eq!(other.vendor, Vendor::Other(0x1234));
        assert_eq!(other.vendor.name(), None);
        assert_eq!(Architecture::Unknown.name(), None);
    }

    #[test]
    fn driver_versions_are_unpacked_as_their_vendor_packs_them() {
        // NVIDIA 566.36: 566 << 22 | 36 << 14.
        let nvidia = (566 << 22) | (36 << 14);
        assert_eq!(
            driver_version(Vendor::Nvidia, nvidia, DriverOs::Windows).to_string(),
            "566.36.0.0"
        );
        // Intel on Windows, 101.6559: 101 << 14 | 6559.
        let intel = (101 << 14) + 6559;
        assert_eq!(
            driver_version(Vendor::Intel, intel, DriverOs::Windows).to_string(),
            "101.6559"
        );
        // Mesa on Linux, 24.3.1 in the Vulkan layout.
        let mesa = (24 << 22) | (3 << 12) | 1;
        assert_eq!(
            driver_version(Vendor::Intel, mesa, DriverOs::Other).to_string(),
            "24.3.1"
        );
        assert_eq!(
            driver_version(Vendor::Amd, mesa, DriverOs::Windows).to_string(),
            "24.3.1"
        );
    }

    #[test]
    fn driver_versions_compare_part_by_part() {
        let old = DriverVersion::new(&[101, 6559]);
        let new = DriverVersion::new(&[101, 9175]);
        assert!(old < new);
        assert_eq!(new.parts(), [101, 9175]);
        assert_eq!(DriverVersion::new(&[1, 2, 3, 4, 5]).to_string(), "1.2.3.4");
    }

    #[test]
    fn pci_addresses_parse_like_linux_names_them() {
        let a: PciAddress = "0000:03:00.0".parse().unwrap();
        assert_eq!(
            a,
            PciAddress {
                domain: 0,
                bus: 3,
                device: 0,
                function: 0
            }
        );
        assert_eq!(a.to_string(), "0000:03:00.0");
        assert_eq!(
            "0a:1f.7".parse::<PciAddress>().unwrap().to_string(),
            "0000:0a:1f.7"
        );
        assert!("0000:03:00".parse::<PciAddress>().is_err());
        assert!("1:2:3:4.0".parse::<PciAddress>().is_err());
    }
}
