//! A GPU's PCI side: its BARs (how much VRAM the CPU can map, and whether
//! that window sits above 4 GiB), the resizable-BAR capability, and the PCIe
//! link. Read from the OS, without a driver API.
//!
//! - **Linux**: sysfs (`/sys/bus/pci/devices/*`): `resource`,
//!   `resource<N>_resize`, `current_link_*` / `max_link_*`, `amdgpu`'s
//!   `mem_info_vram_total`. Untested: the parsers are unit-tested, the
//!   reading never ran on a real system ([`SUPPORT`]).
//! - **Windows**: the configuration manager (`cfgmgr32`): the memory ranges
//!   the OS assigned, `DEVPKEY_PciDevice_*Link*`, and the VRAM size from the
//!   display driver's registry key. Run on an Intel Arc B580.
//!
//! Cards with a PCIe switch on board (Intel Arc, recent AMD) show a x1 link
//! on the GPU itself; [`Gpu::link`] is the switch's upstream link, the one in
//! the slot. Vulkan tells the CPU-visible VRAM from its side with
//! `vulkan::Memory` (and DirectX with
//! `directx::modern::VideoMemory`).

use std::fmt;
use std::io;
use std::sync::OnceLock;

use super::{PciAddress, bytes};
use crate::platform::{Status, Support};

/// Where this was run: Windows (Intel Arc B580); Linux untested.
pub const SUPPORT: Support = Support::new(Status::Untested, Status::Tested);

/// The aperture of a discrete GPU without resizable BAR.
pub const LEGACY_APERTURE: u64 = 256 << 20;
const FOUR_GIB: u64 = 1 << 32;

/// One memory BAR (Windows: one assigned memory range).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Bar {
    /// The BAR number (Linux); Windows lists ranges without it.
    pub index: Option<u8>,
    pub base: u64,
    pub size: u64,
    /// Linux only.
    pub prefetchable: Option<bool>,
    /// A 64-bit BAR (Linux; on Windows known when it lies above 4 GiB).
    pub is_64bit: Option<bool>,
    /// Sizes the resizable-BAR capability allows for it (Linux
    /// `resource<N>_resize`): empty when it is not resizable or the OS
    /// doesn't tell.
    pub resizable: Vec<u64>,
}

impl Bar {
    /// One past the last byte.
    #[must_use]
    pub fn end(&self) -> u64 {
        self.base.saturating_add(self.size)
    }

    /// Mapped above 4 GiB, which takes "Above 4G Decoding" in the firmware.
    #[must_use]
    pub fn above_4g(&self) -> bool {
        self.end() > FOUR_GIB
    }
}

/// A PCIe link.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Link {
    /// PCIe generation, 1 (2.5 GT/s) to 6 (64 GT/s).
    pub generation: u8,
    /// Lanes.
    pub width: u8,
}

impl Link {
    /// Transfers per second per lane, in GT/s.
    #[must_use]
    pub fn gts(self) -> f64 {
        match self.generation {
            1 => 2.5,
            2 => 5.0,
            3 => 8.0,
            4 => 16.0,
            5 => 32.0,
            _ => 64.0,
        }
    }

    /// Usable bytes per second in each direction, after the line code
    /// (8b/10b up to Gen 2, 128b/130b for Gen 3 to 5, 242/256 FLITs in Gen 6).
    #[must_use]
    pub fn bandwidth(self) -> f64 {
        let code = match self.generation {
            1 | 2 => 8.0 / 10.0,
            3..=5 => 128.0 / 130.0,
            _ => 242.0 / 256.0,
        };
        self.gts() * 1e9 * code / 8.0 * f64::from(self.width)
    }

    /// From the PCIe Link Status speed code (1 = 2.5 GT/s ...) and width, as
    /// Windows reports them. `None` for codes outside the spec.
    #[must_use]
    pub fn from_codes(speed: u32, width: u32) -> Option<Self> {
        let generation = u8::try_from(speed).ok().filter(|g| (1..=6).contains(g))?;
        let width = u8::try_from(width).ok().filter(|&w| w > 0)?;
        Some(Self { generation, width })
    }

    /// From sysfs: `current_link_speed` (`16.0 GT/s PCIe`, `8 GT/s`) and
    /// `current_link_width` (`16`).
    #[must_use]
    pub fn from_sysfs(speed: &str, width: &str) -> Option<Self> {
        let gts: f64 = speed.split_whitespace().next()?.parse().ok()?;
        let generation = [2.5, 5.0, 8.0, 16.0, 32.0, 64.0]
            .iter()
            .position(|&g| (g - gts).abs() < 0.01)?;
        let width = width.trim().parse::<u8>().ok().filter(|&w| w > 0)?;
        Some(Self {
            generation: u8::try_from(generation + 1).ok()?,
            width,
        })
    }
}

impl fmt::Display for Link {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "PCIe {}.0 x{} ({:.2} GB/s)",
            self.generation,
            self.width,
            self.bandwidth() / 1e9
        )
    }
}

/// The resizable-BAR state in one word.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ResizableBar {
    /// The CPU sees all of VRAM (resizable BAR on, or a BAR that large
    /// anyway).
    Active,
    /// The card can, but the aperture is smaller than VRAM: turn on "Above
    /// 4G Decoding" and "Resizable BAR" in the firmware. `max` is the
    /// largest size the card offers.
    Inactive { max: u64 },
    /// The card has no resizable-BAR capability.
    NotCapable,
    /// The aperture is small and the OS does not say whether the card could
    /// do more (Windows).
    Unknown,
    /// The OS doesn't show this process the ranges at all
    /// ([`Gpu::ranges_hidden`]).
    Hidden,
}

/// One display-class PCI device.
#[derive(Debug, Clone, PartialEq)]
pub struct Gpu {
    /// Vendor and architecture by the PCI ids.
    pub ids: super::Gpu,
    pub address: Option<PciAddress>,
    /// The device description (Windows).
    pub name: Option<String>,
    pub bars: Vec<Bar>,
    /// The link in the slot: the GPU's own, or its on-board switch's
    /// upstream link. Windows reports links on endpoints only, so behind a
    /// switch it stays `None` there.
    pub link: Option<Link>,
    /// The fastest that slot link could train to.
    pub max_link: Option<Link>,
    /// Whether the GPU sits behind a PCIe switch of its own vendor on the
    /// card (Intel Arc, recent AMD).
    pub behind_switch: bool,
    /// The link the GPU itself reports: behind a switch an internal one
    /// (often x1), which says nothing about the slot.
    pub own_link: Option<Link>,
    /// VRAM in bytes, where the OS tells (Windows: the display driver's
    /// registry key; Linux: `amdgpu` only). Set it from
    /// `vulkan::Memory::vram` otherwise.
    pub vram: Option<u64>,
    /// The OS keeps the assigned memory ranges from this process, so
    /// [`bars`](Self::bars) is empty without saying anything about the
    /// card: a 32-bit process on 64-bit Windows, where the configuration
    /// manager's resource calls return `CR_CALL_NOT_IMPLEMENTED` under
    /// WOW64. Run 64-bit to read them.
    pub ranges_hidden: bool,
}

impl Gpu {
    /// The BAR that maps VRAM: the largest memory range.
    #[must_use]
    pub fn aperture(&self) -> Option<&Bar> {
        self.bars.iter().max_by_key(|b| b.size)
    }

    /// Bytes of VRAM the CPU can map through the aperture.
    #[must_use]
    pub fn cpu_visible(&self) -> Option<u64> {
        self.aperture().map(|b| b.size)
    }

    /// Whether the aperture lies above 4 GiB ("Above 4G Decoding" is on).
    #[must_use]
    pub fn above_4g(&self) -> Option<bool> {
        self.aperture().map(Bar::above_4g)
    }

    /// Whether the card has the resizable-BAR capability: known on Linux;
    /// `None` on Windows, which doesn't tell.
    #[must_use]
    pub fn resizable_bar_capable(&self) -> Option<bool> {
        let bar = self.aperture()?;
        bar.index?;
        Some(self.bars.iter().any(|b| !b.resizable.is_empty()))
    }

    /// Whether the CPU sees all of VRAM. Without a known VRAM size, an
    /// aperture beyond the legacy 256 MiB counts.
    #[must_use]
    pub fn full_access(&self) -> Option<bool> {
        let visible = self.cpu_visible()?;
        Some(match self.vram {
            Some(vram) => visible >= vram,
            None => visible > LEGACY_APERTURE,
        })
    }

    /// The resizable-BAR state.
    #[must_use]
    pub fn resizable_bar(&self) -> ResizableBar {
        if self.ranges_hidden && self.bars.is_empty() {
            return ResizableBar::Hidden;
        }
        if self.full_access() == Some(true) {
            return ResizableBar::Active;
        }
        match self.resizable_bar_capable() {
            Some(true) => ResizableBar::Inactive {
                max: self
                    .bars
                    .iter()
                    .flat_map(|b| b.resizable.iter().copied())
                    .max()
                    .unwrap_or(0),
            },
            Some(false) => ResizableBar::NotCapable,
            None => ResizableBar::Unknown,
        }
    }
}

impl fmt::Display for Gpu {
    /// Multi-line: name and ids, link, VRAM and aperture, resizable BAR,
    /// the BARs.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let vendor = self.ids.vendor.name().unwrap_or("unknown vendor");
        write!(f, "{}", self.name.as_deref().unwrap_or(vendor))?;
        write!(f, " [{:04x}:{:04x}]", self.ids.vendor.id(), self.ids.device)?;
        if let Some(arch) = self.ids.architecture.name() {
            write!(f, " {vendor} {arch}")?;
        }
        if let Some(at) = self.address {
            write!(f, " at {at}")?;
        }
        writeln!(f)?;
        match (self.link, self.max_link) {
            (Some(l), Some(m)) if l != m => write!(f, "  link: {l}, can train to {m}")?,
            (Some(l), _) => write!(f, "  link: {l}")?,
            (None, _) if self.behind_switch => {
                write!(
                    f,
                    "  link: behind the on-board switch, not reported by the OS"
                )?;
                if let Some(own) = self.own_link {
                    write!(f, " (GPU-internal {own})")?;
                }
            }
            (None, _) => write!(f, "  link: not reported")?,
        }
        let switch = self.behind_switch && self.link.is_some();
        writeln!(f, "{}", if switch { " (on-board switch)" } else { "" })?;
        let vram = self.vram.map_or_else(|| "unknown".into(), bytes);
        match self.aperture() {
            Some(bar) => writeln!(
                f,
                "  VRAM {vram}, CPU-visible {} at {:#x} ({} 4 GiB)",
                bytes(bar.size),
                bar.base,
                if bar.above_4g() { "above" } else { "below" }
            )?,
            None if self.ranges_hidden => writeln!(
                f,
                "  VRAM {vram}, memory BARs hidden from this process (32-bit under WOW64)"
            )?,
            None => writeln!(f, "  VRAM {vram}, no memory BAR reported")?,
        }
        let rebar = match self.resizable_bar() {
            ResizableBar::Active => "active".to_string(),
            ResizableBar::Inactive { max } => format!("capable up to {}, not active", bytes(max)),
            ResizableBar::NotCapable => "not capable".into(),
            ResizableBar::Unknown => "unknown (small aperture; the OS doesn't say why)".into(),
            ResizableBar::Hidden => {
                "unknown (the OS hides the ranges from a 32-bit process)".into()
            }
        };
        write!(f, "  resizable BAR: {rebar}")
    }
}

static SCANNED: OnceLock<Vec<Gpu>> = OnceLock::new();

/// Every display-class PCI device (base class 0x03), by address. Read from
/// the OS on the first call and kept for the process (like miraculix's
/// `detect_features`): BARs and links only change with a reboot, a firmware
/// setting or a hot plug. A failed read isn't kept, the next call tries
/// again. [`scan_fresh`] always reads; [`warm_up`] fills it at startup.
///
/// # Errors
///
/// As [`scan_fresh`].
pub fn scan() -> io::Result<&'static [Gpu]> {
    if let Some(gpus) = SCANNED.get() {
        return Ok(gpus);
    }
    let gpus = scan_fresh()?;
    Ok(SCANNED.get_or_init(|| gpus))
}

/// Optional: [`scan`] at startup, so the first use later doesn't pay for
/// it. An error is left for that use to see.
pub fn warm_up() {
    let _ = scan();
}

/// The scanned device at `address`, or else the first one with these PCI
/// ids (two identical cards without an address are told apart by order
/// only). `vulkan::DeviceInfo::bar` asks with a
/// Vulkan device's ids.
#[must_use]
pub fn find(vendor: u32, device: u32, address: Option<PciAddress>) -> Option<&'static Gpu> {
    find_in(scan().ok()?, vendor, device, address)
}

fn find_in(gpus: &[Gpu], vendor: u32, device: u32, address: Option<PciAddress>) -> Option<&Gpu> {
    address
        .and_then(|a| gpus.iter().find(|g| g.address == Some(a)))
        .or_else(|| {
            gpus.iter()
                .find(|g| g.ids.vendor.id() == vendor && g.ids.device == device)
        })
}

/// What the OS reports now, without the cache.
///
/// # Errors
///
/// When the OS's device list can't be read (sysfs missing, configuration
/// manager failing); [`io::ErrorKind::Unsupported`] on other systems.
pub fn scan_fresh() -> io::Result<Vec<Gpu>> {
    #[cfg(target_os = "linux")]
    return linux::scan();
    #[cfg(windows)]
    return windows::scan();
    #[cfg(not(any(target_os = "linux", windows)))]
    Err(io::Error::new(
        io::ErrorKind::Unsupported,
        "pci::bar reads sysfs (Linux) or cfgmgr32 (Windows)",
    ))
}

/// Linux `resource`: one `start end flags` line per resource, BARs 0 to 5
/// first. Memory BARs only.
#[must_use]
pub fn parse_resource(text: &str) -> Vec<Bar> {
    const IORESOURCE_MEM: u64 = 0x200;
    const IORESOURCE_PREFETCH: u64 = 0x2000;
    const IORESOURCE_MEM_64: u64 = 0x0010_0000;
    let hex = |v: &str| u64::from_str_radix(v.trim_start_matches("0x"), 16).ok();
    text.lines()
        .take(6)
        .enumerate()
        .filter_map(|(i, line)| {
            let mut f = line.split_whitespace().map(hex);
            let (start, end, flags) = (f.next()??, f.next()??, f.next()??);
            (flags & IORESOURCE_MEM != 0 && end > start).then(|| Bar {
                index: u8::try_from(i).ok(),
                base: start,
                size: end - start + 1,
                prefetchable: Some(flags & IORESOURCE_PREFETCH != 0),
                is_64bit: Some(flags & IORESOURCE_MEM_64 != 0),
                resizable: Vec::new(),
            })
        })
        .collect()
}

/// Linux `resource<N>_resize`: a hex mask, bit `n` for a size of 1 MiB << n.
#[must_use]
pub fn parse_resize_mask(text: &str) -> Vec<u64> {
    let Ok(mask) = u64::from_str_radix(text.trim().trim_start_matches("0x"), 16) else {
        return Vec::new();
    };
    (0..44)
        .filter(|bit| mask & (1 << bit) != 0)
        .map(|bit| (1u64 << 20) << bit)
        .collect()
}

#[cfg(target_os = "linux")]
mod linux {
    use std::fs;
    use std::io;
    use std::path::{Path, PathBuf};

    use super::{Gpu, Link, parse_resize_mask, parse_resource};
    use crate::pci::{self, PciAddress};

    const ROOT: &str = "/sys/bus/pci/devices";

    fn read(dir: &Path, name: &str) -> Option<String> {
        fs::read_to_string(dir.join(name)).ok()
    }

    fn hex(dir: &Path, name: &str) -> Option<u32> {
        u32::from_str_radix(read(dir, name)?.trim().trim_start_matches("0x"), 16).ok()
    }

    fn link(dir: &Path, which: &str) -> Option<Link> {
        Link::from_sysfs(
            &read(dir, &format!("{which}_link_speed"))?,
            &read(dir, &format!("{which}_link_width"))?,
        )
    }

    /// The topmost bridge of the GPU's own vendor above it: its on-board
    /// switch's upstream port, whose link is the one in the slot.
    fn slot_device(dev: &Path, vendor: u32) -> PathBuf {
        let mut top = dev.to_path_buf();
        let mut at = dev.parent();
        while let Some(dir) = at {
            let is_bridge = hex(dir, "class").is_some_and(|c| c >> 8 == 0x0604);
            if !is_bridge || hex(dir, "vendor") != Some(vendor) {
                break;
            }
            top = dir.to_path_buf();
            at = dir.parent();
        }
        top
    }

    pub(super) fn scan() -> io::Result<Vec<Gpu>> {
        let mut out = Vec::new();
        for entry in fs::read_dir(ROOT)? {
            let dir = fs::canonicalize(entry?.path())?;
            if hex(&dir, "class").is_none_or(|c| c >> 16 != 0x03) {
                continue;
            }
            let (Some(vendor), Some(device)) = (hex(&dir, "vendor"), hex(&dir, "device")) else {
                continue;
            };
            let mut bars = read(&dir, "resource").map_or_else(Vec::new, |t| parse_resource(&t));
            for bar in &mut bars {
                if let Some(i) = bar.index {
                    bar.resizable = read(&dir, &format!("resource{i}_resize"))
                        .map_or_else(Vec::new, |t| parse_resize_mask(&t));
                }
            }
            let slot = slot_device(&dir, vendor);
            out.push(Gpu {
                ids: pci::Gpu::from_pci(vendor, device),
                address: dir
                    .file_name()
                    .and_then(|n| n.to_str())
                    .and_then(|n| n.parse::<PciAddress>().ok()),
                name: None,
                bars,
                link: link(&slot, "current"),
                max_link: link(&slot, "max"),
                behind_switch: slot != dir,
                own_link: link(&dir, "current"),
                vram: read(&dir, "mem_info_vram_total").and_then(|v| v.trim().parse().ok()),
                ranges_hidden: false,
            });
        }
        out.sort_by_key(|g| g.address);
        Ok(out)
    }
}

#[cfg(windows)]
mod windows {
    //! `cfgmgr32` and `advapi32`, declared by hand (no bindings crate).

    use std::ffi::c_void;
    use std::io;

    use super::{Bar, Gpu, Link};
    use crate::pci::{self, PciAddress};

    const CR_SUCCESS: u32 = 0;
    const CR_BUFFER_SMALL: u32 = 0x1A;
    /// What the resource calls return to a 32-bit process under WOW64.
    const CR_CALL_NOT_IMPLEMENTED: u32 = 0x34;
    const CM_GETIDLIST_FILTER_ENUMERATOR: u32 = 0x1;
    const CM_GETIDLIST_FILTER_PRESENT: u32 = 0x100;
    const ALLOC_LOG_CONF: u32 = 0x2;
    const RES_TYPE_ALL: u32 = 0;
    const RES_TYPE_MEM: u32 = 1;
    const RES_TYPE_MEM_LARGE: u32 = 7;
    const DEVPROP_TYPE_UINT32: u32 = 7;
    const DEVPROP_TYPE_STRING: u32 = 0x12;
    const RRF_RT_ANY: u32 = 0xFFFF;
    /// `HKEY_LOCAL_MACHINE`: `(HKEY)(ULONG_PTR)(LONG)0x80000002`, sign-extended.
    const HKEY_LOCAL_MACHINE: isize = 0x8000_0002_u32.cast_signed() as isize;

    #[repr(C)]
    struct DevPropKey {
        fmtid: [u8; 16],
        pid: u32,
    }

    /// A GUID's bytes as Windows lays it out (the first three fields little-endian).
    const fn guid(d1: u32, d2: u16, d3: u16, d4: [u8; 8]) -> [u8; 16] {
        let a = d1.to_le_bytes();
        let b = d2.to_le_bytes();
        let c = d3.to_le_bytes();
        [
            a[0], a[1], a[2], a[3], b[0], b[1], c[0], c[1], d4[0], d4[1], d4[2], d4[3], d4[4],
            d4[5], d4[6], d4[7],
        ]
    }

    /// `DEVPKEY_Device_*`.
    const DEVICE: [u8; 16] = guid(
        0xa45c_254e,
        0xdf1c,
        0x4efd,
        [0x80, 0x20, 0x67, 0xd1, 0x46, 0xa8, 0x50, 0xe0],
    );
    /// `DEVPKEY_PciDevice_*`.
    const PCI: [u8; 16] = guid(
        0x3ab2_2e31,
        0x8264,
        0x4b4e,
        [0x9a, 0xf5, 0xa8, 0xd2, 0xd8, 0xe3, 0x3e, 0x62],
    );
    const DEVICE_DESC: DevPropKey = DevPropKey {
        fmtid: DEVICE,
        pid: 2,
    };
    const DEVICE_DRIVER: DevPropKey = DevPropKey {
        fmtid: DEVICE,
        pid: 11,
    };
    const DEVICE_BUS_NUMBER: DevPropKey = DevPropKey {
        fmtid: DEVICE,
        pid: 23,
    };
    const DEVICE_ADDRESS: DevPropKey = DevPropKey {
        fmtid: DEVICE,
        pid: 30,
    };
    const PCI_BASE_CLASS: DevPropKey = DevPropKey { fmtid: PCI, pid: 3 };
    const PCI_CURRENT_LINK_SPEED: DevPropKey = DevPropKey { fmtid: PCI, pid: 9 };
    const PCI_CURRENT_LINK_WIDTH: DevPropKey = DevPropKey {
        fmtid: PCI,
        pid: 10,
    };
    const PCI_MAX_LINK_SPEED: DevPropKey = DevPropKey {
        fmtid: PCI,
        pid: 11,
    };
    const PCI_MAX_LINK_WIDTH: DevPropKey = DevPropKey {
        fmtid: PCI,
        pid: 12,
    };

    #[link(name = "cfgmgr32")]
    unsafe extern "system" {
        fn CM_Get_Device_ID_List_SizeW(len: *mut u32, filter: *const u16, flags: u32) -> u32;
        fn CM_Get_Device_ID_ListW(filter: *const u16, buf: *mut u16, len: u32, flags: u32) -> u32;
        fn CM_Locate_DevNodeW(inst: *mut u32, id: *const u16, flags: u32) -> u32;
        fn CM_Get_Parent(parent: *mut u32, inst: u32, flags: u32) -> u32;
        fn CM_Get_Device_ID_Size(len: *mut u32, inst: u32, flags: u32) -> u32;
        fn CM_Get_Device_IDW(inst: u32, buf: *mut u16, len: u32, flags: u32) -> u32;
        fn CM_Get_DevNode_PropertyW(
            inst: u32,
            key: *const DevPropKey,
            kind: *mut u32,
            buf: *mut u8,
            size: *mut u32,
            flags: u32,
        ) -> u32;
        fn CM_Get_First_Log_Conf(conf: *mut usize, inst: u32, flags: u32) -> u32;
        fn CM_Get_Next_Res_Des(
            next: *mut usize,
            current: usize,
            for_resource: u32,
            resource: *mut u32,
            flags: u32,
        ) -> u32;
        fn CM_Get_Res_Des_Data_Size(size: *mut u32, des: usize, flags: u32) -> u32;
        fn CM_Get_Res_Des_Data(des: usize, buf: *mut c_void, len: u32, flags: u32) -> u32;
        fn CM_Free_Res_Des_Handle(des: usize) -> u32;
        fn CM_Free_Log_Conf_Handle(conf: usize) -> u32;
    }

    #[link(name = "advapi32")]
    unsafe extern "system" {
        fn RegGetValueW(
            key: isize,
            subkey: *const u16,
            value: *const u16,
            flags: u32,
            kind: *mut u32,
            data: *mut c_void,
            len: *mut u32,
        ) -> i32;
    }

    fn wide(s: &str) -> Vec<u16> {
        s.encode_utf16().chain([0]).collect()
    }

    fn narrow(w: &[u16]) -> String {
        let end = w.iter().position(|&c| c == 0).unwrap_or(w.len());
        String::from_utf16_lossy(&w[..end])
    }

    fn property(inst: u32, key: &DevPropKey) -> Option<(u32, Vec<u8>)> {
        let (mut kind, mut size) = (0, 0);
        // SAFETY: a size query: no buffer, out pointers alive.
        let cr = unsafe {
            CM_Get_DevNode_PropertyW(
                inst,
                key,
                &raw mut kind,
                std::ptr::null_mut(),
                &raw mut size,
                0,
            )
        };
        if cr != CR_BUFFER_SMALL || size == 0 {
            return None;
        }
        let mut buf = vec![0u8; size as usize];
        // SAFETY: `buf` holds `size` bytes.
        let cr = unsafe {
            CM_Get_DevNode_PropertyW(inst, key, &raw mut kind, buf.as_mut_ptr(), &raw mut size, 0)
        };
        (cr == CR_SUCCESS).then(|| {
            buf.truncate(size as usize);
            (kind, buf)
        })
    }

    fn u32_property(inst: u32, key: &DevPropKey) -> Option<u32> {
        match property(inst, key)? {
            (DEVPROP_TYPE_UINT32, b) if b.len() == 4 => {
                Some(u32::from_le_bytes(b.try_into().ok()?))
            }
            // DEVPKEY_PciDevice_BaseClass is documented as a byte.
            (_, b) if b.len() == 1 => Some(u32::from(b[0])),
            _ => None,
        }
    }

    fn string_property(inst: u32, key: &DevPropKey) -> Option<String> {
        let (kind, b) = property(inst, key)?;
        if kind != DEVPROP_TYPE_STRING {
            return None;
        }
        let w: Vec<u16> = b
            .as_chunks::<2>()
            .0
            .iter()
            .map(|&c| u16::from_le_bytes(c))
            .collect();
        Some(narrow(&w))
    }

    fn device_id(inst: u32) -> Option<String> {
        let mut len = 0;
        // SAFETY: out pointer alive.
        if unsafe { CM_Get_Device_ID_Size(&raw mut len, inst, 0) } != CR_SUCCESS {
            return None;
        }
        let mut buf = vec![0u16; len as usize + 1];
        // SAFETY: `buf` holds `len + 1` characters.
        let cr = unsafe { CM_Get_Device_IDW(inst, buf.as_mut_ptr(), len + 1, 0) };
        (cr == CR_SUCCESS).then(|| narrow(&buf))
    }

    /// `VEN_xxxx` and `DEV_xxxx` of a `PCI\VEN_8086&DEV_E20B&...` instance id.
    pub(super) fn ids(instance: &str) -> Option<(u32, u32)> {
        let field = |tag: &str| {
            let at = instance.find(tag)? + tag.len();
            u32::from_str_radix(instance.get(at..at + 4)?, 16).ok()
        };
        Some((field("VEN_")?, field("DEV_")?))
    }

    fn link(inst: u32, speed: &DevPropKey, width: &DevPropKey) -> Option<Link> {
        Link::from_codes(u32_property(inst, speed)?, u32_property(inst, width)?)
    }

    /// The topmost PCI bridge of the GPU's own vendor above it (its on-board
    /// switch's upstream port).
    fn slot_device(inst: u32, vendor: u32) -> u32 {
        let mut top = inst;
        let mut at = inst;
        loop {
            let mut parent = 0;
            // SAFETY: out pointer alive.
            if unsafe { CM_Get_Parent(&raw mut parent, at, 0) } != CR_SUCCESS {
                return top;
            }
            let same_vendor = device_id(parent)
                .and_then(|id| ids(&id))
                .is_some_and(|(v, _)| v == vendor);
            let bridge = u32_property(parent, &PCI_BASE_CLASS) == Some(0x06);
            if !(same_vendor && bridge) {
                return top;
            }
            top = parent;
            at = parent;
        }
    }

    /// The memory ranges the OS assigned (`ALLOC_LOG_CONF`), and whether
    /// the OS hides them from this process (WOW64).
    fn memory_ranges(inst: u32) -> (Vec<Bar>, bool) {
        let mut bars = Vec::new();
        let mut conf = 0usize;
        // SAFETY: out pointer alive.
        match unsafe { CM_Get_First_Log_Conf(&raw mut conf, inst, ALLOC_LOG_CONF) } {
            CR_SUCCESS => {}
            CR_CALL_NOT_IMPLEMENTED => return (bars, true),
            _ => return (bars, false),
        }
        let mut current = conf;
        loop {
            let (mut next, mut kind) = (0usize, 0u32);
            // SAFETY: `current` is the log conf or a descriptor from the last call.
            let cr = unsafe {
                CM_Get_Next_Res_Des(&raw mut next, current, RES_TYPE_ALL, &raw mut kind, 0)
            };
            if current != conf {
                // SAFETY: a descriptor handle we own.
                unsafe { CM_Free_Res_Des_Handle(current) };
            }
            if cr != CR_SUCCESS {
                break;
            }
            current = next;
            if (kind == RES_TYPE_MEM || kind == RES_TYPE_MEM_LARGE)
                && let Some(bar) = memory_range(current)
            {
                bars.push(bar);
            }
        }
        // SAFETY: the log conf handle we own.
        unsafe { CM_Free_Log_Conf_Handle(conf) };
        (bars, false)
    }

    /// `MEM_DES` / `MEM_LARGE_DES`: count, type, `u64` base at 8, `u64`
    /// inclusive end at 16.
    fn memory_range(des: usize) -> Option<Bar> {
        let mut size = 0;
        // SAFETY: out pointer alive, `des` a live descriptor.
        if unsafe { CM_Get_Res_Des_Data_Size(&raw mut size, des, 0) } != CR_SUCCESS || size < 24 {
            return None;
        }
        let mut buf = vec![0u8; size as usize];
        // SAFETY: `buf` holds `size` bytes.
        if unsafe { CM_Get_Res_Des_Data(des, buf.as_mut_ptr().cast(), size, 0) } != CR_SUCCESS {
            return None;
        }
        let at = |o: usize| u64::from_le_bytes(buf[o..o + 8].try_into().expect("8 bytes"));
        let (base, end) = (at(8), at(16));
        (end > base).then(|| Bar {
            index: None,
            base,
            size: end - base + 1,
            prefetchable: None,
            is_64bit: (end >= 1 << 32).then_some(true),
            resizable: Vec::new(),
        })
    }

    /// `HardwareInformation.qwMemorySize` (or the 32-bit `MemorySize`) of
    /// the display driver's key.
    fn vram(inst: u32) -> Option<u64> {
        let driver = string_property(inst, &DEVICE_DRIVER)?;
        let key = wide(&format!(r"SYSTEM\CurrentControlSet\Control\Class\{driver}"));
        for name in [
            "HardwareInformation.qwMemorySize",
            "HardwareInformation.MemorySize",
        ] {
            let value = wide(name);
            let (mut kind, mut data, mut len) = (0u32, 0u64, 8u32);
            // SAFETY: `data` holds `len` bytes; strings are NUL-terminated.
            let status = unsafe {
                RegGetValueW(
                    HKEY_LOCAL_MACHINE,
                    key.as_ptr(),
                    value.as_ptr(),
                    RRF_RT_ANY,
                    &raw mut kind,
                    (&raw mut data).cast(),
                    &raw mut len,
                )
            };
            // Drivers write a REG_QWORD, older ones a 4-byte DWORD or binary.
            match (status, len) {
                (0, 8) => return Some(data),
                (0, 4) => return Some(data & 0xFFFF_FFFF),
                _ => {}
            }
        }
        None
    }

    pub(super) fn scan() -> io::Result<Vec<Gpu>> {
        let filter = wide("PCI");
        let flags = CM_GETIDLIST_FILTER_ENUMERATOR | CM_GETIDLIST_FILTER_PRESENT;
        let mut len = 0;
        // SAFETY: out pointer alive, filter NUL-terminated.
        let cr = unsafe { CM_Get_Device_ID_List_SizeW(&raw mut len, filter.as_ptr(), flags) };
        if cr != CR_SUCCESS {
            return Err(io::Error::other(format!(
                "CM_Get_Device_ID_List_SizeW: {cr:#x}"
            )));
        }
        let mut list = vec![0u16; len as usize];
        // SAFETY: `list` holds `len` characters.
        let cr = unsafe { CM_Get_Device_ID_ListW(filter.as_ptr(), list.as_mut_ptr(), len, flags) };
        if cr != CR_SUCCESS {
            return Err(io::Error::other(format!("CM_Get_Device_ID_ListW: {cr:#x}")));
        }
        let mut out = Vec::new();
        for id in list.split(|&c| c == 0).filter(|s| !s.is_empty()) {
            let id: Vec<u16> = id.iter().copied().chain([0]).collect();
            let mut inst = 0;
            // SAFETY: NUL-terminated id, out pointer alive.
            if unsafe { CM_Locate_DevNodeW(&raw mut inst, id.as_ptr(), 0) } != CR_SUCCESS {
                continue;
            }
            if u32_property(inst, &PCI_BASE_CLASS) != Some(0x03) {
                continue;
            }
            let Some((vendor, device)) = ids(&narrow(&id)) else {
                continue;
            };
            let address = match (
                u32_property(inst, &DEVICE_BUS_NUMBER),
                u32_property(inst, &DEVICE_ADDRESS),
            ) {
                (Some(bus), Some(addr)) => Some(PciAddress {
                    domain: 0,
                    bus: u8::try_from(bus).unwrap_or(u8::MAX),
                    device: u8::try_from(addr >> 16).unwrap_or(u8::MAX),
                    function: u8::try_from(addr & 0xFFFF).unwrap_or(u8::MAX),
                }),
                _ => None,
            };
            let slot = slot_device(inst, vendor);
            let (bars, ranges_hidden) = memory_ranges(inst);
            out.push(Gpu {
                ids: pci::Gpu::from_pci(vendor, device),
                address,
                name: string_property(inst, &DEVICE_DESC),
                bars,
                link: link(slot, &PCI_CURRENT_LINK_SPEED, &PCI_CURRENT_LINK_WIDTH),
                max_link: link(slot, &PCI_MAX_LINK_SPEED, &PCI_MAX_LINK_WIDTH),
                behind_switch: slot != inst,
                own_link: link(inst, &PCI_CURRENT_LINK_SPEED, &PCI_CURRENT_LINK_WIDTH),
                vram: vram(inst),
                ranges_hidden,
            });
        }
        out.sort_by_key(|g| g.address);
        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sysfs_resources_become_memory_bars() {
        // An NVIDIA card with resizable BAR: 16 MiB registers, 16 GiB VRAM
        // aperture above 4 GiB, 32 MiB, an I/O BAR, the ROM.
        let text = "\
0x00000000a1000000 0x00000000a1ffffff 0x0000000000040200
0x0000006000000000 0x00000063ffffffff 0x000000000014220c
0x0000000000000000 0x0000000000000000 0x0000000000000000
0x0000006400000000 0x0000006401ffffff 0x000000000014220c
0x0000000000000000 0x0000000000000000 0x0000000000000000
0x0000000000003000 0x000000000000307f 0x0000000000040101
0x00000000a2000000 0x00000000a207ffff 0x0000000000046200
";
        let bars = parse_resource(text);
        assert_eq!(bars.len(), 3);
        assert_eq!((bars[1].index, bars[1].size), (Some(1), 16 << 30));
        assert_eq!(bars[1].prefetchable, Some(true));
        assert_eq!(bars[1].is_64bit, Some(true));
        assert!(bars[1].above_4g() && !bars[0].above_4g());
    }

    #[test]
    fn resize_masks_list_sizes() {
        // Bits 8 to 14: 256 MiB to 16 GiB.
        let sizes = parse_resize_mask("0000000000007f00\n");
        assert_eq!(sizes.first(), Some(&(256 << 20)));
        assert_eq!(sizes.last(), Some(&(16 << 30)));
        assert_eq!(parse_resize_mask("garbage"), Vec::<u64>::new());
    }

    #[test]
    fn links_parse_and_count_bandwidth() {
        let gen4 = Link::from_sysfs("16.0 GT/s PCIe\n", "16\n").unwrap();
        assert_eq!((gen4.generation, gen4.width), (4, 16));
        assert_eq!(
            Link::from_sysfs("8 GT/s", "4"),
            Some(Link {
                generation: 3,
                width: 4
            })
        );
        assert_eq!(Link::from_sysfs("Unknown", "0"), None);
        assert_eq!(
            Link::from_codes(4, 8),
            Some(Link {
                generation: 4,
                width: 8
            })
        );
        assert_eq!(Link::from_codes(9, 8), None);
        assert_eq!(gen4.to_string(), "PCIe 4.0 x16 (31.51 GB/s)");
        let gen1 = Link::from_codes(1, 1).unwrap();
        assert!((gen1.bandwidth() - 250e6).abs() < 1.0);
    }

    fn gpu(bars: Vec<Bar>, vram: Option<u64>) -> Gpu {
        Gpu {
            ids: crate::pci::Gpu::from_pci(0x8086, 0xE20B),
            address: "0000:03:00.0".parse().ok(),
            name: Some("Intel(R) Arc(TM) B580 Graphics".into()),
            bars,
            link: Link::from_codes(4, 8),
            max_link: Link::from_codes(4, 8),
            behind_switch: true,
            own_link: Link::from_codes(1, 1),
            vram,
            ranges_hidden: false,
        }
    }

    fn bar(index: Option<u8>, base: u64, size: u64, resizable: Vec<u64>) -> Bar {
        Bar {
            index,
            base,
            size,
            prefetchable: index.map(|_| true),
            is_64bit: None,
            resizable,
        }
    }

    #[test]
    fn the_aperture_decides_the_resizable_bar_state() {
        let vram = 12 << 30;
        let regs = bar(None, 0xF600_0000, 16 << 20, vec![]);
        // Windows, resizable BAR on: a 16 GiB range high up.
        let on = gpu(
            vec![regs.clone(), bar(None, 0x40_0000_0000, 16 << 30, vec![])],
            Some(vram),
        );
        assert_eq!(on.resizable_bar(), ResizableBar::Active);
        assert_eq!(
            (on.cpu_visible(), on.above_4g()),
            (Some(16 << 30), Some(true))
        );
        assert_eq!(on.resizable_bar_capable(), None);
        assert_eq!(
            on.to_string(),
            "Intel(R) Arc(TM) B580 Graphics [8086:e20b] Intel Xe2-HPG at 0000:03:00.0\n  \
             link: PCIe 4.0 x8 (15.75 GB/s) (on-board switch)\n  \
             VRAM 12 GiB, CPU-visible 16 GiB at 0x4000000000 (above 4 GiB)\n  \
             resizable BAR: active"
        );
        // Windows, off: 256 MiB below 4 GiB, and no word on the capability.
        let off = gpu(
            vec![regs, bar(None, 0xE000_0000, 256 << 20, vec![])],
            Some(vram),
        );
        assert_eq!(off.resizable_bar(), ResizableBar::Unknown);
        let hidden = Gpu {
            ranges_hidden: true,
            ..gpu(Vec::new(), Some(12 << 30))
        };
        assert_eq!(hidden.resizable_bar(), ResizableBar::Hidden);
        assert!(hidden.to_string().contains("hidden from this process"));
        assert_eq!(off.above_4g(), Some(false));
        // Linux, capable but off.
        let sizes = parse_resize_mask("7f00");
        let linux = gpu(
            vec![bar(Some(2), 0xE000_0000, 256 << 20, sizes)],
            Some(vram),
        );
        assert_eq!(
            linux.resizable_bar(),
            ResizableBar::Inactive { max: 16 << 30 }
        );
        // Linux, not capable.
        let old = gpu(
            vec![bar(Some(1), 0xE000_0000, 256 << 20, vec![])],
            Some(vram),
        );
        assert_eq!(old.resizable_bar(), ResizableBar::NotCapable);
        // VRAM unknown: a large aperture still counts.
        let unknown = gpu(vec![bar(None, 0x40_0000_0000, 8 << 30, vec![])], None);
        assert_eq!(unknown.full_access(), Some(true));
    }

    #[test]
    fn behind_a_switch_without_a_slot_link_the_internal_one_is_named() {
        let windows = Gpu {
            link: None,
            max_link: None,
            ..gpu(vec![], Some(12 << 30))
        };
        let shown = windows.to_string();
        assert!(
            shown.contains(
                "link: behind the on-board switch, not reported by the OS \
                 (GPU-internal PCIe 1.0 x1 (0.25 GB/s))\n"
            ),
            "{shown}"
        );
    }

    #[test]
    fn find_prefers_the_address_then_the_ids() {
        let first = gpu(vec![], Some(12 << 30));
        let second = Gpu {
            address: "0000:04:00.0".parse().ok(),
            ..gpu(vec![], Some(10 << 30))
        };
        let gpus = [first, second];
        let at = |a: &str| find_in(&gpus, 0x8086, 0xE20B, a.parse().ok()).and_then(|g| g.vram);
        assert_eq!(at("0000:04:00.0"), Some(10 << 30));
        // No address, or one not listed: the first with the ids.
        assert_eq!(at(""), Some(12 << 30));
        assert_eq!(at("0000:09:00.0"), Some(12 << 30));
        assert!(find_in(&gpus, 0x10DE, 0x2206, None).is_none());
    }

    #[test]
    fn the_scan_is_read_once_and_kept() {
        // Where the OS can't be read (CI, other systems) there is nothing to
        // keep; the error isn't cached either.
        if let Ok(first) = scan() {
            warm_up();
            assert!(std::ptr::eq(first, scan().unwrap()));
        } else {
            assert!(scan().is_err());
        }
    }

    #[cfg(windows)]
    #[test]
    fn windows_instance_ids_carry_vendor_and_device() {
        assert_eq!(
            windows::ids(r"PCI\VEN_8086&DEV_E20B&SUBSYS_4287172F&REV_00\6&22113E25&0&00080008"),
            Some((0x8086, 0xE20B))
        );
        assert_eq!(windows::ids(r"ROOT\DISPLAY\0000"), None);
    }
}
