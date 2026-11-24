//! Which device a plan is for: its identity from the properties query (tied
//! to [`pci`] for vendor, architecture and driver version),
//! driver [`Quirk`]s, and [`rank`] to pick among several.

use std::ffi::{CStr, c_void};
use std::fmt::Write as _;

use super::caps::Capability;
use super::features::s_type;
use super::subgroup::Subgroup;
use super::{ApiVersion, Plan};
use crate::pci::{self, DriverOs, DriverVersion, Gpu, PciAddress, Vendor};
use crate::platform::Os;

/// `VkPhysicalDeviceType`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum DeviceType {
    Other,
    IntegratedGpu,
    DiscreteGpu,
    VirtualGpu,
    Cpu,
}

impl DeviceType {
    #[must_use]
    pub const fn from_raw(raw: u32) -> Self {
        match raw {
            1 => Self::IntegratedGpu,
            2 => Self::DiscreteGpu,
            3 => Self::VirtualGpu,
            4 => Self::Cpu,
            _ => Self::Other,
        }
    }

    #[must_use]
    pub const fn name(self) -> &'static str {
        match self {
            Self::Other => "other",
            Self::IntegratedGpu => "integrated",
            Self::DiscreteGpu => "discrete",
            Self::VirtualGpu => "virtual",
            Self::Cpu => "CPU",
        }
    }

    /// Lower is preferred by [`rank`]: discrete, integrated, virtual, CPU.
    const fn preference(self) -> u8 {
        match self {
            Self::DiscreteGpu => 0,
            Self::IntegratedGpu => 1,
            Self::VirtualGpu => 2,
            Self::Cpu => 3,
            Self::Other => 4,
        }
    }
}

/// What a device says about itself.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DeviceInfo {
    /// `apiVersion`; `None` for a version this crate can't read (Vulkan SC).
    pub api: Option<ApiVersion>,
    /// `driverVersion` as the vendor packed it; see [`driver`](Self::driver).
    pub driver_version: u32,
    pub vendor_id: u32,
    pub device_id: u32,
    pub device_type: DeviceType,
    pub name: String,
    /// `VkPhysicalDeviceDriverProperties::driverName` (1.2 or
    /// `VK_KHR_driver_properties`).
    pub driver_name: Option<String>,
    /// `VkPhysicalDeviceDriverProperties::driverInfo`.
    pub driver_info: Option<String>,
    /// From `VK_EXT_pci_bus_info`; matches
    /// [`pci::bar::Gpu::address`](crate::pci::bar::Gpu::address).
    pub pci: Option<PciAddress>,
}

/// Offsets in `VkPhysicalDeviceProperties`.
const DEVICE_NAME: usize = 20;
const NAME_LEN: usize = 256;

impl DeviceInfo {
    /// From the start of a `VkPhysicalDeviceProperties`: `None` when its
    /// `apiVersion` is 0, i.e. nobody filled it.
    fn from_header(b: &[u8]) -> Option<Self> {
        let word = |at: usize| u32::from_ne_bytes(b[at..at + 4].try_into().expect("4 bytes"));
        let api = word(0);
        if api == 0 {
            return None;
        }
        Some(Self {
            api: ApiVersion::from_raw(api),
            driver_version: word(4),
            vendor_id: word(8),
            device_id: word(12),
            device_type: DeviceType::from_raw(word(16)),
            name: c_string(&b[DEVICE_NAME..DEVICE_NAME + NAME_LEN]),
            driver_name: None,
            driver_info: None,
            pci: None,
        })
    }

    /// From a `VkPhysicalDeviceProperties` the host read itself.
    ///
    /// # Safety
    ///
    /// `properties` points at a filled `VkPhysicalDeviceProperties`.
    #[must_use]
    pub unsafe fn from_properties(properties: *const c_void) -> Option<Self> {
        // SAFETY: the caller's promise; the header is plain old data.
        let header =
            unsafe { std::slice::from_raw_parts(properties.cast::<u8>(), DEVICE_NAME + NAME_LEN) };
        Self::from_header(header)
    }

    /// A GPU in the CPU package, sharing system RAM.
    #[must_use]
    pub fn is_integrated(&self) -> bool {
        self.device_type == DeviceType::IntegratedGpu
    }

    /// Vendor and architecture by the PCI ids.
    #[must_use]
    pub fn gpu(&self) -> Gpu {
        Gpu::from_pci(self.vendor_id, self.device_id)
    }

    /// The same card on the PCI side (BARs, link, resizable BAR), from the
    /// cached [`pci::bar::scan`]: by the PCI address when the device told
    /// it, else by the ids.
    #[must_use]
    pub fn bar(&self) -> Option<&'static pci::bar::Gpu> {
        pci::bar::find(self.vendor_id, self.device_id, self.pci)
    }

    /// The driver version as the vendor counts it on `os`.
    #[must_use]
    pub fn driver(&self, os: Os) -> DriverVersion {
        let os = if os == Os::Windows {
            DriverOs::Windows
        } else {
            DriverOs::Other
        };
        pci::driver_version(self.gpu().vendor, self.driver_version, os)
    }

    /// One line for logs: name, type, vendor and architecture, ids, PCI
    /// address, driver.
    #[must_use]
    pub fn describe(&self, os: Os) -> String {
        let gpu = self.gpu();
        let mut s = format!("{}, {}", self.name, self.device_type.name());
        match (gpu.vendor.name(), gpu.architecture.name()) {
            (Some(v), Some(a)) => _ = write!(s, ", {v} {a}"),
            (Some(v), None) => _ = write!(s, ", {v}"),
            _ => {}
        }
        _ = write!(s, " [{:04x}:{:04x}]", self.vendor_id, self.device_id);
        if let Some(at) = self.pci {
            _ = write!(s, " at {at}");
        }
        _ = write!(s, ", driver {}", self.driver(os));
        if let Some(name) = &self.driver_name {
            _ = write!(s, " ({name})");
        }
        s
    }
}

fn c_string(b: &[u8]) -> String {
    let end = b.iter().position(|&c| c == 0).unwrap_or(b.len());
    String::from_utf8_lossy(&b[..end]).into_owned()
}

/// A known driver problem: a capability that reports support but must not be
/// used. The host passes its list to [`Probe::with_quirks`](super::Probe::with_quirks);
/// numerobis ships none until one is confirmed on hardware.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Quirk {
    pub capability: Capability,
    /// Only this vendor's drivers (by PCI vendor id).
    pub vendor: Option<Vendor>,
    /// Only on this OS.
    pub os: Option<Os>,
    /// Only drivers older than this (in the vendor's numbering on that OS).
    pub driver_below: Option<DriverVersion>,
    /// Shown in the log.
    pub reason: &'static str,
}

impl Quirk {
    /// Whether it applies to `device` on `os`. A quirk that names a vendor or
    /// driver never applies to a device whose identity is unknown.
    #[must_use]
    pub fn applies(&self, device: Option<&DeviceInfo>, os: Os) -> bool {
        if self.os.is_some_and(|o| o != os) {
            return false;
        }
        if self.vendor.is_none() && self.driver_below.is_none() {
            return true;
        }
        let Some(device) = device else {
            return false;
        };
        self.vendor.is_none_or(|v| device.gpu().vendor == v)
            && self
                .driver_below
                .is_none_or(|below| device.driver(os) < below)
    }
}

/// The indices of `plans` that grant every capability of `required`, best
/// first: discrete before integrated, virtual and CPU devices, then more
/// granted capabilities, then more VRAM, then the newer API version.
#[must_use]
pub fn rank(plans: &[Plan], required: &[Capability]) -> Vec<usize> {
    let mut usable: Vec<usize> = (0..plans.len())
        .filter(|&i| plans[i].lacking(required).is_empty())
        .collect();
    let key = |p: &Plan| {
        let kind = p.device_info().map_or(5, |d| d.device_type.preference());
        let granted = p.outcomes().iter().filter(|o| o.granted()).count();
        let vram = p.memory().map_or(0, super::Memory::vram);
        (
            kind,
            std::cmp::Reverse(granted),
            std::cmp::Reverse(vram),
            std::cmp::Reverse(p.api),
        )
    };
    usable.sort_by_key(|&i| key(&plans[i]));
    usable
}

/// `VkPhysicalDeviceSubgroupProperties`.
#[repr(C)]
struct RawSubgroup {
    s_type: i32,
    p_next: *mut c_void,
    size: u32,
    stages: u32,
    operations: u32,
    quad_operations_in_all_stages: u32,
}

/// `VkPhysicalDeviceDriverProperties`.
#[repr(C)]
struct RawDriver {
    s_type: i32,
    p_next: *mut c_void,
    driver_id: u32,
    name: [u8; NAME_LEN],
    info: [u8; NAME_LEN],
    conformance: [u8; 4],
}

/// `VkPhysicalDevicePCIBusInfoPropertiesEXT`.
#[repr(C)]
struct RawPciBusInfo {
    s_type: i32,
    p_next: *mut c_void,
    domain: u32,
    bus: u32,
    device: u32,
    function: u32,
}

/// `VkPhysicalDeviceProperties2`, with room to spare for the 824 bytes of
/// `VkPhysicalDeviceProperties` (drivers write only their struct's size).
#[repr(C)]
struct RawProperties2 {
    s_type: i32,
    p_next: *mut c_void,
    properties: [u64; 128],
}

// The C layouts; on 64-bit: 16, 536, 32, 32.
const _: () = assert!(std::mem::offset_of!(RawProperties2, properties) == super::HEADER);
const _: () = assert!(size_of::<RawDriver>() == super::HEADER + 4 + 2 * NAME_LEN + 4);
const _: () = assert!(size_of::<RawPciBusInfo>() == super::HEADER + 16);
const _: () = assert!(size_of::<RawSubgroup>() == super::HEADER + 16);

/// Which structs the properties chain carries.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct PropertyChain {
    /// `VkPhysicalDeviceSubgroupProperties` (core 1.1).
    pub subgroup: bool,
    /// `VkPhysicalDeviceDriverProperties` (1.2 or `VK_KHR_driver_properties`).
    pub driver: bool,
    /// `VkPhysicalDevicePCIBusInfoPropertiesEXT` (`VK_EXT_pci_bus_info`).
    pub pci: bool,
}

/// The properties query a probe hands out, boxed so its chain stays put.
pub(crate) struct PropertiesQuery {
    head: RawProperties2,
    subgroup: RawSubgroup,
    driver: RawDriver,
    pci: RawPciBusInfo,
}

impl PropertiesQuery {
    pub(crate) fn new() -> Box<Self> {
        let null = std::ptr::null_mut();
        Box::new(Self {
            head: RawProperties2 {
                s_type: s_type::PROPERTIES_2,
                p_next: null,
                properties: [0; 128],
            },
            subgroup: RawSubgroup {
                s_type: s_type::SUBGROUP_PROPERTIES,
                p_next: null,
                size: 0,
                stages: 0,
                operations: 0,
                quad_operations_in_all_stages: 0,
            },
            driver: RawDriver {
                s_type: s_type::DRIVER_PROPERTIES,
                p_next: null,
                driver_id: 0,
                name: [0; NAME_LEN],
                info: [0; NAME_LEN],
                conformance: [0; 4],
            },
            // u32::MAX: not written (0000:00:00.0 is a real address).
            pci: RawPciBusInfo {
                s_type: s_type::PCI_BUS_INFO_PROPERTIES,
                p_next: null,
                domain: u32::MAX,
                bus: u32::MAX,
                device: u32::MAX,
                function: u32::MAX,
            },
        })
    }

    /// Links the structs `chain` asks for and returns the head.
    pub(crate) fn link(&mut self, chain: PropertyChain) -> *mut c_void {
        let mut next: *mut c_void = std::ptr::null_mut();
        if chain.pci {
            self.pci.p_next = next;
            next = std::ptr::from_mut(&mut self.pci).cast();
        }
        if chain.driver {
            self.driver.p_next = next;
            next = std::ptr::from_mut(&mut self.driver).cast();
        }
        if chain.subgroup {
            self.subgroup.p_next = next;
            next = std::ptr::from_mut(&mut self.subgroup).cast();
        }
        self.head.p_next = next;
        std::ptr::from_mut(&mut self.head).cast()
    }

    /// `subgroupSize` is at least 1 on every device: 0 means not written.
    pub(crate) fn subgroup(&self) -> Option<Subgroup> {
        let s = &self.subgroup;
        (s.size != 0).then_some(Subgroup {
            size: s.size,
            stages: s.stages,
            operations: s.operations,
        })
    }

    pub(crate) fn device(&self) -> Option<DeviceInfo> {
        let bytes: Vec<u8> = self
            .head
            .properties
            .iter()
            .flat_map(|w| w.to_ne_bytes())
            .collect();
        let mut info = DeviceInfo::from_header(&bytes)?;
        if self.driver.driver_id != 0 {
            info.driver_name = Some(c_string(&self.driver.name));
            info.driver_info = Some(c_string(&self.driver.info));
        }
        let p = &self.pci;
        if p.domain != u32::MAX {
            info.pci = Some(PciAddress {
                domain: p.domain,
                bus: u8::try_from(p.bus).ok()?,
                device: u8::try_from(p.device).ok()?,
                function: u8::try_from(p.function).ok()?,
            });
        }
        Some(info)
    }
}

/// `VK_KHR_driver_properties`, promoted to 1.2.
pub(crate) const DRIVER_PROPERTIES: &CStr = c"VK_KHR_driver_properties";
/// `VK_EXT_pci_bus_info`.
pub(crate) const PCI_BUS_INFO: &CStr = c"VK_EXT_pci_bus_info";

#[cfg(test)]
mod tests {
    use super::*;

    fn arc() -> DeviceInfo {
        DeviceInfo {
            api: Some(ApiVersion::V1_4),
            // 101.9175 in Intel's Windows packing.
            driver_version: (101 << 14) + 9175,
            vendor_id: 0x8086,
            device_id: 0xE20B,
            device_type: DeviceType::DiscreteGpu,
            name: "Intel(R) Arc(TM) B580 Graphics".into(),
            driver_name: Some("Intel Corporation".into()),
            driver_info: None,
            pci: Some("0000:03:00.0".parse().unwrap()),
        }
    }

    #[test]
    fn identity_goes_through_the_pci_tables() {
        let d = arc();
        assert_eq!(d.driver(Os::Windows).to_string(), "101.9175");
        assert_eq!(
            d.describe(Os::Windows),
            "Intel(R) Arc(TM) B580 Graphics, discrete, Intel Xe2-HPG [8086:e20b] at 0000:03:00.0, \
             driver 101.9175 (Intel Corporation)"
        );
    }

    #[test]
    fn quirks_match_vendor_os_and_driver() {
        let q = Quirk {
            capability: Capability::MeshShader,
            vendor: Some(Vendor::Intel),
            os: Some(Os::Windows),
            driver_below: Some(DriverVersion::new(&[101, 9000])),
            reason: "test",
        };
        let d = arc();
        assert!(!q.applies(Some(&d), Os::Windows), "9175 is newer");
        let old = DeviceInfo {
            driver_version: (101 << 14) + 6559,
            ..arc()
        };
        assert!(q.applies(Some(&old), Os::Windows));
        assert!(!q.applies(Some(&old), Os::Linux));
        assert!(!q.applies(None, Os::Windows), "unknown device");
        let everywhere = Quirk {
            vendor: None,
            os: None,
            driver_below: None,
            ..q
        };
        assert!(everywhere.applies(None, Os::Linux));
    }

    #[test]
    fn the_properties_chain_carries_what_it_was_asked_for() {
        let mut q = PropertiesQuery::new();
        assert!(q.device().is_none());
        let all = PropertyChain {
            subgroup: true,
            driver: true,
            pci: true,
        };
        // SAFETY: the query owns every struct; written as a driver would.
        unsafe {
            let head = q.link(all).cast::<RawProperties2>();
            let props = std::ptr::addr_of_mut!((*head).properties).cast::<u8>();
            for (at, v) in [
                (0, ApiVersion::V1_4.raw()),
                (4, 7),
                (8, 0x8086),
                (12, 0xE20B),
                (16, 2),
            ] {
                props.add(at).cast::<u32>().write_unaligned(v);
            }
            props.add(DEVICE_NAME).copy_from(c"Arc".as_ptr().cast(), 4);
            let sub = (*head).p_next.cast::<RawSubgroup>();
            assert_eq!((*sub).s_type, s_type::SUBGROUP_PROPERTIES);
            (*sub).size = 32;
            let drv = (*sub).p_next.cast::<RawDriver>();
            assert_eq!((*drv).s_type, s_type::DRIVER_PROPERTIES);
            (*drv).driver_id = 5;
            std::ptr::addr_of_mut!((*drv).name)
                .cast::<u8>()
                .copy_from(b"Intel".as_ptr(), 5);
            let pci = (*drv).p_next.cast::<RawPciBusInfo>();
            assert_eq!((*pci).s_type, s_type::PCI_BUS_INFO_PROPERTIES);
            assert!((*pci).p_next.is_null());
            ((*pci).domain, (*pci).bus, (*pci).device, (*pci).function) = (0, 3, 0, 0);
        }
        let d = q.device().expect("filled");
        assert_eq!(
            (d.name.as_str(), d.device_type),
            ("Arc", DeviceType::DiscreteGpu)
        );
        assert_eq!(d.driver_name.as_deref(), Some("Intel"));
        assert_eq!(
            d.pci.map(|a| a.to_string()).as_deref(),
            Some("0000:03:00.0")
        );
        assert_eq!(q.subgroup().map(|s| s.size), Some(32));
        // A 1.0 device gets the bare head.
        let none = PropertyChain {
            subgroup: false,
            driver: false,
            pci: false,
        };
        // SAFETY: as above.
        unsafe { assert!((*q.link(none).cast::<RawProperties2>()).p_next.is_null()) };
    }
}
