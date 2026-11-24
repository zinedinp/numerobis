//! The DXGI side: which adapter a device is on (tied to [`pci`] like a
//! Vulkan device), its video memory budget, and [`rank`] to pick among
//! several.

use std::fmt::{self, Write as _};
use std::time::{Duration, Instant};

use super::{Capability, Plan};
use crate::pci::{self, Gpu, bytes};

/// `DXGI_ADAPTER_DESC1`, as `IDXGIAdapter1::GetDesc1` fills it.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct AdapterDesc1 {
    pub description: [u16; 128],
    pub vendor_id: u32,
    pub device_id: u32,
    pub sub_sys_id: u32,
    pub revision: u32,
    pub dedicated_video_memory: usize,
    pub dedicated_system_memory: usize,
    pub shared_system_memory: usize,
    /// `LUID`: `LowPart`, `HighPart`.
    pub adapter_luid: [u32; 2],
    pub flags: u32,
}

// The `SIZE_T`s make it 312 bytes on 64-bit, 296 on 32-bit.
#[cfg(target_pointer_width = "64")]
const _: () = assert!(size_of::<AdapterDesc1>() == 312);
#[cfg(target_pointer_width = "32")]
const _: () = assert!(size_of::<AdapterDesc1>() == 296);

impl Default for AdapterDesc1 {
    fn default() -> Self {
        Self {
            description: [0; 128],
            vendor_id: 0,
            device_id: 0,
            sub_sys_id: 0,
            revision: 0,
            dedicated_video_memory: 0,
            dedicated_system_memory: 0,
            shared_system_memory: 0,
            adapter_luid: [0; 2],
            flags: 0,
        }
    }
}

/// `DXGI_ADAPTER_FLAG_SOFTWARE`: WARP or another software rasterizer.
pub const ADAPTER_FLAG_SOFTWARE: u32 = 2;
/// Microsoft's PCI vendor id, which WARP and the Basic Render Driver use.
pub const MICROSOFT_VENDOR: u32 = 0x1414;

/// What DXGI says about an adapter.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Adapter {
    pub name: String,
    pub vendor_id: u32,
    pub device_id: u32,
    pub sub_sys_id: u32,
    pub revision: u32,
    /// VRAM.
    pub dedicated_video: u64,
    /// System memory set aside for the GPU at boot.
    pub dedicated_system: u64,
    /// System memory the GPU may share.
    pub shared_system: u64,
    /// The `LUID` as one number (`HighPart << 32 | LowPart`): what
    /// `VkPhysicalDeviceIDProperties::deviceLUID` holds for the same GPU.
    pub luid: u64,
    /// WARP or another software adapter.
    pub software: bool,
}

impl Adapter {
    #[must_use]
    pub fn from_desc1(desc: &AdapterDesc1) -> Self {
        let end = desc.description.iter().position(|&c| c == 0).unwrap_or(128);
        Self {
            name: String::from_utf16_lossy(&desc.description[..end]),
            vendor_id: desc.vendor_id,
            device_id: desc.device_id,
            sub_sys_id: desc.sub_sys_id,
            revision: desc.revision,
            dedicated_video: desc.dedicated_video_memory as u64,
            dedicated_system: desc.dedicated_system_memory as u64,
            shared_system: desc.shared_system_memory as u64,
            luid: (u64::from(desc.adapter_luid[1]) << 32) | u64::from(desc.adapter_luid[0]),
            software: desc.flags & ADAPTER_FLAG_SOFTWARE != 0 || desc.vendor_id == MICROSOFT_VENDOR,
        }
    }

    /// Vendor and architecture by the PCI ids, as for a Vulkan device.
    #[must_use]
    pub fn gpu(&self) -> Gpu {
        Gpu::from_pci(self.vendor_id, self.device_id)
    }

    /// The same card on the PCI side (BARs, resizable BAR), from the cached
    /// [`pci::bar::scan`], by the ids (DXGI tells no PCI address).
    #[must_use]
    pub fn bar(&self) -> Option<&'static pci::bar::Gpu> {
        pci::bar::find(self.vendor_id, self.device_id, None)
    }

    /// One line for logs.
    #[must_use]
    pub fn describe(&self) -> String {
        let gpu = self.gpu();
        let mut s = self.name.clone();
        if self.software {
            s.push_str(", software");
        }
        match (gpu.vendor.name(), gpu.architecture.name()) {
            (Some(v), Some(a)) => _ = write!(s, ", {v} {a}"),
            (Some(v), None) => _ = write!(s, ", {v}"),
            _ => {}
        }
        _ = write!(
            s,
            " [{:04x}:{:04x}], LUID {:#x}, {} VRAM, {} shared",
            self.vendor_id,
            self.device_id,
            self.luid,
            bytes(self.dedicated_video),
            bytes(self.shared_system)
        );
        s
    }
}

/// `DXGI_MEMORY_SEGMENT_GROUP`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum SegmentGroup {
    /// VRAM on a discrete GPU, everything on an integrated one.
    Local = 0,
    /// System memory a discrete GPU reaches over PCIe.
    NonLocal = 1,
}

/// `DXGI_QUERY_VIDEO_MEMORY_INFO`, as `IDXGIAdapter3::QueryVideoMemoryInfo`
/// fills it: the OS's budget for this process, which changes while it
/// runs, like Vulkan's memory budget.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct VideoMemory {
    pub budget: u64,
    pub current_usage: u64,
    pub available_for_reservation: u64,
    pub current_reservation: u64,
}

const _: () = assert!(size_of::<VideoMemory>() == 32);

impl VideoMemory {
    /// What this process may still allocate before the OS starts demoting.
    #[must_use]
    pub const fn available(&self) -> u64 {
        self.budget.saturating_sub(self.current_usage)
    }
}

impl fmt::Display for VideoMemory {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "budget {}, used {}, available {}",
            bytes(self.budget),
            bytes(self.current_usage),
            bytes(self.available())
        )
    }
}

/// The budget of one segment group, read again once older than
/// `max_age` (per frame, before big allocations).
#[derive(Debug, Clone)]
pub struct BudgetCache {
    max_age: Duration,
    last: Option<(Instant, VideoMemory)>,
}

impl BudgetCache {
    #[must_use]
    pub const fn new(max_age: Duration) -> Self {
        Self {
            max_age,
            last: None,
        }
    }

    /// The kept budget, or a fresh one from `fill` (which calls
    /// `QueryVideoMemoryInfo` and says whether it succeeded) when there is
    /// none or it is too old. A failed read is not kept.
    pub fn get(&mut self, fill: impl FnOnce(&mut VideoMemory) -> bool) -> Option<VideoMemory> {
        match self.last {
            Some((at, memory)) if at.elapsed() <= self.max_age => Some(memory),
            _ => self.refresh(fill),
        }
    }

    /// Reads it now.
    pub fn refresh(&mut self, fill: impl FnOnce(&mut VideoMemory) -> bool) -> Option<VideoMemory> {
        let mut memory = VideoMemory::default();
        if fill(&mut memory) {
            self.last = Some((Instant::now(), memory));
            Some(memory)
        } else {
            self.last = None;
            None
        }
    }

    #[must_use]
    pub fn cached(&self) -> Option<VideoMemory> {
        self.last.map(|(_, m)| m)
    }

    pub fn invalidate(&mut self) {
        self.last = None;
    }
}

/// The indices of `plans` that grant all of `required`, best first:
/// hardware before software, then the newer version and higher level, then
/// more granted capabilities, then more VRAM.
#[must_use]
pub fn rank(plans: &[Plan], required: &[Capability]) -> Vec<usize> {
    let mut usable: Vec<usize> = (0..plans.len())
        .filter(|&i| plans[i].lacking(required).is_empty())
        .collect();
    let key = |p: &Plan| {
        let software = p.adapter().is_none_or(|a| a.software);
        let granted = p.outcomes().iter().filter(|o| o.granted()).count();
        let vram = p.adapter().map_or(0, |a| a.dedicated_video);
        (
            software,
            std::cmp::Reverse(p.api),
            std::cmp::Reverse(p.level),
            std::cmp::Reverse(granted),
            std::cmp::Reverse(vram),
        )
    };
    usable.sort_by_key(|&i| key(&plans[i]));
    usable
}
