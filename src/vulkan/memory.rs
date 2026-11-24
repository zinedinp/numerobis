//! Memory heaps and types, read through the probe: how much VRAM there is,
//! how much of it the CPU can map (all of it with resizable BAR or a large
//! BAR, 256 MiB without), and the budget when `VK_EXT_memory_budget` is
//! there. [`pci::bar`](crate::pci::bar) tells the same from the PCI side.

use std::ffi::c_void;
use std::fmt;

use super::features::s_type;

/// `VkMemoryPropertyFlagBits`.
pub mod flags {
    pub const DEVICE_LOCAL: u32 = 0x1;
    pub const HOST_VISIBLE: u32 = 0x2;
    pub const HOST_COHERENT: u32 = 0x4;
    pub const HOST_CACHED: u32 = 0x8;
    pub const LAZILY_ALLOCATED: u32 = 0x10;
    pub const PROTECTED: u32 = 0x20;
}

/// `VK_MEMORY_HEAP_DEVICE_LOCAL_BIT`.
const HEAP_DEVICE_LOCAL: u32 = 0x1;
const MAX_TYPES: usize = 32;
const MAX_HEAPS: usize = 16;

#[repr(C)]
#[derive(Clone, Copy)]
struct RawType {
    flags: u32,
    heap: u32,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct RawHeap {
    size: u64,
    flags: u32,
}

/// `VkPhysicalDeviceMemoryProperties`.
#[repr(C)]
#[derive(Clone, Copy)]
pub(crate) struct RawMemoryProperties {
    type_count: u32,
    types: [RawType; MAX_TYPES],
    heap_count: u32,
    heaps: [RawHeap; MAX_HEAPS],
}

/// `VkPhysicalDeviceMemoryProperties2`.
#[repr(C)]
struct RawMemoryProperties2 {
    s_type: i32,
    p_next: *mut c_void,
    properties: RawMemoryProperties,
}

/// `VkPhysicalDeviceMemoryBudgetPropertiesEXT`.
#[repr(C)]
struct RawBudget {
    s_type: i32,
    p_next: *mut c_void,
    budget: [u64; MAX_HEAPS],
    usage: [u64; MAX_HEAPS],
}

// The C layouts (64-bit): VkMemoryHeap is 16 bytes, the properties 520.
const _: () = assert!(size_of::<RawHeap>() == 16);
const _: () = assert!(size_of::<RawMemoryProperties>() == 520);
const _: () = assert!(std::mem::offset_of!(RawMemoryProperties, heaps) == 264);
const _: () = assert!(std::mem::offset_of!(RawMemoryProperties2, properties) == 16);
const _: () = assert!(size_of::<RawBudget>() == 16 + 2 * 8 * MAX_HEAPS);

/// The memory query a probe hands out, boxed so its chain stays put.
pub(crate) struct MemoryQuery {
    head: RawMemoryProperties2,
    budget: RawBudget,
}

impl MemoryQuery {
    pub(crate) fn new() -> Box<Self> {
        let zero = RawMemoryProperties {
            type_count: 0,
            types: [RawType { flags: 0, heap: 0 }; MAX_TYPES],
            heap_count: 0,
            heaps: [RawHeap { size: 0, flags: 0 }; MAX_HEAPS],
        };
        Box::new(Self {
            head: RawMemoryProperties2 {
                s_type: s_type::MEMORY_PROPERTIES_2,
                p_next: std::ptr::null_mut(),
                properties: zero,
            },
            budget: RawBudget {
                s_type: s_type::MEMORY_BUDGET_PROPERTIES,
                p_next: std::ptr::null_mut(),
                budget: [u64::MAX; MAX_HEAPS],
                usage: [u64::MAX; MAX_HEAPS],
            },
        })
    }

    /// `VkPhysicalDeviceMemoryProperties2*`, with the budget struct behind
    /// it when the device offers `VK_EXT_memory_budget`.
    pub(crate) fn properties2(&mut self, budget: bool) -> *mut c_void {
        self.head.p_next = if budget {
            std::ptr::from_mut(&mut self.budget).cast()
        } else {
            std::ptr::null_mut()
        };
        std::ptr::from_mut(&mut self.head).cast()
    }

    /// `VkPhysicalDeviceMemoryProperties*`, for the 1.0 entry point.
    pub(crate) fn properties(&mut self) -> *mut c_void {
        std::ptr::from_mut(&mut self.head.properties).cast()
    }

    /// What the driver wrote, if it wrote anything.
    pub(crate) fn read(&self) -> Option<Memory> {
        let budget = (self.budget.budget[0] != u64::MAX).then_some(&self.budget);
        Memory::from_raw(&self.head.properties, budget)
    }
}

/// One memory heap.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Heap {
    pub size: u64,
    /// `VK_MEMORY_HEAP_DEVICE_LOCAL_BIT`: VRAM on a discrete GPU.
    pub device_local: bool,
    /// What the process may allocate now (`VK_EXT_memory_budget`).
    pub budget: Option<u64>,
    /// What the process has allocated (`VK_EXT_memory_budget`).
    pub usage: Option<u64>,
}

/// One memory type: [`flags`] and the heap it allocates from.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MemoryType {
    pub flags: u32,
    pub heap: usize,
}

impl MemoryType {
    #[must_use]
    pub fn has(&self, bits: u32) -> bool {
        self.flags & bits == bits
    }
}

/// The device's memory as Vulkan reports it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Memory {
    pub heaps: Vec<Heap>,
    pub types: Vec<MemoryType>,
}

impl Memory {
    fn from_raw(raw: &RawMemoryProperties, budget: Option<&RawBudget>) -> Option<Self> {
        let heaps = (raw.heap_count as usize).min(MAX_HEAPS);
        let types = (raw.type_count as usize).min(MAX_TYPES);
        if heaps == 0 {
            return None;
        }
        Some(Self {
            heaps: (0..heaps)
                .map(|i| Heap {
                    size: raw.heaps[i].size,
                    device_local: raw.heaps[i].flags & HEAP_DEVICE_LOCAL != 0,
                    budget: budget.map(|b| b.budget[i]),
                    usage: budget.map(|b| b.usage[i]),
                })
                .collect(),
            types: (0..types)
                .map(|i| MemoryType {
                    flags: raw.types[i].flags,
                    heap: raw.types[i].heap as usize,
                })
                .collect(),
        })
    }

    /// From a `VkPhysicalDeviceMemoryProperties` the host read itself.
    ///
    /// # Safety
    ///
    /// `properties` points at a filled `VkPhysicalDeviceMemoryProperties`.
    #[must_use]
    pub unsafe fn from_properties(properties: *const c_void) -> Option<Self> {
        // SAFETY: the caller's promise; the struct is plain old data.
        let raw = unsafe { &*properties.cast::<RawMemoryProperties>() };
        Self::from_raw(raw, None)
    }

    /// The largest device-local heap: VRAM on a discrete GPU, the shared
    /// pool on an integrated one.
    #[must_use]
    pub fn vram(&self) -> u64 {
        self.heaps
            .iter()
            .filter(|h| h.device_local)
            .map(|h| h.size)
            .max()
            .unwrap_or(0)
    }

    /// The largest device-local heap the CPU can map: the BAR aperture
    /// Vulkan exposes. 256 MiB without resizable BAR on most discrete GPUs,
    /// all of [`vram`](Self::vram) with it.
    #[must_use]
    pub fn cpu_visible_vram(&self) -> u64 {
        self.types
            .iter()
            .filter(|t| t.has(flags::DEVICE_LOCAL | flags::HOST_VISIBLE))
            .filter_map(|t| self.heaps.get(t.heap))
            .filter(|h| h.device_local)
            .map(|h| h.size)
            .max()
            .unwrap_or(0)
    }

    /// Whether the CPU can map all of VRAM (resizable BAR, a large fixed
    /// BAR, or unified memory).
    #[must_use]
    pub fn full_cpu_access(&self) -> bool {
        let vram = self.vram();
        vram > 0 && self.cpu_visible_vram() >= vram
    }

    /// Whether every device-local type is also host visible: one memory for
    /// CPU and GPU (integrated GPUs, Apple).
    #[must_use]
    pub fn is_unified(&self) -> bool {
        let mut local = self
            .types
            .iter()
            .filter(|t| t.has(flags::DEVICE_LOCAL))
            .peekable();
        local.peek().is_some() && local.all(|t| t.has(flags::HOST_VISIBLE))
    }

    /// Host memory the GPU can reach: the largest heap that is not device
    /// local.
    #[must_use]
    pub fn host(&self) -> u64 {
        self.heaps
            .iter()
            .filter(|h| !h.device_local)
            .map(|h| h.size)
            .max()
            .unwrap_or(0)
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

impl fmt::Display for Memory {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let (vram, visible) = (self.vram(), self.cpu_visible_vram());
        if self.is_unified() {
            write!(f, "{} unified", bytes(vram))?;
        } else if self.full_cpu_access() {
            write!(f, "{} VRAM, all CPU-visible", bytes(vram))?;
        } else if visible > 0 {
            write!(f, "{} VRAM, {} CPU-visible", bytes(vram), bytes(visible))?;
        } else {
            write!(f, "{} VRAM, none CPU-visible", bytes(vram))?;
        }
        let host = self.host();
        if host > 0 {
            write!(f, ", {} host", bytes(host))?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn memory(heaps: &[(u64, bool)], types: &[(u32, usize)]) -> Memory {
        Memory {
            heaps: heaps
                .iter()
                .map(|&(size, device_local)| Heap {
                    size,
                    device_local,
                    budget: None,
                    usage: None,
                })
                .collect(),
            types: types
                .iter()
                .map(|&(flags, heap)| MemoryType { flags, heap })
                .collect(),
        }
    }

    const DL: u32 = flags::DEVICE_LOCAL;
    const HV: u32 = flags::HOST_VISIBLE | flags::HOST_COHERENT;

    #[test]
    fn resizable_bar_shows_as_a_host_visible_type_on_the_vram_heap() {
        // Intel Arc B580 on Windows (vulkaninfo): heap 0 11.83 GiB VRAM,
        // type 4 DEVICE_LOCAL|HOST_VISIBLE|HOST_COHERENT on it.
        let arc = memory(
            &[(12_706_643_968, true), (17_150_029_824, false)],
            &[
                (DL, 0),
                (DL, 0),
                (HV, 1),
                (HV | flags::HOST_CACHED, 1),
                (DL | HV, 0),
            ],
        );
        assert_eq!(arc.vram(), 12_706_643_968);
        assert_eq!(arc.cpu_visible_vram(), arc.vram());
        assert!(arc.full_cpu_access());
        assert!(!arc.is_unified());
        assert_eq!(
            arc.to_string(),
            "11.83 GiB VRAM, all CPU-visible, 15.97 GiB host"
        );
    }

    #[test]
    fn without_resizable_bar_a_256_mib_heap_is_mapped() {
        let small_bar = memory(
            &[(12 << 30, true), (32 << 30, false), (256 << 20, true)],
            &[(DL, 0), (HV, 1), (DL | HV, 2)],
        );
        assert_eq!(small_bar.cpu_visible_vram(), 256 << 20);
        assert!(!small_bar.full_cpu_access());
        assert_eq!(
            small_bar.to_string(),
            "12 GiB VRAM, 256 MiB CPU-visible, 32 GiB host"
        );
    }

    #[test]
    fn integrated_gpus_are_unified() {
        let igpu = memory(
            &[(16 << 30, true)],
            &[(DL | HV, 0), (DL | HV | flags::HOST_CACHED, 0)],
        );
        assert!(igpu.is_unified());
        assert!(igpu.full_cpu_access());
        assert_eq!(igpu.to_string(), "16 GiB unified");
    }

    #[test]
    fn the_query_reads_what_the_driver_wrote() {
        let mut q = MemoryQuery::new();
        assert!(q.read().is_none());
        // SAFETY: the query owns the struct; written as a driver would.
        unsafe {
            let p = q.properties2(true).cast::<RawMemoryProperties2>();
            assert_eq!((*p).s_type, s_type::MEMORY_PROPERTIES_2);
            let budget = (*p).p_next.cast::<RawBudget>();
            assert_eq!((*budget).s_type, s_type::MEMORY_BUDGET_PROPERTIES);
            (*p).properties.heap_count = 1;
            (*p).properties.heaps[0] = RawHeap {
                size: 8 << 30,
                flags: HEAP_DEVICE_LOCAL,
            };
            (*p).properties.type_count = 1;
            (*p).properties.types[0] = RawType { flags: DL, heap: 0 };
            (*budget).budget[0] = 7 << 30;
            (*budget).usage[0] = 1 << 20;
        }
        let m = q.read().expect("filled");
        assert_eq!(m.heaps[0].budget, Some(7 << 30));
        assert_eq!(m.heaps[0].usage, Some(1 << 20));
        assert_eq!(m.cpu_visible_vram(), 0);
        assert!(
            q.properties2(false)
                .cast::<RawMemoryProperties2>()
                .is_aligned()
        );
    }

    #[test]
    fn byte_counts_read_naturally() {
        assert_eq!(bytes(256 << 20), "256 MiB");
        assert_eq!(bytes(1536), "1.50 KiB");
        assert_eq!(bytes(0), "0 B");
    }
}
