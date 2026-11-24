//! Memory heaps and types, read through the probe: how much VRAM there is,
//! how much of it the CPU can map (all of it with resizable BAR or a large
//! BAR, 256 MiB without), and the budget when `VK_EXT_memory_budget` is
//! there. [`pci::bar`](crate::pci::bar) tells the same from the PCI side.
//!
//! Heaps and types never change while the program runs, so the
//! [`Plan`](super::Plan)'s copy stays right. The budget does (other
//! processes, the shared pool of an integrated GPU, eviction): read it again
//! with a [`MemoryQuery`], or through a [`BudgetCache`] that reads only when
//! its copy is older than a set age.

use std::ffi::c_void;
use std::fmt::{self, Write as _};
use std::time::{Duration, Instant};

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

// The C layouts. No pointers, but `VkDeviceSize` is a u64, and its
// alignment sets the size of VkMemoryHeap: 16 bytes and properties of 520
// where u64 is 8-aligned (64-bit, 32-bit Windows and ARM), 12 and 456 on
// 32-bit x86 System V (i386 Linux), where it is 4-aligned. The chained
// structs grow with `HEADER`.
const _: () = assert!(match align_of::<u64>() {
    8 => size_of::<RawHeap>() == 16 && size_of::<RawMemoryProperties>() == 520,
    4 => size_of::<RawHeap>() == 12 && size_of::<RawMemoryProperties>() == 456,
    _ => false,
});
const _: () = assert!(std::mem::offset_of!(RawMemoryProperties, heaps) == 264);
const _: () = assert!(std::mem::offset_of!(RawMemoryProperties2, properties) == super::HEADER);
const _: () = assert!(size_of::<RawBudget>() == super::HEADER + 2 * 8 * MAX_HEAPS);

/// A `VkPhysicalDeviceMemoryProperties2` with the budget struct behind it,
/// to fill as often as needed (`vkGetPhysicalDeviceMemoryProperties2` is
/// cheap). Boxed so the chain stays put while the driver writes.
pub struct MemoryQuery {
    head: RawMemoryProperties2,
    budget: RawBudget,
}

// SAFETY: the only pointer, `head.p_next`, points into the same box or is
// null; it is set again by every `properties2` call.
unsafe impl Send for MemoryQuery {}
// SAFETY: `&self` methods only read the plain-old-data fields.
unsafe impl Sync for MemoryQuery {}

impl MemoryQuery {
    #[must_use]
    pub fn new() -> Box<Self> {
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
    /// it when the device offers `VK_EXT_memory_budget`. Valid while `self`
    /// is neither moved out of its box nor dropped; what an earlier fill
    /// wrote is cleared, so [`read`](Self::read) only sees the next one.
    pub fn properties2(&mut self, budget: bool) -> *mut c_void {
        self.clear();
        self.head.p_next = if budget {
            std::ptr::from_mut(&mut self.budget).cast()
        } else {
            std::ptr::null_mut()
        };
        std::ptr::from_mut(&mut self.head).cast()
    }

    /// `VkPhysicalDeviceMemoryProperties*`, for the 1.0 entry point (no
    /// budget).
    pub fn properties(&mut self) -> *mut c_void {
        self.clear();
        std::ptr::from_mut(&mut self.head.properties).cast()
    }

    fn clear(&mut self) {
        self.head.properties.heap_count = 0;
        self.head.properties.type_count = 0;
        self.budget.budget = [u64::MAX; MAX_HEAPS];
        self.budget.usage = [u64::MAX; MAX_HEAPS];
    }

    /// What the driver wrote, if it wrote anything.
    #[must_use]
    pub fn read(&self) -> Option<Memory> {
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
    /// CPU and GPU (Apple, Intel and llvmpipe). RADV keeps a device-local
    /// type without host visibility on AMD APUs too, so this alone misses
    /// them; [`summary`](Self::summary) takes the device type for that.
    #[must_use]
    pub fn is_unified(&self) -> bool {
        let mut local = self
            .types
            .iter()
            .filter(|t| t.has(flags::DEVICE_LOCAL))
            .peekable();
        local.peek().is_some() && local.all(|t| t.has(flags::HOST_VISIBLE))
    }

    /// What the process may still allocate in VRAM now: budget minus usage
    /// of the [`vram`](Self::vram) heap. `None` without
    /// `VK_EXT_memory_budget`. Only as fresh as the read; see
    /// [`BudgetCache`].
    #[must_use]
    pub fn available_vram(&self) -> Option<u64> {
        let heap = self
            .heaps
            .iter()
            .filter(|h| h.device_local)
            .max_by_key(|h| h.size)?;
        Some(heap.budget?.saturating_sub(heap.usage?))
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

/// The memory of one device, read again through the host's
/// `vkGetPhysicalDeviceMemoryProperties2` once the copy is older than
/// `max_age`. Per frame or before large allocations: the budget is the
/// share the OS gives this process now, and it shrinks when others
/// allocate (on an integrated GPU, everything else using the same RAM).
///
/// ```
/// use std::time::Duration;
/// use numerobis::vulkan::memory::BudgetCache;
///
/// let mut cache = BudgetCache::new(true, Duration::from_millis(500));
/// // The host's call; here nothing is written, so there is nothing to read.
/// let memory = cache.get(|_properties2| {
///     // get_physical_device_memory_properties2(physical, _properties2.cast())
/// });
/// assert!(memory.is_none());
/// ```
pub struct BudgetCache {
    query: Box<MemoryQuery>,
    budget: bool,
    max_age: Duration,
    last: Option<(Instant, Memory)>,
}

impl BudgetCache {
    /// `budget`: whether the device offers `VK_EXT_memory_budget`
    /// ([`for_plan`](Self::for_plan) takes it from a plan).
    #[must_use]
    pub fn new(budget: bool, max_age: Duration) -> Self {
        Self {
            query: MemoryQuery::new(),
            budget,
            max_age,
            last: None,
        }
    }

    /// For the device `plan` was made for: with the budget when its probe
    /// read one, starting from the plan's copy.
    #[must_use]
    pub fn for_plan(plan: &super::Plan, max_age: Duration) -> Self {
        let memory = plan.memory();
        let budget = memory.is_some_and(|m| m.heaps.iter().any(|h| h.budget.is_some()));
        Self {
            last: memory.map(|m| (Instant::now(), m.clone())),
            ..Self::new(budget, max_age)
        }
    }

    /// The copy, read again through `fill` when there is none or it is
    /// older than `max_age`. `fill` gets the
    /// `VkPhysicalDeviceMemoryProperties2*` to hand to
    /// `vkGetPhysicalDeviceMemoryProperties2`. When it writes nothing, the
    /// old copy is dropped and the result is `None`.
    pub fn get(&mut self, fill: impl FnOnce(*mut c_void)) -> Option<&Memory> {
        if self.age().is_none_or(|age| age > self.max_age) {
            return self.refresh(fill);
        }
        self.cached()
    }

    /// Reads now, whatever the age.
    pub fn refresh(&mut self, fill: impl FnOnce(*mut c_void)) -> Option<&Memory> {
        fill(self.query.properties2(self.budget));
        self.last = self.query.read().map(|m| (Instant::now(), m));
        self.cached()
    }

    /// The copy as it is, however old.
    #[must_use]
    pub fn cached(&self) -> Option<&Memory> {
        self.last.as_ref().map(|(_, m)| m)
    }

    /// How old the copy is.
    #[must_use]
    pub fn age(&self) -> Option<Duration> {
        self.last.as_ref().map(|(at, _)| at.elapsed())
    }

    /// The next [`get`](Self::get) reads again (after a large allocation or
    /// free of its own).
    pub fn invalidate(&mut self) {
        self.last = None;
    }
}

pub use crate::pci::bytes;

impl Memory {
    /// One line, as [`Display`](fmt::Display) but told whether the device is
    /// an integrated GPU (`DeviceInfo::is_integrated`): then its
    /// device-local heap is system RAM, whatever the memory types say.
    #[must_use]
    pub fn summary(&self, integrated: bool) -> String {
        let (vram, visible) = (self.vram(), self.cpu_visible_vram());
        let mut s = if self.is_unified() {
            format!("{} unified", bytes(vram))
        } else if integrated && self.full_cpu_access() {
            format!("{} shared with system RAM, all CPU-visible", bytes(vram))
        } else if integrated {
            format!(
                "{} shared with system RAM, {} CPU-visible",
                bytes(vram),
                bytes(visible)
            )
        } else if self.full_cpu_access() {
            format!("{} VRAM, all CPU-visible", bytes(vram))
        } else if visible > 0 {
            format!("{} VRAM, {} CPU-visible", bytes(vram), bytes(visible))
        } else {
            format!("{} VRAM, none CPU-visible", bytes(vram))
        };
        let host = self.host();
        if host > 0 {
            _ = write!(s, ", {} host", bytes(host));
        }
        s
    }
}

impl fmt::Display for Memory {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.summary(false))
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
    fn radv_on_an_apu_is_shared_memory() {
        // The Radeon 680M under RADV: a device-local heap the CPU maps
        // entirely, but one device-local-only type, so not "unified".
        let apu = memory(
            &[(2_932_359_168, false), (5_864_722_432, true)],
            &[(DL, 1), (HV, 0), (DL | HV, 1), (HV | flags::HOST_CACHED, 0)],
        );
        assert!(!apu.is_unified());
        assert_eq!(
            apu.to_string(),
            "5.46 GiB VRAM, all CPU-visible, 2.73 GiB host"
        );
        assert_eq!(
            apu.summary(true),
            "5.46 GiB shared with system RAM, all CPU-visible, 2.73 GiB host"
        );
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

    /// Writes one VRAM heap with `budget` and `usage`, as a driver would.
    fn fill(p: *mut c_void, budget: u64, usage: u64) {
        // SAFETY: `p` is a query's properties2 with the budget chained.
        unsafe {
            let p = p.cast::<RawMemoryProperties2>();
            (*p).properties.heap_count = 1;
            (*p).properties.heaps[0] = RawHeap {
                size: 8 << 30,
                flags: HEAP_DEVICE_LOCAL,
            };
            (*p).properties.type_count = 1;
            (*p).properties.types[0] = RawType { flags: DL, heap: 0 };
            let b = (*p).p_next.cast::<RawBudget>();
            (*b).budget[0] = budget;
            (*b).usage[0] = usage;
        }
    }

    #[test]
    fn a_query_can_be_filled_again_and_forgets_the_old_fill() {
        let mut q = MemoryQuery::new();
        fill(q.properties2(true), 7 << 30, 1 << 30);
        assert_eq!(q.read().unwrap().available_vram(), Some(6 << 30));
        fill(q.properties2(true), 7 << 30, 3 << 30);
        assert_eq!(q.read().unwrap().available_vram(), Some(4 << 30));
        // A call that writes nothing reads as nothing, not as the last fill.
        let _ = q.properties2(true);
        assert!(q.read().is_none());
        let _ = q.properties();
        assert!(q.read().is_none());
    }

    #[test]
    fn the_budget_cache_reads_only_when_its_copy_is_too_old() {
        let mut calls = 0;
        let mut get = |cache: &mut BudgetCache, usage| {
            cache
                .get(|p| {
                    calls += 1;
                    fill(p, 7 << 30, usage);
                })
                .and_then(Memory::available_vram)
        };
        let mut cache = BudgetCache::new(true, Duration::from_secs(3600));
        assert_eq!(get(&mut cache, 1 << 30), Some(6 << 30));
        // Fresh enough: the second fill isn't asked for.
        assert_eq!(get(&mut cache, 5 << 30), Some(6 << 30));
        cache.invalidate();
        assert_eq!(get(&mut cache, 5 << 30), Some(2 << 30));
        // Too old: every get reads.
        let mut stale = BudgetCache::new(true, Duration::ZERO);
        assert_eq!(get(&mut stale, 1 << 30), Some(6 << 30));
        std::thread::sleep(Duration::from_millis(2));
        assert_eq!(get(&mut stale, 3 << 30), Some(4 << 30));
        assert_eq!(calls, 4);
        assert!(stale.age().is_some());
        assert!(
            stale.refresh(|_| ()).is_none(),
            "a fill that writes nothing drops the copy"
        );
        assert!(stale.cached().is_none());
    }

    #[test]
    fn available_vram_needs_the_budget() {
        let plain = memory(&[(8 << 30, true)], &[(DL, 0)]);
        assert_eq!(plain.available_vram(), None);
    }

    #[test]
    fn byte_counts_read_naturally() {
        assert_eq!(bytes(256 << 20), "256 MiB");
        assert_eq!(bytes(1536), "1.50 KiB");
        assert_eq!(bytes(0), "0 B");
    }
}
