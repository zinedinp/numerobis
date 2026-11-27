//! Memory sub-allocation as logic: which memory types to try for a
//! resource, how large a block to allocate per heap, where in a block a
//! resource goes, and how many `vkAllocateMemory` allocations are live
//! against `maxMemoryAllocationCount`. The host keeps the calls
//! (`vkAllocateMemory`, `vkBind*Memory`, `vkMapMemory`).
//!
//! - [`Usage`] / [`Memory::types_for`]: the memory types for a resource, in
//!   the order to try them, with the fallbacks when a heap is full.
//! - [`block_size`] / [`dedicated`]: one allocation per block of many
//!   resources, a large resource on its own.
//! - [`Block`]: a first-fit range allocator over one allocation, with
//!   alignment and `bufferImageGranularity`.
//! - [`AllocationCount`]: the live count, with a warning near the limit.
//!
//! ```
//! use numerobis::vulkan::alloc::{AllocationCount, Block, Tiling, Usage, block_size};
//! use numerobis::vulkan::memory::{Heap, Memory, MemoryType, flags};
//!
//! // A discrete GPU: 8 GiB VRAM, 16 GiB host memory, a 256 MiB BAR.
//! let heap = |size, device_local| Heap { size, device_local, budget: None, usage: None };
//! let memory = Memory {
//!     heaps: vec![heap(8 << 30, true), heap(16 << 30, false), heap(256 << 20, true)],
//!     types: vec![
//!         MemoryType { flags: flags::DEVICE_LOCAL, heap: 0 },
//!         MemoryType { flags: flags::HOST_VISIBLE | flags::HOST_COHERENT, heap: 1 },
//!         MemoryType {
//!             flags: flags::DEVICE_LOCAL | flags::HOST_VISIBLE | flags::HOST_COHERENT,
//!             heap: 2,
//!         },
//!     ],
//! };
//! // `memoryTypeBits` of the buffer's VkMemoryRequirements: all three.
//! assert_eq!(memory.types_for(0b111, Usage::Upload, false), [2, 1]);
//! assert_eq!(memory.types_for(0b111, Usage::Device, false), [0, 2, 1]);
//!
//! static LIVE: AllocationCount = AllocationCount::new(4096);
//! // vkAllocateMemory(block_size(8 << 30, 256 << 20)) on type 0, then:
//! assert_eq!(LIVE.allocated(), None, "far from the limit");
//! let mut block = Block::new(block_size(8 << 30, 256 << 20), 1024);
//! let vertices = block.alloc(4096, 16, Tiling::Linear).unwrap();
//! let texture = block.alloc(65536, 256, Tiling::Optimal).unwrap();
//! assert_eq!((vertices, texture), (0, 4096));
//! // vkBindBufferMemory(buffer, memory, vertices), ...
//! block.free(vertices);
//! ```

use std::collections::BTreeMap;
use std::sync::atomic::{AtomicBool, AtomicU32, Ordering};

use super::memory::{Memory, flags};

/// `VK_MEMORY_PROPERTY_DEVICE_COHERENT_BIT_AMD` and
/// `VK_MEMORY_PROPERTY_DEVICE_UNCACHED_BIT_AMD`: for debugging GPU crashes
/// (`VK_AMD_device_coherent_memory`), slow, and never what a resource wants.
const AMD_DEBUG: u32 = 0x40 | 0x80;
const HOST: u32 = flags::HOST_VISIBLE | flags::HOST_COHERENT;

/// One step of a [`Usage`]'s waterfall: a memory type with all of `has`
/// and none of `lacks`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Want {
    pub has: u32,
    pub lacks: u32,
}

/// What a resource's memory is for. Mapped memory is always
/// `HOST_COHERENT`, so the host never flushes or invalidates.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Usage {
    /// The GPU reads and writes it, a copy fills it, the CPU never maps it
    /// (meshes, textures, render targets): VRAM, else host memory, which
    /// is slower to read from but keeps the resource while another program
    /// holds the VRAM.
    Device,
    /// As [`Device`](Self::Device), but VRAM the CPU maps first (resizable
    /// BAR, unified memory), so uploads are written straight into it
    /// without a staging copy. Ask for it only when
    /// [`Memory::full_cpu_access`]: on a 256 MiB BAR it fills the window
    /// that [`Upload`](Self::Upload) needs.
    DeviceMapped,
    /// The CPU writes it through a mapping and the GPU reads it every
    /// frame (uniforms, per-frame instances): VRAM it maps, else host
    /// memory. Write-combined either way: written in order, never read.
    /// A buffer only a copy reads is [`Staging`](Self::Staging).
    Upload,
    /// The CPU writes it once through a mapping and a copy reads it
    /// (texture and mesh uploads): host memory, uncached (write-combined)
    /// before cached, so the BAR window stays for [`Upload`](Self::Upload);
    /// mapped VRAM only where there is no host memory (unified memory) or
    /// it is full. The same past the VRAM budget, as it never wants VRAM.
    Staging,
    /// The GPU writes it and the CPU reads it back (queries, screenshots,
    /// occlusion results): cached host memory, as reading write-combined
    /// memory costs the CPU many times more.
    Readback,
    /// Attachments only a render pass touches (a depth buffer nothing
    /// samples): `LAZILY_ALLOCATED` on tilers, which may never back it,
    /// else VRAM.
    Transient,
}

impl Usage {
    /// The waterfall. `over_budget`: VRAM is past
    /// [`Memory::available_vram`], and host memory goes first for what may
    /// live there; the OS would otherwise evict other allocations from
    /// VRAM to make room.
    #[must_use]
    pub const fn waterfall(self, over_budget: bool) -> &'static [Want] {
        const DL: u32 = flags::DEVICE_LOCAL;
        const VRAM: Want = Want { has: DL, lacks: 0 };
        const VRAM_UNMAPPED: Want = Want {
            has: DL,
            lacks: flags::HOST_VISIBLE,
        };
        const VRAM_MAPPED: Want = Want {
            has: DL | HOST,
            lacks: 0,
        };
        const LAZY: Want = Want {
            has: DL | flags::LAZILY_ALLOCATED,
            lacks: 0,
        };
        const MAPPED: Want = Want {
            has: HOST,
            lacks: 0,
        };
        const HOST_ONLY: Want = Want {
            has: HOST,
            lacks: DL,
        };
        const HOST_UNCACHED: Want = Want {
            has: HOST,
            lacks: DL | flags::HOST_CACHED,
        };
        const MAPPED_UNCACHED: Want = Want {
            has: HOST,
            lacks: flags::HOST_CACHED,
        };
        const CACHED: Want = Want {
            has: HOST | flags::HOST_CACHED,
            lacks: 0,
        };
        match (self, over_budget) {
            (Self::Device | Self::DeviceMapped, true) => &[HOST_ONLY, VRAM],
            // VRAM the CPU can't map before VRAM it can: a 256 MiB BAR is
            // better left to `Upload`.
            (Self::Device, false) => &[VRAM_UNMAPPED, VRAM, MAPPED],
            (Self::DeviceMapped, false) => &[VRAM_MAPPED, VRAM, MAPPED],
            (Self::Upload, false) => &[VRAM_MAPPED, MAPPED],
            (Self::Upload, true) => &[HOST_ONLY, MAPPED],
            (Self::Staging, _) => &[HOST_UNCACHED, HOST_ONLY, MAPPED_UNCACHED, MAPPED],
            (Self::Readback, _) => &[CACHED, HOST_ONLY, MAPPED],
            (Self::Transient, _) => &[LAZY, VRAM],
        }
    }
}

impl Memory {
    /// The memory types among `type_bits` (`VkMemoryRequirements::
    /// memoryTypeBits`) for `usage`, in the order to try them: per step of
    /// its [`waterfall`](Usage::waterfall), every matching type in the
    /// device's order, each type once. Every one, not the first per step: a
    /// full heap (the VRAM, the BAR window) can have a type whose flags the
    /// next heap's type shares.
    ///
    /// Never protected memory or AMD's debug types; `LAZILY_ALLOCATED`
    /// only for [`Usage::Transient`].
    #[must_use]
    pub fn types_for(&self, type_bits: u32, usage: Usage, over_budget: bool) -> Vec<u32> {
        let mut order = Vec::new();
        for step in usage.waterfall(over_budget) {
            let lazy = if step.has & flags::LAZILY_ALLOCATED == 0 {
                flags::LAZILY_ALLOCATED
            } else {
                0
            };
            let lacks = step.lacks | flags::PROTECTED | AMD_DEBUG | lazy;
            for (i, t) in (0u32..).zip(&self.types) {
                if i < 32
                    && type_bits & (1 << i) != 0
                    && t.has(step.has)
                    && t.flags & lacks == 0
                    && !order.contains(&i)
                {
                    order.push(i);
                }
            }
        }
        order
    }
}

/// The usual cap of [`block_size`]: 256 MiB, small enough that a block is
/// not hard to find room for, large enough that a few thousand
/// allocations reach any heap.
pub const BLOCK_CAP: u64 = 256 << 20;

/// Bytes per block on a heap of `heap_size`: an eighth of the heap, at most
/// `cap` ([`BLOCK_CAP`] usually). The BAR window of 256 MiB gets 32 MiB
/// blocks, so one half-used block doesn't hold most of it.
#[must_use]
pub const fn block_size(heap_size: u64, cap: u64) -> u64 {
    let eighth = heap_size / 8;
    let size = if eighth < cap { eighth } else { cap };
    if size == 0 { 1 } else { size }
}

/// Whether a resource of `size` bytes gets an allocation of its own rather
/// than a range of a block of `block` bytes: when it takes more than half a
/// block, or the driver prefers it (`VkMemoryDedicatedRequirements::
/// prefersDedicatedAllocation`). `requiresDedicatedAllocation` always gets
/// its own; near the [`AllocationCount`] limit, a preference is better
/// ignored.
#[must_use]
pub const fn dedicated(size: u64, block: u64, prefers: bool) -> bool {
    prefers || size > block / 2
}

/// How a resource lays out its memory, for `bufferImageGranularity`: a
/// linear and an optimal one must not share a page of that size.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Tiling {
    /// Buffers and `VK_IMAGE_TILING_LINEAR` images.
    Linear,
    /// `VK_IMAGE_TILING_OPTIMAL` images.
    Optimal,
}

/// First-fit ranges of one allocation of `size` bytes, coalescing on free.
///
/// Offsets meet each resource's alignment (any non-zero value, so element
/// sizes such as 12 work as well as Vulkan's powers of two) and keep
/// [`Tiling::Linear`] and [`Tiling::Optimal`] resources out of each other's
/// `bufferImageGranularity` pages. A host that keeps linear and optimal
/// resources in blocks of their own passes a granularity of 1.
///
/// What the GPU may still read is the host's to keep: free a range only
/// once the last frame that used it has finished.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Block {
    size: u64,
    granularity: u64,
    /// Free `(offset, len)` regions, sorted by offset, never adjacent.
    free: Vec<(u64, u64)>,
    /// Ranges in use, by offset: size and tiling.
    used: BTreeMap<u64, (u64, Tiling)>,
}

impl Block {
    /// `granularity`: the device's `bufferImageGranularity` (a power of
    /// two), or 1 when the block holds one tiling only.
    #[must_use]
    pub fn new(size: u64, granularity: u64) -> Self {
        Self {
            size,
            granularity: granularity.max(1),
            free: if size > 0 {
                vec![(0, size)]
            } else {
                Vec::new()
            },
            used: BTreeMap::new(),
        }
    }

    /// The offset of a new range of `size` bytes, or `None` if no free
    /// region fits it (or `size` is 0).
    pub fn alloc(&mut self, size: u64, align: u64, tiling: Tiling) -> Option<u64> {
        if size == 0 {
            return None;
        }
        let align = align.max(1);
        let (i, start, end) = self
            .free
            .iter()
            .enumerate()
            .find_map(|(i, &(offset, len))| {
                let (start, end) = self.fit(offset, offset + len, size, align, tiling)?;
                Some((i, start, end))
            })?;
        let (offset, len) = self.free[i];
        let region_end = offset + len;
        match (start > offset, end < region_end) {
            (false, false) => {
                self.free.remove(i);
            }
            (true, false) => self.free[i] = (offset, start - offset),
            (false, true) => self.free[i] = (end, region_end - end),
            (true, true) => {
                self.free[i] = (offset, start - offset);
                self.free.insert(i + 1, (end, region_end - end));
            }
        }
        self.used.insert(start, (size, tiling));
        Some(start)
    }

    /// Where `size` bytes go in the free region `offset..region_end`.
    fn fit(
        &self,
        offset: u64,
        region_end: u64,
        size: u64,
        align: u64,
        tiling: Tiling,
    ) -> Option<(u64, u64)> {
        let g = self.granularity;
        let page = |at: u64| at / g;
        let mut start = offset.checked_next_multiple_of(align)?;
        if g > 1 {
            // A resource of the other tiling ending in the page `start` is
            // in: move to the next page. Ranges before the region end at
            // or before `offset`; walking back stops at the first that
            // ends in an earlier page.
            let clash = self
                .used
                .range(..offset)
                .rev()
                .map(|(&at, &(len, t))| (at + len - 1, t))
                .take_while(|&(last, _)| page(last) == page(start))
                .any(|(_, t)| t != tiling);
            if clash {
                start = start
                    .checked_next_multiple_of(g)?
                    .checked_next_multiple_of(align)?;
            }
        }
        let end = start.checked_add(size)?;
        if end > region_end {
            return None;
        }
        if g > 1 {
            // A resource of the other tiling starting in the page the new
            // one ends in: it would share that page, so not here.
            let clash = self
                .used
                .range(region_end..)
                .take_while(|&(&at, _)| page(at) == page(end - 1))
                .any(|(_, &(_, t))| t != tiling);
            if clash {
                return None;
            }
        }
        Some((start, end))
    }

    /// Gives the range at `offset` back; its size, or `None` if no range
    /// starts there.
    pub fn free(&mut self, offset: u64) -> Option<u64> {
        let (size, _) = self.used.remove(&offset)?;
        let at = self.free.partition_point(|&(o, _)| o < offset);
        self.free.insert(at, (offset, size));
        if let Some(&(next, next_len)) = self.free.get(at + 1)
            && offset + size == next
        {
            self.free[at].1 += next_len;
            self.free.remove(at + 1);
        }
        if at > 0 {
            let (prev, prev_len) = self.free[at - 1];
            if prev + prev_len == offset {
                self.free[at - 1].1 += self.free[at].1;
                self.free.remove(at);
            }
        }
        Some(size)
    }

    /// Bytes of the allocation.
    #[must_use]
    pub fn size(&self) -> u64 {
        self.size
    }

    /// No range in use: the allocation may be freed.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.used.is_empty()
    }

    /// Bytes in use.
    #[must_use]
    pub fn used(&self) -> u64 {
        self.used.values().map(|&(len, _)| len).sum()
    }

    /// The largest free region: a range larger than this can't fit
    /// (alignment and granularity can make the limit lower).
    #[must_use]
    pub fn largest_free(&self) -> u64 {
        self.free.iter().map(|&(_, len)| len).max().unwrap_or(0)
    }

    /// The free `(offset, len)` regions, by offset.
    #[must_use]
    pub fn free_regions(&self) -> &[(u64, u64)] {
        &self.free
    }
}

/// Live `vkAllocateMemory` allocations against `maxMemoryAllocationCount`
/// (4096 on Windows drivers, often more elsewhere). Atomic, so one `static`
/// counts for every thread that allocates.
///
/// [`allocated`](Self::allocated) says once when the count reaches three
/// quarters of the limit, and again only after it fell below half.
#[derive(Debug)]
pub struct AllocationCount {
    live: AtomicU32,
    limit: AtomicU32,
    warned: AtomicBool,
}

impl AllocationCount {
    /// The limit until the device's is known: 4096, the usual Windows value
    /// and the lowest the spec allows.
    pub const DEFAULT_LIMIT: u32 = 4096;

    #[must_use]
    pub const fn new(limit: u32) -> Self {
        Self {
            live: AtomicU32::new(0),
            limit: AtomicU32::new(limit),
            warned: AtomicBool::new(false),
        }
    }

    /// The device's `VkPhysicalDeviceLimits::maxMemoryAllocationCount`.
    pub fn set_limit(&self, limit: u32) {
        self.limit.store(limit, Ordering::Relaxed);
    }

    #[must_use]
    pub fn live(&self) -> u32 {
        self.live.load(Ordering::Relaxed)
    }

    #[must_use]
    pub fn limit(&self) -> u32 {
        self.limit.load(Ordering::Relaxed)
    }

    /// Another allocation stays under the limit.
    #[must_use]
    pub fn has_room(&self) -> bool {
        self.live() < self.limit()
    }

    /// Whether `live` allocations are close to `limit`: three quarters.
    #[must_use]
    pub const fn near_limit(live: u32, limit: u32) -> bool {
        live >= limit - limit / 4
    }

    /// Counts a successful `vkAllocateMemory`. `Some(live)` when the count
    /// reached [`near_limit`](Self::near_limit) and wasn't told yet: time
    /// to warn, or to stop giving resources their own allocation.
    #[must_use]
    pub fn allocated(&self) -> Option<u32> {
        let live = self.live.fetch_add(1, Ordering::Relaxed).saturating_add(1);
        (Self::near_limit(live, self.limit()) && !self.warned.swap(true, Ordering::Relaxed))
            .then_some(live)
    }

    /// Counts a `vkFreeMemory`.
    pub fn freed(&self) {
        let before = self
            .live
            .try_update(Ordering::Relaxed, Ordering::Relaxed, |n| {
                Some(n.saturating_sub(1))
            })
            .unwrap_or(0);
        if before.saturating_sub(1) < self.limit() / 2 {
            self.warned.store(false, Ordering::Relaxed);
        }
    }
}

impl Default for AllocationCount {
    fn default() -> Self {
        Self::new(Self::DEFAULT_LIMIT)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::vulkan::memory::{Heap, MemoryType};

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
    const CACHED: u32 = HOST | flags::HOST_CACHED;

    /// The Arc B580 on Windows: two VRAM types, two host types, the BAR
    /// type last (resizable BAR, so it spans the VRAM heap).
    fn arc() -> Memory {
        memory(
            &[(12 << 30, true), (16 << 30, false)],
            &[(DL, 0), (DL, 0), (HOST, 1), (CACHED, 1), (DL | HOST, 0)],
        )
    }

    #[test]
    fn each_usage_has_its_order() {
        let m = arc();
        let all = 0b1_1111;
        assert_eq!(m.types_for(all, Usage::Device, false), [0, 1, 4, 2, 3]);
        assert_eq!(
            m.types_for(all, Usage::DeviceMapped, false),
            [4, 0, 1, 2, 3]
        );
        assert_eq!(m.types_for(all, Usage::Upload, false), [4, 2, 3]);
        assert_eq!(m.types_for(all, Usage::Staging, false), [2, 3, 4]);
        assert_eq!(m.types_for(all, Usage::Readback, false), [3, 2, 4]);
        assert_eq!(m.types_for(all, Usage::Transient, false), [0, 1, 4]);
        // Past the budget: host memory first, and not the BAR type, which
        // is VRAM too.
        assert_eq!(m.types_for(all, Usage::Device, true), [2, 3, 0, 1, 4]);
        assert_eq!(m.types_for(all, Usage::Upload, true), [2, 3, 4]);
        assert_eq!(m.types_for(all, Usage::Staging, true), [2, 3, 4]);
    }

    #[test]
    fn staging_takes_uncached_host_memory_first() {
        // The cached host type listed before the uncached one.
        let m = memory(
            &[(8 << 30, true), (16 << 30, false), (256 << 20, true)],
            &[(DL, 0), (CACHED, 1), (HOST, 1), (DL | HOST, 2)],
        );
        assert_eq!(m.types_for(0b1111, Usage::Staging, false), [2, 1, 3]);
        // The host types can't hold the buffer: the BAR.
        assert_eq!(m.types_for(0b1001, Usage::Staging, false), [3]);
        // Unified memory: every type is VRAM, uncached before cached.
        let uma = memory(
            &[(16 << 30, true)],
            &[(DL, 0), (DL | CACHED, 0), (DL | HOST, 0)],
        );
        assert_eq!(uma.types_for(0b111, Usage::Staging, false), [2, 1]);
    }

    #[test]
    fn type_bits_protected_and_lazy_types_are_left_out() {
        let m = memory(
            &[(4 << 30, true), (8 << 30, false)],
            &[
                (DL | flags::PROTECTED, 0),
                (DL | flags::LAZILY_ALLOCATED, 0),
                (DL | 0x40, 0),
                (DL, 0),
                (HOST, 1),
            ],
        );
        assert_eq!(m.types_for(0b1_1111, Usage::Device, false), [3, 4]);
        assert_eq!(m.types_for(0b1_1111, Usage::Transient, false), [1, 3]);
        // The image can't live in type 3.
        assert_eq!(m.types_for(0b1_0111, Usage::Transient, false), [1]);
        assert_eq!(m.types_for(0, Usage::Device, false), []);
    }

    #[test]
    fn blocks_are_an_eighth_of_the_heap_capped() {
        assert_eq!(block_size(256 << 20, BLOCK_CAP), 32 << 20);
        assert_eq!(block_size(12 << 30, BLOCK_CAP), 256 << 20);
        assert_eq!(block_size(12 << 30, 64 << 20), 64 << 20);
        assert_eq!(block_size(0, BLOCK_CAP), 1);
        assert!(dedicated(200 << 20, 256 << 20, false));
        assert!(!dedicated(64 << 20, 256 << 20, false));
        assert!(dedicated(1, 256 << 20, true));
    }

    #[test]
    #[allow(clippy::many_single_char_names, reason = "the ranges are x, y, z")]
    fn first_fit_reuses_and_coalesces() {
        let mut b = Block::new(100, 1);
        let l = Tiling::Linear;
        let (x, y, z) = (b.alloc(10, 1, l), b.alloc(20, 1, l), b.alloc(30, 1, l));
        assert_eq!((x, y, z), (Some(0), Some(10), Some(30)));
        assert_eq!(b.free(10), Some(20));
        assert_eq!(b.alloc(15, 1, l), Some(10));
        assert_eq!(b.free(10), Some(15));
        assert_eq!(b.free(0), Some(10));
        assert_eq!(b.free(0), None, "freed twice");
        assert_eq!(b.free(30), Some(30));
        assert_eq!(b.free_regions(), [(0, 100)]);
        assert!(b.is_empty());
        assert_eq!(b.alloc(101, 1, l), None);
        assert_eq!(b.alloc(0, 1, l), None);
        assert_eq!(b.alloc(100, 1, l), Some(0));
        assert_eq!(b.free_regions(), []);
        assert_eq!(b.alloc(1, 1, l), None);
    }

    #[test]
    fn alignment_leaves_the_gap_free() {
        let mut b = Block::new(1024, 1);
        let l = Tiling::Linear;
        assert_eq!(b.alloc(3, 1, l), Some(0));
        assert_eq!(b.alloc(64, 256, l), Some(256));
        assert_eq!(b.free_regions(), [(3, 253), (320, 704)]);
        // Element sizes that aren't powers of two: 12-byte vertices.
        assert_eq!(b.alloc(24, 12, l), Some(12));
        assert_eq!(b.used(), 3 + 64 + 24);
        assert_eq!(b.largest_free(), 704);
    }

    #[test]
    fn linear_and_optimal_keep_out_of_each_others_pages() {
        let (l, o) = (Tiling::Linear, Tiling::Optimal);
        let mut b = Block::new(8192, 1024);
        assert_eq!(b.alloc(100, 16, l), Some(0));
        // Same tiling shares the page.
        assert_eq!(b.alloc(100, 16, l), Some(112));
        // The other one starts on the next page.
        assert_eq!(b.alloc(4096, 256, o), Some(1024));
        // The gap before the image takes buffers: its pages end before
        // the image's first.
        assert_eq!(b.alloc(16, 16, l), Some(224));
        // But no image: it would share the buffers' page. It goes after
        // the first image, which has the same tiling.
        assert_eq!(b.alloc(256, 256, o), Some(5120));
        // A buffer right after an image starts on the next page.
        assert_eq!(b.alloc(800, 16, l), Some(6144));
        // Without granularity the gap takes the image.
        let mut flat = Block::new(8192, 1);
        assert_eq!(flat.alloc(100, 16, l), Some(0));
        assert_eq!(flat.alloc(4096, 256, o), Some(256));
    }

    #[test]
    fn a_range_may_not_end_in_the_page_the_next_one_starts() {
        let (l, o) = (Tiling::Linear, Tiling::Optimal);
        let mut b = Block::new(4096, 1024);
        assert_eq!(b.alloc(16, 16, l), Some(0));
        assert_eq!(b.alloc(16, 16, o), Some(1024));
        assert_eq!(b.alloc(16, 16, o), Some(1040));
        assert_eq!(b.free(1024), Some(16));
        // 16..1040 is free, the image at 1040 is in page 1: a buffer
        // reaching page 1 doesn't fit there, so it goes after the image's page.
        assert_eq!(b.alloc(1010, 16, l), Some(2048));
        // One that stays in page 0 does.
        assert_eq!(b.alloc(900, 16, l), Some(16));
    }

    #[test]
    fn the_count_warns_once_near_the_limit_and_rearms_below_half() {
        let count = AllocationCount::new(8);
        let warnings: Vec<_> = (0..8).filter_map(|_| count.allocated()).collect();
        assert_eq!(warnings, [6]);
        assert!(!count.has_room());
        for _ in 0..4 {
            count.freed();
        }
        assert_eq!(count.allocated(), None, "not below half yet");
        count.freed();
        count.freed();
        assert_eq!(count.live(), 3);
        assert_eq!(count.allocated(), None);
        assert_eq!(count.allocated(), None);
        assert_eq!(count.allocated(), Some(6));
        let empty = AllocationCount::default();
        empty.freed();
        assert_eq!(empty.live(), 0);
        assert!(AllocationCount::near_limit(3072, 4096));
        assert!(!AllocationCount::near_limit(3071, 4096));
        assert!(AllocationCount::near_limit(u32::MAX, u32::MAX));
    }
}
