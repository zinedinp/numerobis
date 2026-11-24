//! Queue global priority (`VK_KHR_global_priority`, core 1.4): how the GPU's
//! scheduler weighs this process's queues against other processes on the
//! same GPU. It is set per queue at `vkCreateDevice` and can't change later.
//!
//! It works on the GPU only: an NPU or another GPU has its own scheduler
//! and doesn't see it. On a chip whose CPU, iGPU and NPU share power and
//! memory bandwidth, a lower priority (or less GPU work) leaves the others
//! more of both, but nothing here sets clocks.
//!
//! Above `Medium` the driver may refuse with `VK_ERROR_NOT_PERMITTED`
//! (Linux: `CAP_SYS_NICE` or root; Windows: realtime usually needs
//! administrator rights). [`create_with_fallback`] then tries the next lower
//! one, down to the driver's default.
//!
//! ```
//! use numerobis::vulkan::priority::{self, GlobalPriority, NOT_PERMITTED};
//! # use numerobis::vulkan::{ApiVersion as V, Capability, Offer, Probe};
//! # let plan = Probe::new(Offer::new(V::V1_4, V::V1_4, []), &[Capability::GlobalPriority]).plan();
//!
//! // One family's allowed priorities, when the plan has global_priority_query.
//! let allowed = None;
//! let order = priority::candidates(&plan, GlobalPriority::Realtime, allowed);
//! assert_eq!(order, [GlobalPriority::Realtime, GlobalPriority::High, GlobalPriority::Medium]);
//! // The host's vkCreateDevice; this driver refuses realtime.
//! let (device, got) = priority::create_with_fallback(&order, |p| {
//!     let info = p.map(GlobalPriority::create_info); // VkDeviceQueueCreateInfo.pNext
//!     match p {
//!         Some(GlobalPriority::Realtime) => Err(NOT_PERMITTED),
//!         _ => Ok("device"),
//!     }
//! })?;
//! assert_eq!((device, got), ("device", Some(GlobalPriority::High)));
//! # Ok::<(), i32>(())
//! ```

use std::ffi::c_void;
use std::fmt;

use super::features::s_type;
use super::{Capability, Plan};

/// `VK_ERROR_NOT_PERMITTED`: the priority needs rights the process lacks.
pub const NOT_PERMITTED: i32 = -1_000_174_001;
/// `VK_MAX_GLOBAL_PRIORITY_SIZE`.
const MAX_PRIORITIES: usize = 16;

/// `VkQueueGlobalPriority`, lowest first.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum GlobalPriority {
    /// Background work: yields to everything else.
    Low,
    /// The default without a priority.
    Medium,
    /// Ahead of normal processes (compositors, VR, latency-bound apps).
    High,
    /// Ahead of everything; usually privileged.
    Realtime,
}

impl GlobalPriority {
    pub const ALL: [Self; 4] = [Self::Low, Self::Medium, Self::High, Self::Realtime];

    #[must_use]
    pub const fn raw(self) -> i32 {
        match self {
            Self::Low => 128,
            Self::Medium => 256,
            Self::High => 512,
            Self::Realtime => 1024,
        }
    }

    #[must_use]
    pub const fn from_raw(raw: i32) -> Option<Self> {
        match raw {
            128 => Some(Self::Low),
            256 => Some(Self::Medium),
            512 => Some(Self::High),
            1024 => Some(Self::Realtime),
            _ => None,
        }
    }

    #[must_use]
    pub const fn name(self) -> &'static str {
        match self {
            Self::Low => "low",
            Self::Medium => "medium",
            Self::High => "high",
            Self::Realtime => "realtime",
        }
    }

    /// The `VkDeviceQueueGlobalPriorityCreateInfo` to chain into a
    /// `VkDeviceQueueCreateInfo`.
    #[must_use]
    pub const fn create_info(self) -> QueuePriorityInfo {
        QueuePriorityInfo {
            s_type: s_type::DEVICE_QUEUE_GLOBAL_PRIORITY_CREATE_INFO,
            p_next: std::ptr::null(),
            priority: self.raw(),
        }
    }
}

impl fmt::Display for GlobalPriority {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.name())
    }
}

impl std::str::FromStr for GlobalPriority {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, String> {
        Self::ALL
            .into_iter()
            .find(|p| p.name().eq_ignore_ascii_case(s.trim()))
            .ok_or_else(|| format!("{s:?}: not low, medium, high or realtime"))
    }
}

/// `VkDeviceQueueGlobalPriorityCreateInfo`: point
/// `VkDeviceQueueCreateInfo::pNext` at it ([`as_ptr`](Self::as_ptr)).
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct QueuePriorityInfo {
    s_type: i32,
    p_next: *const c_void,
    priority: i32,
}

// 24 bytes on 64-bit (padded after `priority`), 12 on 32-bit.
const _: () = assert!(
    size_of::<QueuePriorityInfo>()
        == (super::HEADER + 4).next_multiple_of(align_of::<*const c_void>())
);

// SAFETY: `p_next` is only stored for the host, never dereferenced here.
unsafe impl Send for QueuePriorityInfo {}
// SAFETY: as above.
unsafe impl Sync for QueuePriorityInfo {}

impl QueuePriorityInfo {
    #[must_use]
    pub fn priority(&self) -> Option<GlobalPriority> {
        GlobalPriority::from_raw(self.priority)
    }

    /// Chains another struct of the host's behind this one.
    #[must_use]
    pub const fn with_next(mut self, next: *const c_void) -> Self {
        self.p_next = next;
        self
    }

    #[must_use]
    pub const fn as_ptr(&self) -> *const c_void {
        std::ptr::from_ref(self).cast()
    }
}

/// `VkQueueFlagBits`.
pub mod queue_flags {
    pub const GRAPHICS: u32 = 0x1;
    pub const COMPUTE: u32 = 0x2;
    pub const TRANSFER: u32 = 0x4;
    pub const SPARSE_BINDING: u32 = 0x8;
    pub const PROTECTED: u32 = 0x10;
    pub const VIDEO_DECODE: u32 = 0x20;
    pub const VIDEO_ENCODE: u32 = 0x40;
    pub const OPTICAL_FLOW: u32 = 0x100;
}

/// One queue family.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct QueueFamily {
    /// [`queue_flags`].
    pub flags: u32,
    pub count: u32,
    /// The priorities it allows, lowest first; `None` when they weren't
    /// queried (no `global_priority_query`).
    pub priorities: Option<Vec<GlobalPriority>>,
}

impl QueueFamily {
    #[must_use]
    pub fn has(&self, flags: u32) -> bool {
        self.flags & flags == flags
    }
}

/// `VkQueueFamilyProperties`.
#[repr(C)]
#[derive(Clone, Copy, Default)]
struct RawFamily {
    flags: u32,
    count: u32,
    timestamp_valid_bits: u32,
    granularity: [u32; 3],
}

/// `VkQueueFamilyProperties2`.
#[repr(C)]
#[derive(Clone, Copy)]
struct RawFamily2 {
    s_type: i32,
    p_next: *mut c_void,
    properties: RawFamily,
}

/// `VkQueueFamilyGlobalPriorityProperties`.
#[repr(C)]
#[derive(Clone, Copy)]
struct RawPriorities {
    s_type: i32,
    p_next: *mut c_void,
    count: u32,
    priorities: [i32; MAX_PRIORITIES],
}

const _: () = assert!(size_of::<RawFamily>() == 24);
const _: () = assert!(size_of::<RawFamily2>() == super::HEADER + 24);
const _: () = assert!(std::mem::offset_of!(RawPriorities, priorities) == super::HEADER + 4);

/// The array for `vkGetPhysicalDeviceQueueFamilyProperties2`, with a
/// `VkQueueFamilyGlobalPriorityProperties` behind each entry when asked.
/// Ask the count first (null array), then fill:
///
/// ```text
/// vkGetPhysicalDeviceQueueFamilyProperties2(pd, &count, null);
/// let mut q = QueueFamilyQuery::new(count, plan.has(Capability::GlobalPriorityQuery));
/// vkGetPhysicalDeviceQueueFamilyProperties2(pd, &count, q.as_mut_ptr());
/// let families = q.read();
/// ```
pub struct QueueFamilyQuery {
    families: Vec<RawFamily2>,
    priorities: Option<Vec<RawPriorities>>,
}

// SAFETY: the pointers point into the query's own vectors (whose heap
// buffers don't move with it) or are null; they are set again by every
// `as_mut_ptr` call.
unsafe impl Send for QueueFamilyQuery {}
// SAFETY: `&self` methods only read plain-old-data fields.
unsafe impl Sync for QueueFamilyQuery {}

impl QueueFamilyQuery {
    /// `with_priorities` only when the device has the extension or 1.4
    /// (the plan grants [`Capability::GlobalPriorityQuery`]).
    #[must_use]
    pub fn new(count: u32, with_priorities: bool) -> Self {
        let n = count as usize;
        Self {
            families: vec![
                RawFamily2 {
                    s_type: s_type::QUEUE_FAMILY_PROPERTIES_2,
                    p_next: std::ptr::null_mut(),
                    properties: RawFamily::default(),
                };
                n
            ],
            priorities: with_priorities.then(|| {
                vec![
                    RawPriorities {
                        s_type: s_type::QUEUE_FAMILY_GLOBAL_PRIORITY_PROPERTIES,
                        p_next: std::ptr::null_mut(),
                        count: 0,
                        priorities: [0; MAX_PRIORITIES],
                    };
                    n
                ]
            }),
        }
    }

    /// The entries to hand over; valid while `self` is neither changed nor
    /// dropped.
    pub fn as_mut_ptr(&mut self) -> *mut c_void {
        for (i, family) in self.families.iter_mut().enumerate() {
            family.p_next = self.priorities.as_mut().map_or(std::ptr::null_mut(), |p| {
                std::ptr::from_mut(&mut p[i]).cast()
            });
        }
        self.families.as_mut_ptr().cast()
    }

    /// The `count` the driver wrote back may be smaller; the rest is cut.
    #[must_use]
    pub fn read_first(&self, count: u32) -> Vec<QueueFamily> {
        let n = (count as usize).min(self.families.len());
        (0..n)
            .map(|i| {
                let f = self.families[i].properties;
                QueueFamily {
                    flags: f.flags,
                    count: f.count,
                    priorities: self.priorities.as_ref().map(|p| {
                        let p = &p[i];
                        let mut v: Vec<GlobalPriority> = p.priorities
                            [..(p.count as usize).min(MAX_PRIORITIES)]
                            .iter()
                            .filter_map(|&raw| GlobalPriority::from_raw(raw))
                            .collect();
                        v.sort_unstable();
                        v
                    }),
                }
            })
            .collect()
    }

    #[must_use]
    pub fn read(&self) -> Vec<QueueFamily> {
        self.read_first(u32::try_from(self.families.len()).unwrap_or(u32::MAX))
    }
}

/// The priorities to try at `vkCreateDevice`, best first: `wanted`, then
/// each lower one down to `Medium` (below it, the default would be higher
/// than asked). Empty when the plan has no [`Capability::GlobalPriority`]:
/// then no priority struct goes in.
///
/// With [`Capability::GlobalPriorityQuery`] granted (and so enabled),
/// Vulkan requires the priority to be one the family allows: pass the
/// family's `priorities` as `allowed` and the others are left out.
#[must_use]
pub fn candidates(
    plan: &Plan,
    wanted: GlobalPriority,
    allowed: Option<&[GlobalPriority]>,
) -> Vec<GlobalPriority> {
    if !plan.has(Capability::GlobalPriority) {
        return Vec::new();
    }
    let floor = wanted.min(GlobalPriority::Medium);
    let allowed = allowed.filter(|_| plan.has(Capability::GlobalPriorityQuery));
    GlobalPriority::ALL
        .into_iter()
        .rev()
        .filter(|p| (floor..=wanted).contains(p))
        .filter(|p| allowed.is_none_or(|a| a.contains(p)))
        .collect()
}

/// Calls `create` with each of `candidates` until one isn't refused with
/// [`NOT_PERMITTED`], last with `None` (no priority struct, the driver's
/// default). `create` makes the device with
/// [`GlobalPriority::create_info`] chained into its queues. Returns the
/// device and the priority it got.
///
/// # Errors
///
/// The first error that isn't [`NOT_PERMITTED`], or `create(None)`'s.
pub fn create_with_fallback<T>(
    candidates: &[GlobalPriority],
    mut create: impl FnMut(Option<GlobalPriority>) -> Result<T, i32>,
) -> Result<(T, Option<GlobalPriority>), i32> {
    for &p in candidates {
        match create(Some(p)) {
            Ok(device) => return Ok((device, Some(p))),
            Err(NOT_PERMITTED) => {}
            Err(e) => return Err(e),
        }
    }
    create(None).map(|device| (device, None))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::vulkan::{ApiVersion as V, Offer, Probe};
    use GlobalPriority::{High, Low, Medium, Realtime};

    fn plan(caps: &[Capability], api: V, extensions: &[&'static std::ffi::CStr]) -> Plan {
        Probe::new(Offer::new(api, api, extensions.iter().copied()), caps).plan()
    }

    #[test]
    fn priorities_round_trip_and_parse() {
        for p in GlobalPriority::ALL {
            assert_eq!(GlobalPriority::from_raw(p.raw()), Some(p));
            assert_eq!(p.name().parse(), Ok(p));
            assert_eq!(p.create_info().priority(), Some(p));
        }
        assert_eq!(GlobalPriority::from_raw(300), None);
        assert!("urgent".parse::<GlobalPriority>().is_err());
        let info = High.create_info();
        // SAFETY: reading the C layout just built.
        let s_type = unsafe { *info.as_ptr().cast::<i32>() };
        assert_eq!(s_type, 1_000_174_000);
    }

    #[test]
    fn candidates_go_down_to_medium() {
        let core = plan(&[Capability::GlobalPriority], V::V1_4, &[]);
        assert!(core.has(Capability::GlobalPriority));
        assert_eq!(candidates(&core, Realtime, None), [Realtime, High, Medium]);
        assert_eq!(candidates(&core, Medium, None), [Medium]);
        assert_eq!(candidates(&core, Low, None), [Low]);
        // Without the query capability the family's list doesn't bind.
        assert_eq!(
            candidates(&core, High, Some(&[Low, Medium])),
            [High, Medium]
        );
        // Neither 1.4 nor an extension: no priority struct at all.
        let none = plan(&[Capability::GlobalPriority], V::V1_3, &[]);
        assert_eq!(candidates(&none, High, None), []);
        // The 1.3 extension route.
        let ext = plan(
            &[Capability::GlobalPriority],
            V::V1_3,
            &[c"VK_KHR_global_priority"],
        );
        assert_eq!(ext.extensions(), [c"VK_KHR_global_priority"]);
        assert_eq!(candidates(&ext, High, None), [High, Medium]);
    }

    #[test]
    fn with_the_query_only_allowed_priorities_are_tried() {
        let caps = [Capability::GlobalPriority, Capability::GlobalPriorityQuery];
        let mut probe = Probe::new(Offer::new(V::V1_4, V::V1_4, []), &caps);
        crate::vulkan::tests::fill(
            &mut probe,
            &[(crate::vulkan::Chain::Vulkan14, "globalPriorityQuery")],
        );
        let plan = probe.plan();
        assert!(plan.has(Capability::GlobalPriorityQuery), "{plan}");
        assert_eq!(
            candidates(&plan, Realtime, Some(&[Low, Medium, High])),
            [High, Medium]
        );
        assert_eq!(candidates(&plan, Realtime, None), [Realtime, High, Medium]);
        assert_eq!(candidates(&plan, Low, Some(&[Medium])), []);
    }

    #[test]
    fn refused_priorities_fall_back_and_other_errors_stop() {
        let mut tried = Vec::new();
        let got = create_with_fallback(&[Realtime, High, Medium], |p| {
            tried.push(p);
            if p > Some(Medium) {
                Err(NOT_PERMITTED)
            } else {
                Ok(1)
            }
        });
        assert_eq!(got, Ok((1, Some(Medium))));
        assert_eq!(tried, [Some(Realtime), Some(High), Some(Medium)]);
        // Everything refused: the default.
        let got = create_with_fallback(&[High], |p| p.map_or(Ok(2), |_| Err(NOT_PERMITTED)));
        assert_eq!(got, Ok((2, None)));
        // Another error isn't retried.
        let mut calls = 0;
        let got = create_with_fallback(&[High, Medium], |_| {
            calls += 1;
            Err::<(), _>(-3)
        });
        assert_eq!((got, calls), (Err(-3), 1));
        assert_eq!(create_with_fallback(&[], Ok::<_, i32>), Ok((None, None)));
    }

    #[test]
    fn the_family_query_reads_flags_and_priorities() {
        let mut q = QueueFamilyQuery::new(2, true);
        let p = q.as_mut_ptr().cast::<RawFamily2>();
        // SAFETY: two entries, each with its priority struct, written as a
        // driver would.
        unsafe {
            for i in 0..2 {
                let f = p.add(i);
                assert_eq!((*f).s_type, s_type::QUEUE_FAMILY_PROPERTIES_2);
                let pr = (*f).p_next.cast::<RawPriorities>();
                assert_eq!(
                    (*pr).s_type,
                    s_type::QUEUE_FAMILY_GLOBAL_PRIORITY_PROPERTIES
                );
                (*f).properties.flags = if i == 0 {
                    queue_flags::GRAPHICS | queue_flags::COMPUTE | queue_flags::TRANSFER
                } else {
                    queue_flags::TRANSFER
                };
                (*f).properties.count = 1;
                (*pr).count = 3;
                (*pr).priorities = [0; MAX_PRIORITIES];
                (*pr).priorities[0] = 512;
                (*pr).priorities[1] = 128;
                (*pr).priorities[2] = 256;
            }
        }
        let families = q.read();
        assert!(families[0].has(queue_flags::GRAPHICS | queue_flags::COMPUTE));
        assert!(!families[1].has(queue_flags::COMPUTE));
        assert_eq!(
            families[0].priorities.as_deref(),
            Some(&[Low, Medium, High][..])
        );
        assert_eq!(q.read_first(1).len(), 1);
        let mut plain = QueueFamilyQuery::new(1, false);
        // SAFETY: one entry.
        let next = unsafe { (*plain.as_mut_ptr().cast::<RawFamily2>()).p_next };
        assert!(next.is_null());
        assert_eq!(plain.read()[0].priorities, None);
    }
}
