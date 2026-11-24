//! Direct3D 12 command queue priority: `D3D12_COMMAND_QUEUE_DESC::Priority`.
//!
//! The same idea as Vulkan's global priority, with three levels: normal
//! (the default), high, and global realtime, which needs
//! `SeIncreaseBasePriorityPrivilege` ([`QueuePriority::privilege`]).
//! `D3D12_FEATURE_COMMAND_QUEUE_PRIORITY` says per queue type which ones the
//! device takes, but not whether the process may: it reported global
//! realtime on the B580 without the privilege, and `CreateCommandQueue` then
//! failed (with a debug layer error). [`candidates`] asks the device and the
//! host's privilege check, [`create_with_fallback`] steps down when creation
//! is refused anyway.
//!
//! On vkd3d-proton (Direct3D 12 on Vulkan) the priority is accepted up to
//! high and then ignored: its Vulkan queues are made up front.
//!
//! ```
//! use numerobis::directx::modern::queue::{self, ListType, QueuePriority};
//!
//! // CheckFeatureSupport(D3D12_FEATURE_COMMAND_QUEUE_PRIORITY, ...) through the closure.
//! let device = |_, data: *mut std::ffi::c_void, _| {
//!     unsafe { *data.cast::<u32>().add(2) = 1 }; // PriorityForTypeIsSupported
//!     true
//! };
//! // A process without SeIncreaseBasePriorityPrivilege: no realtime to try.
//! let order = queue::candidates(ListType::Direct, QueuePriority::GlobalRealtime, device, |_| false);
//! assert_eq!(order, [QueuePriority::High, QueuePriority::Normal]);
//! // Letting creation find out instead.
//! let order = queue::candidates(ListType::Direct, QueuePriority::GlobalRealtime, device, |_| true);
//! assert_eq!(order, [QueuePriority::GlobalRealtime, QueuePriority::High, QueuePriority::Normal]);
//! let (queue, got) = queue::create_with_fallback(&order, |p| {
//!     // D3D12_COMMAND_QUEUE_DESC { Priority: p.raw(), .. } -> CreateCommandQueue
//!     if p == QueuePriority::GlobalRealtime { Err(queue::E_ACCESSDENIED) } else { Ok("queue") }
//! })?;
//! assert_eq!((queue, got), ("queue", QueuePriority::High));
//! # Ok::<(), i32>(())
//! ```

use std::ffi::c_void;
use std::fmt;

use super::super::generated::value as v;
use super::D3d12Feature;

/// `E_ACCESSDENIED`, as an `HRESULT`.
#[allow(clippy::cast_possible_wrap, reason = "an HRESULT is its bits")]
pub const E_ACCESSDENIED: i32 = 0x8007_0005_u32 as i32;

/// `D3D12_COMMAND_QUEUE_PRIORITY`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum QueuePriority {
    Normal,
    High,
    GlobalRealtime,
}

impl QueuePriority {
    pub const ALL: [Self; 3] = [Self::Normal, Self::High, Self::GlobalRealtime];

    /// For `D3D12_COMMAND_QUEUE_DESC::Priority`.
    #[must_use]
    pub const fn raw(self) -> u32 {
        match self {
            Self::Normal => v::D3D12_COMMAND_QUEUE_PRIORITY_NORMAL,
            Self::High => v::D3D12_COMMAND_QUEUE_PRIORITY_HIGH,
            Self::GlobalRealtime => v::D3D12_COMMAND_QUEUE_PRIORITY_GLOBAL_REALTIME,
        }
    }

    #[must_use]
    pub const fn from_raw(raw: u32) -> Option<Self> {
        match raw {
            v::D3D12_COMMAND_QUEUE_PRIORITY_NORMAL => Some(Self::Normal),
            v::D3D12_COMMAND_QUEUE_PRIORITY_HIGH => Some(Self::High),
            v::D3D12_COMMAND_QUEUE_PRIORITY_GLOBAL_REALTIME => Some(Self::GlobalRealtime),
            _ => None,
        }
    }

    /// The Windows privilege a process needs to create a queue at this
    /// priority (enabled in its token).
    #[must_use]
    pub const fn privilege(self) -> Option<&'static str> {
        match self {
            Self::GlobalRealtime => Some("SeIncreaseBasePriorityPrivilege"),
            Self::Normal | Self::High => None,
        }
    }

    #[must_use]
    pub const fn name(self) -> &'static str {
        match self {
            Self::Normal => "normal",
            Self::High => "high",
            Self::GlobalRealtime => "global realtime",
        }
    }
}

impl fmt::Display for QueuePriority {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.name())
    }
}

impl std::str::FromStr for QueuePriority {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, String> {
        match s.trim().to_ascii_lowercase().as_str() {
            "normal" | "medium" => Ok(Self::Normal),
            "high" => Ok(Self::High),
            "realtime" | "global_realtime" | "global realtime" => Ok(Self::GlobalRealtime),
            _ => Err(format!("{s:?}: expected normal, high or realtime")),
        }
    }
}

/// `D3D12_COMMAND_LIST_TYPE` of a queue.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ListType {
    Direct,
    Compute,
    Copy,
    VideoDecode,
    VideoProcess,
    VideoEncode,
}

impl ListType {
    #[must_use]
    pub const fn raw(self) -> u32 {
        match self {
            Self::Direct => v::D3D12_COMMAND_LIST_TYPE_DIRECT,
            Self::Compute => v::D3D12_COMMAND_LIST_TYPE_COMPUTE,
            Self::Copy => v::D3D12_COMMAND_LIST_TYPE_COPY,
            Self::VideoDecode => v::D3D12_COMMAND_LIST_TYPE_VIDEO_DECODE,
            Self::VideoProcess => v::D3D12_COMMAND_LIST_TYPE_VIDEO_PROCESS,
            Self::VideoEncode => v::D3D12_COMMAND_LIST_TYPE_VIDEO_ENCODE,
        }
    }
}

/// Whether the device takes `priority` for queues of `list`, through
/// `check(feature, data, size)` (`CheckFeatureSupport`).
pub fn supported(
    list: ListType,
    priority: QueuePriority,
    mut check: impl FnMut(u32, *mut c_void, u32) -> bool,
) -> bool {
    // CommandListType, Priority in; PriorityForTypeIsSupported out.
    let mut data = [list.raw(), priority.raw(), 0u32];
    check(
        D3d12Feature::CommandQueuePriority.id(),
        data.as_mut_ptr().cast(),
        12,
    ) && data[2] != 0
}

/// The priorities to try for a queue of `list`, from `wanted` down to
/// normal, keeping only those the device says it takes (normal always) and
/// whose [`privilege`](QueuePriority::privilege) `has_privilege` grants:
/// the host checks its token (`PrivilegeCheck`), or answers `true` to let
/// [`create_with_fallback`] find out.
pub fn candidates(
    list: ListType,
    wanted: QueuePriority,
    mut check: impl FnMut(u32, *mut c_void, u32) -> bool,
    mut has_privilege: impl FnMut(&str) -> bool,
) -> Vec<QueuePriority> {
    QueuePriority::ALL
        .iter()
        .rev()
        .copied()
        .filter(|&p| p <= wanted)
        .filter(|&p| p.privilege().is_none_or(&mut has_privilege))
        .filter(|&p| p == QueuePriority::Normal || supported(list, p, &mut check))
        .collect()
}

/// Creates the queue at the first priority of `order` that `create`
/// accepts. A refusal above normal steps down; at normal (or the last one)
/// its `HRESULT` is the error.
///
/// # Errors
///
/// The last `HRESULT`, or `E_ACCESSDENIED` for an empty `order`.
pub fn create_with_fallback<T>(
    order: &[QueuePriority],
    mut create: impl FnMut(QueuePriority) -> Result<T, i32>,
) -> Result<(T, QueuePriority), i32> {
    let mut last = E_ACCESSDENIED;
    for &p in order {
        match create(p) {
            Ok(made) => return Ok((made, p)),
            Err(hr) if p > QueuePriority::Normal => last = hr,
            Err(hr) => return Err(hr),
        }
    }
    Err(last)
}
