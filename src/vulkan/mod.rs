//! Vulkan versions, their feature bits, and a waterfall that picks the best
//! way to get each capability a renderer wants.
//!
//! No Vulkan bindings: like [`pci`](crate::pci) this is data plus logic, and
//! the host makes the Vulkan calls with whatever bindings it uses (ash,
//! vulkanalia, ...). The structs handed over are plain C layouts
//! (`sType`, `pNext`, `VkBool32`s).
//!
//! - [`features`]: per core version its feature bits, the struct holding
//!   them, and the extensions it promoted.
//! - [`caps`]: the [`Capability`]s (draw, memory, shader, descriptor, sync,
//!   render), each with its [`Route`]s in fallback order.
//! - [`Instance`]: what `vkCreateInstance` needs (portability enumeration,
//!   `VK_KHR_get_physical_device_properties2` on a 1.0 loader).
//! - [`Probe`] / [`Plan`]: the waterfall and its log.
//! - [`cache`]: the plans kept for the process after one probe at startup;
//!   [`BudgetCache`] reads the memory budget again when it is too old.
//! - [`priority`]: queue global priority at `vkCreateDevice`, with the
//!   fallback when the driver refuses a level.
//! - [`alloc`]: memory sub-allocation as logic: memory types per usage,
//!   block sizes, a range allocator, the live allocation count.
//! - [`swapchain`]: present mode, surface format (SDR, HDR, wide gamut),
//!   extent and image count, and when to recreate after a present.
//! - [`color_space`]: what a `VkColorSpaceKHR` means as a [`keinbonus`]
//!   colour description, and `VkHdrMetadataEXT` from HDR10 metadata.
//! - [`format`](mod@format): the depth format waterfall, the `VkFormat`s used,
//!   and one block of a format per aspect for copy bounds.
//! - [`barrier`]: `synchronization2`'s stages, accesses and layouts lowered
//!   to `vkCmdPipelineBarrier`'s on a device without it.
//!
//! ```
//! use numerobis::vulkan::{ApiVersion, Capability as Cap, Instance, Offer, Probe};
//!
//! // Before vkCreateInstance: what the loader reports (capped by what the
//! // tables know) and its instance extensions.
//! let instance = Instance::new(
//!     Some(ApiVersion::V1_3),
//!     ApiVersion::NEWEST,
//!     [c"VK_KHR_portability_enumeration"],
//! );
//! // vkCreateInstance(apiVersion = instance.api.raw(), flags = instance.flags,
//! //                  extensions = instance.extensions() + the host's own)
//! // After picking the physical device: its version and extensions.
//! let offer = Offer::new(instance.api, ApiVersion::V1_1, [c"VK_KHR_draw_indirect_count"])
//!     .with_instance(instance.extensions().iter().copied());
//! let mut probe = Probe::new(offer, &[Cap::DrawIndirectCount, Cap::BufferDeviceAddress]);
//! let _query = probe.features2(); // hand to vkGetPhysicalDeviceFeatures2
//! let plan = probe.plan();
//! assert!(plan.has(Cap::DrawIndirectCount)); // through the extension on 1.1
//! let required = [Cap::DrawIndirectCount, Cap::BufferDeviceAddress];
//! assert_eq!(plan.lacking(&required), [Cap::BufferDeviceAddress]);
//! assert_eq!(plan.extensions(), [c"VK_KHR_draw_indirect_count"]);
//! println!("{plan}");
//! ```

use std::ffi::CStr;
use std::fmt;

pub mod alloc;
pub mod barrier;
pub mod cache;
pub mod caps;
pub mod color_space;
pub mod device;
pub mod features;
pub mod format;
pub mod memory;
mod plan;
pub mod priority;
pub mod swapchain;

pub use crate::platform::{Os, Status, Support};
pub use caps::{Area, Capability, Route};
pub use device::{DeviceInfo, DeviceType, Quirk, rank};
pub use features::Chain;
pub use memory::{BudgetCache, Memory, MemoryQuery};
pub use plan::{BitError, DeviceChain, FeatureQuery, Offer, Outcome, Plan, Probe, Skip, Verdict};
pub use priority::GlobalPriority;

/// `sType` and `pNext` at the head of every chained struct, with the
/// padding before `pNext`: 16 bytes on 64-bit targets, 8 on 32-bit. The
/// layout checks are written with it, so they hold on both.
pub(crate) const HEADER: usize = 2 * size_of::<*const std::ffi::c_void>();

/// A Vulkan version (`VK_MAKE_API_VERSION` without the variant).
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct ApiVersion {
    pub major: u32,
    pub minor: u32,
    pub patch: u32,
}

impl ApiVersion {
    pub const V1_0: Self = Self::new(1, 0);
    pub const V1_1: Self = Self::new(1, 1);
    pub const V1_2: Self = Self::new(1, 2);
    pub const V1_3: Self = Self::new(1, 3);
    pub const V1_4: Self = Self::new(1, 4);
    /// The newest version the tables know.
    pub const NEWEST: Self = Self::V1_4;

    #[must_use]
    pub const fn new(major: u32, minor: u32) -> Self {
        Self {
            major,
            minor,
            patch: 0,
        }
    }

    /// Unpacks a `uint32_t` version (bits 31-29 variant, 28-22 major, 21-12
    /// minor, 11-0 patch). `None` for a variant other than 0: Vulkan SC and
    /// other variants are not Vulkan versions these tables describe.
    #[must_use]
    pub const fn from_raw(raw: u32) -> Option<Self> {
        if raw >> 29 != 0 {
            return None;
        }
        Some(Self {
            major: (raw >> 22) & 0x7F,
            minor: (raw >> 12) & 0x3FF,
            patch: raw & 0xFFF,
        })
    }

    /// Packs it for `VkApplicationInfo::apiVersion`.
    #[must_use]
    pub const fn raw(self) -> u32 {
        (self.major << 22) | (self.minor << 12) | self.patch
    }
}

impl fmt::Display for ApiVersion {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.patch == 0 {
            write!(f, "{}.{}", self.major, self.minor)
        } else {
            write!(f, "{}.{}.{}", self.major, self.minor, self.patch)
        }
    }
}

/// The `VkApplicationInfo::apiVersion` to create the instance with.
///
/// `loader` is what `vkEnumerateInstanceVersion` returns, `None` when
/// `vkGetInstanceProcAddr` has no such function (a 1.0 loader, which fails
/// any `apiVersion` above 1.0). From 1.1 on the spec lets `apiVersion` exceed
/// both the instance and the device version: each device then works at the
/// lower of `apiVersion` and its own version ([`Offer::api`]).
#[must_use]
pub fn app_version(loader: Option<ApiVersion>, ceiling: ApiVersion) -> ApiVersion {
    match loader {
        Some(version) if version >= ApiVersion::V1_1 => ceiling,
        _ => ApiVersion::V1_0,
    }
}

/// What `vkCreateInstance` needs for the waterfall to see every device and
/// every route: the `apiVersion`, instance extensions and create flags. The
/// host adds its own (surface, debug utils).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Instance {
    /// `VkApplicationInfo::apiVersion` ([`app_version`]).
    pub api: ApiVersion,
    /// `VkInstanceCreateInfo::flags`.
    pub flags: u32,
    extensions: Vec<&'static CStr>,
}

impl Instance {
    /// `loader` as for [`app_version`]; `available` is what
    /// `vkEnumerateInstanceExtensionProperties` lists.
    ///
    /// - On a 1.0 loader, `VK_KHR_get_physical_device_properties2`, which
    ///   opens the feature structs ([`FeatureQuery::Features2Khr`]), and the
    ///   other instance extensions 1.1 made core.
    /// - `VK_KHR_portability_enumeration` with its create flag, without
    ///   which a newer loader hides `MoltenVK` and other portability drivers.
    pub fn new<'a>(
        loader: Option<ApiVersion>,
        ceiling: ApiVersion,
        available: impl IntoIterator<Item = &'a CStr>,
    ) -> Self {
        let api = app_version(loader, ceiling);
        let available: Vec<&CStr> = available.into_iter().collect();
        let mut extensions = Vec::new();
        let mut flags = 0;
        if api < ApiVersion::V1_1 {
            // What 1.1 made core on the instance side, which device
            // extensions build on (`VK_KHR_device_group` on
            // `VK_KHR_device_group_creation`, ...).
            for name in [
                features::GET_PHYSICAL_DEVICE_PROPERTIES2,
                c"VK_KHR_device_group_creation",
                c"VK_KHR_external_memory_capabilities",
                c"VK_KHR_external_semaphore_capabilities",
                c"VK_KHR_external_fence_capabilities",
            ] {
                if available.contains(&name) {
                    extensions.push(name);
                }
            }
        }
        if available.contains(&features::PORTABILITY_ENUMERATION) {
            extensions.push(features::PORTABILITY_ENUMERATION);
            flags |= features::ENUMERATE_PORTABILITY_BIT;
        }
        Self {
            api,
            flags,
            extensions,
        }
    }

    /// Adds `VK_EXT_swapchain_colorspace` when `available` lists it.
    /// Without it a surface lists sRGB non-linear only: no HDR10, HLG,
    /// scRGB or Display P3 ([`swapchain::output_format`]). It depends on
    /// `VK_KHR_surface`, which the host enables with its surface, so it is
    /// not part of [`new`](Self::new), which compute hosts use too.
    #[must_use]
    pub fn with_color_spaces<'a>(mut self, available: impl IntoIterator<Item = &'a CStr>) -> Self {
        let name = color_space::SWAPCHAIN_COLORSPACE;
        if available.into_iter().any(|e| e == name) && !self.extensions.contains(&name) {
            self.extensions.push(name);
        }
        self
    }

    /// Instance extensions to enable; pass the same to
    /// [`Offer::with_instance`].
    #[must_use]
    pub fn extensions(&self) -> &[&'static CStr] {
        &self.extensions
    }
}

/// `VkPhysicalDeviceSubgroupProperties`, core in 1.1.
pub mod subgroup {
    /// `VkSubgroupFeatureFlagBits`.
    pub mod op {
        pub const BASIC: u32 = 0x1;
        pub const VOTE: u32 = 0x2;
        pub const ARITHMETIC: u32 = 0x4;
        pub const BALLOT: u32 = 0x8;
        pub const SHUFFLE: u32 = 0x10;
        pub const SHUFFLE_RELATIVE: u32 = 0x20;
        pub const CLUSTERED: u32 = 0x40;
        pub const QUAD: u32 = 0x80;
    }

    /// `VkShaderStageFlagBits`.
    pub mod stage {
        pub const VERTEX: u32 = 0x1;
        pub const FRAGMENT: u32 = 0x10;
        pub const COMPUTE: u32 = 0x20;
    }

    /// What the device reports.
    #[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
    pub struct Subgroup {
        pub size: u32,
        /// `supportedStages`.
        pub stages: u32,
        /// `supportedOperations`.
        pub operations: u32,
    }
}

#[cfg(test)]
mod tests;
