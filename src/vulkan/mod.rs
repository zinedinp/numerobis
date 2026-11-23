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
//! - [`ops`]: the capabilities (draw, memory, shader, descriptor, sync,
//!   render), each with its [`Route`]s in fallback order.
//! - [`Probe`] / [`Plan`]: the waterfall and its log.
//!
//! ```
//! use numerobis::vulkan::{ApiVersion, Offer, Op, Probe, app_version};
//!
//! // Before vkCreateInstance: what the loader reports, capped by what the
//! // tables know.
//! let app = app_version(Some(ApiVersion::V1_3), ApiVersion::NEWEST);
//! // After picking the physical device: its version and extensions.
//! let offer = Offer::new(app, ApiVersion::V1_1, [c"VK_KHR_draw_indirect_count"]);
//! let mut probe = Probe::new(offer, &[Op::DrawIndirectCount, Op::BufferDeviceAddress]);
//! let _query = probe.features2(); // hand to vkGetPhysicalDeviceFeatures2
//! let plan = probe.plan();
//! assert!(plan.has(Op::DrawIndirectCount)); // through the extension on 1.1
//! assert!(!plan.has(Op::BufferDeviceAddress));
//! assert_eq!(plan.extensions(), [c"VK_KHR_draw_indirect_count"]);
//! println!("{plan}");
//! ```

use std::fmt;

pub mod features;
pub mod ops;
mod plan;

pub use features::Chain;
pub use ops::{Area, Op, Route};
pub use plan::{DeviceChain, Offer, Outcome, Plan, Probe, Skip, Verdict};

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

    /// Unpacks a `uint32_t` version (bits 28-22 major, 21-12 minor, 11-0 patch).
    #[must_use]
    pub const fn from_raw(raw: u32) -> Self {
        Self {
            major: (raw >> 22) & 0x7F,
            minor: (raw >> 12) & 0x3FF,
            patch: raw & 0xFFF,
        }
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
