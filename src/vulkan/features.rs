//! The feature bits of each core version, the structs that carry them, and the
//! extensions each version promoted to core.
//!
//! The tables are generated from the Vulkan registry by
//! `scripts/gen-vulkan.py` (`generated.rs`); this file holds what is written
//! by hand. Field names and their order follow the C declarations: a
//! [`Chain`] block is read and written by index, so the order is the layout.

use std::ffi::CStr;

use super::ApiVersion;

mod generated;

pub(crate) use generated::s_type;
pub use generated::{CORE_1_0, CORE_1_1, CORE_1_2, CORE_1_3, CORE_1_4, Chain, VERSIONS};

/// One `VkBool32` feature member.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Field {
    /// The member name as the spec spells it (`drawIndirectCount`).
    pub name: &'static str,
    /// The extension it came from before it was core, if any.
    pub from: Option<&'static CStr>,
}

const fn f(name: &'static str) -> Field {
    Field { name, from: None }
}

const fn e(name: &'static str, from: &'static CStr) -> Field {
    Field {
        name,
        from: Some(from),
    }
}

/// What one core version added.
#[derive(Debug, Clone, Copy)]
pub struct Version {
    pub version: ApiVersion,
    /// The struct holding the version's bits ([`Chain::Core10`] for 1.0).
    pub chain: Chain,
    /// Its feature bits, in declaration order.
    pub fields: &'static [Field],
    /// The extensions this version made core.
    pub promoted: &'static [&'static CStr],
}

/// `str` equality in `const` contexts.
const fn same(a: &str, b: &str) -> bool {
    let (a, b) = (a.as_bytes(), b.as_bytes());
    if a.len() != b.len() {
        return false;
    }
    let mut i = 0;
    while i < a.len() {
        if a[i] != b[i] {
            return false;
        }
        i += 1;
    }
    true
}

impl Chain {
    /// The index of member `name`. `const`, so route tables are checked
    /// while compiling.
    #[must_use]
    pub const fn index(self, name: &str) -> Option<usize> {
        let fields = self.fields();
        let mut i = 0;
        while i < fields.len() {
            if same(fields[i].name, name) {
                return Some(i);
            }
            i += 1;
        }
        None
    }

    /// The struct that carries this one's bits on a device working at `api`:
    /// itself, or the version struct that absorbed it.
    ///
    /// # Errors
    ///
    /// A version struct the device can't take yet.
    pub fn at(self, api: ApiVersion) -> Result<Chain, ApiVersion> {
        if let Some(since) = self.version_struct_since()
            && api < since
        {
            return Err(since);
        }
        Ok(match self.absorbed_by() {
            Some((since, by)) if since <= api => by,
            _ => self,
        })
    }
}

/// What an extension needs before it can be enabled: the `depends`
/// expression of `vk.xml`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Depends {
    /// The device works at this version or higher.
    Version(ApiVersion),
    /// This extension: core already, enabled on the instance, or offered by
    /// the device with its own dependencies met.
    Extension(&'static CStr),
    /// All of them (`a+b`); empty for an extension without dependencies.
    All(&'static [Depends]),
    /// One of them (`a,b`).
    Any(&'static [Depends]),
    /// Vulkan SC only.
    Never,
}

/// The dependencies of extension `name`. Known for every extension a route
/// names, and for theirs; `None` for others.
#[must_use]
pub fn depends(name: &CStr) -> Option<&'static Depends> {
    generated::DEPENDS
        .iter()
        .find(|(n, _)| *n == name)
        .map(|(_, d)| d)
}

/// Every extension [`depends`] knows.
pub fn known_extensions() -> impl Iterator<Item = &'static CStr> {
    generated::DEPENDS.iter().map(|&(n, _)| n)
}

/// The table of `version` (patch ignored), if known.
#[must_use]
pub fn of(version: ApiVersion) -> Option<&'static Version> {
    VERSIONS
        .iter()
        .find(|v| v.version.major == version.major && v.version.minor == version.minor)
}

/// The version that made extension `name` core, if one did.
#[must_use]
pub fn promoted_in(name: &CStr) -> Option<ApiVersion> {
    VERSIONS
        .iter()
        .find(|v| v.promoted.contains(&name))
        .map(|v| v.version)
}

/// Whether `name` is an instance extension: enabled at `vkCreateInstance`,
/// never at `vkCreateDevice`.
#[must_use]
pub fn is_instance_extension(name: &CStr) -> bool {
    generated::INSTANCE_EXTENSIONS.contains(&name)
}

/// `VK_KHR_get_physical_device_properties2`: on a 1.0 instance, the way to
/// `vkGetPhysicalDeviceFeatures2KHR` and to feature structs at device creation.
pub const GET_PHYSICAL_DEVICE_PROPERTIES2: &CStr = c"VK_KHR_get_physical_device_properties2";
/// `VK_KHR_portability_enumeration`: without it a 1.3.216+ loader hides
/// portability drivers such as `MoltenVK`.
pub const PORTABILITY_ENUMERATION: &CStr = c"VK_KHR_portability_enumeration";
/// `VK_INSTANCE_CREATE_ENUMERATE_PORTABILITY_BIT_KHR`, goes with
/// [`PORTABILITY_ENUMERATION`].
pub const ENUMERATE_PORTABILITY_BIT: u32 = 0x1;
/// `VK_KHR_portability_subset`: must be enabled whenever a device offers it
/// (`VUID-VkDeviceCreateInfo-pProperties-04451`).
pub const PORTABILITY_SUBSET_EXTENSION: &CStr = c"VK_KHR_portability_subset";
