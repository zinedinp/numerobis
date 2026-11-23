//! The waterfall: which route each capability takes on one device, and the log of it.
//!
//! Per capability the routes are tried in order. A route is *available* when the
//! device's API version reaches it (core), or when its extension is offered,
//! is not already core at that version, and its dependencies are core,
//! offered by the device or enabled on the instance. The first available
//! route decides: its bits are read, and if they are off the capability is missing.
//! Later routes are not tried, because they read the same hardware bit
//! through another struct, and chaining both a
//! `VkPhysicalDeviceVulkan1xFeatures` and the single struct it absorbed is
//! invalid (`VUID-VkDeviceCreateInfo-pNext-02829`, `-02830`, `-06532`).

use std::ffi::{CStr, CString, c_void};
use std::fmt;

use super::ApiVersion;
use super::caps::{Capability, Route};
use super::features::{self, Chain};
use super::subgroup::Subgroup;

/// Room for the largest struct (`VkPhysicalDeviceFeatures`, 55 bits).
pub(crate) const BITS: usize = 64;

/// A feature struct as C lays it out: `sType`, `pNext`, `VkBool32`s. Longer
/// than any real one; drivers touch only their struct's own size.
#[repr(C)]
#[derive(Clone)]
struct Block {
    s_type: i32,
    p_next: *mut c_void,
    bits: [u32; BITS],
}

impl Block {
    fn mask(&self) -> u64 {
        self.bits
            .iter()
            .enumerate()
            .filter(|&(_, &b)| b != 0)
            .fold(0, |mask, (i, _)| mask | 1 << i)
    }
}

/// `VkPhysicalDeviceFeatures2` and the structs behind it.
#[derive(Clone)]
struct Blocks {
    chains: Vec<Chain>,
    blocks: Vec<Block>,
}

impl Blocks {
    fn new() -> Self {
        let mut blocks = Self {
            chains: Vec::new(),
            blocks: Vec::new(),
        };
        blocks.add(Chain::Core10);
        blocks
    }

    fn add(&mut self, chain: Chain) -> usize {
        if let Some(at) = self.chains.iter().position(|&c| c == chain) {
            return at;
        }
        self.chains.push(chain);
        self.blocks.push(Block {
            s_type: chain.s_type(),
            p_next: std::ptr::null_mut(),
            bits: [0; BITS],
        });
        self.blocks.len() - 1
    }

    fn set(&mut self, chain: Chain, bit: usize) {
        let block = self.add(chain);
        self.blocks[block].bits[bit] = 1;
    }

    /// Per struct, the bits that are on.
    fn masks(&self) -> Vec<(Chain, u64)> {
        self.chains
            .iter()
            .copied()
            .zip(self.blocks.iter().map(Block::mask))
            .collect()
    }

    /// Links the blocks, the last one to `tail`, and returns the head.
    fn link(&mut self, tail: *mut c_void) -> *mut c_void {
        let next: Vec<*mut c_void> = self
            .blocks
            .iter_mut()
            .skip(1)
            .map(|block| std::ptr::from_mut(block).cast())
            .chain([tail])
            .collect();
        for (block, next) in self.blocks.iter_mut().zip(next) {
            block.p_next = next;
        }
        std::ptr::from_mut(&mut self.blocks[0]).cast()
    }

    fn core_bits(&mut self) -> *mut c_void {
        self.blocks[0].bits.as_mut_ptr().cast()
    }
}

/// How the feature structs are read and handed to `vkCreateDevice`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FeatureQuery {
    /// Core `vkGetPhysicalDeviceFeatures2`; `VkDeviceCreateInfo::pNext` takes
    /// the chain (API 1.1 and up).
    Features2,
    /// `vkGetPhysicalDeviceFeatures2KHR` from
    /// `VK_KHR_get_physical_device_properties2` on a 1.0 instance; the chain
    /// goes into `VkDeviceCreateInfo::pNext` the same way.
    Features2Khr,
    /// Plain `vkGetPhysicalDeviceFeatures` and `pEnabledFeatures`: only the
    /// 1.0 bits exist.
    Features,
}

impl FeatureQuery {
    /// Whether feature structs beyond the 1.0 bits can be read and enabled.
    #[must_use]
    pub fn has_chain(self) -> bool {
        self != Self::Features
    }
}

/// Why a feature bit could not be queried or enabled by hand.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BitError {
    /// The struct has no member of that name.
    UnknownField,
    /// A `VkPhysicalDeviceVulkan1xFeatures` struct above the device's version.
    NeedsVersion(ApiVersion),
    /// Only the 1.0 bits are reachable ([`FeatureQuery::Features`]).
    NeedsFeatures2,
}

impl fmt::Display for BitError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match *self {
            Self::UnknownField => write!(f, "no such member"),
            Self::NeedsVersion(v) => write!(f, "needs {v}"),
            Self::NeedsFeatures2 => write!(
                f,
                "needs 1.1 or {}",
                features::GET_PHYSICAL_DEVICE_PROPERTIES2.to_string_lossy()
            ),
        }
    }
}

impl std::error::Error for BitError {}

/// The struct that holds `chain`'s bits on this device, if it can be reached.
fn resolve(chain: Chain, api: ApiVersion, query: FeatureQuery) -> Result<Chain, BitError> {
    if chain != Chain::Core10 && !query.has_chain() {
        return Err(BitError::NeedsFeatures2);
    }
    chain.at(api).map_err(BitError::NeedsVersion)
}

/// What the device offers before any feature bit is read.
#[derive(Debug, Clone)]
pub struct Offer {
    /// The `VkApplicationInfo::apiVersion` the instance was made with.
    pub app: ApiVersion,
    /// `VkPhysicalDeviceProperties::apiVersion`.
    pub device: ApiVersion,
    extensions: Vec<CString>,
    instance: Vec<CString>,
}

impl Offer {
    /// `extensions`: what `vkEnumerateDeviceExtensionProperties` lists.
    pub fn new<'a>(
        app: ApiVersion,
        device: ApiVersion,
        extensions: impl IntoIterator<Item = &'a CStr>,
    ) -> Self {
        Self {
            app,
            device,
            extensions: extensions.into_iter().map(CStr::to_owned).collect(),
            instance: Vec::new(),
        }
    }

    /// The instance extensions that were enabled at `vkCreateInstance`
    /// ([`Instance::extensions`](super::Instance::extensions)). They count as
    /// dependencies of device extensions, and on a 1.0 instance
    /// `VK_KHR_get_physical_device_properties2` opens the feature structs.
    #[must_use]
    pub fn with_instance<'a>(mut self, enabled: impl IntoIterator<Item = &'a CStr>) -> Self {
        self.instance = enabled.into_iter().map(CStr::to_owned).collect();
        self
    }

    /// The version the device works at: the lower of the two.
    #[must_use]
    pub fn api(&self) -> ApiVersion {
        self.app.min(self.device)
    }

    #[must_use]
    pub fn offers(&self, name: &CStr) -> bool {
        self.extensions.iter().any(|e| e.as_c_str() == name)
    }

    /// Whether instance extension `name` was enabled.
    #[must_use]
    pub fn instance_has(&self, name: &CStr) -> bool {
        self.instance.iter().any(|e| e.as_c_str() == name)
    }

    /// How feature bits are read on this device.
    #[must_use]
    pub fn query(&self) -> FeatureQuery {
        if self.api() >= ApiVersion::V1_1 {
            FeatureQuery::Features2
        } else if self.instance_has(features::GET_PHYSICAL_DEVICE_PROPERTIES2) {
            FeatureQuery::Features2Khr
        } else {
            FeatureQuery::Features
        }
    }

    /// Whether `name` needs no enabling: core at [`api`](Self::api).
    fn is_core(&self, name: &CStr) -> bool {
        features::promoted_in(name).is_some_and(|v| v <= self.api())
    }

    /// Whether dependency `name` is met: core, enabled on the instance, or
    /// offered by the device.
    fn has_dependency(&self, name: &CStr) -> bool {
        if self.is_core(name) {
            return true;
        }
        if features::is_instance_extension(name) {
            self.instance_has(name)
        } else {
            self.offers(name)
        }
    }

    fn check(&self, route: &Route) -> Result<(), Skip> {
        let api = self.api();
        match *route {
            Route::Core { since, .. } | Route::Subgroup { since, .. } => {
                if api < since {
                    return Err(Skip::Version(since));
                }
            }
            Route::Extension {
                name, deps, chain, ..
            } => {
                if let Some(core) = features::promoted_in(name).filter(|&v| v <= api) {
                    return Err(Skip::Promoted(core));
                }
                if !self.offers(name) {
                    return Err(Skip::NotOffered(name));
                }
                // Reading an extension's struct takes vkGetPhysicalDeviceFeatures2.
                if chain.is_some() && !self.query().has_chain() {
                    return Err(Skip::Dependency(features::GET_PHYSICAL_DEVICE_PROPERTIES2));
                }
                if let Some(&dep) = deps.iter().find(|&&dep| !self.has_dependency(dep)) {
                    return Err(Skip::Dependency(dep));
                }
            }
        }
        Ok(())
    }

    /// Whether the device offers `VK_KHR_portability_subset`, which then has
    /// to be enabled.
    fn portability(&self) -> bool {
        self.offers(features::PORTABILITY_SUBSET_EXTENSION)
    }
}

/// Why a route was passed over.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Skip {
    /// Needs this API version; the device works at a lower one.
    Version(ApiVersion),
    /// The extension is core from this version on: an earlier core route
    /// stands for it.
    Promoted(ApiVersion),
    NotOffered(&'static CStr),
    /// A dependency that is neither core nor offered (device extensions) or
    /// enabled (instance extensions).
    Dependency(&'static CStr),
}

/// How the chosen route turned out.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Verdict {
    Granted,
    /// The route was there, but this bit is off.
    Off(&'static str),
    /// The subgroup properties lack the operations or stages.
    SubgroupLacks,
    /// A subgroup route, but the subgroup properties were never read.
    SubgroupUnread,
    /// No route was available.
    NoRoute,
}

/// One capability's result.
#[derive(Debug, Clone)]
pub struct Outcome {
    pub cap: Capability,
    /// The route that decided, if one was available.
    pub route: Option<&'static Route>,
    pub verdict: Verdict,
    /// The routes passed over before it, and why.
    pub skipped: Vec<(&'static Route, Skip)>,
}

impl Outcome {
    #[must_use]
    pub fn granted(&self) -> bool {
        self.verdict == Verdict::Granted
    }
}

struct Pick {
    cap: Capability,
    route: Option<&'static Route>,
    skipped: Vec<(&'static Route, Skip)>,
}

/// `VkPhysicalDeviceSubgroupProperties`.
#[repr(C)]
struct SubgroupProperties {
    s_type: i32,
    p_next: *mut c_void,
    size: u32,
    stages: u32,
    operations: u32,
    quad_operations_in_all_stages: u32,
}

/// `VkPhysicalDeviceProperties2`, with room to spare for the 824 bytes of
/// `VkPhysicalDeviceProperties` (drivers write only their struct's size).
#[repr(C)]
struct Properties2 {
    s_type: i32,
    p_next: *mut c_void,
    properties: [u64; 128],
}

/// The properties query, boxed so the chain's addresses stay put.
#[repr(C)]
struct PropertiesQuery {
    head: Properties2,
    subgroup: SubgroupProperties,
}

const PROPERTIES2_S_TYPE: i32 = 1_000_059_001;
const SUBGROUP_PROPERTIES_S_TYPE: i32 = 1_000_094_000;

/// Step 1: the routes picked from versions and extensions, and the feature
/// query they need.
pub struct Probe {
    offer: Offer,
    picks: Vec<Pick>,
    query: Blocks,
    subgroup: Option<Subgroup>,
    properties: Option<Box<PropertiesQuery>>,
}

impl Probe {
    /// Picks a route for each of `caps` (duplicates count once).
    pub fn new(offer: Offer, caps: &[Capability]) -> Self {
        let mut query = Blocks::new();
        let mut picks: Vec<Pick> = Vec::new();
        for &cap in caps {
            if picks.iter().any(|p| p.cap == cap) {
                continue;
            }
            let mut pick = Pick {
                cap,
                route: None,
                skipped: Vec::new(),
            };
            for route in cap.spec().routes {
                match offer.check(route) {
                    Ok(()) => {
                        pick.route = Some(route);
                        break;
                    }
                    Err(skip) => pick.skipped.push((route, skip)),
                }
            }
            if let Some(chain) = pick.route.and_then(route_chain) {
                query.add(chain);
            }
            picks.push(pick);
        }
        if offer.portability() && offer.query().has_chain() {
            query.add(Chain::PortabilitySubset);
        }
        Self {
            offer,
            picks,
            query,
            subgroup: None,
            properties: None,
        }
    }

    /// How to read the features: which entry point gets
    /// [`features2`](Self::features2) or [`core_features`](Self::core_features).
    #[must_use]
    pub fn query(&self) -> FeatureQuery {
        self.offer.query()
    }

    /// Also reads `chain`, for bits no [`Capability`] covers; ask
    /// [`Plan::supported`] afterwards. A single struct that a version struct
    /// absorbed at this device's version is read through that one.
    ///
    /// # Errors
    ///
    /// The struct can't be read on this device.
    pub fn also_query(&mut self, chain: Chain) -> Result<(), BitError> {
        let chain = resolve(chain, self.offer.api(), self.query())?;
        self.query.add(chain);
        Ok(())
    }

    /// The `VkPhysicalDeviceFeatures2*` to hand to
    /// `vkGetPhysicalDeviceFeatures2` ([`FeatureQuery::Features2`] or
    /// [`Features2Khr`](FeatureQuery::Features2Khr)). Valid until the probe
    /// is used again or dropped.
    pub fn features2(&mut self) -> *mut c_void {
        self.query.link(std::ptr::null_mut())
    }

    /// The `VkPhysicalDeviceFeatures*` inside it, for `vkGetPhysicalDeviceFeatures`
    /// on a 1.0 device ([`FeatureQuery::Features`]).
    pub fn core_features(&mut self) -> *mut c_void {
        self.query.core_bits()
    }

    /// Whether a picked route reads `VkPhysicalDeviceSubgroupProperties`.
    #[must_use]
    pub fn wants_subgroup(&self) -> bool {
        self.picks
            .iter()
            .any(|p| matches!(p.route, Some(Route::Subgroup { .. })))
    }

    /// A `VkPhysicalDeviceProperties2*` for `vkGetPhysicalDeviceProperties2`
    /// (API 1.1 and up), with `VkPhysicalDeviceSubgroupProperties` behind it.
    /// [`plan`](Self::plan) reads the subgroup properties from it, so
    /// [`set_subgroup`](Self::set_subgroup) is not needed. Valid until the
    /// probe is used again or dropped.
    pub fn properties2(&mut self) -> *mut c_void {
        let props = self.properties.get_or_insert_with(|| {
            Box::new(PropertiesQuery {
                head: Properties2 {
                    s_type: PROPERTIES2_S_TYPE,
                    p_next: std::ptr::null_mut(),
                    properties: [0; 128],
                },
                subgroup: SubgroupProperties {
                    s_type: SUBGROUP_PROPERTIES_S_TYPE,
                    p_next: std::ptr::null_mut(),
                    size: 0,
                    stages: 0,
                    operations: 0,
                    quad_operations_in_all_stages: 0,
                },
            })
        });
        props.head.p_next = std::ptr::from_mut(&mut props.subgroup).cast();
        std::ptr::from_mut(&mut props.head).cast()
    }

    /// The subgroup properties, when the host read them itself.
    pub fn set_subgroup(&mut self, subgroup: Subgroup) {
        self.subgroup = Some(subgroup);
    }

    /// What [`properties2`](Self::properties2) brought back. `subgroupSize`
    /// is at least 1 on every device, so 0 means the query never ran.
    fn subgroup(&self) -> Option<Subgroup> {
        self.subgroup.or_else(|| {
            let s = &self.properties.as_ref()?.subgroup;
            (s.size != 0).then_some(Subgroup {
                size: s.size,
                stages: s.stages,
                operations: s.operations,
            })
        })
    }

    /// Step 2, after the query was filled: what each capability gets.
    #[must_use]
    pub fn plan(mut self) -> Plan {
        let query = self.query();
        let subgroup = self.subgroup();
        let supported = self.query.masks();
        let mut outcomes = Vec::new();
        let mut extensions: Vec<&'static CStr> = Vec::new();
        if self.offer.portability() {
            extensions.push(features::PORTABILITY_SUBSET_EXTENSION);
        }
        let mut enabled = Vec::new();
        let picks = std::mem::take(&mut self.picks);
        for pick in picks {
            let verdict = match pick.route {
                None => Verdict::NoRoute,
                Some(route) => verdict(route, &supported, subgroup),
            };
            if let (Verdict::Granted, Some(route)) = (verdict, pick.route) {
                if let Route::Extension { name, deps, .. } = *route {
                    let wanted = std::iter::once(name).chain(deps.iter().copied().filter(|&d| {
                        !self.offer.is_core(d) && !features::is_instance_extension(d)
                    }));
                    for name in wanted {
                        if !extensions.contains(&name) {
                            extensions.push(name);
                        }
                    }
                }
                if let Some(chain) = route_chain(route) {
                    for field in route_fields(route) {
                        let bit = chain.index(field).expect("route fields are in their chain");
                        if !enabled.contains(&(chain, bit)) {
                            enabled.push((chain, bit));
                        }
                    }
                }
            }
            outcomes.push(Outcome {
                cap: pick.cap,
                route: pick.route,
                verdict,
                skipped: pick.skipped,
            });
        }
        Plan {
            app: self.offer.app,
            device: self.offer.device,
            api: self.offer.api(),
            query,
            outcomes,
            extensions,
            enabled,
            supported,
            subgroup,
            portability: self.offer.portability(),
        }
    }
}

fn mask_of(supported: &[(Chain, u64)], chain: Chain) -> Option<u64> {
    supported
        .iter()
        .find(|&&(c, _)| c == chain)
        .map(|&(_, mask)| mask)
}

fn verdict(route: &Route, supported: &[(Chain, u64)], subgroup: Option<Subgroup>) -> Verdict {
    if let Route::Subgroup {
        operations, stages, ..
    } = *route
    {
        return match subgroup {
            None => Verdict::SubgroupUnread,
            Some(s) if s.operations & operations == operations && s.stages & stages == stages => {
                Verdict::Granted
            }
            Some(_) => Verdict::SubgroupLacks,
        };
    }
    let Some(chain) = route_chain(route) else {
        return Verdict::Granted;
    };
    let mask = mask_of(supported, chain).unwrap_or(0);
    match route_fields(route)
        .iter()
        .find(|field| chain.index(field).is_none_or(|bit| mask & (1 << bit) == 0))
    {
        Some(field) => Verdict::Off(field),
        None => Verdict::Granted,
    }
}

fn route_chain(route: &Route) -> Option<Chain> {
    match *route {
        Route::Core { chain, .. } => Some(chain),
        Route::Extension { chain, .. } => chain,
        Route::Subgroup { .. } => None,
    }
}

fn route_fields(route: &Route) -> &'static [&'static str] {
    match *route {
        Route::Core { fields, .. } | Route::Extension { fields, .. } => fields,
        Route::Subgroup { .. } => &[],
    }
}

/// What the device gets: the version, the routes taken, the extensions and
/// feature bits to enable at `vkCreateDevice`.
#[derive(Debug, Clone)]
pub struct Plan {
    pub app: ApiVersion,
    pub device: ApiVersion,
    /// The version the device works at.
    pub api: ApiVersion,
    /// How the features are passed to `vkCreateDevice`.
    pub query: FeatureQuery,
    outcomes: Vec<Outcome>,
    extensions: Vec<&'static CStr>,
    enabled: Vec<(Chain, usize)>,
    /// Per queried struct, the bits the device reported.
    supported: Vec<(Chain, u64)>,
    subgroup: Option<Subgroup>,
    portability: bool,
}

impl Plan {
    #[must_use]
    pub fn has(&self, cap: Capability) -> bool {
        self.outcome(cap).is_some_and(Outcome::granted)
    }

    /// The capabilities of `required` that are not granted (asked for or not): empty
    /// when the device can be used.
    #[must_use]
    pub fn lacking(&self, required: &[Capability]) -> Vec<Capability> {
        required
            .iter()
            .copied()
            .filter(|&cap| !self.has(cap))
            .collect()
    }

    /// The route `cap` was granted through: a host loads core or extension
    /// entry points by it (`vkCmdDrawIndexedIndirectCount` vs its `KHR` twin).
    #[must_use]
    pub fn route(&self, cap: Capability) -> Option<&'static Route> {
        self.outcome(cap)
            .filter(|o| o.granted())
            .and_then(|o| o.route)
    }

    #[must_use]
    pub fn outcome(&self, cap: Capability) -> Option<&Outcome> {
        self.outcomes.iter().find(|o| o.cap == cap)
    }

    #[must_use]
    pub fn outcomes(&self) -> &[Outcome] {
        &self.outcomes
    }

    /// Device extensions to enable, dependencies included (the host adds its
    /// own, such as `VK_KHR_swapchain`). Instance extensions never show up
    /// here; `VK_KHR_portability_subset` always does when offered.
    #[must_use]
    pub fn extensions(&self) -> &[&'static CStr] {
        &self.extensions
    }

    /// The feature bits to turn on, by struct and member.
    pub fn enabled(&self) -> impl Iterator<Item = (Chain, &'static str)> + '_ {
        self.enabled
            .iter()
            .map(|&(chain, bit)| (chain, chain.fields()[bit].name))
    }

    /// Whether the device reported bit `field` of `chain`: `None` when that
    /// struct was not read (see [`Probe::also_query`]) or has no such member.
    /// The 1.0 bits are always read.
    #[must_use]
    pub fn supported(&self, chain: Chain, field: &str) -> Option<bool> {
        let chain = resolve(chain, self.api, self.query).ok()?;
        let bit = chain.index(field)?;
        mask_of(&self.supported, chain).map(|mask| mask & (1 << bit) != 0)
    }

    /// The subgroup properties, if they were read.
    #[must_use]
    pub fn subgroup(&self) -> Option<Subgroup> {
        self.subgroup
    }

    /// Whether the device is a portability implementation
    /// (`VK_KHR_portability_subset`, already in [`extensions`](Self::extensions)).
    #[must_use]
    pub fn is_portability_subset(&self) -> bool {
        self.portability
    }

    /// On a portability implementation, the parts of full Vulkan it reported
    /// as missing (`VkPhysicalDevicePortabilitySubsetFeaturesKHR` bits that
    /// are off). Bits it has must still be enabled before use, through
    /// [`DeviceChain::enable`].
    #[must_use]
    pub fn portability_lacks(&self) -> Vec<&'static str> {
        let Some(mask) = mask_of(&self.supported, Chain::PortabilitySubset) else {
            return Vec::new();
        };
        Chain::PortabilitySubset
            .fields()
            .iter()
            .enumerate()
            .filter(|&(i, _)| mask & (1 << i) == 0)
            .map(|(_, field)| field.name)
            .collect()
    }

    /// The feature structs for `VkDeviceCreateInfo`.
    #[must_use]
    pub fn device_chain(&self) -> DeviceChain {
        let mut blocks = Blocks::new();
        for &(chain, bit) in &self.enabled {
            blocks.set(chain, bit);
        }
        DeviceChain {
            blocks,
            api: self.api,
            query: self.query,
        }
    }
}

/// The enabled feature structs, owned. Pointers from it stay valid until it
/// is used again or dropped.
pub struct DeviceChain {
    blocks: Blocks,
    api: ApiVersion,
    query: FeatureQuery,
}

impl DeviceChain {
    /// Turns on one more bit, for features no [`Capability`] covers. A single struct
    /// that a version struct absorbed at this device's version is set
    /// through that one, so the chain stays valid. Whether the device has the
    /// bit is the host's to check ([`Plan::supported`]).
    ///
    /// # Errors
    ///
    /// An unknown member, or a struct this device can't take.
    pub fn enable(&mut self, chain: Chain, field: &str) -> Result<(), BitError> {
        let chain = resolve(chain, self.api, self.query)?;
        let bit = chain.index(field).ok_or(BitError::UnknownField)?;
        self.blocks.set(chain, bit);
        Ok(())
    }

    /// `VkDeviceCreateInfo::pNext` (with `pEnabledFeatures` null).
    pub fn features2(&mut self) -> *mut c_void {
        self.blocks.link(std::ptr::null_mut())
    }

    /// The same, with the host's own structs (`tail`) chained behind. `tail`
    /// must not hold feature structs: add bits with [`enable`](Self::enable)
    /// instead, which keeps a single struct and the version struct that
    /// absorbed it out of the same chain.
    pub fn features2_then(&mut self, tail: *mut c_void) -> *mut c_void {
        self.blocks.link(tail)
    }

    /// `VkDeviceCreateInfo::pEnabledFeatures` on a 1.0 device, which can't
    /// take a `pNext` feature chain.
    pub fn core_features(&mut self) -> *const c_void {
        self.blocks.core_bits()
    }
}

struct ShowRoute<'a>(&'a Route);

impl fmt::Display for ShowRoute<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match *self.0 {
            Route::Core {
                since,
                chain,
                fields,
            } => write!(f, "core {since} ({}.{})", chain.name(), fields.join("+")),
            Route::Extension {
                name,
                chain: Some(chain),
                fields,
                ..
            } => write!(
                f,
                "{} ({}.{})",
                name.to_string_lossy(),
                chain.name(),
                fields.join("+")
            ),
            Route::Extension { name, .. } => write!(f, "{}", name.to_string_lossy()),
            Route::Subgroup {
                since,
                operations,
                stages,
            } => write!(
                f,
                "core {since} subgroup ops {operations:#x} in stages {stages:#x}"
            ),
        }
    }
}

impl fmt::Display for Skip {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match *self {
            Self::Version(v) => write!(f, "needs {v}"),
            Self::Promoted(v) => write!(f, "core since {v}"),
            Self::NotOffered(_) => write!(f, "not offered"),
            Self::Dependency(dep) => write!(f, "needs {}", dep.to_string_lossy()),
        }
    }
}

impl fmt::Display for Outcome {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let spec = self.cap.spec();
        let sign = if self.granted() { '+' } else { '-' };
        write!(f, "{sign} {}.{}: ", spec.area.name(), spec.name)?;
        match (self.route, self.verdict) {
            (Some(route), Verdict::Granted) => write!(f, "{}", ShowRoute(route))?,
            (Some(route), Verdict::Off(field)) => {
                write!(f, "{field} is off ({})", ShowRoute(route))?;
            }
            (Some(route), Verdict::SubgroupLacks) => {
                write!(f, "not supported ({})", ShowRoute(route))?;
            }
            (Some(route), Verdict::SubgroupUnread) => {
                write!(f, "subgroup properties not read ({})", ShowRoute(route))?;
            }
            _ => write!(f, "no route")?,
        }
        for (i, (route, skip)) in self.skipped.iter().enumerate() {
            let lead = if i == 0 { " [skipped " } else { "; " };
            write!(f, "{lead}{}: {skip}", ShowRoute(route))?;
        }
        if !self.skipped.is_empty() {
            write!(f, "]")?;
        }
        Ok(())
    }
}

impl fmt::Display for Plan {
    /// Multi-line: the version, one line per capability, the extensions.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "Vulkan {} (instance asks {}, device has {})",
            self.api, self.app, self.device
        )?;
        if self.query == FeatureQuery::Features2Khr {
            write!(
                f,
                ", features through VK_KHR_get_physical_device_properties2"
            )?;
        }
        writeln!(f)?;
        for outcome in &self.outcomes {
            writeln!(f, "  {outcome}")?;
        }
        if self.portability {
            let lacks = self.portability_lacks();
            if lacks.is_empty() {
                writeln!(f, "  portability subset")?;
            } else {
                writeln!(f, "  portability subset, lacks {}", lacks.join(", "))?;
            }
        }
        let names: Vec<_> = self
            .extensions
            .iter()
            .map(|name| name.to_string_lossy())
            .collect();
        write!(f, "  extensions: {}", names.join(", "))
    }
}
