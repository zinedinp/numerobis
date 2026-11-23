//! The waterfall: which route each op takes on one device, and the log of it.
//!
//! Per op the routes are tried in order. A route is *available* when the
//! device's API version reaches it (core), or when its extension is offered,
//! is not already core at that version, and its dependencies are core or
//! offered. The first available route decides: its bits are read, and if
//! they are off the op is missing. Later routes are not tried, because they
//! read the same hardware bit through another struct, and chaining both a
//! `VkPhysicalDeviceVulkan1xFeatures` and the single struct it absorbed is
//! invalid (`VUID-VkDeviceCreateInfo-pNext-02829`, `-02830`, `-06532`).

use std::ffi::{CStr, CString, c_void};
use std::fmt;

use super::ApiVersion;
use super::features::{self, Chain};
use super::ops::{Op, Route};
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

    fn bit(&self, chain: Chain, field: &str) -> bool {
        let block = self.chains.iter().position(|&c| c == chain);
        match (block, chain.index(field)) {
            (Some(block), Some(bit)) => self.blocks[block].bits[bit] != 0,
            _ => false,
        }
    }

    fn set(&mut self, chain: Chain, bit: usize) {
        let block = self.add(chain);
        self.blocks[block].bits[bit] = 1;
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

/// What the device offers before any feature bit is read.
#[derive(Debug, Clone)]
pub struct Offer {
    /// The `VkApplicationInfo::apiVersion` the instance was made with.
    pub app: ApiVersion,
    /// `VkPhysicalDeviceProperties::apiVersion`.
    pub device: ApiVersion,
    extensions: Vec<CString>,
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
        }
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

    /// Whether `name` needs no enabling: core at [`api`](Self::api).
    fn is_core(&self, name: &CStr) -> bool {
        features::promoted_in(name).is_some_and(|v| v <= self.api())
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
                if chain.is_some() && api < ApiVersion::V1_1 {
                    return Err(Skip::Version(ApiVersion::V1_1));
                }
                if let Some(&dep) = deps
                    .iter()
                    .find(|&&dep| !self.is_core(dep) && !self.offers(dep))
                {
                    return Err(Skip::Dependency(dep));
                }
            }
        }
        Ok(())
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
    /// A dependency that is neither core nor offered.
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
    /// A subgroup route, but the host never passed the properties.
    SubgroupUnread,
    /// No route was available.
    NoRoute,
}

/// One op's result.
#[derive(Debug, Clone)]
pub struct Outcome {
    pub op: Op,
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
    op: Op,
    route: Option<&'static Route>,
    skipped: Vec<(&'static Route, Skip)>,
}

/// Step 1: the routes picked from versions and extensions, and the feature
/// query they need.
pub struct Probe {
    offer: Offer,
    picks: Vec<Pick>,
    query: Blocks,
    subgroup: Option<Subgroup>,
}

impl Probe {
    /// Picks a route for each of `ops` (duplicates count once).
    pub fn new(offer: Offer, ops: &[Op]) -> Self {
        let mut query = Blocks::new();
        let mut picks: Vec<Pick> = Vec::new();
        for &op in ops {
            if picks.iter().any(|p| p.op == op) {
                continue;
            }
            let mut pick = Pick {
                op,
                route: None,
                skipped: Vec::new(),
            };
            for route in op.spec().routes {
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
        Self {
            offer,
            picks,
            query,
            subgroup: None,
        }
    }

    /// The `VkPhysicalDeviceFeatures2*` to hand to
    /// `vkGetPhysicalDeviceFeatures2` (API 1.1 and up). Valid until the probe
    /// is used again or dropped.
    pub fn features2(&mut self) -> *mut c_void {
        self.query.link(std::ptr::null_mut())
    }

    /// The `VkPhysicalDeviceFeatures*` inside it, for `vkGetPhysicalDeviceFeatures`
    /// on a 1.0 device.
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

    pub fn set_subgroup(&mut self, subgroup: Subgroup) {
        self.subgroup = Some(subgroup);
    }

    /// Step 2, after the query was filled: what each op gets.
    #[must_use]
    pub fn plan(mut self) -> Plan {
        let mut outcomes = Vec::new();
        let mut extensions: Vec<&'static CStr> = Vec::new();
        let mut enabled = Vec::new();
        let picks = std::mem::take(&mut self.picks);
        for pick in picks {
            let verdict = match pick.route {
                None => Verdict::NoRoute,
                Some(route) => self.verdict(route),
            };
            if let (Verdict::Granted, Some(route)) = (verdict, pick.route) {
                if let Route::Extension { name, deps, .. } = *route {
                    let wanted = std::iter::once(name)
                        .chain(deps.iter().copied().filter(|&d| !self.offer.is_core(d)));
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
                op: pick.op,
                route: pick.route,
                verdict,
                skipped: pick.skipped,
            });
        }
        Plan {
            app: self.offer.app,
            device: self.offer.device,
            api: self.offer.api(),
            outcomes,
            extensions,
            enabled,
        }
    }

    fn verdict(&self, route: &Route) -> Verdict {
        if let Route::Subgroup {
            operations, stages, ..
        } = *route
        {
            return match self.subgroup {
                None => Verdict::SubgroupUnread,
                Some(s)
                    if s.operations & operations == operations && s.stages & stages == stages =>
                {
                    Verdict::Granted
                }
                Some(_) => Verdict::SubgroupLacks,
            };
        }
        let Some(chain) = route_chain(route) else {
            return Verdict::Granted;
        };
        match route_fields(route)
            .iter()
            .find(|field| !self.query.bit(chain, field))
        {
            Some(field) => Verdict::Off(field),
            None => Verdict::Granted,
        }
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
    outcomes: Vec<Outcome>,
    extensions: Vec<&'static CStr>,
    enabled: Vec<(Chain, usize)>,
}

impl Plan {
    #[must_use]
    pub fn has(&self, op: Op) -> bool {
        self.outcome(op).is_some_and(Outcome::granted)
    }

    /// The route `op` was granted through: a host loads core or extension
    /// entry points by it (`vkCmdDrawIndexedIndirectCount` vs its `KHR` twin).
    #[must_use]
    pub fn route(&self, op: Op) -> Option<&'static Route> {
        self.outcome(op)
            .filter(|o| o.granted())
            .and_then(|o| o.route)
    }

    #[must_use]
    pub fn outcome(&self, op: Op) -> Option<&Outcome> {
        self.outcomes.iter().find(|o| o.op == op)
    }

    #[must_use]
    pub fn outcomes(&self) -> &[Outcome] {
        &self.outcomes
    }

    /// Device extensions to enable, dependencies included (the host adds its
    /// own, such as `VK_KHR_swapchain`).
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

    /// The feature structs for `VkDeviceCreateInfo`.
    #[must_use]
    pub fn device_chain(&self) -> DeviceChain {
        let mut blocks = Blocks::new();
        for &(chain, bit) in &self.enabled {
            blocks.set(chain, bit);
        }
        DeviceChain { blocks }
    }
}

/// The enabled feature structs, owned. Pointers from it stay valid until it
/// is used again or dropped.
pub struct DeviceChain {
    blocks: Blocks,
}

impl DeviceChain {
    /// `VkDeviceCreateInfo::pNext` (with `pEnabledFeatures` null).
    pub fn features2(&mut self) -> *mut c_void {
        self.blocks.link(std::ptr::null_mut())
    }

    /// The same, with the host's own structs (`tail`) chained behind.
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
        let spec = self.op.spec();
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
    /// Multi-line: the version, one line per op, the extensions.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(
            f,
            "Vulkan {} (instance asks {}, device has {})",
            self.api, self.app, self.device
        )?;
        for outcome in &self.outcomes {
            writeln!(f, "  {outcome}")?;
        }
        let names: Vec<_> = self
            .extensions
            .iter()
            .map(|name| name.to_string_lossy())
            .collect();
        write!(f, "  extensions: {}", names.join(", "))
    }
}
