//! Direct3D 10, 11 and 12: feature levels as Direct3D defines them, and the
//! optional features `CheckFeatureSupport` reports.
//!
//! [`Probe`] drives the queries: it knows which structs the wanted
//! capabilities read, fills the inputs (`HighestShaderModel`,
//! `HighestVersion`, `NodeIndex`) and steps down where an older runtime
//! refuses a value it doesn't know. The host only forwards each call to
//! the device. Direct3D 10 has no such queries; its capabilities are the
//! feature level's.
//!
//! - [`caps`]: the [`Capability`]s by [`Area`], each with its [`Route`]s.
//! - [`dxgi`]: the adapter (ids tied to [`pci`](crate::pci)), video memory
//!   budget, ranking.
//! - [`queue`]: Direct3D 12 command queue priority.
//! - [`cache`]: the plans kept for the process.

use std::ffi::c_void;
use std::fmt;

pub mod cache;
pub mod caps;
pub mod dxgi;
pub mod queue;

pub use super::generated::{D3d11Feature, D3d12Feature};
use super::level::{FeatureLevel, ShaderModel};
use super::{Api, Era};
use crate::platform::Os;
pub use caps::{Area, Capability, Route, Spec, Test};
pub use dxgi::{Adapter, BudgetCache, VideoMemory, rank};
pub use queue::QueuePriority;

/// `D3D12_FEATURE_DATA_FEATURE_LEVELS`, for `D3D12_FEATURE_FEATURE_LEVELS`
/// (it holds a pointer, so it is not among the generated structs).
#[repr(C)]
#[derive(Debug)]
pub struct FeatureLevels {
    pub count: u32,
    pub requested: *const u32,
    pub max_supported: u32,
}

/// `D3D12_FEATURE_FEATURE_LEVELS`.
pub const D3D12_FEATURE_FEATURE_LEVELS: u32 = 2;

impl FeatureLevels {
    /// Asks for the levels in `requested` ([`D3D12_LEVELS`]: all).
    ///
    /// # Panics
    ///
    /// With more than `u32::MAX` levels.
    #[must_use]
    pub fn new(requested: &'static [u32]) -> Self {
        Self {
            count: u32::try_from(requested.len()).expect("few levels"),
            requested: requested.as_ptr(),
            max_supported: 0,
        }
    }

    /// The highest level the device has, once filled.
    #[must_use]
    pub const fn max(&self) -> Option<FeatureLevel> {
        FeatureLevel::from_raw(self.max_supported)
    }
}

/// The graphics levels a Direct3D 12 device can be created at, raw.
pub const D3D12_LEVELS: [u32; 5] = [
    super::generated::value::D3D_FEATURE_LEVEL_11_0,
    super::generated::value::D3D_FEATURE_LEVEL_11_1,
    super::generated::value::D3D_FEATURE_LEVEL_12_0,
    super::generated::value::D3D_FEATURE_LEVEL_12_1,
    super::generated::value::D3D_FEATURE_LEVEL_12_2,
];

/// The compute-only levels, for MCDM devices (NPUs) found through DXCore.
pub const D3D12_COMPUTE_LEVELS: [u32; 2] = [
    super::generated::value::D3D_FEATURE_LEVEL_1_0_GENERIC,
    super::generated::value::D3D_FEATURE_LEVEL_1_0_CORE,
];

/// A struct the probe asks for.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
enum Source {
    D3d12(D3d12Feature),
    D3d11(D3d11Feature),
}

impl Source {
    const fn id(self) -> u32 {
        match self {
            Self::D3d12(f) => f.id(),
            Self::D3d11(f) => f.id(),
        }
    }

    const fn name(self) -> &'static str {
        match self {
            Self::D3d12(f) => f.c_name(),
            Self::D3d11(f) => f.c_name(),
        }
    }

    const fn words(self) -> usize {
        match self {
            Self::D3d12(f) => f.fields().len(),
            Self::D3d11(f) => f.fields().len(),
        }
    }
}

/// What one struct query gave.
#[derive(Debug, Clone, PartialEq, Eq)]
struct Filled {
    source: Source,
    /// `None` when the runtime refused it (it predates the struct).
    words: Option<Vec<u32>>,
}

/// The queries for one device, made after creating it.
#[derive(Debug, Clone)]
pub struct Probe {
    api: Api,
    level: FeatureLevel,
    os: Os,
    adapter: Option<Adapter>,
    wanted: Vec<Capability>,
    filled: Vec<Filled>,
    shader_model: Option<ShaderModel>,
}

impl Probe {
    /// For a device of `api` (10, 11 or 12) created at `level`, the
    /// capabilities in `caps` and what they need.
    ///
    /// # Panics
    ///
    /// When `api` is Direct3D 8 or 9.
    #[must_use]
    pub fn new(api: Api, level: FeatureLevel, caps: &[Capability]) -> Self {
        assert!(api.era() == Era::Modern, "{api} is not Direct3D 10 to 12");
        let mut wanted = Vec::new();
        for &cap in caps {
            push_with_needs(cap, &mut wanted);
        }
        Self {
            api,
            level,
            os: Os::HOST,
            adapter: None,
            wanted,
            filled: Vec::new(),
            shader_model: None,
        }
    }

    /// Plans as on `os` (what is unavailable there is skipped).
    #[must_use]
    pub fn with_os(mut self, os: Os) -> Self {
        self.os = os;
        self
    }

    /// The adapter the device is on; kept in the plan for [`rank`].
    #[must_use]
    pub fn with_adapter(mut self, adapter: Adapter) -> Self {
        self.adapter = Some(adapter);
        self
    }

    /// The structs [`fill`](Self::fill) will ask for, by C name: those the
    /// wanted capabilities read on this version, plus the shader model and
    /// root signature on Direct3D 12.
    #[must_use]
    pub fn structs(&self) -> Vec<&'static str> {
        self.sources().iter().map(|s| s.name()).collect()
    }

    fn sources(&self) -> Vec<Source> {
        let mut out = Vec::new();
        if self.api == Api::D3D12 {
            out.push(Source::D3d12(D3d12Feature::ShaderModel));
        }
        for cap in &self.wanted {
            for route in cap.spec().routes {
                let source = match *route {
                    Route::D3d12 { feature, .. } if self.api == Api::D3D12 => {
                        Source::D3d12(feature)
                    }
                    Route::D3d11 { feature, .. } if self.api == Api::D3D11 => {
                        Source::D3d11(feature)
                    }
                    _ => continue,
                };
                if !out.contains(&source) {
                    out.push(source);
                }
            }
        }
        out
    }

    /// Runs every query through `check(feature, data, size)`, which calls
    /// `CheckFeatureSupport` on the device and says whether it succeeded.
    /// A failure leaves that struct unknown (an older runtime), and what
    /// reads it is not granted.
    ///
    /// # Panics
    ///
    /// Never: the structs are a few words long.
    pub fn fill(&mut self, mut check: impl FnMut(u32, *mut c_void, u32) -> bool) {
        let mut ask = |source: Source, words: &mut Vec<u32>| {
            let size = u32::try_from(4 * words.len()).expect("small struct");
            check(source.id(), words.as_mut_ptr().cast(), size)
        };
        for source in self.sources() {
            let mut words = vec![0u32; source.words()];
            let ok = match source {
                // In/out: the newest the caller knows; a runtime fails one
                // newer than itself, so step down.
                Source::D3d12(D3d12Feature::ShaderModel) => {
                    ShaderModel::d3d12_query_order().into_iter().any(|sm| {
                        words[0] = sm;
                        ask(source, &mut words)
                    })
                }
                Source::D3d12(D3d12Feature::RootSignature) => [3, 2, 1].into_iter().any(|rs| {
                    words[0] = rs;
                    ask(source, &mut words)
                }),
                // NodeIndex 0 (the zeroed first member) for the rest.
                _ => ask(source, &mut words),
            };
            if source == Source::D3d12(D3d12Feature::ShaderModel) && ok {
                self.shader_model = Some(ShaderModel::from_raw(words[0]));
            }
            self.filled.retain(|f| f.source != source);
            self.filled.push(Filled {
                source,
                words: ok.then_some(words),
            });
        }
    }

    /// The verdicts.
    #[must_use]
    pub fn plan(self) -> Plan {
        let shader_model = self.shader_model.or_else(|| {
            // Direct3D 10 and 11 have no query: the level's model.
            (self.api != Api::D3D12)
                .then(|| self.level.shader_model())
                .flatten()
        });
        let mut outcomes: Vec<Outcome> = Vec::with_capacity(self.wanted.len());
        for &cap in &self.wanted {
            let outcome = self.judge(cap, shader_model, &outcomes);
            outcomes.push(outcome);
        }
        Plan {
            api: self.api,
            level: self.level,
            shader_model,
            adapter: self.adapter,
            outcomes,
        }
    }

    fn read(&self, source: Source, field: usize) -> Result<u32, Skip> {
        match self.filled.iter().find(|f| f.source == source) {
            Some(Filled {
                words: Some(words), ..
            }) => Ok(words[field]),
            Some(_) => Err(Skip::Unknown(source.name())),
            None => Err(Skip::NotAsked(source.name())),
        }
    }

    fn judge(&self, cap: Capability, sm: Option<ShaderModel>, done: &[Outcome]) -> Outcome {
        let spec = cap.spec();
        let mut skips = Vec::new();
        if !spec.support.available_on(self.os) {
            skips.push(Skip::Os(self.os));
            return Outcome {
                cap,
                route: None,
                skips,
            };
        }
        let mut route = None;
        for r in spec.routes {
            if !r.on(self.api) {
                continue;
            }
            let pass = match *r {
                Route::Level(level) => {
                    if self.level >= level {
                        Ok(())
                    } else {
                        Err(Skip::Level(level))
                    }
                }
                Route::Api(_) => Ok(()),
                Route::D3d12 {
                    feature,
                    field,
                    test,
                } => self.read(Source::D3d12(feature), field).and_then(|have| {
                    if test.passes(have) {
                        Ok(())
                    } else {
                        Err(Skip::Reported {
                            member: feature.fields()[field],
                            value: have,
                        })
                    }
                }),
                Route::D3d11 {
                    feature,
                    field,
                    test,
                } => self.read(Source::D3d11(feature), field).and_then(|have| {
                    if test.passes(have) {
                        Ok(())
                    } else {
                        Err(Skip::Reported {
                            member: feature.fields()[field],
                            value: have,
                        })
                    }
                }),
            };
            match pass {
                Ok(()) => {
                    route = Some(r);
                    break;
                }
                Err(skip) => skips.push(skip),
            }
        }
        if route.is_none() && skips.is_empty() {
            skips.push(Skip::NotOn(self.api));
        }
        if route.is_some()
            && self.api == Api::D3D12
            && let Some(need) = spec.shader_model
            && sm.is_none_or(|have| have < need)
        {
            skips.push(Skip::ShaderModel(need));
            route = None;
        }
        if route.is_some()
            && let Some(&n) = spec
                .needs
                .iter()
                .find(|n| !done.iter().any(|o| o.cap == **n && o.granted()))
        {
            skips.push(Skip::Needs(n));
            route = None;
        }
        Outcome { cap, route, skips }
    }
}

fn push_with_needs(cap: Capability, order: &mut Vec<Capability>) {
    if order.contains(&cap) {
        return;
    }
    for &n in cap.spec().needs {
        push_with_needs(n, order);
    }
    order.push(cap);
}

/// Why a route didn't give a capability.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Skip {
    /// No route for this version (raytracing on Direct3D 11).
    NotOn(Api),
    /// The device is below this level.
    Level(FeatureLevel),
    /// The runtime refused the struct (it predates it).
    Unknown(&'static str),
    /// [`Probe::fill`] was not run for it.
    NotAsked(&'static str),
    /// The device reported `value` for `member`.
    Reported { member: &'static str, value: u32 },
    /// The device's shader model is below this.
    ShaderModel(ShaderModel),
    /// A capability it needs is missing.
    Needs(Capability),
    /// It doesn't exist on this OS.
    Os(Os),
}

/// One capability's verdict, with why each route before the granting one
/// (or every route) failed.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Outcome {
    pub cap: Capability,
    pub route: Option<&'static Route>,
    pub skips: Vec<Skip>,
}

impl Outcome {
    #[must_use]
    pub const fn granted(&self) -> bool {
        self.route.is_some()
    }
}

/// What a Direct3D 10 to 12 device can do.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Plan {
    pub api: Api,
    pub level: FeatureLevel,
    /// Direct3D 12: what `D3D12_FEATURE_SHADER_MODEL` reported; 10 and 11:
    /// the level's.
    pub shader_model: Option<ShaderModel>,
    adapter: Option<Adapter>,
    outcomes: Vec<Outcome>,
}

impl Plan {
    #[must_use]
    pub fn has(&self, cap: Capability) -> bool {
        self.outcome(cap).is_some_and(Outcome::granted)
    }

    #[must_use]
    pub fn lacking(&self, required: &[Capability]) -> Vec<Capability> {
        required.iter().copied().filter(|&c| !self.has(c)).collect()
    }

    #[must_use]
    pub fn outcome(&self, cap: Capability) -> Option<&Outcome> {
        self.outcomes.iter().find(|o| o.cap == cap)
    }

    /// Every verdict, needs before what needs them.
    #[must_use]
    pub fn outcomes(&self) -> &[Outcome] {
        &self.outcomes
    }

    #[must_use]
    pub fn adapter(&self) -> Option<&Adapter> {
        self.adapter.as_ref()
    }
}

impl fmt::Display for Skip {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NotOn(api) => write!(f, "not in {api}"),
            Self::Level(l) => write!(f, "below {l}"),
            Self::Unknown(s) => write!(f, "runtime lacks {s}"),
            Self::NotAsked(s) => write!(f, "{s} not queried"),
            Self::Reported { member, value } => write!(f, "{member} = {value}"),
            Self::ShaderModel(sm) => write!(f, "needs shader model {sm}"),
            Self::Needs(cap) => write!(f, "needs {}", cap.name()),
            Self::Os(os) => write!(f, "not on {os}"),
        }
    }
}

impl fmt::Display for Route {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Level(l) => write!(f, "level {l}"),
            Self::Api(api) => write!(f, "{api}"),
            Self::D3d12 { feature, field, .. } => {
                write!(f, "{}.{}", feature.c_name(), feature.fields()[*field])
            }
            Self::D3d11 { feature, field, .. } => {
                write!(f, "{}.{}", feature.c_name(), feature.fields()[*field])
            }
        }
    }
}

impl fmt::Display for Plan {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{} at feature level {}", self.api, self.level)?;
        if let Some(sm) = self.shader_model {
            write!(f, ", shader model {sm}")?;
        }
        writeln!(f)?;
        let granted = self.outcomes.iter().filter(|o| o.granted()).count();
        writeln!(f, "  {granted} of {} granted", self.outcomes.len())?;
        for o in &self.outcomes {
            let spec = o.cap.spec();
            match (o.route, o.skips.last()) {
                (Some(r), _) => writeln!(f, "  + {:<16} {} ({r})", spec.area.name(), spec.name)?,
                (None, Some(why)) => {
                    writeln!(f, "  - {:<16} {} ({why})", spec.area.name(), spec.name)?;
                }
                (None, None) => writeln!(f, "  - {:<16} {}", spec.area.name(), spec.name)?,
            }
        }
        Ok(())
    }
}
