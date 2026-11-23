//! Engine model and feature sets.

use crate::tiers::Tier;

/// A hardware engine class on a device.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum Engine {
    /// Matrix engines: XMX, Tensor Cores, MFMA/WMMA, Tensix FPU, AIE, NPUs.
    Matrix,
    /// SIMT/SIMD vector units (stage 2).
    Vector,
    /// Ray tracing units (stage 3).
    RayTracing,
    /// Video / upscaler engines (stage 3).
    Media,
    /// DMA / copy engines.
    Copy,
}

/// Device class, used by [`Placement`](crate::Placement).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum DeviceClass {
    Gpu,
    Npu,
    /// Other accelerator cards (Tenstorrent, …).
    Accel,
}

/// Comfort layers (vendor or portable user-space stacks).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum Comfort {
    Vulkan,
    Cuda,
    Hip,
    /// CIX Zhouyi NPU SDK.
    Cix,
    HailoRt,
}

/// How an engine is reached.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Access {
    /// Kernel driver UAPI only (ioctl/mmap), no vendor user space.
    Raw,
    /// Through a comfort layer, because the raw path is missing or unstable.
    Comfort(Comfort),
    /// Host emulation (mock backend, tests only).
    Host,
}

/// What a device offers, filled in by its backend's `detect`.
#[derive(Debug, Clone, PartialEq)]
pub struct FeatureSet {
    /// Human-readable name (`"Intel Arc B580"`, `"mock"`).
    pub name: String,
    pub class: DeviceClass,
    /// Driver name and version as reported by the kernel.
    pub driver: String,
    /// Reachable engines and the access layer for each.
    pub engines: Vec<(Engine, Access)>,
    /// Matrix tiers in ascending order (a linear ladder); the last one is the best.
    pub tiers: Vec<Tier>,
}

impl FeatureSet {
    /// Access layer for `engine`, if reachable.
    #[must_use]
    pub fn access(&self, engine: Engine) -> Option<Access> {
        self.engines
            .iter()
            .find(|(e, _)| *e == engine)
            .map(|&(_, a)| a)
    }

    #[must_use]
    pub fn has_tier(&self, tier: Tier) -> bool {
        self.tiers.contains(&tier)
    }
}
