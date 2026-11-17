//! Matrix tiers per target (plan B.2, B.2b, B.2c).
//!
//! Each target has a linear ladder; the derived `Ord` within one target
//! enum follows that ladder. A device's [`FeatureSet::tiers`](crate::FeatureSet)
//! lists the tiers it supports in ascending order, always ending above
//! [`Tier::Emulated`], which every backend can run.

/// Intel GPU (xe).
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Intel {
    /// DP4A only.
    XeLp,
    /// XMX F16/BF16/I8, SIMD8.
    XeHpg,
    /// XMX SIMD16.
    XeHpc,
    /// XMX SIMD16, FP8 when the device query reports it.
    Xe2,
}

/// AMD GPU (KFD). CDNA and RDNA are separate ladders.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Amd {
    Gfx908,
    Gfx90a,
    /// FP8 FNUZ.
    Gfx942,
    /// FP8 OCP, MXFP4/6.
    Gfx950,
    /// RDNA3 WMMA.
    Gfx11,
    /// RDNA4 WMMA FP8.
    Gfx12,
}

/// NVIDIA GPU. `Sm120` is a side branch next to `Sm100`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Nvidia {
    /// `mma.sync`.
    Sm80,
    /// FP8.
    Sm89,
    /// `wgmma` (comfort first).
    Sm90,
    /// `tcgen05`, MX, NVFP4 (comfort first).
    Sm100,
    Sm120,
}

/// Tenstorrent.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Tt {
    Wormhole,
    Blackhole,
}

/// AMD XDNA (AIE).
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Xdna {
    Aie2,
    /// BFP16.
    Aie2p,
}

/// Intel NPU (ivpu).
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum IntelNpu {
    Npu3,
    Npu4,
    Npu5,
}

/// Arm Ethos-U.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum EthosU {
    U55,
    U65,
    U85,
}

/// Broadcom VideoCore (Raspberry Pi), Emulated matrix only.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum V3d {
    /// Pi 4 / CM4.
    V3d42,
    /// Pi 5 / CM5.
    V3d71,
}

/// A matrix tier.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum Tier {
    /// Integer bit-logic decode + F32 vector FMA. Runs on every layer.
    Emulated,
    Intel(Intel),
    Amd(Amd),
    Nvidia(Nvidia),
    Tt(Tt),
    Xdna(Xdna),
    IntelNpu(IntelNpu),
    /// Rockchip RK3588.
    Rknpu,
    EthosU(EthosU),
    /// CIX Zhouyi X2.
    Zhouyi,
    V3d(V3d),
}

impl Tier {
    /// Short name used in kernel binary paths (`kernels/bin/<target>/<variant>`).
    pub fn name(self) -> String {
        match self {
            Tier::Emulated => "emulated".into(),
            Tier::Intel(t) => format!("intel-{t:?}").to_ascii_lowercase(),
            Tier::Amd(t) => format!("amd-{t:?}").to_ascii_lowercase(),
            Tier::Nvidia(t) => format!("nvidia-{t:?}").to_ascii_lowercase(),
            Tier::Tt(t) => format!("tt-{t:?}").to_ascii_lowercase(),
            Tier::Xdna(t) => format!("xdna-{t:?}").to_ascii_lowercase(),
            Tier::IntelNpu(t) => format!("intel-npu-{t:?}").to_ascii_lowercase(),
            Tier::Rknpu => "rknpu-rk3588".into(),
            Tier::EthosU(t) => format!("ethosu-{t:?}").to_ascii_lowercase(),
            Tier::Zhouyi => "zhouyi-x2".into(),
            Tier::V3d(t) => format!("v3d-{t:?}").to_ascii_lowercase(),
        }
    }
}
