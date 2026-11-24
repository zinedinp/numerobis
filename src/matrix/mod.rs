//! Matrix engines, one model for every vendor: Intel XMX, NVIDIA Tensor
//! Cores, AMD Matrix Cores (CDNA, `MFMA`) and AI Accelerators (RDNA,
//! `WMMA`), and the Tenstorrent Tensix FPU.
//!
//! They all do the same thing, `D = A·B + C` on small tiles (A `M×K`, B
//! `K×N`, C and D `M×N`) with narrow inputs and a wide accumulator. They
//! differ in the tile [`Shape`], in how many lanes share one instruction
//! ([`Scope`]) and in the formats they take. An [`Mma`] is one such
//! instruction; each [`Engine`] lists its own ([`Engine::mmas`]),
//! generation by generation, from the vendors' ISA documents.
//!
//! Kept apart from the rest of the crate: conversions, activations and the
//! [`Tier::Emulated`] kernels run on the vector units and have no engine
//! here. `gemm` (and convolution, an implicit GEMM) is what a matrix engine
//! runs; [`Engine::route`] says how a `gemm_<a>_<b>_<acc>` gets onto one:
//! natively, after an exact widening of its inputs (`E4M3` to `F16`, `I4` to
//! `I8`, ...), or not at all ([`Route::Vector`]).
//!
//! ```
//! use numerobis::elem::{E4M3Fn, Elem, F16, F32};
//! use numerobis::matrix::{Engine, Route, intel::Xmx};
//! use numerobis::pci::Gpu;
//!
//! let b580 = Engine::from_gpu(Gpu::from_pci(0x8086, 0xE20B)).expect("XMX");
//! assert_eq!(b580, Engine::Xmx(Xmx::Xe2));
//! assert_eq!(b580.to_string(), "Xe2 XMX");
//! // FP8 has no XMX instruction on Xe2; widened to F16 (exact) it has one.
//! match b580.route(E4M3Fn::FORMAT, E4M3Fn::FORMAT, F32::FORMAT) {
//!     Route::Widened { a, mma, .. } => {
//!         assert_eq!(a, F16::FORMAT);
//!         assert_eq!(mma.shape.to_string(), "8x16x16");
//!     }
//!     other => panic!("{other:?}"),
//! }
//! ```
//!
//! What the engine tables say is what the hardware does. A driver may
//! expose less: the B580's Vulkan driver reports four of the Xe2 table's
//! nine type combinations as cooperative matrices (no bf16, int4 or mixed
//! signedness). `matrix::vulkan` (feature `vulkan`) reads the driver's list
//! and sets it next to the table.
//!
//! Not here: NPUs (they run compiled command streams, no instruction-level
//! tiles to name), and `f64` (`DMMA`, `v_mfma_f64`: no 64-bit format in
//! [`elem`](crate::elem)).

use std::fmt;

use crate::elem::Format;
use crate::pci::{Architecture, Gpu};
use crate::platform::Support;
use crate::tiers::Tier;

pub mod amd;
pub mod intel;
pub mod nvidia;
pub mod tt;
#[cfg(feature = "vulkan")]
pub mod vulkan;

#[cfg(test)]
mod tests;

/// The formats the tables use, by their short names.
pub(crate) mod f {
    use crate::elem::Format;
    use miraculix::formats::MiniFloat;
    use miraculix::formats::bfp::BfpElem;
    use miraculix::formats::mx::MxElem;

    pub const F32: Format = Format::F32;
    pub const TF32: Format = Format::Tf32;
    pub const F16: Format = Format::Float(MiniFloat::F16);
    pub const BF16: Format = Format::Float(MiniFloat::Bf16);
    pub const E4M3: Format = Format::Float(MiniFloat::E4M3Fn);
    pub const E5M2: Format = Format::Float(MiniFloat::E5M2);
    pub const E4M3FNUZ: Format = Format::Float(MiniFloat::E4M3Fnuz);
    pub const E5M2FNUZ: Format = Format::Float(MiniFloat::E5M2Fnuz);
    pub const E2M3: Format = Format::Float(MiniFloat::E2M3);
    pub const E3M2: Format = Format::Float(MiniFloat::E3M2);
    pub const E2M1: Format = Format::Float(MiniFloat::E2M1);
    pub const I32: Format = Format::I32;
    pub const I8: Format = Format::I8;
    pub const U8: Format = Format::U8;
    pub const I4: Format = Format::Int {
        bits: 4,
        signed: true,
    };
    pub const U4: Format = Format::Int {
        bits: 4,
        signed: false,
    };
    pub const MXFP8E4M3: Format = Format::Mx(MxElem::Fp8E4M3);
    pub const MXFP8E5M2: Format = Format::Mx(MxElem::Fp8E5M2);
    pub const MXFP6E2M3: Format = Format::Mx(MxElem::Fp6E2M3);
    pub const MXFP6E3M2: Format = Format::Mx(MxElem::Fp6E3M2);
    pub const MXFP4: Format = Format::Mx(MxElem::Fp4);
    pub const NVFP4: Format = Format::Nvfp4;
    pub const BFP8: Format = Format::Bfp(BfpElem::M7);
    pub const BFP4: Format = Format::Bfp(BfpElem::M3);
    pub const BFP2: Format = Format::Bfp(BfpElem::M1);
}

/// One instruction's tile: A is `m×k`, B `k×n`, C and D `m×n`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Shape {
    pub m: u32,
    pub n: u32,
    pub k: u32,
}

impl Shape {
    #[must_use]
    pub const fn new(m: u32, n: u32, k: u32) -> Self {
        Self { m, n, k }
    }

    /// Multiply-adds per instruction.
    #[must_use]
    pub const fn macs(self) -> u64 {
        self.m as u64 * self.n as u64 * self.k as u64
    }

    /// Whether a tile of `self` is made of whole `unit` tiles (a driver's
    /// cooperative matrix over several instructions).
    #[must_use]
    pub const fn is_made_of(self, unit: Shape) -> bool {
        unit.m > 0
            && unit.n > 0
            && unit.k > 0
            && self.m.is_multiple_of(unit.m)
            && self.n.is_multiple_of(unit.n)
            && self.k.is_multiple_of(unit.k)
    }
}

impl fmt::Display for Shape {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}x{}x{}", self.m, self.n, self.k)
    }
}

/// Who issues one instruction and holds its tiles.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Scope {
    /// A subgroup (Intel), warp (NVIDIA) or wave (AMD) of `lanes` lanes,
    /// the tiles spread over their registers.
    Subgroup { lanes: u32 },
    /// A Hopper warpgroup: four warps, 128 lanes (`wgmma`), B (and A) read
    /// from shared memory.
    Warpgroup,
    /// A thread block (`pair`: two, on neighbouring SMs): Blackwell's
    /// `tcgen05`, issued by one thread, the accumulator in tensor memory.
    Cta { pair: bool },
    /// One Tensix core: its FPU, fed by the unpackers from L1.
    Core,
}

impl fmt::Display for Scope {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Subgroup { lanes } => write!(f, "subgroup of {lanes}"),
            Self::Warpgroup => f.write_str("warpgroup of 128"),
            Self::Cta { pair: false } => f.write_str("thread block"),
            Self::Cta { pair: true } => f.write_str("thread block pair"),
            Self::Core => f.write_str("Tensix core"),
        }
    }
}

/// One matrix instruction: `D (acc) = A (a) · B (b) + C (acc)` on a
/// [`Shape`]. Block-scaled inputs (MX, NVFP4, `BFP_B`) carry their scales
/// along K, so `shape.k` is a whole number of blocks.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Mma {
    pub a: Format,
    pub b: Format,
    pub acc: Format,
    pub shape: Shape,
    pub scope: Scope,
    /// The vendor's instruction (`dpas`, `mma.sync.aligned.m16n8k16`,
    /// `v_wmma_f32_16x16x16_f16`, ...).
    pub instr: &'static str,
}

impl Mma {
    #[must_use]
    pub fn takes(&self, a: Format, b: Format, acc: Format) -> bool {
        self.a == a && self.b == b && self.acc == acc
    }
}

impl fmt::Display for Mma {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{} x {} -> {} {} ({}, {})",
            self.a.name(),
            self.b.name(),
            self.acc.name(),
            self.shape,
            self.scope,
            self.instr
        )
    }
}

/// `Mma` table rows: `a, b -> acc, m n k, scope, "instruction"`.
macro_rules! mmas {
    ($($a:ident, $b:ident -> $acc:ident, $m:literal $n:literal $k:literal, $scope:expr, $instr:literal;)*) => {
        &[$($crate::matrix::Mma {
            a: $crate::matrix::f::$a,
            b: $crate::matrix::f::$b,
            acc: $crate::matrix::f::$acc,
            shape: $crate::matrix::Shape::new($m, $n, $k),
            scope: $scope,
            instr: $instr,
        }),*]
    };
}
pub(crate) use mmas;

/// A matrix engine generation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Engine {
    Xmx(intel::Xmx),
    TensorCore(nvidia::TensorCore),
    MatrixCore(amd::MatrixCore),
    Tensix(tt::Tensix),
}

impl Engine {
    pub const ALL: [Self; 18] = [
        Self::Xmx(intel::Xmx::XeHpg),
        Self::Xmx(intel::Xmx::XeHpc),
        Self::Xmx(intel::Xmx::Xe2),
        Self::TensorCore(nvidia::TensorCore::Volta),
        Self::TensorCore(nvidia::TensorCore::Turing),
        Self::TensorCore(nvidia::TensorCore::Ampere),
        Self::TensorCore(nvidia::TensorCore::Ada),
        Self::TensorCore(nvidia::TensorCore::Hopper),
        Self::TensorCore(nvidia::TensorCore::Blackwell),
        Self::TensorCore(nvidia::TensorCore::BlackwellGeForce),
        Self::MatrixCore(amd::MatrixCore::Cdna1),
        Self::MatrixCore(amd::MatrixCore::Cdna2),
        Self::MatrixCore(amd::MatrixCore::Cdna3),
        Self::MatrixCore(amd::MatrixCore::Cdna4),
        Self::MatrixCore(amd::MatrixCore::Rdna3),
        Self::MatrixCore(amd::MatrixCore::Rdna4),
        Self::Tensix(tt::Tensix::Wormhole),
        Self::Tensix(tt::Tensix::Blackhole),
    ];

    /// The vendor's name for the unit: `"XMX"`, `"Tensor Core"`, `"Matrix
    /// Core"` (CDNA), `"AI Accelerator"` (RDNA), `"Tensix FPU"`.
    #[must_use]
    pub const fn unit(self) -> &'static str {
        match self {
            Self::Xmx(_) => "XMX",
            Self::TensorCore(_) => "Tensor Core",
            Self::MatrixCore(g) => g.unit(),
            Self::Tensix(_) => "Tensix FPU",
        }
    }

    /// The generation: `"Xe2"`, `"Ada"`, `"CDNA 3"`, `"Blackhole"`.
    #[must_use]
    pub const fn generation(self) -> &'static str {
        match self {
            Self::Xmx(g) => g.name(),
            Self::TensorCore(g) => g.name(),
            Self::MatrixCore(g) => g.name(),
            Self::Tensix(g) => g.name(),
        }
    }

    /// The instruction groups of this generation, each a table.
    #[must_use]
    pub const fn families(self) -> &'static [&'static [Mma]] {
        match self {
            Self::Xmx(g) => g.families(),
            Self::TensorCore(g) => g.families(),
            Self::MatrixCore(g) => g.families(),
            Self::Tensix(g) => g.families(),
        }
    }

    /// Every instruction, the preferred one first for each type combination.
    pub fn mmas(self) -> impl Iterator<Item = &'static Mma> {
        self.families().iter().flat_map(|family| family.iter())
    }

    /// The first instruction for exactly these formats.
    #[must_use]
    pub fn find(self, a: Format, b: Format, acc: Format) -> Option<&'static Mma> {
        self.mmas().find(|m| m.takes(a, b, acc))
    }

    /// Where its table was checked against real hardware.
    #[must_use]
    pub const fn support(self) -> Support {
        match self {
            // Through Vulkan cooperative matrices on the B580: what the
            // driver exposes matches, and those four ran (`matrix_probe`).
            Self::Xmx(intel::Xmx::Xe2) => Support::WINDOWS_TESTED,
            _ => Support::UNTESTED,
        }
    }

    /// The vendor tier its kernels are built for (`kernels/bin/<vendor>/<tier>`),
    /// if the crate has one.
    #[must_use]
    pub const fn tier(self) -> Option<Tier> {
        match self {
            Self::Xmx(g) => g.tier(),
            Self::TensorCore(g) => g.tier(),
            Self::MatrixCore(g) => Some(g.tier()),
            Self::Tensix(g) => Some(g.tier()),
        }
    }

    /// The engine a tier's kernels run on: `None` for [`Tier::Emulated`]
    /// (the vector units), DP4A-only Xe-LP, `VideoCore` and the NPUs.
    #[must_use]
    pub const fn of(tier: Tier) -> Option<Self> {
        match tier {
            Tier::Intel(t) => match intel::Xmx::of(t) {
                Some(g) => Some(Self::Xmx(g)),
                None => None,
            },
            Tier::Nvidia(t) => Some(Self::TensorCore(nvidia::TensorCore::of(t))),
            Tier::Amd(t) => Some(Self::MatrixCore(amd::MatrixCore::of(t))),
            Tier::Tt(t) => Some(Self::Tensix(tt::Tensix::of(t))),
            _ => None,
        }
    }

    /// The engine of a GPU or card known by its PCI ids; `None` without one
    /// (or when the tables cannot tell: then never a guess).
    #[must_use]
    pub const fn from_gpu(gpu: Gpu) -> Option<Self> {
        match gpu.architecture {
            Architecture::Intel(arch) => match intel::Xmx::from_arch(arch) {
                Some(g) => Some(Self::Xmx(g)),
                None => None,
            },
            Architecture::Nvidia(arch) => match nvidia::TensorCore::from_pci(arch, gpu.device) {
                Some(g) => Some(Self::TensorCore(g)),
                None => None,
            },
            Architecture::Amd(arch) => match amd::MatrixCore::from_pci(arch, gpu.device) {
                Some(g) => Some(Self::MatrixCore(g)),
                None => None,
            },
            Architecture::Tenstorrent(arch) => match tt::Tensix::from_arch(arch) {
                Some(g) => Some(Self::Tensix(g)),
                None => None,
            },
            Architecture::Unknown => None,
        }
    }

    /// How `gemm_<a>_<b>_<acc>` gets onto this engine: natively, after
    /// [`widen`]ing its inputs (exact: every value stays), with an `F16` or
    /// `BF16` accumulator widened to `F32` (closer, not the same bits), or
    /// not at all. Among several options the narrowest inputs win.
    #[must_use]
    pub fn route(self, a: Format, b: Format, acc: Format) -> Route {
        if let Some(mma) = self.find(a, b, acc) {
            return Route::Native(mma);
        }
        let accs: &[Format] = if acc == f::F16 || acc == f::BF16 {
            &[acc, f::F32]
        } else {
            std::slice::from_ref(match acc {
                Format::F32 => &f::F32,
                Format::I32 => &f::I32,
                _ => return Route::Vector,
            })
        };
        let with_self = |x: Format| std::iter::once(x).chain(widen(x).iter().copied());
        for &acc2 in accs {
            let mut best: Option<(usize, Format, Format, &'static Mma)> = None;
            for (i, a2) in with_self(a).enumerate() {
                for (j, b2) in with_self(b).enumerate() {
                    if let Some(mma) = self.find(a2, b2, acc2)
                        && best.is_none_or(|(cost, ..)| i + j < cost)
                    {
                        best = Some((i + j, a2, b2, mma));
                    }
                }
            }
            if let Some((_, a, b, mma)) = best {
                return Route::Widened {
                    a,
                    b,
                    acc: acc2,
                    mma,
                };
            }
        }
        Route::Vector
    }
}

impl fmt::Display for Engine {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{} {}", self.generation(), self.unit())
    }
}

/// How a GEMM reaches a matrix engine ([`Engine::route`]).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Route {
    /// An instruction takes the formats as they are.
    Native(&'static Mma),
    /// Converted first (`ops::convert` / `ops::unpack`, exact), then `mma`.
    /// `acc` differs from the asked one only for `F16`/`BF16` -> `F32`.
    Widened {
        a: Format,
        b: Format,
        acc: Format,
        mma: &'static Mma,
    },
    /// No instruction takes it, even widened: the vector units run it
    /// ([`Tier::Emulated`]). Block formats a generation cannot read come
    /// here too: dequantizing them is not exact.
    Vector,
}

impl Route {
    #[must_use]
    pub const fn mma(self) -> Option<&'static Mma> {
        match self {
            Self::Native(mma) | Self::Widened { mma, .. } => Some(mma),
            Self::Vector => None,
        }
    }
}

impl fmt::Display for Route {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Native(mma) => write!(f, "native: {mma}"),
            Self::Widened { a, b, acc, mma } => write!(
                f,
                "widened to {} x {} -> {}: {mma}",
                a.name(),
                b.name(),
                acc.name()
            ),
            Self::Vector => f.write_str("no matrix instruction: vector units"),
        }
    }
}

/// The formats `x` converts to without changing any value, narrowest first.
///
/// Floats keep sign, NaN and infinity (as far as the target has one; FNUZ
/// codes have no infinity, so nothing turns into one); integers stay
/// integers (an `I32` accumulator does not become a float one). Block
/// formats widen to nothing: their scales reach past `F16` and `BF16`.
#[must_use]
pub const fn widen(x: Format) -> &'static [Format] {
    use f::{BF16, E2M3, E3M2, E4M3, E5M2, F16, F32, I4, I8, TF32, U4, U8};
    match x {
        Format::Float(m) => {
            use miraculix::formats::MiniFloat as M;
            match m {
                M::E2M1 => &[E2M3, E3M2, E4M3, E5M2, F16, BF16],
                M::E2M3 => &[E4M3, F16, BF16],
                M::E3M2 => &[E4M3, E5M2, F16, BF16],
                M::E4M3Fn | M::E5M2 | M::E4M3Fnuz | M::E5M2Fnuz => &[F16, BF16],
                M::F16 | M::Bf16 => &[TF32, F32],
                _ => &[],
            }
        }
        Format::Tf32 => &[F32],
        Format::Int {
            bits: 2,
            signed: true,
        } => &[I4, I8],
        Format::Int {
            bits: 2,
            signed: false,
        } => &[U4, U8, I4, I8],
        Format::Int {
            bits: 4,
            signed: true,
        } => &[I8],
        Format::Int {
            bits: 4,
            signed: false,
        } => &[U8, I8],
        _ => &[],
    }
}

/// The matrix engines behind a device's tiers, best last (as the tiers).
#[must_use]
pub fn engines(tiers: &[Tier]) -> Vec<Engine> {
    let mut out: Vec<Engine> = Vec::new();
    for &tier in tiers {
        if let Some(e) = Engine::of(tier)
            && !out.contains(&e)
        {
            out.push(e);
        }
    }
    out
}
