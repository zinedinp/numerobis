//! The `Backend` trait: the only thing a raw, comfort or mock target implements.

use std::os::fd::{BorrowedFd, OwnedFd};
use std::time::Duration;

use crate::elem::Format;
use crate::engine::FeatureSet;
use crate::runtime::error::Result;
use crate::tiers::Tier;

/// Backend-local buffer handle (GEM handle, KFD BO, …).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct BufId(pub u64);

/// Backend-local loaded-kernel handle.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct KernelId(pub u64);

/// Completion point of submitted work (syncobj, AQL signal, fence value, …).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Fence(pub u64);

/// Structured description of an op. Raw backends look kernels up by
/// [`KernelKey::name`]; the mock backend interprets this signature directly.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum OpSig {
    /// Float -> float, element for element.
    /// Args: `[src, dst, len: U32, overflow: U32 (0 saturate, 1 special)]`.
    Convert { from: Format, to: Format },
    /// `F32` -> block format. Args: `[src, dst, len: U32]`.
    Quantize { to: Format },
    /// Block format -> `F32`. Args: `[src, dst, len: U32]`.
    Dequantize { from: Format },
    /// `I8`/`U8` -> packed sub-byte integer, saturating. Args: `[src, dst, len: U32]`.
    Pack { to: Format },
    /// Packed sub-byte integer -> `I8`/`U8`. Args: `[src, dst, len: U32]`.
    Unpack { from: Format },
    /// `C = alpha*A*B^T + beta*C`, batched. A is `MxK`, B is `NxK` (both K-contiguous),
    /// C is `MxN`, batches back to back.
    /// Args: `[a, b, c, m, n, k, batch: U32, alpha, beta]`.
    Gemm { a: Format, b: Format, acc: Format },
    /// Per-block max of `|x|`. Args: `[src, dst, len: U32, block: U32]`.
    AbsmaxBlock { src: Format },
    /// Element-wise activation. Args: `[src, dst, len: U32, p0: F32, p1: F32]`.
    Act {
        src: Format,
        dst: Format,
        act: ActKind,
    },
    /// Direct 2-D convolution, NCHW / KCRS / NKPQ.
    /// Args: `[x, w, bias (F32, or x when absent), y, params (U32 buffer, `conv::PARAM_WORDS`)]`.
    Conv2d {
        x: Format,
        w: Format,
        y: Format,
        bias: bool,
    },
    /// Image resize of `planes` HxW planes to HoxWo.
    /// Args: `[src, dst, planes, h, w, ho, wo: U32]`.
    Resize {
        src: Format,
        dst: Format,
        mode: ResizeMode,
    },
}

/// Activation function of [`OpSig::Act`] (parameters travel as arguments).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ActKind {
    Relu,
    LeakyRelu,
    Sigmoid,
    Tanh,
    Silu,
    Gelu,
    Clamp,
    Exp,
}

impl ActKind {
    /// Kernel code (`NB_ACT_*` in `kernels/include/nb_math.h`).
    pub fn code(self) -> u32 {
        self as u32
    }
    pub fn name(self) -> &'static str {
        match self {
            ActKind::Relu => "relu",
            ActKind::LeakyRelu => "leaky_relu",
            ActKind::Sigmoid => "sigmoid",
            ActKind::Tanh => "tanh",
            ActKind::Silu => "silu",
            ActKind::Gelu => "gelu",
            ActKind::Clamp => "clamp",
            ActKind::Exp => "exp",
        }
    }
    /// Pure selects/compares: identical on every tier (no rounding inside).
    pub fn is_selection(self) -> bool {
        matches!(self, ActKind::Relu | ActKind::Clamp)
    }
}

/// Sampling of [`OpSig::Resize`], half-pixel centres.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ResizeMode {
    /// Integer index arithmetic only: identical on every tier.
    Nearest,
    Bilinear,
}

impl ResizeMode {
    pub fn code(self) -> u32 {
        self as u32
    }
    pub fn name(self) -> &'static str {
        match self {
            ResizeMode::Nearest => "nearest",
            ResizeMode::Bilinear => "bilinear",
        }
    }
}

/// Kernel lookup key: `<op>_<types>` name plus tier.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct KernelKey {
    pub name: String,
    pub sig: OpSig,
    pub tier: Tier,
}

impl OpSig {
    /// Kernel name following the miraculix `<op>_<type>` convention.
    pub fn name(&self, batched: bool) -> String {
        match *self {
            OpSig::Convert { from, to } => format!("convert_{}_{}", from.name(), to.name()),
            OpSig::Quantize { to } => format!("quantize_{}", to.name()),
            OpSig::Dequantize { from } => format!("dequantize_{}", from.name()),
            OpSig::Pack { to } => format!("pack_{}", to.name()),
            OpSig::Unpack { from } => format!("unpack_{}", from.name()),
            OpSig::Gemm { a, b, acc } => {
                let p = if batched { "gemm_strided" } else { "gemm" };
                format!("{p}_{}_{}_{}", a.name(), b.name(), acc.name())
            }
            OpSig::AbsmaxBlock { src } => format!("absmax_block_{}", src.name()),
            OpSig::Act { src, dst, act } => {
                format!("act_{}_{}_{}", act.name(), src.name(), dst.name())
            }
            OpSig::Conv2d { x, w, y, bias } => {
                format!(
                    "conv2d{}_{}_{}_{}",
                    if bias { "_bias" } else { "" },
                    x.name(),
                    w.name(),
                    y.name()
                )
            }
            OpSig::Resize { src, dst, mode } => {
                format!("resize_{}_{}_{}", mode.name(), src.name(), dst.name())
            }
        }
    }

    /// Does this op return the same bits as the mock oracle on `tier`?
    ///
    /// Format conversion, quantization, packing, absmax, nearest resize and
    /// the select-only activations are exact by definition on every tier.
    /// Everything that rounds (GEMM, convolution, bilinear resize,
    /// transcendental activations) is bit-exact only on [`Tier::Emulated`];
    /// vendor tiers (XMX, MFMA, Tensix FPU, native `exp`) reorder or
    /// approximate. See [`Exactness`](crate::Exactness).
    pub fn exact_on(&self, tier: Tier) -> bool {
        if tier == Tier::Emulated {
            return true;
        }
        match *self {
            OpSig::Convert { .. }
            | OpSig::Quantize { .. }
            | OpSig::Dequantize { .. }
            | OpSig::Pack { .. }
            | OpSig::Unpack { .. }
            | OpSig::AbsmaxBlock { .. } => true,
            OpSig::Act { act, .. } => act.is_selection(),
            OpSig::Resize { mode, .. } => mode == ResizeMode::Nearest,
            OpSig::Gemm { .. } | OpSig::Conv2d { .. } => false,
        }
    }
}

/// A kernel argument.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Arg {
    Buf(BufId),
    U32(u32),
    I32(i32),
    F32(f32),
}

impl From<f32> for Arg {
    fn from(v: f32) -> Arg {
        Arg::F32(v)
    }
}
impl From<i32> for Arg {
    fn from(v: i32) -> Arg {
        Arg::I32(v)
    }
}
impl From<u32> for Arg {
    fn from(v: u32) -> Arg {
        Arg::U32(v)
    }
}

/// One target's device access. Implementations are `Send + Sync`; per-queue
/// state lives behind the backend's own locks.
///
/// Every call that submits work returns a [`Fence`]; [`wait`](Self::wait) has a
/// timeout. Unknown ioctls (`ENOTTY`/`EINVAL`) are reported as
/// [`Error::Unsupported`](crate::Error::Unsupported), not as hard errors.
pub trait Backend: Send + Sync + 'static {
    fn features(&self) -> &FeatureSet;

    /// Allocates `bytes` of device memory (contents unspecified).
    fn alloc(&self, bytes: usize) -> Result<BufId>;
    /// Releases a buffer. Called only after all work using it has retired.
    fn free(&self, id: BufId);

    /// Copies host bytes into the buffer through internal staging.
    fn upload(&self, id: BufId, offset: usize, src: &[u8]) -> Result<Fence>;
    /// Starts a device -> staging copy of `len` bytes.
    fn download(&self, id: BufId, offset: usize, len: usize) -> Result<Fence>;
    /// After `fence` (from [`download`](Self::download)) has signalled, copies the staged bytes out.
    fn take_download(&self, fence: Fence, dst: &mut [u8]) -> Result<()>;

    /// Is there a kernel for this key? Must be cheap; used by `auto_up`.
    fn supports(&self, key: &KernelKey) -> bool;
    fn load_kernel(&self, key: &KernelKey) -> Result<KernelId>;
    fn dispatch(&self, kernel: KernelId, args: &[Arg]) -> Result<Fence>;

    fn wait(&self, fence: Fence, timeout: Duration) -> Result<()>;
    /// Device time between start and end of the work behind `fence`, once signalled.
    fn elapsed(&self, fence: Fence) -> Option<Duration>;

    /// Submits several dispatches in queue order as one unit (a recorded
    /// [`CommandList`](crate::CommandList)); returns the fence of the last.
    /// Backends with a real command buffer override this to submit once.
    fn dispatch_batch(&self, cmds: &[(KernelId, &[Arg])]) -> Result<Fence> {
        let mut last = None;
        for (k, a) in cmds {
            last = Some(self.dispatch(*k, a)?);
        }
        last.ok_or_else(|| crate::Error::InvalidArgument("empty command list".into()))
    }

    /// Exports the buffer as a dma-buf (DRM PRIME, udmabuf, tt-kmd TLB export).
    fn export_dmabuf(&self, _id: BufId) -> Result<OwnedFd> {
        Err(crate::Error::Unsupported(format!(
            "dma-buf export on {}",
            self.features().name
        )))
    }
    /// Imports a dma-buf of `bytes` bytes as a buffer of this device, without a copy.
    fn import_dmabuf(&self, _fd: BorrowedFd<'_>, _bytes: usize) -> Result<BufId> {
        Err(crate::Error::Unsupported(format!(
            "dma-buf import on {}",
            self.features().name
        )))
    }
    /// A kernel `sync_file` for `fence`, if the driver has dma-fences (DRM
    /// drivers). `None`: the runtime bridges through a DRM timeline syncobj.
    fn export_sync_file(&self, _fence: Fence) -> Option<OwnedFd> {
        None
    }
}
