//! Mock backend (feature `mock`, tests only): runs the Emulated semantics on
//! the host with the miraculix codecs, synchronously. It is the oracle for
//! every real tier (plan B.3 step 1, B.5).
//!
//! [`codec`] is split by format kind, [`kernels`] by op kind.

pub mod codec;
mod kernels;

use std::collections::HashMap;
use std::sync::Mutex;
use std::time::{Duration, Instant};

use crate::engine::{Access, DeviceClass, Engine, FeatureSet};
use crate::runtime::backend::{Arg, Backend, BufId, Fence, KernelId, KernelKey, OpSig};
use crate::runtime::error::{Error, Result};
use crate::tiers::Tier;

#[derive(Default)]
struct State {
    next: u64,
    mem: kernels::Mem,
    kernels: Vec<OpSig>,
    downloads: HashMap<u64, Vec<u8>>,
    elapsed: HashMap<u64, Duration>,
}

impl State {
    fn fence(&mut self) -> Fence {
        self.next += 1;
        Fence(self.next)
    }
}

/// Host-emulated device.
pub struct MockBackend {
    features: FeatureSet,
    /// Tiers that have "kernels" (all run the same emulation).
    kernel_tiers: Vec<Tier>,
    state: Mutex<State>,
}

impl MockBackend {
    /// A device with only [`Tier::Emulated`].
    pub fn new(class: DeviceClass) -> Self {
        Self::with_tiers(class, vec![Tier::Emulated], vec![Tier::Emulated])
    }

    /// A device that reports `tiers` but has kernels only for `kernel_tiers`
    /// (to test `auto_up` fallback and placement).
    pub fn with_tiers(class: DeviceClass, tiers: Vec<Tier>, kernel_tiers: Vec<Tier>) -> Self {
        MockBackend {
            features: FeatureSet {
                name: "mock".into(),
                class,
                driver: "host".into(),
                engines: vec![(Engine::Matrix, Access::Host), (Engine::Copy, Access::Host)],
                tiers,
            },
            kernel_tiers,
            state: Mutex::new(State::default()),
        }
    }

    fn lock(&self) -> std::sync::MutexGuard<'_, State> {
        self.state.lock().unwrap_or_else(|e| e.into_inner())
    }
}

/// Is `sig` a stage-1 op of the Emulated tier? Shared with test backends.
pub fn supported(sig: OpSig) -> bool {
    kernels::supported(sig)
}

fn invalid(msg: &str) -> Error {
    Error::InvalidArgument(msg.into())
}

impl Backend for MockBackend {
    fn features(&self) -> &FeatureSet {
        &self.features
    }

    fn alloc(&self, bytes: usize) -> Result<BufId> {
        let mut s = self.lock();
        s.next += 1;
        let id = s.next;
        s.mem.insert(id, vec![0; bytes]);
        Ok(BufId(id))
    }

    fn free(&self, id: BufId) {
        self.lock().mem.remove(&id.0);
    }

    fn upload(&self, id: BufId, offset: usize, src: &[u8]) -> Result<Fence> {
        let mut s = self.lock();
        let b = s.mem.get_mut(&id.0).ok_or_else(|| invalid("unknown buffer"))?;
        b.get_mut(offset..offset + src.len()).ok_or_else(|| invalid("upload out of range"))?.copy_from_slice(src);
        Ok(s.fence())
    }

    fn download(&self, id: BufId, offset: usize, len: usize) -> Result<Fence> {
        let mut s = self.lock();
        let b = s.mem.get(&id.0).ok_or_else(|| invalid("unknown buffer"))?;
        let data = b.get(offset..offset + len).ok_or_else(|| invalid("download out of range"))?.to_vec();
        let f = s.fence();
        s.downloads.insert(f.0, data);
        Ok(f)
    }

    fn take_download(&self, fence: Fence, dst: &mut [u8]) -> Result<()> {
        let data = self.lock().downloads.remove(&fence.0).ok_or_else(|| invalid("no such download"))?;
        dst.copy_from_slice(&data);
        Ok(())
    }

    fn supports(&self, key: &KernelKey) -> bool {
        self.kernel_tiers.contains(&key.tier) && self.features.tiers.contains(&key.tier) && kernels::supported(key.sig)
    }

    fn load_kernel(&self, key: &KernelKey) -> Result<KernelId> {
        if !self.supports(key) {
            return Err(Error::Unsupported(key.name.clone()));
        }
        let mut s = self.lock();
        s.kernels.push(key.sig);
        Ok(KernelId(s.kernels.len() as u64 - 1))
    }

    fn dispatch(&self, kernel: KernelId, args: &[Arg]) -> Result<Fence> {
        let mut s = self.lock();
        let sig = *s.kernels.get(kernel.0 as usize).ok_or_else(|| invalid("unknown kernel"))?;
        let t = Instant::now();
        kernels::run(&mut s.mem, sig, args)?;
        let f = s.fence();
        s.elapsed.insert(f.0, t.elapsed());
        Ok(f)
    }

    fn wait(&self, _fence: Fence, _timeout: Duration) -> Result<()> {
        Ok(())
    }

    fn elapsed(&self, fence: Fence) -> Option<Duration> {
        self.lock().elapsed.get(&fence.0).copied()
    }
}
