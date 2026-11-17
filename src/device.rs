//! Devices and placement (plan B.2b).

use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use crate::backend::{Backend, KernelId, KernelKey};
use crate::engine::{DeviceClass, Engine, FeatureSet};
use crate::error::{Error, Result};
use crate::stream::Stream;

/// Where work should run.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Placement {
    /// Highest matrix tier; GPUs and accelerator cards before NPUs.
    Fastest,
    /// NPUs before GPUs: moves GEMM batches off CPU and iGPU.
    LowPower,
    /// The `n`-th device of [`Device::all`].
    Device(usize),
}

pub(crate) struct Inner {
    pub(crate) backend: Arc<dyn Backend>,
    kernels: Mutex<HashMap<KernelKey, KernelId>>,
}

/// One accelerator. Cheap to clone, `Send + Sync`.
#[derive(Clone)]
pub struct Device {
    pub(crate) inner: Arc<Inner>,
}

impl core::fmt::Debug for Device {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        let fs = self.features();
        write!(f, "Device({:?} {:?}, {})", fs.class, fs.name, fs.driver)
    }
}

impl Device {
    /// Wraps a backend (used by the built-in probes and by out-of-tree backends).
    pub fn from_backend(backend: Arc<dyn Backend>) -> Device {
        Device { inner: Arc::new(Inner { backend, kernels: Mutex::new(HashMap::new()) }) }
    }

    /// A host-emulated device for tests (feature `mock`).
    #[cfg(feature = "mock")]
    pub fn mock() -> Device {
        Device::from_backend(Arc::new(crate::mock::MockBackend::new(DeviceClass::Gpu)))
    }

    /// Every device the compiled-in backends can open.
    ///
    /// Raw backends are added here as their B.3 steps land; for now the list is empty.
    pub fn all() -> Vec<Device> {
        Vec::new()
    }

    pub fn open(placement: Placement) -> Result<Device> {
        placement.pick(&Device::all()).ok_or(Error::NoDevice)
    }

    pub fn features(&self) -> &FeatureSet {
        self.inner.backend.features()
    }

    pub fn stream(&self) -> Stream {
        Stream::new(self.clone())
    }

    pub(crate) fn backend(&self) -> &dyn Backend {
        &*self.inner.backend
    }

    pub(crate) fn kernel(&self, key: &KernelKey) -> Result<KernelId> {
        let mut cache = self.inner.kernels.lock().unwrap_or_else(|e| e.into_inner());
        if let Some(&k) = cache.get(key) {
            return Ok(k);
        }
        let k = self.inner.backend.load_kernel(key)?;
        cache.insert(key.clone(), k);
        Ok(k)
    }
}

impl Placement {
    /// Picks from `devices` (only those with a matrix engine).
    pub fn pick(self, devices: &[Device]) -> Option<Device> {
        let matrix = |d: &&Device| d.features().access(Engine::Matrix).is_some();
        match self {
            Placement::Device(n) => devices.get(n).cloned(),
            Placement::Fastest | Placement::LowPower => {
                let rank = |d: &Device| {
                    let class = match (self, d.features().class) {
                        (Placement::LowPower, DeviceClass::Npu) => 2,
                        (Placement::LowPower, _) => 1,
                        (_, DeviceClass::Npu) => 1,
                        _ => 2,
                    };
                    (class, d.features().tiers.len())
                };
                // max_by_key keeps the last max; reverse so the first device wins ties.
                devices.iter().filter(matrix).rev().max_by_key(|d| rank(d)).cloned()
            }
        }
    }
}
