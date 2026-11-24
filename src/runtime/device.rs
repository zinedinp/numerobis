//! Devices and placement.

use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use crate::engine::{DeviceClass, Engine, FeatureSet};
use crate::runtime::backend::{Backend, KernelId, KernelKey};
use crate::runtime::error::{Error, Result};
use crate::runtime::stream::Stream;

/// Where work should run.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Placement {
    /// Highest matrix tier; GPUs and accelerator cards before NPUs.
    /// Devices without a matrix engine (whose vector units run the Emulated
    /// tier, such as RDNA 2) only when no device has one.
    Fastest,
    /// NPUs before GPUs: moves GEMM batches off CPU and iGPU. Devices
    /// without a matrix engine last, as for [`Fastest`](Self::Fastest).
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
        Device {
            inner: Arc::new(Inner {
                backend,
                kernels: Mutex::new(HashMap::new()),
            }),
        }
    }

    /// A host-emulated device for tests (feature `mock`).
    #[cfg(feature = "mock")]
    #[must_use]
    pub fn mock() -> Device {
        Device::from_backend(Arc::new(crate::mock::MockBackend::new(DeviceClass::Gpu)))
    }

    /// Every device the compiled-in backends can open.
    ///
    /// Every raw backend of the enabled features (`nvidia`, `amd`, `tt`). Devices
    /// that fail to open are skipped.
    #[must_use]
    pub fn all() -> Vec<Device> {
        #[allow(unused_mut)]
        let mut v = Vec::new();
        #[cfg(all(feature = "nvidia", target_os = "linux"))]
        v.extend(crate::nvidia::backend::open_all());
        #[cfg(all(feature = "amd", target_os = "linux"))]
        v.extend(crate::amd::backend::open_all());
        #[cfg(all(feature = "tt", target_os = "linux"))]
        v.extend(crate::tt::backend::open_all());
        v
    }

    /// The device `placement` picks among [`Device::all`].
    ///
    /// # Errors
    ///
    /// [`Error::NoDevice`](crate::Error::NoDevice) when no device of [`Device::all`] matches.
    pub fn open(placement: Placement) -> Result<Device> {
        placement.pick(&Device::all()).ok_or(Error::NoDevice)
    }

    #[must_use]
    pub fn features(&self) -> &FeatureSet {
        self.inner.backend.features()
    }

    #[must_use]
    pub fn stream(&self) -> Stream {
        Stream::new(self.clone())
    }

    pub(crate) fn backend(&self) -> &dyn Backend {
        &*self.inner.backend
    }

    pub(crate) fn kernel(&self, key: &KernelKey) -> Result<KernelId> {
        let mut cache = self
            .inner
            .kernels
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
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
    #[must_use]
    pub fn pick(self, devices: &[Device]) -> Option<Device> {
        match self {
            Placement::Device(n) => devices.get(n).cloned(),
            Placement::Fastest | Placement::LowPower => {
                let rank = |d: &Device| {
                    let class = match (self, d.features().class) {
                        (Placement::LowPower, DeviceClass::Npu) => 2,
                        (Placement::LowPower, _) | (_, DeviceClass::Npu) => 1,
                        _ => 2,
                    };
                    let matrix = d.features().access(Engine::Matrix).is_some();
                    (matrix, class, d.features().tiers.len())
                };
                // max_by_key keeps the last max; reverse so the first device wins ties.
                devices.iter().rev().max_by_key(|d| rank(d)).cloned()
            }
        }
    }
}
