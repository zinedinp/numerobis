//! A pool of HSA completion signals (`amd_signal_t`) for one engine.
//!
//! Fences count submissions from 1; fence `n` owns slot `n % SLOTS`, armed
//! at 1 and brought to 0 by the engine (the CP for AQL packets, an SDMA
//! `ATOMIC` for copies). An engine retires in order, so fence `n` done means
//! every fence before it is done too.
//!
//! A slot is armed again only once its previous fence has signalled *and*
//! the other engine's packet that waits on it (its consumer) has run:
//! otherwise that packet could see the slot re-armed for a later fence and
//! wait for work queued behind it.

use std::sync::atomic::{Ordering, fence};
use std::time::{Duration, Instant};

use super::kfd::{Bo, Gpu, Kind};
use super::sys::signal as sig;
use crate::runtime::error::{Error, Result};
use crate::runtime::os::View;

/// Slots per pool.
pub const SLOTS: u64 = 256;

pub struct Signals {
    bo: Bo,
    /// Fences handed out so far.
    pub last: u64,
    /// Highest fence seen signalled.
    completed: u64,
    /// Per slot: the other engine's fence that waits on this slot's fence.
    consumers: Vec<u64>,
}

pub(crate) fn view(bo: &Bo) -> View {
    bo.cpu().expect("CPU-mapped engine memory")
}

impl Signals {
    /// # Errors
    ///
    /// As [`Gpu::alloc`].
    pub fn new(gpu: &Gpu) -> Result<Signals> {
        Ok(Signals {
            bo: gpu.alloc(SLOTS * sig::SIZE as u64, Kind::Gtt)?,
            last: 0,
            completed: 0,
            consumers: vec![0; SLOTS as usize],
        })
    }

    #[must_use]
    pub fn slot(f: u64) -> usize {
        (f % SLOTS) as usize
    }

    /// GPU address of fence `f`'s `amd_signal_t` (an `hsa_signal_t` handle).
    #[must_use]
    pub fn address(&self, f: u64) -> u64 {
        self.bo.va + (Self::slot(f) * sig::SIZE) as u64
    }

    /// GPU address of fence `f`'s signal value: 0 once done.
    #[must_use]
    pub fn value_address(&self, f: u64) -> u64 {
        self.address(f) + sig::VALUE as u64
    }

    pub(crate) fn value(&self, f: u64) -> i64 {
        view(&self.bo).read64(Self::slot(f) * sig::SIZE + sig::VALUE) as i64
    }

    /// Whether fence `f` has signalled. Fence 0 (nothing) always has.
    pub fn done(&mut self, f: u64) -> bool {
        if f <= self.completed {
            return true;
        }
        if f > self.last {
            return false;
        }
        // A slot is re-armed only after its fence was seen done (and then
        // `completed` covers it), so the slot still holds `f`.
        if self.value(f) <= 0 {
            self.completed = f;
            return true;
        }
        false
    }

    /// Blocks until fence `f` has signalled; `state` describes the engine
    /// for the trace of a timeout.
    ///
    /// # Errors
    ///
    /// [`Error::Timeout`] when it hasn't after `timeout`.
    pub fn wait(
        &mut self,
        f: u64,
        timeout: Duration,
        state: impl Fn(&Self) -> String,
    ) -> Result<()> {
        let start = Instant::now();
        let mut spins = 0u32;
        while !self.done(f) {
            if start.elapsed() > timeout {
                let e = Error::Timeout(timeout);
                super::kfd::trace(&state(self), &e);
                return Err(e);
            }
            spins += 1;
            if spins < 1000 {
                std::hint::spin_loop();
            } else {
                std::thread::sleep(Duration::from_micros(20));
            }
        }
        // Device writes released at system scope; order the host's reads after.
        fence(Ordering::Acquire);
        Ok(())
    }

    /// The other engine's fence that must run before the next fence's slot
    /// is armed again (0: none).
    #[must_use]
    pub fn next_consumer(&self) -> u64 {
        self.consumers[Self::slot(self.last + 1)]
    }

    /// Records that the other engine's fence `consumer` waits on fence `f`.
    pub fn add_consumer(&mut self, f: u64, consumer: u64) {
        let c = &mut self.consumers[Self::slot(f)];
        *c = (*c).max(consumer);
    }

    /// Hands out the next fence with its signal armed at 1; returns the
    /// fence and its signal's address. The caller has made sure the slot's
    /// consumer ran ([`next_consumer`](Self::next_consumer)).
    ///
    /// # Errors
    ///
    /// [`Error::Timeout`] when the slot's previous fence stays busy for 10 s.
    pub fn arm(&mut self, state: impl Fn(&Self) -> String) -> Result<(u64, u64)> {
        let f = self.last + 1;
        self.wait(f.saturating_sub(SLOTS), Duration::from_secs(10), state)?;
        let v = view(&self.bo);
        let at = Self::slot(f) * sig::SIZE;
        v.write64(at + sig::KIND, 1);
        v.write64(at + sig::VALUE, 1);
        v.write64(at + sig::START_TS, 0);
        v.write64(at + sig::END_TS, 0);
        self.consumers[Self::slot(f)] = 0;
        self.last = f;
        Ok((f, self.address(f)))
    }

    /// The start and end timestamps the CP wrote for fence `f`, once
    /// signalled and while its slot holds it.
    pub fn timestamps(&mut self, f: u64) -> Option<(u64, u64)> {
        if !self.done(f) || f + SLOTS <= self.last {
            return None;
        }
        let v = view(&self.bo);
        let at = Self::slot(f) * sig::SIZE;
        let (a, b) = (v.read64(at + sig::START_TS), v.read64(at + sig::END_TS));
        (a != 0 && b >= a).then_some((a, b))
    }

    /// The timestamps in fence `f`'s slot as they are, unchecked.
    pub(crate) fn raw_timestamps(&self, f: u64) -> (u64, u64) {
        let v = view(&self.bo);
        let at = Self::slot(f) * sig::SIZE;
        (v.read64(at + sig::START_TS), v.read64(at + sig::END_TS))
    }

    pub fn free(self, gpu: &Gpu) {
        gpu.free(self.bo);
    }
}
