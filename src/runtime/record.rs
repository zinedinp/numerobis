//! Reusable command lists: record once, replay per frame with other buffers.
//!
//! ```
//! use numerobis::{Bind, Choice, Device, elem::F32, ops::{self, Act}};
//!
//! let dev = Device::mock();
//! let mut s = dev.stream();
//! let (inp, mut out) = (s.alloc::<F32>(4)?, s.alloc::<F32>(4)?);
//! // Translate once (kernel lookup, lowering, checks).
//! let cl = s.record(|s| ops::act(s, Choice::Up, Act::Relu, &inp, &mut out).map(drop))?;
//! // Per frame: only the buffers change.
//! let frame = s.upload::<F32>(&[-1.0, 2.0, -3.0, 4.0])?;
//! let result = s.alloc::<F32>(4)?;
//! s.replay(&cl, &[Bind::new(&inp, &frame)?, Bind::new(&out, &result)?])?.wait()?;
//! assert_eq!(s.download(&result)?.wait()?.to_vec(), [0.0, 2.0, 0.0, 4.0]);
//! # Ok::<(), numerobis::Error>(())
//! ```

use std::collections::HashMap;
use std::sync::Arc;

use crate::elem::Elem;
use crate::runtime::backend::{Arg, BufId, KernelId};
use crate::runtime::buffer::{DeviceBuffer, RawBuffer};
use crate::runtime::device::Device;
use crate::runtime::error::{Result, invalid};
use crate::runtime::stream::Exactness;

pub(crate) struct Cmd {
    kernel: KernelId,
    args: Vec<Arg>,
    reads: Vec<Arc<RawBuffer>>,
    writes: Vec<Arc<RawBuffer>>,
    approx: bool,
}

impl Cmd {
    pub(crate) fn new(
        kernel: KernelId,
        args: &[Arg],
        reads: &[&Arc<RawBuffer>],
        writes: &[&Arc<RawBuffer>],
        approx: bool,
    ) -> Cmd {
        Cmd {
            kernel,
            args: args.to_vec(),
            reads: reads.iter().map(|b| Arc::clone(b)).collect(),
            writes: writes.iter().map(|b| Arc::clone(b)).collect(),
            approx,
        }
    }
}

#[derive(Default)]
pub(crate) struct Recorder {
    cmds: Vec<Cmd>,
}

impl Recorder {
    pub(crate) fn push(&mut self, c: Cmd) {
        self.cmds.push(c);
    }
    pub(crate) fn finish(self, dev: Device) -> CommandList {
        CommandList {
            dev,
            cmds: self.cmds,
        }
    }
}

/// A recorded sequence of launches (see [`Stream::record`](crate::Stream::record)).
/// Holds every recorded buffer alive; replay with [`Bind`]s to substitute them.
pub struct CommandList {
    pub(crate) dev: Device,
    cmds: Vec<Cmd>,
}

impl core::fmt::Debug for CommandList {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(
            f,
            "CommandList({} launches, {})",
            self.cmds.len(),
            if self.is_exact() {
                "bit-exact"
            } else {
                "approx"
            }
        )
    }
}

/// Replace a recorded buffer by another of the same type and length.
pub struct Bind<'a> {
    from: &'a Arc<RawBuffer>,
    to: &'a Arc<RawBuffer>,
}

impl<'a> Bind<'a> {
    pub fn new<E: Elem>(
        recorded: &'a DeviceBuffer<E>,
        actual: &'a DeviceBuffer<E>,
    ) -> Result<Bind<'a>> {
        if recorded.len() != actual.len() || recorded.bytes() != actual.bytes() {
            return Err(invalid(format!(
                "bind: recorded {} elements, actual {}",
                recorded.len(),
                actual.len()
            )));
        }
        Ok(Bind {
            from: &recorded.raw,
            to: &actual.raw,
        })
    }
}

pub(crate) struct Plan {
    pub(crate) cmds: Vec<(KernelId, Vec<Arg>)>,
    pub(crate) keep: Vec<Arc<RawBuffer>>,
    pub(crate) final_approx: Vec<(Arc<RawBuffer>, bool)>,
}

impl CommandList {
    pub fn len(&self) -> usize {
        self.cmds.len()
    }
    pub fn is_empty(&self) -> bool {
        self.cmds.is_empty()
    }
    /// Every launch is on a bit-exact tier.
    pub fn is_exact(&self) -> bool {
        self.cmds.iter().all(|c| !c.approx)
    }

    /// Substitutes the binds, checks exactness in order, and lists the final
    /// approx flag of every written buffer.
    pub(crate) fn plan(&self, binds: &[Bind<'_>], ex: Exactness) -> Result<Plan> {
        let mut map: HashMap<BufId, &Arc<RawBuffer>> = HashMap::new();
        for b in binds {
            if !Arc::ptr_eq(&b.to.backend, &self.dev.inner.backend) {
                return Err(invalid("bind: buffer belongs to a different device"));
            }
            if map.insert(b.from.id, b.to).is_some() {
                return Err(invalid("bind: the same recorded buffer is bound twice"));
            }
        }
        let sub = |r: &Arc<RawBuffer>| -> Arc<RawBuffer> {
            Arc::clone(map.get(&r.id).copied().unwrap_or(r))
        };
        if ex == Exactness::Bit && !self.is_exact() {
            return Err(invalid(
                "command list contains approximate launches; this stream is Exactness::Bit",
            ));
        }
        // Simulated approx state of every buffer as the list runs.
        let mut state: HashMap<BufId, (Arc<RawBuffer>, bool)> = HashMap::new();
        let mut keep = Vec::new();
        let mut cmds = Vec::with_capacity(self.cmds.len());
        for c in &self.cmds {
            let mut in_approx = false;
            for r in &c.reads {
                let b = sub(r);
                let a = state.get(&b.id).map_or_else(|| b.is_approx(), |e| e.1);
                if a && ex == Exactness::Bit {
                    return Err(invalid(
                        "replay: an input holds approximate results; this stream is Exactness::Bit",
                    ));
                }
                in_approx |= a;
                keep.push(b);
            }
            for w in &c.writes {
                let b = sub(w);
                state.insert(b.id, (b.clone(), c.approx || in_approx));
                keep.push(b);
            }
            let args = c
                .args
                .iter()
                .map(|a| match *a {
                    Arg::Buf(id) => Arg::Buf(map.get(&id).map_or(id, |b| b.id)),
                    other => other,
                })
                .collect();
            cmds.push((c.kernel, args));
        }
        Ok(Plan {
            cmds,
            keep,
            final_approx: state.into_values().collect(),
        })
    }
}
