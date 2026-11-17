//! Mock backend (feature `mock`, tests only): runs the Emulated semantics on
//! the host with the miraculix codecs, synchronously. It is the oracle for
//! every real tier (plan B.3.1, B.5).

pub mod codec;

use std::collections::HashMap;
use std::sync::Mutex;
use std::time::{Duration, Instant};

use miraculix::formats::{MiniFloat, Overflow};

use crate::backend::{Arg, Backend, BufId, Fence, KernelId, KernelKey, OpSig};
use crate::elem::Format;
use crate::engine::{Access, DeviceClass, Engine, FeatureSet};
use crate::error::{Error, Result};
use crate::tiers::Tier;

#[derive(Default)]
struct State {
    next: u64,
    bufs: HashMap<u64, Vec<u8>>,
    kernels: Vec<OpSig>,
    downloads: HashMap<u64, Vec<u8>>,
    elapsed: HashMap<u64, Duration>,
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

fn sig_supported(sig: OpSig) -> bool {
    match sig {
        OpSig::Convert { from, to } => codec::convert_supported(from, to),
        OpSig::Gemm { a, b, acc } => {
            let float_in = |f: Format| !f.is_integer();
            match acc {
                Format::I32 => a.is_integer() && b.is_integer(),
                Format::F32 | Format::Float(MiniFloat::F16) => float_in(a) && float_in(b),
                _ => false,
            }
        }
        OpSig::AbsmaxBlock { src } => !src.is_integer(),
        #[allow(unreachable_patterns)]
        _ => false,
    }
}

fn buf(a: &Arg) -> Result<u64> {
    match a {
        Arg::Buf(b) => Ok(b.0),
        _ => Err(Error::InvalidArgument("expected buffer arg".into())),
    }
}
fn u(a: &Arg) -> Result<usize> {
    match a {
        Arg::U32(v) => Ok(*v as usize),
        _ => Err(Error::InvalidArgument("expected u32 arg".into())),
    }
}

impl State {
    fn get(&self, id: u64) -> Result<&Vec<u8>> {
        self.bufs.get(&id).ok_or(Error::InvalidArgument(format!("unknown buffer {id}")))
    }

    fn fence(&mut self) -> Fence {
        self.next += 1;
        Fence(self.next)
    }

    fn run(&mut self, sig: OpSig, args: &[Arg]) -> Result<()> {
        let bad = || Error::Unsupported(format!("mock: {sig:?}"));
        match sig {
            OpSig::Convert { from, to } => {
                let (src, dst, len) = (buf(&args[0])?, buf(&args[1])?, u(&args[2])?);
                let ov = if u(&args[3])? == 0 { Overflow::Saturate } else { Overflow::Special };
                let out = codec::convert(from, to, self.get(src)?, len, ov).ok_or_else(bad)?;
                self.bufs.insert(dst, out);
            }
            OpSig::AbsmaxBlock { src } => {
                let (s, d, len, block) = (buf(&args[0])?, buf(&args[1])?, u(&args[2])?, u(&args[3])?);
                let x = codec::decode_f32(src, self.get(s)?, len).ok_or_else(bad)?;
                let m: Vec<f32> =
                    x.chunks_exact(block).map(|c| c.iter().fold(0.0f32, |m, v| if v.abs() > m { v.abs() } else { m })).collect();
                self.bufs.insert(d, crate::elem::as_bytes(&m).to_vec());
            }
            OpSig::Gemm { a, b, acc } => {
                let (ia, ib, ic) = (buf(&args[0])?, buf(&args[1])?, buf(&args[2])?);
                let (m, n, k, batch) = (u(&args[3])?, u(&args[4])?, u(&args[5])?, u(&args[6])?);
                let out = if acc == Format::I32 {
                    let (Arg::I32(alpha), Arg::I32(beta)) = (args[7], args[8]) else { return Err(bad()) };
                    let av = codec::decode_i32(a, self.get(ia)?, batch * m * k).ok_or_else(bad)?;
                    let bv = codec::decode_i32(b, self.get(ib)?, batch * n * k).ok_or_else(bad)?;
                    let cv = codec::decode_i32(acc, self.get(ic)?, batch * m * n).ok_or_else(bad)?;
                    let mut c = vec![0i32; batch * m * n];
                    for (((cb, ab), bb), cold) in
                        c.chunks_exact_mut(m * n).zip(av.chunks_exact(m * k)).zip(bv.chunks_exact(n * k)).zip(cv.chunks_exact(m * n))
                    {
                        for i in 0..m {
                            for j in 0..n {
                                let mut s = 0i32;
                                for p in 0..k {
                                    s = s.wrapping_add(ab[i * k + p].wrapping_mul(bb[j * k + p]));
                                }
                                let mut r = alpha.wrapping_mul(s);
                                if beta != 0 {
                                    r = r.wrapping_add(beta.wrapping_mul(cold[i * n + j]));
                                }
                                cb[i * n + j] = r;
                            }
                        }
                    }
                    codec::encode_i32(acc, &c).ok_or_else(bad)?
                } else {
                    let (Arg::F32(alpha), Arg::F32(beta)) = (args[7], args[8]) else { return Err(bad()) };
                    let av = codec::decode_f32(a, self.get(ia)?, batch * m * k).ok_or_else(bad)?;
                    let bv = codec::decode_f32(b, self.get(ib)?, batch * n * k).ok_or_else(bad)?;
                    let cv = if beta != 0.0 {
                        codec::decode_f32(acc, self.get(ic)?, batch * m * n).ok_or_else(bad)?
                    } else {
                        vec![0.0; batch * m * n]
                    };
                    let mut c = vec![0f32; batch * m * n];
                    for (((cb, ab), bb), cold) in
                        c.chunks_exact_mut(m * n).zip(av.chunks_exact(m * k)).zip(bv.chunks_exact(n * k)).zip(cv.chunks_exact(m * n))
                    {
                        for i in 0..m {
                            for j in 0..n {
                                let mut s = 0f32;
                                for p in 0..k {
                                    s = ab[i * k + p].mul_add(bb[j * k + p], s);
                                }
                                cb[i * n + j] = if beta != 0.0 { alpha * s + beta * cold[i * n + j] } else { alpha * s };
                            }
                        }
                    }
                    codec::encode_f32(acc, &c, Overflow::Special).ok_or_else(bad)?
                };
                self.bufs.insert(ic, out);
            }
            #[allow(unreachable_patterns)]
            _ => return Err(bad()),
        }
        Ok(())
    }
}

impl Backend for MockBackend {
    fn features(&self) -> &FeatureSet {
        &self.features
    }

    fn alloc(&self, bytes: usize) -> Result<BufId> {
        let mut s = self.lock();
        s.next += 1;
        let id = s.next;
        s.bufs.insert(id, vec![0; bytes]);
        Ok(BufId(id))
    }

    fn free(&self, id: BufId) {
        self.lock().bufs.remove(&id.0);
    }

    fn upload(&self, id: BufId, offset: usize, src: &[u8]) -> Result<Fence> {
        let mut s = self.lock();
        let b = s.bufs.get_mut(&id.0).ok_or(Error::InvalidArgument("unknown buffer".into()))?;
        b.get_mut(offset..offset + src.len()).ok_or(Error::InvalidArgument("upload out of range".into()))?.copy_from_slice(src);
        Ok(s.fence())
    }

    fn download(&self, id: BufId, offset: usize, len: usize) -> Result<Fence> {
        let mut s = self.lock();
        let data = s.get(id.0)?.get(offset..offset + len).ok_or(Error::InvalidArgument("download out of range".into()))?.to_vec();
        let f = s.fence();
        s.downloads.insert(f.0, data);
        Ok(f)
    }

    fn take_download(&self, fence: Fence, dst: &mut [u8]) -> Result<()> {
        let data = self.lock().downloads.remove(&fence.0).ok_or(Error::InvalidArgument("no such download".into()))?;
        dst.copy_from_slice(&data);
        Ok(())
    }

    fn supports(&self, key: &KernelKey) -> bool {
        self.kernel_tiers.contains(&key.tier) && self.features.tiers.contains(&key.tier) && sig_supported(key.sig)
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
        let sig = *s.kernels.get(kernel.0 as usize).ok_or(Error::InvalidArgument("unknown kernel".into()))?;
        let t = Instant::now();
        s.run(sig, args)?;
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
