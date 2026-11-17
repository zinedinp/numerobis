//! `Backend` over tt-kmd: device buffers in DRAM, the Emulated kernels run
//! on the BRISC core of one Tensix tile (`kernels/bin/tt`).
//!
//! First version, deliberately simple:
//! - buffers live in DRAM channel 0; host uploads/downloads go through a TLB
//!   window;
//! - one launch at a time: the core's runtime pulls every argument buffer
//!   into its L1 over the NoC, runs the kernel over all work items, and
//!   writes the buffers back, so a launch's buffers must fit the L1 staging
//!   area (~1.4 MiB);
//! - one Tensix core, scalar soft-float. Spreading work items over the 140
//!   cores and the FPU (matrix) tier are the next steps.
//!
//! The opened core stays owned by this process (one backend instance per
//! card, shared by every `Device`); dropping it puts the core back into reset.

use std::collections::{BTreeMap, HashMap};
use std::sync::{Arc, Mutex, MutexGuard, Weak};
use std::time::{Duration, Instant};

use super::card::{Arch, Card, Noc};
use super::Tier as TtTier;
use crate::engine::{Access, DeviceClass, Engine, FeatureSet};
use crate::loader::elf::Elf;
use crate::runtime::abi::{SYMBOLS, lower};
use crate::runtime::backend::{Arg, Backend, BufId, Fence, KernelId, KernelKey, OpSig};
use crate::runtime::error::{Error, Result};
use crate::runtime::Device;
use crate::tiers::Tier;

static BLACKHOLE: &[u8] = include_bytes!("../../kernels/bin/tt/blackhole-emulated.elf");

/// Tensix debug register: soft reset of the five baby RISC-V cores.
const SOFT_RESET: u64 = 0xFFB1_21B0;
const RESET_ALL: u32 = (1 << 11) | (1 << 12) | (1 << 13) | (1 << 14) | (1 << 18);
/// BRISC running, NCRISC and TRISC0..2 held in reset.
const RUN_BRISC: u32 = RESET_ALL & !(1 << 11);

/// L1 layout shared with `kernels/tt/tensix.ld` and `runtime.c`.
const MAILBOX: u64 = 0x18000;
const STAGING: u64 = 0x20000;
const L1_SIZE: u64 = 0x18_0000;
const MAX_ARGS: usize = 16;
const MAX_BUFS: usize = 8;
/// Mailbox field offsets.
const MB_SEQ: u64 = 0;
const MB_DONE: u64 = 4;
const MB_STATUS: u64 = 8;
const MB_KERNEL: u64 = 12;

/// DRAM channel 0 endpoint (NoC 0) and the range handed out.
const DRAM_XY: (u16, u16) = (0, 0);
const DRAM_BASE: u64 = 64 << 20;
/// Channel size the firmware exposes (4 GiB minus 16 MiB reserved at the top).
const DRAM_END: u64 = 4_278_190_080;
const ALIGN: u64 = 64;

/// Tensix tile used by default (NoC 0 coordinates). Never probe tiles by
/// reading them: a NoC read to a harvested or non-Tensix tile can hang the
/// Blackhole NoC (and with it the ARC), which only a reset clears.
/// `NUMEROBIS_TT_CORE=x,y` picks another tile.
const DEFAULT_CORE: (u16, u16) = (1, 2);

fn core_from_env() -> Result<(u16, u16)> {
    let Some(v) = std::env::var_os("NUMEROBIS_TT_CORE") else { return Ok(DEFAULT_CORE) };
    let v = v.to_string_lossy();
    let parsed = v.split_once(',').and_then(|(x, y)| Some((x.trim().parse().ok()?, y.trim().parse().ok()?)));
    parsed.ok_or_else(|| Error::InvalidArgument(format!("NUMEROBIS_TT_CORE={v}: expected x,y")))
}


struct Dram {
    free: BTreeMap<u64, u64>,
}

impl Dram {
    fn alloc(&mut self, size: u64) -> Option<u64> {
        let size = size.max(1).next_multiple_of(ALIGN);
        let (&start, &len) = self.free.iter().find(|&(_, &len)| len >= size)?;
        self.free.remove(&start);
        if len > size {
            self.free.insert(start + size, len - size);
        }
        Some(start)
    }

    fn release(&mut self, start: u64, size: u64) {
        let (mut start, mut size) = (start, size.max(1).next_multiple_of(ALIGN));
        if let Some((&s, &l)) = self.free.range(..start).next_back()
            && s + l == start
        {
            self.free.remove(&s);
            start = s;
            size += l;
        }
        if let Some(l) = self.free.remove(&(start + size)) {
            size += l;
        }
        self.free.insert(start, size);
    }
}

struct State {
    noc: Noc,
    dram: Dram,
    next: u64,
    /// id -> (DRAM address, bytes)
    bufs: HashMap<u64, (u64, u64)>,
    kernels: Vec<OpSig>,
    downloads: HashMap<u64, Vec<u8>>,
    seq: u32,
}

pub struct TtBackend {
    features: FeatureSet,
    core: (u16, u16),
    _card: Card,
    state: Mutex<State>,
}

fn xy(x: u16, y: u16) -> u32 {
    ((y as u32) << 6) | x as u32
}

/// One backend per card per process (the core is a shared resource).
static OPEN: Mutex<Vec<(String, Weak<TtBackend>)>> = Mutex::new(Vec::new());

/// Tries every `/dev/tenstorrent/N`; errors are kept for diagnostics.
pub fn probe() -> Vec<Result<Device>> {
    let Ok(dir) = std::fs::read_dir("/dev/tenstorrent") else { return Vec::new() };
    let mut paths: Vec<String> = dir
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .filter(|p| p.file_name().and_then(|n| n.to_str()).is_some_and(|n| n.bytes().all(|b| b.is_ascii_digit())))
        .map(|p| p.display().to_string())
        .collect();
    paths.sort();
    paths
        .into_iter()
        .map(|p| {
            let mut open = OPEN.lock().unwrap_or_else(|e| e.into_inner());
            open.retain(|(_, w)| w.strong_count() > 0);
            if let Some(b) = open.iter().find(|(q, _)| *q == p).and_then(|(_, w)| w.upgrade()) {
                return Ok(Device::from_backend(b));
            }
            let b = Arc::new(TtBackend::open(&p)?);
            open.push((p, Arc::downgrade(&b)));
            Ok(Device::from_backend(b))
        })
        .collect()
}

/// Every card that opens on the raw path.
pub fn open_all() -> Vec<Device> {
    probe().into_iter().filter_map(Result::ok).collect()
}

impl TtBackend {
    pub fn open(path: &str) -> Result<TtBackend> {
        if !super::card::arc_alive(path) {
            return Err(Error::DeviceLost);
        }
        let card = Card::open(path)?;
        if card.arch != Arch::Blackhole {
            return Err(Error::Unsupported("Wormhole on the raw path (Blackhole only so far)".into()));
        }
        let mut noc = Noc::new(&card)?;
        let core = core_from_env()?;
        let (x, y) = core;
        if noc.read32(x, y, SOFT_RESET)? == u32::MAX {
            return Err(Error::DeviceLost);
        }

        // Load the image while all five cores are held in reset.
        noc.write32(x, y, SOFT_RESET, RESET_ALL)?;
        let elf = Elf::parse(BLACKHOLE)?;
        for s in elf.sections.iter().filter(|s| s.flags & 2 != 0 && s.size > 0) {
            if s.addr + s.size > MAILBOX {
                return Err(Error::BadBinary(format!("{} overlaps the mailbox", s.name)));
            }
            if s.kind == 8 {
                noc.write(x, y, s.addr, &vec![0; s.size as usize])?;
            } else {
                noc.write(x, y, s.addr, s.data)?;
            }
        }
        noc.write(x, y, MAILBOX, &[0; 16])?;
        noc.write32(x, y, SOFT_RESET, RUN_BRISC)?;

        let (a, b, c) = card.driver;
        let features = FeatureSet {
            name: format!("Tenstorrent Blackhole ({}, core {x}-{y})", card.pci),
            class: DeviceClass::Accel,
            driver: format!("tenstorrent {a}.{b}.{c} (raw tt-kmd)"),
            engines: vec![(Engine::Matrix, Access::Raw), (Engine::Copy, Access::Raw)],
            tiers: vec![Tier::Emulated, Tier::Tt(TtTier::Blackhole)],
        };
        let mut free = BTreeMap::new();
        free.insert(DRAM_BASE, DRAM_END - DRAM_BASE);
        Ok(TtBackend {
            features,
            core,
            _card: card,
            state: Mutex::new(State {
                noc,
                dram: Dram { free },
                next: 0,
                bufs: HashMap::new(),
                kernels: Vec::new(),
                downloads: HashMap::new(),
                seq: 0,
            }),
        })
    }

    fn lock(&self) -> MutexGuard<'_, State> {
        self.state.lock().unwrap_or_else(|e| e.into_inner())
    }

    fn fence(s: &mut State) -> Fence {
        s.next += 1;
        Fence(s.next)
    }

    /// Runs one launch to completion on the core.
    fn launch(&self, s: &mut State, symbol: &str, args: &[Arg], items: u32) -> Result<()> {
        let kernel = SYMBOLS.iter().position(|&k| k == symbol).ok_or_else(|| Error::BadBinary(format!("missing kernel {symbol}")))?;
        if args.len() > MAX_ARGS {
            return Err(Error::InvalidArgument(format!("{symbol}: too many arguments")));
        }
        // Stage every distinct buffer argument in L1.
        let mut bufs: Vec<(u64, u64, u64, u64)> = Vec::new(); // (id, l1, dram, bytes)
        let mut words = [0u32; MAX_ARGS];
        let mut at = STAGING;
        for (i, a) in args.iter().enumerate() {
            words[i] = match *a {
                Arg::Buf(b) => {
                    if let Some(&(_, l1, _, _)) = bufs.iter().find(|e| e.0 == b.0) {
                        l1 as u32
                    } else {
                        let &(dram, bytes) = s.bufs.get(&b.0).ok_or_else(|| Error::InvalidArgument("unknown buffer".into()))?;
                        let l1 = at;
                        at = (at + bytes).next_multiple_of(ALIGN);
                        if at > L1_SIZE || bufs.len() == MAX_BUFS {
                            return Err(Error::Unsupported(format!("{symbol}: buffers exceed the {} KiB L1 staging area", (L1_SIZE - STAGING) >> 10)));
                        }
                        bufs.push((b.0, l1, dram, bytes));
                        l1 as u32
                    }
                }
                Arg::U32(v) => v,
                Arg::I32(v) => v as u32,
                Arg::F32(v) => v.to_bits(),
                #[allow(unreachable_patterns)]
                _ => return Err(Error::InvalidArgument("unsupported argument".into())),
            };
        }
        let mut mb: Vec<u32> = vec![kernel as u32, items, args.len() as u32];
        mb.extend_from_slice(&words);
        mb.push(bufs.len() as u32);
        let dram = xy(DRAM_XY.0, DRAM_XY.1);
        for &(_, l1, addr, bytes) in &bufs {
            // Writes back every staged buffer (inputs come back unchanged).
            mb.extend_from_slice(&[l1 as u32, addr as u32, (addr >> 32) as u32, bytes as u32, dram, 1]);
        }
        let bytes: Vec<u8> = mb.iter().flat_map(|w| w.to_le_bytes()).collect();
        let (x, y) = self.core;
        s.noc.write(x, y, MAILBOX + MB_KERNEL, &bytes)?;
        s.seq = s.seq.wrapping_add(1);
        s.noc.write32(x, y, MAILBOX + MB_SEQ, s.seq)?;
        let t0 = Instant::now();
        // Scalar soft-float on one core: generous, scales with the work.
        let timeout = Duration::from_secs(60);
        loop {
            let done = s.noc.read32(x, y, MAILBOX + MB_DONE)?;
            if done == s.seq {
                break;
            }
            // All ones: the card stopped answering. Every further MMIO read
            // would stall a CPU core for the PCIe completion timeout.
            if done == u32::MAX {
                return Err(Error::DeviceLost);
            }
            if t0.elapsed() > timeout {
                return Err(Error::Timeout(timeout));
            }
            std::thread::yield_now();
        }
        match s.noc.read32(x, y, MAILBOX + MB_STATUS)? {
            0 => Ok(()),
            e => Err(Error::BadBinary(format!("{symbol}: runtime status {e}"))),
        }
    }
}

impl Backend for TtBackend {
    fn features(&self) -> &FeatureSet {
        &self.features
    }

    fn alloc(&self, bytes: usize) -> Result<BufId> {
        let mut s = self.lock();
        let addr = s.dram.alloc(bytes as u64).ok_or(Error::OutOfMemory { requested: bytes })?;
        s.next += 1;
        let id = s.next;
        s.bufs.insert(id, (addr, bytes as u64));
        Ok(BufId(id))
    }

    fn free(&self, id: BufId) {
        let mut s = self.lock();
        if let Some((addr, bytes)) = s.bufs.remove(&id.0) {
            s.dram.release(addr, bytes);
        }
    }

    fn upload(&self, id: BufId, offset: usize, src: &[u8]) -> Result<Fence> {
        let mut s = self.lock();
        let &(addr, bytes) = s.bufs.get(&id.0).ok_or_else(|| Error::InvalidArgument("unknown buffer".into()))?;
        if (offset + src.len()) as u64 > bytes {
            return Err(Error::InvalidArgument("upload out of bounds".into()));
        }
        if !offset.is_multiple_of(4) {
            return Err(Error::InvalidArgument("upload offset must be 4-byte aligned".into()));
        }
        // Whole words: the tail word is read-modify-written.
        let mut data = src.to_vec();
        let tail = data.len() % 4;
        if tail != 0 {
            let at = addr + offset as u64 + (data.len() - tail) as u64;
            let mut w = [0u8; 4];
            s.noc.read(DRAM_XY.0, DRAM_XY.1, at, &mut w)?;
            data.extend_from_slice(&w[tail..]);
        }
        s.noc.write(DRAM_XY.0, DRAM_XY.1, addr + offset as u64, &data)?;
        Ok(Self::fence(&mut s))
    }

    fn download(&self, id: BufId, offset: usize, len: usize) -> Result<Fence> {
        let mut s = self.lock();
        let &(addr, bytes) = s.bufs.get(&id.0).ok_or_else(|| Error::InvalidArgument("unknown buffer".into()))?;
        if (offset + len) as u64 > bytes {
            return Err(Error::InvalidArgument("download out of bounds".into()));
        }
        let mut d = vec![0; len];
        s.noc.read(DRAM_XY.0, DRAM_XY.1, addr + offset as u64, &mut d)?;
        let f = Self::fence(&mut s);
        s.downloads.insert(f.0, d);
        Ok(f)
    }

    fn take_download(&self, fence: Fence, dst: &mut [u8]) -> Result<()> {
        let d = self.lock().downloads.remove(&fence.0).ok_or_else(|| Error::InvalidArgument("unknown download".into()))?;
        if d.len() != dst.len() {
            return Err(Error::InvalidArgument("download size mismatch".into()));
        }
        dst.copy_from_slice(&d);
        Ok(())
    }

    fn supports(&self, key: &KernelKey) -> bool {
        key.tier == Tier::Emulated && crate::runtime::abi::supported(key.sig)
    }

    fn load_kernel(&self, key: &KernelKey) -> Result<KernelId> {
        if !self.supports(key) {
            return Err(Error::Unsupported(format!("{} on {}", key.name, key.tier.name())));
        }
        let mut s = self.lock();
        s.kernels.push(key.sig);
        Ok(KernelId(s.kernels.len() as u64 - 1))
    }

    fn dispatch(&self, kernel: KernelId, args: &[Arg]) -> Result<Fence> {
        let mut s = self.lock();
        let sig = *s.kernels.get(kernel.0 as usize).ok_or_else(|| Error::InvalidArgument("unknown kernel".into()))?;
        for l in lower(sig, args)? {
            if l.items > 0 {
                self.launch(&mut s, l.symbol, &l.args, l.items)?;
            }
        }
        Ok(Self::fence(&mut s))
    }

    fn wait(&self, _: Fence, _: Duration) -> Result<()> {
        // Every call above completes before it returns.
        Ok(())
    }

    fn elapsed(&self, _: Fence) -> Option<Duration> {
        None
    }
}

impl Drop for TtBackend {
    fn drop(&mut self) {
        let (x, y) = self.core;
        let s = self.state.get_mut().unwrap_or_else(|e| e.into_inner());
        let _ = s.noc.write32(x, y, SOFT_RESET, RESET_ALL);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dram_allocator_merges() {
        let mut d = Dram { free: BTreeMap::from([(0, 1024)]) };
        let a = d.alloc(1).unwrap();
        let b = d.alloc(100).unwrap();
        assert_eq!((a, b), (0, 64));
        d.release(a, 1);
        d.release(b, 100);
        assert_eq!(d.free, BTreeMap::from([(0, 1024)]));
    }
}
