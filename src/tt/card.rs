//! One Tenstorrent card: device node, TLB windows into the NoC address
//! space, and word/bulk access to any NoC endpoint (Tensix L1 and registers,
//! DRAM).
//!
//! A 2 MiB TLB window maps `[addr, addr + 2 MiB)` of one NoC endpoint into
//! BAR space; [`Noc`] keeps one window and retargets it on demand.

use std::sync::Arc;

use super::sys::*;
use crate::runtime::error::{Error, Result};
use crate::runtime::os::{Fd, Map, RW};

/// Card generations (PCI device id).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Arch {
    Wormhole,
    Blackhole,
}

pub const TLB_2M: u64 = 2 << 20;

/// ARC heartbeat from tt-kmd's sysfs telemetry for `/dev/tenstorrent/N`.
/// All ones (or no telemetry) means the card's management firmware is not
/// answering: the NoC must not be touched then, because every MMIO read to a
/// hung card stalls the reading CPU core for the PCIe completion timeout.
///
/// The sysfs read is itself such an MMIO read in tt-kmd (~220 ms in kernel
/// mode on a hung card), so a dead result is cached for a minute per process.
pub fn arc_alive(path: &str) -> bool {
    use std::collections::HashMap;
    use std::sync::Mutex;
    use std::time::{Duration, Instant};
    static DEAD: Mutex<Option<HashMap<String, Instant>>> = Mutex::new(None);
    let mut dead = DEAD.lock().unwrap_or_else(|e| e.into_inner());
    let dead = dead.get_or_insert_with(HashMap::new);
    if dead.get(path).is_some_and(|t| t.elapsed() < Duration::from_secs(60)) {
        return false;
    }
    let n = path.rsplit('/').next().unwrap_or("0");
    let alive = std::fs::read_to_string(format!("/sys/class/tenstorrent/tenstorrent!{n}/tt_heartbeat"))
        .ok()
        .and_then(|s| s.trim().parse::<u64>().ok())
        .is_some_and(|v| v != u32::MAX as u64);
    if alive {
        dead.remove(path);
    } else {
        dead.insert(path.to_string(), Instant::now());
    }
    alive
}

/// Board name from sysfs (`p150a`, `p100a`, ...), if the firmware reports one.
pub fn card_type(path: &str) -> Option<String> {
    let n = path.rsplit('/').next().unwrap_or("0");
    let t = std::fs::read_to_string(format!("/sys/class/tenstorrent/tenstorrent!{n}/tt_card_type")).ok()?;
    let t = t.trim();
    (!t.is_empty() && t != "unknown").then(|| t.to_string())
}

pub struct Card {
    pub(crate) fd: Arc<Fd>,
    pub arch: Arch,
    pub device_id: u16,
    pub pci: String,
    pub driver: (u8, u8, u8),
}

impl Card {
    pub fn open(path: &str) -> Result<Card> {
        let fd = Fd::open(path)?;
        let mut info = get_device_info { in_output_size_bytes: 20, ..Default::default() };
        call(&fd, GET_DEVICE_INFO, &mut info)?;
        let arch = match info.device_id {
            0x401e => Arch::Wormhole,
            0xb140 => Arch::Blackhole,
            d => return Err(Error::Unsupported(format!("Tenstorrent device id {d:#x}"))),
        };
        let mut drv = get_driver_info { in_output_size_bytes: 12, ..Default::default() };
        call(&fd, GET_DRIVER_INFO, &mut drv)?;
        let b = info.bus_dev_fn;
        let pci = format!("{:04x}:{:02x}:{:02x}.{}", info.pci_domain, b >> 8, (b >> 3) & 0x1f, b & 7);
        Ok(Card { fd: Arc::new(fd), arch, device_id: info.device_id, pci, driver: (drv.major, drv.minor, drv.patch) })
    }

    /// Allocates a TLB window of `size` bytes (2 MiB on Blackhole/Wormhole).
    pub fn window(&self, size: u64) -> Result<Window> {
        let mut a = allocate_tlb { size, ..Default::default() };
        call(&self.fd, ALLOCATE_TLB, &mut a)?;
        let uc = Map::new(0, size as usize, RW, libc::MAP_SHARED, self.fd.0, a.mmap_offset_uc);
        let wc = Map::new(0, size as usize, RW, libc::MAP_SHARED, self.fd.0, a.mmap_offset_wc);
        match (uc, wc) {
            (Ok(uc), Ok(wc)) => Ok(Window { fd: self.fd.clone(), id: a.id, size, uc, wc, target: None }),
            (Err(e), _) | (_, Err(e)) => {
                let mut f = free_tlb { id: a.id };
                let _ = call(&self.fd, FREE_TLB, &mut f);
                Err(e)
            }
        }
    }
}

/// A TLB window; retargeted with [`Window::point`].
pub struct Window {
    fd: Arc<Fd>,
    id: u32,
    size: u64,
    uc: Map,
    wc: Map,
    target: Option<(u16, u16, u64)>,
}

impl Window {
    /// Points the window at `(x, y)` NoC0, base `addr` (multiple of the window size).
    pub fn point(&mut self, x: u16, y: u16, addr: u64) -> Result<()> {
        debug_assert!(addr.is_multiple_of(self.size));
        if self.target == Some((x, y, addr)) {
            return Ok(());
        }
        let mut c = configure_tlb {
            id: self.id,
            config: noc_tlb_config { addr, x_end: x, y_end: y, ordering: ORDER_STRICT, ..Default::default() },
            ..Default::default()
        };
        call(&self.fd, CONFIGURE_TLB, &mut c)?;
        self.target = Some((x, y, addr));
        Ok(())
    }
}

impl Drop for Window {
    fn drop(&mut self) {
        let mut f = free_tlb { id: self.id };
        let _ = call(&self.fd, FREE_TLB, &mut f);
    }
}

/// Word and bulk access to NoC endpoints through one retargeted window.
pub struct Noc {
    w: Window,
}

impl Noc {
    pub fn new(card: &Card) -> Result<Noc> {
        Ok(Noc { w: card.window(TLB_2M)? })
    }

    fn at(&mut self, x: u16, y: u16, addr: u64, len: usize) -> Result<usize> {
        let base = addr & !(TLB_2M - 1);
        let off = (addr - base) as usize;
        if off + len > TLB_2M as usize {
            return Err(Error::InvalidArgument("NoC access crosses a 2 MiB window".into()));
        }
        self.w.point(x, y, base)?;
        Ok(off)
    }

    pub fn read32(&mut self, x: u16, y: u16, addr: u64) -> Result<u32> {
        let off = self.at(x, y, addr, 4)?;
        Ok(self.w.uc.read32(off))
    }

    pub fn write32(&mut self, x: u16, y: u16, addr: u64, v: u32) -> Result<()> {
        let off = self.at(x, y, addr, 4)?;
        self.w.uc.write32(off, v);
        Ok(())
    }

    /// Bulk write (write-combined, 4-byte granular; `src.len()` padded with zeros).
    pub fn write(&mut self, x: u16, y: u16, addr: u64, src: &[u8]) -> Result<()> {
        let mut done = 0;
        while done < src.len() {
            let a = addr + done as u64;
            let room = (TLB_2M - (a & (TLB_2M - 1))) as usize;
            let n = room.min(src.len() - done);
            let off = self.at(x, y, a, n.next_multiple_of(4))?;
            let chunk = &src[done..done + n];
            let (words, tail) = chunk.as_chunks::<4>();
            for (i, w) in words.iter().enumerate() {
                self.w.wc.write32(off + i * 4, u32::from_le_bytes(*w));
            }
            if !tail.is_empty() {
                let mut last = [0u8; 4];
                last[..tail.len()].copy_from_slice(tail);
                self.w.wc.write32(off + words.len() * 4, u32::from_le_bytes(last));
            }
            std::sync::atomic::fence(std::sync::atomic::Ordering::SeqCst);
            done += n;
        }
        Ok(())
    }

    /// Bulk read (4-byte granular).
    pub fn read(&mut self, x: u16, y: u16, addr: u64, dst: &mut [u8]) -> Result<()> {
        let mut done = 0;
        while done < dst.len() {
            let a = addr + done as u64;
            let room = (TLB_2M - (a & (TLB_2M - 1))) as usize;
            let n = room.min(dst.len() - done);
            let off = self.at(x, y, a, n.next_multiple_of(4))?;
            for i in (0..n).step_by(4) {
                let w = self.w.uc.read32(off + i).to_le_bytes();
                let k = (n - i).min(4);
                dst[done + i..done + i + k].copy_from_slice(&w[..k]);
            }
            done += n;
        }
        Ok(())
    }
}
