//! Tenstorrent on-card memory check (feature `tt`): GDDR6 through every DRAM
//! channel endpoint and the L1 SRAM of the configured Tensix tile, written
//! and read back through TLB windows with address, inverted-address and
//! random patterns; then a round trip through the `Device` API.
//!
//! Skipped (with a note) when no card is present or its ARC heartbeat is
//! dead. Only DRAM endpoints from the Blackhole SoC descriptor and the tile
//! from `NUMEROBIS_TT_CORE` (default 1,2) are touched; never probe others.
//! All 8 channels are tested on p150 boards, channel 0 elsewhere.
//!
//! `NUMEROBIS_TT_MEM_MB=n` sets the bytes tested per DRAM region (default 1).
#![cfg(all(feature = "tt", target_os = "linux"))]

use std::time::Instant;

use numerobis::tt::card::{Card, Noc, arc_alive, card_type};

const PATH: &str = "/dev/tenstorrent/0";
/// First NoC 0 endpoint of each DRAM channel (blackhole_140_arch.yaml).
const DRAM: [(u16, u16); 8] = [
    (0, 0),
    (0, 2),
    (0, 9),
    (0, 5),
    (9, 0),
    (9, 2),
    (9, 9),
    (9, 5),
];
/// Usable bytes per channel (4 GiB minus 16 MiB firmware reserve).
const CHANNEL: u64 = 4_278_190_080;
const SOFT_RESET: u64 = 0xFFB1_21B0;
const RESET_ALL: u32 = (1 << 11) | (1 << 12) | (1 << 13) | (1 << 14) | (1 << 18);

fn pattern(kind: u32, addr: u64, len: usize) -> Vec<u8> {
    let mut r = addr ^ 0x9E37_79B9_7F4A_7C15;
    (0..len / 4)
        .flat_map(|i| {
            let a = (addr + i as u64 * 4) as u32;
            let w = match kind {
                0 => a,
                1 => !a,
                _ => {
                    r ^= r << 13;
                    r ^= r >> 7;
                    r ^= r << 17;
                    r as u32
                }
            };
            w.to_le_bytes()
        })
        .collect()
}

/// Writes and verifies all three patterns at `[addr, addr + len)` of `(x, y)`.
fn check(noc: &mut Noc, what: &str, (x, y): (u16, u16), addr: u64, len: usize) -> f64 {
    let mut back = vec![0u8; len];
    let t = Instant::now();
    for kind in 0..3 {
        let want = pattern(kind, addr, len);
        noc.write(x, y, addr, &want).unwrap();
        noc.read(x, y, addr, &mut back).unwrap();
        if let Some(i) = want.chunks(4).zip(back.chunks(4)).position(|(w, g)| w != g) {
            let (w, g) = (&want[i * 4..i * 4 + 4], &back[i * 4..i * 4 + 4]);
            panic!(
                "{what}: pattern {kind} mismatch at {:#x}: wrote {w:02x?}, read {g:02x?}",
                addr + i as u64 * 4
            );
        }
    }
    (3 * 2 * len) as f64 / t.elapsed().as_secs_f64() / (1 << 20) as f64
}

fn core() -> (u16, u16) {
    std::env::var("NUMEROBIS_TT_CORE")
        .ok()
        .and_then(|v| {
            v.split_once(',')
                .and_then(|(a, b)| Some((a.trim().parse().ok()?, b.trim().parse().ok()?)))
        })
        .unwrap_or((1, 2))
}

#[test]
fn dram_l1_and_device_round_trip() {
    if !std::path::Path::new(PATH).exists() {
        println!("skipped: no {PATH}");
        return;
    }
    if !arc_alive(PATH) {
        println!("skipped: ARC heartbeat dead (reset the card: tt-smi -r)");
        return;
    }
    let card = Card::open(PATH).unwrap();
    let mut noc = Noc::new(&card).unwrap();
    let board = card_type(PATH);
    let channels = if board.as_deref().is_some_and(|b| b.starts_with("p150")) {
        8
    } else {
        1
    };
    let mb: usize = std::env::var("NUMEROBIS_TT_MEM_MB")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(1);
    let len = mb << 20;
    println!("board {board:?}, {channels} DRAM channel(s), {mb} MiB per region");

    for (ch, &ep) in DRAM.iter().take(channels).enumerate() {
        for addr in [64 << 20, CHANNEL / 2, CHANNEL - len as u64] {
            let rate = check(&mut noc, &format!("DRAM ch{ch} {ep:?}"), ep, addr, len);
            println!("DRAM ch{ch} {ep:?} @ {addr:#x}: ok ({rate:.0} MiB/s)");
        }
    }

    // L1: hold the tile's five cores in reset, then test all 1.5 MiB. The
    // tt backend reloads its image the next time it opens the tile.
    let (x, y) = core();
    assert_ne!(
        noc.read32(x, y, SOFT_RESET).unwrap(),
        u32::MAX,
        "tile {x}-{y} does not answer"
    );
    noc.write32(x, y, SOFT_RESET, RESET_ALL).unwrap();
    let rate = check(&mut noc, &format!("L1 {x}-{y}"), (x, y), 0, 0x18_0000);
    println!("L1 {x}-{y}: 1536 KiB ok ({rate:.0} MiB/s)");
    drop(noc);
    drop(card);

    // Through the public API: DRAM buffer upload/download and one kernel that
    // moves data DRAM -> L1 -> DRAM on the core.
    use numerobis::{Choice, Overflow, elem::*, ops};
    let dev = numerobis::tt::backend::open_all()
        .into_iter()
        .next()
        .expect("tt backend did not open");
    let mut s = dev.stream();
    let n = 4 << 20;
    let x: Vec<f32> = (0..n).map(|i| i as f32).collect();
    let b = s.upload::<F32>(&x).unwrap();
    assert_eq!(
        s.download(&b).unwrap().wait().unwrap().to_vec(),
        x,
        "16 MiB device buffer round trip"
    );
    let small = s.upload::<F32>(&x[..4096]).unwrap();
    let mut h = s.alloc::<F16>(4096).unwrap();
    ops::convert(&mut s, Choice::Up, &small, &mut h, Overflow::Saturate)
        .unwrap()
        .wait()
        .unwrap();
    let mut back = s.alloc::<F32>(4096).unwrap();
    ops::convert(&mut s, Choice::Up, &h, &mut back, Overflow::Saturate)
        .unwrap()
        .wait()
        .unwrap();
    let got = s.download(&back).unwrap().wait().unwrap().to_vec();
    for (i, g) in got.iter().enumerate() {
        assert_eq!(*g, half_round(i as f32), "element {i}");
    }
    println!("device API: 16 MiB round trip and F32 -> F16 -> F32 on core ok");
}

/// f32 -> f16 -> f32 for the exactly representable-or-rounded integers used above.
fn half_round(v: f32) -> f32 {
    let mf = miraculix::formats::MiniFloat::F16;
    mf.decode(mf.encode(v, numerobis::Overflow::Saturate))
}
