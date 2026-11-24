//! The raw AMD backend (KFD, AQL) on whatever AMD GPU this machine has.
//! `tests/emulated.rs` runs every shared kernel against the mock oracle on
//! every device of `Device::all`, this one included; here are the parts
//! that are the backend's own: opening, copies, many launches through the
//! signal and kernarg slots, timing, dma-bufs, reopening, placement.
//! Skipped without an AMD GPU that KFD lists.

#![cfg(all(feature = "amd", target_os = "linux", target_pointer_width = "64"))]
#![allow(
    clippy::cast_precision_loss,
    clippy::many_single_char_names,
    reason = "test data counts up in floats; GEMM operands are named as in the math"
)]

mod common;

use std::os::fd::AsFd;
use std::time::Duration;

use common::*;
use numerobis::amd::backend::CopyEngine;
use numerobis::elem::*;
use numerobis::{Choice, Device, GemmShape, Placement, ops};

fn amd() -> Option<Device> {
    let mut found = None;
    for r in numerobis::amd::backend::probe() {
        match r {
            Ok(d) => found = found.or(Some(d)),
            Err(e) => eprintln!("amd: {e}"),
        }
    }
    if found.is_none() {
        eprintln!("skipped: no AMD GPU opens on the raw path");
    }
    found
}

#[test]
fn opens_and_round_trips() {
    let Some(dev) = amd() else { return };
    let f = dev.features();
    assert!(f.driver.contains("KFD"), "{}", f.driver);
    let mut s = dev.stream();
    let vals: Vec<f32> = (0..10_000).map(|i| i as f32 * 0.5 - 7.0).collect();
    let x = s.upload::<F32>(&vals).unwrap();
    assert_eq!(download(&mut s, &x), vals);
    // An empty buffer and an empty upload.
    let e = s.upload::<F32>(&[]).unwrap();
    assert_eq!(download(&mut s, &e), Vec::<f32>::new());
}

#[test]
fn a_kernel_runs_and_is_timed() {
    let Some(dev) = amd() else { return };
    let mut s = dev.stream();
    let vals: Vec<f32> = (0..4097).map(|i| i as f32 - 2048.0).collect();
    let x = s.upload::<F32>(&vals).unwrap();
    let mut y = s.alloc::<F32>(vals.len()).unwrap();
    let p = ops::act(&mut s, Choice::Up, ops::Act::Relu, &x, &mut y).unwrap();
    let ((), elapsed) = p.wait_elapsed().unwrap();
    let want: Vec<f32> = vals.iter().map(|v| v.max(0.0)).collect();
    assert_eq!(download(&mut s, &y), want);
    let elapsed = elapsed.expect("the CP writes timestamps into the signal");
    eprintln!("relu over {} values: {elapsed:?}", vals.len());
    assert!(elapsed.as_secs() < 1, "{elapsed:?}");
}

#[test]
fn many_launches_cycle_the_slots() {
    // More launches than signal / kernarg slots (256) and ring packets
    // (1024): each slot is reused
    // only after its fence, and the queue's order holds throughout.
    let Some(dev) = amd() else { return };
    let mut s = dev.stream();
    let x = s.upload::<F32>(&[-1.0, 2.0, -3.0, 4.0]).unwrap();
    let mut a = s.alloc::<F32>(4).unwrap();
    let mut b = s.alloc::<F32>(4).unwrap();
    // Not waited for: the queue keeps the order.
    drop(ops::act(&mut s, Choice::Up, ops::Act::Relu, &x, &mut a).unwrap());
    for _ in 0..640 {
        drop(ops::act(&mut s, Choice::Up, ops::Act::Relu, &a, &mut b).unwrap());
        ops::act(&mut s, Choice::Up, ops::Act::Relu, &b, &mut a)
            .unwrap()
            .wait()
            .unwrap();
    }
    assert_eq!(download(&mut s, &a), [0.0, 2.0, 0.0, 4.0]);
}

#[test]
fn gemm_matches_the_oracle_bit_for_bit() {
    let Some(dev) = amd() else { return };
    let shape = GemmShape {
        m: 37,
        n: 29,
        k: 61,
        batch: 2,
    };
    let mut rng = Rng(7);
    let a = rng.vec(shape.batch * shape.m * shape.k, 3.0);
    let b = rng.vec(shape.batch * shape.n * shape.k, 3.0);
    let run = |dev: &Device| {
        let mut s = dev.stream();
        let da = s.upload::<F32>(&a).unwrap();
        let db = s.upload::<F32>(&b).unwrap();
        let mut c = s.alloc::<F32>(shape.batch * shape.m * shape.n).unwrap();
        ops::gemm(&mut s, Choice::Up, shape, 1.0, &da, &db, 0.0, &mut c)
            .unwrap()
            .wait()
            .unwrap();
        download(&mut s, &c)
    };
    let want = run(&Device::mock());
    // Timed: a larger GEMM, for the order of magnitude.
    let big = GemmShape::new(512, 512, 512);
    let mut s = dev.stream();
    let x = s.upload::<F32>(&vec![0.5; 512 * 512]).unwrap();
    let mut c = s.alloc::<F32>(512 * 512).unwrap();
    let ((), t) = ops::gemm(&mut s, Choice::Up, big, 1.0, &x, &x, 0.0, &mut c)
        .unwrap()
        .wait_elapsed()
        .unwrap();
    if let Some(t) = t {
        eprintln!(
            "gemm f32 512^3 (Emulated): {t:?}, {:.1} GFLOP/s",
            2.0 * 512f64.powi(3) / t.as_secs_f64() / 1e9
        );
    }
    let got = run(&dev);
    for (i, (w, g)) in want.iter().zip(&got).enumerate() {
        assert!(same(*w, *g), "element {i}: oracle {w}, gfx {g}");
    }
}

#[test]
fn dma_bufs_export_and_import() {
    let Some(dev) = amd() else { return };
    let mut s = dev.stream();
    let x = s.upload::<F32>(&[1.0, -2.0, 3.0, -4.0]).unwrap();
    // Out as a dma-buf, back in as a second buffer on the same memory.
    let fd = x.export_dmabuf().unwrap();
    let y = s.import_dmabuf::<F32>(fd.as_fd(), 4).unwrap();
    assert_eq!(download(&mut s, &y), [1.0, -2.0, 3.0, -4.0]);
    let mut z = s.alloc::<F32>(4).unwrap();
    ops::act(&mut s, Choice::Up, ops::Act::Relu, &y, &mut z)
        .unwrap()
        .wait()
        .unwrap();
    assert_eq!(download(&mut s, &z), [1.0, 0.0, 3.0, 0.0]);
    // Too small a dma-buf for the length asked.
    assert!(s.import_dmabuf::<F32>(fd.as_fd(), 1 << 20).is_err());
}

#[test]
fn reopens_after_the_last_device_is_dropped() {
    // KFD keeps the process VM on the render-node file it was first given;
    // a second backend on the same GPU must reuse it.
    let Some(dev) = amd() else { return };
    drop(dev);
    let Some(again) = amd() else {
        panic!("the GPU opened once but not twice");
    };
    let mut s = again.stream();
    let x = s.upload::<F32>(&[5.0]).unwrap();
    assert_eq!(download(&mut s, &x), [5.0]);
}

#[test]
fn placement_finds_a_gpu_without_a_matrix_engine() {
    let Some(dev) = amd() else { return };
    let name = dev.features().name.clone();
    // Fastest prefers matrix engines, but takes this GPU when it is the
    // best there is.
    let picked = Placement::Fastest.pick(&[dev]).unwrap();
    assert_eq!(picked.features().name, name);
}

/// The backend of every copy engine (each opened on its own).
fn engines() -> Vec<(CopyEngine, Device)> {
    let mut v = Vec::new();
    for e in [CopyEngine::Sdma, CopyEngine::DriverSdma, CopyEngine::Cpu] {
        match numerobis::amd::backend::probe_with(e).into_iter().next() {
            Some(Ok(d)) => v.push((e, d)),
            Some(Err(err)) => eprintln!("amd ({e:?}): {err}"),
            None => {}
        }
    }
    v
}

#[test]
fn every_copy_engine_moves_large_buffers() {
    for (e, dev) in engines() {
        assert!(
            dev.features().driver.contains(e.name()),
            "{}",
            dev.features().driver
        );
        let mut s = dev.stream();
        // 48 MiB: twelve SDMA packets of 4 MiB, and a tail.
        let n = (48 << 20) / 4 + 3;
        let mut rng = Rng(e as u64 + 1);
        let vals: Vec<f32> = (0..n).map(|_| rng.f()).collect();
        // Twice: the second time with staging the backend kept.
        for round in ["first", "again"] {
            let t = std::time::Instant::now();
            let x = s.upload::<F32>(&vals).unwrap();
            let back = download(&mut s, &x);
            let took = t.elapsed();
            assert!(back == vals, "{e:?}: round trip differs");
            eprintln!(
                "{e:?} ({round}): 48 MiB up and down in {took:?} ({:.1} GB/s)",
                2.0 * f64::from(n * 4) / took.as_secs_f64() / 1e9
            );
        }
    }
}

#[test]
fn copies_and_kernels_keep_their_order_without_waiting() {
    // Nothing waits on the host until the end: a copy that overwrites a
    // kernel's input must wait for the kernel, a kernel for the copy before
    // it, a download for the kernel that wrote its buffer.
    for (e, dev) in engines() {
        let mut s = dev.stream();
        let n = 1 << 20;
        let a: Vec<f32> = (0..n).map(|i| (i % 7) as f32 - 3.0).collect();
        let b: Vec<f32> = (0..n).map(|i| 2.0 - (i % 5) as f32).collect();
        let x = s.upload::<F32>(&a).unwrap();
        let mut y = s.alloc::<F32>(n).unwrap();
        let mut z = s.alloc::<F32>(n).unwrap();
        drop(ops::act(&mut s, Choice::Up, ops::Act::Relu, &x, &mut y).unwrap());
        s.write(&x, &b).unwrap(); // after the kernel read `a`
        drop(ops::act(&mut s, Choice::Up, ops::Act::Relu, &x, &mut z).unwrap());
        let got_y = s.download(&y).unwrap();
        let got_z = s.download(&z).unwrap();
        let relu = |v: &[f32]| v.iter().map(|v| v.max(0.0)).collect::<Vec<_>>();
        assert!(got_y.wait().unwrap().to_vec() == relu(&a), "{e:?}: y");
        assert!(got_z.wait().unwrap().to_vec() == relu(&b), "{e:?}: z");
    }
}

#[test]
fn copies_are_timed_on_the_gpu_clock() {
    for (e, dev) in engines() {
        let mut s = dev.stream();
        let x = s.upload::<F32>(&vec![1.5; 16 << 20]).unwrap();
        let t = std::time::Instant::now();
        let (back, elapsed) = s.download(&x).unwrap().wait_elapsed().unwrap();
        let wall = t.elapsed();
        assert!(back.to_vec().iter().all(|v| v.to_bits() == 1.5f32.to_bits()));
        match (e, elapsed) {
            // The host copies itself: nothing for the GPU to time.
            (CopyEngine::Cpu, el) => assert_eq!(el, None),
            (_, Some(el)) => {
                eprintln!("{e:?}: 64 MiB down in {el:?} on the GPU, {wall:?} on the host");
                // The same clock as the CP's: within what the host saw.
                assert!(
                    el > Duration::ZERO && el <= wall,
                    "{e:?}: {el:?} vs {wall:?}"
                );
            }
            (_, None) => panic!("{e:?}: no timestamps for an SDMA copy"),
        }
    }
}

/// Over 1 GiB: more than one submission (256 packets of 4 MiB), the
/// timestamps across them, a buffer the 680M's 1 GiB of VRAM can't hold
/// (system memory then). Needs about 6 GiB of RAM:
/// `cargo test --features amd --test amd_raw -- --ignored`.
#[test]
#[ignore = "needs about 6 GiB of RAM"]
fn copies_over_one_gib() {
    for (e, dev) in engines() {
        let mut s = dev.stream();
        let n = ((1 << 30) + (64 << 20)) / 4 + 5;
        let vals: Vec<f32> = (0..n).map(|i| (i % 65_521) as f32).collect();
        let t = std::time::Instant::now();
        let x = s.upload::<F32>(&vals).unwrap();
        let (back, elapsed) = s.download(&x).unwrap().wait_elapsed().unwrap();
        let took = t.elapsed();
        let back = back.to_vec();
        assert_eq!(back.len(), n);
        if let Some(i) = back.iter().zip(&vals).position(|(a, b)| a.to_bits() != b.to_bits()) {
            panic!("{e:?}: element {i} differs");
        }
        eprintln!(
            "{e:?}: {:.2} GiB up and down in {took:?} (download on the GPU: {elapsed:?})",
            (n * 4) as f64 / f64::from(1 << 30)
        );
        if e != CopyEngine::Cpu {
            assert!(
                elapsed.is_some(),
                "{e:?}: the span of several submissions is timed"
            );
        }
    }
}
