//! Reusable command lists, the exactness rule (display vs. game state),
//! dma-buf sharing and sync with Vulkan-compatible kernel objects.

#![allow(
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap,
    clippy::cast_precision_loss,
    clippy::cast_sign_loss,
    clippy::many_single_char_names,
    reason = "test data is built with bit-level casts; shapes are named as in the math"
)]

mod common;

use std::sync::Arc;
#[cfg(target_os = "linux")]
use std::time::Duration;

use common::*;
use numerobis::elem::*;
use numerobis::engine::DeviceClass;
use numerobis::mock::MockBackend;
use numerobis::ops::{self, Act, Conv2dShape};
use numerobis::{Bind, Choice, Device, Exactness, GemmShape, Tier};
#[cfg(target_os = "linux")]
use std::os::fd::AsFd as _;

/// A device whose top tier is a vendor matrix tier (not bit-exact for GEMM).
fn vendor_device() -> Device {
    let t = Tier::Tt(numerobis::tt::Tier::Blackhole);
    Device::from_backend(Arc::new(MockBackend::with_tiers(
        DeviceClass::Gpu,
        vec![Tier::Emulated, t],
        vec![Tier::Emulated, t],
    )))
}

#[test]
fn command_list_replays_with_swapped_buffers() {
    let dev = Device::mock();
    let mut s = dev.stream();
    let shape = Conv2dShape::new(1, 2, 6, 6, 3, 3, 3).pad(1, 1);
    let mut r = Rng(5);
    let (x0, w) = (
        s.alloc::<F32>(shape.input_len()).unwrap(),
        s.upload::<F32>(&r.vec(shape.weight_len(), 0.5)).unwrap(),
    );
    let mut y0 = s.alloc::<F32>(shape.output_len().unwrap()).unwrap();
    let mut z0 = s.alloc::<Bf16>(y0.len()).unwrap();
    let cl = s
        .record(|s| {
            ops::conv2d(s, Choice::Up, shape, &x0, &w, None, None, &mut y0)?.wait()?;
            ops::act(s, Choice::Up, Act::Gelu, &y0, &mut z0)?.wait()
        })
        .unwrap();
    assert_eq!(cl.len(), 2);
    assert!(cl.is_exact());
    for frame in 0..3 {
        let input = r.vec(shape.input_len(), 1.0);
        let x = s.upload::<F32>(&input).unwrap();
        let z = s.alloc::<Bf16>(y0.len()).unwrap();
        s.replay(
            &cl,
            &[Bind::new(&x0, &x).unwrap(), Bind::new(&z0, &z).unwrap()],
        )
        .unwrap()
        .wait()
        .unwrap();
        // Same result as running the ops directly.
        let mut y = s.alloc::<F32>(y0.len()).unwrap();
        let mut z2 = s.alloc::<Bf16>(y0.len()).unwrap();
        ops::conv2d(&mut s, Choice::Up, shape, &x, &w, None, None, &mut y)
            .unwrap()
            .wait()
            .unwrap();
        ops::act(&mut s, Choice::Up, Act::Gelu, &y, &mut z2)
            .unwrap()
            .wait()
            .unwrap();
        assert_eq!(bytes(&mut s, &z), bytes(&mut s, &z2), "frame {frame}");
    }
    // Wrong length binds and downloads while recording are refused.
    let short = s.alloc::<F32>(4).unwrap();
    assert!(Bind::new(&x0, &short).is_err());
    s.begin_recording().unwrap();
    assert!(s.download(&w).is_err());
    s.end_recording().unwrap();
}

#[test]
fn bit_streams_never_use_or_read_approximate_tiers() {
    let dev = vendor_device();
    let mut s = dev.stream();
    assert_eq!(s.exactness(), Exactness::Bit);
    let a = s.upload::<F32>(&[1.0, 2.0, 3.0, 4.0]).unwrap();
    let mut c = s.alloc::<F32>(4).unwrap();
    // Auto tier on a Bit stream: GEMM falls back to the exact Emulated tier.
    ops::gemm(
        &mut s,
        Choice::Up,
        GemmShape::new(2, 2, 2),
        1.0,
        &a,
        &a,
        0.0,
        &mut c,
    )
    .unwrap()
    .wait()
    .unwrap();
    assert!(c.is_exact());
    // Forcing the vendor tier is refused.
    let t = Tier::Tt(numerobis::tt::Tier::Blackhole);
    assert!(
        ops::gemm(
            &mut s,
            Choice::Down(t),
            GemmShape::new(2, 2, 2),
            1.0,
            &a,
            &a,
            0.0,
            &mut c
        )
        .is_err()
    );
    // Exact on every tier (format conversion) is fine even when forced.
    let mut h = s.alloc::<Bf16>(4).unwrap();
    ops::convert(
        &mut s,
        Choice::Down(t),
        &a,
        &mut h,
        numerobis::Overflow::Saturate,
    )
    .unwrap()
    .wait()
    .unwrap();
    assert!(h.is_exact());

    // A display stream picks the vendor tier and marks the result approximate,
    let mut d = dev.stream();
    d.set_exactness(Exactness::Approx);
    let mut shown = d.alloc::<F32>(4).unwrap();
    ops::gemm(
        &mut d,
        Choice::Up,
        GemmShape::new(2, 2, 2),
        1.0,
        &a,
        &a,
        0.0,
        &mut shown,
    )
    .unwrap()
    .wait()
    .unwrap();
    assert!(!shown.is_exact());
    // and the taint follows the data through exact ops.
    let mut relu = d.alloc::<F32>(4).unwrap();
    ops::act(&mut d, Choice::Up, Act::Relu, &shown, &mut relu)
        .unwrap()
        .wait()
        .unwrap();
    assert!(!relu.is_exact());
    // Game state cannot read it ...
    let mut state = s.alloc::<F32>(4).unwrap();
    assert!(ops::act(&mut s, Choice::Up, Act::Relu, &relu, &mut state).is_err());
    // ... until it is overwritten from the host.
    s.write::<F32>(&relu, &[0.0; 4]).unwrap();
    ops::act(&mut s, Choice::Up, Act::Relu, &relu, &mut state)
        .unwrap()
        .wait()
        .unwrap();

    // Recorded lists follow the same rule.
    let cl = d
        .record(|d| {
            ops::gemm(
                d,
                Choice::Up,
                GemmShape::new(2, 2, 2),
                1.0,
                &a,
                &a,
                0.0,
                &mut shown,
            )
            .map(drop)
        })
        .unwrap();
    assert!(!cl.is_exact());
    assert!(s.replay(&cl, &[]).is_err());
    d.replay(&cl, &[]).unwrap().wait().unwrap();
}

/// dma-bufs and sync files are Linux kernel objects.
#[cfg(target_os = "linux")]
#[test]
fn dmabuf_shares_memory_between_devices() {
    let (d1, d2) = (Device::mock(), Device::mock());
    let (mut s1, mut s2) = (d1.stream(), d2.stream());
    let a = s1.upload::<F32>(&[1.0, -2.0, 3.0, -4.0]).unwrap();
    let fd = a.export_dmabuf().unwrap();
    let b = s2
        .import_dmabuf::<F32>(std::os::fd::AsFd::as_fd(&fd), 4)
        .unwrap();
    assert_eq!(download(&mut s2, &b), [1.0, -2.0, 3.0, -4.0]);
    // Device 2 writes in place; device 1 sees it without a copy.
    let mut tmp = s2.alloc::<F32>(4).unwrap();
    ops::act(&mut s2, Choice::Up, Act::Relu, &b, &mut tmp)
        .unwrap()
        .wait()
        .unwrap();
    ops::act(&mut s2, Choice::Up, Act::Relu, &tmp, &mut { b })
        .unwrap()
        .wait()
        .unwrap();
    assert_eq!(download(&mut s1, &a), [1.0, 0.0, 3.0, 0.0]);
    // Too small a dma-buf is rejected.
    assert!(
        s2.import_dmabuf::<F32>(std::os::fd::AsFd::as_fd(&fd), 1 << 20)
            .is_err()
    );
}

/// dma-bufs and sync files are Linux kernel objects.
#[cfg(target_os = "linux")]
#[test]
fn timeline_and_sync_file_interop() {
    let Ok(tl) = numerobis::sync::Timeline::new() else {
        eprintln!("skipped: no DRM render node with timeline syncobjs");
        return;
    };
    let dev = Device::mock();
    let mut s = dev.stream();
    let x = s.upload::<F32>(&[1.0; 64]).unwrap();
    let mut y = s.alloc::<F32>(64).unwrap();
    let p = ops::act(&mut s, Choice::Up, Act::Exp, &x, &mut y).unwrap();
    p.signal_timeline(&tl, 1).unwrap();
    tl.wait(1, Duration::from_secs(5)).unwrap();
    assert!(tl.value().unwrap() >= 1);
    // The signalled point as a sync_file (Vulkan SYNC_FD), already signalled.
    let sf = tl.sync_file(1).unwrap();
    assert!(sf.is_signaled().unwrap());
    // Export / import the syncobj (Vulkan OPAQUE_FD timeline semaphore).
    let fd = tl.export_fd().unwrap();
    let tl2 = numerobis::sync::Timeline::import_fd(std::os::fd::AsFd::as_fd(&fd)).unwrap();
    assert_eq!(tl2.value().unwrap(), tl.value().unwrap());
    // Vulkan -> device: the stream's next submission waits for point 2,
    // which another thread signals later (wait-before-signal).
    let tl3 = tl.clone();
    let t = std::thread::spawn(move || {
        std::thread::sleep(Duration::from_millis(50));
        tl3.signal(2).unwrap();
    });
    s.wait_timeline(&tl, 2);
    let t0 = std::time::Instant::now();
    ops::act(&mut s, Choice::Up, Act::Relu, &x, &mut y)
        .unwrap()
        .wait()
        .unwrap();
    assert!(t0.elapsed() >= Duration::from_millis(40));
    t.join().unwrap();
    // A sync_file imported into a timeline point.
    tl.import_sync_file(3, &tl.sync_file(2).unwrap()).unwrap();
    tl.wait(3, Duration::from_secs(1)).unwrap();
    // Signalling backwards is refused.
    assert!(
        ops::act(&mut s, Choice::Up, Act::Relu, &x, &mut y)
            .unwrap()
            .signal_timeline(&tl, 1)
            .is_err()
    );
}

/// dma-bufs and sync files are Linux kernel objects.
#[cfg(target_os = "linux")]
#[test]
fn host_dmabuf_api() {
    use numerobis::dmabuf::{self, HostDmabuf};
    use std::os::fd::AsFd;
    let h = HostDmabuf::new(4096 + 8).unwrap();
    assert_eq!(h.len(), 4104);
    h.write(4096, &[1, 2, 3, 4, 5, 6, 7, 8]).unwrap();
    // A second mapping of the same memory (as another process or API would see it).
    let fd = h.try_clone_fd().unwrap();
    assert!(dmabuf::size(fd.as_fd()).unwrap() >= 4104);
    let other = HostDmabuf::map(fd.as_fd(), 4104).unwrap();
    assert_eq!(other.is_dmabuf(), h.is_dmabuf());
    let mut b = [0u8; 8];
    other.read(4096, &mut b).unwrap();
    assert_eq!(b, [1, 2, 3, 4, 5, 6, 7, 8]);
    // Out of range is an error, not a panic; too-large maps are refused.
    assert!(h.read(4100, &mut b).is_err());
    assert!(HostDmabuf::map(fd.as_fd(), 1 << 30).is_err());
    // PRIME on something that is not a DRM device fails cleanly.
    assert!(dmabuf::prime_import(fd.as_fd(), fd.as_fd()).is_err());
    assert!(dmabuf::prime_export(fd.as_fd(), 1).is_err());
}

/// dma-bufs and sync files are Linux kernel objects.
#[cfg(target_os = "linux")]
#[test]
fn host_dmabuf_into_device_without_copy() {
    use numerobis::dmabuf::HostDmabuf;
    let frame = HostDmabuf::new(16).unwrap();
    let vals: Vec<u8> = [-1.0f32, 2.0, -3.0, 4.0]
        .iter()
        .flat_map(|v| v.to_le_bytes())
        .collect();
    frame.write(0, &vals).unwrap();
    let dev = Device::mock();
    let mut s = dev.stream();
    let x = s.import_dmabuf::<F32>(frame.as_fd(), 4).unwrap();
    assert_eq!(download(&mut s, &x), [-1.0, 2.0, -3.0, 4.0]);
    let mut y = s.alloc::<F32>(4).unwrap();
    ops::act(&mut s, Choice::Up, Act::Relu, &x, &mut y)
        .unwrap()
        .wait()
        .unwrap();
    ops::act(&mut s, Choice::Up, Act::Relu, &y, &mut { x })
        .unwrap()
        .wait()
        .unwrap();
    let mut out = [0u8; 16];
    frame.read(0, &mut out).unwrap();
    assert_eq!(
        &out[..8],
        &[0.0f32.to_le_bytes(), 2.0f32.to_le_bytes()].concat()[..]
    );
}

/// PRIME against a real DRM driver: a dumb buffer on the first primary node
/// that hands one out (any KMS driver: amdgpu, i915, xe, nouveau, v3d),
/// exported as a dma-buf, mapped and written through it, and imported on the
/// render node as another GEM handle. Skipped without such a node.
#[cfg(target_os = "linux")]
#[test]
fn prime_round_trip_on_a_real_drm_device() {
    use numerobis::dmabuf::{self, HostDmabuf};
    use std::fs::OpenOptions;
    use std::os::fd::AsFd;
    use std::os::unix::io::AsRawFd;

    /// `struct drm_mode_create_dumb`.
    #[repr(C)]
    #[derive(Default)]
    struct CreateDumb {
        height: u32,
        width: u32,
        bpp: u32,
        flags: u32,
        handle: u32,
        pitch: u32,
        size: u64,
    }
    /// `struct drm_gem_close`.
    #[repr(C)]
    struct GemClose {
        handle: u32,
        pad: u32,
    }
    const CREATE_DUMB: libc::c_ulong = 0xC020_64B2; // DRM_IOWR(0xB2, 32 bytes)
    const DESTROY_DUMB: libc::c_ulong = 0xC004_64B4; // DRM_IOWR(0xB4, 4 bytes)
    const GEM_CLOSE: libc::c_ulong = 0x4008_6409; // DRM_IOW(0x09, 8 bytes)

    let open = |path: &str| OpenOptions::new().read(true).write(true).open(path).ok();
    let nodes = |prefix: &'static str| {
        (0..16).filter_map(move |i| open(&format!("/dev/dri/{prefix}{i}")).map(|f| (i, f)))
    };
    let dumb = nodes("card").find_map(|(i, card)| {
        let mut c = CreateDumb {
            height: 64,
            width: 64,
            bpp: 32,
            ..Default::default()
        };
        // SAFETY: CreateDumb matches DRM_IOCTL_MODE_CREATE_DUMB.
        let ok = unsafe { libc::ioctl(card.as_raw_fd(), CREATE_DUMB, &raw mut c) } == 0;
        ok.then_some((i, card, c))
    });
    let Some((index, card, buf)) = dumb else {
        eprintln!("skipped: no DRM primary node gives this user a dumb buffer");
        return;
    };
    let len = usize::try_from(buf.size).unwrap();
    assert!(len >= 64 * 64 * 4);

    // GEM handle -> dma-buf, mapped by the CPU through the dma-buf itself.
    let fd = dmabuf::prime_export(card.as_fd(), buf.handle).unwrap();
    assert!(dmabuf::size(fd.as_fd()).unwrap() >= len);
    let map = HostDmabuf::map(fd.as_fd(), len).unwrap();
    assert!(
        map.is_dmabuf(),
        "card{index}: an exported buffer is a dma-buf"
    );
    let pattern: Vec<u8> = (0..=255).collect();
    map.write(len - 256, &pattern).unwrap();
    let mut back = vec![0; 256];
    HostDmabuf::map(fd.as_fd(), len)
        .unwrap()
        .read(len - 256, &mut back)
        .unwrap();
    assert_eq!(back, pattern);

    // Re-importing on the same node gives the same handle back (the kernel
    // keeps one handle per buffer and file); on a render node a new one.
    assert_eq!(
        dmabuf::prime_import(card.as_fd(), fd.as_fd()).unwrap(),
        buf.handle
    );
    if let Some((_, render)) = nodes("renderD").next() {
        let handle = dmabuf::prime_import(render.as_fd(), fd.as_fd()).unwrap();
        assert_ne!(handle, 0);
        let mut close = GemClose { handle, pad: 0 };
        // SAFETY: GemClose matches DRM_IOCTL_GEM_CLOSE; we own the handle.
        assert_eq!(
            unsafe { libc::ioctl(render.as_raw_fd(), GEM_CLOSE, &raw mut close) },
            0
        );
    }
    let mut destroy = buf.handle;
    // SAFETY: `struct drm_mode_destroy_dumb` is one u32 handle, ours.
    unsafe { libc::ioctl(card.as_raw_fd(), DESTROY_DUMB, &raw mut destroy) };
}
