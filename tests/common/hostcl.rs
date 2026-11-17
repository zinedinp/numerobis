//! Test backend that runs the real Emulated kernels (`kernels/src/*.cl`)
//! on the CPU: the sources are compiled as C with `-DNB_HOST` into
//! `kernels/host/driver.c`, and every launch from `runtime::abi::lower`
//! runs in that process. Same `Backend` contract as a GPU backend, so the
//! whole test suite checks kernel source against the Rust oracle.

use std::collections::HashMap;
use std::io::Write;
use std::path::PathBuf;
use std::process::{Command, Stdio};
use std::sync::{Arc, Mutex, OnceLock};
use std::time::Duration;

use numerobis::engine::{Access, DeviceClass, Engine, FeatureSet};
use numerobis::runtime::abi::{Launch, lower};
use numerobis::runtime::{Arg, Backend, BufId, Fence, KernelId, KernelKey, OpSig};
use numerobis::{Device, Error, Result, Tier};

fn driver() -> &'static PathBuf {
    static EXE: OnceLock<PathBuf> = OnceLock::new();
    EXE.get_or_init(|| {
        let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        let exe = PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join(format!("nb-host-driver-{}", std::process::id()));
        let cc = std::env::var("CC").unwrap_or_else(|_| "clang".into());
        let out = Command::new(&cc)
            .args(["-std=c11", "-O1", "-ffp-contract=off", "-DNB_HOST", "-Wall", "-Werror", "-Wno-unused-function"])
            .arg("-I")
            .arg(root.join("kernels/include"))
            .arg(root.join("kernels/host/driver.c"))
            .arg("-o")
            .arg(&exe)
            .arg("-lm")
            .output()
            .unwrap_or_else(|e| panic!("cannot run {cc} to build the host kernel driver: {e}"));
        assert!(out.status.success(), "host driver build failed:\n{}", String::from_utf8_lossy(&out.stderr));
        exe
    })
}

#[derive(Default)]
struct State {
    next: u64,
    mem: HashMap<u64, Vec<u8>>,
    kernels: Vec<OpSig>,
    downloads: HashMap<u64, Vec<u8>>,
}

pub struct HostCl {
    features: FeatureSet,
    state: Mutex<State>,
}

impl HostCl {
    pub fn device() -> Device {
        Device::from_backend(Arc::new(HostCl {
            features: FeatureSet {
                name: "host-cl".into(),
                class: DeviceClass::Gpu,
                driver: "kernels/host/driver.c".into(),
                engines: vec![(Engine::Matrix, Access::Host)],
                tiers: vec![Tier::Emulated],
            },
            state: Mutex::new(State::default()),
        }))
    }
}

fn run(l: &Launch, mem: &mut HashMap<u64, Vec<u8>>) -> Result<()> {
    let mut input = Vec::new();
    let put = |v: &mut Vec<u8>, x: u32| v.extend_from_slice(&x.to_le_bytes());
    put(&mut input, l.symbol.len() as u32);
    input.extend_from_slice(l.symbol.as_bytes());
    put(&mut input, l.items);
    put(&mut input, l.args.len() as u32);
    let mut bufs = Vec::new();
    for a in &l.args {
        match *a {
            Arg::Buf(b) => {
                let data = mem.get(&b.0).ok_or_else(|| Error::InvalidArgument("unknown buffer".into()))?;
                input.push(0);
                put(&mut input, data.len() as u32);
                input.extend_from_slice(data);
                bufs.push((b.0, data.len()));
            }
            Arg::U32(v) => {
                input.push(1);
                put(&mut input, v);
            }
            Arg::I32(v) => {
                input.push(2);
                put(&mut input, v as u32);
            }
            Arg::F32(v) => {
                input.push(3);
                put(&mut input, v.to_bits());
            }
            #[allow(unreachable_patterns)]
            _ => return Err(Error::InvalidArgument("unknown arg".into())),
        }
    }
    let mut child = Command::new(driver()).stdin(Stdio::piped()).stdout(Stdio::piped()).spawn().map_err(|e| Error::Os(e.raw_os_error().unwrap_or(0)))?;
    child.stdin.take().unwrap().write_all(&input).unwrap();
    let out = child.wait_with_output().unwrap();
    assert!(out.status.success(), "driver failed for {}", l.symbol);
    let mut off = 0;
    for (id, n) in bufs {
        mem.insert(id, out.stdout[off..off + n].to_vec());
        off += n;
    }
    Ok(())
}

impl HostCl {
    fn lock(&self) -> std::sync::MutexGuard<'_, State> {
        self.state.lock().unwrap_or_else(|e| e.into_inner())
    }
}

impl Backend for HostCl {
    fn features(&self) -> &FeatureSet {
        &self.features
    }
    fn alloc(&self, bytes: usize) -> Result<BufId> {
        let mut s = self.lock();
        s.next += 1;
        let id = s.next;
        s.mem.insert(id, vec![0; bytes]);
        Ok(BufId(id))
    }
    fn free(&self, id: BufId) {
        self.lock().mem.remove(&id.0);
    }
    fn upload(&self, id: BufId, offset: usize, src: &[u8]) -> Result<Fence> {
        let mut s = self.lock();
        s.mem.get_mut(&id.0).unwrap()[offset..offset + src.len()].copy_from_slice(src);
        s.next += 1;
        Ok(Fence(s.next))
    }
    fn download(&self, id: BufId, offset: usize, len: usize) -> Result<Fence> {
        let mut s = self.lock();
        let d = s.mem[&id.0][offset..offset + len].to_vec();
        s.next += 1;
        let f = s.next;
        s.downloads.insert(f, d);
        Ok(Fence(f))
    }
    fn take_download(&self, fence: Fence, dst: &mut [u8]) -> Result<()> {
        dst.copy_from_slice(&self.lock().downloads.remove(&fence.0).unwrap());
        Ok(())
    }
    fn supports(&self, key: &KernelKey) -> bool {
        key.tier == Tier::Emulated && numerobis::mock::supported(key.sig)
    }
    fn load_kernel(&self, key: &KernelKey) -> Result<KernelId> {
        let mut s = self.lock();
        s.kernels.push(key.sig);
        Ok(KernelId(s.kernels.len() as u64 - 1))
    }
    fn dispatch(&self, kernel: KernelId, args: &[Arg]) -> Result<Fence> {
        let mut s = self.lock();
        let sig = s.kernels[kernel.0 as usize];
        for l in lower(sig, args)? {
            run(&l, &mut s.mem)?;
        }
        s.next += 1;
        Ok(Fence(s.next))
    }
    fn wait(&self, _: Fence, _: Duration) -> Result<()> {
        Ok(())
    }
    fn elapsed(&self, _: Fence) -> Option<Duration> {
        None
    }
}
