//! Test backend that runs the real Emulated kernels (`kernels/src/*.cl`)
//! on the CPU: the sources are compiled as C with `-DNB_HOST` into
//! `kernels/host/driver.c`, and every launch from `runtime::abi::lower`
//! runs in that process. Same `Backend` contract as a GPU backend, so the
//! whole test suite checks kernel source against the Rust oracle.
//!
//! [`HostCl::rv32`] builds the Tenstorrent flavour instead: the same
//! sources for RV32IM with `kernels/tt/softfloat.c` (exactly the code of
//! `kernels/bin/tt`), driven by `kernels/tt/qemu_driver.c` under
//! `qemu-riscv32`.

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

/// `clang` from the `PATH`; on Windows, without one there, the one where the
/// LLVM installer puts it (`%ProgramFiles%\LLVM\bin`), since the installer
/// leaves the `PATH` alone by default.
pub fn clang() -> String {
    static CLANG: OnceLock<String> = OnceLock::new();
    CLANG
        .get_or_init(|| {
            let on_path = Command::new("clang")
                .arg("--version")
                .output()
                .is_ok_and(|o| o.status.success());
            if !on_path && cfg!(windows) {
                let installed = std::env::var_os("ProgramFiles")
                    .map(|p| PathBuf::from(p).join(r"LLVM\bin\clang.exe"))
                    .filter(|p| p.is_file());
                if let Some(p) = installed {
                    return p.display().to_string();
                }
            }
            "clang".into()
        })
        .clone()
}

/// The host C compiler: `$CC`, else [`clang`].
pub fn cc() -> String {
    std::env::var("CC").unwrap_or_else(|_| clang())
}

fn build(cc: &str, args: &[&str], what: &str) {
    let out = Command::new(cc).args(args).output().unwrap_or_else(|e| {
        panic!("cannot run {cc} to build the {what}: {e} (install LLVM, or set CC)")
    });
    assert!(
        out.status.success(),
        "{what} build failed:\n{}",
        String::from_utf8_lossy(&out.stderr)
    );
}

fn rv32_driver() -> &'static PathBuf {
    static EXE: OnceLock<PathBuf> = OnceLock::new();
    EXE.get_or_init(|| {
        let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        let exe = PathBuf::from(env!("CARGO_TARGET_TMPDIR"))
            .join(format!("nb-rv32-driver-{}", std::process::id()));
        let inc = root.join("kernels/include");
        let (drv, sf) = (
            root.join("kernels/tt/qemu_driver.c"),
            root.join("kernels/tt/softfloat.c"),
        );
        build(
            &clang(),
            &[
                "--target=riscv32-unknown-linux-gnu",
                "-march=rv32im",
                "-mabi=ilp32",
                "-std=c11",
                "-O2",
                "-ffreestanding",
                "-fno-builtin",
                "-fno-stack-protector",
                "-nostdlib",
                "-static",
                "-ffp-contract=off",
                "-fno-jump-tables",
                "-DNB_HOST",
                "-Wall",
                "-Werror",
                "-Wno-unused-function",
                "-fuse-ld=lld",
                "-I",
                inc.to_str().unwrap(),
                "-o",
                exe.to_str().unwrap(),
                drv.to_str().unwrap(),
                sf.to_str().unwrap(),
            ],
            "RV32 kernel driver",
        );
        exe
    })
}

/// Is `qemu-riscv32` on PATH?
pub fn have_qemu_rv32() -> bool {
    Command::new("qemu-riscv32")
        .arg("--version")
        .output()
        .is_ok_and(|o| o.status.success())
}

fn driver() -> &'static PathBuf {
    static EXE: OnceLock<PathBuf> = OnceLock::new();
    EXE.get_or_init(|| {
        let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        // With the suffix: on Windows `Command` appends `.exe` to a path
        // without an extension.
        let exe = PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join(format!(
            "nb-host-driver-{}{}",
            std::process::id(),
            std::env::consts::EXE_SUFFIX
        ));
        let cc = cc();
        let (inc, drv) = (
            root.join("kernels/include"),
            root.join("kernels/host/driver.c"),
        );
        let mut args = vec![
            "-std=c11",
            "-O1",
            "-ffp-contract=off",
            "-DNB_HOST",
            "-Wall",
            "-Werror",
            "-Wno-unused-function",
            "-I",
            inc.to_str().unwrap(),
            drv.to_str().unwrap(),
            "-o",
            exe.to_str().unwrap(),
        ];
        // The MSVC C runtime has the math functions built in; no libm.
        if !cfg!(target_env = "msvc") {
            args.push("-lm");
        }
        build(&cc, &args, "host kernel driver");
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
    rv32: bool,
    state: Mutex<State>,
}

impl HostCl {
    fn make(name: &str, driver: &str, rv32: bool) -> Device {
        Device::from_backend(Arc::new(HostCl {
            features: FeatureSet {
                name: name.into(),
                class: DeviceClass::Gpu,
                driver: driver.into(),
                engines: vec![(Engine::Matrix, Access::Host)],
                tiers: vec![Tier::Emulated],
            },
            rv32,
            state: Mutex::new(State::default()),
        }))
    }

    /// Kernels compiled for the host CPU.
    pub fn device() -> Device {
        Self::make("host-cl", "kernels/host/driver.c", false)
    }

    /// Kernels compiled for the Tensix RISC-V cores (RV32IM + soft-float), under qemu.
    pub fn rv32() -> Device {
        Self::make("rv32-qemu", "kernels/tt/qemu_driver.c", true)
    }
}

fn run(l: &Launch, mem: &mut HashMap<u64, Vec<u8>>, rv32: bool) -> Result<()> {
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
                let data = mem
                    .get(&b.0)
                    .ok_or_else(|| Error::InvalidArgument("unknown buffer".into()))?;
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
    let mut cmd = if rv32 {
        let mut c = Command::new("qemu-riscv32");
        c.arg(rv32_driver());
        c
    } else {
        Command::new(driver())
    };
    let mut child = cmd
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .spawn()
        .map_err(|e| Error::Os(e.raw_os_error().unwrap_or(0)))?;
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
        self.state
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
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
            run(&l, &mut s.mem, self.rv32)?;
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
