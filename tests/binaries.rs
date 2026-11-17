//! Checked-in kernel binaries (`kernels/bin`, built by scripts/build-kernels.sh):
//! every ELF exports all kernel-ABI symbols and imports nothing (no device
//! library is linked).

use std::path::Path;

use numerobis::loader::elf::{Elf, machine};
use numerobis::runtime::abi::SYMBOLS;

fn check(dir: &str, ext: &str, want_machine: u16, kernel_symbol: impl Fn(&str) -> String) -> usize {
    let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("kernels/bin").join(dir);
    let mut n = 0;
    for e in std::fs::read_dir(&dir).unwrap_or_else(|e| panic!("{}: {e}", dir.display())) {
        let path = e.unwrap().path();
        if path.extension().and_then(|x| x.to_str()) != Some(ext) {
            continue;
        }
        let bytes = std::fs::read(&path).unwrap();
        let elf = Elf::parse(&bytes).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
        assert_eq!(elf.machine, want_machine, "{}", path.display());
        let syms = elf.symbols().unwrap();
        for s in SYMBOLS {
            let want = kernel_symbol(s);
            assert!(syms.iter().any(|x| x.name == want && x.shndx != 0), "{}: missing {want}", path.display());
        }
        // `.nv.*` (e.g. `.nv.reservedSmem.offset0` on sm_120) are loader-resolved
        // reserved symbols, not library imports; the raw NVIDIA loader (B.3 step 6)
        // has to patch them.
        let undefined: Vec<_> = syms
            .iter()
            .filter(|s| s.shndx == 0 && !s.name.is_empty() && !s.name.starts_with(".nv."))
            .map(|s| s.name)
            .collect();
        assert!(undefined.is_empty(), "{}: unresolved {undefined:?}", path.display());
        n += 1;
    }
    n
}

#[test]
fn amd_hsaco() {
    // HSACO kernels are found through their kernel descriptor `<name>.kd`.
    assert_eq!(check("amd", "hsaco", machine::AMDGPU, |s| format!("{s}.kd")), 6);
}

#[test]
fn nvidia_cubin() {
    assert_eq!(check("nvidia", "cubin", machine::CUDA, |s| s.to_string()), 5);
}

#[test]
fn spirv_module() {
    let p = Path::new(env!("CARGO_MANIFEST_DIR")).join("kernels/bin/spirv/emulated.spv");
    let b = std::fs::read(&p).unwrap();
    assert_eq!(&b[..4], &0x0723_0203u32.to_le_bytes(), "SPIR-V magic");
    // Entry point names are stored as NUL-terminated strings.
    for s in SYMBOLS {
        let needle = [s.as_bytes(), &[0]].concat();
        assert!(b.windows(needle.len()).any(|w| w == needle), "missing entry point {s}");
    }
}
