//! `kernels/tt/softfloat.c` (the float arithmetic of the Tensix RISC-V
//! kernels) must match the host FPU bit for bit: fma, add, sub, mul, div,
//! int conversions and compares on random patterns incl. subnormals and
//! specials.

use std::path::PathBuf;
use std::process::Command;

#[test]
fn softfloat_matches_hardware_fpu() {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let exe = PathBuf::from(env!("CARGO_TARGET_TMPDIR"))
        .join(format!("nb-softfloat-{}", std::process::id()));
    let cc = std::env::var("CC").unwrap_or_else(|_| "clang".into());
    let out = Command::new(&cc)
        .args([
            "-std=c11",
            "-O2",
            "-ffp-contract=off",
            "-Wall",
            "-Werror",
            "-Wno-unused-function",
            "-I",
        ])
        .arg(root.join("kernels/tt"))
        .arg(root.join("kernels/tt/softfloat_test.c"))
        .arg("-o")
        .arg(&exe)
        .arg("-lm")
        .output()
        .unwrap_or_else(|e| panic!("cannot run {cc}: {e}"));
    assert!(
        out.status.success(),
        "{}",
        String::from_utf8_lossy(&out.stderr)
    );
    let run = Command::new(&exe).arg("1000000").output().unwrap();
    assert!(
        run.status.success(),
        "{}",
        String::from_utf8_lossy(&run.stdout)
    );
}
