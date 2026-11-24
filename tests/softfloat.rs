//! `kernels/tt/softfloat.c` (the float arithmetic of the Tensix RISC-V
//! kernels) must match the host FPU bit for bit: fma, add, sub, mul, div,
//! int conversions and compares on random patterns incl. subnormals and
//! specials.

mod common;

use std::path::PathBuf;
use std::process::Command;

#[test]
fn softfloat_matches_hardware_fpu() {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    // With the suffix: on Windows `Command` appends `.exe` to a path
    // without an extension.
    let exe = PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join(format!(
        "nb-softfloat-{}{}",
        std::process::id(),
        std::env::consts::EXE_SUFFIX
    ));
    let cc = common::hostcl::cc();
    let mut cmd = Command::new(&cc);
    cmd.args([
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
    .arg(&exe);
    // The MSVC C runtime has the math functions built in; no libm.
    if !cfg!(target_env = "msvc") {
        cmd.arg("-lm");
    }
    let out = cmd
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
