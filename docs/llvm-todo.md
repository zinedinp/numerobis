# LLVM / rustc TODOs

What numerobis builds itself today is the prototype and the test oracle for a
later upstream version. Code marks each spot with `// TODO(llvm): <short>` (or
`/* TODO(llvm) */` in kernel sources) and links here.

| TODO | Prototype in numerobis | Test vectors | LLVM area |
|---|---|---|---|
| MiniFloat IR types / `llvm.fptrunc.sat.*`, `llvm.fpext.*` for FP8/FP6/FP4 with a soft lowering | `kernels/include/nb_codec.h` (`nb_mf_encode` / `nb_mf_decode`) | miraculix `formats` tests, `tests/emulated.rs` | IR, `APFloat`, generic lowering |
| Block-scaled formats (MX, NVFP4, BFP): generic quantize / dequantize intrinsics, native on gfx950 and sm_100 | `kernels/include/nb_block.h`, `kernels/src/quantize.cl` | `tests/quantize.rs`, `tests/emulated.rs` | IR intrinsics, AMDGPU, NVPTX |
| Portable matrix interface instead of the `nb_mma` header layer (MFMA/WMMA, mma/wgmma, `SPV_KHR_cooperative_matrix`, `SPV_INTEL_subgroup_matrix_multiply_accumulate`) | `kernels/include/nb_emulated.h` (+ `nb_xmx.h`, `nb_amd.h`, `nb_nv.h`) | `tests/gemm.rs` | Clang matrix extension, targets |
| SPIR-V backend: OpenCL C -> Vulkan shader SPIR-V (removes clspv) | `scripts/build-kernels.sh` | lavapipe runs | SPIR-V backend |
| V3D QPU backend (Raspberry Pi 4/5) | `src/v3d/qpu.rs` (B.3 step 8) | Pi oracle runs | new target |
| Tenstorrent RISC-V vendor extension (`Xtt...`, SFPI) | `src/tt` (B.3 step 9) | tt oracle runs | RISC-V MC layer |
| Upstream the AIE backend (`llvm-aie` / Peano) | `src/npu/xdna` | XDNA oracle runs | new target |
| Missing target intrinsics found while building tiers (gfx12 WMMA FP8, sm_100/sm_120 block-scale MMA, Xe2 DPAS FP8) | per vendor header | per tier | AMDGPU, NVPTX, SPIR-V |
| rustc: kernels in Rust (`core::arch::{amdgpu,nvptx}` matrix intrinsics, FP8/FP6/FP4 primitives, `spirv` target) | - | whole suite | rustc, stdarch |

Not solvable through LLVM: Ethos-U, RKNPU and Hailo command streams are not
ISA targets; they stay Rust codegen in numerobis.
