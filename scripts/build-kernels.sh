#!/usr/bin/env bash
# Builds the Emulated kernels (kernels/src/all.cl) for every target, and the
# cooperative matrix GEMMs (kernels/vulkan). Output is checked in under
# kernels/bin, so no toolchain is needed at run time.
#
#   scripts/build-kernels.sh           build into kernels/bin
#   scripts/build-kernels.sh --check   rebuild into a temp dir, fail if kernels/bin differs
#
# Needs: clang (amdgcn, nvptx, spirv64, riscv32 targets) and lld, ptxas for
# cubins, glslc (Vulkan SDK) for kernels/vulkan, spirv-val (optional). Intel
# Gen ISA via ocloc is added once ocloc is available; until then Intel loads
# the SPIR-V. Environment: CLANG, PTXAS, GLSLC.
set -euo pipefail

root=$(cd "$(dirname "$0")/.." && pwd)
CLANG=${CLANG:-clang}
PTXAS=${PTXAS:-ptxas}
GLSLC=${GLSLC:-glslc}

# gfx10-3-generic: one code object for all of RDNA 2 (gfx1030..gfx1036, incl.
# the Rembrandt / Van Gogh / Raphael APUs), loaded by the raw `amd` backend.
AMD_ARCHS=(gfx908 gfx90a gfx942 gfx950 gfx10-3-generic gfx1100 gfx1200)
NV_ARCHS=(sm_80 sm_89 sm_90 sm_100 sm_120)

check=0
[[ ${1:-} == --check ]] && check=1

out="$root/kernels/bin"
if ((check)); then
    out=$(mktemp -d)
    trap 'rm -rf "$out"' EXIT
fi

src="$root/kernels/src/all.cl"
common=(-cl-std=CL2.0 -O3 -nogpulib -ffp-contract=off -cl-fp32-correctly-rounded-divide-sqrt
    -I "$root/kernels/include" -Wall -Werror -Wno-unused-function)

mkdir -p "$out/amd" "$out/nvidia" "$out/spirv" "$out/tt"

# clang links amdgcn with ld.lld; without it the rust-lld of the Rust
# toolchain does the same when it is called by that name.
if ! command -v ld.lld >/dev/null; then
    rust_lld=$(ls "$HOME"/.rustup/toolchains/*/lib/rustlib/*/bin/rust-lld 2>/dev/null | head -1 || true)
    if [[ -n $rust_lld ]]; then
        lld_dir=$(mktemp -d)
        ln -s "$rust_lld" "$lld_dir/ld.lld"
        export PATH="$lld_dir:$PATH"
    fi
fi

for a in "${AMD_ARCHS[@]}"; do
    "$CLANG" "${common[@]}" -target amdgcn-amd-amdhsa -mcpu="$a" "$src" -o "$out/amd/$a-emulated.hsaco"
done

if command -v "$PTXAS" >/dev/null; then
    for a in "${NV_ARCHS[@]}"; do
        ptx="$out/nvidia/$a-emulated.ptx"
        # No jump tables: they become .nv.global.init data plus a .nv.constant4
        # relocation, which the raw loader would have to link.
        "$CLANG" "${common[@]}" -fno-jump-tables -target nvptx64-nvidia-cuda -march="$a" -S "$src" -o "$ptx"
        "$PTXAS" -arch="$a" "$ptx" -o "$out/nvidia/$a-emulated.cubin"
        rm "$ptx"
    done
else
    echo "warning: $PTXAS not found, skipping NVIDIA cubins" >&2
fi

"$CLANG" "${common[@]}" -target spirv64 -c "$src" -o "$out/spirv/emulated.spv"

# Matrix engines through Vulkan cooperative matrices (kernels/vulkan): one
# SPIR-V per input / accumulator type, as examples/matrix_probe runs them.
if command -v "$GLSLC" >/dev/null; then
    #        name  acc  A_T        C_T        S_T
    while read -r a acc at ct st; do
        "$GLSLC" --target-env=vulkan1.3 -fshader-stage=compute -O \
            -DA_T="$at" -DC_T="$ct" -DS_T="$st" "$root/kernels/vulkan/gemm_coopmat.comp" \
            -o "$out/spirv/gemm_coopmat_${a}_${a}_${acc}.spv"
    done <<'TYPES'
f16 f32 float16_t float float
f16 f16 float16_t float16_t float
i8 i32 int8_t int32_t int
u8 i32 uint8_t uint32_t int
TYPES
else
    echo "warning: $GLSLC not found, skipping the cooperative matrix kernels" >&2
fi

# Tenstorrent: the same sources as freestanding C for the Tensix BRISC core
# (RV32IM, no FPU: kernels/tt/softfloat.c), linked at L1 address 0 with the
# mailbox runtime (kernels/tt/runtime.c).
"$CLANG" --target=riscv32-unknown-elf -march=rv32im -mabi=ilp32 -std=c11 -O2 -ffreestanding -nostdlib \
    -ffp-contract=off -fno-jump-tables -ffunction-sections -DNB_HOST -Wall -Werror -Wno-unused-function \
    -I "$root/kernels/include" -fuse-ld=lld -Wl,-T,"$root/kernels/tt/tensix.ld" -Wl,--gc-sections -Wl,--build-id=none \
    -o "$out/tt/blackhole-emulated.elf" "$root/kernels/tt/runtime.c" "$root/kernels/tt/softfloat.c"
if command -v spirv-val >/dev/null; then
    spirv-val --target-env spv1.4 "$out/spirv/emulated.spv"
    for f in "$out"/spirv/gemm_coopmat_*.spv; do
        [[ -e $f ]] && spirv-val --target-env vulkan1.3 "$f"
    done
fi

if ((check)); then
    if ! diff -r "$root/kernels/bin" "$out" >/dev/null; then
        diff -rq "$root/kernels/bin" "$out" || true
        echo "kernels/bin is stale: run scripts/build-kernels.sh" >&2
        exit 1
    fi
    echo "kernels/bin is up to date"
else
    du -ach "$out"/*/* | tail -1
fi
