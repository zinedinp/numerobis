#!/usr/bin/env bash
# Builds the Emulated kernels (kernels/src/all.cl) for every SIMT target
# (plan A.3). Output is checked in under kernels/bin, so no toolchain is
# needed at run time.
#
#   scripts/build-kernels.sh           build into kernels/bin
#   scripts/build-kernels.sh --check   rebuild into a temp dir, fail if kernels/bin differs
#
# Needs: clang (amdgcn, nvptx, spirv64 targets), ptxas for cubins, spirv-val
# (optional). Intel Gen ISA via ocloc is added once ocloc is available; until
# then Intel loads the SPIR-V. Environment: CLANG, PTXAS.
set -euo pipefail

root=$(cd "$(dirname "$0")/.." && pwd)
CLANG=${CLANG:-clang}
PTXAS=${PTXAS:-ptxas}

AMD_ARCHS=(gfx908 gfx90a gfx942 gfx950 gfx1100 gfx1200)
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

mkdir -p "$out/amd" "$out/nvidia" "$out/spirv"

for a in "${AMD_ARCHS[@]}"; do
    "$CLANG" "${common[@]}" -target amdgcn-amd-amdhsa -mcpu="$a" "$src" -o "$out/amd/$a-emulated.hsaco"
done

if command -v "$PTXAS" >/dev/null; then
    for a in "${NV_ARCHS[@]}"; do
        ptx="$out/nvidia/$a-emulated.ptx"
        "$CLANG" "${common[@]}" -target nvptx64-nvidia-cuda -march="$a" -S "$src" -o "$ptx"
        "$PTXAS" -arch="$a" "$ptx" -o "$out/nvidia/$a-emulated.cubin"
        rm "$ptx"
    done
else
    echo "warning: $PTXAS not found, skipping NVIDIA cubins" >&2
fi

"$CLANG" "${common[@]}" -target spirv64 -c "$src" -o "$out/spirv/emulated.spv"
if command -v spirv-val >/dev/null; then
    spirv-val --target-env spv1.4 "$out/spirv/emulated.spv"
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
