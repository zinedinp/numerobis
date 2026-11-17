/* Matrix interface, Emulated tier: 1x1 fragments, one work item per C
 * element, decode by integer bit logic and an F32 FMA chain.
 *
 * nb_xmx.h, nb_amd.h and nb_nv.h provide the same names with real
 * fragments (DPAS, MFMA/WMMA, mma.sync); gemm.cl only talks to this
 * interface. TODO(llvm): portable matrix intrinsics instead of this layer
 * (docs/llvm-todo.md). */
#ifndef NB_EMULATED_H
#define NB_EMULATED_H

#include "nb_block.h"

typedef float nb_frag_a;
typedef float nb_frag_b;
typedef float nb_frag_acc;

#define NB_FRAG_M 1
#define NB_FRAG_N 1

NB_INLINE nb_frag_acc nb_mma(nb_frag_acc acc, nb_frag_a a, nb_frag_b b) { return nb_fma(a, b, acc); }

/* Loads element (row, k) of a K-contiguous operand with `rows` rows. */
NB_INLINE nb_frag_a nb_load_tile(const __global uchar *p, uint d, uint row, uint kk, uint K, uint len, uint soff) {
    return nb_load_any(p, d, row * K + kk, len, soff);
}

#endif /* NB_EMULATED_H */
