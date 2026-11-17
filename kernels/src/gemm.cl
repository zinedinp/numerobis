/* gemm[_strided]_<a>_<b>_<acc>: C = alpha*A*B^T + beta*C. A is MxK, B is
 * NxK (both K-contiguous), C is MxN, batches back to back. One work item per
 * C element. With beta == 0, C is not read.
 *
 * The matrix interface comes from nb_emulated.h here; the vendor tiers swap
 * in nb_xmx.h / nb_amd.h / nb_nv.h with the same names. */
#include "nb_emulated.h"

__kernel void nb_gemm_f(const __global uchar *a, const __global uchar *b, __global uchar *c, uint m, uint n,
                        uint k, uint batch, float alpha, float beta, uint fa, uint fb, uint fc, uint sa, uint sb) {
    uint g = NB_GID();
    if (g >= batch * m * n) return;
    uint z = g / (m * n), r = g % (m * n), i = r / n, j = r % n;
    uint la = batch * m * k, lb = batch * n * k;
    nb_frag_acc s = 0.0f;
    for (uint p = 0; p < k; p++)
        s = nb_mma(s, nb_load_tile(a, fa, z * m + i, p, k, la, sa), nb_load_tile(b, fb, z * n + j, p, k, lb, sb));
    float out = beta != 0.0f ? alpha * s + beta * nb_load_f(c, fc, g) : alpha * s;
    nb_store_f(c, fc, g, out, 0);
}

/* Integer inputs, I32 accumulator, wrapping arithmetic (done in uint). */
__kernel void nb_gemm_i(const __global uchar *a, const __global uchar *b, __global uchar *c, uint m, uint n,
                        uint k, uint batch, int alpha, int beta, uint fa, uint fb, uint fc, uint sa, uint sb) {
    uint g = NB_GID();
    if (g >= batch * m * n) return;
    uint z = g / (m * n), r = g % (m * n), i = r / n, j = r % n;
    uint s = 0;
    for (uint p = 0; p < k; p++)
        s += (uint)nb_load_i(a, fa, (z * m + i) * k + p) * (uint)nb_load_i(b, fb, (z * n + j) * k + p);
    uint out = (uint)alpha * s;
    if (beta != 0) out += (uint)beta * (uint)nb_load_i(c, fc, g);
    ((__global int *)c)[g] = (int)out;
    (void)sa;
    (void)sb;
}
