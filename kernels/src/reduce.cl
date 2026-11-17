/* absmax_block_<src>: dst[g] = max |src[g*block .. (g+1)*block]|, NaN ignored.
 * One work item per block. */
#include "nb_block.h"

__kernel void nb_absmax_block(const __global uchar *src, __global uchar *dst, uint len, uint block, uint d,
                              uint soff) {
    uint g = NB_GID();
    if (g * block >= len) return;
    float m = 0.0f;
    for (uint i = g * block; i < (g + 1) * block; i++) {
        float a = nb_fabs(nb_load_any(src, d, i, len, soff));
        if (a > m) m = a;
    }
    ((__global float *)dst)[g] = m;
}
