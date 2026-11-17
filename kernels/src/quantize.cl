/* quantize_<to> / dequantize_<from>: F32 <-> MX, NVFP4, BFP_B (GGUF decode
 * only). One work item per block. Ports of miraculix mx_quantize,
 * nvfp4_quantize, bfp_encode_rows (row_scalar). */
#include "nb_block.h"

NB_INLINE float nb_amax(const __global float *x, uint n) {
    float m = 0.0f;
    for (uint i = 0; i < n; i++) {
        float a = nb_fabs(x[i]);
        if (a > m) m = a; /* NaN never wins */
    }
    return m;
}

/* MXINT8 element: round_even(clamp(v * 64, -127, 127)) as i32 as u8; NaN -> 0. */
NB_INLINE uint nb_mxint8(float v) {
    float c = v * 64.0f;
    c = c < -127.0f ? -127.0f : c > 127.0f ? 127.0f : c;
    if (c != c) return 0;
    const float M = 12582912.0f; /* 1.5 * 2^23 */
    float r = (c + M) - M;
    return (uint)(uchar)(int)r;
}

NB_INLINE void nb_quantize_mx(const __global float *x, __global uchar *out, __global uchar *scale, uint d) {
    float amax = nb_amax(x, NB_MX_BLOCK);
    uint s;
    if (amax == 0.0f || !(amax - amax == 0.0f)) { /* zero or not finite */
        s = 127;
    } else {
        int emax = NB_MX_INT8(d) ? 0 : nb_mf_emax(d);
        int e = ((int)(nb_f2u(amax) >> 23) - 127) - emax;
        e += 127;
        s = (uint)(e < 0 ? 0 : e > 254 ? 254 : e);
    }
    *scale = (uchar)s;
    float inv = 1.0f / nb_e8m0(s);
    uint w = nb_mx_w(d);
    for (uint i = 0; i < NB_MX_BLOCK; i++) {
        float v = x[i] * inv;
        nb_or_bits(out, i, w, NB_MX_INT8(d) ? nb_mxint8(v) : nb_mf_encode(d, v, 1));
    }
}

NB_INLINE void nb_quantize_nvfp4(const __global float *x, __global uchar *out, __global uchar *scale, float ts) {
    float amax = nb_amax(x, NB_NVFP4_BLOCK);
    uint sc = nb_mf_encode(NB_DESC_E4M3FN, amax / 6.0f / ts, 1);
    *scale = (uchar)sc;
    float real = nb_mf_decode(NB_DESC_E4M3FN, sc) * ts;
    for (uint i = 0; i < NB_NVFP4_BLOCK; i++)
        nb_or_bits(out, i, 4, real == 0.0f ? 0u : nb_mf_encode(NB_DESC_E2M1, x[i] / real, 1));
}

NB_INLINE void nb_quantize_bfp(const __global float *x, __global uchar *out, __global uchar *exp_out, uint d) {
    uint bits = NB_BFP_BITS(d), shift = 24 - bits, limit = (1u << bits) - 1;
    uint shared = 0;
    for (uint i = 0; i < NB_BFP_ROW; i++) {
        uint e = (nb_f2u(x[i]) >> 23) & 255u;
        if (e > shared) shared = e;
    }
    *exp_out = (uchar)shared;
    for (uint i = 0; i < NB_BFP_ROW; i++) {
        uint w = nb_f2u(x[i]);
        uint exponent = (w >> 23) & 255u, diff = shared - exponent;
        uint m = (w & 0x7FFFFFu) | 0x800000u;
        m = diff >= 32 ? 0 : m >> diff;
        m = (m + ((1u << (shift - 1)) - 1) + ((m >> shift) & 1u)) >> shift;
        m = exponent != 0 ? (m < limit ? m : limit) : 0;
        nb_or_bits(out, i, bits + 1, m != 0 ? (m | ((w >> 31) << bits)) : 0u);
    }
}

/* NVFP4 per-tensor scale amax / (6 * 448), written to the tensor-scale slot
 * at `ts_off`. Runs as a single work item before nb_quantize.
 * TODO: tree reduction; one work item is fine for the Emulated tier. */
__kernel void nb_nvfp4_tensor_scale(const __global uchar *src, __global uchar *dst, uint len, uint ts_off) {
    if (NB_GID() != 0) return;
    float amax = nb_amax((const __global float *)src, len);
    float ts = amax == 0.0f ? 1.0f : amax / (6.0f * 448.0f);
    uint b = nb_f2u(ts);
    for (uint k = 0; k < 4; k++) dst[ts_off + k] = (uchar)(b >> (8 * k));
}

__kernel void nb_quantize(const __global uchar *src, __global uchar *dst, uint len, uint d, uint soff) {
    uint g = NB_GID(), B = nb_block_len(d);
    if (g * B >= len) return;
    const __global float *x = (const __global float *)src + g * B;
    __global uchar *out = dst + g * nb_block_bytes(d);
    for (uint k = 0; k < nb_block_bytes(d); k++) out[k] = 0;
    switch (NB_KIND(d)) {
    case NB_KIND_MX: nb_quantize_mx(x, out, dst + soff + g, d); break;
    case NB_KIND_NVFP4: nb_quantize_nvfp4(x, out, dst + soff + g, nb_read_f32(dst + soff + len / NB_NVFP4_BLOCK)); break;
    case NB_KIND_BFP: nb_quantize_bfp(x, out, dst + soff + g, d); break;
    default: break;
    }
}

__kernel void nb_dequantize(const __global uchar *src, __global uchar *dst, uint len, uint d, uint soff) {
    uint g = NB_GID(), B = nb_block_len(d);
    if (g * B >= len) return;
    __global float *y = (__global float *)dst;
    for (uint i = g * B; i < (g + 1) * B; i++) y[i] = nb_load_block(src, d, i, len, soff);
}
