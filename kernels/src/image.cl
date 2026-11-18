/* Ops beyond GEMM: activations, direct 2-D convolution, resize.
 *
 * All three read and write element-wise float formats (F32, TF32, F16, BF16,
 * FP8) through nb_load_f / nb_store_f and accumulate in F32 with a fixed
 * order, so they are bit-exact against src/mock/kernels/{act,conv,resize}.rs.
 * NaN results are stored as the canonical quiet NaN (0x7FC00000).
 * Convolution is direct (no im2col buffer): one work item per output. */
#include "nb_math.h"

/* act_<kind>_<src>_<dst>: dst[i] = act(src[i]). One work item per element. */
__kernel void nb_act(const __global uchar *src, __global uchar *dst, uint len, uint act, float p0, float p1, uint fs,
                     uint fd) {
    uint i = NB_GID();
    if (i >= len) return;
    nb_store_f(dst, fd, i, nb_canon(nb_activate(act, nb_load_f(src, fs, i), p0, p1)), 0);
}

/* Parameter block of nb_conv2d (u32 each), uploaded by ops::conv2d. */
#define CV_N 0
#define CV_C 1
#define CV_H 2
#define CV_W 3
#define CV_K 4
#define CV_R 5
#define CV_S 6
#define CV_P 7
#define CV_Q 8
#define CV_SH 9
#define CV_SW 10
#define CV_PH 11
#define CV_PW 12
#define CV_DH 13
#define CV_DW 14
#define CV_G 15
#define CV_ACT 16
#define CV_WORDS 17

/* conv2d_<x>_<w>_<y>: NCHW input, KCRS weights (C per group), NKPQ output,
 * optional F32 bias per output channel, optional fused activation
 * (prm[CV_ACT] = act code + 1, 0 = none). Loop order c, r, s; padding taps
 * are skipped (not added as zero), which keeps -0 and NaN semantics simple. */
__kernel void nb_conv2d(const __global uchar *x, const __global uchar *w, const __global uchar *bias,
                        __global uchar *y, const __global uchar *prm8, uint fx, uint fw, uint fy, uint has_bias) {
    const __global uint *prm = (const __global uint *)prm8;
    uint N = prm[CV_N], C = prm[CV_C], H = prm[CV_H], W = prm[CV_W], K = prm[CV_K], R = prm[CV_R], S = prm[CV_S];
    uint P = prm[CV_P], Q = prm[CV_Q], G = prm[CV_G];
    uint g = NB_GID();
    if (g >= N * K * P * Q) return;
    uint q = g % Q, p = (g / Q) % P, k = (g / (P * Q)) % K, n = g / (K * P * Q);
    uint cg = C / G, grp = k / (K / G);
    float acc = has_bias ? ((const __global float *)bias)[k] : 0.0f;
    for (uint c = 0; c < cg; c++) {
        uint ci = grp * cg + c;
        for (uint r = 0; r < R; r++) {
            int ih = (int)(p * prm[CV_SH] + r * prm[CV_DH]) - (int)prm[CV_PH];
            if (ih < 0 || ih >= (int)H) continue;
            for (uint s = 0; s < S; s++) {
                int iw = (int)(q * prm[CV_SW] + s * prm[CV_DW]) - (int)prm[CV_PW];
                if (iw < 0 || iw >= (int)W) continue;
                float xv = nb_load_f(x, fx, ((n * C + ci) * H + (uint)ih) * W + (uint)iw);
                float wv = nb_load_f(w, fw, ((k * cg + c) * R + r) * S + s);
                acc = nb_fma(xv, wv, acc);
            }
        }
    }
    uint act = prm[CV_ACT];
    if (act != 0) acc = nb_activate(act - 1, acc, 0.0f, 0.0f);
    nb_store_f(y, fy, g, nb_canon(acc), 0);
}

#define NB_RESIZE_NEAREST 0u
#define NB_RESIZE_BILINEAR 1u

/* resize_<mode>_<src>_<dst>: `planes` images of HxW -> HoxWo, half-pixel
 * centres. Nearest uses exact integer arithmetic; bilinear computes the
 * source coordinate in F32 and blends with two fma lerps per axis. */
__kernel void nb_resize(const __global uchar *src, __global uchar *dst, uint planes, uint h, uint w, uint ho, uint wo,
                        uint mode, uint fs, uint fd) {
    uint g = NB_GID();
    if (g >= planes * ho * wo) return;
    uint ox = g % wo, oy = (g / wo) % ho, pl = g / (wo * ho);
    uint base = pl * h * w;
    float v;
    if (mode == NB_RESIZE_NEAREST) {
        uint sy = ((2 * oy + 1) * h) / (2 * ho), sx = ((2 * ox + 1) * w) / (2 * wo);
        if (sy >= h) sy = h - 1;
        if (sx >= w) sx = w - 1;
        v = nb_load_f(src, fs, base + sy * w + sx);
    } else {
        float fy = ((float)oy + 0.5f) * ((float)h / (float)ho) - 0.5f;
        float fx = ((float)ox + 0.5f) * ((float)w / (float)wo) - 0.5f;
        if (fy < 0.0f) fy = 0.0f;
        if (fx < 0.0f) fx = 0.0f;
        uint y0 = (uint)fy, x0 = (uint)fx;
        if (y0 > h - 1) y0 = h - 1;
        if (x0 > w - 1) x0 = w - 1;
        uint y1 = y0 + 1 < h ? y0 + 1 : h - 1, x1 = x0 + 1 < w ? x0 + 1 : w - 1;
        float ly = fy - (float)y0, lx = fx - (float)x0;
        float v00 = nb_load_f(src, fs, base + y0 * w + x0), v01 = nb_load_f(src, fs, base + y0 * w + x1);
        float v10 = nb_load_f(src, fs, base + y1 * w + x0), v11 = nb_load_f(src, fs, base + y1 * w + x1);
        float top = nb_fma(lx, v01 - v00, v00);
        float bot = nb_fma(lx, v11 - v10, v10);
        v = nb_fma(ly, bot - top, top);
    }
    nb_store_f(dst, fd, g, nb_canon(v), 0);
}
