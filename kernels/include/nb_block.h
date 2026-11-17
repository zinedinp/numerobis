/* Block formats (miraculix formats/mx.rs, bfp.rs, gguf.rs) in the numerobis
 * device layout: element data, then one scale byte per block (NVFP4: then
 * the f32 tensor scale). `soff` is Format::scale_offset, passed by the host. */
#ifndef NB_BLOCK_H
#define NB_BLOCK_H

#include "nb_codec.h"

NB_INLINE float nb_e8m0(uint s) { return s == 0xFFu ? NB_NAN : nb_pow2((int)s - 127); }

NB_INLINE uint nb_mx_w(uint d) { return NB_MX_INT8(d) ? 8u : 1u + NB_FE(d) + NB_FM(d); }

/* Elements per block. */
NB_INLINE uint nb_block_len(uint d) {
    switch (NB_KIND(d)) {
    case NB_KIND_MX: return NB_MX_BLOCK;
    case NB_KIND_NVFP4: return NB_NVFP4_BLOCK;
    case NB_KIND_BFP: return NB_BFP_ROW;
    case NB_KIND_Q8_0: return NB_Q8_0_BLOCK;
    case NB_KIND_Q6_K: return NB_Q6_K_BLOCK;
    default: return 1;
    }
}

/* Element-data bytes per block (GGUF: the whole interleaved block). */
NB_INLINE uint nb_block_bytes(uint d) {
    switch (NB_KIND(d)) {
    case NB_KIND_MX: return NB_MX_BLOCK * nb_mx_w(d) / 8;
    case NB_KIND_NVFP4: return NB_NVFP4_BLOCK / 2;
    case NB_KIND_BFP: return NB_BFP_ROW * (NB_BFP_BITS(d) + 1) / 8;
    case NB_KIND_Q8_0: return NB_Q8_0_BYTES;
    case NB_KIND_Q6_K: return NB_Q6_K_BYTES;
    default: return 0;
    }
}

NB_INLINE uint nb_is_block(uint d) { return NB_KIND(d) >= NB_KIND_MX; }

/* Unaligned little-endian f32. */
NB_INLINE float nb_read_f32(const __global uchar *p) {
    return nb_u2f((uint)p[0] | ((uint)p[1] << 8) | ((uint)p[2] << 16) | ((uint)p[3] << 24));
}

NB_INLINE float nb_half(const __global uchar *p) {
    return nb_mf_decode(NB_DESC_F16, (uint)p[0] | ((uint)p[1] << 8));
}

/* Element i of a block-format tensor of `len` elements. */
NB_INLINE float nb_load_block(const __global uchar *p, uint d, uint i, uint len, uint soff) {
    uint B = nb_block_len(d), b = i / B, j = i % B;
    const __global uchar *blk = p + b * nb_block_bytes(d);
    switch (NB_KIND(d)) {
    case NB_KIND_MX: {
        float scale = nb_e8m0(p[soff + b]);
        uint c = nb_get_bits(blk, j, nb_mx_w(d));
        if (NB_MX_INT8(d)) return (float)(signed char)c * (1.0f / 64.0f) * scale;
        return nb_mf_decode(d, c) * scale;
    }
    case NB_KIND_NVFP4: {
        float ts = nb_read_f32(p + soff + len / NB_NVFP4_BLOCK);
        float s = nb_mf_decode(NB_DESC_E4M3FN, p[soff + b]) * ts;
        return nb_mf_decode(NB_DESC_E2M1, nb_get_bits(blk, j, 4)) * s;
    }
    case NB_KIND_BFP: {
        uint bits = NB_BFP_BITS(d), w = bits + 1, per = 8 / w;
        float unit = nb_pow2((int)p[soff + b] - 127 - ((int)bits - 1));
        uint code = ((uint)blk[j / per] >> ((j % per) * w)) & ((1u << w) - 1);
        float mag = (float)(code & ((1u << bits) - 1)) * unit;
        return (code >> bits) ? -mag : mag;
    }
    case NB_KIND_Q8_0:
        return (float)(signed char)blk[2 + j] * nb_half(blk);
    case NB_KIND_Q6_K: {
        uint n = j / 128, r = j % 128, quarter = r / 32, l = r % 32, is = l / 16;
        const __global uchar *ql = blk + n * 64, *qh = blk + 128 + n * 32, *sc = blk + 192 + n * 8;
        uint lo = (quarter & 1) ? ql[l + 32] : ql[l];
        lo = quarter < 2 ? (lo & 0xFu) : (lo >> 4);
        uint hi = ((uint)qh[l] >> (2 * quarter)) & 3u;
        int q = (int)(lo | (hi << 4)) - 32;
        return nb_half(blk + 208) * (float)(signed char)sc[is + 2 * quarter] * (float)q;
    }
    default: return 0.0f;
    }
}

/* Any format, element i. */
NB_INLINE float nb_load_any(const __global uchar *p, uint d, uint i, uint len, uint soff) {
    return nb_is_block(d) ? nb_load_block(p, d, i, len, soff) : nb_load_f(p, d, i);
}

#endif /* NB_BLOCK_H */
