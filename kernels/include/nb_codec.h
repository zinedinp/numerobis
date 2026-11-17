/* Element codecs: a line-by-line port of miraculix::formats (MiniFloat,
 * round_to_tf32, bf16 fast path, packed integers), driven by the format
 * descriptor of numerobis::elem::Format::desc. Bit-exact with the Rust oracle. */
#ifndef NB_CODEC_H
#define NB_CODEC_H

#include "nb_target.h"
#include "nb_formats.h"

#define NB_KIND(d) ((d) & 0xFu)
#define NB_FE(d) (((d) >> 4) & 0xFu)
#define NB_FM(d) (((d) >> 8) & 0xFu)
#define NB_FBIAS(d) ((int)(((d) >> 12) & 0xFFu))
#define NB_FSPECIAL(d) (((d) >> 20) & 0x3u)
#define NB_MX_INT8(d) (((d) >> 22) & 1u)
#define NB_IBITS(d) (((d) >> 4) & 0xFu)
#define NB_ISIGNED(d) (((d) >> 8) & 1u)
#define NB_BFP_BITS(d) (((d) >> 4) & 0xFu)

/* ---- MiniFloat (miraculix formats/minifloat.rs) ---- */

NB_INLINE float nb_pow2(int k) {
    return k >= -126 ? nb_u2f((uint)(k + 127) << 23) : nb_u2f(1u << (k + 149));
}

NB_INLINE uint nb_rne(uint m, uint shift) {
    if (shift >= 32) return 0;
    uint q = m >> shift;
    uint rem = m & ((1u << shift) - 1);
    uint hb = 1u << (shift - 1);
    return (rem > hb || (rem == hb && (q & 1))) ? q + 1 : q;
}

NB_INLINE uint nb_mf_max_code(uint d) {
    uint e = NB_FE(d), m = NB_FM(d);
    switch (NB_FSPECIAL(d)) {
    case NB_SPECIAL_IEEE: return (((1u << e) - 2) << m) | ((1u << m) - 1);
    case NB_SPECIAL_FN: return (1u << (e + m)) - 2;
    default: return (1u << (e + m)) - 1;
    }
}

NB_INLINE int nb_mf_emax(uint d) {
    uint top = (nb_mf_max_code(d) >> NB_FM(d)) & ((1u << NB_FE(d)) - 1);
    return (int)top - NB_FBIAS(d);
}

NB_INLINE float nb_mf_decode(uint d, uint code) {
    uint e = NB_FE(d), m = NB_FM(d), sp = NB_FSPECIAL(d);
    int bias = NB_FBIAS(d);
    uint sign = (code >> (e + m)) & 1;
    uint ex = (code >> m) & ((1u << e) - 1);
    uint man = code & ((1u << m) - 1);
    uint all_ones = (1u << e) - 1;
    if (sp == NB_SPECIAL_IEEE && ex == all_ones)
        return man == 0 ? nb_u2f(sign ? 0xFF800000u : 0x7F800000u) : NB_NAN;
    if (sp == NB_SPECIAL_FN && ex == all_ones && man == (1u << m) - 1) return NB_NAN;
    if (sp == NB_SPECIAL_FNUZ && (code & ((1u << (e + m + 1)) - 1)) == (1u << (e + m))) return NB_NAN;
    float mag = ex == 0 ? (float)man * nb_pow2(1 - bias - (int)m)
                         : (float)((1u << m) | man) * nb_pow2((int)ex - bias - (int)m);
    return sign ? -mag : mag;
}

/* sat != 0: Overflow::Saturate, else Overflow::Special. */
NB_INLINE uint nb_mf_encode(uint d, float x, uint sat) {
    uint e = NB_FE(d), m = NB_FM(d), sp = NB_FSPECIAL(d);
    int bias = NB_FBIAS(d);
    uint bits = nb_f2u(x);
    uint sign_bit = (bits >> 31) << (e + m);
    uint ab = bits & 0x7FFFFFFFu;
    uint nan = sp == NB_SPECIAL_IEEE ? ((((1u << e) - 1) << m) | (1u << (m - 1)))
             : sp == NB_SPECIAL_FN   ? (1u << (e + m)) - 1
             : sp == NB_SPECIAL_FNUZ ? 1u << (e + m)
                                     : 0u;
    if (ab > 0x7F800000u) {
        if (sp == NB_SPECIAL_FNUZ) return nan;
        if (sp == NB_SPECIAL_FINITE) return 0;
        return sign_bit | nan;
    }
    uint max = nb_mf_max_code(d);
    int emin = 1 - bias;
    uint code;
    if (ab >= 0x7F800000u) {
        code = max + 1;
    } else {
        int fexp = (int)(ab >> 23);
        int ee = fexp == 0 ? -126 : fexp - 127;
        uint mant24 = fexp == 0 ? (ab & 0x7FFFFFu) : ((ab & 0x7FFFFFu) | 0x800000u);
        if (ee >= emin && mant24 >= 0x800000u) {
            uint r = nb_rne(mant24, 23 - m);
            code = ((uint)(ee + bias) << m) + (r - (1u << m));
        } else {
            code = nb_rne(mant24, 23 - m + (uint)(emin - ee));
        }
    }
    if (code > max) {
        if (sat || sp == NB_SPECIAL_FINITE) code = max;
        else if (sp == NB_SPECIAL_IEEE) code = ((1u << e) - 1) << m;
        else if (sp == NB_SPECIAL_FN) return sign_bit | nan;
        else return nan;
    }
    return (sp == NB_SPECIAL_FNUZ && code == 0) ? 0 : (sign_bit | code);
}

NB_INLINE uint nb_is_bf16(uint d) { return NB_KIND(d) == NB_KIND_FLOAT && NB_FE(d) == 8 && NB_FM(d) == 7; }

/* miraculix bf16 slice path: integer RNE, NaN keeps its payload (quieted). */
NB_INLINE uint nb_bf16_encode(float x, uint sat) {
    uint b = nb_f2u(x);
    if ((b & 0x7FFFFFFFu) > 0x7F800000u) return ((b >> 16) | 0x40u) & 0xFFFFu;
    uint r = ((b + 0x7FFFu + ((b >> 16) & 1u)) >> 16) & 0xFFFFu;
    if (sat && (r & 0x7FFFu) == 0x7F80u) return (r & 0x8000u) | 0x7F7Fu;
    return r;
}

/* miraculix round_to_tf32. */
NB_INLINE float nb_tf32(float x) {
    uint b = nb_f2u(x);
    if ((b & 0x7F800000u) == 0x7F800000u) return x;
    return nb_u2f((b + 0xFFFu + ((b >> 13) & 1u)) & ~0x1FFFu);
}

/* ---- Packed bit streams, LSB-first ---- */

NB_INLINE uint nb_get_bits(const __global uchar *p, uint i, uint w) {
    uint bit = i * w, byte = bit / 8, off = bit % 8;
    uint v = (uint)p[byte] >> off;
    if (off + w > 8) v |= (uint)p[byte + 1] << (8 - off);
    return v & ((1u << w) - 1);
}

/* ORs into zeroed bytes; the caller owns every byte it touches. */
NB_INLINE void nb_or_bits(__global uchar *p, uint i, uint w, uint v) {
    uint bit = i * w, byte = bit / 8, off = bit % 8;
    p[byte] |= (uchar)(v << off);
    if (off + w > 8) p[byte + 1] |= (uchar)(v >> (8 - off));
}

/* Bits per element of an element-wise format; 0 for block formats. */
NB_INLINE uint nb_elem_bits(uint d) {
    switch (NB_KIND(d)) {
    case NB_KIND_F32: case NB_KIND_TF32: case NB_KIND_I32: return 32;
    case NB_KIND_I8: case NB_KIND_U8: return 8;
    case NB_KIND_FLOAT: return 1 + NB_FE(d) + NB_FM(d);
    case NB_KIND_INT: return NB_IBITS(d);
    default: return 0;
    }
}

/* Clears the bytes of elements [first, first + count) of a packed format. */
NB_INLINE void nb_zero_packed(__global uchar *p, uint d, uint first, uint count) {
    uint w = nb_elem_bits(d);
    if (w >= 8 || w == 0) return;
    for (uint b = first * w / 8; b < (first + count) * w / 8; b++) p[b] = 0;
}

/* ---- Element-wise load/store as float ---- */

NB_INLINE int nb_load_i(const __global uchar *p, uint d, uint i);

NB_INLINE float nb_load_f(const __global uchar *p, uint d, uint i) {
    switch (NB_KIND(d)) {
    case NB_KIND_F32: case NB_KIND_TF32: return ((const __global float *)p)[i];
    case NB_KIND_FLOAT: {
        uint w = 1 + NB_FE(d) + NB_FM(d);
        uint code = w == 16 ? (uint)((const __global ushort *)p)[i] : w == 8 ? (uint)p[i] : nb_get_bits(p, i, w);
        return nb_is_bf16(d) ? nb_u2f(code << 16) : nb_mf_decode(d, code);
    }
    default: return (float)nb_load_i(p, d, i);
    }
}

NB_INLINE void nb_store_f(__global uchar *p, uint d, uint i, float x, uint sat) {
    switch (NB_KIND(d)) {
    case NB_KIND_F32: ((__global float *)p)[i] = x; break;
    case NB_KIND_TF32: ((__global float *)p)[i] = nb_tf32(x); break;
    case NB_KIND_FLOAT: {
        uint w = 1 + NB_FE(d) + NB_FM(d);
        uint code = nb_is_bf16(d) ? nb_bf16_encode(x, sat) : nb_mf_encode(d, x, sat);
        if (w == 16) ((__global ushort *)p)[i] = (ushort)code;
        else if (w == 8) p[i] = (uchar)code;
        else nb_or_bits(p, i, w, code);
        break;
    }
    default: break;
    }
}

/* ---- Integers (miraculix formats/int.rs) ---- */

NB_INLINE int nb_load_i(const __global uchar *p, uint d, uint i) {
    switch (NB_KIND(d)) {
    case NB_KIND_I32: return ((const __global int *)p)[i];
    case NB_KIND_I8: return (int)(signed char)p[i];
    case NB_KIND_U8: return (int)p[i];
    case NB_KIND_INT: {
        uint w = NB_IBITS(d), v = nb_get_bits(p, i, w);
        if (NB_ISIGNED(d) && (v >> (w - 1))) return (int)v - (1 << w);
        return (int)v;
    }
    default: return 0;
    }
}

NB_INLINE int nb_clamp_i(int v, int lo, int hi) { return v < lo ? lo : v > hi ? hi : v; }

/* Saturating, like pack_int / pack_uint. */
NB_INLINE void nb_store_i(__global uchar *p, uint d, uint i, int v) {
    switch (NB_KIND(d)) {
    case NB_KIND_I32: ((__global int *)p)[i] = v; break;
    case NB_KIND_I8: p[i] = (uchar)nb_clamp_i(v, -128, 127); break;
    case NB_KIND_U8: p[i] = (uchar)nb_clamp_i(v, 0, 255); break;
    case NB_KIND_INT: {
        uint w = NB_IBITS(d);
        int c = NB_ISIGNED(d) ? nb_clamp_i(v, -(1 << (w - 1)), (1 << (w - 1)) - 1) : nb_clamp_i(v, 0, (1 << w) - 1);
        nb_or_bits(p, i, w, (uint)c & ((1u << w) - 1));
        break;
    }
    default: break;
    }
}

#endif /* NB_CODEC_H */
