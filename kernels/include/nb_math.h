/* Bit-exact transcendental functions for the Emulated kernels.
 *
 * Only +, -, *, / (correctly rounded) and fma are used, in a fixed order,
 * with constants given as bit patterns. Every IEEE-conforming target
 * (AMDGPU, NVPTX, SPIR-V, host C, RV32 soft-float) therefore returns the same
 * bits, and src/mock/kernels/act.rs mirrors each line. That is the contract
 * that makes these results usable for game state (prediction, physics,
 * server reconciliation); vendor tiers with native exp/tanh are not.
 *
 * Accuracy: exp ~1 ulp on the normal range, sigmoid/tanh/gelu a few ulp. */
#ifndef NB_MATH_H
#define NB_MATH_H

#include "nb_codec.h"

/* NaN payloads and signs differ between targets (x86 makes -inf*0 a
 * negative NaN, RISC-V the canonical one): outputs are canonicalised. */
NB_INLINE float nb_canon(float x) { return x != x ? NB_NAN : x; }

#define NB_LOG2E 0x3FB8AA3Bu  /* 1.44269502 */
#define NB_LN2_HI 0x3F317200u /* 0.693145751953125 (low 12 bits zero) */
#define NB_LN2_LO 0x35BFBE8Eu /* 1.42860677e-06 */
#define NB_EXP_HI 0x42B17218u /* 88.7228394: exp overflows above */
#define NB_EXP_LO 0xC2CFF1B5u /* -103.972084: exp is 0 below */

NB_INLINE float nb_exp(float x) {
    if (x != x) return x;
    if (x > nb_u2f(NB_EXP_HI)) return nb_u2f(0x7F800000u);
    if (x < nb_u2f(NB_EXP_LO)) return 0.0f;
    float t = x * nb_u2f(NB_LOG2E);
    int k = (int)(t >= 0.0f ? t + 0.5f : t - 0.5f);
    float kf = (float)k;
    float r = nb_fma(-kf, nb_u2f(NB_LN2_HI), x);
    r = nb_fma(-kf, nb_u2f(NB_LN2_LO), r);
    /* Taylor to r^7 on |r| <= 0.35: truncation < 2.2e-9 relative. */
    float p = nb_u2f(0x39500D01u);     /* 1/5040 */
    p = nb_fma(p, r, nb_u2f(0x3AB60B61u)); /* 1/720 */
    p = nb_fma(p, r, nb_u2f(0x3C088889u)); /* 1/120 */
    p = nb_fma(p, r, nb_u2f(0x3D2AAAABu)); /* 1/24 */
    p = nb_fma(p, r, nb_u2f(0x3E2AAAABu)); /* 1/6 */
    p = nb_fma(p, r, 0.5f);
    p = nb_fma(p, r, 1.0f);
    p = nb_fma(p, r, 1.0f);
    /* p * 2^k in two exact-or-once-rounded steps (k in [-151, 128]). */
    if (k > 127) return p * nb_pow2(127) * nb_pow2(k - 127);
    if (k < -126) return p * nb_pow2(k + 64) * nb_pow2(-64);
    return p * nb_pow2(k);
}

NB_INLINE float nb_sigmoid(float x) { return 1.0f / (1.0f + nb_exp(-x)); }

NB_INLINE float nb_tanh(float x) {
    if (x != x) return x;
    float a = nb_fabs(x);
    float t;
    if (a < nb_u2f(0x39800000u)) return x; /* |x| < 2^-12: tanh x == x in f32 */
    if (a > 9.0f) t = 1.0f;
    else t = 1.0f - 2.0f / (nb_exp(2.0f * a) + 1.0f);
    return x < 0.0f ? -t : t;
}

/* GELU, tanh form: 0.5 x (1 + tanh(sqrt(2/pi) (x + 0.044715 x^3))). */
NB_INLINE float nb_gelu(float x) {
    float x3 = x * x * x;
    float u = nb_u2f(0x3F4C422Au) * (x + nb_u2f(0x3D372713u) * x3);
    return 0.5f * x * (1.0f + nb_tanh(u));
}

/* Activation codes (src/ops/act.rs: Act::code). */
#define NB_ACT_RELU 0u
#define NB_ACT_LEAKY 1u
#define NB_ACT_SIGMOID 2u
#define NB_ACT_TANH 3u
#define NB_ACT_SILU 4u
#define NB_ACT_GELU 5u
#define NB_ACT_CLAMP 6u
#define NB_ACT_EXP 7u

NB_INLINE float nb_activate(uint act, float x, float p0, float p1) {
    switch (act) {
    case NB_ACT_RELU: return x != x ? x : (x > 0.0f ? x : 0.0f);
    case NB_ACT_LEAKY: return x >= 0.0f ? x : x * p0;
    case NB_ACT_SIGMOID: return nb_sigmoid(x);
    case NB_ACT_TANH: return nb_tanh(x);
    case NB_ACT_SILU: return x * nb_sigmoid(x);
    case NB_ACT_GELU: return nb_gelu(x);
    case NB_ACT_CLAMP: return x != x ? x : (x < p0 ? p0 : (x > p1 ? p1 : x));
    case NB_ACT_EXP: return nb_exp(x);
    default: return NB_NAN;
    }
}

#endif /* NB_MATH_H */
