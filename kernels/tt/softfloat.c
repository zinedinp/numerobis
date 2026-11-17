/* IEEE 754 binary32 soft-float for the Tensix baby RISC-V cores (RV32IM, no
 * F extension): the libgcc/compiler-rt entry points clang emits for float
 * arithmetic, plus fmaf. Round to nearest even, subnormals kept (no FTZ),
 * invalid operations return the default NaN 0x7FC00000, NaN operands are
 * returned quieted.
 *
 * fmaf is the core: a + b = fma(a, 1, b) and a * b = fma(a, b, -0) are
 * exactly rounded, so only fma and division need their own rounding.
 *
 * Built with NB_SF_PREFIX defined, every symbol gets that prefix so the host
 * test (tests/softfloat.rs) can compare it with the hardware FPU. */

#include <stdint.h>

#ifdef NB_SF_PREFIX
#define SF2(p, n) p##n
#define SF1(p, n) SF2(p, n)
#define SF(n) SF1(NB_SF_PREFIX, n)
#else
#define SF(n) n
#endif

typedef uint32_t u32;
typedef uint64_t u64;
typedef int32_t i32;

static inline u32 bits(float x) { u32 u; __builtin_memcpy(&u, &x, 4); return u; }
static inline float flt(u32 u) { float x; __builtin_memcpy(&x, &u, 4); return x; }

#define SIGN 0x80000000u
#define QNAN 0x7FC00000u
static inline int is_nan(u32 a) { return (a & 0x7FFFFFFFu) > 0x7F800000u; }
static inline int is_inf(u32 a) { return (a & 0x7FFFFFFFu) == 0x7F800000u; }
static inline int is_zero(u32 a) { return (a & 0x7FFFFFFFu) == 0; }

static int clz64(u64 x) { return __builtin_clzll(x); }

/* value = m * 2^e with m in [2^23, 2^24) for nonzero finite a. */
static void unpack(u32 a, u32 *m, i32 *e) {
    u32 ex = (a >> 23) & 0xFF, mant = a & 0x7FFFFF;
    if (ex == 0) {
        int s = __builtin_clz(mant) - 8; /* bring the top bit to bit 23 */
        *m = mant << s;
        *e = -126 - 23 - s;
    } else {
        *m = mant | 0x800000u;
        *e = (i32)ex - 127 - 23;
    }
}

/* x >> n, round to nearest even, `sticky` marks nonzero bits below x. */
static u64 rne_shift(u64 x, int n, int sticky) {
    if (n <= 0) return x << -n;
    if (n >= 64) return (n == 64 && (x > (1ull << 63) || (x == (1ull << 63) && sticky))) ? 1 : 0;
    u64 q = x >> n, rem = x & ((1ull << n) - 1), half = 1ull << (n - 1);
    if (rem > half || (rem == half && (sticky || (q & 1)))) q++;
    return q;
}

/* Rounds sign * (s + sticky) * 2^e (s != 0) to binary32. */
static u32 round_pack(u32 sign, u64 s, i32 e, int sticky) {
    int k = 63 - clz64(s);  /* msb position */
    i32 E = e + k;          /* unbiased exponent of the result */
    if (E > 127) return sign | 0x7F800000u;
    if (E >= -126) {
        u64 m = rne_shift(s, k - 23, sticky);
        if (m >> 24) { m >>= 1; E++; }
        if (E > 127) return sign | 0x7F800000u;
        return sign | ((u32)(E + 127) << 23) | ((u32)m & 0x7FFFFF);
    }
    /* Subnormal: value = M * 2^-149. M == 2^23 encodes the smallest normal. */
    u64 m = rne_shift(s, -149 - e, sticky);
    return sign | (u32)m;
}

/* Shifts right by d, ORing lost bits into bit 0. */
static u64 shr_sticky(u64 x, i32 d) {
    if (d <= 0) return x;
    if (d >= 64) return x != 0;
    return (x >> d) | ((x & ((1ull << d) - 1)) != 0);
}

float SF(fmaf)(float fa, float fb, float fc) {
    u32 a = bits(fa), b = bits(fb), c = bits(fc);
    if (is_nan(a)) return flt(a | 0x400000u);
    if (is_nan(b)) return flt(b | 0x400000u);
    u32 ps = (a ^ b) & SIGN;
    if ((is_inf(a) && is_zero(b)) || (is_zero(a) && is_inf(b))) return flt(QNAN);
    if (is_nan(c)) return flt(c | 0x400000u);
    if (is_inf(a) || is_inf(b)) {
        if (is_inf(c) && (c & SIGN) != ps) return flt(QNAN);
        return flt(ps | 0x7F800000u);
    }
    if (is_inf(c)) return fc;
    if (is_zero(a) || is_zero(b)) {
        /* Exact zero product: +0 unless both are -0. */
        if (is_zero(c)) return flt((ps & c & SIGN) ? SIGN : 0);
        return fc;
    }
    u32 ma, mb;
    i32 ea, eb;
    unpack(a, &ma, &ea);
    unpack(b, &mb, &eb);
    u64 P = ((u64)ma * mb) << 15; /* < 2^63 */
    i32 eP = ea + eb - 15;
    if (is_zero(c)) return flt(round_pack(ps, P, eP, 0));
    u32 mc;
    i32 ec;
    unpack(c, &mc, &ec);
    u64 C = (u64)mc << 39; /* < 2^63 */
    i32 eC = ec - 39;
    u32 cs = c & SIGN;
    i32 e = eP > eC ? eP : eC;
    P = shr_sticky(P, e - eP);
    C = shr_sticky(C, e - eC);
    u64 S;
    u32 sign;
    if (ps == cs) {
        S = P + C;
        sign = ps;
    } else if (P >= C) {
        S = P - C;
        sign = ps;
    } else {
        S = C - P;
        sign = cs;
    }
    if (S == 0) return flt(0); /* exact cancellation: +0 */
    return flt(round_pack(sign, S, e, 0));
}

float SF(__addsf3)(float a, float b) { return SF(fmaf)(a, 1.0f, b); }
float SF(__subsf3)(float a, float b) { return SF(fmaf)(a, 1.0f, flt(bits(b) ^ SIGN)); }
float SF(__mulsf3)(float a, float b) { return SF(fmaf)(a, b, flt(SIGN)); }

float SF(__divsf3)(float fa, float fb) {
    u32 a = bits(fa), b = bits(fb);
    if (is_nan(a)) return flt(a | 0x400000u);
    if (is_nan(b)) return flt(b | 0x400000u);
    u32 sign = (a ^ b) & SIGN;
    if (is_inf(a)) return flt(is_inf(b) ? QNAN : (sign | 0x7F800000u));
    if (is_inf(b)) return flt(sign);
    if (is_zero(b)) return flt(is_zero(a) ? QNAN : (sign | 0x7F800000u));
    if (is_zero(a)) return flt(sign);
    u32 ma, mb;
    i32 ea, eb;
    unpack(a, &ma, &ea);
    unpack(b, &mb, &eb);
    /* q = floor(ma * 2^40 / mb) by restoring division (no 64-bit divide on RV32IM). */
    u64 q = 0;
    u32 r = ma;
    for (int i = 0; i <= 40; i++) {
        q <<= 1;
        if (r >= mb) {
            r -= mb;
            q |= 1;
        }
        r <<= 1;
    }
    return flt(round_pack(sign, (q << 1) | (r != 0), ea - eb - 41, 0));
}

static float from_u32(u32 sign, u32 v) {
    if (v == 0) return flt(sign);
    return flt(round_pack(sign, v, 0, 0));
}
float SF(__floatsisf)(i32 v) { return v < 0 ? from_u32(SIGN, 0u - (u32)v) : from_u32(0, (u32)v); }
float SF(__floatunsisf)(u32 v) { return from_u32(0, v); }

/* Truncating, saturating (Rust `as` semantics), NaN -> 0. */
i32 SF(__fixsfsi)(float fa) {
    u32 a = bits(fa);
    if (is_nan(a)) return 0;
    i32 ex = (i32)((a >> 23) & 0xFF) - 127;
    if (ex < 0) return 0;
    if (ex >= 31) return (a & SIGN) ? INT32_MIN : INT32_MAX;
    u32 m = (a & 0x7FFFFF) | 0x800000u;
    u32 v = ex >= 23 ? m << (ex - 23) : m >> (23 - ex);
    return (a & SIGN) ? -(i32)v : (i32)v;
}
u32 SF(__fixunssfsi)(float fa) {
    u32 a = bits(fa);
    if (is_nan(a) || (a & SIGN)) return 0;
    i32 ex = (i32)((a >> 23) & 0xFF) - 127;
    if (ex < 0) return 0;
    if (ex >= 32) return UINT32_MAX;
    u32 m = (a & 0x7FFFFF) | 0x800000u;
    return ex >= 23 ? m << (ex - 23) : m >> (23 - ex);
}

/* -1 less, 0 equal, 1 greater, 2 unordered. */
static int cmp(float fa, float fb) {
    u32 a = bits(fa), b = bits(fb);
    if (is_nan(a) || is_nan(b)) return 2;
    if (is_zero(a) && is_zero(b)) return 0;
    i32 ka = (a & SIGN) ? -(i32)(a & 0x7FFFFFFFu) : (i32)a;
    i32 kb = (b & SIGN) ? -(i32)(b & 0x7FFFFFFFu) : (i32)b;
    return ka < kb ? -1 : ka > kb;
}
int SF(__eqsf2)(float a, float b) { int c = cmp(a, b); return c == 2 ? 1 : c; }
int SF(__nesf2)(float a, float b) { int c = cmp(a, b); return c == 2 ? 1 : c; }
int SF(__ltsf2)(float a, float b) { int c = cmp(a, b); return c == 2 ? 1 : c; }
int SF(__lesf2)(float a, float b) { int c = cmp(a, b); return c == 2 ? 1 : c; }
int SF(__gtsf2)(float a, float b) { int c = cmp(a, b); return c == 2 ? -1 : c; }
int SF(__gesf2)(float a, float b) { int c = cmp(a, b); return c == 2 ? -1 : c; }
int SF(__unordsf2)(float a, float b) { return cmp(a, b) == 2; }
