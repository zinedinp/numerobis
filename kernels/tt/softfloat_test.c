/* Host check of softfloat.c against the hardware FPU (tests/softfloat.rs).
 * Built with -DNB_SF_PREFIX=sf_ -ffp-contract=off; exits nonzero on the
 * first mismatch. NaN results only have to be NaN. */
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define NB_SF_PREFIX sf_
#include "softfloat.c"

static uint64_t s = 0x9E3779B97F4A7C15ull;
static uint32_t rnd(void) {
    s ^= s << 13;
    s ^= s >> 7;
    s ^= s << 17;
    return (uint32_t)(s >> 16);
}

/* Random bit patterns biased towards interesting exponents. */
static float pick(void) {
    uint32_t r = rnd();
    switch (r % 8) {
    case 0: return flt(rnd());
    case 1: return flt((rnd() & 0x807FFFFFu));                       /* subnormal / zero */
    case 2: return flt((rnd() & 0x80FFFFFFu) | (1u << 23));           /* tiny normal */
    case 3: return flt((rnd() & 0x807FFFFFu) | (0xFEu << 23));        /* huge */
    case 4: { static const uint32_t sp[] = {0, SIGN, 0x7F800000u, 0xFF800000u, QNAN, 0x3F800000u, 0xBF800000u, 0x00000001u, 0x7F7FFFFFu};
              return flt(sp[rnd() % 9]); }
    default: return flt((rnd() & 0x807FFFFFu) | ((100 + rnd() % 56) << 23));
    }
}

static int same(float a, float b) { return bits(a) == bits(b) || (isnan(a) && isnan(b)); }

#define CHECK(name, got, want, ...)                                                       \
    if (!same(got, want)) {                                                               \
        printf(name " mismatch: got %08x want %08x for " __VA_ARGS__);                   \
        return 1;                                                                        \
    }

int main(int argc, char **argv) {
    long n = argc > 1 ? atol(argv[1]) : 2000000;
    for (long i = 0; i < n; i++) {
        float a = pick(), b = pick(), c = pick();
        CHECK("fma", sf_fmaf(a, b, c), fmaf(a, b, c), "%08x %08x %08x\n", bits(sf_fmaf(a, b, c)), bits(fmaf(a, b, c)), bits(a), bits(b), bits(c));
        CHECK("add", sf___addsf3(a, b), a + b, "%08x %08x\n", bits(sf___addsf3(a, b)), bits(a + b), bits(a), bits(b));
        CHECK("sub", sf___subsf3(a, b), a - b, "%08x %08x\n", bits(sf___subsf3(a, b)), bits(a - b), bits(a), bits(b));
        CHECK("mul", sf___mulsf3(a, b), a * b, "%08x %08x\n", bits(sf___mulsf3(a, b)), bits(a * b), bits(a), bits(b));
        CHECK("div", sf___divsf3(a, b), a / b, "%08x %08x\n", bits(sf___divsf3(a, b)), bits(a / b), bits(a), bits(b));
        int32_t iv = (int32_t)rnd() >> (rnd() % 31);
        CHECK("i2f", sf___floatsisf(iv), (float)iv, "%d\n", bits(sf___floatsisf(iv)), bits((float)iv), iv);
        uint32_t uv = rnd() >> (rnd() % 31);
        CHECK("u2f", sf___floatunsisf(uv), (float)uv, "%u\n", bits(sf___floatunsisf(uv)), bits((float)uv), uv);
        if (!isnan(a) && fabsf(a) < 2147483520.0f && sf___fixsfsi(a) != (int32_t)a) {
            printf("f2i mismatch for %08x\n", bits(a));
            return 1;
        }
        int lt = sf___ltsf2(a, b) < 0, le = sf___lesf2(a, b) <= 0, gt = sf___gtsf2(a, b) > 0, ge = sf___gesf2(a, b) >= 0;
        int eq = sf___eqsf2(a, b) == 0, ne = sf___nesf2(a, b) != 0, un = sf___unordsf2(a, b) != 0;
        if (lt != (a < b) || le != (a <= b) || gt != (a > b) || ge != (a >= b) || eq != (a == b) || ne != (a != b) || un != (isnan(a) || isnan(b))) {
            printf("compare mismatch for %08x %08x\n", bits(a), bits(b));
            return 1;
        }
    }
    printf("ok %ld\n", n);
    return 0;
}
