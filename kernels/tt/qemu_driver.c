/* Linux user-mode harness for the Tensix kernel build (tests only): the same
 * RV32IM code as kernels/bin/tt (shared kernels + softfloat.c, freestanding),
 * driven through the stdin/stdout protocol of kernels/host/driver.c and run
 * under qemu-riscv32. No libc: raw syscalls, static arena. */
#include "../src/all.cl"

uint nb_host_gid;

/* The compiler may emit these for struct initialisation and copies. */
void *memset(void *d, int c, unsigned long n) {
    uchar *p = d;
    while (n--) *p++ = (uchar)c;
    return d;
}
void *memcpy(void *d, const void *s, unsigned long n) {
    uchar *p = d;
    const uchar *q = s;
    while (n--) *p++ = *q++;
    return d;
}

static long sys3(long n, long a, long b, long c) {
    register long a0 __asm__("a0") = a, a1 __asm__("a1") = b, a2 __asm__("a2") = c, a7 __asm__("a7") = n;
    __asm__ volatile("ecall" : "+r"(a0) : "r"(a1), "r"(a2), "r"(a7) : "memory");
    return a0;
}
static __attribute__((noreturn)) void quit(int code) {
    sys3(93, code, 0, 0);
    for (;;) {}
}
static void rd(void *dst, uint n) {
    uchar *p = dst;
    while (n) {
        long r = sys3(63, 0, (long)p, n);
        if (r <= 0) quit(2);
        p += r;
        n -= (uint)r;
    }
}
static void wr(const void *src, uint n) {
    const uchar *p = src;
    while (n) {
        long r = sys3(64, 1, (long)p, n);
        if (r <= 0) quit(2);
        p += r;
        n -= (uint)r;
    }
}
static uint rd_u32(void) {
    uchar b[4];
    rd(b, 4);
    return (uint)b[0] | ((uint)b[1] << 8) | ((uint)b[2] << 16) | ((uint)b[3] << 24);
}

static uchar arena[32 << 20];
static uint used;

typedef struct {
    uchar tag;
    uchar *p;
    uint len, v;
} Arg;

static int eq(const char *a, const char *b) {
    while (*a && *a == *b) a++, b++;
    return *a == *b;
}

#define P(i) (a[i].p)
#define U(i) (a[i].v)
#define I(i) ((int)a[i].v)
#define F(i) nb_u2f(a[i].v)

static int run(void) {
    char name[64] = {0};
    uint nl = rd_u32();
    if (nl >= sizeof name) return 2;
    rd(name, nl);
    uint items = rd_u32(), nargs = rd_u32();
    Arg a[16] = {0};
    if (nargs > 16) return 2;
    for (uint i = 0; i < nargs; i++) {
        rd(&a[i].tag, 1);
        if (a[i].tag == 0) {
            a[i].len = rd_u32();
            if (used + a[i].len + 64 > sizeof arena) return 4;
            a[i].p = arena + used;
            used = (used + a[i].len + 64) & ~63u;
            rd(a[i].p, a[i].len);
        } else {
            a[i].v = rd_u32();
        }
    }
    for (nb_host_gid = 0; nb_host_gid < items; nb_host_gid++) {
        if (eq(name, "nb_convert")) nb_convert(P(0), P(1), U(2), U(3), U(4), U(5), U(6));
        else if (eq(name, "nb_int_convert")) nb_int_convert(P(0), P(1), U(2), U(3), U(4), U(5));
        else if (eq(name, "nb_nvfp4_tensor_scale")) nb_nvfp4_tensor_scale(P(0), P(1), U(2), U(3));
        else if (eq(name, "nb_quantize")) nb_quantize(P(0), P(1), U(2), U(3), U(4));
        else if (eq(name, "nb_dequantize")) nb_dequantize(P(0), P(1), U(2), U(3), U(4));
        else if (eq(name, "nb_gemm_f"))
            nb_gemm_f(P(0), P(1), P(2), U(3), U(4), U(5), U(6), F(7), F(8), U(9), U(10), U(11), U(12), U(13));
        else if (eq(name, "nb_gemm_i"))
            nb_gemm_i(P(0), P(1), P(2), U(3), U(4), U(5), U(6), I(7), I(8), U(9), U(10), U(11), U(12), U(13));
        else if (eq(name, "nb_absmax_block")) nb_absmax_block(P(0), P(1), U(2), U(3), U(4), U(5));
        else return 3;
    }
    for (uint i = 0; i < nargs; i++)
        if (a[i].tag == 0) wr(a[i].p, a[i].len);
    return 0;
}

/* Static ELF entry: stack is set up by the kernel/qemu. */
__attribute__((noreturn)) void _start(void) { quit(run()); }
