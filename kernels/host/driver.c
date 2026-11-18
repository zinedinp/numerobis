/* Host driver for the Emulated kernels (tests only). Compiled as C with
 * -DNB_HOST; runs one launch per invocation, work items in order.
 *
 * stdin (little-endian): u32 name_len, name, u32 items, u32 nargs, then per
 * arg u8 tag (0 buffer, 1 u32, 2 i32, 3 f32) and either u32 byte_len + bytes
 * or 4 value bytes. stdout: the bytes of every buffer arg, in order. */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "../src/all.cl"

uint nb_host_gid;

typedef struct {
    uchar tag;
    uchar *p;
    uint len;
    uint v;
} Arg;

static void rd(void *dst, size_t n) {
    if (fread(dst, 1, n, stdin) != n) {
        fprintf(stderr, "driver: short read\n");
        exit(2);
    }
}

static uint rd_u32(void) {
    uchar b[4];
    rd(b, 4);
    return (uint)b[0] | ((uint)b[1] << 8) | ((uint)b[2] << 16) | ((uint)b[3] << 24);
}

#define P(i) (a[i].p)
#define U(i) (a[i].v)
#define I(i) ((int)a[i].v)
#define F(i) nb_u2f(a[i].v)

int main(void) {
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
            a[i].p = malloc(a[i].len + 1);
            rd(a[i].p, a[i].len);
        } else {
            a[i].v = rd_u32();
        }
    }
    for (nb_host_gid = 0; nb_host_gid < items; nb_host_gid++) {
        if (!strcmp(name, "nb_convert")) nb_convert(P(0), P(1), U(2), U(3), U(4), U(5), U(6));
        else if (!strcmp(name, "nb_int_convert")) nb_int_convert(P(0), P(1), U(2), U(3), U(4), U(5));
        else if (!strcmp(name, "nb_nvfp4_tensor_scale")) nb_nvfp4_tensor_scale(P(0), P(1), U(2), U(3));
        else if (!strcmp(name, "nb_quantize")) nb_quantize(P(0), P(1), U(2), U(3), U(4));
        else if (!strcmp(name, "nb_dequantize")) nb_dequantize(P(0), P(1), U(2), U(3), U(4));
        else if (!strcmp(name, "nb_gemm_f"))
            nb_gemm_f(P(0), P(1), P(2), U(3), U(4), U(5), U(6), F(7), F(8), U(9), U(10), U(11), U(12), U(13));
        else if (!strcmp(name, "nb_gemm_i"))
            nb_gemm_i(P(0), P(1), P(2), U(3), U(4), U(5), U(6), I(7), I(8), U(9), U(10), U(11), U(12), U(13));
        else if (!strcmp(name, "nb_absmax_block")) nb_absmax_block(P(0), P(1), U(2), U(3), U(4), U(5));
        else if (!strcmp(name, "nb_act")) nb_act(P(0), P(1), U(2), U(3), F(4), F(5), U(6), U(7));
        else if (!strcmp(name, "nb_conv2d")) nb_conv2d(P(0), P(1), P(2), P(3), P(4), U(5), U(6), U(7), U(8));
        else if (!strcmp(name, "nb_resize")) nb_resize(P(0), P(1), U(2), U(3), U(4), U(5), U(6), U(7), U(8), U(9));
        else {
            fprintf(stderr, "driver: unknown kernel %s\n", name);
            return 3;
        }
    }
    for (uint i = 0; i < nargs; i++)
        if (a[i].tag == 0) fwrite(a[i].p, 1, a[i].len, stdout);
    return 0;
}
