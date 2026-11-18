/* Persistent BRISC runtime for the Emulated kernels on one Tensix core
 * (Blackhole). The shared kernel sources are compiled as freestanding C
 * (NB_HOST) for RV32IM; float arithmetic comes from softfloat.c.
 *
 * Protocol (host side: src/tt/backend.rs): the host fills the mailbox at
 * NB_MAILBOX and then bumps `seq`. The core pulls every listed buffer from
 * DRAM into L1 over NoC 0, runs the kernel for `items` work items, writes the
 * buffers marked `writeback` back to DRAM, and stores `seq` into `done`.
 *
 * Blackhole's RISC-V data cache is invalidated with `fence`; L1 written by
 * the NoC is only read after one. */
#include "../src/all.cl"

uint nb_host_gid;

#define NB_MAILBOX 0x18000u
#define NB_STACK_TOP 0x20000u
#define NB_MAX_ARGS 16
#define NB_MAX_BUFS 8

struct nb_buf {
    uint l1, addr_lo, addr_hi, bytes, xy, writeback;
};

struct nb_mailbox {
    volatile uint seq;
    volatile uint done;
    volatile uint status;
    uint kernel; /* index into runtime::abi::SYMBOLS */
    uint items;
    uint nargs;
    uint args[NB_MAX_ARGS];
    uint nbufs;
    struct nb_buf buf[NB_MAX_BUFS];
};

/* ---- NoC 0 command buffers (Blackhole noc_parameters.h) ---- */
#define NOC_REGS 0xFFB20000u
#define NOC_REG(buf, off) (*(volatile uint *)(NOC_REGS + (buf) * 0x800u + (off)))
#define NOC_STATUS(i) (*(volatile uint *)(NOC_REGS + 0x200u + (i) * 4u))
#define TARG_LO 0x00u
#define TARG_MID 0x04u
#define TARG_COORD 0x08u
#define RET_LO 0x0Cu
#define RET_MID 0x10u
#define RET_COORD 0x14u
#define CTRL 0x1Cu
#define AT_LEN 0x20u
#define CMD_CTRL 0x40u
#define NODE_ID 0x44u
#define RD_RESP_RECEIVED 0x2u
#define WR_ACK_RECEIVED 0x1u
#define CMD_RD ((0u << 1) | (1u << 4) | (1u << 7) | (1u << 13)) /* CPY|RD|RESP_MARKED|VC_STATIC|VC 1 */
#define CMD_WR ((1u << 1) | (1u << 4) | (1u << 7) | (1u << 13)) /* CPY|WR|RESP_MARKED|VC_STATIC|VC 1 */
#define RD_BUF 1u
#define WR_BUF 0u
#define CHUNK 8192u

static inline void fence(void) { __asm__ volatile("fence" ::: "memory"); }

static uint my_xy;

static void noc_read(uint xy, uint lo, uint hi, uint l1, uint bytes) {
    uint base = NOC_STATUS(RD_RESP_RECEIVED), n = 0;
    for (uint off = 0; off < bytes; off += CHUNK, n++) {
        uint len = bytes - off < CHUNK ? bytes - off : CHUNK;
        while (NOC_REG(RD_BUF, CMD_CTRL) != 0) {}
        NOC_REG(RD_BUF, CTRL) = CMD_RD;
        NOC_REG(RD_BUF, RET_LO) = l1 + off;
        NOC_REG(RD_BUF, RET_MID) = 0;
        NOC_REG(RD_BUF, RET_COORD) = my_xy;
        NOC_REG(RD_BUF, TARG_LO) = lo + off;
        NOC_REG(RD_BUF, TARG_MID) = hi;
        NOC_REG(RD_BUF, TARG_COORD) = xy;
        NOC_REG(RD_BUF, AT_LEN) = len;
        NOC_REG(RD_BUF, CMD_CTRL) = 1;
    }
    while (NOC_STATUS(RD_RESP_RECEIVED) - base != n) {}
}

static void noc_write(uint l1, uint xy, uint lo, uint hi, uint bytes) {
    uint base = NOC_STATUS(WR_ACK_RECEIVED), n = 0;
    for (uint off = 0; off < bytes; off += CHUNK, n++) {
        uint len = bytes - off < CHUNK ? bytes - off : CHUNK;
        while (NOC_REG(WR_BUF, CMD_CTRL) != 0) {}
        NOC_REG(WR_BUF, CTRL) = CMD_WR;
        NOC_REG(WR_BUF, TARG_LO) = l1 + off;
        NOC_REG(WR_BUF, TARG_MID) = 0;
        NOC_REG(WR_BUF, TARG_COORD) = my_xy;
        NOC_REG(WR_BUF, RET_LO) = lo + off;
        NOC_REG(WR_BUF, RET_MID) = hi;
        NOC_REG(WR_BUF, RET_COORD) = xy;
        NOC_REG(WR_BUF, AT_LEN) = len;
        NOC_REG(WR_BUF, CMD_CTRL) = 1;
    }
    while (NOC_STATUS(WR_ACK_RECEIVED) - base != n) {}
}

#define P(i) ((uchar *)(uintptr_t)m->args[i])
#define U(i) (m->args[i])
#define I(i) ((int)m->args[i])
#define F(i) nb_u2f(m->args[i])

/* Kernel indices follow runtime::abi::SYMBOLS. */
static int run(struct nb_mailbox *m) {
    for (nb_host_gid = 0; nb_host_gid < m->items; nb_host_gid++) {
        switch (m->kernel) {
        case 0: nb_convert(P(0), P(1), U(2), U(3), U(4), U(5), U(6)); break;
        case 1: nb_int_convert(P(0), P(1), U(2), U(3), U(4), U(5)); break;
        case 2: nb_nvfp4_tensor_scale(P(0), P(1), U(2), U(3)); break;
        case 3: nb_quantize(P(0), P(1), U(2), U(3), U(4)); break;
        case 4: nb_dequantize(P(0), P(1), U(2), U(3), U(4)); break;
        case 5: nb_gemm_f(P(0), P(1), P(2), U(3), U(4), U(5), U(6), F(7), F(8), U(9), U(10), U(11), U(12), U(13)); break;
        case 6: nb_gemm_i(P(0), P(1), P(2), U(3), U(4), U(5), U(6), I(7), I(8), U(9), U(10), U(11), U(12), U(13)); break;
        case 7: nb_absmax_block(P(0), P(1), U(2), U(3), U(4), U(5)); break;
        case 8: nb_act(P(0), P(1), U(2), U(3), F(4), F(5), U(6), U(7)); break;
        case 9: nb_conv2d(P(0), P(1), P(2), P(3), P(4), U(5), U(6), U(7), U(8)); break;
        case 10: nb_resize(P(0), P(1), U(2), U(3), U(4), U(5), U(6), U(7), U(8), U(9)); break;
        default: return 1;
        }
    }
    return 0;
}

extern char __bss_start[], __bss_end[];

__attribute__((noreturn)) void nb_main(void) {
    for (char *p = __bss_start; p < __bss_end; p++) *p = 0;
    struct nb_mailbox *m = (struct nb_mailbox *)NB_MAILBOX;
    uint id = NOC_REG(0, NODE_ID);
    my_xy = ((id >> 6) & 0x3Fu) << 6 | (id & 0x3Fu);
    fence();
    uint seen = m->done;
    for (;;) {
        fence();
        uint s = m->seq;
        if (s == seen) continue;
        for (uint i = 0; i < m->nbufs; i++)
            noc_read(m->buf[i].xy, m->buf[i].addr_lo, m->buf[i].addr_hi, m->buf[i].l1, m->buf[i].bytes);
        fence();
        uint st = run(m);
        fence();
        for (uint i = 0; i < m->nbufs; i++)
            if (m->buf[i].writeback)
                noc_write(m->buf[i].l1, m->buf[i].xy, m->buf[i].addr_lo, m->buf[i].addr_hi, m->buf[i].bytes);
        m->status = st;
        fence();
        m->done = s;
        fence();
        seen = s;
    }
}

/* Reset entry at L1 address 0: stack, global pointer, then C. */
__attribute__((naked, section(".init"))) void _start(void) {
    __asm__ volatile(".option push\n.option norelax\n"
                     "li sp, %0\n"
                     ".option pop\n"
                     "j nb_main\n" ::"i"(NB_STACK_TOP));
}
