/* BRISC image for numerobis on Tenstorrent Blackhole. BRISC's reset PC is
 * hard-wired to L1 address 0, so the image is linked there. Layout:
 *   0x00000  code, rodata, data, bss  (< 0x18000)
 *   0x18000  mailbox (struct nb_mailbox, kernels/tt/runtime.c)
 *   0x20000  stack top (grows down)
 *   0x20000.. staging area for launch buffers (runtime::tt) */
ENTRY(_start)
MEMORY { L1 (rwx) : ORIGIN = 0, LENGTH = 0x18000 }
SECTIONS {
    .text : { KEEP(*(.init)) *(.text .text.*) } > L1
    .rodata : { *(.rodata .rodata.* .srodata .srodata.*) } > L1
    .data : { *(.data .data.* .sdata .sdata.*) } > L1
    .bss (NOLOAD) : { __bss_start = .; *(.bss .bss.* .sbss .sbss.* COMMON) __bss_end = .; } > L1
    /DISCARD/ : { *(.comment) *(.riscv.attributes) *(.note*) *(.eh_frame*) }
}
