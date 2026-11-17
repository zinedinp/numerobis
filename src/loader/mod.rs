//! Kernel binary loaders. One module per container format; the per-vendor
//! backends turn the parsed image into a loaded kernel.

pub mod elf;
