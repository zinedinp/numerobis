//! Minimal ELF little-endian reader (ELF64 and ELF32) for kernel binaries:
//! AMD HSACO, NVIDIA cubin, Intel Gen ELF (ocloc), RISC-V ELF32 (Tenstorrent). Only parses what a
//! loader needs (sections, symbols, notes); never executes or relocates.

use crate::runtime::error::{Error, Result};

/// `e_machine` values of the targets numerobis loads.
pub mod machine {
    pub const X86_64: u16 = 62;
    pub const AARCH64: u16 = 183;
    pub const CUDA: u16 = 190;
    pub const AMDGPU: u16 = 224;
    pub const RISCV: u16 = 243;
    /// Intel GPU (ocloc output).
    pub const INTEL_GT: u16 = 205;
}

/// Section header type: symbol table.
pub const SHT_SYMTAB: u32 = 2;
/// Section header type: note.
pub const SHT_NOTE: u32 = 7;
/// Section header type: no file data (`.bss`).
pub const SHT_NOBITS: u32 = 8;
/// Section header type: dynamic symbol table.
pub const SHT_DYNSYM: u32 = 11;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Section<'a> {
    pub name: &'a str,
    pub kind: u32,
    pub flags: u64,
    pub addr: u64,
    pub data: &'a [u8],
    pub link: u32,
    /// `sh_info` (cubin `.text.*`: register count in bits 24..32).
    pub info: u32,
    /// File offset of the section data.
    pub offset: u64,
    /// `sh_size` (also for `SHT_NOBITS`, whose `data` is empty).
    pub size: u64,
    pub entsize: u64,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Symbol<'a> {
    pub name: &'a str,
    pub value: u64,
    pub size: u64,
    pub info: u8,
    pub shndx: u16,
}

/// A parsed ELF image borrowing the input bytes.
#[derive(Debug, Clone)]
pub struct Elf<'a> {
    /// ELF32 (RISC-V kernels) rather than ELF64.
    pub is32: bool,
    pub machine: u16,
    /// Entry point.
    pub entry: u64,
    /// `e_flags` (gfx target on AMDGPU, SM version on CUDA).
    pub flags: u32,
    pub sections: Vec<Section<'a>>,
}

fn bad(msg: &str) -> Error {
    Error::BadBinary(msg.into())
}

fn rd<const N: usize>(b: &[u8], off: usize) -> Result<[u8; N]> {
    off.checked_add(N)
        .and_then(|end| b.get(off..end))
        .and_then(|s| s.try_into().ok())
        .ok_or_else(|| bad("truncated"))
}
fn u16_at(b: &[u8], o: usize) -> Result<u16> {
    Ok(u16::from_le_bytes(rd(b, o)?))
}
fn u32_at(b: &[u8], o: usize) -> Result<u32> {
    Ok(u32::from_le_bytes(rd(b, o)?))
}
fn u64_at(b: &[u8], o: usize) -> Result<u64> {
    Ok(u64::from_le_bytes(rd(b, o)?))
}
fn usize_of(v: u64) -> Result<usize> {
    usize::try_from(v).map_err(|_| bad("offset overflow"))
}

fn cstr(b: &[u8], off: usize) -> Result<&str> {
    let s = b.get(off..).ok_or_else(|| bad("string offset"))?;
    let end = s
        .iter()
        .position(|&c| c == 0)
        .ok_or_else(|| bad("unterminated string"))?;
    core::str::from_utf8(&s[..end]).map_err(|_| bad("non-UTF-8 name"))
}

/// Reads the fields whose offset or width follows the ELF class.
#[derive(Clone, Copy)]
struct Class<'a> {
    b: &'a [u8],
    is32: bool,
}

impl<'a> Class<'a> {
    /// A word-size field: `u32` at `at + o32` in ELF32, `u64` at `at + o64` in ELF64.
    fn word(self, at: usize, o32: usize, o64: usize) -> Result<u64> {
        if self.is32 {
            Ok(u64::from(u32_at(self.b, at + o32)?))
        } else {
            u64_at(self.b, at + o64)
        }
    }

    /// A `u32` field at a class-dependent offset.
    fn u32(self, at: usize, o32: usize, o64: usize) -> Result<u32> {
        u32_at(self.b, at + if self.is32 { o32 } else { o64 })
    }

    /// A `u16` file header field at a class-dependent offset.
    fn u16(self, o32: usize, o64: usize) -> Result<usize> {
        Ok(usize::from(u16_at(
            self.b,
            if self.is32 { o32 } else { o64 },
        )?))
    }

    /// Section header `i`, with its name still an offset into the section
    /// name table.
    fn section(self, shoff: usize, shentsize: usize, i: usize) -> Result<(u32, Section<'a>)> {
        let h = i
            .checked_mul(shentsize)
            .and_then(|o| o.checked_add(shoff))
            .ok_or_else(|| bad("overflow"))?;
        if h.checked_add(shentsize)
            .is_none_or(|end| end > self.b.len())
        {
            return Err(bad("section header out of range"));
        }
        let kind = u32_at(self.b, h + 4)?;
        let offset = self.word(h, 16, 24)?;
        let size = self.word(h, 20, 32)?;
        let (off, len) = (usize_of(offset)?, usize_of(size)?);
        let data = if kind == SHT_NOBITS {
            &self.b[..0]
        } else {
            off.checked_add(len)
                .and_then(|end| self.b.get(off..end))
                .ok_or_else(|| bad("section out of range"))?
        };
        let section = Section {
            name: "",
            kind,
            flags: self.word(h, 8, 8)?,
            addr: self.word(h, 12, 16)?,
            data,
            link: self.u32(h, 24, 40)?,
            info: self.u32(h, 28, 44)?,
            offset,
            size,
            entsize: self.word(h, 36, 56)?,
        };
        Ok((u32_at(self.b, h)?, section))
    }
}

impl<'a> Elf<'a> {
    /// Parses the file header and the section table.
    ///
    /// # Errors
    ///
    /// [`Error::BadBinary`] for anything but little-endian ELF32/ELF64, and
    /// for headers, sections or names outside the file.
    pub fn parse(b: &'a [u8]) -> Result<Elf<'a>> {
        if b.get(..4) != Some(b"ELF") {
            return Err(bad("not ELF"));
        }
        let is32 = match (b.get(4), b.get(5)) {
            (Some(1), Some(1)) => true,
            (Some(2), Some(1)) => false,
            _ => return Err(bad("only little-endian ELF32/ELF64 is supported")),
        };
        let class = Class { b, is32 };
        let mut elf = Elf {
            is32,
            machine: u16_at(b, 18)?,
            entry: class.word(0, 24, 24)?,
            flags: class.u32(0, 36, 48)?,
            sections: Vec::new(),
        };
        let shoff = usize_of(class.word(0, 32, 40)?)?;
        let (shentsize, shnum, shstrndx) =
            (class.u16(46, 58)?, class.u16(48, 60)?, class.u16(50, 62)?);
        if shnum == 0 {
            return Ok(elf);
        }
        if shentsize < if is32 { 40 } else { 64 } {
            return Err(bad("section header size"));
        }
        let names = class.section(shoff, shentsize, shstrndx)?.1.data;
        elf.sections.reserve_exact(shnum);
        for i in 0..shnum {
            let (name, mut section) = class.section(shoff, shentsize, i)?;
            section.name = cstr(names, name as usize)?;
            elf.sections.push(section);
        }
        Ok(elf)
    }

    #[must_use]
    pub fn section(&self, name: &str) -> Option<&Section<'a>> {
        self.sections.iter().find(|s| s.name == name)
    }

    /// Symbols from `.symtab` and `.dynsym`.
    ///
    /// # Errors
    ///
    /// [`Error::BadBinary`] for a symbol table whose string table link or
    /// names are out of range.
    pub fn symbols(&self) -> Result<Vec<Symbol<'a>>> {
        let mut out = Vec::new();
        for s in self
            .sections
            .iter()
            .filter(|s| s.kind == SHT_SYMTAB || s.kind == SHT_DYNSYM)
        {
            let strs = self
                .sections
                .get(s.link as usize)
                .ok_or_else(|| bad("symtab link"))?
                .data;
            if self.is32 {
                for e in s.data.as_chunks::<16>().0 {
                    out.push(Symbol {
                        name: cstr(strs, u32_at(e, 0)? as usize)?,
                        value: u64::from(u32_at(e, 4)?),
                        size: u64::from(u32_at(e, 8)?),
                        info: e[12],
                        shndx: u16_at(e, 14)?,
                    });
                }
            } else {
                for e in s.data.as_chunks::<24>().0 {
                    out.push(Symbol {
                        name: cstr(strs, u32_at(e, 0)? as usize)?,
                        info: e[4],
                        shndx: u16_at(e, 6)?,
                        value: u64_at(e, 8)?,
                        size: u64_at(e, 16)?,
                    });
                }
            }
        }
        Ok(out)
    }

    /// The first symbol called `name`.
    ///
    /// # Errors
    ///
    /// As [`symbols`](Self::symbols).
    pub fn symbol(&self, name: &str) -> Result<Option<Symbol<'a>>> {
        Ok(self.symbols()?.into_iter().find(|s| s.name == name))
    }

    /// Notes as `(owner, type, desc)` from all `SHT_NOTE` sections (HSACO metadata lives here).
    ///
    /// # Errors
    ///
    /// [`Error::BadBinary`] for a note whose name or descriptor runs past its
    /// section, or whose owner is not UTF-8.
    pub fn notes(&self) -> Result<Vec<(&'a str, u32, &'a [u8])>> {
        let mut out = Vec::new();
        for s in self.sections.iter().filter(|s| s.kind == SHT_NOTE) {
            let d = s.data;
            let mut o = 0;
            while o + 12 <= d.len() {
                let (nsz, dsz, ty) = (
                    u32_at(d, o)? as usize,
                    u32_at(d, o + 4)? as usize,
                    u32_at(d, o + 8)?,
                );
                let n0 = o + 12;
                let d0 = n0 + nsz.next_multiple_of(4);
                let name = d.get(n0..n0 + nsz).ok_or_else(|| bad("note name"))?;
                let name = core::str::from_utf8(name.strip_suffix(&[0]).unwrap_or(name))
                    .map_err(|_| bad("note name"))?;
                out.push((
                    name,
                    ty,
                    d.get(d0..d0 + dsz).ok_or_else(|| bad("note desc"))?,
                ));
                o = d0 + dsz.next_multiple_of(4);
            }
        }
        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Its own binary is an ELF file on Linux only.
    #[cfg(target_os = "linux")]
    #[test]
    fn parses_own_test_binary() {
        let exe = std::fs::read("/proc/self/exe").unwrap();
        let elf = Elf::parse(&exe).unwrap();
        #[cfg(target_arch = "x86_64")]
        assert_eq!(elf.machine, machine::X86_64);
        assert!(elf.section(".text").is_some_and(|s| !s.data.is_empty()));
        let syms = elf.symbols().unwrap();
        assert!(
            syms.iter()
                .any(|s| s.name.contains("parses_own_test_binary"))
        );
    }

    #[test]
    fn rejects_garbage() {
        assert!(Elf::parse(b"nope").is_err());
        let mut h = vec![0u8; 64];
        h[..4].copy_from_slice(b"\x7fELF");
        h[4] = 3;
        assert!(Elf::parse(&h).is_err());
        h[4] = 2;
        h[5] = 1;
        h[40] = 0xff; // shoff beyond the file
        h[60] = 1;
        h[58] = 64;
        assert!(Elf::parse(&h).is_err());
    }
}
