//! Minimal ELF64 little-endian reader for kernel binaries: AMD HSACO, NVIDIA
//! cubin, Intel Gen ELF (ocloc), RISC-V ELF (Tenstorrent). Only parses what a
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

/// A parsed ELF64 image borrowing the input bytes.
#[derive(Debug, Clone)]
pub struct Elf<'a> {
    pub machine: u16,
    /// `e_flags` (gfx target on AMDGPU, SM version on CUDA).
    pub flags: u32,
    pub sections: Vec<Section<'a>>,
}

/// `(sh_name, sh_type, sh_flags, sh_addr, data, sh_link, sh_info, sh_offset, sh_size, sh_entsize)`.
type Shdr<'a> = (u32, u32, u64, u64, &'a [u8], u32, u32, u64, u64, u64);

fn bad(msg: &str) -> Error {
    Error::BadBinary(msg.into())
}

fn rd<const N: usize>(b: &[u8], off: usize) -> Result<[u8; N]> {
    b.get(off..off + N).and_then(|s| s.try_into().ok()).ok_or_else(|| bad("truncated"))
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
    let end = s.iter().position(|&c| c == 0).ok_or_else(|| bad("unterminated string"))?;
    core::str::from_utf8(&s[..end]).map_err(|_| bad("non-UTF-8 name"))
}

impl<'a> Elf<'a> {
    pub fn parse(b: &'a [u8]) -> Result<Elf<'a>> {
        if b.get(..4) != Some(b"\x7fELF") {
            return Err(bad("not ELF"));
        }
        if b.get(4) != Some(&2) || b.get(5) != Some(&1) {
            return Err(bad("only ELF64 little-endian is supported"));
        }
        let machine = u16_at(b, 18)?;
        let flags = u32_at(b, 48)?;
        let shoff = usize_of(u64_at(b, 40)?)?;
        let shentsize = u16_at(b, 58)? as usize;
        let shnum = u16_at(b, 60)? as usize;
        let shstrndx = u16_at(b, 62)? as usize;
        if shnum == 0 {
            return Ok(Elf { machine, flags, sections: Vec::new() });
        }
        if shentsize < 64 {
            return Err(bad("section header size"));
        }
        let hdr = |i: usize| -> Result<Shdr<'a>> {
            let h = shoff.checked_add(i.checked_mul(shentsize).ok_or_else(|| bad("overflow"))?).ok_or_else(|| bad("overflow"))?;
            let kind = u32_at(b, h + 4)?;
            let (off, size) = (usize_of(u64_at(b, h + 24)?)?, usize_of(u64_at(b, h + 32)?)?);
            // SHT_NOBITS (.bss) has no file data.
            let data = if kind == 8 { &b[..0] } else { b.get(off..off.checked_add(size).ok_or_else(|| bad("overflow"))?).ok_or_else(|| bad("section out of range"))? };
            Ok((u32_at(b, h)?, kind, u64_at(b, h + 8)?, u64_at(b, h + 16)?, data, u32_at(b, h + 40)?, u32_at(b, h + 44)?, off as u64, size as u64, u64_at(b, h + 56)?))
        };
        let strtab = hdr(shstrndx)?.4;
        let mut sections = Vec::with_capacity(shnum);
        for i in 0..shnum {
            let (name, kind, flags, addr, data, link, info, offset, size, entsize) = hdr(i)?;
            sections.push(Section { name: cstr(strtab, name as usize)?, kind, flags, addr, data, link, info, offset, size, entsize });
        }
        Ok(Elf { machine, flags, sections })
    }

    pub fn section(&self, name: &str) -> Option<&Section<'a>> {
        self.sections.iter().find(|s| s.name == name)
    }

    /// Symbols from `.symtab` and `.dynsym`.
    pub fn symbols(&self) -> Result<Vec<Symbol<'a>>> {
        let mut out = Vec::new();
        for s in self.sections.iter().filter(|s| s.kind == SHT_SYMTAB || s.kind == SHT_DYNSYM) {
            let strs = self.sections.get(s.link as usize).ok_or_else(|| bad("symtab link"))?.data;
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
        Ok(out)
    }

    pub fn symbol(&self, name: &str) -> Result<Option<Symbol<'a>>> {
        Ok(self.symbols()?.into_iter().find(|s| s.name == name))
    }

    /// Notes as `(owner, type, desc)` from all `SHT_NOTE` sections (HSACO metadata lives here).
    pub fn notes(&self) -> Result<Vec<(&'a str, u32, &'a [u8])>> {
        let mut out = Vec::new();
        for s in self.sections.iter().filter(|s| s.kind == SHT_NOTE) {
            let d = s.data;
            let mut o = 0;
            while o + 12 <= d.len() {
                let (nsz, dsz, ty) = (u32_at(d, o)? as usize, u32_at(d, o + 4)? as usize, u32_at(d, o + 8)?);
                let n0 = o + 12;
                let d0 = n0 + nsz.next_multiple_of(4);
                let name = d.get(n0..n0 + nsz).ok_or_else(|| bad("note name"))?;
                let name = core::str::from_utf8(name.strip_suffix(&[0]).unwrap_or(name)).map_err(|_| bad("note name"))?;
                out.push((name, ty, d.get(d0..d0 + dsz).ok_or_else(|| bad("note desc"))?));
                o = d0 + dsz.next_multiple_of(4);
            }
        }
        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_own_test_binary() {
        let exe = std::fs::read("/proc/self/exe").unwrap();
        let elf = Elf::parse(&exe).unwrap();
        #[cfg(target_arch = "x86_64")]
        assert_eq!(elf.machine, machine::X86_64);
        assert!(elf.section(".text").is_some_and(|s| !s.data.is_empty()));
        let syms = elf.symbols().unwrap();
        assert!(syms.iter().any(|s| s.name.contains("parses_own_test_binary")));
    }

    #[test]
    fn rejects_garbage() {
        assert!(Elf::parse(b"nope").is_err());
        let mut h = vec![0u8; 64];
        h[..4].copy_from_slice(b"\x7fELF");
        h[4] = 1;
        assert!(Elf::parse(&h).is_err());
        h[4] = 2;
        h[5] = 1;
        h[40] = 0xff; // shoff beyond the file
        h[60] = 1;
        h[58] = 64;
        assert!(Elf::parse(&h).is_err());
    }
}
