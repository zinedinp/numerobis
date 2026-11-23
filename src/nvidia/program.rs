//! Cubin loading: kernel entry points, register counts, constant banks and
//! parameter layout from the ELF sections and `.nv.info` attributes.
//!
//! The whole cubin is copied to device memory as one image; a kernel's
//! code and its `.nv.constant2` bank are addressed by file offset into it.
//! Cubins that need linking (`.nv.global.init`, `.nv.constant4`, `.rel.*`
//! other than debug info) are rejected; the build uses `-fno-jump-tables` so
//! the shared kernels never need it.

use crate::loader::elf::{Elf, machine};
use crate::runtime::error::{Error, Result};

/// `.nv.info` attribute ids (EIATTR_*).
mod eiattr {
    pub const PARAM_CBANK: u8 = 0x0a;
    pub const FRAME_SIZE: u8 = 0x11;
    pub const MIN_STACK_SIZE: u8 = 0x12;
    pub const KPARAM_INFO: u8 = 0x17;
    pub const MAX_STACK_SIZE: u8 = 0x23;
}
/// `.nv.info` value formats (EIFMT_*); only SVAL carries a payload.
const EIFMT_SVAL: u8 = 4;

/// One kernel entry point inside a cubin image.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Kernel {
    pub name: String,
    /// Code offset/size in the image.
    pub text: (u64, u64),
    pub registers: u32,
    /// Size of constant bank 0 (driver area + parameters).
    pub cbuf0_size: u32,
    /// Parameter block offset/size inside bank 0.
    pub params: (u32, u32),
    /// Parameter offsets by ordinal.
    pub param_offsets: Vec<u32>,
    /// `.nv.constant2.<name>` offset/size in the image (ptxas literal pool).
    pub cbuf2: Option<(u64, u32)>,
    /// Static shared memory.
    pub shared: u32,
}

/// A parsed cubin.
#[derive(Debug, Clone)]
pub struct Program {
    /// SM version from `e_flags` (`0x50` = `sm_80`, ...).
    pub sm: u32,
    pub kernels: Vec<Kernel>,
    /// Largest per-thread stack/frame of any function (local memory).
    pub stack: u32,
}

fn bad(m: String) -> Error {
    Error::BadBinary(m)
}

/// Iterates `(format, attribute, payload)` of a `.nv.info*` section.
fn attrs(d: &[u8]) -> Result<Vec<(u8, u8, &[u8])>> {
    let mut out = Vec::new();
    let mut o = 0;
    while o + 4 <= d.len() {
        let (fmt, attr) = (d[o], d[o + 1]);
        let sz = u16::from_le_bytes([d[o + 2], d[o + 3]]) as usize;
        if fmt == EIFMT_SVAL {
            let p = d
                .get(o + 4..o + 4 + sz)
                .ok_or_else(|| bad("truncated .nv.info".into()))?;
            out.push((fmt, attr, p));
            o += 4 + sz;
        } else {
            out.push((fmt, attr, &d[o + 2..o + 4]));
            o += 4;
        }
    }
    Ok(out)
}

fn u32le(b: &[u8], o: usize) -> u32 {
    u32::from_le_bytes(b[o..o + 4].try_into().unwrap())
}

impl Program {
    /// Parses a cubin: its kernels, constant banks and parameter layout.
    ///
    /// # Errors
    ///
    /// [`Error::BadBinary`] for anything but a linked cubin whose kernels
    /// have constant bank 0 and gap-free parameters (`EIATTR_PARAM_CBANK`).
    pub fn parse(image: &[u8]) -> Result<Program> {
        let elf = Elf::parse(image)?;
        if elf.machine != machine::CUDA {
            return Err(bad(format!("not a cubin (e_machine {})", elf.machine)));
        }
        // e_flags: SM version in bits 8..16 for cubins built by ptxas >= 12 (0x50 -> sm_80).
        let sm = (elf.flags >> 8) & 0xff;
        for s in &elf.sections {
            let needs_link = s.name == ".nv.global.init"
                || s.name.starts_with(".nv.constant4")
                || (s.name.starts_with(".rel")
                    && s.name != ".rel.debug_frame"
                    && s.name != ".rela.debug_frame");
            if needs_link {
                return Err(bad(format!("cubin needs linking ({})", s.name)));
            }
        }
        let mut stack = 0;
        if let Some(info) = elf.section(".nv.info") {
            for (_, a, p) in attrs(info.data)? {
                if matches!(
                    a,
                    eiattr::FRAME_SIZE | eiattr::MIN_STACK_SIZE | eiattr::MAX_STACK_SIZE
                ) && p.len() >= 8
                {
                    stack = stack.max(u32le(p, 4));
                }
            }
        }
        let mut kernels = Vec::new();
        for s in elf.sections.iter().filter(|s| s.name.starts_with(".text.")) {
            let name = &s.name[".text.".len()..];
            let Some(info) = elf.section(&format!(".nv.info.{name}")) else {
                continue;
            };
            let cbuf0 = elf
                .section(&format!(".nv.constant0.{name}"))
                .ok_or_else(|| bad(format!("{name}: no constant bank 0")))?;
            let mut params = None;
            let mut offsets = Vec::new();
            for (_, a, p) in attrs(info.data)? {
                match a {
                    eiattr::PARAM_CBANK if p.len() >= 8 => {
                        let v = u32le(p, 4);
                        params = Some((v & 0xffff, v >> 16));
                    }
                    eiattr::KPARAM_INFO if p.len() >= 12 => {
                        let ordinal = u16::from_le_bytes([p[4], p[5]]) as usize;
                        let off = u32::from(u16::from_le_bytes([p[6], p[7]]));
                        if offsets.len() <= ordinal {
                            offsets.resize(ordinal + 1, u32::MAX);
                        }
                        offsets[ordinal] = off;
                    }
                    _ => {}
                }
            }
            let params = params.ok_or_else(|| bad(format!("{name}: no EIATTR_PARAM_CBANK")))?;
            if offsets.contains(&u32::MAX) {
                return Err(bad(format!("{name}: parameter ordinals have gaps")));
            }
            let cbuf2 = elf
                .section(&format!(".nv.constant2.{name}"))
                .map(|c| (c.offset, c.data.len() as u32));
            let shared = elf
                .section(&format!(".nv.shared.{name}"))
                .map_or(0, |c| c.size as u32);
            kernels.push(Kernel {
                name: name.to_string(),
                text: (s.offset, s.data.len() as u64),
                registers: s.info >> 24,
                cbuf0_size: cbuf0.data.len() as u32,
                params,
                param_offsets: offsets,
                cbuf2,
                shared,
            });
        }
        Ok(Program { sm, kernels, stack })
    }

    #[must_use]
    pub fn kernel(&self, name: &str) -> Option<&Kernel> {
        self.kernels.iter().find(|k| k.name == name)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cubin(sm: &str) -> Vec<u8> {
        std::fs::read(format!(
            "{}/kernels/bin/nvidia/{sm}-emulated.cubin",
            env!("CARGO_MANIFEST_DIR")
        ))
        .unwrap()
    }

    #[test]
    fn parses_shared_kernels() {
        // The raw path opens Ampere/Ada only; Sm90+ cubins carry `.rela.text.*`
        // and go through the cuda comfort backend first.
        for sm in ["sm_80", "sm_89"] {
            let p = Program::parse(&cubin(sm)).unwrap_or_else(|e| panic!("{sm}: {e}"));
            for s in crate::runtime::abi::SYMBOLS {
                let k = p.kernel(s).unwrap_or_else(|| panic!("{sm}: {s}"));
                assert!(
                    k.registers > 0 && k.registers <= 255,
                    "{sm} {s}: {} registers",
                    k.registers
                );
                assert!(k.params.0 + k.params.1 <= k.cbuf0_size);
            }
            // nb_gemm_f(a, b, c, m, n, k, batch, alpha, beta, fa, fb, fc, sa, sb):
            // three pointers, then eleven 32-bit values, naturally aligned.
            let g = p.kernel("nb_gemm_f").unwrap();
            assert_eq!(g.params, (0x160, 3 * 8 + 11 * 4), "{sm}");
            let want: Vec<u32> = (0..14)
                .map(|i| if i < 3 { i * 8 } else { 24 + (i - 3) * 4 })
                .collect();
            assert_eq!(g.param_offsets, want, "{sm}");
        }
    }

    #[test]
    fn rejects_cubins_that_need_linking() {
        assert!(matches!(
            Program::parse(&cubin("sm_90")),
            Err(Error::BadBinary(_))
        ));
    }

    #[test]
    fn sm_version_from_flags() {
        assert_eq!(Program::parse(&cubin("sm_80")).unwrap().sm, 80);
        assert_eq!(Program::parse(&cubin("sm_89")).unwrap().sm, 89);
    }
}
