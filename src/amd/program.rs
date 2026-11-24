//! AMD code objects (HSACO, code object v4 to v6): the loadable image, the
//! kernel descriptors (`<name>.kd`) and the kernel argument layout from the
//! `NT_AMDGPU_METADATA` note, including the implicit arguments that code
//! object v5 reads instead of the dispatch packet (`hidden_block_count_x`,
//! `hidden_group_size_x`, ...).
//!
//! The loader places the image at one GPU address as is: it accepts no
//! relocations and no kernels that need scratch memory (the raw queue sets up
//! none). The shared Emulated kernels need neither.

use crate::loader::elf::{Elf, machine};
use crate::loader::msgpack::{self, Value};
use crate::runtime::error::{Error, Result};

/// `NT_AMDGPU_METADATA`.
const NT_AMDGPU_METADATA: u32 = 32;
/// `SHF_ALLOC`.
const SHF_ALLOC: u64 = 2;
/// `SHT_RELA`, `SHT_REL`.
const SHT_RELA: u32 = 4;
const SHT_REL: u32 = 9;
/// Size of `kernel_descriptor_t`.
pub const DESCRIPTOR_SIZE: usize = 64;

fn bad(what: impl Into<String>) -> Error {
    Error::BadBinary(what.into())
}

/// One dimension of a 3-D launch.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Dim {
    X,
    Y,
    Z,
}

/// An implicit argument the runtime fills (code object v5).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Hidden {
    /// Work groups in that dimension, full ones only.
    BlockCount(Dim),
    GroupSize(Dim),
    /// Work items in the last, partial group (0 when the grid divides).
    Remainder(Dim),
    GlobalOffset(Dim),
    GridDims,
    /// Anything else (`hidden_printf_buffer`, `hidden_queue_ptr`, ...):
    /// left zero, which the shared kernels never read.
    Other(String),
}

/// What an argument slot holds.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ArgKind {
    /// A device pointer (`global_buffer`).
    Buffer,
    /// A value passed as is (`by_value`).
    Value,
    Hidden(Hidden),
}

/// One kernel argument slot in the kernarg segment.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct KernArg {
    pub offset: u32,
    pub size: u32,
    pub kind: ArgKind,
}

/// An explicit argument value for [`Kernel::kernargs`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Word {
    /// A GPU virtual address, for a [`ArgKind::Buffer`] slot.
    Ptr(u64),
    /// 32 bits, for a 4-byte [`ArgKind::Value`] slot.
    B32(u32),
}

/// One kernel of a code object.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Kernel {
    pub name: String,
    /// Offset of the kernel descriptor in [`Program::image`].
    pub descriptor: u64,
    pub kernarg_size: u32,
    pub kernarg_align: u32,
    /// LDS bytes.
    pub group_segment: u32,
    /// Scratch bytes per work item.
    pub private_segment: u32,
    pub dynamic_stack: bool,
    pub wavefront: u32,
    pub max_workgroup: u32,
    pub args: Vec<KernArg>,
}

impl Kernel {
    /// Explicit arguments, in declaration order.
    pub fn explicit(&self) -> impl Iterator<Item = &KernArg> {
        self.args
            .iter()
            .filter(|a| !matches!(a.kind, ArgKind::Hidden(_)))
    }

    /// The kernarg segment for a 1-D launch of `grid` work items (a multiple
    /// of `group`) with `args` for the explicit slots.
    ///
    /// # Errors
    ///
    /// [`Error::InvalidArgument`] when `args` doesn't match the explicit
    /// slots in count or kind, or `grid` isn't a multiple of `group`.
    pub fn kernargs(&self, args: &[Word], grid: u32, group: u16) -> Result<Vec<u8>> {
        let explicit = self.explicit().count();
        if args.len() != explicit {
            return Err(Error::InvalidArgument(format!(
                "{}: {} args for {explicit} parameters",
                self.name,
                args.len()
            )));
        }
        if group == 0 || !grid.is_multiple_of(u32::from(group)) {
            return Err(Error::InvalidArgument(format!(
                "{}: grid {grid} is not a multiple of the group {group}",
                self.name
            )));
        }
        let mut seg = vec![0u8; self.kernarg_size as usize];
        let mut put = |slot: &KernArg, bytes: &[u8]| -> Result<()> {
            let at = slot.offset as usize;
            let dst = seg
                .get_mut(at..at + bytes.len())
                .filter(|_| bytes.len() <= slot.size as usize)
                .ok_or_else(|| {
                    bad(format!(
                        "{}: argument at {at} outside the segment",
                        self.name
                    ))
                })?;
            dst.copy_from_slice(bytes);
            Ok(())
        };
        let mut values = args.iter();
        for slot in &self.args {
            let dim = |d: &Dim| match d {
                Dim::X => 0,
                _ => 1,
            };
            match (&slot.kind, slot.size) {
                (ArgKind::Buffer, 8) => match values.next() {
                    Some(Word::Ptr(va)) => put(slot, &va.to_le_bytes())?,
                    _ => return Err(self.mismatch(slot)),
                },
                (ArgKind::Value, 4) => match values.next() {
                    Some(Word::B32(v)) => put(slot, &v.to_le_bytes())?,
                    _ => return Err(self.mismatch(slot)),
                },
                (ArgKind::Buffer | ArgKind::Value, _) => return Err(self.mismatch(slot)),
                (ArgKind::Hidden(h), _) => match h {
                    // 1-D: one block in Y and Z, groups of 1.
                    Hidden::BlockCount(d) => {
                        let n = if dim(d) == 0 {
                            grid / u32::from(group)
                        } else {
                            1
                        };
                        put(slot, &n.to_le_bytes())?;
                    }
                    Hidden::GroupSize(d) => {
                        let n: u16 = if dim(d) == 0 { group } else { 1 };
                        put(slot, &n.to_le_bytes())?;
                    }
                    Hidden::GridDims => put(slot, &1u16.to_le_bytes())?,
                    // The grid divides, there is no offset; others stay zero.
                    Hidden::Remainder(_) | Hidden::GlobalOffset(_) | Hidden::Other(_) => {}
                },
            }
        }
        Ok(seg)
    }

    fn mismatch(&self, slot: &KernArg) -> Error {
        Error::InvalidArgument(format!(
            "{}: argument at offset {} ({:?}, {} bytes) gets another kind",
            self.name, slot.offset, slot.kind, slot.size
        ))
    }
}

/// A parsed code object.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Program {
    /// `amdhsa.target`, e.g. `amdgcn-amd-amdhsa-unknown-gfx10-3-generic`.
    pub target: String,
    /// Every allocated section at its address, the rest zero: copied to
    /// one GPU address as is.
    pub image: Vec<u8>,
    pub kernels: Vec<Kernel>,
}

fn field<'a>(v: &'a Value<'a>, key: &str, kernel: &str) -> Result<&'a Value<'a>> {
    v.get(key)
        .ok_or_else(|| bad(format!("{kernel}: metadata without {key}")))
}

fn u32_field(v: &Value<'_>, key: &str, kernel: &str) -> Result<u32> {
    field(v, key, kernel)?
        .as_u64()
        .and_then(|n| u32::try_from(n).ok())
        .ok_or_else(|| bad(format!("{kernel}: {key} is not a 32-bit number")))
}

fn arg_kind(kind: &str) -> ArgKind {
    let dim = |s: &str| match s {
        "x" => Some(Dim::X),
        "y" => Some(Dim::Y),
        "z" => Some(Dim::Z),
        _ => None,
    };
    let hidden = |h| ArgKind::Hidden(h);
    match kind {
        "global_buffer" => ArgKind::Buffer,
        "by_value" => ArgKind::Value,
        "hidden_grid_dims" => hidden(Hidden::GridDims),
        _ => {
            let (stem, d) = kind.rsplit_once('_').unwrap_or((kind, ""));
            match (stem, dim(d)) {
                ("hidden_block_count", Some(d)) => hidden(Hidden::BlockCount(d)),
                ("hidden_group_size", Some(d)) => hidden(Hidden::GroupSize(d)),
                ("hidden_remainder", Some(d)) => hidden(Hidden::Remainder(d)),
                ("hidden_global_offset", Some(d)) => hidden(Hidden::GlobalOffset(d)),
                _ => hidden(Hidden::Other(kind.to_string())),
            }
        }
    }
}

impl Program {
    /// Parses an HSACO.
    ///
    /// # Errors
    ///
    /// [`Error::BadBinary`] for a file that isn't an AMDGPU ELF, carries
    /// relocations, lacks the metadata, or whose descriptors or arguments
    /// fall outside the image.
    pub fn parse(bytes: &[u8]) -> Result<Program> {
        let elf = Elf::parse(bytes)?;
        if elf.machine != machine::AMDGPU {
            return Err(bad("not an AMDGPU code object"));
        }
        if elf
            .sections
            .iter()
            .any(|s| (s.kind == SHT_RELA || s.kind == SHT_REL) && s.size > 0)
        {
            return Err(bad("code object with relocations"));
        }
        let mut image = Vec::new();
        for s in elf.sections.iter().filter(|s| s.flags & SHF_ALLOC != 0) {
            let at = usize::try_from(s.addr).map_err(|_| bad("section address"))?;
            let size = usize::try_from(s.size).map_err(|_| bad("section size"))?;
            let end = at.checked_add(size).ok_or_else(|| bad("section end"))?;
            if end > image.len() {
                image.resize(end, 0);
            }
            image[at..at + s.data.len()].copy_from_slice(s.data);
        }

        let (_, _, desc) = elf
            .notes()?
            .into_iter()
            .find(|(owner, ty, _)| *owner == "AMDGPU" && *ty == NT_AMDGPU_METADATA)
            .ok_or_else(|| bad("no AMDGPU metadata note"))?;
        let meta = msgpack::decode(desc)?;
        let target = meta
            .get("amdhsa.target")
            .and_then(Value::as_str)
            .unwrap_or_default()
            .to_string();
        let symbols = elf.symbols()?;
        let mut kernels = Vec::new();
        for k in meta
            .get("amdhsa.kernels")
            .and_then(Value::as_array)
            .ok_or_else(|| bad("metadata without amdhsa.kernels"))?
        {
            let name = field(k, ".name", "?")?
                .as_str()
                .ok_or_else(|| bad("kernel name"))?;
            let symbol = field(k, ".symbol", name)?
                .as_str()
                .ok_or_else(|| bad("kernel symbol"))?;
            let descriptor = symbols
                .iter()
                .find(|s| s.name == symbol && s.shndx != 0)
                .ok_or_else(|| bad(format!("{name}: no descriptor {symbol}")))?
                .value;
            let kd = usize::try_from(descriptor)
                .ok()
                .and_then(|d| image.get(d..d.checked_add(DESCRIPTOR_SIZE)?))
                .ok_or_else(|| bad(format!("{name}: descriptor outside the image")))?;
            // kernel_code_entry_byte_offset, relative to the descriptor.
            let mut entry = [0u8; 8];
            entry.copy_from_slice(&kd[16..24]);
            let entry = i64::from_le_bytes(entry);
            let code = descriptor.checked_add_signed(entry);
            if code.is_none_or(|c| c >= image.len() as u64) {
                return Err(bad(format!("{name}: entry point outside the image")));
            }
            let mut args = Vec::new();
            for a in field(k, ".args", name)?.as_array().unwrap_or_default() {
                let kind = field(a, ".value_kind", name)?
                    .as_str()
                    .ok_or_else(|| bad("value kind"))?;
                args.push(KernArg {
                    offset: u32_field(a, ".offset", name)?,
                    size: u32_field(a, ".size", name)?,
                    kind: arg_kind(kind),
                });
            }
            let kernarg_size = u32_field(k, ".kernarg_segment_size", name)?;
            if args.iter().any(|a| a.offset + a.size > kernarg_size) {
                return Err(bad(format!("{name}: argument outside the kernarg segment")));
            }
            kernels.push(Kernel {
                name: name.to_string(),
                descriptor,
                kernarg_size,
                kernarg_align: u32_field(k, ".kernarg_segment_align", name)?,
                group_segment: u32_field(k, ".group_segment_fixed_size", name)?,
                private_segment: u32_field(k, ".private_segment_fixed_size", name)?,
                dynamic_stack: k.get(".uses_dynamic_stack") == Some(&Value::Bool(true)),
                wavefront: u32_field(k, ".wavefront_size", name)?,
                max_workgroup: u32_field(k, ".max_flat_workgroup_size", name)?,
                args,
            });
        }
        Ok(Program {
            target,
            image,
            kernels,
        })
    }

    #[must_use]
    pub fn kernel(&self, name: &str) -> Option<&Kernel> {
        self.kernels.iter().find(|k| k.name == name)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::runtime::abi::SYMBOLS;

    static RDNA2: &[u8] = include_bytes!("../../kernels/bin/amd/gfx10-3-generic-emulated.hsaco");

    #[test]
    fn the_shared_kernels_load_without_scratch() {
        let p = Program::parse(RDNA2).unwrap();
        assert_eq!(p.target, "amdgcn-amd-amdhsa-unknown-gfx10-3-generic");
        for s in SYMBOLS {
            let k = p.kernel(s).unwrap_or_else(|| panic!("{s}"));
            assert_eq!((k.private_segment, k.dynamic_stack), (0, false), "{s}");
            assert_eq!(k.wavefront, 32);
            assert!(k.max_workgroup >= 64);
            assert_eq!(k.descriptor % 64, 0, "{s}: descriptors are 64-byte aligned");
        }
    }

    #[test]
    fn kernargs_fill_explicit_and_hidden_slots() {
        let p = Program::parse(RDNA2).unwrap();
        let k = p.kernel("nb_convert").unwrap();
        // src, dst, len, saturate, from, to, group.
        let words = [
            Word::Ptr(0x1_0000_1000),
            Word::Ptr(0x1_0000_2000),
            Word::B32(1000),
            Word::B32(1),
            Word::B32(7),
            Word::B32(9),
            Word::B32(1),
        ];
        let seg = k.kernargs(&words, 1024, 256).unwrap();
        assert_eq!(seg.len(), k.kernarg_size as usize);
        let u32_at = |o: usize| u32::from_le_bytes(seg[o..o + 4].try_into().unwrap());
        let u16_at = |o: usize| u16::from_le_bytes(seg[o..o + 2].try_into().unwrap());
        assert_eq!(&seg[0..8], &0x1_0000_1000u64.to_le_bytes());
        assert_eq!(u32_at(16), 1000);
        let at = |h: Hidden| {
            k.args
                .iter()
                .find(|a| a.kind == ArgKind::Hidden(h.clone()))
                .unwrap()
                .offset as usize
        };
        assert_eq!(u32_at(at(Hidden::BlockCount(Dim::X))), 4);
        assert_eq!(u32_at(at(Hidden::BlockCount(Dim::Y))), 1);
        assert_eq!(u16_at(at(Hidden::GroupSize(Dim::X))), 256);
        assert_eq!(u16_at(at(Hidden::GroupSize(Dim::Z))), 1);
        assert_eq!(u16_at(at(Hidden::Remainder(Dim::X))), 0);
        assert_eq!(u16_at(at(Hidden::GridDims)), 1);
        // Wrong count, wrong kind, a grid that doesn't divide.
        assert!(k.kernargs(&words[..6], 1024, 256).is_err());
        let mut swapped = words;
        swapped[0] = Word::B32(0);
        assert!(k.kernargs(&swapped, 1024, 256).is_err());
        assert!(k.kernargs(&words, 1000, 256).is_err());
    }

    #[test]
    fn other_binaries_are_refused() {
        let nv = include_bytes!("../../kernels/bin/nvidia/sm_80-emulated.cubin");
        assert!(matches!(Program::parse(nv), Err(Error::BadBinary(_))));
        assert!(Program::parse(&RDNA2[..1000]).is_err());
    }
}
