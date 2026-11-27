//! Barriers on a device without `synchronization2`: the
//! `VkPipelineStageFlags2`, `VkAccessFlags2` and `VkImageLayout` values a
//! renderer writes once, lowered to what `vkCmdPipelineBarrier` takes.
//!
//! The 1.0 bits keep their values in the 64-bit masks; each bit
//! `synchronization2` added widens to the legacy bits that contain it
//! ([`STAGES`], [`ACCESSES`], from `vk.xml`'s `<sync>` equivalences), and
//! any other new bit to the widest legacy one (`ALL_COMMANDS`,
//! `MEMORY_READ | MEMORY_WRITE`), which is correct, only slower.
//!
//! `PRE_RASTERIZATION_SHADERS` becomes `ALL_GRAPHICS`: the stages it
//! stands for (vertex, tessellation, geometry, task, mesh) may only be
//! named in a 1.0 barrier when their features are enabled, and HUAWEI's
//! cluster culling stage has no legacy bit but `ALL_GRAPHICS`.
//!
//! ```
//! use numerobis::vulkan::barrier::{self, access, access2, layout, stage, stage2};
//!
//! assert_eq!(barrier::stages(stage2::COPY | stage2::BLIT, true), stage::TRANSFER);
//! assert_eq!(barrier::stages(0, true), stage::TOP_OF_PIPE);
//! assert_eq!(barrier::accesses(access2::SHADER_SAMPLED_READ), access::SHADER_READ);
//! let color = numerobis::vulkan::format::aspect::COLOR;
//! assert_eq!(barrier::layout(layout::ATTACHMENT_OPTIMAL, color), layout::COLOR_ATTACHMENT_OPTIMAL);
//! ```

use super::features::sync2;
use super::format::aspect;

/// `VkPipelineStageFlagBits`: the ones the lowering names.
pub mod stage {
    pub const TOP_OF_PIPE: u32 = 0x1;
    pub const VERTEX_INPUT: u32 = 0x4;
    pub const VERTEX_SHADER: u32 = 0x8;
    pub const TESSELLATION_CONTROL_SHADER: u32 = 0x10;
    pub const TESSELLATION_EVALUATION_SHADER: u32 = 0x20;
    pub const GEOMETRY_SHADER: u32 = 0x40;
    pub const TRANSFER: u32 = 0x1000;
    pub const BOTTOM_OF_PIPE: u32 = 0x2000;
    pub const ALL_GRAPHICS: u32 = 0x8000;
    pub const ALL_COMMANDS: u32 = 0x1_0000;
    /// Every bit 1.0 and its extensions name (bits 0 to 25), the same
    /// values in `VkPipelineStageFlags2`. From `vk.xml`.
    pub const LEGACY: u64 = super::sync2::LEGACY_STAGES;
}

/// `VkPipelineStageFlagBits2` that `synchronization2` added: the ones the
/// lowering names.
pub mod stage2 {
    pub const COPY: u64 = 0x1_0000_0000;
    pub const RESOLVE: u64 = 0x2_0000_0000;
    pub const BLIT: u64 = 0x4_0000_0000;
    pub const CLEAR: u64 = 0x8_0000_0000;
    pub const INDEX_INPUT: u64 = 0x10_0000_0000;
    pub const VERTEX_ATTRIBUTE_INPUT: u64 = 0x20_0000_0000;
    pub const PRE_RASTERIZATION_SHADERS: u64 = 0x40_0000_0000;
}

/// `VkAccessFlagBits`: the ones the lowering names.
pub mod access {
    pub const SHADER_READ: u32 = 0x20;
    pub const SHADER_WRITE: u32 = 0x40;
    pub const MEMORY_READ: u32 = 0x8000;
    pub const MEMORY_WRITE: u32 = 0x1_0000;
    /// Every bit 1.0 and its extensions name (bits 0 to 27), the same
    /// values in `VkAccessFlags2`. From `vk.xml`.
    pub const LEGACY: u64 = super::sync2::LEGACY_ACCESSES;
}

/// `VkAccessFlagBits2` that `synchronization2` added: the ones the
/// lowering names.
pub mod access2 {
    pub const SHADER_SAMPLED_READ: u64 = 0x1_0000_0000;
    pub const SHADER_STORAGE_READ: u64 = 0x2_0000_0000;
    pub const SHADER_STORAGE_WRITE: u64 = 0x4_0000_0000;
}

/// `VkImageLayout`s the lowering names.
pub mod layout {
    pub const COLOR_ATTACHMENT_OPTIMAL: u32 = 2;
    pub const DEPTH_STENCIL_ATTACHMENT_OPTIMAL: u32 = 3;
    pub const DEPTH_STENCIL_READ_ONLY_OPTIMAL: u32 = 4;
    pub const SHADER_READ_ONLY_OPTIMAL: u32 = 5;
    /// `synchronization2`'s, by aspect.
    pub const READ_ONLY_OPTIMAL: u32 = 1_000_314_000;
    /// `synchronization2`'s, by aspect.
    pub const ATTACHMENT_OPTIMAL: u32 = 1_000_314_001;
}

/// The new stages and the legacy ones that contain them, generated from
/// `vk.xml`. No two rows share a new bit.
pub const STAGES: &[(u64, u32)] = sync2::STAGES;

/// The new accesses and the legacy ones that contain them, generated from
/// `vk.xml`. No two rows share a new bit.
pub const ACCESSES: &[(u64, u32)] = sync2::ACCESSES;

/// `bits` without the new ones `table` names, and the legacy bits they
/// widen to.
const fn widen(bits: u64, table: &[(u64, u32)]) -> (u64, u32) {
    let mut rest = bits;
    let mut out = 0;
    let mut i = 0;
    while i < table.len() {
        let (from, to) = table[i];
        if rest & from != 0 {
            out |= to;
            rest &= !from;
        }
        i += 1;
    }
    (rest, out)
}

/// `stages` (`VkPipelineStageFlags2`) as `VkPipelineStageFlags`. `src`:
/// the first scope, where an empty mask means `TOP_OF_PIPE` (the second's
/// means `BOTTOM_OF_PIPE`), as 1.0 doesn't allow empty masks.
#[must_use]
pub const fn stages(stages: u64, src: bool) -> u32 {
    let shared = (stages & stage::LEGACY) as u32;
    let (rest, wide) = widen(stages & !stage::LEGACY, STAGES);
    let mut out = shared | wide;
    if rest != 0 {
        out |= stage::ALL_COMMANDS;
    }
    if out == 0 {
        out = if src {
            stage::TOP_OF_PIPE
        } else {
            stage::BOTTOM_OF_PIPE
        };
    }
    out
}

/// `accesses` (`VkAccessFlags2`) as `VkAccessFlags`.
#[must_use]
pub const fn accesses(accesses: u64) -> u32 {
    let shared = (accesses & access::LEGACY) as u32;
    let (rest, wide) = widen(accesses & !access::LEGACY, ACCESSES);
    let mut out = shared | wide;
    if rest != 0 {
        out |= access::MEMORY_READ | access::MEMORY_WRITE;
    }
    out
}

/// `layout` without the layouts `synchronization2` added, for an image of
/// `aspects` (`VkImageAspectFlags`): `ATTACHMENT_OPTIMAL` and
/// `READ_ONLY_OPTIMAL` become the colour or depth-stencil ones.
#[must_use]
pub const fn layout(layout: u32, aspects: u32) -> u32 {
    let depth = aspects & (aspect::DEPTH | aspect::STENCIL) != 0;
    match (layout, depth) {
        (layout::ATTACHMENT_OPTIMAL, false) => layout::COLOR_ATTACHMENT_OPTIMAL,
        (layout::ATTACHMENT_OPTIMAL, true) => layout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
        (layout::READ_ONLY_OPTIMAL, false) => layout::SHADER_READ_ONLY_OPTIMAL,
        (layout::READ_ONLY_OPTIMAL, true) => layout::DEPTH_STENCIL_READ_ONLY_OPTIMAL,
        _ => layout,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn shared_bits_pass_through() {
        // FRAGMENT_SHADER, COLOR_ATTACHMENT_OUTPUT, TRANSFER, MESH_SHADER_EXT.
        let s = 0x80 | 0x400 | stage::TRANSFER | 0x10_0000;
        assert_eq!(stages(u64::from(s), true), s);
        // COLOR_ATTACHMENT_WRITE, TRANSFER_WRITE.
        let a = access::SHADER_READ | 0x100 | 0x1000;
        assert_eq!(accesses(u64::from(a)), a);
    }

    #[test]
    fn new_stages_widen() {
        assert_eq!(stages(stage2::COPY | stage2::BLIT, true), stage::TRANSFER);
        assert_eq!(stages(stage2::INDEX_INPUT, false), stage::VERTEX_INPUT);
        // No feature-gated stage (geometry, tessellation, mesh) in a 1.0 mask.
        assert_eq!(
            stages(stage2::PRE_RASTERIZATION_SHADERS, true),
            stage::ALL_GRAPHICS
        );
        // ACCELERATION_STRUCTURE_COPY_KHR, below bit 32 but not legacy.
        assert_eq!(stages(0x1000_0000, true), stage::TRANSFER);
        // VIDEO_DECODE_KHR: no legacy bit holds it.
        assert_eq!(stages(0x400_0000, true), stage::ALL_COMMANDS);
        assert_eq!(
            stages(stage2::CLEAR | 0x400_0000, true),
            stage::TRANSFER | stage::ALL_COMMANDS
        );
    }

    #[test]
    fn tables_are_new_bits_to_legacy_ones() {
        for (table, legacy) in [(STAGES, stage::LEGACY), (ACCESSES, access::LEGACY)] {
            let mut seen = 0;
            for &(from, to) in table {
                assert_eq!(from & legacy, 0, "{from:#x} holds legacy bits");
                assert_eq!(from & seen, 0, "{from:#x} is in two rows");
                assert_ne!(to, 0);
                assert_eq!(u64::from(to) & !legacy, 0, "{to:#x} is not legacy");
                seen |= from;
            }
        }
        // The bits the named constants say are new.
        let named = stage2::COPY
            | stage2::RESOLVE
            | stage2::BLIT
            | stage2::CLEAR
            | stage2::INDEX_INPUT
            | stage2::VERTEX_ATTRIBUTE_INPUT
            | stage2::PRE_RASTERIZATION_SHADERS;
        let all = STAGES.iter().fold(0, |m, &(from, _)| m | from);
        assert_eq!(all & named, named);
    }

    #[test]
    fn empty_stages() {
        assert_eq!(stages(0, true), stage::TOP_OF_PIPE);
        assert_eq!(stages(0, false), stage::BOTTOM_OF_PIPE);
    }

    #[test]
    fn new_accesses_widen() {
        assert_eq!(
            accesses(access2::SHADER_SAMPLED_READ | access2::SHADER_STORAGE_READ),
            access::SHADER_READ
        );
        assert_eq!(
            accesses(access2::SHADER_STORAGE_WRITE),
            access::SHADER_WRITE
        );
        // SHADER_BINDING_TABLE_READ_KHR.
        assert_eq!(accesses(0x100_0000_0000), access::SHADER_READ);
        // DESCRIPTOR_BUFFER_READ_EXT.
        assert_eq!(
            accesses(0x200_0000_0000),
            access::MEMORY_READ | access::MEMORY_WRITE
        );
        assert_eq!(accesses(0), 0);
    }

    #[test]
    fn layouts_by_aspect() {
        use layout as L;
        assert_eq!(
            layout(L::ATTACHMENT_OPTIMAL, aspect::COLOR),
            L::COLOR_ATTACHMENT_OPTIMAL
        );
        assert_eq!(
            layout(L::ATTACHMENT_OPTIMAL, aspect::DEPTH),
            L::DEPTH_STENCIL_ATTACHMENT_OPTIMAL
        );
        assert_eq!(
            layout(L::READ_ONLY_OPTIMAL, aspect::COLOR),
            L::SHADER_READ_ONLY_OPTIMAL
        );
        assert_eq!(
            layout(L::READ_ONLY_OPTIMAL, aspect::DEPTH | aspect::STENCIL),
            L::DEPTH_STENCIL_READ_ONLY_OPTIMAL
        );
        // GENERAL.
        assert_eq!(layout(1, aspect::COLOR), 1);
    }
}
