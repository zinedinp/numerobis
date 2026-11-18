//! Queue Meta Data (compute launch descriptor), version 03_00 (Ampere/Ada,
//! `AMPERE_COMPUTE_B` / `ADA_COMPUTE_A`). Field positions from NVIDIA
//! `open-gpu-doc` `classes/compute/clc6c0qmd.h` (`NVC6C0_QMDV03_00_*`, `MW(hi:lo)`).

/// 256-byte QMD.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Qmd(pub [u32; 64]);

macro_rules! fields {
    ($($name:ident = $hi:expr, $lo:expr;)*) => {
        $(pub const $name: (u32, u32) = ($hi, $lo);)*
    };
}

fields! {
    QMD_GROUP_ID = 133, 128;
    SM_GLOBAL_CACHING_ENABLE = 134, 134;
    INVALIDATE_TEXTURE_HEADER_CACHE = 186, 186;
    INVALIDATE_TEXTURE_SAMPLER_CACHE = 187, 187;
    INVALIDATE_TEXTURE_DATA_CACHE = 188, 188;
    INVALIDATE_SHADER_DATA_CACHE = 189, 189;
    INVALIDATE_INSTRUCTION_CACHE = 190, 190;
    INVALIDATE_SHADER_CONSTANT_CACHE = 191, 191;
    PROGRAM_PREFETCH_ADDR_LOWER_SHIFTED = 287, 256;
    CWD_MEMBAR_TYPE = 369, 368;
    API_VISIBLE_CALL_LIMIT = 378, 378;
    SAMPLER_INDEX = 382, 382;
    CTA_RASTER_WIDTH = 415, 384;
    CTA_RASTER_HEIGHT = 431, 416;
    CTA_RASTER_DEPTH = 463, 448;
    SHARED_MEMORY_SIZE = 561, 544;
    MIN_SM_CONFIG_SHARED_MEM_SIZE = 567, 562;
    MAX_SM_CONFIG_SHARED_MEM_SIZE = 574, 569;
    QMD_VERSION = 579, 576;
    QMD_MAJOR_VERSION = 583, 580;
    CTA_THREAD_DIMENSION0 = 607, 592;
    CTA_THREAD_DIMENSION1 = 623, 608;
    CTA_THREAD_DIMENSION2 = 639, 624;
    REGISTER_COUNT_V = 656, 648;
    TARGET_SM_CONFIG_SHARED_MEM_SIZE = 662, 657;
    SHADER_LOCAL_MEMORY_LOW_SIZE = 759, 736;
    BARRIER_COUNT = 767, 763;
    PROGRAM_ADDRESS_LOWER = 1567, 1536;
    PROGRAM_ADDRESS_UPPER = 1584, 1568;
    SHADER_LOCAL_MEMORY_HIGH_SIZE = 1623, 1600;
    PROGRAM_PREFETCH_ADDR_UPPER_SHIFTED = 1640, 1632;
    PROGRAM_PREFETCH_SIZE = 1649, 1641;
    SASS_VERSION = 1663, 1656;
}

/// `CONSTANT_BUFFER_VALID(i)`.
pub const fn cb_valid(i: u32) -> (u32, u32) {
    (640 + i, 640 + i)
}
/// `CONSTANT_BUFFER_ADDR_LOWER(i)`.
pub const fn cb_addr_lower(i: u32) -> (u32, u32) {
    (1055 + i * 64, 1024 + i * 64)
}
/// `CONSTANT_BUFFER_ADDR_UPPER(i)`.
pub const fn cb_addr_upper(i: u32) -> (u32, u32) {
    (1072 + i * 64, 1056 + i * 64)
}
/// `CONSTANT_BUFFER_SIZE_SHIFTED4(i)`.
pub const fn cb_size_shifted4(i: u32) -> (u32, u32) {
    (1087 + i * 64, 1075 + i * 64)
}

/// `CWD_MEMBAR_TYPE_L1_SYSMEMBAR`.
pub const CWD_MEMBAR_L1_SYSMEMBAR: u32 = 1;

impl Default for Qmd {
    fn default() -> Self {
        Qmd([0; 64])
    }
}

impl Qmd {
    /// Sets bits `hi..=lo` (may span two words) to `v`.
    pub fn set(&mut self, (hi, lo): (u32, u32), v: u64) {
        let width = hi - lo + 1;
        assert!(
            width <= 64 && (width == 64 || v < (1u64 << width)),
            "QMD field {hi}:{lo} overflow ({v:#x})"
        );
        for b in 0..width {
            let bit = lo + b;
            let (w, s) = ((bit / 32) as usize, bit % 32);
            self.0[w] = (self.0[w] & !(1 << s)) | ((((v >> b) & 1) as u32) << s);
        }
    }

    pub fn get(&self, (hi, lo): (u32, u32)) -> u64 {
        (0..hi - lo + 1)
            .map(|b| (((self.0[((lo + b) / 32) as usize] >> ((lo + b) % 32)) & 1) as u64) << b)
            .sum()
    }

    /// Binds constant buffer `i` at device address `addr` with `size` bytes.
    pub fn constant_buffer(&mut self, i: u32, addr: u64, size: u32) {
        self.set(cb_addr_lower(i), addr & 0xffff_ffff);
        self.set(cb_addr_upper(i), addr >> 32);
        self.set(cb_size_shifted4(i), size.div_ceil(16) as u64);
        self.set(cb_valid(i), 1);
    }

    pub fn to_bytes(&self) -> Vec<u8> {
        self.0.iter().flat_map(|w| w.to_le_bytes()).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fields_round_trip_across_words() {
        let mut q = Qmd::default();
        q.set(PROGRAM_ADDRESS_UPPER, 0x1_2345);
        q.set(cb_addr_upper(2), 0x1_ffff);
        q.set(QMD_MAJOR_VERSION, 3);
        assert_eq!(q.get(PROGRAM_ADDRESS_UPPER), 0x1_2345);
        assert_eq!(q.get(cb_addr_upper(2)), 0x1_ffff);
        assert_eq!(q.get(QMD_MAJOR_VERSION), 3);
        // QMD_MAJOR_VERSION is bits 583:580 -> word 18, bits 4..8.
        assert_eq!(q.0[18], 3 << 4);
    }

    #[test]
    #[should_panic]
    fn overflow_is_caught() {
        Qmd::default().set(QMD_VERSION, 16);
    }
}
