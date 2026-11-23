//! Device-layout codecs for the mock backend, built only from miraculix and
//! split by format kind like [`crate::elem`]. These are the oracle every real
//! tier is compared against.

pub mod block;
pub mod float;
pub mod int;

use miraculix::formats::Overflow;

use crate::elem::Format;

/// Device bytes -> `f32`, for any non-integer format (and integers, exactly).
#[must_use]
pub fn decode_f32(fmt: Format, b: &[u8], len: usize) -> Option<Vec<f32>> {
    if fmt.is_integer() {
        return Some(
            int::decode(fmt, b, len)?
                .into_iter()
                .map(|v| v as f32)
                .collect(),
        );
    }
    if fmt.is_block_scaled() {
        block::decode(fmt, b, len)
    } else {
        float::decode(fmt, b, len)
    }
}

/// `f32` -> device bytes for float and quantizable block formats.
#[must_use]
pub fn encode_f32(fmt: Format, src: &[f32], overflow: Overflow) -> Option<Vec<u8>> {
    if fmt.is_block_scaled() {
        block::encode(fmt, src)
    } else {
        float::encode(fmt, src, overflow)
    }
}
