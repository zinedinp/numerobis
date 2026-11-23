//! MX, NVFP4, `BFP_B` and GGUF in the device layout of [`crate::elem`]
//! (element data, then scales; NVFP4 then the tensor scale).

use miraculix::formats::bfp::{bfp_decode_rows, bfp_encode_rows};
use miraculix::formats::gguf::{dequantize_q6_k, dequantize_q8_0};
use miraculix::formats::mx::{
    mx_dequantize, mx_quantize, nvfp4_dequantize, nvfp4_quantize, nvfp4_tensor_scale,
};

use crate::elem::Format;

#[must_use]
pub fn decode(fmt: Format, b: &[u8], len: usize) -> Option<Vec<f32>> {
    let mut out = vec![0.0f32; len];
    let off = fmt.scale_offset(len);
    match fmt {
        Format::Mx(e) => mx_dequantize(e, &b[off..], &b[..off], &mut out),
        Format::Nvfp4 => {
            let blocks = len / fmt.block();
            let ts = f32::from_le_bytes(b.get(off + blocks..off + blocks + 4)?.try_into().ok()?);
            nvfp4_dequantize(&b[off..off + blocks], ts, &b[..off], &mut out);
        }
        Format::Bfp(e) => bfp_decode_rows(e, &b[off..], &b[..off], &mut out),
        Format::Q8_0 => dequantize_q8_0(b, &mut out),
        Format::Q6K => dequantize_q6_k(b, &mut out),
        _ => return None,
    }
    Some(out)
}

/// `None` for GGUF (decode only).
#[must_use]
pub fn encode(fmt: Format, src: &[f32]) -> Option<Vec<u8>> {
    let len = src.len();
    let mut out = vec![0u8; fmt.storage_bytes(len)?];
    let off = fmt.scale_offset(len);
    match fmt {
        Format::Mx(e) => {
            let (data, scales) = out.split_at_mut(off);
            mx_quantize(e, src, scales, data);
        }
        Format::Nvfp4 => {
            let blocks = len / fmt.block();
            let ts = nvfp4_tensor_scale(src);
            let (data, rest) = out.split_at_mut(off);
            let (scales, tail) = rest.split_at_mut(blocks);
            nvfp4_quantize(src, ts, scales, data);
            tail.copy_from_slice(&ts.to_le_bytes());
        }
        Format::Bfp(e) => {
            let (data, exps) = out.split_at_mut(off);
            bfp_encode_rows(e, src, exps, data);
        }
        _ => return None,
    }
    Some(out)
}
