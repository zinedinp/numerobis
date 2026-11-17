//! Device-layout codecs for the mock backend, built only from miraculix.
//! These are the oracle every real tier is compared against.

use miraculix::formats::bfp::{bfp_decode_rows, bfp_encode_rows};
use miraculix::formats::gguf::{dequantize_q6_k, dequantize_q8_0};
use miraculix::formats::int::{pack_int, pack_uint, unpack_int, unpack_uint};
use miraculix::formats::mx::{mx_dequantize, mx_quantize, nvfp4_dequantize, nvfp4_quantize, nvfp4_tensor_scale};
use miraculix::formats::{Overflow, round_to_tf32};

use crate::elem::{Format, as_bytes, from_bytes};

fn get_bits(d: &[u8], i: usize, w: usize) -> u32 {
    let bit = i * w;
    let (byte, off) = (bit / 8, bit % 8);
    let mut v = (d[byte] as u32) >> off;
    if off + w > 8 {
        v |= (d[byte + 1] as u32) << (8 - off);
    }
    v & ((1 << w) - 1)
}

fn put_bits(d: &mut [u8], i: usize, w: usize, v: u32) {
    let bit = i * w;
    let (byte, off) = (bit / 8, bit % 8);
    d[byte] |= (v << off) as u8;
    if off + w > 8 {
        d[byte + 1] |= (v >> (8 - off)) as u8;
    }
}

/// Device bytes → `f32`. `None` if the format has no float view.
pub fn decode_f32(fmt: Format, b: &[u8], len: usize) -> Option<Vec<f32>> {
    let mut out = vec![0.0f32; len];
    let off = fmt.scale_offset(len);
    match fmt {
        Format::F32 | Format::Tf32 => return Some(from_bytes(b)),
        Format::I32 | Format::I8 | Format::U8 | Format::Int { .. } => {
            return Some(decode_i32(fmt, b, len)?.into_iter().map(|v| v as f32).collect());
        }
        Format::Float(f) if f.bits() == 16 => f.decode_u16_slice(&from_bytes::<u16>(b), &mut out),
        Format::Float(f) if f.bits() == 8 => f.decode_u8_slice(b, &mut out),
        Format::Float(f) => {
            let w = f.bits() as usize;
            out.iter_mut().enumerate().for_each(|(i, o)| *o = f.decode(get_bits(b, i, w)));
        }
        Format::Mx(e) => mx_dequantize(e, &b[off..], &b[..off], &mut out),
        Format::Nvfp4 => {
            let blocks = len / fmt.block();
            let ts = f32::from_le_bytes(b[off + blocks..off + blocks + 4].try_into().ok()?);
            nvfp4_dequantize(&b[off..off + blocks], ts, &b[..off], &mut out);
        }
        Format::Bfp(e) => bfp_decode_rows(e, &b[off..], &b[..off], &mut out),
        Format::Q8_0 => dequantize_q8_0(b, &mut out),
        Format::Q6K => dequantize_q6_k(b, &mut out),
    }
    Some(out)
}

/// `f32` → device bytes. `None` for formats without an encoder (integers, GGUF).
pub fn encode_f32(fmt: Format, src: &[f32], overflow: Overflow) -> Option<Vec<u8>> {
    let len = src.len();
    let mut out = vec![0u8; fmt.storage_bytes(len)?];
    let off = fmt.scale_offset(len);
    match fmt {
        Format::F32 => out.copy_from_slice(as_bytes(src)),
        Format::Tf32 => {
            let r: Vec<f32> = src.iter().map(|&x| round_to_tf32(x)).collect();
            out.copy_from_slice(as_bytes(&r));
        }
        Format::Float(f) if f.bits() == 16 => {
            let mut h = vec![0u16; len];
            f.encode_u16_slice(src, &mut h, overflow);
            out.copy_from_slice(as_bytes(&h));
        }
        Format::Float(f) if f.bits() == 8 => f.encode_u8_slice(src, &mut out, overflow),
        Format::Float(f) => {
            let w = f.bits() as usize;
            src.iter().enumerate().for_each(|(i, &x)| put_bits(&mut out, i, w, f.encode(x, overflow)));
        }
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

/// Device bytes → `i32` for integer formats.
pub fn decode_i32(fmt: Format, b: &[u8], len: usize) -> Option<Vec<i32>> {
    Some(match fmt {
        Format::I32 => from_bytes(b),
        Format::I8 => b.iter().map(|&v| v as i8 as i32).collect(),
        Format::U8 => b.iter().map(|&v| v as i32).collect(),
        Format::Int { bits, signed: true } => {
            let mut v = vec![0i8; len];
            match bits {
                4 => unpack_int::<4>(b, &mut v),
                2 => unpack_int::<2>(b, &mut v),
                _ => unpack_int::<1>(b, &mut v),
            }
            v.into_iter().map(i32::from).collect()
        }
        Format::Int { bits, signed: false } => {
            let mut v = vec![0u8; len];
            match bits {
                4 => unpack_uint::<4>(b, &mut v),
                2 => unpack_uint::<2>(b, &mut v),
                _ => unpack_uint::<1>(b, &mut v),
            }
            v.into_iter().map(i32::from).collect()
        }
        _ => return None,
    })
}

/// `i32` → integer device bytes, saturating.
pub fn encode_i32(fmt: Format, src: &[i32]) -> Option<Vec<u8>> {
    let mut out = vec![0u8; fmt.storage_bytes(src.len())?];
    let s8: Vec<i8> = src.iter().map(|&v| v.clamp(-128, 127) as i8).collect();
    let u8s: Vec<u8> = src.iter().map(|&v| v.clamp(0, 255) as u8).collect();
    match fmt {
        Format::I32 => out.copy_from_slice(as_bytes(src)),
        Format::I8 => out.copy_from_slice(as_bytes(&s8)),
        Format::U8 => out.copy_from_slice(&u8s),
        Format::Int { bits: 4, signed: true } => pack_int::<4>(&s8, &mut out),
        Format::Int { bits: 2, signed: true } => pack_int::<2>(&s8, &mut out),
        Format::Int { bits: 1, signed: true } => pack_int::<1>(&s8, &mut out),
        Format::Int { bits: 4, signed: false } => pack_uint::<4>(&u8s, &mut out),
        Format::Int { bits: 2, signed: false } => pack_uint::<2>(&u8s, &mut out),
        Format::Int { bits: 1, signed: false } => pack_uint::<1>(&u8s, &mut out),
        _ => return None,
    }
    Some(out)
}

/// Is `from → to` a stage-1 convert?
pub fn convert_supported(from: Format, to: Format) -> bool {
    let int = |f: Format| f.is_integer();
    let gguf = |f: Format| matches!(f, Format::Q8_0 | Format::Q6K);
    match (from, to) {
        (Format::F32, t) => !int(t) && !gguf(t),
        (f, Format::F32) => !matches!(f, Format::F32),
        (f, t) => int(f) && int(t) && f != t,
    }
}

/// Runs a supported convert on device bytes.
pub fn convert(from: Format, to: Format, src: &[u8], len: usize, overflow: Overflow) -> Option<Vec<u8>> {
    if from.is_integer() && to.is_integer() {
        return encode_i32(to, &decode_i32(from, src, len)?);
    }
    encode_f32(to, &decode_f32(from, src, len)?, overflow)
}
