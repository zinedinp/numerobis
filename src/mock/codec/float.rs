//! F32, TF32 and [`MiniFloat`] codes (byte-aligned or packed LSB-first).

use miraculix::formats::{Overflow, round_to_tf32};

use crate::elem::{Format, as_bytes, from_bytes};

pub(super) fn get_bits(d: &[u8], i: usize, w: usize) -> u32 {
    let bit = i * w;
    let (byte, off) = (bit / 8, bit % 8);
    let mut v = (d[byte] as u32) >> off;
    if off + w > 8 {
        v |= (d[byte + 1] as u32) << (8 - off);
    }
    v & ((1 << w) - 1)
}

pub(super) fn put_bits(d: &mut [u8], i: usize, w: usize, v: u32) {
    let bit = i * w;
    let (byte, off) = (bit / 8, bit % 8);
    d[byte] |= (v << off) as u8;
    if off + w > 8 {
        d[byte + 1] |= (v >> (8 - off)) as u8;
    }
}

pub fn decode(fmt: Format, b: &[u8], len: usize) -> Option<Vec<f32>> {
    let mut out = vec![0.0f32; len];
    match fmt {
        Format::F32 | Format::Tf32 => return Some(from_bytes(b)),
        Format::Float(f) if f.bits() == 16 => f.decode_u16_slice(&from_bytes::<u16>(b), &mut out),
        Format::Float(f) if f.bits() == 8 => f.decode_u8_slice(b, &mut out),
        Format::Float(f) => {
            let w = f.bits() as usize;
            out.iter_mut().enumerate().for_each(|(i, o)| *o = f.decode(get_bits(b, i, w)));
        }
        _ => return None,
    }
    Some(out)
}

pub fn encode(fmt: Format, src: &[f32], overflow: Overflow) -> Option<Vec<u8>> {
    let len = src.len();
    let mut out = vec![0u8; fmt.storage_bytes(len)?];
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
        _ => return None,
    }
    Some(out)
}
