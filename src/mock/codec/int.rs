//! I32, I8, U8 and packed I4/U4/I2/I1.

use miraculix::formats::int::{pack_int, pack_uint, unpack_int, unpack_uint};

use crate::elem::{Format, as_bytes, from_bytes};

/// Device bytes -> `i32`.
pub fn decode(fmt: Format, b: &[u8], len: usize) -> Option<Vec<i32>> {
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

/// `i32` -> device bytes, saturating.
pub fn encode(fmt: Format, src: &[i32]) -> Option<Vec<u8>> {
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
