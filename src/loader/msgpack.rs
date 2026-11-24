//! Minimal `MessagePack` reader for the AMDGPU code object metadata
//! (`NT_AMDGPU_METADATA` notes, code object v3 and later). Decodes the
//! whole document into a [`Value`] tree; extension types and timestamps,
//! which the metadata never uses, are refused.

use crate::runtime::error::{Error, Result};

/// One decoded `MessagePack` value.
#[derive(Debug, Clone, PartialEq)]
pub enum Value<'a> {
    Nil,
    Bool(bool),
    /// Every integer, signed or not, that fits an `i128` (all of them).
    Int(i128),
    Float(f64),
    Str(&'a str),
    Bin(&'a [u8]),
    Array(Vec<Value<'a>>),
    Map(Vec<(Value<'a>, Value<'a>)>),
}

impl<'a> Value<'a> {
    /// The value under string key `key` of a map.
    #[must_use]
    pub fn get(&self, key: &str) -> Option<&Value<'a>> {
        match self {
            Value::Map(m) => m
                .iter()
                .find(|(k, _)| matches!(k, Value::Str(s) if *s == key))
                .map(|(_, v)| v),
            _ => None,
        }
    }

    #[must_use]
    pub fn as_str(&self) -> Option<&'a str> {
        match *self {
            Value::Str(s) => Some(s),
            _ => None,
        }
    }

    #[must_use]
    pub fn as_u64(&self) -> Option<u64> {
        match *self {
            Value::Int(i) => u64::try_from(i).ok(),
            _ => None,
        }
    }

    #[must_use]
    pub fn as_array(&self) -> Option<&[Value<'a>]> {
        match self {
            Value::Array(a) => Some(a),
            _ => None,
        }
    }
}

fn bad(what: &str) -> Error {
    Error::BadBinary(format!("msgpack: {what}"))
}

/// Nesting deeper than this is refused (the metadata nests four levels).
const MAX_DEPTH: usize = 32;

struct Reader<'a> {
    b: &'a [u8],
    at: usize,
}

impl<'a> Reader<'a> {
    fn take(&mut self, n: usize) -> Result<&'a [u8]> {
        let end = self.at.checked_add(n).ok_or_else(|| bad("length"))?;
        let s = self.b.get(self.at..end).ok_or_else(|| bad("truncated"))?;
        self.at = end;
        Ok(s)
    }

    fn be(&mut self, n: usize) -> Result<u64> {
        Ok(self
            .take(n)?
            .iter()
            .fold(0u64, |v, &b| (v << 8) | u64::from(b)))
    }

    fn len(&mut self, n: usize) -> Result<usize> {
        usize::try_from(self.be(n)?).map_err(|_| bad("length"))
    }

    fn str(&mut self, n: usize) -> Result<Value<'a>> {
        let s = self.take(n)?;
        core::str::from_utf8(s)
            .map(Value::Str)
            .map_err(|_| bad("non-UTF-8 string"))
    }

    fn array(&mut self, n: usize, depth: usize) -> Result<Value<'a>> {
        // Every element takes at least one byte: no huge pre-allocation.
        let mut v = Vec::with_capacity(n.min(self.b.len() - self.at));
        for _ in 0..n {
            v.push(self.value(depth + 1)?);
        }
        Ok(Value::Array(v))
    }

    fn map(&mut self, n: usize, depth: usize) -> Result<Value<'a>> {
        let mut v = Vec::with_capacity(n.min(self.b.len() - self.at));
        for _ in 0..n {
            let k = self.value(depth + 1)?;
            v.push((k, self.value(depth + 1)?));
        }
        Ok(Value::Map(v))
    }

    #[allow(
        clippy::cast_possible_wrap,
        clippy::cast_possible_truncation,
        reason = "`be(n)` holds n bytes; the signed forms reinterpret those bits"
    )]
    fn value(&mut self, depth: usize) -> Result<Value<'a>> {
        if depth > MAX_DEPTH {
            return Err(bad("nested too deep"));
        }
        let t = self.take(1)?[0];
        Ok(match t {
            0x00..=0x7f => Value::Int(i128::from(t)),
            0x80..=0x8f => return self.map(usize::from(t & 0x0f), depth),
            0x90..=0x9f => return self.array(usize::from(t & 0x0f), depth),
            0xa0..=0xbf => return self.str(usize::from(t & 0x1f)),
            0xc0 => Value::Nil,
            0xc2 => Value::Bool(false),
            0xc3 => Value::Bool(true),
            0xc4..=0xc6 => {
                let n = self.len(1 << (t - 0xc4))?;
                Value::Bin(self.take(n)?)
            }
            0xca => Value::Float(f64::from(f32::from_bits(self.be(4)? as u32))),
            0xcb => Value::Float(f64::from_bits(self.be(8)?)),
            0xcc..=0xcf => Value::Int(i128::from(self.be(1 << (t - 0xcc))?)),
            0xd0 => Value::Int(i128::from(self.be(1)? as u8 as i8)),
            0xd1 => Value::Int(i128::from(self.be(2)? as u16 as i16)),
            0xd2 => Value::Int(i128::from(self.be(4)? as u32 as i32)),
            0xd3 => Value::Int(i128::from(self.be(8)? as i64)),
            0xd9..=0xdb => {
                let n = self.len(1 << (t - 0xd9))?;
                return self.str(n);
            }
            0xdc | 0xdd => {
                let n = self.len(if t == 0xdc { 2 } else { 4 })?;
                return self.array(n, depth);
            }
            0xde | 0xdf => {
                let n = self.len(if t == 0xde { 2 } else { 4 })?;
                return self.map(n, depth);
            }
            0xe0..=0xff => Value::Int(i128::from(t as i8)),
            // 0xc1 (never used), extension types and fixext.
            _ => return Err(bad(&format!("unsupported type {t:#04x}"))),
        })
    }
}

/// Decodes one `MessagePack` document; trailing bytes are an error.
///
/// # Errors
///
/// [`Error::BadBinary`] for truncated or malformed input, extension types,
/// and nesting deeper than 32 levels.
pub fn decode(b: &[u8]) -> Result<Value<'_>> {
    let mut r = Reader { b, at: 0 };
    let v = r.value(0)?;
    if r.at != b.len() {
        return Err(bad("trailing bytes"));
    }
    Ok(v)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn decodes_the_forms_the_metadata_uses() {
        // {"a": [1, -1, 300, "xy"], "b": true, "c": nil}
        let doc = [
            0x83, 0xa1, b'a', 0x94, 0x01, 0xff, 0xcd, 0x01, 0x2c, 0xa2, b'x', b'y', 0xa1, b'b',
            0xc3, 0xa1, b'c', 0xc0,
        ];
        let v = decode(&doc).unwrap();
        let a = v.get("a").and_then(Value::as_array).unwrap();
        assert_eq!(a[0].as_u64(), Some(1));
        assert_eq!(a[1], Value::Int(-1));
        assert_eq!(a[2].as_u64(), Some(300));
        assert_eq!(a[3].as_str(), Some("xy"));
        assert_eq!(v.get("b"), Some(&Value::Bool(true)));
        assert_eq!(v.get("c"), Some(&Value::Nil));
        assert_eq!(v.get("d"), None);
    }

    #[test]
    fn malformed_input_is_an_error() {
        assert!(decode(&[0x92, 0x01]).is_err()); // array of 2, one element
        assert!(decode(&[0xa3, b'a']).is_err()); // string past the end
        assert!(decode(&[0x01, 0x02]).is_err()); // trailing bytes
        assert!(decode(&[0xd4, 0x00, 0x00]).is_err()); // fixext
        assert!(decode(&[0x91; 64]).is_err()); // nested too deep
        // A length that claims more than there is doesn't allocate it.
        assert!(decode(&[0xdd, 0xff, 0xff, 0xff, 0xff]).is_err());
    }
}
