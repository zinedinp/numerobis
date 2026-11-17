use miraculix::formats::Overflow;

use super::{Choice, key, len_u32};
use crate::backend::{Arg, OpSig};
use crate::buffer::DeviceBuffer;
use crate::elem::Elem;
use crate::error::{Result, invalid};
use crate::stream::{Pending, Stream};

/// `convert_<from>_<to>`, `quantize_<from>_<to>` or `dequantize_<from>_<to>`,
/// element for element, bit-exact with the miraculix codec of the same pair.
///
/// `overflow` applies to plain minifloat targets; MX, NVFP4 and BFP quantizers
/// always saturate (as in miraculix). Supported pairs (stage 1):
/// - `F32 ->` every minifloat, `Tf32`, and every block format except GGUF;
/// - every minifloat, block and integer format `→ F32`;
/// - `I8 <-> I4/I2`, `U8 <-> U4` (pack / unpack, saturating).
pub fn convert<F: Elem, T: Elem>(
    s: &mut Stream,
    choice: Choice,
    src: &DeviceBuffer<F>,
    dst: &mut DeviceBuffer<T>,
    overflow: Overflow,
) -> Result<Pending<()>> {
    if src.len() != dst.len() {
        return Err(invalid(format!("convert: {} elements into {}", src.len(), dst.len())));
    }
    let k = key(s, choice, OpSig::Convert { from: F::FORMAT, to: T::FORMAT }, false)?;
    let ov = match overflow {
        Overflow::Saturate => 0,
        _ => 1,
    };
    let args = [Arg::Buf(src.raw.id), Arg::Buf(dst.raw.id), Arg::U32(len_u32(src.len())?), Arg::U32(ov)];
    s.launch(&k, &[&src.raw, &dst.raw], &args)
}
