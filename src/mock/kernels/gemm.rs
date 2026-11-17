use miraculix::formats::Overflow;

use super::{Mem, get, put, u};
use crate::elem::Format;
use crate::mock::codec::{self, int};
use crate::runtime::backend::Arg;

/// Emulated tier: decode, then one sequential F32 FMA chain per output
/// (wrapping `i32` for integer inputs).
pub(super) fn run(mem: &mut Mem, fa: Format, fb: Format, acc: Format, args: &[Arg]) -> Option<()> {
    let (m, n, k, batch) = (u(args, 3)?, u(args, 4)?, u(args, 5)?, u(args, 6)?);
    let (la, lb, lc) = (batch * m * k, batch * n * k, batch * m * n);
    let (ab, bb, cb) = (get(mem, args, 0)?, get(mem, args, 1)?, get(mem, args, 2)?);
    let out = if acc == Format::I32 {
        let (Arg::I32(alpha), Arg::I32(beta)) = (*args.get(7)?, *args.get(8)?) else { return None };
        let (av, bv, cv) = (int::decode(fa, &ab, la)?, int::decode(fb, &bb, lb)?, int::decode(acc, &cb, lc)?);
        let c = mm(&av, &bv, &cv, (m, n, k), 0i32, |s, x, y| s.wrapping_add(x.wrapping_mul(y)), |s, old| {
            let r = alpha.wrapping_mul(s);
            if beta != 0 { r.wrapping_add(beta.wrapping_mul(old)) } else { r }
        });
        int::encode(acc, &c)?
    } else {
        let (Arg::F32(alpha), Arg::F32(beta)) = (*args.get(7)?, *args.get(8)?) else { return None };
        let (av, bv) = (codec::decode_f32(fa, &ab, la)?, codec::decode_f32(fb, &bb, lb)?);
        // beta == 0: C is not read, so NaNs in it do not propagate.
        let cv = if beta != 0.0 { codec::decode_f32(acc, &cb, lc)? } else { vec![0.0; lc] };
        let c = mm(&av, &bv, &cv, (m, n, k), 0f32, |s, x, y| x.mul_add(y, s), |s, old| {
            if beta != 0.0 { alpha * s + beta * old } else { alpha * s }
        });
        codec::encode_f32(acc, &c, Overflow::Special)?
    };
    put(mem, args, 2, out)
}

/// Batched `A (MxK) * B^T (B is NxK)`, batches back to back.
fn mm<T: Copy>(
    a: &[T],
    b: &[T],
    c_old: &[T],
    (m, n, k): (usize, usize, usize),
    zero: T,
    fma: impl Fn(T, T, T) -> T,
    epilogue: impl Fn(T, T) -> T,
) -> Vec<T> {
    let mut c = vec![zero; c_old.len()];
    let batches = c.chunks_exact_mut(m * n).zip(a.chunks_exact(m * k)).zip(b.chunks_exact(n * k)).zip(c_old.chunks_exact(m * n));
    for (((cb, ab), bb), ob) in batches {
        for i in 0..m {
            for j in 0..n {
                let s = (0..k).fold(zero, |s, p| fma(s, ab[i * k + p], bb[j * k + p]));
                cb[i * n + j] = epilogue(s, ob[i * n + j]);
            }
        }
    }
    c
}
