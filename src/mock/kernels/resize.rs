//! Mirror of `nb_resize` (half-pixel centres; nearest in integers, bilinear
//! with two fma lerps per axis).

use miraculix::formats::Overflow;

use super::{Mem, get, put, u};
use crate::elem::Format;
use crate::mock::codec;
use crate::runtime::backend::{Arg, ResizeMode};

pub(super) fn run(
    mem: &mut Mem,
    fs: Format,
    fd: Format,
    mode: ResizeMode,
    a: &[Arg],
) -> Option<()> {
    let (planes, h, w, ho, wo) = (u(a, 2)?, u(a, 3)?, u(a, 4)?, u(a, 5)?, u(a, 6)?);
    let x = codec::decode_f32(fs, &get(mem, a, 0)?, planes * h * w)?;
    let mut y = vec![0f32; planes * ho * wo];
    for (g, out) in y.iter_mut().enumerate() {
        let (ox, oy, pl) = (g % wo, (g / wo) % ho, g / (wo * ho));
        let px = &x[pl * h * w..][..h * w];
        *out = super::act::canon(match mode {
            ResizeMode::Nearest => {
                let sy = (((2 * oy + 1) * h) / (2 * ho)).min(h - 1);
                let sx = (((2 * ox + 1) * w) / (2 * wo)).min(w - 1);
                px[sy * w + sx]
            }
            ResizeMode::Bilinear => {
                let mut fy = (oy as u32 as f32 + 0.5) * (h as u32 as f32 / ho as u32 as f32) - 0.5;
                let mut fx = (ox as u32 as f32 + 0.5) * (w as u32 as f32 / wo as u32 as f32) - 0.5;
                if fy < 0.0 {
                    fy = 0.0;
                }
                if fx < 0.0 {
                    fx = 0.0;
                }
                let (y0, x0) = (
                    (fy as u32 as usize).min(h - 1),
                    (fx as u32 as usize).min(w - 1),
                );
                let (y1, x1) = ((y0 + 1).min(h - 1), (x0 + 1).min(w - 1));
                let (ly, lx) = (fy - y0 as u32 as f32, fx - x0 as u32 as f32);
                let (v00, v01, v10, v11) = (
                    px[y0 * w + x0],
                    px[y0 * w + x1],
                    px[y1 * w + x0],
                    px[y1 * w + x1],
                );
                let top = lx.mul_add(v01 - v00, v00);
                let bot = lx.mul_add(v11 - v10, v10);
                ly.mul_add(bot - top, top)
            }
        });
    }
    put(mem, a, 1, codec::encode_f32(fd, &y, Overflow::Special)?)
}
