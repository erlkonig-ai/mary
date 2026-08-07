//! PIL-compatible image preprocessing for Gemma 4 vision.
//!
//! Mirrors CPython's `PIL.Image.resize(size, Resampling.BICUBIC)` byte-for-byte
//! on 8-bit RGB images. PIL's u8 path uses the Keys cubic kernel (a = -0.5),
//! a coordinate mapping of `center = (dst + 0.5) * src/dst`, antialiased filter
//! support (`2 * max(1, scale)`), and integer fixed-point arithmetic with
//! 22-bit precision (`libImaging/Resample.c`). Rust's `image` crate
//! `FilterType::CatmullRom` gets the kernel right but not the coordinate
//! mapping, antialiasing, or rounding.

use image::{Rgb, RgbImage};

const PRECISION_BITS: u32 = 32 - 8 - 2; // 22, matches PIL
const ROUND_OFFSET: i32 = 1 << (PRECISION_BITS - 1);

/// Keys cubic kernel with a = -0.5 (PIL's bicubic filter).
#[inline]
fn bicubic_kernel(x: f64) -> f64 {
    const A: f64 = -0.5;
    let x = x.abs();
    if x < 1.0 {
        ((A + 2.0) * x - (A + 3.0)) * x * x + 1.0
    } else if x < 2.0 {
        (((x - 5.0) * x + 8.0) * x - 4.0) * A
    } else {
        0.0
    }
}

/// Precompute PIL's per-output-pixel sampling bounds + fixed-point weights.
fn precompute_coeffs(src_size: usize, dst_size: usize) -> (Vec<(usize, usize)>, usize, Vec<i32>) {
    let scale = src_size as f64 / dst_size as f64;
    let filterscale = if scale < 1.0 { 1.0 } else { scale };
    let support = 2.0 * filterscale;
    let ksize = (support.ceil() as usize) * 2 + 1;

    let mut bounds = Vec::with_capacity(dst_size);
    let mut weights_f = vec![0.0f64; dst_size * ksize];

    for xx in 0..dst_size {
        let center = (xx as f64 + 0.5) * scale;
        let ww = center - support + 0.5;
        let ee = center + support + 0.5;
        let left = ww.floor().max(0.0) as usize;
        let right = (ee.floor() as usize).min(src_size);
        bounds.push((left, right));

        let mut sum = 0.0;
        let base = xx * ksize;
        for x in left..right {
            let w = bicubic_kernel((x as f64 + 0.5 - center) / filterscale);
            weights_f[base + (x - left)] = w;
            sum += w;
        }
        if sum != 0.0 {
            let inv = 1.0 / sum;
            for x in 0..(right - left) {
                weights_f[base + x] *= inv;
            }
        }
    }

    // PIL's normalize_coeffs_8bpc: round to nearest, away from zero.
    let scale_i = (1u64 << PRECISION_BITS) as f64;
    let mut weights_i = vec![0i32; dst_size * ksize];
    for (i, &w) in weights_f.iter().enumerate() {
        let rounded = if w < 0.0 {
            -0.5 + w * scale_i
        } else {
            0.5 + w * scale_i
        };
        weights_i[i] = rounded as i32;
    }

    (bounds, ksize, weights_i)
}

#[inline]
fn clip8(acc: i32) -> u8 {
    let v = acc >> PRECISION_BITS;
    if v <= 0 {
        0
    } else if v >= 255 {
        255
    } else {
        v as u8
    }
}

/// Horizontal pass on u8 RGB data, shape (src_h, src_w, 3) → (src_h, dst_w, 3).
fn resample_horizontal_u8(src: &[u8], src_w: usize, src_h: usize, dst_w: usize) -> Vec<u8> {
    let (bounds, ksize, weights) = precompute_coeffs(src_w, dst_w);
    let mut out = vec![0u8; src_h * dst_w * 3];
    for y in 0..src_h {
        for xx in 0..dst_w {
            let (left, right) = bounds[xx];
            let base = xx * ksize;
            let mut r = ROUND_OFFSET;
            let mut g = ROUND_OFFSET;
            let mut b = ROUND_OFFSET;
            for x in left..right {
                let k = weights[base + (x - left)];
                let src_off = (y * src_w + x) * 3;
                r += (src[src_off] as i32) * k;
                g += (src[src_off + 1] as i32) * k;
                b += (src[src_off + 2] as i32) * k;
            }
            let dst_off = (y * dst_w + xx) * 3;
            out[dst_off] = clip8(r);
            out[dst_off + 1] = clip8(g);
            out[dst_off + 2] = clip8(b);
        }
    }
    out
}

/// Vertical pass on u8 RGB data, shape (src_h, dst_w, 3) → (dst_h, dst_w, 3).
fn resample_vertical_u8(src: &[u8], dst_w: usize, src_h: usize, dst_h: usize) -> Vec<u8> {
    let (bounds, ksize, weights) = precompute_coeffs(src_h, dst_h);
    let mut out = vec![0u8; dst_h * dst_w * 3];
    for yy in 0..dst_h {
        let (top, bottom) = bounds[yy];
        let base = yy * ksize;
        for x in 0..dst_w {
            let mut r = ROUND_OFFSET;
            let mut g = ROUND_OFFSET;
            let mut b = ROUND_OFFSET;
            for y in top..bottom {
                let k = weights[base + (y - top)];
                let src_off = (y * dst_w + x) * 3;
                r += (src[src_off] as i32) * k;
                g += (src[src_off + 1] as i32) * k;
                b += (src[src_off + 2] as i32) * k;
            }
            let dst_off = (yy * dst_w + x) * 3;
            out[dst_off] = clip8(r);
            out[dst_off + 1] = clip8(g);
            out[dst_off + 2] = clip8(b);
        }
    }
    out
}

/// PIL-compatible BICUBIC resize on an RGB image. Byte-exact match against
/// `PIL.Image.resize(size, Resampling.BICUBIC)`.
pub fn pil_resize_bicubic(img: &RgbImage, dst_w: u32, dst_h: u32) -> RgbImage {
    let src_w = img.width() as usize;
    let src_h = img.height() as usize;
    let dst_w_u = dst_w as usize;
    let dst_h_u = dst_h as usize;

    // Borrow the raw pixel buffer (packed RGB, u8, row-major).
    let src_bytes: &[u8] = img.as_raw();

    let h_out = resample_horizontal_u8(src_bytes, src_w, src_h, dst_w_u);
    let v_out = resample_vertical_u8(&h_out, dst_w_u, src_h, dst_h_u);

    let mut out = RgbImage::new(dst_w, dst_h);
    for y in 0..dst_h_u {
        for x in 0..dst_w_u {
            let off = (y * dst_w_u + x) * 3;
            out.put_pixel(
                x as u32,
                y as u32,
                Rgb([v_out[off], v_out[off + 1], v_out[off + 2]]),
            );
        }
    }
    out
}
