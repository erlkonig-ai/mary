//! `inkling_quant_levels` — does a second scale level fix the activation
//! quantiser, and by how much?
//!
//! The one-level quantiser sets `block_scale = amax_block / 6` and rounds that
//! straight to E4M3. E4M3 is a *tiny* type — 3 mantissa bits, normals only from
//! 2^-6, subnormals to 2^-9 — so when activations are small the block scale
//! falls off the bottom of its range and the whole block is destroyed, and when
//! the scale rounds up the elements land systematically low.
//!
//! The checkpoint's weights do not have this problem, because NVFP4 is a
//! **two-level** format: an F32 per-tensor `scale2` normalises the tensor so
//! the E4M3 block scales sit in the middle of their range. `hf_quant_config`
//! asks for the same treatment on activations (`*input_quantizer`), and the
//! one-level version was simply an incomplete implementation of it.
//!
//! Two-level, the standard recipe:
//!
//!   global      = amax_tensor / (6 * 448)          (448 = largest E4M3)
//!   block_scale = e4m3(amax_block / 6 / global)
//!   code        = e2m1(x / (global * block_scale))
//!
//! and the matmul multiplies by `global` afterwards, exactly as it already
//! multiplies by the weight's `scale2`.
//!
//! Build: `--features cuda-backend,inkling`

use std::path::PathBuf;

use anyhow::Result;

use mary::models::inkling::fp4gemm::{f32_to_e4m3, quantize_act_host, ExpertSource, GROUP};
use mary::models::inkling::nvfp4::{e4m3_to_f32, FP4_E2M1};

const LAYER: usize = 10;
const E4M3_MAX: f32 = 448.0;
const FP4_MAX: f32 = 6.0;

fn decode(codes: &[u8], scales: &[u8], rows: usize, k: usize, scale2: f32) -> Vec<f32> {
    let mut out = vec![0f32; rows * k];
    for r in 0..rows {
        for j in 0..k {
            let byte = codes[r * (k / 2) + j / 2];
            let c = if j % 2 == 0 { byte & 0x0F } else { byte >> 4 };
            out[r * k + j] =
                FP4_E2M1[c as usize] * e4m3_to_f32(scales[r * (k / GROUP) + j / GROUP]) * scale2;
        }
    }
    out
}

fn e2m1_code(q: f32) -> u8 {
    let a = q.abs();
    let m: u8 = if a < 0.25 {
        0
    } else if a < 0.75 {
        1
    } else if a < 1.25 {
        2
    } else if a < 1.75 {
        3
    } else if a < 2.5 {
        4
    } else if a < 3.5 {
        5
    } else if a < 5.0 {
        6
    } else {
        7
    };
    if q < 0.0 { m + 8 } else { m }
}

/// Two-level quantisation. Returns (codes, scale bytes, global).
fn quantize_two_level(x: &[f32], k: usize) -> (Vec<u8>, Vec<u8>, f32) {
    let amax_t = x.iter().fold(0.0f32, |m, v| m.max(v.abs()));
    let global = if amax_t > 0.0 { amax_t / (FP4_MAX * E4M3_MAX) } else { 1.0 };
    let nblocks = x.len() / GROUP;
    let mut codes = vec![0u8; x.len() / 2];
    let mut scales = vec![0u8; nblocks];
    for b in 0..nblocks {
        let base = b * GROUP;
        let amax = (0..GROUP).map(|i| x[base + i].abs()).fold(0.0f32, f32::max);
        let sb = f32_to_e4m3(amax / FP4_MAX / global);
        scales[b] = sb;
        let s = e4m3_to_f32(sb) * global;
        if !(s > 0.0) {
            continue;
        }
        for i in 0..GROUP {
            let c = e2m1_code(x[base + i] / s);
            let j = base + i;
            if j % 2 == 0 {
                codes[j / 2] |= c;
            } else {
                codes[j / 2] |= c << 4;
            }
        }
    }
    (codes, scales, global)
}


/// The floor: an EXACT f32 block scale, no E4M3 rounding anywhere. Whatever
/// error survives this is the E2M1 grid itself, and no scale scheme -- one
/// level, two levels, or a perfect one -- can remove it.
fn quantize_ideal(x: &[f32]) -> Vec<f32> {
    let mut out = vec![0f32; x.len()];
    for b in 0..x.len() / GROUP {
        let base = b * GROUP;
        let amax = (0..GROUP).map(|i| x[base + i].abs()).fold(0.0f32, f32::max);
        let s = amax / FP4_MAX;
        if !(s > 0.0) {
            continue;
        }
        for i in 0..GROUP {
            let c = e2m1_code(x[base + i] / s);
            out[base + i] = FP4_E2M1[(c & 0x0F) as usize] * s;
        }
    }
    out
}


/// Pick the E4M3 block scale that minimises the block's error, instead of
/// assuming `amax/6` is the best one.
///
/// `amax/6` is the scale that makes the largest element land exactly on the top
/// E2M1 code. That is the only scale which cannot clip -- but clipping one
/// element is often much cheaper than mis-rounding the other fifteen, and
/// rounding `amax/6` itself to E4M3 (3 mantissa bits) usually moves it anyway.
/// So search a small neighbourhood of E4M3 codes around it and keep whichever
/// minimises squared error over the block.
fn quantize_searched(x: &[f32]) -> Vec<f32> {
    let mut out = vec![0f32; x.len()];
    for b in 0..x.len() / GROUP {
        let base = b * GROUP;
        let amax = (0..GROUP).map(|i| x[base + i].abs()).fold(0.0f32, f32::max);
        if !(amax > 0.0) {
            continue;
        }
        let start = f32_to_e4m3(amax / FP4_MAX);
        let mut best = start;
        let mut best_err = f32::INFINITY;
        // +-4 E4M3 codes is about +-50% in scale, which brackets the rounding.
        for d in -4i32..=4 {
            let cand = (start as i32 + d).clamp(1, 126) as u8;
            let sc = e4m3_to_f32(cand);
            if !(sc > 0.0) {
                continue;
            }
            let mut err = 0.0f32;
            for i in 0..GROUP {
                let c = e2m1_code(x[base + i] / sc);
                let d = FP4_E2M1[(c & 0x0F) as usize] * sc - x[base + i];
                err += d * d;
            }
            if err < best_err {
                best_err = err;
                best = cand;
            }
        }
        let sc = e4m3_to_f32(best);
        for i in 0..GROUP {
            let c = e2m1_code(x[base + i] / sc);
            out[base + i] = FP4_E2M1[(c & 0x0F) as usize] * sc;
        }
    }
    out
}

fn stats(orig: &[f32], deq: &[f32]) -> (f64, f64, f64) {
    let mut sum = 0.0f64;
    let mut worst = 0.0f64;
    let mut zeros = 0usize;
    let mut n = 0usize;
    for (o, d) in orig.iter().zip(deq) {
        if o.abs() < 1e-12 {
            continue;
        }
        let r = ((d - o).abs() / o.abs()) as f64;
        sum += r;
        if r > worst {
            worst = r;
        }
        if d.abs() < 1e-20 {
            zeros += 1;
        }
        n += 1;
    }
    (sum / n as f64, worst, zeros as f64 / n as f64)
}

fn main() -> Result<()> {
    let dir = std::env::args()
        .nth(1)
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("models/thinkingmachines-inkling-small-nvfp4"));
    let src = ExpertSource::open(&dir)?;
    let b13 = format!("model.llm.layers.{LAYER}.mlp.experts.w13_weight");

    println!("=== activation quantiser: one level vs two ===");
    println!("(input = real decoded expert rows, which have the dynamic range the");
    println!(" real activations have; K=4096)\n");
    println!(
        "{:<10} {:>12} {:>12} {:>12} {:>12} {:>10} {:>12} {:>12}",
        "expert", "1-lvl mean", "1-lvl max", "2-lvl mean", "2-lvl max", "improve",
        "exact mean", "exact max"
    );

    let mut tot1 = 0.0;
    let mut tot2 = 0.0;
    let mut n = 0;
    for e in [0usize, 3, 7, 11, 19] {
        let w = src.expert(&b13, e)?;
        let k = w.cols * 2;
        let rows = 16;
        let x = decode(w.codes, w.scales, rows, k, w.scale2);

        let (c1, s1) = quantize_act_host(&x, k);
        let d1 = decode(&c1, &s1, rows, k, 1.0);
        let (m1, w1, _) = stats(&x, &d1);

        let (c2, s2, g) = quantize_two_level(&x, k);
        let d2 = decode(&c2, &s2, rows, k, g);
        let (m2, w2v, _) = stats(&x, &d2);

        let d3 = quantize_ideal(&x);
        let (m3, w3, _) = stats(&x, &d3);
        let d4 = quantize_searched(&x);
        let (m4, _w4, _) = stats(&x, &d4);
        println!(
            "{:<10} {:>12.3e} {:>12.3e} {:>12.3e} {:>12.3e} {:>9.1}x  {:>12.3e} {:>12.3e}",
            e, m1, w1, m2, w2v, m1 / m2, m3, w3
        );
        println!("{:<10} searched-scale mean {:.3e}   ({:.2}x better than 1-level, floor is {:.3e})",
                 "", m4, m1 / m4, m3);
        tot1 += m1;
        tot2 += m2;
        n += 1;
    }
    println!();
    println!(
        "mean relative error: one level {:.3e}  ->  two level {:.3e}   ({:.1}x better)",
        tot1 / n as f64,
        tot2 / n as f64,
        tot1 / tot2
    );

    // Why: where do the block scales actually land in E4M3's range?
    let w = src.expert(&b13, 0)?;
    let k = w.cols * 2;
    let x = decode(w.codes, w.scales, 16, k, w.scale2);
    let (_, s1) = quantize_act_host(&x, k);
    let (_, s2, _) = quantize_two_level(&x, k);
    let sub1 = s1.iter().filter(|&&b| (b >> 3) & 0x0F == 0).count();
    let sub2 = s2.iter().filter(|&&b| (b >> 3) & 0x0F == 0).count();
    let zero1 = s1.iter().filter(|&&b| b == 0).count();
    let zero2 = s2.iter().filter(|&&b| b == 0).count();
    println!();
    println!("block scales that fell into E4M3 SUBNORMALS (3 bits of precision or fewer):");
    println!("  one level : {sub1} of {} ({:.1}%), of which {zero1} rounded to exactly zero",
             s1.len(), 100.0 * sub1 as f64 / s1.len() as f64);
    println!("  two level : {sub2} of {} ({:.1}%), of which {zero2} rounded to exactly zero",
             s2.len(), 100.0 * sub2 as f64 / s2.len() as f64);
    Ok(())
}
