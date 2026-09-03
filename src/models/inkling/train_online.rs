//! Online, in-place learning on the model's own 4-bit codes (compass 84605490, S3).
//!
//! The question this answers is JP's: can Inkling learn from the user turns it
//! serves with NO master weights and NO optimizer state -- the update landing
//! directly on the NVFP4 codes by unbiased stochastic rounding, block scales
//! frozen -- and the only score is the prequential loss on the next user turns.
//!
//! Two kinds of override for a routed expert: `F32` (a master copy, the ceiling
//! of what the same gradient could do) and `Fp4` (the checkpoint's own packed
//! format, re-encoded after each step). Both are keyed by expert within ONE
//! layer; the driver keeps the rest of the model as the checkpoint has it.
use crate::models::inkling::load::{Checkpoint, PackedExpert};
use crate::models::inkling::nvfp4::{GROUP, decode_row, e4m3_to_f32};
use anyhow::Result;
use std::collections::HashMap;

/// A packed expert whose w13 rows are in GATE-FIRST order (the checkpoint interleaves them).
pub struct Fp4Expert { pub w13: PackedExpert, pub w2: PackedExpert }

pub enum ExpertState {
    F32 { w13: Vec<f32>, w2: Vec<f32> },
    Fp4(Fp4Expert),
}

/// Reorder the rows of a packed matrix from interleaved (g0,u0,g1,u1,...) to gate-first.
pub fn deinterleave_packed_rows(p: &PackedExpert) -> PackedExpert {
    assert!(p.rows % 2 == 0);
    let half = p.rows / 2;
    let cb = p.cols;               // code bytes per row
    let sb = p.cols * 2 / GROUP;   // scale bytes per row
    let mut codes = vec![0u8; p.codes.len()];
    let mut scales = vec![0u8; p.scales.len()];
    for r in 0..p.rows {
        let dst = if r % 2 == 0 { r / 2 } else { half + r / 2 };
        codes[dst * cb..(dst + 1) * cb].copy_from_slice(&p.codes[r * cb..(r + 1) * cb]);
        scales[dst * sb..(dst + 1) * sb].copy_from_slice(&p.scales[r * sb..(r + 1) * sb]);
    }
    PackedExpert { codes, scales, scale2: p.scale2, rows: p.rows, cols: p.cols }
}

pub fn fetch_fp4(cp: &Checkpoint, layer: usize, e: usize) -> Result<Fp4Expert> {
    let pfx = format!("model.llm.layers.{layer}.");
    let w13 = deinterleave_packed_rows(&cp.expert_slice_packed(&format!("{pfx}mlp.experts.w13_weight"), e)?);
    let w2 = cp.expert_slice_packed(&format!("{pfx}mlp.experts.w2_weight"), e)?;
    Ok(Fp4Expert { w13, w2 })
}

pub fn decode_packed(p: &PackedExpert) -> Vec<f32> {
    let logical = p.cols * 2;
    let sb = logical / GROUP;
    let mut out = vec![0f32; p.rows * logical];
    for r in 0..p.rows {
        decode_row(&p.codes[r * p.cols..(r + 1) * p.cols], &p.scales[r * sb..(r + 1) * sb], p.scale2, &mut out[r * logical..(r + 1) * logical]);
    }
    out
}

/// xorshift64*: cheap, seedable, good enough for rounding coins.
pub struct Coin(u64);
impl Coin {
    pub fn new(seed: u64) -> Self { Coin(seed.max(1) ^ 0x9E37_79B9_7F4A_7C15) }
    #[inline] pub fn unit(&mut self) -> f32 {
        let mut x = self.0; x ^= x >> 12; x ^= x << 25; x ^= x >> 27; self.0 = x;
        ((x.wrapping_mul(0x2545_F491_4F6C_DD1D) >> 40) as f32) / 16_777_216.0
    }
}

/// Unbiased stochastic rounding of one value onto the E2M1 grid at a fixed scale.
/// Returns the 4-bit code. Magnitudes beyond 6*scale clip (the one biased region).
#[inline]
pub fn sr_code(v: f32, scale: f32, coin: &mut Coin) -> u8 {
    if !(scale > 0.0) || !v.is_finite() { return 0; }
    let x = v / scale;
    let neg = x < 0.0;
    let a = x.abs().min(6.0);
    // grid magnitudes by code 0..8: 0 .5 1 1.5 2 3 4 6
    const G: [f32; 8] = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0];
    let mut hi = 1usize;
    while hi < 7 && G[hi] < a { hi += 1; }
    let lo = hi - 1;
    let code = if a <= G[lo] { lo } else if a >= G[hi] { hi } else {
        let p = (a - G[lo]) / (G[hi] - G[lo]);
        if coin.unit() < p { hi } else { lo }
    };
    (code as u8) | if neg && code != 0 { 8 } else { 0 }
}

/// Deterministic round-to-nearest onto the E2M1 grid at a fixed scale (the control for `sr_code`:
/// a step smaller than half a grid gap vanishes entirely, so a learner rounding this way
/// learns nothing until its step is large).
#[inline]
pub fn nearest_code(v: f32, scale: f32) -> u8 {
    if !(scale > 0.0) || !v.is_finite() { return 0; }
    let x = v / scale;
    let neg = x < 0.0;
    let a = x.abs().min(6.0);
    const G: [f32; 8] = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0];
    let mut best = 0usize;
    for (i, g) in G.iter().enumerate() { if (a - g).abs() < (a - G[best]).abs() { best = i; } }
    (best as u8) | if neg && best != 0 { 8 } else { 0 }
}

/// Re-encode a whole packed matrix from new f32 values, keeping every block scale.
/// `coin: None` rounds to nearest (the deterministic control); `Some` rounds stochastically.
pub fn encode_into(p: &mut PackedExpert, values: &[f32], mut coin: Option<&mut Coin>) -> (u64, u64) {
    let logical = p.cols * 2;
    let sb = logical / GROUP;
    assert_eq!(values.len(), p.rows * logical);
    let (mut changed, mut clipped) = (0u64, 0u64);
    for r in 0..p.rows {
        for b in 0..sb {
            let scale = e4m3_to_f32(p.scales[r * sb + b]) * p.scale2;
            for i in 0..GROUP / 2 {
                let j = r * logical + b * GROUP + 2 * i;
                let (v0, v1) = (values[j], values[j + 1]);
                if scale > 0.0 && (v0.abs() > 6.0 * scale || v1.abs() > 6.0 * scale) { clipped += 1; }
                let (c0, c1) = match coin.as_deref_mut() {
                    Some(c) => (sr_code(v0, scale, c), sr_code(v1, scale, c)),
                    None => (nearest_code(v0, scale), nearest_code(v1, scale)),
                };
                let byte = c0 | (c1 << 4);
                let k = r * p.cols + b * (GROUP / 2) + i;
                if p.codes[k] != byte { changed += 1; }
                p.codes[k] = byte;
            }
        }
    }
    (changed, clipped)
}

/// Re-encode a whole packed matrix from new f32 values, keeping every block scale.
pub fn sr_encode_into(p: &mut PackedExpert, values: &[f32], coin: &mut Coin) -> (u64, u64) {
    let logical = p.cols * 2;
    let sb = logical / GROUP;
    assert_eq!(values.len(), p.rows * logical);
    let (mut changed, mut clipped) = (0u64, 0u64);
    for r in 0..p.rows {
        for b in 0..sb {
            let scale = e4m3_to_f32(p.scales[r * sb + b]) * p.scale2;
            for i in 0..GROUP / 2 {
                let j = r * logical + b * GROUP + 2 * i;
                let (v0, v1) = (values[j], values[j + 1]);
                if scale > 0.0 && (v0.abs() > 6.0 * scale || v1.abs() > 6.0 * scale) { clipped += 1; }
                let c0 = sr_code(v0, scale, coin);
                let c1 = sr_code(v1, scale, coin);
                let byte = c0 | (c1 << 4);
                let k = r * p.cols + b * (GROUP / 2) + i;
                if p.codes[k] != byte { changed += 1; }
                p.codes[k] = byte;
            }
        }
    }
    (changed, clipped)
}

/// The per-layer override table for one arm.
pub struct Arm {
    pub name: String,
    pub lr: f32,
    pub kind: ArmKind,
    /// keyed by (layer, expert)
    pub states: HashMap<(usize, usize), ExpertState>,
    pub coin: Coin,
    pub codes_changed: u64,
    pub codes_clipped: u64,
}
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum ArmKind { None, F32, Fp4Sr, Fp4Nearest }

impl Arm {
    pub fn new(name: &str, kind: ArmKind, lr: f32, seed: u64) -> Self {
        Arm { name: name.into(), lr, kind, states: HashMap::new(), coin: Coin::new(seed), codes_changed: 0, codes_clipped: 0 }
    }
    /// The f32 weights this arm currently holds for expert `e` of `layer`, or None if untouched (checkpoint).
    pub fn current(&self, layer: usize, e: usize) -> Option<(Vec<f32>, Vec<f32>)> {
        match self.states.get(&(layer, e))? {
            ExpertState::F32 { w13, w2 } => Some((w13.clone(), w2.clone())),
            ExpertState::Fp4(q) => Some((decode_packed(&q.w13), decode_packed(&q.w2))),
        }
    }
    /// Apply one SGD step to expert `e` in this arm's representation. `base` is the checkpoint's f32
    /// (gate-first) used to seed an untouched expert; `g13`/`g2` the gradients.
    pub fn step(&mut self, cp: &Checkpoint, layer: usize, e: usize, base: &(Vec<f32>, Vec<f32>), g13: &[f32], g2: &[f32]) -> Result<()> {
        let lr = self.lr;
        match self.kind {
            ArmKind::None => {}
            ArmKind::F32 => {
                let st = self.states.entry((layer, e)).or_insert_with(|| ExpertState::F32 { w13: base.0.clone(), w2: base.1.clone() });
                if let ExpertState::F32 { w13, w2 } = st {
                    for (w, g) in w13.iter_mut().zip(g13) { *w -= lr * g; }
                    for (w, g) in w2.iter_mut().zip(g2) { *w -= lr * g; }
                }
            }
            ArmKind::Fp4Sr | ArmKind::Fp4Nearest => {
                let stochastic = self.kind == ArmKind::Fp4Sr;
                if !self.states.contains_key(&(layer, e)) { self.states.insert((layer, e), ExpertState::Fp4(fetch_fp4(cp, layer, e)?)); }
                let coin = &mut self.coin;
                if let Some(ExpertState::Fp4(q)) = self.states.get_mut(&(layer, e)) {
                    let mut w13 = decode_packed(&q.w13);
                    let mut w2 = decode_packed(&q.w2);
                    for (w, g) in w13.iter_mut().zip(g13) { *w -= lr * g; }
                    for (w, g) in w2.iter_mut().zip(g2) { *w -= lr * g; }
                    let (c1, k1) = encode_into(&mut q.w13, &w13, if stochastic { Some(&mut *coin) } else { None });
                    let (c2, k2) = encode_into(&mut q.w2, &w2, if stochastic { Some(&mut *coin) } else { None });
                    self.codes_changed += c1 + c2; self.codes_clipped += k1 + k2;
                }
            }
        }
        Ok(())
    }
}
