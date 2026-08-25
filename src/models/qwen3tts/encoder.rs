//! The 12 Hz codec **encoder** (`speech_tokenizer/model.safetensors`,
//! `encoder.*`) — a transformers `MimiModel` encoder path: reference audio →
//! 16 codes/frame, making arbitrary-reference voice cloning self-contained
//! (previously the reference codes were a captured oracle artifact).
//!
//! Architecture (measured from the checkpoint + transformers `modeling_mimi`):
//!   1. SEANet conv encoder, all causal (zero left-pad `k−stride`, right-pad
//!      to full frames): stem 1→64 k7, then per ratio r ∈ [4,5,6,8]:
//!      { residual unit (ELU → conv dim→dim/2 k3 → ELU → conv dim/2→dim k1,
//!      identity shortcut), ELU, downsample conv dim→2·dim k=2r stride r },
//!      then ELU + final conv 1024→512 k3. Total ×960 (24 kHz → 25 Hz).
//!   2. 8-layer transformer at hidden 512: LayerNorm (biased, eps 1e-5),
//!      8 heads × 64, biasless q/k/v/o, RoPE θ=10 000, **full causal** (the
//!      config says sliding_window 250, but the reference's eager/sdpa path
//!      never applies it — see the note in `transformer`), GELU MLP
//!      512→2048→512 (biasless), LayerScale on both residual branches.
//!   3. `downsample` conv 512→512 k4 stride 2 (biasless, causal) → 12.5 Hz.
//!   4. Split-RVQ **encode**: input_proj k1 512→256 (biasless) per bank;
//!      semantic bank = 1 quantizer, acoustic bank = residual chain (15 of
//!      its 31 quantizers used); EuclideanCodebook argmin over
//!      `embed_sum / clamp(cluster_usage, 1e-5)`.
//!
//! Runs **on the CPU** (Accelerate sgemm im2col convs): encoding happens once
//! per reference clip, so determinism and gateability beat throughput.

use super::config::*;
use super::cpu::{sgemm, sgemm_nt};
use crate::nn::weight_loader::WeightLoader;

const ENC_HIDDEN: usize = 512;
const ENC_HEADS: usize = 8;
const ENC_HEAD_DIM: usize = 64;
const ENC_LAYERS: usize = 8;
const ENC_INTER: usize = 2048;
const ENC_EPS: f64 = 1e-5;
const ENC_ROPE_THETA: f64 = 10_000.0;
/// Encoder ratios, outermost-first as built (`reversed(upsampling_ratios)`).
const ENC_RATIOS: [usize; 4] = [4, 5, 6, 8];

/// Causal Conv1d on the CPU: left-pad `k−stride`, right-pad to a whole number
/// of output frames (the Mimi "extra padding"), im2col + sgemm. Padding is
/// zeros except the `downsample` conv, which Mimi builds with
/// `pad_mode="replicate"` (edge values repeat). No dilation in this encoder.
struct CpuConv {
    w: Vec<f32>, // [out, in·k] (im2col layout: (c, j) → c·k + j)
    b: Option<Vec<f32>>,
    out: usize,
    inc: usize,
    k: usize,
    stride: usize,
    replicate_pad: bool,
}

impl CpuConv {
    fn load(
        loader: &WeightLoader,
        prefix: &str,
        stride: usize,
        bias: bool,
        replicate_pad: bool,
    ) -> Self {
        let (w, shape) = loader.load_f32(&format!("{prefix}.weight"));
        let (out, inc, k) = (shape[0], shape[1], shape[2]);
        Self {
            w,
            b: bias.then(|| loader.load_f32(&format!("{prefix}.bias")).0),
            out,
            inc,
            k,
            stride,
            replicate_pad,
        }
    }

    /// `x: [in, L]` → `[out, T]`.
    fn forward(&self, x: &[f32], l: usize) -> (Vec<f32>, usize) {
        let (k, s) = (self.k, self.stride);
        let pad_left = k - s; // padding_total, all left (causal)
        // extra right padding to a whole number of frames
        let n_frames =
            ((l + pad_left).saturating_sub(k) as f64 / s as f64 + 1.0).ceil() as usize - 1;
        let ideal = n_frames * s + k - pad_left;
        let pad_right = ideal.saturating_sub(l);
        let lp = l + pad_left + pad_right;
        let t = (lp - k) / s + 1;

        // im2col [in·k, T]
        let mut col = vec![0f32; self.inc * k * t];
        for c in 0..self.inc {
            let row = &x[c * l..(c + 1) * l];
            for j in 0..k {
                let dst = &mut col[(c * k + j) * t..(c * k + j) * t + t];
                for (ti, d) in dst.iter_mut().enumerate() {
                    let src = (ti * s + j) as isize - pad_left as isize;
                    if src >= 0 && (src as usize) < l {
                        *d = row[src as usize];
                    } else if self.replicate_pad {
                        *d = if src < 0 { row[0] } else { row[l - 1] };
                    }
                }
            }
        }
        let mut y = vec![0f32; self.out * t];
        sgemm(&self.w, &col, self.out, self.inc * k, t, &mut y);
        if let Some(b) = &self.b {
            for (o, bo) in b.iter().enumerate() {
                for v in &mut y[o * t..(o + 1) * t] {
                    *v += bo;
                }
            }
        }
        (y, t)
    }
}

fn elu(x: &mut [f32]) {
    for v in x.iter_mut() {
        if *v < 0.0 {
            *v = v.exp() - 1.0;
        }
    }
}

/// Exact (erf-based) GELU, matching torch's default.
fn gelu(x: &mut [f32]) {
    for v in x.iter_mut() {
        *v = 0.5 * *v * (1.0 + libm::erf(*v as f64 * std::f64::consts::FRAC_1_SQRT_2) as f32);
    }
}

/// One SEANet stage: residual unit + downsample.
struct EncBlock {
    res1: CpuConv, // dim → dim/2, k3
    res2: CpuConv, // dim/2 → dim, k1
    down: CpuConv, // dim → 2·dim, k=2r stride r
}

struct EncTrLayer {
    ln1_w: Vec<f32>,
    ln1_b: Vec<f32>,
    ln2_w: Vec<f32>,
    ln2_b: Vec<f32>,
    q: Vec<f32>, // [512, 512]
    k: Vec<f32>,
    v: Vec<f32>,
    o: Vec<f32>,
    fc1: Vec<f32>, // [2048, 512]
    fc2: Vec<f32>, // [512, 2048]
    ls_attn: Vec<f32>,
    ls_mlp: Vec<f32>,
}

/// One RVQ bank's encode side: input_proj + pre-divided codebooks (+ their
/// squared row norms, for the argmin).
struct RvqEncoder {
    input_proj: Vec<f32>, // [256, 512]
    codebooks: Vec<Vec<f32>>,
    norms: Vec<Vec<f32>>, // per codebook: [2048] row ‖e‖²
}

impl RvqEncoder {
    fn load(loader: &WeightLoader, prefix: &str, n_q: usize) -> Self {
        let mut codebooks = Vec::with_capacity(n_q);
        let mut norms = Vec::with_capacity(n_q);
        for i in 0..n_q {
            let (sum, _) = loader.load_f32(&format!("{prefix}.layers.{i}.codebook.embed_sum"));
            let (usage, _) =
                loader.load_f32(&format!("{prefix}.layers.{i}.codebook.cluster_usage"));
            let mut cb = sum;
            for (r, &u) in usage.iter().enumerate() {
                let d = u.max(1e-5);
                for v in &mut cb[r * DEC_CODE_DIM..(r + 1) * DEC_CODE_DIM] {
                    *v /= d;
                }
            }
            norms.push(
                (0..usage.len())
                    .map(|r| {
                        cb[r * DEC_CODE_DIM..(r + 1) * DEC_CODE_DIM]
                            .iter()
                            .map(|&v| v * v)
                            .sum::<f32>()
                    })
                    .collect(),
            );
            codebooks.push(cb);
        }
        Self {
            input_proj: loader.load_f32(&format!("{prefix}.input_proj.weight")).0,
            codebooks,
            norms,
        }
    }

    /// Residual-VQ encode `x: [T, 512]` → per-quantizer codes `[n_q][T]`.
    fn encode(&self, x: &[f32], t: usize) -> Vec<Vec<u32>> {
        // input_proj (k1 conv ≡ matmul): [T, 512] → [T, 256]
        let mut residual = vec![0f32; t * DEC_CODE_DIM];
        sgemm_nt(
            x,
            &self.input_proj,
            t,
            ENC_HIDDEN,
            DEC_CODE_DIM,
            &mut residual,
        );

        let mut out = Vec::with_capacity(self.codebooks.len());
        for (cb, norms) in self.codebooks.iter().zip(&self.norms) {
            let rows = norms.len();
            // dists² = ‖r‖² − 2·r·e + ‖e‖²; ‖r‖² is row-constant → argmin over
            // (‖e‖² − 2·r·e)
            let mut dots = vec![0f32; t * rows];
            sgemm_nt(&residual, cb, t, DEC_CODE_DIM, rows, &mut dots);
            let mut codes = Vec::with_capacity(t);
            for ti in 0..t {
                let row = &dots[ti * rows..(ti + 1) * rows];
                let mut best = (f32::MAX, 0usize);
                for (i, (&d, &n)) in row.iter().zip(norms).enumerate() {
                    let dist = n - 2.0 * d;
                    if dist < best.0 {
                        best = (dist, i);
                    }
                }
                codes.push(best.1 as u32);
                let e = &cb[best.1 * DEC_CODE_DIM..(best.1 + 1) * DEC_CODE_DIM];
                for (r, &ev) in residual[ti * DEC_CODE_DIM..(ti + 1) * DEC_CODE_DIM]
                    .iter_mut()
                    .zip(e)
                {
                    *r -= ev;
                }
            }
            out.push(codes);
        }
        out
    }
}

pub struct CodecEncoder {
    stem: CpuConv,
    blocks: Vec<EncBlock>,
    final_conv: CpuConv,
    tr_layers: Vec<EncTrLayer>,
    downsample: CpuConv,
    semantic: RvqEncoder,
    acoustic: RvqEncoder,
}

impl CodecEncoder {
    pub fn load(loader: &WeightLoader) -> Self {
        let p = "encoder.encoder.layers";
        // layer indices in the checkpoint: 0 stem; per stage i: 3i+1 residual
        // block (block.1, block.3), 3i+3 downsample; 14 final conv.
        let blocks = ENC_RATIOS
            .iter()
            .enumerate()
            .map(|(i, &r)| EncBlock {
                res1: CpuConv::load(
                    loader,
                    &format!("{p}.{}.block.1.conv", 3 * i + 1),
                    1,
                    true,
                    false,
                ),
                res2: CpuConv::load(
                    loader,
                    &format!("{p}.{}.block.3.conv", 3 * i + 1),
                    1,
                    true,
                    false,
                ),
                down: CpuConv::load(loader, &format!("{p}.{}.conv", 3 * i + 3), r, true, false),
            })
            .collect();
        let t = "encoder.encoder_transformer.layers";
        let tr_layers = (0..ENC_LAYERS)
            .map(|i| EncTrLayer {
                ln1_w: loader
                    .load_f32(&format!("{t}.{i}.input_layernorm.weight"))
                    .0,
                ln1_b: loader.load_f32(&format!("{t}.{i}.input_layernorm.bias")).0,
                ln2_w: loader
                    .load_f32(&format!("{t}.{i}.post_attention_layernorm.weight"))
                    .0,
                ln2_b: loader
                    .load_f32(&format!("{t}.{i}.post_attention_layernorm.bias"))
                    .0,
                q: loader
                    .load_f32(&format!("{t}.{i}.self_attn.q_proj.weight"))
                    .0,
                k: loader
                    .load_f32(&format!("{t}.{i}.self_attn.k_proj.weight"))
                    .0,
                v: loader
                    .load_f32(&format!("{t}.{i}.self_attn.v_proj.weight"))
                    .0,
                o: loader
                    .load_f32(&format!("{t}.{i}.self_attn.o_proj.weight"))
                    .0,
                fc1: loader.load_f32(&format!("{t}.{i}.mlp.fc1.weight")).0,
                fc2: loader.load_f32(&format!("{t}.{i}.mlp.fc2.weight")).0,
                ls_attn: loader
                    .load_f32(&format!("{t}.{i}.self_attn_layer_scale.scale"))
                    .0,
                ls_mlp: loader.load_f32(&format!("{t}.{i}.mlp_layer_scale.scale")).0,
            })
            .collect();
        Self {
            stem: CpuConv::load(loader, &format!("{p}.0.conv"), 1, true, false),
            blocks,
            final_conv: CpuConv::load(loader, &format!("{p}.14.conv"), 1, true, false),
            tr_layers,
            downsample: CpuConv::load(loader, "encoder.downsample.conv", 2, false, true),
            semantic: RvqEncoder::load(
                loader,
                "encoder.quantizer.semantic_residual_vector_quantizer",
                1,
            ),
            acoustic: RvqEncoder::load(
                loader,
                "encoder.quantizer.acoustic_residual_vector_quantizer",
                NUM_CODE_GROUPS - 1,
            ),
        }
    }

    /// LayerNorm over the last dim of `x: [T, 512]`, biased, eps 1e-5.
    fn layer_norm(x: &[f32], w: &[f32], b: &[f32], out: &mut [f32]) {
        let d = w.len();
        for (row_in, row_out) in x.chunks_exact(d).zip(out.chunks_exact_mut(d)) {
            let mean = row_in.iter().map(|&v| v as f64).sum::<f64>() / d as f64;
            let var = row_in
                .iter()
                .map(|&v| (v as f64 - mean).powi(2))
                .sum::<f64>()
                / d as f64;
            let inv = ((var + ENC_EPS).sqrt().recip()) as f32;
            let mean = mean as f32;
            for i in 0..d {
                row_out[i] = (row_in[i] - mean) * inv * w[i] + b[i];
            }
        }
    }

    /// The 8-layer transformer over `h: [T, 512]`, in place.
    fn transformer(&self, h: &mut [f32], t: usize) {
        let (hd, nh, d) = (ENC_HIDDEN, ENC_HEADS, ENC_HEAD_DIM);
        let half = d / 2;
        let scale = ((d as f64).powf(-0.5)) as f32;
        // RoPE tables [t, half]
        let mut cos = vec![0f32; t * half];
        let mut sin = vec![0f32; t * half];
        for pos in 0..t {
            for i in 0..half {
                let r = pos as f64 * ENC_ROPE_THETA.powf(-2.0 * i as f64 / d as f64);
                cos[pos * half + i] = r.cos() as f32;
                sin[pos * half + i] = r.sin() as f32;
            }
        }
        let rope = |x: &mut [f32], pos: usize| {
            for i in 0..half {
                let (c, s) = (cos[pos * half + i], sin[pos * half + i]);
                let (a, b) = (x[i], x[i + half]);
                x[i] = a * c - b * s;
                x[i + half] = b * c + a * s;
            }
        };

        let mut xin = vec![0f32; t * hd];
        let mut q = vec![0f32; t * hd];
        let mut k = vec![0f32; t * hd];
        let mut v = vec![0f32; t * hd];
        let mut attn = vec![0f32; t * hd];
        let mut proj = vec![0f32; t * hd];
        let mut ff = vec![0f32; t * ENC_INTER];
        for l in &self.tr_layers {
            Self::layer_norm(h, &l.ln1_w, &l.ln1_b, &mut xin);
            sgemm_nt(&xin, &l.q, t, hd, hd, &mut q);
            sgemm_nt(&xin, &l.k, t, hd, hd, &mut k);
            sgemm_nt(&xin, &l.v, t, hd, hd, &mut v);
            for pos in 0..t {
                for hh in 0..nh {
                    rope(&mut q[pos * hd + hh * d..pos * hd + (hh + 1) * d], pos);
                    rope(&mut k[pos * hd + hh * d..pos * hd + (hh + 1) * d], pos);
                }
            }
            // causal sliding-window attention
            attn.fill(0.0);
            let mut scores = vec![0f32; t];
            for hh in 0..nh {
                for qp in 0..t {
                    // NOTE: the config declares sliding_window=250, but the
                    // reference (transformers eager/sdpa path) builds a PLAIN
                    // causal mask — `create_causal_mask` only honors the
                    // window on the flash-attn path, which the oracle didn't
                    // run. Full causal matches the oracle bit-for-bit; with
                    // the window, positions ≥250 diverge.
                    let lo = 0;
                    let qrow = &q[qp * hd + hh * d..qp * hd + (hh + 1) * d];
                    for (si, kp) in (lo..=qp).enumerate() {
                        let krow = &k[kp * hd + hh * d..kp * hd + (hh + 1) * d];
                        scores[si] =
                            qrow.iter().zip(krow).map(|(&a, &b)| a * b).sum::<f32>() * scale;
                    }
                    let n = qp - lo + 1;
                    super::cpu::softmax(&mut scores[..n]);
                    let out = &mut attn[qp * hd + hh * d..qp * hd + (hh + 1) * d];
                    for (si, kp) in (lo..=qp).enumerate() {
                        let vrow = &v[kp * hd + hh * d..kp * hd + (hh + 1) * d];
                        let p = scores[si];
                        for (o, &vv) in out.iter_mut().zip(vrow) {
                            *o += p * vv;
                        }
                    }
                }
            }
            sgemm_nt(&attn, &l.o, t, hd, hd, &mut proj);
            for pos in 0..t {
                for i in 0..hd {
                    h[pos * hd + i] += proj[pos * hd + i] * l.ls_attn[i];
                }
            }
            Self::layer_norm(h, &l.ln2_w, &l.ln2_b, &mut xin);
            sgemm_nt(&xin, &l.fc1, t, hd, ENC_INTER, &mut ff);
            gelu(&mut ff);
            sgemm_nt(&ff, &l.fc2, t, ENC_INTER, hd, &mut proj);
            for pos in 0..t {
                for i in 0..hd {
                    h[pos * hd + i] += proj[pos * hd + i] * l.ls_mlp[i];
                }
            }
        }
    }

    /// Encode a 24 kHz mono waveform into codec frames (T × 16 codes,
    /// codebook 0 = semantic). `T = ceil(len / 1920)`.
    pub fn encode(&self, samples: &[f32]) -> Vec<[u32; NUM_CODE_GROUPS]> {
        // SEANet
        let (mut x, mut l) = self.stem.forward(samples, samples.len());
        for blk in &self.blocks {
            // residual unit (identity shortcut)
            let mut h = x.clone();
            elu(&mut h);
            let (mut h, hl) = blk.res1.forward(&h, l);
            elu(&mut h);
            let (h, hl2) = blk.res2.forward(&h, hl);
            debug_assert_eq!(hl2, l);
            for (xv, hv) in x.iter_mut().zip(&h) {
                *xv += hv;
            }
            elu(&mut x);
            let (nx, nl) = blk.down.forward(&x, l);
            x = nx;
            l = nl;
        }
        elu(&mut x);
        let (x, l) = self.final_conv.forward(&x, l);

        // transformer works [T, C]; convs produced [C, T]
        let mut h = vec![0f32; l * ENC_HIDDEN];
        for c in 0..ENC_HIDDEN {
            for ti in 0..l {
                h[ti * ENC_HIDDEN + c] = x[c * l + ti];
            }
        }
        self.transformer(&mut h, l);
        let mut ht = vec![0f32; ENC_HIDDEN * l];
        for c in 0..ENC_HIDDEN {
            for ti in 0..l {
                ht[c * l + ti] = h[ti * ENC_HIDDEN + c];
            }
        }

        // ×2 downsample → 12.5 Hz, back to [T, C] for the quantizer
        let (x, t) = self.downsample.forward(&ht, l);
        let mut emb = vec![0f32; t * ENC_HIDDEN];
        for c in 0..ENC_HIDDEN {
            for ti in 0..t {
                emb[ti * ENC_HIDDEN + c] = x[c * t + ti];
            }
        }

        let sem = self.semantic.encode(&emb, t);
        let ac = self.acoustic.encode(&emb, t);
        // truncate to ceil(samples / 1920) frames (the oracle's padding-mask cut)
        let frames = samples.len().div_ceil(SAMPLES_PER_FRAME).min(t);
        (0..frames)
            .map(|ti| {
                let mut f = [0u32; NUM_CODE_GROUPS];
                f[0] = sem[0][ti];
                for (qi, codes) in ac.iter().enumerate() {
                    f[qi + 1] = codes[ti];
                }
                f
            })
            .collect()
    }
}
