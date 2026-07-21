//! Mimi **encoder** (moshi `encoder.*`, `encoder_transformer.*`, `downsample.*`,
//! `quantizer.*`): 24 kHz mono → `T×8` integer codes.
//!
//! Pipeline (moshi `_encode_to_unquantized_latent` + `encode`):
//!   1. `pad_for_conv1d`, then SEANet conv encoder (`encoder.model.*`), all
//!      causal (left-pad `k−stride`, right-pad to whole frames): stem 1→64 k7,
//!      then per ratio r ∈ [8,6,5,4]: { residual unit (ELU → conv dim→dim/2 k3
//!      → ELU → conv dim/2→dim k1, identity shortcut), ELU, downsample conv
//!      dim→2·dim k=2r stride r }, then ELU + final conv 1024→512 k3. ×960.
//!   2. `encoder_transformer`: 8-layer, width 512, 8 heads × 64, LayerNorm
//!      (biased, eps 1e-5), fused biasless `in_projs.0` (qkv) / `out_projs.0`,
//!      RoPE θ=10000, **causal sliding-window 250** (moshi applies it), GELU
//!      MLP 512→2048→512 (biasless), LayerScale on both residual branches.
//!   3. `downsample` conv 512→512 k4 stride 2 (biasless, causal, replicate pad)
//!      → 12.5 Hz.
//!   4. Split-RVQ **encode**: rvq_first (1 semantic quantizer) + rvq_rest (7 of
//!      31 acoustic quantizers), each `input_proj` k1 512→256 (biasless);
//!      EuclideanCodebook argmin over `embedding_sum / clamp(cluster_usage,1e-5)`.
//!
//! CPU (Accelerate sgemm im2col convs): encoding is once-per-clip, so
//! determinism + gateability beat throughput. Mirrors the qwen3tts CPU encoder.

use super::config::*;
use crate::models::qwen3tts::cpu::{sgemm, sgemm_nt, softmax};
use crate::nn::weight_loader::{HostF32, WeightLoader};

/// Causal Conv1d on the CPU (im2col + sgemm). Left-pad `k−stride`, right-pad to
/// a whole number of output frames (moshi "extra padding"). Zeros except the
/// `downsample` conv (`pad_mode="replicate"`, edge values repeat).
///
/// Weights are consumed UNMODIFIED in the safetensors `[out, in, k]` row-major
/// flatten, so on an mmap-capable loader they are zero-copy pile views
/// ([`HostF32::Mapped`]) — no read/copy pass at load.
struct CpuConv {
    w: HostF32, // [out, in·k]
    b: Option<HostF32>,
    out: usize,
    inc: usize,
    k: usize,
    stride: usize,
    replicate_pad: bool,
}

impl CpuConv {
    fn load(loader: &WeightLoader, prefix: &str, stride: usize, bias: bool, replicate_pad: bool) -> Self {
        let (w, shape) = loader.load_host_f32(&format!("{prefix}.weight"));
        let (out, inc, k) = (shape[0], shape[1], shape[2]);
        Self {
            w,
            b: bias.then(|| loader.load_host_f32(&format!("{prefix}.bias")).0),
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
        let pad_left = k - s;
        let n_frames = ((l + pad_left).saturating_sub(k) as f64 / s as f64 + 1.0).ceil() as usize - 1;
        let ideal = n_frames * s + k - pad_left;
        let pad_right = ideal.saturating_sub(l);
        let lp = l + pad_left + pad_right;
        let t = (lp - k) / s + 1;

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

pub(super) struct TrLayer {
    ln1_w: HostF32,
    ln1_b: HostF32,
    ln2_w: HostF32,
    ln2_b: HostF32,
    in_proj: HostF32, // [3·512, 512] fused qkv
    out_proj: HostF32, // [512, 512]
    fc1: HostF32,      // [2048, 512]
    fc2: HostF32,      // [512, 2048]
    ls1: HostF32,
    ls2: HostF32,
}

/// One RVQ bank's encode side: input_proj + pre-divided codebooks (+ squared
/// row norms for the argmin). The codebooks/norms are DERIVED at load
/// (`embedding_sum / clamp(cluster_usage)` + row norms, ~17 MB/bank,
/// milliseconds) and stay computed; `input_proj` is consumed as shipped, so
/// it is a zero-copy pile view.
struct RvqEncoder {
    input_proj: HostF32, // [256, 512]
    codebooks: Vec<Vec<f32>>,
    norms: Vec<Vec<f32>>,
}

impl RvqEncoder {
    fn load(loader: &WeightLoader, prefix: &str, n_q: usize) -> Self {
        let mut codebooks = Vec::with_capacity(n_q);
        let mut norms = Vec::with_capacity(n_q);
        for i in 0..n_q {
            let (sum, _) = loader.load_f32(&format!("{prefix}.vq.layers.{i}._codebook.embedding_sum"));
            let (usage, _) = loader.load_f32(&format!("{prefix}.vq.layers.{i}._codebook.cluster_usage"));
            let mut cb = sum;
            for (r, &u) in usage.iter().enumerate() {
                let d = u.max(1e-5);
                for v in &mut cb[r * CODE_DIM..(r + 1) * CODE_DIM] {
                    *v /= d;
                }
            }
            norms.push(
                (0..usage.len())
                    .map(|r| cb[r * CODE_DIM..(r + 1) * CODE_DIM].iter().map(|&v| v * v).sum::<f32>())
                    .collect(),
            );
            codebooks.push(cb);
        }
        // input_proj weight is [256, 512, 1] (k1 conv) → flatten trailing 1.
        Self {
            input_proj: loader.load_host_f32(&format!("{prefix}.input_proj.weight")).0,
            codebooks,
            norms,
        }
    }

    /// Residual-VQ encode `x: [T, 512]` → per-quantizer codes `[n_q][T]`.
    fn encode(&self, x: &[f32], t: usize) -> Vec<Vec<u32>> {
        // input_proj (k1 conv ≡ matmul): [T, 512] → [T, 256]
        let mut residual = vec![0f32; t * CODE_DIM];
        sgemm_nt(x, &self.input_proj, t, HIDDEN, CODE_DIM, &mut residual);

        let mut out = Vec::with_capacity(self.codebooks.len());
        for (cb, norms) in self.codebooks.iter().zip(&self.norms) {
            let rows = norms.len();
            let mut dots = vec![0f32; t * rows];
            sgemm_nt(&residual, cb, t, CODE_DIM, rows, &mut dots);
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
                let e = &cb[best.1 * CODE_DIM..(best.1 + 1) * CODE_DIM];
                for (r, &ev) in residual[ti * CODE_DIM..(ti + 1) * CODE_DIM].iter_mut().zip(e) {
                    *r -= ev;
                }
            }
            out.push(codes);
        }
        out
    }
}

pub struct MimiEncoder {
    stem: CpuConv,
    blocks: Vec<EncBlock>,
    final_conv: CpuConv,
    tr_layers: Vec<TrLayer>,
    downsample: CpuConv,
    rvq_first: RvqEncoder,
    rvq_rest: RvqEncoder,
}

impl MimiEncoder {
    pub fn load(loader: &WeightLoader) -> Self {
        let p = "encoder.model";
        // per stage i: 3i+1 residual block (block.1, block.3), 3i+3 downsample;
        // 0 stem; 14 final conv.
        let blocks = ENC_RATIOS
            .iter()
            .enumerate()
            .map(|(i, &r)| EncBlock {
                res1: CpuConv::load(loader, &format!("{p}.{}.block.1.conv.conv", 3 * i + 1), 1, true, false),
                res2: CpuConv::load(loader, &format!("{p}.{}.block.3.conv.conv", 3 * i + 1), 1, true, false),
                down: CpuConv::load(loader, &format!("{p}.{}.conv.conv", 3 * i + 3), r, true, false),
            })
            .collect();
        let t = "encoder_transformer.transformer.layers";
        let tr_layers = (0..TR_LAYERS).map(|i| Self::load_layer(loader, t, i)).collect();
        Self {
            stem: CpuConv::load(loader, &format!("{p}.0.conv.conv"), 1, true, false),
            blocks,
            final_conv: CpuConv::load(loader, &format!("{p}.14.conv.conv"), 1, true, false),
            tr_layers,
            downsample: CpuConv::load(loader, "downsample.conv.conv.conv", 2, false, true),
            rvq_first: RvqEncoder::load(loader, "quantizer.rvq_first", 1),
            rvq_rest: RvqEncoder::load(loader, "quantizer.rvq_rest", N_ACOUSTIC),
        }
    }

    fn load_layer(loader: &WeightLoader, prefix: &str, i: usize) -> TrLayer {
        TrLayer {
            ln1_w: loader.load_host_f32(&format!("{prefix}.{i}.norm1.weight")).0,
            ln1_b: loader.load_host_f32(&format!("{prefix}.{i}.norm1.bias")).0,
            ln2_w: loader.load_host_f32(&format!("{prefix}.{i}.norm2.weight")).0,
            ln2_b: loader.load_host_f32(&format!("{prefix}.{i}.norm2.bias")).0,
            in_proj: loader.load_host_f32(&format!("{prefix}.{i}.self_attn.in_proj_weight")).0,
            out_proj: loader.load_host_f32(&format!("{prefix}.{i}.self_attn.out_proj.weight")).0,
            fc1: loader.load_host_f32(&format!("{prefix}.{i}.linear1.weight")).0,
            fc2: loader.load_host_f32(&format!("{prefix}.{i}.linear2.weight")).0,
            ls1: loader.load_host_f32(&format!("{prefix}.{i}.layer_scale_1.scale")).0,
            ls2: loader.load_host_f32(&format!("{prefix}.{i}.layer_scale_2.scale")).0,
        }
    }

    /// LayerNorm over the last dim of `x: [T, 512]`, biased, eps 1e-5.
    fn layer_norm(x: &[f32], w: &[f32], b: &[f32], out: &mut [f32]) {
        let d = w.len();
        for (row_in, row_out) in x.chunks_exact(d).zip(out.chunks_exact_mut(d)) {
            let mean = row_in.iter().map(|&v| v as f64).sum::<f64>() / d as f64;
            let var = row_in.iter().map(|&v| (v as f64 - mean).powi(2)).sum::<f64>() / d as f64;
            let inv = ((var + TR_EPS).sqrt().recip()) as f32;
            let mean = mean as f32;
            for i in 0..d {
                row_out[i] = (row_in[i] - mean) * inv * w[i] + b[i];
            }
        }
    }

    /// The 8-layer transformer over `h: [T, 512]`, in place. Shared with the
    /// decoder side via [`transformer_forward`].
    fn transformer(&self, h: &mut [f32], t: usize) {
        transformer_forward(&self.tr_layers, h, t);
    }

    /// Encode a 24 kHz mono waveform into `T×8` codes (codebook 0 = semantic).
    pub fn encode(&self, samples: &[f32]) -> Vec<[u32; NUM_CODEBOOKS]> {
        self.encode_stages(samples).3
    }

    /// Encode returning per-stage intermediates for parity gating: SEANet output
    /// `[C, T25]`, transformer output `[C, T25]`, downsample output `[C, T12.5]`,
    /// and the `T×8` codes. All conv-side tensors are `[channel-major, time]`.
    pub fn encode_stages(
        &self,
        samples: &[f32],
    ) -> (Vec<f32>, Vec<f32>, Vec<f32>, Vec<[u32; NUM_CODEBOOKS]>) {
        // moshi pads the wav to a whole number of frames before the SEANet
        // stack (`pad_for_conv1d`); the per-conv causal formula reproduces the
        // same total padding, so feeding the raw samples is equivalent.
        let (mut x, mut l) = self.stem.forward(samples, samples.len());
        for blk in &self.blocks {
            let mut h = x.clone();
            elu(&mut h);
            let (mut h, hl) = blk.res1.forward(&h, l);
            elu(&mut h);
            let (h, _hl2) = blk.res2.forward(&h, hl);
            for (xv, hv) in x.iter_mut().zip(&h) {
                *xv += hv;
            }
            elu(&mut x);
            let (nx, nl) = blk.down.forward(&x, l);
            x = nx;
            l = nl;
        }
        elu(&mut x);
        let (seanet, l) = self.final_conv.forward(&x, l); // [C, T25]

        // convs produced [C, T]; transformer works [T, C].
        let mut h = vec![0f32; l * HIDDEN];
        for c in 0..HIDDEN {
            for ti in 0..l {
                h[ti * HIDDEN + c] = seanet[c * l + ti];
            }
        }
        self.transformer(&mut h, l);
        let mut tr = vec![0f32; HIDDEN * l];
        for c in 0..HIDDEN {
            for ti in 0..l {
                tr[c * l + ti] = h[ti * HIDDEN + c];
            }
        }

        // ×2 downsample → 12.5 Hz, back to [T, C] for the quantizer.
        let (ds, t) = self.downsample.forward(&tr, l);
        let mut emb = vec![0f32; t * HIDDEN];
        for c in 0..HIDDEN {
            for ti in 0..t {
                emb[ti * HIDDEN + c] = ds[c * t + ti];
            }
        }

        let sem = self.rvq_first.encode(&emb, t);
        let ac = self.rvq_rest.encode(&emb, t);
        let codes = (0..t)
            .map(|ti| {
                let mut f = [0u32; NUM_CODEBOOKS];
                f[0] = sem[0][ti];
                for (qi, codes) in ac.iter().enumerate() {
                    f[qi + 1] = codes[ti];
                }
                f
            })
            .collect();
        (seanet, tr, ds, codes)
    }
}

/// Mimi transformer bottleneck (shared enc/dec): 8 layers, fused qkv `in_proj`,
/// RoPE θ=10000, causal sliding-window 250, GELU MLP, LayerScale. In place over
/// `h: [T, 512]`.
pub(super) fn transformer_forward(layers: &[TrLayer], h: &mut [f32], t: usize) {
    let (hd, nh, d) = (HIDDEN, TR_HEADS, TR_HEAD_DIM);
    let half = d / 2;
    let scale = ((d as f64).powf(-0.5)) as f32;
    let mut cos = vec![0f32; t * half];
    let mut sin = vec![0f32; t * half];
    for pos in 0..t {
        for i in 0..half {
            let r = pos as f64 * TR_ROPE_THETA.powf(-2.0 * i as f64 / d as f64);
            cos[pos * half + i] = r.cos() as f32;
            sin[pos * half + i] = r.sin() as f32;
        }
    }
    // moshi RoPE is INTERLEAVED (`interleave=True`): pair (x[2i], x[2i+1]),
    // not the split-half (x[i], x[i+half]).
    let rope = |x: &mut [f32], pos: usize| {
        for i in 0..half {
            let (c, s) = (cos[pos * half + i], sin[pos * half + i]);
            let (a, b) = (x[2 * i], x[2 * i + 1]);
            x[2 * i] = a * c - b * s;
            x[2 * i + 1] = a * s + b * c;
        }
    };

    let mut xin = vec![0f32; t * hd];
    let mut qkv = vec![0f32; t * 3 * hd];
    let mut attn = vec![0f32; t * hd];
    let mut proj = vec![0f32; t * hd];
    let mut ff = vec![0f32; t * TR_INTER];
    for l in layers {
        MimiEncoder::layer_norm(h, &l.ln1_w, &l.ln1_b, &mut xin);
        // fused qkv: [T, 512] @ [1536, 512]ᵀ → [T, 1536] laid out [q(512)|k|v]
        sgemm_nt(&xin, &l.in_proj, t, hd, 3 * hd, &mut qkv);
        // RoPE q and k per head
        for pos in 0..t {
            for hh in 0..nh {
                rope(&mut qkv[pos * 3 * hd + hh * d..pos * 3 * hd + (hh + 1) * d], pos);
                rope(&mut qkv[pos * 3 * hd + hd + hh * d..pos * 3 * hd + hd + (hh + 1) * d], pos);
            }
        }
        attn.fill(0.0);
        let mut scores = vec![0f32; t];
        for hh in 0..nh {
            for qp in 0..t {
                // causal sliding window: keys [max(0, qp+1-W) ..= qp]
                let lo = (qp + 1).saturating_sub(TR_WINDOW);
                let qrow = &qkv[qp * 3 * hd + hh * d..qp * 3 * hd + (hh + 1) * d];
                for (si, kp) in (lo..=qp).enumerate() {
                    let krow = &qkv[kp * 3 * hd + hd + hh * d..kp * 3 * hd + hd + (hh + 1) * d];
                    scores[si] = qrow.iter().zip(krow).map(|(&a, &b)| a * b).sum::<f32>() * scale;
                }
                let n = qp - lo + 1;
                softmax(&mut scores[..n]);
                let out = &mut attn[qp * hd + hh * d..qp * hd + (hh + 1) * d];
                for (si, kp) in (lo..=qp).enumerate() {
                    let vrow = &qkv[kp * 3 * hd + 2 * hd + hh * d..kp * 3 * hd + 2 * hd + (hh + 1) * d];
                    let p = scores[si];
                    for (o, &vv) in out.iter_mut().zip(vrow) {
                        *o += p * vv;
                    }
                }
            }
        }
        sgemm_nt(&attn, &l.out_proj, t, hd, hd, &mut proj);
        for pos in 0..t {
            for i in 0..hd {
                h[pos * hd + i] += proj[pos * hd + i] * l.ls1[i];
            }
        }
        MimiEncoder::layer_norm(h, &l.ln2_w, &l.ln2_b, &mut xin);
        sgemm_nt(&xin, &l.fc1, t, hd, TR_INTER, &mut ff);
        gelu(&mut ff);
        sgemm_nt(&ff, &l.fc2, t, TR_INTER, hd, &mut proj);
        for pos in 0..t {
            for i in 0..hd {
                h[pos * hd + i] += proj[pos * hd + i] * l.ls2[i];
            }
        }
    }
}

impl MimiEncoder {
    /// Load just the transformer layers (for the decoder's own bottleneck).
    pub(super) fn load_tr_layers(loader: &WeightLoader, prefix: &str) -> Vec<TrLayer> {
        (0..TR_LAYERS).map(|i| Self::load_layer(loader, prefix, i)).collect()
    }
}
