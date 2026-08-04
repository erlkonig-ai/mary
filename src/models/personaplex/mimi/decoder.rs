//! Mimi **decoder** (moshi `quantizer.*`, `upsample.*`, `decoder_transformer.*`,
//! `decoder.model.*`): `T×8` codes → 24 kHz mono waveform, 1920× upsample.
//!
//! Pipeline (moshi `MimiModel.decode`):
//!   1. `quantizer.decode`: rvq_first (semantic) + rvq_rest (7 acoustic), each
//!      EuclideanCodebook lookup (`embedding_sum / clamp(cluster_usage,1e-5)`),
//!      residual-summed per bank, then `output_proj` k1 256→512; the two banks
//!      are added → `[512, T]`.
//!   2. `upsample`: **depthwise** causal ConvTranspose1d 512→512 k4 stride 2
//!      (groups=512) → 25 Hz.
//!   3. `decoder_transformer`: the same 8-layer bottleneck as the encoder.
//!   4. SEANet decoder (`decoder.model.*`): conv 512→1024 k7, then per ratio
//!      r ∈ [8,6,5,4]: { ELU, ConvTranspose1d dim→dim/2 k=2r stride r (causal
//!      right-trim k−r), residual unit (ELU → conv k3 → ELU → conv k1, identity
//!      shortcut) }, then ELU + conv 64→1 k3.
//!
//! CPU (Accelerate sgemm), reusing the encoder's `CpuConv`/`transformer_forward`
//! for a single deterministic numeric path (parity-first Phase-1 port). A Burn
//! port is the throughput follow-up — see the module note.

use super::config::*;
use super::encoder::{transformer_forward, TrLayer as MimiTrLayer};
use crate::models::qwen3tts::cpu::{sgemm, sgemm_nt};
use crate::nn::weight_loader::{HostF32, WeightLoader};

/// Causal Conv1d (im2col + sgemm), zero left-pad `k−stride`. Depthwise when
/// `groups == inc` (each channel independent). Mirrors the encoder's CpuConv but
/// exposed here for the decoder's own convs. Weights consumed as shipped →
/// zero-copy pile views.
struct CpuConv {
    w: HostF32,
    b: Option<HostF32>,
    out: usize,
    inc: usize,
    k: usize,
    stride: usize,
}

impl CpuConv {
    fn load(loader: &WeightLoader, prefix: &str, stride: usize) -> Self {
        let (w, shape) = loader.load_host_f32(&format!("{prefix}.weight"));
        let (out, inc, k) = (shape[0], shape[1], shape[2]);
        Self {
            w,
            b: Some(loader.load_host_f32(&format!("{prefix}.bias")).0),
            out,
            inc,
            k,
            stride,
        }
    }

    /// `x: [in, L]` → `[out, T]`. Dense (`groups=1`).
    fn forward(&self, x: &[f32], l: usize) -> (Vec<f32>, usize) {
        let (k, s) = (self.k, self.stride);
        let pad_left = k - s;
        let n_frames =
            ((l + pad_left).saturating_sub(k) as f64 / s as f64 + 1.0).ceil() as usize - 1;
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

/// Causal ConvTranspose1d on the CPU. Weight `[in, out/groups, k]`, stride s,
/// k = 2·s here. Output length = L·s (causal right-trim of the `k−s` tail).
///
/// Dense (`groups == 1`) transconvs run as ONE sgemm + a col2im scatter-add
/// (`wt` is the `[out·k, in]` re-layout built at load): the SEANet upsample
/// stages are ~105 MMACs/frame, which the original scalar loop paid at
/// ~46 ms/frame — the whole realtime frame budget (measured 2026-07-11,
/// framebench). The depthwise `upsample` (groups == in) keeps the scalar
/// path — it is a few k·L MACs.
struct CpuTransConv {
    /// Raw `[in, out/groups, k]` as shipped → zero-copy pile view.
    w: HostF32,
    /// groups == 1: `[out·k, in]`, `wt[(oc·k+j)·in+ic] = w[ic,oc,j]` — a
    /// DERIVED re-layout for the one-sgemm dense path (~50 MB across the
    /// SEANet stages, built in milliseconds; stays computed at load).
    wt: Option<Vec<f32>>,
    b: Option<HostF32>, // [out]
    inc: usize,
    out: usize,
    k: usize,
    stride: usize,
    groups: usize,
}

impl CpuTransConv {
    fn load(loader: &WeightLoader, prefix: &str, stride: usize, groups: usize, bias: bool) -> Self {
        let (w, shape) = loader.load_host_f32(&format!("{prefix}.weight"));
        let (inc, opg, k) = (shape[0], shape[1], shape[2]);
        Self {
            wt: (groups == 1).then(|| {
                let out = opg * groups;
                let mut wt = vec![0f32; out * k * inc];
                for ic in 0..inc {
                    for oc in 0..out {
                        for j in 0..k {
                            wt[(oc * k + j) * inc + ic] = w[(ic * opg + oc) * k + j];
                        }
                    }
                }
                wt
            }),
            w,
            b: bias.then(|| loader.load_host_f32(&format!("{prefix}.bias")).0),
            inc,
            out: opg * groups,
            k,
            stride,
            groups,
        }
    }

    /// `x: [in, L]` → `[out, L·stride]` (causal). Full transposed length is
    /// `(L−1)·s + k`; moshi trims `padding_total = k − s` split all to the right
    /// (causal), leaving exactly `L·s`.
    fn forward(&self, x: &[f32], l: usize) -> (Vec<f32>, usize) {
        let (k, s) = (self.k, self.stride);
        let full = (l - 1) * s + k;
        let acc = match &self.wt {
            // dense: G[oc·k+j, ti] = Σ_ic wt[oc·k+j, ic]·x[ic, ti] in one
            // sgemm, then scatter-add G into the overlapping output taps
            Some(wt) => {
                let mut gbuf = vec![0f32; self.out * k * l];
                sgemm(wt, x, self.out * k, self.inc, l, &mut gbuf);
                let mut acc = vec![0f32; self.out * full];
                for oc in 0..self.out {
                    for j in 0..k {
                        let grow = &gbuf[(oc * k + j) * l..(oc * k + j) * l + l];
                        let arow = &mut acc[oc * full + j..];
                        for (ti, &gv) in grow.iter().enumerate() {
                            arow[ti * s] += gv;
                        }
                    }
                }
                acc
            }
            None => self.scatter_scalar(x, l),
        };
        // trim right, add bias
        let out_len = l * s; // trim (k - s) from the right
        let mut y = vec![0f32; self.out * out_len];
        for oc in 0..self.out {
            let src = &acc[oc * full..oc * full + out_len];
            let dst = &mut y[oc * out_len..(oc + 1) * out_len];
            let bo = self.b.as_ref().map_or(0.0, |b| b[oc]);
            for (d, &sv) in dst.iter_mut().zip(src) {
                *d = sv + bo;
            }
        }
        (y, out_len)
    }

    /// The reference scalar scatter (all groups); untrimmed `[out, full]`.
    fn scatter_scalar(&self, x: &[f32], l: usize) -> Vec<f32> {
        let (k, s, g) = (self.k, self.stride, self.groups);
        let full = (l - 1) * s + k;
        let ipg = self.inc / g; // in per group
        let opg = self.out / g; // out per group
        let mut acc = vec![0f32; self.out * full];
        // conv_transpose: out[oc, ti*s + j] += Σ_ic w[ic, oc_local, j] · x[ic, ti]
        for gi in 0..g {
            for ic_local in 0..ipg {
                let ic = gi * ipg + ic_local;
                let xrow = &x[ic * l..(ic + 1) * l];
                for oc_local in 0..opg {
                    let oc = gi * opg + oc_local;
                    let wbase = (ic * opg + oc_local) * k;
                    let arow_base = oc * full;
                    for ti in 0..l {
                        let xv = xrow[ti];
                        if xv == 0.0 {
                            continue;
                        }
                        let base = ti * s;
                        for j in 0..k {
                            acc[arow_base + base + j] += self.w[wbase + j] * xv;
                        }
                    }
                }
            }
        }
        acc
    }
}

fn elu(x: &mut [f32]) {
    for v in x.iter_mut() {
        if *v < 0.0 {
            *v = v.exp() - 1.0;
        }
    }
}

/// One RVQ bank decode side: pre-divided codebooks `[2048, 256]` (DERIVED at
/// load, stays computed) + output_proj (as shipped → zero-copy pile view).
struct RvqDecoder {
    codebooks: Vec<Vec<f32>>, // per q: [2048·256]
    output_proj: HostF32,     // [512, 256]
}

impl RvqDecoder {
    fn load(loader: &WeightLoader, prefix: &str, n_q: usize) -> Self {
        let codebooks = (0..n_q)
            .map(|i| {
                let (mut cb, _) =
                    loader.load_f32(&format!("{prefix}.vq.layers.{i}._codebook.embedding_sum"));
                let (usage, _) =
                    loader.load_f32(&format!("{prefix}.vq.layers.{i}._codebook.cluster_usage"));
                for (r, &u) in usage.iter().enumerate() {
                    let d = u.max(1e-5);
                    for v in &mut cb[r * CODE_DIM..(r + 1) * CODE_DIM] {
                        *v /= d;
                    }
                }
                cb
            })
            .collect();
        Self {
            codebooks,
            output_proj: loader
                .load_host_f32(&format!("{prefix}.output_proj.weight"))
                .0,
        }
    }

    /// `codes[q][t]` → `[512, T]` (output_proj already applied).
    fn decode(&self, codes: &[Vec<u32>], t: usize) -> Vec<f32> {
        // sum embeddings over quantizers → [T, 256]
        let mut emb = vec![0f32; t * CODE_DIM];
        for (q, row) in codes.iter().enumerate() {
            let cb = &self.codebooks[q];
            for (ti, &c) in row.iter().enumerate() {
                let src = &cb[c as usize * CODE_DIM..(c as usize + 1) * CODE_DIM];
                for (e, &sv) in emb[ti * CODE_DIM..(ti + 1) * CODE_DIM].iter_mut().zip(src) {
                    *e += sv;
                }
            }
        }
        // output_proj k1 conv (256→512): [T, 256] @ [512, 256]ᵀ → [T, 512], then
        // transpose to [512, T].
        let mut proj = vec![0f32; t * HIDDEN];
        sgemm_nt(&emb, &self.output_proj, t, CODE_DIM, HIDDEN, &mut proj);
        let mut out = vec![0f32; HIDDEN * t];
        for c in 0..HIDDEN {
            for ti in 0..t {
                out[c * t + ti] = proj[ti * HIDDEN + c];
            }
        }
        out
    }
}

/// SEANet decoder stage: ELU → upsample transconv → residual unit.
struct DecBlock {
    up: CpuTransConv, // dim → dim/2, k=2r stride r
    res1: CpuConv,    // dim/2 → dim/4, k3
    res2: CpuConv,    // dim/4 → dim/2, k1
}

pub struct MimiDecoder {
    rvq_first: RvqDecoder,
    rvq_rest: RvqDecoder,
    upsample: CpuTransConv, // depthwise 512→512 k4 s2
    tr_layers: Vec<MimiTrLayer>,
    head: CpuConv, // 512→1024 k7
    blocks: Vec<DecBlock>,
    tail_conv: CpuConv, // 64→1 k3
}

impl MimiDecoder {
    pub fn load(loader: &WeightLoader) -> Self {
        // decoder.model chain: 0 head conv; per stage i: 3i+2 convtr, 3i+3
        // resblock; 14 tail conv.
        let blocks = DEC_RATIOS
            .iter()
            .enumerate()
            .map(|(i, &r)| DecBlock {
                up: CpuTransConv::load(
                    loader,
                    &format!("decoder.model.{}.convtr.convtr", 3 * i + 2),
                    r,
                    1,
                    true,
                ),
                res1: CpuConv::load(
                    loader,
                    &format!("decoder.model.{}.block.1.conv.conv", 3 * i + 3),
                    1,
                ),
                res2: CpuConv::load(
                    loader,
                    &format!("decoder.model.{}.block.3.conv.conv", 3 * i + 3),
                    1,
                ),
            })
            .collect();
        Self {
            rvq_first: RvqDecoder::load(loader, "quantizer.rvq_first", 1),
            rvq_rest: RvqDecoder::load(loader, "quantizer.rvq_rest", N_ACOUSTIC),
            upsample: CpuTransConv::load(loader, "upsample.convtr.convtr.convtr", 2, HIDDEN, false),
            tr_layers: super::encoder::MimiEncoder::load_tr_layers(
                loader,
                "decoder_transformer.transformer.layers",
            ),
            head: CpuConv::load(loader, "decoder.model.0.conv.conv", 1),
            blocks,
            tail_conv: CpuConv::load(loader, "decoder.model.14.conv.conv", 1),
        }
    }

    /// `codes[t][q]` → `[512, T]` quantized latent (both banks summed).
    pub fn quantizer_decode(&self, codes: &[[u32; NUM_CODEBOOKS]]) -> (Vec<f32>, usize) {
        let t = codes.len();
        let sem: Vec<Vec<u32>> = vec![codes.iter().map(|f| f[0]).collect()];
        let ac: Vec<Vec<u32>> = (1..NUM_CODEBOOKS)
            .map(|q| codes.iter().map(|f| f[q]).collect())
            .collect();
        let a = self.rvq_first.decode(&sem, t);
        let b = self.rvq_rest.decode(&ac, t);
        let mut out = a;
        for (o, bv) in out.iter_mut().zip(&b) {
            *o += bv;
        }
        (out, t)
    }

    /// Full decode: `T×8` codes → waveform samples (clamped [−1, 1]).
    pub fn decode(&self, codes: &[[u32; NUM_CODEBOOKS]]) -> Vec<f32> {
        let (q, t) = self.quantizer_decode(codes);

        // upsample 512→512 k4 s2 depthwise → 25 Hz.
        let (up, ul) = self.upsample.forward(&q, t);

        // decoder_transformer works [T, C]; up is [C, T].
        let mut h = vec![0f32; ul * HIDDEN];
        for c in 0..HIDDEN {
            for ti in 0..ul {
                h[ti * HIDDEN + c] = up[c * ul + ti];
            }
        }
        transformer_forward(&self.tr_layers, &mut h, ul);
        let mut x = vec![0f32; HIDDEN * ul];
        for c in 0..HIDDEN {
            for ti in 0..ul {
                x[c * ul + ti] = h[ti * HIDDEN + c];
            }
        }

        // SEANet decoder: head conv, then per stage ELU→convtr→resunit.
        let (mut x, mut l) = self.head.forward(&x, ul);
        for blk in &self.blocks {
            elu(&mut x);
            let (up, ul) = blk.up.forward(&x, l);
            // residual unit (identity shortcut)
            let mut h = up.clone();
            elu(&mut h);
            let (mut h, hl) = blk.res1.forward(&h, ul);
            elu(&mut h);
            let (h, _hl2) = blk.res2.forward(&h, hl);
            x = up;
            for (xv, hv) in x.iter_mut().zip(&h) {
                *xv += hv;
            }
            l = ul;
        }
        elu(&mut x);
        let (w, wl) = self.tail_conv.forward(&x, l);
        // single channel, clamp to [−1, 1]
        let mut out = w[..wl].to_vec();
        for v in &mut out {
            *v = v.clamp(-1.0, 1.0);
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The sgemm+col2im dense path must reproduce the reference scalar
    /// scatter (fp-association differences only) — SEANet-stage-shaped case.
    #[test]
    fn transconv_gemm_matches_scalar() {
        let (inc, out, k, s, l) = (16usize, 8usize, 16usize, 8usize, 7usize);
        let val = |i: usize| ((i * 2654435761 % 1000) as f32 / 500.0) - 1.0;
        let w: Vec<f32> = (0..inc * out * k).map(val).collect();
        let b: Vec<f32> = (0..out).map(|i| val(i + 13)).collect();
        let x: Vec<f32> = (0..inc * l).map(|i| val(i + 31)).collect();

        let mut c = CpuTransConv {
            wt: None,
            w: HostF32::Owned(w),
            b: Some(HostF32::Owned(b)),
            inc,
            out,
            k,
            stride: s,
            groups: 1,
        };
        let (y_scalar, l_scalar) = c.forward(&x, l);
        c.wt = Some({
            let mut wt = vec![0f32; out * k * inc];
            for ic in 0..inc {
                for oc in 0..out {
                    for j in 0..k {
                        wt[(oc * k + j) * inc + ic] = c.w[(ic * out + oc) * k + j];
                    }
                }
            }
            wt
        });
        let (y_gemm, l_gemm) = c.forward(&x, l);

        assert_eq!(l_scalar, l * s);
        assert_eq!(l_gemm, l_scalar);
        assert_eq!(y_gemm.len(), y_scalar.len());
        for (i, (a, g)) in y_scalar.iter().zip(&y_gemm).enumerate() {
            assert!((a - g).abs() <= 1e-4, "tap {i}: scalar {a} vs gemm {g}");
        }
    }
}
