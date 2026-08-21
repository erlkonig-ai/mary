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
//! Two paths over the same graph:
//!
//! - [`MimiDecoder`] — CPU (Accelerate sgemm), reusing the encoder's
//!   `CpuConv`/`transformer_forward`. It is the parity oracle: `mimi_probe`
//!   gates it against the moshi CPU-f32 goldens, and the GPU path is gated
//!   against IT.
//! - [`MimiDecoderGpu`] (feature `q4`) — hand-launched cubecl kernels on the
//!   raw wgpu/Metal device, or CUDA under `cuda-backend`. Everything is f32;
//!   see its docs for the exact numerics difference and the dispatch shape.

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

// ===========================================================================
// GPU decoder (cubecl) — one source, Metal (wgpu) and CUDA
// ===========================================================================

#[cfg(feature = "q4")]
pub use gpu::MimiDecoderGpu;

/// The Mimi decoder on the GPU: the same graph as [`MimiDecoder`], rebuilt as
/// hand-launched cubecl kernels on the raw wgpu/Metal (or CUDA, under
/// `cuda-backend`) device — the same lane `temporal_metal` runs on, so decode
/// can share one client and one queue with the LM
/// ([`MimiDecoderGpu::load_on`]) instead of contending for cores with it.
/// One source, both backends: the `mimi_gpu_synthetic_matches_cpu` gate runs
/// on either without a weight pile, which is how the CUDA side is checked on
/// a box that has no 34 GB PersonaPlex pile.
///
/// ## What is exact and what is not
///
/// Everything is f32 — no quantization anywhere — so the only numeric
/// difference from the CPU reference is **summation order** plus two
/// intrinsics:
///
/// - reductions are tree reductions across 32 lanes (or 64 threads in
///   attention) rather than the CPU's strictly ascending single accumulator;
/// - LayerNorm accumulates the mean and the variance in f32 across a shared
///   tree, where the CPU path uses f64 scalars;
/// - GELU calls the device `erf` in f32, where the CPU calls `libm::erf` in
///   f64.
///
/// None of that is bit-exact and it is not meant to be: the port is judged on
/// waveform agreement (see the `gpu_matches_cpu_*` tests), not on bits.
///
/// ## Dispatch shape (per `decode` call, T frames → 1920·T samples)
///
/// 3 quantizer + upsample + transpose, an entry LayerNorm, 8 per transformer
/// layer (×8), a transpose back, the SEANet head conv, 4 per SEANet stage
/// (×4), and the tail conv = **91 dispatches**, one blocking readback of the
/// waveform. Every stage is shaped so the whole T batch rides one launch;
/// the linear kernel walks token PAIRS so a layer's weights are streamed once
/// per two tokens rather than once per token.
#[cfg(feature = "q4")]
mod gpu {
    use super::{CpuConv, CpuTransConv, RvqDecoder};
    use crate::models::personaplex::mimi::config::*;
    use crate::nn::q4::{self, Rt};
    use crate::nn::weight_loader::WeightLoader;
    use cubecl::prelude::*;
    use cubecl::server::Handle;

    /// Threads cooperating on one output element (one Apple simdgroup / one
    /// CUDA warp).
    const LANES: u32 = 32;
    /// Output rows per cube when the output channel count allows it.
    const ROWS: u32 = 8;
    /// Shared-memory cap for one query's attention scores — the transformer's
    /// causal window is [`TR_WINDOW`] (250), rounded up to a power of two.
    const SCORE_CAP: u32 = 256;
    /// RoPE table depth in transformer positions (= 2 per Mimi frame), i.e.
    /// a 4096-frame (5.5 min) batch decode. Asserted, never wrapped.
    const MAX_POS: usize = 8192;

    fn as_bytes<T>(v: &[T]) -> &[u8] {
        unsafe { core::slice::from_raw_parts(v.as_ptr() as *const u8, std::mem::size_of_val(v)) }
    }

    // -----------------------------------------------------------------------
    // kernels
    // -----------------------------------------------------------------------

    /// RVQ bank gather: `emb[t, d] = Σ_q codebook_q[codes[q, t], d]`, with the
    /// codebooks already divided by their cluster usage at load. One thread
    /// per `(t, d)`.
    #[cube(launch_unchecked)]
    fn rvq_gather_kernel(
        codes: &Array<u32>,
        cb: &Array<f32>,
        emb: &mut Array<f32>,
        t_len: u32,
        #[comptime] n_q: u32,
        #[comptime] dim: u32,
        #[comptime] cb_size: u32,
    ) {
        let idx = ABSOLUTE_POS as u32;
        if idx < t_len * dim {
            let ti = idx / dim;
            let d = idx % dim;
            let mut acc = f32::new(0.0);
            for q in 0..n_q {
                let c = codes[(q * t_len + ti) as usize];
                acc += cb[(q * cb_size * dim + c * dim + d) as usize];
            }
            emb[idx as usize] = acc;
        }
    }

    /// The two banks' `output_proj` k1 convs, summed:
    /// `y[oc, t] = Σ_d w_sem[oc, d]·emb_sem[t, d] + w_ac[oc, d]·emb_ac[t, d]`.
    /// Output is channel-major `[512, T]`, the layout the SEANet side wants.
    #[cube(launch_unchecked)]
    fn rvq_proj_kernel(
        emb_sem: &Array<f32>,
        emb_ac: &Array<f32>,
        w_sem: &Array<f32>,
        w_ac: &Array<f32>,
        y: &mut Array<f32>,
        t_len: u32,
        #[comptime] dim: u32,
        #[comptime] rows: u32,
        #[comptime] lanes: u32,
    ) {
        let u = UNIT_POS_X;
        let lane = u % lanes;
        let oc = CUBE_POS_Y * rows + u / lanes;
        let ti = CUBE_POS_X;
        let mut red = SharedMemory::<f32>::new(comptime!((rows * lanes) as usize));

        let mut acc = f32::new(0.0);
        let mut d = lane;
        while d < dim {
            acc += w_sem[(oc * dim + d) as usize] * emb_sem[(ti * dim + d) as usize];
            acc += w_ac[(oc * dim + d) as usize] * emb_ac[(ti * dim + d) as usize];
            d += lanes;
        }
        red[u as usize] = acc;
        sync_cube();
        let mut stride = u32::new((lanes / 2) as i64);
        while stride > 0 {
            if lane < stride {
                red[u as usize] = red[u as usize] + red[(u + stride) as usize];
            }
            sync_cube();
            stride /= 2;
        }
        if lane == 0 {
            y[(oc * t_len + ti) as usize] = red[(u - lane) as usize];
        }
    }

    /// DEPTHWISE causal ConvTranspose1d, `k = 2·s`, no bias — the `upsample`
    /// stage (512→512 k4 s2, groups=512). `k = 2s` means exactly two taps
    /// land on each output position: `j ∈ {o mod s, (o mod s) + s}` with
    /// `ti = ⌊o/s⌋ − n`. The causal right-trim is implicit: the launch only
    /// covers `o < L·s`.
    #[cube(launch_unchecked)]
    fn dw_transconv_kernel(
        x: &Array<f32>,
        w: &Array<f32>,
        y: &mut Array<f32>,
        l: u32,
        channels: u32,
        #[comptime] k: u32,
        #[comptime] s: u32,
    ) {
        let idx = ABSOLUTE_POS as u32;
        let out_len = l * s;
        if idx < channels * out_len {
            let c = idx / out_len;
            let o = idx % out_len;
            let j0 = o % s;
            let ti0 = o / s;
            let mut acc = w[(c * k + j0) as usize] * x[(c * l + ti0) as usize];
            if ti0 >= 1 {
                acc += w[(c * k + j0 + s) as usize] * x[(c * l + ti0 - 1) as usize];
            }
            y[idx as usize] = acc;
        }
    }

    /// Dense causal ConvTranspose1d, `k = 2·s` (every SEANet upsample stage).
    /// Weights arrive in the `wt[(oc·k + j)·inc + ic]` re-layout the CPU path
    /// already builds at load, so the 32 lanes of a row group sweep `ic`
    /// consecutively. `elu_in` applies ELU to the input as it is read — every
    /// SEANet stage opens with one, so it costs no dispatch of its own.
    #[cube(launch_unchecked)]
    fn transconv_kernel(
        x: &Array<f32>,
        wt: &Array<f32>,
        b: &Array<f32>,
        y: &mut Array<f32>,
        l: u32,
        inc: u32,
        #[comptime] k: u32,
        #[comptime] s: u32,
        #[comptime] rows: u32,
        #[comptime] lanes: u32,
        #[comptime] elu_in: bool,
    ) {
        let u = UNIT_POS_X;
        let lane = u % lanes;
        let oc = CUBE_POS_Y * rows + u / lanes;
        let o = CUBE_POS_X;
        let j0 = o % s;
        let ti0 = o / s;
        let mut red = SharedMemory::<f32>::new(comptime!((rows * lanes) as usize));

        let base0 = (oc * k + j0) * inc;
        let base1 = (oc * k + j0 + s) * inc;
        let mut acc = f32::new(0.0);
        let mut ic = lane;
        if ti0 >= 1 {
            while ic < inc {
                let mut x0 = x[(ic * l + ti0) as usize];
                let mut x1 = x[(ic * l + ti0 - 1) as usize];
                if elu_in {
                    if x0 < 0.0 {
                        x0 = x0.exp() - 1.0;
                    }
                    if x1 < 0.0 {
                        x1 = x1.exp() - 1.0;
                    }
                }
                acc += wt[(base0 + ic) as usize] * x0;
                acc += wt[(base1 + ic) as usize] * x1;
                ic += lanes;
            }
        } else {
            while ic < inc {
                let mut x0 = x[(ic * l + ti0) as usize];
                if elu_in {
                    if x0 < 0.0 {
                        x0 = x0.exp() - 1.0;
                    }
                }
                acc += wt[(base0 + ic) as usize] * x0;
                ic += lanes;
            }
        }
        red[u as usize] = acc;
        sync_cube();
        let mut stride = u32::new((lanes / 2) as i64);
        while stride > 0 {
            if lane < stride {
                red[u as usize] = red[u as usize] + red[(u + stride) as usize];
            }
            sync_cube();
            stride /= 2;
        }
        if lane == 0 {
            y[(oc * (l * s) + o) as usize] = red[(u - lane) as usize] + b[oc as usize];
        }
    }

    /// Causal Conv1d, **stride 1** (every conv in the decoder is stride 1):
    /// left-pad `k−1`, output length `L`. Weights are `[out, inc, k]` as
    /// shipped, so the lanes sweep the flattened `(c, j)` reduction axis
    /// consecutively. `elu_in` applies ELU to the INPUT as it is read (the
    /// SEANet stages always precede a conv with one, so it costs no
    /// dispatch); `clamp_out` clamps the result to [−1, 1] (the tail conv).
    #[cube(launch_unchecked)]
    fn conv1d_kernel(
        x: &Array<f32>,
        w: &Array<f32>,
        b: &Array<f32>,
        y: &mut Array<f32>,
        l: u32,
        inc: u32,
        #[comptime] k: u32,
        #[comptime] rows: u32,
        #[comptime] lanes: u32,
        #[comptime] elu_in: bool,
        #[comptime] clamp_out: bool,
    ) {
        let u = UNIT_POS_X;
        let lane = u % lanes;
        let oc = CUBE_POS_Y * rows + u / lanes;
        let ti = CUBE_POS_X;
        let mut red = SharedMemory::<f32>::new(comptime!((rows * lanes) as usize));

        let total = inc * k;
        let mut acc = f32::new(0.0);
        let mut idx = lane;
        while idx < total {
            let c = idx / k;
            let j = idx % k;
            // stride 1, left pad k−1: src = ti + j − (k−1)
            if ti + j + 1 >= k {
                let src = ti + j + 1 - k;
                let mut xv = x[(c * l + src) as usize];
                if elu_in {
                    if xv < 0.0 {
                        xv = xv.exp() - 1.0;
                    }
                }
                acc += w[(oc * total + idx) as usize] * xv;
            }
            idx += lanes;
        }
        red[u as usize] = acc;
        sync_cube();
        let mut stride = u32::new((lanes / 2) as i64);
        while stride > 0 {
            if lane < stride {
                red[u as usize] = red[u as usize] + red[(u + stride) as usize];
            }
            sync_cube();
            stride /= 2;
        }
        if lane == 0 {
            let mut v = red[(u - lane) as usize] + b[oc as usize];
            if clamp_out {
                if v > 1.0 {
                    v = f32::new(1.0);
                }
                if v < -1.0 {
                    v = f32::new(-1.0);
                }
            }
            y[(oc * l + ti) as usize] = v;
        }
    }

    /// `y[c, t] = x[t, c]` (transformer row-major → SEANet channel-major) or
    /// its inverse, selected by `to_channel_major`.
    #[cube(launch_unchecked)]
    fn transpose_kernel(
        x: &Array<f32>,
        y: &mut Array<f32>,
        l: u32,
        #[comptime] channels: u32,
        #[comptime] to_channel_major: bool,
    ) {
        let idx = ABSOLUTE_POS as u32;
        if idx < l * channels {
            if to_channel_major {
                let ti = idx / channels;
                let c = idx % channels;
                y[(c * l + ti) as usize] = x[idx as usize];
            } else {
                let ti = idx / channels;
                let c = idx % channels;
                y[idx as usize] = x[(c * l + ti) as usize];
            }
        }
    }

    /// Biased LayerNorm over the last dim of `x: [T, d]`, eps [`TR_EPS`].
    /// Two passes (mean, then Σ(v−mean)²) rather than the E[x²]−E[x]²
    /// shortcut: the shortcut cancels catastrophically in f32 at this width.
    #[cube(launch_unchecked)]
    fn layernorm_kernel(
        x: &Array<f32>,
        w: &Array<f32>,
        b: &Array<f32>,
        y: &mut Array<f32>,
        eps: f32,
        #[comptime] d: u32,
        #[comptime] cube_dim: u32,
    ) {
        let i = UNIT_POS_X;
        let t = CUBE_POS_X;
        let row = t * d;
        let mut red = SharedMemory::<f32>::new(comptime!(cube_dim as usize));

        let mut acc = f32::new(0.0);
        let mut k = i;
        while k < d {
            acc += x[(row + k) as usize];
            k += cube_dim;
        }
        red[i as usize] = acc;
        sync_cube();
        let mut stride = u32::new((cube_dim / 2) as i64);
        while stride > 0 {
            if i < stride {
                red[i as usize] = red[i as usize] + red[(i + stride) as usize];
            }
            sync_cube();
            stride /= 2;
        }
        let mean = red[0] / (d as f32);
        sync_cube();

        let mut acc2 = f32::new(0.0);
        let mut k = i;
        while k < d {
            let dv = x[(row + k) as usize] - mean;
            acc2 += dv * dv;
            k += cube_dim;
        }
        red[i as usize] = acc2;
        sync_cube();
        let mut stride = u32::new((cube_dim / 2) as i64);
        while stride > 0 {
            if i < stride {
                red[i as usize] = red[i as usize] + red[(i + stride) as usize];
            }
            sync_cube();
            stride /= 2;
        }
        let inv = 1.0 / (red[0] / (d as f32) + eps).sqrt();
        sync_cube();

        let mut k = i;
        while k < d {
            y[(row + k) as usize] = (x[(row + k) as usize] - mean) * inv * w[k as usize]
                + b[k as usize];
            k += cube_dim;
        }
    }

    /// Bias-free linear over a sequence: `y[t, o] = Σ_i x[t, i]·w[o, i]`, with
    /// an optional exact-erf GELU epilogue. Each 32-lane group owns one output
    /// row and walks the sequence in token PAIRS (the sequence length is
    /// always even — 2 per Mimi frame), so a weight row is fetched once per
    /// two tokens instead of once per token: at the streaming shape (T=1,
    /// n=2) that halves the transformer's whole weight traffic, and at batch
    /// shapes it divides it by n/2.
    #[cube(launch_unchecked)]
    fn linear_kernel(
        x: &Array<f32>,
        w: &Array<f32>,
        y: &mut Array<f32>,
        n: u32,
        #[comptime] in_dim: u32,
        #[comptime] out_dim: u32,
        #[comptime] rows: u32,
        #[comptime] lanes: u32,
        #[comptime] gelu: bool,
    ) {
        let u = UNIT_POS_X;
        let lane = u % lanes;
        let oc = CUBE_POS_X * rows + u / lanes;
        let mut red0 = SharedMemory::<f32>::new(comptime!((rows * lanes) as usize));
        let mut red1 = SharedMemory::<f32>::new(comptime!((rows * lanes) as usize));
        let wbase = oc * in_dim;

        let mut t = u32::new(0);
        while t < n {
            let x0 = t * in_dim;
            let x1 = x0 + in_dim;
            let mut a0 = f32::new(0.0);
            let mut a1 = f32::new(0.0);
            let mut i = lane;
            while i < in_dim {
                let wv = w[(wbase + i) as usize];
                a0 += wv * x[(x0 + i) as usize];
                a1 += wv * x[(x1 + i) as usize];
                i += lanes;
            }
            sync_cube();
            red0[u as usize] = a0;
            red1[u as usize] = a1;
            sync_cube();
            let mut stride = u32::new((lanes / 2) as i64);
            while stride > 0 {
                if lane < stride {
                    red0[u as usize] = red0[u as usize] + red0[(u + stride) as usize];
                    red1[u as usize] = red1[u as usize] + red1[(u + stride) as usize];
                }
                sync_cube();
                stride /= 2;
            }
            if lane == 0 {
                let base = u - lane;
                let mut v0 = red0[base as usize];
                let mut v1 = red1[base as usize];
                if gelu {
                    v0 = 0.5 * v0 * (1.0 + Erf::erf(v0 * 0.70710678));
                    v1 = 0.5 * v1 * (1.0 + Erf::erf(v1 * 0.70710678));
                }
                y[(t * out_dim + oc) as usize] = v0;
                y[((t + 1) * out_dim + oc) as usize] = v1;
            }
            t += 2;
        }
    }

    /// Interleaved RoPE on the fused `qkv: [n, 3·512]` buffer — moshi rotates
    /// the PAIR `(x[2i], x[2i+1])` per head, not the split halves. One thread
    /// per `(pos, head, i)`; it rotates q and k at that index together.
    #[cube(launch_unchecked)]
    fn rope_kernel(
        qkv: &mut Array<f32>,
        cos: &Array<f32>,
        sin: &Array<f32>,
        n: u32,
        #[comptime] hidden: u32,
        #[comptime] d: u32,
        #[comptime] half: u32,
    ) {
        let idx = ABSOLUTE_POS as u32;
        let pairs = hidden / 2;
        if idx < n * pairs {
            let pos = idx / pairs;
            let p = idx % pairs;
            let head = p / half;
            let i = p % half;
            let c = cos[(pos * half + i) as usize];
            let s = sin[(pos * half + i) as usize];
            let qb = pos * 3 * hidden + head * d + 2 * i;
            let a = qkv[qb as usize];
            let b = qkv[(qb + 1) as usize];
            qkv[qb as usize] = a * c - b * s;
            qkv[(qb + 1) as usize] = a * s + b * c;
            let kb = qb + hidden;
            let a = qkv[kb as usize];
            let b = qkv[(kb + 1) as usize];
            qkv[kb as usize] = a * c - b * s;
            qkv[(kb + 1) as usize] = a * s + b * c;
        }
    }

    /// Causal sliding-window attention over the whole sequence: one cube per
    /// `(query position, head)`, `d` threads. Keys are `[max(0, qp+1−W) ..=
    /// qp]` — moshi's `causal=True, context=250`.
    #[cube(launch_unchecked)]
    fn attn_kernel(
        qkv: &Array<f32>,
        out: &mut Array<f32>,
        scale: f32,
        #[comptime] hidden: u32,
        #[comptime] d: u32,
        #[comptime] window: u32,
        #[comptime] score_cap: u32,
    ) {
        let i = UNIT_POS_X;
        let qp = CUBE_POS_X;
        let h = CUBE_POS_Y;
        let mut lo = u32::new(0);
        if qp + 1 > window {
            lo = qp + 1 - window;
        }
        let cnt = qp - lo + 1;

        let mut qsh = SharedMemory::<f32>::new(comptime!(d as usize));
        let mut red = SharedMemory::<f32>::new(comptime!(d as usize));
        let mut scores = SharedMemory::<f32>::new(comptime!(score_cap as usize));

        qsh[i as usize] = qkv[(qp * 3 * hidden + h * d + i) as usize];
        sync_cube();

        let mut si = i;
        while si < cnt {
            let kbase = (lo + si) * 3 * hidden + hidden + h * d;
            let mut acc = f32::new(0.0);
            for dd in 0..d {
                acc += qsh[dd as usize] * qkv[(kbase + dd) as usize];
            }
            scores[si as usize] = acc * scale;
            si += d;
        }
        sync_cube();

        let mut m = f32::new(-3.40282e38);
        let mut si = i;
        while si < cnt {
            let sv = scores[si as usize];
            if sv > m {
                m = sv;
            }
            si += d;
        }
        red[i as usize] = m;
        sync_cube();
        let mut stride = u32::new((d / 2) as i64);
        while stride > 0 {
            if i < stride {
                let other = red[(i + stride) as usize];
                if other > red[i as usize] {
                    red[i as usize] = other;
                }
            }
            sync_cube();
            stride /= 2;
        }
        let mx = red[0];
        sync_cube();

        let mut sum = f32::new(0.0);
        let mut si = i;
        while si < cnt {
            let p = (scores[si as usize] - mx).exp();
            scores[si as usize] = p;
            sum += p;
            si += d;
        }
        red[i as usize] = sum;
        sync_cube();
        let mut stride = u32::new((d / 2) as i64);
        while stride > 0 {
            if i < stride {
                red[i as usize] = red[i as usize] + red[(i + stride) as usize];
            }
            sync_cube();
            stride /= 2;
        }
        let total = red[0];

        let mut acc = f32::new(0.0);
        let mut si = u32::new(0);
        while si < cnt {
            acc += scores[si as usize]
                * qkv[((lo + si) * 3 * hidden + 2 * hidden + h * d + i) as usize];
            si += 1;
        }
        out[(qp * hidden + h * d + i) as usize] = acc / total;
    }

    /// LayerScale residual fused with the CONSUMING LayerNorm:
    /// `h += delta·ls; y = LN(h)·w + b`. One cube per token, each thread
    /// owning the same element subset for the add and both reductions, so
    /// there is no cross-thread hazard between them — the same shape
    /// `temporal_metal`'s `add_rms_kernel` uses. `ls_w`/`ls_b` are the norm
    /// that consumes the residual: `norm2` after attention, the NEXT layer's
    /// `norm1` after the MLP.
    #[cube(launch_unchecked)]
    fn ls_add_ln_kernel(
        h: &mut Array<f32>,
        delta: &Array<f32>,
        ls: &Array<f32>,
        w: &Array<f32>,
        b: &Array<f32>,
        y: &mut Array<f32>,
        eps: f32,
        #[comptime] d: u32,
        #[comptime] cube_dim: u32,
    ) {
        let i = UNIT_POS_X;
        let t = CUBE_POS_X;
        let row = t * d;
        let mut red = SharedMemory::<f32>::new(comptime!(cube_dim as usize));

        let mut acc = f32::new(0.0);
        let mut k = i;
        while k < d {
            let v = h[(row + k) as usize] + delta[(row + k) as usize] * ls[k as usize];
            h[(row + k) as usize] = v;
            acc += v;
            k += cube_dim;
        }
        red[i as usize] = acc;
        sync_cube();
        let mut stride = u32::new((cube_dim / 2) as i64);
        while stride > 0 {
            if i < stride {
                red[i as usize] = red[i as usize] + red[(i + stride) as usize];
            }
            sync_cube();
            stride /= 2;
        }
        let mean = red[0] / (d as f32);
        sync_cube();

        let mut acc2 = f32::new(0.0);
        let mut k = i;
        while k < d {
            let dv = h[(row + k) as usize] - mean;
            acc2 += dv * dv;
            k += cube_dim;
        }
        red[i as usize] = acc2;
        sync_cube();
        let mut stride = u32::new((cube_dim / 2) as i64);
        while stride > 0 {
            if i < stride {
                red[i as usize] = red[i as usize] + red[(i + stride) as usize];
            }
            sync_cube();
            stride /= 2;
        }
        let inv = 1.0 / (red[0] / (d as f32) + eps).sqrt();
        sync_cube();

        let mut k = i;
        while k < d {
            y[(row + k) as usize] =
                (h[(row + k) as usize] - mean) * inv * w[k as usize] + b[k as usize];
            k += cube_dim;
        }
    }

    /// LayerScale residual with no consuming norm — the last layer's MLP add.
    #[cube(launch_unchecked)]
    fn layerscale_add_kernel(
        h: &mut Array<f32>,
        delta: &Array<f32>,
        ls: &Array<f32>,
        n: u32,
        #[comptime] hidden: u32,
    ) {
        let idx = ABSOLUTE_POS as u32;
        if idx < n * hidden {
            let i = idx % hidden;
            h[idx as usize] = h[idx as usize] + delta[idx as usize] * ls[i as usize];
        }
    }

    /// `y[i] = x[i] + h[i]` — the SEANet residual unit's identity shortcut.
    #[cube(launch_unchecked)]
    fn add_kernel(x: &mut Array<f32>, h: &Array<f32>, n: u32) {
        let idx = ABSOLUTE_POS as u32;
        if idx < n {
            x[idx as usize] = x[idx as usize] + h[idx as usize];
        }
    }

    // -----------------------------------------------------------------------
    // host side
    // -----------------------------------------------------------------------

    struct GpuConv {
        w: Handle,
        b: Handle,
        out: usize,
        inc: usize,
        k: usize,
    }

    impl GpuConv {
        fn upload(client: &q4::Client, c: &CpuConv) -> Self {
            assert_eq!(c.stride, 1, "decoder convs are all stride 1");
            let b = c.b.as_ref().expect("decoder convs are biased");
            Self {
                w: client.create_from_slice(as_bytes(&c.w[..])),
                b: client.create_from_slice(as_bytes(&b[..])),
                out: c.out,
                inc: c.inc,
                k: c.k,
            }
        }
    }

    struct GpuTransConv {
        wt: Handle,
        b: Handle,
        out: usize,
        inc: usize,
        k: usize,
        stride: usize,
    }

    impl GpuTransConv {
        fn upload(client: &q4::Client, c: &CpuTransConv) -> Self {
            assert_eq!(c.groups, 1, "dense transconv only");
            assert_eq!(c.k, 2 * c.stride, "SEANet transconvs are k = 2·stride");
            let wt = c.wt.as_ref().expect("dense transconv keeps the [out·k, in] re-layout");
            let b = c.b.as_ref().expect("SEANet transconvs are biased");
            Self {
                wt: client.create_from_slice(as_bytes(&wt[..])),
                b: client.create_from_slice(as_bytes(&b[..])),
                out: c.out,
                inc: c.inc,
                k: c.k,
                stride: c.stride,
            }
        }
    }

    struct GpuBlock {
        up: GpuTransConv,
        res1: GpuConv,
        res2: GpuConv,
    }

    struct GpuTrLayer {
        ln1_w: Handle,
        ln1_b: Handle,
        ln2_w: Handle,
        ln2_b: Handle,
        in_proj: Handle,
        out_proj: Handle,
        fc1: Handle,
        fc2: Handle,
        ls1: Handle,
        ls2: Handle,
    }

    /// See the module docs on [`MimiDecoderGpu`].
    pub struct MimiDecoderGpu {
        client: q4::Client,
        cb_sem: Handle,
        cb_ac: Handle,
        proj_sem: Handle,
        proj_ac: Handle,
        upsample_w: Handle,
        cos: Handle,
        sin: Handle,
        tr: Vec<GpuTrLayer>,
        head: GpuConv,
        blocks: Vec<GpuBlock>,
        tail: GpuConv,
    }

    impl MimiDecoderGpu {
        /// Load straight to the device. Host weights are materialised one
        /// tensor at a time (through the same `CpuConv`/`CpuTransConv`/
        /// `RvqDecoder` loaders the CPU decoder uses, so the derived layouts
        /// — usage-divided codebooks, the `[out·k, in]` transconv re-layout —
        /// are shared code) and dropped as soon as they are uploaded.
        pub fn load(loader: &WeightLoader) -> Self {
            Self::load_on(loader, q4::client_for_default_device())
        }

        /// Load onto an EXISTING client — the point of the port: decode
        /// shares the LM's device and queue rather than a second one.
        pub fn load_on(loader: &WeightLoader, client: q4::Client) -> Self {
            let rvq_first = RvqDecoder::load(loader, "quantizer.rvq_first", 1);
            let rvq_rest = RvqDecoder::load(loader, "quantizer.rvq_rest", N_ACOUSTIC);
            let flat = |cbs: &[Vec<f32>]| -> Vec<f32> {
                let mut v = Vec::with_capacity(cbs.len() * CODEBOOK_SIZE * CODE_DIM);
                for cb in cbs {
                    v.extend_from_slice(cb);
                }
                v
            };
            let cb_sem = client.create_from_slice(as_bytes(&flat(&rvq_first.codebooks)));
            let cb_ac = client.create_from_slice(as_bytes(&flat(&rvq_rest.codebooks)));
            let proj_sem = client.create_from_slice(as_bytes(&rvq_first.output_proj[..]));
            let proj_ac = client.create_from_slice(as_bytes(&rvq_rest.output_proj[..]));
            drop(rvq_first);
            drop(rvq_rest);

            let ups = CpuTransConv::load(loader, "upsample.convtr.convtr.convtr", 2, HIDDEN, false);
            assert_eq!(ups.groups, HIDDEN);
            assert_eq!((ups.k, ups.stride), (4, 2));
            let upsample_w = client.create_from_slice(as_bytes(&ups.w[..]));
            drop(ups);

            // RoPE half-tables, interleaved convention (θ = 10000).
            let half = TR_HEAD_DIM / 2;
            let mut cos = vec![0f32; MAX_POS * half];
            let mut sin = vec![0f32; MAX_POS * half];
            for p in 0..MAX_POS {
                for i in 0..half {
                    let r = p as f64 * TR_ROPE_THETA.powf(-2.0 * i as f64 / TR_HEAD_DIM as f64);
                    cos[p * half + i] = r.cos() as f32;
                    sin[p * half + i] = r.sin() as f32;
                }
            }

            let up = |name: &str| {
                let (v, _) = loader.load_host_f32(name);
                client.create_from_slice(as_bytes(&v[..]))
            };
            let tr = (0..TR_LAYERS)
                .map(|i| {
                    let p = format!("decoder_transformer.transformer.layers.{i}");
                    GpuTrLayer {
                        ln1_w: up(&format!("{p}.norm1.weight")),
                        ln1_b: up(&format!("{p}.norm1.bias")),
                        ln2_w: up(&format!("{p}.norm2.weight")),
                        ln2_b: up(&format!("{p}.norm2.bias")),
                        in_proj: up(&format!("{p}.self_attn.in_proj_weight")),
                        out_proj: up(&format!("{p}.self_attn.out_proj.weight")),
                        fc1: up(&format!("{p}.linear1.weight")),
                        fc2: up(&format!("{p}.linear2.weight")),
                        ls1: up(&format!("{p}.layer_scale_1.scale")),
                        ls2: up(&format!("{p}.layer_scale_2.scale")),
                    }
                })
                .collect();

            let blocks = DEC_RATIOS
                .iter()
                .enumerate()
                .map(|(i, &r)| {
                    let up_c = CpuTransConv::load(
                        loader,
                        &format!("decoder.model.{}.convtr.convtr", 3 * i + 2),
                        r,
                        1,
                        true,
                    );
                    let res1 = CpuConv::load(
                        loader,
                        &format!("decoder.model.{}.block.1.conv.conv", 3 * i + 3),
                        1,
                    );
                    let res2 = CpuConv::load(
                        loader,
                        &format!("decoder.model.{}.block.3.conv.conv", 3 * i + 3),
                        1,
                    );
                    GpuBlock {
                        up: GpuTransConv::upload(&client, &up_c),
                        res1: GpuConv::upload(&client, &res1),
                        res2: GpuConv::upload(&client, &res2),
                    }
                })
                .collect();

            let head = GpuConv::upload(&client, &CpuConv::load(loader, "decoder.model.0.conv.conv", 1));
            let tail = GpuConv::upload(&client, &CpuConv::load(loader, "decoder.model.14.conv.conv", 1));

            Self {
                cos: client.create_from_slice(as_bytes(&cos)),
                sin: client.create_from_slice(as_bytes(&sin)),
                cb_sem,
                cb_ac,
                proj_sem,
                proj_ac,
                upsample_w,
                tr,
                head,
                blocks,
                tail,
                client,
            }
        }

        /// The client this decoder runs on — so a caller can hand the same one
        /// to the LM lane and keep both on a single queue.
        pub fn client(&self) -> &q4::Client {
            &self.client
        }

        /// Submit the whole decode graph without reading back. Returns the
        /// waveform handle `[1920·T]` and its length.
        pub fn decode_submit(&self, codes: &[[u32; NUM_CODEBOOKS]]) -> (Handle, usize) {
            let t = codes.len();
            assert!(t > 0, "decode needs at least one frame");
            let n = 2 * t; // transformer positions (upsample doubles 12.5 → 25 Hz)
            assert!(n <= MAX_POS, "RoPE table holds {MAX_POS} positions, got {n}");
            let c = &self.client;
            let arr = |h: &Handle, len: usize| unsafe { ArrayArg::from_raw_parts(h.clone(), len) };
            let elems = |n: usize| c.empty(n * 4);

            // ---- quantizer ----
            let sem: Vec<u32> = codes.iter().map(|f| f[0]).collect();
            let mut ac: Vec<u32> = Vec::with_capacity(N_ACOUSTIC * t);
            for q in 1..NUM_CODEBOOKS {
                ac.extend(codes.iter().map(|f| f[q]));
            }
            let sem_h = c.create_from_slice(as_bytes(&sem));
            let ac_h = c.create_from_slice(as_bytes(&ac));
            let emb_sem = elems(t * CODE_DIM);
            let emb_ac = elems(t * CODE_DIM);
            let gather = |codes: &Handle, cb: &Handle, out: &Handle, n_q: usize| unsafe {
                let total = (t * CODE_DIM) as u32;
                rvq_gather_kernel::launch_unchecked::<Rt>(
                    c,
                    CubeCount::new_1d(total.div_ceil(256)),
                    CubeDim::new_1d(256),
                    arr(codes, n_q * t),
                    arr(cb, n_q * CODEBOOK_SIZE * CODE_DIM),
                    arr(out, t * CODE_DIM),
                    t as u32,
                    n_q as u32,
                    CODE_DIM as u32,
                    CODEBOOK_SIZE as u32,
                );
            };
            gather(&sem_h, &self.cb_sem, &emb_sem, 1);
            gather(&ac_h, &self.cb_ac, &emb_ac, N_ACOUSTIC);

            let q_lat = elems(HIDDEN * t);
            unsafe {
                rvq_proj_kernel::launch_unchecked::<Rt>(
                    c,
                    CubeCount::new_2d(t as u32, (HIDDEN as u32) / ROWS),
                    CubeDim::new_1d(ROWS * LANES),
                    arr(&emb_sem, t * CODE_DIM),
                    arr(&emb_ac, t * CODE_DIM),
                    arr(&self.proj_sem, HIDDEN * CODE_DIM),
                    arr(&self.proj_ac, HIDDEN * CODE_DIM),
                    arr(&q_lat, HIDDEN * t),
                    t as u32,
                    CODE_DIM as u32,
                    ROWS,
                    LANES,
                );
            }

            // ---- upsample (depthwise k4 s2) → [512, n] ----
            let up = elems(HIDDEN * n);
            unsafe {
                let total = (HIDDEN * n) as u32;
                dw_transconv_kernel::launch_unchecked::<Rt>(
                    c,
                    CubeCount::new_1d(total.div_ceil(256)),
                    CubeDim::new_1d(256),
                    arr(&q_lat, HIDDEN * t),
                    arr(&self.upsample_w, HIDDEN * 4),
                    arr(&up, HIDDEN * n),
                    t as u32,
                    HIDDEN as u32,
                    4,
                    2,
                );
            }

            // ---- decoder transformer, [n, 512] row-major ----
            let h = elems(n * HIDDEN);
            self.transpose(&up, &h, n, HIDDEN, false);
            self.transformer(&h, n);

            // ---- SEANet decoder ----
            let mut x = elems(HIDDEN * n);
            self.transpose(&h, &x, n, HIDDEN, true);
            let mut l = n;
            let mut y = elems(self.head.out * l);
            self.conv1d(&self.head, &x, &y, l, false, false);
            x = y;
            let mut ch = self.head.out;

            for blk in &self.blocks {
                let out_len = l * blk.up.stride;
                let upb = elems(blk.up.out * out_len);
                unsafe {
                    transconv_kernel::launch_unchecked::<Rt>(
                        c,
                        CubeCount::new_2d(out_len as u32, (blk.up.out as u32) / ROWS),
                        CubeDim::new_1d(ROWS * LANES),
                        arr(&x, ch * l),
                        arr(&blk.up.wt, blk.up.out * blk.up.k * blk.up.inc),
                        arr(&blk.up.b, blk.up.out),
                        arr(&upb, blk.up.out * out_len),
                        l as u32,
                        blk.up.inc as u32,
                        blk.up.k as u32,
                        blk.up.stride as u32,
                        ROWS,
                        LANES,
                        true, // the stage's opening ELU
                    );
                }
                l = out_len;
                ch = blk.up.out;
                // residual unit: ELU → conv k3 → ELU → conv k1, identity shortcut
                let h1 = elems(blk.res1.out * l);
                self.conv1d(&blk.res1, &upb, &h1, l, true, false);
                let h2 = elems(blk.res2.out * l);
                self.conv1d(&blk.res2, &h1, &h2, l, true, false);
                unsafe {
                    let total = (ch * l) as u32;
                    add_kernel::launch_unchecked::<Rt>(
                        c,
                        CubeCount::new_1d(total.div_ceil(256)),
                        CubeDim::new_1d(256),
                        arr(&upb, ch * l),
                        arr(&h2, ch * l),
                        total,
                    );
                }
                x = upb;
            }

            y = elems(l);
            self.conv1d(&self.tail, &x, &y, l, true, true);
            (y, l)
        }

        /// Full decode: `T×8` codes → 24 kHz waveform, clamped to [−1, 1].
        pub fn decode(&self, codes: &[[u32; NUM_CODEBOOKS]]) -> Vec<f32> {
            let (h, l) = self.decode_submit(codes);
            self.read(&h, l)
        }

        /// Blocking readback of a [`Self::decode_submit`] handle.
        pub fn read(&self, h: &Handle, l: usize) -> Vec<f32> {
            read_f32(&self.client, h, l)
        }

        fn transpose(&self, src: &Handle, dst: &Handle, l: usize, ch: usize, to_channel_major: bool) {
            let total = (l * ch) as u32;
            unsafe {
                transpose_kernel::launch_unchecked::<Rt>(
                    &self.client,
                    CubeCount::new_1d(total.div_ceil(256)),
                    CubeDim::new_1d(256),
                    ArrayArg::from_raw_parts(src.clone(), l * ch),
                    ArrayArg::from_raw_parts(dst.clone(), l * ch),
                    l as u32,
                    ch as u32,
                    to_channel_major,
                );
            }
        }

        fn conv1d(&self, conv: &GpuConv, x: &Handle, y: &Handle, l: usize, elu_in: bool, clamp: bool) {
            let rows = if conv.out % ROWS as usize == 0 { ROWS } else { 1 };
            unsafe {
                conv1d_kernel::launch_unchecked::<Rt>(
                    &self.client,
                    CubeCount::new_2d(l as u32, (conv.out as u32) / rows),
                    CubeDim::new_1d(rows * LANES),
                    ArrayArg::from_raw_parts(x.clone(), conv.inc * l),
                    ArrayArg::from_raw_parts(conv.w.clone(), conv.out * conv.inc * conv.k),
                    ArrayArg::from_raw_parts(conv.b.clone(), conv.out),
                    ArrayArg::from_raw_parts(y.clone(), conv.out * l),
                    l as u32,
                    conv.inc as u32,
                    conv.k as u32,
                    rows,
                    LANES,
                    elu_in,
                    clamp,
                );
            }
        }

        fn transformer(&self, h: &Handle, n: usize) {
            let c = &self.client;
            let arr = |hh: &Handle, len: usize| unsafe { ArrayArg::from_raw_parts(hh.clone(), len) };
            let hd = HIDDEN;
            let xin = c.empty(n * hd * 4);
            let qkv = c.empty(n * 3 * hd * 4);
            let attn = c.empty(n * hd * 4);
            let proj = c.empty(n * hd * 4);
            let ff = c.empty(n * TR_INTER * 4);
            let scale = (TR_HEAD_DIM as f32).powf(-0.5);

            // Only layer 0's norm1 stands alone; every later norm rides the
            // residual add that feeds it (`ls_add_ln`).
            self.layernorm(h, &self.tr[0].ln1_w, &self.tr[0].ln1_b, &xin, n);
            for (i, l) in self.tr.iter().enumerate() {
                self.linear(&xin, &l.in_proj, &qkv, n, hd, 3 * hd, false);
                unsafe {
                    let total = (n * hd / 2) as u32;
                    rope_kernel::launch_unchecked::<Rt>(
                        c,
                        CubeCount::new_1d(total.div_ceil(256)),
                        CubeDim::new_1d(256),
                        arr(&qkv, n * 3 * hd),
                        arr(&self.cos, MAX_POS * TR_HEAD_DIM / 2),
                        arr(&self.sin, MAX_POS * TR_HEAD_DIM / 2),
                        n as u32,
                        hd as u32,
                        TR_HEAD_DIM as u32,
                        (TR_HEAD_DIM / 2) as u32,
                    );
                    attn_kernel::launch_unchecked::<Rt>(
                        c,
                        CubeCount::new_2d(n as u32, TR_HEADS as u32),
                        CubeDim::new_1d(TR_HEAD_DIM as u32),
                        arr(&qkv, n * 3 * hd),
                        arr(&attn, n * hd),
                        scale,
                        hd as u32,
                        TR_HEAD_DIM as u32,
                        TR_WINDOW as u32,
                        SCORE_CAP,
                    );
                }
                self.linear(&attn, &l.out_proj, &proj, n, hd, hd, false);
                self.ls_add_ln(h, &proj, &l.ls1, &l.ln2_w, &l.ln2_b, &xin, n);
                self.linear(&xin, &l.fc1, &ff, n, hd, TR_INTER, true);
                self.linear(&ff, &l.fc2, &proj, n, TR_INTER, hd, false);
                match self.tr.get(i + 1) {
                    Some(next) => {
                        self.ls_add_ln(h, &proj, &l.ls2, &next.ln1_w, &next.ln1_b, &xin, n)
                    }
                    None => self.layerscale_add(h, &proj, &l.ls2, n),
                }
            }
        }

        fn layernorm(&self, x: &Handle, w: &Handle, b: &Handle, y: &Handle, n: usize) {
            unsafe {
                layernorm_kernel::launch_unchecked::<Rt>(
                    &self.client,
                    CubeCount::new_1d(n as u32),
                    CubeDim::new_1d(256),
                    ArrayArg::from_raw_parts(x.clone(), n * HIDDEN),
                    ArrayArg::from_raw_parts(w.clone(), HIDDEN),
                    ArrayArg::from_raw_parts(b.clone(), HIDDEN),
                    ArrayArg::from_raw_parts(y.clone(), n * HIDDEN),
                    TR_EPS as f32,
                    HIDDEN as u32,
                    256,
                );
            }
        }

        fn linear(
            &self,
            x: &Handle,
            w: &Handle,
            y: &Handle,
            n: usize,
            in_dim: usize,
            out_dim: usize,
            gelu: bool,
        ) {
            assert_eq!(n % 2, 0, "sequence length is 2 per Mimi frame");
            assert_eq!(out_dim % ROWS as usize, 0);
            unsafe {
                linear_kernel::launch_unchecked::<Rt>(
                    &self.client,
                    CubeCount::new_1d((out_dim as u32) / ROWS),
                    CubeDim::new_1d(ROWS * LANES),
                    ArrayArg::from_raw_parts(x.clone(), n * in_dim),
                    ArrayArg::from_raw_parts(w.clone(), out_dim * in_dim),
                    ArrayArg::from_raw_parts(y.clone(), n * out_dim),
                    n as u32,
                    in_dim as u32,
                    out_dim as u32,
                    ROWS,
                    LANES,
                    gelu,
                );
            }
        }

        #[allow(clippy::too_many_arguments)]
        fn ls_add_ln(
            &self,
            h: &Handle,
            delta: &Handle,
            ls: &Handle,
            w: &Handle,
            b: &Handle,
            y: &Handle,
            n: usize,
        ) {
            unsafe {
                ls_add_ln_kernel::launch_unchecked::<Rt>(
                    &self.client,
                    CubeCount::new_1d(n as u32),
                    CubeDim::new_1d(256),
                    ArrayArg::from_raw_parts(h.clone(), n * HIDDEN),
                    ArrayArg::from_raw_parts(delta.clone(), n * HIDDEN),
                    ArrayArg::from_raw_parts(ls.clone(), HIDDEN),
                    ArrayArg::from_raw_parts(w.clone(), HIDDEN),
                    ArrayArg::from_raw_parts(b.clone(), HIDDEN),
                    ArrayArg::from_raw_parts(y.clone(), n * HIDDEN),
                    TR_EPS as f32,
                    HIDDEN as u32,
                    256,
                );
            }
        }

        fn layerscale_add(&self, h: &Handle, delta: &Handle, ls: &Handle, n: usize) {
            let total = (n * HIDDEN) as u32;
            unsafe {
                layerscale_add_kernel::launch_unchecked::<Rt>(
                    &self.client,
                    CubeCount::new_1d(total.div_ceil(256)),
                    CubeDim::new_1d(256),
                    ArrayArg::from_raw_parts(h.clone(), n * HIDDEN),
                    ArrayArg::from_raw_parts(delta.clone(), n * HIDDEN),
                    ArrayArg::from_raw_parts(ls.clone(), HIDDEN),
                    n as u32,
                    HIDDEN as u32,
                );
            }
        }
    }

    fn read_f32(client: &q4::Client, h: &Handle, n: usize) -> Vec<f32> {
        use cubecl::CubeElement;
        let bytes = client.read_one(h.clone()).expect("readback");
        let mut v = vec![0f32; n];
        v.copy_from_slice(&f32::from_bytes(&bytes)[..n]);
        v
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

/// GPU-vs-CPU decoder gate and bench. Needs real weights: `MIMI_TEST_PILE`,
/// else `$MARY_MODELS/personaplex.pile`. Both tests FAIL (never silently
/// skip) when a pile is named but unusable; when no pile can be located at
/// all they print a SKIP line naming the fix.
///
///   cargo test --release --features qwen3tts,q4,import --lib mimi_gpu \
///       -- --nocapture --test-threads=1
#[cfg(all(test, feature = "q4"))]
mod gpu_tests {
    use super::*;
    use crate::models::personaplex::mimi::decoder::MimiDecoderGpu;
    use std::path::PathBuf;
    use std::time::Instant;

    fn pile_path() -> Option<PathBuf> {
        if let Ok(p) = std::env::var("MIMI_TEST_PILE") {
            let p = PathBuf::from(p);
            assert!(p.exists(), "MIMI_TEST_PILE={} does not exist", p.display());
            return Some(p);
        }
        crate::paths::model_opt(None, "personaplex.pile")
    }

    fn loader() -> Option<WeightLoader> {
        let p = pile_path()?;
        eprintln!("[weights] {}", p.display());
        Some(WeightLoader::from_pile(&p).unwrap_or_else(|e| panic!("open {}: {e}", p.display())))
    }

    /// Deterministic pseudo-random code stream in the valid `[0, 2048)` range.
    pub(super) fn codes(t: usize, seed: u64) -> Vec<[u32; NUM_CODEBOOKS]> {
        let mut s = seed | 1;
        let mut next = || {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            (s % CODEBOOK_SIZE as u64) as u32
        };
        (0..t)
            .map(|_| std::array::from_fn(|_| next()))
            .collect()
    }

    /// In-place iterative radix-2 FFT (magnitude spectrum is all we need).
    fn fft(re: &mut [f32], im: &mut [f32]) {
        let n = re.len();
        let mut j = 0usize;
        for i in 1..n {
            let mut bit = n >> 1;
            while j & bit != 0 {
                j ^= bit;
                bit >>= 1;
            }
            j |= bit;
            if i < j {
                re.swap(i, j);
                im.swap(i, j);
            }
        }
        let mut len = 2;
        while len <= n {
            let ang = -2.0 * std::f64::consts::PI / len as f64;
            for i in (0..n).step_by(len) {
                for k in 0..len / 2 {
                    let (wr, wi) = ((ang * k as f64).cos() as f32, (ang * k as f64).sin() as f32);
                    let (ur, ui) = (re[i + k], im[i + k]);
                    let (vr, vi) = (
                        re[i + k + len / 2] * wr - im[i + k + len / 2] * wi,
                        re[i + k + len / 2] * wi + im[i + k + len / 2] * wr,
                    );
                    re[i + k] = ur + vr;
                    im[i + k] = ui + vi;
                    re[i + k + len / 2] = ur - vr;
                    im[i + k + len / 2] = ui - vi;
                }
            }
            len <<= 1;
        }
    }

    /// Mean log-spectral distance in dB over 512-sample Hann frames, hop 256
    /// — the natural perceptual-ish measure for a codec decoder (a waveform
    /// can differ in phase and still be the same sound; a spectrum that
    /// differs is a different sound).
    fn log_spectral_distance(a: &[f32], b: &[f32]) -> f64 {
        const N: usize = 512;
        let win: Vec<f32> = (0..N)
            .map(|i| {
                0.5 - 0.5 * (2.0 * std::f64::consts::PI * i as f64 / N as f64).cos() as f32
            })
            .collect();
        let (mut acc, mut cnt) = (0f64, 0usize);
        let mut off = 0;
        while off + N <= a.len().min(b.len()) {
            let mut spec = |x: &[f32]| {
                let mut re: Vec<f32> = (0..N).map(|i| x[off + i] * win[i]).collect();
                let mut im = vec![0f32; N];
                fft(&mut re, &mut im);
                (0..N / 2)
                    .map(|i| (re[i] * re[i] + im[i] * im[i]).sqrt() as f64)
                    .collect::<Vec<f64>>()
            };
            let (sa, sb) = (spec(a), spec(b));
            for (x, y) in sa.iter().zip(&sb) {
                acc += (20.0 * ((x + 1e-9) / (y + 1e-9)).log10()).abs();
                cnt += 1;
            }
            off += N / 2;
        }
        if cnt == 0 {
            return 0.0;
        }
        acc / cnt as f64
    }

    pub(super) fn report(name: &str, cpu: &[f32], gpu: &[f32]) {
        assert_eq!(cpu.len(), gpu.len(), "{name}: length {} vs {}", cpu.len(), gpu.len());
        let (mut dot, mut na, mut nb, mut sd, mut maxabs) = (0f64, 0f64, 0f64, 0f64, 0f64);
        for (&x, &y) in cpu.iter().zip(gpu) {
            let (x, y) = (x as f64, y as f64);
            dot += x * y;
            na += x * x;
            nb += y * y;
            sd += (x - y) * (x - y);
            maxabs = maxabs.max((x - y).abs());
        }
        let cos = dot / (na.sqrt() * nb.sqrt());
        let rel_rms = (sd / na).sqrt();
        let lsd = log_spectral_distance(cpu, gpu);
        println!(
            "  {name:14} n={:6}  cos={cos:.9}  relRMS={rel_rms:.3e}  max|Δ|={maxabs:.3e}  LSD={lsd:.4} dB",
            cpu.len()
        );
        assert!(cos > 0.9999_9, "{name}: cosine {cos} — not the same waveform");
        assert!(rel_rms < 1e-3, "{name}: relative RMS {rel_rms} too large");
        assert!(lsd < 0.5, "{name}: log-spectral distance {lsd} dB too large");
    }

    #[test]
    fn mimi_gpu_matches_cpu() {
        let Some(loader) = loader() else {
            println!("SKIP mimi_gpu_matches_cpu: {}", crate::paths::skip_reason("personaplex.pile"));
            return;
        };
        let cpu = MimiDecoder::load(&loader);
        let gpu = MimiDecoderGpu::load(&loader);
        println!("mimi decoder GPU vs CPU (f32 both; order-of-summation differences only):");
        for (name, t, seed) in [("1 frame", 1usize, 12345u64), ("5 frames", 5, 999), ("25 frames", 25, 7)] {
            let c = codes(t, seed);
            report(name, &cpu.decode(&c), &gpu.decode(&c));
        }
        // The stream the live loop actually feeds when the agent is silent.
        let sil = vec![[0u32; NUM_CODEBOOKS]; 8];
        report("zeros x8", &cpu.decode(&sil), &gpu.decode(&sil));
    }

    #[test]
    fn mimi_gpu_bench() {
        let Some(loader) = loader() else {
            println!("SKIP mimi_gpu_bench: {}", crate::paths::skip_reason("personaplex.pile"));
            return;
        };
        let cpu = MimiDecoder::load(&loader);
        let gpu = MimiDecoderGpu::load(&loader);
        let reps: usize = std::env::var("MIMI_BENCH_REPS")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(30);
        println!(
            "mimi decode ms/frame — {reps} timed repeats after 3 warm-up passes (cold pass discarded).\n\
             `cpu-s` is process CPU consumed per frame (getrusage user+sys): the number that\n\
             says how much core the encoder and depth stages get back. `submit` amortizes 8\n\
             back-to-back GPU decodes over ONE readback — the cost when nothing blocks on the\n\
             waveform, which is how the live loop runs it."
        );
        println!(
            "  {:>6}  {:>24} {:>7}  {:>24} {:>7} {:>7}  {:>7}",
            "frames", "CPU wall p50 [min..max]", "cpu-s", "GPU wall p50 [min..max]", "cpu-s", "submit", "speedup"
        );
        for t in [1usize, 5, 25] {
            let c = codes(t, 4242);
            let run = |f: &dyn Fn()| -> (f64, f64, f64, f64) {
                for _ in 0..3 {
                    f();
                }
                let cpu0 = process_cpu_secs();
                let mut ms: Vec<f64> = (0..reps)
                    .map(|_| {
                        let t0 = Instant::now();
                        f();
                        t0.elapsed().as_secs_f64() * 1e3
                    })
                    .collect();
                let cpu_ms = (process_cpu_secs() - cpu0) * 1e3 / reps as f64;
                ms.sort_by(|a, b| a.partial_cmp(b).unwrap());
                (ms[ms.len() / 2], ms[0], ms[ms.len() - 1], cpu_ms)
            };
            let (cp, cmin, cmax, ccpu) = run(&|| {
                let _ = cpu.decode(&c);
            });
            let (gp, gmin, gmax, gcpu) = run(&|| {
                let _ = gpu.decode(&c);
            });
            // 8 submits, one readback: the queue drains once, so this is the
            // per-decode cost with the blocking sync amortized away.
            let (sp, _, _, _) = run(&|| {
                let hs: Vec<_> = (0..8).map(|_| gpu.decode_submit(&c)).collect();
                let (h, l) = hs.last().unwrap();
                let _ = gpu.read(h, *l);
            });
            let f = t as f64;
            println!(
                "  {t:>6}  {:>7.2} [{:>6.2}..{:>6.2}] {:>7.2}  {:>7.2} [{:>6.2}..{:>6.2}] {:>7.2} {:>7.2}  {:>6.2}x",
                cp / f, cmin / f, cmax / f, ccpu / f,
                gp / f, gmin / f, gmax / f, gcpu / f, sp / (8.0 * f),
                cp / gp
            );
        }
    }

    /// Whole-process CPU seconds (user + sys). The CPU decoder farms its
    /// sgemms out to Accelerate's pool, so a per-thread clock would undercount
    /// it; nothing else runs in this test binary.
    fn process_cpu_secs() -> f64 {
        unsafe {
            let mut ru: libc::rusage = std::mem::zeroed();
            libc::getrusage(libc::RUSAGE_SELF, &mut ru);
            let s = |t: libc::timeval| t.tv_sec as f64 + t.tv_usec as f64 * 1e-6;
            s(ru.ru_utime) + s(ru.ru_stime)
        }
    }
}

/// Weights-free GPU-vs-CPU gate. Builds a full-size decoder from
/// deterministic pseudo-random weights in an in-memory [`WeightLoader::Pile`]
/// and runs both paths over it, so the kernels can be checked on ANY device
/// — in particular on a CUDA box that has no 34 GB PersonaPlex pile on it.
/// Same graph, same shapes, same dispatches as the real thing; only the
/// numbers are synthetic.
///
///   cargo test --release --features qwen3tts,q4 --lib mimi_gpu_synthetic \
///       -- --nocapture
///   # …and on CUDA:
///   cargo test --release --features qwen3tts,q4,cuda-backend --lib \
///       mimi_gpu_synthetic -- --nocapture
#[cfg(all(test, feature = "q4"))]
mod gpu_synth_tests {
    use super::gpu_tests::{codes, report};
    use super::*;
    use crate::models::personaplex::mimi::decoder::MimiDecoderGpu;
    use std::collections::HashMap;

    struct Rng(u64);
    impl Rng {
        fn f32(&mut self) -> f32 {
            self.0 ^= self.0 << 13;
            self.0 ^= self.0 >> 7;
            self.0 ^= self.0 << 17;
            ((self.0 >> 40) as f32 / (1u32 << 24) as f32) * 2.0 - 1.0
        }
    }

    type Weights = HashMap<String, (Vec<f32>, Vec<usize>)>;

    fn put(m: &mut Weights, r: &mut Rng, name: &str, shape: &[usize], scale: f32) {
        let n: usize = shape.iter().product();
        let v = (0..n).map(|_| r.f32() * scale).collect();
        m.insert(name.to_string(), (v, shape.to_vec()));
    }

    fn put_const(m: &mut Weights, name: &str, shape: &[usize], val: f32) {
        let n: usize = shape.iter().product();
        m.insert(name.to_string(), (vec![val; n], shape.to_vec()));
    }

    /// Fan-in scaled so eight transformer layers and four SEANet stages of
    /// random weights neither explode nor collapse — a degenerate activation
    /// range would make the comparison vacuous.
    fn fan(n: usize) -> f32 {
        (n as f32).sqrt().recip()
    }

    fn synth_weights() -> WeightLoader {
        let mut m = Weights::new();
        let mut r = Rng(0x1234_5678_9abc_def0);

        for (prefix, n_q) in [("quantizer.rvq_first", 1), ("quantizer.rvq_rest", N_ACOUSTIC)] {
            for q in 0..n_q {
                put(
                    &mut m,
                    &mut r,
                    &format!("{prefix}.vq.layers.{q}._codebook.embedding_sum"),
                    &[CODEBOOK_SIZE, CODE_DIM],
                    0.5,
                );
                put_const(
                    &mut m,
                    &format!("{prefix}.vq.layers.{q}._codebook.cluster_usage"),
                    &[CODEBOOK_SIZE],
                    1.0,
                );
            }
            put(
                &mut m,
                &mut r,
                &format!("{prefix}.output_proj.weight"),
                &[HIDDEN, CODE_DIM],
                fan(CODE_DIM),
            );
        }
        put(&mut m, &mut r, "upsample.convtr.convtr.convtr.weight", &[HIDDEN, 1, 4], 0.5);

        for i in 0..TR_LAYERS {
            let p = format!("decoder_transformer.transformer.layers.{i}");
            put_const(&mut m, &format!("{p}.norm1.weight"), &[HIDDEN], 1.0);
            put(&mut m, &mut r, &format!("{p}.norm1.bias"), &[HIDDEN], 0.05);
            put_const(&mut m, &format!("{p}.norm2.weight"), &[HIDDEN], 1.0);
            put(&mut m, &mut r, &format!("{p}.norm2.bias"), &[HIDDEN], 0.05);
            put(&mut m, &mut r, &format!("{p}.self_attn.in_proj_weight"), &[3 * HIDDEN, HIDDEN], fan(HIDDEN));
            put(&mut m, &mut r, &format!("{p}.self_attn.out_proj.weight"), &[HIDDEN, HIDDEN], fan(HIDDEN));
            put(&mut m, &mut r, &format!("{p}.linear1.weight"), &[TR_INTER, HIDDEN], fan(HIDDEN));
            put(&mut m, &mut r, &format!("{p}.linear2.weight"), &[HIDDEN, TR_INTER], fan(TR_INTER));
            put(&mut m, &mut r, &format!("{p}.layer_scale_1.scale"), &[HIDDEN], 0.1);
            put(&mut m, &mut r, &format!("{p}.layer_scale_2.scale"), &[HIDDEN], 0.1);
        }

        put(&mut m, &mut r, "decoder.model.0.conv.conv.weight", &[2 * HIDDEN, HIDDEN, 7], fan(HIDDEN * 7));
        put(&mut m, &mut r, "decoder.model.0.conv.conv.bias", &[2 * HIDDEN], 0.01);
        let mut dim = 2 * HIDDEN;
        for (i, &ratio) in DEC_RATIOS.iter().enumerate() {
            let (k, half, quarter) = (2 * ratio, dim / 2, dim / 4);
            let up = format!("decoder.model.{}.convtr.convtr", 3 * i + 2);
            put(&mut m, &mut r, &format!("{up}.weight"), &[dim, half, k], fan(dim * k));
            put(&mut m, &mut r, &format!("{up}.bias"), &[half], 0.01);
            let res = format!("decoder.model.{}", 3 * i + 3);
            put(&mut m, &mut r, &format!("{res}.block.1.conv.conv.weight"), &[quarter, half, 3], fan(half * 3));
            put(&mut m, &mut r, &format!("{res}.block.1.conv.conv.bias"), &[quarter], 0.01);
            put(&mut m, &mut r, &format!("{res}.block.3.conv.conv.weight"), &[half, quarter, 1], fan(quarter));
            put(&mut m, &mut r, &format!("{res}.block.3.conv.conv.bias"), &[half], 0.01);
            dim = half;
        }
        put(&mut m, &mut r, "decoder.model.14.conv.conv.weight", &[1, dim, 3], fan(dim * 3));
        put(&mut m, &mut r, "decoder.model.14.conv.conv.bias", &[1], 0.01);

        WeightLoader::Pile(m)
    }

    #[test]
    fn mimi_gpu_synthetic_matches_cpu() {
        let loader = synth_weights();
        let cpu = MimiDecoder::load(&loader);
        let gpu = MimiDecoderGpu::load(&loader);
        println!("mimi decoder GPU vs CPU on SYNTHETIC weights (device-portability gate):");
        for (name, t, seed) in [("1 frame", 1usize, 5u64), ("3 frames", 3, 77), ("9 frames", 9, 31337)] {
            let c = codes(t, seed);
            let a = cpu.decode(&c);
            // a decoder that emitted silence would pass any comparison
            let peak = a.iter().fold(0f32, |m, v| m.max(v.abs()));
            assert!(peak > 1e-3, "{name}: CPU reference is silent (peak {peak}) — vacuous gate");
            report(name, &a, &gpu.decode(&c));
        }
    }
}
