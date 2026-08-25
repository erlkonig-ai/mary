//! The 12 Hz codec **decoder** (`speech_tokenizer/model.safetensors`,
//! `decoder.*`): 16 codebooks/frame → 24 kHz waveform, 1920× upsample.
//! Fully causal ("lightweight non-DiT"):
//!   split-RVQ decode (semantic + 15 acoustic) → causal conv 512→1024 k3
//!   → 8-layer sliding-window (72) transformer at width 512 with LayerScale
//!   → 2× (transconv ×2 + ConvNeXt) → SEANet-style SnakeBeta decoder stack
//!   (upsample 8·5·4·3) → 1-ch wav, clamp [−1,1].
//!
//! The codec **encoder** is a transformers MimiModel and is *not* ported —
//! reference-audio codes are captured once from the oracle (see PORT_NOTES).

use burn::prelude::*;
use burn::tensor::activation::gelu;
use burn::tensor::module::conv1d;
use burn::tensor::ops::ConvOptions;

use super::config::*;
use super::layers::{AttnConfig, DecoderLayer, KvCache, Linear, RopeTable};
use crate::nn::weight_loader::WeightLoader;

fn dec_attn_config() -> AttnConfig {
    AttnConfig {
        hidden: DEC_HIDDEN,
        heads: DEC_TR_HEADS,
        kv_heads: DEC_TR_HEADS,
        head_dim: DEC_TR_HEAD_DIM,
        rope_theta: DEC_TR_ROPE_THETA,
        eps: DEC_TR_EPS,
        window: Some(DEC_TR_WINDOW),
        qk_norm: false,
        layer_scale: true,
    }
}

/// Causal Conv1d: left-pad `(k−1)·dilation` zeros (stride 1 everywhere here).
///
/// Dense (groups=1) convs run as **one GEMM over k shifted views** (im2col):
/// wgpu's direct conv1d path measured ~3% efficiency on the SEANet stack
/// (7.5 s per 12 s of audio); the same MACs through matmul are ~50× faster.
/// Depthwise (groups=C, the ConvNeXt dwconv) stays on `conv1d` — it's tiny.
struct CausalConv<B: Backend> {
    w: Tensor<B, 3>, // [out, in/groups, k]
    /// Pre-flattened GEMM weight [out, k·in] (order j-major, matching the
    /// cat-of-shifts), for the groups=1 path.
    w2: Option<Tensor<B, 2>>,
    b: Tensor<B, 1>,
    dilation: usize,
    groups: usize,
}

impl<B: Backend> CausalConv<B> {
    fn load(
        loader: &WeightLoader,
        prefix: &str,
        dilation: usize,
        groups: usize,
        device: &B::Device,
    ) -> Self {
        let w: Tensor<B, 3> = loader.load_tensor(&format!("{prefix}.weight"), device);
        let [o, i, k] = w.dims();
        let w2 = (groups == 1).then(|| w.clone().swap_dims(1, 2).reshape([o, k * i]));
        Self {
            w,
            w2,
            b: loader.load_tensor(&format!("{prefix}.bias"), device),
            dilation,
            groups,
        }
    }

    fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        let k = self.w.dims()[2];
        let pad = (k - 1) * self.dilation;
        let [b, c, l] = x.dims();
        let x = if pad > 0 {
            Tensor::cat(vec![Tensor::zeros([b, c, pad], &x.device()), x], 2)
        } else {
            x
        };
        match &self.w2 {
            Some(w2) if k > 1 => {
                // im2col: rows [x[·, j·dil ..][:l] for j in 0..k] → [1, k·C, L]
                let shifts: Vec<Tensor<B, 3>> = (0..k)
                    .map(|j| x.clone().narrow(2, j * self.dilation, l))
                    .collect();
                let xs = Tensor::cat(shifts, 1);
                let o = w2.dims()[0];
                w2.clone().unsqueeze::<3>().matmul(xs) + self.b.clone().reshape([1, o, 1])
            }
            Some(w2) => {
                let o = w2.dims()[0];
                w2.clone().unsqueeze::<3>().matmul(x) + self.b.clone().reshape([1, o, 1])
            }
            None => conv1d(
                x,
                self.w.clone(),
                Some(self.b.clone()),
                ConvOptions::new([1], [0], [self.dilation], self.groups),
            ),
        }
    }
}

/// Causal ConvTranspose1d (k = m·stride here, m ∈ {1,2}): one GEMM
/// `[k·out, in] @ [in, T]`, then overlap-add of the m stride-blocks and
/// interleave — the causal right-trim (k−stride) drops the dangling tail.
struct CausalTransConv<B: Backend> {
    w2: Tensor<B, 2>, // [k·out, in]
    b: Tensor<B, 1>,
    out_ch: usize,
    k: usize,
    stride: usize,
}

impl<B: Backend> CausalTransConv<B> {
    fn load(loader: &WeightLoader, prefix: &str, stride: usize, device: &B::Device) -> Self {
        let w: Tensor<B, 3> = loader.load_tensor(&format!("{prefix}.weight"), device); // [in, out, k]
        let [i, o, k] = w.dims();
        assert!(
            k % stride == 0 && k / stride <= 2,
            "transconv k={k} stride={stride}"
        );
        // [in, out, k] → [k, out, in] → [k·out, in]
        let w2 = w.permute([2, 1, 0]).reshape([k * o, i]);
        Self {
            w2,
            b: loader.load_tensor(&format!("{prefix}.bias"), device),
            out_ch: o,
            k,
            stride,
        }
    }

    fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        let [_, _, t] = x.dims();
        let (o, k, s) = (self.out_ch, self.k, self.stride);
        let z = self.w2.clone().unsqueeze::<3>().matmul(x); // [1, k·o, T]
        // out[t·s + j] = Σ_b z[j + s·b, ·, t − b]  (b < k/s), trimmed to T·s
        let z = z.reshape([k, o, t]);
        let mut acc = z.clone().narrow(0, 0, s); // b = 0 block [s, o, T]
        if k == 2 * s {
            let z1 = z.narrow(0, s, s);
            let shifted = Tensor::cat(
                vec![
                    Tensor::zeros([s, o, 1], &z1.device()),
                    z1.narrow(2, 0, t - 1),
                ],
                2,
            );
            acc = acc + shifted;
        }
        // [s, o, T] → [o, T, s] → [1, o, T·s]
        let out = acc.permute([1, 2, 0]).reshape([1, o, t * s]);
        out + self.b.clone().reshape([1, o, 1])
    }
}

/// SnakeBeta: `x + sin²(x·e^α)/(e^β+1e-9)`, per channel.
struct SnakeBeta<B: Backend> {
    alpha: Tensor<B, 1>,
    beta: Tensor<B, 1>,
}

impl<B: Backend> SnakeBeta<B> {
    fn load(loader: &WeightLoader, prefix: &str, device: &B::Device) -> Self {
        Self {
            alpha: loader.load_tensor(&format!("{prefix}.alpha"), device),
            beta: loader.load_tensor(&format!("{prefix}.beta"), device),
        }
    }

    fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        let c = self.alpha.dims()[0];
        let a = self.alpha.clone().exp().reshape([1, c, 1]);
        let b = self.beta.clone().exp().reshape([1, c, 1]);
        x.clone() + x.mul(a).sin().powf_scalar(2.0).div(b.add_scalar(1e-9))
    }
}

/// ConvNeXt block: causal dwconv k7 → LayerNorm → pw 4× GELU pw → γ → residual.
struct ConvNeXt<B: Backend> {
    dw: CausalConv<B>,
    norm_w: Tensor<B, 1>,
    norm_b: Tensor<B, 1>,
    pw1: Linear<B>,
    pw2: Linear<B>,
    gamma: Tensor<B, 1>,
}

impl<B: Backend> ConvNeXt<B> {
    fn load(loader: &WeightLoader, prefix: &str, device: &B::Device) -> Self {
        let dim = 1024; // latent
        let _ = dim;
        Self {
            dw: CausalConv::load(
                loader,
                &format!("{prefix}.dwconv.conv"),
                1,
                DEC_LATENT,
                device,
            ),
            norm_w: loader.load_tensor(&format!("{prefix}.norm.weight"), device),
            norm_b: loader.load_tensor(&format!("{prefix}.norm.bias"), device),
            pw1: Linear::load(loader, &format!("{prefix}.pwconv1"), true, device),
            pw2: Linear::load(loader, &format!("{prefix}.pwconv2"), true, device),
            gamma: loader.load_tensor(&format!("{prefix}.gamma"), device),
        }
    }

    fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        let input = x.clone();
        let h = self.dw.forward(x).swap_dims(1, 2); // [B,T,C]
        // LayerNorm eps 1e-6 over channels
        let d = self.norm_w.dims()[0];
        let mean = h.clone().mean_dim(2);
        let var = (h.clone() - mean.clone()).powf_scalar(2.0).mean_dim(2);
        let h = (h - mean)
            .div(var.add_scalar(1e-6).sqrt())
            .mul(self.norm_w.clone().reshape([1, 1, d]))
            .add(self.norm_b.clone().reshape([1, 1, d]));
        let h = self.pw2.forward(gelu(self.pw1.forward(h)));
        let h = h.mul(self.gamma.clone().reshape([1, 1, d]));
        input + h.swap_dims(1, 2)
    }
}

/// One residual VQ bank: per-quantizer EuclideanCodebook (embedding =
/// embedding_sum / clamp(cluster_usage, 1e-5)), summed, then output_proj k1.
struct Rvq<B: Backend> {
    /// Per-quantizer decode tables [2048, 256] (pre-divided).
    codebooks: Vec<Tensor<B, 2>>,
    output_proj: Tensor<B, 3>, // [512, 256, 1]
}

impl<B: Backend> Rvq<B> {
    fn load(loader: &WeightLoader, prefix: &str, n_q: usize, device: &B::Device) -> Self {
        let codebooks = (0..n_q)
            .map(|i| {
                let sum: Tensor<B, 2> = loader.load_tensor(
                    &format!("{prefix}.vq.layers.{i}._codebook.embedding_sum"),
                    device,
                );
                let usage: Tensor<B, 1> = loader.load_tensor(
                    &format!("{prefix}.vq.layers.{i}._codebook.cluster_usage"),
                    device,
                );
                let n = usage.dims()[0];
                sum.div(usage.clamp_min(1e-5).reshape([n, 1]))
            })
            .collect();
        Self {
            codebooks,
            output_proj: loader.load_tensor(&format!("{prefix}.output_proj.weight"), device),
        }
    }

    /// `codes[q][t]` → `[1, 512, T]`.
    fn decode(&self, codes: &[Vec<u32>], device: &B::Device) -> Tensor<B, 3> {
        let t = codes[0].len();
        let mut sum: Option<Tensor<B, 2>> = None;
        for (q, row) in codes.iter().enumerate() {
            let idx: Vec<i32> = row.iter().map(|&c| c as i32).collect();
            let idx = Tensor::<B, 1, Int>::from_ints(idx.as_slice(), device);
            let e = self.codebooks[q].clone().select(0, idx); // [T, 256]
            sum = Some(match sum {
                Some(s) => s + e,
                None => e,
            });
        }
        let q = sum.unwrap().swap_dims(0, 1).reshape([1, DEC_CODE_DIM, t]); // [1,256,T]
        conv1d(
            q,
            self.output_proj.clone(),
            None,
            ConvOptions::new([1], [0], [1], 1),
        )
    }
}

/// SEANet decoder block: SnakeBeta → transconv (k=2·rate, s=rate) → 3 residual
/// units (SnakeBeta → conv k7 dil {1,3,9} → SnakeBeta → conv k1, residual).
struct DecoderBlock<B: Backend> {
    act: SnakeBeta<B>,
    up: CausalTransConv<B>,
    units: Vec<(SnakeBeta<B>, CausalConv<B>, SnakeBeta<B>, CausalConv<B>)>,
}

impl<B: Backend> DecoderBlock<B> {
    fn load(loader: &WeightLoader, prefix: &str, rate: usize, device: &B::Device) -> Self {
        let units = [1usize, 3, 9]
            .iter()
            .enumerate()
            .map(|(i, &dil)| {
                let u = format!("{prefix}.block.{}", i + 2);
                (
                    SnakeBeta::load(loader, &format!("{u}.act1"), device),
                    CausalConv::load(loader, &format!("{u}.conv1.conv"), dil, 1, device),
                    SnakeBeta::load(loader, &format!("{u}.act2"), device),
                    CausalConv::load(loader, &format!("{u}.conv2.conv"), 1, 1, device),
                )
            })
            .collect();
        Self {
            act: SnakeBeta::load(loader, &format!("{prefix}.block.0"), device),
            up: CausalTransConv::load(loader, &format!("{prefix}.block.1.conv"), rate, device),
            units,
        }
    }

    fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        let mut h = self.up.forward(self.act.forward(x));
        for (a1, c1, a2, c2) in &self.units {
            let r = h.clone();
            h = c2.forward(a2.forward(c1.forward(a1.forward(h)))) + r;
        }
        h
    }
}

pub struct CodecDecoder<B: Backend> {
    rvq_first: Rvq<B>,
    rvq_rest: Rvq<B>,
    pre_conv: CausalConv<B>,
    tr_input: Linear<B>,
    tr_layers: Vec<DecoderLayer<B>>,
    /// output_proj with the pre_transformer's final norm weight folded in.
    tr_output: Linear<B>,
    tr_rope: RopeTable<B>,
    upsample: Vec<(CausalTransConv<B>, ConvNeXt<B>)>,
    dec_head: CausalConv<B>, // decoder.0: 1024→1536 k7
    dec_blocks: Vec<DecoderBlock<B>>,
    dec_tail_act: SnakeBeta<B>,
    dec_tail_conv: CausalConv<B>, // 96→1 k7
}

impl<B: Backend> CodecDecoder<B> {
    pub fn load(loader: &WeightLoader, device: &B::Device) -> Self {
        let cfg = dec_attn_config();
        Self {
            rvq_first: Rvq::load(loader, "decoder.quantizer.rvq_first", 1, device),
            rvq_rest: Rvq::load(
                loader,
                "decoder.quantizer.rvq_rest",
                NUM_CODE_GROUPS - 1,
                device,
            ),
            pre_conv: CausalConv::load(loader, "decoder.pre_conv.conv", 1, 1, device),
            tr_input: Linear::load(loader, "decoder.pre_transformer.input_proj", true, device),
            tr_layers: (0..DEC_TR_LAYERS)
                .map(|i| {
                    DecoderLayer::load(
                        loader,
                        &format!("decoder.pre_transformer.layers.{i}"),
                        cfg,
                        device,
                    )
                })
                .collect(),
            tr_output: Linear::load(loader, "decoder.pre_transformer.output_proj", true, device)
                .fold_in(loader.load_tensor("decoder.pre_transformer.norm.weight", device)),
            tr_rope: RopeTable::new(DEC_TR_ROPE_THETA, DEC_TR_HEAD_DIM, 512, device),
            upsample: (0..DEC_UPSAMPLING_RATIOS.len())
                .map(|i| {
                    (
                        CausalTransConv::load(
                            loader,
                            &format!("decoder.upsample.{i}.0.conv"),
                            DEC_UPSAMPLING_RATIOS[i],
                            device,
                        ),
                        ConvNeXt::load(loader, &format!("decoder.upsample.{i}.1"), device),
                    )
                })
                .collect(),
            dec_head: CausalConv::load(loader, "decoder.decoder.0.conv", 1, 1, device),
            dec_blocks: (0..DEC_UPSAMPLE_RATES.len())
                .map(|i| {
                    DecoderBlock::load(
                        loader,
                        &format!("decoder.decoder.{}", i + 1),
                        DEC_UPSAMPLE_RATES[i],
                        device,
                    )
                })
                .collect(),
            dec_tail_act: SnakeBeta::load(loader, "decoder.decoder.5", device),
            dec_tail_conv: CausalConv::load(loader, "decoder.decoder.6.conv", 1, 1, device),
        }
    }

    /// Split-RVQ decode: `codes[t][q]` frames → `[1, 512, T]`.
    pub fn quantizer_decode(
        &self,
        codes: &[[u32; NUM_CODE_GROUPS]],
        device: &B::Device,
    ) -> Tensor<B, 3> {
        let sem: Vec<Vec<u32>> = vec![codes.iter().map(|f| f[0]).collect()];
        let ac: Vec<Vec<u32>> = (1..NUM_CODE_GROUPS)
            .map(|q| codes.iter().map(|f| f[q]).collect())
            .collect();
        self.rvq_first.decode(&sem, device) + self.rvq_rest.decode(&ac, device)
    }

    /// Full decode of `codes` (T frames × 16 codebooks) → waveform samples.
    pub fn decode(&self, codes: &[[u32; NUM_CODE_GROUPS]], device: &B::Device) -> Vec<f32> {
        let bench = std::env::var("QWEN3TTS_BENCH").is_ok();
        let sync = |label: &str, t: &mut std::time::Instant, x: &Tensor<B, 3>| {
            if bench {
                let _ = x.clone().slice([0..1, 0..1, 0..1]).into_data();
                eprintln!("  codec {label}: {:.2}s", t.elapsed().as_secs_f64());
                *t = std::time::Instant::now();
            }
        };
        let mut t = std::time::Instant::now();
        let h = self.quantizer_decode(codes, device);
        let h = self.pre_conv.forward(h).swap_dims(1, 2); // [1,T,1024]
        sync("rvq+preconv", &mut t, &h);

        // pre_transformer (fresh caches; full-sequence forward)
        let mut hs = self.tr_input.forward(h);
        let (cos, sin) = self.tr_rope.slices(0, hs.dims()[1]);
        let mut caches: Vec<KvCache<B>> = (0..DEC_TR_LAYERS).map(|_| KvCache::empty()).collect();
        for (layer, cache) in self.tr_layers.iter().zip(caches.iter_mut()) {
            hs = layer.forward(hs, &cos, &sin, cache, device);
        }
        let h = self
            .tr_output
            .forward(super::layers::rms(hs, DEC_TR_EPS))
            .swap_dims(1, 2); // [1,1024,T]
        sync("pre_transformer", &mut t, &h);

        let mut h = h;
        for (up, cn) in &self.upsample {
            h = cn.forward(up.forward(h));
        }
        sync("upsample", &mut t, &h);
        let mut w = self.dec_head.forward(h);
        for (i, blk) in self.dec_blocks.iter().enumerate() {
            w = blk.forward(w);
            sync(&format!("seanet block {i}"), &mut t, &w);
        }
        let w = self.dec_tail_conv.forward(self.dec_tail_act.forward(w));
        let w = w.clamp(-1.0, 1.0);
        let n = w.dims()[2];
        w.reshape([n])
            .into_data()
            .convert::<f32>()
            .to_vec()
            .unwrap()
    }

    /// Reference `chunked_decode`: 300-frame chunks, 25-frame left context.
    pub fn chunked_decode(&self, codes: &[[u32; NUM_CODE_GROUPS]], device: &B::Device) -> Vec<f32> {
        let (chunk, ctx) = (300usize, 25usize);
        let mut out = Vec::with_capacity(codes.len() * SAMPLES_PER_FRAME);
        let mut start = 0;
        while start < codes.len() {
            let end = (start + chunk).min(codes.len());
            let c = if start >= ctx { ctx } else { start };
            let wav = self.decode(&codes[start - c..end], device);
            out.extend_from_slice(&wav[c * SAMPLES_PER_FRAME..]);
            start = end;
        }
        out
    }
}
