//! `KimiDeltaAttention` — the whole KDA sublayer, projections included.
//!
//! [`super::kda`] is the *core*: the decay gate, the gated delta-rule
//! recurrence, the short convolution and the gated output norm, all as
//! dependency-free arithmetic over slices, gated against `flash-linear-attention`
//! 0.5.2's own Triton kernels. What it deliberately does not contain is the
//! eight projections around it. This module is that shell, and nothing else —
//! every line of arithmetic below is either a [`super::ops::linear`] or a call
//! into `kda`.
//!
//! # The seam, stated plainly
//!
//! The projections are 32×7168×12288 GEMMs and belong on a backend. The
//! recurrence is a strictly sequential chain of `H·K·V` updates where op
//! submission would dominate the arithmetic, and is pure Rust over slices. So
//! this block **crosses a representation boundary twice per forward**: burn
//! tensors out to `Vec<f32>` before the recurrence, and back after it. That is
//! a real cost (two copies of `[tokens, 12288]`) and it is deliberate — the
//! alternative is either a GPU launch per token or a hand-written GEMM.
//!
//! # What runs in which precision
//!
//! The checkpoint stores `A_log`, `dt_bias`, the three `*_conv1d.weight`s and
//! `o_norm.weight` as **F32** while every other weight is BF16. This port keeps
//! them at F32, matching the oracle's primary lane. That choice is **not
//! settled by any measurement** — a `from_pretrained(dtype=torch.bfloat16)`
//! load would cast them down and produce a different, self-consistent model
//! (`MANIFEST_layer_oracle.md` §5.1 measures the difference at 3.91e-3
//! absolute). It is recorded here so the choice is visible rather than
//! accidental.
//!
//! Everything else follows the shipped bf16 run: each `nn.Linear` output, each
//! convolution output, the recurrence output and the gated norm output are
//! rounded to bfloat16, because that is what a torch module storing bf16
//! activations does. The arithmetic *inside* each of them is f32.

use burn::prelude::*;

use super::config::K3TextConfig;
use super::kda::{
    rms_norm_gated, Kda, KdaConfig, KdaParams, KdaScratch, KdaState, KdaToken, ShortConv,
    ShortConvState,
};
use super::ops::{linear, ActRound};

/// Shape and hyper-parameters of one KDA sublayer.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct KdaAttnConfig {
    /// Residual-stream width (7168).
    pub hidden_size: usize,
    /// The recurrence's own shape and gate bounds.
    pub kda: KdaConfig,
    /// Epsilon of the gated output RMSNorm (`config.rms_norm_eps`, 1e-5) —
    /// **not** the q/k L2-norm epsilon, which lives in [`KdaConfig`] and is a
    /// different number for a different operation three lines away.
    pub rms_norm_eps: f64,
    /// `linear_attn_config.use_full_rank_gate`. True in this checkpoint, so the
    /// output gate is a single `g_proj` rather than a `g_a_proj`/`g_b_proj`
    /// low-rank pair.
    pub use_full_rank_gate: bool,
    /// Rank of the decay-gate bottleneck (`f_a_proj` out, 128).
    pub gate_rank: usize,
}

impl KdaAttnConfig {
    /// `num_heads · head_dim` — the width every q/k/v projection produces.
    pub fn proj_size(&self) -> usize {
        self.kda.num_heads * self.kda.head_k_dim
    }

    /// Read the sublayer's shape out of a parsed `config.json`.
    ///
    /// Every `Err` is a shape of KDA layer this port does not implement.
    /// Refusing is the point: the low-rank output gate would otherwise run to
    /// completion against weights that are not in this checkpoint.
    pub fn from_text_config(cfg: &K3TextConfig) -> Result<Self, String> {
        let lac = &cfg.linear_attn_config;
        if !lac.use_full_rank_gate {
            return Err(
                "use_full_rank_gate is false; this port implements only the full-rank \
                 `g_proj` output gate, and this checkpoint carries no g_a_proj/g_b_proj \
                 to check the other branch against"
                    .to_string(),
            );
        }
        if lac.head_dim == 0 || lac.num_heads == 0 {
            return Err(format!(
                "degenerate linear_attn_config: num_heads {} head_dim {}",
                lac.num_heads, lac.head_dim
            ));
        }
        Ok(Self {
            hidden_size: cfg.hidden_size,
            kda: KdaConfig {
                num_heads: lac.num_heads,
                head_k_dim: lac.head_dim,
                head_v_dim: lac.head_dim,
                conv_kernel: lac.short_conv_kernel_size,
                gate_lower_bound: lac.gate_lower_bound.map(|v| v as f64),
                l2norm_eps: 1e-6,
            },
            rms_norm_eps: cfg.rms_norm_eps,
            use_full_rank_gate: lac.use_full_rank_gate,
            gate_rank: lac.head_dim,
        })
    }
}

/// The ten weights of one KDA sublayer, at their checkpoint ranks.
#[derive(Debug, Clone)]
pub struct KdaAttnWeights<B: Backend> {
    /// `[proj, hidden]`.
    pub q_proj: Tensor<B, 2>,
    /// `[proj, hidden]`.
    pub k_proj: Tensor<B, 2>,
    /// `[proj, hidden]`.
    pub v_proj: Tensor<B, 2>,
    /// `[proj · width]` — `nn.Conv1d`'s `[D, 1, W]` depthwise weight, flat. F32.
    pub q_conv1d: Vec<f32>,
    /// `[proj · width]`. F32.
    pub k_conv1d: Vec<f32>,
    /// `[proj · width]`. F32.
    pub v_conv1d: Vec<f32>,
    /// `[num_heads]` — the **live** entries only. The checkpoint parameter is
    /// zero-padded to 128; taking the first `num_heads` is correct and using
    /// all 128 is a bug, because `exp(0) = 1` makes every padded head's decay
    /// rate 1 (i.e. no decay at all). F32.
    pub a_log: Vec<f32>,
    /// `[num_heads · head_dim]`, viewed as `[head, channel]`. F32.
    pub dt_bias: Vec<f32>,
    /// `[gate_rank, hidden]`.
    pub f_a_proj: Tensor<B, 2>,
    /// `[proj, gate_rank]`.
    pub f_b_proj: Tensor<B, 2>,
    /// `[num_heads, hidden]`.
    pub b_proj: Tensor<B, 2>,
    /// `[proj, hidden]` — the full-rank output gate.
    pub g_proj: Tensor<B, 2>,
    /// `[head_dim]` — one gain shared by all heads. F32.
    pub o_norm: Vec<f32>,
    /// `[hidden, proj]`.
    pub o_proj: Tensor<B, 2>,
}

/// Per-sequence recurrent state: the three convolution windows and the
/// `[H, K, V]` delta-rule state, one set per batch element.
///
/// Fixed size in `T` by construction — that is the whole point of a linear
/// attention layer, and it is expressed in the type: nothing here can grow.
#[derive(Debug, Clone)]
pub struct KdaCache {
    conv_q: Vec<ShortConvState<f32>>,
    conv_k: Vec<ShortConvState<f32>>,
    conv_v: Vec<ShortConvState<f32>>,
    state: Vec<KdaState<f32>>,
}

impl KdaCache {
    /// A zero cache for `batch` sequences — exactly the causal zero-padding at
    /// the start of a sequence, and a zero initial recurrent state.
    pub fn zeros<B: Backend>(block: &KdaAttention<B>, batch: usize) -> Self {
        assert!(batch > 0, "KdaCache over zero sequences");
        Self {
            conv_q: (0..batch).map(|_| ShortConvState::zeros(&block.q_conv)).collect(),
            conv_k: (0..batch).map(|_| ShortConvState::zeros(&block.k_conv)).collect(),
            conv_v: (0..batch).map(|_| ShortConvState::zeros(&block.v_conv)).collect(),
            state: (0..batch).map(|_| KdaState::zeros(&block.cfg.kda)).collect(),
        }
    }

    /// Number of sequences this cache tracks.
    pub fn batch(&self) -> usize {
        self.state.len()
    }

    /// One sequence's recurrent state, `[H][K][V]` in C order.
    pub fn state(&self, b: usize) -> &super::kda::KdaState<f32> {
        &self.state[b]
    }

    /// Bytes of state, all sequences. Independent of sequence length.
    pub fn byte_len(&self) -> usize {
        self.state.iter().map(|s| s.byte_len()).sum::<usize>()
            + self
                .conv_q
                .iter()
                .chain(self.conv_k.iter())
                .chain(self.conv_v.iter())
                .map(|c| c.byte_len())
                .sum::<usize>()
    }
}

/// Every boundary inside the KDA sublayer, so a gate can localise a failure to
/// one operation instead of to "attention".
#[derive(Debug, Clone)]
pub struct KdaTrace<B: Backend> {
    /// `[tokens, proj]` — before the convolution.
    pub q_proj_out: Tensor<B, 2>,
    pub k_proj_out: Tensor<B, 2>,
    pub v_proj_out: Tensor<B, 2>,
    /// `[tokens, proj]` — `silu(depthwise conv)`, what the recurrence sees.
    pub q_conv_out: Tensor<B, 2>,
    pub k_conv_out: Tensor<B, 2>,
    pub v_conv_out: Tensor<B, 2>,
    /// `[tokens, gate_rank]`.
    pub f_a_proj_out: Tensor<B, 2>,
    /// `[tokens, proj]` — the decay gate's raw input, pre-`dt_bias`.
    pub f_b_proj_out: Tensor<B, 2>,
    /// `[tokens, num_heads]` — the delta-rule step size, pre-sigmoid. The
    /// shipped code takes `.float()` of the bf16 Linear output, so this is a
    /// bf16 value in an f32 tensor.
    pub b_proj_out: Tensor<B, 2>,
    /// `[tokens, proj]` — the output gate, pre-sigmoid.
    pub g_proj_out: Tensor<B, 2>,
    /// `[tokens, proj]` — the recurrence output, before the gated norm.
    pub core_out: Tensor<B, 2>,
    /// `[tokens, proj]` — after `FusedRMSNormGated`.
    pub o_norm_out: Tensor<B, 2>,
    /// `[tokens, hidden]` — the sublayer output.
    pub out: Tensor<B, 2>,
}

/// One KDA sublayer.
#[derive(Debug, Clone)]
pub struct KdaAttention<B: Backend> {
    pub cfg: KdaAttnConfig,
    pub w: KdaAttnWeights<B>,
    pub round: ActRound,
    core: Kda<f32>,
    q_conv: ShortConv<f32>,
    k_conv: ShortConv<f32>,
    v_conv: ShortConv<f32>,
}

fn vec2<B: Backend>(t: &Tensor<B, 2>) -> Vec<f32> {
    t.clone().into_data().convert::<f32>().to_vec().expect("tensor -> f32")
}

fn t2<B: Backend>(v: Vec<f32>, rows: usize, cols: usize, device: &B::Device) -> Tensor<B, 2> {
    assert_eq!(v.len(), rows * cols, "t2: {} values into [{rows}, {cols}]", v.len());
    Tensor::from_data(TensorData::new(v, [rows, cols]), device)
}

impl<B: Backend> KdaAttention<B> {
    /// Build the sublayer, checking every weight against the config's shape.
    ///
    /// The shape assertions are the load-bearing part: a KDA layer's weights
    /// are eight matrices and five vectors of five distinct widths, and the
    /// only thing standing between a mis-ordered load and a plausible wrong
    /// answer is that the widths disagree.
    pub fn new(cfg: KdaAttnConfig, w: KdaAttnWeights<B>, round: ActRound) -> Self {
        let p = cfg.proj_size();
        let h = cfg.hidden_size;
        let (nh, hd, cw) = (cfg.kda.num_heads, cfg.kda.head_k_dim, cfg.kda.conv_kernel);
        assert_eq!(w.q_proj.dims(), [p, h], "q_proj");
        assert_eq!(w.k_proj.dims(), [p, h], "k_proj");
        assert_eq!(w.v_proj.dims(), [p, h], "v_proj");
        assert_eq!(w.f_a_proj.dims(), [cfg.gate_rank, h], "f_a_proj");
        assert_eq!(w.f_b_proj.dims(), [p, cfg.gate_rank], "f_b_proj");
        assert_eq!(w.b_proj.dims(), [nh, h], "b_proj");
        assert_eq!(w.g_proj.dims(), [p, h], "g_proj: full-rank output gate");
        assert_eq!(w.o_proj.dims(), [h, p], "o_proj");
        assert_eq!(w.a_log.len(), nh, "A_log must be the {nh} LIVE heads, not the padded parameter");
        assert_eq!(w.dt_bias.len(), p, "dt_bias");
        assert_eq!(w.o_norm.len(), hd, "o_norm gain is per-channel within a head");
        for (name, c) in [("q", &w.q_conv1d), ("k", &w.k_conv1d), ("v", &w.v_conv1d)] {
            assert_eq!(c.len(), p * cw, "{name}_conv1d must be [D, 1, W] = [{p}, 1, {cw}]");
        }
        assert!(
            cfg.use_full_rank_gate,
            "KdaAttention holds a full-rank g_proj; a low-rank config must not reach here"
        );

        let core = Kda::new(cfg.kda, KdaParams::new(&cfg.kda, &w.a_log, &w.dt_bias));
        let q_conv = ShortConv::new(p, cw, &w.q_conv1d);
        let k_conv = ShortConv::new(p, cw, &w.k_conv1d);
        let v_conv = ShortConv::new(p, cw, &w.v_conv1d);
        Self { cfg, w, round, core, q_conv, k_conv, v_conv }
    }

    /// Run the sublayer over `[tokens, hidden]`, where `tokens = batch · seq`
    /// in row-major `[b, t] -> b·seq + t` order.
    ///
    /// `cache` carries the convolution windows and the recurrent state and is
    /// advanced in place, so prefill and decode are the same code path.
    pub fn forward(&self, hidden: Tensor<B, 2>, cache: &mut KdaCache) -> KdaTrace<B> {
        let [tokens, dh] = hidden.dims();
        let batch = cache.batch();
        assert_eq!(dh, self.cfg.hidden_size, "hidden width {dh}");
        assert!(tokens > 0, "KDA forward over zero tokens");
        assert_eq!(
            tokens % batch,
            0,
            "{tokens} tokens do not divide into {batch} sequences"
        );
        let seq = tokens / batch;
        let device = hidden.device();
        let p = self.cfg.proj_size();
        let (nh, hd) = (self.cfg.kda.num_heads, self.cfg.kda.head_k_dim);

        // --- the six projections -----------------------------------------
        let q_proj_out = linear(hidden.clone(), &self.w.q_proj, self.round);
        let k_proj_out = linear(hidden.clone(), &self.w.k_proj, self.round);
        let v_proj_out = linear(hidden.clone(), &self.w.v_proj, self.round);
        let f_a_proj_out = linear(hidden.clone(), &self.w.f_a_proj, self.round);
        let f_b_proj_out = linear(f_a_proj_out.clone(), &self.w.f_b_proj, self.round);
        // `self.b_proj(hidden_states).float()`: the Linear output is in the
        // model dtype and the upcast happens after it, so the rounding is real
        // and this is a bf16 value living in an f32 tensor.
        let b_proj_out = linear(hidden.clone(), &self.w.b_proj, self.round);
        let g_proj_out = linear(hidden.clone(), &self.w.g_proj, self.round);

        // --- across the seam ----------------------------------------------
        let mut qc = vec![0f32; tokens * p];
        let mut kc = vec![0f32; tokens * p];
        let mut vc = vec![0f32; tokens * p];
        let qp = vec2(&q_proj_out);
        let kp = vec2(&k_proj_out);
        let vp = vec2(&v_proj_out);
        for b in 0..batch {
            let r = b * seq * p..(b + 1) * seq * p;
            self.q_conv.forward(&mut cache.conv_q[b], seq, &qp[r.clone()], &mut qc[r.clone()]);
            self.k_conv.forward(&mut cache.conv_k[b], seq, &kp[r.clone()], &mut kc[r.clone()]);
            self.v_conv.forward(&mut cache.conv_v[b], seq, &vp[r.clone()], &mut vc[r]);
        }
        // The convolution output is a module output, so it is stored at the
        // model dtype like every other one.
        let bf = |v: Vec<f32>| -> Vec<f32> {
            match self.round {
                ActRound::None => v,
                ActRound::Bf16 => v.into_iter().map(|x| half::bf16::from_f32(x).to_f32()).collect(),
            }
        };
        let (qc, kc, vc) = (bf(qc), bf(kc), bf(vc));

        let g_raw = vec2(&f_b_proj_out);
        let beta_raw = vec2(&b_proj_out);
        let mut core = vec![0f32; tokens * p];
        let mut scratch = KdaScratch::new(&self.cfg.kda);
        for b in 0..batch {
            for t in 0..seq {
                let i = b * seq + t;
                self.core.step(
                    &mut cache.state[b],
                    &mut scratch,
                    KdaToken {
                        q_raw: &qc[i * p..(i + 1) * p],
                        k_raw: &kc[i * p..(i + 1) * p],
                        v: &vc[i * p..(i + 1) * p],
                        g_raw: &g_raw[i * p..(i + 1) * p],
                        beta_raw: &beta_raw[i * nh..(i + 1) * nh],
                    },
                    &mut core[i * p..(i + 1) * p],
                );
            }
        }
        let core = bf(core);

        // --- the gated output norm, per (token, head) ----------------------
        let g_gate = vec2(&g_proj_out);
        let mut normed = vec![0f32; tokens * p];
        for i in 0..tokens {
            for h in 0..nh {
                let r = i * p + h * hd..i * p + (h + 1) * hd;
                let (x, g) = (&core[r.clone()], &g_gate[r.clone()]);
                rms_norm_gated(x, g, &self.w.o_norm, self.cfg.rms_norm_eps, &mut normed[r]);
            }
        }
        let normed = bf(normed);

        // --- back across the seam ------------------------------------------
        let core_out = t2::<B>(core, tokens, p, &device);
        let o_norm_out = t2::<B>(normed, tokens, p, &device);
        let out = linear(o_norm_out.clone(), &self.w.o_proj, self.round);

        let (qc, kc, vc) = (
            t2::<B>(qc, tokens, p, &device),
            t2::<B>(kc, tokens, p, &device),
            t2::<B>(vc, tokens, p, &device),
        );
        KdaTrace {
            q_proj_out,
            k_proj_out,
            v_proj_out,
            q_conv_out: qc,
            k_conv_out: kc,
            v_conv_out: vc,
            f_a_proj_out,
            f_b_proj_out,
            b_proj_out,
            g_proj_out,
            core_out,
            o_norm_out,
            out,
        }
    }
}
