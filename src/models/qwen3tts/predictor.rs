//! The code predictor (sub-talker) — a 5-layer, hidden-1024 Qwen3-style
//! decoder that fills in codebooks 1..15 for the current frame, conditioned on
//! the talker's last hidden state and the sampled codebook-0 token.
//!
//! Reference flow (fresh KV cache per frame):
//!   prefill  [proj(h_talker), proj(emb₀(code0))]  → lm_head[0] → code1
//!   step i   proj(embᵢ₋₁(codeᵢ))                  → lm_head[i] → codeᵢ₊₁
//! where embᵢ are the predictor's 15 talker-width (2048) embeddings, `proj`
//! is small_to_mtp_projection (2048→1024, biased), and lm_head[i] maps
//! 1024→2048.
//!
//! Runs **on the CPU** (Accelerate sgemv, see `cpu.rs`): the 15 steps are
//! strictly sequential single-token matvecs — ~1 GB of weight traffic per
//! frame — and the GPU path spent ~10× the actual math time in per-op
//! submission overhead. On the CPU the whole frame is a few milliseconds,
//! deterministic, and exactly gateable against the f32 oracle.
//!
//! Sampling nit (carried over from the GPU version): the reference's
//! sub-talker top-k=50 truncation is replaced by full-vocab gumbel-max at the
//! same temperature; the tail mass at T=0.9 is negligible for the acoustic
//! codebooks and it keeps sampling one argmax.

use super::config::*;
use super::cpu::{rms_norm, sgemv, sgemv_mt, softmax};
use crate::nn::weight_loader::WeightLoader;

use std::cell::Cell;
use std::sync::OnceLock;

// ── bench-only decomposition (QWEN3TTS_BENCH) ──
// The predictor runs on one thread; thread-locals keep the accounting off the
// struct (&self stays immutable) and cost two Instant reads per section only
// when the env var is set. Sections: proj = the 16 small_to_mtp gemvs, stack =
// the 16 5-layer forward_pos calls (gemv share tracked separately — the rest
// is the hand-rolled scalar attention/norm/rope), head = the 15 lm_head gemvs
// + sampling.
fn bench_enabled() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var("QWEN3TTS_BENCH").is_ok())
}

/// A/B arm for the `down` gemv, the sibling of `MARY_PRED_THREADS`: set
/// `MARY_PRED_DOWN_SERIAL` to put `down` [1024×3072] back on the plain serial
/// full-matrix cblas call it used before 2026-08-19. It exists because the
/// measurement that retired the serial call could only be taken off the pile
/// (see `qwen3tts_pred_bench`), and re-taking it in situ needs both arms in
/// one process — hence [`set_down_serial`], which lets a bench interleave
/// them instead of comparing two separately-loaded runs. Off by default; one
/// relaxed atomic load against a ~100 µs gemv.
fn down_flag() -> &'static std::sync::atomic::AtomicBool {
    static F: OnceLock<std::sync::atomic::AtomicBool> = OnceLock::new();
    F.get_or_init(|| {
        std::sync::atomic::AtomicBool::new(std::env::var("MARY_PRED_DOWN_SERIAL").is_ok())
    })
}
fn down_serial() -> bool {
    down_flag().load(std::sync::atomic::Ordering::Relaxed)
}
/// Flip the `down` A/B arm for subsequent calls. See [`down_flag`].
pub fn set_down_serial(v: bool) {
    down_flag().store(v, std::sync::atomic::Ordering::Relaxed);
}
thread_local! {
    static T_PROJ: Cell<f64> = const { Cell::new(0.0) };
    static T_STACK: Cell<f64> = const { Cell::new(0.0) };
    static T_STACK_GEMV: Cell<f64> = const { Cell::new(0.0) };
    static T_HEAD: Cell<f64> = const { Cell::new(0.0) };
    static N_FRAMES: Cell<u64> = const { Cell::new(0) };
    /// Per-gemv split of T_STACK_GEMV: qkv, o, gate_up, down.
    static T_GEMV_PARTS: Cell<[f64; 4]> = const { Cell::new([0.0; 4]) };
}
fn add(cell: &'static std::thread::LocalKey<Cell<f64>>, dt: f64) {
    cell.with(|c| c.set(c.get() + dt));
}

const D: usize = PRED_HEAD_DIM;
const HALF: usize = D / 2;
const Q_DIM: usize = PRED_HEADS * D; // 2048
const KV_DIM: usize = PRED_KV_HEADS * D; // 1024
const INTER: usize = 3072;
/// 2 prefill positions + 14 single-token steps.
const MAX_POS: usize = NUM_CODE_GROUPS + 1;

struct PredLayer {
    in_norm: Vec<f32>,   // [1024]
    post_norm: Vec<f32>, // [1024]
    q_norm: Vec<f32>,    // [128]
    k_norm: Vec<f32>,    // [128]
    qkv: Vec<f32>,       // [(Q+2KV)×1024], rows q‖k‖v
    o: Vec<f32>,         // [1024×2048]
    gate_up: Vec<f32>,   // [2·INTER×1024], rows gate‖up
    down: Vec<f32>,      // [1024×INTER]
}

pub struct CodePredictor {
    /// Talker hidden width, from the checkpoint (1.7B: 2048, 0.6B: 1024) —
    /// the width of `embeddings` rows and `proj`'s input.
    talker_width: usize,
    /// 15 embeddings [2048 vocab × talker_width] — talker-width, also used to
    /// build the talker's next-frame input.
    embeddings: Vec<Vec<f32>>,
    /// small_to_mtp_projection talker_width → 1024, biased — present only
    /// when the widths differ (1.7B). The 0.6B talker is already
    /// predictor-width, and the reference uses `nn.Identity()` there.
    proj: Option<(Vec<f32>, Vec<f32>)>,
    layers: Vec<PredLayer>,
    norm_w: Vec<f32>,
    /// 15 heads 1024 → 2048.
    lm_heads: Vec<Vec<f32>>,
    /// RoPE tables [MAX_POS × HALF] (θ = 1e6).
    cos: Vec<f32>,
    sin: Vec<f32>,
    /// The device engine, once [`use_gpu`](Self::use_gpu) has built it. While
    /// it is present [`predict_frame`](Self::predict_frame) runs there and
    /// the f32 host weights below stay resident only as the parity oracle.
    #[cfg(feature = "predictor-gpu")]
    gpu: Option<super::predictor_gpu::PredictorEngine>,
}

/// Borrowed view of one layer's weights, for a consumer that lays them
/// out differently — see [`super::predictor_gpu`], which folds the two
/// layernorms into the projections they precede and rounds to f16.
pub struct LayerView<'a> {
    pub in_norm: &'a [f32],
    pub post_norm: &'a [f32],
    pub q_norm: &'a [f32],
    pub k_norm: &'a [f32],
    pub qkv: &'a [f32],
    pub o: &'a [f32],
    pub gate_up: &'a [f32],
    pub down: &'a [f32],
}

impl CodePredictor {
    pub fn load(loader: &WeightLoader) -> Self {
        let p = "talker.code_predictor";
        let w = |n: String| -> Vec<f32> { loader.load_f32(&n).0 };
        let layers = (0..PRED_LAYERS)
            .map(|i| {
                let lp = format!("{p}.model.layers.{i}");
                let mut qkv = w(format!("{lp}.self_attn.q_proj.weight"));
                qkv.extend(w(format!("{lp}.self_attn.k_proj.weight")));
                qkv.extend(w(format!("{lp}.self_attn.v_proj.weight")));
                let mut gate_up = w(format!("{lp}.mlp.gate_proj.weight"));
                gate_up.extend(w(format!("{lp}.mlp.up_proj.weight")));
                PredLayer {
                    in_norm: w(format!("{lp}.input_layernorm.weight")),
                    post_norm: w(format!("{lp}.post_attention_layernorm.weight")),
                    q_norm: w(format!("{lp}.self_attn.q_norm.weight")),
                    k_norm: w(format!("{lp}.self_attn.k_norm.weight")),
                    qkv,
                    o: w(format!("{lp}.self_attn.o_proj.weight")),
                    gate_up,
                    down: w(format!("{lp}.mlp.down_proj.weight")),
                }
            })
            .collect();
        let mut cos = vec![0f32; MAX_POS * HALF];
        let mut sin = vec![0f32; MAX_POS * HALF];
        for pos in 0..MAX_POS {
            for i in 0..HALF {
                let r = pos as f64 * TALKER_ROPE_THETA.powf(-2.0 * i as f64 / D as f64);
                cos[pos * HALF + i] = r.cos() as f32;
                sin[pos * HALF + i] = r.sin() as f32;
            }
        }
        let embeddings: Vec<Vec<f32>> = (0..NUM_CODE_GROUPS - 1)
            .map(|i| w(format!("{p}.model.codec_embedding.{i}.weight")))
            .collect();
        Self {
            talker_width: embeddings[0].len() / PRED_VOCAB,
            embeddings,
            proj: loader
                .has_weight(&format!("{p}.small_to_mtp_projection.weight"))
                .then(|| {
                    (
                        w(format!("{p}.small_to_mtp_projection.weight")),
                        w(format!("{p}.small_to_mtp_projection.bias")),
                    )
                }),
            layers,
            norm_w: w(format!("{p}.model.norm.weight")),
            lm_heads: (0..NUM_CODE_GROUPS - 1)
                .map(|i| w(format!("{p}.lm_head.{i}.weight")))
                .collect(),
            cos,
            sin,
            #[cfg(feature = "predictor-gpu")]
            gpu: None,
        }
    }

    /// Move the frame loop onto the GPU: fold the layernorms, round the stack
    /// to f16 and upload (see [`super::predictor_gpu`]). Idempotent.
    ///
    /// The host f32 weights are deliberately kept. They are what
    /// [`predict_frame_cpu`](Self::predict_frame_cpu) — the parity oracle, and
    /// the `MARY_PRED_GATE` dual-run — reads, and they cost RAM, not frame
    /// time.
    #[cfg(feature = "predictor-gpu")]
    pub fn use_gpu(&mut self) {
        if self.gpu.is_none() {
            let client = crate::nn::q4::client_for_default_device();
            self.gpu = Some(super::predictor_gpu::PredictorEngine::new(client, self));
        }
    }

    /// Whether the frame loop is on the device.
    pub fn on_gpu(&self) -> bool {
        #[cfg(feature = "predictor-gpu")]
        {
            self.gpu.is_some()
        }
        #[cfg(not(feature = "predictor-gpu"))]
        {
            false
        }
    }

    /// Talker hidden width, as measured from the checkpoint's embedding rows.
    pub fn talker_width(&self) -> usize {
        self.talker_width
    }

    /// The 5 layers' weights, in stack order.
    pub fn layer_weights(&self) -> impl Iterator<Item = LayerView<'_>> {
        self.layers.iter().map(|l| LayerView {
            in_norm: &l.in_norm,
            post_norm: &l.post_norm,
            q_norm: &l.q_norm,
            k_norm: &l.k_norm,
            qkv: &l.qkv,
            o: &l.o,
            gate_up: &l.gate_up,
            down: &l.down,
        })
    }

    /// The 15 `lm_head` matrices `[2048 × 1024]`, in step order.
    pub fn lm_head_weights(&self) -> impl Iterator<Item = &[f32]> {
        self.lm_heads.iter().map(|w| w.as_slice())
    }

    /// The 15 codec embedding tables `[2048 × talker_width]`, in step order.
    pub fn embedding_tables(&self) -> impl Iterator<Item = &[f32]> {
        self.embeddings.iter().map(|w| w.as_slice())
    }

    /// The final `model.norm` weight `[1024]`.
    pub fn norm_weight(&self) -> &[f32] {
        &self.norm_w
    }

    /// small_to_mtp_projection `(weight [1024 × talker_width], bias [1024])`,
    /// absent on the 0.6B checkpoint (`nn.Identity()` there).
    pub fn proj_weights(&self) -> Option<(&[f32], &[f32])> {
        self.proj
            .as_ref()
            .map(|(w, b)| (w.as_slice(), b.as_slice()))
    }

    /// Σ of the 15 non-codebook-0 embedding rows for a full frame, added into
    /// `out: [2048]` — the predictor's share of a talker input position.
    pub fn accumulate_frame(&self, frame: &[u32; NUM_CODE_GROUPS], out: &mut [f32]) {
        for i in 1..NUM_CODE_GROUPS {
            let row = &self.embeddings[i - 1][frame[i] as usize * self.talker_width..]
                [..self.talker_width];
            for (o, &r) in out.iter_mut().zip(row) {
                *o += r;
            }
        }
    }

    /// Per-head q/k RMSNorm + RoPE, in place over one head `[D]`.
    fn norm_rope_head(&self, x: &mut [f32], w: &[f32], pos: usize) {
        let mean: f64 = x.iter().map(|&v| (v as f64) * (v as f64)).sum::<f64>() / D as f64;
        let s = ((mean + TALKER_EPS).sqrt().recip()) as f32;
        for (v, &wi) in x.iter_mut().zip(w) {
            *v *= s * wi;
        }
        for i in 0..HALF {
            let (c, sn) = (self.cos[pos * HALF + i], self.sin[pos * HALF + i]);
            let (a, b) = (x[i], x[i + HALF]);
            x[i] = a * c - b * sn;
            x[i + HALF] = b * c + a * sn;
        }
    }

    /// One position through the 5-layer stack (updates `x: [1024]` in place;
    /// `kc`/`vc` are the per-frame caches `[layers × MAX_POS × KV_DIM]`).
    fn forward_pos(&self, x: &mut [f32], pos: usize, kc: &mut [f32], vc: &mut [f32]) {
        let bench = bench_enabled();
        let t_all = std::time::Instant::now();
        let mut gemv_s = 0f64;
        // per-gemv split: 0 = qkv, 1 = o, 2 = gate_up, 3 = down
        let mut parts = [0f64; 4];
        let mut tg = |slot: usize, f: &mut dyn FnMut()| {
            if bench {
                let t = std::time::Instant::now();
                f();
                let d = t.elapsed().as_secs_f64();
                gemv_s += d;
                parts[slot] += d;
            } else {
                f();
            }
        };
        let hd = PRED_HIDDEN;
        let groups = PRED_HEADS / PRED_KV_HEADS;
        let scale = ((D as f64).powf(-0.5)) as f32;
        let mut xin = vec![0f32; hd];
        let mut qkv = vec![0f32; Q_DIM + 2 * KV_DIM];
        let mut attn = vec![0f32; Q_DIM];
        let mut proj = vec![0f32; hd];
        let mut gu = vec![0f32; 2 * INTER];
        let mut act = vec![0f32; INTER];
        for (li, l) in self.layers.iter().enumerate() {
            rms_norm(x, &l.in_norm, TALKER_EPS, &mut xin);
            tg(0, &mut || {
                sgemv_mt(&l.qkv, Q_DIM + 2 * KV_DIM, hd, &xin, &mut qkv)
            });
            let (q, rest) = qkv.split_at_mut(Q_DIM);
            let (k, v) = rest.split_at_mut(KV_DIM);
            for h in 0..PRED_HEADS {
                self.norm_rope_head(&mut q[h * D..(h + 1) * D], &l.q_norm, pos);
            }
            for h in 0..PRED_KV_HEADS {
                self.norm_rope_head(&mut k[h * D..(h + 1) * D], &l.k_norm, pos);
            }
            let co = (li * MAX_POS + pos) * KV_DIM;
            kc[co..co + KV_DIM].copy_from_slice(k);
            vc[co..co + KV_DIM].copy_from_slice(v);

            attn.fill(0.0);
            let mut scores = [0f32; MAX_POS];
            for h in 0..PRED_HEADS {
                let kvh = h / groups;
                let q = &q[h * D..(h + 1) * D];
                for (t, s) in scores[..=pos].iter_mut().enumerate() {
                    let k = &kc[(li * MAX_POS + t) * KV_DIM + kvh * D..][..D];
                    *s = q.iter().zip(k).map(|(&a, &b)| a * b).sum::<f32>() * scale;
                }
                softmax(&mut scores[..=pos]);
                let out = &mut attn[h * D..(h + 1) * D];
                for (t, &p) in scores[..=pos].iter().enumerate() {
                    let v = &vc[(li * MAX_POS + t) * KV_DIM + kvh * D..][..D];
                    for (o, &vv) in out.iter_mut().zip(v) {
                        *o += p * vv;
                    }
                }
            }
            tg(1, &mut || sgemv_mt(&l.o, hd, Q_DIM, &attn, &mut proj));
            for (xi, &p) in x.iter_mut().zip(&proj) {
                *xi += p;
            }

            rms_norm(x, &l.post_norm, TALKER_EPS, &mut xin);
            tg(2, &mut || {
                sgemv_mt(&l.gate_up, 2 * INTER, hd, &xin, &mut gu)
            });
            for i in 0..INTER {
                let g = gu[i];
                act[i] = g / (1.0 + (-g).exp()) * gu[INTER + i];
            }
            // `down` [1024×3072] goes through the pool like the other three.
            // It was held out as a serial full-matrix call to keep the bytes
            // the serial lane produced; that gate is retired, and the serial
            // call turned out to be the single most expensive gemv in the
            // predictor — the wide-n cblas kernel it selects runs at about
            // half the bandwidth of a row-block call (~118 vs ~230 GB/s
            // measured), so it cost 8.4 ms of a 28.4 ms frame while carrying
            // 20% of the weight traffic. Pooling it: 8.4 → 4.4 ms, −14% on
            // the whole predictor frame (`qwen3tts_pred_bench`; the numbers
            // and the one thing NOT checked are in cpu::sgemv_mt's block).
            tg(3, &mut || {
                if down_serial() {
                    sgemv(&l.down, hd, INTER, &act, &mut proj)
                } else {
                    sgemv_mt(&l.down, hd, INTER, &act, &mut proj)
                }
            });
            for (xi, &p) in x.iter_mut().zip(&proj) {
                *xi += p;
            }
        }
        if bench {
            add(&T_STACK, t_all.elapsed().as_secs_f64());
            add(&T_STACK_GEMV, gemv_s);
            T_GEMV_PARTS.with(|c| {
                let mut v = c.get();
                for (a, b) in v.iter_mut().zip(parts) {
                    *a += b;
                }
                c.set(v);
            });
        }
    }

    /// `proj(x)` — small_to_mtp_projection talker_width → 1024, or the
    /// 0.6B's identity.
    fn project(&self, x: &[f32]) -> Vec<f32> {
        let Some((proj_w, proj_b)) = &self.proj else {
            return x.to_vec();
        };
        let t = std::time::Instant::now();
        let mut y = vec![0f32; PRED_HIDDEN];
        sgemv_mt(proj_w, PRED_HIDDEN, self.talker_width, x, &mut y);
        for (yi, &b) in y.iter_mut().zip(proj_b) {
            *yi += b;
        }
        if bench_enabled() {
            add(&T_PROJ, t.elapsed().as_secs_f64());
        }
        y
    }

    /// Drain the bench decomposition accumulated since the last call (None
    /// when `QWEN3TTS_BENCH` is unset or nothing ran). Per-frame ms.
    pub fn take_bench(&self) -> Option<String> {
        if !bench_enabled() {
            return None;
        }
        let n = N_FRAMES.with(|c| c.replace(0)).max(1) as f64;
        let (proj, stack, gemv, head) = (
            T_PROJ.with(|c| c.replace(0.0)),
            T_STACK.with(|c| c.replace(0.0)),
            T_STACK_GEMV.with(|c| c.replace(0.0)),
            T_HEAD.with(|c| c.replace(0.0)),
        );
        let p = T_GEMV_PARTS.with(|c| c.replace([0.0; 4]));
        Some(format!(
            "{:.1}ms proj + {:.1}ms stack (gemv {:.1}ms [qkv {:.2} | o {:.2} | gate_up {:.2} | \
             down {:.2} = {:.1}% of gemv, {:.1}% of predictor], scalar attn/norm {:.1}ms) + \
             {:.1}ms lm_head+sample per frame",
            proj / n * 1e3,
            stack / n * 1e3,
            gemv / n * 1e3,
            p[0] / n * 1e3,
            p[1] / n * 1e3,
            p[2] / n * 1e3,
            p[3] / n * 1e3,
            if gemv > 0.0 { p[3] / gemv * 100.0 } else { 0.0 },
            if proj + stack + head > 0.0 {
                p[3] / (proj + stack + head) * 100.0
            } else {
                0.0
            },
            (stack - gemv) / n * 1e3,
            head / n * 1e3
        ))
    }

    /// Predict codebooks 1..15 for one frame. `talker_hidden` and
    /// `code0_embed` are talker-width `[2048]` slices; returns the 15 codes
    /// plus Σ of their talker-width embeddings (the predictor's share of the
    /// talker's next-frame input).
    pub fn predict_frame_cpu(
        &self,
        talker_hidden: &[f32],
        code0_embed: &[f32],
        do_sample: bool,
        temperature: f64,
        rng: &mut impl rand::Rng,
    ) -> ([u32; NUM_CODE_GROUPS - 1], Vec<f32>) {
        let mut kc = vec![0f32; PRED_LAYERS * MAX_POS * KV_DIM];
        let mut vc = vec![0f32; PRED_LAYERS * MAX_POS * KV_DIM];

        let mut warm = self.project(talker_hidden);
        self.forward_pos(&mut warm, 0, &mut kc, &mut vc);
        let mut h = self.project(code0_embed);
        self.forward_pos(&mut h, 1, &mut kc, &mut vc);

        let bench = bench_enabled();
        if bench {
            N_FRAMES.with(|c| c.set(c.get() + 1));
        }
        let mut codes = [0u32; NUM_CODE_GROUPS - 1];
        let mut embed_sum = vec![0f32; self.talker_width];
        let mut normed = vec![0f32; PRED_HIDDEN];
        let mut logits = vec![0f32; PRED_VOCAB];
        for step in 0..NUM_CODE_GROUPS - 1 {
            let th = std::time::Instant::now();
            rms_norm(&h, &self.norm_w, TALKER_EPS, &mut normed);
            sgemv_mt(
                &self.lm_heads[step],
                PRED_VOCAB,
                PRED_HIDDEN,
                &normed,
                &mut logits,
            );
            let code = if do_sample {
                // full-vocab gumbel-max: argmax(v/T + g) ~ multinomial(softmax(v/T))
                let mut best = (f32::MIN, 0usize);
                for (i, &v) in logits.iter().enumerate() {
                    let u: f64 = rng.gen_range(1e-12..1.0);
                    let z = (v as f64 / temperature) as f32 + (-(-u.ln()).ln()) as f32;
                    if z > best.0 {
                        best = (z, i);
                    }
                }
                best.1 as u32
            } else {
                let mut best = 0usize;
                for (i, &v) in logits.iter().enumerate() {
                    if v > logits[best] {
                        best = i;
                    }
                }
                best as u32
            };
            codes[step] = code;
            if bench {
                add(&T_HEAD, th.elapsed().as_secs_f64());
            }
            let row =
                &self.embeddings[step][code as usize * self.talker_width..][..self.talker_width];
            for (o, &r) in embed_sum.iter_mut().zip(row) {
                *o += r;
            }
            if step + 1 < NUM_CODE_GROUPS - 1 {
                h = self.project(row);
                self.forward_pos(&mut h, step + 2, &mut kc, &mut vc);
            }
        }
        (codes, embed_sum)
    }
}

impl CodePredictor {
    /// Predict codebooks 1..15 for one frame — on the device when
    /// [`use_gpu`](Self::use_gpu) has been called, on the host otherwise.
    ///
    /// Both engines consume `rng` identically (15 × 2048 draws per sampled
    /// frame, in the same order), so switching engines does not shift the
    /// talker's own sampling stream.
    pub fn predict_frame(
        &self,
        talker_hidden: &[f32],
        code0_embed: &[f32],
        do_sample: bool,
        temperature: f64,
        rng: &mut impl rand::Rng,
    ) -> ([u32; NUM_CODE_GROUPS - 1], Vec<f32>) {
        #[cfg(feature = "predictor-gpu")]
        if let Some(g) = &self.gpu {
            return g.predict_frame(talker_hidden, code0_embed, do_sample, temperature, rng);
        }
        self.predict_frame_cpu(talker_hidden, code0_embed, do_sample, temperature, rng)
    }
}
