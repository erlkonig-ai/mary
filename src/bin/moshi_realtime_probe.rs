//! moshi_realtime_probe — the decisive REALTIME SPIKE for the PersonaPlex-7B
//! (Moshi) voice port. Measures the ONE unproven risk before we commit to the
//! full 7B LM port: can the Moshi **temporal transformer** run at 12.5 Hz
//! (80 ms/frame budget) on this M4 Max, in f16 on the Metal (fused) backend?
//!
//! No weights are downloaded. The transformer stacks are built at the REAL
//! Moshi dims but RANDOM-initialized, by feeding mary's existing
//! `qwen3tts::layers::{DecoderLayer, Attention}` primitives a synthetic
//! `WeightLoader::Pile` whose every requested key resolves to random f32 at the
//! correct shape. Those primitives already implement full MHA with KV-cache,
//! folded RoPE, fused qkv, and RMSNorm pre-norm — exactly the Moshi block — so
//! we reconfigure rather than rewrite.
//!
//! Dims (Moshi / PersonaPlex-7B, Helium-7B temporal + depth transformer):
//!   temporal: d_model 4096, 32 layers, 32 heads, head_dim 128,
//!             FULL MHA (kv_heads == heads == 32, NOT GQA), ffn 16384,
//!             RMSNorm pre-norm, SiLU-gated MLP, RoPE θ=10000, KV-cache,
//!             context ~3000.
//!   depth:    d_model 1024, 6 layers, 16 heads, head_dim 64, ffn 4096;
//!             runs 8 SEQUENTIAL single-token steps per frame (per codebook).
//!
//! Regime measured: single-token DECODE (KV cache pre-filled). We sweep a few
//! context lengths (256 / 1024 / 3000) for the temporal transformer, and time
//! one 8-step frame for the depth transformer. Methodology mirrors
//! `qwen3tts_megakernel_probe`: warm-up discarded, min-of-medians over rounds,
//! and we split each step into SUBMIT time (host returns, work still on the
//! GPU queue) vs FULL time (a forced `into_data()` sync / readback). The
//! submit-vs-full gap is the diagnostic: if submit ≈ full and both are large,
//! we're GPU-bound (compute or bandwidth); if submit is a large fraction of a
//! small full, we're host-submission-bound (the qwen3tts megakernel lever).
//! A tiny control matmul is interleaved each round to catch contention.

use burn::prelude::*;
use burn::tensor::backend::BackendTypes;
use mary::models::qwen3tts::layers::{AttnConfig, DecoderLayer, KvCache, RopeTable};
use mary::nn::backend::{BFused, BFusedHalf, BHalf, B as BMetal};
use mary::nn::weight_loader::WeightLoader;
use std::collections::HashMap;
use std::time::Instant;

/// A cheap deterministic pseudo-random f32 in roughly [-1/√fan, 1/√fan] — small
/// enough that f16 activations stay in range across 32 layers. `seed` decorrelates
/// tensors; `scale` ~ 1/√fan_in.
fn fill(n: usize, seed: u64, scale: f32) -> Vec<f32> {
    let mut s = seed.wrapping_mul(0x9E3779B97F4A7C15).wrapping_add(1);
    (0..n)
        .map(|_| {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            // map to (-1, 1)
            let u = ((s >> 11) as f64 / (1u64 << 53) as f64) as f32; // [0,1)
            (u * 2.0 - 1.0) * scale
        })
        .collect()
}

/// Geometry of one transformer stack we synthesize.
struct StackGeom {
    hidden: usize,
    layers: usize,
    heads: usize,
    kv_heads: usize,
    head_dim: usize,
    inter: usize,
    rope_theta: f64,
    eps: f64,
}

/// Build a synthetic `WeightLoader::Pile` holding random weights for every key
/// `DecoderLayer::load`/`Attention::load` will request for `g.layers` layers
/// under `prefix.{i}`. `qk_norm=false`, `layer_scale=false` (Moshi/Helium block),
/// so no q_norm/k_norm/*_scale keys are needed.
fn synth_loader(g: &StackGeom, prefix: &str) -> WeightLoader {
    let mut m: HashMap<String, (Vec<f32>, Vec<usize>)> = HashMap::new();
    let hd = g.head_dim;
    let q_out = g.heads * hd;
    let kv_out = g.kv_heads * hd;
    let mut seed = 1u64;
    let mut put = |name: String, shape: Vec<usize>, scale: f32| {
        let n: usize = shape.iter().product();
        seed = seed.wrapping_add(0x1234_5678);
        m.insert(name, (fill(n, seed, scale), shape));
    };
    let in_scale = |fan: usize| (fan as f32).powf(-0.5);
    for i in 0..g.layers {
        let p = format!("{prefix}.{i}");
        // attention projections: weights are [out, in]
        put(format!("{p}.self_attn.q_proj.weight"), vec![q_out, g.hidden], in_scale(g.hidden));
        put(format!("{p}.self_attn.k_proj.weight"), vec![kv_out, g.hidden], in_scale(g.hidden));
        put(format!("{p}.self_attn.v_proj.weight"), vec![kv_out, g.hidden], in_scale(g.hidden));
        put(format!("{p}.self_attn.o_proj.weight"), vec![g.hidden, q_out], in_scale(q_out));
        // pre-norm weights [hidden]
        put(format!("{p}.input_layernorm.weight"), vec![g.hidden], 0.02);
        put(format!("{p}.post_attention_layernorm.weight"), vec![g.hidden], 0.02);
        // MLP: gate/up [inter, hidden], down [hidden, inter]
        put(format!("{p}.mlp.gate_proj.weight"), vec![g.inter, g.hidden], in_scale(g.hidden));
        put(format!("{p}.mlp.up_proj.weight"), vec![g.inter, g.hidden], in_scale(g.hidden));
        put(format!("{p}.mlp.down_proj.weight"), vec![g.hidden, g.inter], in_scale(g.inter));
    }
    // small offset so folded weights (near 0.02) don't zero everything
    for (_, (v, _)) in m.iter_mut() {
        for x in v.iter_mut() {
            *x += 0.001;
        }
    }
    WeightLoader::Pile(m)
}

/// A synthetic transformer stack of `DecoderLayer`s + shared RoPE table.
struct SynthStack<B: Backend> {
    layers: Vec<DecoderLayer<B>>,
    rope: RopeTable<B>,
    hidden: usize,
}

impl<B: Backend> SynthStack<B> {
    fn build(g: &StackGeom, max_len: usize, device: &B::Device) -> Self {
        let loader = synth_loader(g, "layers");
        let cfg = AttnConfig {
            hidden: g.hidden,
            heads: g.heads,
            kv_heads: g.kv_heads,
            head_dim: g.head_dim,
            rope_theta: g.rope_theta,
            eps: g.eps,
            window: None, // full causal context
            qk_norm: false,
            layer_scale: false,
        };
        let layers = (0..g.layers)
            .map(|i| DecoderLayer::<B>::load(&loader, &format!("layers.{i}"), cfg, device))
            .collect();
        let rope = RopeTable::<B>::new(g.rope_theta, g.head_dim, max_len, device);
        Self { layers, rope, hidden: g.hidden }
    }

    fn new_caches(&self) -> Vec<KvCache<B>> {
        (0..self.layers.len()).map(|_| KvCache::empty()).collect()
    }

    /// One forward over `l` new tokens at cache offset. Returns the last hidden.
    fn forward(&self, x: Tensor<B, 3>, caches: &mut [KvCache<B>], device: &B::Device) -> Tensor<B, 3> {
        let offset = caches[0].seq_len();
        let l = x.dims()[1];
        let (cos, sin) = self.rope.slices(offset, l);
        let mut h = x;
        for (layer, cache) in self.layers.iter().zip(caches.iter_mut()) {
            h = layer.forward(h, &cos, &sin, cache, device);
        }
        h
    }
}

/// Force a host sync by reading back the last position's hidden state — mirrors
/// the qwen3tts talker's `last_hidden` (narrow + into_data), the proven readback
/// shape for the fused Metal backend.
fn sync_scalar<B: Backend>(t: &Tensor<B, 3>) -> f32 {
    let [_, l, d] = t.dims();
    let v = t
        .clone()
        .narrow(1, l - 1, 1)
        .reshape([d])
        .into_data()
        .convert::<f32>()
        .to_vec::<f32>()
        .unwrap();
    v.first().copied().unwrap_or(0.0)
}

fn median(mut v: Vec<f64>) -> f64 {
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    v[v.len() / 2]
}

/// Prefill chunk size — small chunks keep each fused decode graph shallow
/// (burn-fusion 0.21 can panic on very deep single-graph prefills). Overridable
/// via MOSHI_PREFILL_CHUNK.
fn prefill_chunk() -> usize {
    std::env::var("MOSHI_PREFILL_CHUNK")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(32)
}

/// Prefill a fresh cache to `ctx` positions, syncing after each chunk.
fn prefill<B: Backend>(stack: &SynthStack<B>, ctx: usize, device: &B::Device) -> Vec<KvCache<B>> {
    let mut caches = stack.new_caches();
    let cs = prefill_chunk();
    let mut pos = 0;
    while pos < ctx {
        let chunk = (ctx - pos).min(cs).max(1);
        let x = Tensor::<B, 1>::from_floats(
            fill(chunk * stack.hidden, 7 + pos as u64, 0.1).as_slice(),
            device,
        )
        .reshape([1, chunk, stack.hidden]);
        let h = stack.forward(x, &mut caches, device);
        let _ = sync_scalar(&h);
        pos += chunk;
    }
    caches
}

/// Prefill a cache to `ctx` positions (chunked), then measure single-token
/// decode steps. Returns (submit_ms, full_ms) min-of-medians over `rounds`.
fn bench_decode<B: Backend>(
    stack: &SynthStack<B>,
    ctx: usize,
    rounds: usize,
    steps_per_round: usize,
    device: &B::Device,
) -> (f64, f64) {
    let mut caches = prefill(stack, ctx, device);
    // warm the decode-shape JIT (discarded)
    for w in 0..3 {
        let e = Tensor::<B, 1>::from_floats(
            fill(stack.hidden, 100 + w, 0.1).as_slice(),
            device,
        )
        .reshape([1, 1, stack.hidden]);
        let h = stack.forward(e, &mut caches, device);
        let _ = sync_scalar(&h);
    }
    let mut submit_meds = Vec::new();
    let mut full_meds = Vec::new();
    for r in 0..rounds {
        let mut subs = Vec::new();
        let mut fulls = Vec::new();
        for s in 0..steps_per_round {
            let e = Tensor::<B, 1>::from_floats(
                fill(stack.hidden, 1000 + (r * steps_per_round + s) as u64, 0.1).as_slice(),
                device,
            )
            .reshape([1, 1, stack.hidden]);
            let t0 = Instant::now();
            let h = stack.forward(e, &mut caches, device);
            let submit = t0.elapsed().as_secs_f64();
            let _ = sync_scalar(&h);
            let full = t0.elapsed().as_secs_f64();
            subs.push(submit * 1e3);
            fulls.push(full * 1e3);
        }
        submit_meds.push(median(subs));
        full_meds.push(median(fulls));
        // Rebuild the cache so ctx stays ~fixed across rounds once it has grown
        // materially past the target. Per-round growth is only steps_per_round.
        if caches[0].seq_len() > ctx + steps_per_round + 4 {
            caches = prefill(stack, ctx, device);
        }
    }
    (
        submit_meds.iter().cloned().fold(f64::INFINITY, f64::min),
        full_meds.iter().cloned().fold(f64::INFINITY, f64::min),
    )
}

/// One depth-transformer FRAME = 8 sequential single-token decode steps, each
/// appending to a FRESH cache (the depth transformer restarts every audio
/// frame, walking the 8 codebooks). Returns (submit_ms, full_ms) for the whole
/// 8-step frame, min-of-medians.
fn bench_depth_frame<B: Backend>(
    stack: &SynthStack<B>,
    rounds: usize,
    device: &B::Device,
) -> (f64, f64) {
    // warm
    for _ in 0..3 {
        let mut caches = stack.new_caches();
        for _k in 0..8 {
            let e = Tensor::<B, 1>::from_floats(
                fill(stack.hidden, 55, 0.1).as_slice(),
                device,
            )
            .reshape([1, 1, stack.hidden]);
            let h = stack.forward(e, &mut caches, device);
            let _ = sync_scalar(&h);
        }
    }
    let mut submit_meds = Vec::new();
    let mut full_meds = Vec::new();
    for r in 0..rounds {
        let mut subs = Vec::new();
        let mut fulls = Vec::new();
        // measure several frames per round, take median
        for f in 0..8 {
            let mut caches = stack.new_caches();
            let t0 = Instant::now();
            let mut last = None;
            for k in 0..8u64 {
                let e = Tensor::<B, 1>::from_floats(
                    fill(stack.hidden, 2000 + r as u64 * 64 + f as u64 * 8 + k, 0.1).as_slice(),
                    device,
                )
                .reshape([1, 1, stack.hidden]);
                last = Some(stack.forward(e, &mut caches, device));
            }
            let submit = t0.elapsed().as_secs_f64();
            let _ = sync_scalar(last.as_ref().unwrap());
            let full = t0.elapsed().as_secs_f64();
            subs.push(submit * 1e3);
            fulls.push(full * 1e3);
        }
        submit_meds.push(median(subs));
        full_meds.push(median(fulls));
    }
    (
        submit_meds.iter().cloned().fold(f64::INFINITY, f64::min),
        full_meds.iter().cloned().fold(f64::INFINITY, f64::min),
    )
}

/// Tiny control matmul on the same device to detect contention between rounds.
fn control_op<B: Backend>(device: &B::Device) -> f64 {
    let a = Tensor::<B, 1>::from_floats(fill(256 * 256, 42, 0.1).as_slice(), device)
        .reshape([1, 256, 256]);
    let b = Tensor::<B, 1>::from_floats(fill(256 * 256, 43, 0.1).as_slice(), device)
        .reshape([1, 256, 256]);
    let t0 = Instant::now();
    let c = a.matmul(b);
    let _ = sync_scalar(&c);
    t0.elapsed().as_secs_f64() * 1e3
}

fn run_backend<B: Backend>(label: &str, rounds: usize)
where
    B: BackendTypes,
{
    let device: <B as BackendTypes>::Device = Default::default();
    println!("\n=== backend: {label} ===");
    let ctrl = control_op::<B>(&device);
    println!("control matmul (256³): {ctrl:.3} ms  (baseline for contention)");

    // ---- Temporal transformer: Moshi/Helium-7B dims, FULL MHA ----
    let temporal = StackGeom {
        hidden: 4096,
        layers: 32,
        heads: 32,
        kv_heads: 32, // full MHA, NOT GQA
        head_dim: 128,
        inter: 16384,
        rope_theta: 10000.0,
        eps: 1e-5,
    };
    println!(
        "\nTEMPORAL transformer: d_model {}, {} layers, {} heads (kv {}), head_dim {}, ffn {}",
        temporal.hidden, temporal.layers, temporal.heads, temporal.kv_heads, temporal.head_dim, temporal.inter
    );
    let max_len = 3072;
    println!("  building synthetic stack (random weights, max_len {max_len})...");
    let t_build = Instant::now();
    let tstack = SynthStack::<B>::build(&temporal, max_len, &device);
    println!("  built in {:.2}s", t_build.elapsed().as_secs_f64());

    let mut temporal_results: Vec<(usize, f64, f64)> = Vec::new();
    for &ctx in &[256usize, 1024, 3000] {
        let c0 = control_op::<B>(&device);
        let (sub, full) = bench_decode::<B>(&tstack, ctx, rounds, 16, &device);
        let c1 = control_op::<B>(&device);
        println!(
            "  ctx {ctx:>4}: submit {sub:6.2} ms/step  full {full:6.2} ms/step   (control {c0:.2}→{c1:.2} ms)"
        );
        temporal_results.push((ctx, sub, full));
    }

    // ---- Depth transformer: 1024 dim, 6 layers, 16 heads, 8 steps/frame ----
    let depth = StackGeom {
        hidden: 1024,
        layers: 6,
        heads: 16,
        kv_heads: 16,
        head_dim: 64,
        inter: 4096,
        rope_theta: 10000.0,
        eps: 1e-5,
    };
    println!(
        "\nDEPTH transformer: d_model {}, {} layers, {} heads, head_dim {}, ffn {}  (8 steps/frame)",
        depth.hidden, depth.layers, depth.heads, depth.head_dim, depth.inter
    );
    let dstack = SynthStack::<B>::build(&depth, 32, &device);
    let (dsub, dfull) = bench_depth_frame::<B>(&dstack, rounds, &device);
    let d_per_step_full = dfull / 8.0;
    println!(
        "  8-step frame: submit {dsub:6.2} ms/frame  full {dfull:6.2} ms/frame  ({d_per_step_full:.2} ms/step)"
    );

    // ---- Frame budget ----
    let mimi_allowance = 5.0; // rough allowance for Mimi decode (streaming conv)
    println!("\n--- FRAME BUDGET (80 ms @ 12.5 Hz) ---");
    for (ctx, sub, full) in &temporal_results {
        let total = full + dfull + mimi_allowance;
        let verdict = if total <= 80.0 {
            format!("CLEARS by {:.1} ms", 80.0 - total)
        } else {
            format!("OVER by {:.1} ms ({:.1}× budget)", total - 80.0, total / 80.0)
        };
        println!(
            "  ctx {ctx:>4}: temporal {full:5.2} + depth {dfull:5.2} + mimi ~{mimi_allowance:.0} = {total:5.2} ms/frame  →  {verdict}"
        );
        let _ = sub;
    }
    println!(
        "  (submit-only temporal @ ctx 3000 = {:.2} ms; gap to full = host-submit share)",
        temporal_results.last().map(|r| r.1).unwrap_or(0.0)
    );
}

fn main() {
    let rounds: usize = std::env::var("MOSHI_ROUNDS").ok().and_then(|s| s.parse().ok()).unwrap_or(7);
    println!("moshi_realtime_probe — PersonaPlex-7B (Moshi) temporal+depth realtime spike");
    println!("synthetic random weights, single-token DECODE regime, {rounds} rounds, min-of-medians");
    println!("budget: 80 ms/frame @ 12.5 Hz");

    // Backend selection via MOSHI_BACKEND (default: half = raw f16 Metal).
    //   half       — Metal<f16>, the plain (non-fused) f16 GPU path
    //   fused-half — BFusedHalf, the production fused f16 path (megakernel/fusion)
    //   f32        — Metal<f32>, raw f32 reference
    //   fused-f32  — BFused, fused f32 reference
    // The fused backends give the megakernel op-count win but burn-fusion 0.21
    // can miscompile deep decode graphs at these dims (GlobalArgsLaunch::strides
    // codegen panic) — the raw path is the robust measurement here.
    let backends: Vec<&str> = std::env::var("MOSHI_BACKEND")
        .map(|s| s.split(',').map(|x| Box::leak(x.trim().to_string().into_boxed_str()) as &str).collect())
        .unwrap_or_else(|_| vec!["half"]);
    for b in backends {
        match b {
            "half" => run_backend::<BHalf>("Metal<f16> (raw f16 Metal — robust path)", rounds),
            "fused-half" => run_backend::<BFusedHalf>("BFusedHalf (fused f16 Metal — production)", rounds),
            "f32" => run_backend::<BMetal>("Metal<f32> (raw f32 reference)", rounds),
            "fused-f32" => run_backend::<BFused>("BFused (fused f32 reference)", rounds),
            other => eprintln!("unknown backend '{other}', skipping"),
        }
    }

    println!("\nWHERE THE TIME GOES: compare submit vs full per step. If submit ≈ full,");
    println!("the GPU queue is the bottleneck (compute/bandwidth). If submit is a large");
    println!("fraction of a small full, it's host-submission-bound (the megakernel lever).");
}
