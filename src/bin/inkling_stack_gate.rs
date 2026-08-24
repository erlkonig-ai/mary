//! Parity gate for the Inkling text stack's ends, and for composing layers.
//!
//! Budget, written down first: worst absolute error over the tensor's own
//! scale, `1e-5` — the same criterion and reason as the other real-weight
//! gates.
//!
//! Four checks written to be non-vacuous rather than passed:
//!
//! 1. `embed_norm` must MOVE the embedding, or skipping it would pass. On the
//!    released weights it moves it by ~5.9 absolute.
//! 2. The muP division must change the logits, checked against a reference
//!    computed without it.
//! 3. The vocabulary truncation must actually drop columns (966 here); where
//!    `vocab_size == unpadded_vocab_size` it would be untested.
//! 4. The second layer must move the residual stream, or "composed two layers"
//!    is indistinguishable from having run one.
//!
//!   cargo run --release --features inkling --bin inkling_stack_gate -- <ckpt> <oracle>

use std::path::PathBuf;

use anyhow::{Context, Result};

use mary::models::inkling::attn::{AttnDims, AttnWeights, LogScaling, causal_mask};
use mary::models::inkling::block::rms_norm;
use mary::models::inkling::config::AttnKind;
use mary::models::inkling::layer::{LayerMlp, LayerWeights, decoder_layer};
use mary::models::inkling::load::{Checkpoint, split_gate_up};
use mary::models::inkling::stack::{embed_and_norm, head};

const BUDGET: f32 = 1e-5;

fn read_f32(p: &std::path::Path) -> Result<Vec<f32>> {
    let b = std::fs::read(p).with_context(|| format!("reading {}", p.display()))?;
    Ok(b.chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect())
}

fn num(text: &str, key: &str) -> Result<f64> {
    let pat = format!("\"{key}\"");
    let at = text
        .find(&pat)
        .with_context(|| format!("manifest has no {key}"))?;
    let rest = &text[at + pat.len()..];
    let colon = rest.find(':').context("malformed manifest")?;
    let s: String = rest[colon + 1..]
        .chars()
        .skip_while(|c| c.is_whitespace())
        .take_while(|c| c.is_ascii_digit() || *c == '.' || *c == '-' || *c == 'e')
        .collect();
    s.parse()
        .with_context(|| format!("{key} is not a number: {s:?}"))
}

struct D {
    abs: f32,
    scale: f32,
    rel: f32,
    n: usize,
}
impl D {
    fn scaled(&self) -> f32 {
        self.abs / self.scale.max(f32::MIN_POSITIVE)
    }
}
fn cmp(a: &[f32], b: &[f32]) -> D {
    let mut d = D {
        abs: 0.0,
        scale: 0.0,
        rel: 0.0,
        n: a.len().min(b.len()),
    };
    for (&x, &y) in a.iter().zip(b) {
        let e = (x - y).abs();
        d.abs = d.abs.max(e);
        d.scale = d.scale.max(y.abs());
        d.rel = d.rel.max(e / y.abs().max(1e-6));
    }
    d
}
fn report(label: &str, d: &D, checks: &mut usize, fails: &mut usize) {
    *checks += d.n;
    println!(
        "  {label}: {} values, worst abs {:e} / scale {:e} = {:e}, rel {:e}",
        d.n,
        d.abs,
        d.scale,
        d.scaled(),
        d.rel
    );
    if d.scaled() > BUDGET {
        println!("    FAIL  over budget {BUDGET:e}");
        *fails += 1;
    }
}

fn main() -> Result<()> {
    let ckpt = std::env::args()
        .nth(1)
        .map(PathBuf::from)
        .context("usage: <ckpt> <oracle>")?;
    let oracle = std::env::args()
        .nth(2)
        .map(PathBuf::from)
        .context("usage: <ckpt> <oracle>")?;
    let man = String::from_utf8(std::fs::read(oracle.join("stk_manifest.json"))?)?;

    let t = num(&man, "tokens")? as usize;
    let h = num(&man, "hidden")? as usize;
    let vocab = num(&man, "vocab_size")? as usize;
    let unpadded = num(&man, "unpadded_vocab_size")? as usize;
    let mup = num(&man, "logits_mup_width_multiplier")? as f32;
    let eps = num(&man, "rms_norm_eps")?;
    let di = num(&man, "dense_intermediate")? as usize;
    let window = num(&man, "sliding_window")? as usize;
    let kernel = num(&man, "kernel")? as usize;
    let heads = num(&man, "heads")? as usize;
    let kv_heads = num(&man, "kv_heads")? as usize;
    let head_dim = num(&man, "head_dim")? as usize;
    let d_rel = num(&man, "d_rel")? as usize;
    let rel_extent = num(&man, "rel_extent")? as usize;
    let n_floor = num(&man, "log_scaling_n_floor")? as f32;
    let alpha = num(&man, "log_scaling_alpha")? as f32;

    let cp = Checkpoint::open(&ckpt)?;
    println!("=== stack ===");
    println!("  tokens {t}  hidden {h}  vocab {vocab} -> unpadded {unpadded}  mup {mup}");
    println!("  tensors in index: {}", cp.len());
    let mut fails = 0usize;
    let mut checks = 0usize;

    // ---- embedding ---------------------------------------------------------
    println!("\n=== 1. embedding + embed_norm ===");
    let ids: Vec<usize> = std::fs::read(oracle.join("stk_ids.bin"))?
        .chunks_exact(8)
        .map(|c| i64::from_le_bytes(c.try_into().unwrap()) as usize)
        .collect();
    anyhow::ensure!(ids.len() == t, "{} ids for {t} tokens", ids.len());
    println!("  ids: {ids:?}");
    let table = cp.tensor("model.llm.embed.weight")?.data;
    let en = cp.tensor("model.llm.embed_norm.weight")?.data;
    let raw_ref = read_f32(&oracle.join("stk_raw_embed.bin"))?;
    let emb_ref = read_f32(&oracle.join("stk_inputs_embeds.bin"))?;
    anyhow::ensure!(
        !emb_ref.is_empty(),
        "no embedding reference — gate would be vacuous"
    );

    let raw = mary::models::inkling::stack::embed(&ids, &table, vocab, h);
    report("raw lookup", &cmp(&raw, &raw_ref), &mut checks, &mut fails);
    let embedded = embed_and_norm(&ids, &table, &en, eps, vocab, h);
    report(
        "after embed_norm",
        &cmp(&embedded, &emb_ref),
        &mut checks,
        &mut fails,
    );
    // Non-vacuity: the norm must do something.
    let moved = cmp(&embedded, &raw);
    checks += 1;
    println!(
        "  embed_norm moves the embedding by {:e} absolute",
        moved.abs
    );
    if moved.abs <= 0.0 {
        println!("  FAIL  embed_norm is inert here — skipping it would pass");
        fails += 1;
    }

    // ---- head --------------------------------------------------------------
    println!("\n=== 2. final norm, muP division, head, truncation ===");
    let fnorm = cp.tensor("model.llm.norm.weight")?.data;
    let unembed = cp.tensor("model.llm.unembed.weight")?.data;
    println!("  unembed is {} floats = [{vocab}, {h}]", unembed.len());
    checks += 1;
    if unembed.len() != vocab * h {
        println!("  FAIL  unembed is the wrong size");
        fails += 1;
    }
    let hin = read_f32(&oracle.join("stk_head_in.bin"))?;
    let logits_ref = read_f32(&oracle.join("stk_logits.bin"))?;
    let logits_nomup = read_f32(&oracle.join("stk_logits_nomup.bin"))?;
    anyhow::ensure!(!logits_ref.is_empty(), "no logits reference");

    let logits = head(&hin, &fnorm, &unembed, mup, vocab, unpadded, eps, t, h);
    report(
        "logits",
        &cmp(&logits, &logits_ref),
        &mut checks,
        &mut fails,
    );

    checks += 1;
    let dropped = vocab - unpadded;
    println!("  truncation drops {dropped} columns ({vocab} -> {unpadded})");
    if dropped == 0 {
        println!("  FAIL  nothing is dropped — the truncation is untested here");
        fails += 1;
    }
    let d_nomup = cmp(&logits, &logits_nomup);
    checks += 1;
    println!(
        "  vs logits computed WITHOUT the muP division: {:e} scale-relative",
        d_nomup.scaled()
    );
    if d_nomup.scaled() <= BUDGET {
        println!("  FAIL  the muP division changes nothing — dropping it would pass");
        fails += 1;
    } else {
        println!("  the muP division is genuinely under test (it divides by {mup})");
    }

    // ---- composition -------------------------------------------------------
    println!("\n=== 3. two layers composed, on the real embedding ===");
    let after0_ref = read_f32(&oracle.join("stk_after0.bin"))?;
    let after1_ref = read_f32(&oracle.join("stk_after1.bin"))?;
    let mask = causal_mask(t, Some(window));
    let dims = AttnDims {
        hidden: h,
        heads,
        kv_heads,
        head_dim,
        d_rel,
        rel_extent,
        kernel,
        rms_eps: eps,
        kind: AttnKind::Local,
    };
    let ls = LogScaling { n_floor, alpha };

    let mut hstate = embedded.clone();
    let mut afters: Vec<Vec<f32>> = Vec::new();
    for layer in 0..2usize {
        let p = format!("model.llm.layers.{layer}.");
        let g = |n: &str| -> Result<Vec<f32>> { Ok(cp.tensor(&format!("{p}{n}"))?.data) };
        let attn_norm = g("attn_norm.weight")?;
        let mlp_norm = g("mlp_norm.weight")?;
        let attn_sconv = g("attn_sconv.weight")?;
        let mlp_sconv = g("mlp_sconv.weight")?;
        let wq = g("attn.wq_du.weight")?;
        let wk = g("attn.wk_dv.weight")?;
        let wv = g("attn.wv_dv.weight")?;
        let wr = g("attn.wr_du.weight")?;
        let wo = g("attn.wo_ud.weight")?;
        let qn = g("attn.q_norm.weight")?;
        let kn = g("attn.k_norm.weight")?;
        let ks = g("attn.k_sconv.weight")?;
        let vs = g("attn.v_sconv.weight")?;
        let rp = g("attn.rel_logits_proj.proj")?;
        let fused = g("mlp.w13_dn.weight")?;
        let down = g("mlp.w2_md.weight")?;
        let gscale = g("mlp.global_scale")?;
        let (gate, up) = split_gate_up(&fused, h);

        let aw = AttnWeights {
            wq: &wq,
            wk: &wk,
            wv: &wv,
            wr: &wr,
            wo: &wo,
            k_sconv: &ks,
            v_sconv: &vs,
            q_norm: &qn,
            k_norm: &kn,
            rel_proj: &rp,
        };
        let lw = LayerWeights {
            attn_norm: &attn_norm,
            mlp_norm: &mlp_norm,
            attn_sconv: &attn_sconv,
            mlp_sconv: &mlp_sconv,
        };
        let mlp = LayerMlp {
            gate: &gate,
            up: &up,
            down: &down,
            global_scale: gscale[0],
            inter: di,
        };
        let out = decoder_layer(&hstate, &lw, &aw, &dims, Some(ls), &mlp, &mask, t);
        hstate = out;
        afters.push(hstate.clone());
    }

    report(
        "after layer 0",
        &cmp(&afters[0], &after0_ref),
        &mut checks,
        &mut fails,
    );
    report(
        "after layer 1",
        &cmp(&afters[1], &after1_ref),
        &mut checks,
        &mut fails,
    );
    let moved = cmp(&afters[1], &afters[0]);
    checks += 1;
    println!("  layer 1 moves the stream by {:e} absolute", moved.abs);
    if moved.abs <= 0.0 {
        println!("  FAIL  the second layer changes nothing — composition is untested");
        fails += 1;
    }

    println!("\n=== what this does not prove ===");
    println!("  the name->module mapping is authored on both sides; a shared misreading passes.");
    println!("  only two layers compose here, and both are dense — the reference holds the whole");
    println!("  stack in memory while mary pages experts, so a deeper stack is bounded by torch.");
    println!("  log scaling is inert at {t} tokens with n_floor {n_floor}.");

    println!("\n=== verdict ===");
    println!("  checks: {checks}");
    if fails == 0 {
        println!(
            "GATE PASSED — {checks} checks, the text stack's ends and two composed layers match"
        );
        Ok(())
    } else {
        println!("GATE FAILED — {checks} checks, {fails} FAILURES");
        std::process::exit(1);
    }
}
