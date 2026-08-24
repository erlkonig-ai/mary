//! Parity gate for the Inkling MTP blocks — and an explicit statement of the
//! part that has no oracle.
//!
//! `transformers` ships the MTP config surface and no implementation:
//! `_keys_to_ignore_on_load_unexpected` discards every `model.mtp.*` weight on
//! load, and no class in the package uses `input_proj` or `hidden_norm`. So the
//! blocks can be checked and the composition cannot.
//!
//! What IS gated: each MTP layer's `transformer_block.*` is shape-identical to
//! an ordinary decoder layer, so mary's `decoder_layer` runs on real MTP
//! weights against the reference loaded with the same weights. MTP layer 0 is
//! sliding and layer 1 is global, so both attention kinds are covered — they
//! differ in `rel_extent` (512 against 1024) and in whether log scaling applies.
//!
//! What is NOT gated, and is reported rather than quietly skipped: how the
//! wrapper composes. `input_proj` is `[hidden, 2 * hidden]` and
//! `mtp_hidden_states_first` is true, which reads as "hidden state and
//! embedding concatenated, hidden first" — but that is a flag, not an observed
//! computation, and nothing upstream defines it. The wrapper tensors are
//! fingerprinted so their presence and identity are confirmed; their SEMANTICS
//! are not, and this gate says so instead of implying otherwise.
//!
//! Budget, written down first: worst absolute error over the tensor's own
//! scale, `1e-5`.
//!
//!   cargo run --release --features inkling --bin inkling_mtp_gate -- <ckpt> <oracle>

use std::path::PathBuf;

use anyhow::{Context, Result};

use mary::models::inkling::attn::{AttnDims, AttnWeights, LogScaling, causal_mask};
use mary::models::inkling::config::AttnKind;
use mary::models::inkling::layer::{LayerMlp, LayerWeights, decoder_layer};
use mary::models::inkling::load::{Checkpoint, split_gate_up};

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

fn num_after(text: &str, key: &str, from: usize) -> Result<f64> {
    let pat = format!("\"{key}\"");
    let at = text[from..]
        .find(&pat)
        .with_context(|| format!("no {key} after {from}"))?
        + from;
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

fn main() -> Result<()> {
    let ckpt = std::env::args()
        .nth(1)
        .map(PathBuf::from)
        .context("usage: <ckpt> <oracle>")?;
    let oracle = std::env::args()
        .nth(2)
        .map(PathBuf::from)
        .context("usage: <ckpt> <oracle>")?;
    let man = String::from_utf8(std::fs::read(oracle.join("mtp_manifest.json"))?)?;

    let t = num(&man, "tokens")? as usize;
    let h = num(&man, "hidden")? as usize;
    let n_mtp = num(&man, "n_mtp")? as usize;
    let kernel = num(&man, "kernel")? as usize;
    let eps = num(&man, "rms_norm_eps")?;
    let di = num(&man, "dense_intermediate")? as usize;
    let d_rel = num(&man, "d_rel")? as usize;
    let window = num(&man, "sliding_window")? as usize;
    let n_floor = num(&man, "log_scaling_n_floor")? as f32;
    let alpha = num(&man, "log_scaling_alpha")? as f32;

    let cp = Checkpoint::open(&ckpt)?;
    println!("=== MTP ===");
    println!("  {n_mtp} MTP layers, hidden {h}, tokens {t}");
    println!("  tensors in index: {}", cp.len());

    let x = read_f32(&oracle.join("mtp_x.bin"))?;
    anyhow::ensure!(x.len() == t * h, "input is {} not {}", x.len(), t * h);

    let mut fails = 0usize;
    let mut checks = 0usize;
    let mut kinds: Vec<(String, usize)> = Vec::new();

    for tag in ["local", "global"] {
        let at = man
            .find(&format!("\"{tag}\""))
            .context("manifest lacks the layer")?;
        let idx = num_after(&man, "mtp_index", at)? as usize;
        let heads = num_after(&man, "num_heads", at)? as usize;
        let kv_heads = num_after(&man, "num_kv_heads", at)? as usize;
        let head_dim = num_after(&man, "head_dim", at)? as usize;
        let rel_extent = num_after(&man, "rel_extent", at)? as usize;
        let is_local = tag == "local";
        kinds.push((tag.to_string(), rel_extent));

        println!(
            "\n=== MTP layer {idx} ({tag}): heads {heads}/{kv_heads}x{head_dim}, rel_extent {rel_extent} ==="
        );
        let p = format!("model.mtp.layers.{idx}.transformer_block.");
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

        // The relative table's width is the layer kind, observable on real
        // weights: the window on a local layer, rel_extent on a global one.
        checks += 1;
        let want_rel = d_rel * rel_extent;
        if rp.len() != want_rel {
            println!(
                "  FAIL  rel_logits_proj is {} floats, expected {d_rel}x{rel_extent}",
                rp.len()
            );
            fails += 1;
        } else {
            println!(
                "  rel_logits_proj is [{d_rel}, {rel_extent}] — the table spans the layer's reach"
            );
        }

        let (gate, up) = split_gate_up(&fused, h);
        let dims = AttnDims {
            hidden: h,
            heads,
            kv_heads,
            head_dim,
            d_rel,
            rel_extent,
            kernel,
            rms_eps: eps,
            kind: if is_local {
                AttnKind::Local
            } else {
                AttnKind::Global
            },
        };
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
        let mask = causal_mask(t, if is_local { Some(window) } else { None });
        let mine = decoder_layer(
            &x,
            &lw,
            &aw,
            &dims,
            Some(LogScaling { n_floor, alpha }),
            &mlp,
            &mask,
            t,
        );

        let y_ref = read_f32(&oracle.join(format!("mtp_{tag}_y.bin")))?;
        anyhow::ensure!(
            !y_ref.is_empty(),
            "{tag}: no reference output — the gate would be vacuous"
        );
        let mut abs = 0f32;
        let mut scale = 0f32;
        let mut rel = 0f32;
        for (&a, &b) in mine.iter().zip(&y_ref) {
            checks += 1;
            abs = abs.max((a - b).abs());
            scale = scale.max(b.abs());
            rel = rel.max((a - b).abs() / b.abs().max(1e-6));
        }
        let scaled = abs / scale.max(f32::MIN_POSITIVE);
        println!("  values compared   : {}", mine.len());
        println!("  worst abs / scale : {scaled:e}  (abs {abs:e}, scale {scale:e})");
        println!("  worst relative    : {rel:e}   (reported, not gated)");
        if scaled > BUDGET {
            println!("  FAIL  over budget {BUDGET:e}");
            fails += 1;
        }
    }

    // The two kinds must genuinely differ, or gating both proved one thing twice.
    checks += 1;
    println!("\n=== are the two MTP kinds distinguishable? ===");
    println!("  local rel_extent {} vs global {}", kinds[0].1, kinds[1].1);
    if kinds[0].1 == kinds[1].1 {
        println!("  FAIL  both kinds have the same reach — gating both tested one thing twice");
        fails += 1;
    }

    // ---- the wrapper: present, identified, and NOT interpreted -------------
    println!("\n=== wrapper tensors (no reference consumes these) ===");
    let mut present = 0usize;
    for i in 0..n_mtp {
        for nm in [
            "embed_norm.weight",
            "hidden_norm.weight",
            "input_proj.weight",
        ] {
            checks += 1;
            let name = format!("model.mtp.layers.{i}.{nm}");
            match cp.tensor(&name) {
                Ok(v) => {
                    present += 1;
                    if i == 0 {
                        println!("  layer 0 {nm}: {:?}", v.shape);
                    }
                }
                Err(e) => {
                    println!("  FAIL  {name}: {e}");
                    fails += 1;
                }
            }
        }
    }
    println!("  wrapper tensors found: {present} of {}", n_mtp * 3);
    checks += 1;
    let ip = cp.tensor("model.mtp.layers.0.input_proj.weight")?;
    if ip.shape != vec![h, 2 * h] {
        println!(
            "  FAIL  input_proj is {:?}, expected [{h}, {}]",
            ip.shape,
            2 * h
        );
        fails += 1;
    } else {
        println!(
            "  input_proj is [{h}, {}] — two hidden-width vectors concatenated on the input side",
            2 * h
        );
    }

    println!("\n=== what this does not prove ===");
    println!("  transformers ships the MTP config surface and NO implementation: every");
    println!("  model.mtp.* weight is discarded on load, and no class uses input_proj or");
    println!("  hidden_norm. So the BLOCKS are checked above against a real reference, and");
    println!("  the COMPOSITION has no oracle at all. mtp_hidden_states_first is true and");
    println!("  input_proj takes 2*hidden, which reads as hidden-then-embedding — that is a");
    println!("  flag, not an observed computation, and mary does not implement a guess at it.");
    println!("  The checkpoint-name -> module mapping is also authored on both sides here.");

    println!("\n=== verdict ===");
    println!("  checks: {checks}");
    if fails == 0 {
        println!(
            "GATE PASSED — {checks} checks, MTP blocks match transformers on both attention kinds"
        );
        Ok(())
    } else {
        println!("GATE FAILED — {checks} checks, {fails} FAILURES");
        std::process::exit(1);
    }
}
