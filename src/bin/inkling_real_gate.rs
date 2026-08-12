//! End-to-end gate: one REAL layer of the released checkpoint, in mary.
//!
//! Everything before this ran on random weights. This loads layer weights from
//! `Inkling-Small-NVFP4` through `layout` and `nvfp4`, runs mary's
//! `decoder_layer`, and compares against `transformers` running the same layer
//! on the same input.
//!
//! **What this proves and what it does not.** It exercises the shard plumbing,
//! the BF16 widening, the layout's names, and the whole layer's arithmetic
//! against an independent implementation. It does NOT independently establish
//! the checkpoint-name to module-parameter MAPPING: that mapping is authored on
//! both sides by the same person, so a shared misreading passes. Two things
//! narrow the gap — `load_state_dict(strict=True)` on the reference side makes
//! the mapping's totality machine-checked, and the per-tensor fingerprints here
//! localise any loading error to one tensor. The residual assumption is the
//! `w13` split ([gate; up] rather than [up; gate]), which no comparison of two
//! same-author lanes can falsify; a full forward producing coherent text would,
//! and that needs more memory than one machine here has.
//!
//! Budget, written down first: worst absolute error over the tensor's own
//! scale, `1e-5`. Looser than the random-weight gates because a real layer is
//! ~20k accumulations deep per output with weights spanning several orders of
//! magnitude, so f32 summation order matters more; the number is still reported
//! rather than asserted away.
//!
//!   cargo run --release --features inkling --bin inkling_real_gate -- <ckpt> <oracle>

use std::path::PathBuf;

use anyhow::{Context, Result};

use mary::models::inkling::attn::{causal_mask, AttnDims, AttnWeights, LogScaling};
use mary::models::inkling::config::AttnKind;
use mary::models::inkling::layer::{decoder_layer, LayerMlp, LayerWeights};
use mary::models::inkling::load::{split_gate_up, split_shared_w13, Checkpoint};

const BUDGET: f32 = 1e-5;

/// The settled reading of the SHARED experts' `shared_w13_weight`: INTERLEAVED.
///
/// Settled by A/B on a real forward with everything else held fixed
/// (`INK_SHARED_W13_HALVED` selects the other reading in `inkling_forward`),
/// inkling-small-nvfp4, prompt `"The capital of France is"`, routed experts on
/// the native NVFP4 lane:
///
/// ```text
///   INTERLEAVED  ' Paris'  ' a'  '...'  ' the'  ' $\'          top-1 18.69
///                continuation: "Paris. The capital of Germany is Berlin. …"
///   HALVED       '<|begin_of_text|>' '<|audio_end|>' '.' 'a' ' or'  top-1 8.94
///                continuation: special tokens, no English at all
/// ```
///
/// 9.75 logits is not numerical drift; it is the noise a swapped gate/up half
/// produces. Keep this constant and the toggle: a square tensor whose
/// orientation rests on an experiment nobody can re-run is one refactor from
/// drifting back.
const SHARED_W13_HALVED: bool = false;

fn read_f32(p: &std::path::Path) -> Result<Vec<f32>> {
    let b = std::fs::read(p).with_context(|| format!("reading {}", p.display()))?;
    Ok(b.chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect())
}

fn num(text: &str, key: &str) -> Result<f64> {
    let pat = format!("\"{key}\"");
    let at = text.find(&pat).with_context(|| format!("manifest has no {key}"))?;
    let rest = &text[at + pat.len()..];
    let colon = rest.find(':').context("malformed manifest")?;
    let s: String = rest[colon + 1..]
        .chars()
        .skip_while(|c| c.is_whitespace())
        .take_while(|c| c.is_ascii_digit() || *c == '.' || *c == '-' || *c == 'e')
        .collect();
    s.parse().with_context(|| format!("{key} is not a number: {s:?}"))
}

/// The fingerprint block for one tensor, located by its module name.
fn fingerprint(man: &str, key: &str, field: &str) -> Result<f64> {
    let at = man
        .find(&format!("\"{key}\""))
        .with_context(|| format!("no fingerprint for {key}"))?;
    let pat = format!("\"{field}\"");
    let rel = man[at..].find(&pat).with_context(|| format!("{key} has no {field}"))?;
    let rest = &man[at + rel + pat.len()..];
    let colon = rest.find(':').context("malformed fingerprint")?;
    let s: String = rest[colon + 1..]
        .chars()
        .skip_while(|c| c.is_whitespace())
        .take_while(|c| c.is_ascii_digit() || *c == '.' || *c == '-' || *c == 'e')
        .collect();
    s.parse().with_context(|| format!("{key}.{field} is not a number: {s:?}"))
}

fn check_fp(man: &str, key: &str, v: &[f32], checks: &mut usize, fails: &mut usize) {
    let sum: f64 = v.iter().map(|&x| x as f64).sum();
    let want = match fingerprint(man, key, "sum") {
        Ok(w) => w,
        Err(e) => {
            println!("  FAIL  {key}: {e}");
            *fails += 1;
            return;
        }
    };
    *checks += 1;
    // The sums are of millions of terms in a different order; compare relative
    // to the magnitude of the sum itself.
    let denom = want.abs().max(1.0);
    let rel = (sum - want).abs() / denom;
    if rel > 1e-4 {
        println!("  FAIL  {key}: sum {sum:+.6e}, reference {want:+.6e} (rel {rel:e})");
        *fails += 1;
    } else {
        println!("  ok    {:-42} sum {:+.6e}  n={}", key, sum, v.len());
    }
}

/// Which of the two readings of `shared_w13_weight` does the ORACLE agree with?
///
/// The block is square, so a wrong split is shape-legal, loads without
/// complaint, and computes nonsense; and the two readings have the SAME total
/// sum, so every fingerprint that does not separate the halves agrees with
/// both. A gate that simply splits one way and compares therefore certifies its
/// own assumption whenever the oracle shares it — which is exactly what
/// happened here: this gate halved, `capture_inkling_real.py` documented halved
/// while its code de-interleaved, and the shipped `inkling_oracle_fp4` manifest
/// records the halved sums.
///
/// So compute BOTH candidates and ask which one the oracle matches. That turns
/// a shared misreading from a green light into a named failure.
fn assert_shared_orientation(
    man: &str,
    fused: &[f32],
    n_shared: usize,
    inter: usize,
    hidden: usize,
    checks: &mut usize,
    fails: &mut usize,
) {
    let want = match fingerprint(man, "mlp.shared_experts.gate_proj", "sum") {
        Ok(w) => w,
        Err(e) => {
            println!("  FAIL  shared w13 orientation: {e}");
            *fails += 1;
            return;
        }
    };
    *checks += 1;
    let sum = |v: &[f32]| v.iter().map(|&x| x as f64).sum::<f64>();
    let si = sum(&split_shared_w13(fused, n_shared, inter, hidden, false).0);
    let sh = sum(&split_shared_w13(fused, n_shared, inter, hidden, true).0);
    let rel = |s: f64| (s - want).abs() / want.abs().max(1.0);
    println!(
        "  shared w13 gate-half sum: INTERLEAVED {si:+.6e}  HALVED {sh:+.6e}  oracle {want:+.6e}"
    );

    match (rel(si) <= 1e-4, rel(sh) <= 1e-4) {
        (true, true) => {
            println!(
                "  FAIL  both readings match the oracle — this tensor cannot discriminate them, \
                 so the orientation is NOT established here"
            );
            *fails += 1;
        }
        (false, false) => {
            println!(
                "  FAIL  neither reading matches the oracle — the shared w13 mapping is wrong in \
                 some third way"
            );
            *fails += 1;
        }
        (i_ok, _) => {
            let oracle_halved = !i_ok;
            if oracle_halved == SHARED_W13_HALVED {
                println!(
                    "  ok    oracle was captured {}, which is the reading a real forward settled",
                    if oracle_halved { "HALVED" } else { "INTERLEAVED" }
                );
            } else {
                println!(
                    "  FAIL  oracle was captured {}, but a real forward settled {} \
                     (' Paris' at top-1 logit 18.69 versus '<|begin_of_text|>' at 8.94).\n  \
                     \x20     The ORACLE is stale, not the loader: golden/capture_inkling_real.py \
                     already calls _deint for shared_w13 — it is that file's docstring and the \
                     manifest's \"w13_split\" string that still say halved. Re-run it to \
                     regenerate this oracle; until then this gate cannot certify the layer, \
                     because its reference layer was built with a swapped gate/up half.",
                    if oracle_halved { "HALVED" } else { "INTERLEAVED" },
                    if SHARED_W13_HALVED { "HALVED" } else { "INTERLEAVED" }
                );
                *fails += 1;
            }
        }
    }
}

fn main() -> Result<()> {
    let ckpt = std::env::args().nth(1).map(PathBuf::from).context("usage: <ckpt> <oracle>")?;
    let oracle = std::env::args().nth(2).map(PathBuf::from).context("usage: <ckpt> <oracle>")?;
    let man = String::from_utf8(std::fs::read(oracle.join("real_manifest.json"))?)?;

    let layer = num(&man, "layer")? as usize;
    let t = num(&man, "tokens")? as usize;
    let h = num(&man, "hidden")? as usize;
    let heads = num(&man, "heads")? as usize;
    let kv_heads = num(&man, "kv_heads")? as usize;
    let head_dim = num(&man, "head_dim")? as usize;
    let d_rel = num(&man, "d_rel")? as usize;
    let rel_extent = num(&man, "rel_extent")? as usize;
    let window = num(&man, "sliding_window")? as usize;
    let kernel = num(&man, "kernel")? as usize;
    let eps = num(&man, "rms_norm_eps")?;
    let di = num(&man, "dense_intermediate")? as usize;
    let n_floor = num(&man, "log_scaling_n_floor")? as f32;
    let alpha = num(&man, "log_scaling_alpha")? as f32;
    let is_sliding = man.contains("\"is_sliding\": true");

    let cp = Checkpoint::open(&ckpt)?;
    println!("=== checkpoint ===");
    println!("  dir              : {}", ckpt.display());
    println!("  tensors in index : {}", cp.len());
    println!("  layer {layer}: hidden {h}, heads {heads}/{kv_heads}x{head_dim}, {}",
             if is_sliding { "sliding" } else { "global" });
    anyhow::ensure!(cp.len() > 0, "empty index — the gate would be vacuous");

    let mut fails = 0usize;
    let mut checks = 0usize;

    let pfx = format!("model.llm.layers.{layer}.");
    let g = |n: &str| -> Result<Vec<f32>> { Ok(cp.tensor(&format!("{pfx}{n}"))?.data) };

    println!("\n=== 1. loaded tensors against the reference's fingerprints ===");
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

    check_fp(&man, "input_layernorm.weight", &attn_norm, &mut checks, &mut fails);
    check_fp(&man, "post_attention_layernorm.weight", &mlp_norm, &mut checks, &mut fails);
    check_fp(&man, "attn_sconv.conv1d.weight", &attn_sconv, &mut checks, &mut fails);
    check_fp(&man, "mlp_sconv.conv1d.weight", &mlp_sconv, &mut checks, &mut fails);
    check_fp(&man, "self_attn.q_proj.weight", &wq, &mut checks, &mut fails);
    check_fp(&man, "self_attn.k_proj.weight", &wk, &mut checks, &mut fails);
    check_fp(&man, "self_attn.v_proj.weight", &wv, &mut checks, &mut fails);
    check_fp(&man, "self_attn.r_proj.weight", &wr, &mut checks, &mut fails);
    check_fp(&man, "self_attn.o_proj.weight", &wo, &mut checks, &mut fails);
    check_fp(&man, "self_attn.q_norm.weight", &qn, &mut checks, &mut fails);
    check_fp(&man, "self_attn.k_norm.weight", &kn, &mut checks, &mut fails);
    check_fp(&man, "self_attn.k_sconv.conv1d.weight", &ks, &mut checks, &mut fails);
    check_fp(&man, "self_attn.v_sconv.conv1d.weight", &vs, &mut checks, &mut fails);
    check_fp(&man, "self_attn.rel_logits_proj.proj", &rp, &mut checks, &mut fails);
    let is_dense = man.contains("\"is_dense\": true");

    // These outlive the LayerMlp that borrows them.
    let (fused, dgate, dup, ddown, dscale);
    let (rw, rb, rg, sgate, sup, sdown, cgu, cdn);

    let mlp = if is_dense {
        fused = g("mlp.w13_dn.weight")?;
        ddown = g("mlp.w2_md.weight")?;
        dscale = g("mlp.global_scale")?;
        check_fp(&man, "mlp.down_proj.weight", &ddown, &mut checks, &mut fails);
        check_fp(&man, "mlp.global_scale", &dscale, &mut checks, &mut fails);

        println!("\n=== 2. the fused w13 split ===");
        let (a, b) = split_gate_up(&fused, h);
        dgate = a;
        dup = b;
        check_fp(&man, "mlp.gate_proj.weight", &dgate, &mut checks, &mut fails);
        check_fp(&man, "mlp.up_proj.weight", &dup, &mut checks, &mut fails);
        checks += 1;
        let gs: f64 = dgate.iter().map(|&x| x as f64).sum();
        let us: f64 = dup.iter().map(|&x| x as f64).sum();
        if (gs - us).abs() / gs.abs().max(1.0) < 1e-3 {
            println!("  FAIL  the halves have indistinguishable sums — a swap would pass");
            fails += 1;
        } else {
            println!("  halves differ (gate {gs:+.6e} vs up {us:+.6e}), so a swap is visible");
        }
        LayerMlp::Dense { gate: &dgate, up: &dup, down: &ddown, global_scale: dscale[0], inter: di }
    } else {
        let mi = num(&man, "moe_intermediate")? as usize;
        let n_routed = num(&man, "n_routed")? as usize;
        let n_shared = num(&man, "n_shared")? as usize;
        let top_k = num(&man, "top_k")? as usize;
        let route_scale = num(&man, "route_scale")? as f32;
        println!("\n=== 2. router, shared experts, and the orientation argument ===");
        println!("  moe_intermediate {mi}: note 2*{mi} = {} vs hidden {h}", 2 * mi);
        println!("  w2 is [experts, hidden, intermediate] and NON-square, which pins");
        println!("  the checkpoint's convention to [experts, out, in]; w13's squareness");
        println!("  is then not an open question but a consequence of that convention.");

        rw = g("mlp.gate.weight")?;
        rb = g("mlp.gate.bias")?;
        rg = g("mlp.gate.global_scale")?;
        check_fp(&man, "mlp.gate.weight", &rw, &mut checks, &mut fails);
        check_fp(&man, "mlp.gate.e_score_correction_bias", &rb, &mut checks, &mut fails);
        check_fp(&man, "mlp.gate.global_scale", &rg, &mut checks, &mut fails);
        checks += 1;
        if rw.len() != (n_routed + n_shared) * h {
            println!("  FAIL  router is {} floats, expected {}", rw.len(), (n_routed + n_shared) * h);
            fails += 1;
        } else {
            println!("  router is [{}+{}, {h}] — the shared experts have their own rows",
                     n_routed, n_shared);
        }

        // shared_w13 is [n_shared, 2*inter, hidden]; split each expert's block
        // on the OUT dimension. WHICH split is not a shape question — the block
        // is square — so it is settled by experiment and then asserted here.
        let sfused = g("mlp.shared_experts.shared_w13_weight")?;
        let (a, b) = split_shared_w13(&sfused, n_shared, mi, h, SHARED_W13_HALVED);
        sgate = a;
        sup = b;
        assert_shared_orientation(&man, &sfused, n_shared, mi, h, &mut checks, &mut fails);
        drop(sfused);
        sdown = g("mlp.shared_experts.shared_w2_weight")?;
        check_fp(&man, "mlp.shared_experts.gate_proj", &sgate, &mut checks, &mut fails);
        check_fp(&man, "mlp.shared_experts.up_proj", &sup, &mut checks, &mut fails);
        check_fp(&man, "mlp.shared_experts.down_proj", &sdown, &mut checks, &mut fails);

        println!("\n=== 3. expert slabs ===");
        println!("  which experts a layer uses is not knowable before its attention runs,");
        println!("  so all {n_routed} are fetched, one slab at a time out of the mapping.");
        let mut gu: Vec<f32> = Vec::with_capacity(n_routed * 2 * mi * h);
        let mut dv: Vec<f32> = Vec::with_capacity(n_routed * h * mi);
        for e in 0..n_routed {
            gu.extend_from_slice(&cp.expert_slice(&format!("{pfx}mlp.experts.w13_weight"), e)?.data);
            dv.extend_from_slice(&cp.expert_slice(&format!("{pfx}mlp.experts.w2_weight"), e)?.data);
        }
        cgu = gu;
        cdn = dv;
        println!("  gate_up {} floats, down {} floats", cgu.len(), cdn.len());
        checks += 2;
        if cgu.len() != n_routed * 2 * mi * h {
            println!("  FAIL  gate_up is {} floats, expected {}", cgu.len(), n_routed * 2 * mi * h);
            fails += 1;
        }
        if cdn.len() != n_routed * h * mi {
            println!("  FAIL  down is {} floats, expected {}", cdn.len(), n_routed * h * mi);
            fails += 1;
        }
        check_fp(&man, "mlp.experts.gate_up_proj", &cgu, &mut checks, &mut fails);
        check_fp(&man, "mlp.experts.down_proj", &cdn, &mut checks, &mut fails);

        LayerMlp::Sparse {
            router_weight: &rw, router_bias: &rb, router_global_scale: rg[0],
            route_scale, top_k, gate_up: &cgu, down: &cdn,
            shared_gate: &sgate, shared_up: &sup, shared_down: &sdown,
            experts: n_routed, n_shared, inter: mi,
        }
    };

    println!("\n=== 3. run the layer ===");
    let x = read_f32(&oracle.join("real_x.bin"))?;
    let y_ref = read_f32(&oracle.join("real_y.bin"))?;
    anyhow::ensure!(x.len() == t * h, "input is {} not {}", x.len(), t * h);
    anyhow::ensure!(!y_ref.is_empty(), "no reference output — the gate would be vacuous");

    let dims = AttnDims {
        hidden: h, heads, kv_heads, head_dim, d_rel, rel_extent, kernel,
        rms_eps: eps,
        kind: if is_sliding { AttnKind::Local } else { AttnKind::Global },
    };
    let aw = AttnWeights {
        wq: &wq, wk: &wk, wv: &wv, wr: &wr, wo: &wo,
        k_sconv: &ks, v_sconv: &vs, q_norm: &qn, k_norm: &kn, rel_proj: &rp,
    };
    let lw = LayerWeights {
        attn_norm: &attn_norm, mlp_norm: &mlp_norm,
        attn_sconv: &attn_sconv, mlp_sconv: &mlp_sconv,
    };
    let mask = causal_mask(t, if is_sliding { Some(window) } else { None });

    let (mine, routing) = decoder_layer(&x, &lw, &aw, &dims, Some(LogScaling { n_floor, alpha }), &mlp, &mask, t);

    // Compare the routing itself, so a routing disagreement is reported as one.
    if let Some(r) = &routing {
        let mut sel: Vec<usize> = r.iter().flat_map(|x| x.experts.clone()).collect();
        sel.sort_unstable();
        sel.dedup();
        println!("  the layer routed through {} distinct experts", sel.len());
        checks += 1;
        if sel.len() < 2 {
            println!("  FAIL  fewer than two experts used — per-expert indexing barely exercised");
            fails += 1;
        }
        if let Ok(bytes) = std::fs::read(oracle.join("real_topk_idx.bin")) {
            let refidx: Vec<i64> = bytes
                .chunks_exact(8)
                .map(|c| i64::from_le_bytes(c.try_into().unwrap()))
                .collect();
            let k = refidx.len() / t;
            let mut bad = 0usize;
            for ti in 0..t {
                checks += 1;
                let mut a: Vec<usize> = r[ti].experts.clone();
                let mut b: Vec<usize> =
                    refidx[ti * k..(ti + 1) * k].iter().map(|&v| v as usize).collect();
                a.sort_unstable();
                b.sort_unstable();
                if a != b {
                    if bad < 3 {
                        println!("  FAIL  token {ti} routed to {a:?}, reference {b:?}");
                    }
                    bad += 1;
                }
            }
            println!("  expert-set mismatches vs the reference: {bad} of {t} tokens");
            fails += bad;
        }
    }

    let mut worst_abs = 0f32;
    let mut scale = 0f32;
    let mut worst_rel = 0f32;
    for (&a, &b) in mine.iter().zip(&y_ref) {
        checks += 1;
        worst_abs = worst_abs.max((a - b).abs());
        scale = scale.max(b.abs());
        worst_rel = worst_rel.max((a - b).abs() / b.abs().max(1e-6));
    }
    let scaled = worst_abs / scale.max(f32::MIN_POSITIVE);
    println!("  values compared   : {}", mine.len());
    println!("  worst absolute    : {worst_abs:e}  (scale {scale:e})");
    println!("  worst abs / scale : {scaled:e}   <- the criterion, budget {BUDGET:e}");
    println!("  worst relative    : {worst_rel:e}   (reported, not gated)");
    if scaled > BUDGET {
        println!("  FAIL  over budget");
        fails += 1;
    }

    println!("\n=== what this does not prove ===");
    println!("  the checkpoint-name -> module mapping is authored on BOTH sides here,");
    println!("  so a shared misreading passes — as the w13 split order did, on both");
    println!("  sides, until a real forward settled it (INTERLEAVED). That one is now");
    println!("  asserted above rather than assumed; the rest of the mapping is not.");
    println!("  log scaling is inert at {t} tokens with n_floor {n_floor} — untested here.");

    println!("\n=== verdict ===");
    println!("  checks: {checks}");
    if fails == 0 {
        println!("GATE PASSED — {checks} checks, a real layer runs in mary and matches transformers");
        Ok(())
    } else {
        println!("GATE FAILED — {checks} checks, {fails} FAILURES");
        std::process::exit(1);
    }
}
