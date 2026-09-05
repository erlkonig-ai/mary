//! inkling_train_step -- stage 2 of the training path (compass 84605490).
//!
//! One full-stack forward + backward of Inkling-Small on ONE box, streaming one
//! layer's weights at a time. The loss is next-token cross-entropy on a REAL user
//! turn through the real head; the gradient is pushed down the stack layer by
//! layer and consumed as it is produced. No update yet (S3).
//!
//! Why this shape: there is no training phase. In the resident design the serving
//! prefill of a user turn IS the forward, so all a step must keep from it is each
//! layer's residual input plus its routing (16 KiB per token per layer at f32).
//! The backward is the only extra work, and it is preemptible at every layer
//! boundary -- its entire state is the upstream gradient (T x hidden) and a layer
//! index -- so it can be sliced between decode steps of the reply. This binary
//! measures that extra work with load and compute timed separately, because the
//! per-layer weight LOAD here is prototype cost only: a resident model already
//! holds its weights.
//!
//! Env: INK_CKPT, INK_STEP_TOKENS (default 16), INK_STEP_TURN (corpus line index,
//! default 0; the first turn at or after it with enough tokens is used),
//! INK_TURNS (corpus path), INK_STEP_LAYERS "a:b" (default the whole stack).
use anyhow::{Context, Result, anyhow};
use burn::prelude::*;
use mary::models::inkling::load::Checkpoint;
use mary::models::inkling::train::*;
use std::time::Instant;

fn env_or<T: std::str::FromStr>(k: &str, d: T) -> T { std::env::var(k).ok().and_then(|s| s.parse().ok()).unwrap_or(d) }

fn main() -> Result<()> {
    let ckpt = std::env::var("INK_CKPT").unwrap_or_else(|_| format!("{}/models/thinkingmachines-inkling-small-nvfp4", std::env::var("HOME").unwrap()));
    let cp = Checkpoint::open(&ckpt).with_context(|| format!("open {ckpt}"))?;
    let t = InklingConfig::from_json(&std::fs::read_to_string(std::path::Path::new(&ckpt).join("config.json"))?)?.text_config;
    let n: usize = env_or("INK_STEP_TOKENS", 16);
    let turn0: usize = env_or("INK_STEP_TURN", 0);
    let turns_path = std::env::var("INK_TURNS").expect("INK_TURNS names the user-turn corpus file");
    let layers_env = std::env::var("INK_STEP_LAYERS").unwrap_or_else(|_| format!("0:{}", t.num_hidden_layers));
    let (l0, l1) = { let mut it = layers_env.split(':'); (it.next().unwrap().parse::<usize>()?, it.next().unwrap().parse::<usize>()?) };
    anyhow::ensure!(l0 < l1 && l1 <= t.num_hidden_layers, "bad INK_STEP_LAYERS {layers_env}");
    let full_stack = l0 == 0 && l1 == t.num_hidden_layers;
    mem_guard("start");

    // --- a real user turn, tokenized by the checkpoint's own tokenizer
    let tok = tokenizers::Tokenizer::from_file(std::path::Path::new(&ckpt).join("tokenizer.json")).map_err(|e| anyhow!("tokenizer: {e}"))?;
    let corpus = std::fs::read_to_string(&turns_path).with_context(|| format!("read {turns_path}"))?;
    let mut chosen: Option<(usize, Vec<usize>)> = None;
    for (i, line) in corpus.lines().enumerate().skip(turn0) {
        let enc = tok.encode(line, false).map_err(|e| anyhow!("encode: {e}"))?;
        let ids: Vec<usize> = enc.get_ids().iter().map(|&u| u as usize).collect();
        if ids.len() >= n + 1 { chosen = Some((i, ids)); break; }
    }
    let (turn_idx, text_ids) = chosen.ok_or_else(|| anyhow!("no turn at or after line {turn0} has {} tokens", n + 1))?;
    // The turn as the resident meets it: the template's effort line, then the user message
    // markers, then the text, then the end marker. Only the text and its end marker are scored.
    let special = |name: &str| -> Result<usize> { tok.token_to_id(name).map(|v| v as usize).ok_or_else(|| anyhow!("tokenizer lacks {name}")) };
    let (msg_system, msg_user, content_text, end_message) = (special("<|message_system|>")?, special("<|content_text|>")?, special("<|content_text|>")?, special("<|end_message|>")?);
    let msg_user = special("<|message_user|>")?; let _ = msg_system; let _ = content_text;
    let effort = tok.encode("Thinking effort level: 0.9", false).map_err(|e| anyhow!("encode: {e}"))?;
    let mut ids: Vec<usize> = vec![special("<|message_system|>")?, special("<|content_text|>")?];
    ids.extend(effort.get_ids().iter().map(|&u| u as usize));
    ids.push(end_message);
    ids.extend([msg_user, special("<|content_text|>")?]);
    let scored_from = ids.len() - 1; // the position that predicts the first text token
    ids.extend_from_slice(&text_ids[..n]);
    ids.push(end_message);
    let total = ids.len() - 1; // positions
    let inputs = ids[..total].to_vec();
    let targets = ids[1..].to_vec();
    let n_scored = total - scored_from;
    let h = t.hidden_size;
    let eps = t.rms_norm_eps;
    println!("=== inkling_train_step: layers {l0}..{l1} of {}, user turn #{turn_idx}: {n} text tokens (+1 end marker) scored, {scored_from} template positions conditioned on, {total} positions total ===", t.num_hidden_layers);
    let n = total; // every layer sees all positions
    println!("  MemAvailable at start: {:.1} GiB", mem_available_gib());

    let dev = Dev::default();
    println!("  device pool at start: {}", pool_release(&dev));
    let t_head = Instant::now();
    let hh = load_head(&cp, &t)?;
    println!("  head tables (embed {} x {h}, unembed) loaded in {:.1}s; MemAvailable {:.1} GiB", hh.vocab, t_head.elapsed().as_secs_f64(), mem_available_gib());
    mem_guard("head tables");
    let mut min_avail = mem_available_gib();

    // ---------------------------------------------------------------- forward
    // Stand-in for the serving prefill: keep each layer's residual input and routing, nothing else.
    let mut x: Vec<f32> = embed_host(&hh, &inputs, &t);
    let mut layer_inputs: Vec<Vec<f32>> = Vec::with_capacity(l1 - l0);
    let mut layer_routing: Vec<Vec<mary::models::inkling::block::Routing>> = Vec::with_capacity(l1 - l0);
    let (mut fwd_load, mut fwd_compute) = (0f64, 0f64);
    for layer in l0..l1 {
        let tl = Instant::now();
        let (mut hw, g) = load_layer(&cp, &t, layer, n)?;
        set_router_global_scale(hw.rg);
        // graph-free forward: plain CUDA backend, nothing retained but the residual input and routing
        let mut dw: DevW<Bk> = build_dev(&dev, &hw, &g);
        let xin = c2::<Bk>(&dev, x.clone(), n, h);
        let routing = if g.is_dense { Vec::new() } else {
            let (_x1, _hn2, logits) = pre_moe(&dev, &dw, &g, xin.clone());
            let lg: Vec<f32> = logits.expect("router logits").into_data().to_vec::<f32>().unwrap();
            select_routing(&lg, &hw.rb, &g)
        };
        let n_exp = load_experts(&cp, &g, &mut hw, &routing)?;
        bind_experts(&dev, &mut dw, &hw, &g);
        let load_s = tl.elapsed().as_secs_f64();
        let tc = Instant::now();
        let y = burn_layer(&dev, &dw, &g, xin, &routing);
        let yv: Vec<f32> = y.into_data().to_vec::<f32>().unwrap();
        let compute_s = tc.elapsed().as_secs_f64();
        anyhow::ensure!(yv.iter().all(|v| v.is_finite()), "layer {layer} produced non-finite output");
        let rms = (yv.iter().map(|v| (v * v) as f64).sum::<f64>() / yv.len() as f64).sqrt();
        println!("  fwd L{layer:02} {:<6} experts {n_exp:>3}  load {load_s:>6.2}s  compute {compute_s:>5.2}s  |y|rms {rms:.3}  MemAvailable {:.1} GiB",
            if g.is_dense { "dense" } else { "moe" }, mem_available_gib());
        fwd_load += load_s; fwd_compute += compute_s;
        layer_inputs.push(std::mem::replace(&mut x, yv));
        layer_routing.push(routing);
        drop(dw); drop(hw);
        let pool = pool_release(&dev);
        min_avail = min_avail.min(mem_available_gib());
        println!("         after release: MemAvailable {:.1} GiB, {pool}", mem_available_gib());
        mem_guard(&format!("after fwd layer {layer}"));
    }

    // ---------------------------------------------------------------- head: the loss that matters
    let t_loss = Instant::now();
    let (loss, per_token, mut g_up) = if full_stack {
        head_loss(&dev, &hh, &x, &targets, n, h, eps, scored_from)
    } else {
        // partial stack: no meaningful loss; push a unit-scale random gradient so the backward still runs
        println!("  (partial stack: loss is NOT the model's; backward driven by a fixed pseudo-gradient)");
        let g: Vec<f32> = (0..n * h).map(|i| ((i * 2654435761usize) % 1000) as f32 / 1000.0 - 0.5).collect();
        (f32::NAN, vec![], g)
    };
    let loss_s = t_loss.elapsed().as_secs_f64();
    if full_stack {
        let ppl = (loss as f64).exp();
        println!("  LOSS {loss:.4} nats/token over {n_scored} scored user-turn targets (ppl {ppl:.1}), head fwd+bwd {loss_s:.2}s");
        let shown: Vec<String> = per_token[scored_from..].iter().take(12).map(|v| format!("{v:.2}")).collect();
        println!("  per-token nll of the scored tokens (first 12): {}", shown.join(" "));
        let spelled: Vec<String> = targets[scored_from..].iter().zip(&per_token[scored_from..]).map(|(&id, nll)| format!("{:?}:{nll:.1}", tok.decode(&[id as u32], false).unwrap_or_default())).collect();
        println!("  tokens with nll: {}", spelled.join(" "));
        let mut sorted: Vec<f32> = per_token[scored_from..].to_vec(); sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
        println!("  scored nll median {:.2}, p90 {:.2}, max {:.2}", sorted[sorted.len() / 2], sorted[sorted.len() * 9 / 10], sorted[sorted.len() - 1]);
    }
    let gnorm = |v: &[f32]| (v.iter().map(|x| (x * x) as f64).sum::<f64>()).sqrt();
    println!("  |dL/dh_top| = {:.3e}", gnorm(&g_up));

    // ---------------------------------------------------------------- backward, top down, one layer resident at a time
    let (mut bwd_load, mut bwd_compute) = (0f64, 0f64);
    let mut total_wgrad_sq = 0f64;
    for layer in (l0..l1).rev() {
        let i = layer - l0;
        let tl = Instant::now();
        let (mut hw, g) = load_layer(&cp, &t, layer, n)?;
        set_router_global_scale(hw.rg);
        let n_exp = load_experts(&cp, &g, &mut hw, &layer_routing[i])?;
        let dw: DevW<Ad> = build_dev(&dev, &hw, &g);
        let load_s = tl.elapsed().as_secs_f64();
        let tc = Instant::now();
        let xin = c2::<Ad>(&dev, layer_inputs[i].clone(), n, h).require_grad();
        let y = burn_layer(&dev, &dw, &g, xin.clone(), &layer_routing[i]);
        let gout = c2::<Ad>(&dev, g_up.clone(), n, h);
        let pseudo = (y * gout).sum();
        let grads = pseudo.backward();
        let gin: Vec<f32> = xin.grad(&grads).expect("input gradient").into_data().to_vec::<f32>().unwrap();
        // weight-gradient norms, consumed here and never stored
        let attn_sq: f64 = [grad_norm(&dw.wq, &grads), grad_norm(&dw.wk, &grads), grad_norm(&dw.wv, &grads), grad_norm(&dw.wr, &grads), grad_norm(&dw.wo, &grads),
            grad_norm(&dw.ks, &grads), grad_norm(&dw.vs, &grads), grad_norm(&dw.rp, &grads), grad_norm(&dw.qn, &grads), grad_norm(&dw.kn, &grads),
            grad_norm(&dw.attn_norm, &grads), grad_norm(&dw.attn_sconv, &grads)].iter().map(|v| (*v as f64).powi(2)).sum();
        let mut mlp_sq: f64 = [grad_norm(&dw.mlp_norm, &grads), grad_norm(&dw.mlp_sconv, &grads)].iter().map(|v| (*v as f64).powi(2)).sum();
        if let Some(dd) = &dw.dense {
            mlp_sq += [grad_norm(&dd.gate, &grads), grad_norm(&dd.up, &grads), grad_norm(&dd.down, &grads)].iter().map(|v| (*v as f64).powi(2)).sum::<f64>();
        } else {
            mlp_sq += (grad_norm(&dw.rw, &grads) as f64).powi(2);
            for s in 0..dw.sgate.len() { mlp_sq += [grad_norm(&dw.sgate[s], &grads), grad_norm(&dw.sup[s], &grads), grad_norm(&dw.sdown[s], &grads)].iter().map(|v| (*v as f64).powi(2)).sum::<f64>(); }
            for (w13, w2) in dw.experts.values() { mlp_sq += (grad_norm(w13, &grads) as f64).powi(2) + (grad_norm(w2, &grads) as f64).powi(2); }
        }
        drop(grads);
        let compute_s = tc.elapsed().as_secs_f64();
        anyhow::ensure!(gin.iter().all(|v| v.is_finite()), "layer {layer} produced a non-finite input gradient");
        total_wgrad_sq += attn_sq + mlp_sq;
        println!("  bwd L{layer:02} {:<6} experts {n_exp:>3}  load {load_s:>6.2}s  compute {compute_s:>5.2}s  |g_in| {:.3e}  |g_attn| {:.3e}  |g_mlp| {:.3e}  MemAvailable {:.1} GiB",
            if g.is_dense { "dense" } else { "moe" }, gnorm(&gin), attn_sq.sqrt(), mlp_sq.sqrt(), mem_available_gib());
        bwd_load += load_s; bwd_compute += compute_s;
        g_up = gin;
        drop(dw); drop(hw);
        let pool = pool_release(&dev);
        min_avail = min_avail.min(mem_available_gib());
        println!("         after release: MemAvailable {:.1} GiB, {pool}", mem_available_gib());
        mem_guard(&format!("after bwd layer {layer}"));
    }
    println!("=== step done: layers {l0}..{l1}, {n} positions ({n_scored} scored); loss {loss:.4}; |grad all weights| {:.3e}; |dL/dx_embed| {:.3e}", total_wgrad_sq.sqrt(), gnorm(&g_up));
    println!("    forward  load {fwd_load:.1}s  compute {fwd_compute:.1}s   (load is prototype-only: a resident model holds its weights)");
    println!("    backward load {bwd_load:.1}s  compute {bwd_compute:.1}s   (compute is the extra work a live turn costs; preemptible per layer)");
    println!("    per-layer backward compute mean {:.2}s; lowest MemAvailable seen {min_avail:.1} GiB", bwd_compute / (l1 - l0) as f64);
    Ok(())
}
