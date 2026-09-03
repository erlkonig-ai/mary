//! inkling_train_online -- S3 of the training path (compass 84605490).
//!
//! Prequential online learning on ONE MoE layer's routed experts, three arms on
//! the same sequence of real user turns: `none` (the checkpoint), `f32` (a master
//! copy updated by plain SGD -- the ceiling for this gradient), and `fp4sr` (the
//! update landed directly on the NVFP4 codes by unbiased stochastic rounding,
//! block scales frozen, no master weights, no optimizer state). For each turn k
//! every arm is SCORED on turn k before it LEARNS from turn k. The gate is the
//! running loss, arm against `none`, and nothing else.
//!
//! Layers below the trained one are identical across arms, so their forward runs
//! once per turn and is shared; the trained layer, the head and the backward run
//! per arm. Non-expert layer weights are cached on the host across turns.
//!
//! Env: INK_CKPT, INK_TURNS, INK_STEP_TURN (first corpus line), INK_ONLINE_TURNS
//! (default 8), INK_ONLINE_TOKENS (cap per turn, default 48), INK_ONLINE_MIN_TOKENS
//! (default 12), INK_ONLINE_LAYER (default the last layer), INK_ONLINE_LR (default 0.1),
//! INK_ONLINE_SEED (default 7).
use anyhow::{Context, Result, anyhow};
use burn::prelude::*;
use mary::models::inkling::block::Routing;
use mary::models::inkling::load::Checkpoint;
use mary::models::inkling::train::*;
use mary::models::inkling::train_online::*;
use std::time::Instant;

fn env_or<T: std::str::FromStr>(k: &str, d: T) -> T { std::env::var(k).ok().and_then(|s| s.parse().ok()).unwrap_or(d) }

struct Turn { line: usize, ids: Vec<usize>, scored_from: usize, n_text: usize }

fn main() -> Result<()> {
    let ckpt = std::env::var("INK_CKPT").unwrap_or_else(|_| format!("{}/models/thinkingmachines-inkling-small-nvfp4", std::env::var("HOME").unwrap()));
    let cp = Checkpoint::open(&ckpt).with_context(|| format!("open {ckpt}"))?;
    let t = InklingConfig::from_json(&std::fs::read_to_string(std::path::Path::new(&ckpt).join("config.json"))?)?.text_config;
    let k_turns: usize = env_or("INK_ONLINE_TURNS", 8);
    let cap: usize = env_or("INK_ONLINE_TOKENS", 48);
    let min_tokens: usize = env_or("INK_ONLINE_MIN_TOKENS", 12);
    let layer: usize = env_or("INK_ONLINE_LAYER", t.num_hidden_layers - 1);
    let lr: f32 = env_or("INK_ONLINE_LR", 0.1);
    let seed: u64 = env_or("INK_ONLINE_SEED", 7);
    let turn0: usize = env_or("INK_STEP_TURN", 0);
    let turns_path = std::env::var("INK_TURNS").unwrap_or_else(|_| "/tmp/claude-1000/-home-liora-liora/f03d61a4-efd1-4cac-982c-c155f6c0d5ab/scratchpad/userturns/user_turns.txt".into());
    anyhow::ensure!(!t.is_dense(layer) && layer < t.num_hidden_layers, "INK_ONLINE_LAYER {layer} must be a MoE layer");
    mem_guard("start");

    // --- turns, each rendered as the resident meets it
    let tok = tokenizers::Tokenizer::from_file(std::path::Path::new(&ckpt).join("tokenizer.json")).map_err(|e| anyhow!("tokenizer: {e}"))?;
    let special = |name: &str| -> Result<usize> { tok.token_to_id(name).map(|v| v as usize).ok_or_else(|| anyhow!("tokenizer lacks {name}")) };
    let (msg_system, msg_user, content_text, end_message) = (special("<|message_system|>")?, special("<|message_user|>")?, special("<|content_text|>")?, special("<|end_message|>")?);
    let effort: Vec<usize> = tok.encode("Thinking effort level: 0.9", false).map_err(|e| anyhow!("encode: {e}"))?.get_ids().iter().map(|&u| u as usize).collect();
    let corpus = std::fs::read_to_string(&turns_path).with_context(|| format!("read {turns_path}"))?;
    let mut turns: Vec<Turn> = Vec::new();
    for (i, line) in corpus.lines().enumerate().skip(turn0) {
        if turns.len() == k_turns { break; }
        let text: Vec<usize> = tok.encode(line, false).map_err(|e| anyhow!("encode: {e}"))?.get_ids().iter().map(|&u| u as usize).collect();
        if text.len() < min_tokens { continue; }
        let n_text = text.len().min(cap);
        let mut ids = vec![msg_system, content_text];
        ids.extend(&effort); ids.push(end_message); ids.extend([msg_user, content_text]);
        let scored_from = ids.len() - 1;
        ids.extend_from_slice(&text[..n_text]); ids.push(end_message);
        turns.push(Turn { line: i, ids, scored_from, n_text });
    }
    anyhow::ensure!(turns.len() == k_turns, "only {} usable turns from line {turn0}", turns.len());
    println!("=== inkling_train_online: layer {layer}, {k_turns} turns (cap {cap} text tokens), lr {lr}, seed {seed}, arms none/f32/fp4sr ===");
    for (k, tr) in turns.iter().enumerate() { println!("  turn {k}: corpus line {} , {} text tokens", tr.line, tr.n_text); }

    let dev = Dev::default();
    let hh = load_head(&cp, &t)?;
    let h = t.hidden_size; let eps = t.rms_norm_eps;
    mem_guard("head tables");

    // host cache of non-expert layer weights (experts are per turn)
    let mut cache: Vec<Option<HostW>> = (0..=layer).map(|_| None).collect();
    let mut rel_extent: Vec<usize> = vec![0; layer + 1];
    let mut arms = vec![Arm::new("none", ArmKind::None, 0.0, seed), Arm::new("f32", ArmKind::F32, lr, seed), Arm::new("fp4sr", ArmKind::Fp4Sr, lr, seed)];
    let mut losses: Vec<Vec<f32>> = vec![Vec::new(); arms.len()];
    let t_all = Instant::now();

    for (k, tr) in turns.iter().enumerate() {
        let n = tr.ids.len() - 1;
        let inputs = &tr.ids[..n]; let targets = &tr.ids[1..];
        let t_turn = Instant::now();
        let mut x: Vec<f32> = embed_host(&hh, inputs, &t);
        // ---- shared forward below the trained layer (identical in every arm)
        for l in 0..layer {
            if cache[l].is_none() { let (hw, g) = load_layer(&cp, &t, l, n)?; rel_extent[l] = g.rel_extent; cache[l] = Some(hw); }
            let g = geom_for(&t, l, n, rel_extent[l]);
            let mut hw = cache[l].clone().unwrap();
            set_router_global_scale(hw.rg);
            let mut dw: DevW<Bk> = build_dev(&dev, &hw, &g);
            let xin = c2::<Bk>(&dev, x.clone(), n, h);
            let routing = if g.is_dense { Vec::new() } else {
                let (_a, _b, logits) = pre_moe(&dev, &dw, &g, xin.clone());
                let lg: Vec<f32> = logits.unwrap().into_data().to_vec::<f32>().unwrap();
                select_routing(&lg, &hw.rb, &g)
            };
            load_experts(&cp, &g, &mut hw, &routing)?;
            bind_experts(&dev, &mut dw, &hw, &g);
            x = burn_layer(&dev, &dw, &g, xin, &routing).into_data().to_vec::<f32>().unwrap();
            drop(dw); drop(hw);
            pool_release(&dev);
        }
        let t_shared = t_turn.elapsed().as_secs_f64();
        // ---- trained layer: routing is shared (router and attention are not trained)
        if cache[layer].is_none() { let (hw, g) = load_layer(&cp, &t, layer, n)?; rel_extent[layer] = g.rel_extent; cache[layer] = Some(hw); }
        let g = geom_for(&t, layer, n, rel_extent[layer]);
        let mut base = cache[layer].clone().unwrap();
        set_router_global_scale(base.rg);
        let routing: Vec<Routing> = {
            let dw0: DevW<Bk> = build_dev(&dev, &base, &g);
            let (_a, _b, logits) = pre_moe(&dev, &dw0, &g, c2::<Bk>(&dev, x.clone(), n, h));
            let lg: Vec<f32> = logits.unwrap().into_data().to_vec::<f32>().unwrap();
            select_routing(&lg, &base.rb, &g)
        };
        load_experts(&cp, &g, &mut base, &routing)?;
        let base_no_experts = HostW { experts: Default::default(), ..base.clone() };
        let mut touched: Vec<usize> = routing.iter().flat_map(|r| r.experts.clone()).collect();
        touched.sort_unstable(); touched.dedup();
        let mut line = format!("turn {k:>2} ({:>3} scored):", tr.n_text + 1);
        for (a, arm) in arms.iter_mut().enumerate() {
            let t_arm = Instant::now();
            // this arm's view of the layer: the checkpoint's experts, overridden where the arm has
            // learned. Built straight onto the device -- no host clone of the layer per arm (the
            // clone cost ~4 GB per arm per turn and pushed turn 2 to 32 GiB free).
            let mut dw: DevW<Ad> = build_dev(&dev, &base_no_experts, &g);
            for &e in &touched {
                let (w13, w2) = match arm.current(e) { Some(w) => w, None => base.experts.get(&e).unwrap().clone() };
                dw.experts.insert(e, (t2(&dev, &w13, 2 * g.mi, g.h), t2(&dev, &w2, g.h, g.mi)));
            }
            let xin = c2::<Ad>(&dev, x.clone(), n, h);
            let y = burn_layer(&dev, &dw, &g, xin, &routing);
            let yv: Vec<f32> = y.clone().into_data().to_vec::<f32>().unwrap();
            // SCORE before LEARN
            let (loss, per_token, g_up) = head_loss(&dev, &hh, &yv, targets, n, h, eps, tr.scored_from);
            losses[a].push(loss);
            if arm.kind == ArmKind::None {
                let spelled: Vec<String> = targets[tr.scored_from..].iter().zip(&per_token[tr.scored_from..]).map(|(&id, nll)| format!("{:?}:{nll:.1}", tok.decode(&[id as u32], false).unwrap_or_default())).collect();
                println!("  turn {k} tokens with nll (checkpoint): {}", spelled.join(" "));
            }
            if arm.kind != ArmKind::None {
                let gout = c2::<Ad>(&dev, g_up, n, h);
                let grads = (y * gout).sum().backward();
                for &e in &touched {
                    let (w13, w2) = dw.experts.get(&e).unwrap();
                    let g13: Vec<f32> = w13.grad(&grads).expect("w13 grad").into_data().to_vec::<f32>().unwrap();
                    let g2: Vec<f32> = w2.grad(&grads).expect("w2 grad").into_data().to_vec::<f32>().unwrap();
                    arm.step(&cp, layer, e, base.experts.get(&e).unwrap(), &g13, &g2)?;
                }
                drop(grads);
            }
            drop(dw);
            pool_release(&dev);
            line.push_str(&format!("  {} {:.4} ({:.1}s, {:.0} GiB free)", arm.name, loss, t_arm.elapsed().as_secs_f64(), mem_available_gib()));
        }
        println!("{line}   [shared fwd {t_shared:.0}s, {} experts touched, MemAvailable {:.1} GiB]", touched.len(), mem_available_gib());
        mem_guard(&format!("after turn {k}"));
    }
    println!("=== {k_turns} turns in {:.0}s ===", t_all.elapsed().as_secs_f64());
    let mean = |v: &[f32]| v.iter().sum::<f32>() / v.len() as f32;
    let none = losses[0].clone();
    for (a, arm) in arms.iter().enumerate() {
        let per: Vec<String> = losses[a].iter().map(|v| format!("{v:.3}")).collect();
        let delta: f32 = losses[a].iter().zip(&none).map(|(x, y)| x - y).sum::<f32>() / none.len() as f32;
        let delta_after1: f32 = if none.len() > 1 { losses[a][1..].iter().zip(&none[1..]).map(|(x, y)| x - y).sum::<f32>() / (none.len() - 1) as f32 } else { 0.0 };
        println!("  {:<6} mean {:.4}  vs none {:+.4} (turns 1.. {:+.4})  per turn: {}{}", arm.name, mean(&losses[a]), delta, delta_after1, per.join(" "),
            if arm.kind == ArmKind::Fp4Sr { format!("   [codes changed {}, clipped {}]", arm.codes_changed, arm.codes_clipped) } else { String::new() });
    }
    println!("  prequential gate: an arm WINS if its 'vs none' over turns 1.. is negative on turns it did not learn from yet");
    Ok(())
}
