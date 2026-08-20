//! PersonaPlex-7B LM parity gates — CPU-f32 Rust port vs the moshi CPU-f32
//! oracle goldens captured by `golden/capture_personaplex.py` (under
//! /tmp/mary-personaplex/golden, deterministic/greedy). Mirrors the
//! mimi_probe discipline: per-component cos gates, honest report.
//!
//!   cargo run --release --features personaplex --bin personaplex_probe -- \
//!     <temporal|depth|e2e> [pile-path]
//!
//! The parity window is the oracle's full 113-step stream: 50 voice-prompt
//! steps (pre-recorded embeddings), 6 silence, 26 text-prompt, 6 silence,
//! 25 user-audio steps (2 s of ref_voice.wav as Mimi codes).
//!
//! - `temporal` — LM part 1: per-step hidden/text-logits cos ≥ 0.99999 for
//!   the 7B temporal transformer (recomputes all 113 steps, ~26 GiB f32).
//! - `depth`  — LM part 2, teacher-forced: drives the full `StreamCache`
//!   delay bookkeeping from goldens (temporal hidden + text-argmax replayed
//!   from `tt_hidden`/`tt_text_logits`, depformer inputs pinned to the
//!   oracle trajectory via `dep_tokens`), gates the 113 × 16 per-codebook
//!   logit rows (cos ≥ 0.99999) + argmax rate vs `dep_logits`/`dep_tokens`,
//!   `next_text_token` vs `dep_in_text`, and the undelayed output frames vs
//!   `out_tokens` (integer-exact). Loads ONLY the depth weights — fast.
//! - `e2e`    — LM parts 1+2 free-running: the real `LmGen` step machine
//!   (temporal + depth + bookkeeping) fed exactly what the oracle was fed
//!   (voice-prompt embeddings, silence/sine frames, text prompt tokens,
//!   `user_codes`); gates the assembled model inputs vs `step_tokens`, the
//!   depformer tokens vs `dep_tokens`, and the output stream vs
//!   `out_tokens` — ALL integer-exact (greedy).
//! - `pipeline` — LM part 3, the full audio chain through `VoicePipeline`:
//!   input WAV → Mimi encode (gated integer-exact vs `user_codes`) → prompt
//!   phases → LM free-run (out frames gated vs `out_tokens`) → agent streams
//!   1..=8 → Mimi decode → 24 kHz PCM, gated (cos ≥ 0.999) against the
//!   oracle's streaming per-frame decode `out_audio` (cross-checked against
//!   `out_audio_batch`), and written to /tmp/mary-personaplex/
//!   pipeline_out.wav for listening.
//! - `prompt` — Phase 5, the prompt machinery from PRIMARY sources: mary's
//!   pure-Rust SPM tokenizer vs `text_prompt_tokens` (26/26) + the oracle
//!   battery (`spm_battery.json`, `golden/capture_spm_battery.py`); the
//!   `.pt` voice-prompt parser vs `vp_embeddings` (BIT-exact) + `vp_cache`;
//!   then the full 113-step free-run with the prompt assembled from the
//!   `.pt` file + the raw prompt text — model inputs vs `step_tokens`/
//!   `step_token_idx`, dep tokens, out frames, ALL integer-exact. `e2e`
//!   (golden-fed) is the ablation that separates LM wiring from prompt
//!   assembly; this gate proves the assembly.
//! - `ownprompt <voice.pt> [text]` — model-free smoke: load ANY packaged
//!   voice through mary's parser (e.g. the ref_voice.pt built by
//!   `golden/build_voice_prompt.py`), tokenize a system prompt, drive the
//!   bare `StreamCache` through the full prompt flow, and check the stream
//!   invariants (all prompt inputs in-vocabulary, nothing ungenerated).

use mary::models::f5::wav;
use mary::models::personaplex::config as cfg;
use mary::models::personaplex::depth::{argmax, DepthTransformer};
use mary::models::personaplex::lmgen::{LmGen, StreamCache};
use mary::models::personaplex::mimi::config as mimi_cfg;
use mary::models::personaplex::pipeline::{agent_codes, VoicePipeline, SILENCE, SINE};
use mary::models::personaplex::prompt::{wrap_with_system_tags, Prompt, SILENCE_FRAMES};
use mary::models::personaplex::spm::SpmTokenizer;
use mary::models::personaplex::temporal::TemporalTransformer;
use mary::models::personaplex::voice_prompt::VoicePrompt;
use mary::nn::npy;
use mary::nn::weight_loader::WeightLoader;
use std::path::Path;
use std::time::Instant;

type B = burn_ndarray::NdArray<f32>;

const GOLD: &str = "/tmp/mary-personaplex/golden";
const GATE: f64 = 0.99999;

fn golden_f32(name: &str) -> (Vec<f32>, Vec<usize>) {
    npy::load_npy(&Path::new(GOLD).join(format!("{name}.npy")))
        .unwrap_or_else(|e| panic!("golden {name}: {e}"))
}

fn golden_i64(name: &str) -> (Vec<i64>, Vec<usize>) {
    npy::load_npy_i64(&Path::new(GOLD).join(format!("{name}.npy")))
        .unwrap_or_else(|e| panic!("golden {name}: {e}"))
}

/// (cosine similarity, max |Δ|) in f64 accumulation.
fn cos_maxd(a: &[f32], b: &[f32]) -> (f64, f64) {
    assert_eq!(a.len(), b.len());
    let (mut dot, mut na, mut nb, mut maxd) = (0f64, 0f64, 0f64, 0f64);
    for (&x, &y) in a.iter().zip(b) {
        let (x, y) = (x as f64, y as f64);
        dot += x * y;
        na += x * x;
        nb += y * y;
        maxd = maxd.max((x - y).abs());
    }
    (dot / (na.sqrt() * nb.sqrt()), maxd)
}

/// Lazy handle-indexed pile loader (nothing materialized wholesale).
fn pile_loader(pile: &str) -> WeightLoader {
    mary::persist::personaplex_loader(Path::new(pile)).unwrap_or_else(|e| panic!("pile load: {e}"))
}

/// One oracle prompt-flow step: what `capture_personaplex.py` fed at each of
/// the 113 temporal steps.
enum Phase {
    /// Voice-prompt embedding replay (index into `vp_embeddings`).
    Vp(usize),
    /// Silence spacer: agent SILENCE_TOKENS + user SINE_TOKENS + text PAD.
    Silence,
    /// Text-prompt token (agent SILENCE + user SINE alongside).
    Text(i64),
    /// User-audio generation step (Mimi codes of the input wav).
    User([i64; 8]),
}

/// The oracle's prompt schedule: vp(50) → silence(6) → text → silence(6) →
/// user audio. Counts asserted against the goldens by the caller.
fn schedule(n_vp: usize, n_silence: usize, text: &[i64], user: &[i64]) -> Vec<Phase> {
    let mut s: Vec<Phase> = Vec::new();
    s.extend((0..n_vp).map(Phase::Vp));
    s.extend((0..n_silence).map(|_| Phase::Silence));
    s.extend(text.iter().map(|&t| Phase::Text(t)));
    s.extend((0..n_silence).map(|_| Phase::Silence));
    s.extend(user.chunks(8).map(|c| Phase::User(c.try_into().unwrap())));
    s
}

const PAD: i64 = cfg::TEXT_PAD_TOKEN as i64;

// ───────────────────────────── temporal gate ─────────────────────────────

fn temporal_gate(pile: &str) {
    let device = Default::default();

    let (vp, vps) = golden_f32("vp_embeddings"); // [50, 1, 1, 4096]
    assert_eq!(&vps[1..], &[1, 1, cfg::DIM], "vp_embeddings shape");
    let n_vp = vps[0];
    let (toks, ts) = golden_i64("step_tokens"); // [63, 17]
    assert_eq!(ts[1], cfg::NUM_STREAMS, "step_tokens shape");
    let n_tok = ts[0];
    let (tok_idx, _) = golden_i64("step_token_idx");
    let (gh, ghs) = golden_f32("tt_hidden"); // [113, 4096]
    let (gl, gls) = golden_f32("tt_text_logits"); // [113, 32000]
    let steps = ghs[0];
    assert_eq!(steps, n_vp + n_tok, "temporal steps = vp + token steps");
    assert_eq!(ghs[1], cfg::DIM);
    assert_eq!(gls, vec![steps, cfg::TEXT_LOGITS]);
    for (r, &i) in tok_idx.iter().enumerate() {
        assert_eq!(
            i as usize,
            n_vp + r,
            "token rows must be contiguous after the vp phase"
        );
    }
    println!("goldens: {steps} temporal steps ({n_vp} embedding-fed + {n_tok} token-fed)");

    println!("loading temporal transformer from {pile} …");
    let t0 = Instant::now();
    let loader = pile_loader(pile);
    let mut tt = TemporalTransformer::<B>::load(&loader, &device);
    println!("loaded in {:.1}s", t0.elapsed().as_secs_f64());

    let (mut min_hcos, mut min_lcos) = (1f64, 1f64);
    let (mut max_hd, mut max_ld) = (0f64, 0f64);
    let (mut worst_h, mut worst_l) = (0usize, 0usize);
    let mut argmax_hits = 0usize;
    let mut per_step: Vec<(f64, f64)> = Vec::with_capacity(steps);
    let t0 = Instant::now();
    for s in 0..steps {
        let x = if s < n_vp {
            let row = &vp[s * cfg::DIM..(s + 1) * cfg::DIM];
            burn::tensor::Tensor::<B, 1>::from_floats(row, &device).reshape([1, 1, cfg::DIM])
        } else {
            let row = &toks[(s - n_vp) * cfg::NUM_STREAMS..(s - n_vp + 1) * cfg::NUM_STREAMS];
            tt.embed_codes(row, &device)
        };
        let (hidden, logits) = tt.forward_embeddings(x, &device);
        let h: Vec<f32> = hidden.into_data().to_vec::<f32>().unwrap();
        let l: Vec<f32> = logits.into_data().to_vec::<f32>().unwrap();

        let (hcos, hd) = cos_maxd(&h, &gh[s * cfg::DIM..(s + 1) * cfg::DIM]);
        let glrow = &gl[s * cfg::TEXT_LOGITS..(s + 1) * cfg::TEXT_LOGITS];
        let (lcos, ld) = cos_maxd(&l, glrow);
        if hcos < min_hcos {
            min_hcos = hcos;
            worst_h = s;
        }
        if lcos < min_lcos {
            min_lcos = lcos;
            worst_l = s;
        }
        max_hd = max_hd.max(hd);
        max_ld = max_ld.max(ld);
        if argmax(&l) == argmax(glrow) {
            argmax_hits += 1;
        }
        per_step.push((hcos, lcos));
        if (s + 1) % 16 == 0 || s + 1 == steps {
            eprintln!(
                "  step {:3}/{steps}  hidden cos={hcos:.9}  logits cos={lcos:.9}  ({:.2} s/step)",
                s + 1,
                t0.elapsed().as_secs_f64() / (s + 1) as f64
            );
        }
    }
    let secs = t0.elapsed().as_secs_f64();

    // per-phase minima (localize any divergence to a prompt phase)
    let phase = |name: &str, lo: usize, hi: usize| {
        let hmin = per_step[lo..hi].iter().map(|p| p.0).fold(1f64, f64::min);
        let lmin = per_step[lo..hi].iter().map(|p| p.1).fold(1f64, f64::min);
        println!("  phase {name:<22} steps {lo:3}..{hi:3}  min hidden cos={hmin:.9}  min logits cos={lmin:.9}");
    };
    println!("phases:");
    phase("voice prompt (embed)", 0, 50);
    phase("silence 1", 50, 56);
    phase("text prompt", 56, 82);
    phase("silence 2", 82, 88);
    phase("user audio (gen)", 88, steps);

    let ok_h = min_hcos >= GATE;
    let ok_l = min_lcos >= GATE;
    println!(
        "  {} tt hidden       min cos={min_hcos:.9} (step {worst_h})  max|Δ|={max_hd:.3e}",
        if ok_h { "OK" } else { "XX" }
    );
    println!(
        "  {} tt text logits  min cos={min_lcos:.9} (step {worst_l})  max|Δ|={max_ld:.3e}",
        if ok_l { "OK" } else { "XX" }
    );
    println!(
        "  -- text argmax     {argmax_hits}/{steps} match ({:.1}%)",
        100.0 * argmax_hits as f64 / steps as f64
    );
    println!(
        "ran {steps} steps in {secs:.1}s ({:.2} s/step)",
        secs / steps as f64
    );

    verdict("PERSONAPLEX TEMPORAL PARITY", ok_h && ok_l);
}

// ─────────────────────────── shared golden set ───────────────────────────

struct StreamGoldens {
    steps: usize,
    n_vp: usize,
    sched: Vec<Phase>,
    dep_logits: Vec<f32>,  // [S, 16, 2048]
    dep_tokens: Vec<i64>,  // [S, 16]
    dep_in_text: Vec<i64>, // [S]
    out_tokens: Vec<i64>,  // [G, 17]
    gen_start: usize,      // first user-audio step
    vp_cache: Vec<i64>,    // [17, CT]
}

fn stream_goldens() -> StreamGoldens {
    let (dep_logits, s) = golden_f32("dep_logits");
    assert_eq!(&s[1..], &[cfg::DEP_Q, cfg::CARD], "dep_logits shape");
    let steps = s[0];
    let (dep_tokens, s) = golden_i64("dep_tokens");
    assert_eq!(s, vec![steps, cfg::DEP_Q], "dep_tokens shape");
    let (dep_in_text, s) = golden_i64("dep_in_text");
    assert_eq!(s, vec![steps], "dep_in_text shape");
    let (out_tokens, s) = golden_i64("out_tokens");
    assert_eq!(s[1], cfg::NUM_STREAMS, "out_tokens shape");
    let n_out = s[0];
    let (user, s) = golden_i64("user_codes");
    assert_eq!(s, vec![n_out, 8], "user_codes shape");
    let (text, _) = golden_i64("text_prompt_tokens");
    let (vp_cache, s) = golden_i64("vp_cache");
    assert_eq!(
        s,
        vec![1, cfg::NUM_STREAMS, mary::models::personaplex::lmgen::CT]
    );

    let n_silence = 6; // int(0.5 s × 12.5 Hz), meta.json phases
    let n_vp = steps - 2 * n_silence - text.len() - n_out;
    let sched = schedule(n_vp, n_silence, &text, &user);
    assert_eq!(sched.len(), steps, "schedule covers all temporal steps");
    let gen_start = steps - n_out;
    println!(
        "goldens: {steps} steps = vp {n_vp} + silence {n_silence} + text {} + silence {n_silence} + user {n_out}",
        text.len()
    );
    StreamGoldens {
        steps,
        n_vp,
        sched,
        dep_logits,
        dep_tokens,
        dep_in_text,
        out_tokens,
        gen_start,
        vp_cache,
    }
}

fn verdict(name: &str, ok: bool) {
    if ok {
        println!("{name}: PASS");
    } else {
        println!("{name}: FAIL");
        std::process::exit(1);
    }
}

// ────────────────────────────── depth gate ───────────────────────────────

fn depth_gate(pile: &str) {
    let device = Default::default();
    let g = stream_goldens();
    let (gh, s) = golden_f32("tt_hidden");
    assert_eq!(s, vec![g.steps, cfg::DIM]);
    let (gl, s) = golden_f32("tt_text_logits");
    assert_eq!(s, vec![g.steps, cfg::TEXT_LOGITS]);

    println!("loading depth transformer from {pile} …");
    let t0 = Instant::now();
    let loader = pile_loader(pile);
    let depth = DepthTransformer::<B>::load(&loader, &device);
    println!("loaded in {:.1}s", t0.elapsed().as_secs_f64());

    let mut sc = StreamCache::new();
    let mut min_cos_cb = [1f64; cfg::DEP_Q];
    let (mut min_cos, mut max_d, mut worst) = (1f64, 0f64, (0usize, 0usize));
    let mut argmax_hits = 0usize;
    let (mut text_hits, mut out_hits) = (0usize, 0usize);
    let t0 = Instant::now();
    for s in 0..g.steps {
        let p = match &g.sched[s] {
            Phase::Vp(_) => loop {
                let dummy = [cfg::CARD as i64; 8];
                if let Some(p) = sc.prepare(Some(&dummy), Some(&dummy), Some(PAD)) {
                    break p;
                }
            },
            Phase::Silence => sc.prepare(Some(&SINE), Some(&SILENCE), Some(PAD)).unwrap(),
            Phase::Text(t) => sc.prepare(Some(&SINE), Some(&SILENCE), Some(*t)).unwrap(),
            Phase::User(c) => sc.prepare(Some(c), None, None).unwrap(),
        };

        let hidden = burn::tensor::Tensor::<B, 1>::from_floats(
            &gh[s * cfg::DIM..(s + 1) * cfg::DIM],
            &device,
        )
        .reshape([1, 1, cfg::DIM]);
        let sampled_text = argmax(&gl[s * cfg::TEXT_LOGITS..(s + 1) * cfg::TEXT_LOGITS]) as i64;
        let next_text = if p.provided[0] {
            p.target[0]
        } else {
            sampled_text
        };
        text_hits += (next_text == g.dep_in_text[s]) as usize;

        let teacher: [i64; cfg::DEP_Q] = g.dep_tokens[s * cfg::DEP_Q..(s + 1) * cfg::DEP_Q]
            .try_into()
            .unwrap();
        let (toks, logits) = depth.frame(
            &hidden,
            next_text,
            &p.forced(),
            Some(&teacher),
            None,
            &device,
        );
        for cb in 0..cfg::DEP_Q {
            let grow = &g.dep_logits
                [(s * cfg::DEP_Q + cb) * cfg::CARD..(s * cfg::DEP_Q + cb + 1) * cfg::CARD];
            let (c, d) = cos_maxd(&logits[cb], grow);
            min_cos_cb[cb] = min_cos_cb[cb].min(c);
            if c < min_cos {
                min_cos = c;
                worst = (s, cb);
            }
            max_d = max_d.max(d);
            argmax_hits += (toks[cb] == teacher[cb]) as usize;
        }

        // teacher-forced commit: the cache follows the oracle trajectory
        let out = sc.commit(&p, sampled_text, &teacher);
        if s >= g.gen_start {
            let r = s - g.gen_start;
            let grow = &g.out_tokens[r * cfg::NUM_STREAMS..(r + 1) * cfg::NUM_STREAMS];
            out_hits += (out.as_ref().map(|o| &o[..]) == Some(grow)) as usize;
        }
        if s + 1 == g.n_vp {
            sc.overwrite(&g.vp_cache); // oracle: state.cache.copy_(voice_prompt_cache)
        }
        if (s + 1) % 16 == 0 || s + 1 == g.steps {
            eprintln!(
                "  step {:3}/{}  min cos={min_cos:.9}  ({:.2} s/step)",
                s + 1,
                g.steps,
                t0.elapsed().as_secs_f64() / (s + 1) as f64
            );
        }
    }
    let secs = t0.elapsed().as_secs_f64();

    println!("per-codebook min cos over {} steps:", g.steps);
    for (cb, mc) in min_cos_cb.iter().enumerate() {
        let kind = if cb == 0 || cb == 8 {
            "semantic"
        } else {
            "acoustic"
        };
        let side = if cb < 8 { "agent" } else { "user-pred" };
        println!("  cb {cb:2} ({side:9} {kind:8})  min cos={mc:.9}");
    }
    let n_gen = g.steps - g.gen_start;
    let n_cb = g.steps * cfg::DEP_Q;
    let ok_cos = min_cos >= GATE;
    let ok_int = argmax_hits == n_cb && text_hits == g.steps && out_hits == n_gen;
    println!(
        "  {} dep logits   min cos={min_cos:.9} (step {}, cb {})  max|Δ|={max_d:.3e}",
        if ok_cos { "OK" } else { "XX" },
        worst.0,
        worst.1
    );
    println!(
        "  {} dep argmax   {argmax_hits}/{n_cb} match ({:.2}%)",
        if argmax_hits == n_cb { "OK" } else { "XX" },
        100.0 * argmax_hits as f64 / n_cb as f64
    );
    println!(
        "  {} next_text    {text_hits}/{} match (dep_in_text)",
        if text_hits == g.steps { "OK" } else { "XX" },
        g.steps
    );
    println!(
        "  {} out frames   {out_hits}/{n_gen} exact (out_tokens, all 17 streams)",
        if out_hits == n_gen { "OK" } else { "XX" }
    );
    println!(
        "ran {} steps in {secs:.1}s ({:.2} s/step)",
        g.steps,
        secs / g.steps as f64
    );

    verdict("PERSONAPLEX DEPTH PARITY", ok_cos && ok_int);
}

// ─────────────────────────────── e2e gate ────────────────────────────────

fn e2e_gate(pile: &str) {
    let device = Default::default();
    let g = stream_goldens();
    let (vp, vps) = golden_f32("vp_embeddings");
    assert_eq!(vps, vec![g.n_vp, 1, 1, cfg::DIM], "vp_embeddings shape");
    let (step_toks, s) = golden_i64("step_tokens");
    assert_eq!(
        s,
        vec![g.steps - g.n_vp, cfg::NUM_STREAMS],
        "step_tokens shape"
    );

    println!("loading temporal + depth transformers from {pile} …");
    let t0 = Instant::now();
    let loader = pile_loader(pile);
    let mut lm = LmGen::new(
        TemporalTransformer::<B>::load(&loader, &device),
        DepthTransformer::<B>::load(&loader, &device),
    );
    println!("loaded in {:.1}s", t0.elapsed().as_secs_f64());

    let (mut input_hits, mut text_hits, mut dep_hits, mut out_hits) =
        (0usize, 0usize, 0usize, 0usize);
    let mut first_bad: Option<(usize, &'static str)> = None;
    let t0 = Instant::now();
    for s in 0..g.steps {
        let trace = match &g.sched[s] {
            Phase::Vp(i) => {
                let row = &vp[i * cfg::DIM..(i + 1) * cfg::DIM];
                let x = burn::tensor::Tensor::<B, 1>::from_floats(row, &device).reshape([
                    1,
                    1,
                    cfg::DIM,
                ]);
                lm.step_embeddings(x)
            }
            Phase::Silence => lm.step(Some(&SINE), Some(&SILENCE), Some(PAD), &device),
            Phase::Text(t) => lm.step(Some(&SINE), Some(&SILENCE), Some(*t), &device),
            Phase::User(c) => lm.step(Some(c), None, None, &device),
        };
        let mut bad = |what: &'static str| {
            if first_bad.is_none() {
                first_bad = Some((s, what));
            }
        };

        if s >= g.n_vp {
            let r = s - g.n_vp;
            let grow = &step_toks[r * cfg::NUM_STREAMS..(r + 1) * cfg::NUM_STREAMS];
            if trace.input.as_ref().map(|i| &i[..]) == Some(grow) {
                input_hits += 1;
            } else {
                bad("model input vs step_tokens");
            }
        }
        if trace.next_text == g.dep_in_text[s] {
            text_hits += 1;
        } else {
            bad("next_text vs dep_in_text");
        }
        if trace.dep_tokens[..] == g.dep_tokens[s * cfg::DEP_Q..(s + 1) * cfg::DEP_Q] {
            dep_hits += 1;
        } else {
            bad("dep tokens vs dep_tokens");
        }
        if s >= g.gen_start {
            let r = s - g.gen_start;
            let grow = &g.out_tokens[r * cfg::NUM_STREAMS..(r + 1) * cfg::NUM_STREAMS];
            if trace.out.as_ref().map(|o| &o[..]) == Some(grow) {
                out_hits += 1;
            } else {
                bad("out frame vs out_tokens");
            }
        }
        if s + 1 == g.n_vp {
            lm.stream.overwrite(&g.vp_cache); // oracle: cache.copy_(voice_prompt_cache)
        }
        if (s + 1) % 8 == 0 || s + 1 == g.steps {
            eprintln!(
                "  step {:3}/{}  dep frames exact {dep_hits}/{}  ({:.2} s/step)",
                s + 1,
                g.steps,
                s + 1,
                t0.elapsed().as_secs_f64() / (s + 1) as f64
            );
        }
    }
    let secs = t0.elapsed().as_secs_f64();

    let n_tok = g.steps - g.n_vp;
    let n_gen = g.steps - g.gen_start;
    let ok =
        input_hits == n_tok && text_hits == g.steps && dep_hits == g.steps && out_hits == n_gen;
    println!(
        "  {} model inputs  {input_hits}/{n_tok} frames exact (step_tokens, 17 streams)",
        if input_hits == n_tok { "OK" } else { "XX" }
    );
    println!(
        "  {} next_text     {text_hits}/{} exact (dep_in_text)",
        if text_hits == g.steps { "OK" } else { "XX" },
        g.steps
    );
    println!(
        "  {} dep tokens    {dep_hits}/{} frames exact (dep_tokens, 16 codebooks)",
        if dep_hits == g.steps { "OK" } else { "XX" },
        g.steps
    );
    println!(
        "  {} out frames    {out_hits}/{n_gen} exact (out_tokens, 17 streams)",
        if out_hits == n_gen { "OK" } else { "XX" }
    );
    if let Some((s, what)) = first_bad {
        println!("  first divergence: step {s} — {what}");
    }
    println!(
        "ran {} steps in {secs:.1}s ({:.2} s/step)",
        g.steps,
        secs / g.steps as f64
    );

    verdict("PERSONAPLEX E2E TOKEN PARITY", ok);
}

// ───────────────────────────── pipeline gate ─────────────────────────────

/// The oracle's user-side input WAV (2 s of ref_voice.wav — `meta.json`
/// `input_wav`/`input_seconds`); the pipeline re-encodes it with mary's Mimi
/// encoder instead of replaying `user_codes`.
const INPUT_WAV: &str = "ref_voice.wav";
const OUT_WAV: &str = "/tmp/mary-personaplex/pipeline_out.wav";
/// Audio bar: the Mimi-decoder parity bar. The tokens are integer-exact, so
/// any gap here is decoder wiring (or streaming-vs-batch fp — see the
/// `out_audio_batch` cross-check).
const AUDIO_GATE: f64 = 0.999;

fn pipeline_gate(pile: &str) {
    let device = Default::default();
    let g = stream_goldens();
    let n_gen = g.steps - g.gen_start;
    let (vp, vps) = golden_f32("vp_embeddings");
    assert_eq!(vps, vec![g.n_vp, 1, 1, cfg::DIM], "vp_embeddings shape");
    let (text, _) = golden_i64("text_prompt_tokens");
    let (user, s) = golden_i64("user_codes");
    assert_eq!(s, vec![n_gen, 8], "user_codes shape");
    let (gaudio, s) = golden_f32("out_audio");
    assert_eq!(
        s,
        vec![n_gen * mimi_cfg::SAMPLES_PER_FRAME],
        "out_audio shape"
    );
    let (gaudio_batch, s) = golden_f32("out_audio_batch");
    assert_eq!(
        s,
        vec![n_gen * mimi_cfg::SAMPLES_PER_FRAME],
        "out_audio_batch shape"
    );

    let (mut samples, sr) = wav::read_pcm16_mono(Path::new(INPUT_WAV));
    assert_eq!(sr, mimi_cfg::SAMPLE_RATE, "input wav sample rate");
    let n_samples = n_gen * mimi_cfg::SAMPLES_PER_FRAME;
    assert!(
        samples.len() >= n_samples,
        "input wav shorter than the oracle window"
    );
    samples.truncate(n_samples);
    println!("input: {INPUT_WAV} ({n_samples} samples = {n_gen} frames)");

    println!("loading pipeline (temporal + depth + mimi encoder/decoder) from {pile} …");
    let t0 = Instant::now();
    let loader = pile_loader(pile);
    let mut p = VoicePipeline::<B>::load(&loader, &device);
    println!("loaded in {:.1}s", t0.elapsed().as_secs_f64());

    // 1. Mimi encode — user codes, gated integer-exact vs the oracle's
    // streaming encode.
    let codes = p.encoder.encode(&samples);
    assert_eq!(codes.len(), n_gen, "encoded frame count");
    let mut enc_mism = 0usize;
    for (t, frame) in codes.iter().enumerate() {
        for (q, &c) in frame.iter().enumerate() {
            enc_mism += (c as i64 != user[t * 8 + q]) as usize;
        }
    }
    let n_codes = n_gen * 8;
    let ok_enc = enc_mism == 0;
    println!(
        "  {} mimi encode   {}/{n_codes} codes exact (user_codes)",
        if ok_enc { "OK" } else { "XX" },
        n_codes - enc_mism
    );

    // The production encoder consumes one 80 ms frame per LM step. Gate that
    // stateful path against both the established full-clip path and the
    // NVIDIA streaming oracle captured in `user_codes.npy`.
    let mut encoder_state = p.encoder.stream_state();
    let streaming_codes: Vec<_> = samples
        .chunks_exact(mimi_cfg::SAMPLES_PER_FRAME)
        .map(|frame| {
            p.encoder.encode_stream_frame(
                &mut encoder_state,
                frame.try_into().expect("exact Mimi input frame"),
            )
        })
        .collect();
    let ok_stream = streaming_codes == codes;
    println!(
        "  {} mimi stream   {}/{} frames exact (batch + user_codes)",
        if ok_stream { "OK" } else { "XX" },
        streaming_codes
            .iter()
            .zip(&codes)
            .filter(|(stream, batch)| stream == batch)
            .count(),
        n_gen
    );

    encoder_state.reset();
    let reset_codes: Vec<_> = samples
        .chunks_exact(mimi_cfg::SAMPLES_PER_FRAME)
        .map(|frame| {
            p.encoder.encode_stream_frame(
                &mut encoder_state,
                frame.try_into().expect("exact Mimi input frame"),
            )
        })
        .collect();
    let ok_reset = reset_codes == streaming_codes;
    println!(
        "  {} mimi reset    deterministic streaming replay",
        if ok_reset { "OK" } else { "XX" }
    );

    // 2. Prompt phases (oracle flow: voice → silence → text → silence).
    let t0 = Instant::now();
    p.prompt_voice(&vp, &g.vp_cache, &device);
    p.prompt_silence(6, &device);
    p.prompt_text(&text, &device);
    p.prompt_silence(6, &device);
    println!(
        "prompt phases done in {:.1}s ({} steps)",
        t0.elapsed().as_secs_f64(),
        g.gen_start
    );

    // 3. Free-run over the user frames; undelayed out frames gated vs
    // out_tokens, agent streams 1..=8 collected for the decoder.
    let mut agent: Vec<[u32; mimi_cfg::NUM_CODEBOOKS]> = Vec::with_capacity(n_gen);
    let mut out_hits = 0usize;
    let t0 = Instant::now();
    for (r, frame) in codes.iter().enumerate() {
        let cf: [i64; 8] = frame.map(|c| c as i64);
        let out = p
            .step_user_frame(&cf, &device)
            .expect("past the delay horizon — prompt flow consumed the None steps");
        let grow = &g.out_tokens[r * cfg::NUM_STREAMS..(r + 1) * cfg::NUM_STREAMS];
        out_hits += (out[..] == *grow) as usize;
        agent.push(agent_codes(&out));
        if (r + 1) % 8 == 0 || r + 1 == n_gen {
            eprintln!(
                "  step {:3}/{n_gen}  out frames exact {out_hits}/{}  ({:.2} s/step)",
                r + 1,
                r + 1,
                t0.elapsed().as_secs_f64() / (r + 1) as f64
            );
        }
    }
    let ok_out = out_hits == n_gen;
    println!(
        "  {} out frames    {out_hits}/{n_gen} exact (out_tokens, all 17 streams)",
        if ok_out { "OK" } else { "XX" }
    );

    // 4. Mimi decode → PCM, gated vs the oracle's streaming per-frame decode
    // (production semantics) with the batch decode as fp cross-check.
    let t0 = Instant::now();
    let pcm = p.decode(&agent);
    assert_eq!(pcm.len(), gaudio.len(), "decoded sample count");
    println!(
        "decoded {} samples in {:.1}s",
        pcm.len(),
        t0.elapsed().as_secs_f64()
    );
    let (cos_s, maxd_s) = cos_maxd(&pcm, &gaudio);
    let (cos_b, maxd_b) = cos_maxd(&pcm, &gaudio_batch);
    let ok_audio = cos_s >= AUDIO_GATE;
    println!(
        "  {} audio (streaming oracle)  cos={cos_s:.9}  max|Δ|={maxd_s:.3e}",
        if ok_audio { "OK" } else { "XX" }
    );
    println!("  -- audio (batch oracle)      cos={cos_b:.9}  max|Δ|={maxd_b:.3e}");

    wav::write_pcm16_mono(Path::new(OUT_WAV), &pcm, mimi_cfg::SAMPLE_RATE);
    println!("wrote {OUT_WAV}");

    verdict(
        "PERSONAPLEX PIPELINE PARITY",
        ok_enc && ok_stream && ok_reset && ok_out && ok_audio,
    );
}

// ────────────────────────────── prompt gate ──────────────────────────────

/// Primary prompt sources — the artifacts the oracle capture itself consumed
/// (Phase 0 downloads under /tmp/personaplex_scratch; the pile stays a
/// weights-only artifact for now).
const SPM_MODEL: &str = "/tmp/personaplex_scratch/ckpt/tokenizer_spm_32k_3.model";
const VOICE_PT: &str = "/tmp/personaplex_scratch/voices/NATM0.pt";

/// Fast Phase-5 gates (no model): tokenizer battery + exact prompt tokens,
/// and the `.pt` parser vs the vp goldens (bit-exact). Returns the assembled
/// [`Prompt`] for the free-run gate.
fn prompt_sources_gate() -> (Prompt, bool) {
    let mut ok = true;

    // 1. Tokenizer battery vs the oracle venv (golden/capture_spm_battery.py).
    let spm = SpmTokenizer::load(Path::new(SPM_MODEL));
    let battery: serde_json::Value = serde_json::from_slice(
        &std::fs::read(Path::new(GOLD).join("spm_battery.json")).expect("spm_battery.json"),
    )
    .expect("battery json");
    let battery = battery.as_array().expect("battery array");
    let mut hits = 0usize;
    for case in battery {
        let text = case["text"].as_str().unwrap();
        let want: Vec<i64> = case["ids"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_i64().unwrap())
            .collect();
        let got = spm.encode(text);
        if got == want {
            hits += 1;
        } else {
            println!("  XX battery {text:?}\n     want {want:?}\n     got  {got:?}");
        }
    }
    let ok_bat = hits == battery.len();
    ok &= ok_bat;
    println!(
        "  {} spm battery   {hits}/{} strings token-exact (oracle venv)",
        if ok_bat { "OK" } else { "XX" },
        battery.len()
    );

    // 2. The capture's exact system prompt (meta.json) vs text_prompt_tokens.
    let meta: serde_json::Value = serde_json::from_slice(
        &std::fs::read(Path::new(GOLD).join("meta.json")).expect("meta.json"),
    )
    .expect("meta json");
    let text_prompt = meta["text_prompt"].as_str().expect("meta text_prompt");
    let (gtok, _) = golden_i64("text_prompt_tokens");
    let tokens = spm.encode(&wrap_with_system_tags(text_prompt));
    let tok_hits = tokens.iter().zip(&gtok).filter(|(a, b)| a == b).count();
    let ok_tok = tokens.len() == gtok.len() && tok_hits == gtok.len();
    ok &= ok_tok;
    println!(
        "  {} text prompt   {tok_hits}/{} tokens exact (text_prompt_tokens; got {} tokens)",
        if ok_tok { "OK" } else { "XX" },
        gtok.len(),
        tokens.len()
    );

    // 3. The packaged voice .pt vs the vp goldens — embeddings BIT-exact
    // (bf16→f32 is a bit-shift; the golden is the f32 dump of that tensor).
    let voice = VoicePrompt::load(Path::new(VOICE_PT));
    let (gvp, gvps) = golden_f32("vp_embeddings");
    let (gcache, _) = golden_i64("vp_cache");
    let ok_shape = gvps[0] == voice.n_frames && voice.embeddings.len() == gvp.len();
    let bit_hits = voice
        .embeddings
        .iter()
        .zip(&gvp)
        .filter(|(a, b)| a.to_bits() == b.to_bits())
        .count();
    let ok_emb = ok_shape && bit_hits == gvp.len();
    ok &= ok_emb;
    println!(
        "  {} vp embeddings {bit_hits}/{} f32 values BIT-exact ({} frames)",
        if ok_emb { "OK" } else { "XX" },
        gvp.len(),
        voice.n_frames
    );
    let cache_hits = voice
        .cache
        .iter()
        .zip(&gcache)
        .filter(|(a, b)| a == b)
        .count();
    let ok_cache = voice.cache.len() == gcache.len() && cache_hits == gcache.len();
    ok &= ok_cache;
    println!(
        "  {} vp cache      {cache_hits}/{} tokens exact",
        if ok_cache { "OK" } else { "XX" },
        gcache.len()
    );

    let prompt = Prompt {
        voice,
        text_tokens: tokens,
        silence_frames: SILENCE_FRAMES,
    };
    (prompt, ok)
}

fn prompt_gate(pile: &str) {
    println!("prompt sources: {SPM_MODEL} + {VOICE_PT}");
    let (prompt, ok_src) = prompt_sources_gate();

    // The free-run compare targets (the goldens describe the same schedule
    // the primary sources must reproduce).
    let g = stream_goldens();
    let (step_toks, s) = golden_i64("step_tokens");
    assert_eq!(
        s,
        vec![g.steps - g.n_vp, cfg::NUM_STREAMS],
        "step_tokens shape"
    );
    let (tok_idx, _) = golden_i64("step_token_idx");
    assert_eq!(
        prompt.total_steps() + g.out_tokens.len() / cfg::NUM_STREAMS,
        g.steps,
        "assembled prompt covers the oracle's prompt phases"
    );
    // step_token_idx semantics: token-fed steps are exactly the post-vp rows.
    let ok_idx = tok_idx.len() == g.steps - g.n_vp
        && tok_idx
            .iter()
            .enumerate()
            .all(|(r, &i)| i as usize == g.n_vp + r);
    println!(
        "  {} step_token_idx {} token-fed steps contiguous after {} vp steps",
        if ok_idx { "OK" } else { "XX" },
        tok_idx.len(),
        g.n_vp
    );
    let (user, s) = golden_i64("user_codes");
    assert_eq!(s[1], 8, "user_codes shape");

    println!("loading temporal + depth transformers from {pile} …");
    let device = Default::default();
    let t0 = Instant::now();
    let loader = pile_loader(pile);
    let mut lm = LmGen::new(
        TemporalTransformer::<B>::load(&loader, &device),
        DepthTransformer::<B>::load(&loader, &device),
    );
    println!("loaded in {:.1}s", t0.elapsed().as_secs_f64());

    // Free-run: the assembled prompt (primary sources), then the golden user
    // codes (the live-audio input; Mimi encode has its own gate in
    // `pipeline`). Gate every surface exactly as e2e does.
    let (mut input_hits, mut dep_hits, mut out_hits) = (0usize, 0usize, 0usize);
    let mut first_bad: Option<(usize, &'static str)> = None;
    let mut step_no = 0usize;
    let t0 = Instant::now();
    let mut check = |trace: &mary::models::personaplex::lmgen::StepTrace,
                     s: usize,
                     first_bad: &mut Option<(usize, &'static str)>| {
        let mut bad = |what: &'static str| {
            if first_bad.is_none() {
                *first_bad = Some((s, what));
            }
        };
        if s >= g.n_vp {
            let r = s - g.n_vp;
            let grow = &step_toks[r * cfg::NUM_STREAMS..(r + 1) * cfg::NUM_STREAMS];
            if trace.input.as_ref().map(|i| &i[..]) == Some(grow) {
                input_hits += 1;
            } else {
                bad("model input vs step_tokens");
            }
        }
        if trace.dep_tokens[..] == g.dep_tokens[s * cfg::DEP_Q..(s + 1) * cfg::DEP_Q] {
            dep_hits += 1;
        } else {
            bad("dep tokens vs dep_tokens");
        }
        if s >= g.gen_start {
            let r = s - g.gen_start;
            let grow = &g.out_tokens[r * cfg::NUM_STREAMS..(r + 1) * cfg::NUM_STREAMS];
            if trace.out.as_ref().map(|o| &o[..]) == Some(grow) {
                out_hits += 1;
            } else {
                bad("out frame vs out_tokens");
            }
        }
    };

    for row in prompt.voice.embeddings.chunks_exact(cfg::DIM) {
        let x = burn::tensor::Tensor::<B, 1>::from_floats(row, &device).reshape([1, 1, cfg::DIM]);
        let trace = lm.step_embeddings(x);
        check(&trace, step_no, &mut first_bad);
        step_no += 1;
    }
    lm.stream.overwrite(&prompt.voice.cache);
    for _ in 0..prompt.silence_frames {
        let trace = lm.step(Some(&SINE), Some(&SILENCE), Some(PAD), &device);
        check(&trace, step_no, &mut first_bad);
        step_no += 1;
    }
    for &t in &prompt.text_tokens {
        let trace = lm.step(Some(&SINE), Some(&SILENCE), Some(t), &device);
        check(&trace, step_no, &mut first_bad);
        step_no += 1;
    }
    for _ in 0..prompt.silence_frames {
        let trace = lm.step(Some(&SINE), Some(&SILENCE), Some(PAD), &device);
        check(&trace, step_no, &mut first_bad);
        step_no += 1;
    }
    eprintln!(
        "  prompt phases done ({step_no} steps, {:.2} s/step)",
        t0.elapsed().as_secs_f64() / step_no as f64
    );
    for c in user.chunks_exact(8) {
        let codes: [i64; 8] = c.try_into().unwrap();
        let trace = lm.step(Some(&codes), None, None, &device);
        check(&trace, step_no, &mut first_bad);
        step_no += 1;
    }
    let secs = t0.elapsed().as_secs_f64();
    assert_eq!(step_no, g.steps, "step count");

    let n_tok = g.steps - g.n_vp;
    let n_gen = g.steps - g.gen_start;
    let ok_run = input_hits == n_tok && dep_hits == g.steps && out_hits == n_gen;
    println!(
        "  {} model inputs  {input_hits}/{n_tok} frames exact (step_tokens, 17 streams)",
        if input_hits == n_tok { "OK" } else { "XX" }
    );
    println!(
        "  {} dep tokens    {dep_hits}/{} frames exact (dep_tokens, 16 codebooks)",
        if dep_hits == g.steps { "OK" } else { "XX" },
        g.steps
    );
    println!(
        "  {} out frames    {out_hits}/{n_gen} exact (out_tokens, 17 streams)",
        if out_hits == n_gen { "OK" } else { "XX" }
    );
    if let Some((s, what)) = first_bad {
        println!("  first divergence: step {s} — {what}");
    }
    println!(
        "ran {} steps in {secs:.1}s ({:.2} s/step)",
        g.steps,
        secs / g.steps as f64
    );

    verdict("PERSONAPLEX PROMPT ASSEMBLY", ok_src && ok_idx && ok_run);
}

// ───────────────────────────── ownprompt smoke ────────────────────────────

/// Default system prompt for the own-voice assembly smoke.
const OWN_TEXT: &str = "You are a helpful assistant. You speak with a warm, \
                        curious, and direct voice. Answer in a clear and engaging way.";

fn ownprompt_smoke(voice_pt: &str, text: &str) {
    println!("voice:  {voice_pt}");
    let voice = VoicePrompt::load(Path::new(voice_pt));
    let bad_emb = voice.embeddings.iter().filter(|v| !v.is_finite()).count();
    println!(
        "  loaded: {} embedding-replay frames ({} f32 values, {bad_emb} non-finite), cache [17,{}]",
        voice.n_frames,
        voice.embeddings.len(),
        mary::models::personaplex::lmgen::CT
    );
    // Cache snapshot sanity: audio streams ≤ initial (2048), text ≤ 32000,
    // nothing ungenerated (−2).
    let cache_ok = voice.cache.iter().enumerate().all(|(i, &t)| {
        let k = i / mary::models::personaplex::lmgen::CT;
        let hi = if k == 0 {
            cfg::TEXT_CARD as i64
        } else {
            cfg::CARD as i64
        };
        (0..=hi).contains(&t)
    });
    println!(
        "  {} cache snapshot tokens in range",
        if cache_ok { "OK" } else { "XX" }
    );

    let spm = SpmTokenizer::load(Path::new(SPM_MODEL));
    let wrapped = wrap_with_system_tags(text);
    let tokens = spm.encode(&wrapped);
    println!("text:   {wrapped:?}");
    println!("  {} spm tokens: {tokens:?}", tokens.len());

    // Drive the bare StreamCache through the full prompt flow (no model —
    // the token machine is exact by construction, gated in `prompt`).
    let prompt = Prompt {
        voice,
        text_tokens: tokens,
        silence_frames: SILENCE_FRAMES,
    };
    let mut sc = StreamCache::new();
    let mut inputs_ok = 0usize;
    let mut n_model_inputs = 0usize;
    let dummy = [cfg::CARD as i64; 8];
    let token_step = |sc: &mut StreamCache,
                      user: &[i64; 8],
                      agent: &[i64; 8],
                      text_tok: i64,
                      n_model_inputs: &mut usize,
                      inputs_ok: &mut usize| {
        let p = sc.prepare(Some(user), Some(agent), Some(text_tok)).unwrap();
        *n_model_inputs += 1;
        let valid = p.input.iter().enumerate().all(|(k, &t)| {
            let hi = if k == 0 {
                cfg::TEXT_CARD as i64
            } else {
                cfg::CARD as i64
            };
            (0..=hi).contains(&t)
        });
        *inputs_ok += valid as usize;
        // Prompt steps are fully provided — commit target tokens verbatim.
        let audio: [i64; cfg::DEP_Q] = std::array::from_fn(|s| p.target[1 + s]);
        sc.commit(&p, p.target[0], &audio);
    };
    for _ in 0..prompt.voice.n_frames {
        loop {
            if let Some(p) = sc.prepare(Some(&dummy), Some(&dummy), Some(PAD)) {
                let audio: [i64; cfg::DEP_Q] = std::array::from_fn(|s| p.target[1 + s]);
                sc.commit(&p, p.target[0], &audio);
                break;
            }
        }
    }
    sc.overwrite(&prompt.voice.cache);
    for _ in 0..prompt.silence_frames {
        token_step(
            &mut sc,
            &SINE,
            &SILENCE,
            PAD,
            &mut n_model_inputs,
            &mut inputs_ok,
        );
    }
    for &t in &prompt.text_tokens {
        token_step(
            &mut sc,
            &SINE,
            &SILENCE,
            t,
            &mut n_model_inputs,
            &mut inputs_ok,
        );
    }
    for _ in 0..prompt.silence_frames {
        token_step(
            &mut sc,
            &SINE,
            &SILENCE,
            PAD,
            &mut n_model_inputs,
            &mut inputs_ok,
        );
    }
    let ok_inputs = inputs_ok == n_model_inputs;
    println!(
        "  {} stream: {} prompt steps assembled ({} token-fed model inputs, {inputs_ok} in-vocabulary)",
        if ok_inputs { "OK" } else { "XX" },
        prompt.total_steps(),
        n_model_inputs
    );
    verdict("PERSONAPLEX OWN-PROMPT ASSEMBLY", cache_ok && ok_inputs);
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let sub = args.get(1).map(String::as_str).unwrap_or("");
    match sub {
        "ownprompt" => {
            let voice = args.get(2).map(String::as_str).unwrap_or_else(|| {
                eprintln!("usage: personaplex_probe ownprompt <voice.pt> [system text]");
                std::process::exit(2);
            });
            ownprompt_smoke(voice, args.get(3).map(String::as_str).unwrap_or(OWN_TEXT));
            return;
        }
        _ => {}
    }
    let pile = mary::paths::model(args.get(2).map(String::as_str), "personaplex.pile")
        .unwrap_or_else(|e| {
            eprintln!("{e}");
            std::process::exit(2)
        })
        .to_string_lossy()
        .into_owned();
    match sub {
        "temporal" => temporal_gate(&pile),
        "depth" => depth_gate(&pile),
        "e2e" => e2e_gate(&pile),
        "pipeline" => pipeline_gate(&pile),
        "prompt" => prompt_gate(&pile),
        _ => {
            eprintln!("usage: personaplex_probe <temporal|depth|e2e|pipeline|prompt> [pile-path]");
            eprintln!("       personaplex_probe ownprompt <voice.pt> [system text]");
            eprintln!("  temporal  LM part 1: 7B temporal transformer cos gates (113 steps)");
            eprintln!("  depth     LM part 2: depformer teacher-forced logit gates + bookkeeping");
            eprintln!("  e2e       LM parts 1+2 free-running: integer-exact token stream");
            eprintln!("  pipeline  LM part 3: WAV → encode → LM → decode → WAV, audio cos gate");
            eprintln!("  prompt    Phase 5: SPM tokenizer + voice .pt + assembled prompt stream");
            eprintln!("  ownprompt Phase 5 smoke: assemble a prompt from any packaged voice");
            std::process::exit(2);
        }
    }
}
