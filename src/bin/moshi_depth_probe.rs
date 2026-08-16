//! `depth_fast` (Lane B realtime depformer predictor) — gate + bench.
//!
//!   cargo run --release --features personaplex --bin moshi_depth_probe -- \
//!     gate  [--f16] [--skip-burn] [pile-path]
//!   cargo run --release --features personaplex --bin moshi_depth_probe -- \
//!     bench [--f16] [--synth] [pile-path]
//!
//! `gate` replays the oracle's full 113-step teacher-forced stream (same
//! `StreamCache` drive as `personaplex_probe depth`) through
//! [`DepthFast::frame`] and gates:
//!   - per-codebook logits cos vs `dep_logits` (bar 0.99999, expect ~1.0),
//!   - argmax 1808/1808 vs `dep_tokens`,
//!   - `next_text_token` 113/113 vs `dep_in_text`,
//!   - undelayed out frames 25/25 vs `out_tokens`,
//! and (unless `--skip-burn`) runs the burn `depth.rs` reference on the SAME
//! inputs in the same process: exact-token agreement 1808/1808 plus the
//! fast-vs-burn logits cos / max|Δ| (the same-numerics-family evidence).
//!
//! `bench` free-runs frames (greedy, no forcing) over the golden temporal
//! hiddens and reports min-of-medians ms per whole 16-step frame
//! (`MOSHI_ROUNDS` × `MOSHI_FRAMES`, defaults 8 × 25), the effective weight
//! bandwidth, and the cost decomposition (cond gemv / stack gemv / head /
//! scalar). `--synth` uses deterministic synthetic weights at the real
//! shapes — pile-free timing for thread sweeps (`MARY_DEPTH_THREADS` for the
//! f16 NEON pool, `MARY_PRED_THREADS` for the f32 Accelerate pool).

use mary::models::personaplex::config as cfg;
use mary::models::personaplex::depth::{argmax, DepthTransformer};
use mary::models::personaplex::depth_fast::DepthFast;
use mary::models::personaplex::lmgen::StreamCache;
use mary::models::personaplex::pipeline::{SILENCE, SINE};
use mary::models::qwen3tts::cpu;
use mary::nn::npy;
use mary::nn::weight_loader::WeightLoader;
use std::path::Path;
use std::time::Instant;

type B = burn_ndarray::NdArray<f32>;

const GOLD: &str = "/tmp/mary-personaplex/golden";
const GATE: f64 = 0.99999;
const PAD: i64 = cfg::TEXT_PAD_TOKEN as i64;

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

fn pile_loader(pile: &str) -> WeightLoader {
    mary::persist::personaplex_loader(Path::new(pile)).unwrap_or_else(|e| panic!("pile load: {e}"))
}

/// One oracle prompt-flow step (what `capture_personaplex.py` fed).
enum Phase {
    Vp,
    Silence,
    Text(i64),
    User([i64; 8]),
}

struct Goldens {
    steps: usize,
    n_vp: usize,
    sched: Vec<Phase>,
    dep_logits: Vec<f32>,  // [S, 16, 2048]
    dep_tokens: Vec<i64>,  // [S, 16]
    dep_in_text: Vec<i64>, // [S]
    out_tokens: Vec<i64>,  // [G, 17]
    gen_start: usize,
    vp_cache: Vec<i64>,
    tt_hidden: Vec<f32>,      // [S, 4096]
    tt_text_logits: Vec<f32>, // [S, 32000]
}

fn goldens() -> Goldens {
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
    let (tt_hidden, s) = golden_f32("tt_hidden");
    assert_eq!(s, vec![steps, cfg::DIM], "tt_hidden shape");
    let (tt_text_logits, s) = golden_f32("tt_text_logits");
    assert_eq!(s, vec![steps, cfg::TEXT_LOGITS], "tt_text_logits shape");

    let n_silence = 6; // int(0.5 s × 12.5 Hz), meta.json phases
    let n_vp = steps - 2 * n_silence - text.len() - n_out;
    let mut sched: Vec<Phase> = Vec::new();
    sched.extend((0..n_vp).map(|_| Phase::Vp));
    sched.extend((0..n_silence).map(|_| Phase::Silence));
    sched.extend(text.iter().map(|&t| Phase::Text(t)));
    sched.extend((0..n_silence).map(|_| Phase::Silence));
    sched.extend(user.chunks(8).map(|c| Phase::User(c.try_into().unwrap())));
    assert_eq!(sched.len(), steps, "schedule covers all temporal steps");
    println!(
        "goldens: {steps} steps = vp {n_vp} + silence {n_silence} + text {} + silence {n_silence} + user {n_out}",
        text.len()
    );
    Goldens {
        steps,
        n_vp,
        sched,
        dep_logits,
        dep_tokens,
        dep_in_text,
        out_tokens,
        gen_start: steps - n_out,
        vp_cache,
        tt_hidden,
        tt_text_logits,
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

// ──────────────────────────────── gate ────────────────────────────────────

fn gate(pile: &str, f16: bool, vs_burn: bool) {
    let device: burn_ndarray::NdArrayDevice = Default::default();
    let g = goldens();

    println!(
        "loading depth_fast ({}) from {pile} …",
        if f16 {
            "f16 storage, f32 accumulate"
        } else {
            "f32"
        }
    );
    let t0 = Instant::now();
    let loader = pile_loader(pile);
    let mut fast = DepthFast::load(&loader, f16);
    println!("loaded in {:.1}s", t0.elapsed().as_secs_f64());
    let burn_depth = vs_burn.then(|| {
        println!("loading burn depth.rs reference …");
        let t0 = Instant::now();
        let d = DepthTransformer::<B>::load(&loader, &device);
        println!("loaded in {:.1}s", t0.elapsed().as_secs_f64());
        d
    });

    let mut sc = StreamCache::new();
    let mut min_cos_cb = [1f64; cfg::DEP_Q];
    let (mut min_cos, mut max_d, mut worst) = (1f64, 0f64, (0usize, 0usize));
    let mut argmax_hits = 0usize;
    let (mut text_hits, mut out_hits) = (0usize, 0usize);
    // fast-vs-burn agreement
    let mut vs_token_hits = 0usize;
    let (mut vs_min_cos, mut vs_max_d) = (1f64, 0f64);
    let t0 = Instant::now();
    for s in 0..g.steps {
        let p = match &g.sched[s] {
            Phase::Vp => loop {
                let dummy = [cfg::CARD as i64; 8];
                if let Some(p) = sc.prepare(Some(&dummy), Some(&dummy), Some(PAD)) {
                    break p;
                }
            },
            Phase::Silence => sc.prepare(Some(&SINE), Some(&SILENCE), Some(PAD)).unwrap(),
            Phase::Text(t) => sc.prepare(Some(&SINE), Some(&SILENCE), Some(*t)).unwrap(),
            Phase::User(c) => sc.prepare(Some(c), None, None).unwrap(),
        };

        let hidden = &g.tt_hidden[s * cfg::DIM..(s + 1) * cfg::DIM];
        let sampled_text =
            argmax(&g.tt_text_logits[s * cfg::TEXT_LOGITS..(s + 1) * cfg::TEXT_LOGITS]) as i64;
        let next_text = if p.provided[0] {
            p.target[0]
        } else {
            sampled_text
        };
        text_hits += (next_text == g.dep_in_text[s]) as usize;

        let teacher: [i64; cfg::DEP_Q] = g.dep_tokens[s * cfg::DEP_Q..(s + 1) * cfg::DEP_Q]
            .try_into()
            .unwrap();
        let toks = fast.frame(hidden, next_text, &p.forced(), Some(&teacher), None);
        for cb in 0..cfg::DEP_Q {
            let grow = &g.dep_logits
                [(s * cfg::DEP_Q + cb) * cfg::CARD..(s * cfg::DEP_Q + cb + 1) * cfg::CARD];
            let (c, d) = cos_maxd(&fast.logits()[cb * cfg::CARD..(cb + 1) * cfg::CARD], grow);
            min_cos_cb[cb] = min_cos_cb[cb].min(c);
            if c < min_cos {
                min_cos = c;
                worst = (s, cb);
            }
            max_d = max_d.max(d);
            argmax_hits += (toks[cb] == teacher[cb]) as usize;
        }

        if let Some(depth) = &burn_depth {
            let ht = burn::tensor::Tensor::<B, 1>::from_floats(hidden, &device).reshape([
                1,
                1,
                cfg::DIM,
            ]);
            let (btoks, blogits) =
                depth.frame(&ht, next_text, &p.forced(), Some(&teacher), None, &device);
            for cb in 0..cfg::DEP_Q {
                vs_token_hits += (toks[cb] == btoks[cb]) as usize;
                let (c, d) = cos_maxd(
                    &fast.logits()[cb * cfg::CARD..(cb + 1) * cfg::CARD],
                    &blogits[cb],
                );
                vs_min_cos = vs_min_cos.min(c);
                vs_max_d = vs_max_d.max(d);
            }
        }

        // teacher-forced commit: the cache follows the oracle trajectory
        let out = sc.commit(&p, sampled_text, &teacher);
        if s >= g.gen_start {
            let r = s - g.gen_start;
            let grow = &g.out_tokens[r * cfg::NUM_STREAMS..(r + 1) * cfg::NUM_STREAMS];
            out_hits += (out.as_ref().map(|o| &o[..]) == Some(grow)) as usize;
        }
        if s + 1 == g.n_vp {
            sc.overwrite(&g.vp_cache);
        }
        if (s + 1) % 16 == 0 || s + 1 == g.steps {
            eprintln!(
                "  step {:3}/{}  min cos={min_cos:.9}  ({:.0} ms/step)",
                s + 1,
                g.steps,
                t0.elapsed().as_secs_f64() / (s + 1) as f64 * 1e3
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
    let mut ok_burn = true;
    if burn_depth.is_some() {
        ok_burn = vs_token_hits == n_cb;
        println!(
            "  {} vs depth.rs  tokens {vs_token_hits}/{n_cb} exact, logits min cos={vs_min_cos:.9}, max|Δ|={vs_max_d:.3e}",
            if ok_burn { "OK" } else { "XX" }
        );
    }
    let (frames, total, cond, gemv, head, scalar) = fast.take_bench();
    println!(
        "timing (gate replay, {frames} frames): {total:.1} ms/frame = cond {cond:.1} + stack gemv {gemv:.1} + head {head:.1} + scalar {scalar:.1}"
    );
    println!(
        "ran {} steps in {secs:.1}s ({:.0} ms/step incl. burn ref)",
        g.steps,
        secs / g.steps as f64 * 1e3
    );

    verdict("DEPTH_FAST PARITY", ok_cos && ok_int && ok_burn);
}

// ──────────────────────────────── bench ───────────────────────────────────

fn median(mut v: Vec<f64>) -> f64 {
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    v[v.len() / 2]
}

fn bench(pile: &str, f16: bool, synth: bool) {
    let rounds: usize = std::env::var("MOSHI_ROUNDS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(8);
    let frames: usize = std::env::var("MOSHI_FRAMES")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(25);

    let mut fast = if synth {
        println!(
            "synthetic weights ({}) at real shapes …",
            if f16 { "f16" } else { "f32" }
        );
        let t0 = Instant::now();
        let f = DepthFast::synthetic(f16);
        println!("built in {:.1}s", t0.elapsed().as_secs_f64());
        f
    } else {
        println!(
            "loading depth_fast ({}) from {pile} …",
            if f16 { "f16" } else { "f32" }
        );
        let t0 = Instant::now();
        let loader = pile_loader(pile);
        let f = DepthFast::load(&loader, f16);
        println!("loaded in {:.1}s", t0.elapsed().as_secs_f64());
        f
    };

    // Drive: golden temporal hiddens when present (cycled), else a synthetic
    // hidden — the frame cost is shape-fixed either way. Free-run greedy.
    let (hiddens, n_h) = if Path::new(GOLD).join("tt_hidden.npy").exists() {
        let (h, s) = golden_f32("tt_hidden");
        assert_eq!(s[1], cfg::DIM);
        let n = s[0];
        (h, n)
    } else {
        println!("(no goldens found — synthetic hidden)");
        (
            (0..cfg::DIM)
                .map(|i| ((i * 37 % 101) as f32 - 50.0) / 50.0)
                .collect(),
            1,
        )
    };
    let forced: [Option<i64>; cfg::DEP_Q] = [None; cfg::DEP_Q];

    let gib = fast.frame_weight_bytes() as f64 / (1u64 << 30) as f64;
    println!(
        "bench: {rounds} rounds × {frames} frames, {gib:.2} GiB weights/frame, \
         MARY_DEPTH_THREADS={} MARY_PRED_THREADS={}",
        std::env::var("MARY_DEPTH_THREADS").unwrap_or_else(|_| "6 (default)".into()),
        std::env::var("MARY_PRED_THREADS").unwrap_or_else(|_| "2 (default)".into()),
    );

    // warmup: touch every weight twice (page-in + pool spin-up)
    for w in 0..2 {
        fast.frame(
            &hiddens[(w % n_h) * cfg::DIM..][..cfg::DIM],
            PAD,
            &forced,
            None,
            None,
        );
    }
    let _ = fast.take_bench();

    let mut meds: Vec<f64> = Vec::with_capacity(rounds);
    let mut i = 0usize;
    for r in 0..rounds {
        let mut times: Vec<f64> = Vec::with_capacity(frames);
        for _ in 0..frames {
            let h = &hiddens[(i % n_h) * cfg::DIM..][..cfg::DIM];
            i += 1;
            let t0 = Instant::now();
            fast.frame(h, PAD, &forced, None, None);
            times.push(t0.elapsed().as_secs_f64() * 1e3);
        }
        let med = median(times);
        meds.push(med);
        eprintln!("  round {:2}/{rounds}: median {med:.2} ms/frame", r + 1);
    }
    let best = meds.iter().cloned().fold(f64::MAX, f64::min);
    let (nf, total, cond, gemv, head, scalar) = fast.take_bench();
    let bw = fast.frame_weight_bytes() as f64 / (best / 1e3) / 1e9;

    println!(
        "decomposition (mean over {nf} frames): {total:.2} ms = cond {cond:.2} + stack gemv {gemv:.2} + head {head:.2} + scalar attn/norm/silu {scalar:.2}"
    );
    println!(
        "RESULT depth_fast {} : min-of-medians {best:.2} ms / 16-step frame  ({bw:.0} GB/s effective weight bandwidth)",
        if f16 { "f16" } else { "f32" }
    );
}

fn main() {
    cpu::set_interactive_qos();
    let args: Vec<String> = std::env::args().collect();
    let sub = args.get(1).map(String::as_str).unwrap_or("");
    let f16 = args.iter().any(|a| a == "--f16");
    let skip_burn = args.iter().any(|a| a == "--skip-burn");
    let synth = args.iter().any(|a| a == "--synth");
    let pile = mary::paths::model(
        args.iter().skip(2).find(|a| !a.starts_with("--")).map(String::as_str),
        "personaplex.pile",
    )
    .unwrap_or_else(|e| {
        eprintln!("{e}");
        std::process::exit(2)
    })
    .to_string_lossy()
    .into_owned();
    match sub {
        "gate" => gate(&pile, f16, !skip_burn),
        "bench" => bench(&pile, f16, synth),
        _ => {
            eprintln!("usage: moshi_depth_probe <gate|bench> [--f16] [--skip-burn|--synth] [pile]");
            eprintln!("  gate   teacher-forced parity vs goldens + exact tokens vs depth.rs");
            eprintln!("  bench  min-of-medians ms per 16-step frame (MOSHI_ROUNDS × MOSHI_FRAMES)");
            eprintln!("  env    MARY_DEPTH_THREADS (f16 NEON pool, default 6)");
            eprintln!("         MARY_PRED_THREADS  (f32 Accelerate pool, default 2)");
            std::process::exit(2);
        }
    }
}
