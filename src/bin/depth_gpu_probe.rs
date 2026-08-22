//! PersonaPlex depth transformer on cubecl ([`mary::models::personaplex::depth_gpu`])
//! — dispatch-cost probe, GPU-vs-CPU parity gate, per-frame bench.
//!
//!   cargo run --release --features personaplex,q4 --bin depth_gpu_probe -- \
//!     dispatch
//!   cargo run --release --features personaplex,q4 --bin depth_gpu_probe -- \
//!     bench <pile-path> [--fmt q4|q8|f16] [--nq N]
//!   cargo run --release --features personaplex,q4 --bin depth_gpu_probe -- \
//!     bench --synth [--fmt q4|q8|f16] [--nq N]
//!   cargo run --release --features personaplex,q4 --bin depth_gpu_probe -- \
//!     gate <pile-path> [--fmt q4|q8|f16] [--frames N]
//!
//! `dispatch` measures the per-dispatch floor of the device — the number the
//! depth port's kernel layout is designed against (its matvecs are small
//! enough that launch cost and streaming cost are the same order).
//!
//! `bench` free-runs whole frames and reports **p50 / min / max ms per frame**
//! over `--rounds × --frames` (cold first pass discarded), for the GPU port and
//! — unless `--synth` — the CPU predictor `depth_fast` on the SAME weights, so
//! the before/after is one process and one machine state.
//!
//! `gate` is the parity measure. There is no bit-exactness bar (quantization
//! is a real numerics change): the CPU f32 `depth_fast` is the reference and
//! the report is (a) per-codebook logit cosine and (b) **codebook-token
//! agreement rate**, both teacher-forced onto the CPU reference's trajectory —
//! so the 16 comparisons stay independent of upstream token flips — plus the
//! free-running agreement, which is what the realtime loop actually gets.
//!
//! The moshi golden captures (`/tmp/mary-personaplex/golden`) are session
//! scratch and were gone when this was written, so the gate does NOT re-derive
//! oracle parity; it inherits it. `depth_fast` is gated at ~1.0 cosine and
//! full argmax agreement against those goldens by `moshi_depth_probe gate`,
//! and this probe measures the GPU port against `depth_fast`. Inputs are
//! synthesized at the temporal stack's OWN output profile — a random direction
//! shaped by the checkpoint's `out_norm.alpha`, which is what gives
//! `transformer_out` its per-dimension scale — not white noise.

use mary::models::personaplex::config as cfg;
use mary::models::personaplex::depth_fast::DepthFast;
use mary::models::personaplex::depth_gpu::{DepthGpu, NO_FORCE};
use mary::models::personaplex::temporal_metal::WeightFmt;
use mary::nn::q4;
use mary::nn::weight_loader::WeightLoader;
use std::path::Path;
use std::time::Instant;

fn pile_arg(args: &[String]) -> Option<&str> {
    let mut pile = None;
    let mut i = 1; // mode
    while i < args.len() {
        match args[i].as_str() {
            "--fmt" | "--nq" | "--frames" | "--rounds" => {
                assert!(i + 1 < args.len(), "{} requires a value", args[i]);
                i += 2;
            }
            "--synth" | "--skip-cpu" => i += 1,
            flag if flag.starts_with("--") => panic!("unknown flag {flag}"),
            path => {
                assert!(
                    pile.replace(path).is_none(),
                    "more than one pile path supplied"
                );
                i += 1;
            }
        }
    }
    pile
}

fn require_pile<'a>(mode: &str, pile: Option<&'a str>) -> &'a str {
    pile.unwrap_or_else(|| {
        panic!("{mode} requires an explicit PersonaPlex pile path (or use bench --synth)")
    })
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let mode = args.first().cloned().unwrap_or_else(|| "dispatch".into());
    let flag = |name: &str| args.iter().any(|a| a == name);
    let val = |name: &str| -> Option<String> {
        args.iter()
            .position(|a| a == name)
            .and_then(|i| args.get(i + 1))
            .cloned()
    };
    let fmt = val("--fmt")
        .map(|s| WeightFmt::parse(&s).unwrap_or_else(|| panic!("bad --fmt {s}")))
        .unwrap_or(WeightFmt::Q8);
    let pile = pile_arg(&args);
    let nq: usize = val("--nq")
        .map(|s| s.parse().unwrap())
        .unwrap_or(cfg::DEP_Q);
    let frames: usize = val("--frames").map(|s| s.parse().unwrap()).unwrap_or(25);
    let rounds: usize = val("--rounds").map(|s| s.parse().unwrap()).unwrap_or(8);
    let synth = flag("--synth");

    match mode.as_str() {
        "dispatch" => {
            assert!(pile.is_none(), "dispatch does not accept a pile path");
            dispatch();
        }
        "bench" => {
            if synth {
                assert!(pile.is_none(), "bench --synth does not accept a pile path");
            }
            bench(fmt, nq, frames, rounds, synth, pile, flag("--skip-cpu"));
        }
        "gate" => gate(fmt, frames, require_pile("gate", pile)),
        m => panic!("unknown mode {m} (dispatch | bench | gate)"),
    }
}

// ---------------------------------------------------------------------------

/// Per-dispatch floor: N launches of a trivially small matvec, one sync at the
/// end. Isolates submit + inter-kernel gap from any bandwidth term.
fn dispatch() {
    let c = q4::client_for_default_device();
    for (out, inn) in [(8usize, 32usize), (1024, 1024)] {
        let w = c.empty(out * inn * 2); // f16
        let x = c.empty(inn * 4);
        let y = c.empty(out * 4);
        for _ in 0..64 {
            q4::f16_matvec(&c, &x, &w, &y, out, inn);
        }
        let _ = c.read_one(y.clone()).unwrap();
        let mut best = f64::MAX;
        for _ in 0..8 {
            let n = 1000;
            let t = Instant::now();
            for _ in 0..n {
                q4::f16_matvec(&c, &x, &w, &y, out, inn);
            }
            let submit = t.elapsed().as_secs_f64();
            let _ = c.read_one(y.clone()).unwrap();
            let total = t.elapsed().as_secs_f64();
            best = best.min(total / n as f64);
            eprintln!(
                "  [{out}x{inn}] submit {:.2} us/disp  total {:.2} us/disp",
                submit / n as f64 * 1e6,
                total / n as f64 * 1e6
            );
        }
        println!(
            "RESULT dispatch [{out}x{inn}] best {:.2} us/dispatch",
            best * 1e6
        );
    }
}

// ---------------------------------------------------------------------------

fn pile_loader(pile: &str) -> WeightLoader {
    mary::persist::personaplex_loader(Path::new(pile)).unwrap_or_else(|e| panic!("pile load: {e}"))
}

/// Deterministic uniform-ish noise in `[-0.5, 0.5)`.
struct Rnd(u64);
impl Rnd {
    fn next(&mut self) -> f32 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((self.0 >> 40) as f32) / (1u32 << 24) as f32 - 0.5
    }
}

/// `transformer_out` at the temporal stack's own output profile: a random unit
/// direction RMS-normalized to 1 and then scaled per dimension by
/// `out_norm.alpha` — the last thing the temporal stack applies, and what
/// gives its hidden its per-dimension scale. Falls back to a plain
/// RMS-1 direction when the alpha isn't available.
fn hidden_like(rnd: &mut Rnd, alpha: Option<&[f32]>) -> Vec<f32> {
    let mut v: Vec<f32> = (0..cfg::DIM).map(|_| rnd.next()).collect();
    let rms = (v.iter().map(|x| (x * x) as f64).sum::<f64>() / v.len() as f64).sqrt() as f32;
    for (i, x) in v.iter_mut().enumerate() {
        *x = *x / rms * alpha.map_or(1.0, |a| a[i]);
    }
    v
}

fn out_norm_alpha(loader: &WeightLoader) -> Vec<f32> {
    let (a, s) = loader.load_f32("out_norm.alpha");
    assert_eq!(s, vec![1, 1, cfg::DIM], "out_norm.alpha shape");
    a
}

fn fmt_name(fmt: WeightFmt) -> &'static str {
    match fmt {
        WeightFmt::Q4 => "q4",
        WeightFmt::Q8 => "q8",
        WeightFmt::F16 => "f16",
    }
}

fn stats(mut v: Vec<f64>) -> (f64, f64, f64) {
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    (v[v.len() / 2], v[0], v[v.len() - 1])
}

// ---------------------------------------------------------------------------

#[allow(clippy::too_many_arguments)]
fn bench(
    fmt: WeightFmt,
    nq: usize,
    frames: usize,
    rounds: usize,
    synth: bool,
    pile: Option<&str>,
    skip_cpu: bool,
) {
    let t0 = Instant::now();
    let (mut gpu, mut cpu, alpha) = if synth {
        eprintln!("building synthetic depth_gpu ({}) …", fmt_name(fmt));
        (DepthGpu::synthetic(fmt), None, None)
    } else {
        let pile = require_pile("bench", pile);
        eprintln!("loading depth_gpu ({}) from {pile} …", fmt_name(fmt));
        let loader = pile_loader(pile);
        let alpha = out_norm_alpha(&loader);
        let g = DepthGpu::load(&loader, fmt);
        let c = if skip_cpu {
            None
        } else {
            eprintln!("loading depth_fast (f16 storage, the CPU lane's fastest) …");
            Some(DepthFast::load(&loader, true))
        };
        (g, c, Some(alpha))
    };
    eprintln!("  loaded in {:.1} s", t0.elapsed().as_secs_f64());

    let mut rnd = Rnd(0xC0FFEE);
    let hiddens: Vec<Vec<f32>> = (0..frames)
        .map(|_| hidden_like(&mut rnd, alpha.as_deref()))
        .collect();
    let free = vec![NO_FORCE; cfg::DEP_Q];
    let none = [None; cfg::DEP_Q];

    // ── GPU ──
    let mut per_frame: Vec<f64> = Vec::new();
    for r in 0..=rounds {
        let mut round: Vec<f64> = Vec::with_capacity(frames);
        for h in &hiddens {
            let t = Instant::now();
            gpu.frame_submit(h, 0, &free, nq);
            let _ = gpu.tokens(nq); // the frame's single sync
            round.push(t.elapsed().as_secs_f64() * 1e3);
        }
        if r == 0 {
            eprintln!("  (cold pass discarded: p50 {:.2} ms)", stats(round).0);
            continue; // discard the cold pass (shader compile + first-touch)
        }
        per_frame.extend(round);
    }
    let (p50, lo, hi) = stats(per_frame.clone());
    let bytes = gpu.frame_weight_bytes() as f64 * (nq as f64 / cfg::DEP_Q as f64);
    println!(
        "RESULT depth_gpu {} n_q={nq} : p50 {p50:.2} ms/frame  (min {lo:.2}, max {hi:.2}, n={})  \
         {:.2} GB/frame, {:.0} GB/s effective",
        fmt_name(fmt),
        per_frame.len(),
        bytes / 1e9,
        bytes / (p50 * 1e-3) / 1e9,
    );

    // ── CPU reference, same process, same machine state ──
    if let Some(cpu) = cpu.as_mut() {
        let mut per_frame: Vec<f64> = Vec::new();
        for r in 0..=rounds.min(3) {
            let mut round: Vec<f64> = Vec::with_capacity(frames);
            for h in &hiddens {
                let t = Instant::now();
                cpu.frame(h, 0, &none, None, None);
                round.push(t.elapsed().as_secs_f64() * 1e3);
            }
            if r == 0 {
                continue;
            }
            per_frame.extend(round);
        }
        let (c50, clo, chi) = stats(per_frame.clone());
        println!(
            "RESULT depth_fast cpu-f16 n_q=16 : p50 {c50:.2} ms/frame  (min {clo:.2}, max {chi:.2}, n={})",
            per_frame.len()
        );
        println!(
            "RESULT speedup : {:.2}x  ({c50:.2} ms CPU -> {p50:.2} ms GPU)",
            c50 / p50
        );
    }
}

// ---------------------------------------------------------------------------

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

fn gate(fmt: WeightFmt, frames: usize, pile: &str) {
    eprintln!(
        "loading depth_gpu ({}) + depth_fast (f32) from {pile} …",
        fmt_name(fmt)
    );
    let loader = pile_loader(pile);
    let alpha = out_norm_alpha(&loader);
    let mut gpu = DepthGpu::load(&loader, fmt);
    // f32 storage: the CPU reference should carry no width penalty of its own.
    let mut cpu = DepthFast::load(&loader, false);
    drop(loader);

    let mut rnd = Rnd(0x5EED);
    let nq = cfg::DEP_Q;
    let none = [None; cfg::DEP_Q];

    let mut cos_min = f64::MAX;
    let mut cos_sum = 0f64;
    let mut cos_n = 0usize;
    let mut maxd_all = 0f64;
    let mut tf_agree = 0usize;
    let mut tf_total = 0usize;
    let mut free_agree = 0usize;
    let mut free_total = 0usize;
    let mut first_div: Option<(usize, usize)> = None;

    for f in 0..frames {
        let h = hidden_like(&mut rnd, Some(&alpha));
        let text_token = (rnd.next().abs() * 2.0 * cfg::TEXT_CARD as f32) as usize % cfg::TEXT_CARD;

        // 1. CPU reference, free-running — this run defines the teacher.
        let cpu_tokens = cpu.frame(&h, text_token as i64, &none, None, None);
        let cpu_logits = cpu.logits().to_vec();

        // 2. GPU teacher-forced onto the CPU trajectory: every step's INPUT is
        //    pinned, so the 16 logit rows are independent comparisons.
        let teacher: Vec<u32> = cpu_tokens.iter().map(|&t| t as u32).collect();
        gpu.frame_submit(&h, text_token as u32, &teacher, nq);
        let gpu_tokens = gpu.tokens(nq);
        let gpu_logits = gpu.logits(nq);
        for s in 0..nq {
            let (c, d) = cos_maxd(
                &cpu_logits[s * cfg::CARD..(s + 1) * cfg::CARD],
                &gpu_logits[s * cfg::CARD..(s + 1) * cfg::CARD],
            );
            cos_min = cos_min.min(c);
            cos_sum += c;
            cos_n += 1;
            maxd_all = maxd_all.max(d);
            if gpu_tokens[s] == cpu_tokens[s] {
                tf_agree += 1;
            } else if first_div.is_none() {
                first_div = Some((f, s));
            }
            tf_total += 1;
        }

        // 3. GPU free-running — what the realtime loop gets, chain divergence
        //    included.
        let free = vec![NO_FORCE; cfg::DEP_Q];
        let gpu_free = gpu.frame(&h, text_token as u32, &free, nq);
        for s in 0..nq {
            if gpu_free[s] == cpu_tokens[s] {
                free_agree += 1;
            }
            free_total += 1;
        }
    }

    println!(
        "RESULT gate {} : teacher-forced logits cos min {cos_min:.6} mean {:.6}, max|Δ| {maxd_all:.4}",
        fmt_name(fmt),
        cos_sum / cos_n as f64
    );
    println!(
        "RESULT gate {} : codebook-token agreement teacher-forced {tf_agree}/{tf_total} ({:.2}%){}",
        fmt_name(fmt),
        100.0 * tf_agree as f64 / tf_total as f64,
        match first_div {
            Some((f, s)) => format!(", first divergence frame {f} step {s}"),
            None => String::new(),
        }
    );
    println!(
        "RESULT gate {} : codebook-token agreement free-running {free_agree}/{free_total} ({:.2}%)",
        fmt_name(fmt),
        100.0 * free_agree as f64 / free_total as f64
    );
}

#[cfg(test)]
mod cli_tests {
    use super::*;

    fn args(values: &[&str]) -> Vec<String> {
        values.iter().map(|value| (*value).to_owned()).collect()
    }

    #[test]
    fn pile_path_is_explicit_and_flags_do_not_become_positionals() {
        let values = args(&[
            "gate",
            "--fmt",
            "q8",
            "/models/personaplex.pile",
            "--frames",
            "2",
        ]);
        assert_eq!(pile_arg(&values), Some("/models/personaplex.pile"));
    }

    #[test]
    fn synthetic_bench_needs_no_pile() {
        let values = args(&["bench", "--synth", "--fmt", "f16"]);
        assert_eq!(pile_arg(&values), None);
    }

    #[test]
    #[should_panic(expected = "gate requires an explicit PersonaPlex pile path")]
    fn gate_without_pile_is_rejected() {
        let values = args(&["gate", "--frames", "2"]);
        let _ = require_pile("gate", pile_arg(&values));
    }
}
