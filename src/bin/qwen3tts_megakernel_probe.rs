//! Megakernel probe — parity + performance for the hand-fused talker decode
//! step (`qwen3tts::megakernel`), plus the persistent-kernel microbenchmarks
//! that motivated/bounded the design (transferable to jerky/GPU-succinct).
//!
//!   cargo run --release --features megakernel --bin qwen3tts_megakernel_probe
//!     [-- --prefill 400 --steps 16 --bench 50 --skip-fused --skip-micro]
//!
//! Phases:
//!   1. parity   — synthetic prefill through the Burn talker; N teacher-forced
//!                 decode steps through BOTH paths (same inputs); gate
//!                 cos(hidden) ≥ 0.999 per step (expected ≈ 1.0 — same math,
//!                 different fp association), logits argmax match reported.
//!   2. bench    — M frames per path, submit time and full frame time
//!                 (submit + one-sync readback), interleaved:
//!                   burn-raw   (non-fused Metal, the engine's host backend)
//!                   engine     (141 dispatches/frame, this module)
//!                   burn-fused (BFused — the production baseline)
//!   3. micro    — per-dispatch host overhead (encode-only vs sync-per-op) and
//!                 the persistent-kernel experiment: a K-step dependent matvec
//!                 chain as K multi-cube dispatches vs ONE single-cube
//!                 persistent dispatch (the only legal persistent form on
//!                 wgpu/Metal — no grid-wide barrier exists).

use burn::prelude::*;
use mary::models::qwen3tts::config::*;
use mary::models::qwen3tts::megakernel::{self, TalkerEngine};
use mary::models::qwen3tts::talker::Talker;
use mary::nn::backend::{BFused, BFusedHalf};
use mary::nn::weight_loader::WeightLoader;
use std::path::Path;
use std::time::Instant;

const PILE: &str = "models/qwen3tts.pile";

type Raw = megakernel::Raw;

fn cos(a: &[f32], b: &[f32]) -> f64 {
    let (mut dot, mut na, mut nb) = (0f64, 0f64, 0f64);
    for (x, y) in a.iter().zip(b) {
        let (x, y) = (*x as f64, *y as f64);
        dot += x * y;
        na += x * x;
        nb += y * y;
    }
    dot / (na.sqrt() * nb.sqrt())
}

fn argmax(v: &[f32]) -> usize {
    let mut best = 0;
    for (i, &x) in v.iter().enumerate() {
        if x > v[best] {
            best = i;
        }
    }
    best
}

/// Deterministic pseudo-random codec ids in [0, 2048).
fn synth_ids(n: usize, mut seed: u64) -> Vec<u32> {
    (0..n)
        .map(|_| {
            seed = seed
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((seed >> 33) % 2048) as u32
        })
        .collect()
}

fn arg(name: &str, default: usize) -> usize {
    let args: Vec<String> = std::env::args().collect();
    args.iter()
        .position(|a| a == name)
        .and_then(|i| args.get(i + 1))
        .and_then(|v| v.parse().ok())
        .unwrap_or(default)
}

fn flag(name: &str) -> bool {
    std::env::args().any(|a| a == name)
}

fn main() {
    mary::models::qwen3tts::cpu::set_interactive_qos();
    let prefill_len = arg("--prefill", 400);
    let steps = arg("--steps", 16);
    let bench = arg("--bench", 50);
    let max_seq = 2048;

    let dev: <Raw as burn::tensor::backend::BackendTypes>::Device = Default::default();
    // Weights come from the durable qwen3tts pile (same source as qwen3tts_say;
    // pile-vs-safetensors is bit-identical per qwen3tts_pile_test).
    let pile = std::env::var("QWEN3TTS_PILE").unwrap_or_else(|_| PILE.to_string());
    let loader = mary::persist::load_aliased_loader_from_pile(Path::new(&pile), "talker_f16")
        .unwrap_or_else(|e| panic!("load qwen3tts pile {pile}: {e:?}"));
    println!("loading talker (raw f32 backend)...");
    let t0 = Instant::now();
    let talker = Talker::<Raw>::load(&loader, &dev);
    println!("  {:.1}s", t0.elapsed().as_secs_f32());

    // ---- prefill (Burn path) --------------------------------------------
    let ids = synth_ids(prefill_len, 7);
    let embeds = talker.embed_codec(&ids, &dev);
    let mut caches = talker.new_caches();
    let hidden = talker.forward(embeds, &mut caches, &dev);
    let _ = talker.last_hidden(hidden); // drain the prefill before timing anything

    // ---- engine build + cache import ------------------------------------
    println!("building engine (aliasing talker weight buffers)...");
    let mut engine = TalkerEngine::new(&talker, max_seq);
    engine.import_caches(&caches);
    assert_eq!(engine.len(), prefill_len);
    println!(
        "  {} dispatches/frame (burn path: ~18/layer ≈ {}+)",
        TalkerEngine::DISPATCHES_PER_STEP,
        TALKER_LAYERS * 18
    );

    // ---- phase 1: teacher-forced parity ----------------------------------
    println!("\nparity: {steps} teacher-forced steps (identical inputs to both paths)");
    let step_ids = synth_ids(steps, 99);
    let mut ok = true;
    let mut min_cos = 1.0f64;
    let mut tok_match = 0;
    for (s, &id) in step_ids.iter().enumerate() {
        let row = talker.codec_row(id).to_vec();

        // Burn path
        let e = Tensor::<Raw, 1>::from_floats(row.as_slice(), &dev).reshape([1, 1, TALKER_HIDDEN]);
        let h_ref = talker.last_hidden(talker.forward(e, &mut caches, &dev));
        // engine path
        let h_eng = engine.step(&row);

        let c = cos(&h_ref, &h_eng);
        min_cos = min_cos.min(c);
        let (a_ref, a_eng) = (
            argmax(&talker.logits_from(&h_ref)),
            argmax(&talker.logits_from(&h_eng)),
        );
        if a_ref == a_eng {
            tok_match += 1;
        }
        let pass = c > 0.999;
        ok &= pass;
        if s < 4 || !pass {
            println!(
                "  {} step {s:2} cos={c:.9} argmax {}/{}",
                if pass { "✓" } else { "✗" },
                a_ref,
                a_eng
            );
        }
    }
    println!(
        "  {} min cos={min_cos:.9} over {steps} steps; argmax match {tok_match}/{steps}",
        if ok { "✓" } else { "✗" }
    );

    // ---- phase 2: bench ---------------------------------------------------
    // The machine is shared (ambient daemons + other concurrent sessions swing
    // throughput 4-10x, see PORT_NOTES). Protocol: sequential per-path blocks
    // (interleaving per-frame makes the paths fight each other for CPU),
    // repeated over `--rounds`, per-block medians, and a final min-of-medians
    // per path — min because contention only ever inflates.
    let rounds = arg("--rounds", 3);
    println!(
        "\nbench: {bench} frames/path x {rounds} rounds at seq≈{}",
        prefill_len + steps
    );
    let bench_ids = synth_ids(bench, 1234);

    let mut fused = if flag("--skip-fused") {
        None
    } else {
        println!("  loading talker again on BFused (production baseline)...");
        let fdev: <BFused as burn::tensor::backend::BackendTypes>::Device = Default::default();
        let ftalker = Talker::<BFused>::load(&loader, &fdev);
        let fids = synth_ids(prefill_len, 7);
        let fembeds = ftalker.embed_codec(&fids, &fdev);
        let mut fcaches = ftalker.new_caches();
        let fh = ftalker.forward(fembeds, &mut fcaches, &fdev);
        let _ = ftalker.last_hidden(fh);
        // warm the decode-shape JIT like the other paths' parity steps did
        for &id in &step_ids {
            let row = ftalker.codec_row(id).to_vec();
            let e = Tensor::<BFused, 1>::from_floats(row.as_slice(), &fdev).reshape([
                1,
                1,
                TALKER_HIDDEN,
            ]);
            let h = ftalker.forward(e, &mut fcaches, &fdev);
            let _ = ftalker.last_hidden(h);
        }
        Some((ftalker, fcaches, fdev))
    };

    let med = |v: &mut Vec<(f64, f64)>| -> (f64, f64) {
        let mut a: Vec<f64> = v.iter().map(|x| x.0).collect();
        let mut b: Vec<f64> = v.iter().map(|x| x.1).collect();
        a.sort_by(|x, y| x.partial_cmp(y).unwrap());
        b.sort_by(|x, y| x.partial_cmp(y).unwrap());
        (a[a.len() / 2] * 1e3, b[b.len() / 2] * 1e3)
    };
    let mut best_raw = (f64::MAX, f64::MAX);
    let mut best_eng = (f64::MAX, f64::MAX);
    let mut best_fus = (f64::MAX, f64::MAX);
    for round in 0..rounds {
        let mut m_raw: Vec<(f64, f64)> = Vec::new();
        let mut m_eng: Vec<(f64, f64)> = Vec::new();
        let mut m_fus: Vec<(f64, f64)> = Vec::new();
        for &id in &bench_ids {
            let row = talker.codec_row(id).to_vec();
            let t0 = Instant::now();
            let e =
                Tensor::<Raw, 1>::from_floats(row.as_slice(), &dev).reshape([1, 1, TALKER_HIDDEN]);
            let h = talker.forward(e, &mut caches, &dev);
            let t1 = t0.elapsed().as_secs_f64();
            let _ = talker.last_hidden(h);
            m_raw.push((t1, t0.elapsed().as_secs_f64()));
        }
        for &id in &bench_ids {
            let row = talker.codec_row(id).to_vec();
            let t0 = Instant::now();
            engine.step_submit(&row);
            let t1 = t0.elapsed().as_secs_f64();
            let _ = engine.read_hidden();
            m_eng.push((t1, t0.elapsed().as_secs_f64()));
        }
        if let Some((ftalker, fcaches, fdev)) = fused.as_mut() {
            for &id in &bench_ids {
                let frow = ftalker.codec_row(id).to_vec();
                let t0 = Instant::now();
                let e = Tensor::<BFused, 1>::from_floats(frow.as_slice(), fdev).reshape([
                    1,
                    1,
                    TALKER_HIDDEN,
                ]);
                let h = ftalker.forward(e, fcaches, fdev);
                let t1 = t0.elapsed().as_secs_f64();
                let _ = ftalker.last_hidden(h);
                m_fus.push((t1, t0.elapsed().as_secs_f64()));
            }
        }
        let (r, e2) = (med(&mut m_raw), med(&mut m_eng));
        best_raw = (best_raw.0.min(r.0), best_raw.1.min(r.1));
        best_eng = (best_eng.0.min(e2.0), best_eng.1.min(e2.1));
        if !m_fus.is_empty() {
            let f = med(&mut m_fus);
            best_fus = (best_fus.0.min(f.0), best_fus.1.min(f.1));
            println!(
                "  round {round}: raw {:6.2}/{:6.2}  engine {:5.2}/{:6.2}  fused {:6.2}/{:6.2}  (submit/full ms)",
                r.0, r.1, e2.0, e2.1, f.0, f.1
            );
        } else {
            println!(
                "  round {round}: raw {:6.2}/{:6.2}  engine {:5.2}/{:6.2}  (submit/full ms)",
                r.0, r.1, e2.0, e2.1
            );
        }
    }
    println!("  min-of-medians:");
    println!(
        "  burn-raw   submit {:6.2} ms/frame  full {:6.2} ms/frame",
        best_raw.0, best_raw.1
    );
    println!(
        "  engine     submit {:6.2} ms/frame  full {:6.2} ms/frame",
        best_eng.0, best_eng.1
    );
    if best_fus.0 < f64::MAX {
        println!(
            "  burn-fused submit {:6.2} ms/frame  full {:6.2} ms/frame",
            best_fus.0, best_fus.1
        );
        println!(
            "\n  engine vs fused: submit {:.1}x, full frame {:.1}x",
            best_fus.0 / best_eng.0,
            best_fus.1 / best_eng.1
        );
    }

    // production-exact baseline: the f16 fused talker (what qwen3tts_say runs)
    if flag("--fused-f16") {
        println!("\nfused-f16 (BFusedHalf, the production talker):");
        let fids = synth_ids(prefill_len, 7);
        let meds = bench_fused::<BFusedHalf>(&loader, &fids, &step_ids, &bench_ids, rounds);
        let mut best = (f64::MAX, f64::MAX);
        for (r, m) in meds.iter().enumerate() {
            println!(
                "  round {r}: fused-f16 {:6.2}/{:6.2}  (submit/full ms)",
                m.0, m.1
            );
            best = (best.0.min(m.0), best.1.min(m.1));
        }
        println!(
            "  min-of-medians: submit {:6.2} ms/frame  full {:6.2} ms/frame",
            best.0, best.1
        );
    }

    // ---- phase 3: microbenchmarks ----------------------------------------
    if !flag("--skip-micro") {
        micro();
    }

    if !ok {
        std::process::exit(1);
    }
}

/// Bench one Burn-fused talker variant (f32 `BFused` or the production-exact
/// f16 `BFusedHalf`): fresh load + prefill + JIT warm, then `rounds` blocks of
/// `bench_ids` frames. Returns per-round (submit, full) medians in ms.
fn bench_fused<B: Backend>(
    loader: &WeightLoader,
    prefill_ids: &[u32],
    warm_ids: &[u32],
    bench_ids: &[u32],
    rounds: usize,
) -> Vec<(f64, f64)> {
    let dev: <B as burn::tensor::backend::BackendTypes>::Device = Default::default();
    let talker = Talker::<B>::load(loader, &dev);
    let embeds = talker.embed_codec(prefill_ids, &dev);
    let mut caches = talker.new_caches();
    let h = talker.forward(embeds, &mut caches, &dev);
    let _ = talker.last_hidden(h);
    for &id in warm_ids {
        let row = talker.codec_row(id).to_vec();
        let e = Tensor::<B, 1>::from_floats(row.as_slice(), &dev).reshape([1, 1, TALKER_HIDDEN]);
        let h = talker.forward(e, &mut caches, &dev);
        let _ = talker.last_hidden(h);
    }
    let mut out = Vec::new();
    for _ in 0..rounds {
        let mut m: Vec<(f64, f64)> = Vec::new();
        for &id in bench_ids {
            let row = talker.codec_row(id).to_vec();
            let t0 = Instant::now();
            let e =
                Tensor::<B, 1>::from_floats(row.as_slice(), &dev).reshape([1, 1, TALKER_HIDDEN]);
            let h = talker.forward(e, &mut caches, &dev);
            let t1 = t0.elapsed().as_secs_f64();
            let _ = talker.last_hidden(h);
            m.push((t1, t0.elapsed().as_secs_f64()));
        }
        let mut a: Vec<f64> = m.iter().map(|x| x.0).collect();
        let mut b: Vec<f64> = m.iter().map(|x| x.1).collect();
        a.sort_by(|x, y| x.partial_cmp(y).unwrap());
        b.sort_by(|x, y| x.partial_cmp(y).unwrap());
        out.push((a[a.len() / 2] * 1e3, b[b.len() / 2] * 1e3));
    }
    out
}

/// Per-dispatch overhead + the persistent-kernel experiment.
fn micro() {
    use cubecl::prelude::*;
    use mary::models::qwen3tts::megakernel::{
        chain_matvec_kernel, client_for_device, persistent_chain_kernel, touch_kernel,
    };

    println!("\nmicro: dispatch overhead + persistent-kernel experiment");
    let client = client_for_device();
    let n: u32 = 1024;
    let k_steps: u32 = 100;

    let touch = client.create_from_slice(&[0u8; 4]);
    let w_host: Vec<f32> = (0..n * n)
        .map(|i| ((i % 61) as f32 - 30.0) / 900.0)
        .collect();
    let x_host: Vec<f32> = (0..2 * n).map(|i| ((i % 17) as f32 - 8.0) / 8.0).collect();
    let bytes =
        |v: &[f32]| unsafe { core::slice::from_raw_parts(v.as_ptr() as *const u8, v.len() * 4) };
    let w = client.create_from_slice(bytes(&w_host));
    let buf = client.create_from_slice(bytes(&x_host));

    // warm compile all three kernels
    unsafe {
        touch_kernel::launch_unchecked::<burn::backend::wgpu::WgpuRuntime>(
            &client,
            CubeCount::new_single(),
            CubeDim::new_1d(32),
            ArrayArg::from_raw_parts(touch.clone(), 1),
        );
        chain_matvec_kernel::launch_unchecked::<burn::backend::wgpu::WgpuRuntime>(
            &client,
            CubeCount::new_1d(n / 128),
            CubeDim::new_1d(128),
            ArrayArg::from_raw_parts(buf.clone(), 2 * n as usize),
            ArrayArg::from_raw_parts(w.clone(), (n * n) as usize),
            0,
            n,
            n,
        );
        persistent_chain_kernel::launch_unchecked::<burn::backend::wgpu::WgpuRuntime>(
            &client,
            CubeCount::new_single(),
            CubeDim::new_1d(256),
            ArrayArg::from_raw_parts(buf.clone(), 2 * n as usize),
            ArrayArg::from_raw_parts(w.clone(), (n * n) as usize),
            2,
            n,
            256,
        );
    }
    let _ = client.read_one(touch.clone()).unwrap();

    // (a) encode-only cost per dispatch (one flush at the end)
    let reps: u32 = 1000;
    let t0 = Instant::now();
    for _ in 0..reps {
        unsafe {
            touch_kernel::launch_unchecked::<burn::backend::wgpu::WgpuRuntime>(
                &client,
                CubeCount::new_single(),
                CubeDim::new_1d(32),
                ArrayArg::from_raw_parts(touch.clone(), 1),
            );
        }
    }
    let encode_us = t0.elapsed().as_secs_f64() / reps as f64 * 1e6;
    let _ = client.read_one(touch.clone()).unwrap();
    let batched_us = t0.elapsed().as_secs_f64() / reps as f64 * 1e6;

    // (b) full round trip per dispatch (sync each)
    let reps_rt: u32 = 100;
    let t0 = Instant::now();
    for _ in 0..reps_rt {
        unsafe {
            touch_kernel::launch_unchecked::<burn::backend::wgpu::WgpuRuntime>(
                &client,
                CubeCount::new_single(),
                CubeDim::new_1d(32),
                ArrayArg::from_raw_parts(touch.clone(), 1),
            );
        }
        let _ = client.read_one(touch.clone()).unwrap();
    }
    let roundtrip_us = t0.elapsed().as_secs_f64() / reps_rt as f64 * 1e6;
    println!(
        "  dispatch cost: encode {encode_us:.1} µs, encode+drain(batched) {batched_us:.1} µs, sync-per-op {roundtrip_us:.1} µs"
    );

    // (c) K-step dependent matvec chain: K dispatches vs 1 persistent dispatch
    let chain_bytes = (k_steps as f64) * (n as f64) * (n as f64) * 4.0;
    let t0 = Instant::now();
    for s in 0..k_steps {
        let (src, dst) = if s % 2 == 0 { (0, n) } else { (n, 0) };
        unsafe {
            chain_matvec_kernel::launch_unchecked::<burn::backend::wgpu::WgpuRuntime>(
                &client,
                CubeCount::new_1d(n / 128),
                CubeDim::new_1d(128),
                ArrayArg::from_raw_parts(buf.clone(), 2 * n as usize),
                ArrayArg::from_raw_parts(w.clone(), (n * n) as usize),
                src,
                dst,
                n,
            );
        }
    }
    let _ = client.read_one(buf.clone()).unwrap();
    let multi_s = t0.elapsed().as_secs_f64();

    for dim in [128u32, 256, 512, 1024] {
        let t0 = Instant::now();
        unsafe {
            persistent_chain_kernel::launch_unchecked::<burn::backend::wgpu::WgpuRuntime>(
                &client,
                CubeCount::new_single(),
                CubeDim::new_1d(dim),
                ArrayArg::from_raw_parts(buf.clone(), 2 * n as usize),
                ArrayArg::from_raw_parts(w.clone(), (n * n) as usize),
                k_steps,
                n,
                dim,
            );
        }
        let _ = client.read_one(buf.clone()).unwrap();
        let pers_s = t0.elapsed().as_secs_f64();
        println!(
            "  chain n={n} K={k_steps}: multi-dispatch {:.2} ms ({:.0} GB/s) vs persistent 1-cube dim={dim} {:.2} ms ({:.0} GB/s)",
            multi_s * 1e3,
            chain_bytes / multi_s / 1e9,
            pers_s * 1e3,
            chain_bytes / pers_s / 1e9,
        );
    }
}
