//! Mimi **encoder** CPU-vs-GPU bench + parity probe.
//!
//! The streaming encoder is on the realtime critical path: one 80 ms frame of
//! 24 kHz audio must become eight codes inside the frame budget. The CPU
//! (Accelerate im2col) build costs ~27 ms/frame; this probe measures that lane
//! and the cubecl lane side by side on the same input, and reports how far the
//! two agree.
//!
//! Parity is judged **statistically, not bit-exactly**: the GPU reduction order
//! differs from Accelerate's, so the argmin over 2048 codebook rows can and
//! does flip when two rows are within reduction noise. What matters is the
//! agreement rate on the emitted tokens and the relative error of the
//! continuous latent that feeds them.
//!
//!   cargo run --release --features qwen3tts,q4 --bin mimi_gpu_probe -- <pile>

use std::time::Instant;

use mary::models::personaplex::mimi::config::*;
use mary::nn::weight_loader::WeightLoader;
use mary::models::personaplex::mimi::{MimiEncoder, MimiEncoderGpu};
use std::collections::HashMap;
use std::path::Path;

/// A deterministic stand-in checkpoint with the real shapes.
///
/// The point is portability of the GATE, not of the numbers: the CPU and GPU
/// encoders are both built from this map, so CPU-vs-GPU parity can be measured
/// on any device — including a CUDA box with no PersonaPlex pile on it — which
/// is the whole reason the port targets cubecl rather than Metal.
///
/// Weights are scaled `1/sqrt(fan_in)` so activations neither die nor explode
/// through fourteen convolutions; codebooks are unit-ish, which makes the RVQ
/// argmin HARDER than the real one (random rows sit closer together than
/// trained ones), so a high agreement rate here is not a flattering result.
fn synthetic_weights() -> WeightLoader {
    let mut rng = 0x9E37_79B9_7F4A_7C15u64;
    let mut next = move || {
        rng ^= rng << 13;
        rng ^= rng >> 7;
        rng ^= rng << 17;
        ((rng >> 40) as f32 / 8388608.0) - 1.0
    };
    let mut m: HashMap<String, (Vec<f32>, Vec<usize>)> = HashMap::new();
    let mut put = |m: &mut HashMap<_, _>, name: String, shape: Vec<usize>, gain: f32, next: &mut dyn FnMut() -> f32| {
        let n: usize = shape.iter().product();
        let v: Vec<f32> = (0..n).map(|_| next() * gain).collect();
        m.insert(name, (v, shape));
    };
    let g = |fan: usize| (3.0f32 / fan as f32).sqrt();

    let p = "encoder.model";
    put(&mut m, format!("{p}.0.conv.conv.weight"), vec![64, 1, 7], g(7), &mut next);
    put(&mut m, format!("{p}.0.conv.conv.bias"), vec![64], 0.01, &mut next);
    for (i, &r) in ENC_RATIOS.iter().enumerate() {
        let dim = 64usize << i;
        put(&mut m, format!("{p}.{}.block.1.conv.conv.weight", 3 * i + 1), vec![dim / 2, dim, 3], g(dim * 3), &mut next);
        put(&mut m, format!("{p}.{}.block.1.conv.conv.bias", 3 * i + 1), vec![dim / 2], 0.01, &mut next);
        put(&mut m, format!("{p}.{}.block.3.conv.conv.weight", 3 * i + 1), vec![dim, dim / 2, 1], g(dim / 2), &mut next);
        put(&mut m, format!("{p}.{}.block.3.conv.conv.bias", 3 * i + 1), vec![dim], 0.01, &mut next);
        put(&mut m, format!("{p}.{}.conv.conv.weight", 3 * i + 3), vec![2 * dim, dim, 2 * r], g(dim * 2 * r), &mut next);
        put(&mut m, format!("{p}.{}.conv.conv.bias", 3 * i + 3), vec![2 * dim], 0.01, &mut next);
    }
    put(&mut m, format!("{p}.14.conv.conv.weight"), vec![HIDDEN, 1024, 3], g(1024 * 3), &mut next);
    put(&mut m, format!("{p}.14.conv.conv.bias"), vec![HIDDEN], 0.01, &mut next);

    let t = "encoder_transformer.transformer.layers";
    for i in 0..TR_LAYERS {
        for (n, sh, gg) in [
            (format!("{t}.{i}.norm1.weight"), vec![HIDDEN], 0.0),
            (format!("{t}.{i}.norm1.bias"), vec![HIDDEN], 0.01),
            (format!("{t}.{i}.norm2.weight"), vec![HIDDEN], 0.0),
            (format!("{t}.{i}.norm2.bias"), vec![HIDDEN], 0.01),
            (format!("{t}.{i}.self_attn.in_proj_weight"), vec![3 * HIDDEN, HIDDEN], g(HIDDEN)),
            (format!("{t}.{i}.self_attn.out_proj.weight"), vec![HIDDEN, HIDDEN], g(HIDDEN)),
            (format!("{t}.{i}.linear1.weight"), vec![TR_INTER, HIDDEN], g(HIDDEN)),
            (format!("{t}.{i}.linear2.weight"), vec![HIDDEN, TR_INTER], g(TR_INTER)),
            (format!("{t}.{i}.layer_scale_1.scale"), vec![HIDDEN], 0.02),
            (format!("{t}.{i}.layer_scale_2.scale"), vec![HIDDEN], 0.02),
        ] {
            put(&mut m, n, sh, gg, &mut next);
        }
        // LayerNorm gains sit at 1 + noise, not at noise.
        for n in [format!("{t}.{i}.norm1.weight"), format!("{t}.{i}.norm2.weight")] {
            let e = m.get_mut(&n).unwrap();
            for v in e.0.iter_mut() {
                *v = 1.0 + 0.05 * next();
            }
        }
    }
    put(&mut m, "downsample.conv.conv.conv.weight".into(), vec![HIDDEN, HIDDEN, 4], g(HIDDEN * 4), &mut next);
    for (bank, nq) in [("quantizer.rvq_first", 1usize), ("quantizer.rvq_rest", N_ACOUSTIC)] {
        put(&mut m, format!("{bank}.input_proj.weight"), vec![CODE_DIM, HIDDEN, 1], g(HIDDEN), &mut next);
        for q in 0..nq {
            put(&mut m, format!("{bank}.vq.layers.{q}._codebook.embedding_sum"), vec![CODEBOOK_SIZE, CODE_DIM], 1.0, &mut next);
            let n = format!("{bank}.vq.layers.{q}._codebook.cluster_usage");
            let v: Vec<f32> = (0..CODEBOOK_SIZE).map(|_| 1.0 + 0.2 * next()) .collect();
            m.insert(n, (v, vec![CODEBOOK_SIZE]));
        }
    }
    WeightLoader::Pile(m)
}

/// Deterministic speech-like probe signal: a syllable-rate amplitude envelope
/// over a drifting glottal pulse train plus a breath-noise floor. Not real
/// speech, but broadband and non-stationary, so it exercises every codebook
/// rather than parking on one entry (which a sine would).
fn probe_signal(n: usize) -> Vec<f32> {
    let mut rng = 0x2545_F491_4F6C_DD1Du64;
    let mut next = || {
        rng ^= rng << 13;
        rng ^= rng >> 7;
        rng ^= rng << 17;
        ((rng >> 40) as f32 / 8388608.0) - 1.0
    };
    let sr = SAMPLE_RATE as f32;
    let mut phase = 0f32;
    (0..n)
        .map(|i| {
            let t = i as f32 / sr;
            let f0 = 110.0 + 35.0 * (2.0 * std::f32::consts::PI * 0.7 * t).sin();
            phase += 2.0 * std::f32::consts::PI * f0 / sr;
            let env = 0.35 * (1.0 + (2.0 * std::f32::consts::PI * 3.7 * t).sin()) * 0.5 + 0.05;
            let mut v = 0f32;
            for h in 1..=18 {
                let a = 1.0 / (h as f32).powf(1.4);
                v += a * (phase * h as f32).sin();
            }
            (env * v + 0.02 * next()) * 0.3
        })
        .collect()
}

fn stats(mut v: Vec<f64>) -> (f64, f64, f64) {
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    (v[0], v[v.len() / 2], v[v.len() - 1])
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let pile = args
        .get(1)
        .cloned()
        .unwrap_or_else(|| "/Volumes/pile_backup/models/personaplex.pile".to_string());
    let frames: usize = args
        .get(2)
        .and_then(|s| s.parse().ok())
        .unwrap_or(200usize);

    let t0 = Instant::now();
    let loader = if pile == "--synthetic" {
        println!("synthetic checkpoint (shapes only — see synthetic_weights)");
        synthetic_weights()
    } else {
        println!("loading Mimi encoder from {pile} …");
        mary::persist::personaplex_loader(Path::new(&pile))
            .unwrap_or_else(|e| panic!("pile load: {e}"))
    };
    let enc = MimiEncoder::load(&loader);
    println!("loaded in {:.1}s", t0.elapsed().as_secs_f64());

    let samples = probe_signal(frames * SAMPLES_PER_FRAME);
    println!("probe signal: {frames} frames ({} samples)", samples.len());

    // ── CPU streaming lane ──
    let mut state = enc.stream_state();
    let mut cpu_codes = Vec::with_capacity(frames);
    let mut cpu_ms = Vec::with_capacity(frames);
    for chunk in samples.chunks_exact(SAMPLES_PER_FRAME) {
        let t = Instant::now();
        let c = enc.encode_stream_frame(&mut state, chunk.try_into().unwrap());
        cpu_ms.push(t.elapsed().as_secs_f64() * 1e3);
        cpu_codes.push(c);
    }
    // Discard the cold prefix AND the ring-fill ramp: the 250-position sliding
    // window is only full after 125 frames, so earlier frames understate the
    // steady-state attention cost.
    let warm = frames.min(125);
    let (lo, med, hi) = stats(cpu_ms[warm..].to_vec());
    println!(
        "CPU streaming encode: p50 {med:.2} ms/frame  (min {lo:.2}, max {hi:.2}, n={})",
        cpu_ms.len() - warm
    );
    let (alo, amed, ahi) = stats(cpu_ms.clone());
    println!("  all frames incl. ramp: p50 {amed:.2} (min {alo:.2}, max {ahi:.2}, n={frames})");
    println!("  first frame (cold): {:.2} ms", cpu_ms[0]);
    let mut hist = [0usize; NUM_CODEBOOKS];
    for f in &cpu_codes {
        for (q, &c) in f.iter().enumerate() {
            hist[q] = hist[q].max(c as usize);
        }
    }
    println!("  max code per quantizer: {hist:?}");

    // The batch path is integer-exact against the streaming path on the CPU
    // (`mimi_probe`), so its 12.5 Hz pre-quantizer latent is a valid reference
    // for the per-frame streaming latent the GPU emits.
    let (_, _, cpu_ds, batch_codes) = enc.encode_stages(&samples);
    assert_eq!(batch_codes, cpu_codes, "CPU batch vs streaming disagreed");

    // ── GPU streaming lane ──
    println!("\nloading GPU encoder …");
    let t0 = Instant::now();
    let mut gpu = MimiEncoderGpu::load(&loader);
    println!("loaded in {:.1}s", t0.elapsed().as_secs_f64());

    let mut gpu_codes = Vec::with_capacity(frames);
    let mut gpu_ms = Vec::with_capacity(frames);
    let mut gpu_submit_ms = Vec::with_capacity(frames);
    let mut lat_num = 0f64;
    let mut lat_den = 0f64;
    let mut lat_dot = 0f64;
    let mut lat_na = 0f64;
    let mut lat_nb = 0f64;
    for (fi, chunk) in samples.chunks_exact(SAMPLES_PER_FRAME).enumerate() {
        let t = Instant::now();
        gpu.submit_frame(chunk.try_into().unwrap());
        let sub = t.elapsed().as_secs_f64() * 1e3;
        let c = gpu.read_codes();
        gpu_ms.push(t.elapsed().as_secs_f64() * 1e3);
        gpu_submit_ms.push(sub);
        gpu_codes.push(c);
        let lat = gpu.read_latent();
        for (ch, &g) in lat.iter().enumerate() {
            let r = cpu_ds[ch * frames + fi] as f64;
            let g = g as f64;
            lat_num += (g - r) * (g - r);
            lat_den += r * r;
            lat_dot += g * r;
            lat_na += g * g;
            lat_nb += r * r;
        }
    }
    let (glo, gmed, ghi) = stats(gpu_ms[warm..].to_vec());
    println!(
        "GPU streaming encode: p50 {gmed:.2} ms/frame  (min {glo:.2}, max {ghi:.2}, n={})",
        gpu_ms.len() - warm
    );
    let (aglo, agmed, aghi) = stats(gpu_ms.clone());
    println!("  all frames incl. warmup: p50 {agmed:.2} (min {aglo:.2}, max {aghi:.2}, n={frames})");
    println!("  first frame (shader compile): {:.2} ms", gpu_ms[0]);
    let (slo, smed, shi) = stats(gpu_submit_ms[warm..].to_vec());
    println!("  of which host submit: p50 {smed:.2} ms (min {slo:.2}, max {shi:.2}); drain {:.2} ms", gmed - smed);
    println!("  speedup vs CPU (p50, steady state): {:.2}x", med / gmed);

    // ── parity, judged statistically ──
    println!("\nparity (no bit-exactness gate — see encoder_gpu module docs):");
    println!(
        "  latent (pre-quantizer 12.5 Hz, {} frames x {HIDDEN}): rel RMS {:.3e}  cos {:.8}",
        frames,
        (lat_num / lat_den).sqrt(),
        lat_dot / (lat_na.sqrt() * lat_nb.sqrt())
    );
    let mut agree = [0usize; NUM_CODEBOOKS];
    for (g, c) in gpu_codes.iter().zip(&cpu_codes) {
        for q in 0..NUM_CODEBOOKS {
            if g[q] == c[q] {
                agree[q] += 1;
            }
        }
    }
    let total: usize = agree.iter().sum();
    println!(
        "  code agreement: {}/{} ({:.2}%)",
        total,
        frames * NUM_CODEBOOKS,
        100.0 * total as f64 / (frames * NUM_CODEBOOKS) as f64
    );
    for q in 0..NUM_CODEBOOKS {
        println!(
            "    q{q}: {}/{frames} ({:.1}%)",
            agree[q],
            100.0 * agree[q] as f64 / frames as f64
        );
    }
    for fi in 0..frames.min(3) {
        println!("    frame {fi}: gpu {:?}", gpu_codes[fi]);
        println!("    frame {fi}: cpu {:?}", cpu_codes[fi]);
    }

    // Throughput vs. per-frame synchronisation: submit a run of frames and
    // drain once. The codes are meaningless (only the last frame's survive in
    // the buffer) — this measures how much of the per-frame wall clock is GPU
    // work and how much is the round trip.
    gpu.reset();
    let mut burst = Vec::new();
    for rep in 0..6 {
        let t = Instant::now();
        for chunk in samples.chunks_exact(SAMPLES_PER_FRAME).skip(rep * 8).take(8) {
            gpu.submit_frame(chunk.try_into().unwrap());
        }
        let _ = gpu.read_codes();
        burst.push(t.elapsed().as_secs_f64() * 1e3 / 8.0);
    }
    let (blo, bmed, bhi) = stats(burst);
    println!(
        "  pipelined (8 frames per drain): p50 {bmed:.2} ms/frame (min {blo:.2}, max {bhi:.2}, n=6)"
    );

    // Determinism of the GPU lane against itself across a reset.
    gpu.reset();
    let replay: Vec<_> = samples
        .chunks_exact(SAMPLES_PER_FRAME)
        .take(20)
        .map(|c| gpu.encode_frame(c.try_into().unwrap()))
        .collect();
    let same = replay.iter().zip(&gpu_codes).filter(|(a, b)| a == b).count();
    println!("  reset replay: {same}/20 frames identical");
}
