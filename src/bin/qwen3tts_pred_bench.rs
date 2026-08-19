//! Bench the **real** `CodePredictor` — the CPU half of the Qwen3-TTS voice
//! lane — without a pile, and A/B the `down` gemv's two arms in one process.
//!
//!   cargo run --release --features qwen3tts --bin qwen3tts_pred_bench -- [rounds]
//!
//! Why synthetic weights: the predictor is ~5 GB of strictly sequential weight
//! traffic per frame through Accelerate, and Accelerate's sgemv timing is
//! data-independent. What a bench of it has to get right is the shapes, the
//! traffic and the dispatch pattern — all of which are fixed by
//! `CodePredictor::load`, not by the values. So this fills a
//! `WeightLoader::Pile` map with random f32 at exactly the checkpoint's names
//! and shapes and runs the shipped `predict_frame`. That also makes the bench
//! runnable on a machine that has no Qwen3-TTS pile mounted, which is how it
//! came to exist (2026-08-19: the canonical `qwen3tts.pile` was on an
//! unmounted volume and only the 173-tensor f16 talker fold was local).
//!
//! It is NOT an output check: random weights say nothing about how the voice
//! sounds. Ear gates ride `speak_check` on a real pile.
//!
//! Arms are interleaved round by round (`MARY_PRED_DOWN_SERIAL`'s two states,
//! flipped in process), so a change in ambient load hits both equally instead
//! of only whichever ran last — this machine's background daemons swing a
//! sequential A/B by more than the effect being measured. `MARY_PRED_THREADS`
//! sets the pool width as usual.

use mary::models::qwen3tts::config::*;
use mary::models::qwen3tts::predictor::{set_down_serial, CodePredictor};
use mary::nn::weight_loader::WeightLoader;
use std::collections::HashMap;
use std::time::Instant;

const INTER: usize = 3072;
const TALKER_W: usize = 2048;
const Q_DIM: usize = PRED_HEADS * PRED_HEAD_DIM;
const KV_DIM: usize = PRED_KV_HEADS * PRED_HEAD_DIM;

/// xorshift; ~N(0, 0.02), the scale real weights sit at — no denormals to
/// stall on and no overflow in the long reductions.
struct Rng(u64);
impl Rng {
    fn vec(&mut self, n: usize) -> Vec<f32> {
        (0..n)
            .map(|_| {
                self.0 ^= self.0 << 13;
                self.0 ^= self.0 >> 7;
                self.0 ^= self.0 << 17;
                ((self.0 >> 40) as f32 / 8388608.0 - 0.5) * 0.04
            })
            .collect()
    }
}

fn pct(v: &mut [f64], p: f64) -> f64 {
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    v[((v.len() - 1) as f64 * p).round() as usize]
}

fn main() {
    mary::models::qwen3tts::cpu::set_interactive_qos();
    let rounds: usize = std::env::args()
        .nth(1)
        .and_then(|s| s.parse().ok())
        .unwrap_or(60);

    let mut rng = Rng(0x2026_0819_0000_0001);
    let mut m: HashMap<String, (Vec<f32>, Vec<usize>)> = HashMap::new();
    let mut put = |m: &mut HashMap<_, _>, rng: &mut Rng, name: String, shape: Vec<usize>| {
        let n = shape.iter().product();
        m.insert(name, (rng.vec(n), shape));
    };
    let p = "talker.code_predictor";
    for i in 0..PRED_LAYERS {
        let lp = format!("{p}.model.layers.{i}");
        let h = PRED_HIDDEN;
        put(&mut m, &mut rng, format!("{lp}.self_attn.q_proj.weight"), vec![Q_DIM, h]);
        put(&mut m, &mut rng, format!("{lp}.self_attn.k_proj.weight"), vec![KV_DIM, h]);
        put(&mut m, &mut rng, format!("{lp}.self_attn.v_proj.weight"), vec![KV_DIM, h]);
        put(&mut m, &mut rng, format!("{lp}.self_attn.o_proj.weight"), vec![h, Q_DIM]);
        put(&mut m, &mut rng, format!("{lp}.mlp.gate_proj.weight"), vec![INTER, h]);
        put(&mut m, &mut rng, format!("{lp}.mlp.up_proj.weight"), vec![INTER, h]);
        put(&mut m, &mut rng, format!("{lp}.mlp.down_proj.weight"), vec![h, INTER]);
        put(&mut m, &mut rng, format!("{lp}.input_layernorm.weight"), vec![h]);
        put(&mut m, &mut rng, format!("{lp}.post_attention_layernorm.weight"), vec![h]);
        put(&mut m, &mut rng, format!("{lp}.self_attn.q_norm.weight"), vec![PRED_HEAD_DIM]);
        put(&mut m, &mut rng, format!("{lp}.self_attn.k_norm.weight"), vec![PRED_HEAD_DIM]);
    }
    for i in 0..NUM_CODE_GROUPS - 1 {
        put(
            &mut m,
            &mut rng,
            format!("{p}.model.codec_embedding.{i}.weight"),
            vec![PRED_VOCAB, TALKER_W],
        );
        put(
            &mut m,
            &mut rng,
            format!("{p}.lm_head.{i}.weight"),
            vec![PRED_VOCAB, PRED_HIDDEN],
        );
    }
    // the 1.7B talker's 2048 -> 1024 projection (the 0.6B has none)
    put(
        &mut m,
        &mut rng,
        format!("{p}.small_to_mtp_projection.weight"),
        vec![PRED_HIDDEN, TALKER_W],
    );
    put(&mut m, &mut rng, format!("{p}.small_to_mtp_projection.bias"), vec![PRED_HIDDEN]);
    put(&mut m, &mut rng, format!("{p}.model.norm.weight"), vec![PRED_HIDDEN]);

    let bytes: usize = m.values().map(|(v, _)| v.len() * 4).sum();
    println!(
        "qwen3tts_pred_bench: {} tensors / {} MiB synthetic, MARY_PRED_THREADS={}",
        m.len(),
        bytes / (1 << 20),
        std::env::var("MARY_PRED_THREADS").unwrap_or_else(|_| "2 (default)".into())
    );
    let predictor = CodePredictor::load(&WeightLoader::Pile(m));

    let talker_hidden = rng.vec(TALKER_W);
    let code0 = rng.vec(TALKER_W);
    let mut r = <rand::rngs::StdRng as rand::SeedableRng>::seed_from_u64(7);
    let mut run = |serial: bool, rng: &mut rand::rngs::StdRng| -> f64 {
        set_down_serial(serial);
        let t = Instant::now();
        let _ = predictor.predict_frame(&talker_hidden, &code0, true, 0.9, rng);
        t.elapsed().as_secs_f64() * 1e3
    };

    for _ in 0..3 {
        run(true, &mut r);
        run(false, &mut r);
    }
    let (mut ser, mut pooled) = (Vec::new(), Vec::new());
    for _ in 0..rounds {
        ser.push(run(true, &mut r));
        pooled.push(run(false, &mut r));
    }
    let _ = predictor.take_bench(); // drain whatever the interleaved rounds put there

    println!("\n{rounds} interleaved rounds, ms per predictor frame:");
    for (name, v) in [("down=serial", &mut ser), ("down=pooled", &mut pooled)] {
        println!(
            "  {name:<12} p10 {:6.2} | p50 {:6.2} | p90 {:6.2} | min {:6.2}",
            pct(v, 0.10),
            pct(v, 0.50),
            pct(v, 0.90),
            pct(v, 0.0),
        );
    }
    // The decomposition is descriptive, not a comparison, so it runs as two
    // separate blocks — `take_bench` accumulates into thread-locals and cannot
    // be attributed round by round.
    if std::env::var("QWEN3TTS_BENCH").is_ok() {
        println!();
        for (name, serial) in [("down=serial", true), ("down=pooled", false)] {
            for _ in 0..20 {
                run(serial, &mut r);
            }
            if let Some(line) = predictor.take_bench() {
                println!("  {name}: {line}");
            }
        }
    }

    let (a, b) = (pct(&mut ser, 0.5), pct(&mut pooled, 0.5));
    println!(
        "\n  pooled vs serial: {:+.2}% per predictor frame ({a:.2} -> {b:.2} ms)\n  \
         a frame is 80 ms of audio, so the predictor alone runs at {:.2}x audio-rate \
         serial and {:.2}x pooled",
        (b - a) / a * 100.0,
        80.0 / a,
        80.0 / b,
    );
}
