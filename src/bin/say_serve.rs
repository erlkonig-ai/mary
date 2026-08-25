//! Warm-weight voice server — the T2 prototype. Loads F5 + Vocos + the mel
//! extractor + the reference voice ONCE, then serves synthesis requests from
//! stdin in a loop, so the ~10s fixed cost (shader JIT + setup) is paid once
//! and amortized across every utterance. Transport-agnostic on purpose: stdin
//! here is a stand-in for a message seam (an intent is a message) or a local
//! socket a calling service uses instead of spawning a 50s `say` process.
//!
//!   say_serve <f5.pile> <ref.wav> <ref_text>
//!   then on stdin, one request per line:  <out.wav>\t<text to speak>
//!
//! Weights (F5 + Vocos) come entirely from the pile (see `f5_persist`).
//!
//! Prints per-request timing so the warm-vs-cold amortization is measurable.

use burn::prelude::*;
use burn::tensor::Distribution;
use mary::models::f5::cfm;
use mary::models::f5::config::{CfmConfig, F5Config};
use mary::models::f5::mel::MelExtractor;
use mary::models::f5::model::F5Transformer;
use mary::models::f5::tokenizer::Tokenizer;
use mary::models::f5::vocos::Vocos;
use mary::models::f5::wav;
use mary::nn::backend::{B, WgpuDevice};
use mary::nn::weight_loader::WeightLoader;
use std::io::{BufRead, Write};
use std::path::Path;

fn synth(
    model: &F5Transformer<B>,
    vocos: &Vocos<B>,
    ref_mel: &Tensor<B, 3>,
    ref_len: usize,
    ref_text: &str,
    text: &str,
    device: &WgpuDevice,
) -> Vec<f32> {
    let nfe = std::env::var("MARY_NFE")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(32);
    let cfg_strength = std::env::var("MARY_CFG")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(2.0);
    let toks = Tokenizer::new().encode_tensor::<B>(&format!("{ref_text} {text}"), device);
    let (rb, gb) = (ref_text.len() as f64, text.len() as f64);
    let local_speed = if text.len() < 10 { 0.3 } else { 1.0 };
    let duration = ref_len + (ref_len as f64 / rb * gb / local_speed) as usize;
    let r#gen = duration - ref_len;
    let cond = Tensor::cat(
        vec![
            ref_mel.clone(),
            Tensor::<B, 3>::zeros([1, r#gen, 100], device),
        ],
        1,
    );
    let y0 = Tensor::random([1, duration, 100], Distribution::Normal(0.0, 1.0), device);
    let cfg = CfmConfig {
        nfe,
        sway_coef: -1.0,
        cfg_strength,
    };
    let sampled = cfm::integrate(model, y0, cond, toks, &cfg, device);
    let gen_mel = sampled
        .slice([0..1, ref_len..duration, 0..100])
        .swap_dims(1, 2);
    vocos.forward(gen_mel).into_data().to_vec().unwrap()
}

fn main() {
    let a: Vec<String> = std::env::args().collect();
    if a.len() < 4 {
        eprintln!("usage: say_serve <f5.pile> <ref.wav> <ref_text>");
        std::process::exit(1);
    }
    let (pile_path, ref_wav, ref_text) = (&a[1], &a[2], &a[3]);
    let device: WgpuDevice = Default::default();

    let t_load = std::time::Instant::now();
    let (samples, sr) = wav::read_pcm16_mono(Path::new(ref_wav));
    assert_eq!(sr, 24000, "reference clip must be 24 kHz mono");
    let n = samples.len();
    let wavt = Tensor::<B, 1>::from_floats(samples.as_slice(), &device).reshape([1, n]);
    let ref_mel = MelExtractor::<B>::new(&device)
        .forward(wavt)
        .swap_dims(1, 2);
    let ref_len = ref_mel.dims()[1];
    let loader = WeightLoader::from_pile(Path::new(pile_path))
        .unwrap_or_else(|e| panic!("load voice pile {pile_path}: {e:?}"));
    let model = F5Transformer::<B>::load(&loader, F5Config::v1_base(), &device);
    let vocos = Vocos::<B>::load(&loader, &device);
    eprintln!(
        "[load] model + ref warm in {:.2}s — ready, ref_len {ref_len}",
        t_load.elapsed().as_secs_f32()
    );

    // Warm the kernels once with a throwaway synth so request #1 is also fast
    // (this is where the shader JIT is paid). Comment-measured below instead:
    let stdin = std::io::stdin();
    let mut req = 0usize;
    for line in stdin.lock().lines() {
        let line = match line {
            Ok(l) => l,
            Err(_) => break,
        };
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        let (out_path, text) = match line.split_once('\t') {
            Some((o, t)) => (o, t),
            None => {
                eprintln!("[skip] expected <out.wav>\\t<text>");
                continue;
            }
        };
        req += 1;
        let t = std::time::Instant::now();
        let mut audio = synth(&model, &vocos, &ref_mel, ref_len, ref_text, text, &device);
        let peak = audio.iter().fold(0f32, |m, &x| m.max(x.abs())).max(1e-6);
        audio.iter_mut().for_each(|x| *x = *x / peak * 0.95);
        wav::write_pcm16_mono(Path::new(out_path), &audio, 24000);
        let synth_s = t.elapsed().as_secs_f32();
        let audio_s = audio.len() as f32 / 24000.0;
        eprintln!(
            "[req {req}] {:.2}s synth for {:.2}s audio ({:.2}x realtime) -> {out_path}",
            synth_s,
            audio_s,
            synth_s / audio_s
        );
        std::io::stderr().flush().ok();
    }
    eprintln!("[done] served {req} request(s)");
}
