//! `mary::say` — self-contained F5-TTS in Burn: speak `gen_text` in the voice of
//! a reference clip, no Python. Loads weights from a durable standalone pile
//! (both F5 and Vocos live in it — written once by the `f5_persist` importer),
//! extracts the ref mel, tokenizes, runs CFM → Vocos, returns 24 kHz mono audio.
//!
//! This is the LIBRARY seam: callers use [`synthesize`] in-process, so there is
//! no separately-built production artifact that can drift stale against the
//! pile's blob format. (A stale standalone `say` binary once wrote through an
//! outdated pile format; the fix for that whole class is "no separate
//! production binary in the path".) No safetensors sit in this path either —
//! the weights come entirely from the pile.
//!
//! Long text is split into sentence-sized passes (F5 drifts — degenerates to
//! Mandarin — on single passes past ~30 s / ~1000 chars); each pass clones the
//! same reference so the voice stays consistent, and the audio is concatenated.

use crate::models::f5::cfm;
use crate::models::f5::config::{CfmConfig, F5Config};
use crate::models::f5::mel::MelExtractor;
use crate::models::f5::model::F5Transformer;
use crate::models::f5::tokenizer::Tokenizer;
use crate::models::f5::vocos::Vocos;
use crate::models::f5::wav;
use crate::nn::backend::{WgpuDevice, B};
use crate::nn::weight_loader::WeightLoader;
use burn::prelude::*;
use burn::tensor::Distribution;
use std::path::Path;

/// Output sample rate of the synthesizer (Vocos is a 24 kHz vocoder).
pub const SAMPLE_RATE: u32 = 24000;

/// Keep each F5 pass well under the drift threshold (~1000 chars / ~30 s).
const MAX_CHARS: usize = 300;

/// Split text into sentence-aware passes, each ≤ `max` chars. Sentences are
/// packed greedily; an over-long sentence is split on word boundaries.
/// Shared with `mary::speak` (Qwen3-TTS), which chunks for the same reason.
pub(crate) fn chunk_text(s: &str, max: usize) -> Vec<String> {
    let mut sentences: Vec<String> = Vec::new();
    let mut cur = String::new();
    for c in s.chars() {
        cur.push(c);
        if matches!(c, '.' | '!' | '?') {
            let t = cur.trim();
            if !t.is_empty() {
                sentences.push(t.to_string());
            }
            cur.clear();
        }
    }
    if !cur.trim().is_empty() {
        sentences.push(cur.trim().to_string());
    }
    if sentences.is_empty() {
        return vec![s.trim().to_string()];
    }

    let mut chunks: Vec<String> = Vec::new();
    let mut buf = String::new();
    let push = |buf: &mut String, chunks: &mut Vec<String>| {
        if !buf.is_empty() {
            chunks.push(std::mem::take(buf));
        }
    };
    for sent in sentences {
        if sent.len() > max {
            push(&mut buf, &mut chunks);
            let mut wbuf = String::new();
            for word in sent.split_whitespace() {
                if !wbuf.is_empty() && wbuf.len() + 1 + word.len() > max {
                    chunks.push(std::mem::take(&mut wbuf));
                }
                if !wbuf.is_empty() {
                    wbuf.push(' ');
                }
                wbuf.push_str(word);
            }
            push(&mut wbuf, &mut chunks);
        } else {
            if !buf.is_empty() && buf.len() + 1 + sent.len() > max {
                push(&mut buf, &mut chunks);
            }
            if !buf.is_empty() {
                buf.push(' ');
            }
            buf.push_str(&sent);
        }
    }
    push(&mut buf, &mut chunks);
    chunks
}

/// Synthesise one pass (ref ⊕ chunk), returning just the generated audio.
fn synth_chunk(
    model: &F5Transformer<B>,
    vocos: &Vocos<B>,
    ref_mel: &Tensor<B, 3>,
    ref_len: usize,
    ref_text: &str,
    chunk: &str,
    device: &WgpuDevice,
) -> Vec<f32> {
    let text = Tokenizer::new().encode_tensor::<B>(&format!("{ref_text} {chunk}"), device);
    let (rb, gb) = (ref_text.len() as f64, chunk.len() as f64);
    let local_speed = if chunk.len() < 10 { 0.3 } else { 1.0 };
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
    // nfe / cfg overridable via env for empirical latency/quality sweeps.
    let nfe = std::env::var("MARY_NFE")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(32);
    let cfg_strength = std::env::var("MARY_CFG")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(2.0);
    let cfg = CfmConfig {
        nfe,
        sway_coef: -1.0,
        cfg_strength,
    };
    let sampled = cfm::integrate(model, y0, cond, text, &cfg, device);
    let gen_mel = sampled
        .slice([0..1, ref_len..duration, 0..100])
        .swap_dims(1, 2);
    vocos.forward(gen_mel).into_data().to_vec().unwrap()
}

/// Speak `gen_text` in the voice of `ref_wav` (24 kHz mono PCM16, transcript
/// `ref_text`), using the F5-TTS + Vocos weights persisted in the pile at
/// `weights_pile` (written by the `f5_persist` importer; the tensor-name
/// namespaces — `ema_model.*` vs `backbone.*`/`head.*` — are disjoint, so one
/// union keymap serves both models). Returns peak-normalized 24 kHz mono audio
/// in `[-1, 1]`.
///
/// In-process, no subprocess and no safetensors — this is the shared library
/// seam. Loads weights from the pile, splits the text into drift-safe passes,
/// runs CFM → Vocos per pass, and concatenates with a short gap between passes.
pub fn synthesize(weights_pile: &Path, ref_wav: &Path, ref_text: &str, gen_text: &str) -> Vec<f32> {
    let device: WgpuDevice = Default::default();

    // reference mel (the voice to clone — shared across passes)
    let (samples, sr) = wav::read_pcm16_mono(ref_wav);
    assert_eq!(sr, SAMPLE_RATE, "reference clip must be 24 kHz mono");
    let n = samples.len();
    let wavt = Tensor::<B, 1>::from_floats(samples.as_slice(), &device).reshape([1, n]);
    let ref_mel = MelExtractor::<B>::new(&device)
        .forward(wavt)
        .swap_dims(1, 2);
    let ref_len = ref_mel.dims()[1];

    let t_load = std::time::Instant::now();
    let loader = WeightLoader::from_pile(weights_pile)
        .unwrap_or_else(|e| panic!("load voice pile {weights_pile:?}: {e:?}"));
    let model = F5Transformer::<B>::load(&loader, F5Config::v1_base(), &device);
    let vocos = Vocos::<B>::load(&loader, &device);
    eprintln!(
        "[timing] weight load (pile): {:.2}s",
        t_load.elapsed().as_secs_f32()
    );

    let chunks = chunk_text(gen_text, MAX_CHARS);
    eprintln!("{} pass(es), ref_len {ref_len}", chunks.len());
    let t_synth = std::time::Instant::now();
    let mut audio: Vec<f32> = Vec::new();
    for (i, chunk) in chunks.iter().enumerate() {
        eprintln!("  pass {}/{}: {} chars …", i + 1, chunks.len(), chunk.len());
        let w = synth_chunk(&model, &vocos, &ref_mel, ref_len, ref_text, chunk, &device);
        audio.extend_from_slice(&w);
        if i + 1 < chunks.len() {
            audio.extend(std::iter::repeat(0.0).take(SAMPLE_RATE as usize / 6));
            // ~167 ms gap
        }
    }

    let synth_s = t_synth.elapsed().as_secs_f32();
    let audio_s = audio.len() as f32 / SAMPLE_RATE as f32;
    eprintln!(
        "[timing] synth: {:.2}s for {:.2}s audio ({:.2}x realtime)",
        synth_s,
        audio_s,
        synth_s / audio_s.max(1e-6)
    );

    let peak = audio.iter().fold(0f32, |m, &x| m.max(x.abs())).max(1e-6);
    audio.iter_mut().for_each(|x| *x = *x / peak * 0.95);
    audio
}

/// Convenience: [`synthesize`] then write the result to `out_path` as a 24 kHz
/// mono PCM16 WAV. Returns the number of samples written.
pub fn synthesize_to_wav(
    weights_pile: &Path,
    ref_wav: &Path,
    ref_text: &str,
    gen_text: &str,
    out_path: &Path,
) -> usize {
    let audio = synthesize(weights_pile, ref_wav, ref_text, gen_text);
    wav::write_pcm16_mono(out_path, &audio, SAMPLE_RATE);
    audio.len()
}
