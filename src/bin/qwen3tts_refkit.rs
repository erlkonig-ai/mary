//! `qwen3tts_refkit <ref_24k.wav> <out_code.npy>` — the third file of a
//! reference kit: the clip's codec frames, `(T, 16)` f32 npy, exactly what the
//! voice faculty reads beside the 24 kHz mono clip and its transcript.
//!
//! The kit the shipping voice clones was made on the Mac; this makes one
//! anywhere the qwen3tts pile is (`QWEN3TTS_PILE`, else `$MARY_MODELS/
//! qwen3tts.pile`). The codec ENCODER is the CPU-f32 port (it runs once, on a
//! ten-second clip, to make a file); nothing here is on the serving path.
use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use mary::models::f5::wav;
use mary::models::qwen3tts::config::NUM_CODE_GROUPS;
use mary::models::qwen3tts::encoder::CodecEncoder;

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    anyhow::ensure!(args.len() == 2, "usage: qwen3tts_refkit <ref_24k.wav> <out_code.npy>");
    let ref_wav = PathBuf::from(&args[0]);
    let out = PathBuf::from(&args[1]);

    let (samples, sr) = wav::read_pcm16_mono(&ref_wav);
    anyhow::ensure!(
        sr == 24_000,
        "reference clip must be 24 kHz mono PCM16 (got {sr} Hz): {}",
        ref_wav.display()
    );
    let pile = mary::paths::model(std::env::var("QWEN3TTS_PILE").ok().as_deref(), "qwen3tts.pile")?;
    let loader = mary::persist::load_aliased_loader_from_pile(&pile, "talker_f16")
        .with_context(|| format!("load {}", pile.display()))?;
    let encoder = CodecEncoder::load(&loader);
    let codes = encoder.encode(&samples);
    let flat: Vec<f32> = codes.iter().flat_map(|f| f.iter().map(|&c| c as f32)).collect();
    mary::nn::npy::save_npy(&out, &flat, &[codes.len(), NUM_CODE_GROUPS])
        .with_context(|| format!("write {}", out.display()))?;
    println!(
        "{}: {} frames ({:.2}s at 12.5 Hz) -> {}",
        ref_wav.display(),
        codes.len(),
        codes.len() as f32 / 12.5,
        Path::new(&out).display()
    );
    Ok(())
}
