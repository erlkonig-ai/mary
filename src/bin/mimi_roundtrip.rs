//! Mimi codec round-trip: 24 kHz mono WAV in → `MimiEncoder::encode` →
//! `MimiDecoder::decode` → 24 kHz mono WAV out.
//!
//! Isolates the CODEC from the LM: whatever survives here is the ceiling on
//! what any Mimi-token pipeline (PersonaPlex, Moshi) can reproduce. Built to
//! answer "does Mimi preserve singing, or only speech" — run it on a sung
//! excerpt and on a speech control, then measure the two outputs offline.
//!
//!   cargo run --release --features qwen3tts,import --bin mimi_roundtrip \
//!       -- <in.wav> <out.wav> [weights.pile] [--codes <out.csv>]

use mary::models::f5::wav;
use mary::models::personaplex::mimi::config::*;
use mary::models::personaplex::mimi::{MimiDecoder, MimiEncoder};
use mary::nn::weight_loader::WeightLoader;
use std::io::Write;
use std::path::Path;
use std::time::Instant;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 3 {
        eprintln!(
            "usage: mimi_roundtrip <in.wav> <out.wav> [weights.pile] [--codes <codes.csv>]\n\
             \n\
             Reads a 24 kHz mono PCM16 WAV, Mimi-encodes it to discrete codes,\n\
             Mimi-decodes those codes back to 24 kHz PCM, and writes the result.\n\
             Weights: [weights.pile], else $MARY_MODELS/personaplex_typed.pile"
        );
        std::process::exit(2);
    }
    let in_path = args[1].clone();
    let out_path = args[2].clone();
    let mut pile: Option<String> = None;
    let mut codes_csv: Option<String> = None;
    let mut i = 3;
    while i < args.len() {
        match args[i].as_str() {
            "--codes" => {
                codes_csv = Some(args.get(i + 1).expect("--codes needs a path").clone());
                i += 2;
            }
            other => {
                pile = Some(other.to_string());
                i += 1;
            }
        }
    }
    let pile = mary::paths::model(pile.as_deref(), "personaplex_typed.pile")
        .unwrap_or_else(|e| {
            eprintln!("{e}");
            std::process::exit(2)
        })
        .to_string_lossy()
        .into_owned();

    // ---- weights ----
    println!("[weights] {pile}");
    let t = Instant::now();
    let loader = WeightLoader::from_pile(Path::new(&pile))
        .unwrap_or_else(|e| panic!("open weight pile {pile}: {e}"));
    // fail loudly and early if this pile simply doesn't carry the codec
    for k in [
        "encoder.model.0.conv.conv.weight",
        "downsample.conv.conv.conv.weight",
        "quantizer.rvq_first.vq.layers.0._codebook.embedding_sum",
        "decoder.model.0.conv.conv.weight",
    ] {
        if !loader.has_weight(k) {
            eprintln!("[warn] weight pile has no `{k}` — load will likely panic");
        }
    }
    println!(
        "[weights] index opened in {:.1}s",
        t.elapsed().as_secs_f32()
    );

    // ---- input ----
    let (samples, sr) = wav::read_pcm16_mono(Path::new(&in_path));
    assert_eq!(sr, SAMPLE_RATE, "input must be {SAMPLE_RATE} Hz");
    let secs = samples.len() as f32 / sr as f32;
    println!(
        "[in ] {in_path}: {} samples, {sr} Hz, {secs:.2}s",
        samples.len()
    );

    // ---- encode ----
    let t = Instant::now();
    let enc = MimiEncoder::load(&loader);
    println!("[enc ] loaded in {:.1}s", t.elapsed().as_secs_f32());
    let t = Instant::now();
    let codes = enc.encode(&samples);
    let enc_s = t.elapsed().as_secs_f32();
    let frames = codes.len();
    println!(
        "[enc ] {frames} frames × {NUM_CODEBOOKS} codebooks in {enc_s:.1}s ({:.2}× realtime)",
        secs / enc_s
    );

    // ---- observed codec configuration ----
    let frame_rate = SAMPLE_RATE as f64 / SAMPLES_PER_FRAME as f64;
    let bits_per_code = (CODEBOOK_SIZE as f64).log2();
    let toks_per_s = frame_rate * NUM_CODEBOOKS as f64;
    println!("[cfg ] codebooks/frame      = {NUM_CODEBOOKS}");
    println!("[cfg ] codebook size        = {CODEBOOK_SIZE} ({bits_per_code:.1} bits/code)");
    println!("[cfg ] samples/frame        = {SAMPLES_PER_FRAME} @ {SAMPLE_RATE} Hz");
    println!("[cfg ] frame rate           = {frame_rate:.4} Hz");
    println!("[cfg ] tokens/second        = {toks_per_s:.1}");
    println!(
        "[cfg ] bitrate              = {:.1} bits/s",
        toks_per_s * bits_per_code
    );
    println!(
        "[cfg ] observed frames/sec  = {:.4} ({frames} frames / {secs:.2}s)",
        frames as f64 / secs as f64
    );

    // per-codebook code-usage spread (how much of each 2048-entry book the
    // signal actually exercises) — cheap, and it is a measured fact.
    for q in 0..NUM_CODEBOOKS {
        let mut seen = vec![false; CODEBOOK_SIZE];
        let mut hist = std::collections::HashMap::<u32, u32>::new();
        for f in &codes {
            seen[f[q] as usize] = true;
            *hist.entry(f[q]).or_insert(0) += 1;
        }
        let distinct = seen.iter().filter(|&&b| b).count();
        let n = frames as f64;
        let h: f64 = hist
            .values()
            .map(|&c| {
                let p = c as f64 / n;
                -p * p.log2()
            })
            .sum();
        println!("[book] q{q}: {distinct} distinct codes, entropy {h:.2} bits");
    }

    if let Some(path) = &codes_csv {
        let mut f = std::fs::File::create(path).expect("create codes csv");
        for fr in &codes {
            let row: Vec<String> = fr.iter().map(|c| c.to_string()).collect();
            writeln!(f, "{}", row.join(",")).unwrap();
        }
        println!("[codes] wrote {path}");
    }

    // ---- decode ----
    let t = Instant::now();
    let dec = MimiDecoder::load(&loader);
    println!("[dec ] loaded in {:.1}s", t.elapsed().as_secs_f32());
    let t = Instant::now();
    let out = dec.decode(&codes);
    let dec_s = t.elapsed().as_secs_f32();
    println!(
        "[dec ] {} samples in {dec_s:.1}s ({:.2}× realtime)",
        out.len(),
        (out.len() as f32 / sr as f32) / dec_s
    );

    wav::write_pcm16_mono(Path::new(&out_path), &out, SAMPLE_RATE);
    println!(
        "[out ] {out_path}: {} samples ({:.2}s)",
        out.len(),
        out.len() as f32 / sr as f32
    );
}
