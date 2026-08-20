//! Mimi codec parity gates — CPU Rust port vs the moshi CPU-f32 oracle goldens
//! captured by `/tmp/mimi_work/capture_mimi.py`. Mirrors the Flux/qwen3tts
//! parity discipline: per-component .npy goldens, cos gate on continuous
//! stages, INTEGER-EXACT gate on the codes.
//!
//!   1. encoder SEANet        cos vs `enc_seanet`
//!   2. encoder transformer   cos vs `enc_transformer`
//!   3. encoder downsample    cos vs `enc_downsample`
//!   4. encode → codes        INTEGER-EXACT vs `codes`
//!   5. quantizer decode      cos vs `dec_quantized`
//!   6. decode → waveform     cos vs `decode_wav`
//!
//!   cargo run --release --features qwen3tts,import --bin mimi_probe

use mary::models::f5::wav;
use mary::models::personaplex::mimi::config::*;
use mary::models::personaplex::mimi::{MimiDecoder, MimiEncoder};
use mary::nn::npy;
use mary::nn::weight_loader::{SingleFileLoader, WeightLoader};
use std::path::Path;

const CKPT: &str = "/tmp/mimi_work/mimi.safetensors";
const GOLD: &str = "/tmp/mimi_work/golden";
const REF_WAV: &str = "ref_voice.wav";

fn golden(name: &str) -> (Vec<f32>, Vec<usize>) {
    npy::load_npy(&Path::new(GOLD).join(format!("{name}.npy")))
        .unwrap_or_else(|e| panic!("golden {name}: {e}"))
}

fn metrics(name: &str, a: &[f32], b: &[f32], cos_gate: f64) -> bool {
    assert_eq!(a.len(), b.len(), "{name}: len {} vs {}", a.len(), b.len());
    let (mut dot, mut na, mut nb, mut maxabs) = (0f64, 0f64, 0f64, 0f64);
    for (x, y) in a.iter().zip(b) {
        let (x, y) = (*x as f64, *y as f64);
        dot += x * y;
        na += x * x;
        nb += y * y;
        maxabs = maxabs.max((x - y).abs());
    }
    let cos = dot / (na.sqrt() * nb.sqrt());
    let ok = cos > cos_gate;
    println!(
        "  {} {name:22} cos={cos:.8}  max|Δ|={maxabs:.3e}",
        if ok { "OK" } else { "XX" }
    );
    ok
}

fn main() {
    let loader = WeightLoader::SingleFile(SingleFileLoader::new(Path::new(CKPT)));
    let (samples, sr) = wav::read_pcm16_mono(Path::new(REF_WAV));
    assert_eq!(sr, SAMPLE_RATE);
    println!(
        "ref wav: {} samples ({} frames)",
        samples.len(),
        samples.len() / SAMPLES_PER_FRAME
    );

    let mut ok = true;

    // ---- ENCODER ----
    println!("loading encoder…");
    let enc = MimiEncoder::load(&loader);
    let (seanet, tr, ds, codes) = enc.encode_stages(&samples);

    // sub-stage gates (localize any divergence before the argmin)
    let (g_seanet, _) = golden("enc_seanet");
    ok &= metrics("enc seanet", &seanet, &g_seanet, 0.9999);
    let (g_tr, _) = golden("enc_transformer");
    ok &= metrics("enc transformer", &tr, &g_tr, 0.9999);
    let (g_ds, _) = golden("enc_downsample");
    ok &= metrics("enc downsample", &ds, &g_ds, 0.9999);

    // integer-exact codes gate
    let (gc, gcs) = golden("codes_f32");
    let (gt, gk) = (gcs[0], gcs[1]);
    let m = codes.len() == gt;
    let mut mism = 0usize;
    let mut per_q = [0usize; NUM_CODEBOOKS];
    let n = codes.len().min(gt);
    for t in 0..n {
        for q in 0..NUM_CODEBOOKS {
            if codes[t][q] != gc[t * gk + q] as u32 {
                mism += 1;
                per_q[q] += 1;
            }
        }
    }
    let total = n * NUM_CODEBOOKS;
    let em = m && mism == 0;
    ok &= em;
    println!(
        "  {} encode codes          {}/{} exact ({} frames vs {})",
        if em { "OK" } else { "XX" },
        total - mism,
        total,
        codes.len(),
        gt
    );
    if mism > 0 {
        println!("    per-quantizer mismatches: {per_q:?}");
        for t in 0..n.min(3) {
            let got: Vec<u32> = codes[t].to_vec();
            let want: Vec<u32> = (0..NUM_CODEBOOKS).map(|q| gc[t * gk + q] as u32).collect();
            println!("    frame {t}: got  {got:?}");
            println!("    frame {t}: want {want:?}");
        }
    }

    // The production path consumes one 80 ms frame at a time.  Gate it both
    // against the existing full-clip implementation and, transitively, the
    // exact upstream integer oracle above.
    assert_eq!(samples.len() % SAMPLES_PER_FRAME, 0);
    let mut stream_state = enc.stream_state();
    let stream_codes: Vec<_> = samples
        .chunks_exact(SAMPLES_PER_FRAME)
        .map(|frame| {
            enc.encode_stream_frame(
                &mut stream_state,
                frame.try_into().expect("exact Mimi input frame"),
            )
        })
        .collect();
    let stream_exact = stream_codes == codes;
    ok &= stream_exact;
    println!(
        "  {} streaming encode      {}/{} frames exact vs batch/oracle",
        if stream_exact { "OK" } else { "XX" },
        stream_codes
            .iter()
            .zip(&codes)
            .filter(|(stream, batch)| stream == batch)
            .count(),
        codes.len()
    );

    stream_state.reset();
    let reset_codes: Vec<_> = samples
        .chunks_exact(SAMPLES_PER_FRAME)
        .map(|frame| {
            enc.encode_stream_frame(
                &mut stream_state,
                frame.try_into().expect("exact Mimi input frame"),
            )
        })
        .collect();
    let reset_exact = reset_codes == stream_codes;
    ok &= reset_exact;
    println!(
        "  {} streaming reset       deterministic replay",
        if reset_exact { "OK" } else { "XX" }
    );

    // ---- DECODER ----
    println!("loading decoder…");
    let dec = MimiDecoder::load(&loader);

    // gate the decoder on the ORACLE codes (isolates decode from encode).
    let ref_codes: Vec<[u32; NUM_CODEBOOKS]> = (0..gt)
        .map(|t| {
            let mut f = [0u32; NUM_CODEBOOKS];
            for q in 0..NUM_CODEBOOKS {
                f[q] = gc[t * gk + q] as u32;
            }
            f
        })
        .collect();

    let (q, qt) = dec.quantizer_decode(&ref_codes);
    // golden is [1, 512, T]; our layout is [512, T] flat — same order.
    let (gq, gqs) = golden("dec_quantized");
    assert_eq!(qt, gqs[2]);
    ok &= metrics("quantizer decode", &q, &gq, 0.9999);

    let wav_out = dec.decode(&ref_codes);
    let (gwav, _) = golden("decode_wav");
    ok &= metrics("decode waveform", &wav_out, &gwav, 0.9999);
    wav::write_pcm16_mono(
        Path::new("/tmp/mimi_work/mimi_parity.wav"),
        &wav_out,
        SAMPLE_RATE,
    );

    // end-to-end: decode(OUR encode) vs the oracle roundtrip.
    let rt = dec.decode(&codes);
    let (grt, _) = golden("roundtrip_wav");
    ok &= metrics("roundtrip wav", &rt, &grt, 0.9999);

    println!(
        "{}",
        if ok {
            "ALL GATES PASSED"
        } else {
            "GATES FAILED"
        }
    );
    std::process::exit(if ok { 0 } else { 1 });
}
