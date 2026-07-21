//! Self-contained F5-TTS in Burn: speak `gen_text` in the voice of a reference
//! clip — no Python. A thin CLI wrapper over the [`mary::say`] library seam; the
//! actual synthesis lives in `mary::say::synthesize` so production callers use
//! the SAME code in-process (no separately-built production binary that can
//! drift stale against the pile blob format).
//!
//!   cargo run --release --example say -- <f5.safetensors> <ref.wav> <ref_text> <gen_text> [out.wav]

use std::path::{Path, PathBuf};

fn main() {
    let a: Vec<String> = std::env::args().collect();
    if a.len() < 5 {
        eprintln!("usage: say <f5.safetensors> <ref.wav> <ref_text> <gen_text> [out.wav]");
        std::process::exit(1);
    }
    let (f5_path, ref_wav, ref_text, gen_text) = (&a[1], &a[2], &a[3], &a[4]);
    let out_path = PathBuf::from(a.get(5).cloned().unwrap_or_else(|| "say_out.wav".into()));

    let n = mary::say::synthesize_to_wav(
        Path::new(f5_path),
        Path::new(ref_wav),
        ref_text,
        gen_text,
        &out_path,
    );
    println!(
        "✓ {} ({:.2}s) — \"{}\"",
        out_path.display(),
        n as f32 / mary::say::SAMPLE_RATE as f32,
        gen_text
    );
}
