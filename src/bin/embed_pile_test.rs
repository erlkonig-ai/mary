//! Embedder pile round-trip parity gate (the shell-is-physics endpoint for the
//! mary embedders). For BOTH CLIP and SigLIP:
//!
//!   1. resolve the HF snapshot dir from the local cache (no download),
//!   2. persist its `*.safetensors` into a TEMP pile FILE on disk
//!      (`persist_safetensors_to_pile` — each tensor a content-addressed f32 leaf),
//!   3. load the embedder from JUST the pile (`load_*_from_pile`) AND the normal
//!      safetensors way (`load_*_from_hf`),
//!   4. embed a test image + a text with both,
//!   5. assert cosine(pile_vec, safetensors_vec) > 0.999999 for image AND text.
//!
//! The f32 pile round-trip is LOSSLESS, so the two builds must be ~identical.
//!
//!   cargo run --release --features embed --bin embed_pile_test

use mary::embed::{
    load_clip_from_hf, load_clip_from_pile, load_siglip_from_hf, load_siglip_from_pile,
    LocalEmbedder,
};
use mary::nn::backend::WgpuDevice;
use mary::persist::persist_safetensors_to_pile;
use std::path::{Path, PathBuf};

const IMAGE: &str = "test_image.png"; // any local test image (repo-relative or absolute)
const TEXT: &str = "a cartoon mouse";
const CLIP_ID: &str = "openai/clip-vit-base-patch32";
const SIGLIP_ID: &str = "google/siglip2-so400m-patch14-384";
const THRESHOLD: f32 = 0.999999;

fn cosine(a: &[f32], b: &[f32]) -> f32 {
    let dot: f32 = a.iter().zip(b).map(|(x, y)| x * y).sum();
    let na: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let nb: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    dot / (na * nb).max(1e-12)
}

fn mark(ok: bool) -> &'static str {
    if ok {
        "OK"
    } else {
        "<-- FAIL"
    }
}

/// Find a file in any snapshot of a cached HF model — robust to HF's
/// split-snapshot layout (config/tokenizer and weights can land in different
/// snapshot dirs). Pure cache lookup, no download.
fn hf_cache_resolve(model_id: &str, filename: &str) -> Option<PathBuf> {
    let home = std::env::var("HOME").ok()?;
    let repo = format!("models--{}", model_id.replace('/', "--"));
    let snapshots = Path::new(&home)
        .join(".cache/huggingface/hub")
        .join(repo)
        .join("snapshots");
    for snap in std::fs::read_dir(&snapshots).ok()?.flatten() {
        let p = snap.path().join(filename);
        if p.exists() {
            return Some(p);
        }
    }
    None
}

fn gib(bytes: u64) -> f64 {
    bytes as f64 / (1u64 << 30) as f64
}

fn main() {
    let device: WgpuDevice = Default::default();
    let img_bytes = std::fs::read(IMAGE).expect("read test image");
    let mut all_pass = true;

    // ── CLIP ────────────────────────────────────────────────────────────────
    {
        println!("=== CLIP ({CLIP_ID}) ===");
        let weights = hf_cache_resolve(CLIP_ID, "model.safetensors")
            .expect("clip model.safetensors not in HF cache — fetch it first");
        let tokenizer = hf_cache_resolve(CLIP_ID, "tokenizer.json")
            .expect("clip tokenizer.json not in HF cache");
        let snapshot_dir = weights.parent().unwrap();

        let pile_path =
            std::env::temp_dir().join(format!("mary_embed_clip_{}.pile", std::process::id()));
        let _ = std::fs::remove_file(&pile_path);
        eprintln!("[clip] persisting {snapshot_dir:?} → {pile_path:?} ...");
        persist_safetensors_to_pile(snapshot_dir, &pile_path, mary::ingest::LeafDtype::F32)
            .expect("persist clip to pile");
        let pile_size = std::fs::metadata(&pile_path).unwrap().len();
        eprintln!(
            "[clip] pile is {} bytes ({:.3} GiB).",
            pile_size,
            gib(pile_size)
        );

        let from_safe = load_clip_from_hf(CLIP_ID, device.clone()).expect("load clip from hf");
        let from_pile = load_clip_from_pile(&pile_path, &tokenizer, device.clone())
            .expect("load clip from pile");

        let safe_img = from_safe.embed_image(&img_bytes).expect("safe image");
        let pile_img = from_pile.embed_image(&img_bytes).expect("pile image");
        let safe_txt = from_safe.embed_text(TEXT).expect("safe text");
        let pile_txt = from_pile.embed_text(TEXT).expect("pile text");

        let c_img = cosine(&pile_img, &safe_img);
        let c_txt = cosine(&pile_txt, &safe_txt);
        let img_ok = c_img > THRESHOLD;
        let txt_ok = c_txt > THRESHOLD;
        println!(
            "  image cos(pile, safetensors) = {c_img:.8}  {}",
            mark(img_ok)
        );
        println!(
            "  text  cos(pile, safetensors) = {c_txt:.8}  {}",
            mark(txt_ok)
        );
        println!(
            "  pile file: {pile_path:?}  ({} bytes, {:.3} GiB)",
            pile_size,
            gib(pile_size)
        );
        all_pass &= img_ok && txt_ok;

        let _ = std::fs::remove_file(&pile_path);
    }

    // ── SigLIP ──────────────────────────────────────────────────────────────
    {
        println!("\n=== SigLIP ({SIGLIP_ID}) ===");
        let weights = hf_cache_resolve(SIGLIP_ID, "model.safetensors")
            .expect("siglip model.safetensors not in HF cache — fetch it first");
        let tokenizer = hf_cache_resolve(SIGLIP_ID, "tokenizer.json")
            .expect("siglip tokenizer.json not in HF cache");
        let snapshot_dir = weights.parent().unwrap();

        let pile_path =
            std::env::temp_dir().join(format!("mary_embed_siglip_{}.pile", std::process::id()));
        let _ = std::fs::remove_file(&pile_path);
        eprintln!("[siglip] persisting {snapshot_dir:?} → {pile_path:?} ...");
        persist_safetensors_to_pile(snapshot_dir, &pile_path, mary::ingest::LeafDtype::F32)
            .expect("persist siglip to pile");
        let pile_size = std::fs::metadata(&pile_path).unwrap().len();
        eprintln!(
            "[siglip] pile is {} bytes ({:.3} GiB).",
            pile_size,
            gib(pile_size)
        );

        let from_safe =
            load_siglip_from_hf(SIGLIP_ID, device.clone()).expect("load siglip from hf");
        let from_pile = load_siglip_from_pile(&pile_path, &tokenizer, device.clone())
            .expect("load siglip from pile");

        let safe_img = from_safe.embed_image(&img_bytes).expect("safe image");
        let pile_img = from_pile.embed_image(&img_bytes).expect("pile image");
        let safe_txt = from_safe.embed_text(TEXT).expect("safe text");
        let pile_txt = from_pile.embed_text(TEXT).expect("pile text");

        let c_img = cosine(&pile_img, &safe_img);
        let c_txt = cosine(&pile_txt, &safe_txt);
        let img_ok = c_img > THRESHOLD;
        let txt_ok = c_txt > THRESHOLD;
        println!(
            "  image cos(pile, safetensors) = {c_img:.8}  {}",
            mark(img_ok)
        );
        println!(
            "  text  cos(pile, safetensors) = {c_txt:.8}  {}",
            mark(txt_ok)
        );
        println!(
            "  pile file: {pile_path:?}  ({} bytes, {:.3} GiB)",
            pile_size,
            gib(pile_size)
        );
        all_pass &= img_ok && txt_ok;

        let _ = std::fs::remove_file(&pile_path);
    }

    println!("\n=== {} ===", if all_pass { "PASS" } else { "FAIL" });
    if !all_pass {
        std::process::exit(1);
    }
}
