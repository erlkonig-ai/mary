//! CLIP embedder parity gate.
//!
//!   cargo run --release --features embed --bin clip_embed_test
//!
//! 1. Shell out to python (transformers `CLIPModel`/`CLIPProcessor` on
//!    openai/clip-vit-base-patch32) to dump L2-normalized reference vectors for
//!    a test image + 3 texts to /tmp/clip_ref.json.
//! 2. Embed the same image + texts via `mary::embed::ClipEmbedder`.
//! 3. Assert PARITY (cosine(rust, hf) > 0.99 per item) and SEMANTIC ordering
//!    (the image's best-matching text is the true caption).
//! 4. Print every cosine + PASS/FAIL.

use mary::embed::{load_clip_from_hf, LocalEmbedder, CLIP_DIM};
use mary::nn::backend::WgpuDevice;
use std::process::Command;

const IMAGE: &str = "test_image.png"; // any local test image (repo-relative or absolute)
const TEXTS: [&str; 3] = [
    "a cartoon mouse",
    "a photo of a truck",
    "a screenshot of code",
];
const REF_JSON: &str = "/tmp/clip_ref.json";

fn cosine(a: &[f32], b: &[f32]) -> f32 {
    let dot: f32 = a.iter().zip(b).map(|(x, y)| x * y).sum();
    let na: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let nb: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    dot / (na * nb).max(1e-12)
}

fn gen_reference() {
    let texts_py = format!(
        "[{}]",
        TEXTS
            .iter()
            .map(|t| format!("{t:?}"))
            .collect::<Vec<_>>()
            .join(", ")
    );
    let script = format!(
        r#"
import json, numpy as np, torch
from transformers import CLIPModel, CLIPProcessor
from PIL import Image

model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").eval()
proc = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

img = Image.open({image:?}).convert("RGB")
texts = {texts}

# transformers 5.x: get_*_features returns BaseModelOutputWithPooling whose
# pooler_output is the projected 512-d feature; older returns a bare tensor.
def feat(out):
    return out.pooler_output if hasattr(out, "pooler_output") else out

with torch.no_grad():
    pix = proc(images=img, return_tensors="pt")
    imf = feat(model.get_image_features(**pix))
    imf = (imf / imf.norm(dim=-1, keepdim=True)).squeeze(0).cpu().numpy().tolist()

    tok = proc(text=texts, return_tensors="pt", padding=True, truncation=True)
    txf = feat(model.get_text_features(**tok))
    txf = (txf / txf.norm(dim=-1, keepdim=True)).cpu().numpy().tolist()

json.dump({{"image": imf, "texts": txf}}, open({ref:?}, "w"))
print("ref written")
"#,
        image = IMAGE,
        texts = texts_py,
        ref = REF_JSON,
    );
    let status = Command::new("python3")
        .arg("-c")
        .arg(&script)
        .status()
        .expect("spawn python3 (need transformers + torch installed)");
    assert!(status.success(), "python reference generation failed");
}

fn main() {
    // load_clip_from_hf resolves weights+tokenizer across snapshots — no symlink needed.

    println!("generating HF reference (python transformers)...");
    gen_reference();
    let refv: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(REF_JSON).unwrap()).unwrap();
    let hf_image: Vec<f32> = refv["image"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_f64().unwrap() as f32)
        .collect();
    let hf_texts: Vec<Vec<f32>> = refv["texts"]
        .as_array()
        .unwrap()
        .iter()
        .map(|row| {
            row.as_array()
                .unwrap()
                .iter()
                .map(|v| v.as_f64().unwrap() as f32)
                .collect()
        })
        .collect();

    println!("loading ClipEmbedder (mary)...");
    let device: WgpuDevice = Default::default();
    let clip = load_clip_from_hf("openai/clip-vit-base-patch32", device).expect("load clip");
    assert_eq!(clip.dim(), CLIP_DIM);

    let img_bytes = std::fs::read(IMAGE).expect("read image");
    let rust_image = clip.embed_image(&img_bytes).expect("embed image");
    let rust_texts: Vec<Vec<f32>> = TEXTS
        .iter()
        .map(|t| clip.embed_text(t).expect("embed text"))
        .collect();

    // ---- PARITY ----
    let mut pass = true;
    println!("\n=== PARITY (cosine rust-vs-HF, must be > 0.99) ===");
    let c_img = cosine(&rust_image, &hf_image);
    println!(
        "  image                         : {c_img:.6}  {}",
        mark(c_img > 0.99)
    );
    pass &= c_img > 0.99;
    for (i, t) in TEXTS.iter().enumerate() {
        let c = cosine(&rust_texts[i], &hf_texts[i]);
        println!("  text[{i}] {t:<28}: {c:.6}  {}", mark(c > 0.99));
        pass &= c > 0.99;
    }

    // ---- SEMANTIC ----
    println!("\n=== SEMANTIC (image vs each text; best should be text[0]) ===");
    let mut best = 0usize;
    let mut best_c = f32::MIN;
    for (i, t) in TEXTS.iter().enumerate() {
        let c = cosine(&rust_image, &rust_texts[i]);
        println!("  cos(image, \"{t}\") = {c:.6}");
        if c > best_c {
            best_c = c;
            best = i;
        }
    }
    let semantic_ok = best == 0;
    println!(
        "  best match: text[{best}] \"{}\"  {}",
        TEXTS[best],
        mark(semantic_ok)
    );
    pass &= semantic_ok;

    println!("\n=== {} ===", if pass { "PASS" } else { "FAIL" });
    if !pass {
        std::process::exit(1);
    }
}

fn mark(ok: bool) -> &'static str {
    if ok {
        "OK"
    } else {
        "<-- FAIL"
    }
}
