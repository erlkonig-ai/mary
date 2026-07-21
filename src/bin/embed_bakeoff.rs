//! Cross-modal bake-off: Nomic (vision↔text, shared 768-d) vs SigLIP2 (1152-d).
//!
//!   cargo run --release --features embed --bin embed_bakeoff
//!
//! For a set of local test images with known content and a few text queries,
//! print each model's cosine ranking of images per query, side by side — to SEE
//! whether Nomic's text→image ranking is sensible where SigLIP's was reportedly
//! wrong (SigLIP ranked "a face" LAST for portrait renders). Also print the
//! image-image cosine matrix per model. Qualitative; no hard assert.
//! Point IMAGES at any three local test images with matching descriptions.

use mary::embed::{
    load_nomic_text_from_hf, load_nomic_vision_from_hf, load_siglip_from_hf, LocalEmbedder,
};
use mary::nn::backend::WgpuDevice;

const NOMIC_TEXT: &str = "nomic-ai/nomic-embed-text-v1.5";
const NOMIC_VISION: &str = "nomic-ai/nomic-embed-vision-v1.5";
const SIGLIP: &str = "google/siglip2-so400m-patch14-384";

/// (path, short human description)
const IMAGES: &[(&str, &str)] = &[
    ("test_images/cartoon_mouse.png", "render: a cartoon mouse face"),
    ("test_images/portrait.png", "render: a generated face/portrait"),
    ("test_images/shapes.jpg", "photo: colored geometric shapes"),
];

const QUERIES: &[&str] = &[
    "a face",
    "a logo",
    "a screenshot",
    "a cartoon character",
    // clear-case probes (unambiguous best match in braces) to test whether
    // cross-modal alignment works at all per model:
    "a cartoon mouse",            // -> mickey
    "a glowing portrait of a woman", // -> burn21
    "colored geometric shapes",   // -> shapes (image.jpg)
];

fn cosine(a: &[f32], b: &[f32]) -> f32 {
    let dot: f32 = a.iter().zip(b).map(|(x, y)| x * y).sum();
    let na: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let nb: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    dot / (na * nb).max(1e-12)
}

fn short(path: &str) -> &str {
    path.rsplit('/').next().unwrap_or(path)
}

/// Print "query: img(score) > img(score) > ..." ranked descending.
fn print_ranking(label: &str, _query: &str, img_paths: &[&str], q_vec: &[f32], img_vecs: &[Vec<f32>]) {
    let mut scored: Vec<(usize, f32)> =
        img_vecs.iter().enumerate().map(|(i, v)| (i, cosine(q_vec, v))).collect();
    scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    let ranked: Vec<String> =
        scored.iter().map(|(i, s)| format!("{} ({s:+.4})", short(img_paths[*i]))).collect();
    println!("    {label:7}: {}", ranked.join("  >  "));
}

fn image_matrix(label: &str, img_paths: &[&str], vecs: &[Vec<f32>]) {
    println!("  [{label}] image-image cosine:");
    print!("    {:<28}", "");
    for p in img_paths {
        print!("{:>14}", short(p).chars().take(13).collect::<String>());
    }
    println!();
    for (i, p) in img_paths.iter().enumerate() {
        print!("    {:<28}", short(p).chars().take(27).collect::<String>());
        for j in 0..img_paths.len() {
            print!("{:>14.4}", cosine(&vecs[i], &vecs[j]));
        }
        println!();
    }
}

fn main() {
    let device: WgpuDevice = Default::default();
    let img_paths: Vec<&str> = IMAGES.iter().map(|(p, _)| *p).collect();
    let bytes: Vec<Vec<u8>> = img_paths
        .iter()
        .map(|p| std::fs::read(p).unwrap_or_else(|e| panic!("read {p}: {e}")))
        .collect();

    println!("=== IMAGES ===");
    for (p, desc) in IMAGES {
        println!("  {}  —  {desc}", short(p));
    }

    // ---- NOMIC (cross-modal: vision tower + text tower, shared 768-d) ----
    println!("\nloading Nomic vision + text ...");
    let nomic_v = load_nomic_vision_from_hf(NOMIC_VISION, device.clone()).expect("nomic vision");
    let nomic_t = load_nomic_text_from_hf(NOMIC_TEXT, device.clone()).expect("nomic text");
    let nomic_img: Vec<Vec<f32>> =
        bytes.iter().map(|b| nomic_v.embed_image(b).expect("nomic embed_image")).collect();
    let nomic_q: Vec<Vec<f32>> =
        QUERIES.iter().map(|q| nomic_t.embed_query(q).expect("nomic embed_query")).collect();

    // ---- SigLIP (image+text, 1152-d) ----
    println!("loading SigLIP2 ...");
    let siglip = load_siglip_from_hf(SIGLIP, device.clone()).expect("siglip");
    let siglip_img: Vec<Vec<f32>> =
        bytes.iter().map(|b| siglip.embed_image(b).expect("siglip embed_image")).collect();
    let siglip_q: Vec<Vec<f32>> =
        QUERIES.iter().map(|q| siglip.embed_text(q).expect("siglip embed_text")).collect();

    println!("\n=== TEXT->IMAGE RANKINGS (per query, descending cosine) ===");
    for (qi, q) in QUERIES.iter().enumerate() {
        println!("  query: {q:?}");
        print_ranking("NOMIC", q, &img_paths, &nomic_q[qi], &nomic_img);
        print_ranking("SigLIP", q, &img_paths, &siglip_q[qi], &siglip_img);
    }

    println!("\n=== IMAGE-IMAGE COSINE MATRICES ===");
    image_matrix("NOMIC", &img_paths, &nomic_img);
    println!();
    image_matrix("SigLIP", &img_paths, &siglip_img);
}
