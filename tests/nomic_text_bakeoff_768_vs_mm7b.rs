//! Quality bake-off: is `nomic-embed-multimodal-7b`'s TEXT→TEXT retrieval as
//! good as the dedicated `nomic-embed-text-v1.5` on note-sized prose?
//!
//! Decides an architecture call: migrate all text to the 7b's 3584-dim space, vs
//! keep nomic-text-768 for text and use the 7b only for images. This test is
//! MEASUREMENT only — no assertions on which model wins; it prints a side-by-side
//! table of per-query top-1 correctness and top1−top2 cosine margin for both
//! embedders, plus totals, means, and per-embed latency.
//!
//! Corpus + queries are copied verbatim from `src/bin/nomic_semantic_probe.rs`
//! (6 note-sized prose docs; 6 paraphrase queries that share NO keywords
//! with the docs, so a hit can only come from meaning).
//!
//! Both embedders run on the Metal GPU (`B = Metal`, `WgpuDevice`). The 7b loads
//! aliased-from-pile (mmap f16 -> Metal); the 768 loads safetensors from the HF
//! cache. Disk-gated: SKIPS cleanly if the 7b pile / tokenizer / 768 cache are
//! absent.
//!
//!   cargo test --release --features "gemma embed" \
//!       --test nomic_text_bakeoff_768_vs_mm7b -- --nocapture
//!
//! macOS / Metal only.
#![cfg(all(target_os = "macos", feature = "gemma", feature = "embed"))]

use mary::embed::load_nomic_text_from_hf;
use mary::models::gemma::metal_device::init_metal_device_16gb;
use std::path::PathBuf;
use std::time::Instant;

const MODEL_768: &str = "nomic-ai/nomic-embed-text-v1.5";

fn cosine(a: &[f32], b: &[f32]) -> f32 {
    let dot: f32 = a.iter().zip(b).map(|(x, y)| x * y).sum();
    let na: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let nb: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    dot / (na * nb + 1e-9)
}

/// Rank `query` against the doc embeddings; return (top_id, ok, margin = top1−top2).
fn rank<'a>(qv: &[f32], doc_vecs: &[(&'a str, Vec<f32>)], want: &str) -> (&'a str, bool, f32) {
    let mut scored: Vec<(&str, f32)> = doc_vecs
        .iter()
        .map(|(id, v)| (*id, cosine(qv, v)))
        .collect();
    scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    let top = scored[0].0;
    let margin = scored[0].1 - scored[1].1;
    (top, top == want, margin)
}

/// Best-effort tokenizer for the 7b (we feed text; any nomic-mm7b/Qwen2.5-VL
/// tokenizer works). Honors NOMIC_MM7B_TOKENIZER, else scans the HF cache.
fn nomic_mm7b_tokenizer() -> Option<PathBuf> {
    if let Ok(p) = std::env::var("NOMIC_MM7B_TOKENIZER") {
        let p = PathBuf::from(p);
        if p.exists() {
            return Some(p);
        }
    }
    let home = std::env::var("HOME").ok()?;
    for pat in [
        "models--nomic-ai--nomic-embed-multimodal-7b",
        "models--Qwen--Qwen2.5-VL-7B-Instruct",
    ] {
        let base = PathBuf::from(&home)
            .join(".cache/huggingface/hub")
            .join(pat)
            .join("snapshots");
        if let Ok(rd) = std::fs::read_dir(&base) {
            for e in rd.flatten() {
                let p = e.path().join("tokenizer.json");
                if p.exists() {
                    return Some(p);
                }
            }
        }
    }
    None
}

#[test]
fn text_retrieval_bakeoff() {
    // --- corpus + queries: copied verbatim from src/bin/nomic_semantic_probe.rs ---
    let docs: Vec<(&str, &str)> = vec![
        ("gpu", "A scattered-read kernel ran 14x faster than a 16-thread CPU by hiding the memory latency the host processor stalls on."),
        ("database", "The engine answers lookups by walking a compact on-disk index instead of scanning; each record resolves in near-constant time as the table grows."),
        ("compression", "Entropy coders model how often each symbol appears and spend fewer bits on the common ones; the original stream reconstructs exactly from the table."),
        ("networking", "Routers forward each packet hop by hop toward its destination, and senders back off when queues fill so shared links do not collapse under load."),
        ("cryptography", "Two parties who have never met agree on a shared secret over an open line, protected by math an eavesdropper cannot reverse in practice."),
        ("weather", "Simulations advance the equations of moving air on a planet-sized grid; tiny errors in the starting state grow until forecasts lose skill after two weeks."),
    ];
    let cases: &[(&str, &str)] = &[
        ("CUDA shaders and parallel compute acceleration", "gpu"),
        ("fast record lookups and storage efficiency", "database"),
        ("shrinking files without losing information", "compression"),
        ("how data travels across the internet", "networking"),
        (
            "sending private messages over a public channel",
            "cryptography",
        ),
        ("predicting storms and rain with computers", "weather"),
    ];

    // --- disk gates ---
    let Some(pile_path) = mary::paths::model_opt(
        std::env::var("NOMIC_MM7B_PILE").ok().as_deref(),
        "nomic_mm7b.pile",
    ) else {
        eprintln!("SKIP: {}", mary::paths::skip_reason("nomic_mm7b.pile"));
        return;
    };
    let Some(tok_path) = nomic_mm7b_tokenizer() else {
        eprintln!("SKIP: no nomic-mm7b/Qwen2.5-VL tokenizer.json in HF cache");
        return;
    };

    let device = init_metal_device_16gb();

    // ===== 768: nomic-embed-text-v1.5 =====
    eprintln!("[bakeoff] loading 768 ({MODEL_768}) from HF cache ...");
    let t0 = Instant::now();
    let nomic768 = match load_nomic_text_from_hf(MODEL_768, device.clone()) {
        Ok(m) => m,
        Err(e) => {
            eprintln!("SKIP: 768 not in HF cache ({e})");
            return;
        }
    };
    let load_768_ms = t0.elapsed().as_millis();

    let mut dvecs_768: Vec<(&str, Vec<f32>)> = Vec::new();
    let t = Instant::now();
    for (id, txt) in &docs {
        dvecs_768.push((*id, nomic768.embed_document(txt).expect("768 doc")));
    }
    let doc_768_ms = t.elapsed().as_millis();
    let dim_768 = dvecs_768[0].1.len();

    let mut res_768: Vec<(&str, bool, f32)> = Vec::new();
    let t = Instant::now();
    for (q, want) in cases {
        let qv = nomic768.embed_query(q).expect("768 query");
        res_768.push(rank(&qv, &dvecs_768, want));
    }
    let q_768_ms = t.elapsed().as_millis();
    drop(nomic768);

    // ===== 7b: nomic-embed-multimodal-7b (text path) =====
    eprintln!(
        "[bakeoff] loading 7b (aliased f16 -> Metal) from {} ...",
        pile_path.display()
    );
    let t0 = Instant::now();
    let snapshot = mary::model_collection::load_model_collection_local_latest(
        &pile_path,
        mary::model_collection::model_graph_team_at(&pile_path).expect("sole model-graph team"),
    )
    .expect("load native model collection snapshot");
    let nomic7b = mary::persist::load_nomic_mm7b_aliased_from_snapshot(
        snapshot,
        &tok_path,
        device.clone(),
    )
    .expect("7b aliased load");
    let load_7b_ms = t0.elapsed().as_millis();

    let mut dvecs_7b: Vec<(&str, Vec<f32>)> = Vec::new();
    let t = Instant::now();
    for (id, txt) in &docs {
        dvecs_7b.push((*id, nomic7b.embed_document(txt).expect("7b doc")));
    }
    let doc_7b_ms = t.elapsed().as_millis();
    let dim_7b = dvecs_7b[0].1.len();

    let mut res_7b: Vec<(&str, bool, f32)> = Vec::new();
    let t = Instant::now();
    for (q, want) in cases {
        let qv = nomic7b.embed_query(q).expect("7b query");
        res_7b.push(rank(&qv, &dvecs_7b, want));
    }
    let q_7b_ms = t.elapsed().as_millis();

    // ===== side-by-side table =====
    println!("\n================ TEXT→TEXT RETRIEVAL BAKE-OFF ================");
    println!("768 = nomic-embed-text-v1.5 (dim {dim_768})   |   7b = nomic-embed-multimodal-7b text path (dim {dim_7b})");
    println!(
        "\n{:<46} | {:<11} {:<5} {:<8} | {:<11} {:<5} {:<8}",
        "query (want)", "768 top", "ok?", "margin", "7b top", "ok?", "margin"
    );
    println!("{}", "-".repeat(46 + 3 + 26 + 3 + 26));
    let mut ok768 = 0usize;
    let mut ok7b = 0usize;
    let mut m768_sum = 0.0f32;
    let mut m7b_sum = 0.0f32;
    for (i, (q, want)) in cases.iter().enumerate() {
        let (t8, o8, mg8) = res_768[i];
        let (t7, o7, mg7) = res_7b[i];
        ok768 += o8 as usize;
        ok7b += o7 as usize;
        m768_sum += mg8;
        m7b_sum += mg7;
        let qlabel = {
            let mut s = format!("{q} ({want})");
            if s.len() > 46 {
                s.truncate(43);
                s.push_str("...");
            }
            s
        };
        println!(
            "{:<46} | {:<11} {:<5} {:<8.4} | {:<11} {:<5} {:<8.4}",
            qlabel,
            t8,
            if o8 { "yes" } else { "NO" },
            mg8,
            t7,
            if o7 { "yes" } else { "NO" },
            mg7,
        );
    }
    let n = cases.len() as f32;
    println!("{}", "-".repeat(46 + 3 + 26 + 3 + 26));
    println!(
        "{:<46} | {:<11} {:<5} {:<8.4} | {:<11} {:<5} {:<8.4}",
        "TOTALS",
        format!("{ok768}/{}", cases.len()),
        "",
        m768_sum / n,
        format!("{ok7b}/{}", cases.len()),
        "",
        m7b_sum / n,
    );
    println!("(margin column = mean top1−top2 cosine on the TOTALS row)");

    // ===== latency =====
    println!("\n---- latency (Metal GPU) ----");
    println!(
        "768: cold load {load_768_ms} ms | 6 docs {doc_768_ms} ms ({:.1} ms/text) | 6 queries {q_768_ms} ms ({:.1} ms/text)",
        doc_768_ms as f64 / 6.0,
        q_768_ms as f64 / 6.0
    );
    println!(
        "7b : cold load {load_7b_ms} ms | 6 docs {doc_7b_ms} ms ({:.1} ms/text) | 6 queries {q_7b_ms} ms ({:.1} ms/text)",
        doc_7b_ms as f64 / 6.0,
        q_7b_ms as f64 / 6.0
    );
    println!("=============================================================\n");
}
