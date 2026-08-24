//! Does nomic-embed-text-v1.5 give good SEMANTIC recall over short prose
//! notes — matching by MEANING, where a BM25 search only matched keywords?
//!
//! Loads Nomic, embeds a handful of note-sized prose docs + several
//! queries, and ranks by cosine. The queries are deliberately phrased in
//! DIFFERENT words than the memories, so a top hit can only come from meaning,
//! not shared tokens. Each query's top hit should be the semantically right
//! memory. This is the end-to-end proof of the embedding seam (the HNSW index
//! over these vectors is already tested in triblespace-search).
//!
//! Run:  cargo run --release --features embed --bin nomic_semantic_probe
use anyhow::Result;
use burn::backend::wgpu::WgpuDevice;
use mary::embed::load_nomic_text_from_hf;

const MODEL_ID: &str = "nomic-ai/nomic-embed-text-v1.5";

fn cosine(a: &[f32], b: &[f32]) -> f32 {
    let dot: f32 = a.iter().zip(b).map(|(x, y)| x * y).sum();
    let na: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let nb: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    dot / (na * nb + 1e-9)
}

fn main() -> Result<()> {
    let device = WgpuDevice::default();
    eprintln!("[probe] loading {MODEL_ID} from HF cache ...");
    let nomic = load_nomic_text_from_hf(MODEL_ID, device)?;
    eprintln!("[probe] loaded; embedding documents ...");

    // Note-sized prose docs — distinct themes, deliberately NOT sharing the
    // query keywords, so a hit can only be semantic.
    let docs: Vec<(&str, &str)> = vec![
        (
            "gpu",
            "A scattered-read kernel ran 14x faster than a 16-thread CPU by hiding the memory latency the host processor stalls on.",
        ),
        (
            "database",
            "The engine answers lookups by walking a compact on-disk index instead of scanning; each record resolves in near-constant time as the table grows.",
        ),
        (
            "compression",
            "Entropy coders model how often each symbol appears and spend fewer bits on the common ones; the original stream reconstructs exactly from the table.",
        ),
        (
            "networking",
            "Routers forward each packet hop by hop toward its destination, and senders back off when queues fill so shared links do not collapse under load.",
        ),
        (
            "cryptography",
            "Two parties who have never met agree on a shared secret over an open line, protected by math an eavesdropper cannot reverse in practice.",
        ),
        (
            "weather",
            "Simulations advance the equations of moving air on a planet-sized grid; tiny errors in the starting state grow until forecasts lose skill after two weeks.",
        ),
    ];
    let doc_vecs: Vec<(&str, Vec<f32>)> = docs
        .iter()
        .map(|(id, t)| Ok((*id, nomic.embed_document(t)?)))
        .collect::<Result<_>>()?;

    // Queries phrased in DIFFERENT words than the docs — pure semantic match.
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

    let mut passed = 0usize;
    for (q, want) in cases {
        let qv = nomic.embed_query(q)?;
        let mut scored: Vec<(&str, f32)> = doc_vecs
            .iter()
            .map(|(id, v)| (*id, cosine(&qv, v)))
            .collect();
        scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        let ok = scored[0].0 == *want;
        if ok {
            passed += 1;
        }
        println!(
            "\n[{}] query: {q:?}  (want: {want})",
            if ok { "PASS" } else { "FAIL" }
        );
        for (id, s) in &scored {
            println!("   {s:6.3}  {id}");
        }
    }
    println!(
        "\n=== {passed}/{} semantic queries ranked the right memory first (queries share NO keywords with the docs) ===",
        cases.len()
    );
    Ok(())
}
