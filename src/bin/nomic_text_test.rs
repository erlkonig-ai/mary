//! nomic-embed-text-v1.5 parity gate.
//!
//!   cargo run --release --features embed,import --bin nomic_text_test
//!
//! 1. Shell out to python (sentence-transformers, or transformers AutoModel as a
//!    fallback) to dump L2-normalized reference vectors for a short text, a
//!    sentence, and a ~150-word multi-paragraph blob — all with the
//!    "search_document: " prefix — to /tmp/nomic_ref.json.
//! 2. Embed the same texts via `mary::embed::NomicTextEmbedder::embed_document`.
//! 3. Assert PARITY (cosine(rust, ref) > 0.99 per text, incl the long one).
//! 4. Native collection round-trip: publish weights + tokenizer atomically,
//!    fresh-load one frozen snapshot, and assert cosine(native, HF) ~ 1.0.
//! 5. Print every cosine + token counts + PASS/FAIL.

#[path = "support/native_embedding_collection.rs"]
mod native_embedding_collection;

use ed25519_dalek::SigningKey;
use mary::embed::{
    EmbeddingArchitecture, LocalEmbedder, NOMIC_TEXT_DIM, hf_cache_main_snapshot,
    load_nomic_text_from_files, nomic_text_from_parts,
};
use mary::nn::backend::WgpuDevice;
use mary::selection::{ModelSelector, TokenizerSelector};
use std::process::Command;
use triblespace::core::repo::pile::Pile;

const MODEL_ID: &str = "nomic-ai/nomic-embed-text-v1.5";
const REF_JSON: &str = "/tmp/nomic_ref.json";

const SHORT: &str = "a cartoon mouse";
const SENTENCE: &str =
    "The quick brown fox jumps over the lazy dog while the sun sets behind the hills.";
const LONG: &str = "A glacier is not a single block of ice but a slow river of compacted \
snow, each season adding a distinct layer that records the climate of its year. \
Some layers carry volcanic ash, others trap ancient air, and together they flow \
downhill under their own weight, deforming and fracturing as the terrain demands. \
No central plan coordinates the movement; instead, the flow emerges from the local \
pressure and temperature of countless ice crystals sharing a common mass. Geologists \
have long studied where accumulation ends and ablation begins, because the boundary \
migrates with every season. This ambiguity makes glaciers a favorite subject for \
studying systems that change faster than they appear to, where enormous consequences \
arise from slow, distributed processes. The great valley glaciers of the Alps are the \
most famous examples, grinding rock into flour and leaving moraines that trace their \
former extent. What looks like one static object is, on closer inspection, a dynamic \
archive of centuries of weather, an elegant demonstration that permanence at one scale \
can dissolve into constant motion at another.";

fn texts() -> [&'static str; 3] {
    [SHORT, SENTENCE, LONG]
}

fn cosine(a: &[f32], b: &[f32]) -> f32 {
    let dot: f32 = a.iter().zip(b).map(|(x, y)| x * y).sum();
    let na: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let nb: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    dot / (na * nb).max(1e-12)
}

fn mark(ok: bool) -> &'static str {
    if ok { "OK" } else { "<-- FAIL" }
}

fn gib(bytes: u64) -> f64 {
    bytes as f64 / (1u64 << 30) as f64
}

fn gen_reference() {
    let texts_py = format!(
        "[{}]",
        texts()
            .iter()
            .map(|t| format!("{t:?}"))
            .collect::<Vec<_>>()
            .join(", ")
    );
    // Prefer sentence-transformers; fall back to transformers AutoModel with a
    // hand-rolled masked-mean-pool + normalize if sentence_transformers is absent.
    let script = format!(
        r#"
import json
texts = {texts}
prefixed = ["search_document: " + t for t in texts]
vecs = None
try:
    from sentence_transformers import SentenceTransformer
    m = SentenceTransformer("{model}", trust_remote_code=True)
    vecs = m.encode(prefixed, normalize_embeddings=True).tolist()
    print("ref via sentence_transformers")
except Exception as e:
    print("sentence_transformers unavailable (%s); falling back to transformers" % e)
    import torch, torch.nn.functional as F
    from transformers import AutoTokenizer, AutoModel
    tok = AutoTokenizer.from_pretrained("{model}")
    model = AutoModel.from_pretrained("{model}", trust_remote_code=True).eval()
    enc = tok(prefixed, padding=True, truncation=True, max_length=8192, return_tensors="pt")
    with torch.no_grad():
        out = model(**enc)[0]  # last_hidden_state [b,s,d]
        mask = enc["attention_mask"].unsqueeze(-1).float()
        pooled = (out * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
        pooled = F.normalize(pooled, p=2, dim=1)
    vecs = pooled.cpu().numpy().tolist()
json.dump({{"texts": vecs}}, open("{ref}", "w"))
print("ref written")
"#,
        texts = texts_py,
        model = MODEL_ID,
        ref = REF_JSON,
    );
    let status = Command::new("python3")
        .arg("-c")
        .arg(&script)
        .status()
        .expect("spawn python3 (need sentence-transformers or transformers+torch)");
    assert!(status.success(), "python reference generation failed");
}

fn main() {
    println!("generating reference (python)...");
    gen_reference();
    let refv: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(REF_JSON).unwrap()).unwrap();
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

    let source_snapshot =
        hf_cache_main_snapshot(MODEL_ID).expect("resolve one cached Nomic text main revision");
    let weights = source_snapshot.join("model.safetensors");
    let tokenizer = source_snapshot.join("tokenizer.json");
    println!("loading NomicTextEmbedder (mary) from {source_snapshot:?}...");
    let device: WgpuDevice = Default::default();
    let nomic = load_nomic_text_from_files(&weights, &tokenizer, device.clone())
        .expect("load Nomic text baseline from pinned snapshot");
    assert_eq!(nomic.dim(), NOMIC_TEXT_DIM);

    let rust_texts: Vec<Vec<f32>> = texts()
        .iter()
        .map(|t| nomic.embed_document(t).expect("embed_document"))
        .collect();

    let mut pass = true;
    println!("\n=== PARITY (cosine rust-vs-reference, must be > 0.99) ===");
    let labels = ["short   ", "sentence", "long    "];
    for (i, t) in texts().iter().enumerate() {
        let c = cosine(&rust_texts[i], &hf_texts[i]);
        let ntok = nomic.token_count("search_document: ", t);
        println!(
            "  text[{i}] {} ({ntok:>3} tok): {c:.6}  {}",
            labels[i],
            mark(c > 0.99)
        );
        pass &= c > 0.99;
    }

    // ---- NATIVE COLLECTION ROUND-TRIP ----
    println!("\n=== NATIVE ROUND-TRIP (cosine collection-vs-HF, must be ~1.0) ===");
    let pile_path =
        std::env::temp_dir().join(format!("mary_embed_nomic_{}.pile", std::process::id()));
    let _ = std::fs::remove_file(&pile_path);
    std::fs::File::create(&pile_path).expect("create fresh Nomic text collection pile");
    eprintln!("[nomic] publishing {source_snapshot:?} → {pile_path:?} ...");
    let mut pile = Pile::open(&pile_path).expect("open fresh Nomic text collection pile");
    let signing_key = SigningKey::from_bytes(&[0x4e; 32]);
    let publication = native_embedding_collection::publish_embedding_candidate(
        &mut pile,
        &signing_key,
        &weights,
        mary::formats::WeightFormat::Safetensors,
        Some(&tokenizer),
        MODEL_ID,
        EmbeddingArchitecture::NomicTextV15,
    );
    let close = pile.close();
    publication.expect("publish native Nomic text cohort");
    close.expect("close native Nomic text collection pile");
    let pile_size = std::fs::metadata(&pile_path).unwrap().len();
    eprintln!(
        "[nomic] pile is {} bytes ({:.3} GiB).",
        pile_size,
        gib(pile_size)
    );

    let (_, snapshot) = mary::model_collection::load_sole_model_collection_local_latest(&pile_path)
        .expect("fresh-load native Nomic text collection");
    let keymap = mary::selection::load_keymap_from_graph(
        snapshot.facts(),
        snapshot.store(),
        ModelSelector::Source {
            source: MODEL_ID,
            quantization: mary::persist::QUANTIZATION_NATIVE,
        },
    )
    .expect("select native Nomic text weights");
    let native_tokenizer = mary::selection::load_tokenizer_from_graph(
        snapshot.facts(),
        snapshot.store(),
        TokenizerSelector::Name(MODEL_ID),
    )
    .expect("select native Nomic text tokenizer");
    let from_native = nomic_text_from_parts(keymap, native_tokenizer, device.clone())
        .expect("build Nomic text embedder from native snapshot");
    drop(snapshot);
    let native_vec = from_native
        .embed_document(SENTENCE)
        .expect("native collection embed");
    let c_native = cosine(&native_vec, &rust_texts[1]);
    let native_ok = c_native > 0.999999;
    println!(
        "  cos(collection, HF) [sentence] = {c_native:.8}  {}",
        mark(native_ok)
    );
    pass &= native_ok;
    let _ = std::fs::remove_file(&pile_path);

    println!("\n=== {} ===", if pass { "PASS" } else { "FAIL" });
    if !pass {
        std::process::exit(1);
    }
}
