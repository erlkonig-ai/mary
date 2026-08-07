//! nomic-embed-text-v1.5 parity gate.
//!
//!   cargo run --release --features embed --bin nomic_text_test
//!
//! 1. Shell out to python (sentence-transformers, or transformers AutoModel as a
//!    fallback) to dump L2-normalized reference vectors for a short text, a
//!    sentence, and a ~150-word multi-paragraph blob — all with the
//!    "search_document: " prefix — to /tmp/nomic_ref.json.
//! 2. Embed the same texts via `mary::embed::NomicTextEmbedder::embed_document`.
//! 3. Assert PARITY (cosine(rust, ref) > 0.99 per text, incl the long one).
//! 4. Pile round-trip: persist nomic safetensors → temp pile → load from pile,
//!    assert cosine(pile, safetensors) ~ 1.0 for one text, clean up.
//! 5. Print every cosine + token counts + PASS/FAIL.

use mary::embed::{
    load_nomic_text_from_hf, load_nomic_text_from_pile, LocalEmbedder, NOMIC_TEXT_DIM,
};
use mary::nn::backend::WgpuDevice;
use mary::persist::persist_safetensors_to_pile;
use std::path::{Path, PathBuf};
use std::process::Command;

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
    if ok {
        "OK"
    } else {
        "<-- FAIL"
    }
}

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

    println!("loading NomicTextEmbedder (mary)...");
    let device: WgpuDevice = Default::default();
    let nomic = load_nomic_text_from_hf(MODEL_ID, device.clone()).expect("load nomic");
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

    // ---- PILE ROUND-TRIP ----
    println!("\n=== PILE ROUND-TRIP (cosine pile-vs-safetensors, must be ~1.0) ===");
    let weights = hf_cache_resolve(MODEL_ID, "model.safetensors").expect("nomic weights in cache");
    let tokenizer = hf_cache_resolve(MODEL_ID, "tokenizer.json").expect("nomic tokenizer in cache");
    let snapshot_dir = weights.parent().unwrap();
    let pile_path =
        std::env::temp_dir().join(format!("mary_embed_nomic_{}.pile", std::process::id()));
    let _ = std::fs::remove_file(&pile_path);
    eprintln!("[nomic] persisting {snapshot_dir:?} → {pile_path:?} ...");
    persist_safetensors_to_pile(snapshot_dir, &pile_path, mary::ingest::LeafDtype::F32)
        .expect("persist nomic to pile");
    let pile_size = std::fs::metadata(&pile_path).unwrap().len();
    eprintln!(
        "[nomic] pile is {} bytes ({:.3} GiB).",
        pile_size,
        gib(pile_size)
    );

    let from_pile =
        load_nomic_text_from_pile(&pile_path, &tokenizer, device.clone()).expect("load from pile");
    let pile_vec = from_pile.embed_document(SENTENCE).expect("pile embed");
    let c_pile = cosine(&pile_vec, &rust_texts[1]);
    let pile_ok = c_pile > 0.999999;
    println!(
        "  cos(pile, safetensors) [sentence] = {c_pile:.8}  {}",
        mark(pile_ok)
    );
    pass &= pile_ok;
    let _ = std::fs::remove_file(&pile_path);

    println!("\n=== {} ===", if pass { "PASS" } else { "FAIL" });
    if !pass {
        std::process::exit(1);
    }
}
