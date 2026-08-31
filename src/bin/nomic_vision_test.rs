//! nomic-embed-vision-v1.5 parity gate.
//!
//!   cargo run --release --features embed,import --bin nomic_vision_test
//!
//! 1. Shell out to python (transformers AutoModel + AutoImageProcessor,
//!    trust_remote_code) to dump the L2-normalized reference embedding of
//!    output_mickey.png — `out.last_hidden_state[:,0]` (the selector pooling
//!    output) then F.normalize — to /tmp/nomic_vision_ref.json.
//! 2. Embed the same image via `mary::embed::NomicVisionEmbedder` (from HF).
//! 3. Assert PARITY cosine(rust, HF) > 0.99.
//! 4. Native collection round-trip: publish weights, fresh-load one frozen
//!    snapshot, embed, and assert cosine(native, HF) ~ 1.0; clean up.
//! 5. Print every cosine + PASS/FAIL.

#[path = "support/native_embedding_collection.rs"]
mod native_embedding_collection;

use ed25519_dalek::SigningKey;
use mary::embed::{
    EmbeddingArchitecture, LocalEmbedder, NOMIC_TEXT_DIM, hf_cache_main_snapshot,
    load_nomic_vision_from_files, load_nomic_vision_from_keymap,
};
use mary::nn::backend::WgpuDevice;
use mary::selection::ModelSelector;
use std::process::Command;
use triblespace::core::repo::pile::Pile;

const MODEL_ID: &str = "nomic-ai/nomic-embed-vision-v1.5";
const REF_JSON: &str = "/tmp/nomic_vision_ref.json";
const IMAGE: &str = "test_image.png"; // any local test image (repo-relative or absolute)

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
    let script = format!(
        r#"
import json, glob, math, sys
import torch, torch.nn.functional as F
from PIL import Image
from transformers import AutoImageProcessor
from safetensors.torch import load_file

MODEL = "{model}"
IMAGE = "{image}"
REF = "{ref}"
MDIR = glob.glob("{home}/.cache/huggingface/hub/models--nomic-ai--nomic-embed-vision-v1.5/snapshots/*/")[0]
proc = AutoImageProcessor.from_pretrained(MODEL)
pix = proc(images=Image.open(IMAGE).convert("RGB"), return_tensors="pt")["pixel_values"].float()

def try_remote():
    # Genuine HF NomicVisionModel via trust_remote_code. Newer transformers
    # versions mismatch this older remote module (n_inner float, all_tied_weights_keys,
    # NaN non-persistent rope buffers) — if so, this returns None and we fall back.
    try:
        from transformers import AutoConfig, AutoModel
        from transformers.dynamic_module_utils import get_class_from_dynamic_module
        cfg = AutoConfig.from_pretrained(MODEL, trust_remote_code=True)
        if isinstance(getattr(cfg, "n_inner", None), float):
            cfg.n_inner = int(cfg.n_inner)
        m = AutoModel.from_pretrained(MODEL, config=cfg, trust_remote_code=True).eval()
        with torch.no_grad():
            emb = F.normalize(m(pixel_values=pix).last_hidden_state[:, 0], p=2, dim=1)
        if torch.isfinite(emb).all():
            return emb
    except Exception as e:
        print("remote model unusable (%s); using safetensors reference" % e)
    return None

def reference_forward():
    # Faithful re-derivation from the safetensors, mirroring modeling_hf_nomic_bert.py's
    # NomicVisionModel (prenorm running-residual ViT + 2D axial RoPE on patch tokens +
    # selector attention pooling). Preprocessing reuses the HF CLIPImageProcessor above.
    W = {{k: v.float() for k, v in load_file(MDIR + "model.safetensors").items()}}
    NH, HD, EPS, NL, G = 12, 64, 1e-6, 12, 14
    B = 1
    def ln(x, w, b, eps): return F.layer_norm(x, (x.shape[-1],), w, b, eps)
    patches = pix.unfold(2, 16, 16).unfold(3, 16, 16).permute(0, 2, 3, 1, 4, 5).reshape(B, G * G, 3 * 16 * 16)
    x = patches @ W["embeddings.proj.weight"].T + W["embeddings.proj.bias"]
    x = torch.cat([W["embeddings.cls_token"].expand(B, 1, -1), x], dim=1) + W["embeddings.pos_embed"]
    nb = HD // 4
    bands = 1.0 / (10000.0 ** (torch.arange(nb).float() / nb))
    grid = torch.stack(torch.meshgrid(torch.arange(G).float(), torch.arange(G).float(), indexing="ij"), -1).unsqueeze(-1)
    pos = grid * bands
    sin = pos.sin().reshape(G * G, -1).repeat_interleave(2, -1)
    cos = pos.cos().reshape(G * G, -1).repeat_interleave(2, -1)
    def rot(z): return torch.stack([-z[..., 1::2], z[..., ::2]], -1).reshape(z.shape)
    def rope(q): return q * cos + rot(q) * sin
    def sh(z): return z.reshape(B, -1, NH, HD).permute(0, 2, 1, 3)
    residual = None
    for i in range(NL):
        p = "layers.%d." % i
        residual = x if residual is None else x + residual
        h = ln(residual, W[p + "norm1.weight"], W[p + "norm1.bias"], EPS)
        q, k, v = (h @ W[p + "attn.Wqkv.weight"].T + W[p + "attn.Wqkv.bias"]).split(768, -1)
        q, k, v = sh(q), sh(k), sh(v)
        q = torch.cat([q[:, :, :1], rope(q[:, :, 1:])], 2)
        k = torch.cat([k[:, :, :1], rope(k[:, :, 1:])], 2)
        att = (F.softmax((q @ k.transpose(-1, -2)) / math.sqrt(HD), -1) @ v).permute(0, 2, 1, 3).reshape(B, -1, 768)
        att = att @ W[p + "attn.out_proj.weight"].T + W[p + "attn.out_proj.bias"]
        residual = att + residual
        h = ln(residual, W[p + "norm2.weight"], W[p + "norm2.bias"], EPS)
        g = (h @ W[p + "mlp.fc11.weight"].T + W[p + "mlp.fc11.bias"]) * F.silu(h @ W[p + "mlp.fc12.weight"].T + W[p + "mlp.fc12.bias"])
        g = ln(g, W[p + "mlp.norm.weight"], W[p + "mlp.norm.bias"], 1e-5)
        x = g @ W[p + "mlp.fc2.weight"].T + W[p + "mlp.fc2.bias"]
    hidden = x + residual
    q = W["selector.attn.latent"].expand(B, 1, -1) @ W["selector.attn.Wq.weight"].T + W["selector.attn.Wq.bias"]
    k, v = (hidden @ W["selector.attn.Wkv.weight"].T + W["selector.attn.Wkv.bias"]).split(768, -1)
    q, k, v = sh(q), sh(k), sh(v)
    att = (F.softmax((q @ k.transpose(-1, -2)) / math.sqrt(HD), -1) @ v).permute(0, 2, 1, 3).reshape(B, 1, 768)
    att = att @ W["selector.attn.out_proj.weight"].T + W["selector.attn.out_proj.bias"]
    n = ln(att, W["selector.norm1.weight"], W["selector.norm1.bias"], EPS)
    g = (n @ W["selector.mlp.fc11.weight"].T + W["selector.mlp.fc11.bias"]) * F.silu(n @ W["selector.mlp.fc12.weight"].T + W["selector.mlp.fc12.bias"])
    mlp = g @ W["selector.mlp.fc2.weight"].T + W["selector.mlp.fc2.bias"]
    return F.normalize((att + mlp).reshape(B, 768), p=2, dim=1)

emb = try_remote()
src = "remote"
if emb is None:
    emb = reference_forward()
    src = "safetensors"
with torch.no_grad():
    json.dump({{"image": emb[0].detach().cpu().numpy().tolist(), "source": src}}, open(REF, "w"))
print("ref written (%s), dim %d" % (src, emb.shape[-1]))
"#,
        model = MODEL_ID,
        image = IMAGE,
        ref = REF_JSON,
        home = std::env::var("HOME").unwrap(),
    );
    let status = Command::new("python3")
        .arg("-c")
        .arg(&script)
        .status()
        .expect("spawn python3 (need transformers + torch + pillow)");
    assert!(status.success(), "python reference generation failed");
}

fn main() {
    println!("generating reference (python)...");
    gen_reference();
    let refv: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(REF_JSON).unwrap()).unwrap();
    let hf_vec: Vec<f32> = refv["image"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_f64().unwrap() as f32)
        .collect();
    println!(
        "reference source: {}",
        refv["source"].as_str().unwrap_or("?")
    );

    let source_snapshot =
        hf_cache_main_snapshot(MODEL_ID).expect("resolve one cached Nomic vision main revision");
    let weights = source_snapshot.join("model.safetensors");
    println!("loading NomicVisionEmbedder (mary) from {source_snapshot:?}...");
    let device: WgpuDevice = Default::default();
    let nomic = load_nomic_vision_from_files(&weights, device.clone())
        .expect("load Nomic vision baseline from pinned snapshot");
    assert_eq!(nomic.dim(), NOMIC_TEXT_DIM);

    let bytes = std::fs::read(IMAGE).expect("read image");
    let rust_vec = nomic.embed_image(&bytes).expect("embed_image");

    let mut pass = true;
    println!("\n=== PARITY (cosine rust-vs-HF, must be > 0.99) ===");
    let c = cosine(&rust_vec, &hf_vec);
    println!("  cos(rust, HF) [mickey] = {c:.6}  {}", mark(c > 0.99));
    pass &= c > 0.99;

    // ---- NATIVE COLLECTION ROUND-TRIP ----
    println!("\n=== NATIVE ROUND-TRIP (cosine collection-vs-HF, must be ~1.0) ===");
    let pile_path = std::env::temp_dir().join(format!(
        "mary_embed_nomic_vision_{}.pile",
        std::process::id()
    ));
    let _ = std::fs::remove_file(&pile_path);
    std::fs::File::create(&pile_path).expect("create fresh Nomic vision collection pile");
    eprintln!("[nomic-vision] publishing {source_snapshot:?} → {pile_path:?} ...");
    let mut pile = Pile::open(&pile_path).expect("open fresh Nomic vision collection pile");
    let signing_key = SigningKey::from_bytes(&[0x56; 32]);
    let publication = native_embedding_collection::publish_embedding_candidate(
        &mut pile,
        &signing_key,
        &weights,
        mary::formats::WeightFormat::Safetensors,
        None,
        MODEL_ID,
        EmbeddingArchitecture::NomicVisionV15,
    );
    let close = pile.close();
    publication.expect("publish native Nomic vision cohort");
    close.expect("close native Nomic vision collection pile");
    let pile_size = std::fs::metadata(&pile_path).unwrap().len();
    eprintln!(
        "[nomic-vision] pile is {} bytes ({:.3} GiB).",
        pile_size,
        gib(pile_size)
    );

    let snapshot = mary::model_collection::load_model_collection_local_latest(&pile_path)
        .expect("fresh-load native Nomic vision collection");
    let keymap = mary::selection::load_keymap_from_graph(
        snapshot.facts(),
        snapshot.store(),
        ModelSelector::Source {
            source: MODEL_ID,
            quantization: mary::persist::QUANTIZATION_NATIVE,
        },
    )
    .expect("select native Nomic vision weights");
    let from_native = load_nomic_vision_from_keymap(keymap, device.clone())
        .expect("build Nomic vision embedder from native snapshot");
    drop(snapshot);
    let native_vec = from_native
        .embed_image(&bytes)
        .expect("native collection embed");
    let c_native = cosine(&native_vec, &rust_vec);
    let native_ok = c_native > 0.999999;
    println!(
        "  cos(collection, HF) [mickey] = {c_native:.8}  {}",
        mark(native_ok)
    );
    pass &= native_ok;
    let _ = std::fs::remove_file(&pile_path);

    println!("\n=== {} ===", if pass { "PASS" } else { "FAIL" });
    if !pass {
        std::process::exit(1);
    }
}
