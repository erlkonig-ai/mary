//! SmolVLA → mary pile → SmolVLA round-trip. Ingest the smolvla_base
//! safetensors into a pile (each tensor a content-addressed leaf in a module),
//! materialize the model back out, run the full image→action pipeline from the
//! pile-loaded weights, and assert parity against the PyTorch golden.
//!
//!   cargo run --release --features smolvla --bin smolvla_pile_test
//!
//! This is the "everything is a trible" proof: the model is stored as facts +
//! mmap-able f32 blobs, not an opaque file.

use burn::prelude::*;
use burn::tensor::TensorData;
use mary::ingest::{load_keymap, save_safetensors};
use mary::models::smolvla::pipeline::SmolVla;
use mary::nn::backend::{B, WgpuDevice};
use mary::nn::npy;
use mary::nn::weight_loader::{WeightLoader, read_safetensors_file};
use std::path::Path;
use triblespace::prelude::*;

const CKPT: &str = concat!(
    env!("HOME"),
    "/.cache/huggingface/hub/models--lerobot--smolvla_base/snapshots/c83c3163b8ca9b7e67c509fffd9121e66cb96205/model.safetensors"
);
const G: &str = "/tmp/codex_outputs/smolvla_probe";

fn loadt<const D: usize>(rel: &str, dev: &WgpuDevice) -> Tensor<B, D> {
    let (d, s) = npy::load_npy(&Path::new(G).join(rel)).unwrap();
    Tensor::<B, D>::from_data(TensorData::new(d, s), dev)
}

fn main() {
    let dev: WgpuDevice = Default::default();

    // 1. ingest safetensors → pile
    let bytes = read_safetensors_file(Path::new(CKPT));
    let mut blobs = MemoryBlobStore::new();
    let frag = save_safetensors(
        &bytes,
        "smolvla_base",
        &mut blobs,
        mary::ingest::LeafDtype::F32,
    )
    .expect("ingest");
    let model_id = frag.root().expect("model root");
    let mut tribles = TribleSet::new();
    tribles += frag;
    eprintln!("ingested → {} tribles", tribles.len());

    // 2. materialize back out of the pile
    let reader = SnapshotSource::snapshot(&mut blobs).expect("reader");
    let keymap = load_keymap(&tribles, &reader, model_id);
    eprintln!("materialized {} tensors from pile", keymap.len());
    let loader = WeightLoader::Pile(keymap);

    // 3. assemble SmolVLA from the pile-loaded weights
    let model = SmolVla::<B>::load(&loader, &dev);

    // 4. full image→action pipeline — self-contained (masks/positions computed
    //    in Rust; only raw observation tensors come from disk)
    let image = loadt::<4>("inputs/images.npy", &dev);
    let lang_ids = loadt::<2>("inputs/lang_tokens_f32.npy", &dev).int();
    let state = loadt::<3>("inputs/state_3d.npy", &dev);
    let (ck, cv) = model.perceive(image, lang_ids, state, &dev);

    let noise = loadt::<3>("inputs/noise.npy", &dev);
    let actions = model.act(ck, cv, noise, &dev);

    // 5. parity vs the PyTorch golden
    let got = actions.into_data().to_vec::<f32>().unwrap();
    let gold = npy::load_npy(&Path::new(G).join("golden/actions_final.npy"))
        .unwrap()
        .0;
    let (mut dot, mut na, mut nb, mut maxabs) = (0f64, 0f64, 0f64, 0f64);
    for (x, y) in got.iter().zip(&gold) {
        let (x, y) = (*x as f64, *y as f64);
        dot += x * y;
        na += x * x;
        nb += y * y;
        maxabs = maxabs.max((x - y).abs());
    }
    let cos = dot / (na.sqrt() * nb.sqrt());
    println!("pile-loaded SmolVLA → actions: cos={cos:.8}  max|Δ|={maxabs:.3e}");
    assert!(
        cos > 0.9999 && maxabs < 1e-3,
        "pile load diverges from golden"
    );
    println!("✓ SmolVLA round-trips through the mary pile — end-to-end parity exact.");
}
