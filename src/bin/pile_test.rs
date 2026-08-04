//! F5 → mary pile → F5 round-trip, validated by forward-parity: load the model
//! from safetensors and from the pile, run the same input through both, assert
//! the velocity field is identical.
//!
//!   cargo run --release --bin pile_test -- <F5TTS_v1_Base.safetensors>

use burn::prelude::*;
use mary::ingest::{load_keymap, save_safetensors};
use mary::models::f5::config::F5Config;
use mary::models::f5::model::F5Transformer;
use mary::nn::backend::B;
use mary::nn::weight_loader::{read_safetensors_file, SingleFileLoader, WeightLoader};
use std::path::Path;
use triblespace::prelude::*;

fn main() {
    let path = std::env::args()
        .nth(1)
        .expect("usage: pile_test <f5.safetensors>");
    let device = Default::default();
    let cfg = F5Config::v1_base();

    // 1. baseline: load from safetensors
    let st_loader = WeightLoader::SingleFile(SingleFileLoader::new(Path::new(&path)));
    let model_st = F5Transformer::<B>::load(&st_loader, cfg.clone(), &device);
    eprintln!("loaded from safetensors");

    // 2. ingest safetensors → pile (each tensor a content-addressed leaf in a module)
    let bytes = read_safetensors_file(Path::new(&path));
    let mut blobs = MemoryBlobStore::new();
    let frag = save_safetensors(
        &bytes,
        "F5TTS_v1_Base",
        &mut blobs,
        mary::ingest::LeafDtype::F32,
    )
    .expect("ingest");
    let model_id = frag.root().expect("model root");
    let mut tribles = TribleSet::new();
    tribles += frag;
    eprintln!("ingested → {} tribles", tribles.len());

    // 3. materialize the model out of the pile → Pile loader
    let reader = BlobStore::reader(&mut blobs).expect("reader");
    let keymap = load_keymap(&tribles, &reader, model_id);
    eprintln!("materialized {} tensors from pile", keymap.len());
    let pile_loader = WeightLoader::Pile(keymap);

    // 4. reconstruct from the pile (reuses F5Transformer::load unchanged)
    let model_pile = F5Transformer::<B>::load(&pile_loader, cfg.clone(), &device);
    eprintln!("reconstructed from pile");

    // 5. forward-parity on a fixed input
    let (bn, t, m) = (1usize, 64usize, cfg.mel.n_mel);
    let noised = Tensor::<B, 3>::zeros([bn, t, m], &device);
    let cond = Tensor::<B, 3>::zeros([bn, t, m], &device);
    let text = Tensor::<B, 2, Int>::ones([bn, t], &device);
    let time = Tensor::<B, 1>::full([bn], 0.5, &device);
    let v_st = model_st.forward(noised.clone(), cond.clone(), text.clone(), time.clone());
    let v_pile = model_pile.forward(noised, cond, text, time);
    let d: f32 = (v_st - v_pile).abs().max().into_scalar().elem();
    println!("forward max|Δ| safetensors-vs-pile = {d:.3e}");
    assert!(d < 1e-5, "pile load diverges");
    println!("✓ F5 round-trips through the mary pile — forward-parity exact.");
}
