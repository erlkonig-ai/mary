//! Flux.2 transformer → mary pile → Flux.2 transformer round-trip, validated by
//! forward-parity: load the DiT from the FLUX.2-klein-4B safetensors and from a
//! mary pile (ingested via `mary::ingest`), run the same fixed input through
//! both, assert the output is bit-identical. The flux analogue of `pile_test`.
//!
//!   cargo run --release --features flux --bin flux_pile_test [-- <transformer_dir>]

use burn::prelude::*;
use mary::ingest::{load_keymap, save_safetensors};
use mary::models::flux::transformer::Flux2Transformer2DModel;
use mary::models::flux::transformer::config::Flux2TransformerConfig;
use mary::nn::backend::B;
use mary::nn::weight_loader::{WeightLoader, read_safetensors_file};
use std::path::PathBuf;
use triblespace::prelude::*;

/// Resolve the FLUX.2-klein-4B transformer dir: CLI arg, or glob the HF cache.
fn transformer_dir() -> PathBuf {
    if let Some(arg) = std::env::args().nth(1) {
        return PathBuf::from(arg);
    }
    let home = std::env::var("HOME").expect("HOME");
    let snapshots = PathBuf::from(home)
        .join(".cache/huggingface/hub/models--black-forest-labs--FLUX.2-klein-4B/snapshots");
    let snap = std::fs::read_dir(&snapshots)
        .unwrap_or_else(|e| panic!("no FLUX.2-klein snapshots at {}: {e}", snapshots.display()))
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .find(|p| p.join("transformer").join("config.json").exists())
        .expect("no snapshot with transformer/config.json");
    snap.join("transformer")
}

fn main() {
    let dir = transformer_dir();
    let device = Default::default();
    let cfg = Flux2TransformerConfig::load(&dir.join("config.json"));
    eprintln!(
        "transformer dir: {} ({} double + {} single blocks)",
        dir.display(),
        cfg.num_layers,
        cfg.num_single_layers
    );

    // 1. baseline: load the DiT straight from safetensors
    let st_loader = WeightLoader::from_dir(&dir);
    let model_st = Flux2Transformer2DModel::<B>::load(&st_loader, cfg.clone(), &device);
    eprintln!("loaded from safetensors");

    // 2. ingest the transformer safetensors → pile (each tensor a content-addressed
    //    leaf inside a module, gathered under one model entity). The transformer is
    //    a single-file safetensors, so one ingest covers all its tensors.
    let st_path = dir.join("diffusion_pytorch_model.safetensors");
    let bytes = read_safetensors_file(&st_path);
    let mut blobs = MemoryBlobStore::new();
    let frag = save_safetensors(
        &bytes,
        "FLUX.2-klein-4B-transformer",
        &mut blobs,
        mary::ingest::LeafDtype::F32,
    )
    .expect("ingest");
    let model_id = frag.root().expect("model root");
    let mut tribles = TribleSet::new();
    tribles += frag;
    eprintln!("ingested → {} tribles", tribles.len());

    // 3. materialize the model out of the pile → Pile loader keymap (all tensors)
    let reader = SnapshotSource::snapshot(&mut blobs).expect("reader");
    let keymap = load_keymap(&tribles, &reader, model_id);
    eprintln!("materialized {} tensors from pile", keymap.len());
    let pile_loader = WeightLoader::Pile(keymap);

    // 4. reconstruct the DiT from the pile (reuses Flux2Transformer2DModel::load)
    let model_pile = Flux2Transformer2DModel::<B>::load(&pile_loader, cfg.clone(), &device);
    eprintln!("reconstructed from pile");

    // 5. forward-parity on a fixed zero input.
    //    forward(hidden_states[B,S_img,in_ch], encoder_hidden_states[B,S_txt,joint_attn_dim],
    //            timestep[B], guidance, img_ids[S_img,4], txt_ids[S_txt,4], device)
    let (bn, s_img, s_txt) = (1usize, 4usize, 4usize);
    let hidden = Tensor::<B, 3>::zeros([bn, s_img, cfg.in_channels], &device);
    let enc = Tensor::<B, 3>::zeros([bn, s_txt, cfg.joint_attention_dim], &device);
    let timestep = Tensor::<B, 1>::full([bn], 0.5, &device);
    let img_ids = Tensor::<B, 2>::zeros([s_img, 4], &device);
    let txt_ids = Tensor::<B, 2>::zeros([s_txt, 4], &device);

    let v_st = model_st.forward(
        hidden.clone(),
        enc.clone(),
        timestep.clone(),
        None,
        img_ids.clone(),
        txt_ids.clone(),
        &device,
    );
    let v_pile = model_pile.forward(hidden, enc, timestep, None, img_ids, txt_ids, &device);

    let d: f32 = (v_st - v_pile).abs().max().into_scalar().elem();
    println!("forward max|Δ| safetensors-vs-pile = {d:.3e}");
    assert!(d == 0.0, "pile load diverges (max|Δ| = {d:.3e})");
    println!("✓ FLUX.2 transformer round-trips through the mary pile — bit-identical.");
}
