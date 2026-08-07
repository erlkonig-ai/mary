//! Qwen3-TTS → mary pile → Qwen3-TTS round-trip, validated by forward parity:
//! ingest both checkpoints (talker+predictor+speaker encoder, and the codec)
//! into a content-addressed model graph, materialize them back, and assert the
//! reconstructed components produce identical outputs. The qwen3tts analogue of
//! `flux_pile_test` / `gemma_pile_test`. Uses an in-memory blob store — no
//! on-disk pile is touched.
//!
//!   cargo run --release --features qwen3tts --bin qwen3tts_pile_test

use burn::prelude::*;
use burn::tensor::TensorData;
use mary::ingest::{load_keymap, save_safetensors, LeafDtype};
use mary::models::qwen3tts::codec::CodecDecoder;
use mary::models::qwen3tts::config::*;
use mary::models::qwen3tts::talker::Talker;
use mary::nn::backend::BFused as B;
use mary::nn::npy;
use mary::nn::weight_loader::{read_safetensors_file, SingleFileLoader, WeightLoader};
use std::path::Path;
use triblespace::prelude::*;

const WEIGHTS: &str = "/tmp/qwen3tts-weights/base";
const GOLD: &str = "/tmp/mary-qwen3tts/golden";

/// safetensors bytes → pile → `WeightLoader::Pile`.
fn roundtrip(bytes: &[u8], name: &str) -> WeightLoader {
    let mut blobs = MemoryBlobStore::new();
    let frag = save_safetensors(bytes, name, &mut blobs, LeafDtype::F32).expect("ingest");
    let model_id = frag.root().expect("model root");
    let mut tribles = TribleSet::new();
    tribles += frag;
    eprintln!("{name}: ingested → {} tribles", tribles.len());
    let reader = BlobStore::reader(&mut blobs).expect("reader");
    let keymap = load_keymap(&tribles, &reader, model_id);
    eprintln!("{name}: materialized {} tensors from pile", keymap.len());
    WeightLoader::Pile(keymap)
}

fn main() {
    let dev = Default::default();

    // ── codec first (small): decode the reference codes both ways ──
    let codec_path = Path::new(WEIGHTS).join("speech_tokenizer/model.safetensors");
    let (rc, rcs) = npy::load_npy(&Path::new(GOLD).join("ref_code_f32.npy")).expect("ref codes");
    let frames: Vec<[u32; NUM_CODE_GROUPS]> = (0..rcs[0].min(16))
        .map(|t| {
            let mut f = [0u32; NUM_CODE_GROUPS];
            for q in 0..NUM_CODE_GROUPS {
                f[q] = rc[t * NUM_CODE_GROUPS + q] as u32;
            }
            f
        })
        .collect();

    let st = WeightLoader::SingleFile(SingleFileLoader::new(&codec_path));
    let codec_st = CodecDecoder::<B>::load(&st, &dev);
    let wav_st = codec_st.decode(&frames, &dev);
    drop(codec_st);

    let pile = roundtrip(
        &read_safetensors_file(&codec_path),
        "qwen3-tts-tokenizer-12hz",
    );
    let codec_pile = CodecDecoder::<B>::load(&pile, &dev);
    let wav_pile = codec_pile.decode(&frames, &dev);
    drop(codec_pile);

    let d = wav_st
        .iter()
        .zip(&wav_pile)
        .map(|(a, b)| (a - b).abs())
        .fold(0f32, f32::max);
    println!("codec decode max|Δ| safetensors-vs-pile = {d:.3e}");
    assert_eq!(d, 0.0, "codec pile load diverges");

    // ── talker (the 1.7B): prefill forward on the golden embeds both ways ──
    let base_path = Path::new(WEIGHTS).join("model.safetensors");
    let (ge, ges) =
        npy::load_npy(&Path::new(GOLD).join("prefill_embeds.npy")).expect("golden embeds");
    let embeds = Tensor::<B, 3>::from_data(TensorData::new(ge, ges), &dev);

    let st = WeightLoader::SingleFile(SingleFileLoader::new(&base_path));
    let talker_st = Talker::<B>::load(&st, &dev);
    let mut caches = talker_st.new_caches();
    let h_st = talker_st.forward(embeds.clone(), &mut caches, &dev);
    let l_st = talker_st.logits_last(h_st);
    drop(caches);
    drop(talker_st);
    drop(st);

    let pile = roundtrip(
        &read_safetensors_file(&base_path),
        "qwen3-tts-12hz-1.7b-base",
    );
    let talker_pile = Talker::<B>::load(&pile, &dev);
    let mut caches = talker_pile.new_caches();
    let h_pile = talker_pile.forward(embeds, &mut caches, &dev);
    let l_pile = talker_pile.logits_last(h_pile);

    let d = l_st
        .iter()
        .zip(&l_pile)
        .map(|(a, b)| (a - b).abs())
        .fold(0f32, f32::max);
    println!("talker prefill logits max|Δ| safetensors-vs-pile = {d:.3e}");
    assert_eq!(d, 0.0, "talker pile load diverges");

    println!("✓ Qwen3-TTS round-trips through the mary pile — bit-identical.");
}
