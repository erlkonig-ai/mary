//! End-to-end F5 speech in Burn: load the reference mel + tokenized text + fixed
//! noise from scripts/prep_infer.py, run CFM (32 steps) → slice generated mel →
//! Vocos → 24 kHz waveform, and write it for playback / parity vs the reference.
//!
//!   python3 scripts/prep_infer.py <model.safetensors>
//!   cargo run --release --bin speak -- <model.safetensors>

use burn::prelude::*;
use burn::tensor::TensorData;
use mary::models::f5::cfm;
use mary::models::f5::config::{CfmConfig, F5Config};
use mary::models::f5::model::F5Transformer;
use mary::models::f5::vocos::Vocos;
use mary::nn::backend::B;
use mary::nn::npy;
use mary::nn::weight_loader::{SingleFileLoader, WeightLoader};
use std::path::{Path, PathBuf};

fn main() {
    let path = std::env::args()
        .nth(1)
        .expect("usage: speak <model.safetensors>");
    let device = Default::default();
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let dir = root.join("probes").join("infer");

    let meta = std::fs::read_to_string(dir.join("meta.txt")).unwrap();
    let mut it = meta.split_whitespace();
    let ref_len: usize = it.next().unwrap().parse().unwrap();
    let duration: usize = it.next().unwrap().parse().unwrap();

    let ld = |n: &str| npy::load_npy(&dir.join(format!("{n}.npy"))).unwrap();
    let (rd, rs) = ld("ref_mel"); // [1, ref_len, 100]
    let (yd, ys) = ld("y0"); // [1, duration, 100]
    let ref_mel = Tensor::<B, 3>::from_data(TensorData::new(rd, rs), &device);
    let y0 = Tensor::<B, 3>::from_data(TensorData::new(yd, ys), &device);

    // tokenize ref+gen text in Rust (self-contained; matches prep_infer.py)
    const REF: &str = "Some call me nature, others call me mother nature.";
    const GEN: &str = "I don't really care what you call me. I am the voice of the body now.";
    let text_raw = mary::models::f5::tokenizer::Tokenizer::new()
        .encode_tensor::<B>(&format!("{REF} {GEN}"), &device);

    // cond = ref_mel padded with zeros to `duration`
    let gen = duration - ref_len;
    let cond = Tensor::cat(
        vec![ref_mel, Tensor::<B, 3>::zeros([1, gen, 100], &device)],
        1,
    );
    // text padded to `duration` with -1 (→ +1 = 0 = filler, matches F5 padding)
    let n_chars = text_raw.dims()[1];
    let pad_text = Tensor::<B, 2, Int>::zeros([1, duration - n_chars], &device) - 1;
    let text = Tensor::cat(vec![text_raw, pad_text], 1);

    let loader = WeightLoader::SingleFile(SingleFileLoader::new(Path::new(&path)));
    let model = F5Transformer::<B>::load(&loader, F5Config::v1_base(), &device);
    let vweights = root.join("weights").join("vocos.safetensors");
    let vocos = Vocos::<B>::load(
        &WeightLoader::SingleFile(SingleFileLoader::new(&vweights)),
        &device,
    );

    // single-forward parity check on the real input (matches prep's v0_cond)
    {
        let v0 = model.forward(
            y0.clone(),
            cond.clone(),
            text.clone(),
            Tensor::<B, 1>::zeros([1], &device),
        );
        let vd = v0.into_data();
        let sh = vd.shape.to_vec();
        npy::save_npy(
            &dir.join("v0_cond_burn.npy"),
            &vd.to_vec::<f32>().unwrap(),
            &sh,
        )
        .unwrap();
    }

    let cfg = CfmConfig {
        nfe: 32,
        sway_coef: -1.0,
        cfg_strength: 2.0,
    };
    eprintln!("sampling {duration} frames, {} steps …", cfg.nfe);
    let sampled = cfm::integrate(&model, y0, cond, text, &cfg, &device); // [1,dur,100]
    {
        let sd = sampled.clone().into_data();
        let sh = sd.shape.to_vec();
        npy::save_npy(
            &dir.join("sampled_mel_burn.npy"),
            &sd.to_vec::<f32>().unwrap(),
            &sh,
        )
        .unwrap();
    }
    let gen_mel = sampled
        .slice([0..1, ref_len..duration, 0..100])
        .swap_dims(1, 2); // [1,100,gen]
    let wave = vocos.forward(gen_mel); // [1, (gen-1)*256]

    let data = wave.into_data();
    let shape = data.shape.to_vec();
    let v = data.to_vec::<f32>().unwrap();
    npy::save_npy(&dir.join("burn_wave.npy"), &v, &shape).unwrap();
    println!(
        "✓ burn wave {shape:?} → {}",
        dir.join("burn_wave.npy").display()
    );
}
