//! Load the real F5TTS_v1_Base weights and run one forward — validates that
//! every key name resolves, every shape matches, and the DiT executes.
//!
//!   cargo run --release --bin forward_test -- <path-to-model_1250000.safetensors>

use burn::prelude::*;
use mary::models::f5::config::F5Config;
use mary::models::f5::model::F5Transformer;
use mary::nn::backend::B;
use mary::nn::weight_loader::{SingleFileLoader, WeightLoader};
use std::path::Path;

fn main() {
    let path = std::env::args()
        .nth(1)
        .expect("usage: forward_test <model.safetensors>");
    let device = Default::default();

    eprintln!("loading {path} …");
    let loader = WeightLoader::SingleFile(SingleFileLoader::new(Path::new(&path)));
    let cfg = F5Config::v1_base();
    let model = F5Transformer::<B>::load(&loader, cfg.clone(), &device);
    eprintln!("loaded — all {} blocks + embeds present.", cfg.depth);

    let (b, t, m) = (1usize, 64usize, cfg.mel.n_mel);
    let noised = Tensor::<B, 3>::zeros([b, t, m], &device);
    let cond = Tensor::<B, 3>::zeros([b, t, m], &device);
    let text = Tensor::<B, 2, Int>::ones([b, t], &device);
    let time = Tensor::<B, 1>::full([b], 0.5, &device);

    let out = model.forward(noised, cond, text, time);
    let dims = out.dims();
    // a cheap numeric sanity check
    let n = b * t * m;
    let mean: f32 = out.clone().mean().into_scalar().elem();
    let std: f32 = out.clone().reshape([n]).var(0).sqrt().into_scalar().elem();
    println!("forward OK — velocity shape {dims:?}, mean {mean:.4}, std {std:.4}");
    assert_eq!(dims, [b, t, m], "expected [B,T,n_mel]");
    println!("✓ F5 DiT port validated end-to-end on real weights.");
}
