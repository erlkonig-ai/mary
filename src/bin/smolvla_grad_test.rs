//! De-risk the SmolVLA retarget: confirm Burn autodiff flows gradients into the
//! (tensor-based, not Module/Param) expert weights. Loads a trainable projection
//! under Autodiff<Metal>, runs a flow-matching-style MSE forward, backprops, and
//! checks the weight received a non-zero gradient — the one unknown in the
//! retarget plan (compass d37b6067). If this is green, the fine-tune is just the
//! loss + optimizer loop over real demos.
//!
//!   cargo run --release --features smolvla --bin smolvla_grad_test

use burn::prelude::*;
use burn::tensor::Distribution;
use burn::tensor::backend::AutodiffBackend;
use mary::nn::backend::BTrain as B;
use mary::nn::weight_loader::{SingleFileLoader, WeightLoader};
use std::path::Path;

const CKPT: &str = concat!(
    env!("HOME"),
    "/.cache/huggingface/hub/models--lerobot--smolvla_base/snapshots/c83c3163b8ca9b7e67c509fffd9121e66cb96205/model.safetensors"
);

fn main() {
    let dev = Default::default();
    let loader = WeightLoader::SingleFile(SingleFileLoader::new(Path::new(CKPT)));

    // action_out_proj: the expert head (720 -> 32), a trainable leaf
    let w = loader
        .load_tensor::<B, 2>("model.action_out_proj.weight", &dev)
        .require_grad(); // [32,720]
    let bias = loader
        .load_tensor::<B, 1>("model.action_out_proj.bias", &dev)
        .require_grad(); // [32]

    // a flow-matching-style step: predict v from a noised suffix-out, regress to u_t
    let x = Tensor::<B, 3>::random([1, 50, 720], Distribution::Normal(0.0, 1.0), &dev);
    let v = x
        .matmul(w.clone().transpose().unsqueeze())
        .add(bias.clone().unsqueeze()); // [1,50,32]
    let u = Tensor::<B, 3>::random([1, 50, 32], Distribution::Normal(0.0, 1.0), &dev);
    let loss = (v - u).powf_scalar(2.0).mean();
    let loss_val: f32 = loss.clone().into_scalar().elem();

    let grads = loss.backward();
    let gw = w.grad(&grads).expect("weight gradient");
    let gb = bias.grad(&grads).expect("bias gradient");
    let gw_norm: f32 = gw.powf_scalar(2.0).sum().sqrt().into_scalar().elem();
    let gb_norm: f32 = gb.powf_scalar(2.0).sum().sqrt().into_scalar().elem();

    println!("loss = {loss_val:.4}");
    println!(
        "|grad action_out_proj.weight| = {gw_norm:.4}  (shape {:?})",
        w.dims()
    );
    println!("|grad action_out_proj.bias|   = {gb_norm:.4}");
    assert!(
        gw_norm > 0.0 && gb_norm > 0.0,
        "no gradient flowed — autodiff path broken"
    );
    println!(
        "✓ gradients flow into the expert weights under Autodiff<Metal> — retarget autodiff path is viable."
    );
}
