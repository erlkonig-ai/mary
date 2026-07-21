//! De-risk the SmolVLA retarget, step 2: prove the *optimizer loop* learns, not
//! just that one gradient flows (smolvla_grad_test). Loads the real expert head
//! (action_out_proj) as a trainable leaf, fixes a synthetic (input, target)
//! pair, and runs manual SGD under Autodiff<Metal> until the MSE collapses — the
//! end-to-end finetune mechanic (forward → loss → backward → update), minus the
//! real demos. If the loss drives to ~0, the only thing T3 still needs is data.
//!
//!   cargo run --release --features smolvla --bin smolvla_overfit
//!
//! The expert head is overparameterized for a single example (32×720 weights vs
//! 50×32 targets), so an exact fit is reachable — loss → ~0 is the pass signal.

use burn::prelude::*;
use burn::tensor::Distribution;
use mary::nn::backend::BTrain as B;
use mary::nn::weight_loader::{SingleFileLoader, WeightLoader};
use std::path::Path;

const CKPT: &str = concat!(env!("HOME"), "/.cache/huggingface/hub/models--lerobot--smolvla_base/snapshots/c83c3163b8ca9b7e67c509fffd9121e66cb96205/model.safetensors");

fn main() {
    let dev = Default::default();
    let loader = WeightLoader::SingleFile(SingleFileLoader::new(Path::new(CKPT)));

    // The real expert head (720 -> 32), lifted to trainable leaves.
    let mut w = loader.load_tensor::<B, 2>("model.action_out_proj.weight", &dev).require_grad(); // [32,720]
    let mut b = loader.load_tensor::<B, 1>("model.action_out_proj.bias", &dev).require_grad(); // [32]

    // One fixed synthetic demo: a frozen 50-step suffix-out, regress to a frozen
    // target action chunk. Detached so they are constants, not graph leaves.
    let x = Tensor::<B, 3>::random([1, 50, 720], Distribution::Normal(0.0, 1.0), &dev).detach();
    let target = Tensor::<B, 3>::random([1, 50, 32], Distribution::Normal(0.0, 1.0), &dev).detach();

    let lr = 0.05;
    let steps = 300;
    let mut first = 0f32;
    for step in 0..=steps {
        let v = x.clone().matmul(w.clone().transpose().unsqueeze()).add(b.clone().unsqueeze()); // [1,50,32]
        let loss = (v - target.clone()).powf_scalar(2.0).mean();
        let loss_val: f32 = loss.clone().into_scalar().elem();
        if step == 0 { first = loss_val; }
        if step % 30 == 0 {
            println!("step {step:>3}  MSE = {loss_val:.6}");
        }

        let grads = loss.backward();
        let gw = w.grad(&grads).expect("weight gradient");
        let gb = b.grad(&grads).expect("bias gradient");

        // Manual SGD: update on the inner backend, re-lift as fresh trainable
        // leaves so the autodiff graph does not accumulate across steps.
        w = Tensor::from_inner(w.inner() - gw * lr).require_grad();
        b = Tensor::from_inner(b.inner() - gb * lr).require_grad();
    }

    let final_v = x.matmul(w.clone().transpose().unsqueeze()).add(b.unsqueeze());
    let final_loss: f32 = (final_v - target).powf_scalar(2.0).mean().into_scalar().elem();
    println!("\nMSE {first:.4} -> {final_loss:.6}  ({:.0}x reduction over {steps} SGD steps)", first / final_loss.max(1e-9));
    assert!(final_loss < first * 0.01, "loss did not collapse — optimizer loop broken");
    println!("✓ the finetune loop learns: forward → loss → backward → SGD drives the real expert head to fit. T3 mechanics are de-risked; only demos remain.");
}
