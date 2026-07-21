//! Run SmolVLA on a real image + a natural-language instruction, all in Rust:
//! decode the image, preprocess, tokenize, and roll out a 50-step action chunk.
//!
//!   cargo run --release --features smolvla --bin smolvla_infer -- <image> "<instruction>"
//!
//! Weights come ONLY from a persisted pile (write one with `smolvla_persist`);
//! `SMOLVLA_PILE` overrides the default path. The tokenizer stays a small file.
//!
//! NOTE: with the stock smolvla_base weights the action numbers are in the
//! SO100/SO101 training space — meaningful Reachy poses need the policy
//! retargeted on Reachy demonstrations. This proves the deployable engine:
//! image + sentence → motor plan, no Python in the loop.

use burn::prelude::*;
use burn::tensor::{Distribution, TensorData};
use mary::models::smolvla::pipeline::{preprocess_image, SmolVla};
use mary::nn::backend::{WgpuDevice, B};
use mary::nn::weight_loader::WeightLoader;
use std::path::Path;

const PILE: &str = "models/smolvla.pile";
const TOKENIZER: &str = concat!(env!("HOME"), "/.cache/huggingface/hub/models--HuggingFaceTB--SmolVLM2-500M-Video-Instruct/snapshots/7b375e1b73b11138ff12fe22c8f2822d8fe03467/tokenizer.json");
const JOINTS: [&str; 9] = ["head_x", "head_y", "head_z", "roll", "pitch", "yaw", "body_yaw", "ant_l", "ant_r"];

fn main() {
    let img_path = std::env::args().nth(1).expect("usage: smolvla_infer <image> <instruction>");
    let instruction = std::env::args().nth(2).expect("usage: smolvla_infer <image> <instruction>");
    let dev: WgpuDevice = Default::default();

    // image → [1,3,H,W] in [0,1] → resize-with-pad + normalize
    let rgb = image::open(&img_path).expect("open image").to_rgb8();
    let (w, h) = rgb.dimensions();
    let (w, h) = (w as usize, h as usize);
    let mut data = vec![0f32; 3 * h * w];
    for (x, y, px) in rgb.enumerate_pixels() {
        for c in 0..3 {
            data[c * h * w + (y as usize) * w + (x as usize)] = px[c] as f32 / 255.0;
        }
    }
    let raw = Tensor::<B, 4>::from_data(TensorData::new(data, [1, 3, h, w]), &dev);
    let image = preprocess_image::<B>(raw, 512);

    // instruction → token ids
    let tk = tokenizers::Tokenizer::from_file(TOKENIZER).expect("tokenizer");
    let enc = tk.encode(instruction.as_str(), false).expect("encode");
    let ids: Vec<f32> = enc.get_ids().iter().map(|&i| i as f32).collect();
    let nt = ids.len();
    let lang_ids = Tensor::<B, 2>::from_data(TensorData::new(ids, [1, nt]), &dev).int();

    // neutral state, fresh flow noise
    let state = Tensor::<B, 3>::zeros([1, 1, 32], &dev);
    let noise = Tensor::<B, 3>::random([1, 50, 32], Distribution::Normal(0.0, 1.0), &dev);

    // perceive → act — weights from the durable pile
    let pile = std::env::var("SMOLVLA_PILE").unwrap_or_else(|_| PILE.to_string());
    let loader = WeightLoader::from_pile(Path::new(&pile))
        .unwrap_or_else(|e| panic!("load smolvla pile {pile}: {e:?}"));
    let model = SmolVla::<B>::load(&loader, &dev);
    let (ck, cv) = model.perceive(image, lang_ids, state, &dev);
    let actions = model.act(ck, cv, noise, &dev); // [1,50,32]

    // report: the 9 expressive joints over the chunk (first / mid / last pose)
    let a = actions.into_data().to_vec::<f32>().unwrap(); // 50×32 row-major
    println!("\nSmolVLA · \"{instruction}\" · {img_path}");
    println!("action chunk: 50 poses × 32 dims (expressive 9 shown)\n");
    println!("  {:>9} {:>9} {:>9} {:>9}", "joint", "pose[0]", "pose[24]", "pose[49]");
    for (j, name) in JOINTS.iter().enumerate() {
        let at = |p: usize| a[p * 32 + j];
        println!("  {name:>9} {:>9.4} {:>9.4} {:>9.4}", at(0), at(24), at(49));
    }
    println!("\n(stock smolvla_base — retarget on Reachy demos for real poses)");
}
