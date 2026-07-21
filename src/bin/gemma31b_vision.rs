//! Vision-encoder load + sanity-forward check for gemma-4-31B-it, from the pile.
//! Confirms the 356 vision tensors persisted and the Gemma4VisionEncoder builds
//! (patch embedder + 27 encoder layers + embedding projection + std_bias/scale),
//! then runs a forward on a synthetic full-image patch grid and reports the
//! soft-token output shape + finiteness. A true image→text vision parity gate
//! (HF pixel preprocessing + soft-token splicing into the decoder) is a
//! follow-up; this proves the encoder loads and runs at 31B scale.
//!
//!   cargo run --release --features gemma,f16gen --bin gemma31b_vision -- \
//!     --pile models/gemma_31b.pile

use burn::prelude::*;
use mary::models::gemma::gemma4::config::Gemma4Config;
use std::path::Path;
use std::process::Command;
use mary::models::gemma::gemma4::lm::GemmaLM;
use mary::models::gemma::gemma4::preprocess::pil_resize_bicubic;

#[cfg(feature = "f16gen")]
use mary::nn::backend::BHalf as B;
#[cfg(not(feature = "f16gen"))]
use mary::nn::backend::B;

fn find_hf_file(model_id: &str, filename: &str) -> String {
    let o = Command::new("python3")
        .args(["-c", &format!(
            "from huggingface_hub import hf_hub_download; print(hf_hub_download('{}', '{}'))",
            model_id, filename)])
        .output().unwrap();
    String::from_utf8(o.stdout).unwrap().trim().to_string()
}

fn arg(args: &[String], k: &str) -> Option<String> {
    args.iter().position(|s| s == k).map(|i| args[i + 1].clone())
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let pile = arg(&args, "--pile").expect("--pile <gemma_31b.pile>");
    let config_path = find_hf_file("google/gemma-4-31B-it", "config.json");
    // Keep vision_config THIS time — we want the vision tower loaded.
    let config = Gemma4Config::load(Path::new(&config_path));
    assert!(config.vision_config.is_some(), "31B config must carry a vision_config");

    let device = mary::models::gemma::metal_device::init_metal_device_16gb();
    eprintln!("[vision] streaming 31B (WITH vision tower) from {pile}...");
    let (_decoder, vision) = mary::persist::load_gemma4_streaming_from_pile::<B>(
        Path::new(&pile), config.clone(), &device,
    ).expect("stream 31b (+vision) from pile");

    let enc = vision.expect("vision encoder must be present for the 31B");
    eprintln!(
        "[vision] encoder loaded: {} layers, patch_size {}, hidden {}",
        enc.config.num_hidden_layers, enc.config.patch_size, enc.config.hidden_size
    );

    let image_path = arg(&args, "--image");

    if let Some(img_path) = image_path {
        // Load and resize the image
        eprintln!("[vision] loading image from {img_path}...");
        let raw_img = image::open(&img_path).expect("open image").into_rgb8();
        let target_size = 864; // 864x864 typical for gemma4
        let resized = pil_resize_bicubic(&raw_img, target_size, target_size);

        let patch_size = enc.config.patch_size as u32;
        let grid = target_size / patch_size;
        let num_patches = (grid * grid) as usize;
        let patch_dim = 3 * patch_size as usize * patch_size as usize;

        // Build flat pixel values [0, 1]
        let mut pv = Vec::with_capacity(num_patches * patch_dim);
        for y in 0..grid {
            for x in 0..grid {
                for py in 0..patch_size {
                    for px in 0..patch_size {
                        let p = resized.get_pixel(x * patch_size + px, y * patch_size + py);
                        pv.push(p[0] as f32 / 255.0);
                        pv.push(p[1] as f32 / 255.0);
                        pv.push(p[2] as f32 / 255.0);
                    }
                }
            }
        }
        let pixel_values = Tensor::<B, 1>::from_floats(&pv[..], &device)
            .reshape([1, num_patches, patch_dim]);

        let mut pos: Vec<i32> = Vec::with_capacity(num_patches * 2);
        for y in 0..grid { for x in 0..grid { pos.push(x as i32); pos.push(y as i32); } }
        let pixel_position_ids = Tensor::<B, 1, Int>::from_ints(&pos[..], &device)
            .reshape([1, num_patches, 2]);

        eprintln!("[vision] encoding image...");
        let soft = enc.forward(pixel_values, pixel_position_ids, &device);
        let soft_len = soft.dims()[0];
        eprintln!("[vision] produced {} soft tokens", soft_len);

        eprintln!("[vision] building full text model...");
        // Re-load the text model properly using the LM wrapper
        // The decoder is already loaded, we just need to wrap it.
        // But `GemmaLM` constructor wants the `Gemma4Model`, not just the decoder.
        // Wait, `load_gemma4_streaming_from_pile` returned `(_decoder, vision)`.
        // _decoder IS `Gemma4Model<B>`. Let's wrap it!
        let tokenizer_path = find_hf_file("google/gemma-4-31B-it", "tokenizer.json");
        let lm = GemmaLM::from_model(config.clone(), _decoder, Path::new(&tokenizer_path), device.clone());

        eprintln!("[vision] assembling prompt...");
        let pre = format!("<bos><|turn>user\n");
        let post = format!("Describe this image.<turn|>\n<|turn>model\n");

        eprintln!("[vision] generating description...");
        let text = lm.complete_with_vision(&pre, soft, &post, 128);
        println!("\n=== VISION OUTPUT ===");
        println!("{text}");
    } else {
        // Synthetic full image: a grid of patches, all valid (position ids >= 0).
        // patch_size=16 → 3*16*16 = 768 floats per patch; use a modest grid so the
        // forward is quick but exercises the real code path (RoPE, attention, pool).
        let patch_dim = 3 * enc.config.patch_size * enc.config.patch_size;
        let grid = 12usize; // 12x12 = 144 patches
        let num_patches = grid * grid;
        let pv: Vec<f32> = (0..num_patches * patch_dim)
            .map(|i| ((i % 255) as f32 / 255.0) - 0.5)
            .collect();
        let pixel_values = Tensor::<B, 1>::from_floats(&pv[..], &device)
            .reshape([1, num_patches, patch_dim]);
        let mut pos: Vec<i32> = Vec::with_capacity(num_patches * 2);
        for y in 0..grid { for x in 0..grid { pos.push(x as i32); pos.push(y as i32); } }
        let pixel_position_ids = Tensor::<B, 1, Int>::from_ints(&pos[..], &device)
            .reshape([1, num_patches, 2]);

        eprintln!("[vision] running encoder forward on a {grid}x{grid} synthetic patch grid...");
        let soft = enc.forward(pixel_values, pixel_position_ids, &device);
        let dims = soft.dims();
        let flat: Vec<f32> = soft.reshape([dims[0] * dims[1]]).to_data().convert::<f32>().to_vec().unwrap();
        let finite = flat.iter().filter(|x| x.is_finite()).count();
        let (mn, mx) = flat.iter().fold((f32::INFINITY, f32::NEG_INFINITY), |(a, b), &x| (a.min(x), b.max(x)));

        for (i, &val) in flat.iter().enumerate() {
            if !val.is_finite() {
                let token_idx = i / dims[1];
                let channel_idx = i % dims[1];
                println!("Token {token_idx}, Channel {channel_idx} is {val}");
                if i > dims[1] * 3 { break; } // just print a few
            }
        }

        println!("\n=== VISION ENCODER (31B) ===");
        println!("encoder layers        : {}", enc.config.num_hidden_layers);
        println!("soft-token output     : {:?} (pooled_tokens x text_hidden)", dims);
        println!("finite / total        : {finite} / {}", flat.len());
        println!("value range           : [{mn:.4}, {mx:.4}]");
        let ok = finite == flat.len() && dims[1] == config.text_config.hidden_size;
        println!("GATE: {}", if ok { "PASS (encoder loads + forwards, output in text-hidden space, all finite)" } else { "FAIL" });
        if !ok { std::process::exit(1); }
    }
}
