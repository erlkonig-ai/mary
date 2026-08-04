//! Rust side of the CFM sampler parity probe. Loads the fixed y0/cond/text from
//! probes/cfm/, runs `cfm::integrate` (8 steps, cfg 2, sway −1), writes the
//! final mel for compare_cfm.py.
//!
//!   python3 scripts/probe_cfm.py <model.safetensors>
//!   cargo run --release --bin probe_cfm -- <model.safetensors>

use burn::prelude::*;
use burn::tensor::TensorData;
use mary::models::f5::cfm;
use mary::models::f5::config::{CfmConfig, F5Config};
use mary::models::f5::model::F5Transformer;
use mary::nn::backend::B;
use mary::nn::npy;
use mary::nn::weight_loader::{SingleFileLoader, WeightLoader};
use std::path::{Path, PathBuf};

fn main() {
    let path = std::env::args()
        .nth(1)
        .expect("usage: probe_cfm <model.safetensors>");
    let device = Default::default();
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let dir = root.join("probes").join("cfm");
    let rs = root.join("probes").join("cfm_rust");
    std::fs::create_dir_all(&rs).unwrap();

    let load = |n: &str| npy::load_npy(&dir.join(format!("{n}.npy"))).unwrap();
    let (yd, ys) = load("y0");
    let (cd, cs) = load("cond");
    let (td, ts) = load("text");
    let y0 = Tensor::<B, 3>::from_data(TensorData::new(yd, ys), &device);
    let cond = Tensor::<B, 3>::from_data(TensorData::new(cd, cs), &device);
    let text = Tensor::<B, 2>::from_data(TensorData::new(td, ts), &device).int();

    let loader = WeightLoader::SingleFile(SingleFileLoader::new(Path::new(&path)));
    let model = F5Transformer::<B>::load(&loader, F5Config::v1_base(), &device);
    let cfg = CfmConfig {
        nfe: 8,
        sway_coef: -1.0,
        cfg_strength: 2.0,
    };

    let mel = cfm::integrate(&model, y0, cond, text, &cfg, &device);
    let data = mel.into_data();
    let shape = data.shape.to_vec();
    npy::save_npy(&rs.join("mel.npy"), &data.to_vec::<f32>().unwrap(), &shape).unwrap();
    println!("✓ cfm rust mel {shape:?} → {}", rs.display());
}
