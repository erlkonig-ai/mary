//! Rust side of the F5 numerical-parity probe. Loads the SAME fixed input
//! that `scripts/probe_f5.py` saved, runs `F5Transformer::forward_probed`, and
//! writes each intermediate to probes/rust/ for `scripts/compare_probes.py`.
//!
//!   python3 scripts/probe_f5.py <model.safetensors>      # reference first
//!   cargo run --release --bin probe -- <model.safetensors>

use burn::prelude::*;
use burn::tensor::TensorData;
use mary::models::f5::config::F5Config;
use mary::models::f5::model::F5Transformer;
use mary::nn::backend::B;
use mary::nn::npy;
use mary::nn::weight_loader::{SingleFileLoader, WeightLoader};
use std::path::{Path, PathBuf};

fn load(dir: &Path, name: &str) -> (Vec<f32>, Vec<usize>) {
    npy::load_npy(&dir.join(format!("{name}.npy"))).unwrap()
}

fn main() {
    let path = std::env::args()
        .nth(1)
        .expect("usage: probe <model.safetensors>");
    let device = Default::default();
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("probes");
    let py = root.join("python");
    let rs = root.join("rust");
    std::fs::create_dir_all(&rs).unwrap();

    // identical fixed input (as written by the reference script)
    let (xd, xs) = load(&py, "x");
    let (cd, cs) = load(&py, "cond");
    let (td, ts) = load(&py, "text");
    let (md, ms) = load(&py, "time");
    let noised = Tensor::<B, 3>::from_data(TensorData::new(xd, xs), &device);
    let cond = Tensor::<B, 3>::from_data(TensorData::new(cd, cs), &device);
    let text = Tensor::<B, 2>::from_data(TensorData::new(td, ts), &device).int();
    let time = Tensor::<B, 1>::from_data(TensorData::new(md, ms), &device);

    let loader = WeightLoader::SingleFile(SingleFileLoader::new(Path::new(&path)));
    let model = F5Transformer::<B>::load(&loader, F5Config::v1_base(), &device);

    let (_out, probes) = model.forward_probed(noised, cond, text, time);
    for (name, data, shape) in &probes {
        npy::save_npy(&rs.join(format!("{name}.npy")), data, shape).unwrap();
        println!("{name}: {shape:?}");
    }
    println!("✓ rust probes written to {}", rs.display());
}
