//! Rust side of the Vocos numerical-parity probe. Loads the fixed mel from
//! probes/vocos/mel.npy and the exported weights, runs `Vocos::forward_probed`,
//! and writes intermediates to probes/vocos_rust/ for compare_vocos.py.
//!
//!   python3 scripts/probe_vocos.py            # reference + weight export
//!   cargo run --release --bin probe_vocos

use mary::nn::backend::B;
use mary::nn::npy;
use mary::nn::weight_loader::{SingleFileLoader, WeightLoader};
use burn::prelude::*;
use burn::tensor::TensorData;
use std::path::{Path, PathBuf};
use mary::models::f5::vocos::Vocos;

fn main() {
    let device = Default::default();
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let py = root.join("probes").join("vocos");
    let rs = root.join("probes").join("vocos_rust");
    std::fs::create_dir_all(&rs).unwrap();

    let (md, ms) = npy::load_npy(&py.join("mel.npy")).unwrap();
    let mel = Tensor::<B, 3>::from_data(TensorData::new(md, ms), &device);

    let wpath = root.join("weights").join("vocos.safetensors");
    let loader = WeightLoader::SingleFile(SingleFileLoader::new(&wpath));
    let model = Vocos::<B>::load(&loader, &device);

    let (_audio, probes) = model.forward_probed(mel);
    for (name, data, shape) in &probes {
        npy::save_npy(&rs.join(format!("{name}.npy")), data, shape).unwrap();
        println!("{name}: {shape:?}");
    }
    println!("✓ vocos rust probes → {}", rs.display());
}
