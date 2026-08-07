//! Mel-extraction parity probe: load the ref audio saved by prep_infer.py, run
//! MelExtractor, compare to the reference get_vocos_mel_spectrogram (ref_mel.npy).
//!
//!   cargo run --release --bin probe_mel

use burn::prelude::*;
use burn::tensor::TensorData;
use mary::models::f5::mel::MelExtractor;
use mary::nn::backend::B;
use mary::nn::npy;
use std::path::PathBuf;

fn main() {
    let device = Default::default();
    let dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("probes")
        .join("infer");
    let (ad, as_) = npy::load_npy(&dir.join("ref_audio.npy")).unwrap(); // [1, n_samples]
    let wav = Tensor::<B, 2>::from_data(TensorData::new(ad, as_), &device);
    let mel = MelExtractor::<B>::new(&device).forward(wav).swap_dims(1, 2); // [1,T,100]
    let d = mel.into_data();
    let sh = d.shape.to_vec();
    npy::save_npy(
        &dir.join("ref_mel_burn.npy"),
        &d.to_vec::<f32>().unwrap(),
        &sh,
    )
    .unwrap();
    println!("✓ burn ref_mel {sh:?}");
}
