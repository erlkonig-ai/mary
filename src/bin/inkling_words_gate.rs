//! `inkling_words_gate` — does the word-packed dequant agree with the byte one?
//!
//! I wrote "gated against `dequant_nvfp4`" in `dequant_nvfp4_words`' own
//! docstring and then did not write the gate. That is a comment claiming
//! verification and standing in for it, which is the failure this whole project
//! keeps catching in other places. This is the missing check.
//!
//! It matters right now because the device expert lane diverges from the host
//! lane by 5.1e-4 while the host's own reassociation noise is 2.0e-6 — 262x
//! apart, so something real differs. Exactly one of two things can explain it:
//! this dequant, or the device matmul. Comparing the two dequants on the SAME
//! backend with NO matmul anywhere isolates the first completely; whatever
//! survives belongs to the second.
//!
//! BITWISE, not a tolerance. Both paths gather through the same host-built
//! lookup tables and multiply in the same order; the only difference is how the
//! nibbles are extracted, which is exact integer work. Anything but equality is
//! a defect, so 0 is the honest bar.
//!
//! Non-vacuity, since a gate that cannot fail is decoration:
//!   * runs on REAL expert bytes from the released checkpoint, not a fixture —
//!     a synthetic corpus would very likely miss the E4M3 subnormals and the
//!     top-nibble sign-extension case that make this arithmetic interesting;
//!   * prints how many values it compared;
//!   * `--mutate` perturbs one shift and REQUIRES the comparison to fail, so the
//!     equality has been watched rejecting something before it is believed.
//!
//!   cargo run --release --features inkling-cuda --bin inkling_words_gate -- <ckpt> [--mutate]

use anyhow::{Context, Result};
use burn::prelude::Backend;
use burn::tensor::{Int, Tensor, TensorData};

use mary::models::inkling::burn::{dequant_nvfp4, dequant_nvfp4_words};
use mary::models::inkling::load::Checkpoint;

type Bk = burn::backend::Cuda<f32>;

/// The byte-per-element upload the original gated path uses.
fn as_bytes<B: Backend>(
    codes: &[u8],
    scales: &[u8],
    scale2: f32,
    rows: usize,
    cols: usize,
    dev: &B::Device,
) -> Tensor<B, 2> {
    let n_scales = scales.len() / rows;
    let c: Vec<i32> = codes.iter().map(|&b| b as i32).collect();
    let s: Vec<i32> = scales.iter().map(|&b| b as i32).collect();
    dequant_nvfp4(
        Tensor::<B, 2, Int>::from_data(TensorData::new(c, [rows, cols]), dev),
        Tensor::<B, 2, Int>::from_data(TensorData::new(s, [rows, n_scales]), dev),
        Tensor::<B, 1>::from_data(TensorData::new(vec![scale2; rows], [rows]), dev),
    )
}

/// The word-packed upload, optionally with one shift perturbed.
fn as_words<B: Backend>(
    codes: &[u8],
    scales: &[u8],
    scale2: f32,
    rows: usize,
    cols: usize,
    dev: &B::Device,
    mutate: bool,
) -> Tensor<B, 2> {
    let n_scales = scales.len() / rows;
    let word = |b: &[u8]| -> Vec<i32> {
        b.chunks_exact(4)
            .map(|c| i32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect()
    };
    let mut cw = word(codes);
    if mutate {
        // Rotate one word by a nibble: still a valid word, wrong values, and
        // invisible to any shape or size check.
        if let Some(v) = cw.get_mut(0) {
            *v = v.rotate_left(4);
        }
    }
    dequant_nvfp4_words(
        Tensor::<B, 2, Int>::from_data(TensorData::new(cw, [rows, cols / 4]), dev),
        Tensor::<B, 2, Int>::from_data(TensorData::new(word(scales), [rows, n_scales / 4]), dev),
        Tensor::<B, 1>::from_data(TensorData::new(vec![scale2; rows], [rows]), dev),
    )
}

fn main() -> Result<()> {
    let ckpt = std::env::args().nth(1).context("usage: <ckpt> [--mutate]")?;
    let mutate = std::env::args().any(|a| a == "--mutate");
    let cp = Checkpoint::open(&ckpt)?;
    let dev = burn::backend::cuda::CudaDevice::default();

    println!("=== word-packed dequant vs byte-packed dequant ===");
    println!("  bar    : BITWISE equality (both gather the same tables, multiply in");
    println!("           the same order; only nibble extraction differs, which is exact)");
    if mutate {
        println!("  MODE   : --mutate (one word rotated; a PASS here is the bug)");
    }

    // Real experts from real layers, including several so a single unlucky
    // weight slice cannot carry the verdict.
    let mut checked = 0usize;
    let mut mismatched = 0usize;
    let mut worst = 0f32;
    for (layer, e) in [(3usize, 0usize), (3, 17), (20, 5), (41, 200)] {
        let base = format!("model.llm.layers.{layer}.mlp.experts.w13_weight");
        if !cp.is_nvfp4(&base) {
            println!("  layer {layer} expert {e}: not NVFP4, skipped");
            continue;
        }
        let q = cp
            .expert_slice_packed(&base, e)
            .with_context(|| format!("slicing layer {layer} expert {e}"))?;

        let a = as_bytes::<Bk>(&q.codes, &q.scales, q.scale2, q.rows, q.cols, &dev);
        let b = as_words::<Bk>(&q.codes, &q.scales, q.scale2, q.rows, q.cols, &dev, mutate);
        let av = a.into_data().convert::<f32>().to_vec::<f32>().unwrap();
        let bv = b.into_data().convert::<f32>().to_vec::<f32>().unwrap();
        anyhow::ensure!(av.len() == bv.len(), "lengths differ: {} vs {}", av.len(), bv.len());
        anyhow::ensure!(!av.is_empty(), "empty tensor — the check would be vacuous");

        let mut bad = 0usize;
        let mut w = 0f32;
        for (x, y) in av.iter().zip(&bv) {
            if x.to_bits() != y.to_bits() {
                bad += 1;
                w = w.max((x - y).abs());
            }
        }
        checked += av.len();
        mismatched += bad;
        worst = worst.max(w);
        println!(
            "  layer {layer:2} expert {e:3}: {} values, {bad} differing, worst {w:.3e}",
            av.len()
        );
    }

    println!("\n  values compared : {checked}");
    println!("  differing       : {mismatched}");
    anyhow::ensure!(checked > 0, "GATE VACUOUS — compared nothing");

    if mutate {
        if mismatched > 0 {
            println!("MUTATION CAUGHT — {mismatched} values differ, worst {worst:.3e}");
            println!("The equality can fail, so a clean run means something.");
            return Ok(());
        }
        anyhow::bail!("MUTATION SURVIVED — a rotated word produced identical output. \
                       This gate cannot discriminate.");
    }

    if mismatched > 0 {
        anyhow::bail!(
            "GATE FAILED — {mismatched} of {checked} values differ, worst {worst:.3e}. \
             The word-packed dequant is NOT equivalent to the gated byte one."
        );
    }
    println!("GATE PASSED — {checked} values bitwise identical");
    println!("\nSo the dequant is exonerated, and any host-vs-device divergence in the");
    println!("forward has to live in the matmul rather than here.");
    Ok(())
}
