//! Parity gate for the Inkling Burn lane against the f32 slice lane.
//!
//! The slice lane is the reference here, and it is itself gated against
//! `transformers` by `inkling_layer_gate` and `inkling_real_gate`. So this
//! needs no torch: the oracle is code that already has one.
//!
//! Budget, written down before any number was read: worst absolute error over
//! the tensor's own scale, `1e-5`. Looser than the slice-vs-torch gates because
//! a backend matmul blocks and reorders its accumulations, which is a bigger
//! reordering than the one between two scalar loops. The per-element relative
//! figure is printed and NOT gated on — these outputs cancel, and dividing by a
//! near-zero reference is meaningless, which cost a false failure once already.
//!
//! Non-vacuity: the shapes are the real model's (hidden 4096, intermediate
//! 2048, dense intermediate 16384), not toys, and every check prints how many
//! values it compared. A gate that ran on 4x4 tensors would pass without
//! touching the blocking behaviour that makes a backend matmul differ at all.
//!
//! MEASURED 2026-08-08 on the GB10. ndarray passes; CUDA does NOT, and the
//! shape of the failure is the finding:
//!
//! ```text
//!   rms_norm    1.1e-6   passes
//!   expert_ffn  4.5e-4   FAILS, 45x over
//!   dense_mlp   4.3e-4   FAILS, 43x over
//! ```
//!
//! Only the matmul-bearing checks fail, both at the same magnitude, while
//! RMSNorm under the identical metric passes at 1.1e-6 — so this is not a
//! cancellation artifact and not the elementwise path. About 4.3e-4 relative is
//! roughly eleven bits of mantissa, which is what a TF32 tensor-core matmul
//! gives. The likely cause is cubecl dispatching f32 matmul to TF32 on this
//! backend; confirming that and finding the switch to force full f32 accumulate
//! is open work.
//!
//! The budget is deliberately NOT widened to accommodate it. A GPU lane that
//! silently carries eleven mantissa bits is a fact worth failing over, and the
//! whole point of gating the Burn lane against the slice lane is to surface
//! exactly this before anything depends on it.
//!
//!   cargo run --release --features inkling-burn --bin inkling_burn_gate
//!   cargo run --release --features inkling-cuda --bin inkling_burn_gate -- cuda

use burn::prelude::*;
use burn::tensor::{Tensor, TensorData};

use mary::models::inkling::mlp as slice_lane;

const BUDGET: f32 = 1e-5;

/// Deterministic pseudo-random values — no rng dependency, and the same numbers
/// on every run so a failure can be re-examined.
fn fill(n: usize, seed: u64) -> Vec<f32> {
    let mut s = seed | 1;
    (0..n)
        .map(|_| {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            ((s >> 40) as f32 / 16_777_216.0) - 0.5
        })
        .collect()
}

fn t2<B: Backend>(v: &[f32], r: usize, c: usize, dev: &B::Device) -> Tensor<B, 2> {
    Tensor::from_data(TensorData::new(v.to_vec(), [r, c]), dev)
}

struct D {
    abs: f32,
    scale: f32,
    rel: f32,
    n: usize,
}
impl D {
    fn scaled(&self) -> f32 {
        self.abs / self.scale.max(f32::MIN_POSITIVE)
    }
}
fn cmp(a: &[f32], b: &[f32]) -> D {
    let mut d = D { abs: 0.0, scale: 0.0, rel: 0.0, n: a.len().min(b.len()) };
    for (&x, &y) in a.iter().zip(b) {
        let e = (x - y).abs();
        d.abs = d.abs.max(e);
        d.scale = d.scale.max(y.abs());
        d.rel = d.rel.max(e / y.abs().max(1e-6));
    }
    d
}

fn run<B: Backend>(dev: &B::Device, label: &str) -> (usize, usize) {
    let mut checks = 0usize;
    let mut fails = 0usize;
    let mut report = |name: &str, d: D, checks: &mut usize, fails: &mut usize| {
        *checks += d.n;
        println!("  {name}: {} values, worst abs {:e} / scale {:e} = {:e}, rel {:e}",
                 d.n, d.abs, d.scale, d.scaled(), d.rel);
        if d.n == 0 {
            println!("    FAIL  compared nothing");
            *fails += 1;
        }
        if d.scaled() > BUDGET {
            println!("    FAIL  over budget {BUDGET:e}");
            *fails += 1;
        }
    };

    // Real model dimensions: a toy size would not exercise a backend matmul's
    // blocking, which is the only reason the two lanes differ at all.
    let (tokens, h, inter, dense_inter) = (8usize, 4096usize, 2048usize, 16384usize);
    println!("\n=== {label}: tokens {tokens}, hidden {h}, inter {inter}, dense {dense_inter} ===");

    let x = fill(tokens * h, 0x51ED);
    let gain = fill(h, 0xA17);
    let eps = 1e-6f64;

    // ---- RMSNorm ----------------------------------------------------------
    let mine = {
        let g: Tensor<B, 1> = Tensor::from_data(TensorData::new(gain.clone(), [h]), dev);
        mary::models::inkling::burn::rms_norm(t2::<B>(&x, tokens, h, dev), g, eps)
            .into_data()
            .convert::<f32>()
            .to_vec::<f32>()
            .unwrap()
    };
    let theirs = mary::models::inkling::block::rms_norm(&x, &gain, eps, tokens, h);
    report("rms_norm", cmp(&mine, &theirs), &mut checks, &mut fails);

    // ---- one expert's feed-forward ----------------------------------------
    let gu = fill(2 * inter * h, 0xBEEF);
    let dn = fill(h * inter, 0xF00D);
    let mine = mary::models::inkling::burn::expert_ffn(
        t2::<B>(&x, tokens, h, dev),
        t2::<B>(&gu, 2 * inter, h, dev),
        t2::<B>(&dn, h, inter, dev),
    )
    .into_data()
    .convert::<f32>()
    .to_vec::<f32>()
    .unwrap();
    let theirs = {
        let mut out = vec![0f32; tokens * h];
        for t in 0..tokens {
            let xt = &x[t * h..(t + 1) * h];
            let contrib = slice_lane::expert_ffn_one(xt, &gu, &dn, h, inter);
            out[t * h..(t + 1) * h].copy_from_slice(&contrib);
        }
        out
    };
    report("expert_ffn", cmp(&mine, &theirs), &mut checks, &mut fails);

    // ---- dense MLP ---------------------------------------------------------
    let g = fill(dense_inter * h, 0x1234);
    let u = fill(dense_inter * h, 0x5678);
    let d = fill(h * dense_inter, 0x9ABC);
    let gs = 1.7f32;
    let mine = mary::models::inkling::burn::dense_mlp(
        t2::<B>(&x, tokens, h, dev),
        t2::<B>(&g, dense_inter, h, dev),
        t2::<B>(&u, dense_inter, h, dev),
        t2::<B>(&d, h, dense_inter, dev),
        gs,
    )
    .into_data()
    .convert::<f32>()
    .to_vec::<f32>()
    .unwrap();
    let theirs = slice_lane::dense_mlp(&x, &g, &u, &d, gs, tokens, h, dense_inter);
    report("dense_mlp", cmp(&mine, &theirs), &mut checks, &mut fails);

    (checks, fails)
}

fn main() {
    let want_cuda = std::env::args().nth(1).as_deref() == Some("cuda");
    println!("=== Inkling Burn lane vs the f32 slice lane ===");
    println!("  the slice lane is itself gated against transformers, so this needs no torch");
    println!("  budget: worst-absolute-over-scale {BUDGET:e}, written down first");

    #[allow(unused_mut)]
    let mut total = (0usize, 0usize);

    #[cfg(feature = "inkling-cuda")]
    if want_cuda {
        type C = burn::backend::Cuda<f32>;
        let (c, f) = run::<C>(&Default::default(), "cuda");
        total = (total.0 + c, total.1 + f);
    }
    #[cfg(not(feature = "inkling-cuda"))]
    if want_cuda {
        println!("  (cuda requested but this build has no inkling-cuda feature)");
    }

    if !want_cuda {
        type N = burn::backend::NdArray<f32>;
        let (c, f) = run::<N>(&Default::default(), "ndarray");
        total = (total.0 + c, total.1 + f);
    }

    println!("\n=== verdict ===");
    println!("  checks: {}", total.0);
    if total.1 == 0 {
        println!("GATE PASSED — {} checks, the Burn lane matches the slice lane", total.0);
    } else {
        println!("GATE FAILED — {} checks, {} FAILURES", total.0, total.1);
        std::process::exit(1);
    }
}
