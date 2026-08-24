//! backend_matmul_probe — isolates WHICH burn backend the Moshi-decode-shaped
//! M=1 matmul chain actually runs on, and what each one costs.
//!
//! `moshi_q4_probe`'s "burn f16 matmul" baseline uses `mary::nn::backend::BHalf`
//! = `burn::backend::Metal<f16>`, and burn's `Metal` is a plain alias for
//! `Wgpu` (burn-wgpu/src/lib.rs:110). On the Spark that baseline is therefore
//! wgpu/Vulkan, never burn's CUDA backend. This probe runs the same chain on
//! Wgpu<f16> and on Cuda<f16> side by side, plus an M-sweep to expose tile
//! quantization and a single-shape micro-bench to separate launch overhead
//! from bandwidth.
//!
//! Run: cargo run --release --features burn-cuda-bench --bin backend_matmul_probe

use burn::prelude::*;
use burn::tensor::backend::{Backend, BackendTypes};
use burn::tensor::Distribution;
use std::time::Instant;

const LAYERS: usize = 32;
const HIDDEN: usize = 4096;
const INTER: usize = 16384;

const SHAPES: [(usize, usize); 7] = [
    (HIDDEN, HIDDEN),
    (HIDDEN, HIDDEN),
    (HIDDEN, HIDDEN),
    (HIDDEN, HIDDEN),
    (INTER, HIDDEN),
    (INTER, HIDDEN),
    (HIDDEN, INTER),
];

fn median(mut v: Vec<f64>) -> f64 {
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    v[v.len() / 2]
}

fn params_per_step() -> usize {
    SHAPES.iter().map(|(o, i)| o * i).sum::<usize>() * LAYERS
}

fn bench(
    rounds: usize,
    steps: usize,
    mut submit: impl FnMut(),
    mut sync: impl FnMut(),
) -> (f64, f64) {
    for _ in 0..3 {
        submit();
        sync();
    }
    let mut sm = Vec::new();
    let mut fm = Vec::new();
    for _ in 0..rounds {
        let mut s = Vec::new();
        let mut f = Vec::new();
        for _ in 0..steps {
            let t0 = Instant::now();
            submit();
            s.push(t0.elapsed().as_secs_f64() * 1e3);
            sync();
            f.push(t0.elapsed().as_secs_f64() * 1e3);
        }
        sm.push(median(s));
        fm.push(median(f));
    }
    (
        sm.iter().cloned().fold(f64::INFINITY, f64::min),
        fm.iter().cloned().fold(f64::INFINITY, f64::min),
    )
}

/// The moshi_q4_probe chain, verbatim, generic over backend.
fn chain<B: Backend>(
    device: &B::Device,
    rounds: usize,
    steps: usize,
    m: usize,
    transposed: bool,
) -> (f64, f64) {
    // `transposed`: store weights [out, in] and hand `matmul` a swap_dims view,
    // so cubek sees MildlyPermuted{transposed} rhs (its "col major vecmat",
    // PRIORITY_MAX in the gemv tune group) instead of a Contiguous rhs (the
    // "we don't have good algos for row major vecmat" case).
    let layers: Vec<Vec<Tensor<B, 2>>> = (0..LAYERS)
        .map(|_| {
            SHAPES
                .iter()
                .map(|&(o, i)| {
                    if transposed {
                        Tensor::<B, 2>::random([o, i], Distribution::Uniform(-0.01, 0.01), device)
                            .transpose()
                    } else {
                        Tensor::<B, 2>::random([i, o], Distribution::Uniform(-0.01, 0.01), device)
                    }
                })
                .collect()
        })
        .collect();
    let x0 = Tensor::<B, 2>::random([m, HIDDEN], Distribution::Uniform(-0.1, 0.1), device);
    let last: std::cell::RefCell<Option<Tensor<B, 2>>> = std::cell::RefCell::new(None);
    bench(
        rounds,
        steps,
        || {
            let mut x = x0.clone();
            for l in &layers {
                let yq = x.clone().matmul(l[0].clone());
                let _yk = x.clone().matmul(l[1].clone());
                let _yv = x.matmul(l[2].clone());
                let yo = yq.matmul(l[3].clone());
                let yg = yo.clone().matmul(l[4].clone());
                let _yu = yo.matmul(l[5].clone());
                x = yg.matmul(l[6].clone());
            }
            *last.borrow_mut() = Some(x);
        },
        || {
            let _ = last
                .borrow()
                .as_ref()
                .unwrap()
                .clone()
                .into_data()
                .convert::<f32>()
                .to_vec::<f32>()
                .unwrap();
        },
    )
}

/// One shape, repeated `reps` times with independent weights (so it streams
/// `reps` distinct weight copies, not a cached one), M rows.
fn single_shape<B: Backend>(
    device: &B::Device,
    out: usize,
    inn: usize,
    reps: usize,
    m: usize,
    rounds: usize,
    transposed: bool,
) -> f64 {
    let ws: Vec<Tensor<B, 2>> = (0..reps)
        .map(|_| {
            if transposed {
                Tensor::<B, 2>::random([out, inn], Distribution::Uniform(-0.01, 0.01), device)
                    .transpose()
            } else {
                Tensor::<B, 2>::random([inn, out], Distribution::Uniform(-0.01, 0.01), device)
            }
        })
        .collect();
    let x = Tensor::<B, 2>::random([m, inn], Distribution::Uniform(-0.1, 0.1), device);
    let last: std::cell::RefCell<Option<Tensor<B, 2>>> = std::cell::RefCell::new(None);
    let (_s, f) = bench(
        rounds,
        5,
        || {
            let mut acc: Option<Tensor<B, 2>> = None;
            for w in &ws {
                let y = x.clone().matmul(w.clone());
                acc = Some(match acc {
                    None => y,
                    Some(a) => a + y,
                });
            }
            *last.borrow_mut() = acc;
        },
        || {
            let _ = last
                .borrow()
                .as_ref()
                .unwrap()
                .clone()
                .into_data()
                .convert::<f32>()
                .to_vec::<f32>()
                .unwrap();
        },
    );
    f
}

fn report<B: Backend>(name: &str, device: &B::Device, rounds: usize, steps: usize) {
    let params = params_per_step();
    let f16_gb = params as f64 * 2.0 / 1e9;
    println!("\n=== {name} ===");

    for (tag, tr) in [("[in,out] contig", false), ("[out,in] transp", true)] {
        for m in [1usize, 8, 64] {
            let (s, f) = chain::<B>(device, rounds, steps, m, tr);
            println!(
                "  chain {tag} M={m:<3} submit {s:7.2} ms  full {f:7.2} ms   {:6.1} GB/s (weights {:.1} GB)",
                f16_gb / (f / 1e3),
                f16_gb
            );
        }
    }

    // single shape: 4096x4096 x 64 reps = 2.15 GB of weights
    let reps = 64;
    let bytes = (HIDDEN * HIDDEN * 2 * reps) as f64 / 1e9;
    for (tag, tr) in [("[in,out] contig", false), ("[out,in] transp", true)] {
        for m in [1usize, 8, 64, 256] {
            let f = single_shape::<B>(device, HIDDEN, HIDDEN, reps, m, rounds, tr);
            println!(
                "  4096^2 x{reps} {tag} M={m:<4} full {f:7.2} ms   {:6.1} GB/s   ({:.2} ms/matmul)",
                bytes / (f / 1e3),
                f / reps as f64
            );
        }
    }
}

fn main() {
    let rounds: usize = std::env::var("Q4_ROUNDS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(5);
    let steps: usize = std::env::var("Q4_STEPS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(8);
    let which = std::env::var("BACKENDS").unwrap_or_else(|_| "wgpu,cuda".into());
    println!("backend_matmul_probe — {rounds} rounds x {steps} steps, min-of-medians");
    println!(
        "params/step {:.2}G, f16 {:.2} GB/step",
        params_per_step() as f64 / 1e9,
        params_per_step() as f64 * 2.0 / 1e9
    );

    if which.contains("wgpu") {
        type W = burn::backend::Metal<half::f16>;
        let d: <W as BackendTypes>::Device = Default::default();
        report::<W>(
            "burn Wgpu<f16>  (BHalf — what moshi_q4_probe measures)",
            &d,
            rounds,
            steps,
        );
    }
    if which.contains("cuda") {
        type C = burn::backend::Cuda<half::f16>;
        let d: <C as BackendTypes>::Device = Default::default();
        report::<C>(
            "burn Cuda<f16>  (burn's real CUDA backend)",
            &d,
            rounds,
            steps,
        );
    }
}
