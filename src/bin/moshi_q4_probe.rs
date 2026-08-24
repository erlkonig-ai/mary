//! moshi_q4_probe — the q4 weight-quantization spike for the PersonaPlex-7B
//! (Moshi) realtime lane. `moshi_realtime_probe` (2e6ae2e) proved the decode
//! step's hard floor is streaming 17.2 GB of f16 weights every 80 ms frame
//! (32–42 ms of pure bandwidth on M4 Max). This probe answers the follow-up:
//! does a q4 dequant-in-kernel matvec (`mary::nn::q4`) actually deliver the
//! projected ~3.5× bandwidth win at the real Moshi decode shapes?
//!
//! Three parts:
//!   1. CORRECTNESS — quantize random f32 Linears at real shapes, compare the
//!      GPU kernel against (a) the CPU dequantized-weight oracle (kernel
//!      implements the format: rel RMS ≲ 1e-5, f32 sum-order only) and (b) the
//!      unquantized f64 reference (pure q4 noise: rel RMS ~1e-2 class), plus a
//!      bitwise determinism check (two launches, identical bits).
//!   2. BANDWIDTH — a decode-step-shaped benchmark: the full temporal-layer
//!      matmul set (q/k/v/o 4096², gate/up 16384×4096, down 4096×16384) × 32
//!      layers, M=1, chained through intermediate buffers like a real step.
//!      Three variants: burn f16 Tensor matmul (the production baseline the
//!      realtime probe used), the custom f16 matvec kernel (same kernel shape
//!      as q4 — isolates bytes from kernel design), and q4. Methodology
//!      mirrors the realtime probe: warm-up discarded, min-of-medians over
//!      rounds, submit-vs-full split, interleaved control op.
//!      Bench weights are zeroed `client.empty` buffers — GPU memory traffic
//!      is value-independent, and 22 GB of host-side RNG buys nothing.
//!   3. PROJECTION — plug the measured q4 time into the realtime probe's frame
//!      model (temporal + 21.6 ms depth + ~5 ms mimi vs 80 ms @ 12.5 Hz), both
//!      as a naive substitution into the probe's raw measurements and as the
//!      with-levers floor (megakernel-class submission + static-slot KV).
//!
//! Run: `cargo run --release --features q4 --bin moshi_q4_probe`
//! Env: Q4_ROUNDS (default 5), Q4_STEPS (default 8).

use burn::prelude::*;
use burn::tensor::backend::BackendTypes;
use burn::tensor::Distribution;
use cubecl::server::Handle;
use cubecl::CubeElement;
use mary::nn::backend::BHalf;
use mary::nn::q4::{self, dequantize_q4, f16_matvec, quantize_q4, Q4Linear};
use std::time::Instant;

/// Deterministic pseudo-random f32 in (-scale, scale) — same generator as the
/// realtime probe.
fn fill(n: usize, seed: u64, scale: f32) -> Vec<f32> {
    let mut s = seed.wrapping_mul(0x9E3779B97F4A7C15).wrapping_add(1);
    (0..n)
        .map(|_| {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            let u = ((s >> 11) as f64 / (1u64 << 53) as f64) as f32;
            (u * 2.0 - 1.0) * scale
        })
        .collect()
}

fn median(mut v: Vec<f64>) -> f64 {
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    v[v.len() / 2]
}

fn rel_rms(y: &[f32], y_ref: &[f64]) -> f64 {
    let mut num = 0f64;
    let mut den = 0f64;
    for (a, b) in y.iter().zip(y_ref) {
        num += (*a as f64 - b).powi(2);
        den += b.powi(2);
    }
    (num / den).sqrt()
}

fn read_f32(client: &q4::Client, h: &Handle, n: usize) -> Vec<f32> {
    let bytes = client.read_one(h.clone()).expect("readback");
    let mut v = vec![0f32; n];
    v.copy_from_slice(f32::from_bytes(&bytes[..n * 4]));
    v
}

fn as_bytes<T>(v: &[T]) -> &[u8] {
    unsafe { core::slice::from_raw_parts(v.as_ptr() as *const u8, std::mem::size_of_val(v)) }
}

// ---------------------------------------------------------------------------
// 1. correctness
// ---------------------------------------------------------------------------

/// Quantize a random Linear at `[out, in]`, run the GPU kernel, and gate it
/// against the CPU format-oracle and the unquantized reference.
fn correctness(client: &q4::Client, out_dim: usize, in_dim: usize, seed: u64) {
    let w = fill(out_dim * in_dim, seed, (in_dim as f32).powf(-0.5));
    let x = fill(in_dim, seed + 99, 0.1);

    let lin = Q4Linear::from_f32(client, &w, out_dim, in_dim);
    let xh = client.create_from_slice(as_bytes(&x));
    let yh = client.empty(out_dim * 4);
    lin.forward(client, &xh, &yh);
    let y_gpu = read_f32(client, &yh, out_dim);

    // determinism: same input, second launch, bitwise identical
    let yh2 = client.empty(out_dim * 4);
    lin.forward(client, &xh, &yh2);
    let y_gpu2 = read_f32(client, &yh2, out_dim);
    assert!(
        y_gpu
            .iter()
            .zip(&y_gpu2)
            .all(|(a, b)| a.to_bits() == b.to_bits()),
        "q4 matvec is not deterministic at [{out_dim}, {in_dim}]"
    );

    // CPU format oracle: matvec against the dequantized weight, grouped f32
    // accumulation matching the kernel (per-group sum, then scale)
    let (wq, scales) = quantize_q4(&w, out_dim, in_dim);
    let wd = dequantize_q4(&wq, &scales, out_dim, in_dim);
    let y_fmt: Vec<f64> = (0..out_dim)
        .map(|j| {
            let mut acc = 0f64;
            for g in 0..in_dim / q4::GROUP {
                let mut gsum = 0f32;
                for i in 0..q4::GROUP {
                    let k = g * q4::GROUP + i;
                    gsum += wd[j * in_dim + k] * x[k];
                }
                acc += gsum as f64;
            }
            acc
        })
        .collect();

    // unquantized reference (f64 accumulation)
    let y_ref: Vec<f64> = (0..out_dim)
        .map(|j| {
            (0..in_dim)
                .map(|k| w[j * in_dim + k] as f64 * x[k] as f64)
                .sum()
        })
        .collect();

    let e_kernel = rel_rms(&y_gpu, &y_fmt); // kernel vs format (should be ~f32 eps)
    let e_quant = rel_rms(&y_gpu, &y_ref); // total q4 error vs f32 Linear
    println!(
        "  [{out_dim:>5} x {in_dim:>5}]  kernel-vs-format {e_kernel:.2e}   q4-vs-f32 {e_quant:.3e}   deterministic ok"
    );
    assert!(
        e_kernel < 1e-4,
        "kernel does not implement the q4 format: rel RMS {e_kernel}"
    );
    // analytic q4_0 rel RMS for uniform random weights is ~0.063 (step/√12 over
    // signal RMS); the matvec inherits it ~1:1 (signal and noise both scale √N)
    assert!(
        e_quant < 0.09,
        "q4 error out of expected class: rel RMS {e_quant} (expect ~6e-2 on uniform)"
    );
}

// ---------------------------------------------------------------------------
// 2. decode-step-shaped benchmark
// ---------------------------------------------------------------------------

const LAYERS: usize = 32;
const HIDDEN: usize = 4096;
const INTER: usize = 16384;

/// One temporal layer's matmul set: (out, in) for q, k, v, o, gate, up, down.
const SHAPES: [(usize, usize); 7] = [
    (HIDDEN, HIDDEN), // q
    (HIDDEN, HIDDEN), // k
    (HIDDEN, HIDDEN), // v
    (HIDDEN, HIDDEN), // o
    (INTER, HIDDEN),  // gate
    (INTER, HIDDEN),  // up
    (HIDDEN, INTER),  // down
];

fn params_per_step() -> usize {
    SHAPES.iter().map(|(o, i)| o * i).sum::<usize>() * LAYERS
}

struct Bufs {
    x: Handle,
    q: Handle,
    k: Handle,
    v: Handle,
    o: Handle,
    g: Handle,
    u: Handle,
}

impl Bufs {
    fn new(client: &q4::Client) -> Self {
        Self {
            x: client.empty(HIDDEN * 4),
            q: client.empty(HIDDEN * 4),
            k: client.empty(HIDDEN * 4),
            v: client.empty(HIDDEN * 4),
            o: client.empty(HIDDEN * 4),
            g: client.empty(INTER * 4),
            u: client.empty(INTER * 4),
        }
    }
}

/// min-of-medians (submit, full) over `rounds` × `steps` of `step_submit` +
/// one readback (`sync`) per step.
fn bench(
    rounds: usize,
    steps: usize,
    mut step_submit: impl FnMut(),
    mut sync: impl FnMut(),
) -> (f64, f64) {
    for _ in 0..3 {
        step_submit();
        sync();
    }
    let mut submit_meds = Vec::new();
    let mut full_meds = Vec::new();
    for _ in 0..rounds {
        let mut subs = Vec::new();
        let mut fulls = Vec::new();
        for _ in 0..steps {
            let t0 = Instant::now();
            step_submit();
            let submit = t0.elapsed().as_secs_f64();
            sync();
            let full = t0.elapsed().as_secs_f64();
            subs.push(submit * 1e3);
            fulls.push(full * 1e3);
        }
        submit_meds.push(median(subs));
        full_meds.push(median(fulls));
    }
    (
        submit_meds.iter().cloned().fold(f64::INFINITY, f64::min),
        full_meds.iter().cloned().fold(f64::INFINITY, f64::min),
    )
}

/// Tiny burn control matmul on the shared device — contention canary,
/// identical to the realtime probe's.
fn control_op(device: &<BHalf as BackendTypes>::Device) -> f64 {
    let a = Tensor::<BHalf, 2>::random([256, 256], Distribution::Uniform(-0.1, 0.1), device);
    let b = Tensor::<BHalf, 2>::random([256, 256], Distribution::Uniform(-0.1, 0.1), device);
    let t0 = Instant::now();
    let c = a.matmul(b);
    let _ = c.into_data().convert::<f32>().to_vec::<f32>().unwrap();
    t0.elapsed().as_secs_f64() * 1e3
}

/// burn f16 Tensor-matmul step: the production-path baseline (what the
/// realtime probe's DecoderLayers use for their projections).
fn bench_burn_f16(
    device: &<BHalf as BackendTypes>::Device,
    rounds: usize,
    steps: usize,
) -> (f64, f64) {
    // pre-transposed [in, out] so the chain is pure matmuls
    let layers: Vec<Vec<Tensor<BHalf, 2>>> = (0..LAYERS)
        .map(|_| {
            SHAPES
                .iter()
                .map(|&(o, i)| {
                    Tensor::<BHalf, 2>::random([i, o], Distribution::Uniform(-0.01, 0.01), device)
                })
                .collect()
        })
        .collect();
    let x0 = Tensor::<BHalf, 2>::random([1, HIDDEN], Distribution::Uniform(-0.1, 0.1), device);

    let last: std::cell::RefCell<Option<Tensor<BHalf, 2>>> = std::cell::RefCell::new(None);
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

/// Custom-kernel step (shared by the f16-kernel and q4 variants): the 7-matvec
/// chain per layer, dependencies mirroring a real block (q→o→gate→down→next).
fn submit_custom_step(client: &q4::Client, layers: &[Vec<Q4Linear>], bufs: &Bufs, is_q4: bool) {
    for l in layers {
        let launch = |lin: &Q4Linear, x: &Handle, y: &Handle| {
            if is_q4 {
                lin.forward(client, x, y);
            } else {
                // wq handle doubles as the f16 buffer in the f16 variant
                f16_matvec(client, x, &lin.wq, y, lin.out_dim, lin.in_dim);
            }
        };
        launch(&l[0], &bufs.x, &bufs.q);
        launch(&l[1], &bufs.x, &bufs.k);
        launch(&l[2], &bufs.x, &bufs.v);
        launch(&l[3], &bufs.q, &bufs.o);
        launch(&l[4], &bufs.o, &bufs.g);
        launch(&l[5], &bufs.o, &bufs.u);
        launch(&l[6], &bufs.g, &bufs.x);
    }
}

fn alloc_custom_layers(client: &q4::Client, is_q4: bool) -> Vec<Vec<Q4Linear>> {
    (0..LAYERS)
        .map(|_| {
            SHAPES
                .iter()
                .map(|&(o, i)| {
                    if is_q4 {
                        Q4Linear::empty(client, o, i)
                    } else {
                        // raw f16 rows in the wq slot; scales unused (1 byte min alloc)
                        Q4Linear {
                            wq: client.empty(o * i * 2),
                            scales: client.empty(4),
                            out_dim: o,
                            in_dim: i,
                        }
                    }
                })
                .collect()
        })
        .collect()
}

// ---------------------------------------------------------------------------
// 3. main
// ---------------------------------------------------------------------------

/// Which burn backend the `burn f16 matmul` row runs on. `BHalf` is
/// `burn::backend::Metal`, which is an alias for `Wgpu`, so this is wgpu on
/// every host — it does NOT follow `cuda-backend`.
const BURN_BACKEND: &str = "burn-wgpu";

/// Which cubecl runtime the hand-written matvec rows run on. This one DOES
/// follow `cuda-backend` (see `mary::nn::q4::Rt`).
#[cfg(feature = "cuda-backend")]
const LANE_BACKEND: &str = "cubecl-cuda";
#[cfg(not(feature = "cuda-backend"))]
const LANE_BACKEND: &str = "cubecl-wgpu";

fn main() {
    let rounds: usize = std::env::var("Q4_ROUNDS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(5);
    let steps: usize = std::env::var("Q4_STEPS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(8);
    println!("moshi_q4_probe — q4 weight-quant matvec spike (PersonaPlex-7B temporal decode)");
    println!("{rounds} rounds x {steps} steps, min-of-medians\n");

    let client = q4::client_for_default_device();
    let device: <BHalf as BackendTypes>::Device = Default::default();

    // ---- correctness ----
    println!("CORRECTNESS (random f32 Linear, q4 round-trip through the GPU kernel):");
    correctness(&client, HIDDEN, HIDDEN, 1);
    correctness(&client, INTER, HIDDEN, 2);
    correctness(&client, HIDDEN, INTER, 3);

    // ---- benchmark ----
    let params = params_per_step();
    let f16_gb = params as f64 * 2.0 / 1e9;
    let q4_gb = params as f64 * 0.5625 / 1e9; // 4 bit + f16 scale / 32
    println!("\nBANDWIDTH (temporal decode shapes: 7 matvecs x {LAYERS} layers, M=1, chained):");
    println!(
        "  {:.2}G params/step  =>  f16 {:.2} GB/step, q4 {:.2} GB/step (4.5 bit/w)",
        params as f64 / 1e9,
        f16_gb,
        q4_gb
    );

    let c0 = control_op(&device);
    println!("  control matmul (256^2, f16): {c0:.2} ms");

    // burn f16 Tensor matmul (production baseline) — scoped so its 17 GB drop
    //
    // NAME THE BACKEND. This row runs `mary::nn::backend::BHalf`, which is
    // `burn::backend::Metal<f16>`, and burn's `Metal` is a plain alias for
    // `Wgpu` (burn-wgpu/src/lib.rs:110). `cuda-backend` swaps only the cubecl
    // q4 lane below and deliberately does not touch the burn dep, so on a CUDA
    // host this row is wgpu/Vulkan while the rows under it are cubecl-CUDA.
    // Unlabelled, that reads as a burn-CUDA-vs-hand-kernel comparison and is
    // not one — it cost exactly that misreading on 2026-08-22.
    let (burn_sub, burn_full) = {
        println!("  building burn f16 stack ({f16_gb:.1} GB on device)...");
        let r = bench_burn_f16(&device, rounds, steps);
        let c = control_op(&device);
        println!(
            "  burn f16 matmul [{BURN_BACKEND}] : submit {:7.2} ms  full {:7.2} ms   {:6.1} GB/s   (control {c:.2} ms)",
            r.0,
            r.1,
            f16_gb / (r.1 / 1e3)
        );
        r
    };

    // custom f16 matvec kernel — same kernel shape as q4, 3.56x the bytes
    let (f16k_sub, f16k_full) = {
        println!("  building custom-kernel f16 stack ({f16_gb:.1} GB on device)...");
        let layers = alloc_custom_layers(&client, false);
        let bufs = Bufs::new(&client);
        let r = bench(
            rounds,
            steps,
            || submit_custom_step(&client, &layers, &bufs, false),
            || {
                let _ = read_f32(&client, &bufs.x, HIDDEN);
            },
        );
        let c = control_op(&device);
        println!(
            "  f16 matvec kern [{LANE_BACKEND}] : submit {:7.2} ms  full {:7.2} ms   {:6.1} GB/s   (control {c:.2} ms)",
            r.0,
            r.1,
            f16_gb / (r.1 / 1e3)
        );
        r
    };

    // q4 dequant-in-kernel matvec
    let (q4_sub, q4_full) = {
        println!("  building q4 stack ({q4_gb:.1} GB on device)...");
        let layers = alloc_custom_layers(&client, true);
        let bufs = Bufs::new(&client);
        let r = bench(
            rounds,
            steps,
            || submit_custom_step(&client, &layers, &bufs, true),
            || {
                let _ = read_f32(&client, &bufs.x, HIDDEN);
            },
        );
        let c = control_op(&device);
        println!(
            "  q4 matvec kern  [{LANE_BACKEND}] : submit {:7.2} ms  full {:7.2} ms   {:6.1} GB/s effective ({:6.1} GB/s f16-equiv)   (control {c:.2} ms)",
            r.0,
            r.1,
            q4_gb / (r.1 / 1e3),
            f16_gb / (r.1 / 1e3)
        );
        r
    };
    let _ = (burn_sub, f16k_sub, q4_sub);

    // ---- frame projection ----
    // Constants from moshi_realtime_probe (2e6ae2e), M4 Max, raw Metal<f16>:
    let probe_temporal: [(usize, f64); 3] = [(256, 96.8), (1024, 144.7), (3000, 221.1)];
    const DEPTH_MS: f64 = 21.6; // measured 8-step depth frame
    const MIMI_MS: f64 = 5.0; // probe's mimi allowance
    const SUBMIT_LEVER_MS: f64 = 5.0; // megakernel-class submission budget (qwen3tts-proven)

    println!("\nFRAME PROJECTION (80 ms @ 12.5 Hz; depth {DEPTH_MS} ms + mimi ~{MIMI_MS} ms):");
    println!("  naive plug: probe temporal-full - burn-f16-matmul + q4-matmul (submission");
    println!("  overlap makes this pessimistic; KV-cat churn still included):");
    for (ctx, t) in probe_temporal {
        let temporal = t - burn_full + q4_full;
        let total = temporal + DEPTH_MS + MIMI_MS;
        let verdict = if total <= 80.0 {
            format!("CLEARS by {:.1} ms", 80.0 - total)
        } else {
            format!("OVER by {:.1} ms", total - 80.0)
        };
        println!(
            "    ctx {ctx:>4}: temporal {temporal:6.1} + depth {DEPTH_MS} + mimi {MIMI_MS} = {total:6.1} ms  ->  {verdict}"
        );
    }
    println!("  with-levers floor: q4 matmuls + true KV reads at measured bandwidth +");
    println!(
        "  megakernel-class submission ({SUBMIT_LEVER_MS} ms) + attention compute (~free at M=1):"
    );
    let eff_gbps = q4_gb / (q4_full / 1e3);
    for (ctx, _) in probe_temporal {
        // full-MHA KV: 32 layers x 2 (K+V) x ctx x 4096 x 2 B, f16
        let kv_gb = (LAYERS * 2 * ctx * HIDDEN * 2) as f64 / 1e9;
        let kv_ms = kv_gb / eff_gbps * 1e3;
        let total = q4_full + kv_ms + SUBMIT_LEVER_MS + DEPTH_MS + MIMI_MS;
        let verdict = if total <= 80.0 {
            format!("CLEARS by {:.1} ms", 80.0 - total)
        } else {
            format!("OVER by {:.1} ms", total - 80.0)
        };
        println!(
            "    ctx {ctx:>4}: q4 {q4_full:5.1} + kv {kv_ms:4.1} + submit {SUBMIT_LEVER_MS} + depth {DEPTH_MS} + mimi {MIMI_MS} = {total:6.1} ms  ->  {verdict}"
        );
    }
    println!(
        "\n  q4 vs f16 (same kernel shape): {:.2}x faster ({:.1} -> {:.1} ms); bytes ratio 3.56x",
        f16k_full / q4_full,
        f16k_full,
        q4_full
    );
}
