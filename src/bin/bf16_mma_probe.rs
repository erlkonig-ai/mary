//! `bf16_mma_probe` — can CubeCL emit the **unscaled BF16** tensor-core MMA on
//! sm_121a, and does it compute the right answer on real Inkling layer-2
//! weights?
//!
//! Same question `nvfp4_mma_probe` asked of the block-scaled FP4 instruction,
//! asked before anything is built on the answer. It is worth asking separately:
//! `mma.sync…bf16` is a different instruction with a different k (16, not 64),
//! reached through a different constructor
//! ([`cmma::MmaDefinition::new`] rather than `new_scaled`), and CubeCL's CUDA
//! backend registers it from `arch >= 80` — a declaration this file turns into a
//! measurement.
//!
//! This is a REACHABILITY probe, not the correctness authority. It launches the
//! real kernel [`mary::models::inkling::bf16gemm::bf16_linear`] — there is no
//! second, probe-only transcription of the tiling to disagree with — against an
//! f64 accumulation of the same BF16 values, which is what both f32 lanes are
//! approximating. Whether the LANE computes the right thing (the right expert,
//! the right gate/up split, the right transpose) is `inkling_bf16_expert_gate`'s
//! question, and it is answered against Python.
//!
//! Build: `--features inkling-cuda,cuda-backend,import`
//! Run:   `bf16_mma_probe [<checkpoint dir>]`

use anyhow::{bail, Result};
use cubecl::prelude::*;
use half::bf16;

use mary::models::inkling::bf16gemm::{bf16_linear_launch, KTILE, MTILE, NTILE};

type Rt = cubecl::cuda::CudaRuntime;

/// The BF16 layer. Not a parameter: the point is to probe the instruction on the
/// bytes it will actually be fed.
const LAYER: usize = 2;

/// Decode a `[rows, k]` BF16 slab's row `r` to f32 — the exact values, since
/// every BF16 is an f32 with a zeroed mantissa tail.
fn row(bytes: &[u8], r: usize, k: usize) -> Vec<f32> {
    (0..k)
        .map(|j| {
            let o = (r * k + j) * 2;
            bf16::from_le_bytes([bytes[o], bytes[o + 1]]).to_f32()
        })
        .collect()
}

fn main() -> Result<()> {
    let dir = mary::paths::model(
        std::env::args().nth(1).as_deref(),
        "inkling-small-complete.pile",
    )?;

    let (m, n) = (MTILE, NTILE);

    let client = Rt::client(&Default::default());

    // --- is the BF16 combination even registered? -------------------------
    // Matched by inspection of the advertised set, exactly as the NVFP4 probe
    // does, so this does not depend on the public path of the config type.
    let props = client.properties();
    let registered = props.features.matmul.mma.iter().any(|c| {
        c.a_type == bf16::cube_type()
            && c.b_type == bf16::cube_type()
            && c.cd_type == f32::cube_type()
            && c.m == m as u32
            && c.n == n as u32
            && c.k == KTILE as u32
    });
    println!(
        "CubeCL reports the BF16 combination (bf16 x bf16 -> f32, m{m}n{n}k{KTILE}): {}",
        if registered { "REGISTERED" } else { "NOT registered" }
    );
    if !registered {
        bail!(
            "CubeCL does not advertise the unscaled BF16 MMA on this device. \
             Do NOT widen to f32 to work around this -- report it."
        );
    }

    // --- real layer-2 expert rows -----------------------------------------
    let src = mary::models::inkling::source::Weights::open(&dir)?;
    let base = format!("model.llm.layers.{LAYER}.mlp.experts.w13_weight");
    if src.is_nvfp4(&base) {
        bail!("{base} is packed NVFP4 -- layer {LAYER} is supposed to be the BF16 one");
    }
    // VIEWS into the pile's mapping, not copies: 33.6 MB an expert, and this
    // probe reads 24 rows of them.
    let e0 = src.expert_bf16(&base, 0)?;
    let e7 = src.expert_bf16(&base, 7)?;
    let k = e0.cols;
    println!(
        "{base}: experts are [{}, {k}] BF16 -- A = 16 rows of expert 7, B = 8 rows of expert 0",
        e0.rows
    );

    // A is 16 rows of one expert and B is 8 rows of another: both operands are
    // genuine checkpoint BF16 with a genuine dynamic range, which synthetic
    // values would not have.
    let a_ref: Vec<Vec<f32>> = (0..m).map(|r| row(&e7.bytes, r, k)).collect();
    let b_ref: Vec<Vec<f32>> = (0..n).map(|r| row(&e0.bytes, r, k)).collect();

    // --- correctness ------------------------------------------------------
    // Against an f64 sum of the same BF16 values, not against another f32 lane:
    // a bare GPU-vs-CPU-f32 delta cannot tell "the MMA is wrong" from "the two
    // summed in a different order".
    //
    // TWO metrics, because one of them lies. Per-element RELATIVE error is
    // meaningless on a dot product that cancels -- two weight rows of 4096
    // terms sum to a few times one term's magnitude, so the denominator is
    // nearly zero by construction and every lane looks terrible. The metric the
    // gates in this repo use is worst-absolute over the TENSOR's scale, and
    // that is the one the budget is on.
    //
    // A K sweep separates the two failure modes: an indexing bug is already
    // wrong at K = 16 (one instruction, nothing accumulated), while f32
    // accumulation drift grows like sqrt(K). Beside it, a host lane that sums
    // in the MMA's OWN order -- exact 16-term tiles added into an f32
    // accumulator -- which is what the instruction is specified to do, so
    // agreeing with it is the positive statement that nothing but rounding is
    // left.
    let mut all_ok = true;
    for &k_use in &[KTILE, 64usize, 512, k] {
        if k_use > k {
            continue;
        }
        // Rows of the SUB-matrix, not a flat prefix of the full-width one.
        let sub = |rows: &Vec<Vec<f32>>| -> Vec<u8> {
            let mut v = Vec::with_capacity(rows.len() * k_use * 2);
            for r in rows {
                for &x in &r[..k_use] {
                    v.extend_from_slice(&bf16::from_f32(x).to_le_bytes());
                }
            }
            v
        };
        let ah = client.create_from_slice(&sub(&a_ref));
        let bh = client.create_from_slice(&sub(&b_ref));
        let oh = bf16_linear_launch(&client, &ah, &bh, m, k_use, n);
        let got = f32::from_bytes(&client.read_one(oh).expect("read")).to_vec();

        let mut gpu_abs = 0.0f64;
        let mut cpu_abs = 0.0f64;
        let mut tile_abs = 0.0f64;
        let mut scale = 0.0f64;
        let mut gpu_rel = 0.0f64;
        let mut worst_at = (0usize, 0usize);
        for i in 0..m {
            for j in 0..n {
                let mut s32 = 0.0f32;
                let mut s64 = 0.0f64;
                let mut stile = 0.0f32;
                for t in 0..k_use / KTILE {
                    let mut inner = 0.0f64;
                    for l in t * KTILE..(t + 1) * KTILE {
                        s32 += a_ref[i][l] * b_ref[j][l];
                        s64 += a_ref[i][l] as f64 * b_ref[j][l] as f64;
                        inner += a_ref[i][l] as f64 * b_ref[j][l] as f64;
                    }
                    stile += inner as f32;
                }
                let g = got[i * n + j] as f64;
                if !g.is_finite() {
                    println!("NON-FINITE at ({i},{j}) — FAIL");
                    std::process::exit(1);
                }
                scale = scale.max(s64.abs());
                gpu_abs = gpu_abs.max((g - s64).abs());
                cpu_abs = cpu_abs.max((s32 as f64 - s64).abs());
                tile_abs = tile_abs.max((stile as f64 - s64).abs());
                let r = (g - s64).abs() / s64.abs().max(1e-12);
                if r > gpu_rel {
                    gpu_rel = r;
                    worst_at = (i, j);
                }
            }
        }
        let gpu_scaled = gpu_abs / scale;
        // f32 half-ulp is 2^-24; a random walk of it over K products is
        // eps*sqrt(K) of the sum's own magnitude. That is the number this
        // instruction cannot beat and should not badly miss.
        let predicted = 2f64.powi(-24) * (k_use as f64).sqrt();
        println!(
            "K = {k_use:5} ({:3} tiles)  |ref|max {scale:.4e}   worst abs / tensor scale:\n\
             \x20   GPU BF16 MMA        {gpu_scaled:.3e}   (eps*sqrt(K) predicts {predicted:.3e})\n\
             \x20   host f32, 16-tiled  {:.3e}   (the MMA's own summation order)\n\
             \x20   host f32, flat sum  {:.3e}\n\
             \x20   worst per-element relative (cancellation, not error): {gpu_rel:.3e} at {worst_at:?}",
            k_use / KTILE,
            tile_abs / scale,
            cpu_abs / scale,
        );
        // Budget: 4 * eps * sqrt(K). Derived rather than picked -- it is the
        // f32 accumulator's own random walk with a factor of four for the
        // worst of 128 dot products, and it is orders of magnitude under the
        // O(1) error any indexing or transpose mistake produces.
        if !(gpu_scaled <= 4.0 * predicted) {
            all_ok = false;
            println!("    -> over budget ({:.3e})", 4.0 * predicted);
        }
    }

    if !all_ok {
        println!("FAIL — CubeCL's BF16 MMA did not reproduce the reference");
        std::process::exit(1);
    }
    println!(
        "PASS — CubeCL emits and executes the unscaled BF16 tensor-core MMA on real \
         Inkling layer-{LAYER} expert weights"
    );
    Ok(())
}
