//! `inkling_fp4_expert_gate` — is the native NVFP4 expert matmul right?
//!
//! `nvfp4_mma_probe` established that CubeCL reaches the instruction and gets a
//! 16x8x4096 product right. This gates the thing the forward actually calls:
//! [`mary::models::inkling::fp4gemm::fp4_linear`] over a whole real expert
//! (N = 4096 output rows, K = 4096), driven from
//! [`mary::models::inkling::source::Weights`], with activations quantised
//! by the same host recipe the device kernel implements.
//!
//! ## What it is compared against, and why not the f32 lane
//!
//! Not against the existing decode-then-f32 lane. That lane's activations are
//! f32, so it computes a DIFFERENT quantity — the checkpoint specifies E2M1
//! activations in dynamic per-16 blocks (`hf_quant_config.json`,
//! `*input_quantizer`), which is what this path feeds the tensor cores.
//! Comparing the two would blend "is the kernel correct" with "what does
//! 4-bit activation quantisation cost", and those need separate answers.
//!
//! So the reference here accumulates the SAME quantised operands in f64. That
//! isolates the kernel: any disagreement is the kernel's, not the format's.
//! `nvfp4_mma_probe` showed the tensor core lands within 1.973e-6 of an f64 sum
//! over K=4096 while an f32 host lane over the same products is 3.705e-5 out,
//! so the tolerance here is set at 1e-4 relative — comfortably above the
//! former and still far below anything a real indexing bug could hide under.
//!
//! Batching every token of an expert into one matmul reassociates the sums, so
//! this is deliberately NOT a bitwise gate.
//!
//! Build: `--features cuda-backend,inkling`

use anyhow::{Context, Result};
use cubecl::prelude::*;

use mary::models::inkling::fp4gemm::{
    fp4_linear_launch, gate_up_silu_launch, quantize_act_host, upload_quantized_act,
    GROUP, MTILE,
};
use mary::models::inkling::nvfp4::{e4m3_to_f32, FP4_E2M1};

type Rt = cubecl::cuda::CudaRuntime;

const LAYER: usize = 10;

/// Decode packed NVFP4 rows to f32 (the audited scalar path).
fn decode(codes: &[u8], scales: &[u8], rows: usize, k: usize, scale2: f32) -> Vec<f32> {
    let mut out = vec![0f32; rows * k];
    for r in 0..rows {
        for j in 0..k {
            let byte = codes[r * (k / 2) + j / 2];
            let c = if j % 2 == 0 { byte & 0x0F } else { byte >> 4 };
            out[r * k + j] =
                FP4_E2M1[c as usize] * e4m3_to_f32(scales[r * (k / GROUP) + j / GROUP]) * scale2;
        }
    }
    out
}

fn main() -> Result<()> {
    let dir = mary::paths::model(
        std::env::args().nth(1).as_deref(),
        "inkling-small-complete.pile",
    )?;

    let src = mary::models::inkling::source::Weights::open(&dir, "inkling")?;
    let b13 = format!("model.llm.layers.{LAYER}.mlp.experts.w13_weight");
    anyhow::ensure!(src.is_nvfp4(&b13), "{b13} is not packed NVFP4");

    let w13 = src.expert_packed(&b13, 0)?;
    let (n, k) = (w13.rows, w13.cols * 2);
    println!("expert 0 of {b13}: N={n}  K={k}  scale2={:e}", w13.scale2);

    // ---- a real activation ------------------------------------------------
    // Real decoded expert rows, not synthetic values: the quantiser's whole job
    // is to cope with a real dynamic range, and uniform test data would not
    // exercise the E4M3 block-scale rounding at all.
    let tokens = 5usize;
    let probe = src.expert_packed(&b13, 7)?;
    let x = decode(&probe.codes, &probe.scales, tokens, k, probe.scale2);

    let client = Rt::client(&Default::default());

    // ---- device -----------------------------------------------------------
    let (a_h, a_sc_h, m_pad) = upload_quantized_act(&client, &x, tokens, k);
    let b_h = client.create_from_slice(&w13.codes);
    let b_sc_h = client.create_from_slice(&w13.scales);
    let out_h = fp4_linear_launch(&client, &a_h, &a_sc_h, &b_h, &b_sc_h, m_pad, k, n, w13.scale2);
    let got = f32::from_bytes(&client.read_one(out_h).expect("read")).to_vec();
    println!("launched fp4_linear: m_pad={m_pad}  {} planes", (n / 8) * (m_pad / MTILE));

    // ---- f64 reference over the SAME quantised operands --------------------
    let mut padded = vec![0f32; m_pad * k];
    padded[..tokens * k].copy_from_slice(&x);
    let (a_codes, a_scales) = quantize_act_host(&padded, k);
    let a_deq = decode(&a_codes, &a_scales, tokens, k, 1.0);
    let b_deq = decode(&w13.codes, &w13.scales, n, k, w13.scale2);

    let mut worst = 0.0f64;
    let mut worst_at = (0usize, 0usize);
    let mut checked = 0usize;
    // Every token row, and a stride over the 4096 output columns: the full
    // 5x4096 f64 reference is 84 M products, which is a minute of host time for
    // no extra confidence over a strided sample of the same kernel.
    for r in 0..tokens {
        for c in (0..n).step_by(37) {
            let mut s = 0.0f64;
            for j in 0..k {
                s += a_deq[r * k + j] as f64 * b_deq[c * k + j] as f64;
            }
            let g = got[r * n + c] as f64;
            anyhow::ensure!(g.is_finite(), "non-finite at ({r},{c})");
            let e = (g - s).abs() / s.abs().max(1e-12);
            if e > worst {
                worst = e;
                worst_at = (r, c);
            }
            checked += 1;
        }
    }
    println!("fp4_linear vs f64 over the same quantised operands:");
    println!("  max relative error {worst:.3e} at {worst_at:?}   ({checked} dot products of K={k})");

    // ---- what the padded rows did -----------------------------------------
    // Rows past `tokens` are zero-padding; if the kernel wrote anything but
    // zero there, the m-tiling is wrong in a way the sampled rows would miss.
    let mut pad_max = 0.0f32;
    for r in tokens..m_pad {
        for c in 0..n {
            pad_max = pad_max.max(got[r * n + c].abs());
        }
    }
    println!("  max |value| in the {} zero-padded rows: {pad_max:e}", m_pad - tokens);

    // ---- the FFN chain -----------------------------------------------------
    let b2name = format!("model.llm.layers.{LAYER}.mlp.experts.w2_weight");
    let w2 = src.expert_packed(&b2name, 0)?;
    let inter = n / 2;
    anyhow::ensure!(w2.cols * 2 == inter, "w2 K={} but inter={inter}", w2.cols * 2);

    let act_h = gate_up_silu_launch(&client, &out_h_clone(&client, &got, m_pad, n), m_pad, inter);
    let act = f32::from_bytes(&client.read_one(act_h).expect("read")).to_vec();

    let mut ref_act = vec![0f32; m_pad * inter];
    for r in 0..tokens {
        for i in 0..inter {
            let g = got[r * n + 2 * i];
            let u = got[r * n + 2 * i + 1];
            ref_act[r * inter + i] = (g / (1.0 + (-g).exp())) * u;
        }
    }
    let mut act_err = 0.0f32;
    for r in 0..tokens {
        for i in 0..inter {
            let d = (act[r * inter + i] - ref_act[r * inter + i]).abs();
            let s = ref_act[r * inter + i].abs().max(1e-6);
            act_err = act_err.max(d / s);
        }
    }
    println!("  gate_up_silu (deinterleave + SiLU) max rel err: {act_err:.3e}");

    let ok = worst <= 1e-4 && pad_max == 0.0 && act_err <= 1e-5;
    if !ok {
        println!("FAIL");
        std::process::exit(1);
    }
    println!("PASS — the native NVFP4 expert matmul reproduces an f64 sum of the same operands");
    Ok(())
}

/// Re-upload a host copy of the fused result (the gate reads it back to build
/// its own reference, so the device copy has already been consumed).
fn out_h_clone(
    client: &cubecl::prelude::ComputeClient<Rt>,
    got: &[f32],
    m_pad: usize,
    n: usize,
) -> cubecl::server::Handle {
    assert_eq!(got.len(), m_pad * n);
    client.create_from_slice(f32::as_bytes(got))
}
