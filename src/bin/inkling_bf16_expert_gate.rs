//! `inkling_bf16_expert_gate` — is the native BF16 expert matmul right, against
//! Python?
//!
//! `bf16_mma_probe` established that CubeCL reaches `mma.sync…bf16` on sm_121a
//! and that one instruction is exact. This gates the thing the forward actually
//! calls — [`mary::models::inkling::bf16gemm::bf16_linear`] over a whole real
//! layer-2 expert, plus the de-interleave, the SiLU and the second GEMM — and it
//! gates it against a bundle **Python** wrote.
//!
//! That distinction is the point. Layer 2 is not like the MTP heads, which
//! `transformers` does not implement: it is an ordinary BF16 MoE layer that
//! `transformers` implements fully, so a Rust reference standing in for an
//! oracle would be a second transcription by the same author of the same
//! misreading. `golden/capture_inkling_bf16_experts.py` writes the bundle, and
//! it writes the BUDGETS too — this file does not get to pick the number it is
//! judged by.
//!
//! ## What is compared, and against which of the two references
//!
//! | check | against | what it can catch |
//! |---|---|---|
//! | the inbound cast | torch's own BF16 bits | a rounding mode that is not RNE |
//! | GEMM 1 | f64 over the same BF16 bits | any indexing, tiling or transpose error |
//! | the intermediate | torch's own BF16 bits | the gate/up split, the SiLU |
//! | GEMM 2, isolated | f64 over the CAPTURED intermediate | the second GEMM alone |
//! | the whole chain | f64 | the composition |
//! | the whole chain | `transformers` in BF16 | what the layer MEANS |
//! | the padded rows | zero | an m-tiling that writes outside the tokens |
//!
//! The f64 arbiter carries the tight budgets because both lanes multiply the
//! same BF16 bits, so nothing but the f32 accumulator can differ. The
//! `transformers` comparison is looser BY ARITHMETIC, not by convenience: its
//! chain rounds to BF16 at points this one does not, and the capture measures
//! that floor (4.3e-3) beside the budget (1e-2) so the headroom is visible
//! rather than asserted.
//!
//! ## Making it fail
//!
//! `INK_BF16_GATE_MUTATE=transpose|expert|halved` injects a deliberate mistake —
//! a transposed `w13` (square, so it loads without complaint), the neighbouring
//! expert's weights, or the HALVED reading of the gate/up split. A check that
//! has never failed is not evidence; each of these must be reported as FAIL.
//!
//! Build: `--features inkling-cuda,cuda-backend,import`
//! Run:   `inkling_bf16_expert_gate [<oracle dir>] [<checkpoint dir>]`

use std::path::PathBuf;

use anyhow::{Context, Result};
use cubecl::prelude::*;
use half::bf16;

use mary::models::inkling::bf16gemm::{bf16_linear_launch, upload_bf16_act, MTILE};
use mary::models::inkling::fp4gemm::gate_up_silu_bf16_launch;

type Rt = cubecl::cuda::CudaRuntime;

fn read_f32(p: &PathBuf) -> Result<Vec<f32>> {
    let b = std::fs::read(p).with_context(|| format!("reading {}", p.display()))?;
    Ok(b.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect())
}

fn read_f64(p: &PathBuf) -> Result<Vec<f64>> {
    let b = std::fs::read(p).with_context(|| format!("reading {}", p.display()))?;
    Ok(b.chunks_exact(8)
        .map(|c| f64::from_le_bytes([c[0], c[1], c[2], c[3], c[4], c[5], c[6], c[7]]))
        .collect())
}

fn read_u16(p: &PathBuf) -> Result<Vec<u16>> {
    let b = std::fs::read(p).with_context(|| format!("reading {}", p.display()))?;
    Ok(b.chunks_exact(2).map(|c| u16::from_le_bytes([c[0], c[1]])).collect())
}

/// Worst absolute difference over the reference tensor's own scale.
///
/// The metric every budget here is stated in. Per-element RELATIVE error is
/// meaningless on dot products that cancel — a sum of 4096 terms lands a few
/// times one term's magnitude from zero, so the denominator is nearly zero by
/// construction and every lane looks broken.
fn scaled(got: &[f32], reference: &[f64]) -> (f64, f64, usize) {
    let mut worst = 0.0f64;
    let mut scale = 0.0f64;
    let mut at = 0usize;
    for (i, (&g, &r)) in got.iter().zip(reference).enumerate() {
        scale = scale.max(r.abs());
        let d = (g as f64 - r).abs();
        if d > worst {
            worst = d;
            at = i;
        }
    }
    (worst / scale.max(1e-300), scale, at)
}

fn main() -> Result<()> {
    let oracle = std::env::args()
        .nth(1)
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("inkling_oracle_bf16"));
    let ckpt = std::env::args()
        .nth(2)
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("converted/inkling-small-complete.pile"));

    let man: serde_json::Value =
        serde_json::from_slice(&std::fs::read(oracle.join("bf16_manifest.json"))?)
            .context("parsing bf16_manifest.json")?;
    let layer = man["layer"].as_u64().context("layer")? as usize;
    let tokens = man["tokens"].as_u64().context("tokens")? as usize;
    let h = man["hidden"].as_u64().context("hidden")? as usize;
    let inter = man["inter"].as_u64().context("inter")? as usize;
    let experts: Vec<usize> = man["experts"]
        .as_array()
        .context("experts")?
        .iter()
        .map(|v| v.as_u64().unwrap() as usize)
        .collect();
    let b = &man["budgets"];
    let b_gemm1 = b["gemm1_vs_f64"].as_f64().context("budget gemm1")?;
    let b_gemm2 = b["gemm2_isolated_vs_f64"].as_f64().context("budget gemm2")?;
    let b_chain = b["chain_vs_f64"].as_f64().context("budget chain")?;
    let b_act = b["act_vs_torch"].as_f64().context("budget act")?;
    let b_flip = b["act_flip_fraction"].as_f64().context("budget flip")?;
    let b_tf = b["chain_vs_transformers"].as_f64().context("budget tf")?;

    let mutate = std::env::var("INK_BF16_GATE_MUTATE").unwrap_or_default();
    if !mutate.is_empty() {
        println!("!! MUTATED: {mutate} — this run is EXPECTED to fail\n");
    }
    if mutate == "halved" {
        // The gate/up split is read out of the environment by the launcher, so
        // the mutation is injected the same way the A/B always was.
        unsafe { std::env::set_var("INK_W13_HALVED", "1") };
    }

    println!(
        "oracle  : {} (torch {}, transformers {}, on {})",
        oracle.display(),
        man["torch"].as_str().unwrap_or("?"),
        man["transformers"].as_str().unwrap_or("?"),
        man["device"].as_str().unwrap_or("?")
    );
    println!("layer {layer}: {tokens} tokens, hidden {h}, inter {inter}, experts {experts:?}");
    println!(
        "budgets : gemm1 {b_gemm1:.3e}  gemm2 {b_gemm2:.3e}  chain {b_chain:.3e}  \
         transformers {b_tf:.3e}  (from the capture, not from here)"
    );
    println!(
        "          the capture measured transformers(bf16) vs its own f64 arbiter at {:.3e}\n",
        man["transformers_vs_f64_measured"].as_f64().unwrap_or(f64::NAN)
    );

    let x_f32 = read_f32(&oracle.join("bf16_x_f32.bin"))?;
    let x_bits = read_u16(&oracle.join("bf16_x_bf16.bin"))?;
    anyhow::ensure!(x_f32.len() == tokens * h, "x is {} f32", x_f32.len());
    anyhow::ensure!(x_bits.len() == tokens * h, "x bits are {}", x_bits.len());

    let src = mary::models::inkling::source::Weights::open(&ckpt, "inkling")?;
    let n13 = format!("model.llm.layers.{layer}.mlp.experts.w13_weight");
    let n2 = format!("model.llm.layers.{layer}.mlp.experts.w2_weight");
    anyhow::ensure!(!src.is_nvfp4(&n13), "{n13} is packed NVFP4; layer {layer} should be BF16");

    let client = Rt::client(&Default::default());

    // ---- how the slabs actually reach the device -------------------------
    // Not decoration: the MMA's operands are 32-bit vectors of two BF16, so
    // `Aliases::slice` refuses anything under 4-byte alignment, and layer 2's
    // w13 sits at file offset 434 335 018 -- 2 mod 4. So it is COPIED and w2
    // (offset 182 686 804, 0 mod 4) is aliased, every expert, every token. That
    // is the right call and not a bug -- an aliased 2-mod-4 pointer would make
    // every 32-bit operand load misaligned -- but it is 33.6 MB of copy per
    // expert, and reporting it here is what keeps it a known cost rather than a
    // mystery in the profile. Measured, not inferred: the alias is attempted.
    let aliases = mary::models::inkling::fp4gemm::Aliases::register(&client, src.mappings()?);

    let mut ok = true;

    for &e in &experts {
        // The weights the LANE will use. Under `expert` the neighbour's are
        // fetched while the reference stays expert `e`'s — the mistake a lane
        // makes when a routing index is off by one.
        let w_e = if mutate == "expert" { e + 1 } else { e };
        let w13 = src.expert_bf16(&n13, w_e)?;
        let w2 = src.expert_bf16(&n2, w_e)?;
        anyhow::ensure!(w13.rows == 2 * inter && w13.cols == h, "w13 is {}x{}", w13.rows, w13.cols);
        anyhow::ensure!(w2.rows == h && w2.cols == inter, "w2 is {}x{}", w2.rows, w2.cols);

        let w13_h = if mutate == "transpose" {
            // Square, so a transposed w13 loads without complaint and computes
            // nonsense — exactly the class of mistake shape checks cannot see.
            let src_b = &w13.bytes;
            let mut t = vec![0u8; src_b.len()];
            for r in 0..w13.rows {
                for c in 0..w13.cols {
                    let (from, to) = ((r * w13.cols + c) * 2, (c * w13.rows + r) * 2);
                    t[to] = src_b[from];
                    t[to + 1] = src_b[from + 1];
                }
            }
            client.create_from_slice(&t)
        } else {
            client.create_from_slice(&w13.bytes)
        };
        let w2_h = client.create_from_slice(&w2.bytes);

        println!("expert {e}{}:", if w_e != e { format!(" (weights from {w_e})") } else { String::new() });
        if let Some(al) = aliases.as_ref() {
            println!(
                "  zero-copy binding         : w13 {}, w2 {}",
                if al.slice(&w13.bytes).is_some() { "aliased" } else { "COPIED (alignment)" },
                if al.slice(&w2.bytes).is_some() { "aliased" } else { "COPIED (alignment)" },
            );
        }

        // ---- the inbound cast, bitwise -------------------------------------
        let (a_h, m_pad) = upload_bf16_act(&client, &x_f32, tokens, h);
        anyhow::ensure!(m_pad == tokens.div_ceil(MTILE) * MTILE, "m_pad {m_pad}");
        let a_back = client.read_one(a_h.clone()).expect("read a");
        let a_bits: Vec<u16> =
            a_back.chunks_exact(2).map(|c| u16::from_le_bytes([c[0], c[1]])).collect();
        let cast_bad = (0..tokens * h).filter(|&i| a_bits[i] != x_bits[i]).count();
        let pad_bad = (tokens * h..m_pad * h).filter(|&i| a_bits[i] != 0).count();
        println!(
            "  cast f32 -> bf16 vs torch : {} of {} bits differ, {} non-zero padded",
            cast_bad,
            tokens * h,
            pad_bad
        );
        ok &= cast_bad == 0 && pad_bad == 0;

        // ---- GEMM 1 --------------------------------------------------------
        let both_h = bf16_linear_launch(&client, &a_h, &w13_h, m_pad, h, 2 * inter);
        let both = f32::from_bytes(&client.read_one(both_h.clone()).expect("read both")).to_vec();
        let both_ref = read_f64(&oracle.join(format!("bf16_e{e}_both_f64.bin")))?;
        anyhow::ensure!(both_ref.len() == tokens * 2 * inter, "both ref is {}", both_ref.len());
        let (d1, s1, at1) = scaled(&both[..tokens * 2 * inter], &both_ref);
        println!(
            "  GEMM 1  x @ w13^T         : {d1:.3e}  (budget {b_gemm1:.3e}, |ref|max {s1:.4e}, worst at {at1})"
        );
        ok &= d1 <= b_gemm1;

        // ---- the intermediate, bitwise -------------------------------------
        let act_h = gate_up_silu_bf16_launch(&client, &both_h, m_pad, inter);
        let act_back = client.read_one(act_h.clone()).expect("read act");
        let act_bits: Vec<u16> =
            act_back.chunks_exact(2).map(|c| u16::from_le_bytes([c[0], c[1]])).collect();
        let act_ref = read_u16(&oracle.join(format!("bf16_e{e}_act_bf16.bin")))?;
        let mut act_diff = 0usize;
        let mut act_worst_ulp = 0i64;
        let mut act_absd = 0.0f64;
        let mut act_scale = 0.0f64;
        let mut act_at = (0.0f64, 0.0f64, 0i64);
        for i in 0..tokens * inter {
            let r = bf16::from_bits(act_ref[i]).to_f32();
            act_scale = act_scale.max(r.abs() as f64);
            if act_bits[i] != act_ref[i] {
                act_diff += 1;
                let d = (bf16::from_bits(act_bits[i]).to_f32() - r).abs();
                act_absd = act_absd.max(d as f64);
                // Distance in units of the reference's own last place.
                let ulp = (r.abs() * 2f32.powi(-8)).max(f32::MIN_POSITIVE);
                // Clamped: when a mutation makes the reference element 1e-13
                // and the lane 282, "ulp" is not a number worth printing.
                let u = ((d / ulp).round() as i64).min(9999);
                if u > act_worst_ulp {
                    act_worst_ulp = u;
                    act_at = (r as f64, d as f64, u);
                }
            }
        }
        // Not a bitwise check, and deliberately not an ulp check either. The
        // device's SiLU consumes an f32 accumulator whose error is ABSOLUTE
        // (eps*sqrt(K) of the whole tensor's scale) while a bf16 ulp is
        // RELATIVE, so it is the small elements that reround, by many ulps and
        // by nothing in absolute terms. The criteria are therefore the same
        // scaled-deviation metric as everywhere else, plus a bound on HOW MANY
        // elements moved -- a halved gate/up split or the wrong expert moves
        // essentially all of them, which is what this check is for.
        println!(
            "  silu(g)*u -> bf16 vs torch: {} of {} differ ({:.2e}), worst {} ulp              (at |act|={:.3e}, abs {:.3e}); max abs / |act|max = {:.3e}              (budgets: fraction {:.1e}, scaled {:.1e})",
            act_diff,
            tokens * inter,
            act_diff as f64 / (tokens * inter) as f64,
            act_worst_ulp,
            act_at.0.abs(),
            act_at.1,
            act_absd / act_scale.max(1e-300),
            b_flip,
            b_act
        );
        let act_scaled = act_absd / act_scale.max(1e-300);
        let act_frac = act_diff as f64 / (tokens * inter) as f64;
        ok &= act_scaled <= b_act && act_frac <= b_flip;

        // ---- GEMM 2, isolated on the CAPTURED intermediate ------------------
        let mut act_pad = vec![0u8; m_pad * inter * 2];
        for i in 0..tokens * inter {
            act_pad[2 * i..2 * i + 2].copy_from_slice(&act_ref[i].to_le_bytes());
        }
        let iso_a = client.create_from_slice(&act_pad);
        let iso_h = bf16_linear_launch(&client, &iso_a, &w2_h, m_pad, inter, h);
        let iso = f32::from_bytes(&client.read_one(iso_h).expect("read iso")).to_vec();
        let y_ref = read_f64(&oracle.join(format!("bf16_e{e}_y_f64.bin")))?;
        anyhow::ensure!(y_ref.len() == tokens * h, "y ref is {}", y_ref.len());
        let (d2, s2, at2) = scaled(&iso[..tokens * h], &y_ref);
        println!(
            "  GEMM 2  act @ w2^T        : {d2:.3e}  (budget {b_gemm2:.3e}, |ref|max {s2:.4e}, worst at {at2})"
        );
        ok &= d2 <= b_gemm2;

        // ---- the whole chain ------------------------------------------------
        let y_h = bf16_linear_launch(&client, &act_h, &w2_h, m_pad, inter, h);
        let y = f32::from_bytes(&client.read_one(y_h).expect("read y")).to_vec();
        let (d3, _, at3) = scaled(&y[..tokens * h], &y_ref);
        println!("  chain vs f64              : {d3:.3e}  (budget {b_chain:.3e}, worst at {at3})");
        ok &= d3 <= b_chain;

        let tf = read_f32(&oracle.join(format!("bf16_e{e}_y_tf_f32.bin")))?;
        let tf64: Vec<f64> = tf.iter().map(|&v| v as f64).collect();
        let (d4, _, at4) = scaled(&y[..tokens * h], &tf64);
        println!("  chain vs transformers     : {d4:.3e}  (budget {b_tf:.3e}, worst at {at4})");
        ok &= d4 <= b_tf;

        // ---- the padded rows -------------------------------------------------
        let pad_max = y[tokens * h..m_pad * h].iter().fold(0.0f32, |a, v| a.max(v.abs()));
        println!("  padded rows ({:2})          : max |value| {pad_max:e}", m_pad - tokens);
        ok &= pad_max == 0.0;
    }

    if !ok {
        println!("\nFAIL");
        std::process::exit(1);
    }
    println!(
        "\nPASS — the native BF16 expert lane reproduces what Python computes, \
         within the budgets Python wrote down"
    );
    Ok(())
}
