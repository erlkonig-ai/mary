//! `inkling_attn_bf16_gate` — the attention layer with its five projections as
//! the BF16 the checkpoint stores, measured against the `transformers` f32
//! reference at the widths the forward actually runs.
//!
//! ## What this gate is for
//!
//! Moving `wq/wk/wv/wr/wo` off the f32 device lane and onto `mma.sync…bf16` is
//! not bit-exact against the f32 lane and cannot be. It was once refused for
//! exactly that, which is the wrong test: agreeing with a previous
//! implementation says which one was written first, not which one is right.
//! The right test is a DELTA against the reference, inside a budget stated
//! before it is measured, and that is this binary.
//!
//! ## The corpus, and what it does and does not cover
//!
//! `golden/capture_inkling_attn_real.py` writes `areal_*` into the oracle
//! directory: the input, every weight, and the output `y` of
//! `transformers`' own `InklingAttention`, at hidden 4096 / 32 heads / 8 KV
//! heads / head_dim 128 over 109 positions.
//!
//! **Its weights are randomly initialised, not the checkpoint's.** The script
//! reads `config.json` for the SHAPES and then fills every parameter from
//! `normal_(std = 1/sqrt(fan_in))`, with the input `randn` at unit scale. So
//! this is a reference for the ARITHMETIC at the real widths — which is what a
//! tensor-core matmul can get wrong — and it is not a claim about the trained
//! model's activation distribution. The capture's own docstring says as much,
//! and two branches are inert here by construction: `log_scaling_n_floor` is
//! 128000 so tau is exactly 1 at 109 tokens, and `rel_extent` and
//! `sliding_window_size` both exceed the sequence, so neither the out-of-range
//! relative bias nor the window ever fires. `inkling_attn_gate`'s deliberately
//! shrunken corpus is what tests those.
//!
//! ## Three numbers, because one of them is a control
//!
//! The reference ran in f32 (`torch.set_default_dtype(torch.float32)`), so the
//! BF16 arm here pays a rounding on the WEIGHTS that production does not: the
//! pile already stores them as BF16 and `transformers` runs the released model
//! in BF16 as well. This gate rounds f32 weights down to BF16 to feed the MMA,
//! so its BF16 figure is a conservative upper bound on the change's cost, not
//! an estimate of it.
//!
//! The f32 host lane (`attn::attention`, untouched by this change) is measured
//! against the same reference in the same run, so the report separates what
//! the port already cost from what BF16 adds.
//!
//! Build: `--features inkling-cuda,cuda-backend,import`
//! Run:   `inkling_attn_bf16_gate [<oracle dir>]`

use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use burn::tensor::{Tensor, TensorData};
use cubecl::cuda::CudaRuntime;
use cubecl::prelude::ComputeClient;
use half::bf16;

use mary::models::inkling::attn::{AttnDims, AttnWeights, LogScaling, attention, causal_mask};
use mary::models::inkling::bf16gemm::Bf16W;
use mary::models::inkling::burn as dev_lane;
use mary::models::inkling::config::AttnKind;
use mary::models::inkling::seam::{Bk, client_of};

/// Relative L2 budget for the BF16 lane against the f32 reference.
///
/// Stated here, before the measurement, and derived rather than tuned:
///
/// * BF16 keeps 8 explicit mantissa bits, so rounding an operand is a relative
///   perturbation of at most `2^-9 = 1.95e-3` and about `2^-9/sqrt(3) = 1.1e-3`
///   RMS.
/// * Both operands are rounded — the weight and, on the device, the activation
///   — and the two are independent, so about `1.6e-3` RMS goes into each
///   product. A `k = 4096` dot product does not amplify that: numerator and
///   denominator both grow as `sqrt(k)`, so the relative error of the result
///   stays at the per-term figure.
/// * A layer chains four such projections (q/k/v, then o) around a softmax,
///   which is contractive. Call it `2-3e-3` expected.
///
/// `5e-3` leaves room for that without reaching the `1e-2` the whole 42-layer
/// stack is held to. A measurement past this is a defect to find, not a number
/// to move.
const BUDGET: f64 = 5e-3;

fn read_f32(dir: &Path, name: &str) -> Result<Vec<f32>> {
    let b = std::fs::read(dir.join(name)).with_context(|| format!("reading {name}"))?;
    anyhow::ensure!(
        b.len() % 4 == 0,
        "{name} is {} bytes, not a whole f32 count",
        b.len()
    );
    Ok(b.chunks_exact(4)
        .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
        .collect())
}

/// One top-level number out of the manifest.
fn num(man: &str, key: &str) -> Result<f64> {
    let pat = format!("\"{key}\":");
    let i = man
        .find(&pat)
        .with_context(|| format!("{key} missing from the manifest"))?;
    let rest = &man[i + pat.len()..];
    let end = rest
        .find(|c: char| {
            !(c.is_ascii_digit()
                || c == '.'
                || c == '-'
                || c == '+'
                || c == 'e'
                || c == 'E'
                || c.is_whitespace())
        })
        .unwrap_or(rest.len());
    rest[..end]
        .trim()
        .parse()
        .with_context(|| format!("{key} is not a number"))
}

/// Relative L2 and worst absolute of `got` against `want`.
fn err(got: &[f32], want: &[f32]) -> (f64, f64, f64) {
    assert_eq!(
        got.len(),
        want.len(),
        "{} values against {}",
        got.len(),
        want.len()
    );
    let mut num = 0f64;
    let mut den = 0f64;
    let mut worst = 0f64;
    for (&g, &w) in got.iter().zip(want) {
        let d = (g as f64 - w as f64).abs();
        num += d * d;
        den += (w as f64) * (w as f64);
        worst = worst.max(d);
    }
    (num.sqrt() / den.sqrt().max(1e-30), worst, den.sqrt())
}

/// An `[n, k]` f32 slab as the BF16 the MMA takes.
///
/// `bf16::from_f32` is round-to-nearest-even, the same rounding
/// `torch.Tensor.to(torch.bfloat16)` performs and the same one the device cast
/// in `bf16gemm::to_bf16` performs, so the operand BITS here are the operand
/// bits a production run multiplies.
fn bf16w(client: &ComputeClient<CudaRuntime>, v: &[f32], n: usize, k: usize) -> Bf16W {
    assert_eq!(v.len(), n * k, "{n}x{k} is not {} values", v.len());
    assert!(Bf16W::tileable(n, k), "{n}x{k} does not tile as m16n8k16");
    let mut bytes = Vec::with_capacity(v.len() * 2);
    for &f in v {
        bytes.extend_from_slice(&bf16::from_f32(f).to_le_bytes());
    }
    Bf16W {
        h: client.create_from_slice(&bytes),
        n,
        k,
        align: 16,
    }
}

#[allow(clippy::too_many_arguments)]
fn one(
    dir: &Path,
    tag: &str,
    tokens: usize,
    hidden: usize,
    kernel: usize,
    eps: f64,
    ls: LogScaling,
    window: usize,
    checks: &mut usize,
    fails: &mut usize,
) -> Result<()> {
    let is_local = tag == "local";
    let p = |s: &str| format!("areal_{tag}_{s}.bin");

    let x = read_f32(dir, "areal_x.bin")?;
    let wq = read_f32(dir, &p("wq"))?;
    let wk = read_f32(dir, &p("wk"))?;
    let wv = read_f32(dir, &p("wv"))?;
    let wr = read_f32(dir, &p("wr"))?;
    let wo = read_f32(dir, &p("wo"))?;
    let ks = read_f32(dir, &p("k_sconv"))?;
    let vs = read_f32(dir, &p("v_sconv"))?;
    let qn = read_f32(dir, &p("q_norm"))?;
    let kn = read_f32(dir, &p("k_norm"))?;
    let rp = read_f32(dir, &p("rel_proj"))?;
    let y = read_f32(dir, &p("y"))?;

    // Shapes come out of the files rather than being asserted from the
    // manifest: a dump that disagrees with the config is exactly the failure
    // this would otherwise discover as a plausible wrong answer.
    let head_dim = qn.len();
    let heads = wq.len() / hidden / head_dim;
    let kv_heads = wk.len() / hidden / head_dim;
    let d_rel = wr.len() / hidden / heads;
    let rel_extent = rp.len() / d_rel;
    anyhow::ensure!(x.len() == tokens * hidden, "x is not [{tokens}, {hidden}]");
    anyhow::ensure!(y.len() == tokens * hidden, "y is not [{tokens}, {hidden}]");
    anyhow::ensure!(
        wo.len() == hidden * heads * head_dim,
        "wo is not [{hidden}, {}]",
        heads * head_dim
    );
    anyhow::ensure!(
        ks.len() == kv_heads * head_dim * kernel,
        "k_sconv is not [{}, {kernel}]",
        kv_heads * head_dim
    );

    println!(
        "\n=== {tag} : heads {heads} kv {kv_heads} head_dim {head_dim} d_rel {d_rel} rel_extent {rel_extent} ==="
    );

    let dims = AttnDims {
        hidden,
        heads,
        kv_heads,
        head_dim,
        d_rel,
        rel_extent,
        kernel,
        rms_eps: eps,
        kind: if is_local {
            AttnKind::Local
        } else {
            AttnKind::Global
        },
    };
    let mask = causal_mask(tokens, if is_local { Some(window) } else { None });

    // ---- control: the f32 host lane, which this change does not touch ----
    let hw = AttnWeights {
        wq: &wq,
        wk: &wk,
        wv: &wv,
        wr: &wr,
        wo: &wo,
        k_sconv: &ks,
        v_sconv: &vs,
        q_norm: &qn,
        k_norm: &kn,
        rel_proj: &rp,
    };
    let host = attention(&x, &hw, &dims, Some(ls), &mask, tokens);
    let (h_l2, h_worst, scale) = err(&host, &y);

    // ---- the arm under test: the device lane, projections in BF16 --------
    let dev = Default::default();
    let xt: Tensor<Bk, 2> = Tensor::from_data(TensorData::new(x.clone(), [tokens, hidden]), &dev);
    let client = client_of(&xt);
    let t2 = |v: &[f32], r: usize, c: usize| -> Tensor<Bk, 2> {
        Tensor::from_data(TensorData::new(v.to_vec(), [r, c]), &dev)
    };
    let dw = dev_lane::AttnWeightsDev {
        wq: bf16w(&client, &wq, heads * head_dim, hidden),
        wk: bf16w(&client, &wk, kv_heads * head_dim, hidden),
        wv: bf16w(&client, &wv, kv_heads * head_dim, hidden),
        wr: bf16w(&client, &wr, heads * d_rel, hidden),
        wqkvr: None,
        wo: bf16w(&client, &wo, hidden, heads * head_dim),
        k_sconv: t2(&ks, kv_heads * head_dim, kernel),
        v_sconv: t2(&vs, kv_heads * head_dim, kernel),
        q_norm: Tensor::from_data(TensorData::new(qn.clone(), [head_dim]), &dev),
        k_norm: Tensor::from_data(TensorData::new(kn.clone(), [head_dim]), &dev),
        rel_proj: t2(&rp, d_rel, rel_extent),
    };
    let devy = dev_lane::attention(
        xt,
        &dw,
        &dims,
        Some(ls),
        if is_local { Some(window) } else { None },
    )
    .into_data()
    .to_vec::<f32>()
    .expect("device attention output");
    let (d_l2, d_worst, _) = err(&devy, &y);
    let (gap_l2, _, _) = err(&devy, &host);

    println!("  |y| (L2)                       : {scale:e}");
    println!(
        "  f32 host lane   vs reference   : rel L2 {h_l2:e}   worst abs {h_worst:e}   <- control"
    );
    println!(
        "  BF16 device lane vs reference  : rel L2 {d_l2:e}   worst abs {d_worst:e}   <- the criterion"
    );
    println!("  BF16 device lane vs f32 host   : rel L2 {gap_l2:e}   (what the change moved)");
    println!("  budget                         : {BUDGET:e}");

    *checks += 1;
    if d_l2 > BUDGET {
        println!("  FAIL  the BF16 lane is {d_l2:e} from the reference, past the budget");
        *fails += 1;
    } else {
        println!("  PASS  inside budget");
    }

    // Non-vacuity: a lane that ignored its weights would also be "close" if the
    // reference happened to be small. It is not, and the control proves the
    // corpus discriminates.
    *checks += 1;
    if scale <= 0.0 || !d_l2.is_finite() {
        println!("  FAIL  degenerate corpus or non-finite output");
        *fails += 1;
    }
    Ok(())
}

fn main() -> Result<()> {
    let dir = mary::paths::model(std::env::args().nth(1).as_deref(), "inkling-oracle")?;
    let man = String::from_utf8(std::fs::read(dir.join("areal_manifest.json"))?)?;
    let tokens = num(&man, "tokens")? as usize;
    let hidden = num(&man, "hidden")? as usize;
    let kernel = num(&man, "kernel")? as usize;
    let eps = num(&man, "rms_norm_eps")?;
    let window = num(&man, "sliding_window")? as usize;
    let ls = LogScaling {
        n_floor: num(&man, "log_scaling_n_floor")? as f32,
        alpha: num(&man, "log_scaling_alpha")? as f32,
    };

    println!("=== inkling attention, projections as stored BF16, vs the f32 reference ===");
    println!("  oracle    : {}", dir.display());
    println!("  corpus    : {tokens} tokens, hidden {hidden}, kernel {kernel}, window {window}");
    println!("  provenance: RANDOM weights at the checkpoint's SHAPES (see the module docs)");
    println!(
        "  inert here: log scaling (n_floor {}) and the out-of-range relative branch",
        ls.n_floor
    );

    let (mut checks, mut fails) = (0usize, 0usize);
    for tag in ["local", "global"] {
        one(
            &dir,
            tag,
            tokens,
            hidden,
            kernel,
            eps,
            ls,
            window,
            &mut checks,
            &mut fails,
        )?;
    }

    println!("\n=== {checks} checks, {fails} failures ===");
    if fails > 0 {
        std::process::exit(1);
    }
    Ok(())
}
