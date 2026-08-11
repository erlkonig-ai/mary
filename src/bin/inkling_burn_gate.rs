//! Parity gate for the Inkling Burn lane against `transformers`.
//!
//! Covers every stage that runs on the device: RMSNorm, the short convolution,
//! attention (local and global), one routed expert's feed-forward, the shared
//! experts, the dense MLP, the NVFP4 decode, and the gate/up de-interleave.
//!
//! The reference is PYTHON, read from the oracle bundles that
//! `golden/capture_inkling_*.py` dump. Those are raw f32 arrays of
//! transformers' own outputs, so they gate any implementation -- there is no
//! reason to compare Burn against the slice lane and inherit a second hop.
//!
//! It is also the slow hop: the slice lane is scalar CPU at ~95s a forward
//! while the Python reference is GPU-accelerated. Gating against the dumps
//! keeps the iteration loop in seconds.
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
//! Attention is gated TWICE for this reason: once on the toy capture, whose
//! configuration is chosen so that every branch engages, and once at hidden
//! 4096 over 109 tokens, where the branches are inert but the matmul is real.
//!
//! MEASURED 2026-08-11 on the GB10. ndarray passes everything; CUDA fails
//! exactly the matmul-bearing checks, and the shape of the failure is the
//! finding:
//!
//! ```text
//!                      ndarray     cuda
//!   rms_norm            2.2e-7    2.2e-7   passes
//!   short_conv          8.1e-8    8.1e-8   passes
//!   nvfp4 dequant        exact     exact   bitwise
//!   expert_ffn          2.2e-6    4.6e-4   FAILS on cuda
//!   dense_mlp           4.7e-6    5.5e-4   FAILS on cuda
//!   shared_experts      2.0e-7    3.3e-4   FAILS on cuda
//!   attention (toy)     2.0e-7    5.4e-4   FAILS on cuda
//!   attention (real)    2.4e-6    4.5e-4   FAILS on cuda
//! ```
//!
//! Every ported stage lands at the same 3-6e-4 on CUDA and passes on ndarray,
//! which places the discrepancy in the backend's matmul and not in any of the
//! ports. The elementwise checks — RMSNorm, the short convolution, the FP4
//! decode — are unaffected on both.
//!
//! Only the matmul-bearing checks fail, all at the same magnitude, while
//! RMSNorm under the identical metric passes at 1.1e-6 — so this is not a
//! cancellation artifact and not the elementwise path. About 4.3e-4 relative is
//! roughly eleven bits of mantissa, which is what a TF32 tensor-core matmul
//! gives. CONFIRMED by the discriminator this gate now runs first: 2049 has a
//! 12-bit significand, so it is exact in f32 and absent from the TF32 grid,
//! whose neighbours here are 2048 and 2050. ndarray returns 2049; CUDA returns
//! 2050. The inputs are being reduced to an 11-bit significand.
//!
//! (2050 rather than 2048 because 2049 is an exact tie and NVIDIA's
//! `cvt.rna.tf32.f32` rounds ties away from zero. The first version of this
//! check predicted 2048 from a ties-to-even assumption and misclassified the
//! result as "neither"; the classifier now tests grid membership instead of one
//! guessed value.)
//!
//! Finding the switch to force full f32 accumulate is open work, but it is no
//! longer a mystery where the switch would go. `cubek-matmul`'s
//! `definition::blueprint::adjust_dtypes` rewrites the stage and register
//! element types from f32 to tf32 — and *only* when the chosen tile matmul
//! reports `requires_accelerator()`. The unit routines (`SimpleUnitAlgorithm`,
//! `DoubleUnitAlgorithm`) do not, so they keep f32 end to end. What is missing
//! is a way to ask for one: burn's `MatmulStrategy` offers `Cube` (which passes
//! `cubek`'s `Strategy::Auto`) and `Autotune` (which picks by speed, and the
//! accelerator is faster), and neither can name a routine. The fix belongs
//! upstream in burn as a precision knob on the matmul, not here.
//!
//! The budget is deliberately NOT widened to accommodate it. A GPU lane that
//! silently carries eleven mantissa bits is a fact worth failing over, and the
//! whole point of gating the Burn lane against `transformers` is to surface
//! exactly this before anything depends on it.
//!
//! What it costs end to end, measured on the 109-token forward: 106 of 109
//! positions keep the same argmax against the all-host run, the three that move
//! are near-ties inside the top five, and the prompt's own continuation is
//! unchanged. So this is a real loss of precision with a small consequence —
//! which is a statement about this prompt, not a licence to stop measuring.
//!
//!   cargo run --release --features inkling-burn --bin inkling_burn_gate -- <oracle>
//!   cargo run --release --features inkling-cuda --bin inkling_burn_gate -- cuda <oracle>

use burn::prelude::*;
use burn::tensor::{Tensor, TensorData};


const BUDGET: f32 = 1e-5;

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

fn read_f32(p: &std::path::Path) -> Vec<f32> {
    let b = std::fs::read(p).unwrap_or_else(|e| panic!("reading {}: {e}", p.display()));
    assert!(!b.is_empty(), "{} is empty -- the gate would be vacuous", p.display());
    b.chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

/// One of the capture scripts' manifests.
fn manifest(oracle: &std::path::Path, name: &str) -> serde_json::Value {
    let p = oracle.join(name);
    let s = std::fs::read_to_string(&p).unwrap_or_else(|e| panic!("reading {}: {e}", p.display()));
    serde_json::from_str(&s).unwrap_or_else(|e| panic!("parsing {}: {e}", p.display()))
}

fn mu(v: &serde_json::Value, k: &str) -> usize {
    v.get(k)
        .and_then(|x| x.as_u64())
        .unwrap_or_else(|| panic!("manifest has no unsigned {k}")) as usize
}

fn mf(v: &serde_json::Value, k: &str) -> f64 {
    v.get(k)
        .and_then(|x| x.as_f64())
        .unwrap_or_else(|| panic!("manifest has no number {k}"))
}

fn run<B: Backend>(dev: &B::Device, oracle: &std::path::Path, label: &str) -> (usize, usize) {
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
    let (h, inter, dense_inter) = (4096usize, 2048usize, 16384usize);
    println!("\n=== {label}: hidden {h}, inter {inter}, dense {dense_inter} ===");

    // Inputs come from the oracle so the comparison is against the same numbers
    // transformers saw, not merely the same shapes.
    let x = read_f32(&oracle.join("blk_rms_x.bin"));
    let gain = read_f32(&oracle.join("blk_rms_w.bin"));
    let eps = 1e-6f64;
    let tokens = x.len() / h;

    // ---- precision discriminator ------------------------------------------
    // Exact in f32, unrepresentable in TF32. Reports the matmul's input
    // precision outright rather than inferring it from an error magnitude.
    {
        let a: Tensor<B, 2> =
            Tensor::from_data(TensorData::new(vec![2049.0f32, 1.0], [1, 2]), dev);
        let b: Tensor<B, 2> =
            Tensor::from_data(TensorData::new(vec![1.0f32, 0.0], [2, 1]), dev);
        let got = a.matmul(b).into_data().convert::<f32>().to_vec::<f32>().unwrap()[0];
        checks += 1;
        // Membership in the TF32 grid, not equality with a guessed value. Near
        // 2049 an 11-bit significand can represent 2048 and 2050 and nothing
        // between, and which of the two a tie lands on is the rounding mode's
        // business, not ours: cvt.rna.tf32.f32 rounds ties away from zero and
        // gives 2050, while ties-to-even would give 2048. Both are TF32.
        let on_tf32_grid = got == 2048.0 || got == 2050.0;
        let bits = if got == 2049.0 {
            "f32 (24-bit significand)"
        } else if on_tf32_grid {
            "TF32 (11-bit significand) — inputs ARE being truncated"
        } else {
            "neither f32 nor the TF32 grid — unexpected, investigate"
        };
        println!("  matmul precision: 2049*1 -> {got}  => {bits}");
        if got != 2049.0 {
            println!("    this is the cause of any matmul-only budget failure below");
        }
    }

    // ---- RMSNorm ----------------------------------------------------------
    let mine = {
        let g: Tensor<B, 1> = Tensor::from_data(TensorData::new(gain.clone(), [h]), dev);
        mary::models::inkling::burn::rms_norm(t2::<B>(&x, tokens, h, dev), g, eps)
            .into_data()
            .convert::<f32>()
            .to_vec::<f32>()
            .unwrap()
    };
    let theirs = read_f32(&oracle.join("blk_rms_y.bin"));
    report("rms_norm vs python", cmp(&mine, &theirs), &mut checks, &mut fails);

    // ---- one expert's feed-forward ----------------------------------------
    let gu = read_f32(&oracle.join("bop_expert_gate_up.bin"));
    let dn = read_f32(&oracle.join("bop_expert_down.bin"));
    let xb = read_f32(&oracle.join("bop_x.bin"));
    let tok_b = xb.len() / h;
    let mine = mary::models::inkling::burn::expert_ffn(
        t2::<B>(&xb, tok_b, h, dev),
        t2::<B>(&gu, 2 * inter, h, dev),
        t2::<B>(&dn, h, inter, dev),
    )
    .into_data()
    .convert::<f32>()
    .to_vec::<f32>()
    .unwrap();
    let theirs = read_f32(&oracle.join("bop_expert_y.bin"));
    report("expert_ffn vs python", cmp(&mine, &theirs), &mut checks, &mut fails);

    // ---- NVFP4 dequant on device ------------------------------------------
    // Same oracle the CPU decode was gated on: compressed_tensors' own answer.
    {
        let man = std::fs::read_to_string(oracle.join("nvfp4_manifest.json")).unwrap();
        let numf = |k: &str| -> usize {
            let at = man.find(&format!("\"{k}\"")).unwrap();
            man[at..].split(':').nth(1).unwrap()
                .chars().skip_while(|c| c.is_whitespace())
                .take_while(|c| c.is_ascii_digit()).collect::<String>().parse().unwrap()
        };
        let experts = numf("experts");
        let rws = numf("rows");
        let bpr = numf("bytes_per_row");
        let rows = experts * rws;
        let codes_b = std::fs::read(oracle.join("nvfp4_codes.bin")).unwrap();
        let scal_b = std::fs::read(oracle.join("nvfp4_scale_e4m3.bin")).unwrap();
        let s2 = read_f32(&oracle.join("nvfp4_scale2_f32.bin"));
        let want = read_f32(&oracle.join("nvfp4_expected_f32.bin"));
        let nsc = scal_b.len() / rows;

        let ci: Vec<i32> = codes_b.iter().map(|&b| b as i32).collect();
        let si: Vec<i32> = scal_b.iter().map(|&b| b as i32).collect();
        let codes: Tensor<B, 2, burn::tensor::Int> =
            Tensor::from_data(TensorData::new(ci, [rows, bpr]), dev);
        let scales: Tensor<B, 2, burn::tensor::Int> =
            Tensor::from_data(TensorData::new(si, [rows, nsc]), dev);
        // scale2 is per EXPERT; every row of an expert shares it.
        let per_row: Vec<f32> = (0..rows).map(|r| s2[r / rws]).collect();
        let s2t: Tensor<B, 1> = Tensor::from_data(TensorData::new(per_row, [rows]), dev);

        let got = mary::models::inkling::burn::dequant_nvfp4(codes, scales, s2t)
            .into_data().convert::<f32>().to_vec::<f32>().unwrap();
        let d = cmp(&got, &want);
        checks += d.n;
        println!("  nvfp4 dequant vs compressed_tensors: {} values, worst abs {:e}, rel {:e}",
                 d.n, d.abs, d.rel);
        if d.n == 0 {
            println!("    FAIL  compared nothing");
            fails += 1;
        } else if d.abs != 0.0 {
            println!("    NOTE  not bitwise; both sides multiply the same exact values in the");
            println!("          same order, so the bar here is 0 and this is a real difference");
            if d.scaled() > BUDGET {
                println!("    FAIL  and over budget");
                fails += 1;
            }
        } else {
            println!("    bitwise identical");
        }
    }

    // ---- dense MLP ---------------------------------------------------------
    let g = read_f32(&oracle.join("bop_dense_gate.bin"));
    let u = read_f32(&oracle.join("bop_dense_up.bin"));
    let d = read_f32(&oracle.join("bop_dense_down.bin"));
    let gs = 1.7f32;
    let mine = mary::models::inkling::burn::dense_mlp(
        t2::<B>(&xb, tok_b, h, dev),
        t2::<B>(&g, dense_inter, h, dev),
        t2::<B>(&u, dense_inter, h, dev),
        t2::<B>(&d, h, dense_inter, dev),
        gs,
    )
    .into_data()
    .convert::<f32>()
    .to_vec::<f32>()
    .unwrap();
    let theirs = read_f32(&oracle.join("bop_dense_y.bin"));
    report("dense_mlp vs python", cmp(&mine, &theirs), &mut checks, &mut fails);

    // ---- short convolution -------------------------------------------------
    // At the real width, from `InklingShortConvolution` itself. The oracle also
    // carries the residual-free convolution, so the gate can say WHICH of the
    // two an implementation matches instead of reporting "wrong numbers".
    {
        let m = manifest(oracle, "blk_manifest.json");
        let k = mu(&m, "kernel");
        let xs = read_f32(&oracle.join("blk_sconv_x.bin"));
        let ws = read_f32(&oracle.join("blk_sconv_w.bin"));
        let ys = read_f32(&oracle.join("blk_sconv_y.bin"));
        let pure = read_f32(&oracle.join("blk_sconv_y_noresid.bin"));
        let toks = xs.len() / h;
        let mine = mary::models::inkling::burn::short_conv(
            t2::<B>(&xs, toks, h, dev),
            t2::<B>(&ws, h, k, dev),
        )
        .into_data()
        .convert::<f32>()
        .to_vec::<f32>()
        .unwrap();
        report("short_conv vs python", cmp(&mine, &ys), &mut checks, &mut fails);
        let d = cmp(&mine, &pure);
        checks += 1;
        println!("  short_conv against the RESIDUAL-FREE convolution: {:e}", d.scaled());
        if d.scaled() <= BUDGET {
            println!("    FAIL  indistinguishable from dropping the module's own residual");
            fails += 1;
        }
    }

    // ---- the device gate/up split ------------------------------------------
    // A permutation, so the bar is bitwise. The host side is gated against the
    // real checkpoint by `inkling_real_gate`; this is the two lanes agreeing,
    // which is the only claim a permutation can be wrong about.
    {
        let (rows, cols) = (2 * 12usize, 7usize);
        let ramp: Vec<f32> = (0..rows * cols).map(|i| i as f32).collect();
        let (hg, hu) = mary::models::inkling::load::deinterleave_rows(&ramp, rows, cols);
        let (dg, du) = mary::models::inkling::burn::split_shared_fused(
            t2::<B>(&ramp, rows, cols, dev),
            1,
        );
        let dg = dg.into_data().convert::<f32>().to_vec::<f32>().unwrap();
        let du = du.into_data().convert::<f32>().to_vec::<f32>().unwrap();
        checks += dg.len() + du.len();
        let bad = dg.iter().zip(&hg).filter(|(a, b)| a != b).count()
            + du.iter().zip(&hu).filter(|(a, b)| a != b).count();
        println!("  split_shared_fused vs load::deinterleave_rows: {} values, {bad} differ",
                 dg.len() + du.len());
        if bad != 0 {
            println!("    FAIL  the device split and the host split disagree");
            fails += 1;
        }
    }

    // ---- shared experts ----------------------------------------------------
    // Against `InklingMoE.shared_experts` run on its own, which the capture
    // dumps separately from the routed half precisely so the two cannot be
    // confused for each other.
    {
        let m = manifest(oracle, "lyr_manifest.json");
        let (tok, hid) = (mu(&m, "tokens"), mu(&m, "hidden"));
        let inter = mu(&m, "moe_intermediate");
        let n_shared = mu(&m, "n_shared");
        let x = read_f32(&oracle.join("lyr_x.bin"));
        let g = read_f32(&oracle.join("lyr_moe_shared_experts_gate_proj.bin"));
        let up = read_f32(&oracle.join("lyr_moe_shared_experts_up_proj.bin"));
        let dn = read_f32(&oracle.join("lyr_moe_shared_experts_down_proj.bin"));
        let gam = read_f32(&oracle.join("lyr_moe_gammas.bin"));
        let want = read_f32(&oracle.join("lyr_moe_shared.bin"));
        let routed = read_f32(&oracle.join("lyr_moe_routed.bin"));
        let mine = mary::models::inkling::burn::shared_experts(
            t2::<B>(&x, tok, hid, dev),
            t2::<B>(&g, n_shared * inter, hid, dev),
            t2::<B>(&up, n_shared * inter, hid, dev),
            t2::<B>(&dn, n_shared * hid, inter, dev),
            t2::<B>(&gam, tok, n_shared, dev),
            n_shared,
        )
        .into_data()
        .convert::<f32>()
        .to_vec::<f32>()
        .unwrap();
        report("shared_experts vs python", cmp(&mine, &want), &mut checks, &mut fails);
        // Non-vacuity: the shared half must not be a rounding error beside the
        // routed half, or matching it would prove nothing about either.
        let scale = |v: &[f32]| v.iter().fold(0f32, |a, b| a.max(b.abs()));
        checks += 1;
        println!("  shared |max| {:e} against routed |max| {:e}", scale(&want), scale(&routed));
        if scale(&want) < 0.05 * scale(&routed) {
            println!("    FAIL  the shared half is negligible here; the check is near-vacuous");
            fails += 1;
        }
    }

    // ---- attention ---------------------------------------------------------
    // Twice over: the toy capture, where every branch engages, and the real
    // capture at hidden 4096, where a backend matmul's blocking and input
    // precision do.
    for (man_name, prefix, label, kinds_must_differ) in [
        ("attn_manifest.json", "attn_", "toy config, every branch live", true),
        ("areal_manifest.json", "areal_", "real config, hidden 4096", false),
    ] {
        let m = manifest(oracle, man_name);
        let tok = mu(&m, "tokens");
        let hid = mu(&m, "hidden");
        let d_rel = mu(&m, "d_rel");
        let kernel = mu(&m, "kernel");
        let eps = mf(&m, "rms_norm_eps");
        let window = mu(&m, "sliding_window");
        let ls = mary::models::inkling::attn::LogScaling {
            n_floor: mf(&m, "log_scaling_n_floor") as f32,
            alpha: mf(&m, "log_scaling_alpha") as f32,
        };
        let x = read_f32(&oracle.join(format!("{prefix}x.bin")));
        assert_eq!(x.len(), tok * hid, "{prefix}x.bin is not [tokens, hidden]");
        println!("\n  -- attention: {label} ({tok} tokens, hidden {hid}) --");

        for tag in ["local", "global"] {
            let l = &m["layers"][tag];
            let heads = mu(l, "num_heads");
            let kv_heads = mu(l, "num_kv_heads");
            let head_dim = mu(l, "head_dim");
            let rel_extent = mu(l, "rel_extent");
            let is_local = tag == "local";
            let p = |n: &str| oracle.join(format!("{prefix}{tag}_{n}"));

            let dims = mary::models::inkling::attn::AttnDims {
                hidden: hid,
                heads,
                kv_heads,
                head_dim,
                d_rel,
                rel_extent,
                kernel,
                rms_eps: eps,
                kind: if is_local {
                    mary::models::inkling::config::AttnKind::Local
                } else {
                    mary::models::inkling::config::AttnKind::Global
                },
            };
            checks += 1;
            if (dims.scaling() as f64 - mf(l, "scaling")).abs() > 1e-9 {
                println!("    FAIL  scaling {} != reference {}", dims.scaling(), mf(l, "scaling"));
                fails += 1;
            }

            let load = |n: &str, r: usize, c: usize| t2::<B>(&read_f32(&p(n)), r, c, dev);
            let w = mary::models::inkling::burn::AttnWeightsDev::<B> {
                wq: load("wq.bin", heads * head_dim, hid),
                wk: load("wk.bin", kv_heads * head_dim, hid),
                wv: load("wv.bin", kv_heads * head_dim, hid),
                wr: load("wr.bin", heads * d_rel, hid),
                wo: load("wo.bin", hid, heads * head_dim),
                k_sconv: load("k_sconv.bin", kv_heads * head_dim, kernel),
                v_sconv: load("v_sconv.bin", kv_heads * head_dim, kernel),
                q_norm: Tensor::<B, 1>::from_data(
                    TensorData::new(read_f32(&p("q_norm.bin")), [head_dim]), dev),
                k_norm: Tensor::<B, 1>::from_data(
                    TensorData::new(read_f32(&p("k_norm.bin")), [head_dim]), dev),
                rel_proj: load("rel_proj.bin", d_rel, rel_extent),
            };
            let mask = mary::models::inkling::attn::causal_mask(
                tok, if is_local { Some(window) } else { None });
            let run_kind = |w: &mary::models::inkling::burn::AttnWeightsDev<B>,
                            d: &mary::models::inkling::attn::AttnDims,
                            mask: &[f32]| {
                mary::models::inkling::burn::attention(
                    t2::<B>(&x, tok, hid, dev), w, d, Some(ls), t2::<B>(mask, tok, tok, dev),
                )
                .into_data()
                .convert::<f32>()
                .to_vec::<f32>()
                .unwrap()
            };
            let mine = run_kind(&w, &dims, &mask);
            let theirs = read_f32(&p("y.bin"));
            report(&format!("attention {tag} vs python"), cmp(&mine, &theirs), &mut checks, &mut fails);

            // Non-vacuity: on the toy corpus the same weights under the OTHER
            // kind must disagree, or the `kind` argument is untested.
            //
            // On the real config at this length they CANNOT disagree, and that
            // is a fact about the model rather than a weakness of the gate: the
            // window is 512 and both relative tables reach at least 512, so at
            // 109 tokens neither bites, and `log_scaling_n_floor` is 128000, so
            // tau is exactly 1. A local and a global layer are the same
            // function until the sequence is long enough to tell them apart.
            // Demanding a difference here would only teach the gate to lie.
            let other = mary::models::inkling::attn::AttnDims {
                kind: if is_local {
                    mary::models::inkling::config::AttnKind::Global
                } else {
                    mary::models::inkling::config::AttnKind::Local
                },
                ..dims
            };
            let other_mask = mary::models::inkling::attn::causal_mask(
                tok, if is_local { None } else { Some(window) });
            let flipped = run_kind(&w, &other, &other_mask);
            let fd = cmp(&flipped, &theirs);
            checks += 1;
            println!("    same weights under the OTHER kind: {:e}{}", fd.scaled(),
                     if kinds_must_differ { "  <- must differ" } else { "  (cannot differ at this length)" });
            if kinds_must_differ && fd.scaled() <= BUDGET {
                println!("    FAIL  the two kinds are indistinguishable here");
                fails += 1;
            }
        }
    }

    (checks, fails)
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let want_cuda = args.iter().any(|a| a == "cuda");
    let oracle = std::path::PathBuf::from(
        args.iter().find(|a| a.starts_with('/')).cloned()
            .unwrap_or_else(|| "./inkling-oracle".to_string()));
    println!("  python oracle: {}", oracle.display());
    println!("=== Inkling Burn lane vs the f32 slice lane ===");
    println!("  the slice lane is itself gated against transformers, so this needs no torch");
    println!("  budget: worst-absolute-over-scale {BUDGET:e}, written down first");

    #[allow(unused_mut)]
    let mut total = (0usize, 0usize);

    #[cfg(feature = "inkling-cuda")]
    if want_cuda {
        type C = burn::backend::Cuda<f32>;
        let (c, f) = run::<C>(&Default::default(), &oracle, "cuda");
        total = (total.0 + c, total.1 + f);
    }
    #[cfg(not(feature = "inkling-cuda"))]
    if want_cuda {
        println!("  (cuda requested but this build has no inkling-cuda feature)");
    }

    if !want_cuda {
        type N = burn::backend::NdArray<f32>;
        let (c, f) = run::<N>(&Default::default(), &oracle, "ndarray");
        total = (total.0 + c, total.1 + f);
    }

    println!("\n=== verdict ===");
    println!("  checks: {}", total.0);
    if total.1 == 0 {
        println!("GATE PASSED — {} checks, the Burn lane matches python", total.0);
    } else {
        println!("GATE FAILED — {} checks, {} FAILURES", total.0, total.1);
        std::process::exit(1);
    }
}
