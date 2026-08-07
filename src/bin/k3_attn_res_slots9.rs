//! ADVERSARIAL REVIEW probe — AttnRes at NINE slots, and the full 93-layer schedule.
//!
//! The shipped gate's oracle is a 13-layer prefix. Its bank never exceeds two
//! entries, so `AttnResParams::mix` has only ever been exercised at <= 3 slots
//! and `DepthMixer` has only ever been driven across 2 of the 8 boundary
//! crossings. Production is 9 slots and 93 layers. This binary closes that gap
//! from a second oracle (`slots9_oracle.npz`) built by running the SHIPPED
//! `_apply_attn_res` — extracted verbatim from the checkpoint's own
//! `modeling_kimi_linear.py` — on real late-layer weights and real activations.

use std::path::PathBuf;

use anyhow::{Context, Result};
use burn::prelude::*;

use mary::models::k3::attn_res::{round_bf16, stack_candidates, AttnResParams, DepthMixer};
use mary::models::k3::config::K3Config;
use mary::nn::npz::Npz;

const TOKENS: usize = 32;
const HIDDEN: usize = 7168;
const SLOTS: usize = 9;
/// Same budget the shipped gate uses for a bfloat16 mixture output.
const OUT_REL: f64 = 0.003_906_25;
const SCORE_RTOL: f64 = 1e-5;
const PROB_ATOL: f64 = 1e-6;

fn bf16_bits(x: f32) -> u16 {
    let b = x.to_bits();
    if x.is_nan() {
        return ((b >> 16) as u16) | 0x0040;
    }
    let lsb = (b >> 16) & 1;
    ((b.wrapping_add(0x7FFF + lsb)) >> 16) as u16
}

fn bf16_to_f32(b: u16) -> f32 {
    f32::from_bits((b as u32) << 16)
}

fn worse(a: f64, b: f64) -> f64 {
    if a.is_nan() || b.is_nan() {
        f64::NAN
    } else if b > a {
        b
    } else {
        a
    }
}

fn max_abs_diff(what: &str, a: &[f32], b: &[f32]) -> f64 {
    assert!(!a.is_empty(), "{what}: EMPTY");
    assert_eq!(a.len(), b.len(), "{what}: len {} vs {}", a.len(), b.len());
    let mut m = 0.0f64;
    let mut nan = false;
    for (x, y) in a.iter().zip(b) {
        let d = (*x as f64) - (*y as f64);
        if d.is_nan() {
            nan = true;
        } else if d.abs() > m {
            m = d.abs();
        }
    }
    if nan {
        f64::NAN
    } else {
        m
    }
}

fn max_abs(what: &str, a: &[f32]) -> f64 {
    assert!(!a.is_empty(), "{what}: EMPTY");
    let mut m = 0.0f64;
    for x in a {
        let v = (*x as f64).abs();
        if v.is_nan() {
            return f64::NAN;
        }
        if v > m {
            m = v;
        }
    }
    m
}

fn max_rel_diff(what: &str, got: &[f32], want: &[f32]) -> f64 {
    let s = max_abs(what, want);
    let d = max_abs_diff(what, got, want);
    if s == 0.0 {
        d
    } else {
        d / s
    }
}

fn bits_equal(a: &[f32], b: &[f32]) -> bool {
    a.len() == b.len() && a.iter().zip(b).all(|(x, y)| x.to_bits() == y.to_bits())
}

fn differing_bf16(got: &[f32], want: &[f32]) -> usize {
    got.iter().zip(want).filter(|(g, w)| bf16_bits(**g) != bf16_bits(**w)).count()
}

/// float64 transcription of the same formula — attribution, not a second opinion.
fn mix_f64(v: &[f32], sw: &[f32], eps: f64, slots: usize) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    let mut scores = vec![0.0f32; TOKENS * slots];
    let mut probs = vec![0.0f32; TOKENS * slots];
    let mut out = vec![0.0f32; TOKENS * HIDDEN];
    for t in 0..TOKENS {
        let mut sc = vec![0.0f64; slots];
        for s in 0..slots {
            let row = &v[(t * slots + s) * HIDDEN..(t * slots + s + 1) * HIDDEN];
            let mut sq = 0.0f64;
            for x in row {
                sq += (*x as f64) * (*x as f64);
            }
            let scale = 1.0 / (sq / HIDDEN as f64 + eps).sqrt();
            let mut acc = 0.0f64;
            for (x, w) in row.iter().zip(sw) {
                acc += (*x as f64) * scale * (*w as f64);
            }
            sc[s] = acc;
            scores[t * slots + s] = acc as f32;
        }
        let m = sc.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        let mut z = 0.0f64;
        let mut p = vec![0.0f64; slots];
        for s in 0..slots {
            p[s] = (sc[s] - m).exp();
            z += p[s];
        }
        for s in 0..slots {
            p[s] /= z;
            probs[t * slots + s] = p[s] as f32;
        }
        for h in 0..HIDDEN {
            let mut acc = 0.0f64;
            for s in 0..slots {
                acc += p[s] * (v[(t * slots + s) * HIDDEN + h] as f64);
            }
            out[t * HIDDEN + h] = bf16_to_f32(bf16_bits(acc as f32));
        }
    }
    (scores, probs, out)
}

fn t1<B: Backend>(v: &[f32], dev: &Device<B>) -> Tensor<B, 1> {
    Tensor::from_data(TensorData::new(v.to_vec(), [v.len()]), dev)
}

fn t2<B: Backend>(v: &[f32], r: usize, c: usize, dev: &Device<B>) -> Tensor<B, 2> {
    assert_eq!(v.len(), r * c);
    Tensor::from_data(TensorData::new(v.to_vec(), [r, c]), dev)
}

fn host<B: Backend, const D: usize>(t: Tensor<B, D>) -> Vec<f32> {
    t.into_data().to_vec::<f32>().expect("host readback")
}

struct G {
    fails: Vec<String>,
    passed: usize,
}

impl G {
    fn le(&mut self, id: &str, got: f64, max: f64, unit: &str) {
        let ok = !(got > max) && !got.is_nan();
        self.rec(id, ok, format!("{got:.4e} {unit} (budget {max:.4e})"));
    }
    fn truth(&mut self, id: &str, ok: bool, d: impl Into<String>) {
        self.rec(id, ok, d.into());
    }
    fn rec(&mut self, id: &str, ok: bool, d: String) {
        if ok {
            self.passed += 1;
            println!("    ok   {id:<44} {d}");
        } else {
            println!("    FAIL {id:<44} {d}");
            self.fails.push(format!("{id}: {d}"));
        }
    }
}

fn bfarr(z: &Npz, key: &str) -> Vec<f32> {
    let v: Vec<f32> = z.get(key).bf16_to_f64().into_iter().map(|x| x as f32).collect();
    assert!(!v.is_empty(), "{key}: EMPTY");
    v
}

fn run_lane<B: Backend>(lane: &str, dev: &Device<B>, z: &Npz, cfg: &K3Config) -> bool {
    println!("\n=== lane: {lane} ===");
    let mut g = G { fails: Vec::new(), passed: 0 };
    let eps = cfg.text_config.rms_norm_eps;

    for tag in ["L84_mlp", "L92_sa", "MODEL_output"] {
        let slots = z.get(&format!("{tag}_nslots")).scalar() as usize;
        g.truth(&format!("{tag}.slots"), slots == SLOTS, format!("{slots} slots"));

        let v_ref = bfarr(z, &format!("{tag}_v_bf16bits"));
        let bank_flat = bfarr(z, &format!("{tag}_block_residual_bf16bits"));
        let prefix = bfarr(z, &format!("{tag}_prefix_sum_bf16bits"));
        let sw = z.get(&format!("{tag}_score_weight")).to_f32();
        let ref_scores = z.get(&format!("{tag}_scores")).to_f32();
        let ref_probs = z.get(&format!("{tag}_probs")).to_f32();
        let ref_out = bfarr(z, &format!("{tag}_out_bf16bits"));

        // Rebuild the bank as [tokens, hidden] slices; the npz stores [T, 8, H].
        let nb = slots - 1;
        let bank: Vec<Tensor<B, 2>> = (0..nb)
            .map(|s| {
                let mut o = Vec::with_capacity(TOKENS * HIDDEN);
                for t in 0..TOKENS {
                    let off = t * nb * HIDDEN + s * HIDDEN;
                    o.extend_from_slice(&bank_flat[off..off + HIDDEN]);
                }
                t2::<B>(&o, TOKENS, HIDDEN, dev)
            })
            .collect();

        let v = stack_candidates(&bank, t2::<B>(&prefix, TOKENS, HIDDEN, dev));
        g.truth(
            &format!("{tag}.stack_bitexact_9slots"),
            v.dims() == [TOKENS, SLOTS, HIDDEN] && bits_equal(&host(v.clone()), &v_ref),
            format!("{:?}", v.dims()),
        );

        let p = AttnResParams::<B>::new(
            t1::<B>(&sw, dev), // score weight is the product; feed it as norm with a ones proj
            Tensor::<B, 2>::ones([1, HIDDEN], dev),
            eps,
        );
        g.truth(
            &format!("{tag}.score_weight_roundtrip"),
            bits_equal(&host(p.score_weight()), &sw),
            "port's premultiplied weight == the shipped product",
        );

        let m = p.mix(v);
        let scores = host(m.scores);
        let probs = host(m.probs);
        let out = host(m.out);

        let (f64_scores, f64_probs, f64_out) = mix_f64(&v_ref, &sw, eps, SLOTS);

        g.le(&format!("{tag}.scores"), max_rel_diff("sc", &scores, &ref_scores), SCORE_RTOL, "rel");
        g.le(&format!("{tag}.probs"), max_abs_diff("pr", &probs, &ref_probs), PROB_ATOL, "abs");
        let mut sum_err = 0.0f64;
        for t in 0..TOKENS {
            let s: f64 = probs[t * SLOTS..(t + 1) * SLOTS].iter().map(|x| *x as f64).sum();
            sum_err = worse(sum_err, (s - 1.0).abs());
        }
        g.le(&format!("{tag}.probs_sum_to_one"), sum_err, PROB_ATOL, "abs");
        g.le(&format!("{tag}.out"), max_rel_diff("out", &out, &ref_out), OUT_REL, "rel-to-max");

        // Attribution: the port vs a float64 transcription, next to torch's own f32.
        let e_port = max_rel_diff("sc/f64", &scores, &f64_scores);
        let e_torch = max_rel_diff("sc/f64", &ref_scores, &f64_scores);
        g.le(
            &format!("{tag}.scores_no_worse_than_torch"),
            e_port / e_torch.max(1e-30),
            2.0,
            "x torch's own f32 error",
        );
        let d_out = differing_bf16(&out, &ref_out);
        let d_f64 = differing_bf16(&f64_out, &ref_out);
        println!(
            "         bf16 elements differing: port-vs-shipped {d_out} / {}, \
             float64-vs-shipped {d_f64}   (probs: port {:.2e} torch {:.2e} vs f64)",
            TOKENS * HIDDEN,
            max_abs_diff("p", &probs, &f64_probs),
            max_abs_diff("p", &ref_probs, &f64_probs)
        );
        // The strong criterion the shipped gate declines to make: the port must be
        // no further from the shipped output, in bf16 bit terms, than a float64
        // transcription of the shipped formula is.
        g.truth(
            &format!("{tag}.out_no_more_bitdiffs_than_f64"),
            d_out <= d_f64.max(1),
            format!("{d_out} vs {d_f64}"),
        );
    }

    section_sm93::<B>(&mut g, cfg, dev);
    // ---- the whole 93-layer schedule, driven ----
    println!("  -- 93-layer drive --");
    let mut mixer = DepthMixer::<B>::from_config(&cfg.text_config).expect("schedule");
    let n = cfg.text_config.num_hidden_layers;
    g.truth("drive.layers", mixer.num_layers() == n, format!("{n}"));

    // synthetic but bf16-exact activations, deterministic
    let mut seed = 0x51ED_270B_7A2Fu64;
    let mut next = || {
        seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        let u = ((seed >> 40) as f32) / (1u32 << 24) as f32 - 0.5;
        bf16_to_f32(bf16_bits(u * 4.0))
    };
    let sw_ones: Vec<f32> = (0..HIDDEN).map(|_| 1.0f32 / HIDDEN as f32).collect();
    let params = AttnResParams::<B>::new(
        t1::<B>(&sw_ones, dev),
        Tensor::<B, 2>::ones([1, HIDDEN], dev),
        eps,
    );

    let mut hidden: Tensor<B, 2> = {
        let d: Vec<f32> = (0..TOKENS * HIDDEN).map(|_| next()).collect();
        t2::<B>(&d, TOKENS, HIDDEN, dev)
    };
    let mut max_bank = 0usize;
    let mut max_slots = 0usize;
    let mut crossings: Vec<usize> = Vec::new();
    let mut run = 0usize;
    let mut longest_run = 0usize;
    let mut finite = true;
    for l in 0..n {
        let before = mixer.bank_len();
        let e = mixer.enter_layer(hidden.clone(), &params);
        if let Some(m) = &e.mix {
            max_slots = max_slots.max(m.probs.dims()[1]);
        }
        if mixer.bank_len() > before {
            crossings.push(l);
            run = 1;
        } else {
            run += 1;
        }
        longest_run = longest_run.max(run);
        max_bank = max_bank.max(mixer.bank_len());
        let attn: Tensor<B, 2> = {
            let d: Vec<f32> = (0..TOKENS * HIDDEN).map(|_| next()).collect();
            t2::<B>(&d, TOKENS, HIDDEN, dev)
        };
        let m = mixer.after_attention(attn, &params);
        max_slots = max_slots.max(m.probs.dims()[1]);
        let mlp: Tensor<B, 2> = {
            let d: Vec<f32> = (0..TOKENS * HIDDEN).map(|_| next()).collect();
            t2::<B>(&d, TOKENS, HIDDEN, dev)
        };
        hidden = mixer.after_mlp(mlp);
        if l % 16 == 0 {
            let h = host(hidden.clone());
            if !h.iter().all(|x| x.is_finite()) {
                finite = false;
            }
        }
    }
    let fin = mixer.finish(hidden.clone(), &params);
    g.truth("drive.crossings", crossings == vec![0, 12, 24, 36, 48, 60, 72, 84], format!("{crossings:?}"));
    g.truth("drive.max_bank", max_bank == 8, format!("{max_bank}"));
    g.truth("drive.max_slots", max_slots == 9, format!("{max_slots} (production width)"));
    g.truth("drive.longest_run", longest_run == 12, format!("{longest_run}"));
    g.truth("drive.output_slots", fin.probs.dims() == [TOKENS, 9], format!("{:?}", fin.probs.dims()));
    let fo = host(fin.out);
    g.truth("drive.finite", finite && fo.iter().all(|x| x.is_finite()), "no NaN/inf across 93 layers");
    let ho = host(hidden);
    g.truth(
        "drive.bf16_exact",
        ho.iter().all(|x| bf16_to_f32(bf16_bits(*x)).to_bits() == x.to_bits()),
        "layer-92 output is exactly bfloat16",
    );
    let pv = host(fin.probs.clone());
    let worst = (0..TOKENS)
        .map(|t| {
            let s: f64 = pv[t * 9..(t + 1) * 9].iter().map(|x| *x as f64).sum();
            (s - 1.0).abs()
        })
        .fold(0.0f64, f64::max);
    g.le("drive.output_probs_sum_to_one", worst, PROB_ATOL, "abs");
    // the mixture output is rounded to bf16 even at 9 slots
    g.truth(
        "drive.output_rounded",
        fo.iter().all(|x| bf16_to_f32(bf16_bits(*x)).to_bits() == x.to_bits()),
        "9-slot mixture output is exactly bfloat16",
    );
    let _ = round_bf16(t1::<B>(&[1.0f32], dev));

    println!("  lane {lane}: {} passed, {} failed", g.passed, g.fails.len());
    for f in &g.fails {
        println!("    - {f}");
    }
    g.fails.is_empty()
}

/// Section 2 — the SHIPPED state machine, all 93 layers, all 8 crossings.
///
/// `sm93_oracle.npz` is produced by running `KimiDecoderLayer::_forward_attn_residual`
/// and `_apply_attn_res` extracted verbatim from the checkpoint's own
/// modeling_kimi_linear.py, over the full 93-layer schedule with the real
/// per-layer AttnRes weights. The gate's own oracle stops at layer 12.
fn section_sm93<B: Backend>(g: &mut G, cfg: &K3Config, dev: &Device<B>) {
    println!("  -- shipped state machine, 93 layers --");
    let path = std::env::var("K3_SM93")
        .unwrap_or_else(|_| "./k3-oracle/sm93_oracle.npz".into());
    let z = Npz::open(std::path::Path::new(&path)).expect("opening sm93 oracle");
    let nl = z.get("meta_layers").scalar() as usize;
    let eps = cfg.text_config.rms_norm_eps;
    g.truth("sm93.layers", nl == cfg.text_config.num_hidden_layers, format!("{nl}"));

    let mut mixer = DepthMixer::<B>::from_config(&cfg.text_config).expect("schedule");
    let mut hidden = t2::<B>(&bfarr(&z, "L00_layer_in_bf16bits"), TOKENS, HIDDEN, dev);

    let mut bank_exact = true;
    let mut layer_out_exact = true;
    let mut worst_entry = 0.0f64;
    let mut worst_mlp = 0.0f64;
    let mut entry_bitdiffs = 0usize;
    let mut mlp_bitdiffs = 0usize;
    let mut crossings: Vec<usize> = Vec::new();
    let mut max_slots = 0usize;
    let mut mix_presence_ok = true;

    for l in 0..nl {
        let lp = format!("L{l:02}");
        let ones = Tensor::<B, 2>::ones([1, HIDDEN], dev);
        let sa = AttnResParams::<B>::new(
            t1::<B>(&z.get(&format!("{lp}_sa_score_weight")).to_f32(), dev),
            ones.clone(),
            eps,
        );
        let mlp = AttnResParams::<B>::new(
            t1::<B>(&z.get(&format!("{lp}_mlp_score_weight")).to_f32(), dev),
            ones,
            eps,
        );
        let nb_in = z.get(&format!("{lp}_nb_in")).scalar() as usize;
        let nb_out = z.get(&format!("{lp}_nb_out")).scalar() as usize;

        let e = mixer.enter_layer(hidden.clone(), &sa);
        if e.mix.is_some() != (nb_in > 0) {
            mix_presence_ok = false;
        }
        if let Some(m) = &e.mix {
            max_slots = max_slots.max(m.probs.dims()[1]);
        }
        let got_entry = host(e.to_attention);
        let want_entry = bfarr(&z, &format!("{lp}_input_layernorm_in_bf16bits"));
        worst_entry = worse(worst_entry, max_rel_diff("entry", &got_entry, &want_entry));
        entry_bitdiffs += differing_bf16(&got_entry, &want_entry);

        if nb_out > nb_in {
            crossings.push(l);
        }
        if mixer.bank_len() != nb_out {
            bank_exact = false;
            g.truth(&format!("sm93.{lp}.bank_len"), false, format!("port {} shipped {nb_out}", mixer.bank_len()));
        } else {
            let flat = bfarr(&z, &format!("{lp}_blockres_out_bf16bits"));
            assert_eq!(flat.len(), TOKENS * nb_out * HIDDEN, "{lp}: bank size");
            for (i, entry) in mixer.bank().iter().enumerate() {
                let mut want = Vec::with_capacity(TOKENS * HIDDEN);
                for t in 0..TOKENS {
                    let off = t * nb_out * HIDDEN + i * HIDDEN;
                    want.extend_from_slice(&flat[off..off + HIDDEN]);
                }
                if !bits_equal(&host(entry.clone()), &want) {
                    bank_exact = false;
                    g.truth(&format!("sm93.{lp}.bank[{i}]"), false, "differs");
                }
            }
        }

        let m = mixer.after_attention(
            t2::<B>(&bfarr(&z, &format!("{lp}_attn_out_bf16bits")), TOKENS, HIDDEN, dev),
            &mlp,
        );
        max_slots = max_slots.max(m.probs.dims()[1]);
        let got_mlp = host(m.out);
        let want_mlp = bfarr(&z, &format!("{lp}_post_attention_layernorm_in_bf16bits"));
        worst_mlp = worse(worst_mlp, max_rel_diff("mlp", &got_mlp, &want_mlp));
        mlp_bitdiffs += differing_bf16(&got_mlp, &want_mlp);

        hidden = mixer
            .after_mlp(t2::<B>(&bfarr(&z, &format!("{lp}_mlp_out_bf16bits")), TOKENS, HIDDEN, dev));
        let want_out = bfarr(&z, &format!("{lp}_layer_out_bf16bits"));
        if !bits_equal(&host(hidden.clone()), &want_out) {
            layer_out_exact = false;
            g.truth(
                &format!("sm93.{lp}.layer_out"),
                false,
                format!("max |d| {:e}", max_abs_diff("out", &host(hidden.clone()), &want_out)),
            );
        }
    }

    g.truth("sm93.mix_presence", mix_presence_ok, "mixture iff the shipped bank was non-empty");
    g.truth(
        "sm93.crossings",
        crossings == vec![0, 12, 24, 36, 48, 60, 72, 84],
        format!("{crossings:?} — measured against the SHIPPED forward, all 8"),
    );
    g.truth("sm93.max_slots", max_slots == 9, format!("{max_slots} (production mixture width)"));
    g.truth("sm93.bank_bitexact_all_93_layers", bank_exact, "every snapshot, every layer");
    g.truth("sm93.layer_out_bitexact_all_93_layers", layer_out_exact, "93 layers chained on the depth axis");
    g.le("sm93.entry_mix", worst_entry, OUT_REL, "rel-to-max");
    g.le("sm93.mlp_mix", worst_mlp, OUT_REL, "rel-to-max");
    println!(
        "         bf16 elements differing across all 93 layers: entry {entry_bitdiffs} / {}, \
         mlp {mlp_bitdiffs} / {}",
        nl * TOKENS * HIDDEN,
        nl * TOKENS * HIDDEN
    );

    let out_p = AttnResParams::<B>::new(
        t1::<B>(&z.get("MODEL_output_score_weight").to_f32(), dev),
        Tensor::<B, 2>::ones([1, HIDDEN], dev),
        eps,
    );
    g.truth(
        "sm93.final_hidden_matches",
        bits_equal(&host(hidden.clone()), &bfarr(&z, "MODEL_output_prefix_sum_bf16bits")),
        "the port's layer-92 output is what the shipped output mixture accumulates",
    );
    let fin = mixer.finish(hidden, &out_p);
    let nslots = z.get("MODEL_output_nslots").scalar() as usize;
    g.truth("sm93.output_slots", fin.probs.dims() == [TOKENS, nslots] && nslots == 9, format!("{nslots}"));
    let want_fin = bfarr(&z, "MODEL_output_out_bf16bits");
    let got_fin = host(fin.out);
    g.le("sm93.output_attn_res", max_rel_diff("fin", &got_fin, &want_fin), OUT_REL, "rel-to-max");
    println!(
        "         model-level 9-slot mixture: {} of {} bf16 elements differ",
        differing_bf16(&got_fin, &want_fin),
        TOKENS * HIDDEN
    );
}

fn main() -> Result<()> {
    let ck = PathBuf::from(
        std::env::var("K3_CHECKPOINT").unwrap_or_else(|_| "./kimi-k3".into()),
    );
    let path = PathBuf::from(
        std::env::var("K3_SLOTS9").unwrap_or_else(|_| "./k3-oracle/slots9_oracle.npz".into()),
    );
    let cfg = K3Config::from_json(&std::fs::read_to_string(ck.join("config.json"))?)
        .map_err(|e| anyhow::anyhow!(e))?;
    let z = Npz::open(&path).context("opening the 9-slot oracle")?;
    println!("k3_attn_res_slots9 — AttnRes at NINE slots and across all 93 layers");
    println!("oracle: {} ({} arrays)", path.display(), z.len());

    let mut lanes = Vec::new();
    {
        type Cpu = burn::backend::NdArray;
        lanes.push(("ndarray-cpu", run_lane::<Cpu>("ndarray-cpu", &Device::<Cpu>::default(), &z, &cfg)));
    }
    #[cfg(feature = "k3-attn-res-cuda")]
    {
        type Gpu = burn::backend::Cuda;
        lanes.push(("cuda", run_lane::<Gpu>("cuda", &Device::<Gpu>::default(), &z, &cfg)));
    }
    println!("\n=== summary ===");
    for (n, r) in &lanes {
        println!("  {n:<14} {}", if *r { "PASS" } else { "FAIL" });
    }
    let pass = !lanes.is_empty() && lanes.iter().all(|(_, r)| *r);
    println!("\nSLOTS9: {}  ({} lane(s))", if pass { "PASS" } else { "FAIL" }, lanes.len());
    if !pass {
        std::process::exit(1);
    }
    Ok(())
}
