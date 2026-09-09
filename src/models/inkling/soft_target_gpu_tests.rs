//! A controlled device-only soft-target update through learn_last_layer.
//! No model pile, numerical host model, stochastic success assumption, or policy bypass.

use super::super::assembly::{
    HostT, bytes_of, dev_lane_resid, devroute_new, quantized_bf16, routed_experts_fp4_dev,
    row_nll_dev, up1r, up2, w4a16_bind,
};
use super::super::fp4gemm::{MTILE, swizzle_b_codes, swizzle_b_scales};
use super::super::seam::client_of;
use super::*;

const H: usize = 256;
const INTER: usize = 256;
const VOCAB: usize = 256;
const EXPERTS: usize = 3;
const N: usize = 2;

struct Fixture {
    table: ExpertTable,
    before: Vec<u8>,
    experts: Vec<std::ops::Range<usize>>,
    scales: Vec<std::ops::Range<usize>>,
    w13: Vec<std::ops::Range<usize>>,
    w2: Vec<std::ops::Range<usize>>,
}

fn read(client: &Client, handle: &Handle) -> Vec<u8> {
    client
        .read_one(handle.clone())
        .expect("soft-target test readback")
        .to_vec()
}

fn fixture(client: &Client) -> Fixture {
    fixture_with_w2(client, false)
}

fn fixture_with_w2(client: &Client, nonzero: bool) -> Fixture {
    use rand::{Rng, SeedableRng};
    let mut rng = rand::rngs::StdRng::seed_from_u64(42);
    let mut bytes = vec![0xa5; 32];
    let mut off13 = Vec::<u64>::new();
    let mut off2 = Vec::<u64>::new();
    let mut experts = Vec::new();
    let mut scales = Vec::new();
    let mut w13_ranges = Vec::new();
    let mut w2_ranges = Vec::new();
    for _ in 0..EXPERTS {
        let start = bytes.len();
        let mut codes = vec![0u8; 2 * INTER * H / 2];
        // Every gate/up row reads feature zero with weight 1. Other columns
        // are zero. hn[0]=1 therefore gives gate=up=1, and SiLU(1)>0.
        for row in 0..2 * INTER {
            codes[row * H / 2] = 2;
        }
        off13.push(bytes.len() as u64);
        bytes.extend(swizzle_b_codes(&codes, 2 * INTER, H));
        w13_ranges.push(start..bytes.len());
        off13.push(bytes.len() as u64);
        let scale_start = bytes.len();
        bytes.extend(swizzle_b_scales(
            &vec![0x38; 2 * INTER * H / 16],
            2 * INTER,
            H,
        ));
        scales.push(scale_start..bytes.len());
        let w2_start = bytes.len();
        off2.push(w2_start as u64);
        let codes: Vec<u8> = (0..H * INTER / 2)
            .map(|_| {
                if nonzero {
                    let low = if rng.gen_bool(0.5) { 2 } else { 10 };
                    let high = if rng.gen_bool(0.5) { 2 } else { 10 };
                    low | (high << 4)
                } else {
                    0
                }
            })
            .collect();
        bytes.extend(swizzle_b_codes(&codes, H, INTER));
        w2_ranges.push(w2_start..bytes.len());
        off2.push(bytes.len() as u64);
        let scale_start = bytes.len();
        // Nonzero fixture: +/-1 codes at scale 1/16, not a zero-W2 shortcut.
        let scale = if nonzero { 0x18 } else { 0x38 };
        bytes.extend(swizzle_b_scales(&vec![scale; H * INTER / 16], H, INTER));
        scales.push(scale_start..bytes.len());
        experts.push(start..bytes.len());
    }
    bytes.extend([0x5a; 32]);
    let ones = [1.0f32; EXPERTS];
    let table = ExpertTable {
        off13: client.create_from_slice(bytes_of(&off13)),
        off2: client.create_from_slice(bytes_of(&off2)),
        sc13: client.create_from_slice(bytes_of(&ones)),
        sc2: client.create_from_slice(bytes_of(&ones)),
        wmap: client.create_from_slice(&bytes),
        wmap_bytes: bytes.len(),
        expert_bytes: experts[0].len(),
        n_routed: EXPERTS,
        stride: 2,
        scaled: true,
    };
    Fixture {
        table,
        before: bytes,
        experts,
        scales,
        w13: w13_ranges,
        w2: w2_ranges,
    }
}

#[test]
#[ignore = "requires native NVFP4 CUDA; reserve the shared GPU and run this test explicitly"]
fn explicit_response_distribution_updates_only_its_student_expert() {
    assert!(
        dev_lane::act_bf16(),
        "this fixture requires the normal BF16 activation lane"
    );
    assert_eq!(H % LANE_K, 0);
    assert_eq!(INTER % LANE_K, 0);
    let dev = Dev::default();
    let mut inputs = vec![0.0f32; N * H];
    inputs[0] = 1.0;
    inputs[H] = 1.0;
    let hn = up2::<Bk>(inputs, N, H, &dev).cast(burn::tensor::DType::BF16);
    let client = client_of(&hn);
    let data = fixture(&client);
    let table = &data.table;
    let route = devroute_new(&client, 1, N);
    // Prompt-only row -> expert1, response-prediction row -> expert0.
    // Expert2 is never routed. This makes DistRange's mask observable in bytes.
    let topk = client.create_from_slice(bytes_of(&[1.0f32, 1.0, 0.0, 0.0, 1.0, 0.0]));
    let plan = || {
        super::super::devplan::plan_from_topk_launch(
            &client,
            &topk,
            table,
            &route.fault,
            route.kmax,
            MTILE,
            3,
            N,
        )
    };
    let mut residual = vec![0.0f32; N * H];
    residual[1] = 1.0;
    residual[H + 1] = 1.0;
    let residual = up2::<Bk>(residual, N, H, &dev);
    // Identity short convolution: no legitimate response gradient propagates
    // into the prompt row, so a changed prompt-only expert would be a mask bug.
    // Keep a nonempty history plane like the real model: the backend's
    // zero-sized fill for a one-tap convolution is not a valid CUDA launch.
    let sconv = BT::<Bk, 2>::zeros([H, 2], &dev);
    let forward = |dp: &DevRowPlan| {
        let y = routed_experts_fp4_dev(
            &client,
            &dev,
            "soft-target-fixture.",
            table,
            dp,
            &route,
            &hn,
            N,
            H,
            INTER,
            true,
            std::time::Instant::now(),
            &mut HostT::default(),
        );
        dev_lane_resid::add_resid(residual.clone(), dev_lane::short_conv(y, sconv.clone()))
    };
    let mut unembed = vec![0.0f32; VOCAB * H];
    unembed[0] = 1.0;
    let unembed: Vec<u8> = unembed
        .iter()
        .flat_map(|&x| half::bf16::from_f32(x).to_le_bytes())
        .collect();
    let head_weight = w4a16_bind(&client, quantized_bf16(&client, &unembed, VOCAB, H), true);
    let norm = up1r::<Bk>(&vec![1.0; H], H, &dev);
    let head = |x: T2| {
        dev_lane::linear_w(
            dev_lane_resid::rms_norm(x, norm.clone(), 1e-6),
            &head_weight,
        )
    };
    let nll = |x: T2| {
        <Bk as burn::tensor::backend::Backend>::sync(&dev).expect("routed fixture forward");
        let logits = head(x.slice([1..2, 0..H]));
        <Bk as burn::tensor::backend::Backend>::sync(&dev).expect("fixture head forward");
        let losses = row_nll_dev(BT::<Bk, 2>::cat(vec![logits.clone(), logits], 0), &[0, 1]);
        0.9 * losses[0] + 0.1 * losses[1]
    };
    let dp = plan();
    let before_x = forward(&dp);
    let before_nll = nll(before_x.clone());
    let mut learner = Learner::bind(&client, &unembed, VOCAB, H, 0.05, false);
    learner.steps = 0;
    let mut teacher =
        ema::EmaBank::new(&client, table, H, INTER, true, 0.25, learner.step()).unwrap();
    let teacher_before = read(&client, &teacher.teacher_table().wmap);
    let sc13_before = read(&client, &table.sc13);
    let sc2_before = read(&client, &table.sc2);
    let mut target = vec![0.0f32; VOCAB];
    target[0] = 0.9;
    target[1] = 0.1;
    let target = up2::<Bk>(target, 1, VOCAB, &dev);
    let keep = LearnKeep {
        layer: 0,
        hn: hn.clone(),
        dp,
        sconv: Some(sconv.clone()),
        x_pre: Some(residual.clone()),
        hist0: None,
    };
    let report = learn_last_layer(
        &client,
        &dev,
        &mut learner,
        keep,
        &route,
        table,
        true,
        &before_x,
        &norm,
        &head,
        Target::DistRange {
            dist: &target,
            start: 1,
            weight: 1.0,
        },
        1.0,
        1e-6,
        VOCAB,
        VOCAB,
        EXPERTS,
        INTER,
        1,
    )
    .unwrap();
    assert_eq!(learner.step(), 1);
    assert_eq!((report.rows, report.slots), (1, 2));
    let after = read(&client, &table.wmap);
    for expert in [1, 2] {
        let range = data.experts[expert].clone();
        assert_eq!(
            &after[range.clone()],
            &data.before[range],
            "prompt-only or unused expert changed"
        );
    }
    for range in data.scales.iter().chain(&data.w13) {
        assert_eq!(
            &after[range.clone()],
            &data.before[range.clone()],
            "scale/W13 changed on a zero W13-gradient fixture"
        );
    }
    assert_ne!(
        &after[data.w2[0].clone()],
        &data.before[data.w2[0].clone()],
        "response expert did not update"
    );
    assert_eq!(&after[..32], &data.before[..32]);
    assert_eq!(
        &after[after.len() - 32..],
        &data.before[data.before.len() - 32..]
    );
    assert_eq!(read(&client, &table.sc13), sc13_before);
    assert_eq!(read(&client, &table.sc2), sc2_before);
    let after_nll = nll(forward(&plan()));
    eprintln!("controlled response soft-target NLL: {before_nll} -> {after_nll}");
    assert!(
        before_nll.is_finite() && after_nll.is_finite() && after_nll < before_nll,
        "the controlled nearest-rounding update did not move toward the 90/10 teacher signal"
    );
    // The residual is orthogonal to class0's head feature. Initially its
    // gradient is about -14 and SiLU(1) about .73: lr .05 moves W2 zero
    // codes to roughly +.51, safely past the +.25 nearest-rounding boundary.
    // Pre-update W2 is zero, so W13 receives exactly zero input gradient.
    assert_eq!(
        read(&client, &teacher.teacher_table().wmap),
        teacher_before,
        "student optimization changed the teacher before EMA"
    );
    assert_eq!(teacher.report().version, 0);
    let report = teacher.advance(&client, 0, learner.step()).unwrap();
    assert_eq!((report.version, report.student_step), (1, 1));
    assert_ne!(
        read(&client, &teacher.teacher_table().wmap),
        teacher_before,
        "explicit EMA did not move the teacher on this fixture"
    );
    assert_eq!(
        read(&client, &table.wmap),
        after,
        "EMA mutated student weights"
    );
}

/// A measurement, not a claim that unbiased rounding must improve nonlinear
/// loss. Every trial starts from identical bytes and the same fixed, nearby
/// soft teacher. Only optimizer rounding changes across seeds. No checkpoint,
/// token rollout, EMA update, foreground workload, or retention task is here.
#[test]
#[ignore = "tiny CUDA research diagnostic; reserve GPU and run explicitly with --nocapture"]
fn stochastic_subcode_loss_diagnostic() {
    measure_subcode_loss(0.125, true);
}

#[test]
#[ignore = "tiny CUDA research diagnostic; reserve GPU and run explicitly with --nocapture"]
fn weak_signal_stochastic_subcode_loss_diagnostic() {
    measure_subcode_loss(1.0 / 128.0, true);
}

#[test]
#[ignore = "tiny CUDA research diagnostic; reserve GPU and run explicitly with --nocapture"]
fn wide_residual_stochastic_subcode_loss_diagnostic() {
    measure_subcode_loss(0.125, false);
}

#[test]
#[ignore = "tiny CUDA research diagnostic; reserve GPU and run explicitly with --nocapture"]
fn weak_signal_wide_residual_stochastic_subcode_loss_diagnostic() {
    measure_subcode_loss(1.0 / 128.0, false);
}

fn measure_subcode_loss(teacher_tilt: f32, residual_bf16: bool) {
    use rand::{Rng, SeedableRng};
    const SEEDS: u32 = 32;
    let dev = Dev::default();
    let mut rng = rand::rngs::StdRng::seed_from_u64(17);
    let mut inputs: Vec<f32> = (0..N * H).map(|_| rng.gen_range(-0.1..0.1)).collect();
    inputs[0] = 0.875;
    inputs[H] = 1.125;
    let hn = up2::<Bk>(inputs, N, H, &dev).cast(burn::tensor::DType::BF16);
    let client = client_of(&hn);
    let route = devroute_new(&client, 1, N);
    let topk = client.create_from_slice(bytes_of(&[1.0f32, 1.0, 0.0, 0.0, 1.0, 0.0]));
    let plan = |table: &ExpertTable| {
        super::super::devplan::plan_from_topk_launch(
            &client,
            &topk,
            table,
            &route.fault,
            route.kmax,
            MTILE,
            3,
            N,
        )
    };
    let residual = up2::<Bk>(
        (0..N * H).map(|_| rng.gen_range(-1.0..1.0)).collect(),
        N,
        H,
        &dev,
    );
    let residual = if residual_bf16 {
        residual.cast(burn::tensor::DType::BF16)
    } else {
        residual
    };
    // A response gradient legitimately reaches the preceding prompt expert.
    let taps: Vec<f32> = (0..H).flat_map(|_| [0.25, -0.125]).collect();
    let sconv = up2::<Bk>(taps, H, 2, &dev);
    let forward = |table: &ExpertTable, dp: &DevRowPlan| {
        let y = routed_experts_fp4_dev(
            &client,
            &dev,
            "subcode-fixture.",
            table,
            dp,
            &route,
            &hn,
            N,
            H,
            INTER,
            true,
            std::time::Instant::now(),
            &mut HostT::default(),
        );
        dev_lane_resid::add_resid(residual.clone(), dev_lane::short_conv(y, sconv.clone()))
    };
    // A dense head exercises the independently packed transposed-head
    // backward, unlike the old one-column head fixture.
    let unembed: Vec<u8> = (0..VOCAB * H)
        .flat_map(|_| half::bf16::from_f32(rng.gen_range(-0.0625..0.0625)).to_le_bytes())
        .collect();
    let head_weight = w4a16_bind(&client, quantized_bf16(&client, &unembed, VOCAB, H), true);
    let norm = up1r::<Bk>(&vec![1.0; H], H, &dev);
    let head = |x: T2| {
        dev_lane::linear_w(
            dev_lane_resid::rms_norm(x, norm.clone(), 1e-6),
            &head_weight,
        )
    };
    let logits = |x: T2| head(x.slice([1..2, 0..H])).cast(burn::tensor::DType::F32);
    let host = |x: T2| x.into_data().to_vec::<f32>().unwrap();
    let initial = fixture_with_w2(&client, true);
    let initial_logits = logits(forward(&initial.table, &plan(&initial.table)));
    let original = burn::tensor::activation::softmax(initial_logits.clone(), 1);
    let tilt = up2::<Bk>(
        (0..VOCAB)
            .map(|v| {
                if v % 2 == 0 {
                    teacher_tilt
                } else {
                    -teacher_tilt
                }
            })
            .collect(),
        1,
        VOCAB,
        &dev,
    );
    let target = burn::tensor::activation::softmax(initial_logits.clone() + tilt, 1);
    let target_host = host(target.clone());
    let original_host = host(original.clone());
    let initial_host = host(initial_logits);
    // Score the actual F32 targets without renormalizing them into a different
    // objective. F64 log-softmax preserves small post-step loss differences.
    let kl = |scores: &[f32], probabilities: &[f32]| -> f64 {
        let max = scores
            .iter()
            .map(|&x| f64::from(x))
            .fold(f64::NEG_INFINITY, f64::max);
        let log_z = max
            + scores
                .iter()
                .map(|&x| (f64::from(x) - max).exp())
                .sum::<f64>()
                .ln();
        scores
            .iter()
            .zip(probabilities)
            .map(|(&x, &q)| {
                let q = f64::from(q);
                if q == 0.0 {
                    0.0
                } else {
                    q * (q.ln() - f64::from(x) + log_z)
                }
            })
            .sum()
    };
    let baseline_kl = kl(&initial_host, &target_host);
    // +0 and -0 are different nibbles, not different parameter values.
    let value_code = |code: u8| if code & 7 == 0 { 0 } else { code };
    let count_codes = |before: &[u8], after: &[u8], ranges: &[std::ops::Range<usize>]| -> usize {
        ranges
            .iter()
            .map(|range| {
                before[range.clone()]
                    .iter()
                    .zip(&after[range.clone()])
                    .map(|(&a, &b)| {
                        usize::from(value_code(a & 15) != value_code(b & 15))
                            + usize::from(value_code(a >> 4) != value_code(b >> 4))
                    })
                    .sum::<usize>()
            })
            .sum()
    };
    eprintln!(
        "subcode {}",
        serde_json::json!({
            "kind": "fixture", "hidden": H, "intermediate": INTER, "vocab": VOCAB,
            "experts": EXPERTS, "rows": N, "response_rows": 1, "seeds": SEEDS,
            "act_bf16": dev_lane::act_bf16(), "residual_bf16": residual_bf16,
            "captured_hn_dtype": "BF16", "w13_initialization": "rank_one_identical_gate_up_rows",
            "target_mass": target_host.iter().map(|&x| f64::from(x)).sum::<f64>(),
            "baseline_kl": baseline_kl, "loss_direction": "teacher_to_student",
            "w2_scale": 0.0625, "teacher_logit_tilt": teacher_tilt,
            "weight_fixture_seed": 42, "input_and_head_seed": 17,
            "scope": "one actual optimizer step; fixed synthetic teacher; no retention or throughput claim"
        })
    );
    for (arm, stochastic, has_signal) in [
        ("nearest", false, true),
        ("stochastic", true, true),
        ("zero_signal", true, false),
    ] {
        for lr in [0.001f32, 0.01, 0.1, 1.0] {
            let mut deltas = Vec::new();
            let mut code_counts = [0usize; 2];
            let seed_count = if stochastic { SEEDS } else { 1 };
            for seed in 0..seed_count {
                let data = fixture_with_w2(&client, true);
                assert_eq!(data.before, initial.before, "trial changed initial weights");
                let dp = plan(&data.table);
                let before_x = forward(&data.table, &dp);
                let before_scores = host(logits(before_x.clone()));
                assert_eq!(
                    before_scores, initial_host,
                    "trial changed baseline forward"
                );
                let mut learner = Learner::bind(&client, &unembed, VOCAB, H, lr, stochastic);
                learner.steps = seed;
                let keep = LearnKeep {
                    layer: 0,
                    hn: hn.clone(),
                    dp,
                    sconv: Some(sconv.clone()),
                    x_pre: Some(residual.clone()),
                    hist0: None,
                };
                let want = if has_signal { &target } else { &original };
                learn_last_layer(
                    &client,
                    &dev,
                    &mut learner,
                    keep,
                    &route,
                    &data.table,
                    true,
                    &before_x,
                    &norm,
                    &head,
                    Target::DistRange {
                        dist: want,
                        start: 1,
                        weight: 1.0,
                    },
                    1.0,
                    1e-6,
                    VOCAB,
                    VOCAB,
                    EXPERTS,
                    INTER,
                    1,
                )
                .unwrap();
                assert_eq!(learner.steps, seed + 1);
                let after = read(&client, &data.table.wmap);
                for range in data.scales.iter().chain(std::iter::once(&data.experts[2])) {
                    assert_eq!(
                        &after[range.clone()],
                        &data.before[range.clone()],
                        "scale or unused expert changed"
                    );
                }
                assert_eq!(&after[..32], &data.before[..32]);
                assert_eq!(
                    &after[after.len() - 32..],
                    &data.before[data.before.len() - 32..]
                );
                if !has_signal {
                    assert_eq!(after, data.before, "zero signal moved weights");
                }
                let w13_changed = count_codes(&data.before, &after, &data.w13);
                let w2_changed = count_codes(&data.before, &after, &data.w2);
                let after_scores = host(logits(forward(&data.table, &plan(&data.table))));
                let after_kl = kl(&after_scores, &target_host);
                let delta = after_kl - baseline_kl;
                assert!(after_kl.is_finite());
                let original_drift =
                    kl(&after_scores, &original_host) - kl(&initial_host, &original_host);
                eprintln!(
                    "subcode {}",
                    serde_json::json!({
                        "kind": "trial", "arm": arm, "lr": lr, "seed": seed,
                        "teacher_logit_tilt": teacher_tilt, "residual_bf16": residual_bf16,
                        "after_kl": after_kl, "delta_kl": delta,
                        "original_distribution_kl_increase": original_drift,
                        "w13_changed": w13_changed, "w2_changed": w2_changed
                    })
                );
                deltas.push(delta);
                code_counts[0] += w13_changed;
                code_counts[1] += w2_changed;
            }
            let n = f64::from(seed_count);
            let mean = deltas.iter().sum::<f64>() / n;
            let se = if seed_count > 1 {
                (deltas.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / (n - 1.0) / n).sqrt()
            } else {
                0.0
            };
            eprintln!(
                "subcode {}",
                serde_json::json!({
                    "kind": "summary", "arm": arm, "lr": lr, "trials": seed_count,
                    "teacher_logit_tilt": teacher_tilt, "residual_bf16": residual_bf16,
                    "mean_delta_kl": mean, "standard_error": se,
                    "improved": deltas.iter().filter(|&&x| x < 0.0).count(),
                    "unchanged": deltas.iter().filter(|&&x| x == 0.0).count(),
                    "w13_changes_total": code_counts[0], "w2_changes_total": code_counts[1]
                })
            );
        }
    }
}
