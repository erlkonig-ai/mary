//! A controlled device-only soft-target update through learn_last_layer.
//! No model pile, numerical host model, stochastic success assumption, or policy bypass.

use super::*;
use super::super::assembly::{HostT, bytes_of, dev_lane_resid, devroute_new, quantized_bf16,
    routed_experts_fp4_dev, row_nll_dev, up1r, up2, w4a16_bind};
use super::super::fp4gemm::{MTILE, swizzle_b_codes, swizzle_b_scales};
use super::super::seam::client_of;

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
    client.read_one(handle.clone()).expect("soft-target test readback").to_vec()
}

fn fixture(client: &Client) -> Fixture {
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
        for row in 0..2 * INTER { codes[row * H / 2] = 2; }
        off13.push(bytes.len() as u64);
        bytes.extend(swizzle_b_codes(&codes, 2 * INTER, H));
        w13_ranges.push(start..bytes.len());
        off13.push(bytes.len() as u64);
        let scale_start = bytes.len();
        bytes.extend(swizzle_b_scales(&vec![0x38; 2 * INTER * H / 16], 2 * INTER, H));
        scales.push(scale_start..bytes.len());
        let w2_start = bytes.len();
        off2.push(w2_start as u64);
        bytes.extend(swizzle_b_codes(&vec![0; H * INTER / 2], H, INTER));
        w2_ranges.push(w2_start..bytes.len());
        off2.push(bytes.len() as u64);
        let scale_start = bytes.len();
        bytes.extend(swizzle_b_scales(&vec![0x38; H * INTER / 16], H, INTER));
        scales.push(scale_start..bytes.len());
        experts.push(start..bytes.len());
    }
    bytes.extend([0x5a; 32]);
    let ones = [1.0f32; EXPERTS];
    let table = ExpertTable {
        off13: client.create_from_slice(bytes_of(&off13)), off2: client.create_from_slice(bytes_of(&off2)),
        sc13: client.create_from_slice(bytes_of(&ones)), sc2: client.create_from_slice(bytes_of(&ones)),
        wmap: client.create_from_slice(&bytes), wmap_bytes: bytes.len(),
        expert_bytes: experts[0].len(), n_routed: EXPERTS, stride: 2, scaled: true,
    };
    Fixture { table, before: bytes, experts, scales, w13: w13_ranges, w2: w2_ranges }
}

#[test]
#[ignore = "requires native NVFP4 CUDA; reserve the shared GPU and run this test explicitly"]
fn explicit_response_distribution_updates_only_its_student_expert() {
    assert!(dev_lane::act_bf16(), "this fixture requires the normal BF16 activation lane");
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
    let plan = || super::super::devplan::plan_from_topk_launch(&client, &topk, table,
        &route.fault, route.kmax, MTILE, 3, N);
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
        let y = routed_experts_fp4_dev(&client, &dev, "soft-target-fixture.", table, dp,
            &route, &hn, N, H, INTER, true, std::time::Instant::now(), &mut HostT::default());
        dev_lane_resid::add_resid(residual.clone(), dev_lane::short_conv(y, sconv.clone()))
    };
    let mut unembed = vec![0.0f32; VOCAB * H];
    unembed[0] = 1.0;
    let unembed: Vec<u8> = unembed.iter()
        .flat_map(|&x| half::bf16::from_f32(x).to_le_bytes()).collect();
    let head_weight = w4a16_bind(&client, quantized_bf16(&client, &unembed, VOCAB, H), true);
    let norm = up1r::<Bk>(&vec![1.0; H], H, &dev);
    let head = |x: T2| dev_lane::linear_w(dev_lane_resid::rms_norm(x, norm.clone(), 1e-6), &head_weight);
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
    let mut teacher = ema::EmaBank::new(&client, table, H, INTER, true, 0.25, learner.step()).unwrap();
    let teacher_before = read(&client, &teacher.teacher_table().wmap);
    let sc13_before = read(&client, &table.sc13);
    let sc2_before = read(&client, &table.sc2);
    let mut target = vec![0.0f32; VOCAB];
    target[0] = 0.9;
    target[1] = 0.1;
    let target = up2::<Bk>(target, 1, VOCAB, &dev);
    let keep = LearnKeep { layer: 0, hn: hn.clone(), dp,
        sconv: Some(sconv.clone()), x_pre: Some(residual.clone()), hist0: None };
    let report = learn_last_layer(&client, &dev, &mut learner, keep, &route, table, true,
        &before_x, &norm, &head, Target::DistRange { dist: &target, start: 1, weight: 1.0 },
        1.0, 1e-6, VOCAB, VOCAB, EXPERTS, INTER, 1).unwrap();
    assert_eq!(learner.step(), 1);
    assert_eq!((report.rows, report.slots), (1, 2));
    let after = read(&client, &table.wmap);
    for expert in [1, 2] {
        let range = data.experts[expert].clone();
        assert_eq!(&after[range.clone()], &data.before[range], "prompt-only or unused expert changed");
    }
    for range in data.scales.iter().chain(&data.w13) {
        assert_eq!(&after[range.clone()], &data.before[range.clone()], "scale/W13 changed on a zero W13-gradient fixture");
    }
    assert_ne!(&after[data.w2[0].clone()], &data.before[data.w2[0].clone()], "response expert did not update");
    assert_eq!(&after[..32], &data.before[..32]);
    assert_eq!(&after[after.len() - 32..], &data.before[data.before.len() - 32..]);
    assert_eq!(read(&client, &table.sc13), sc13_before);
    assert_eq!(read(&client, &table.sc2), sc2_before);
    let after_nll = nll(forward(&plan()));
    eprintln!("controlled response soft-target NLL: {before_nll} -> {after_nll}");
    assert!(before_nll.is_finite() && after_nll.is_finite() && after_nll < before_nll,
        "the controlled nearest-rounding update did not move toward the 90/10 teacher signal");
    // The residual is orthogonal to class0's head feature. Initially its
    // gradient is about -14 and SiLU(1) about .73: lr .05 moves W2 zero
    // codes to roughly +.51, safely past the +.25 nearest-rounding boundary.
    // Pre-update W2 is zero, so W13 receives exactly zero input gradient.
    assert_eq!(read(&client, &teacher.teacher_table().wmap), teacher_before,
        "student optimization changed the teacher before EMA");
    assert_eq!(teacher.report().version, 0);
    let report = teacher.advance(&client, 0, learner.step()).unwrap();
    assert_eq!((report.version, report.student_step), (1, 1));
    assert_ne!(read(&client, &teacher.teacher_table().wmap), teacher_before,
        "explicit EMA did not move the teacher on this fixture");
    assert_eq!(read(&client, &table.wmap), after, "EMA mutated student weights");
}
