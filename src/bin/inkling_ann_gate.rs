//! The approximate head against the exact one, at the head's own shape, on one
//! box and without a checkpoint.
//!
//! This is the DEV LOOP, and it is deliberately not the evidence. A full-model
//! run takes minutes to load and needs both Sparks; this builds a `[n, 4096]`
//! table, quantises it exactly as the runtime does, builds the sketch, and puts
//! `w4a16_linear` and [`annhead::ann_logits`] side by side on the same bytes.
//! What it can settle is whether the kernels are CORRECT and where the scan sits
//! against the bus. What it cannot settle is recall on real hidden states, which
//! is a property of the model's own logit geometry — that is `INK_ANN_VERIFY=1`
//! on a real prompt, and nothing here substitutes for it.
//!
//! # The synthetic table is not uniform noise, on purpose
//!
//! A table of i.i.d. Gaussians is the one case where a sign sketch cannot fail:
//! every coordinate carries the same mass, so throwing away magnitudes throws
//! away nothing systematic, and the rotation this module argues is essential
//! would measure as free. Real embedding tables are not like that — they have
//! rogue dimensions carrying disproportionate mass, and row norms that vary by
//! token frequency — so the table here is built with both, and the
//! `INK_ANN_ROT=0` arm exists to show what happens without the rotation. A gate
//! whose synthetic input cannot exhibit the failure it is gating on is a gate
//! that passes for the wrong reason.
//!
//! # The queries are near-ties, on purpose
//!
//! A query drawn independently of the table has a top-1 that beats its top-2 by
//! a wide margin, and any estimator gets those right. The hard case — and the
//! only one that matters, since it is the one that decides a token — is a query
//! sitting between two plausible rows. So queries are drawn as a blend of two
//! random rows plus noise, which manufactures exactly that.
//!
//! # Environment
//!
//! ```text
//!   INK_ANN_N        rows (default 201024, the unembedding's own)
//!   INK_ANN_Q        queries to score (default 64)
//!   INK_ANN_BUDGET   shortlist size (default 1024)
//!   INK_ANN_RANGE    floor histogram window, logits (default 12)
//!   INK_ANN_ROT      1 rotated (default), 0 the raw-coordinate ablation
//! ```

use std::time::Instant;

use cubecl::future;
use cubecl::prelude::*;
use half::bf16;

use mary::models::inkling::annhead;
use mary::models::inkling::fp4quant::quantize_nvfp4_bf16;
use mary::models::inkling::w4a16gemm::w4a16_linear_launch;

type Rt = cubecl::cuda::CudaRuntime;

const K: usize = 4096;
/// Coordinates given ten times the usual mass, standing in for the rogue
/// dimensions every published embedding table has.
const ROGUE: usize = 24;

fn splitmix(z: &mut u64) -> u64 {
    *z = z.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut x = *z;
    x = (x ^ (x >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    x = (x ^ (x >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    x ^ (x >> 31)
}

fn normal(z: &mut u64) -> f32 {
    let u1 = ((splitmix(z) >> 11) as f64 + 0.5) / (1u64 << 53) as f64;
    let u2 = ((splitmix(z) >> 11) as f64 + 0.5) / (1u64 << 53) as f64;
    ((-2.0 * u1.ln()).sqrt() * (std::f64::consts::TAU * u2).cos()) as f32
}

fn env_usize(name: &str, d: usize) -> usize {
    std::env::var(name)
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(d)
}

fn main() {
    let device = Default::default();
    let client = Rt::client(&device);

    let n = env_usize("INK_ANN_N", 201024);
    let q_count = env_usize("INK_ANN_Q", 64);
    let budget = env_usize("INK_ANN_BUDGET", 1024);
    let range: f32 = std::env::var("INK_ANN_RANGE")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(12.0);
    let rotated = std::env::var("INK_ANN_ROT")
        .map(|v| v != "0")
        .unwrap_or(true);
    assert_eq!(n % 8, 0, "n must tile as m16n8k16");

    println!(
        "table [{n}, {K}] -> NVFP4 {:.3} GiB, sign sketch {:.3} GiB, basis {}",
        (n * K / 2 + n * K / 16) as f64 / (1u64 << 30) as f64,
        (n * K / 8 + n * 4) as f64 / (1u64 << 30) as f64,
        if rotated { "ROTATED" } else { "RAW (ablation)" }
    );

    // ---- the table ---------------------------------------------------------
    //
    // Built on the host in BF16 and quantised on the device by the same kernel
    // the runtime's own bind calls, so the codes this gate scans are the codes
    // a real run would scan. A table generated straight into NVFP4 would be a
    // different distribution -- the quantiser's rounding is part of what the
    // sketch has to survive.
    let t0 = Instant::now();
    let mut w = vec![bf16::ZERO; n * K];
    let mut z = 0xA11C_E5EEu64;
    for i in 0..n {
        // Row norms spread over a decade, the way token frequency spreads them.
        let s = (0.3 + 2.7 * ((splitmix(&mut z) >> 40) as f32 / 16777216.0)) / (K as f32).sqrt();
        for d in 0..K {
            let heavy = if d % (K / ROGUE) == 0 { 10.0 } else { 1.0 };
            w[i * K + d] = bf16::from_f32(normal(&mut z) * s * heavy);
        }
    }
    println!(
        "  table built on the host in {:.1} s",
        t0.elapsed().as_secs_f64()
    );

    let wh = client.create_from_slice(bf16::as_bytes(&w));
    let packed = quantize_nvfp4_bf16(&client, &wh, n, K);
    let _ = future::block_on(client.sync());
    drop(wh);

    // ---- the sketch --------------------------------------------------------
    let t0 = Instant::now();
    let sketch = annhead::build_sketch(
        &client,
        &packed.0,
        &packed.1,
        n,
        K,
        1.0,
        0x414E_4E01,
        rotated,
    );
    let _ = future::block_on(client.sync());
    let build_s = t0.elapsed().as_secs_f64();
    println!(
        "  sketch built in {build_s:.2} s ({:.1} M rows/s), {} live rows, mean norm {:.4}",
        n as f64 / build_s / 1e6,
        sketch.live_rows,
        sketch.mean_norm
    );

    // ---- queries: near-ties between two rows -------------------------------
    let mut queries = Vec::with_capacity(q_count * K);
    for _ in 0..q_count {
        let a = (splitmix(&mut z) % n as u64) as usize;
        let b = (splitmix(&mut z) % n as u64) as usize;
        for d in 0..K {
            let mix = 0.5 * (w[a * K + d].to_f32() + w[b * K + d].to_f32());
            queries.push(mix * 400.0 + normal(&mut z) * 0.05);
        }
    }
    drop(w);

    // ---- exact, and approximate, on the same bytes -------------------------
    let mut agree = 0usize;
    let mut shortlisted = 0usize;
    let mut short_sum = 0usize;
    let mut gap_sum = 0f64;
    let mut err_sum = 0f64;
    let mut exact_s = f64::MAX;
    let mut ann_s = f64::MAX;

    for qi in 0..q_count {
        let qslice = &queries[qi * K..(qi + 1) * K];
        // The exact lane wants a BF16 activation padded to one m-tile, which is
        // what the runtime hands it.
        let qb: Vec<bf16> = qslice.iter().map(|v| bf16::from_f32(*v)).collect();
        let mut pad = vec![bf16::ZERO; 16 * K];
        pad[..K].copy_from_slice(&qb);
        let ah = client.create_from_slice(bf16::as_bytes(&pad));
        let qh = client.create_from_slice(bf16::as_bytes(&qb));

        let t0 = Instant::now();
        let ex = w4a16_linear_launch::<Rt>(&client, &ah, &packed.0, &packed.1, 16, K, n, 1.0);
        let _ = future::block_on(client.sync());
        let dt = t0.elapsed().as_secs_f64();
        if qi >= 2 {
            exact_s = exact_s.min(dt);
        }

        let t0 = Instant::now();
        let (ap, stat) = annhead::ann_logits::<Rt, bf16>(
            &client, &sketch, &packed.0, &packed.1, &qh, 1.0, budget, range,
        );
        let _ = future::block_on(client.sync());
        let dt = t0.elapsed().as_secs_f64();
        if qi >= 2 {
            ann_s = ann_s.min(dt);
        }

        let exr = client.read_one_unchecked(ex);
        let exact: &[f32] = f32::from_bytes(&exr);
        let apr = client.read_one_unchecked(ap);
        let approx: &[f32] = f32::from_bytes(&apr);

        let mut best = 0usize;
        let mut second = f32::NEG_INFINITY;
        for j in 0..n {
            if exact[j] > exact[best] {
                second = exact[best];
                best = j;
            } else if exact[j] > second {
                second = exact[j];
            }
        }
        let mut abest = 0usize;
        for j in 0..n {
            if approx[j] > approx[abest] {
                abest = j;
            }
        }
        if abest == best {
            agree += 1;
        }
        if approx[best] >= stat.floor {
            shortlisted += 1;
        }
        short_sum += stat.shortlist;
        gap_sum += (exact[best] - second) as f64;
        err_sum += (exact[best] - approx[best]).abs() as f64;
    }

    let q = q_count as f64;
    println!("\n=== recall, {q_count} near-tie queries, n = {n} ===");
    println!(
        "  recall@1                     {:.4}  ({agree}/{q_count})",
        agree as f64 / q
    );
    println!(
        "  exact winner shortlisted     {:.4}  ({shortlisted}/{q_count})",
        shortlisted as f64 / q
    );
    println!(
        "  mean shortlist               {:.0} rows (budget {budget})",
        short_sum as f64 / q
    );
    println!("  mean exact top1-top2 gap     {:.4} logits", gap_sum / q);
    println!("  mean |exact - approx| at top {:.4} logits", err_sum / q);

    // The framing rule, in the same breath as the numbers. Both are the MINIMUM
    // of the warm launches in this process, one query per launch, launch and
    // sync, GPU otherwise idle; the bytes are per QUERY and the achieved rate
    // divides them by that minimum.
    let codes = (n * K / 2 + n * K / 16) as f64;
    let bits = (n * K / 8 + n * 4) as f64;
    println!(
        "\n=== time, min of {} warm launches, launch + sync, per query ===",
        q_count - 2
    );
    println!(
        "  exact w4a16   {:8.3} ms  {:6.1} GB/s over {:.3} GiB of codes+scales",
        exact_s * 1e3,
        codes / exact_s / 1e9,
        codes / (1u64 << 30) as f64
    );
    println!(
        "  aNN           {:8.3} ms  {:6.1} GB/s over {:.3} GiB of signs+alpha  ({:.2}x)",
        ann_s * 1e3,
        bits / ann_s / 1e9,
        bits / (1u64 << 30) as f64,
        exact_s / ann_s
    );
    println!(
        "  NOTE the two GB/s columns are over DIFFERENT tables and are not a \n  \
         like-for-like efficiency comparison; the ratio in the last column is."
    );
}
