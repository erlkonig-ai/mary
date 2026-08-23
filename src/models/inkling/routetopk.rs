//! The router DECISION on the device: sigmoid, top-k, and the log-softmax that
//! turns the chosen logits into weights.
//!
//! # Why this moved
//!
//! [`super::block::route_from_logits`] is the host twin of this kernel and it
//! is still the reference — the rule is subtle (the gate bias shifts the scores
//! used to *pick* and takes no part in the weights; the shared experts compete
//! in the same log-softmax as the chosen routed ones) and one transcription of
//! it is what both lanes are checked against.
//!
//! What made it worth a kernel is the SHAPE, not the rule. At decode the host
//! twin does one row of 256 scores per layer and costs nothing measurable. At a
//! 512-token prefill it does 512 rows per layer, and it did them with a full
//! `sort_by` over 256 experts whose comparator called `sigmoid` on both sides —
//! ~2048 comparisons and ~4096 `exp` calls per row. Measured on this box, layers
//! 0:8, 512-token prefill: 66 ms of a 448 ms pass, and the whole of it lands
//! between a BLOCKING readback and the launch of the expert lane, so the device
//! is idle for every millisecond of it. An `INK_ROUTE_PROBE=1` arm that reuses
//! the previous pass's decision (identical inputs under `INK_REPEAT=1`, so the
//! same decision) put the wall-clock price at 69 ms, interleaved.
//!
//! # The shape of the kernel
//!
//! One cube per token, one unit per routed expert. The units compute the whole
//! row of selection scores into shared memory in parallel; unit 0 then walks
//! that row `top_k` times, taking the next-largest score each time. `top_k` is
//! six and `n_routed` is 256, so that is 1536 comparisons in one thread and no
//! `exp` at all — the transcendentals are all in the parallel half.
//!
//! A sort would be the wrong shape here twice over: it is `n log n` where a
//! `k`-pass scan is `k n` with `k = 6 << log2(256)`, and it would have to be a
//! cooperative sort to use the other 255 units, for a row of 256.
//!
//! # What comes back
//!
//! ONE `[tokens, 2 * top_k + n_shared + 1]` f32 buffer, so the host does one
//! readback instead of three. The expert ids ride in it as f32 because they are
//! integers below 2^24 and a second buffer costs a second round trip; the host
//! casts them back. The last column is the non-finite flag — `row + 1` of a
//! logit that was NaN or infinite, or zero — which is how the host twin's panic
//! survives the move. Several units can write that flag in the same pass and
//! the last writer wins, so it names *a* bad row rather than the first one; the
//! host twin named the first. That is the one promise this weakens, and it
//! weakens it in a case that is already a panic.
//!
//! # Numerics
//!
//! `exp(lp_j - lse)` with `lse = m + ln(sum(exp(lp - m)))` is `exp(lp_j - m)`
//! over `sum(exp(lp - m))`, so the `ln` and the subtraction cancel and neither
//! is computed. The max subtraction stays: it is what keeps a row of very
//! negative logits from underflowing the sum to zero.
//!
//! Device `expf`/`logf` are not the host's libm, so a score within an ULP of
//! its neighbour can select a different expert here than on the host. There is
//! no bit-exactness gate on this runtime (it disagrees with itself on 8.55% of
//! argmax positions between two runs of the same binary); the gate is
//! capability.

use cubecl::prelude::*;
use cubecl::server::Handle;

/// `1 / (1 + e^-x)`, in the branch that does not overflow.
#[cube]
fn sigmoid_dev(x: f32) -> f32 {
    let mut r = f32::new(0.0);
    if x >= 0.0f32 {
        r = 1.0f32 / (1.0f32 + Exp::exp(-x));
    } else {
        let e = Exp::exp(x);
        r = e / (1.0f32 + e);
    }
    r
}

/// `ln(sigmoid(x))`, in the branch that does not underflow.
#[cube]
fn log_sigmoid_dev(x: f32) -> f32 {
    let mut r = f32::new(0.0);
    if x >= 0.0f32 {
        r = -Log::ln(1.0f32 + Exp::exp(-x));
    } else {
        r = x - Log::ln(1.0f32 + Exp::exp(x));
    }
    r
}

/// True for NaN and for either infinity, without an `is_finite` intrinsic.
#[cube]
fn not_finite(v: f32) -> bool {
    let mut bad = false;
    if v != v {
        bad = true;
    }
    if v > 3.4028235e38f32 {
        bad = true;
    }
    if v < -3.4028235e38f32 {
        bad = true;
    }
    bad
}

/// One cube per token; `logits` is `[tokens, stride]` with the `n_routed`
/// routed rows first and the `n_shared` shared rows after them.
///
/// `stride` is a separate argument from `n_routed + n_shared` because the BF16
/// router arm's weight carries the instruction's n padding, so the row is wider
/// than the model's. The pad columns sit past the shared rows and are never
/// read here — which is the same thing `drop_pad_cols` did on the host, done by
/// not indexing instead of by copying.
#[cube(launch_unchecked)]
#[allow(clippy::too_many_arguments)]
fn router_topk_kernel(
    logits: &Array<f32>,
    bias: &Array<f32>,
    out: &mut Array<f32>,
    scale: f32,
    #[comptime] stride: u32,
    #[comptime] n_routed: u32,
    #[comptime] n_shared: u32,
    #[comptime] top_k: u32,
) {
    let t = CUBE_POS_X;
    let u = UNIT_POS_X;
    let base = (t * stride) as usize;
    let width = comptime!(2 * top_k + n_shared + 1);

    let mut sh = SharedMemory::<f32>::new(comptime!(n_routed as usize));
    let mut flag = SharedMemory::<u32>::new(1usize);

    if u == 0 {
        flag[0] = 0u32;
    }
    sync_cube();

    // The selection score of one routed expert, in parallel.
    if u < n_routed {
        let v = logits[base + u as usize];
        if not_finite(v) {
            flag[0] = u + 1;
        }
        sh[u as usize] = sigmoid_dev(v) + bias[u as usize];
    }
    // The shared rows take no part in the selection but do take part in the
    // softmax, so they are checked here and read again below.
    if u < n_shared {
        if not_finite(logits[base + (n_routed + u) as usize]) {
            flag[0] = n_routed + u + 1;
        }
    }
    sync_cube();

    if u == 0 {
        // `top_k` passes of "the largest score left", masking each pick out of
        // the shared row as it is taken. Ascending `e` with a strict `>` keeps
        // the lowest index among equal scores, which is the tie-break the host
        // comparator has; masking is what keeps a pick from being taken twice.
        //
        // The row is scratch by this point — nothing below reads a score — so
        // the mask is written into it rather than into a second array. An
        // earlier version carried the previous pick and admitted only scores
        // strictly below it, which is the same rule expressed as a filter; it
        // is not written that way any more because a filter built out of `<`
        // and `==` also silently excludes a NaN, and a selection that drops an
        // expert is not the failure mode a non-finite logit should have.
        let mut pick = Array::<u32>::new(comptime!(top_k as usize));
        for j in 0..top_k {
            let mut bs = f32::new(-3.4028235e38);
            let mut be = u32::new(0);
            for e in 0..n_routed {
                let s = sh[e as usize];
                if s > bs {
                    bs = s;
                    be = e;
                }
            }
            pick[j as usize] = be;
            sh[be as usize] = -3.4028235e38f32;
        }

        // The chosen routed logits and every shared logit, through the same
        // log-softmax. The bias took part in the selection above and takes no
        // part here.
        let total = comptime!(top_k + n_shared);
        let mut lp = Array::<f32>::new(comptime!(total as usize));
        for j in 0..top_k {
            lp[j as usize] = log_sigmoid_dev(logits[base + pick[j as usize] as usize]);
        }
        for j in 0..n_shared {
            lp[(top_k + j) as usize] = log_sigmoid_dev(logits[base + (n_routed + j) as usize]);
        }
        let mut m = f32::new(-3.4028235e38);
        for j in 0..total {
            if lp[j as usize] > m {
                m = lp[j as usize];
            }
        }
        let mut sum = f32::new(0.0);
        for j in 0..total {
            let p = Exp::exp(lp[j as usize] - m);
            lp[j as usize] = p;
            sum += p;
        }
        let norm = scale / sum;

        let ob = (t * width) as usize;
        for j in 0..top_k {
            out[ob + j as usize] = f32::cast_from(pick[j as usize]);
            out[ob + (top_k + j) as usize] = lp[j as usize] * norm;
        }
        for j in 0..n_shared {
            out[ob + (2 * top_k + j) as usize] = lp[(top_k + j) as usize] * norm;
        }
        out[ob + (width - 1) as usize] = f32::cast_from(flag[0]);
    }
}

/// Launch [`router_topk_kernel`] over a whole pass of router logits.
///
/// Returns the `[tokens, 2 * top_k + n_shared + 1]` f32 buffer described in the
/// module doc. The cube is `n_routed` units wide, which is 256 here.
#[allow(clippy::too_many_arguments)]
pub fn router_topk_launch<R: Runtime>(
    client: &ComputeClient<R>,
    logits: &Handle,
    bias: &Handle,
    tokens: usize,
    stride: usize,
    n_routed: usize,
    n_shared: usize,
    top_k: usize,
    scale: f32,
) -> Handle {
    assert!(
        n_routed <= 1024,
        "one unit per routed expert, and {n_routed} is past a cube"
    );
    assert!(
        top_k <= n_routed,
        "cannot pick {top_k} of {n_routed} experts"
    );
    assert!(
        n_routed + n_shared <= stride,
        "the row holds {stride}, the model wants {}",
        n_routed + n_shared
    );
    let width = 2 * top_k + n_shared + 1;
    let out = client.empty(tokens * width * core::mem::size_of::<f32>());
    unsafe {
        router_topk_kernel::launch_unchecked::<R>(
            client,
            CubeCount::Static(tokens as u32, 1, 1),
            CubeDim::new_1d(n_routed as u32),
            ArrayArg::from_raw_parts(logits.clone(), tokens * stride),
            ArrayArg::from_raw_parts(bias.clone(), n_routed),
            ArrayArg::from_raw_parts(out.clone(), tokens * width),
            scale,
            stride as u32,
            n_routed as u32,
            n_shared as u32,
            top_k as u32,
        )
    };
    out
}
