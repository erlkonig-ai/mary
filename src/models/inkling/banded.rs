//! Sliding-window attention that never builds `[heads, n, n]`.
//!
//! # Why this is the only lane that reaches a long input
//!
//! Thirty-five of Inkling-Small's forty-two attention layers are local, with a
//! sliding window of 512. They were computing the full `n x n` score matrix and
//! then masking 1 - 512/n of it away: at 7,000 tokens that is 13.7 times more
//! attention arithmetic than the model asks for, at 14,124 it is 27.6, and the
//! discarded part is not free -- it is the part that costs `[32, n, n]` f32 of
//! memory, which is 23.8 GiB at 14,124 and 1.3 TB at 100,623.
//!
//! A window is a BAND. Query `q` reads keys `q-511 ..= q`, and there are never
//! more than 512 of them, so the work per query is constant in `n` and the
//! memory is `O(n * head_dim)` -- 1.65 GB at 100,623 tokens against the 1.3 TB
//! the dense form wants for one layer. That is the whole difference between an
//! input this box can run and one it cannot.
//!
//! # The shape
//!
//! One cube per `(head, query)`, `head_dim` units wide, and nothing quadratic
//! is written down at any point. Each cube:
//!
//! 1. loads its query row into shared memory;
//! 2. computes the band's `len <= window` scores into shared memory, unit `u`
//!    taking keys `u, u + head_dim, ...`, each a `head_dim` dot product plus
//!    the relative-position bias for that distance;
//! 3. reduces the band to its maximum and its exponential sum, in shared
//!    memory, as a tree over the units;
//! 4. writes `out[q, h, :]` -- unit `u` owning output dimension `u`, summing
//!    `p_j * v[key_j, u]` over the band.
//!
//! The scores live in `window` floats of shared memory (4 KB at the ceiling)
//! and are gone when the cube ends. There is no score TENSOR.
//!
//! # Why the grid is `(query, head)` in that order
//!
//! Adjacent queries share `window - 1` of their `window` keys, so the whole
//! working set of a run of cubes is one band of K and V -- 512 KB at
//! `head_dim = 128` -- and it stays in cache. Putting the head on X instead
//! would make consecutive cubes read disjoint K, and the same kernel would
//! become bandwidth-bound. This is the cheapest 30x in the file and it is one
//! argument order.
//!
//! # Reading K and V where they already are
//!
//! The dense lane built `[heads, n, head_dim]` copies of K and V by repeating
//! each KV head `groups` times, because a batched matmul needs the head counts
//! to agree. This kernel indexes `h / groups` instead, so the GQA expansion --
//! two more tensors, `heads * n * head_dim` f32 each -- does not happen at all
//! on a local layer. Q, K and V are read in the `[tokens, ...]` layout the
//! projections already produce, and the output is written in the layout `wo`
//! wants, so the transposes go too.
//!
//! # K is transposed, and it is the difference between 187 ms and 6
//!
//! In the score phase unit `u` owns key `lo + u` and walks all `head_dim`
//! dimensions of it. With K in the `[tokens, kv_heads * head_dim]` layout the
//! projection produces, the 32 units of a warp read 32 addresses
//! `kv_heads * head_dim` floats apart: one memory transaction per unit per
//! instruction, thirty-two per warp, for 128 bytes of payload each time. The
//! first version of this kernel did exactly that and ran at **31 GFLOP/s** --
//! 187.6 ms per layer at 7,000 tokens, 41% of all GPU time in the pass, for
//! 5.87 GFLOP of arithmetic. It was not compute-bound and it was not even
//! bandwidth-bound; it was transaction-bound.
//!
//! Handing the kernel K as `[kv_heads, head_dim, tokens]` makes consecutive
//! units read consecutive addresses, which is one transaction per warp instead
//! of thirty-two. The transpose costs one linear pass over
//! `tokens * kv_heads * head_dim` f32 per layer -- 28 MB at 7,000 tokens --
//! against a quadratic saving.
//!
//! V needs no such treatment: in the value phase unit `u` owns output
//! DIMENSION `u` at a fixed key, so `[tokens, kv_heads * head_dim]` already has
//! the units reading one contiguous line.
//!
//! # Every index here is 32 bits
//!
//! A cubecl `usize` inside a kernel is 32 bits on this runtime. The largest
//! index this kernel forms is `tokens * heads * head_dim`, which is 4.1e8 at
//! 100,623 tokens and reaches `u32::MAX` only at 1,048,576 -- exactly the
//! model's `model_max_length`, which is not a coincidence worth relying on.
//! [`banded_attention_launch`] asserts it rather than trusting it, because the
//! last kernel in this module tree that formed a 32-bit product wrong agreed
//! with the reference at 512, 3,732 and 7,000 tokens and was silently wrong
//! above 11,586.

use cubecl::prelude::*;
use cubecl::server::Handle;

/// The widest window this kernel holds scores for, in units of shared f32.
///
/// 1024 floats is 4 KB per cube, which is nothing against the 228 KB a
/// Blackwell SM will give a block; the limit exists so the shared array can be
/// a compile-time size instead of specialising the kernel on every window a
/// config might carry. Inkling's window is 512.
const MAX_WINDOW: u32 = 1024;

/// Negative infinity as the softmax's identity, spelled the way the rest of
/// this module tree spells it.
const NEG_INF: f32 = -3.4028235e38;

/// One `(head, query)` band: scores, softmax and the value average, fused.
///
/// `units` is `head_dim` again, as a RUNTIME scalar. The tree reductions halve
/// a stride down to one, and a comptime value cannot be assigned to inside a
/// kernel -- cubecl refuses it outright, "Can't have a mutable operation on a
/// const variable". One scalar argument is cheaper than unrolling the
/// reduction at compile time and reads the same.
#[cube(launch_unchecked)]
#[allow(clippy::too_many_arguments)]
fn banded_attention_kernel(
    q: &Array<f32>,
    k: &Array<f32>,
    v: &Array<f32>,
    rel: &Array<f32>,
    out: &mut Array<f32>,
    scaling: f32,
    tokens: u32,
    eff: u32,
    window: u32,
    units: u32,
    #[comptime] heads: u32,
    #[comptime] kv_heads: u32,
    #[comptime] head_dim: u32,
) {
    let qi = CUBE_POS_X;
    let h = CUBE_POS_Y;
    let u = UNIT_POS_X;

    let groups = comptime!(heads / kv_heads);
    let kvh = h / groups;
    let q_row = comptime!(heads * head_dim);
    let kv_row = comptime!(kv_heads * head_dim);

    // The band: keys `lo ..= qi`, at most `window` of them.
    let mut lo = u32::new(0);
    if qi + 1 > window {
        lo = qi + 1 - window;
    }
    let len = qi - lo + 1;

    let mut sq = SharedMemory::<f32>::new(comptime!(head_dim as usize));
    let mut ss = SharedMemory::<f32>::new(comptime!(MAX_WINDOW as usize));
    let mut red = SharedMemory::<f32>::new(comptime!(head_dim as usize));

    sq[u as usize] = q[(qi * q_row + h * head_dim + u) as usize];
    sync_cube();

    // --- the band's scores ---------------------------------------------------
    // Unit `u` takes keys `u, u + head_dim, ...`. Each is a full `head_dim` dot
    // product read serially by one unit; the units read different key rows, so
    // the reads are `head_dim` contiguous floats each and the band as a whole is
    // touched once.
    let mut j = u;
    while j < len {
        let key = lo + j;
        // K is DIMENSION-MAJOR here: `[kv_heads, head_dim, tokens]`. Unit `u`
        // holds key `lo + u`, so consecutive units read consecutive addresses
        // and one warp instruction is one memory transaction. In the
        // `[tokens, kv_heads * head_dim]` layout the projection produces, the
        // same read is 32 addresses `kv_heads * head_dim` floats apart -- 32
        // transactions per instruction, and that alone was 187 ms of a 190 ms
        // kernel. See the module doc.
        let kb = kvh * head_dim * tokens + key;
        let mut acc = f32::new(0.0);
        for d in 0..head_dim {
            acc += sq[d as usize] * k[(kb + d * tokens) as usize];
        }
        // The relative-position bias for this distance. Zero past the table,
        // which is the same rule the dense epilogue applies.
        let dist = qi - key;
        let mut b = f32::new(0.0);
        if dist < eff {
            b = rel[(qi * heads * eff + h * eff + dist) as usize];
        }
        ss[j as usize] = acc * scaling + b;
        j += head_dim;
    }
    sync_cube();

    // --- max over the band ---------------------------------------------------
    let mut m = f32::new(NEG_INF);
    let mut jm = u;
    while jm < len {
        if ss[jm as usize] > m {
            m = ss[jm as usize];
        }
        jm += head_dim;
    }
    red[u as usize] = m;
    sync_cube();
    let mut step = units / 2;
    while step > 0 {
        if u < step {
            if red[(u + step) as usize] > red[u as usize] {
                red[u as usize] = red[(u + step) as usize];
            }
        }
        sync_cube();
        step /= 2u32;
    }
    let mx = red[0];

    // --- exponentiate in place, and sum -------------------------------------
    let mut s = f32::new(0.0);
    let mut je = u;
    while je < len {
        let e = Exp::exp(ss[je as usize] - mx);
        ss[je as usize] = e;
        s += e;
        je += head_dim;
    }
    sync_cube();
    red[u as usize] = s;
    sync_cube();
    let mut step2 = units / 2;
    while step2 > 0 {
        if u < step2 {
            red[u as usize] += red[(u + step2) as usize];
        }
        sync_cube();
        step2 /= 2u32;
    }
    let inv = 1.0f32 / red[0];

    // --- the value average ---------------------------------------------------
    // Unit `u` owns output dimension `u`, so the V reads are `head_dim`
    // consecutive floats across the units: one coalesced line per key.
    let mut o = f32::new(0.0);
    for jj in 0..len {
        let key = lo + jj;
        o += ss[jj as usize] * v[(key * kv_row + kvh * head_dim + u) as usize];
    }
    out[(qi * q_row + h * head_dim + u) as usize] = o * inv;
}

/// Run banded attention over a whole prefill and return `[tokens, heads * head_dim]`.
///
/// `q` is `[tokens, heads * head_dim]`, `v` is `[tokens, kv_heads * head_dim]`
/// and `rel` is `[tokens, heads, eff]` -- the layouts the projections already
/// produce, so nothing is padded or repeated. `k` is the one exception: it must
/// be `[kv_heads, head_dim, tokens]`, dimension-major, for the reason in the
/// module doc. That transpose is linear in the sequence and the caller pays it
/// once per layer.
#[allow(clippy::too_many_arguments)]
pub fn banded_attention_launch<R: Runtime>(
    client: &ComputeClient<R>,
    q: &Handle,
    k: &Handle,
    v: &Handle,
    rel: &Handle,
    tokens: usize,
    heads: usize,
    kv_heads: usize,
    head_dim: usize,
    eff: usize,
    window: usize,
    scaling: f32,
) -> Handle {
    assert!(
        tokens > 0 && heads > 0 && kv_heads > 0,
        "an empty attention has no band"
    );
    assert!(
        applies(heads, kv_heads, head_dim, window),
        "this shape is not banded here"
    );
    assert!(
        eff > 0,
        "the relative table must reach at least one distance"
    );
    let q_elems = tokens * heads * head_dim;
    let kv_elems = tokens * kv_heads * head_dim;
    assert!(
        q_elems <= u32::MAX as usize,
        "{tokens} tokens x {heads} heads x {head_dim} is {q_elems} elements, past the 32-bit \
         index every cubecl usize is on this runtime"
    );
    assert!(
        tokens * heads * eff <= u32::MAX as usize,
        "the relative table is {} elements, past the 32-bit index",
        tokens * heads * eff
    );
    let out = client.empty(q_elems * core::mem::size_of::<f32>());
    unsafe {
        banded_attention_kernel::launch_unchecked::<R>(
            client,
            // Query on X so consecutive cubes share their band; see the module
            // doc. Heads on Y, which is 32 against a 65,535 limit.
            CubeCount::Static(tokens as u32, heads as u32, 1),
            CubeDim::new_1d(head_dim as u32),
            ArrayArg::from_raw_parts(q.clone(), q_elems),
            ArrayArg::from_raw_parts(k.clone(), kv_elems),
            ArrayArg::from_raw_parts(v.clone(), kv_elems),
            ArrayArg::from_raw_parts(rel.clone(), tokens * heads * eff),
            ArrayArg::from_raw_parts(out.clone(), q_elems),
            scaling,
            tokens as u32,
            eff as u32,
            window as u32,
            head_dim as u32,
            heads as u32,
            kv_heads as u32,
            head_dim as u32,
        )
    };
    out
}

/// Whether a layer's shape is one this kernel handles.
///
/// Three conditions, and each of them is a real limit rather than caution: the
/// tree reductions halve `head_dim` down to one, so it must be a power of two;
/// the cube is `head_dim` units wide, so it must fit a cube; and the band's
/// scores live in a fixed-size shared array.
pub fn applies(heads: usize, kv_heads: usize, head_dim: usize, window: usize) -> bool {
    head_dim.is_power_of_two()
        && head_dim <= 1024
        && window > 0
        && window <= MAX_WINDOW as usize
        && kv_heads > 0
        && heads % kv_heads == 0
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn inklings_shape_is_banded() {
        assert!(applies(32, 8, 128, 512));
    }

    #[test]
    fn refuses_what_it_cannot_reduce() {
        // head_dim not a power of two: the tree reduction would drop lanes.
        assert!(!applies(32, 8, 96, 512));
        // A window past the shared array.
        assert!(!applies(32, 8, 128, 4096));
        // Heads that do not divide into KV heads: `h / groups` would alias.
        assert!(!applies(32, 7, 128, 512));
    }
}

#[cfg(test)]
mod device_tests {
    use super::*;
    use cubecl::cuda::CudaRuntime;

    /// Deterministic filler. A fixed pattern rather than a seeded RNG so a
    /// failure is reproducible from the source alone.
    fn fill(n: usize, seed: f32) -> Vec<f32> {
        (0..n)
            .map(|i| (i as f32 * 0.7919 + seed).sin() * 0.5 + (i as f32 * 0.1237).cos() * 0.25)
            .collect()
    }

    /// Run the band once and read its output back.
    fn band(
        tokens: usize,
        heads: usize,
        kv_heads: usize,
        head_dim: usize,
        window: usize,
        q: &[f32],
        k: &[f32],
        v: &[f32],
        rel: &[f32],
        eff: usize,
    ) -> Vec<f32> {
        let scaling = 1.0 / head_dim as f32;
        // The kernel wants K dimension-major; the projection produces it
        // token-major. Transposing here rather than through a helper keeps the
        // test's copy of the layout independent of the caller's.
        let mut kt = vec![0f32; k.len()];
        for j in 0..tokens {
            for c in 0..kv_heads {
                for d in 0..head_dim {
                    kt[c * head_dim * tokens + d * tokens + j] =
                        k[j * kv_heads * head_dim + c * head_dim + d];
                }
            }
        }
        let client = <CudaRuntime as Runtime>::client(&Default::default());
        let qh = client.create_from_slice(f32::as_bytes(q));
        let kh = client.create_from_slice(f32::as_bytes(&kt));
        let vh = client.create_from_slice(f32::as_bytes(v));
        let rh = client.create_from_slice(f32::as_bytes(rel));
        let oh = banded_attention_launch(
            &client, &qh, &kh, &vh, &rh, tokens, heads, kv_heads, head_dim, eff, window, scaling,
        );
        f32::from_bytes(&client.read_one(oh).expect("read the band's output")).to_vec()
    }

    /// The inputs one shape needs, as `(q, k, v, rel, eff)`.
    fn inputs(
        tokens: usize,
        heads: usize,
        kv_heads: usize,
        head_dim: usize,
        window: usize,
    ) -> (Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>, usize) {
        let eff = window.min(tokens);
        (
            fill(tokens * heads * head_dim, 0.1),
            fill(tokens * kv_heads * head_dim, 0.3),
            fill(tokens * kv_heads * head_dim, 0.5),
            fill(tokens * heads * eff, 0.7),
            eff,
        )
    }

    /// The band's DEFINING property, checked against the kernel itself rather
    /// than against a second implementation of it.
    ///
    /// A key more than `window - 1` positions behind a query cannot reach it.
    /// So changing that key must leave the query's output exactly where it was,
    /// and changing a key INSIDE the window must move it. Both halves are
    /// necessary: the first alone passes for a kernel that reads nothing, and
    /// the second alone passes for a dense kernel with no band at all.
    ///
    /// This replaces a host f64 transcription of the whole attention. That
    /// transcription was not ground truth -- it was a second, more expensive
    /// computation of a model whose weights are four bits -- and its presence
    /// invited every later reader to treat numerical distance from it as the
    /// question. What decides whether this kernel is right is `golden/paired/`:
    /// thirty-five of forty-two layers run banded, so the capability comparison
    /// exercises it on every item.
    #[test]
    fn the_window_is_exactly_what_the_query_can_reach() {
        for (tokens, heads, kv_heads, head_dim, window) in [
            (40usize, 4usize, 2usize, 8usize, 8usize),
            (70, 8, 2, 16, 12),
            (11, 4, 2, 4, 5),
        ] {
            let (q, k, v, rel, eff) = inputs(tokens, heads, kv_heads, head_dim, window);
            let base = band(
                tokens, heads, kv_heads, head_dim, window, &q, &k, &v, &rel, eff,
            );
            let row =
                |o: &[f32], i: usize| o[i * heads * head_dim..(i + 1) * heads * head_dim].to_vec();

            // The last query, and a key it cannot see.
            let last = tokens - 1;
            let outside = last - window; // window - 1 is the furthest it reaches
            let inside = last;
            for (j, must_move) in [(outside, false), (inside, true)] {
                let mut kk = k.clone();
                let mut vv = v.clone();
                for c in 0..kv_heads * head_dim {
                    kk[j * kv_heads * head_dim + c] += 0.75;
                    vv[j * kv_heads * head_dim + c] -= 0.75;
                }
                let got = band(
                    tokens, heads, kv_heads, head_dim, window, &q, &kk, &vv, &rel, eff,
                );
                let moved = row(&got, last)
                    .into_iter()
                    .zip(row(&base, last))
                    .map(|(a, b)| (a - b).abs())
                    .fold(0f32, f32::max);
                if must_move {
                    assert!(
                        moved > 1e-4,
                        "key {j} is inside the window of query {last} and perturbing it moved                          the answer by {moved:e}"
                    );
                } else {
                    assert_eq!(
                        moved, 0.0,
                        "key {j} is {window} back from query {last} and outside its window,                          but perturbing it moved the answer by {moved:e}"
                    );
                }
            }
        }
    }

    /// GQA indexing, again against the kernel itself: perturbing KV head `c`
    /// may move only the query heads in `c`'s group.
    ///
    /// This is the failure a reference implementation catches by accident and
    /// this catches on purpose -- `h / groups` off by one is a wrong ANSWER
    /// everywhere and a wrong SHAPE nowhere.
    #[test]
    fn a_kv_head_reaches_only_its_own_group() {
        let (tokens, heads, kv_heads, head_dim, window) =
            (24usize, 8usize, 2usize, 16usize, 12usize);
        let groups = heads / kv_heads;
        let (q, k, v, rel, eff) = inputs(tokens, heads, kv_heads, head_dim, window);
        let base = band(
            tokens, heads, kv_heads, head_dim, window, &q, &k, &v, &rel, eff,
        );
        // ONE key, not every key of the head. Adding the same constant to all of
        // a head's keys shifts every score by the same amount and the softmax
        // cancels it exactly -- a perturbation that leaves the answer alone for
        // a reason that has nothing to do with indexing, and the first version
        // of this test failed on precisely that.
        let at = tokens / 2;
        for c in 0..kv_heads {
            let (mut kk, mut vv) = (k.clone(), v.clone());
            for d in 0..head_dim {
                kk[at * kv_heads * head_dim + c * head_dim + d] += 0.5;
                vv[at * kv_heads * head_dim + c * head_dim + d] -= 0.5;
            }
            let got = band(
                tokens, heads, kv_heads, head_dim, window, &q, &kk, &vv, &rel, eff,
            );
            for h in 0..heads {
                let mine = h / groups == c;
                let moved = (0..tokens)
                    .flat_map(|i| {
                        (0..head_dim).map(move |d| i * heads * head_dim + h * head_dim + d)
                    })
                    .map(|o| (got[o] - base[o]).abs())
                    .fold(0f32, f32::max);
                if mine {
                    assert!(moved > 1e-4, "head {h} shares KV head {c} and did not move");
                } else {
                    assert_eq!(
                        moved, 0.0,
                        "head {h} does not share KV head {c} but moved by {moved:e}"
                    );
                }
            }
        }
    }

    /// The model's own shape, as a smoke check: it runs, and what comes out is
    /// finite and is not the same number everywhere.
    ///
    /// 128 units, 32 heads over 8 KV heads, window 512, at a length where the
    /// band is a small fraction of the sequence. There is no tolerance here on
    /// purpose -- a kernel that has gone wrong at this size produces NaNs, zeros
    /// or one repeated value, and a kernel that is subtly off produces a number
    /// only the capability harness can judge.
    #[test]
    fn inklings_shape_at_a_real_length_runs() {
        let (tokens, heads, kv_heads, head_dim, window) =
            (1500usize, 32usize, 8usize, 128usize, 512usize);
        let (q, k, v, rel, eff) = inputs(tokens, heads, kv_heads, head_dim, window);
        let out = band(
            tokens, heads, kv_heads, head_dim, window, &q, &k, &v, &rel, eff,
        );
        assert_eq!(out.len(), tokens * heads * head_dim);
        assert!(
            out.iter().all(|x| x.is_finite()),
            "the band produced a non-finite value"
        );
        let lo = out.iter().copied().fold(f32::INFINITY, f32::min);
        let hi = out.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        assert!(
            hi - lo > 1e-3,
            "every output is {lo}: the band computed nothing"
        );
    }
}
