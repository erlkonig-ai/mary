//! The unembedding as an approximate maximum-inner-product search.
//!
//! The head multiplies one `[1, 4096]` hidden state against a `[201024, 4096]`
//! table and takes the argmax. That is an exhaustive MIPS brute-forced as a
//! GEMM, and every byte of the table is read to produce one integer. At NVFP4
//! the table is 0.43 GiB and the stage measures 4.6 ms; the whole of that is the
//! DRAM read, so the only way past it is to read FEWER BYTES PER ROW.
//!
//! This lane reads **one bit** per row per dimension instead of four, scans the
//! 0.103 GiB of signs, and rescores a shortlist against the full NVFP4 codes.
//!
//! # Why narrower codes and not fewer rows
//!
//! The obvious aNN answer is a graph or an IVF index: visit 1% of the rows and
//! skip the rest. It does not pay at this shape. `n = 201024` is small for an
//! index and `d = 4096` is enormous; a graph walk pointer-chases 16 KiB rows
//! with no coalescing, and the arithmetic that decides where to go next is
//! serial. Streaming the whole table with perfect coalescing is faster than
//! visiting a twentieth of it at random. So the lever is the CODE WIDTH:
//!
//! ```text
//!   BF16   1.53 GiB   10.1 ms  (measured, tuned cubek lane)
//!   NVFP4  0.43 GiB    4.6 ms  (measured, hand mma lane)
//!   1-bit  0.103 GiB   ~0.6 ms (this lane's scan, at the coalesced ceiling)
//! ```
//!
//! and the shortlist rescore is what buys the accuracy back.
//!
//! # The estimator, and the one scalar it needs per row
//!
//! A sign sketch keeps `sign(w_id)` and throws away every magnitude, so a naive
//! `sum_d q_d * sign(w_id)` is not an estimate of `<q, w_i>` at all — it is off
//! by a per-row factor that depends on how the row's mass is distributed. The
//! correction is one scalar, and it is exact in the sense that matters: it makes
//! the estimator unbiased under the standard RaBitQ argument.
//!
//! Write `u_i = w_i / ||w_i||` and `b_i = sign(w_i) / sqrt(D)`, both unit
//! vectors. The projection of `u_i` onto `b_i` has length `<u_i, b_i>`, so
//!
//! ```text
//!   <q, u_i>  ~  <q, b_i> / <u_i, b_i>
//!   <q, w_i>  ~  ||w_i|| * <q, b_i> / <u_i, b_i>
//!             =  (||w_i||^2 / ||w_i||_1) * sum_d q_d * sign(w_id)
//! ```
//!
//! because `<u_i, b_i> = ||w_i||_1 / (sqrt(D) * ||w_i||_2)` and the two
//! `sqrt(D)` cancel. So **`alpha_i = ||w_i||^2 / ||w_i||_1`** is the whole
//! per-row correction: one f32 per row, 0.8 MB against the sketch's 103 MB,
//! and `est_i = alpha_i * <q, sign(w_i)>`.
//!
//! It is worth being clear about what this does and does not buy. It removes the
//! *systematic* per-row scale error, which is the one that would otherwise make
//! the sketch rank rows by how spiky they are rather than by how well they match.
//! The residual is the projection of `q` onto the subspace `b_i` discards, and
//! that is what the shortlist is for.
//!
//! # The rotation, and why it is not optional
//!
//! Everything above assumes the quantisation error is unstructured. It is not:
//! embedding tables have strongly anisotropic coordinates (a handful of
//! "rogue dimensions" carry disproportionate mass in every published LLM), and a
//! sign sketch on raw coordinates throws away exactly the dimensions that
//! discriminate. Worse, the error is then correlated with vocabulary structure —
//! whole regions of the space get systematically under-estimated, and a token
//! that lives there is never shortlisted, never rescored, and therefore CAN NEVER
//! BE EMITTED.
//!
//! A fixed random rotation `R` fixes that. `R` is orthogonal, so
//! `<Rq, Rw> = <q, w>` exactly — the rescore is untouched — but the sketch is now
//! taken in a basis where no coordinate is special, and the error spreads evenly
//! over the vocabulary instead of pooling on whichever tokens happen to load the
//! rogue dimensions. This is the RaBitQ construction and it costs nothing at
//! decode: `R = (1/sqrt(D)) * H * D_s` with `H` the Hadamard matrix and `D_s` a
//! seeded random sign diagonal is `O(D log D)` — twelve butterfly stages on 4096
//! elements, microseconds against the scan's milliseconds.
//!
//! `D_s` is essential and not decoration. `H` alone is a fixed, highly
//! structured map: its first row is all-ones, so `(Hw)_0` is just the coordinate
//! sum and a sketch of `Hw` would have its own privileged directions. Seeding
//! `D_s` keeps the rotation reproducible across a rebuild while making the
//! *structure* per-checkpoint rather than universal.
//!
//! # Query noise is the third mitigation, and it does a different job
//!
//! The rotation decorrelates the error from vocabulary structure. It does not
//! make the error non-deterministic: given a hidden state, a row that this
//! sketch under-estimates is under-estimated EVERY time, excluded from the
//! shortlist every time, and the exact rescore never sees it. Over a long
//! generation that is a permanently invisible region of the vocabulary.
//!
//! Perturbing the QUERY re-rolls it. `sign(R(h + eps))` differs from `sign(Rh)`
//! in a few coordinates, which changes which bits agree, which changes the error
//! for every row at once. Nothing is permanently invisible.
//!
//! Adding noise to the SCORES cannot do this, and the distinction is the whole
//! point: `argmax(est_i + g_i)` still under-selects a row whose ESTIMATE is
//! biased low, because the noise rides on top of the bias instead of re-rolling
//! it. Gumbel-in-the-scan also presupposes a scan — it is only defined if you
//! visit every row — so it silently constrains the retrieval structure to be
//! linear, which is the thing aNN exists to escape. Query noise perturbs the
//! INPUT and composes with any structure.
//!
//! The knob lives in [`super::super::super`]'s forward pass rather than here,
//! because it is the model's sampling temperature and applies to the exact lane
//! too. See `INK_TEMP`.
//!
//! # The shortlist, and why the floor is chosen by histogram
//!
//! The scan produces `est` for every row. Selecting the top `S` of 201024 on the
//! device could be a sort (too slow), a `k`-pass scan (fine at
//! [`super::routetopk`]'s 256 experts, hopeless at 201024), or a THRESHOLD.
//!
//! A threshold is the right shape here for a reason beyond speed: the question
//! the rescore actually asks is not "give me `S` rows", it is "give me every row
//! whose estimate could still beat the best one" — `candidates_above(floor)`.
//! `S` is a budget, not a semantic. So: one atomic max gives `M`, a 1024-bin
//! histogram over `[M - RANGE, M]` gives the distribution, a one-cube kernel
//! walks the histogram down from the top until the cumulative count reaches the
//! budget and writes the floor, and a compaction pass emits the indices. Four
//! kernels, each touching 0.8 MB or less, against a 103 MB scan — they do not
//! appear in the timing.
//!
//! The floor is derived rather than fixed because a fixed `M - tau` cannot know
//! how crowded the top is, and it is exactly the crowded steps (a genuine
//! near-tie between two plausible tokens) where the shortlist must not overflow.
//!
//! # What the lane returns
//!
//! A full `[1, n]` f32 row, exactly as the exact lane does — **not** a sparse
//! candidate list. Shortlisted rows carry their exact NVFP4 score; every other
//! row carries its sketch estimate. That makes this a drop-in for
//! [`super::burn::linear_w`]: the host argmax, the top-5 report and any softmax
//! downstream all read the row they already read.
//!
//! The blend is not a compromise, it is the honest thing to return. The estimate
//! IS this lane's opinion about a row it did not rescore, and writing `-inf`
//! there would be a stronger claim than the lane can support (and would break
//! any softmax that reads the row). The rows that decide the argmax are exact.
//!
//! # Scope: one row
//!
//! [`ann_logits`] handles `m == 1`. A verify pass with `m > 1` falls back to the
//! exact lane, and that is the right default rather than a gap: the exact GEMM
//! amortises ONE table read over all `m` rows, so at `m = 4` it is already
//! within 2x of what four independent scans would cost, and the decode step —
//! which is `m = 1`, and which is where the 4.6 ms lives — is the whole target.
//! Widening the scan to `m` queries is a change to one inner loop (an
//! accumulator per query, the same bits) and is left until a verify-width
//! measurement asks for it.

use cubecl::e4m3;
use cubecl::prelude::*;
use cubecl::server::Handle;

/// Sign bits packed into one `u32` word.
pub const BITS_PER_WORD: usize = 32;
/// Units in a build cube. One cube transforms one row, so this is also the
/// parallelism of the 4096-point Hadamard transform.
pub const BUILD_UNITS: u32 = 256;
/// Units in a scan cube. One unit owns one row, so this is also the number of
/// consecutive rows a cube reads per bit-plane word — 256 `u32` = 1024 bytes,
/// which is what makes the scan's global reads fully coalesced.
pub const SCAN_UNITS: u32 = 256;
/// Bins in the floor histogram.
pub const HIST_BINS: usize = 1024;
/// Units in the histogram and compaction cubes.
pub const AUX_UNITS: u32 = 256;
/// Lanes in a rescore plane. One plane rescores one candidate row.
pub const PLANE: u32 = 32;

/// A large finite negative, not `-inf`: this value reaches a softmax downstream
/// and `-inf` there produces a NaN if it is ever the row maximum.
const VERY_LOW: f32 = -3.0e38;

/// The 1-bit sign sketch of one weight table, in the rotated basis.
///
/// `bits` is `[k / 32, n]` `u32` — **bit-plane order, not row order**. That
/// transpose is the single most important decision in this module. In row order
/// a scan cube's 256 units would read 256 words 512 bytes apart, which is the
/// same uncoalesced 32-bytes-per-2048 pattern that holds
/// [`super::w4a16gemm`] to 98 GB/s against a 158-172 GB/s coalesced ceiling. In
/// bit-plane order the same 256 units read 256 CONSECUTIVE words, and the scan
/// runs at the streaming ceiling because there is nothing else for it to do.
pub struct Sketch {
    /// `[k / 32, n]` `u32`, bit `j` of word `w` = `sign(R w_i)_(32w + j) < 0`.
    pub bits: Handle,
    /// `[n]` f32, `||R w_i||^2 / ||R w_i||_1` — the unbiasing scalar.
    pub alpha: Handle,
    /// `[k]` f32, the seeded `+-1` diagonal of the rotation.
    pub dsign: Handle,
    pub n: usize,
    pub k: usize,
    /// Mean `||w_i||` over the rows that carry weight. The temperature knob
    /// needs it: query noise of standard deviation `sigma` induces logit noise
    /// of standard deviation `sigma * ||w_i||`, so a temperature stated in
    /// LOGIT units has to be divided by a row norm to become a hidden-state
    /// sigma. Recorded at build because this is the only place the norms exist.
    pub mean_norm: f32,
    /// Whether the sketch was taken in the rotated basis.
    ///
    /// A field and not a constant because "the rotation is not optional" is a
    /// claim, and a claim wants an ablation that can be RUN rather than an
    /// assertion in a doc comment. `false` builds the sketch on raw
    /// coordinates, which is what every argument above says should be worse,
    /// and `inkling_ann_gate` is where the two are put beside each other.
    pub rotated: bool,
    /// Rows with any weight at all. The checkpoint pads the vocabulary to a
    /// multiple of the MMA's `n` tile and the padding rows are exactly zero;
    /// they are excluded here rather than sliced off afterwards, because a
    /// shortlist that spends its budget on padding has already lost.
    pub live_rows: usize,
}

impl Sketch {
    /// Bytes the sketch occupies on the device.
    pub fn bytes(&self) -> usize {
        self.n * self.k / 8 + self.n * 4
    }
}

/// What one [`ann_logits`] call did, for the report and for the recall gate.
#[derive(Clone, Copy, Debug, Default)]
pub struct AnnStat {
    /// Rows whose estimate cleared the floor and were rescored exactly.
    pub shortlist: usize,
    /// The floor, in logit units.
    pub floor: f32,
    /// The largest sketch estimate.
    pub est_max: f32,
}

// ---------------------------------------------------------------------------
// Build
// ---------------------------------------------------------------------------

/// One cube per weight row: dequantise it, rotate it, keep the signs and the two
/// norms the estimator needs.
///
/// The row is materialised in shared memory as f32 (16 KiB at `k = 4096`) and
/// nowhere else — this is the one place in the lane where a full-width row
/// exists, and it exists for microseconds inside a cube rather than as a buffer.
/// The alternative, dequantising the whole table to a `[201024, 4096]` scratch
/// and transforming that, would write and re-read 3.3 GB to save nothing.
///
/// `k` must be a power of two: the transform below is a radix-2 Hadamard
/// butterfly and there is no padding path. 4096 is what this model has.
#[cube(launch_unchecked)]
#[allow(clippy::too_many_arguments)]
fn sketch_build_kernel(
    codes: &Array<u32>,
    scales: &Array<e4m3>,
    dsign: &Array<f32>,
    bits: &mut Array<u32>,
    alpha: &mut Array<f32>,
    norm: &mut Array<f32>,
    n: u32,
    scale2: f32,
    inv_sqrt_k: f32,
    hlim: u32,
    #[comptime] k: u32,
    #[comptime] units: u32,
) {
    let row = CUBE_POS_X;
    let u = UNIT_POS_X;
    let per = comptime!(k / units);
    let wpr = comptime!((k / 8) as usize);
    let spr = comptime!((k / 16) as usize);

    let mut x = SharedMemory::<f32>::new(comptime!(k as usize));
    let mut red = SharedMemory::<f32>::new(comptime!(2 * units as usize));

    // Dequantise into shared memory, applying the rotation's sign diagonal on
    // the way in. Eight consecutive `i` share one packed word and sixteen share
    // one scale, so the units of a warp broadcast rather than issuing 32 loads.
    let cbase = row as usize * wpr;
    let sbase = row as usize * spr;
    for t in 0..per {
        let i = (u + t * units) as usize;
        let word = codes[cbase + i / 8];
        let s = f32::cast_from(scales[sbase + i / 16]);
        let code = (word >> (4 * (i % 8)) as u32) & 15u32;
        x[i] = crate::models::inkling::fp4quant::e2m1_value(code) * s * scale2 * dsign[i];
    }
    sync_cube();

    // In-place Hadamard transform, log2(k) stages of k/2 butterflies. Butterfly
    // `p` at stage `len` pairs `i` with `i + len` where `i` is `p` with a zero
    // inserted at bit position log2(len) -- written as the div/mod form because
    // `len` is a runtime value here and a shift by a runtime amount is the same
    // instruction anyway.
    // `hlim` is `k` in the rotated basis and 1 in the raw one, where the loop
    // does not run at all. A runtime bound rather than a comptime flag because
    // the two arms differ only in how many butterfly stages happen, and one
    // kernel that can do zero of them is less code than two kernels.
    // `u32::new` and not `1u32`: a literal is a COMPTIME value in a cube, and
    // assigning to one inside a runtime loop is the "mutable operation on a
    // const variable" the expander refuses. The same reason `stride` below
    // comes off `CUBE_DIM_X` rather than off the comptime `units`.
    let mut len = u32::new(1);
    while len < hlim {
        let half = comptime!(k / 2);
        let bper = comptime!(half / units);
        for t in 0..bper {
            let p = u + t * units;
            let i = (p / len) * 2 * len + (p % len);
            let a = x[i as usize];
            let b = x[(i + len) as usize];
            x[i as usize] = a + b;
            x[(i + len) as usize] = a - b;
        }
        sync_cube();
        len *= 2;
    }

    // `H` is `sqrt(k)` times orthogonal, so this restores `||Rw|| == ||w||`. The
    // SIGNS do not care, but `alpha` does: it is a ratio of a square to a
    // first power, so it scales with the vector and would be off by 64 at
    // `k = 4096` if the transform were left unnormalised. That error is a
    // uniform factor on every logit -- invisible in an argmax, and a silent 64x
    // temperature change in anything that reads the row.
    let mut l1 = f32::new(0.0);
    let mut l2 = f32::new(0.0);
    for t in 0..per {
        let i = (u + t * units) as usize;
        let v = x[i] * inv_sqrt_k;
        x[i] = v;
        l1 += Abs::abs(v);
        l2 += v * v;
    }
    red[u as usize] = l1;
    red[(units + u) as usize] = l2;
    sync_cube();
    let mut stride = CUBE_DIM_X / 2;
    while stride > 0 {
        if u < stride {
            red[u as usize] += red[(u + stride) as usize];
            red[(units + u) as usize] += red[(units + u + stride) as usize];
        }
        sync_cube();
        stride /= 2;
    }

    // One unit per 32 elements packs the signs. `bits` is bit-plane ordered, so
    // this write is a scatter with stride `n` -- 128 separate lines per row. It
    // is paid once at load and it is what buys the scan its coalesced read.
    let words = comptime!(k / 32);
    if u < words {
        let base = (u * 32) as usize;
        let mut w = u32::new(0);
        #[unroll]
        for j in 0..32u32 {
            if x[base + j as usize] < 0.0 {
                w |= 1u32 << j;
            }
        }
        bits[(u * n + row) as usize] = w;
    }

    if u == 0 {
        let s1 = red[0];
        let s2 = red[units as usize];
        // A padding row is exactly zero and has no direction; `alpha = 0` makes
        // its estimate zero, and `live_rows` on the host side keeps it out of
        // the shortlist regardless. Guarding on `s1` and not on `s2` is
        // deliberate: `s1` is the denominator.
        alpha[row as usize] = if s1 > 0.0 { s2 / s1 } else { f32::new(0.0) };
        norm[row as usize] = Sqrt::sqrt(s2);
    }
}

/// Build the sign sketch of an NVFP4 table.
///
/// `seed` fixes the rotation. It is stored nowhere: the sketch is rebuilt from
/// the table on every load, so the only requirement is that the same seed
/// produces the same `D_s` within a process — which it does, from a splitmix
/// step with no dependence on anything but the seed and the index.
pub fn build_sketch<R: Runtime>(
    client: &ComputeClient<R>,
    codes: &Handle,
    scales: &Handle,
    n: usize,
    k: usize,
    scale2: f32,
    seed: u64,
    rotated: bool,
) -> Sketch {
    assert!(
        k.is_power_of_two(),
        "the sign sketch rotates by a radix-2 Hadamard transform and k = {k} is not a power of two"
    );
    assert!(
        k as u32 >= BUILD_UNITS * 2,
        "k = {k} is too narrow for a {BUILD_UNITS}-unit build cube"
    );
    assert_eq!(k % 32, 0, "k = {k} does not pack into 32-bit sign words");

    let words = k / BITS_PER_WORD;
    let bits = client.empty(words * n * core::mem::size_of::<u32>());
    let alpha = client.empty(n * core::mem::size_of::<f32>());
    let norm = client.empty(n * core::mem::size_of::<f32>());

    // The `+-1` diagonal. Uploaded as f32 rather than as a bitmask because the
    // build kernel multiplies by it once per element and a branch there would
    // cost more than the 16 KiB.
    let ds: Vec<f32> = (0..k)
        .map(|i| {
            let mut z = seed.wrapping_add(0x9E37_79B9_7F4A_7C15u64.wrapping_mul(i as u64 + 1));
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
            z ^= z >> 31;
            if !rotated || z & 1 == 0 {
                1.0f32
            } else {
                -1.0f32
            }
        })
        .collect();
    let dsign = client.create_from_slice(f32::as_bytes(&ds));

    unsafe {
        sketch_build_kernel::launch_unchecked::<R>(
            client,
            CubeCount::Static(n as u32, 1, 1),
            CubeDim::new_1d(BUILD_UNITS),
            ArrayArg::from_raw_parts(codes.clone(), n * k / 8),
            ArrayArg::from_raw_parts(scales.clone(), n * k / 16),
            ArrayArg::from_raw_parts(dsign.clone(), k),
            ArrayArg::from_raw_parts(bits.clone(), words * n),
            ArrayArg::from_raw_parts(alpha.clone(), n),
            ArrayArg::from_raw_parts(norm.clone(), n),
            n as u32,
            scale2,
            if rotated {
                1.0f32 / (k as f32).sqrt()
            } else {
                1.0f32
            },
            if rotated { k as u32 } else { 1u32 },
            k as u32,
            BUILD_UNITS,
        )
    };

    // The norms exist only here, and only two numbers are wanted off them. Read
    // once, at load, and let the buffer die.
    let raw = client.read_one_unchecked(norm);
    let norms: &[f32] = f32::from_bytes(&raw);
    let live: Vec<f32> = norms.iter().copied().filter(|v| *v > 0.0).collect();
    let live_rows = live.len();
    let mean_norm = if live_rows == 0 {
        1.0
    } else {
        live.iter().sum::<f32>() / live_rows as f32
    };

    Sketch {
        bits,
        alpha,
        dsign,
        n,
        k,
        mean_norm,
        rotated,
        live_rows,
    }
}

// ---------------------------------------------------------------------------
// Rotate the query
// ---------------------------------------------------------------------------

/// `out = (1/sqrt(k)) * H * (D_s . q)`, one cube for the one query row.
///
/// The same transform as the build's, on 16 KiB instead of 0.43 GiB. It is here
/// as its own kernel rather than folded into the scan because the scan has 786
/// cubes and this has one: fusing them would make 785 cubes redo the transform.
#[cube(launch_unchecked)]
fn rotate_query_kernel<E: Scalar>(
    q: &Array<E>,
    dsign: &Array<f32>,
    out: &mut Array<f32>,
    inv_sqrt_k: f32,
    hlim: u32,
    #[comptime] k: u32,
    #[comptime] units: u32,
) {
    let u = UNIT_POS_X;
    let per = comptime!(k / units);
    let mut x = SharedMemory::<f32>::new(comptime!(k as usize));

    for t in 0..per {
        let i = (u + t * units) as usize;
        x[i] = f32::cast_from(q[i]) * dsign[i];
    }
    sync_cube();

    let mut len = u32::new(1);
    while len < hlim {
        let bper = comptime!(k / 2 / units);
        for t in 0..bper {
            let p = u + t * units;
            let i = (p / len) * 2 * len + (p % len);
            let a = x[i as usize];
            let b = x[(i + len) as usize];
            x[i as usize] = a + b;
            x[(i + len) as usize] = a - b;
        }
        sync_cube();
        len *= 2;
    }

    for t in 0..per {
        let i = (u + t * units) as usize;
        out[i] = x[i] * inv_sqrt_k;
    }
}

// ---------------------------------------------------------------------------
// Scan
// ---------------------------------------------------------------------------

/// Map an f32 to a `u32` whose unsigned order is the float's order.
///
/// Needed because the device has an atomic max on integers and not on floats,
/// and logits are signed. Flip the sign bit for non-negatives, flip everything
/// for negatives — the standard total-order embedding, and its inverse is
/// [`from_ordered`].
#[cube]
fn to_ordered(v: f32) -> u32 {
    let b = u32::reinterpret(v);
    let mut o = b ^ 0x8000_0000u32;
    if b >= 0x8000_0000u32 {
        o = !b;
    }
    o
}

/// The DEVICE-side inverse of [`to_ordered`].
///
/// It exists because both the histogram and the floor kernel read the peak, and
/// both of them read it as `f32::reinterpret(peak[0])` at first -- which is the
/// ORDERED word, not the float. The failure was silent and total: the histogram
/// binned against a nonsense maximum, no bin ever reached the budget, the floor
/// clamped to the bottom of the window, and the "shortlist" became the entire
/// vocabulary. Every number the lane produced was still plausible, because
/// rescoring everything is CORRECT -- just slower than the head it replaces.
/// A total-order embedding needs its inverse written down beside it.
#[cube]
fn from_ordered_dev(o: u32) -> f32 {
    let mut b = !o;
    if o >= 0x8000_0000u32 {
        b = o ^ 0x8000_0000u32;
    }
    f32::reinterpret(b)
}

/// The host-side inverse of [`to_ordered`].
fn from_ordered(o: u32) -> f32 {
    let b = if o >= 0x8000_0000 {
        o ^ 0x8000_0000
    } else {
        !o
    };
    f32::from_bits(b)
}

/// One unit per weight row: dot the query against the row's signs.
///
/// The inner loop is the whole lane. A unit holds the query's f32 BIT PATTERNS
/// in shared memory and, for each sign bit, XORs that bit into the float's sign
/// position: `+q_d` when the weight is positive and `-q_d` when it is negative,
/// with no branch, no conversion and no multiply. Every unit in a warp reads the
/// SAME shared address at the same step, so the load is a broadcast rather than
/// 32 bank accesses.
///
/// The global read is `bits[w * n + row]`. Across a cube's 256 units that is 256
/// consecutive words per step — one fully coalesced 1024-byte read — which is
/// the property the bit-plane layout exists for.
#[cube(launch_unchecked)]
#[allow(clippy::too_many_arguments)]
fn ann_scan_kernel(
    bits: &Array<u32>,
    alpha: &Array<f32>,
    qrot: &Array<f32>,
    est: &mut Array<f32>,
    peak: &mut Array<Atomic<u32>>,
    n: u32,
    live: u32,
    #[comptime] k: u32,
    #[comptime] units: u32,
) {
    let u = UNIT_POS_X;
    let row = CUBE_POS_X * units + u;

    let mut qs = SharedMemory::<u32>::new(comptime!(k as usize));
    let mut red = SharedMemory::<u32>::new(comptime!(units as usize));
    let per = comptime!(k / units);
    for t in 0..per {
        let i = (u + t * units) as usize;
        qs[i] = u32::reinterpret(qrot[i]);
    }
    sync_cube();

    let words = comptime!(k / 32);
    let mut acc = f32::new(0.0);
    if row < n {
        for w in 0..words {
            let word = bits[(w * n + row) as usize];
            let base = (w * 32) as usize;
            #[unroll]
            for j in 0..32u32 {
                acc += f32::reinterpret(qs[base + j as usize] ^ (((word >> j) & 1u32) << 31u32));
            }
        }
    }

    let mut v = f32::new(VERY_LOW);
    if row < live {
        v = alpha[row as usize] * acc;
    }
    if row < n {
        est[row as usize] = v;
    }

    // One atomic per CUBE, not per row: the cube maxes in shared memory first.
    red[u as usize] = to_ordered(v);
    sync_cube();
    let mut stride = CUBE_DIM_X / 2;
    while stride > 0 {
        if u < stride {
            let o = red[(u + stride) as usize];
            if o > red[u as usize] {
                red[u as usize] = o;
            }
        }
        sync_cube();
        stride /= 2;
    }
    if u == 0 {
        peak[0].fetch_max(red[0]);
    }
}

// ---------------------------------------------------------------------------
// Floor: histogram, pick, compact
// ---------------------------------------------------------------------------

/// Count estimates into `HIST_BINS` bins spanning `[M - range, M]`.
///
/// Anything below the window lands in no bin at all, which is correct: the floor
/// is never below `M - range`, so a row down there is not a candidate under any
/// budget and counting it would only cost an atomic.
#[cube(launch_unchecked)]
#[allow(clippy::too_many_arguments)]
fn ann_hist_kernel(
    est: &Array<f32>,
    peak: &Array<u32>,
    hist: &mut Array<Atomic<u32>>,
    n: u32,
    range: f32,
    #[comptime] bins: u32,
) {
    let i = ABSOLUTE_POS as u32;
    if i < n {
        let m = from_ordered_dev(peak[0]);
        let v = est[i as usize];
        let d = m - v;
        if d >= 0.0 && d < range {
            // Bin 0 is the TOP of the range. Walking down from the max is what
            // the pick below does, so ordering the bins that way makes it a
            // forward scan.
            let b = u32::cast_from(d * (f32::cast_from(bins) / range));
            if b < bins {
                hist[b as usize].fetch_add(1u32);
            }
        }
    }
}

/// Walk the histogram down from the maximum until `budget` rows are covered, and
/// write the floor that admits them.
///
/// One unit. `bins` is 1024 and this runs once per decode step; a parallel scan
/// here would be more code than the thing it replaces costs.
#[cube(launch_unchecked)]
fn ann_floor_kernel(
    hist: &Array<u32>,
    peak: &Array<u32>,
    floor: &mut Array<f32>,
    budget: u32,
    range: f32,
    #[comptime] bins: u32,
) {
    if UNIT_POS == 0 {
        let m = from_ordered_dev(peak[0]);
        let w = range / f32::cast_from(bins);
        let mut acc = u32::new(0);
        let mut b = u32::new(0);
        let mut chosen = u32::cast_from(bins);
        while b < bins {
            acc += hist[b as usize];
            if acc >= budget && chosen == bins {
                chosen = b;
            }
            b += 1;
        }
        // The floor is the BOTTOM of the chosen bin, so every row the count was
        // taken over clears it. Overshoot is one bin's worth of rows, which at
        // 1024 bins over a range the top of the vocabulary occupies is small,
        // and the compaction's cap catches it either way.
        //
        // `chosen` is clamped to the last bin when the window holds FEWER than
        // `budget` rows. Without the clamp the floor fell one bin BELOW the
        // window, which admits every row in the table -- the shortlist went from
        // "the budget" to "the whole vocabulary" precisely when the top was
        // sparse enough that a shortlist was easiest, and the rescore then cost
        // more than the exact head. The semantics this lane wants are
        // `candidates_above(m - range)`: a budget is a ceiling on the answer,
        // never a reason to widen it.
        if chosen >= bins {
            chosen = bins - 1;
        }
        floor[0] = m - w * f32::cast_from(chosen + 1);
    }
}

/// Emit the index of every row whose estimate clears the floor.
///
/// This is `candidates_above(score_floor)` and nothing more. The cap is a
/// safety rail rather than a policy: the floor was chosen to land under it, and
/// a step that hits it has a pathologically flat top which the report names.
#[cube(launch_unchecked)]
#[allow(clippy::too_many_arguments)]
fn ann_compact_kernel(
    est: &Array<f32>,
    floor: &Array<f32>,
    cand: &mut Array<u32>,
    count: &mut Array<Atomic<u32>>,
    live: u32,
    cap: u32,
) {
    let i = ABSOLUTE_POS as u32;
    if i < live && est[i as usize] >= floor[0] {
        let slot = count[0].fetch_add(1u32);
        if slot < cap {
            cand[slot as usize] = i;
        }
    }
}

// ---------------------------------------------------------------------------
// Exact rescore
// ---------------------------------------------------------------------------

/// One plane per shortlisted row: the exact NVFP4 inner product, written back
/// over the row's estimate.
///
/// Lane `l` of the plane walks the row's packed words at stride 32, so the plane
/// reads 32 consecutive words (128 bytes) per step and the eight query elements
/// each lane needs are the 256 consecutive f32 the plane covers. Both sides
/// coalesce, which is why a gather of a few thousand 2 KiB rows costs tens of
/// microseconds and not milliseconds.
///
/// The `scale2` fold matches [`super::w4a16gemm::w4a16_linear`]: applied once to
/// the accumulated f32 rather than per element. That is a deliberate
/// non-associativity deviation, and it is the same one both device lanes make,
/// so this rescore and the exact head agree with each other rather than with
/// [`super::nvfp4::decode_row`].
#[cube(launch_unchecked)]
#[allow(clippy::too_many_arguments)]
fn ann_rescore_kernel<E: Scalar>(
    codes: &Array<u32>,
    scales: &Array<e4m3>,
    cand: &Array<u32>,
    count: &Array<u32>,
    q: &Array<E>,
    est: &mut Array<f32>,
    scale2: f32,
    #[comptime] k: u32,
) {
    let c = CUBE_POS_X;
    if c < count[0] {
        let row = cand[c as usize];
        let l = UNIT_POS_PLANE;
        let wpr = comptime!(k / 8);
        let spr = comptime!(k / 16);
        let cbase = (row * wpr) as usize;
        let sbase = (row * spr) as usize;

        let mut acc = f32::new(0.0);
        let steps = comptime!(k / 8 / 32);
        for t in 0..steps {
            let i = (l + t * 32) as usize;
            let word = codes[cbase + i];
            // Word `i` covers elements `8i..8i+7`; a 16-element scale block
            // starts at a multiple of 16, so all eight share block `i / 2` and
            // the scale is one load rather than eight.
            let s = f32::cast_from(scales[sbase + i / 2]);
            let qb = i * 8;
            #[unroll]
            for j in 0..8usize {
                let code = (word >> (4 * j) as u32) & 15u32;
                acc += crate::models::inkling::fp4quant::e2m1_value(code)
                    * s
                    * f32::cast_from(q[qb + j]);
            }
        }
        let total = plane_sum(acc);
        if l == 0 {
            est[row as usize] = total * scale2;
        }
    }
}

// ---------------------------------------------------------------------------
// The lane
// ---------------------------------------------------------------------------

/// `q @ w^T` for one query row, by sign-sketch scan and exact shortlist rescore.
///
/// `q` is `[k]` in `E` (f32 or BF16 — whichever the residual stream arrived in).
/// Returns a `[n]` f32 handle: exact where it was rescored, estimated
/// everywhere else, and never `-inf` anywhere.
///
/// `budget` is the shortlist target. `range` is the logit window the floor
/// histogram spans below the maximum; it must be wide enough to hold `budget`
/// rows or the floor saturates at the bottom bin and the shortlist overshoots.
#[allow(clippy::too_many_arguments)]
pub fn ann_logits<R: Runtime, E: Scalar>(
    client: &ComputeClient<R>,
    sketch: &Sketch,
    codes: &Handle,
    scales: &Handle,
    q: &Handle,
    scale2: f32,
    budget: usize,
    range: f32,
) -> (Handle, AnnStat) {
    let n = sketch.n;
    let k = sketch.k;
    let words = k / BITS_PER_WORD;
    let cap = budget * 4;

    let qrot = client.empty(k * core::mem::size_of::<f32>());
    unsafe {
        rotate_query_kernel::launch_unchecked::<E, R>(
            client,
            CubeCount::Static(1, 1, 1),
            CubeDim::new_1d(BUILD_UNITS),
            ArrayArg::from_raw_parts(q.clone(), k),
            ArrayArg::from_raw_parts(sketch.dsign.clone(), k),
            ArrayArg::from_raw_parts(qrot.clone(), k),
            if sketch.rotated {
                1.0f32 / (k as f32).sqrt()
            } else {
                1.0f32
            },
            if sketch.rotated { k as u32 } else { 1u32 },
            k as u32,
            BUILD_UNITS,
        )
    };

    let est = client.empty(n * core::mem::size_of::<f32>());
    // `to_ordered(VERY_LOW)` would be the honest identity; 0 is below it and is
    // what an empty buffer is anyway, so the max starts from the floor of the
    // ordering and the first real row wins.
    let peak = client.create_from_slice(&0u32.to_ne_bytes());
    unsafe {
        ann_scan_kernel::launch_unchecked::<R>(
            client,
            CubeCount::Static((n as u32).div_ceil(SCAN_UNITS), 1, 1),
            CubeDim::new_1d(SCAN_UNITS),
            ArrayArg::from_raw_parts(sketch.bits.clone(), words * n),
            ArrayArg::from_raw_parts(sketch.alpha.clone(), n),
            ArrayArg::from_raw_parts(qrot.clone(), k),
            ArrayArg::from_raw_parts(est.clone(), n),
            ArrayArg::from_raw_parts(peak.clone(), 1),
            n as u32,
            sketch.live_rows as u32,
            k as u32,
            SCAN_UNITS,
        )
    };

    let hist = client.create_from_slice(&vec![0u8; HIST_BINS * 4]);
    unsafe {
        ann_hist_kernel::launch_unchecked::<R>(
            client,
            CubeCount::Static((n as u32).div_ceil(AUX_UNITS), 1, 1),
            CubeDim::new_1d(AUX_UNITS),
            ArrayArg::from_raw_parts(est.clone(), n),
            ArrayArg::from_raw_parts(peak.clone(), 1),
            ArrayArg::from_raw_parts(hist.clone(), HIST_BINS),
            n as u32,
            range,
            HIST_BINS as u32,
        )
    };

    let floor = client.empty(core::mem::size_of::<f32>());
    unsafe {
        ann_floor_kernel::launch_unchecked::<R>(
            client,
            CubeCount::Static(1, 1, 1),
            CubeDim::new_1d(1),
            ArrayArg::from_raw_parts(hist.clone(), HIST_BINS),
            ArrayArg::from_raw_parts(peak.clone(), 1),
            ArrayArg::from_raw_parts(floor.clone(), 1),
            budget as u32,
            range,
            HIST_BINS as u32,
        )
    };

    let cand = client.empty(cap * core::mem::size_of::<u32>());
    let count = client.create_from_slice(&0u32.to_ne_bytes());
    unsafe {
        ann_compact_kernel::launch_unchecked::<R>(
            client,
            CubeCount::Static((n as u32).div_ceil(AUX_UNITS), 1, 1),
            CubeDim::new_1d(AUX_UNITS),
            ArrayArg::from_raw_parts(est.clone(), n),
            ArrayArg::from_raw_parts(floor.clone(), 1),
            ArrayArg::from_raw_parts(cand.clone(), cap),
            ArrayArg::from_raw_parts(count.clone(), 1),
            sketch.live_rows as u32,
            cap as u32,
        )
    };

    unsafe {
        ann_rescore_kernel::launch_unchecked::<E, R>(
            client,
            CubeCount::Static(cap as u32, 1, 1),
            CubeDim::new_1d(PLANE),
            ArrayArg::from_raw_parts(codes.clone(), n * k / 8),
            ArrayArg::from_raw_parts(scales.clone(), n * k / 16),
            ArrayArg::from_raw_parts(cand.clone(), cap),
            ArrayArg::from_raw_parts(count.clone(), 1),
            ArrayArg::from_raw_parts(q.clone(), k),
            ArrayArg::from_raw_parts(est.clone(), n),
            scale2,
            k as u32,
        )
    };

    // Two 4-byte reads. They are a SYNC, and they are the reason this returns a
    // stat at all: without them the lane could not report its own shortlist and
    // an overflow would be silent. Both are already-computed values sitting in
    // device memory when the rescore ends, so this is the pass's existing
    // readback and not a second one.
    let stat = AnnStat {
        shortlist: u32::from_ne_bytes(
            client.read_one_unchecked(count)[..4]
                .try_into()
                .expect("the shortlist counter is one u32"),
        ) as usize,
        floor: f32::from_ne_bytes(
            client.read_one_unchecked(floor)[..4]
                .try_into()
                .expect("the floor is one f32"),
        ),
        est_max: from_ordered(u32::from_ne_bytes(
            client.read_one_unchecked(peak)[..4]
                .try_into()
                .expect("the peak is one ordered u32"),
        )),
    };
    (est, stat)
}

// ---------------------------------------------------------------------------
// The recall gate
// ---------------------------------------------------------------------------

/// What the paired exact/approximate runs have found so far.
///
/// A static rather than a plumbed accumulator for the same reason
/// [`super::bf16gemm::HAND`] is one: the thing being counted happens inside a
/// lane and is read by a report thirty call frames away, and threading a
/// counter through that is more edit than the counter is worth. Indices:
///
/// ```text
///   0  steps on which both lanes ran
///   1  steps whose argmax AGREED
///   2  steps where the exact winner cleared the floor and was rescored
///   3  sum of |exact - approx| at the exact winner, in millionths of a logit
///   4  sum of the exact top-1 to top-2 gap, in millionths of a logit
/// ```
///
/// Index 4 is there because index 1 alone cannot be read: a disagreement at a
/// 0.001-logit gap and a disagreement at a 2-logit gap are not the same event,
/// and the W4A16 head's one-token-in-24 change happened at 0.08. A recall
/// number without the gap distribution beside it is a claim whose evidence has
/// been discarded.
pub static VERIFY: [core::sync::atomic::AtomicU64; 5] =
    [const { core::sync::atomic::AtomicU64::new(0) }; 5];

/// Compare one approximate logit row against the exact one and fold the result
/// into [`VERIFY`].
///
/// Takes both rows as the host already has them, because the host already has
/// them: the exact lane's readback is what the verify arm exists to produce and
/// re-reading it on the device would be measuring a different thing.
pub fn verify_row(exact: &[f32], approx: &[f32], floor: f32) {
    use core::sync::atomic::Ordering::Relaxed;
    assert_eq!(
        exact.len(),
        approx.len(),
        "the two lanes returned rows of different widths"
    );
    let mut best = 0usize;
    let mut second = f32::NEG_INFINITY;
    for (j, &v) in exact.iter().enumerate() {
        if v > exact[best] {
            second = exact[best];
            best = j;
        } else if v > second {
            second = v;
        }
    }
    let mut abest = 0usize;
    for (j, &v) in approx.iter().enumerate() {
        if v > approx[abest] {
            abest = j;
        }
    }
    let scale = 1.0e6f64;
    VERIFY[0].fetch_add(1, Relaxed);
    if abest == best {
        VERIFY[1].fetch_add(1, Relaxed);
    }
    // "Was it rescored" is exactly "did its estimate clear the floor", and the
    // row carries the answer: a rescored entry holds an exact score, so it is
    // at or above the floor by construction. A row that did NOT clear the floor
    // is the failure this lane can have that the exact lane cannot -- the token
    // was never even considered -- and it is worth counting apart from a
    // near-tie lost on the rescore.
    if approx[best] >= floor {
        VERIFY[2].fetch_add(1, Relaxed);
    }
    VERIFY[3].fetch_add(
        (((exact[best] - approx[best]) as f64).abs() * scale) as u64,
        Relaxed,
    );
    if second.is_finite() {
        VERIFY[4].fetch_add(
            (((exact[best] - second) as f64).max(0.0) * scale) as u64,
            Relaxed,
        );
    }
}

/// The verify arm's report, or `None` if it never ran.
pub fn verify_report() -> Option<String> {
    use core::sync::atomic::Ordering::Relaxed;
    let n = VERIFY[0].load(Relaxed);
    if n == 0 {
        return None;
    }
    let agree = VERIFY[1].load(Relaxed);
    let seen = VERIFY[2].load(Relaxed);
    let err = VERIFY[3].load(Relaxed) as f64 / 1.0e6 / n as f64;
    let gap = VERIFY[4].load(Relaxed) as f64 / 1.0e6 / n as f64;
    Some(format!(
        "aNN recall@1 {:.4} ({agree}/{n}), exact winner shortlisted {:.4} ({seen}/{n}), \
         mean |exact - approx| at the winner {err:.4} logits, mean exact top1-top2 gap \
         {gap:.4} logits",
        agree as f64 / n as f64,
        seen as f64 / n as f64,
    ))
}
