//! Global attention that never builds `[heads, n, n]` — online softmax, one pass.
//!
//! # The gap this fills
//!
//! Thirty-five of Inkling-Small's forty-two attention layers are local, and
//! [`super::banded`] fuses them. The other **seven are global**, and until this
//! file they went through burn's generic matmul with a MATERIALISED softmax
//! over the whole key axis: `q @ k^T` into a `[heads, rows, tokens]` block, a
//! scale/bias/mask epilogue over that block, `softmax(scores, 2)` over it
//! again, then `p @ v`. The block is bounded only by `INK_QBLOCK`, which
//! bounds `rows` and nothing else, so the key axis is the sequence and the
//! whole thing is linear in `tokens` per query row with a large constant.
//!
//! ## Why the seven matter more than the thirty-five
//!
//! Not at the context we benchmark. At ~3.7k attention is 7.6 ms of a 54 ms
//! decode step and fusing it buys almost nothing; this file is not for that
//! number and should not be tuned against it.
//!
//! It is for the two regimes where the global layers are the constraint:
//!
//! * **Decode at long context.** A global layer reads the whole cache:
//!   `8 kv_heads * 128 * 2 (K and V) * 2 B` is 4 KiB per token per global
//!   layer, so at 1M tokens the seven of them move **28 GB per decode step** in
//!   BF16 against 5.77 GB of weights — attention becomes ~5x the entire rest of
//!   the model. NVFP4 KV (unconditional now) takes that to ~8.1 GB. The
//!   crossover where global-layer KV traffic equals weight traffic is ~400k
//!   tokens. A fused pass does not reduce the KV read — nothing can, the keys
//!   must be looked at — but it removes everything ELSE the old lane moved: the
//!   `[heads, 1, slots]` score row written by the matmul, read and rewritten by
//!   the epilogue, read twice and written once by the softmax, and read again
//!   by `p @ v`. At 1M that row is 128 MB and it is touched six times.
//! * **Prefill.** The global layers cannot materialise `[n, n]` at long `n` at
//!   all, and the old lane's peak working set was never just the score block:
//!   it GQA-expands K and V to `[heads, tokens, head_dim]` first, which is two
//!   819 MB buffers at 100k tokens in BF16, linear in the sequence, with no
//!   knob on them. This kernel indexes `h / groups` instead, so the expansion
//!   does not happen, and the relative-bias table is the only thing left that
//!   scales with the query block.
//!
//! # The shape
//!
//! One cube per `(query tile, KV head, key split)`, 128 units — four planes of
//! thirty-two. A cube holds `ROWS` query rows, and those rows are a
//! `(head, query)` RECTANGLE: `groups` query heads (all the heads that share
//! this KV head) by `ROWS / groups` consecutive queries. That is the whole GQA
//! saving expressed as a tile shape — K and V are read once for all `groups`
//! heads that need them, rather than once per head off an expanded copy.
//!
//! Within a cube, plane `w` owns rows `[w * rpp, (w+1) * rpp)` and lane `l`
//! owns key `l` of the current key tile and output dimensions
//! `l, l+32, l+64, l+96`. So:
//!
//! * a row's thirty-two scores live in ONE plane, and the online softmax's row
//!   max and row sum are `plane_max` / `plane_sum` — no shared-memory tree, no
//!   `sync_cube` per reduction. (`plane_max` lowers to a shuffle loop on this
//!   backend, not to `redux.sync.max.f32`, which sm_121a does not have.)
//! * the output accumulator is `rpp * head_dim / 32` REGISTERS per lane — 32 at
//!   the prefill tile, 4 at the decode tile — and never touches memory until
//!   the cube ends.
//!
//! Nothing quadratic is written down. The largest thing this file allocates is
//! the partial-output buffer, which is `splits * nq * heads * head_dim`.
//!
//! # Online softmax is the one part that is not ordinary GEMM practice
//!
//! Tiling a product over `k` is what every GEMM does. What flash attention
//! genuinely invented is running the softmax's normalisation as a RUNNING
//! quantity: each cube carries `m` (the max seen so far) and `l` (the sum of
//! `exp(s - m)` so far) per row, and when a tile pushes the max up it rescales
//! the accumulator by `exp(m_old - m_new)` before adding the new tile's
//! contribution. The answer is the same as a two-pass softmax to the last bit
//! of the accumulation order, and the key axis is never materialised.
//!
//! `NEG_INF` here is a large FINITE constant, the way the rest of this module
//! tree spells it, and that forces every guard to be explicit: `-3.4e38 -
//! -3.4e38` is `0`, so `exp(s - m)` for a fully-masked tile would be `1` rather
//! than `0` if it were left to the arithmetic. Both the rescale and the
//! exponential are branch-guarded instead of relying on infinity's algebra.
//!
//! # Staging K, and not staging V
//!
//! The score phase has lane `l` walking all `head_dim` dimensions of key
//! `t + l`. In the `[tokens, kv_heads * head_dim]` layout both the projections
//! and the paged cache produce, that is thirty-two addresses `kv_heads *
//! head_dim` floats apart — thirty-two memory transactions per warp
//! instruction, which is the failure [`super::banded`] measured at 187 ms of a
//! 190 ms kernel and fixed by transposing K to `[kv_heads, head_dim, tokens]`.
//!
//! A transpose is not available here. The decode lane reads PAGES of a cache
//! that is stored token-major and quantised token-major, and transposing them
//! per layer per step would cost a full read and a full write of the very
//! bytes the fusion exists to stop moving. So this kernel STAGES instead: the
//! key tile is copied into shared memory by all 128 units in coalesced
//! `head_dim`-wide lines, and the score phase reads it from there, where the
//! access pattern costs nothing. That is the same class of fix a sibling
//! measured as ~2x on the 4-bit weight GEMMs (`fp4_linear_smem`), and it is why
//! a flash-shaped kernel wants no transposed input: it stages by construction.
//!
//! The shared tile is padded to `head_dim + 1` floats a row. Without the pad,
//! lane `l` reading `sk[l * 128 + d]` puts all thirty-two lanes on bank
//! `d % 32` — a thirty-two-way conflict on every single shared read of the
//! inner loop. With it, lane `l` lands on bank `(l + d) % 32`, which is
//! conflict-free.
//!
//! V is NOT staged, for [`super::banded`]'s reason: in the value phase lane `l`
//! owns output DIMENSIONS at a fixed key, so `[tokens, kv_heads * head_dim]`
//! already has consecutive lanes on consecutive addresses. The four planes read
//! the same V line, which L1 serves.
//!
//! # Splitting the key axis, and why decode needs it
//!
//! At decode `nq` is 1, so the grid without a split is `1 x kv_heads x 1` —
//! eight cubes, on a device with dozens of SMs, each walking a million keys
//! serially. That is not a kernel, it is a queue. So the key axis is SPLIT:
//! split `s` runs the same online softmax over its own slice and writes an
//! UNNORMALISED `(o, m, l)` triple, and [`flash_combine_kernel`] merges them
//! with the same rescale the tile loop uses — `exp(m_s - M)` weighting, one
//! division at the end. This is flash-decoding, and it is what turns eight
//! cubes into as many as the device will take.
//!
//! The combine also runs when there is one split, where it is a normalisation
//! pass. That costs one extra read and write of `nq * heads * head_dim` f32 —
//! nothing next to the key axis at the lengths this file is for, and one code
//! path instead of two.
//!
//! # Burn already has a flash kernel, and it was rejected on evidence
//!
//! `cubek-attention` 0.2.0 is in our `Cargo.lock` and already compiled into
//! every mary build — one feature word away from `cubek::attention::launch::
//! launch_ref`. It is not used here, and the reasons are three properties of
//! its interface rather than a preference:
//!
//! * **The relative-position bias is not expressible.** Its mask is a BOOLEAN
//!   predicate — `MaterializedTileMask::should_mask() -> bool`, with a `u8`/
//!   `u32` mask dtype — not an addend. (`MaskTile`'s "additive mask" doc
//!   comment is stale.) And our own pinned burn fork closes the door
//!   independently: `burn-cubecl/src/ops/module.rs:325-330` routes to
//!   `attention_fallback` the moment `attn_bias`, `softcap` or `scale` is
//!   present, and the fallback materialises `[b, h, n, n]` — which is the exact
//!   thing this file exists to stop. Inkling's every layer carries a learned
//!   per-head bias gathered by backward distance; there is no arm of this model
//!   without one.
//! * **A paged, NVFP4 KV cache cannot be handed to it.** K and V must be
//!   strided float tensors read through `AttentionGlobalLayout`; the crate
//!   contains no `paged`, `fp4` or `quant`. Using it means materialising one
//!   contiguous BF16 cache per layer per step — the copy `KvStore::parts` was
//!   changed to stop making.
//! * **GQA 32/8 is silently wrong rather than refused.** `AttentionDims` takes
//!   ONE `num_heads`, from `query.shape[1]`; `key.shape[1]` is never read and
//!   nothing validates the pair.
//!
//! Structurally it also has no split over the key axis — its hypercube is
//! `(seq_q_tile, batch * num_heads)` — so at decode it is thirty-two cubes each
//! walking the whole cache serially, which is the shape this file's splits
//! exist to fix. 0.3.0-pre.3 keeps all four and adds a training backward pass,
//! at the cost of rebasing our cubecl fork off 0.10. Upstream is aware: cubek
//! #284 ("Paged flash attention support") has been open since 2026-05-19 and
//! #310 migrates the kernel to the Tile DSL. This is the same answer
//! [`super::banded`] reached for the thirty-five local layers, for the same
//! kind of reason.
//!
//! One thing upstream has that is worth borrowing later: 0.3's
//! `AttentionGlobalLayout::new` takes separate batch and head strides with
//! explicit broadcast, which makes 32/8 GQA expressible as a strided VIEW — Q
//! as `[8, 4, sq, d]` against K/V as `[8, 1, n, d]` with the head axis
//! broadcast. That is the same correspondence this kernel spells as a cube tile
//! (`groups` heads by `rows / groups` queries), and it is the tidier
//! formulation of the two.
//!
//! # What this does NOT do
//!
//! It does not dequantise NVFP4 in registers. The paged cache hands back
//! dequantised pages (`Fp4PageStore::parts`, one page-sized dequant launch
//! each) and the kernel consumes those. Reading the packed codes directly would
//! save the page-sized round trip through L2, which is real but is not the
//! 28 GB; it is a later change and it is a change to the READER, not to this
//! kernel's shape.
//!
//! It also has no MMA path. At decode `m = 1` and tensor cores buy nothing —
//! the product is a GEMV and the kernel is bandwidth-bound. At prefill `m` is
//! large and they would; see the measurements in the commit for where that
//! leaves the prefill lane against burn's cmma matmul.

use cubecl::prelude::*;
use cubecl::server::Handle;

/// Units per cube. Four planes of thirty-two on this backend.
pub const UNITS: u32 = 128;

/// Lanes in a plane, and therefore keys in one key tile.
///
/// The two are the same number on purpose: it is what makes a row's whole tile
/// of scores live in one plane, so the online softmax's two reductions are
/// `plane_max` and `plane_sum` rather than a shared-memory tree with a
/// `sync_cube` per level.
pub const PLANE: u32 = 32;

/// Planes per cube.
pub const PLANES: u32 = UNITS / PLANE;

/// Query rows in a PREFILL cube: `groups` heads by `ROWS / groups` queries.
///
/// 32 at Inkling's `groups = 4` is eight consecutive queries across all four
/// heads of a KV head. It sets the arithmetic intensity of the whole kernel —
/// one key tile costs `2 * PLANE * head_dim * 4` bytes of K and V and buys
/// `ROWS * PLANE * head_dim * 2` flops, i.e. `ROWS / 4` flops a byte — and it
/// sets the shared budget, which is what stops it being larger.
pub const ROWS_PREFILL: u32 = 32;

/// Query rows in a DECODE cube: one query, all `groups` heads of a KV head.
///
/// A decode step has one query row, so `ROWS` cannot exceed `groups` without
/// padding the tile with rows that do not exist. Four rows is four accumulator
/// registers a lane and the highest occupancy this kernel has, which is what a
/// bandwidth-bound pass wants.
pub const ROWS_DECODE: u32 = 4;

/// Negative infinity as the softmax's identity, spelled the way the rest of
/// this module tree spells it: a large FINITE constant, so every guard against
/// it has to be explicit rather than riding on infinity's algebra.
const NEG_INF: f32 = -3.4028235e38;

/// The shared floats one cube of this shape needs.
///
/// Q tile, key tile padded to `head_dim + 1` a row, and the probability tile.
/// The CUDA static `__shared__` ceiling is 48 KiB and this kernel stays under
/// it deliberately: going above needs an opt-in the runtime does not make, and
/// the occupancy a bigger tile would cost is worth more here than the tile is.
pub fn shared_floats(rows: u32, head_dim: u32) -> usize {
    (rows * head_dim + PLANE * (head_dim + 1) + rows * PLANE) as usize
}

/// One `(query tile, KV head, key split)` of global attention, fused.
///
/// Writes an UNNORMALISED partial: `o = sum exp(s - m) v`, beside `m` and
/// `l = sum exp(s - m)`. [`flash_combine_kernel`] turns a column of those into
/// the answer.
///
/// `k` and `v` are generic in their element type because the cache is BF16 (and
/// the prefill operands are too, under `INK_ACT_BF16`) while everything the
/// softmax touches is f32. The cast happens on the way into shared memory and
/// into a register, which is where the reference implementation puts it as
/// well: narrow operands, wide accumulation.
#[cube(launch_unchecked)]
#[allow(clippy::too_many_arguments)]
fn flash_kernel<KV: Numeric>(
    q: &Array<f32>,
    k: &Array<KV>,
    v: &Array<KV>,
    rel: &Array<f32>,
    po: &mut Array<f32>,
    pml: &mut Array<f32>,
    scaling: f32,
    nq: u32,
    q0: u32,
    k0: u32,
    klo: u32,
    khi: u32,
    eff: u32,
    window: u32,
    slot0: u32,
    splits: u32,
    slots: u32,
    #[comptime] heads: u32,
    #[comptime] kv_heads: u32,
    #[comptime] head_dim: u32,
    #[comptime] rows: u32,
) {
    let qt = CUBE_POS_X;
    let kvh = CUBE_POS_Y;
    let split = CUBE_POS_Z;
    let u = UNIT_POS_X;
    let lane = u % PLANE;
    let plane = u / PLANE;

    let groups = comptime!(heads / kv_heads);
    let bq = comptime!(rows / groups);
    let rpp = comptime!(rows / PLANES);
    // Output dimensions a lane owns. `div_ceil`, not `/`: the model's
    // `head_dim` is 128 and divides exactly, but the test shapes are 4 wide and
    // a kernel that only ran on multiples of the plane could not be checked
    // against the harness that decides whether it is right.
    let dpl = comptime!(head_dim.div_ceil(PLANE));
    let q_row = comptime!(heads * head_dim);
    let kv_row = comptime!(kv_heads * head_dim);
    let skw = comptime!(head_dim + 1);

    let mut sq = SharedMemory::<f32>::new(comptime!((rows * head_dim) as usize));
    let mut sk = SharedMemory::<f32>::new(comptime!((PLANE * (head_dim + 1)) as usize));
    let mut sp = SharedMemory::<f32>::new(comptime!((rows * PLANE) as usize));

    // --- the query tile ------------------------------------------------------
    // Row `r` is `(head group r / bq, query qt * bq + r % bq)`. Consecutive
    // units take consecutive dimensions of one row, so the load is coalesced.
    let mut i = u;
    while i < rows * head_dim {
        let r = i / head_dim;
        let d = i % head_dim;
        let qrow = qt * bq + r % bq;
        let h = kvh * groups + r / bq;
        let mut val = f32::new(0.0);
        if qrow < nq {
            val = q[(qrow * q_row + h * head_dim + d) as usize];
        }
        sq[i as usize] = val;
        i += UNITS;
    }

    // --- this split's slice of the key axis ----------------------------------
    // The tile's queries span absolute positions `q0 + qt*bq ..= q_hi`. No key
    // past `q_hi` can be seen by ANY row here (causality), and with a window no
    // key before `q_lo - window + 1` can be seen by any of them. Trimming the
    // slice by both is not the mask — every cell is still masked individually
    // below — it is what keeps the tile loop off ranges that are entirely dead.
    let q_lo = q0 + qt * bq;
    let mut q_hi = q0 + nq - 1;
    if qt * bq + bq <= nq {
        q_hi = q_lo + bq - 1;
    }

    let mut lo = klo;
    let mut hi = khi;
    // Causality, in chunk-local rows. `q_hi + 1 <= k0` means the whole chunk is
    // in this tile's future.
    if q_hi + 1 <= k0 {
        hi = lo;
    } else if q_hi + 1 - k0 < hi {
        hi = q_hi + 1 - k0;
    }
    // The window's lower edge, likewise.
    if window > 0 && q_lo + 1 > window + k0 {
        let wlo = q_lo + 1 - window - k0;
        if wlo > lo {
            lo = wlo;
        }
    }
    if hi < lo {
        hi = lo;
    }

    // Split the surviving range. `per` is rounded up so `splits` slices cover
    // it; the last ones may be empty, and an empty slice still writes its
    // partial (`m = NEG_INF`, `l = 0`), which the combine weights to zero.
    let span = hi - lo;
    let per = (span + splits - 1) / splits;
    let mut s_lo = lo + split * per;
    if s_lo > hi {
        s_lo = hi;
    }
    let mut s_hi = s_lo + per;
    if s_hi > hi {
        s_hi = hi;
    }

    // --- running softmax state, per row this plane owns ----------------------
    let mut m = Array::<f32>::new(comptime!(rpp as usize));
    let mut l = Array::<f32>::new(comptime!(rpp as usize));
    let mut acc = Array::<f32>::new(comptime!(rpp as usize));
    let mut o = Array::<f32>::new(comptime!((rpp * dpl) as usize));
    #[unroll]
    for ri in 0..rpp {
        m[ri as usize] = f32::new(NEG_INF);
        l[ri as usize] = f32::new(0.0);
    }
    #[unroll]
    for oi in 0..rpp * dpl {
        o[oi as usize] = f32::new(0.0);
    }

    sync_cube();

    let mut t = s_lo;
    while t < s_hi {
        // --- stage the key tile ---------------------------------------------
        // 128 units over `PLANE * head_dim` floats: consecutive units take
        // consecutive dimensions of one key, which is a coalesced line in the
        // token-major layout the projections and the cache both produce. See
        // the module doc for why the destination row is `head_dim + 1` wide.
        let mut ki = u;
        while ki < PLANE * head_dim {
            let j = ki / head_dim;
            let d = ki % head_dim;
            let key = t + j;
            let mut val = f32::new(0.0);
            if key < s_hi {
                val = f32::cast_from(k[(key * kv_row + kvh * head_dim + d) as usize]);
            }
            sk[(j * skw + d) as usize] = val;
            ki += UNITS;
        }
        sync_cube();

        // --- scores: lane `l` owns key `t + l`, for every row this plane holds
        // The dimension loop is outermost so the staged key is read ONCE per
        // dimension and reused across the plane's rows, which is `rpp` times
        // fewer shared reads than the natural nesting.
        #[unroll]
        for ri in 0..rpp {
            acc[ri as usize] = f32::new(0.0);
        }
        for d in 0..head_dim {
            let kd = sk[(lane * skw + d) as usize];
            #[unroll]
            for ri in 0..rpp {
                acc[ri as usize] += sq[((plane * rpp + ri) * head_dim + d) as usize] * kd;
            }
        }

        let key = t + lane;
        let key_abs = k0 + key;
        let visible_key = key < s_hi;

        #[unroll]
        for ri in 0..rpp {
            let r = plane * rpp + ri;
            let qrow = qt * bq + r % bq;
            let h = kvh * groups + r / bq;
            let q_abs = q0 + qrow;

            let mut s = acc[ri as usize] * scaling;
            let mut vis = visible_key && qrow < nq && key_abs <= q_abs;
            if vis && window > 0 && q_abs - key_abs >= window {
                vis = false;
            }
            if vis {
                let dist = q_abs - key_abs;
                if dist < eff {
                    s += rel[((qrow * heads + h) * eff + dist) as usize];
                }
            } else {
                s = f32::new(NEG_INF);
            }

            // The online update. Every step is guarded rather than left to
            // infinity's algebra, because NEG_INF here is finite.
            let tile_max = plane_max(s);
            let mut m_new = m[ri as usize];
            if tile_max > m_new {
                m_new = tile_max;
            }
            let mut alpha = f32::new(1.0);
            if m_new > f32::new(NEG_INF) {
                alpha = Exp::exp(m[ri as usize] - m_new);
            }
            let mut p = f32::new(0.0);
            if vis {
                p = Exp::exp(s - m_new);
            }
            let tile_sum = plane_sum(p);
            l[ri as usize] = l[ri as usize] * alpha + tile_sum;
            m[ri as usize] = m_new;
            #[unroll]
            for di in 0..dpl {
                o[(ri * dpl + di) as usize] *= alpha;
            }
            sp[(r * PLANE + lane) as usize] = p;
        }
        sync_cube();

        // --- the value average ----------------------------------------------
        // Lane `l` owns output dimensions `l, l+32, ...`, so the V read is
        // consecutive across the plane: one coalesced line per key per
        // dimension group, and the four planes share it through L1.
        for j in 0..PLANE {
            let vkey = t + j;
            if vkey < s_hi {
                let vbase = vkey * kv_row + kvh * head_dim;
                #[unroll]
                for di in 0..dpl {
                    let d = lane + di * PLANE;
                    if d < head_dim {
                        let vv = f32::cast_from(v[(vbase + d) as usize]);
                        #[unroll]
                        for ri in 0..rpp {
                            o[(ri * dpl + di) as usize] +=
                                sp[((plane * rpp + ri) * PLANE + j) as usize] * vv;
                        }
                    }
                }
            }
        }
        sync_cube();

        t += PLANE;
    }

    // --- the partial ---------------------------------------------------------
    let slot = slot0 + split;
    #[unroll]
    for ri in 0..rpp {
        let r = plane * rpp + ri;
        let qrow = qt * bq + r % bq;
        let h = kvh * groups + r / bq;
        if qrow < nq {
            let base = ((slot * nq + qrow) * heads + h) * head_dim;
            #[unroll]
            for di in 0..dpl {
                let d = lane + di * PLANE;
                if d < head_dim {
                    po[(base + d) as usize] = o[(ri * dpl + di) as usize];
                }
            }
            if lane == 0 {
                let mb = ((slot * nq + qrow) * heads + h) * 2;
                pml[mb as usize] = m[ri as usize];
                pml[(mb + 1) as usize] = l[ri as usize];
            }
        }
    }
    let _ = slots;
}

/// Merge a column of partials into the answer.
///
/// One cube per `(query row, head)`, `head_dim` units, unit `d` owning output
/// dimension `d`. The merge is the tile loop's rescale one level up: every
/// split's `o` was accumulated against its own `m`, so weighting by
/// `exp(m_s - M)` puts them all on the same exponent and the sum of the
/// likewise-weighted `l` is the denominator.
#[cube(launch_unchecked)]
fn flash_combine_kernel(
    po: &Array<f32>,
    pml: &Array<f32>,
    out: &mut Array<f32>,
    slots: u32,
    nq: u32,
    #[comptime] heads: u32,
    #[comptime] head_dim: u32,
) {
    let qrow = CUBE_POS_X;
    let h = CUBE_POS_Y;
    let d = UNIT_POS_X;

    let mut mx = f32::new(NEG_INF);
    for s in 0..slots {
        let mb = ((s * nq + qrow) * heads + h) * 2;
        if pml[mb as usize] > mx {
            mx = pml[mb as usize];
        }
    }

    let mut num = f32::new(0.0);
    let mut den = f32::new(0.0);
    for s in 0..slots {
        let mb = ((s * nq + qrow) * heads + h) * 2;
        let ms = pml[mb as usize];
        let ls = pml[(mb + 1) as usize];
        let mut wt = f32::new(0.0);
        if ms > f32::new(NEG_INF) {
            wt = Exp::exp(ms - mx);
        }
        den += wt * ls;
        num += wt * po[(((s * nq + qrow) * heads + h) * head_dim + d) as usize];
    }

    let mut res = f32::new(0.0);
    if den > f32::new(0.0) {
        res = num / den;
    }
    out[(qrow * heads * head_dim + h * head_dim + d) as usize] = res;
}

/// The element type a key/value buffer is held in.
///
/// Not `burn::tensor::DType`: this file is runtime-generic the way
/// [`super::banded`] is, and the one thing it needs to know about the cache's
/// dtype is which instantiation of [`flash_kernel`] to launch.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum KvElem {
    F32,
    Bf16,
}

/// One run of contiguous key/value rows the kernel may read.
///
/// A prefill hands one of these — the whole freshly-projected K and V. A decode
/// step hands one PER PAGE, because that is how the cache is stored and
/// rejoining them would be the copy the paged read exists to avoid.
///
/// `base` is the ABSOLUTE sequence position of row 0, which is what causality,
/// the window and the relative distance are all functions of; `lo .. hi` are
/// the rows of this buffer that are live keys, which is how a page's dropped
/// prefix and its tail padding are excluded without slicing the buffer and
/// making its shape walk.
#[derive(Debug, Clone, Copy)]
pub struct KeyRun<'a> {
    pub k: &'a Handle,
    pub v: &'a Handle,
    /// Rows in the buffer.
    pub rows: usize,
    /// Absolute sequence position of row 0.
    pub base: usize,
    /// First live row.
    pub lo: usize,
    /// One past the last live row.
    pub hi: usize,
}

/// How many cubes the launcher aims for before it stops splitting the key axis.
///
/// A decode step's natural grid is `1 x kv_heads x 1` — eight cubes — and a
/// device with dozens of SMs will run that at a few percent of its bandwidth.
/// Splitting to roughly this many is what fills it. Prefill reaches it on the
/// query axis alone and never splits.
const TARGET_CUBES: usize = 1024;

/// The fewest keys worth giving a split.
///
/// Four key tiles. Below that the per-split fixed cost — the query tile load,
/// the partial write, and the combine's extra column — outweighs the
/// parallelism. It matters most at the SHORT end: a local layer's decode step
/// reads a 512-key window, and at 512 keys a split floor of 512 would give the
/// whole layer eight cubes.
const MIN_KEYS_PER_SPLIT: usize = 128;

/// Whether a layer's shape is one this kernel handles.
///
/// Each condition is a real limit. `head_dim` must divide into the plane
/// because a lane owns `head_dim / 32` output dimensions; `rows` must divide
/// into both the planes and the head groups because a cube's rows are a
/// `groups x (rows / groups)` rectangle spread over four planes; and the shared
/// tile must fit the 48 KiB a static `__shared__` gets.
pub fn applies(heads: usize, kv_heads: usize, head_dim: usize, rows: usize) -> bool {
    if kv_heads == 0 || heads % kv_heads != 0 || head_dim == 0 || rows == 0 {
        return false;
    }
    let groups = heads / kv_heads;
    head_dim <= 1024
        && rows % PLANES as usize == 0
        && rows % groups == 0
        && shared_floats(rows as u32, head_dim as u32) * core::mem::size_of::<f32>() <= 48 * 1024
}

/// The cube row count a DECODE step wants: the smallest multiple of [`PLANES`]
/// that also splits into whole head groups.
///
/// One query row across all `groups` heads of a KV head is what a decode step
/// actually has, and at the model's `groups = 4` that is exactly [`PLANES`].
/// Where `groups` does not divide the planes the tile carries query rows that
/// do not exist; they mask out and cost their share of a plane, which is the
/// price of keeping one kernel for both regimes.
pub fn decode_rows(groups: usize) -> usize {
    let mut r = PLANES as usize;
    while !r.is_multiple_of(groups) {
        r += PLANES as usize;
    }
    r
}

/// The cube row count a PREFILL wants: [`ROWS_PREFILL`], rounded up to whole
/// head groups.
pub fn prefill_rows(groups: usize) -> usize {
    let mut r = ROWS_PREFILL as usize;
    while !r.is_multiple_of(groups) {
        r += PLANES as usize;
    }
    r
}

/// How many queries a PREFILL block should hold, from what a block allocates.
///
/// A block allocates three things linear in its rows: the relative-bias table
/// `[rows, heads, eff]`, the partial output `[slots, rows, heads, head_dim]`
/// (one slot at prefill, where the grid is already full on the query axis), and
/// the output itself. That is `heads * (eff + 2 * head_dim)` f32 a row, and
/// none of it is a function of `tokens` — which is the whole point of the
/// change, because the arm this replaces held `heads * tokens` f32 a row and
/// therefore had to shrink `rows` as the sequence grew, until at 1M it could
/// afford eight.
///
/// The cap is 512 MiB, chosen against nothing better than "the same order as
/// the operands": at Inkling's global shape (`heads` 32, `eff` 1024, `head_dim`
/// 128) it comes out at 10,922 rows, and it is a ceiling rather than a target —
/// short sequences take one block.
pub fn query_block(heads: usize, eff: usize, head_dim: usize, tokens: usize) -> usize {
    const CAP: usize = 512 << 20;
    let per_row = heads * (eff + 2 * head_dim) * core::mem::size_of::<f32>();
    (CAP / per_row.max(1)).clamp(1, tokens.max(1))
}

/// The query rows a cube of `rows` holds, which is the grid's X granularity.
pub fn queries_per_cube(heads: usize, kv_heads: usize, rows: usize) -> usize {
    rows / (heads / kv_heads)
}

/// Fused global attention over one query block and any number of key runs.
///
/// `q` is `[nq, heads * head_dim]` f32 and `rel` is `[nq, heads, eff]` f32 —
/// the layouts the projections already produce, block-local, with `q0` giving
/// the absolute position of row 0. The key runs are `[rows, kv_heads *
/// head_dim]` in `kv` element type, token-major, exactly as both the projection
/// and the paged cache leave them.
///
/// Returns `[nq, heads * head_dim]` f32.
#[allow(clippy::too_many_arguments)]
pub fn flash_attention_launch<R: Runtime>(
    client: &ComputeClient<R>,
    q: &Handle,
    runs: &[KeyRun<'_>],
    rel: &Handle,
    kv: KvElem,
    nq: usize,
    q0: usize,
    heads: usize,
    kv_heads: usize,
    head_dim: usize,
    eff: usize,
    window: Option<usize>,
    scaling: f32,
    rows: usize,
) -> Handle {
    assert!(nq > 0, "an empty query block has no attention");
    assert!(!runs.is_empty(), "attention over no keys at all");
    assert!(
        applies(heads, kv_heads, head_dim, rows),
        "this shape is not fused here: {heads}/{kv_heads} heads, head_dim {head_dim}, rows {rows}"
    );
    let groups = heads / kv_heads;
    let bq = rows / groups;
    let q_tiles = nq.div_ceil(bq);
    let q_elems = nq * heads * head_dim;
    let rel_elems = (nq * heads * eff).max(1);
    assert!(
        q_elems <= u32::MAX as usize && rel_elems <= u32::MAX as usize,
        "{nq} queries x {heads} heads is past the 32-bit index every cubecl usize is on this \
         runtime"
    );

    // How far to split each run's key axis. Prefill fills the grid on the query
    // axis and lands on one split everywhere; a decode step has `q_tiles == 1`
    // and splits until the device is busy. Proportional to the run's span, so a
    // merged 1M-row page and a 128-row tail page do not get the same treatment.
    let spans: Vec<usize> = runs.iter().map(|r| r.hi.saturating_sub(r.lo)).collect();
    let total_span: usize = spans.iter().sum();
    let want = (TARGET_CUBES / (q_tiles * kv_heads)).max(1);
    let mut split_of = Vec::with_capacity(runs.len());
    let mut slots = 0usize;
    for span in &spans {
        let by_span = if total_span == 0 {
            1
        } else {
            (want * span).div_ceil(total_span)
        };
        let by_work = span.div_ceil(MIN_KEYS_PER_SPLIT).max(1);
        let s = by_span.min(by_work).max(1);
        split_of.push(s);
        slots += s;
    }

    let po_elems = slots * nq * heads * head_dim;
    let pml_elems = slots * nq * heads * 2;
    assert!(
        po_elems <= u32::MAX as usize,
        "{slots} partial slots x {nq} queries is past the 32-bit index"
    );
    let po = client.empty(po_elems * core::mem::size_of::<f32>());
    let pml = client.empty(pml_elems * core::mem::size_of::<f32>());

    let mut slot0 = 0usize;
    for (run, splits) in runs.iter().zip(split_of.iter().copied()) {
        let kv_elems = run.rows * kv_heads * head_dim;
        assert!(
            kv_elems <= u32::MAX as usize,
            "{} key rows x {kv_heads} x {head_dim} is past the 32-bit index",
            run.rows
        );
        assert!(
            run.lo <= run.hi && run.hi <= run.rows,
            "a key run's live range {}..{} is not inside its {} rows",
            run.lo,
            run.hi,
            run.rows
        );
        let count = CubeCount::Static(q_tiles as u32, kv_heads as u32, splits as u32);
        let dim = CubeDim::new_1d(UNITS);
        macro_rules! go {
            ($e:ty) => {
                unsafe {
                    flash_kernel::launch_unchecked::<$e, R>(
                        client,
                        count.clone(),
                        dim,
                        ArrayArg::from_raw_parts(q.clone(), q_elems),
                        ArrayArg::from_raw_parts(run.k.clone(), kv_elems),
                        ArrayArg::from_raw_parts(run.v.clone(), kv_elems),
                        ArrayArg::from_raw_parts(rel.clone(), rel_elems),
                        ArrayArg::from_raw_parts(po.clone(), po_elems),
                        ArrayArg::from_raw_parts(pml.clone(), pml_elems),
                        scaling,
                        nq as u32,
                        q0 as u32,
                        run.base as u32,
                        run.lo as u32,
                        run.hi as u32,
                        eff as u32,
                        window.unwrap_or(0) as u32,
                        slot0 as u32,
                        splits as u32,
                        slots as u32,
                        heads as u32,
                        kv_heads as u32,
                        head_dim as u32,
                        rows as u32,
                    )
                }
            };
        }
        match kv {
            KvElem::F32 => go!(f32),
            KvElem::Bf16 => go!(half::bf16),
        };
        slot0 += splits;
    }

    let out = client.empty(q_elems * core::mem::size_of::<f32>());
    unsafe {
        flash_combine_kernel::launch_unchecked::<R>(
            client,
            CubeCount::Static(nq as u32, heads as u32, 1),
            CubeDim::new_1d(head_dim as u32),
            ArrayArg::from_raw_parts(po, po_elems),
            ArrayArg::from_raw_parts(pml, pml_elems),
            ArrayArg::from_raw_parts(out.clone(), q_elems),
            slots as u32,
            nq as u32,
            heads as u32,
            head_dim as u32,
        )
    };
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn inklings_global_shape_is_fused() {
        // 32 heads over 8 KV heads, head_dim 128: groups 4, so a 32-row cube is
        // eight consecutive queries across all four heads of a KV head.
        assert!(applies(32, 8, 128, prefill_rows(4)));
        assert!(applies(32, 8, 128, decode_rows(4)));
        assert_eq!(prefill_rows(4), 32);
        assert_eq!(decode_rows(4), 4);
        assert_eq!(queries_per_cube(32, 8, 32), 8);
        assert_eq!(queries_per_cube(32, 8, 4), 1);
    }

    #[test]
    fn the_row_counts_always_split_into_planes_and_groups() {
        for groups in [1usize, 2, 4, 8, 16] {
            for rows in [decode_rows(groups), prefill_rows(groups)] {
                assert_eq!(rows % PLANES as usize, 0, "groups {groups}, rows {rows}");
                assert_eq!(rows % groups, 0, "groups {groups}, rows {rows}");
            }
        }
    }

    #[test]
    fn refuses_what_it_cannot_tile() {
        // Rows that do not fill four planes: three of them would idle and the
        // fourth would own rows that do not exist.
        assert!(!applies(32, 8, 128, 6));
        // Rows that do not split into head groups. At 32/8 every multiple of
        // the planes is also a multiple of `groups`, so this needs a geometry
        // where the two do not divide each other: 24 heads over 4 KV heads is
        // `groups = 6`, and a four-row cube cannot hold whole groups of six.
        assert!(!applies(24, 4, 128, 4));
        // Heads that do not divide into KV heads: `h / groups` would alias.
        assert!(!applies(32, 7, 128, 32));
        // A head_dim whose tile does not fit the 48 KiB a static `__shared__`
        // gets.
        assert!(!applies(32, 8, 2048, 32));
    }

    #[test]
    fn the_shared_tile_fits_the_static_ceiling() {
        for (rows, head_dim) in [(32u32, 128u32), (4, 128), (32, 64)] {
            let bytes = shared_floats(rows, head_dim) * 4;
            assert!(bytes <= 48 * 1024, "{rows}x{head_dim} wants {bytes} bytes");
        }
    }

    /// The block a prefill takes never depends on the sequence length, which is
    /// the whole difference from the arm it replaces.
    #[test]
    fn the_query_block_does_not_shrink_with_the_sequence() {
        let a = query_block(32, 1024, 128, 1 << 20);
        let b = query_block(32, 1024, 128, 1 << 14);
        assert_eq!(a, b.min(a), "the cap must not move with `tokens`");
        assert!(a > 1024, "a 512 MiB cap at Inkling's shape is {a} rows");
    }
}

#[cfg(test)]
mod device_tests {
    use super::*;
    use cubecl::cuda::CudaRuntime;

    /// Deterministic filler, the same one [`super::super::banded`]'s device
    /// tests use: a fixed pattern rather than a seeded RNG, so a failure is
    /// reproducible from the source alone.
    fn fill(n: usize, seed: f32) -> Vec<f32> {
        (0..n)
            .map(|i| (i as f32 * 0.7919 + seed).sin() * 0.5 + (i as f32 * 0.1237).cos() * 0.25)
            .collect()
    }

    struct Shape {
        nq: usize,
        q0: usize,
        keys: usize,
        heads: usize,
        kv_heads: usize,
        head_dim: usize,
        eff: usize,
        window: Option<usize>,
    }

    /// Run the fused kernel over `cuts` contiguous runs of the key axis.
    ///
    /// `cuts` is what makes this harness worth having: the answer must not
    /// depend on where the key axis is chopped, because the online softmax is
    /// exactly the claim that it does not.
    fn run(sh: &Shape, q: &[f32], k: &[f32], v: &[f32], rel: &[f32], cuts: &[usize]) -> Vec<f32> {
        let kv_row = sh.kv_heads * sh.head_dim;
        let groups = sh.heads / sh.kv_heads;
        let rows = if sh.nq == 1 {
            decode_rows(groups)
        } else {
            prefill_rows(groups)
        };
        let client = <CudaRuntime as Runtime>::client(&Default::default());
        let qh = client.create_from_slice(f32::as_bytes(q));
        let rh = client.create_from_slice(f32::as_bytes(rel));
        // Each cut is uploaded as its OWN buffer, which is what a paged cache
        // hands over and what a single prefill projection does not.
        let mut bounds = vec![0usize];
        bounds.extend_from_slice(cuts);
        bounds.push(sh.keys);
        let held: Vec<(cubecl::server::Handle, cubecl::server::Handle, usize, usize)> = bounds
            .windows(2)
            .map(|w| {
                let (lo, hi) = (w[0], w[1]);
                let kh = client.create_from_slice(f32::as_bytes(&k[lo * kv_row..hi * kv_row]));
                let vh = client.create_from_slice(f32::as_bytes(&v[lo * kv_row..hi * kv_row]));
                (kh, vh, hi - lo, lo)
            })
            .collect();
        let runs: Vec<KeyRun<'_>> = held
            .iter()
            .map(|(kh, vh, rows, base)| KeyRun {
                k: kh,
                v: vh,
                rows: *rows,
                base: *base,
                lo: 0,
                hi: *rows,
            })
            .collect();
        let oh = flash_attention_launch(
            &client,
            &qh,
            &runs,
            &rh,
            KvElem::F32,
            sh.nq,
            sh.q0,
            sh.heads,
            sh.kv_heads,
            sh.head_dim,
            sh.eff,
            sh.window,
            1.0 / sh.head_dim as f32,
            rows,
        );
        f32::from_bytes(&client.read_one(oh).expect("read the fused output")).to_vec()
    }

    fn inputs(sh: &Shape) -> (Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>) {
        (
            fill(sh.nq * sh.heads * sh.head_dim, 0.1),
            fill(sh.keys * sh.kv_heads * sh.head_dim, 0.3),
            fill(sh.keys * sh.kv_heads * sh.head_dim, 0.5),
            fill(sh.nq * sh.heads * sh.eff, 0.7),
        )
    }

    fn worst(a: &[f32], b: &[f32]) -> f32 {
        a.iter()
            .zip(b)
            .map(|(x, y)| (x - y).abs())
            .fold(0f32, f32::max)
    }

    /// THE defining property of an online softmax, and the one a two-pass
    /// implementation cannot fail: where the key axis is cut must not change
    /// the answer.
    ///
    /// Every rescale bug lives here. A missing `exp(m_old - m_new)` on the
    /// accumulator, a running sum that is not rescaled with it, a combine that
    /// weights by `l` instead of by `exp(m - M) * l` — none of them shows on a
    /// single tile, all of them show the moment a later tile pushes the max up.
    /// The cuts below are deliberately not multiples of the 32-key tile, so the
    /// split boundaries land mid-tile as a page boundary would.
    #[test]
    fn where_the_key_axis_is_cut_does_not_move_the_answer() {
        for (nq, keys) in [(1usize, 700usize), (16, 200), (37, 91), (1, 33)] {
            let sh = Shape {
                nq,
                q0: keys - nq,
                keys,
                heads: 4,
                kv_heads: 2,
                head_dim: 8,
                eff: 5,
                window: None,
            };
            let (q, k, v, rel) = inputs(&sh);
            let whole = run(&sh, &q, &k, &v, &rel, &[]);
            for cuts in [
                vec![keys / 2],
                vec![1],
                vec![keys - 1],
                vec![7, keys / 3, keys / 2, keys - 5],
            ] {
                let cut = run(&sh, &q, &k, &v, &rel, &cuts);
                let w = worst(&whole, &cut);
                assert!(
                    w < 2e-6,
                    "{nq} queries over {keys} keys, cuts {cuts:?}: {w:e}"
                );
            }
        }
    }

    /// Causality, checked against the kernel itself rather than against a
    /// second implementation of it.
    ///
    /// A key AFTER the query cannot reach it, so perturbing that key must leave
    /// the query's output exactly where it was; a key at or before it must move
    /// it. Both halves are necessary: the first alone passes for a kernel that
    /// reads nothing, the second alone for one with no mask at all.
    #[test]
    fn a_key_after_the_query_cannot_reach_it() {
        let sh = Shape {
            nq: 24,
            q0: 0,
            keys: 40,
            heads: 4,
            kv_heads: 2,
            head_dim: 8,
            eff: 6,
            window: None,
        };
        let (q, k, v, rel) = inputs(&sh);
        let base = run(&sh, &q, &k, &v, &rel, &[]);
        let kv_row = sh.kv_heads * sh.head_dim;
        let last = sh.nq - 1;
        let row = |o: &[f32], i: usize| {
            o[i * sh.heads * sh.head_dim..(i + 1) * sh.heads * sh.head_dim].to_vec()
        };
        for (j, must_move) in [(last + 1, false), (sh.keys - 1, false), (last, true)] {
            let (mut kk, mut vv) = (k.clone(), v.clone());
            for c in 0..kv_row {
                kk[j * kv_row + c] += 0.75;
                vv[j * kv_row + c] -= 0.75;
            }
            let got = run(&sh, &q, &kk, &vv, &rel, &[]);
            let moved = row(&got, last)
                .into_iter()
                .zip(row(&base, last))
                .map(|(a, b)| (a - b).abs())
                .fold(0f32, f32::max);
            if must_move {
                assert!(
                    moved > 1e-4,
                    "key {j} is visible to query {last} and did not move"
                );
            } else {
                assert_eq!(
                    moved, 0.0,
                    "key {j} is after query {last} but perturbing it moved the answer by {moved:e}"
                );
            }
        }
    }

    /// GQA indexing, again against the kernel itself: perturbing KV head `c`
    /// may move only the query heads in `c`'s group.
    ///
    /// This is the failure a reference implementation catches by accident and
    /// this catches on purpose — a cube whose row-to-head map is off by one is
    /// a wrong ANSWER everywhere and a wrong SHAPE nowhere. It is the one thing
    /// this kernel does that the arm it replaces did not: the rows of a cube
    /// span `groups` heads, so the map from a plane's row index to a head is
    /// arithmetic that can be wrong.
    #[test]
    fn a_kv_head_reaches_only_its_own_group() {
        let sh = Shape {
            nq: 20,
            q0: 0,
            keys: 20,
            heads: 8,
            kv_heads: 2,
            head_dim: 8,
            eff: 6,
            window: None,
        };
        let groups = sh.heads / sh.kv_heads;
        let (q, k, v, rel) = inputs(&sh);
        let base = run(&sh, &q, &k, &v, &rel, &[]);
        let kv_row = sh.kv_heads * sh.head_dim;
        // ONE key, not every key of the head: adding the same constant to all
        // of a head's keys shifts every score equally and the softmax cancels
        // it exactly.
        let at = sh.keys / 2;
        for c in 0..sh.kv_heads {
            let (mut kk, mut vv) = (k.clone(), v.clone());
            for d in 0..sh.head_dim {
                kk[at * kv_row + c * sh.head_dim + d] += 0.5;
                vv[at * kv_row + c * sh.head_dim + d] -= 0.5;
            }
            let got = run(&sh, &q, &kk, &vv, &rel, &[]);
            for h in 0..sh.heads {
                let mine = h / groups == c;
                let moved = (0..sh.nq)
                    .flat_map(|i| {
                        (0..sh.head_dim)
                            .map(move |d| i * sh.heads * sh.head_dim + h * sh.head_dim + d)
                    })
                    .map(|o| (got[o] - base[o]).abs())
                    .fold(0f32, f32::max);
                if mine {
                    assert!(moved > 1e-5, "head {h} shares KV head {c} and did not move");
                } else {
                    assert_eq!(
                        moved, 0.0,
                        "head {h} does not share KV head {c} but moved by {moved:e}"
                    );
                }
            }
        }
    }

    /// The window, when one is asked for: a key `window` or more positions back
    /// is outside, and one inside must move the answer.
    ///
    /// The global layers pass `None` here, but the kernel takes a window
    /// because the DECODE lane runs every layer through it, local ones
    /// included, and a window that is applied but not tested is a window that
    /// is wrong on thirty-five layers of forty-two.
    #[test]
    fn the_window_is_exactly_what_the_query_can_reach() {
        let win = 9usize;
        let sh = Shape {
            nq: 30,
            q0: 0,
            keys: 30,
            heads: 4,
            kv_heads: 2,
            head_dim: 8,
            eff: 5,
            window: Some(win),
        };
        let (q, k, v, rel) = inputs(&sh);
        let base = run(&sh, &q, &k, &v, &rel, &[]);
        let kv_row = sh.kv_heads * sh.head_dim;
        let last = sh.nq - 1;
        for (j, must_move) in [(last - win, false), (last - win + 1, true)] {
            let (mut kk, mut vv) = (k.clone(), v.clone());
            for c in 0..kv_row {
                kk[j * kv_row + c] += 0.75;
                vv[j * kv_row + c] -= 0.75;
            }
            let got = run(&sh, &q, &kk, &vv, &rel, &[]);
            let moved = (0..sh.heads * sh.head_dim)
                .map(|c| {
                    let o = last * sh.heads * sh.head_dim + c;
                    (got[o] - base[o]).abs()
                })
                .fold(0f32, f32::max);
            if must_move {
                assert!(
                    moved > 1e-4,
                    "key {j} is inside query {last}'s window and did not move"
                );
            } else {
                assert_eq!(
                    moved, 0.0,
                    "key {j} is {win} back from query {last} but moved the answer by {moved:e}"
                );
            }
        }
    }

    /// A decode-shaped launch at a length that forces the key axis to split,
    /// and a prefill-shaped one at the model's own head geometry: both run,
    /// both produce finite output that is not one repeated number.
    ///
    /// No tolerance here on purpose — a kernel that has gone wrong at this size
    /// produces NaNs, zeros or a constant, and one that is subtly off produces
    /// a number only the paired harness can judge.
    #[test]
    fn inklings_shape_at_a_real_length_runs() {
        for (nq, keys) in [(1usize, 20_000usize), (256, 4_000)] {
            let sh = Shape {
                nq,
                q0: keys - nq,
                keys,
                heads: 32,
                kv_heads: 8,
                head_dim: 128,
                eff: 512,
                window: None,
            };
            let (q, k, v, rel) = inputs(&sh);
            let out = run(&sh, &q, &k, &v, &rel, &[]);
            assert_eq!(out.len(), nq * sh.heads * sh.head_dim);
            assert!(
                out.iter().all(|x| x.is_finite()),
                "{nq} queries over {keys} keys produced a non-finite value"
            );
            let lo = out.iter().copied().fold(f32::INFINITY, f32::min);
            let hi = out.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            assert!(
                hi - lo > 1e-3,
                "every output is {lo}: the kernel computed nothing"
            );
        }
    }
}
