//! `ptx_fp4_probe` — the hand-written PTX arm of the routed NVFP4 GEMM against
//! the cubecl one, in ONE process, on the same handles.
//!
//! [`mary::models::inkling::ptxgemm`] emits
//! [`mary::models::inkling::fp4gemm::fp4_linear_swz`] as PTX: same operands,
//! same `(m_tile, n_tile)` decomposition, same lane roles, same MMA in the same
//! order, same output layout, with the K loop unrolled into the text and every
//! address a base register plus an immediate. Two things have to be shown about
//! it, and they are different questions:
//!
//! 1. **Does it compute the same thing?** Elementwise, bit for bit. The two
//!    arms issue the identical instruction with the identical operands and
//!    accumulate in the identical order, so the answer must be *exactly equal*
//!    — not "close". Anything else is a bug in the address derivation, and the
//!    interesting output is then WHERE it differs, so this prints the max abs
//!    difference and the count of differing elements rather than a verdict.
//! 2. **What does it cost?** Which is a separate measurement and must not be
//!    read off the same launch, because the identity check reads the output
//!    back and a readback is host time.
//!
//! ## The harness, and what it copies from `w4a16_swz_grid`
//!
//! That probe's header records the three ways a two-arm GEMM comparison lies,
//! all of which apply here — the routed shape's weight table is 9.4 MiB against
//! a 24 MiB L2, so the naive form measures cache residency and arm order:
//!
//! * **Rotating tables.** Enough distinct B planes that the gap between two
//!   reads of one clears L2 by a wide margin. Both arms rotate identically over
//!   the SAME set, so neither is the warm one.
//! * **Pipelined launches.** `REPS` launches enqueued back to back and ONE sync
//!   at the end, divided. A per-launch sync at this shape would be timing the
//!   host round trip.
//! * **Interleaved arms with the order reversed on odd rounds**, so a clock
//!   ramp or a neighbour lands on both arms equally, and p50 over the warm
//!   rounds.
//!
//! Every rotating table holds the SAME random bytes, so the identity check is
//! valid against any of them and the timing is unaffected by which one a rep
//! draws.
//!
//! ## The shape
//!
//! One real routed-expert GEMM. `w13` is `[2 * intermediate_size,
//! hidden_size] = [4096, 4096]` and `w2` is `[hidden_size, intermediate_size] =
//! [4096, 2048]` on the 42-layer checkpoint (`InklingTextConfig`:
//! `hidden_size` 4096, `intermediate_size` 2048), so the default here is
//! `k = 4096`, `n = 4096`, `m_pad = 16` — the decode case, where one expert
//! gets a handful of tokens and [`mary::models::inkling::moegroup::RowPlan`]
//! gives it a single 16-row tile. `INK_K`, `INK_N`, `INK_M` override;
//! `INK_SWZ_SC=0` runs with the scale plane left row-major.
//!
//! The codes and scales are RANDOM, not checkpoint rows: this probe asks
//! whether two arms agree with each other and what they cost, and for both
//! questions random NVFP4 is the same exercise as real NVFP4. Whether the
//! instruction itself is right against a decoded reference is
//! `nvfp4_mma_probe`'s job, on real expert rows, and it is already answered.
//!
//! Build: `--features inkling-cuda`
//! Run:   `ptx_fp4_probe`  (`INK_PTX_DUMP=<path>` also writes the generated PTX)

use std::time::Instant;

use cubecl::future;
use cubecl::prelude::*;
use cubecl::server::Handle;

use mary::models::inkling::fp4gemm::{GROUP, fp4_linear_swz_launch};
use mary::models::inkling::ptxgemm::{
    fp4_linear_swz_ptx, fp4_linear_swz_ptx_launch, fp4_linear_swz_ptx_name,
};

type Rt = cubecl::cuda::CudaRuntime;

/// Launches enqueued back to back before one sync.
const REPS: usize = 20;
/// Rounds kept for the p50, after [`WARM`] discarded.
const ROUNDS: usize = 9;
/// Rounds thrown away first.
const WARM: usize = 2;

/// The `[n, k]` NVFP4 weight plane's bytes: `k/2` codes + `k/16` E4M3 scales
/// per row. The GB/s below is over exactly this and nothing else.
fn table_bytes(n: usize, k: usize) -> usize {
    n * (k / 2) + n * (k / 16)
}

fn env_usize(name: &str, default: usize) -> usize {
    std::env::var(name)
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(default)
}

/// Deterministic bytes: the same table on every machine and every run, so a
/// disagreement is reproducible.
struct Lcg(u64);

impl Lcg {
    fn new(seed: u64) -> Self {
        Lcg(seed
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407))
    }
    fn next(&mut self) -> u64 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        self.0
    }
    /// Packed E2M1 code bytes — every nibble is a legal code, so any byte is.
    fn codes(&mut self, n: usize) -> Vec<u8> {
        (0..n).map(|_| (self.next() >> 33) as u8).collect()
    }
    /// E4M3 block-scale bytes in a deliberately narrow band: exponent field 5
    /// to 8 (bias 7), sign clear, so every scale is finite and in `[0.25, 4)`
    /// and no product can leave f32's range over `k` terms. A uniform byte
    /// would include `0x7F`/`0xFF`, which are NaN in E4M3FN and would make
    /// "the two arms agree" vacuous wherever a NaN swallowed the difference.
    fn scales(&mut self, n: usize) -> Vec<u8> {
        (0..n)
            .map(|_| {
                let r = self.next() >> 32;
                let exp = 5 + (r % 4) as u8;
                let man = ((r >> 8) % 8) as u8;
                (exp << 3) | man
            })
            .collect()
    }
}

fn main() {
    let client = Rt::client(&Default::default());

    let k = env_usize("INK_K", 4096);
    let n = env_usize("INK_N", 4096);
    let m_pad = env_usize("INK_M", 16);
    let swz_sc = env_usize("INK_SWZ_SC", 1) == 1;
    // Not 1.0: the epilogue's multiply by the expert's second-level F32
    // constant is part of both kernels, and at 1.0 a wrong one is invisible.
    let scale = 0.734_251_f32;

    let ptx = fp4_linear_swz_ptx(k, n, swz_sc);
    let name = fp4_linear_swz_ptx_name(k, n, swz_sc);
    println!("=== NVFP4 routed GEMM: hand-written PTX vs cubecl, one process ===");
    println!(
        "shape [m_pad {m_pad}, k {k}] x [n {n}, k {k}]^T, scale plane {}, scale {scale}",
        if swz_sc { "PERMUTED" } else { "row-major" }
    );
    println!(
        "PTX entry {name}: {} bytes, {} lines, {} MMAs, {} global loads, {} cubes of 32 threads",
        ptx.len(),
        ptx.lines().count(),
        ptx.matches("mma.sync").count(),
        ptx.matches("ld.global").count(),
        (m_pad / 16) * (n / 8),
    );
    if let Ok(p) = std::env::var("INK_PTX_DUMP") {
        std::fs::write(&p, &ptx).expect("writing the PTX dump");
        println!("PTX written to {p} — check it with `ptxas -arch=sm_121a -v -o /dev/null {p}`");
    }

    // --- operands ---------------------------------------------------------
    let mut rng = Lcg::new(0x5eed_1234);
    let a_codes = rng.codes(m_pad * (k / 2));
    let a_scales = rng.scales(m_pad * (k / GROUP));
    let b_codes = rng.codes(n * (k / 2));
    let b_scales = rng.scales(n * (k / GROUP));

    let a = client.create_from_slice(&a_codes);
    let a_sc = client.create_from_slice(&a_scales);

    // The GAP between two reads of one table must clear the 24 MiB L2 by a wide
    // margin, or the small shapes are quietly served from cache -- which is the
    // bias this harness exists to remove. Same rule as `w4a16_swz_grid`.
    let bytes = table_bytes(n, k);
    let rot = (1 + (256usize << 20).div_ceil(bytes.max(1))).clamp(2, 130);
    let rot = rot.min(((3usize << 30) / bytes.max(1)).max(2));
    let tables: Vec<(Handle, Handle)> = (0..rot)
        .map(|_| {
            (
                client.create_from_slice(&b_codes),
                client.create_from_slice(&b_scales),
            )
        })
        .collect();
    println!(
        "{rot} rotating B tables of {:.2} MiB ({:.2} MiB in flight), A is {:.2} KiB\n",
        bytes as f64 / (1u64 << 20) as f64,
        (rot * bytes) as f64 / (1u64 << 20) as f64,
        (a_codes.len() + a_scales.len()) as f64 / 1024.0,
    );

    // --- 1. elementwise identity ------------------------------------------
    // Untimed, on one table, read back in full.
    let (b0, bs0) = &tables[0];
    let o_cc = fp4_linear_swz_launch::<Rt>(&client, &a, &a_sc, b0, bs0, m_pad, k, n, scale, swz_sc);
    let o_px = fp4_linear_swz_ptx_launch(&client, &a, &a_sc, b0, bs0, m_pad, k, n, scale, swz_sc);
    let want = f32::from_bytes(&client.read_one(o_cc.clone()).expect("read cubecl")).to_vec();
    let got = f32::from_bytes(&client.read_one(o_px.clone()).expect("read ptx")).to_vec();
    assert_eq!(
        want.len(),
        got.len(),
        "the two arms wrote different lengths"
    );

    let mut diff = 0usize;
    let mut max_abs = 0.0f64;
    let mut first: Option<(usize, f32, f32)> = None;
    let mut nonfinite = 0usize;
    for (i, (&w, &g)) in want.iter().zip(&got).enumerate() {
        if !g.is_finite() || !w.is_finite() {
            nonfinite += 1;
        }
        if w.to_bits() != g.to_bits() {
            diff += 1;
            let d = (w as f64 - g as f64).abs();
            if d > max_abs {
                max_abs = d;
            }
            if first.is_none() {
                first = Some((i, w, g));
            }
        }
    }
    println!(
        "identity: {} of {} elements differ, max abs diff {max_abs:e}{}",
        diff,
        want.len(),
        if nonfinite > 0 {
            format!(" ({nonfinite} non-finite on one side or the other)")
        } else {
            String::new()
        }
    );
    match first {
        None => println!(
            "  -> BIT-IDENTICAL. Same instruction, same operand order, same accumulate order."
        ),
        Some((i, w, g)) => println!(
            "  -> DIFFERS. First at element {i} (row {}, col {}): cubecl {w:e} ({:#010x}) vs ptx \
             {g:e} ({:#010x})",
            i / n,
            i % n,
            w.to_bits(),
            g.to_bits()
        ),
    }
    drop(o_cc);
    drop(o_px);

    // --- 2. cost ----------------------------------------------------------
    let mut t_cc: Vec<f64> = Vec::new();
    let mut t_px: Vec<f64> = Vec::new();

    let bench_cc = |client: &ComputeClient<Rt>, tables: &[(Handle, Handle)]| -> f64 {
        let t0 = Instant::now();
        for j in 0..REPS {
            let (b, bs) = &tables[j % tables.len()];
            let o =
                fp4_linear_swz_launch::<Rt>(client, &a, &a_sc, b, bs, m_pad, k, n, scale, swz_sc);
            drop(o);
        }
        let _ = future::block_on(client.sync());
        t0.elapsed().as_secs_f64() / REPS as f64
    };
    let bench_px = |client: &ComputeClient<Rt>, tables: &[(Handle, Handle)]| -> f64 {
        let t0 = Instant::now();
        for j in 0..REPS {
            let (b, bs) = &tables[j % tables.len()];
            let o = fp4_linear_swz_ptx_launch(client, &a, &a_sc, b, bs, m_pad, k, n, scale, swz_sc);
            drop(o);
        }
        let _ = future::block_on(client.sync());
        t0.elapsed().as_secs_f64() / REPS as f64
    };

    for r in 0..WARM + ROUNDS {
        // Reversed on odd rounds: whichever arm runs first pays for whatever
        // the other one left in L2, and over an even split neither does.
        let (cc, px) = if r % 2 == 0 {
            let cc = bench_cc(&client, &tables);
            let px = bench_px(&client, &tables);
            (cc, px)
        } else {
            let px = bench_px(&client, &tables);
            let cc = bench_cc(&client, &tables);
            (cc, px)
        };
        if r >= WARM {
            t_cc.push(cc);
            t_px.push(px);
        }
    }

    let p50 = |v: &mut Vec<f64>| -> f64 {
        v.sort_by(|a, b| a.partial_cmp(b).expect("no NaN in a duration"));
        v[v.len() / 2]
    };
    let cc = p50(&mut t_cc);
    let px = p50(&mut t_px);

    let framing = format!(
        "per LAUNCH of one [{m_pad}, {k}] x [{n}, {k}]^T product, GB/s over the B plane only \
         ({:.2} MiB of codes + scales), p50 of {ROUNDS} warm rounds of {REPS} pipelined launches \
         over {rot} rotating tables, arms interleaved and reversed on odd rounds, ONE GB10 \
         (sm_121a); not a step figure and not a two-node figure",
        bytes as f64 / (1u64 << 20) as f64
    );
    println!(
        "\ncubecl  fp4_linear_swz        {:>9.4} ms  {:>7.1} GB/s   [{framing}]",
        cc * 1e3,
        bytes as f64 / cc / 1e9
    );
    println!(
        "PTX     {name:<22} {:>9.4} ms  {:>7.1} GB/s   [{framing}]",
        px * 1e3,
        bytes as f64 / px / 1e9
    );
    println!(
        "ratio   cubecl / PTX          {:>9.3}x  ({} at this shape)",
        cc / px,
        if cc / px > 1.0 {
            "the hand-written arm is FASTER"
        } else {
            "the hand-written arm is SLOWER"
        }
    );
}
