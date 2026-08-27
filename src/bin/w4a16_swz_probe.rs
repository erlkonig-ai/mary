//! Does the `m16n8k16` B permutation buy anything, and on which plane?
//!
//! `mma16_lane_dump` settled the map and, with it, what the permutation can and
//! cannot be for. It CANNOT be a bandwidth fix: over a whole k loop the
//! row-major form already reaches 100% sector and 100% line utilisation,
//! because the k loop walks each of the eight weight rows forward and a
//! half-used sector is finished by the next few k tiles out of L1. What it
//! removes is REQUESTS — 4096 sector requests per warp k loop against 512
//! distinct sectors on the codes plane, and the same 8x on the scales.
//!
//! Whether removing them is worth anything is a question about the machine, not
//! about the map, so it is measured here rather than argued.
//!
//! ## The arms
//!
//! * `row-major` — `w4a16_linear`, the lane the head runs today.
//! * `swz codes+scales` — `w4a16_linear_swz` against both planes permuted.
//! * `swz codes only` — the same kernel with the scale plane left row-major, so
//!   the two planes' contributions separate instead of being one number.
//! * `stream ceiling` — the same bytes read fully coalesced with no arithmetic.
//!   Not a rival implementation: the bus figure the other three are read
//!   against.
//!
//! Arms are INTERLEAVED (rep 1 of every arm, then rep 2 of every arm) so drift
//! over the run lands on all of them equally; the first two reps are discarded;
//! every rep is printed, and the spread with it.
//!
//! # DO NOT RUN THIS AT A SINK SHAPE. It measures the L2 there, not the kernel.
//!
//! The four arms SHARE BUFFERS: `stream ceiling` reads the same `b_row` /
//! `bs_row` that `row-major` reads, immediately before it, and the two plane
//! sets together are 36 MiB against a 24 MiB L2. So at a shape whose table fits
//! in L2 the row-major arm runs L2-WARM every rep and the two swizzled arms run
//! L2-COLD every rep, and the gap between them is an artifact of the arm ORDER.
//! At `[8192, 4096]` it duly reports 0.176 against 0.306 ms -- a 74% "loss"
//! that reverses under any harness that treats the arms alike. `main` refuses
//! such a shape rather than printing it; `INK_SWZ_FORCE=1` overrides, and then
//! the number is a cache measurement and must be labelled one.
//!
//! At the head's 463 MiB nothing is resident, the confound cannot occur, and the
//! table this probe produced there stands. For anything smaller use
//! `w4a16_swz_grid`, which rotates buffers and pipelines launches for exactly
//! this reason.
//!
//! `INK_SWZ_N` / `INK_SWZ_K` set the shape (default the unembedding's
//! `[201024, 4096]`), `INK_SWZ_M` the padded row count (default 16, one m-tile,
//! which is decode), `INK_SWZ_REPS` the rep count (default 8).
//!
//! `INK_SWZ_MLIVE` sets how many of those `INK_SWZ_M` rows are LIVE, and with
//! it the three GEMM arms take `w4a16gemm::live_row_mask` -- the A operand
//! declining to load a fragment row that is M padding. Unset is "all live",
//! which is what this probe has always measured. `INK_SWZ_MLIVE=1` against
//! `INK_SWZ_M=16` is the real decode shape: one token, fifteen rows of padding.
//! The mask is a REQUEST-count change on A and not a bandwidth one -- A is
//! L2-resident at this `m_pad`, and the same "the bytes were never the problem"
//! that `mma16_lane_dump` established for B applies -- so read it in the ncu
//! sector counts first and in this probe's ms second.
//!
//! The last TWO sections are GATES, not observations, and the probe exits 2 if
//! either fails.
//!
//! * The A-side live-row mask: masked and unmasked must agree to the last bit,
//!   at four `(m_pad, live)` pairs including a multi-tile shape whose LAST tile
//!   is partly padding.
//! * The COALESCED LOAD + WARP SHUFFLE weight read
//!   (`w4a16gemm::swz_shuffle`, `INK_W4A16_SWZ_SHUFFLE`, default off): flag on
//!   and flag off must agree to the last bit, at four shapes that between them
//!   cover decode, a partly-padded last tile, a ROW-MAJOR scale plane the
//!   redistribution must decline, and a `k` whose k-tile count the load depth
//!   does not divide, where it must fall back to the fragment-shaped load
//!   rather than to half a warp step. Both arms are launched from ONE process
//!   by `w4a16_linear_swz_launch_redist`, so neither the environment nor two
//!   builds can be the difference between them. It moves the same words into
//!   the same slots, so the only admissible deviation is zero: anything else is
//!   a wrong source lane, which would look like plausible numbers.

use std::time::Instant;

use cubecl::future;
use cubecl::prelude::*;
use mary::models::inkling::w4a16gemm::{
    swizzle_w4a16_codes_into, swizzle_w4a16_scales_into, w4a16_linear_launch,
    w4a16_linear_swz_launch, w4a16_linear_swz_launch_redist,
};

type Rt = cubecl::cuda::CudaRuntime;

const PER: usize = 8;
const BLOCK: u32 = 256;

/// The coalesced control: the same two planes, 128-bit loads, no arithmetic.
#[cube(launch)]
pub fn stream_planes<NW: Size>(
    w: &Tensor<Vector<u32, NW>>,
    sc: &Tensor<Vector<u32, NW>>,
    out: &mut Tensor<u32>,
    #[comptime] threads: usize,
    #[comptime] per: usize,
) {
    let t = ABSOLUTE_POS as usize;
    let mut acc = u32::new(0i64);
    #[unroll]
    for i in 0..per {
        let v = w[t + i * threads];
        acc += v[0];
    }
    let s = sc[t % sc.len()];
    acc += s[0];
    if acc == u32::new(0x5AFE_5AFEi64) {
        out[t % out.len()] = acc;
    }
}

fn env(name: &str, default: usize) -> usize {
    std::env::var(name)
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(default)
}

/// p50 of a slice, and the spread as (min, max).
fn stats(v: &[f64]) -> (f64, f64, f64) {
    let mut s = v.to_vec();
    s.sort_by(|a, b| a.partial_cmp(b).unwrap());
    (s[s.len() / 2], s[0], s[s.len() - 1])
}

fn main() {
    let device = Default::default();
    let client = Rt::client(&device);

    let n = env("INK_SWZ_N", 201024);
    let k = env("INK_SWZ_K", 4096);
    let m_pad = env("INK_SWZ_M", 16);
    // How many of `m_pad`'s rows are LIVE, for the timing arms. Unset means
    // `None` -- load every row, which is what this probe has always done and
    // what the shipped default still does. `INK_SWZ_MLIVE=1` is decode: one real
    // row against a 16-row tile, and the arm the A-side mask exists for.
    let mlive: Option<usize> = std::env::var("INK_SWZ_MLIVE")
        .ok()
        .and_then(|v| v.parse::<usize>().ok());
    if let Some(v) = mlive {
        assert!(v <= m_pad, "INK_SWZ_MLIVE {v} exceeds INK_SWZ_M {m_pad}");
    }
    let reps = env("INK_SWZ_REPS", 8);
    let codes = n * k / 2;
    let scales = n * (k / 16);
    let bytes = codes + scales;

    // See the header. Both plane sets must exceed L2 by a margin or the arm
    // ORDER decides the answer. GB10's L2 is 24 MiB and this holds two of them,
    // so the table has to clear it on its own before the interleave is honest.
    const L2_BYTES: usize = 24 << 20;
    if 2 * bytes < 3 * L2_BYTES && std::env::var("INK_SWZ_FORCE").as_deref() != Ok("1") {
        eprintln!(
            "w4a16_swz_probe: REFUSING [{n}, {k}] -- its two plane sets are {:.1} MiB against a \n\
             {} MiB L2, so `stream ceiling` leaves the row-major arm warm and the swizzled arms \n\
             cold, and the interleave measures the arm order. Use `w4a16_swz_grid` at this shape \n\
             (rotating buffers, pipelined launches). `INK_SWZ_FORCE=1` overrides, and then the \n\
             result is a CACHE measurement and must be labelled one.",
            2.0 * bytes as f64 / (1u64 << 20) as f64,
            L2_BYTES >> 20,
        );
        std::process::exit(2);
    }

    println!(
        "shape [{n}, {k}] m_pad {m_pad}: codes {:.3} GiB, scales {:.3} GiB, total {:.3} GiB",
        codes as f64 / (1u64 << 30) as f64,
        scales as f64 / (1u64 << 30) as f64,
        bytes as f64 / (1u64 << 30) as f64,
    );

    // Real bytes, really permuted -- not the same buffer read two ways. The
    // footprints are identical either way, so this changes no timing; it
    // removes the objection rather than the effect.
    let t_host = Instant::now();
    let src_c: Vec<u8> = (0..codes)
        .map(|i| (i.wrapping_mul(37) % 251) as u8)
        .collect();
    let src_s: Vec<u8> = (0..scales)
        .map(|i| [0x38u8, 0x40, 0x30, 0x3Cu8][i % 4])
        .collect();
    let mut swz_c = vec![0u8; codes];
    let mut swz_s = vec![0u8; scales];
    swizzle_w4a16_codes_into(&src_c, &mut swz_c, n, k);
    swizzle_w4a16_scales_into(&src_s, &mut swz_s, n, k);
    println!(
        "host permutation of {:.3} GiB: {:.2} s (STARTUP, once, and in the real load path it \
         rides inside a memcpy that already happens)",
        bytes as f64 / (1u64 << 30) as f64,
        t_host.elapsed().as_secs_f64()
    );

    let a = client.empty(m_pad * k * 2);
    let b_row = client.create_from_slice(&src_c);
    let bs_row = client.create_from_slice(&src_s);
    let b_swz = client.create_from_slice(&swz_c);
    let bs_swz = client.create_from_slice(&swz_s);
    drop(src_c);
    drop(swz_c);

    let vectors = codes / 16;
    let threads = vectors / PER;
    let blocks = (threads as u32).div_ceil(BLOCK);
    let dst = client.empty(4096);

    let names = [
        "row-major        ",
        "swz codes+scales ",
        "swz codes only   ",
        "stream ceiling   ",
    ];
    let mut per_rep: Vec<Vec<f64>> = vec![Vec::new(); names.len()];

    for rep in 0..reps {
        for arm in 0..names.len() {
            let t0 = Instant::now();
            match arm {
                0 => {
                    let o = w4a16_linear_launch::<Rt>(
                        &client, &a, &b_row, &bs_row, m_pad, k, n, 1.0, mlive,
                    );
                    let _ = future::block_on(client.sync());
                    drop(o);
                }
                1 => {
                    let o = w4a16_linear_swz_launch::<Rt>(
                        &client, &a, &b_swz, &bs_swz, m_pad, k, n, true, 1.0, mlive,
                    );
                    let _ = future::block_on(client.sync());
                    drop(o);
                }
                2 => {
                    let o = w4a16_linear_swz_launch::<Rt>(
                        &client, &a, &b_swz, &bs_row, m_pad, k, n, false, 1.0, mlive,
                    );
                    let _ = future::block_on(client.sync());
                    drop(o);
                }
                _ => {
                    unsafe {
                        stream_planes::launch::<Rt>(
                            &client,
                            CubeCount::Static(blocks, 1, 1),
                            CubeDim::new_1d(BLOCK),
                            4,
                            TensorArg::from_raw_parts(b_row.clone(), [1].into(), [vectors].into()),
                            TensorArg::from_raw_parts(
                                bs_row.clone(),
                                [1].into(),
                                [scales / 16].into(),
                            ),
                            TensorArg::from_raw_parts(dst.clone(), [1].into(), [1024].into()),
                            threads,
                            PER,
                        )
                    };
                    let _ = future::block_on(client.sync());
                }
            }
            per_rep[arm].push(t0.elapsed().as_secs_f64());
        }
        println!(
            "  rep {rep:>2}{}{}",
            if rep < 2 { "  (discarded, cold)" } else { "" },
            per_rep
                .iter()
                .zip(names)
                .map(|(v, nm)| format!("   {}{:.3} ms", nm.trim(), v[rep] * 1e3))
                .collect::<String>()
        );
    }

    println!("\n  arm                    p50 ms      min      max      GB/s(p50)   vs row-major");
    let warm: Vec<Vec<f64>> = per_rep.iter().map(|v| v[2..].to_vec()).collect();
    let (base, _, _) = stats(&warm[0]);
    for (i, nm) in names.iter().enumerate() {
        let (p50, lo, hi) = stats(&warm[i]);
        println!(
            "  {nm}   {:8.3} {:8.3} {:8.3}   {:8.1}      {:+6.1}%",
            p50 * 1e3,
            lo * 1e3,
            hi * 1e3,
            bytes as f64 / p50 / 1e9,
            100.0 * (base - p50) / base,
        );
    }
    println!(
        "\n  framing: per LAUNCH of one [{m_pad}, {k}] x [{n}, {k}]^T product with {} of those \
         rows live, {} warm reps of {reps} \
         (first 2 discarded), arms interleaved, one GB10 box, one process, same buffers.\n  \
         GB/s counts the weight planes only ({:.3} GiB), which is the traffic the permutation is \
         about; it is not a step figure and not a two-node figure.",
        match mlive {
            Some(v) => format!("{v} (A-side live-row mask ON)"),
            None => format!("{m_pad} (mask off)"),
        },
        reps - 2,
        bytes as f64 / (1u64 << 30) as f64,
    );

    // Numerics, as an OBSERVATION. Not a gate: the permutation moves bytes and
    // changes nothing about what they mean, so the two arms should agree to the
    // last bit -- but a deviation here would be a wrong index, not a rounding
    // difference, and the number says which.
    {
        let (cm, ck, cn) = (16usize, 256usize, 64usize);
        let cc = cn * ck / 2;
        let cs = cn * (ck / 16);
        let wb: Vec<u8> = (0..cc).map(|i| (i.wrapping_mul(97) % 253) as u8).collect();
        let sb: Vec<u8> = (0..cs).map(|i| [0x38u8, 0x40, 0x30][i % 3]).collect();
        let ab: Vec<u8> = (0..cm * ck)
            .flat_map(|i| half::bf16::from_f32((i % 13) as f32 * 0.25 - 1.5).to_le_bytes())
            .collect();
        let mut pc = vec![0u8; cc];
        let mut ps = vec![0u8; cs];
        swizzle_w4a16_codes_into(&wb, &mut pc, cn, ck);
        swizzle_w4a16_scales_into(&sb, &mut ps, cn, ck);
        let ha = client.create_from_slice(&ab);
        let hb = client.create_from_slice(&wb);
        let hs = client.create_from_slice(&sb);
        let hbp = client.create_from_slice(&pc);
        let hsp = client.create_from_slice(&ps);
        let o1 = w4a16_linear_launch::<Rt>(&client, &ha, &hb, &hs, cm, ck, cn, 0.75, None);
        let o2 =
            w4a16_linear_swz_launch::<Rt>(&client, &ha, &hbp, &hsp, cm, ck, cn, true, 0.75, None);
        let o3 =
            w4a16_linear_swz_launch::<Rt>(&client, &ha, &hbp, &hs, cm, ck, cn, false, 0.75, None);
        let f = |h| -> Vec<f32> {
            client
                .read_one(h)
                .unwrap()
                .chunks_exact(4)
                .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                .collect()
        };
        let (v1, v2, v3) = (f(o1), f(o2), f(o3));
        let dev = |x: &[f32], y: &[f32]| {
            x.iter()
                .zip(y)
                .map(|(p, q)| (p - q).abs())
                .fold(0.0f32, f32::max)
        };
        let mag = v1.iter().fold(0.0f32, |m, x| m.max(x.abs()));
        println!(
            "\n  numerics (observation, not a gate), [{cm}, {ck}] x [{cn}, {ck}]^T over {} outputs:\n\
             \x20   max |swz codes+scales - row-major| = {:.3e}   (largest output magnitude {mag:.3})\n\
             \x20   max |swz codes only   - row-major| = {:.3e}",
            v1.len(),
            dev(&v1, &v2),
            dev(&v1, &v3),
        );
    }

    // The A-side LIVE-ROW MASK, as a GATE and not an observation.
    //
    // The mask makes the kernel decline to LOAD an A fragment row that is M
    // padding and feed the MMA a register zero instead. Two independent reasons
    // it cannot move a kept output bit -- the padding rows are zero, and
    // `D[r][c]` reads A row `r` and no other -- so the only admissible deviation
    // is 0.000e0 exactly, over EVERY output including the padding rows the
    // caller slices away. Anything else is a wrong predicate, and this refuses
    // rather than printing it.
    //
    // Two shapes, because the failure modes are different. `[16, ...]` is DECODE
    // (one m-tile, `m_base` always 0). `[48, ...]` at 37 live rows is the LAST
    // TILE case: tiles 0 and 1 are entirely live and tile 2 has five live rows
    // and eleven padding, so a predicate that forgot `m_base` passes the first
    // shape and fails this one.
    {
        let (ck, cn) = (256usize, 64usize);
        let cc = cn * ck / 2;
        let cs = cn * (ck / 16);
        let wb: Vec<u8> = (0..cc).map(|i| (i.wrapping_mul(97) % 253) as u8).collect();
        let sb: Vec<u8> = (0..cs).map(|i| [0x38u8, 0x40, 0x30][i % 3]).collect();
        let mut pc = vec![0u8; cc];
        let mut ps = vec![0u8; cs];
        swizzle_w4a16_codes_into(&wb, &mut pc, cn, ck);
        swizzle_w4a16_scales_into(&sb, &mut ps, cn, ck);
        let hb = client.create_from_slice(&wb);
        let hs = client.create_from_slice(&sb);
        let hbp = client.create_from_slice(&pc);
        let hsp = client.create_from_slice(&ps);

        let read = |h| -> Vec<f32> {
            client
                .read_one(h)
                .unwrap()
                .chunks_exact(4)
                .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                .collect()
        };
        let bits = |x: &[f32], y: &[f32]| -> (f32, usize) {
            let mut worst = 0.0f32;
            let mut n = 0usize;
            for (p, q) in x.iter().zip(y) {
                if p.to_bits() != q.to_bits() {
                    n += 1;
                }
                worst = worst.max((p - q).abs());
            }
            (worst, n)
        };

        let mut bad = 0usize;
        println!("\n  A-side live-row mask, bit-identity (GATE):");
        for (cm, live) in [(16usize, 1usize), (16, 8), (16, 9), (48, 37)] {
            // The activation as the runtime hands it over: `live` real rows,
            // then ZERO to the tile boundary. That zeroing is `pad_bf16`'s and
            // is the thing the mask is allowed to assume.
            let ab: Vec<u8> = (0..cm * ck)
                .flat_map(|i| {
                    let v = if i / ck < live {
                        half::bf16::from_f32((i % 13) as f32 * 0.25 - 1.5)
                    } else {
                        half::bf16::ZERO
                    };
                    v.to_le_bytes()
                })
                .collect();
            let ha = client.create_from_slice(&ab);

            let row_off = read(w4a16_linear_launch::<Rt>(
                &client, &ha, &hb, &hs, cm, ck, cn, 0.75, None,
            ));
            let row_on = read(w4a16_linear_launch::<Rt>(
                &client,
                &ha,
                &hb,
                &hs,
                cm,
                ck,
                cn,
                0.75,
                Some(live),
            ));
            let swz_off = read(w4a16_linear_swz_launch::<Rt>(
                &client, &ha, &hbp, &hsp, cm, ck, cn, true, 0.75, None,
            ));
            let swz_on = read(w4a16_linear_swz_launch::<Rt>(
                &client,
                &ha,
                &hbp,
                &hsp,
                cm,
                ck,
                cn,
                true,
                0.75,
                Some(live),
            ));
            let (dr, nr) = bits(&row_off, &row_on);
            let (ds, ns) = bits(&swz_off, &swz_on);
            bad += nr + ns;
            println!(
                "\x20   [{cm}, {ck}] x [{cn}, {ck}]^T, {live} live of {cm}, {} outputs: \
                 row-major {:.3e} ({nr} bits differ)   swizzled {:.3e} ({ns} bits differ)",
                row_off.len(),
                dr,
                ds,
            );
        }
        println!(
            "\x20   framing: masked vs unmasked, SAME binary and SAME kernel source, the two arms \
             differing only in the comptime `mask_rows`; every output compared, padding rows \
             included; deviation is max |on - off| and the count is exact f32 bit inequality."
        );
        if bad != 0 {
            eprintln!(
                "REFUSED: the live-row mask is not bit-identical ({bad} outputs differ). \
                 That is a wrong predicate, not a rounding difference."
            );
            std::process::exit(2);
        }
        println!("\x20   0 outputs differ anywhere: the mask is bit-identical.");
    }

    // The COALESCED LOAD + WARP SHUFFLE weight load, as a GATE.
    //
    // `w4a16gemm::swz_shuffle` reads the SAME words out of the SAME swizzled
    // planes into the SAME `w_buf` slots -- it changes which lane's load
    // instruction fetches a word, and nothing about which word or when it is
    // consumed. So this is a pure data-movement change and the only admissible
    // deviation is 0.000e0 exactly, on every output. An approximation here
    // would be a wrong source lane, which is a permutation error and would look
    // like plausible numbers.
    //
    // Both arms are launched from ONE process by
    // `w4a16_linear_swz_launch_redist`, so neither the environment nor two
    // builds can be the difference between them.
    //
    // Four shapes, and the last two are the ones with teeth:
    //   * `[16, 256]` decode and `[48, 256]` last-tile, `swz_sc` ON: the
    //     shipped configuration, `k_tiles = 16`, load depth 4.
    //   * `swz_sc` OFF: swizzled codes against a ROW-MAJOR scale plane, which
    //     the redistribution must decline -- that plane strides by `spr` per
    //     lane and is not a warp-contiguous read, so only the codes may move.
    //   * `[16, 96]`: `k_tiles = 6`, which the load depth of 4 does not divide,
    //     so the launch falls back to depth 1 and MUST also fall back to the
    //     fragment-shaped load -- one 32-word step is two k-tiles and there is
    //     no half step. A silent failure here fills `w_buf` not at all.
    {
        let read = |h| -> Vec<f32> {
            client
                .read_one(h)
                .unwrap()
                .chunks_exact(4)
                .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                .collect()
        };
        let mut bad = 0usize;
        println!("\n  coalesced load + warp shuffle, bit-identity (GATE):");
        for (cm, live, ck, sc_swz) in [
            (16usize, 1usize, 256usize, true),
            (48, 37, 256, true),
            (16, 1, 256, false),
            (16, 1, 96, true),
        ] {
            let cn = 64usize;
            let cc = cn * ck / 2;
            let cs = cn * (ck / 16);
            let wb: Vec<u8> = (0..cc).map(|i| (i.wrapping_mul(97) % 253) as u8).collect();
            let sb: Vec<u8> = (0..cs).map(|i| [0x38u8, 0x40, 0x30][i % 3]).collect();
            let mut pc = vec![0u8; cc];
            let mut ps = vec![0u8; cs];
            swizzle_w4a16_codes_into(&wb, &mut pc, cn, ck);
            swizzle_w4a16_scales_into(&sb, &mut ps, cn, ck);
            let hbp = client.create_from_slice(&pc);
            let hsc = if sc_swz {
                client.create_from_slice(&ps)
            } else {
                client.create_from_slice(&sb)
            };
            let ab: Vec<u8> = (0..cm * ck)
                .flat_map(|i| {
                    let v = if i / ck < live {
                        half::bf16::from_f32((i % 13) as f32 * 0.25 - 1.5)
                    } else {
                        half::bf16::ZERO
                    };
                    v.to_le_bytes()
                })
                .collect();
            let ha = client.create_from_slice(&ab);

            let off = read(w4a16_linear_swz_launch_redist::<Rt>(
                &client,
                &ha,
                &hbp,
                &hsc,
                cm,
                ck,
                cn,
                sc_swz,
                0.75,
                Some(live),
                false,
            ));
            let on = read(w4a16_linear_swz_launch_redist::<Rt>(
                &client,
                &ha,
                &hbp,
                &hsc,
                cm,
                ck,
                cn,
                sc_swz,
                0.75,
                Some(live),
                true,
            ));
            let mut worst = 0.0f32;
            let mut n = 0usize;
            for (p, q) in off.iter().zip(&on) {
                if p.to_bits() != q.to_bits() {
                    n += 1;
                }
                worst = worst.max((p - q).abs());
            }
            let nonzero = off.iter().filter(|v| **v != 0.0).count();
            bad += n;
            println!(
                "\x20   [{cm}, {ck}] x [{cn}, {ck}]^T, {live} live of {cm}, swz_sc {sc_swz}, \
                 {} outputs ({nonzero} nonzero): {worst:.3e} ({n} bits differ)",
                off.len(),
            );
        }
        println!(
            "\x20   framing: shuffle off vs shuffle on, SAME binary and SAME kernel source, the \
             two arms differing only in the comptime `redist`; every output compared, padding \
             rows included; deviation is max |on - off| and the count is exact f32 bit inequality."
        );
        if bad != 0 {
            eprintln!(
                "REFUSED: the redistributed weight load is not bit-identical ({bad} outputs \
                 differ). It moves the same words to the same slots, so anything but 0 is a \
                 wrong source lane, not a rounding difference."
            );
            std::process::exit(2);
        }
        println!("\x20   0 outputs differ anywhere: the redistributed load is bit-identical.");
    }
}
