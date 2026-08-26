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
//! `INK_SWZ_N` / `INK_SWZ_K` set the shape (default the unembedding's
//! `[201024, 4096]`), `INK_SWZ_M` the padded row count (default 16, one m-tile,
//! which is decode), `INK_SWZ_REPS` the rep count (default 8).

use std::time::Instant;

use cubecl::future;
use cubecl::prelude::*;
use mary::models::inkling::w4a16gemm::{
    swizzle_w4a16_codes_into, swizzle_w4a16_scales_into, w4a16_linear_launch,
    w4a16_linear_swz_launch,
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
    let reps = env("INK_SWZ_REPS", 8);
    let codes = n * k / 2;
    let scales = n * (k / 16);
    let bytes = codes + scales;

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
                    let o =
                        w4a16_linear_launch::<Rt>(&client, &a, &b_row, &bs_row, m_pad, k, n, 1.0);
                    let _ = future::block_on(client.sync());
                    drop(o);
                }
                1 => {
                    let o = w4a16_linear_swz_launch::<Rt>(
                        &client, &a, &b_swz, &bs_swz, m_pad, k, n, true, 1.0,
                    );
                    let _ = future::block_on(client.sync());
                    drop(o);
                }
                2 => {
                    let o = w4a16_linear_swz_launch::<Rt>(
                        &client, &a, &b_swz, &bs_row, m_pad, k, n, false, 1.0,
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
        "\n  framing: per LAUNCH of one [{m_pad}, {k}] x [{n}, {k}]^T product, {} warm reps of {reps} \
         (first 2 discarded), arms interleaved, one GB10 box, one process, same buffers.\n  \
         GB/s counts the weight planes only ({:.3} GiB), which is the traffic the permutation is \
         about; it is not a step figure and not a two-node figure.",
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
        let o1 = w4a16_linear_launch::<Rt>(&client, &ha, &hb, &hs, cm, ck, cn, 0.75);
        let o2 = w4a16_linear_swz_launch::<Rt>(&client, &ha, &hbp, &hsp, cm, ck, cn, true, 0.75);
        let o3 = w4a16_linear_swz_launch::<Rt>(&client, &ha, &hbp, &hs, cm, ck, cn, false, 0.75);
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
}
