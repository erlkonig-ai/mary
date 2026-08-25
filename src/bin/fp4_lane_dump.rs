//! The hand 4-bit GEMM lanes at the HEAD shape, for disassembly and for one
//! ceiling number.
//!
//! Two jobs, both short.
//!
//! **Dump.** Launching `w4a16_linear` and `fp4_linear` once at `m_pad = 16`,
//! `k = 4096` makes CubeCL emit their CUDA (the shape's comptime constants pick
//! the loop trip count and the strides, so they have to be the real ones).
//! `CUBECL_DEBUG_LOG=… CUBECL_DEBUG_OPTION=debug` writes it, and it is then
//! compiled offline for `sm_121a` and disassembled — no GPU needed for that
//! half.
//!
//! **Ceiling.** `stream_packed` reads the SAME `[n, k/8]` packed table and the
//! SAME `[n, k/16]` scales with 128-bit coalesced loads and no arithmetic. It
//! is what a four-bit head would be bound by if it were bound by the bus, so it
//! is the number the hand lanes' 74 GB/s should be read against — not the BF16
//! lane's 163 GB/s, which is a different table and a different kernel.
//!
//! It read 168-172 GB/s until 2026-08-25 and that figure was WRONG in both
//! directions at once: the thread stride was a u32-word count applied to a
//! tensor CubeCL indexes in 16-byte vectors, so every thread strode four times
//! past the end of the buffer, and the store was unconditional, so the control
//! also paid a 12.5% write tax it charged to a read figure. Corrected it reads
//! ~248 GB/s — the SAME as f32 and BF16 over the same bytes. There is no
//! four-bit read penalty on this part; see `inkling_membw` with
//! `INK_BW_AXES=1`, which puts all of those side by side in one process.
//!
//! `INK_HEAD_N` sets `n` (default 201024, the unembedding's own). Set it to
//! something small to make this a compile-only dump.

use std::time::Instant;

use cubecl::future;

use cubecl::prelude::*;
use mary::models::inkling::bf16gemm::{
    Lane, try_bf16_linear_cubek_launch, try_w4a16_linear_cubek_launch,
};
use mary::models::inkling::fp4gemm::fp4_linear_launch;
use mary::models::inkling::w4a16gemm::{w4a16_linear_launch, w4a16_linear_wide_launch};

type Rt = cubecl::cuda::CudaRuntime;

/// `u32` words each thread reads per grid-stride step. Unrolled, so it is a run
/// of independent 128-bit loads and the memory pipe stays fed.
const PER: usize = 8;
const BLOCK: u32 = 256;

/// Stream the packed table and its scales once, reducing so nothing is elided.
///
/// Thread `t` reads `w[4t..4t+4]`, `w[4t + 4T ..]`, … — at every step a warp
/// covers 512 consecutive bytes, which is the fully-coalesced case the hand
/// lanes are being measured against.
#[cube(launch)]
pub fn stream_packed<NW: Size>(
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
    // The scale plane is 1/8 the words; one step over it is the right ratio.
    let s = sc[t % sc.len()];
    acc += s[0];
    // Sentinel-guarded, so the control pays NO write traffic. Unconditional it
    // stored one word per eight 16-byte loads -- a 12.5% write tax the kernels
    // it is a ceiling for do not pay, charged to a figure that counts only
    // reads.
    if acc == u32::new(0x5AFE_5AFEi64) {
        out[t % out.len()] = acc;
    }
}

fn main() {
    let device = Default::default();
    let client = Rt::client(&device);

    let n: usize = std::env::var("INK_HEAD_N")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(201024);
    // The head decodes at m = 1 padded to 16, i.e. ONE m-tile, so every weight
    // row has exactly one consumer and nothing re-reads it out of L2. Raising
    // m_pad adds m-tiles that share the same rows; if the achieved rate climbs
    // with it, the shortfall is missing reuse rather than a bad kernel.
    let m_pad: usize = std::env::var("INK_HEAD_M")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(16);
    let k = 4096usize;
    let codes = n * (k / 8) * 4;
    let scales = n * (k / 16);
    println!("head shape m_pad={m_pad} k={k} n={n}: codes {codes} B, scales {scales} B");

    // W4A16: BF16 activation, packed-u32 weight, e4m3 scales.
    let a = client.empty(m_pad * k * 2);
    let b = client.empty(codes);
    let b_sc = client.empty(scales);
    let mut w4a16_s = f64::MAX;
    for i in 0..6 {
        let t0 = Instant::now();
        let out = w4a16_linear_launch::<Rt>(&client, &a, &b, &b_sc, m_pad, k, n, 1.0);
        let _ = future::block_on(client.sync());
        let dt = t0.elapsed().as_secs_f64();
        drop(out);
        if i >= 2 {
            w4a16_s = w4a16_s.min(dt);
        }
    }

    // The same product, four planes a cube and a 16-byte B load.
    let mut wide_s = f64::MAX;
    for i in 0..6 {
        let tw = Instant::now();
        let out = w4a16_linear_wide_launch::<Rt>(&client, &a, &b, &b_sc, m_pad, k, n, 1.0);
        let _ = future::block_on(client.sync());
        let dt = tw.elapsed().as_secs_f64();
        drop(out);
        if i >= 2 {
            wide_s = wide_s.min(dt);
        }
    }

    // The control. Same tuned lanes, same m and n, but a plain BF16 weight —
    // 4x the bytes. If these are fast and the quantised ones are not, the cost
    // is the dequantising read and not the shape; if BOTH are slow, the tuned
    // tiling simply does not like m = 16 against n = 201024 and the comparison
    // says nothing about quantisation. `INK_SKIP_BF16=1` skips the 1.5 GiB
    // allocation.
    if std::env::var("INK_SKIP_BF16").is_err() {
        let bbf = client.empty(n * k * 2);
        for lane in [Lane::Auto, Lane::SimpleCyclicMma, Lane::DoubleTmaMma] {
            let mut best = f64::MAX;
            let mut err = None;
            for i in 0..6 {
                let tb = Instant::now();
                match try_bf16_linear_cubek_launch::<Rt>(&client, &a, &bbf, m_pad, k, n, lane) {
                    Ok(o) => {
                        let _ = future::block_on(client.sync());
                        let dt = tb.elapsed().as_secs_f64();
                        drop(o);
                        if i >= 2 {
                            best = best.min(dt);
                        }
                    }
                    Err(e) => {
                        err = Some(format!("{e:?}"));
                        break;
                    }
                }
            }
            match err {
                Some(e) => println!(
                    "cubek bf16 {:<16} declined: {}",
                    lane.name(),
                    e.chars().take(80).collect::<String>()
                ),
                None => println!(
                    "cubek bf16 {:<16} {:8.3} ms   {:6.1} GB/s over its own {:.2} GiB",
                    lane.name(),
                    best * 1e3,
                    (n * k * 2) as f64 / best / 1e9,
                    (n * k * 2) as f64 / (1u64 << 30) as f64
                ),
            }
        }
        drop(bbf);
    }

    // The tuned lane with a four-bit RHS: cubek dequantises on the global read,
    // inside its own staged tiling. The TMA lanes decline a quantised operand,
    // so only the sync ones are worth asking.
    for lane in [
        Lane::Auto,
        Lane::SimpleCyclicMma,
        Lane::DoubleCyclicMma,
        Lane::OrderedDoubleMma,
        Lane::SimpleCyclicCmma,
        Lane::DoubleTmaMma,
    ] {
        let mut best = f64::MAX;
        let mut err = None;
        for i in 0..6 {
            let tq = Instant::now();
            match try_w4a16_linear_cubek_launch::<Rt>(&client, &a, &b, &b_sc, m_pad, k, n, lane) {
                Ok(o) => {
                    let _ = future::block_on(client.sync());
                    let dt = tq.elapsed().as_secs_f64();
                    drop(o);
                    if i >= 2 {
                        best = best.min(dt);
                    }
                }
                Err(e) => {
                    err = Some(format!("{e:?}"));
                    break;
                }
            }
        }
        match err {
            Some(e) => println!(
                "cubek q4 {:<18} declined: {}",
                lane.name(),
                e.chars().take(90).collect::<String>()
            ),
            None => println!(
                "cubek q4 {:<18} {:8.3} ms   {:6.1} GB/s",
                lane.name(),
                best * 1e3,
                (codes + scales) as f64 / best / 1e9
            ),
        }
    }

    // W4A4: both operands packed E2M1 with e4m3 block scales.
    let qa = client.empty(m_pad * k / 2);
    let qa_sc = client.empty(m_pad * (k / 16));
    let mut fp4_s = f64::MAX;
    for i in 0..6 {
        let t1 = Instant::now();
        let o = fp4_linear_launch::<Rt>(&client, &qa, &qa_sc, &b, &b_sc, m_pad, k, n, 1.0);
        let _ = future::block_on(client.sync());
        let dt = t1.elapsed().as_secs_f64();
        drop(o);
        if i >= 2 {
            fp4_s = fp4_s.min(dt);
        }
    }

    // The ceiling: the same bytes, read coalesced, with no work on them.
    // `codes / 4` is the count of u32 WORDS; the tensor is bound as
    // `Vector<u32, 4>`, and CubeCL indexes a vectorised tensor in VECTORS, so
    // the thread stride has to be the vector count. It was the word count,
    // which strode `t + i * threads` four times past the end of a 0.431 GiB
    // buffer -- so the control issued four times the traffic it charged itself
    // for, over addresses it did not own, and the 168-172 GB/s it reported was
    // an artefact of that and not a property of reading four-bit data. The
    // corrected control reads ~248 GB/s, the same as f32 and BF16 over the same
    // bytes (`inkling_membw`, INK_BW_AXES=1, rows 1-5).
    let vectors = codes / 16;
    let threads = vectors / PER;
    let blocks = (threads as u32).div_ceil(BLOCK);
    let dst = client.empty(4096);
    let mut stream_s = f64::MAX;
    for i in 0..6 {
        let t2 = Instant::now();
        unsafe {
            stream_packed::launch::<Rt>(
                &client,
                CubeCount::Static(blocks, 1, 1),
                CubeDim::new_1d(BLOCK),
                4,
                TensorArg::from_raw_parts(b.clone(), [1].into(), [vectors].into()),
                TensorArg::from_raw_parts(b_sc.clone(), [1].into(), [scales / 16].into()),
                TensorArg::from_raw_parts(dst.clone(), [1].into(), [1024].into()),
                threads,
                PER,
            )
        };
        let _ = future::block_on(client.sync());
        let dt = t2.elapsed().as_secs_f64();
        if i >= 2 {
            stream_s = stream_s.min(dt);
        }
    }

    // What a launch pays before it has read a byte: the output allocation the
    // two GEMM launchers do internally, plus the sync. Subtract it and the
    // remainder is the read.
    let mut alloc_s = f64::MAX;
    for i in 0..6 {
        let t3 = Instant::now();
        let h = client.empty(m_pad * n * 4);
        let _ = future::block_on(client.sync());
        let dt = t3.elapsed().as_secs_f64();
        drop(h);
        if i >= 2 {
            alloc_s = alloc_s.min(dt);
        }
    }

    let gib = (codes + scales) as f64 / (1u64 << 30) as f64;
    let gbs = |s: f64| (codes + scales) as f64 / s / 1e9;
    println!("table                {gib:.3} GiB (codes + scales)");
    println!(
        "w4a16_linear (hand)  {:8.3} ms   {:6.1} GB/s",
        w4a16_s * 1e3,
        gbs(w4a16_s)
    );
    println!(
        "w4a16 wide   (hand)  {:8.3} ms   {:6.1} GB/s",
        wide_s * 1e3,
        gbs(wide_s)
    );
    println!(
        "fp4_linear   (hand)  {:8.3} ms   {:6.1} GB/s",
        fp4_s * 1e3,
        gbs(fp4_s)
    );
    println!(
        "stream_packed (ceil) {:8.3} ms   {:6.1} GB/s",
        stream_s * 1e3,
        gbs(stream_s)
    );
    println!(
        "output alloc + sync  {:8.3} ms   (paid by both GEMM rows, not by the ceiling)",
        alloc_s * 1e3
    );
    println!(
        "net of alloc:  w4a16 {:6.1} GB/s   fp4 {:6.1} GB/s   ceiling {:6.1} GB/s",
        gbs(w4a16_s - alloc_s),
        gbs(fp4_s - alloc_s),
        gbs(stream_s)
    );

    // The wide lane has to be the SAME product as the one it is timed against,
    // or the timing is a measurement of a bug. Run LAST: the check's own
    // create/read_one traffic moved the timings above by up to 30% when it ran
    // first, which is a measurement of the allocator, not of the kernels.
    {
        let (cm, ck, cn) = (16usize, 256usize, 64usize);
        let mut wb = Vec::with_capacity(cn * (ck / 8) * 4);
        for i in 0..cn * (ck / 8) {
            wb.extend_from_slice(&(0x1234_5678u32.wrapping_mul(i as u32 + 1)).to_le_bytes());
        }
        // E4M3 1.0 is 0x38; a couple of other exponents keep the scale path live.
        let sb: Vec<u8> = (0..cn * (ck / 16))
            .map(|i| [0x38u8, 0x40, 0x30][i % 3])
            .collect();
        let ab: Vec<u8> = (0..cm * ck)
            .flat_map(|i| half::bf16::from_f32((i % 13) as f32 * 0.25 - 1.5).to_le_bytes())
            .collect();
        let ha = client.create_from_slice(&ab);
        let hb = client.create_from_slice(&wb);
        let hs = client.create_from_slice(&sb);
        let o1 = w4a16_linear_launch::<Rt>(&client, &ha, &hb, &hs, cm, ck, cn, 0.75);
        let o2 = w4a16_linear_wide_launch::<Rt>(&client, &ha, &hb, &hs, cm, ck, cn, 0.75);
        let r1 = client.read_one(o1).unwrap();
        let r2 = client.read_one(o2).unwrap();
        let f = |b: &[u8]| -> Vec<f32> {
            b.chunks_exact(4)
                .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                .collect()
        };
        let (v1, v2) = (f(&r1), f(&r2));
        let bad = v1
            .iter()
            .zip(&v2)
            .enumerate()
            .find(|(_, (x, y))| (**x - **y).abs() > 1e-4 * x.abs().max(1.0));
        match bad {
            Some((i, (x, y))) => panic!("wide lane differs at {i}: {x} vs {y}"),
            None => println!("wide lane matches the original over {} outputs", v1.len()),
        }
    }
}
