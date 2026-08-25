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
//! `INK_HEAD_N` sets `n` (default 201024, the unembedding's own). Set it to
//! something small to make this a compile-only dump.

use std::time::Instant;

use cubecl::prelude::*;
use mary::models::inkling::fp4gemm::fp4_linear_launch;
use mary::models::inkling::w4a16gemm::w4a16_linear_launch;

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
pub fn stream_packed(
    w: &Tensor<Vector<u32, 4>>,
    sc: &Tensor<Vector<u32, 4>>,
    out: &mut Tensor<u32>,
    #[comptime] threads: usize,
    #[comptime] per: usize,
) {
    let t = ABSOLUTE_POS as usize;
    let mut acc = u32::new(0u32);
    #[unroll]
    for i in 0..per {
        let v = w[t + i * threads];
        acc += v[0] ^ v[1] ^ v[2] ^ v[3];
    }
    // The scale plane is 1/8 the words; one step over it is the right ratio.
    let s = sc[t % sc.len()];
    acc += s[0] ^ s[1] ^ s[2] ^ s[3];
    if t < out.len() {
        out[t] = acc;
    }
}

fn main() {
    let device = Default::default();
    let client = Rt::client(&device);

    let n: usize = std::env::var("INK_HEAD_N")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(201024);
    let (m_pad, k) = (16usize, 4096usize);
    let codes = n * (k / 8) * 4;
    let scales = n * (k / 16);
    println!("head shape m_pad={m_pad} k={k} n={n}: codes {codes} B, scales {scales} B");

    // W4A16: BF16 activation, packed-u32 weight, e4m3 scales.
    let a = client.empty(m_pad * k * 2);
    let b = client.empty(codes);
    let b_sc = client.empty(scales);
    for _ in 0..3 {
        let out = w4a16_linear_launch::<Rt>(&client, &a, &b, &b_sc, m_pad, k, n, 1.0);
        let _ = client.read_one(out.clone());
    }
    let t0 = Instant::now();
    let out = w4a16_linear_launch::<Rt>(&client, &a, &b, &b_sc, m_pad, k, n, 1.0);
    let _ = client.read_one(out.clone());
    let w4a16_s = t0.elapsed().as_secs_f64();

    // W4A4: both operands packed E2M1 with e4m3 block scales.
    let qa = client.empty(m_pad * k / 2);
    let qa_sc = client.empty(m_pad * (k / 16));
    for _ in 0..3 {
        let o = fp4_linear_launch::<Rt>(&client, &qa, &qa_sc, &b, &b_sc, m_pad, k, n, 1.0);
        let _ = client.read_one(o.clone());
    }
    let t1 = Instant::now();
    let out2 = fp4_linear_launch::<Rt>(&client, &qa, &qa_sc, &b, &b_sc, m_pad, k, n, 1.0);
    let _ = client.read_one(out2.clone());
    let fp4_s = t1.elapsed().as_secs_f64();

    // The ceiling: the same bytes, read coalesced, with no work on them.
    let words = codes / 4;
    let threads = words / PER;
    let blocks = (threads as u32).div_ceil(BLOCK);
    let dst = client.empty(threads * 4);
    let mut stream_s = f64::MAX;
    for _ in 0..4 {
        let t2 = Instant::now();
        unsafe {
            stream_packed::launch::<Rt>(
                &client,
                CubeCount::Static(blocks, 1, 1),
                CubeDim::new_1d(BLOCK),
                4,
                4,
                TensorArg::from_raw_parts(b.clone(), [1].into(), [words / 4].into()),
                TensorArg::from_raw_parts(b_sc.clone(), [1].into(), [scales / 16].into()),
                TensorArg::from_raw_parts(dst.clone(), [1].into(), [threads].into()),
                threads,
                PER,
            )
        };
        let _ = client.read_one(dst.clone());
        stream_s = stream_s.min(t2.elapsed().as_secs_f64());
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
        "fp4_linear   (hand)  {:8.3} ms   {:6.1} GB/s",
        fp4_s * 1e3,
        gbs(fp4_s)
    );
    println!(
        "stream_packed (ceil) {:8.3} ms   {:6.1} GB/s",
        stream_s * 1e3,
        gbs(stream_s)
    );
}
