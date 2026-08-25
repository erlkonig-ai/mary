//! Compile-only dump of the hand 4-bit GEMM lanes at the HEAD shape.
//!
//! Launches `w4a16_linear` and `fp4_linear` once each at `m_pad = 16`,
//! `k = 4096` (the shape's comptime constants are what pick the loop trip
//! count and the strides, so they have to be the real ones) against a
//! deliberately tiny `n`, so the grid is ~8 cubes and the GPU is touched for
//! microseconds. The point is not the result — the buffers are zeroed — it is
//! the generated CUDA that `CUBECL_DEBUG_LOG` writes on the way past, which is
//! then compiled offline for `sm_121a` and disassembled.
//!
//! `CUBECL_DEBUG_LOG=/tmp/lanes.log CUBECL_DEBUG_OPTION=debug fp4_lane_dump`

use cubecl::prelude::*;
use mary::models::inkling::fp4gemm::fp4_linear_launch;
use mary::models::inkling::w4a16gemm::w4a16_linear_launch;

type Rt = cubecl::cuda::CudaRuntime;

fn main() {
    let device = Default::default();
    let client = Rt::client(&device);

    // The head's own k, and an n just wide enough to tile.
    let (m_pad, k, n) = (16usize, 4096usize, 64usize);

    // W4A16: BF16 activation, packed-u32 weight, e4m3 scales.
    let a = client.empty(m_pad * k * 2);
    let b = client.empty(n * (k / 8) * 4);
    let b_sc = client.empty(n * (k / 16));
    let out = w4a16_linear_launch::<Rt>(&client, &a, &b, &b_sc, m_pad, k, n, 1.0);
    client.read_one(out.binding());
    println!("w4a16_linear launched at m_pad={m_pad} k={k} n={n}");

    // W4A4: both operands packed E2M1 with e4m3 block scales.
    let qa = client.empty(m_pad * k / 2);
    let qa_sc = client.empty(m_pad * (k / 16));
    let qb = client.empty(n * k / 2);
    let qb_sc = client.empty(n * (k / 16));
    let out2 = fp4_linear_launch::<Rt>(&client, &qa, &qa_sc, &qb, &qb_sc, m_pad, k, n, 1.0);
    client.read_one(out2.binding());
    println!("fp4_linear launched at m_pad={m_pad} k={k} n={n}");
}
