//! Does the pre-permuted weight layout help a SMALL launch as much as a big one?
//!
//! The permutation ([`fp4_linear_swz`]) fixes the *access pattern*: it makes a
//! lane's B loads coalesce, and on `inkling_membw` it is worth 114 -> 205 GB/s
//! at the routed-expert shape. That is a fix to how many bytes a memory
//! transaction delivers.
//!
//! The sink shortfall is a different axis. `w4a16_shape_sweep` shows the same
//! hand kernel reading 71 GB/s at the down shape (512 cubes) and 120 GB/s at
//! 24576 cubes over byte-identical tables — a fix to how many warps are in
//! flight. Both lanes put ONE 32-thread cube on each (m_tile 16, n_tile 8), so a
//! decode-shaped launch is exactly `n / 8` warps either way, and the permutation
//! does not add a single one.
//!
//! Whether the two compose is not deducible — wider transactions per lane raise
//! the memory-level parallelism a single warp carries, which is exactly what a
//! warp-starved launch is short of, so the permutation could plausibly cover
//! part of the grid deficit. This measures it: base and swizzled at the same n,
//! swept from 512 cubes to 25128, and the column that matters is the RATIO.
//!
//! * ratio roughly constant  -> the two axes are independent. Permuting the
//!   sinks buys the same multiple it buys the head, on top of a still-starved
//!   grid, and the grid fix is worth its own separate multiple.
//! * ratio collapsing at small n -> the permutation cannot be spent on a launch
//!   that has no warps to spend it on, and the sinks need the grid fixed first.
//!
//! Timing only: the kernels read the same bytes in a different ORDER, so an
//! uninitialised table exercises the identical access pattern. Correctness of
//! the permutation is `fp4gemm`'s own gate, not this one's.

use std::time::Instant;

use cubecl::future;
use cubecl::prelude::*;
use cubecl::server::Handle;
use mary::models::inkling::fp4gemm::{fp4_linear_launch, fp4_linear_swz_launch};

type Rt = cubecl::cuda::CudaRuntime;

const REPS: usize = 20;
const ROUNDS: usize = 6;

/// Weight-side bytes of a `[n, k]` NVFP4 operand: `k/2` codes + `k/16` scales.
fn table_bytes(n: usize, k: usize) -> usize {
    n * (k / 2) + n * (k / 16)
}

struct Arm {
    a: Handle,
    a_sc: Handle,
    rot: Vec<(Handle, Handle)>,
}

impl Arm {
    fn new(client: &ComputeClient<Rt>, m_pad: usize, k: usize, n: usize) -> Self {
        let bytes = table_bytes(n, k);
        // The GAP between two reads of one buffer must clear L2 by a wide
        // margin, or the small shapes are quietly served from cache -- which is
        // exactly the bias that would fake this result.
        let rot = (1 + (256usize << 20).div_ceil(bytes.max(1))).clamp(2, 130);
        let rot = rot.min(((4usize << 30) / bytes.max(1)).max(1));
        Arm {
            a: client.empty(m_pad * k / 2),
            a_sc: client.empty(m_pad * (k / 16)),
            rot: (0..rot)
                .map(|_| (client.empty(n * (k / 2)), client.empty(n * (k / 16))))
                .collect(),
        }
    }
}

fn main() {
    let client = Rt::client(&Default::default());
    let k: usize = std::env::var("INK_K")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(4096);
    let m_pad = 16usize;

    println!("=== pre-permuted vs row-major, swept across grid size ===");
    println!("both lanes: one 32-thread cube per (m_tile 16, n_tile 8), so cubes = n/8");
    println!("k = {k}, m_pad = {m_pad}; GB/s over the weight table; min of {ROUNDS} rounds\n");
    println!(
        "{:>8} {:>7} {:>9} {:>10} {:>10} {:>9} {:>9} {:>7}",
        "n", "cubes", "MiB", "base ms", "swz ms", "base GB/s", "swz GB/s", "ratio"
    );

    let ns: Vec<usize> = [4096usize, 8192, 16384, 32768, 65536, 131072, 201024].to_vec();
    let arms: Vec<Arm> = ns.iter().map(|&n| Arm::new(&client, m_pad, k, n)).collect();
    let mut base = vec![f64::MAX; ns.len()];
    let mut swz = vec![f64::MAX; ns.len()];

    // Round-robin over shapes AND over the two lanes, so a clock ramp or a
    // neighbour on the box cannot land on one arm or one end of the curve.
    for r in 0..ROUNDS + 2 {
        for (i, (&n, arm)) in ns.iter().zip(&arms).enumerate() {
            let t0 = Instant::now();
            for j in 0..REPS {
                let (b, sc) = &arm.rot[j % arm.rot.len()];
                let o =
                    fp4_linear_launch::<Rt>(&client, &arm.a, &arm.a_sc, b, sc, m_pad, k, n, 1.0);
                drop(o);
            }
            let _ = future::block_on(client.sync());
            let db = t0.elapsed().as_secs_f64() / REPS as f64;

            let t1 = Instant::now();
            for j in 0..REPS {
                let (b, sc) = &arm.rot[j % arm.rot.len()];
                let o = fp4_linear_swz_launch::<Rt>(
                    &client, &arm.a, &arm.a_sc, b, sc, m_pad, k, n, 1.0, true,
                );
                drop(o);
            }
            let _ = future::block_on(client.sync());
            let ds = t1.elapsed().as_secs_f64() / REPS as f64;

            if r >= 2 {
                base[i] = base[i].min(db);
                swz[i] = swz[i].min(ds);
            }
        }
    }

    for (i, &n) in ns.iter().enumerate() {
        let bytes = table_bytes(n, k);
        println!(
            "{:>8} {:>7} {:>9.2} {:>10.4} {:>10.4} {:>9.1} {:>9.1} {:>7.2}",
            n,
            n / 8,
            bytes as f64 / (1u64 << 20) as f64,
            base[i] * 1e3,
            swz[i] * 1e3,
            bytes as f64 / base[i] / 1e9,
            bytes as f64 / swz[i] / 1e9,
            base[i] / swz[i]
        );
    }
    println!("\nratio flat  -> permutation and grid are independent axes");
    println!("ratio falls -> a starved grid cannot spend the permutation");
}
