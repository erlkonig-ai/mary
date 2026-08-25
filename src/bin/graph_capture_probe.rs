//! `graph_capture_probe` — does a CUDA graph capture survive cubecl's dispatch
//! path at all, and what does a replay cost per node?
//!
//! ## What is measured, and per what
//!
//! One shape of the real BF16 GEMM lane (`bf16_linear_launch`), issued `CHAIN`
//! times back to back, on one GB10 (sm_121a, CUDA 13.0, driver 580.173.02).
//! Two arms, INTERLEAVED per rep, first `COLD` reps discarded:
//!
//! * `eager`  — `CHAIN` ordinary launches. `host` is the wall time of the
//!   enqueue loop alone (cubecl's `submit` is fire-and-forget, so this is the
//!   caller-thread Rust/CubeCL path, NOT the driver's launch cost); `e2e` adds
//!   a read-back so the device work is included.
//! * `replay` — ONE `graph_replay` of a graph captured from the same `CHAIN`.
//!   `host` is the wall time of that single call; `e2e` adds the same read-back.
//!
//! Both `host` figures are quoted PER LAUNCH (eager) / PER NODE (replay) by
//! dividing by the node count the graph itself reports, so the two columns are
//! the same unit and can be compared. `e2e` is per rep, not per node, because
//! the device work does not divide.
//!
//! This probe deliberately does NOT model the decode loop: it answers only
//! "does the seam work, and what is the ceiling". A shape this small is
//! L2-resident and its device time is not representative of anything.
//!
//! Run: `graph_capture_probe [chain] [reps] [m]`

use anyhow::Result;
use cubecl::prelude::*;
use half::bf16;

use mary::models::inkling::bf16gemm::{KTILE, MTILE, NTILE, bf16_linear_launch};

type Rt = cubecl::cuda::CudaRuntime;

fn slab(n: usize, seed: f32) -> Vec<u8> {
    let mut v = Vec::with_capacity(n * 2);
    for i in 0..n {
        let f = ((i as f32 * 0.017_31 + seed).sin() * 0.5) as f32;
        v.extend_from_slice(&bf16::from_f32(f).to_le_bytes());
    }
    v
}

fn main() -> Result<()> {
    let mut a = std::env::args().skip(1);
    let chain: usize = a.next().and_then(|s| s.parse().ok()).unwrap_or(64);
    let reps: usize = a.next().and_then(|s| s.parse().ok()).unwrap_or(7);
    let m: usize = a.next().and_then(|s| s.parse().ok()).unwrap_or(MTILE);
    const COLD: usize = 2;

    // A shape the forward actually issues, rounded to the lane's tiles.
    let k = 8 * KTILE;
    let n = 8 * NTILE;

    let device = Default::default();
    let client = Rt::client(&device);

    println!("graph_capture_probe: chain={chain} reps={reps} m={m} k={k} n={n}");
    println!("capture supported: {}", client.graph_capture_supported());
    if !client.graph_capture_supported() {
        anyhow::bail!("this backend cannot capture");
    }

    let ah = client.create_from_slice(&slab(m * k, 0.3));
    let bh = client.create_from_slice(&slab(k * n, 1.7));

    // WARM. The first launch of a shape compiles it (NVRTC + cuModuleLoadData)
    // and the first reserve of a size allocates the pool page. Both are exactly
    // what must not happen inside a capture, so both happen here.
    for _ in 0..8 {
        let out = bf16_linear_launch::<Rt>(&client, &ah, &bh, m, k, n);
        let _ = client.read_one(out);
    }

    let mut eager_host = Vec::new();
    let mut eager_e2e = Vec::new();
    let mut rep_host = Vec::new();
    let mut rep_e2e = Vec::new();
    let mut nodes = 0usize;
    let mut graph: Option<u64> = None;

    for rep in 0..reps {
        // ---- eager arm ----
        let t = std::time::Instant::now();
        let mut last = None;
        for _ in 0..chain {
            last = Some(bf16_linear_launch::<Rt>(&client, &ah, &bh, m, k, n));
        }
        let h = t.elapsed();
        let _ = client.read_one(last.take().unwrap());
        let e = t.elapsed();

        // ---- capture, once, on the first rep ----
        if graph.is_none() {
            // Drain anything the drop queue is holding BEFORE the capture opens:
            // inside it the flush is suppressed, so it must not be due.
            client.flush();
            client.graph_capture_begin();
            let mut broke_at = None;
            for i in 0..chain {
                let _ = bf16_linear_launch::<Rt>(&client, &ah, &bh, m, k, n);
                if broke_at.is_none() && client.graph_capture_status() != 1 {
                    broke_at = Some(i);
                }
            }
            if let Some(i) = broke_at {
                println!("capture INVALIDATED after launch {i} of {chain}");
            }
            let g = client.graph_capture_end();
            nodes = client.graph_node_count(g);
            println!("captured: {nodes} nodes from {chain} launches");
            graph = Some(g);
        }
        let g = graph.unwrap();

        // ---- replay arm ----
        let t = std::time::Instant::now();
        client.graph_replay(g);
        let h2 = t.elapsed();
        let probe = client.empty(4);
        let _ = client.read_one(probe);
        let e2 = t.elapsed();

        if rep >= COLD {
            eager_host.push(h.as_secs_f64() * 1e6);
            eager_e2e.push(e.as_secs_f64() * 1e6);
            rep_host.push(h2.as_secs_f64() * 1e6);
            rep_e2e.push(e2.as_secs_f64() * 1e6);
        }
    }

    let stat = |v: &[f64]| {
        let n = v.len() as f64;
        let mu = v.iter().sum::<f64>() / n;
        let sd = (v.iter().map(|x| (x - mu).powi(2)).sum::<f64>() / n).sqrt();
        (mu, sd)
    };
    let per = nodes.max(1) as f64;

    println!("\n=== {} reps kept of {reps} (first {COLD} discarded) ===", eager_host.len());
    println!("nodes in graph: {nodes}   (launches issued: {chain})");
    let (mu, sd) = stat(&eager_host);
    println!("eager  host, per launch : {:8.3} us  (+/- {:.3})   [{} launches/rep]", mu / per, sd / per, chain);
    let (mu, sd) = stat(&rep_host);
    println!("replay host, per node   : {:8.3} us  (+/- {:.3})   [1 replay/rep]", mu / per, sd / per);
    let (mu, sd) = stat(&eager_host);
    println!("eager  host, per rep    : {:8.1} us  (+/- {:.1})", mu, sd);
    let (mu, sd) = stat(&rep_host);
    println!("replay host, per rep    : {:8.1} us  (+/- {:.1})", mu, sd);
    let (mu, sd) = stat(&eager_e2e);
    println!("eager  e2e,  per rep    : {:8.1} us  (+/- {:.1})", mu, sd);
    let (mu, sd) = stat(&rep_e2e);
    println!("replay e2e,  per rep    : {:8.1} us  (+/- {:.1})", mu, sd);

    println!("\nper-rep eager host us : {:?}", eager_host.iter().map(|x| (x * 10.0).round() / 10.0).collect::<Vec<_>>());
    println!("per-rep replay host us: {:?}", rep_host.iter().map(|x| (x * 10.0).round() / 10.0).collect::<Vec<_>>());
    println!("per-rep eager e2e us  : {:?}", eager_e2e.iter().map(|x| (x * 10.0).round() / 10.0).collect::<Vec<_>>());
    println!("per-rep replay e2e us : {:?}", rep_e2e.iter().map(|x| (x * 10.0).round() / 10.0).collect::<Vec<_>>());

    client.graph_destroy(graph.unwrap());
    Ok(())
}
