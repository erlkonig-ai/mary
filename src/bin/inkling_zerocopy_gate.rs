//! `inkling_zerocopy_gate` — does aliasing the checkpoint's mmap'd pages give
//! the same answer as copying them, and what keeps the pages mapped?
//!
//! GB10 reports `pageableMemoryAccessUsesHostPageTables = 1`: the GPU walks the
//! host page tables, so an ordinary file-backed `mmap` is addressable by a
//! kernel at its host address, and `cudaHostGetDevicePointer` hands the same
//! address back. The CUDA `register_external_aliased` seam therefore allocates
//! nothing, copies nothing, and translates nothing — it records the pointer.
//!
//! That makes the interesting failure mode a lifetime bug rather than an
//! addressing one, and lifetime bugs here are nasty: a kernel reading unmapped
//! pages does not reliably fault, it reads whatever the address holds now. So
//! the third check below deliberately destroys the weight source — every
//! mapping it owns — while a handle is still alive, and then reads through that
//! handle. It passes only if the keepalive really is holding the mapping.
//!
//! Runs against EITHER backing: a checkpoint directory or a `.pile`. The seam
//! is the same one — [`Aliases`] registers whatever mappings the source reads
//! through and locates a slab inside them by pointer containment — so a pile,
//! being one file, is one registration where the checkpoint is nine.
//!
//!   inkling_zerocopy_gate <ckpt-dir | pile> [branch]
//!
//! Build: `--features cuda-backend,inkling`

use std::path::PathBuf;
use std::time::Instant;

use anyhow::Result;
use cubecl::prelude::*;

use mary::models::inkling::fp4gemm::{fp4_linear_launch, upload_quantized_act, Aliases};
use mary::models::inkling::nvfp4::{e4m3_to_f32, FP4_E2M1};
use mary::models::inkling::source::Weights;

type Rt = cubecl::cuda::CudaRuntime;
const LAYER: usize = 10;

fn decode(codes: &[u8], scales: &[u8], rows: usize, k: usize, scale2: f32) -> Vec<f32> {
    let mut out = vec![0f32; rows * k];
    for r in 0..rows {
        for j in 0..k {
            let byte = codes[r * (k / 2) + j / 2];
            let c = if j % 2 == 0 { byte & 0x0F } else { byte >> 4 };
            out[r * k + j] =
                FP4_E2M1[c as usize] * e4m3_to_f32(scales[r * (k / 16) + j / 16]) * scale2;
        }
    }
    out
}

fn main() -> Result<()> {
    let dir = std::env::args()
        .nth(1)
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("models/thinkingmachines-inkling-small-nvfp4"));
    let branch = std::env::args().nth(2).unwrap_or_else(|| "inkling".to_string());
    let b13 = format!("model.llm.layers.{LAYER}.mlp.experts.w13_weight");
    // One line decides the backing, and nothing below it knows which won.
    let open = || -> Result<Weights> {
        if dir.extension().map(|e| e == "pile").unwrap_or(false) {
            Weights::open_pile(&dir, &branch)
        } else {
            Weights::open_ckpt(&dir)
        }
    };

    let client = Rt::client(&Default::default());
    println!("  source : {}", open()?.kind());
    println!("=== zero-copy seam ===");
    println!(
        "  device can address host memory directly : {}",
        cubecl::cuda::supports_zero_copy_host(0)
    );

    // ---------------------------------------------------------------- 1
    // Alignment of what we are actually going to alias.
    {
        let src = open()?;
        let w = src.expert_packed(&b13, 0)?;
        println!(
            "  w13 expert-0 slab: {:.1} MB at {:p}  (mod 4 = {}, mod 16 = {})",
            w.codes().len() as f64 / 1e6,
            w.codes().as_ptr(),
            w.codes().as_ptr() as usize % 4,
            w.codes().as_ptr() as usize % 16,
        );
        println!(
            "  scale slab:        {:.1} MB at {:p}  (mod 16 = {})",
            w.scales().len() as f64 / 1e6,
            w.scales().as_ptr(),
            w.scales().as_ptr() as usize % 16,
        );
        println!("  mappings to register: {}", src.mappings()?.len());
    }

    // ---------------------------------------------------------------- 2
    // Aliased vs copied, through the real expert GEMM. Same bytes and the same
    // kernel, so anything but a bitwise match is the seam's fault.
    let (n, k, worst, bitmatch, gbps_alias, gbps_copy) = {
        let src = open()?;
        let al = Aliases::register(&client, src.mappings()?)
            .expect("the device cannot address host memory directly");
        let w13 = src.expert_packed(&b13, 0)?;
        let (n, k) = (w13.rows(), w13.cols() * 2);

        let probe = src.expert_packed(&b13, 7)?;
        let tokens = 5usize;
        let x = decode(probe.codes(), probe.scales(), tokens, k, probe.scale2());
        let (a, asc, m_pad) = upload_quantized_act(&client, &x, tokens, k);

        let alias_codes = al
            .slice(w13.codes())
            .expect("aliasing refused -- see the alignment line above");
        let alias_scales = al.slice(w13.scales()).expect("aliasing refused for scales");
        let copy_codes = client.create_from_slice(w13.codes());
        let copy_scales = client.create_from_slice(w13.scales());

        let run = |bc: &cubecl::server::Handle, bs: &cubecl::server::Handle| {
            fp4_linear_launch(&client, &a, &asc, bc, bs, m_pad, k, n, w13.scale2())
        };

        let ya = f32::from_bytes(&client.read_one(run(&alias_codes, &alias_scales)).unwrap()).to_vec();
        let yc = f32::from_bytes(&client.read_one(run(&copy_codes, &copy_scales)).unwrap()).to_vec();
        let bitmatch = ya.iter().zip(&yc).all(|(p, q)| p.to_bits() == q.to_bits());

        // f64 reference, so "they agree" cannot mean "both wrong the same way".
        let a_deq = {
            let mut padded = vec![0f32; m_pad * k];
            padded[..tokens * k].copy_from_slice(&x);
            let (ac, asb) = mary::models::inkling::fp4gemm::quantize_act_host(&padded, k);
            decode(&ac, &asb, tokens, k, 1.0)
        };
        let b_deq = decode(w13.codes(), w13.scales(), n, k, w13.scale2());
        let mut worst = 0.0f64;
        for r in 0..tokens {
            for c in (0..n).step_by(211) {
                let mut s = 0.0f64;
                for j in 0..k {
                    s += a_deq[r * k + j] as f64 * b_deq[c * k + j] as f64;
                }
                let e = (ya[r * n + c] as f64 - s).abs() / s.abs().max(1e-12);
                if e > worst {
                    worst = e;
                }
            }
        }

        // Bandwidth: the GEMM streams the whole weight once per call.
        let bytes = (w13.codes().len() + w13.scales().len()) as f64;
        let reps = 20;
        let bench = |bc: &cubecl::server::Handle, bs: &cubecl::server::Handle| {
            let h = run(bc, bs);
            client.sync();
            core::hint::black_box(&h);
            let t = Instant::now();
            for _ in 0..reps {
                let h = run(bc, bs);
                core::hint::black_box(&h);
            }
            client.sync();
            let secs = t.elapsed().as_secs_f64() / reps as f64;
            (bytes / secs / 1e9, secs * 1e3)
        };
        let (ga, ma) = bench(&alias_codes, &alias_scales);
        let (gc, mc) = bench(&copy_codes, &copy_scales);
        println!();
        println!("  GEMM reading ALIASED weights : {ma:6.3} ms  {ga:7.2} GB/s");
        println!("  GEMM reading COPIED weights  : {mc:6.3} ms  {gc:7.2} GB/s");
        (n, k, worst, bitmatch, ga, gc)
    };

    println!();
    println!("  aliased vs copied, bitwise identical : {bitmatch}");
    println!("  aliased vs f64 reference (N={n}, K={k}) : {worst:.3e}");

    // ---------------------------------------------------------------- 3
    // The lifetime gate. Destroy every mapping the source owns while a handle
    // derived from it is still alive, then read through that handle.
    let (handle, expected) = {
        let src = open()?;
        let al = Aliases::register(&client, src.mappings()?).expect("zero copy");
        let w = src.expert_packed(&b13, 3)?;
        let expected = w.codes()[..4096].to_vec();
        let h = al.slice(w.codes()).expect("alias");
        (h, expected)
        // `src`, `al` — and every mapping they own — are dropped HERE. The only
        // thing left holding those pages is the Arc inside cubecl's storage
        // entry.
    };
    // Churn the address space so a stale mapping would be likely to show up as
    // something other than the bytes we want.
    let _noise: Vec<Vec<u8>> = (0..64).map(|i| vec![(i % 251) as u8; 1 << 20]).collect();
    let after = client.read_one(handle.clone()).unwrap();
    let survived = after[..4096] == expected[..];
    println!();
    println!("  mapping dropped, handle still live -> bytes intact : {survived}");

    let ok = bitmatch && worst <= 1e-4 && survived;
    if !ok {
        println!("FAIL");
        std::process::exit(1);
    }
    println!(
        "PASS — aliased weights are byte-identical to copied ones, correct against f64, \
         and survive their source being dropped ({gbps_alias:.1} vs {gbps_copy:.1} GB/s)"
    );
    Ok(())
}
