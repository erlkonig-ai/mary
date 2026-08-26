//! `graph_patch_probe` — can one launch inside a captured graph have its
//! SCALARS and its GEOMETRY rewritten, and does the replay then compute the
//! rewritten thing?
//!
//! ## Why this probe exists
//!
//! Cross-step replay of the decode region needs exactly two things: the region's
//! pointers must not move between steps, and the ~4% of its launches that bake a
//! per-step VALUE into their arguments must be able to take a new one without
//! re-capturing. This answers the second, on a kernel small enough that the
//! answer is unambiguous, before any of it is pointed at the layer loop.
//!
//! It is a CORRECTNESS probe and reports no time. Three questions:
//!
//! 1. Does a capture record one node per launch, in launch order, so that
//!    "launch 1 of this region" names a node? (`graph_launch_count`.)
//! 2. Is a scalar findable in the packed argument blob a launch was recorded
//!    with -- uniquely, by its value, rather than by re-deriving cubecl's
//!    packing order by hand? (`graph_launch_params`.)
//! 3. Does rewriting that slot, or the cube count, change what a REPLAY
//!    computes -- and change nothing else? (`graph_patch_launch`.)
//!
//! Run: `graph_patch_probe [n]`

use anyhow::Result;
use cubecl::prelude::*;
use cubecl::server::{GraphLaunchPatch, Handle};

type Rt = cubecl::cuda::CudaRuntime;

/// One thread per element: a ramp from `base`, scaled. Two runtime scalars of
/// different types on purpose -- cubecl packs scalars by type, and a probe that
/// only ever had one would not notice the packing.
#[cube(launch_unchecked)]
fn ramp_kernel(out: &mut Array<f32>, base: usize, scale: f32) {
    if ABSOLUTE_POS < out.len() {
        out[ABSOLUTE_POS] = f32::cast_from(ABSOLUTE_POS + base) * scale;
    }
}

const CUBE: u32 = 64;

fn launch(client: &ComputeClient<Rt>, out: &Handle, n: usize, cubes: u32, base: usize, scale: f32) {
    unsafe {
        ramp_kernel::launch_unchecked::<Rt>(
            client,
            CubeCount::new_1d(cubes),
            CubeDim::new_1d(CUBE),
            ArrayArg::from_raw_parts(out.clone(), n),
            base,
            scale,
        );
    }
}

fn read(client: &ComputeClient<Rt>, h: &Handle, n: usize) -> Vec<f32> {
    let bytes = client.read_one(h.clone()).expect("device readback");
    bytes
        .chunks_exact(4)
        .take(n)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

/// Every slot of `info` whose low 32 bits equal `v`, and every slot whose high
/// 32 bits do. A 32-bit scalar rides inside a 64-bit word and which half is
/// cubecl's business, so the search reports both and the caller insists on
/// exactly one hit.
fn find_u32(info: &[u64], v: u32) -> Vec<(usize, bool)> {
    let mut hits = Vec::new();
    for (i, w) in info.iter().enumerate() {
        if (*w & 0xffff_ffff) as u32 == v {
            hits.push((i, false));
        }
        if (*w >> 32) as u32 == v {
            hits.push((i, true));
        }
    }
    hits
}

fn put_u32(info: &mut [u64], slot: usize, high: bool, v: u32) {
    if high {
        info[slot] = (info[slot] & 0xffff_ffff) | ((v as u64) << 32);
    } else {
        info[slot] = (info[slot] & !0xffff_ffff) | (v as u64);
    }
}

fn main() -> Result<()> {
    let n: usize = std::env::args()
        .nth(1)
        .and_then(|s| s.parse().ok())
        .unwrap_or(256);
    let cubes = (n as u32).div_ceil(CUBE);

    let device = Default::default();
    let client = Rt::client(&device);
    if !client.graph_capture_supported() {
        anyhow::bail!("this backend cannot capture");
    }

    let out = client.empty(n * 4);

    // WARM: compile the kernel and reserve the pages OUTSIDE the capture.
    for _ in 0..4 {
        launch(&client, &out, n, cubes, 0, 1.0);
    }
    let _ = read(&client, &out, n);

    // ---- capture three launches with distinguishable scalars ----
    client.flush();
    client.graph_capture_begin();
    let first = client.graph_capture_launch_count();
    launch(&client, &out, n, cubes, 1_000, 1.0);
    launch(&client, &out, n, cubes, 2_000, 1.0);
    launch(&client, &out, n, cubes, 3_000, 1.0);
    let after = client.graph_capture_launch_count();
    let g = client.graph_capture_end();

    println!("launch indices {first}..{after}, nodes {}", client.graph_node_count(g));
    assert_eq!(after - first, 3, "three launches must record three launches");
    assert_eq!(
        client.graph_launch_count(g),
        3,
        "the closed graph must hold the launches the open capture counted"
    );

    // ---- 1. the region replays what it recorded ----
    client.graph_replay(g);
    let v = read(&client, &out, n);
    let want_last = 3_000.0f32;
    assert_eq!(v[0], want_last, "the LAST launch wins: got {}", v[0]);
    assert_eq!(v[n - 1], want_last + (n - 1) as f32);
    println!("replay: out[0]={} out[{}]={} (as captured)", v[0], n - 1, v[n - 1]);

    // ---- 2. find each launch's `base` in its packed blob ----
    let mut slots = Vec::new();
    for (k, want) in [1_000u32, 2_000, 3_000].iter().enumerate() {
        let p = client.graph_launch_params(g, first + k);
        assert!(
            p.info_is_grid_constant,
            "launch {k} passes its scalars in a device buffer; a parameter rewrite cannot reach them"
        );
        let hits = find_u32(&p.info, *want);
        println!(
            "launch {k}: grid {:?} block {:?} info {} words, ptrs {:?}, `base`={want} at {hits:?}",
            p.grid,
            p.block,
            p.info.len(),
            p.ptrs
        );
        assert_eq!(
            hits.len(),
            1,
            "`base` must be locatable in the blob by its value, and it matched {} slots",
            hits.len()
        );
        slots.push((hits[0], p.info.clone()));
    }

    // ---- 3. rewrite the LAST launch's `base` and replay ----
    let ((slot, high), info) = slots[2].clone();
    let mut patched = info.clone();
    put_u32(&mut patched, slot, high, 7_777);
    client.graph_patch_launch(
        g,
        first + 2,
        GraphLaunchPatch {
            info: Some(patched),
            ..Default::default()
        },
    );
    client.graph_replay(g);
    let v = read(&client, &out, n);
    assert_eq!(v[0], 7_777.0, "the rewritten scalar must be what the replay uses: got {}", v[0]);
    assert_eq!(v[n - 1], 7_777.0 + (n - 1) as f32);
    println!("after scalar patch: out[0]={} out[{}]={}", v[0], n - 1, v[n - 1]);

    // ---- 4. rewrite the GEOMETRY and replay ----
    //
    // Half the cubes: the tail of `out` keeps whatever the previous replay left
    // there, which is what makes a grid change VISIBLE rather than merely
    // accepted. The kernel's own bound is what stops the short grid from
    // reading past the array.
    let half = cubes.div_ceil(2);
    client.graph_patch_launch(
        g,
        first + 2,
        GraphLaunchPatch {
            grid: Some((half, 1, 1)),
            ..Default::default()
        },
    );
    // Give the tail a value only the SECOND launch could have written, so the
    // check below cannot pass by accident.
    client.graph_patch_launch(
        g,
        first + 1,
        GraphLaunchPatch {
            info: Some({
                let ((s, h), i) = slots[1].clone();
                let mut i = i;
                put_u32(&mut i, s, h, 500_000);
                i
            }),
            ..Default::default()
        },
    );
    client.graph_replay(g);
    let v = read(&client, &out, n);
    let covered = (half * CUBE) as usize;
    assert_eq!(v[0], 7_777.0, "the short grid still writes the head");
    assert_eq!(
        v[n - 1],
        500_000.0 + (n - 1) as f32,
        "past the short grid the previous launch's value must survive: got {}",
        v[n - 1]
    );
    println!(
        "after grid patch to {half} cubes ({covered} of {n} elements): out[0]={} out[{}]={}",
        v[0],
        n - 1,
        v[n - 1]
    );

    // ---- 5. a patch that restores the captured values restores the answer ----
    for k in 0..3 {
        let (_, info) = slots[k].clone();
        client.graph_patch_launch(
            g,
            first + k,
            GraphLaunchPatch {
                grid: Some((cubes, 1, 1)),
                info: Some(info),
                ..Default::default()
            },
        );
    }
    client.graph_replay(g);
    let v = read(&client, &out, n);
    assert_eq!(v[0], want_last, "restoring the captured params restores the captured answer");
    assert_eq!(v[n - 1], want_last + (n - 1) as f32);
    println!("after restore: out[0]={} out[{}]={}", v[0], n - 1, v[n - 1]);

    client.graph_destroy(g);
    println!("OK: launches are indexable, scalars are locatable, and both scalars and geometry rewrite");
    Ok(())
}
