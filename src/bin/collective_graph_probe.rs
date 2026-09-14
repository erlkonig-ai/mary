//! `collective_graph_probe` — does a cubecl `all_reduce` fork/join CAPTURE into
//! a CUDA graph, and does a REPLAY of that graph run it?
//!
//! # The question, and why it needs a probe
//!
//! `inkling_forward` used to refuse `INK_GRAPH_LANE=1` under a tensor-parallel
//! group on the reasoning that the NCCL collective is issued on cubecl's separate
//! `comm_stream`, outside the launch bookkeeping, and is therefore "very likely
//! not in the graph" — so a replayed step would silently skip the cross-node
//! reduction and each rank would carry its own partial sum forward. That is the
//! fluent-and-wrong failure, and a refusal was the only safe reading until
//! someone measured it.
//!
//! What the cubecl source actually does (`cubecl-cuda/src/compute/server.rs`,
//! `all_reduce` and `sync_collective`):
//!
//! ```text
//!   Fence::new(compute).wait_async(comm_stream)   // fork:  comm waits on compute
//!   ncclAllReduce(..., comm_stream)               // the collective, on comm
//!   Fence::new(comm_stream).wait_async(compute)   // join:  compute waits on comm
//! ```
//!
//! Under stream capture that is the canonical CUDA fork/join: an event recorded
//! on a capturing stream and waited on by another stream pulls the second
//! stream INTO the capture, and everything issued on it until the join becomes
//! nodes of the same graph. An unjoined fork fails `cuStreamEndCapture` with
//! `CUDA_ERROR_STREAM_CAPTURE_UNJOINED`, which `graph_capture_end` turns into
//! an assertion. So the reasoning that produced the refusal has no path to the
//! failure it feared: the collective is either IN the graph or the capture does
//! not close. This probe is the measurement that replaces the reasoning.
//!
//! # What it does — ONE GPU, world = 1
//!
//! One process, a world-of-one NCCL communicator installed through the same
//! `set_external_comm` seam `tpcomm::Group::form` uses, so `all_reduce` is the
//! identity and needs no peer. Then a region of three device operations:
//!
//! ```text
//!   add_k(a, 1)                        a += 1        cubecl launch, compute stream
//!   all_reduce(a -> b) ; sync           b  = a        NCCL, comm stream, fork/join
//!   add_k(b, 10)                        b += 10       cubecl launch, compute stream
//! ```
//!
//! run once EAGERLY (a = 2, b = 12), then CAPTURED into a graph and REPLAYED
//! `N` times. The three outcomes are separated by `b` alone:
//!
//! ```text
//!   b = 12 + N       the collective is in the graph and replays in order
//!   b = 12 + 10 N    the collective is NOT in the graph: b never sees a again
//!   b =  2 + 10 N    the collective ran ONCE, eagerly, while the capture was
//!                    open, and was not recorded
//! ```
//!
//! With `N = 5` those are 17, 62 and 52, and the `a = 2 + N` line is the
//! control that the graph replayed at all. The node census
//! (`graph_node_kinds`) says what the collective became in the graph — with a
//! world of one NCCL may reduce `a -> b` to a `cudaMemcpyAsync` (one MEMCPY
//! node) rather than a kernel, and an IN-PLACE world-of-one reduce may become
//! nothing at all; both are reported, neither is the verdict. The verdict is
//! `b`.
//!
//! # Measured, 2026-08-30, one GB10 (spark2), NCCL 2.31.2, THREAD_LOCAL mode
//!
//! ```text
//!   captured graph  : 3 nodes -- kernel 2, memcpy 1; 2 of them cubecl launches
//!   after 5 replays : a = 7, b = 17          -> IN THE GRAPH
//!   in-place graph  : 2 nodes -- kernel 2    (world-of-one in-place: NCCL issues nothing)
//! ```
//!
//! The world-of-one `a -> b` reduce is NCCL's single-rank `cudaMemcpyAsync`,
//! and it landed in the graph as the one memcpy node, between the two launches,
//! through the fork/join alone. No `INK_GRAPH_CAPTURE_MODE=relaxed` needed.
//!
//! # What it does NOT answer
//!
//! Whether NCCL's real two-rank kernel, with its network proxy, captures and
//! replays. A world of one never launches that kernel. This is the cheap half;
//! the two-rank half is `INK_GRAPH_LANE=1 INK_GRAPH_CARRY=1 INK_TP=r:2` on the
//! boxes with the token stream compared against an un-graphed run.
//!
//! # Run
//!
//! ```text
//!   NCCL_DEBUG=INFO collective_graph_probe [N]
//!   INK_GRAPH_CAPTURE_MODE=relaxed collective_graph_probe   # if THREAD_LOCAL invalidates
//! ```

use anyhow::Result;
use cubecl::ir::{ElemType, FloatKind};
use cubecl::prelude::*;
use cubecl::server::{Handle, ReduceOperation};

type Rt = cubecl::cuda::CudaRuntime;

const N_ELEMS: usize = 4096;

#[cube(launch_unchecked)]
fn add_k(x: &mut Array<f32>, n: u32, k: f32) {
    let i = ABSOLUTE_POS as u32;
    if i < n {
        x[i as usize] = x[i as usize] + k;
    }
}

fn floats(v: f32) -> Vec<u8> {
    (0..N_ELEMS).flat_map(|_| v.to_le_bytes()).collect()
}

fn read_f32(client: &ComputeClient<Rt>, h: &Handle) -> Result<Vec<f32>> {
    let bytes = client
        .read_one(h.clone())
        .map_err(|e| anyhow::anyhow!("read-back: {e:?}"))?;
    Ok(bytes
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect())
}

/// Every element equal, or the buffer is reported whole.
fn uniform(name: &str, v: &[f32]) -> Result<f32> {
    let first = v[0];
    anyhow::ensure!(
        v.iter().all(|&x| x == first),
        "{name} is not uniform: first {first}, min {}, max {}",
        v.iter().cloned().fold(f32::INFINITY, f32::min),
        v.iter().cloned().fold(f32::NEG_INFINITY, f32::max)
    );
    Ok(first)
}

fn launch_add(client: &ComputeClient<Rt>, h: &Handle, k: f32) {
    unsafe {
        add_k::launch_unchecked::<Rt>(
            client,
            CubeCount::Static(N_ELEMS.div_ceil(256) as u32, 1, 1),
            CubeDim::new_1d(256),
            ArrayArg::from_raw_parts(h.clone(), N_ELEMS),
            N_ELEMS as u32,
            k,
        )
    };
}

fn reduce(
    client: &ComputeClient<Rt>,
    src: &Handle,
    dst: &Handle,
    key: &[cubecl::device::DeviceId],
) {
    // EXACTLY `tpcomm::Group::all_reduce_f32`: the collective, then the return
    // fence. Both go through the client so they are ordered against the
    // launches on the same stream.
    let mut c = client.clone();
    c.all_reduce(
        src.clone(),
        dst.clone(),
        ElemType::Float(FloatKind::F32),
        key.to_vec(),
        ReduceOperation::Sum,
    );
    c.sync_collective();
}

fn kinds_named(kinds: &[(u32, usize)]) -> String {
    let named: Vec<String> = kinds
        .iter()
        .map(|(k, c)| {
            let n = match k {
                0 => "kernel",
                1 => "memcpy",
                2 => "memset",
                3 => "host",
                4 => "child-graph",
                5 => "empty",
                6 => "wait-event",
                7 => "event-record",
                10 => "mem-alloc",
                11 => "mem-free",
                _ => "other",
            };
            format!("{n}({k}) {c}")
        })
        .collect();
    named.join(", ")
}

/// One captured region: `add_k(a,1)`, reduce `a -> dst`, `add_k(dst,10)`.
/// Returns `(graph id, node kinds, cubecl launch count, capture status before
/// end)`.
fn capture_region(
    client: &ComputeClient<Rt>,
    a: &Handle,
    dst: &Handle,
    key: &[cubecl::device::DeviceId],
) -> (u64, Vec<(u32, usize)>, usize, u32) {
    // The arena is what `graph_capture_begin` asserts on by default; the
    // region allocates nothing, so it is a formality here, but it is the same
    // formality `inkling_forward` observes.
    client.graph_arena_begin();
    let _ = client.flush();
    client.graph_capture_begin();
    launch_add(client, a, 1.0);
    reduce(client, a, dst, key);
    launch_add(client, dst, 10.0);
    // 1 = still capturing, 2 = INVALIDATED (something the capture could not
    // hold was issued -- the answer would then be "refuses", and it is a good
    // one). Read BEFORE end, because end asserts on a null graph.
    let status = client.graph_capture_status();
    client.graph_arena_end();
    if status != 1 {
        println!(
            "  capture status before end: {status} (1 = active, 2 = invalidated) -- the \
             collective INVALIDATED the capture; graph_capture_end will assert next"
        );
    }
    let g = client.graph_capture_end();
    let kinds = client.graph_node_kinds(g);
    let launches = client.graph_launch_count(g);
    (g, kinds, launches, status)
}

fn main() -> Result<()> {
    let n: usize = std::env::args()
        .nth(1)
        .and_then(|s| s.parse().ok())
        .unwrap_or(5);
    let mode =
        std::env::var("INK_GRAPH_CAPTURE_MODE").unwrap_or_else(|_| "thread-local (default)".into());
    println!("collective_graph_probe: world 1, one GPU, {n} replays, capture mode {mode}");

    let device = Default::default();
    let client = Rt::client(&device);
    anyhow::ensure!(
        client.graph_capture_supported(),
        "this client does not support graph capture"
    );

    // The group, formed the way `tpcomm::Group::form` forms it -- a minted id
    // installed as an external comm -- so `comm_init` takes the same branch it
    // takes on the boxes. Rank 0 of a world of 1: `ncclCommInitRank` returns
    // without a peer and every collective is the identity.
    let id = cubecl::cuda::collective::mint_unique_id().map_err(|e| anyhow::anyhow!("{e}"))?;
    cubecl::cuda::collective::set_external_comm(id, 0, 1);
    let key = vec![cubecl::device::DeviceId {
        type_id: 0,
        index_id: 0,
    }];

    // ---- 0. the communicator, built and proven OUTSIDE any capture --------
    let t0 = std::time::Instant::now();
    let a = client.create_from_slice(&floats(1.0));
    let b = client.create_from_slice(&floats(0.0));
    reduce(&client, &a, &b, &key);
    let got = uniform("warm b", &read_f32(&client, &b)?)?;
    anyhow::ensure!(
        got == 1.0,
        "world-of-one all_reduce(a -> b) gave {got}, not 1.0"
    );
    println!(
        "  communicator    : built and verified in {:.1} ms (a -> b copied 1.0)",
        t0.elapsed().as_secs_f64() * 1e3
    );

    // ---- 1. the region, EAGERLY: a = 2, b = 12 ------------------------------
    let a = client.create_from_slice(&floats(1.0));
    let b = client.create_from_slice(&floats(0.0));
    client.graph_arena_begin();
    launch_add(&client, &a, 1.0);
    reduce(&client, &a, &b, &key);
    launch_add(&client, &b, 10.0);
    client.graph_arena_end();
    let (ea, eb) = (
        uniform("eager a", &read_f32(&client, &a)?)?,
        uniform("eager b", &read_f32(&client, &b)?)?,
    );
    anyhow::ensure!(
        (ea, eb) == (2.0, 12.0),
        "the eager region gave a = {ea}, b = {eb}; expected 2, 12"
    );
    println!("  eager region    : a = {ea}, b = {eb}   (add 1; reduce a -> b; add 10)");

    // ---- 2. the region, CAPTURED, then replayed N times ---------------------
    let (g, kinds, launches, status) = capture_region(&client, &a, &b, &key);
    let total: usize = kinds.iter().map(|(_, c)| c).sum();
    println!(
        "  captured graph  : {total} nodes -- {}; {launches} of them cubecl launches; status \
         before end {status}",
        kinds_named(&kinds)
    );
    // Nothing ran during the capture: a and b must still be 2 and 12.
    let (ca, cb) = (
        uniform("post-capture a", &read_f32(&client, &a)?)?,
        uniform("post-capture b", &read_f32(&client, &b)?)?,
    );
    println!(
        "  after capture   : a = {ca}, b = {cb}   (expected 2, 12: a capture records, it does not run)"
    );
    for _ in 0..n {
        client.graph_replay(g);
    }
    let (ra, rb) = (
        uniform("replayed a", &read_f32(&client, &a)?)?,
        uniform("replayed b", &read_f32(&client, &b)?)?,
    );
    let want_a = 2.0 + n as f32;
    let in_graph = 12.0 + n as f32;
    let skipped = 12.0 + 10.0 * n as f32;
    let once_eager = 2.0 + 10.0 * n as f32;
    println!("  after {n} replays : a = {ra}, b = {rb}");
    println!(
        "  expected        : a = {want_a}; b = {in_graph} if the collective is IN the graph, \
         {skipped} if it is NOT, {once_eager} if it ran once eagerly during the capture"
    );
    anyhow::ensure!(
        ra == want_a,
        "a = {ra}, not {want_a}: the graph did not replay the launches"
    );
    let verdict = if rb == in_graph {
        "IN THE GRAPH: the fork/join collective captured and replayed in order, every replay"
    } else if rb == skipped {
        "NOT IN THE GRAPH: the replay skipped the collective (the refusal was right)"
    } else if rb == once_eager {
        "RAN ONCE, EAGERLY: the collective executed during the capture and was not recorded"
    } else {
        "UNCLASSIFIED: b matches none of the three outcomes"
    };
    println!("  VERDICT         : {verdict}");

    // ---- 3. the in-place variant, census only -------------------------------
    //
    // The forward reduces IN PLACE (`sendbuff == recvbuff`). A world of one has
    // nothing to do for that and NCCL may issue nothing at all, in which case
    // the graph holds only the two launches. That is a fact about world = 1,
    // not about the mechanism, and it is printed so nobody reads a two-node
    // census against the wrong expectation.
    let c = client.create_from_slice(&floats(1.0));
    client.graph_arena_begin();
    let _ = client.flush();
    client.graph_capture_begin();
    launch_add(&client, &c, 1.0);
    reduce(&client, &c, &c, &key);
    launch_add(&client, &c, 10.0);
    let st = client.graph_capture_status();
    client.graph_arena_end();
    let g2 = client.graph_capture_end();
    let kinds2 = client.graph_node_kinds(g2);
    let total2: usize = kinds2.iter().map(|(_, c)| c).sum();
    for _ in 0..n {
        client.graph_replay(g2);
    }
    let rc = uniform("in-place c", &read_f32(&client, &c)?)?;
    println!(
        "  in-place graph  : {total2} nodes -- {}; status {st}; after {n} replays c = {rc} \
         (expected {}: each replay adds 11 whatever the identity reduce became)",
        kinds_named(&kinds2),
        1.0 + 11.0 * n as f32
    );
    anyhow::ensure!(rb == in_graph, "{verdict}");
    Ok(())
}
