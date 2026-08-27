//! `tp_allreduce_probe` — what a token's worth of collectives costs THROUGH THE
//! PATH THE MODEL WILL USE, rather than through a hand-written `.cu`.
//!
//! # Why this exists when the interconnect is already measured
//!
//! `scripts/interconnect_probe.sh` measures NCCL with a purpose-built CUDA
//! program: one all-reduce, one `cudaStreamSynchronize`, repeat. That is the
//! right way to get a LATENCY, and 29.56 us at 4096 f32 is the number
//! `mary::models::inkling::tp` budgets against. It is not the number the
//! forward will pay, for two reasons, and both move the answer:
//!
//! 1. **The forward does not sync between collectives.** It issues 86 of them a
//!    token onto a stream and syncs once, at the end, if at all. A sync-per-
//!    collective figure multiplied by 86 assumes none of the per-message
//!    overhead overlaps. Some of it does.
//! 2. **The forward goes through cubecl**, which brackets every collective with
//!    two `cudaStreamWaitEvent`s and a submission-queue flush. That is host work
//!    the `.cu` never does, and host work is precisely the term the projection
//!    says becomes binding after the split.
//!
//! So this measures the same wire through `ComputeClient::all_reduce`, and
//! separates the host cost from the device cost, because they land in different
//! places in the projection: device cost adds to the ~49 ms of halved
//! streaming, host cost adds to the ~52 ms of doubled enqueue.
//!
//! # The arms, and what each is per
//!
//! Interleaved per rep, first `COLD` reps discarded, median and full series
//! reported. Both ranks barrier between arms so neither is timing the other's
//! arrival.
//!
//! * **`one`** — one all-reduce of `[1, 4096]` f32 followed by a blocking
//!   read-back. Per COLLECTIVE. This is the arm comparable to
//!   `interconnect_probe.sh`'s 29.56 us, and it exists so that a difference
//!   between the two can be attributed to cubecl rather than to the wire.
//! * **`token`** — `COLLECTIVES` all-reduces issued back to back, then one
//!   blocking read-back. Reported twice:
//!   - `host`, the wall time of the ISSUING LOOP alone, per token. This is what
//!     the collectives cost the CPU that is already the suspected bottleneck.
//!   - `e2e`, issuing plus the read-back, per token. This is the whole
//!     collective term of a decode step.
//!
//! `COLLECTIVES` defaults to 84, which is `tp::collectives_per_token(42, 0)`:
//! two per layer and nothing at the ends. It was 86 while the design expected
//! to cut the embedding and the unembedding as well; neither is cut, because
//! neither sits under a reduce -- see `tp::collectives_per_token`.
//!
//! # What a result means
//!
//! The design budgets 2.48 ms a token (84 x 29.56 us). If `token e2e` comes in
//! at or under that, the budget is confirmed through the real path and the
//! 19:1 trade stands as costed. If `token host` is a large fraction of it, the
//! collectives are competing with the enqueue term rather than hiding behind
//! the device, and the 1.5-1.9x projection's lower end gets likelier.
//!
//! **A run with `NCCL_DEBUG=INFO` is worth more than this probe's numbers.**
//! The socket path and the RDMA path differ by 25x on these boxes, which
//! inverts the decision, and only NCCL's own banner says which one it chose.
//! Grep it for `NET/IB` against `NET/Socket`.
//!
//! Run, rank 0 first:
//!
//! ```text
//!   # on rank 0's box (it BINDS the address)
//!   NCCL_SOCKET_IFNAME=<fast-iface> INK_TP=0:2 tp_allreduce_probe 0.0.0.0:7899
//!   # on rank 1's box
//!   NCCL_SOCKET_IFNAME=<fast-iface> INK_TP=1:2 tp_allreduce_probe <rank0-ip>:7899
//! ```

use anyhow::Result;
use cubecl::prelude::*;
use mary::models::inkling::tp::{Tp, collectives_per_token};
use mary::models::inkling::tpcomm::{Group, transport_note};

type Rt = cubecl::cuda::CudaRuntime;

/// The residual stream's width: every one of the 86 is `[1, hidden]`.
const WIDTH: usize = 4096;
const COLD: usize = 2;

fn median(v: &mut [f64]) -> f64 {
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    v[v.len() / 2]
}

fn report(label: &str, unit: &str, series: &[f64]) {
    let mut w = series.to_vec();
    let med = median(&mut w);
    println!(
        "  {label:<12} median {med:8.3} {unit}   min {:8.3}  max {:8.3}  spread {:6.3}",
        w[0],
        w[w.len() - 1],
        w[w.len() - 1] - w[0]
    );
    let each: Vec<String> = series.iter().map(|x| format!("{x:.3}")).collect();
    println!("  {:<12} series [{}]", "", each.join(", "));
}

fn main() -> Result<()> {
    let mut a = std::env::args().skip(1);
    let addr = a
        .next()
        .ok_or_else(|| anyhow::anyhow!("usage: tp_allreduce_probe <rendezvous addr> [reps] [n]"))?;
    let reps: usize = a.next().and_then(|s| s.parse().ok()).unwrap_or(9);
    let per_token: usize = a
        .next()
        .and_then(|s| s.parse().ok())
        .unwrap_or(collectives_per_token(42, 0));

    let tp = Tp::from_env()?;
    anyhow::ensure!(
        tp.is_split(),
        "INK_TP=rank:world with world > 1 is required; a group of one measures nothing"
    );

    let device = Default::default();
    let client = Rt::client(&device);

    println!("tp_allreduce_probe: rank {} of {}", tp.rank(), tp.world());
    println!("  rendezvous  {addr}");
    println!("  payload     [1, {WIDTH}] f32 = {} B", WIDTH * 4);
    println!("  per token   {per_token} collectives");
    println!("  {}", transport_note());

    let t0 = std::time::Instant::now();
    let mut group = Group::form(tp, client.clone(), &addr)?;
    println!("  formed in   {:.1} ms", t0.elapsed().as_secs_f64() * 1e3);

    let t0 = std::time::Instant::now();
    group.warm()?;
    println!(
        "  warmed in   {:.1} ms  (communicator built, sum verified)",
        t0.elapsed().as_secs_f64() * 1e3
    );

    // One buffer, reused. Allocating inside the timed region would measure the
    // memory pool, and the forward reduces a residual that is already there.
    let payload: Vec<u8> = (0..WIDTH)
        .flat_map(|i| ((i % 17) as f32 * 0.25 + (tp.rank() + 1) as f32).to_le_bytes())
        .collect();

    let mut one = Vec::new();
    let mut tok_host = Vec::new();
    let mut tok_e2e = Vec::new();

    for rep in 0..reps {
        // ---- arm `one` ------------------------------------------------
        group.barrier()?;
        let h = client.create_from_slice(&payload);
        let t = std::time::Instant::now();
        group.all_reduce_f32(&h);
        let got = client
            .read_one(h)
            .map_err(|e| anyhow::anyhow!("read-back after one collective: {e:?}"))?;
        let us = t.elapsed().as_secs_f64() * 1e6;

        // Correctness, on the arm where it is EXACT. Element 0 is `rank + 1` on
        // every rank, so one sum must be `world * (world + 1) / 2`. The failure
        // this catches is a group that formed but did not pair -- two ranks each
        // believing they are rank 0 reduce against themselves and return their
        // own value, which is finite, plausible, and would go on to generate
        // fluent text. Checked every rep, not once, because a communicator that
        // degrades mid-run degrades silently too.
        let v0 = f32::from_le_bytes([got[0], got[1], got[2], got[3]]);
        let want = (tp.world() * (tp.world() + 1) / 2) as f32;
        anyhow::ensure!(
            (v0 - want).abs() < 1e-4,
            "one all-reduce of element 0 gave {v0}, not {want}: the ranks are not summing \
             each other. {}",
            transport_note()
        );

        // ---- arm `token` ----------------------------------------------
        group.barrier()?;
        let h = client.create_from_slice(&payload);
        let t = std::time::Instant::now();
        for _ in 0..per_token {
            group.all_reduce_f32(&h);
        }
        let host_ms = t.elapsed().as_secs_f64() * 1e3;
        let got = client
            .read_one(h)
            .map_err(|e| anyhow::anyhow!("read-back after a token of collectives: {e:?}"))?;
        let e2e_ms = t.elapsed().as_secs_f64() * 1e3;

        // No value check on this arm, deliberately. `per_token` IN-PLACE sums
        // multiply element 0 by `world` each time, so it is `world^86` and has
        // been +inf since the twentieth collective. The arm is a timing arm;
        // `one` above is where correctness is established, exactly, every rep.
        let _ = &got;

        if rep >= COLD {
            one.push(us);
            tok_host.push(host_ms);
            tok_e2e.push(e2e_ms);
        }
        println!(
            "  rep {rep:<2} one {us:8.2} us   token host {host_ms:7.3} ms  e2e {e2e_ms:7.3} ms{}",
            if rep < COLD {
                "   (cold, discarded)"
            } else {
                ""
            }
        );
    }

    println!("\n  --- {} reps, {COLD} cold discarded ---", reps - COLD);
    report("one", "us", &one);
    report("token host", "ms", &tok_host);
    report("token e2e", "ms", &tok_e2e);

    let mut o = one.clone();
    let mut e = tok_e2e.clone();
    let (m_one, m_e2e) = (median(&mut o), median(&mut e));
    let mut hh = tok_host.clone();
    let m_host = median(&mut hh);
    println!("\n  per collective, synced      {m_one:8.2} us   (interconnect_probe.sh: 29.56 us)");
    println!(
        "  per collective, pipelined   {:8.2} us   ({m_e2e:.3} ms / {per_token})",
        m_e2e * 1e3 / per_token as f64
    );
    println!(
        "  host share of a token       {:8.1} %   ({m_host:.3} of {m_e2e:.3} ms)",
        100.0 * m_host / m_e2e
    );
    println!(
        "  tp.rs budgets               {:8.3} ms a token at 29.56 us each",
        mary::models::inkling::tp::collective_ms_per_token(42, 0, 29.56)
    );
    println!("  measured here               {m_e2e:8.3} ms a token");
    println!(
        "\n  Against the 105.2 ms/token PP2 baseline that is {:.2}% of a step.",
        100.0 * m_e2e / 105.2
    );
    println!("  {}", transport_note());
    println!("  Re-run with NCCL_DEBUG=INFO and grep NET/IB vs NET/Socket: the two");
    println!("  differ by 25x on these boxes and only the banner says which you got.");

    group.barrier()?;
    Ok(())
}
