//! The collective that puts [`super::tp`]'s slices back together, and the
//! rendezvous that forms the group across two BOXES.
//!
//! [`super::tp`] is arithmetic: which rows of which tensor a rank owns. It can
//! be reasoned about, tested and got wrong entirely on the host. This module is
//! the other half — the part that has to touch the network — and it exists
//! separately for exactly that reason: `tp`'s twelve tests run anywhere, and
//! nothing here runs without two GPUs and a wire between them.
//!
//! # The property the whole design rests on, which is not the latency
//!
//! The quoted 29.56 us per all-reduce is necessary but it is not what decides
//! this. What decides it is that the collective is **stream-ordered and never
//! touches the host**. cubecl's CUDA server implements `all_reduce` as:
//!
//! ```text
//!   Fence::new(compute_stream).wait_async(comm_stream)   // comm waits for data
//!   ncclAllReduce(..., comm_stream)                      // launch, do not wait
//!   -- and, on sync_collective --
//!   Fence::new(comm_stream).wait_async(compute_stream)   // compute waits for result
//! ```
//!
//! Two `cudaStreamWaitEvent`s and a launch. **No `cudaStreamSynchronize`, no
//! read-back, nothing that blocks the calling thread on the device.** That is
//! the difference between a within-layer split that pays for itself and one
//! that cannot: the projection in [`super::tp`] has the pass bounded by
//! `max(host enqueue, device streaming)` because "nothing in the layer loop
//! synchronises". A collective implemented the obvious way — sync the compute
//! stream, run NCCL, sync again — would synchronise the layer loop **84 times a
//! token**, and the pass would become `enqueue + device` instead of
//! `max(enqueue, device)`. On the same measured parts that is ~52 + ~49 = ~101
//! ms against a 105.2 ms baseline: a rewrite of the whole forward to buy 4%.
//!
//! So the host-side rule for every caller here: **issue and move on.** Do not
//! read a reduced buffer back to decide anything, do not call `client.sync()`
//! between layers, and do not "just check" an intermediate. Each of those turns
//! the run-ahead off, and the cost does not show up as a wrong number — it
//! shows up as the speedup quietly not being there.
//!
//! # Why the group has to be formed by hand
//!
//! cubecl mints a communicator id into a process-global map and derives each
//! rank from its device index. For one process holding several GPUs that is
//! right and needs no help. For two processes on two boxes it fails twice: each
//! mints its OWN id, so `ncclCommInitRank` never pairs them and **hangs** in
//! the rendezvous rather than returning an error; and each derives rank 0 from
//! its own local device 0, so there is no rank 1. [`Group::form`] does what a
//! launcher would normally do — one rank mints, the 128 bytes cross a socket,
//! every rank installs the same group — through the
//! `cubecl::cuda::collective` seam added for it.
//!
//! The rendezvous socket is used ONCE, at startup, and then only kept for the
//! barrier in [`Group::barrier`]. It is not on the token path; every collective
//! after `form` goes through NCCL on the ConnectX pair.
//!
//! # If `warm` HANGS: the seam has two halves and only one of them is here
//!
//! [`Group::form`] calls `cubecl::cuda::collective::set_external_comm`, which
//! only STORES the group. The half that matters is in cubecl's CUDA server:
//! `comm_init` has to read it back and pass those three numbers to
//! `ncclCommInitRank` instead of deriving them from the local device list. A
//! cubecl that has the setter but not the reader COMPILES AND LINKS, accepts
//! the group, and ignores it -- and then both ranks derive rank 0 from their
//! own local device 0, each mints its own id, and `ncclCommInitRank` waits
//! forever for a peer that is calling itself rank 0 too. Diagnosed 2026-08-27,
//! having cost about forty minutes: the boxes' `cubecl-graph` was one commit
//! short of `c8380a03 "comm_init must USE the external group, not just accept
//! one"`, and there is no compile error and no runtime error to see.
//!
//! The symptom is exact and worth memorising: **both processes stop after
//! printing their `tensor parallel : rank R of W` line, the TCP rendezvous is
//! `ESTAB` (so `form` itself succeeded), the GPUs are at 0% and neither ever
//! reaches `tp group : paired and verified`.**
//!
//! The confirmation takes one environment variable, and it is NOT `NCCL_DEBUG`
//! alone: NCCL writes to `stdout` with libc block buffering, so when stdout is
//! a redirected log its banner sits in a 4 KB buffer that a hung process never
//! flushes, and `NCCL_DEBUG=INFO` looks like it produced nothing at all. Add
//! `NCCL_DEBUG_FILE=/tmp/nccl.%h.%p.log`, which NCCL opens per rank and
//! flushes, then read the `ncclCommInitRank` line on BOTH boxes:
//!
//! ```text
//!   box A   [Rank 0] ncclCommInitRank ... rank 0 nranks 2 ... commId 0x551d98...
//!   box B   [Rank 0] ncclCommInitRank ... rank 0 nranks 2 ... commId 0x8a5a1f...
//!                                          ^^^^^^                    ^^^^^^^^
//!            two rank 0s, two different ids -- the external group was ignored
//! ```
//!
//! A healthy pair reads `rank 0` on one and `rank 1` on the other, with the
//! SAME `commId`. The check [`Group::warm`] makes -- every rank contributes 1.0
//! so the sum must be the world size -- cannot catch this one, because it
//! catches a group that PAIRED WRONGLY and this group never pairs at all.
//!
//! # Sum is the only reduction, and that is not a limitation
//!
//! cubecl exposes `Sum` and `Mean`. Every collective this design needs is a
//! sum:
//!
//! * **after `wo`** and **after the MoE down-projection** — genuine sums of
//!   partial products, which is what a column-then-row split produces.
//! * **the embedding broadcast** — rank 0 holds the table (2.40 GiB of
//!   residency to read one 8 KB row, so it is not replicated), rank 1
//!   contributes zeros, and the sum IS the broadcast. One collective either
//!   way, and it needs no second primitive.
//! * **the sharded-unembed argmax** — see [`Group::all_gather_small`]. A max is
//!   built out of a sum by giving each rank its own slot, which costs one
//!   collective of 4 floats rather than an all-gather of 201024.
//!
//! So no `Max`, no `AllGather`, no `Broadcast`: one primitive, and every use of
//! it is 16 KB or less and therefore latency-bound and the same price.

use anyhow::{Context, Result};
use cubecl::cuda::CudaRuntime;
use cubecl::ir::{ElemType, FloatKind};
use cubecl::prelude::ComputeClient;
use cubecl::server::{Handle, ReduceOperation};
use std::io::{Read, Write};
use std::net::{TcpListener, TcpStream};
use std::time::Duration;

use burn::tensor::Tensor;

use super::tp::Tp;

/// How long a rank waits for its peer at the rendezvous before giving up.
///
/// Generous because the peer is usually still mapping 70-odd GiB of pile when
/// this runs, and mean because the failure it replaces is an unbounded hang
/// inside NCCL that looks exactly like a slow load.
const RENDEZVOUS: Duration = Duration::from_secs(180);

/// The devices this group names.
///
/// Purely a KEY: cubecl hashes the sorted list into a `CommunicationId` and
/// looks the communicator up by it, and once [`Group::form`] has installed an
/// external group the list is not consulted for anything else. Both ranks must
/// pass the SAME list or they build two different keys and never meet, which is
/// why this is a constant and not derived from the local device — the local
/// device is 0 on both boxes.
fn group_key(world: usize) -> Vec<cubecl::device::DeviceId> {
    (0..world as u16)
        .map(|i| cubecl::device::DeviceId {
            type_id: 0,
            index_id: i,
        })
        .collect()
}

/// A formed NCCL group, and the client its collectives are ordered against.
///
/// Holds a clone of the compute client deliberately: the collectives must land
/// on the SAME client the kernels are launched on, or the fences order nothing.
/// [`super::seam::client_of`] exists to make that guarantee available, and this
/// takes its result rather than calling `CudaRuntime::client` again.
pub struct Group {
    tp: Tp,
    client: ComputeClient<CudaRuntime>,
    /// The Burn device paired with `client` when this Group was formed through
    /// [`Group::form_default`]. The lower-level [`Group::form`] predates
    /// Session and deliberately carries no such claim.
    default_device: Option<burn::backend::cuda::CudaDevice>,
    /// The allocator selected before `default_device` created its client.
    /// Session uses this exact value for admission instead of independently
    /// pricing a policy the runtime may no longer be able to adopt.
    allocator: Option<super::pool::AllocatorConfig>,
    key: Vec<cubecl::device::DeviceId>,
    /// The rendezvous sockets, kept for [`Group::barrier`]. A star: rank 0
    /// holds one per peer, every other rank holds one, to rank 0.
    socks: Vec<TcpStream>,
}

impl Group {
    /// Form a group on Burn's default CUDA device.
    ///
    /// This is the serving entry point. It derives the raw cubecl client FROM
    /// a Burn tensor on the same device a [`super::session::Session`] uses, so
    /// the group cannot accidentally be formed on a second client whose
    /// stream fences would not order the model's kernels. Form and
    /// [`Group::warm`] this value before handing it to
    /// [`super::session::Session::load_with_group`].
    pub fn form_default(tp: Tp, addr: &str) -> Result<Self> {
        // Both are process-global and must precede the very first CUDA client.
        // `fatal::arm` is idempotent; allocator selection records the policy in
        // the environment before CubeCL reads it once during runtime init.
        super::fatal::arm();
        let allocator = super::pool::choose_memory_config();
        let device = burn::backend::cuda::CudaDevice::default();
        let probe = Tensor::<super::seam::Bk, 2>::zeros([1, 1], &device);
        let mut group = Self::form(tp, super::seam::client_of(&probe), addr)?;
        group.default_device = Some(device);
        group.allocator = Some(allocator);
        Ok(group)
    }

    /// Form the group: mint on rank 0, ship the id, install it everywhere.
    ///
    /// `addr` is `HOST:PORT`. Rank 0 BINDS it and every other rank CONNECTS to
    /// it, so rank 0's address is the one both sides name — which is also why
    /// rank 0 should be the box whose ConnectX address is stable.
    ///
    /// This does not itself create the NCCL communicator. `ncclCommInitRank` is
    /// collective and would block here; cubecl builds it lazily on the first
    /// collective instead. Call [`Group::warm`] once, at a point where blocking
    /// is free, so the rendezvous does not happen inside the first token.
    ///
    /// This is the lower-level harness API and cannot prove which Burn device
    /// produced `client`. A Session deliberately refuses such a Group; serving
    /// code uses [`Group::form_default`] so device and client travel together.
    pub fn form(tp: Tp, client: ComputeClient<CudaRuntime>, addr: &str) -> Result<Self> {
        anyhow::ensure!(
            tp.is_split(),
            "a group of one needs no rendezvous; check `Tp::is_split` before calling"
        );

        let (id, socks) = if tp.rank() == 0 {
            let id =
                cubecl::cuda::collective::mint_unique_id().map_err(|e| anyhow::anyhow!("{e}"))?;
            let l = TcpListener::bind(addr).with_context(|| format!("binding {addr}"))?;
            // Every peer gets the same 128 bytes. `world - 1` of them, in any
            // order; NCCL sorts out who is who from the rank we each install.
            let mut peers = Vec::new();
            for _ in 1..tp.world() {
                let (mut s, _) = l.accept().with_context(|| format!("accepting on {addr}"))?;
                s.set_nodelay(true).ok();
                s.write_all(&id).context("sending the communicator id")?;
                peers.push(s);
            }
            (id, peers)
        } else {
            let mut s = connect_with_deadline(addr, RENDEZVOUS)?;
            s.set_nodelay(true).ok();
            let mut id = [0u8; 128];
            s.read_exact(&mut id)
                .context("receiving the communicator id")?;
            (id, vec![s])
        };

        cubecl::cuda::collective::set_external_comm(id, tp.rank() as i32, tp.world() as i32);

        Ok(Self {
            tp,
            client,
            default_device: None,
            allocator: None,
            key: group_key(tp.world()),
            socks,
        })
    }

    pub fn tp(&self) -> Tp {
        self.tp
    }

    /// The exact client the communicator was installed on.
    ///
    /// Kept crate-visible rather than making a serving caller carry a second
    /// copy of this fact beside the group. A tensor-parallel Session takes its
    /// client from the Group; rank, communicator and kernel stream therefore
    /// cannot drift independently.
    pub(crate) fn client(&self) -> ComputeClient<CudaRuntime> {
        self.client.clone()
    }

    /// The device whose Burn client this group owns.
    ///
    /// A Session refuses a lower-level Group without this witness rather than
    /// guessing that `CudaDevice::default()` names the same server. The guess
    /// is usually true and, when false, makes every collective's stream fences
    /// order a client other than the one running the model.
    pub(crate) fn default_device(&self) -> Result<burn::backend::cuda::CudaDevice> {
        self.default_device.clone().context(
            "this Group was formed from an arbitrary ComputeClient. A Session needs \
             Group::form_default so its Burn device and collective client are one fact",
        )
    }

    /// The allocator CubeCL observed before this Group created its client.
    pub(crate) fn allocator(&self) -> Result<super::pool::AllocatorConfig> {
        self.allocator.context(
            "this Group was formed without an allocator witness. A Session needs \
             Group::form_default so admission and the CUDA runtime price one policy",
        )
    }

    /// Build the communicator and prove the wire works, somewhere it is safe to
    /// block.
    ///
    /// The first collective pays `ncclCommInitRank`, which is a rendezvous of
    /// its own and takes tens of milliseconds. Paying it inside the first
    /// decoded token does not corrupt anything, but it does put a large
    /// one-off into a measurement that is trying to resolve two, so warm it
    /// here and discard it.
    ///
    /// This is also the one place a `sync` is CORRECT: it is the only way to
    /// turn "NCCL has not failed yet" into "NCCL worked", and a mismatched
    /// group that is going to hang should hang here — at startup, next to the
    /// message that says what it was doing — rather than in the layer loop.
    /// It is also a CORRECTNESS gate, and cheaply: every rank contributes 1.0,
    /// so the sum must be exactly the world size. A group that formed but did
    /// not actually pair — the classic being two ranks that each think they are
    /// rank 0 — reduces a rank against itself and returns 1.0, which is a
    /// perfectly finite number that would go on to produce fluent text. Check
    /// it here, once, where the check costs nothing.
    pub fn warm(&self) -> Result<()> {
        // Said BEFORE the blocking call, because the failure this can hit is a
        // HANG and a hang has no message. A run that stops on this line has the
        // problem the module header describes; a run that stops on the line
        // before it never got here. One `println!` buys that distinction.
        println!(
            "  tp group          : building the communicator (blocks; if this never returns, \
             see the `warm` HANGS section of `tpcomm`)"
        );
        let ones: Vec<u8> = (0..4).flat_map(|_| 1f32.to_le_bytes()).collect();
        let probe = self.client.create_from_slice(&ones);
        self.all_reduce_f32(&probe);
        let got = self
            .client
            .read_one(probe)
            .map_err(|e| anyhow::anyhow!("the warm-up collective never completed: {e:?}"))?;
        let sum = f32::from_le_bytes([got[0], got[1], got[2], got[3]]);
        anyhow::ensure!(
            (sum - self.tp.world() as f32).abs() < 1e-6,
            "the warm-up all-reduce summed to {sum}, not {}: the group did not pair. \
             Every rank contributed 1.0, so {} means this rank reduced against itself \
             (two ranks both believing they are rank 0 is the usual cause). {}",
            self.tp.world(),
            sum,
            transport_note()
        );
        Ok(())
    }

    /// Sum `h` across every rank, IN PLACE.
    ///
    /// In place because the callers are all "this rank computed a partial
    /// residual, make it the whole one", and threading a second buffer through
    /// the forward would mean every caller also decides which of two handles is
    /// live afterwards. NCCL supports `sendbuff == recvbuff` natively.
    ///
    /// Issues and returns. The result is NOT readable on the host when this
    /// returns and must not be read — see the module header.
    pub fn all_reduce_f32(&self, h: &Handle) {
        let mut client = self.client.clone();
        client.all_reduce(
            h.clone(),
            h.clone(),
            ElemType::Float(FloatKind::F32),
            self.key.clone(),
            ReduceOperation::Sum,
        );
        // The return fence. `all_reduce` only makes the COMM stream wait for
        // the data; without this the compute stream reads `h` back while NCCL
        // is still writing it. The name is cubecl's and it is misleading: this
        // is a device-side `cudaStreamWaitEvent`, not a host sync.
        client.sync_collective();
    }

    /// An all-gather of a few floats, built out of the sum.
    ///
    /// `local` is this rank's contribution and lands at slot `rank * stride`;
    /// every other slot is zero, so the sum leaves each rank holding all
    /// `world` contributions side by side. Used for the sharded unembedding,
    /// where the pair is `(best value, local row)` and the winner cannot be
    /// found by summing or averaging.
    ///
    /// Returns the gathered buffer, `world * stride` floats. This one DOES get
    /// read back on the host — it is the argmax, the host needs the token id to
    /// append it, and the step is ending anyway.
    pub fn all_gather_small(&self, local: &[f32], stride: usize) -> Result<Vec<f32>> {
        anyhow::ensure!(
            local.len() == stride,
            "all_gather_small: {} floats into a stride of {stride}",
            local.len()
        );
        let mut staged = vec![0f32; self.tp.world() * stride];
        staged[self.tp.rank() * stride..(self.tp.rank() + 1) * stride].copy_from_slice(local);
        let bytes: Vec<u8> = staged.iter().flat_map(|f| f.to_le_bytes()).collect();
        let h = self.client.create_from_slice(&bytes);
        self.all_reduce_f32(&h);
        let out = self
            .client
            .read_one(h)
            .map_err(|e| anyhow::anyhow!("reading the gathered pairs: {e:?}"))?;
        Ok(out
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect())
    }

    /// Whichever rank holds the largest value, as a global row.
    ///
    /// The tie rule is the shard-boundary version of cubek's ArgMax rule ("the
    /// smallest coordinate in case of equality"): a tie goes to the LOWER rank,
    /// because that rank owns the lower rows. [`super::tp`] has a test pinning
    /// exactly this, including the failure it exists to prevent — a local index
    /// returned unlifted is a real token on rank 0, so forgetting the lift
    /// produces fluent nonsense rather than an error.
    pub fn argmax_across(&self, best: f32, local_row: usize, vocab: usize) -> Result<usize> {
        let pairs = self.all_gather_small(&[best, local_row as f32], 2)?;
        let mut win_rank = 0usize;
        let mut win_val = f32::NEG_INFINITY;
        for r in 0..self.tp.world() {
            let v = pairs[r * 2];
            if v > win_val {
                win_val = v;
                win_rank = r;
            }
        }
        let row = pairs[win_rank * 2 + 1] as usize;
        let off = Tp::new(win_rank, self.tp.world())?
            .unembed_offset(vocab)
            .map_err(|e| anyhow::anyhow!("{e}"))?;
        Ok(off + row)
    }

    /// Host-side barrier on the rendezvous socket.
    ///
    /// Deliberately NOT an NCCL collective: this exists for the points where
    /// the two ranks must agree about something on the HOST — the end of a
    /// benchmark rep, a decision to stop generating — and using a device
    /// collective for those would mean a device sync, which is the one thing
    /// the module header forbids on the token path. Never call it inside the
    /// layer loop.
    pub fn barrier(&mut self) -> Result<()> {
        // Write to every peer BEFORE reading from any, or a star of more than
        // two deadlocks: rank 0 blocked reading peer 1 while peer 2 is blocked
        // reading rank 0. One byte fits any socket buffer, so the write phase
        // cannot block on a peer that has not read yet.
        for s in self.socks.iter_mut() {
            s.write_all(&[0xB1]).context("barrier send")?;
            s.flush().ok();
        }
        for s in self.socks.iter_mut() {
            let mut b = [0u8; 1];
            s.read_exact(&mut b).context("barrier recv")?;
            anyhow::ensure!(b[0] == 0xB1, "barrier got {:#x}, not a barrier byte", b[0]);
        }
        Ok(())
    }

    // ── the host-side rank link ─────────────────────────────────────────────
    //
    // The rendezvous socket was already kept for `barrier`, whose doc says why
    // it exists: "the points where the two ranks must agree about something on
    // the HOST — the end of a benchmark rep, A DECISION TO STOP GENERATING —
    // and using a device collective for those would mean a device sync, which
    // is the one thing the module header forbids on the token path."
    //
    // The one-binary collapse is exactly that use, made continuous. There is no
    // proxy process fanning one input stream out to two rank processes any
    // more; rank 0 owns the Drive loop and rank 1 owns nothing but a Session,
    // so rank 1 has to be TOLD which pass to make. It is told here, on the
    // socket that already exists, in nine bytes.
    //
    // Why this is not a new wire: it carries one small message per model PASS,
    // on the fast fabric, with no JSON, no framing library and no second
    // process in the middle. The thing it replaced carried a framed-stream
    // CONSULT envelope per token through a fan-out proxy and an `ssh` channel
    // — about 26 ms of a measured 82 ms p50 token. And it is STRICTLY MORE
    // honest about lockstep: the proxy mirrored input BYTES and relied on both
    // ranks deriving the same passes from them, while this names the passes.

    /// Name the pass rank 0 is about to make, so every other rank makes it too.
    ///
    /// Write-and-continue: this does NOT wait for the peer. The collective
    /// inside the pass is the synchronisation, and the kernel's socket buffer
    /// absorbs any skew; blocking here would serialise what NCCL already
    /// orders. Only rank 0 may call it.
    pub fn lead(&mut self, pass: &Pass) -> Result<()> {
        anyhow::ensure!(
            self.tp.rank() == 0,
            "only rank 0 leads passes; rank {} must follow",
            self.tp.rank()
        );
        let frame = pass.encode();
        for peer in self.socks.iter_mut() {
            peer.write_all(&frame).context("send the pass command")?;
            peer.flush().ok();
        }
        Ok(())
    }

    /// Block until rank 0 names the next pass. Only a non-zero rank may call it.
    ///
    /// An end-of-stream here is not an error to paper over: it means rank 0's
    /// process is GONE. That is the same signal the deleted framed-stream proxy
    /// used ("a truncated stream is distinguishable from a finished one"),
    /// obtained from TCP for free, and it must terminate this rank rather than
    /// leave it holding a 121 GiB arena and half a communicator.
    pub fn follow(&mut self) -> Result<Pass> {
        anyhow::ensure!(
            self.tp.rank() != 0,
            "rank 0 leads passes; it has nobody to follow"
        );
        let peer = self
            .socks
            .first_mut()
            .context("this rank has no rendezvous socket to rank 0")?;
        let mut header = [0u8; 5];
        match peer.read_exact(&mut header) {
            Ok(()) => {}
            Err(error) if error.kind() == std::io::ErrorKind::UnexpectedEof => {
                anyhow::bail!(
                    "rank 0 closed the rank link without a Finish: its process is gone, so \
                     this rank is stopping rather than blocking in NCCL forever"
                )
            }
            Err(error) => return Err(error).context("read the next pass command"),
        }
        let count = u32::from_be_bytes([header[1], header[2], header[3], header[4]]) as usize;
        let mut ids = Vec::with_capacity(count);
        let mut word = [0u8; 4];
        for _ in 0..count {
            peer.read_exact(&mut word)
                .context("read a pass command's token ids")?;
            ids.push(u32::from_be_bytes(word) as usize);
        }
        Pass::decode(header[0], ids)
    }

    /// Exchange one 32-byte sequence digest with every peer and require
    /// agreement.
    ///
    /// Both sides WRITE before either READS, exactly as [`Group::barrier`]
    /// does and for the same reason: 32 bytes fit any socket buffer, so the
    /// write phase cannot block on a peer that has not read yet, and a star of
    /// more than two cannot deadlock.
    ///
    /// This is what remains of the deleted `ServePair`'s byte-for-byte
    /// cross-rank check. It is weaker in one specific way, recorded here rather
    /// than only in the commit that removed the old one: the proxy WITHHELD
    /// each token fragment until the peer confirmed it, so a divergent byte was
    /// never spoken; this compares after the fact, at a turn boundary. It is
    /// stronger in another: the digest covers every pass, including context
    /// passes the old check never saw. Never call it inside the layer loop, and
    /// never while a collective is in flight.
    pub fn agree(&mut self, digest: [u8; 32]) -> Result<()> {
        for peer in self.socks.iter_mut() {
            peer.write_all(&digest)
                .context("send the sequence digest")?;
            peer.flush().ok();
        }
        for (index, peer) in self.socks.iter_mut().enumerate() {
            let mut theirs = [0u8; 32];
            peer.read_exact(&mut theirs)
                .context("receive a peer's sequence digest")?;
            anyhow::ensure!(
                theirs == digest,
                "TENSOR RANKS DIVERGED: peer {index} holds sequence digest {} where this rank \
                 (rank {}) holds {}. Both ranks decode from the same all-reduced logits, so a \
                 disagreement here means the collective delivered different bytes to each rank \
                 — a fabric or NCCL fault, not a sampling difference. {}",
                hex32(&theirs),
                self.tp.rank(),
                hex32(&digest),
                transport_note()
            );
        }
        Ok(())
    }

    /// Send rank 0 this rank's learned cuts, in answer to [`Pass::Export`].
    /// Only a non-zero rank may call it.
    ///
    /// `[count u32be][count x LearnedCut]` on the socket to rank 0. A whole
    /// layer's worth on this model is under two gibibytes, and the write
    /// blocks against rank 0's read of it, which rank 0 makes after its own
    /// export -- so the socket buffer, not a second thread, absorbs the skew.
    pub fn send_cuts(&mut self, cuts: &[super::learned::LearnedCut]) -> Result<()> {
        anyhow::ensure!(
            self.tp.rank() != 0,
            "rank 0 collects learned cuts; it has nobody to send them to"
        );
        let peer = self
            .socks
            .first_mut()
            .context("this rank has no rendezvous socket to rank 0")?;
        peer.write_all(&(cuts.len() as u32).to_be_bytes())
            .context("send the learned-cut count")?;
        let mut frame = Vec::new();
        for cut in cuts {
            frame.clear();
            cut.encode_into(&mut frame);
            peer.write_all(&frame)
                .with_context(|| format!("send {}[{}] rank {}", cut.name, cut.expert, cut.rank))?;
        }
        peer.flush().ok();
        Ok(())
    }

    /// Receive every other rank's learned cuts after leading [`Pass::Export`].
    /// Only rank 0 may call it.
    pub fn recv_cuts(&mut self) -> Result<Vec<super::learned::LearnedCut>> {
        anyhow::ensure!(
            self.tp.rank() == 0,
            "only rank 0 collects learned cuts; rank {} sends its own",
            self.tp.rank()
        );
        let mut cuts = Vec::new();
        for (index, peer) in self.socks.iter_mut().enumerate() {
            let mut count = [0u8; 4];
            peer.read_exact(&mut count)
                .with_context(|| format!("receive peer {index}'s learned-cut count"))?;
            let count = u32::from_be_bytes(count) as usize;
            for i in 0..count {
                cuts.push(
                    super::learned::LearnedCut::decode(peer)
                        .with_context(|| format!("receive peer {index}'s learned cut {i}/{count}"))?,
                );
            }
        }
        Ok(cuts)
    }

    /// Whether every peer's end of the rank link is still open.
    ///
    /// A zero-timeout `poll` for `POLLHUP`/`POLLERR`, checked at pass
    /// boundaries. It catches a peer that died BETWEEN passes; a peer that dies
    /// while both ranks are inside a collective still hangs the survivor,
    /// because nothing watches the link concurrently. The deleted `ServePair`
    /// bounded that case with a reader thread per rank. Restoring it needs
    /// either a watchdog thread owning a `try_clone` of this socket that
    /// force-exits on EOF, or NCCL's own `NCCL_ASYNC_ERROR_HANDLING=1` plus
    /// `ncclCommGetAsyncError` polled from the layer loop. Neither is here.
    pub fn peer_alive(&self) -> bool {
        #[cfg(unix)]
        {
            use std::os::fd::AsRawFd as _;

            for peer in &self.socks {
                let mut descriptor = libc::pollfd {
                    fd: peer.as_raw_fd(),
                    events: 0,
                    revents: 0,
                };
                // SAFETY: `descriptor` is valid for the one-element poll array.
                let polled = unsafe { libc::poll(&mut descriptor, 1, 0) };
                if polled > 0
                    && descriptor.revents & (libc::POLLHUP | libc::POLLERR | libc::POLLNVAL) != 0
                {
                    return false;
                }
            }
        }
        true
    }
}

/// Connect, retrying, because the peer is usually still loading its pile.
fn connect_with_deadline(addr: &str, wait: Duration) -> Result<TcpStream> {
    let start = std::time::Instant::now();
    loop {
        match TcpStream::connect(addr) {
            Ok(s) => return Ok(s),
            Err(e) => {
                if start.elapsed() > wait {
                    return Err(anyhow::anyhow!(
                        "no rank-0 rendezvous at {addr} after {:?}: {e}. Rank 0 BINDS this \
                         address and every other rank connects to it, so check that rank 0 \
                         is up, that this is rank 0's address on the fast NIC, and that \
                         INK_TP names the same world on both ends.",
                        wait
                    ));
                }
                std::thread::sleep(Duration::from_millis(200));
            }
        }
    }
}

/// One model PASS, named by rank 0 before it makes it.
///
/// This is the whole vocabulary of the rank link, and it is deliberately the
/// vocabulary of [`super::session::Session`] rather than of a conversation:
/// every rank runs the same passes over the same ids, and nothing about turns,
/// tokenizers, typed context or Drive crosses this socket. Rank 1 needs no
/// tokenizer, no context codec, no detokenizer and no pile — it needs the ids
/// and the order.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Pass {
    /// `Session::prefill` over these ids: the first pass of a sequence.
    Prefill(Vec<usize>),
    /// `Session::extend` over these ids: carry, then whatever is new.
    Extend(Vec<usize>),
    /// `Session::prefill_scored`: the same pass, and every rank also scores
    /// what it appends — which is what runs the online learner on every rank,
    /// so a tensor-parallel pair updates both cuts of every expert. The scores
    /// are rank-local (the head is replicated) and only rank 0 keeps them.
    PrefillScored(Vec<usize>),
    /// `Session::extend_scored`; see [`Pass::PrefillScored`].
    ExtendScored(Vec<usize>),
    /// `Session::step`: one decode, no new rows.
    Step,
    /// `Session::reset`: the sequence is being replaced.
    Reset,
    /// Exchange sequence digests and require agreement ([`Group::agree`]).
    Agree,
    /// The run is over. Every rank should shut down cleanly.
    Finish,
    /// Rank 0 failed terminally. Stop now rather than wait in a collective.
    Abort,
    /// Every rank sends rank 0 its cut of every expert the learner has moved
    /// ([`Group::send_cuts`]); rank 0 joins them back into whole experts.
    /// Not a pass: no collective, and the reply comes back on this socket.
    Export,
}

impl Pass {
    const PREFILL: u8 = 0x01;
    const EXTEND: u8 = 0x02;
    const STEP: u8 = 0x03;
    const RESET: u8 = 0x04;
    const AGREE: u8 = 0x05;
    const FINISH: u8 = 0x06;
    const ABORT: u8 = 0x07;
    const PREFILL_SCORED: u8 = 0x08;
    const EXTEND_SCORED: u8 = 0x09;
    const EXPORT: u8 = 0x0A;

    /// `[tag u8][count u32be][count x u32be ids]`.
    ///
    /// Fixed width and self-delimiting, so a short read is an error rather than
    /// a resynchronisation problem. A one-row extend — which is what EVERY
    /// generated token after the first costs — is NINE bytes: one tag, one
    /// u32be count, one u32be id.
    fn encode(&self) -> Vec<u8> {
        const NONE: &[usize] = &[];
        let (tag, ids): (u8, &[usize]) = match self {
            Pass::Prefill(ids) => (Self::PREFILL, ids.as_slice()),
            Pass::Extend(ids) => (Self::EXTEND, ids.as_slice()),
            Pass::PrefillScored(ids) => (Self::PREFILL_SCORED, ids.as_slice()),
            Pass::ExtendScored(ids) => (Self::EXTEND_SCORED, ids.as_slice()),
            Pass::Step => (Self::STEP, NONE),
            Pass::Reset => (Self::RESET, NONE),
            Pass::Agree => (Self::AGREE, NONE),
            Pass::Finish => (Self::FINISH, NONE),
            Pass::Abort => (Self::ABORT, NONE),
            Pass::Export => (Self::EXPORT, NONE),
        };
        let mut frame = Vec::with_capacity(5 + 4 * ids.len());
        frame.push(tag);
        frame.extend_from_slice(&(ids.len() as u32).to_be_bytes());
        for id in ids {
            frame.extend_from_slice(&(*id as u32).to_be_bytes());
        }
        frame
    }

    fn decode(tag: u8, ids: Vec<usize>) -> Result<Self> {
        // Takes the ids as an argument rather than capturing them, so the
        // arms below are free to MOVE the vector into a Prefill/Extend.
        fn empty(ids: &[usize], what: &str) -> Result<()> {
            anyhow::ensure!(
                ids.is_empty(),
                "a {what} pass carries no token ids, but {} arrived",
                ids.len()
            );
            Ok(())
        }
        Ok(match tag {
            Self::PREFILL => Pass::Prefill(ids),
            Self::EXTEND => Pass::Extend(ids),
            Self::PREFILL_SCORED => Pass::PrefillScored(ids),
            Self::EXTEND_SCORED => Pass::ExtendScored(ids),
            Self::STEP => {
                empty(&ids, "Step")?;
                Pass::Step
            }
            Self::RESET => {
                empty(&ids, "Reset")?;
                Pass::Reset
            }
            Self::AGREE => {
                empty(&ids, "Agree")?;
                Pass::Agree
            }
            Self::FINISH => {
                empty(&ids, "Finish")?;
                Pass::Finish
            }
            Self::ABORT => {
                empty(&ids, "Abort")?;
                Pass::Abort
            }
            Self::EXPORT => {
                empty(&ids, "Export")?;
                Pass::Export
            }
            // An unknown tag is a version skew between two boxes that are
            // supposed to be running the SAME binary. Refusing loudly is the
            // point: the one-binary deployment exists so this cannot happen,
            // and if it does, the run must stop rather than desynchronise.
            other => anyhow::bail!(
                "unknown rank-link pass tag {other:#04x}: the two ranks are not the same build"
            ),
        })
    }
}

fn hex32(bytes: &[u8; 32]) -> String {
    use std::fmt::Write as _;

    let mut text = String::with_capacity(64);
    for byte in bytes {
        write!(&mut text, "{byte:02X}").expect("writing into a String is infallible");
    }
    text
}

/// WHICH RANK THIS BOX IS, decided by comparing its own addresses to the
/// rendezvous address the operator named.
///
/// # The discriminator was already here, unread
///
/// [`Group::form`] has rank 0 `TcpListener::bind(addr)` while every other rank
/// `connect`s to it, and its doc already says the consequence: *"rank 0's
/// address is the one both sides name — which is also why rank 0 should be the
/// box whose ConnectX address is stable."* So the rendezvous address is not
/// merely configuration that happens to differ per box; it NAMES RANK 0. A box
/// that holds that address is rank 0. A box that does not, is not.
///
/// That is the whole election. It needs no new configuration (the rendezvous
/// address is configuration that already had to exist), no new state, no
/// consensus protocol, and no lease. And it means the two INVOCATIONS are
/// identical, not just the two binaries — which is the actual deployment
/// property being bought, since `--tp-rank 0` on one box and `--tp-rank 1` on
/// the other is two commands to keep in sync forever.
///
/// # It fails correctly in both directions
///
/// A /30 has two hosts, so BOTH boxes matching is impossible. NEITHER matching
/// means the operator named an address that lives on neither box — and that is
/// refused here rather than discovered 180 seconds later as a rendezvous
/// timeout on two processes that both dialled and neither bound.
///
/// The one case this cannot detect from inside a single box is "the other box
/// also thinks it is rank 1", because a box knows only its own addresses. What
/// catches that is the existing 180 s `connect` deadline in
/// [`connect_with_deadline`], whose message already names the right cause, and
/// then [`Group::warm`]'s all-reduce sum, which refuses a group that formed
/// but did not pair.
///
/// # Alternatives, and why not
///
/// * **Hostname match** — adds machine names as new configuration, when the
///   rendezvous address is configuration that already exists.
/// * **An explicit `--tp-rank`** — today's arrangement. It is not discovery,
///   and it makes the two invocations differ even when the binary does not.
/// * **A lease file** — new state, and there is no shared storage between the
///   boxes to put it on.
/// * **The `gb10` box lock** — an advisory reservation with a 90-minute
///   staleness timeout. Coupling model topology to a benchmark-arbitration
///   lock makes a stale lock a WRONG MODEL, silently.
pub fn elect_rank(rendezvous: &str, world: usize) -> Result<Tp> {
    use std::net::ToSocketAddrs as _;

    anyhow::ensure!(
        world == 2,
        "address-match rank election decides exactly one bit — this box either holds the \
         rendezvous address or it does not — so it can only elect a PAIR. A world of {world} \
         needs a launcher that assigns ranks explicitly."
    );

    let named = rendezvous
        .to_socket_addrs()
        .with_context(|| {
            format!(
                "resolve the rendezvous address {rendezvous:?}. It must be HOST:PORT on the fast \
                 fabric, and the HOST must be rank 0's address on that fabric."
            )
        })?
        .map(|addr| addr.ip())
        .collect::<Vec<_>>();
    anyhow::ensure!(
        !named.is_empty(),
        "the rendezvous address {rendezvous:?} resolved to nothing"
    );
    for address in &named {
        anyhow::ensure!(
            !address.is_loopback(),
            "the rendezvous address {rendezvous:?} resolves to loopback ({address}). Loopback \
             names THIS box on both boxes, so the two ranks would never meet; name rank 0's \
             address on the fabric."
        );
        anyhow::ensure!(
            !address.is_unspecified(),
            "the rendezvous address {rendezvous:?} resolves to the unspecified address \
             ({address}), which names no box"
        );
    }

    let local = local_addresses().context("enumerate this box's own addresses")?;
    let matched = named.iter().find(|address| local.contains(address));
    let rank = match matched {
        Some(address) => {
            println!(
                "  rank election      : rank 0 — this box holds {address}, which is the \
                 rendezvous address; it BINDS, owns the Drive loop and owns the pile"
            );
            0
        }
        None => {
            println!(
                "  rank election      : rank 1 — this box holds none of {named:?}, so rank 0 is \
                 elsewhere; it DIALS and runs as a pure model rank"
            );
            1
        }
    };
    Tp::new(rank, world)
}

/// Every address this box holds, on every interface, minus loopback.
///
/// Deliberately not "the address of the fabric interface": naming the
/// interface would be more configuration, and the question being asked is only
/// whether the operator's rendezvous address is one of ours. An unrelated
/// management address matching would mean the operator named the management
/// address, which is a real answer — rank 0 would then bind it and the
/// rendezvous would work, slowly, and `transport_note` would say so.
fn local_addresses() -> Result<Vec<std::net::IpAddr>> {
    let mut addresses = Vec::new();
    #[cfg(unix)]
    {
        let mut list: *mut libc::ifaddrs = std::ptr::null_mut();
        // SAFETY: `getifaddrs` fills `list` with an allocation freed by
        // `freeifaddrs` below on every path out of this block.
        let got = unsafe { libc::getifaddrs(&mut list) };
        anyhow::ensure!(
            got == 0,
            "getifaddrs failed: {}",
            std::io::Error::last_os_error()
        );
        let mut entry = list;
        while !entry.is_null() {
            // SAFETY: `entry` is a node of the list `getifaddrs` built.
            let node = unsafe { &*entry };
            if !node.ifa_addr.is_null() {
                // SAFETY: the family tag selects which concrete sockaddr the
                // kernel stored, and each branch reads only that type's fields.
                let family = unsafe { (*node.ifa_addr).sa_family } as libc::c_int;
                if family == libc::AF_INET {
                    let raw = node.ifa_addr as *const libc::sockaddr_in;
                    // SAFETY: family said AF_INET.
                    let octets = unsafe { (*raw).sin_addr.s_addr };
                    addresses.push(std::net::IpAddr::V4(std::net::Ipv4Addr::from(
                        u32::from_be(octets),
                    )));
                } else if family == libc::AF_INET6 {
                    let raw = node.ifa_addr as *const libc::sockaddr_in6;
                    // SAFETY: family said AF_INET6.
                    let octets = unsafe { (*raw).sin6_addr.s6_addr };
                    addresses.push(std::net::IpAddr::V6(std::net::Ipv6Addr::from(octets)));
                }
            }
            entry = node.ifa_next;
        }
        // SAFETY: `list` came from the successful `getifaddrs` above.
        unsafe { libc::freeifaddrs(list) };
    }
    addresses.retain(|address| !address.is_loopback() && !address.is_unspecified());
    anyhow::ensure!(
        !addresses.is_empty(),
        "this box reports no non-loopback address, so it cannot decide whether it is rank 0"
    );
    Ok(addresses)
}

/// What NCCL chose for the wire, as far as the environment can say.
///
/// Not a measurement and does not pretend to be — it reports what was ASKED
/// for. The socket path and the RDMA path differ by 25x on these boxes
/// (~185 us against 3.68 us), which is enough to invert the entire decision, so
/// a run that does not know which one it got has not measured anything. The
/// only way to know for certain is `NCCL_DEBUG=INFO`, whose banner names the
/// transport per channel; this returns the line a report should print beside
/// the numbers so the question is at least always asked.
pub fn transport_note() -> String {
    let ifname = std::env::var("NCCL_SOCKET_IFNAME").unwrap_or_else(|_| "<unset>".into());
    let hca = std::env::var("NCCL_IB_HCA").unwrap_or_else(|_| "<unset>".into());
    let dbg = std::env::var("NCCL_DEBUG").unwrap_or_else(|_| "<unset>".into());
    format!("NCCL_SOCKET_IFNAME={ifname} NCCL_IB_HCA={hca} NCCL_DEBUG={dbg}")
}

// ---------------------------------------------------------------------------
// The activation-level reduce the forward actually calls.
// ---------------------------------------------------------------------------

/// Sum a `[rows, cols]` activation across every rank, on the stream.
///
/// This is the whole TP2 forward's contact with the network, and it is issued
/// and forgotten: [`Group::all_reduce_f32`] enqueues NCCL on the comm stream
/// between two device-side fences and returns immediately. Nothing here reads
/// the buffer, and nothing here may be made to — see the module header.
///
/// # Where this may be called, and where it may NOT
///
/// A column-then-row split leaves each rank holding a PARTIAL sum of the whole
/// hidden vector: rank 0's contribution from its heads, rank 1's from its own.
/// The partials are only meaningful once added, so this must run before the
/// first operation that is not linear in that sum -- and in this stack there is
/// one immediately downstream of both reduce sites, which is the trap:
///
/// **The short convolution is NOT commutable with a partial sum.** It mixes a
/// rank's partial with CACHED HISTORY from previous tokens, which is already
/// whole. `conv(a) + conv(b) != conv(a + b)` the moment the history term is
/// non-zero, so a reduce placed after the convolution -- which is where it
/// looks like it belongs, next to the residual add -- convolves each rank's
/// half against the full history, sums two wrong answers, and returns a finite,
/// plausible, WRONG hidden state. There is no crash and no NaN to notice.
///
/// So the order is fixed and is not a preference:
///
/// ```text
///   attention (this rank's heads)  ->  REDUCE  ->  short conv  ->  residual
///   MoE       (this rank's half)   ->  REDUCE  ->  short conv  ->  residual
/// ```
pub fn reduce_activation(
    g: &Group,
    device: &burn::backend::cuda::CudaDevice,
    x: Tensor<crate::models::inkling::seam::Bk, 2>,
) -> Tensor<crate::models::inkling::seam::Bk, 2> {
    use crate::models::inkling::seam;
    let [rows, cols] = x.dims();
    let client = seam::client_of(&x);
    // `handle_of` contiguises and asserts f32, which is what NCCL is told the
    // buffer is. A BF16 activation reaching here is a loud panic rather than a
    // reduce of reinterpreted bytes.
    let h = seam::handle_of(x);
    g.all_reduce_f32(&h);
    seam::tensor_of(client, device.clone(), h, rows, cols)
}

#[cfg(test)]
mod tests {
    use super::Pass;

    /// The rank link's framing, which is the ONLY thing the two boxes now say
    /// to each other outside NCCL.
    ///
    /// It round-trips because a desynchronised follower is the failure mode
    /// with no symptom: it would make the wrong pass, block in a collective its
    /// peer is not in, and hang both boxes with nothing in either log.
    #[test]
    fn every_pass_round_trips_and_an_unknown_tag_refuses() {
        for pass in [
            Pass::Prefill(vec![1, 2, 3]),
            Pass::Extend(vec![7]),
            Pass::Extend(Vec::new()),
            Pass::Step,
            Pass::Reset,
            Pass::Agree,
            Pass::Finish,
            Pass::Abort,
        ] {
            let frame = pass.encode();
            let count = u32::from_be_bytes([frame[1], frame[2], frame[3], frame[4]]) as usize;
            assert_eq!(frame.len(), 5 + 4 * count, "{pass:?} is self-delimiting");
            let ids = frame[5..]
                .chunks_exact(4)
                .map(|word| u32::from_be_bytes([word[0], word[1], word[2], word[3]]) as usize)
                .collect::<Vec<_>>();
            assert_eq!(Pass::decode(frame[0], ids).unwrap(), pass);
        }

        // A one-row extend is what EVERY generated token after the first costs
        // on the link. NINE bytes -- `[tag u8][count u32be][one id u32be]` --
        // against the framed-stream CONSULT envelope plus a JSON TurnEnd it
        // replaced. (This literal said 13 and the comment said "Thirteen" until
        // 2026-08-30, when the test was first RUN: the self-delimiting
        // assertion directly above already pins the length at `5 + 4 * count`,
        // which is 9 for one id, so the two assertions contradicted each other
        // and the arithmetic slip was in this one. The encoder was always
        // right. The stale figure had already propagated into the branch's
        // own write-up as the per-token wire cost, overstating it by 44%.)
        assert_eq!(Pass::Extend(vec![9]).encode().len(), 9);

        let error = Pass::decode(0xFE, Vec::new()).expect_err("an unknown tag must refuse");
        assert!(
            error.to_string().contains("not the same build"),
            "{error:#}"
        );

        let error = Pass::decode(0x03, vec![1]).expect_err("a Step pass carries no ids");
        assert!(error.to_string().contains("no token ids"), "{error:#}");
    }
}
