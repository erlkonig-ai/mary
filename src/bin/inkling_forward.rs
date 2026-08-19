//! A real forward pass of Inkling-Small across a CLUSTER, on the device.
//!
//! Every gate ends with the same disclaimer: the checkpoint-name to module
//! mapping is authored on both sides, so a shared misreading would pass. This
//! is the check that can settle it. Coherent continuations cannot come out of
//! a wrong mapping — a transposed projection or a swapped gate/up half
//! produces noise, not English.
//!
//! # There is no single-node mode, and that is deliberate
//!
//! One box cannot hold this model: 144 GiB of weights against 119 GiB of RAM.
//! It USED to run anyway, by streaming — re-reading each token's expert slabs
//! off the SSD because the page cache had evicted them — and every measurement
//! taken in that mode was a measurement of a disk, not of a model. So
//! `INK_LAYERS` is now REQUIRED and must name a strict subrange: a process that
//! would run the whole stack refuses to start.
//!
//! Not "exactly two". Two is what THIS model needs (byte-balanced at layer 20:
//! layer 2 carries BF16 experts, 12.7 GiB against 3.55 GiB for an NVFP4 layer,
//! so an even 21/21 split is a lopsided 85/71 GiB one). The 66-layer sibling
//! wants five to seven. The rule is that no node runs the whole stack.
//!
//! `INK_LAYERS=LO:HI` runs that half-open range; `INK_PIPE=head:HOST:PORT`
//! sends the residual stream on when the range ends, and `INK_PIPE=tail:ADDR`
//! receives it, finishes the stack and returns the argmax. Only `[n, 4096]` f32
//! crosses — 16 KB per token per boundary, once — which is why the split is by
//! layer and not within one: splitting a layer needs an all-reduce per layer and
//! 1 GbE cannot carry it.
//!
//!   # tail, on the second box
//!   INK_LAYERS=20:42 INK_PIPE=tail:0.0.0.0:7654 inkling_forward <pile> <ids> <out>
//!   # head, on the first
//!   INK_LAYERS=0:20  INK_PIPE=head:<tail-host>:7654 inkling_forward <pile> <ids> <out>
//!
//! # Drafting happens on the tail
//!
//! `INK_MTP=k` used to refuse a pipe, on the reasoning that "the head owns no
//! unembedding and the tail owns no embedding table, so neither end can draft
//! alone". Half of that is right. An MTP head takes the stack's FINAL hidden
//! state and the embedding of the token one step ahead, and the tail computes
//! the last layer, owns the final norm, owns the unembedding, and follows the
//! sequence by recomputing the argmax rather than being told. The embedding
//! table was the only missing piece, and a tail with `INK_MTP` set now loads it
//! — deliberately, for this configuration and no other, since the point of the
//! split is that neither box pins a table it never reads.
//!
//! Set `INK_MTP` on the TAIL process only. It is refused on a head, which holds
//! neither the final hidden state nor a way to turn one into a token.
//!
//! # One lane, all the way down
//!
//! There is nothing to select. Attention, the dense and shared MLPs, the head
//! and the routed experts are all on the device, always; residency is
//! unconditional; the routed experts go through the NVFP4 tensor cores as the
//! NVFP4 they are stored as. `INK_GPU`, `INK_ATTN`, `INK_DENSE`, `INK_HEAD`,
//! `INK_EXPERTS`, `INK_RESIDENT`, `INK_DEQUANT`, `INK_ZEROCOPY`, `INK_WARM` and
//! `INK_HOST_SUM` all named a second, slower, host-or-widening implementation
//! of something, and each of them is gone with it. A lane nobody selects is a
//! lane nobody tests; a host lane you CAN select is one you will run by
//! accident, which is how a 401 s forward happened.
//!
//! The host lane is now gone from the LIBRARY as well, not just from the
//! switches here. `mlp::routed_experts`, `mlp::shared_experts`, `mlp::moe` and
//! `mlp::expert_ffn_one` were unreachable from this file and still cost real
//! work: they were the readable transcription of the routed algorithm, so an
//! agent asked to make the experts faster found THEM, and twice in one day
//! someone analysed how to parallelise a function no forward calls. Legibility
//! attracts effort, which means the legible version has to be the live one.
//! [`routed_experts_fp4`]'s doc comment is where that statement of the
//! algorithm lives now. What survives on the host is `mlp::dense_mlp` and
//! `layer::decoder_layer`, and neither is a fallback: they are the only
//! implementation the MTP heads have.
//!
//! Mixed precision is not a second lane either. Layer 2's experts are BF16 and
//! the other 41 layers' are NVFP4, so the routed block picks the instruction the
//! stored format calls for — `mma.sync…bf16` or the block-scaled
//! `…mxf4nvf4.block_scale` — and nothing else changes. Both accumulate in f32,
//! which is the MMA's own output type and not a widening; widening the BF16
//! weight to f32 to reuse the f32 path is the exact thing this file does not do.
//!
//! # Where the weights come from
//!
//! The pile, positionally, on branch `INK_PILE_BRANCH` (default `inkling`). All
//! of it: the weights AND `config.json` AND the chat template, so a run reads
//! nothing off a checkpoint directory and there is no path it could.
//!
//! It used to be the other way round — a checkpoint directory positionally,
//! with `INK_PILE=<path>` as an opt-in swap — and by the end that argument was
//! vestigial theatre: the last runs passed `--` for it because the pile
//! answered for everything. An opt-in that every real invocation opts into is
//! a default written backwards, and a source that CAN be a directory of shards
//! is one that will be by accident. `mary` already made this move for qwen3tts
//! (8475167) and for the runtime at large (906cbc9, "safetensors gated behind
//! `import`, runtime is pile-only"); Inkling is the last holdout.
//!
//!   cargo run --release --features inkling-cuda,cuda-backend,import \
//!       --bin inkling_forward -- <pile> <ids.bin> <out.bin>

use std::collections::BTreeMap;
use std::io::{Read as _, Write as _};
use std::net::{TcpListener, TcpStream};
use std::path::PathBuf;
use std::time::Instant;

use anyhow::{Context, Result};

use mary::models::inkling::attn::{causal_mask, AttnDims, AttnWeights, LogScaling};
use mary::models::inkling::block::{rms_norm, route_from_logits, Routing};
use mary::models::inkling::config::{AttnKind, InklingConfig};
use mary::models::inkling::load::{split_gate_up, Held};
use mary::models::inkling::bf16gemm::Bf16W;
use mary::models::inkling::pile::Elem;
use mary::models::inkling::source::Weights;
use mary::models::inkling::mtp::{
    mtp_block, mtp_block_prefill, mtp_block_step, Concat as MtpConcat, MtpCache, MtpHead,
};
use mary::models::inkling::layer::{LayerMlp, LayerWeights};
use mary::models::inkling::stack::{embed_and_norm_bf16, embed_row_bf16};

/// One gibibyte, as the divisor every byte count here is printed against.
const GIB: f64 = (1u64 << 30) as f64;

/// Bytes this process has actually pulled off the block device.
///
/// NOT the same as bytes read: a page-cache hit costs nothing here. That is
/// precisely what makes it the right instrument for residency — a working set
/// that is genuinely held reads ~0 on the second pass, and one that is merely
/// "probably still in the page cache" does not.
fn io_read_bytes() -> u64 {
    std::fs::read_to_string("/proc/self/io")
        .ok()
        .and_then(|s| {
            s.lines()
                .find(|l| l.starts_with("read_bytes:"))
                .and_then(|l| l.split_whitespace().nth(1).and_then(|v| v.parse().ok()))
        })
        .unwrap_or(0)
}

/// A 95% Wilson score interval for `hits` of `n`.
///
/// Wilson rather than the textbook normal interval, because this measurement
/// produces rates at both ends — depth 1 near 3/4, depth 4 at or near 0 — and
/// there the normal interval runs off the end of [0, 1] and reports a bound that
/// is not a bound. It is printed beside every rate so the counts can be read as
/// evidence: 15/20 and 150/200 are the same percentage and not the same claim.
fn wilson95(hits: usize, n: usize) -> (f64, f64) {
    if n == 0 {
        return (0.0, 1.0);
    }
    let z = 1.959_963_984_540_054_f64;
    let (h, n) = (hits as f64, n as f64);
    let p = h / n;
    let d = 1.0 + z * z / n;
    let centre = (p + z * z / (2.0 * n)) / d;
    let half = z * ((p * (1.0 - p) / n) + z * z / (4.0 * n * n)).sqrt() / d;
    ((centre - half).max(0.0), (centre + half).min(1.0))
}

/// Which end of a two-machine split this process is, if it is one.
///
/// The head owns the embedding and the token loop; the tail owns the final norm,
/// the unembedding and the argmax. Neither loads the other's tables — on a box
/// chosen because the working set only just fits, pinning 3.3 GB of unembedding
/// on the machine that will never read it is the whole problem in miniature.
enum Pipe {
    Head(TcpStream),
    Tail(TcpStream),
}

/// The residual stream, on the wire.
///
/// `pos0` travels with it because the tail's half of the stack needs it and
/// cannot derive it: with a KV cache a decode step feeds ONE token whose
/// absolute position is what log scaling and the relative bias are functions of,
/// and the tail has never seen the sequence it belongs to.
fn send_stream(s: &mut TcpStream, n: usize, pos0: usize, x: &[f32]) -> Result<()> {
    let mut b = Vec::with_capacity(16 + x.len() * 4);
    b.extend_from_slice(&(n as u64).to_le_bytes());
    b.extend_from_slice(&(pos0 as u64).to_le_bytes());
    for v in x {
        b.extend_from_slice(&v.to_le_bytes());
    }
    s.write_all(&b)?;
    s.flush()?;
    Ok(())
}

/// The other side of [`send_stream`]. `None` when the peer is done.
fn recv_stream(s: &mut TcpStream, h: usize) -> Result<Option<(usize, usize, Vec<f32>)>> {
    let mut hdr = [0u8; 16];
    if s.read_exact(&mut hdr).is_err() {
        return Ok(None);
    }
    let n = u64::from_le_bytes(hdr[..8].try_into().unwrap()) as usize;
    let pos0 = u64::from_le_bytes(hdr[8..].try_into().unwrap()) as usize;
    if n == 0 {
        return Ok(None);
    }
    let mut buf = vec![0u8; n * h * 4];
    s.read_exact(&mut buf)?;
    let x = buf
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
        .collect();
    Ok(Some((n, pos0, x)))
}

/// The backend the device lane runs on.
type Bk = burn::backend::Cuda<f32>;
use burn::prelude::Backend;
use burn::tensor::{Tensor as BT, TensorData as BTD};
use mary::models::inkling::burn as dev_lane;

/// Move a host `[rows, cols]` matrix to the device, consuming it.
///
/// Takes the `Vec` by value on purpose: the dense `w13` is 537 MB at f32 and a
/// borrowing helper would hold two copies of it at once.
fn up2<B: Backend>(v: Vec<f32>, rows: usize, cols: usize, dev: &B::Device) -> BT<B, 2> {
    assert_eq!(v.len(), rows * cols, "{} values are not [{rows}, {cols}]", v.len());
    BT::from_data(BTD::new(v, [rows, cols]), dev)
}

fn up1r<B: Backend>(v: &[f32], len: usize, dev: &B::Device) -> BT<B, 1> {
    assert_eq!(v.len(), len, "{} values are not [{len}]", v.len());
    BT::from_data(BTD::new(v.to_vec(), [len]), dev)
}

fn up1<B: Backend>(v: Vec<f32>, len: usize, dev: &B::Device) -> BT<B, 1> {
    assert_eq!(v.len(), len, "{} values are not [{len}]", v.len());
    BT::from_data(BTD::new(v, [len]), dev)
}

/// Read a `[rows, cols]` device tensor back to the host. This is also the sync,
/// so a timer around the call measures work rather than enqueueing.
fn down<B: Backend>(t: BT<B, 2>) -> Vec<f32> {
    t.into_data().convert::<f32>().to_vec::<f32>().expect("device readback")
}

/// A device tensor of this run's backend, named once so the residency types
/// below do not have to repeat it.
type T2 = burn::tensor::Tensor<Bk, 2>;

/// Host seconds inside the routed-expert lane, split by WHAT THE HOST DID.
///
/// One bucket used to cover binding, quantising, four enqueues and the layer's
/// blocking read, and it was called "upload"; a profiling session went looking
/// for a transfer that turned out to be 4% of it. So each field names one kind
/// of work and the sync has a field of its own, because a bucket that mixes
/// "issued a kernel" with "waited for the GPU" cannot answer whether the host
/// is the bottleneck.
#[derive(Default, Clone, Copy)]
struct HostT {
    /// `expert_packed` / `expert_bf16`: a hash lookup and a view of the pile.
    slice: f64,
    /// Copying this expert's tokens out of the residual stream, on the host.
    gather: f64,
    /// Binding the weight, quantising the activation, issuing the kernels.
    /// Non-blocking by construction.
    enqueue: f64,
    /// `read_one`: BLOCKING, so this is the layer's device time as much as the
    /// host's.
    drain: f64,
    /// Scattering each expert's rows back into the accumulator, weighted.
    accum: f64,
    /// Layers the GROUPED lane took: one launch per stage for the whole layer.
    grouped: usize,
    /// Layers that fell back to the per-expert loop, because their weights are
    /// not offsets into one registered mapping.
    per_expert: usize,
}

/// One layer's shared experts, on the device, as the BF16 the pile stores.
///
/// `gate_up` is ONE weight, `[2 * n_shared * inter, hidden]`, holding every
/// shared expert's gate block followed by every up block. It used to be four
/// `Bf16W` — a gate and an up per expert — and four separate GEMMs against the
/// same activation, which is four launches and, more to the point, four grids
/// of 256 cubes.
///
/// The `m16n8k16` kernel gives one warp each `(m_tile, n_tile)`, so with M
/// padded to one tile the grid IS `n / 8` and nothing else. 256 cubes of one
/// warp cannot cover DRAM latency on this part: measured, those calls ran at
/// 79 GB/s while the unembed — same kernel, same instruction, 25128 cubes —
/// ran at 175. Concatenating along `n` is the one axis that adds cubes without
/// touching the arithmetic: each warp still computes the same output tile by
/// the same k-loop, so this is a scheduling change and not a numerical one.
struct SharedOnDevice {
    gate_up: Bf16W,
    down: Vec<Bf16W>,
}

/// The dense weights that live in DEVICE memory for the whole run, unwidened.
///
/// Distinct from the host-resident set, and the difference is the point. Host
/// residency stops the re-read; device residency moves the weight into the
/// memory the GPU reads fastest and lets the matmul run there.
///
/// What changed with rule 3: these used to be `Tensor<Bk, 2>` — Burn f32 — so
/// every BF16 leaf on the way here was doubled, once into a host `Vec<f32>` and
/// again into a device buffer. 4.88 GiB of f32 on the 20-layer head to hold
/// 2.44 GiB of stored weight, and twice the bytes for the GEMM to read on every
/// token. They are [`Bf16W`] now: a handle over BF16, multiplied by the
/// `mma.sync…bf16` instruction whose f32 accumulator is its own output type and
/// not a widening.
#[derive(Default)]
struct DeviceDense {
    shared: std::collections::BTreeMap<String, SharedOnDevice>,
    dense: std::collections::BTreeMap<String, (Bf16W, Bf16W, Bf16W, f32)>,
    bytes: u64,
}

/// One MTP head's weights, OWNED, so the borrowed [`MtpHead`] handed to
/// `mtp_block` can be rebuilt per draft without re-reading anything.
///
/// The split exists because `MtpHead` borrows and the loop needs the owner to
/// outlive every borrow. `gate`/`up` are materialised rather than held because
/// `split_gate_up` de-interleaves the fused matrix into two, and doing that once
/// per head at load beats doing it once per draft.
struct MtpOwned {
    embed_norm: Held,
    hidden_norm: Held,
    input_proj: Held,
    attn_norm: Held,
    mlp_norm: Held,
    attn_sconv: Held,
    mlp_sconv: Held,
    wq: Held,
    wk: Held,
    wv: Held,
    wr: Held,
    wo: Held,
    q_norm: Held,
    k_norm: Held,
    k_sconv: Held,
    v_sconv: Held,
    rel_proj: Held,
    gate: Vec<f32>,
    up: Vec<f32>,
    down: Held,
    global_scale: f32,
    dims: AttnDims,
    local: bool,
}

impl MtpOwned {
    /// The sliding window this head attends within, or `None` on a global
    /// one.
    ///
    /// One fact in one place: the causal mask and the cache's idea of which
    /// keys can never be read again are the same distinction. A head that
    /// masked as local while caching as global would still answer correctly
    /// and grow its cache without bound, which is the kind of bug that only
    /// ever shows up as a memory graph.
    fn window(&self, sliding: usize) -> Option<usize> {
        if self.local {
            Some(sliding)
        } else {
            None
        }
    }

    /// Borrow this head in the shape `mtp_block` wants.
    fn borrow(&self, inter: usize) -> MtpHead<'_> {
        MtpHead {
            embed_norm: &self.embed_norm.data,
            hidden_norm: &self.hidden_norm.data,
            input_proj: &self.input_proj.data,
            lw: LayerWeights {
                attn_norm: &self.attn_norm.data,
                mlp_norm: &self.mlp_norm.data,
                attn_sconv: &self.attn_sconv.data,
                mlp_sconv: &self.mlp_sconv.data,
            },
            aw: AttnWeights {
                wq: &self.wq.data,
                wk: &self.wk.data,
                wv: &self.wv.data,
                wr: &self.wr.data,
                wo: &self.wo.data,
                k_sconv: &self.k_sconv.data,
                v_sconv: &self.v_sconv.data,
                q_norm: &self.q_norm.data,
                k_norm: &self.k_norm.data,
                rel_proj: &self.rel_proj.data,
            },
            mlp: LayerMlp {
                gate: &self.gate,
                up: &self.up,
                down: &self.down.data,
                global_scale: self.global_scale,
                inter,
            },
        }
    }
}

/// Bind a `[rows, cols]` BF16 byte block to the device, aliasing where it can.
///
/// `bytes` is either a view of the pile's mapping — in which case this is a
/// pointer, not a transfer — or a de-interleaved copy, which cannot be aliased
/// because it is not IN the mapping. `Aliases::slice_or_copy` decides by
/// pointer containment and counts which happened, so the report can say.
fn bind_bf16(
    client: &cubecl::prelude::ComputeClient<cubecl::cuda::CudaRuntime>,
    aliases: Option<&mary::models::inkling::fp4gemm::Aliases>,
    bytes: &[u8],
    rows: usize,
    cols: usize,
) -> Bf16W {
    assert_eq!(bytes.len(), rows * cols * 2, "{rows}x{cols} BF16 is not {} bytes", bytes.len());
    assert!(
        Bf16W::tileable(rows, cols),
        "{rows}x{cols} does not tile as m16n8k16; this lane multiplies BF16 by BF16 and \
         widening to reach a shape is what rule 3 forbids"
    );
    let align = note_align(bytes, rows, cols);
    // `INK_ALIGN_COPY=1`: pay a COPY for a weight the alias seam can only place
    // at 4 or 8, and buy the tuned GEMM lane for it. The trade is device bytes
    // against kernel throughput and it is measured both ways rather than
    // assumed; the default is to alias and take the hand kernel on those.
    let copy_to_align = align < 16
        && std::env::var("INK_ALIGN_COPY").map(|v| v == "1").unwrap_or(false);
    let h = match aliases {
        Some(al) if !copy_to_align => al.slice_or_copy(client, bytes),
        _ => client.create_from_slice(bytes),
    };
    Bf16W { h, n: rows, k: cols, align: if copy_to_align { 16 } else { align } }
}

/// How every plain-BF16 weight is aligned, and how much of the model that is.
///
/// The tuned matmul picks its load width from the SHAPE and never from the
/// pointer, so a `[4096, 4096]` operand gets 16-byte loads whatever address it
/// sits at. The aliasing seam only promises 4 (the checkpoint packs tensors
/// back to back and puts the expert slabs at 4 mod 16), so this counts what the
/// dense lane actually gets before anything is decided on the strength of it.
static ALIGN: [core::sync::atomic::AtomicU64; 8] = [const { core::sync::atomic::AtomicU64::new(0) }; 8];

fn note_align(bytes: &[u8], rows: usize, cols: usize) -> usize {
    use core::sync::atomic::Ordering::Relaxed;
    let p = bytes.as_ptr() as usize;
    let (slot, align) = if p % 16 == 0 {
        (0, 16)
    } else if p % 8 == 0 {
        (1, 8)
    } else if p % 4 == 0 {
        (2, 4)
    } else {
        (3, 1)
    };
    ALIGN[slot].fetch_add(1, Relaxed);
    ALIGN[4 + slot].fetch_add((rows * cols * 2) as u64, Relaxed);
    align
}

/// Print the alignment census gathered by [`note_align`].
fn report_align() {
    use core::sync::atomic::Ordering::Relaxed;
    let a: Vec<u64> = ALIGN.iter().map(|c| c.load(Relaxed)).collect();
    let n = a[0] + a[1] + a[2] + a[3];
    if n == 0 {
        return;
    }
    println!(
        "  plain-BF16 weight binds: {n} weights, {:.2} GiB -- 16B {} ({:.2} GiB), 8B {} ({:.2} GiB), \
         4B {} ({:.2} GiB), worse {} ({:.2} GiB)",
        (a[4] + a[5] + a[6] + a[7]) as f64 / GIB,
        a[0],
        a[4] as f64 / GIB,
        a[1],
        a[5] as f64 / GIB,
        a[2],
        a[6] as f64 / GIB,
        a[3],
        a[7] as f64 / GIB
    );
}

impl DeviceDense {
    /// One layer's shared experts, bound on first use.
    #[allow(clippy::too_many_arguments)]
    fn shared_for(
        &mut self,
        cp: &Weights,
        client: &cubecl::prelude::ComputeClient<cubecl::cuda::CudaRuntime>,
        aliases: Option<&mary::models::inkling::fp4gemm::Aliases>,
        p: &str,
        n_shared: usize,
        inter: usize,
        h: usize,
        halved: bool,
    ) -> Result<&SharedOnDevice> {
        if !self.shared.contains_key(p) {
            let fused = cp.stored(&format!("{p}mlp.shared_experts.shared_w13_weight"))?;
            anyhow::ensure!(fused.elem == Elem::Bf16, "shared_w13 is {:?}", fused.elem);
            let (g, u) = mary::models::inkling::load::split_shared_w13_bytes(
                &fused.bytes, n_shared, inter, h, halved, 2,
            );
            let d = cp.stored(&format!("{p}mlp.shared_experts.shared_w2_weight"))?;
            anyhow::ensure!(d.elem == Elem::Bf16, "shared_w2 is {:?}", d.elem);
            let per_d = h * inter * 2;
            // Gate blocks then up blocks, one buffer. `split_shared_w13_bytes`
            // already returns each side with every expert contiguous, so this
            // is a concatenation and not a second de-interleave; the row order
            // is what `shared_experts_bf16` slices the result by.
            let mut gu = g;
            gu.extend_from_slice(&u);
            let mut sd = SharedOnDevice {
                gate_up: bind_bf16(client, aliases, &gu, 2 * n_shared * inter, h),
                down: Vec::new(),
            };
            for e in 0..n_shared {
                // `w2` is NOT de-interleaved, so this one is a view of the pile
                // and aliases outright.
                sd.down.push(bind_bf16(
                    client, aliases, &d.bytes[e * per_d..(e + 1) * per_d], h, inter,
                ));
            }
            self.bytes += (gu.len() + n_shared * per_d) as u64;
            self.shared.insert(p.to_string(), sd);
        }
        Ok(&self.shared[p])
    }

    /// One dense layer's MLP, bound on first use.
    fn dense_for(
        &mut self,
        cp: &Weights,
        client: &cubecl::prelude::ComputeClient<cubecl::cuda::CudaRuntime>,
        aliases: Option<&mary::models::inkling::fp4gemm::Aliases>,
        p: &str,
        h: usize,
    ) -> Result<&(Bf16W, Bf16W, Bf16W, f32)> {
        if !self.dense.contains_key(p) {
            let fused = cp.stored(&format!("{p}mlp.w13_dn.weight"))?;
            anyhow::ensure!(fused.elem == Elem::Bf16, "dense w13 is {:?}", fused.elem);
            let (g, u) = mary::models::inkling::load::split_gate_up_bytes(&fused.bytes, h, 2);
            let down = cp.stored(&format!("{p}mlp.w2_md.weight"))?;
            anyhow::ensure!(down.elem == Elem::Bf16, "dense w2 is {:?}", down.elem);
            let (drows, dcols) = (down.dims[0] as usize, down.dims[1] as usize);
            let inter = g.len() / (h * 2);
            // The global scale is one f32 and is a SCALAR the product is
            // multiplied by, not a weight, so it comes through the widening
            // accessor and costs four bytes.
            let gs = cp.tensor(&format!("{p}mlp.global_scale"))?.data[0];
            self.bytes += (g.len() + u.len() + down.bytes.len()) as u64;
            let trip = (
                bind_bf16(client, aliases, &g, inter, h),
                bind_bf16(client, aliases, &u, inter, h),
                bind_bf16(client, aliases, &down.bytes, drows, dcols),
                gs,
            );
            self.dense.insert(p.to_string(), trip);
        }
        Ok(&self.dense[p])
    }
}

/// The dense MLP, with every weight the BF16 it is stored as.
///
/// `dev_lane::dense_mlp`'s twin: `down(silu(gate(x)) * up(x)) * global_scale`,
/// with the elementwise half still in Burn.
fn dense_mlp_bf16(x: T2, w: &(Bf16W, Bf16W, Bf16W, f32)) -> T2 {
    let g = dev_lane::linear_bf16(x.clone(), &w.0);
    let u = dev_lane::linear_bf16(x, &w.1);
    dev_lane::linear_bf16(dev_lane::silu(g) * u, &w.2).mul_scalar(w.3)
}

/// The shared experts, with every weight the BF16 it is stored as.
///
/// `dev_lane::shared_experts_dev`'s twin, same reason. The gamma multiplies the
/// ACTIVATION, before the down projection — not the block's output, which is
/// algebraically the same only because `down` is linear and is a different
/// function the moment anything else is inserted.
fn shared_experts_bf16(
    dev: &burn::backend::cuda::CudaDevice,
    x: T2,
    sw: &SharedOnDevice,
    gammas: &[f32],
    n_shared: usize,
) -> T2 {
    let [n, _] = x.dims();
    assert_eq!(gammas.len(), n * n_shared, "{} gammas for {n} tokens", gammas.len());
    // ONE projection for every gate and every up in the layer. See
    // [`SharedOnDevice`] for why: four GEMMs against the same activation are
    // four grids of 256 cubes, and this is one of 1024.
    let inter = sw.gate_up.n / (2 * n_shared);
    let gu = dev_lane::linear_bf16(x, &sw.gate_up);
    let mut out: Option<T2> = None;
    for s in 0..n_shared {
        let g = gu.clone().slice([0..n, s * inter..(s + 1) * inter]);
        let u = gu.clone().slice([0..n, (n_shared + s) * inter..(n_shared + s + 1) * inter]);
        let col: Vec<f32> = (0..n).map(|tk| gammas[tk * n_shared + s]).collect();
        let gam = BT::<Bk, 2>::from_data(BTD::new(col, [n, 1]), dev);
        let c = dev_lane::linear_bf16(dev_lane::silu(g) * u * gam, &sw.down[s]);
        out = Some(match out {
            Some(o) => o + c,
            None => c,
        });
    }
    out.expect("a MoE layer with no shared experts")
}

// `lin_bf16` was here. It is `dev_lane::linear_bf16` now: once the attention
// projections needed the same bridge, keeping a second copy of it in the binary
// meant two places to get the M padding right. `dev_lane` can host it because
// nothing about it was generic -- `Bf16W` is a raw cubecl handle and the seam
// that produces one is concrete on `Bk`, which is exactly what `short_conv_step`
// already established.

/// One layer's router, on the device except for the part that is a decision.
///
/// `proj` is the projection, `[n_routed + n_shared, hidden]` however it is
/// oriented, and it is a matmul so it lives on the device. `bias` and
/// `global_scale` never multiply an
/// activation: the bias shifts the 256 scores used to PICK the top-k and takes
/// no part in the weights, and the global scale scales the weights themselves.
/// They are control plane, so they stay host, where the decision is made.
struct RouterDev {
    proj: RouterProj,
    /// `INK_ROUTER_DIFF=1` only: the f32 `[rows, hidden]` lane, held BESIDE the
    /// active one so the two selections can be compared on the same activation.
    /// It is `None` otherwise, so the ordinary run carries neither the weight
    /// nor the second matmul.
    reference: Option<T2>,
    bias: Vec<f32>,
    global_scale: f32,
}

/// What `INK_ROUTER_DIFF=1` counted for one layer, comparing the arm the run
/// ACTED ON against the f32 `[rows, hidden]` lane every earlier run shipped.
///
/// Top-k is discrete, so a changed logit CAN flip an expert, and that is a
/// behavioural change rather than a rounding one. Neither assuming it happens
/// nor assuming it does not is a measurement, so this counts it: per layer, per
/// token, with the examined count printed beside every other number so a zero
/// says how much was looked at.
///
/// The reference is computed, compared and DISCARDED. It never reaches
/// `by_expert`, so the run this instruments is the arm's own run and the
/// counters describe it rather than some third thing.
#[derive(Default, Clone, Copy)]
struct RouteDiff {
    /// Token-positions whose selection was compared.
    examined: usize,
    /// ...where the SET of chosen experts differs.
    set_differs: usize,
    /// ...where the set agrees but the order the top-k came out in does not.
    /// Harmless by itself -- the weights follow the experts -- but it is the
    /// near miss that says how close the ordering is to flipping.
    order_differs: usize,
    /// Individual (position, slot) pairs naming a different expert.
    slots_differ: usize,
    /// Largest `|active - reference|` over every logit compared.
    max_abs_logit: f32,
    /// Largest `|active - reference|` over the weights the chosen experts got.
    /// Only defined where the sets agree, since otherwise the weights are not
    /// the same quantity.
    max_abs_weight: f32,
}

impl RouteDiff {
    /// One token-position, both selections in hand.
    fn note(&mut self, a: &Routing, b: &Routing, la: &[f32], lb: &[f32]) {
        self.examined += 1;
        for (x, y) in la.iter().zip(lb) {
            self.max_abs_logit = self.max_abs_logit.max((x - y).abs());
        }
        let differ = a.experts.iter().zip(&b.experts).filter(|(x, y)| x != y).count();
        self.slots_differ += differ;
        let mut sa = a.experts.clone();
        let mut sb = b.experts.clone();
        sa.sort_unstable();
        sb.sort_unstable();
        if sa != sb {
            self.set_differs += 1;
        } else if differ != 0 {
            self.order_differs += 1;
        } else {
            // Slot for slot the same experts, so slot for slot the same
            // quantity. Anywhere else the two `weights` vectors are indexed by
            // different experts and subtracting them compares nothing.
            for (x, y) in a.weights.iter().zip(&b.weights) {
                self.max_abs_weight = self.max_abs_weight.max((x - y).abs());
            }
        }
    }
}

/// Drop a `[n, cols]` row-major block down to its first `keep` columns.
///
/// The BF16 arm's weight is padded to the `mma` instruction's n tile, so the
/// logits come back six columns wide of what the router asked for. Sliced HERE,
/// on the host, rather than with a device `slice`: on a decode step this is one
/// row of 264 floats and a device slice would be another kernel launch in the
/// one stage of the layer that is already launch-bound.
///
/// Returns its argument untouched when there is no pad, so the f32 arms pay
/// nothing for the BF16 arm's shape.
fn drop_pad_cols(v: Vec<f32>, n: usize, cols: usize, keep: usize) -> Vec<f32> {
    assert!(keep <= cols, "cannot keep {keep} of {cols} columns");
    assert_eq!(v.len(), n * cols, "{} values are not [{n}, {cols}]", v.len());
    if cols == keep {
        return v;
    }
    let mut out = Vec::with_capacity(n * keep);
    for t in 0..n {
        out.extend_from_slice(&v[t * cols..t * cols + keep]);
    }
    out
}

/// `[rows, cols]` row-major to `[cols, rows]` row-major.
///
/// The router's projection is uploaded once per layer and multiplied once per
/// token per layer for the whole run, so the orientation the matmul wants is
/// the orientation to store it in. This is that permutation, paid once on the
/// host at upload, where the device lane paid it on every call.
fn transpose_rows(v: &[f32], rows: usize, cols: usize) -> Vec<f32> {
    assert_eq!(v.len(), rows * cols, "{} values are not [{rows}, {cols}]", v.len());
    let mut out = vec![0f32; rows * cols];
    for r in 0..rows {
        for c in 0..cols {
            out[c * rows + r] = v[r * cols + c];
        }
    }
    out
}

/// Which router lane `INK_ROUTER` selected.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum RouterArm {
    /// `[rows, hidden]` f32, `.transpose()` on the device per call.
    Transpose,
    /// `[hidden, rows]` f32, transposed once on the host at upload.
    Pre,
    /// The BF16 the pile stores, into `mma.sync…bf16`. THE DEFAULT: it is the
    /// precision the model is in and the precision the reference computes in.
    Bf16,
}

impl RouterArm {
    /// `INK_ROUTER=transpose|pre|bf16`.
    ///
    /// `bf16` is the default, and the reason is precision rather than speed.
    /// Inkling is a bfloat16 model with NVFP4 experts, and the official
    /// implementation multiplies the router in bf16 on the tensor cores. The
    /// f32 the other two arms multiply in was never the model's: `gv` widened
    /// the pile's stored BF16 on the way out of the mapping, and that widening
    /// was OUR addition. So the arm that agrees with the reference is this one,
    /// and the f32 arms are the deviation.
    ///
    /// The previous round gated this arm on bitwise agreement with the widened
    /// f32 arm and the gate did not pass: on all eight `fp4_rep2` prompts the
    /// greedy continuation diverges, and 0.46% of 5048 router selections chose
    /// a different SET of six experts. Those numbers still stand and are still
    /// worth having -- they are a measured property of NVFP4 routing, which is
    /// how close the 6th and 7th expert scores sit -- but they are not evidence
    /// against this arm. Demanding that a bf16 model reproduce an f32 lane's
    /// exact top-k demands a precision the model does not have, which is the
    /// same error the deleted f32 CPU reference made.
    ///
    /// Both f32 arms stay reachable so the comparison stays runnable.
    fn from_env() -> Self {
        match std::env::var("INK_ROUTER").as_deref() {
            Ok("transpose") => RouterArm::Transpose,
            Ok("pre") => RouterArm::Pre,
            Ok("bf16") | Err(_) => RouterArm::Bf16,
            Ok(other) => panic!("INK_ROUTER={other:?} is not one of: transpose, pre, bf16"),
        }
    }

    fn label(self) -> &'static str {
        match self {
            RouterArm::Transpose => "f32 [rows,hidden], transposed on the device PER CALL",
            RouterArm::Pre => "f32 [hidden,rows], transposed ONCE on the host at upload",
            RouterArm::Bf16 => "the STORED BF16, into mma.sync...bf16, nothing widened",
        }
    }
}

/// The router projection, in the orientation this run multiplies it in.
///
/// One enum rather than a flag because the two arms hold DIFFERENT TENSORS, not
/// the same tensor used differently: only the arm the run selected is uploaded,
/// so switching arms cannot leave a second copy of every layer's projection on
/// the device unnoticed.
enum RouterProj {
    /// `[rows, hidden]`, the checkpoint's own orientation, transposed on the
    /// device on EVERY call. `INK_ROUTER=transpose`, and what every run before
    /// this one did.
    PerCall(T2),
    /// `[hidden, rows]`, transposed ONCE on the host at upload.
    Pre(T2),
    /// The pile's own BF16 bytes, `[rows, hidden]`, into `mma.sync…bf16` — the
    /// same lane 71b837b put the attention projections on, and no f32 copy of
    /// the weight exists anywhere. The default.
    ///
    /// Inkling is a bfloat16 model with NVFP4 experts. The f32 the two arms
    /// above multiply in is not the model's precision, it is ours: the widen
    /// happened on the host on the way out of the mapping, doubled the bytes,
    /// and bought a precision the checkpoint does not have. This arm also casts
    /// the ACTIVATION to BF16, by the hardware's round-to-nearest-even, which is
    /// what the official implementation's `bfloat16` linear does. So the logits
    /// differ from the f32 arms' and are not meant not to; what has to hold is
    /// the SELECTION, and `INK_ROUTER_DIFF=1` below counts whether it does.
    ///
    /// `n` is the row count PADDED to the instruction's n tile: 258 is not a
    /// multiple of 8, so six zero rows are appended and their logits sliced off
    /// on the host. That pad is why this arm copies rather than aliases -- a
    /// padded buffer is not inside the pile's mapping -- but the copy is at the
    /// STORED width. Nothing here widens; 2.16 MB a layer moves once at upload
    /// where the f32 arms materialised 4.23 MB a layer and uploaded that. The
    /// bind counters see it as exactly eight more `UNMAPPED` copies and 16 MiB.
    Bf16 {
        w: Bf16W,
        /// Real rows, `n_routed + n_shared`, before the pad.
        rows: usize,
    },
}

/// Everything one layer multiplies by, in DEVICE memory, for the whole run.
///
/// This used to be two attention tensors in a map and everything else read out
/// of the host residency cache per token: `attn_norm`, `mlp_norm` and
/// `mlp_sconv` were `Vec<f32>` because the operations consuming them were host
/// operations. They are not any more. A weight the device multiplies by lives
/// on the device, and the host copy is dropped after the upload rather than
/// held beside it -- holding both doubles a budget for no gain, since after the
/// upload nothing on the host reads them again.
struct LayerDev {
    attn: dev_lane::AttnWeightsDev,
    attn_sconv: T2,
    mlp_sconv: T2,
    attn_norm: BT<Bk, 1>,
    mlp_norm: BT<Bk, 1>,
    /// `None` on a dense layer, which has no experts to route to.
    router: Option<RouterDev>,
}

/// Every routed expert for one layer, on the NATIVE NVFP4 tensor-core path.
///
/// # What a routed MoE block computes
///
/// This is written out because it used to be written out somewhere else. A
/// scalar f32 host transcription (`mlp::routed_experts`, over
/// `mlp::expert_ffn_one`) served as the readable statement of the algorithm and
/// was deleted; it had no caller in the data plane, and being the legible
/// version of a hot function is exactly what made people optimise it instead of
/// this. So the legibility moves here, to the code that runs.
///
/// The router has already chosen, per token, `top_k` experts and a weight for
/// each. `by_expert` is that decision INVERTED — keyed by expert, listing
/// `(token row, routing weight)` — because a weight is 12.6 MB and a token row
/// is 16 KB, so the loop that must not be repeated is the one over weights.
/// Each expert then computes, for every token routed to it:
///
/// ```text
/// both = x  · w13ᵀ            w13 is [2 * intermediate, hidden], GATE rows first
/// act  = silu(both[..I]) * both[I..]                       I = intermediate
/// y    = act · w2ᵀ            w2  is [hidden, intermediate]
/// out[token] += y * weight
/// ```
///
/// Four things in that are easy to get wrong and are each pinned somewhere:
///
/// * the gate half comes FIRST in `w13`, and the checkpoint stores the two
///   halves INTERLEAVED (`g0, u0, g1, u1, …`). `2 * intermediate == hidden` in
///   both releases, so the fused matrix is square and a transposed or
///   un-deinterleaved reading loads without complaint and computes nonsense.
///   The de-interleave happens once, at import, and `inkling_fp4_expert_gate`
///   holds one whole real expert to an f64 arbiter over the same operands.
/// * the routing weight multiplies the expert's OUTPUT, once, after `w2` — not
///   the activation, and not the input.
/// * a token's contributions from its `top_k` experts are SUMMED, so the
///   scatter-add below is an add and not a write.
/// * the shared experts do NOT read this function's output. They run beside it
///   on the same normed input — see [`shared_experts_bf16`].
///
/// # This lane, and the two it replaced
///
/// The only routed lane there is. The packed bytes go straight into
/// `mma.sync…kind::mxf4nvf4…ue4m3`. The device lane this replaced
/// (`INK_EXPERTS=gpu`) decoded each expert into a 67.1 + 33.6 MB f32 pair,
/// multiplied THAT, and dropped it — 100 MB of device memory materialised per
/// expert to hold a weight the pile stores in 12.6, four times a token per
/// layer. It is gone rather than kept as a control, because the control was a
/// widening and the whole point of an NVFP4 model is that the weight is never
/// widened. The host lane is gone for the sibling reason: an f32 reference
/// demands agreement at a precision a bfloat16 model with 4-bit weights does not
/// have, so "the device matches the host" would have been a statement about
/// which of the two was written first.
///
/// Activations are quantised to E2M1 in dynamic per-16 blocks with E4M3
/// scales, which the instruction requires and which is what the checkpoint's
/// own `hf_quant_config.json` specifies for `*input_quantizer`.
///
/// # Device in, device out
///
/// `hn` is the normed residual stream ON THE DEVICE and the return is the
/// accumulated expert output ON THE DEVICE. It used to be `&[f32]` and
/// `Vec<f32>`, and the two conversions that implied were not free bookkeeping:
/// the gather copied each expert's rows out of a host buffer, and the drain
/// BLOCKED on `read_one` per expert to get 16 KB back so the host could do a
/// weighted add. Now `select` gathers, `select_assign` scatter-adds, and
/// nothing in the layer waits.
///
/// The accumulation order is unchanged and deliberately so: `by_expert` is a
/// `BTreeMap`, the scatter-adds are issued in its order, and each token appears
/// at most once per expert, so `acc` is the same sum in the same order the host
/// loop made. That is what keeps this a plumbing change rather than a numerics
/// one.
#[allow(clippy::too_many_arguments)]
fn routed_experts_fp4(
    src: &Weights,
    aliases: Option<&mary::models::inkling::fp4gemm::Aliases>,
    client: &cubecl::prelude::ComputeClient<cubecl::cuda::CudaRuntime>,
    dev: &burn::backend::cuda::CudaDevice,
    prefix: &str,
    by_expert: &BTreeMap<usize, Vec<(usize, f32)>>,
    hn: &T2,
    n: usize,
    h: usize,
    inter: usize,
    host: &mut HostT,
) -> Result<T2> {
    // Which of the two lanes runs the layer. The grouped one computes the same
    // thing with one launch per STAGE instead of one sequence per EXPERT, and
    // it returns `None` exactly when its premise -- every active expert's
    // weight is an offset into one registered mapping -- does not hold.
    //
    // `INK_GROUPED=0` takes the per-expert loop, which is how the two are held
    // to the same bits over a whole run; `INK_GROUPED=2` runs BOTH per layer
    // and prints where they part company, which is how a disagreement gets
    // located instead of argued about.
    let mode = std::env::var("INK_GROUPED").unwrap_or_else(|_| "1".to_string());
    if mode != "0" {
        if let Some(al) = aliases {
            if let Some(acc) =
                grouped_experts_fp4(src, al, client, dev, prefix, by_expert, hn, n, h, inter, host)?
            {
                if mode == "2" {
                    let reference = per_expert_fp4(
                        src, aliases, client, dev, prefix, by_expert, hn, n, h, inter, host,
                    )?;
                    report_ab(prefix, &acc, &reference, h);
                }
                host.grouped += 1;
                return Ok(acc);
            }
        }
    }
    host.per_expert += 1;
    per_expert_fp4(src, aliases, client, dev, prefix, by_expert, hn, n, h, inter, host)
}

/// Where the two lanes' accumulators differ, and by how much.
///
/// A MEASUREMENT, not a verdict, and that distinction is the whole point. The
/// grouped lane fuses the routing-weight multiply into the accumulating add and
/// the per-expert lane does not, so the two round in different PLACES and a gap
/// of order an ulp is EXPECTED here -- see
/// [`mary::models::inkling::moegroup`], and `91f81b4` for the time this tree
/// mistook that expectation for a defect and paid a launch to hide it.
///
/// What the number is for is the SIZE and SHAPE of the gap. An ulp-scale
/// difference on a fraction of the elements is the fused multiply. Anything
/// larger, anything structured, or any difference at all in the routing, the
/// gather, the operand order or the accumulation order is a real defect --
/// those are bit-exact and stay that way.
///
/// Compared as BITS, so a `-0.0` against a `0.0` counts and a NaN is not
/// silently equal to itself.
fn report_ab(prefix: &str, a: &T2, b: &T2, h: usize) {
    let av = down(a.clone());
    let bv = down(b.clone());
    let mut differ = 0usize;
    let mut worst = 0.0f32;
    let mut worst_rel = 0.0f32;
    let mut first: Option<(usize, usize, f32, f32)> = None;
    for i in 0..av.len() {
        if av[i].to_bits() == bv[i].to_bits() {
            continue;
        }
        differ += 1;
        let d = (av[i] - bv[i]).abs();
        worst = worst.max(d);
        let scale = av[i].abs().max(bv[i].abs()).max(f32::MIN_POSITIVE);
        worst_rel = worst_rel.max(d / scale);
        if first.is_none() {
            first = Some((i / h, i % h, av[i], bv[i]));
        }
    }
    let ln = prefix.trim_end_matches('.');
    match first {
        None => println!("  [A/B] {ln}: {} elements, ALL BITS EQUAL", av.len()),
        Some((r, c, x, y)) => println!(
            "  [A/B] {ln}: rounding gap on {differ} of {}, max abs {worst:.3e} rel {worst_rel:.3e}, first row {r} col {c}: grouped {x:.9e} per-expert {y:.9e}",
            av.len()
        ),
    }
}

/// Every routed expert for one layer in a handful of launches, or `None` if
/// this lane cannot take the layer.
///
/// Same weights, same kernels, same arithmetic and -- by construction rather
/// than by hope -- the same accumulation order as [`per_expert_fp4`]. See
/// [`mary::models::inkling::moegroup`] for why the last of those had to be
/// designed for. What differs is who walks the expert list: the host did, and
/// now `CUBE_POS_Y` does.
///
/// # What makes it possible, and when it is not
///
/// The kernel reaches every active expert's weight through ONE bound buffer
/// plus a table of byte offsets, so the whole layer's weights have to live in a
/// single registered mapping. A pile is one file and the zero-copy seam
/// registers it once, so that is the ordinary case. `INK_ZEROCOPY=0`, a device
/// that cannot address host memory, or a slab whose offset does not land on the
/// 4-byte vector the packed plane is read in -- each of those is `None`, and
/// the per-expert loop runs the layer instead. That is not a comfort blanket
/// kept beside a better lane: it is the only lane there is when this one's
/// premise fails, and the per-pass report counts which ran.
#[allow(clippy::too_many_arguments)]
fn grouped_experts_fp4(
    src: &Weights,
    al: &mary::models::inkling::fp4gemm::Aliases,
    client: &cubecl::prelude::ComputeClient<cubecl::cuda::CudaRuntime>,
    dev: &burn::backend::cuda::CudaDevice,
    prefix: &str,
    by_expert: &BTreeMap<usize, Vec<(usize, f32)>>,
    hn: &T2,
    n: usize,
    h: usize,
    inter: usize,
    host: &mut HostT,
) -> Result<Option<T2>> {
    use mary::models::inkling::fp4gemm::gate_up_silu_launch;
    use mary::models::inkling::fp4quant::quantize_nvfp4;
    use mary::models::inkling::moegroup::{
        fp4_linear_grouped_launch, gather_grouped, scatter_weighted, BlockPlanDev, RowPlan,
    };
    use mary::models::inkling::seam::{handle_of, tensor_of};

    let n13 = format!("{prefix}mlp.experts.w13_weight");
    let n2 = format!("{prefix}mlp.experts.w2_weight");
    let slots = by_expert.len();
    if slots == 0 {
        return Ok(None);
    }

    // Where every plane of every active expert lives, as a byte offset into the
    // one mapping they must share. This IS the bind, and it is arithmetic.
    let t_s = Instant::now();
    let mut off13: Vec<u64> = Vec::with_capacity(2 * slots);
    let mut off2: Vec<u64> = Vec::with_capacity(2 * slots);
    let mut sc13: Vec<f32> = Vec::with_capacity(slots);
    let mut sc2: Vec<f32> = Vec::with_capacity(slots);
    let mut plane_bytes: Vec<usize> = Vec::with_capacity(4 * slots);
    let mut which: Option<usize> = None;
    for &e in by_expert.keys() {
        let w13 = src.expert_packed(&n13, e)?;
        let w2 = src.expert_packed(&n2, e)?;
        let planes: [&[u8]; 4] = [&w13.codes, &w13.scales, &w2.codes, &w2.scales];
        let mut o = [0u64; 4];
        for (i, plane) in planes.into_iter().enumerate() {
            match al.locate(plane) {
                Some((m, byte)) if which.map_or(true, |w| w == m) => {
                    which = Some(m);
                    o[i] = byte;
                }
                _ => return Ok(None),
            }
            plane_bytes.push(plane.len());
        }
        // The packed planes are read as 4-byte vectors out of the mapping, so
        // an offset that does not land on one cannot be expressed as an index.
        // The SCALE planes are read the same way now -- the instruction takes
        // its four E4M3 block scales as one 32-bit register, so the kernel
        // fetches them as one -- which puts `o[1]` and `o[3]` under the same
        // rule. The startup copy packs every view to a 4-byte boundary, so this
        // refuses nothing it did not already refuse; it is here because the
        // kernel is launched unchecked and an unaligned scale plane would be
        // read one vector to the left of where it starts.
        if o.iter().any(|v| v % 4 != 0) {
            return Ok(None);
        }
        off13.push(o[0]);
        off13.push(o[1]);
        off2.push(o[2]);
        off2.push(o[3]);
        sc13.push(w13.scale2);
        sc2.push(w2.scale2);
    }
    let (wmap, wmap_bytes) = match which.and_then(|i| al.map(i)) {
        Some(m) => m,
        None => return Ok(None),
    };
    // Charged only now that the lane is committed: a probe that turned back
    // moved nothing and must not report that it did.
    for b in plane_bytes {
        al.note_alias(b);
    }
    host.slice += t_s.elapsed().as_secs_f64();

    // The row plan, and the only host->device traffic in the layer: `[M]`
    // indices and weights, `[M/16]` slots, `[2*slots]` offsets, `[slots]`
    // second-level scales and the token->rows table. Nine small uploads for the
    // whole layer, against two per expert before.
    let t_g = Instant::now();
    let plan = RowPlan::build(by_expert.values(), n, RowPlan::planes());
    let m_total = plan.m_total();
    let hn_h = handle_of(hn.clone());
    let h_rowtok = client.create_from_slice(bytes_of(&plan.row_tok));
    let h_rowwgt = client.create_from_slice(bytes_of(&plan.row_wgt));
    let blk = BlockPlanDev {
        slot: client.create_from_slice(bytes_of(&plan.blk_slot)),
        tile0: client.create_from_slice(bytes_of(&plan.blk_tile0)),
        cnt: client.create_from_slice(bytes_of(&plan.blk_cnt)),
        blocks: plan.blk_slot.len(),
        planes: RowPlan::planes(),
    };
    let h_off13 = client.create_from_slice(bytes_of(&off13));
    let h_off2 = client.create_from_slice(bytes_of(&off2));
    let h_sc13 = client.create_from_slice(bytes_of(&sc13));
    let h_sc2 = client.create_from_slice(bytes_of(&sc2));
    let h_tokrows = client.create_from_slice(bytes_of(&plan.tok_rows));
    let h_tokcnt = client.create_from_slice(bytes_of(&plan.tok_cnt));
    let x_h = gather_grouped(client, &hn_h, &h_rowtok, n, m_total, h);
    host.gather += t_g.elapsed().as_secs_f64();

    let t_w = Instant::now();
    let (a, asc) = quantize_nvfp4(client, &x_h, m_total, h);
    let both = fp4_linear_grouped_launch(
        client, &a, &asc, &wmap, wmap_bytes, &blk, &h_off13, &h_sc13, slots, m_total, h,
        2 * inter,
    );
    let act_h = gate_up_silu_launch(client, &both, m_total, inter);
    let (a2, asc2) = quantize_nvfp4(client, &act_h, m_total, inter);
    let y_h = fp4_linear_grouped_launch(
        client, &a2, &asc2, &wmap, wmap_bytes, &blk, &h_off2, &h_sc2, slots, m_total, inter, h,
    );
    host.enqueue += t_w.elapsed().as_secs_f64();

    let t_c = Instant::now();
    let acc_h = scatter_weighted(
        client, &y_h, &h_rowwgt, &h_tokrows, &h_tokcnt, &h_rowtok, m_total, n, h, plan.kmax,
    );
    let acc = tensor_of(client.clone(), dev.clone(), acc_h, n, h);
    host.accum += t_c.elapsed().as_secs_f64();

    Ok(Some(acc))
}

/// A slice of POD as the bytes `create_from_slice` uploads.
fn bytes_of<T: cubecl::prelude::CubeElement>(v: &[T]) -> &[u8] {
    T::as_bytes(v)
}

/// The per-expert lane: one launch sequence per active expert, in `BTreeMap`
/// order, which is the order the accumulation is defined by.
#[allow(clippy::too_many_arguments)]
fn per_expert_fp4(
    src: &Weights,
    aliases: Option<&mary::models::inkling::fp4gemm::Aliases>,
    client: &cubecl::prelude::ComputeClient<cubecl::cuda::CudaRuntime>,
    dev: &burn::backend::cuda::CudaDevice,
    prefix: &str,
    by_expert: &BTreeMap<usize, Vec<(usize, f32)>>,
    hn: &T2,
    n: usize,
    h: usize,
    inter: usize,
    host: &mut HostT,
) -> Result<T2> {
    use mary::models::inkling::fp4gemm::{fp4_linear_launch, gate_up_silu_launch, MTILE};
    use mary::models::inkling::fp4quant::quantize_nvfp4;
    use mary::models::inkling::pad::gather_rows_pad;
    use mary::models::inkling::seam::{handle_of, int_handle_of, tensor_of};

    // Zero copy where the hardware allows it: the GPU reads the source's own
    // mapped pages in place. The mappings were registered ONCE at startup, so
    // this is offset arithmetic on a pointer, not a device round trip.
    let bind = |data: &[u8]| match aliases {
        Some(al) => al.slice_or_copy(client, data),
        None => client.create_from_slice(data),
    };

    let n13 = format!("{prefix}mlp.experts.w13_weight");
    let n2 = format!("{prefix}mlp.experts.w2_weight");
    let mut acc: T2 = burn::tensor::Tensor::zeros([n, h], dev);
    // The residual stream's buffer, taken once: every expert gathers out of the
    // same rows and re-deriving the handle per expert would be six clones of a
    // refcount for nothing.
    let hn_h = handle_of(hn.clone());

    for (&e, toks) in by_expert {
        let t_s = Instant::now();
        let w13 = src.expert_packed(&n13, e)?;
        let w2 = src.expert_packed(&n2, e)?;
        host.slice += t_s.elapsed().as_secs_f64();

        let t_g = Instant::now();
        let m = toks.len();
        let m_pad = m.div_ceil(MTILE) * MTILE;
        let (idx, wgt) = expert_rows::<Bk>(toks, dev);
        // One kernel reads this expert's rows out of the residual stream where
        // they already lie and writes them into the `[m_pad, hidden]` buffer the
        // MMA wants, zeros and all. It was a `select`, a `zeros` and a `cat` —
        // four launches to move one row on a decode step.
        let x_h = gather_rows_pad(client, &hn_h, &int_handle_of(idx.clone()), n, m, m_pad, h);
        host.gather += t_g.elapsed().as_secs_f64();

        let t_w = Instant::now();
        // Quantise on the DEVICE, both times. The host lane this replaces had
        // to bring the intermediate activation back across the bus between the
        // two GEMMs purely to requantise it; `act_h` never leaves the device.
        let (a, asc) = quantize_nvfp4(client, &x_h, m_pad, h);

        let (b, bsc) = (bind(&w13.codes), bind(&w13.scales));
        let both = fp4_linear_launch(client, &a, &asc, &b, &bsc, m_pad, h, 2 * inter, w13.scale2);

        let act_h = gate_up_silu_launch(client, &both, m_pad, inter);
        let (a2, asc2) = quantize_nvfp4(client, &act_h, m_pad, inter);

        let (b2, bsc2) = (bind(&w2.codes), bind(&w2.scales));
        let y_h = fp4_linear_launch(client, &a2, &asc2, &b2, &bsc2, m_pad, inter, h, w2.scale2);
        host.enqueue += t_w.elapsed().as_secs_f64();

        let t_c = Instant::now();
        let y = tensor_of(client.clone(), dev.clone(), y_h, m_pad, h);
        let y = y.slice([0..m, 0..h]) * wgt;
        acc = acc.select_assign(0, idx, y, burn::tensor::IndexingUpdateOp::Add);
        host.accum += t_c.elapsed().as_secs_f64();
    }
    Ok(acc)
}

/// The same lane for layer 2, whose experts are BF16.
///
/// Deliberately the same shape as [`routed_experts_fp4`], line for line — and
/// that doc comment is where the algorithm is written down, since the host
/// transcription that used to hold it is deleted. The
/// same `select` gather, the same pointer-containment binding, the same
/// `select_assign` in `BTreeMap` order. What the format takes away is all that
/// differs — no block scales to bind, no `scale2` to fold in, no activation
/// quantiser. What it puts back is one cast: the MMA takes the same type on
/// both operands, so the f32 residual stream is rounded to BF16 on the device
/// before it enters. That is not a liberty — the reference implementation runs
/// this layer in BF16 throughout, so a BF16 activation is what `transformers`
/// multiplies too.
#[allow(clippy::too_many_arguments)]
fn routed_experts_bf16(
    src: &Weights,
    aliases: Option<&mary::models::inkling::fp4gemm::Aliases>,
    client: &cubecl::prelude::ComputeClient<cubecl::cuda::CudaRuntime>,
    dev: &burn::backend::cuda::CudaDevice,
    prefix: &str,
    by_expert: &BTreeMap<usize, Vec<(usize, f32)>>,
    hn: &T2,
    n: usize,
    h: usize,
    inter: usize,
    host: &mut HostT,
) -> Result<T2> {
    // Same dispatch as the packed lane, and it earns its keep on the same
    // measurement: eight grouped NVFP4 layers cost the host 0.8 ms a pass and
    // this ONE layer, still looping, cost about eight.
    let mode = std::env::var("INK_GROUPED").unwrap_or_else(|_| "1".to_string());
    if mode != "0" {
        if let Some(al) = aliases {
            if let Some(acc) = grouped_experts_bf16(
                src, al, client, dev, prefix, by_expert, hn, n, h, inter, host,
            )? {
                if mode == "2" {
                    let reference = per_expert_bf16(
                        src, aliases, client, dev, prefix, by_expert, hn, n, h, inter, host,
                    )?;
                    report_ab(prefix, &acc, &reference, h);
                }
                host.grouped += 1;
                return Ok(acc);
            }
        }
    }
    host.per_expert += 1;
    per_expert_bf16(src, aliases, client, dev, prefix, by_expert, hn, n, h, inter, host)
}

/// Layer 2's routed experts in a handful of launches, or `None` if this lane
/// cannot take the layer.
///
/// [`grouped_experts_fp4`] with the format's differences and no others: one
/// weight plane per expert instead of two, no second-level scale to carry, and
/// a cast in place of the activation quantiser. The offsets are BF16 elements
/// because that is the unit the unscaled MMA indexes its B operand in.
#[allow(clippy::too_many_arguments)]
fn grouped_experts_bf16(
    src: &Weights,
    al: &mary::models::inkling::fp4gemm::Aliases,
    client: &cubecl::prelude::ComputeClient<cubecl::cuda::CudaRuntime>,
    dev: &burn::backend::cuda::CudaDevice,
    prefix: &str,
    by_expert: &BTreeMap<usize, Vec<(usize, f32)>>,
    hn: &T2,
    n: usize,
    h: usize,
    inter: usize,
    host: &mut HostT,
) -> Result<Option<T2>> {
    use mary::models::inkling::bf16gemm::to_bf16_launch;
    use mary::models::inkling::fp4gemm::gate_up_silu_bf16_launch;
    use mary::models::inkling::moegroup::{
        bf16_linear_grouped_launch, gather_grouped, scatter_weighted, BlockPlanDev, RowPlan,
    };
    use mary::models::inkling::seam::{handle_of, tensor_of};

    let n13 = format!("{prefix}mlp.experts.w13_weight");
    let n2 = format!("{prefix}mlp.experts.w2_weight");
    let slots = by_expert.len();
    if slots == 0 {
        return Ok(None);
    }

    let t_s = Instant::now();
    let mut off13: Vec<u64> = Vec::with_capacity(slots);
    let mut off2: Vec<u64> = Vec::with_capacity(slots);
    let mut plane_bytes: Vec<usize> = Vec::with_capacity(2 * slots);
    let mut which: Option<usize> = None;
    for &e in by_expert.keys() {
        let w13 = src.expert_bf16(&n13, e)?;
        let w2 = src.expert_bf16(&n2, e)?;
        let planes: [&[u8]; 2] = [&w13.bytes, &w2.bytes];
        let mut o = [0u64; 2];
        for (i, plane) in planes.into_iter().enumerate() {
            match al.locate(plane) {
                Some((m, byte)) if which.map_or(true, |w| w == m) => {
                    which = Some(m);
                    o[i] = byte;
                }
                _ => return Ok(None),
            }
            plane_bytes.push(plane.len());
        }
        // Read as two BF16 per 32-bit vector, so the offset has to be a whole
        // number of them -- which is the same 4-byte rule the alias predicate
        // already applies to the pointer.
        if o[0] % 4 != 0 || o[1] % 4 != 0 {
            return Ok(None);
        }
        off13.push(o[0] / 2);
        off2.push(o[1] / 2);
    }
    let (wmap, wmap_bytes) = match which.and_then(|i| al.map(i)) {
        Some(m) => m,
        None => return Ok(None),
    };
    for b in plane_bytes {
        al.note_alias(b);
    }
    host.slice += t_s.elapsed().as_secs_f64();

    let t_g = Instant::now();
    let plan = RowPlan::build(by_expert.values(), n, RowPlan::planes());
    let m_total = plan.m_total();
    let hn_h = handle_of(hn.clone());
    let h_rowtok = client.create_from_slice(bytes_of(&plan.row_tok));
    let h_rowwgt = client.create_from_slice(bytes_of(&plan.row_wgt));
    let blk = BlockPlanDev {
        slot: client.create_from_slice(bytes_of(&plan.blk_slot)),
        tile0: client.create_from_slice(bytes_of(&plan.blk_tile0)),
        cnt: client.create_from_slice(bytes_of(&plan.blk_cnt)),
        blocks: plan.blk_slot.len(),
        planes: RowPlan::planes(),
    };
    let h_off13 = client.create_from_slice(bytes_of(&off13));
    let h_off2 = client.create_from_slice(bytes_of(&off2));
    let h_tokrows = client.create_from_slice(bytes_of(&plan.tok_rows));
    let h_tokcnt = client.create_from_slice(bytes_of(&plan.tok_cnt));
    let x_h = gather_grouped(client, &hn_h, &h_rowtok, n, m_total, h);
    host.gather += t_g.elapsed().as_secs_f64();

    let t_w = Instant::now();
    let a = to_bf16_launch(client, &x_h, m_total * h, m_total * h);
    let both = bf16_linear_grouped_launch(
        client, &a, &wmap, wmap_bytes, &blk, &h_off13, slots, m_total, h, 2 * inter,
    );
    let act = gate_up_silu_bf16_launch(client, &both, m_total, inter);
    let y_h = bf16_linear_grouped_launch(
        client, &act, &wmap, wmap_bytes, &blk, &h_off2, slots, m_total, inter, h,
    );
    host.enqueue += t_w.elapsed().as_secs_f64();

    let t_c = Instant::now();
    let acc_h = scatter_weighted(
        client, &y_h, &h_rowwgt, &h_tokrows, &h_tokcnt, &h_rowtok, m_total, n, h, plan.kmax,
    );
    let acc = tensor_of(client.clone(), dev.clone(), acc_h, n, h);
    host.accum += t_c.elapsed().as_secs_f64();

    Ok(Some(acc))
}

/// The per-expert BF16 lane: one launch sequence per active expert, in
/// `BTreeMap` order.
#[allow(clippy::too_many_arguments)]
fn per_expert_bf16(
    src: &Weights,
    aliases: Option<&mary::models::inkling::fp4gemm::Aliases>,
    client: &cubecl::prelude::ComputeClient<cubecl::cuda::CudaRuntime>,
    dev: &burn::backend::cuda::CudaDevice,
    prefix: &str,
    by_expert: &BTreeMap<usize, Vec<(usize, f32)>>,
    hn: &T2,
    n: usize,
    h: usize,
    inter: usize,
    host: &mut HostT,
) -> Result<T2> {
    use mary::models::inkling::bf16gemm::{bf16_linear_launch, to_bf16_launch, MTILE};
    use mary::models::inkling::fp4gemm::gate_up_silu_bf16_launch;
    use mary::models::inkling::pad::gather_rows_pad;
    use mary::models::inkling::seam::{handle_of, int_handle_of, tensor_of};

    let bind = |data: &[u8]| match aliases {
        Some(al) => al.slice_or_copy(client, data),
        None => client.create_from_slice(data),
    };

    let n13 = format!("{prefix}mlp.experts.w13_weight");
    let n2 = format!("{prefix}mlp.experts.w2_weight");
    let mut acc: T2 = burn::tensor::Tensor::zeros([n, h], dev);
    let hn_h = handle_of(hn.clone());

    for (&e, toks) in by_expert {
        let t_s = Instant::now();
        let w13 = src.expert_bf16(&n13, e)?;
        let w2 = src.expert_bf16(&n2, e)?;
        host.slice += t_s.elapsed().as_secs_f64();

        let t_g = Instant::now();
        let m = toks.len();
        let m_pad = m.div_ceil(MTILE) * MTILE;
        let (idx, wgt) = expert_rows::<Bk>(toks, dev);
        let x_h = gather_rows_pad(client, &hn_h, &int_handle_of(idx.clone()), n, m, m_pad, h);
        host.gather += t_g.elapsed().as_secs_f64();

        let t_w = Instant::now();
        let a = to_bf16_launch(client, &x_h, m_pad * h, m_pad * h);
        let b = bind(&w13.bytes);
        let both = bf16_linear_launch(client, &a, &b, m_pad, h, 2 * inter);

        // The intermediate never leaves the device and never becomes f32 on the
        // host: `gate_up_silu` writes BF16 straight into the second MMA's A
        // operand.
        let act = gate_up_silu_bf16_launch(client, &both, m_pad, inter);
        let b2 = bind(&w2.bytes);
        let y_h = bf16_linear_launch(client, &act, &b2, m_pad, inter, h);
        host.enqueue += t_w.elapsed().as_secs_f64();

        let t_c = Instant::now();
        let y = tensor_of(client.clone(), dev.clone(), y_h, m_pad, h);
        let y = y.slice([0..m, 0..h]) * wgt;
        acc = acc.select_assign(0, idx, y, burn::tensor::IndexingUpdateOp::Add);
        host.accum += t_c.elapsed().as_secs_f64();
    }
    Ok(acc)
}

/// One expert's `(token rows, routing weights)` as device tensors.
///
/// The only host->device traffic left in the routed lane, and it is `m` indices
/// and `m` floats — 8 bytes a token against the 14.2 MB of weight the same
/// expert is about to read. This is control plane crossing into the data plane,
/// which is the direction that is allowed.
fn expert_rows<B: Backend>(
    toks: &[(usize, f32)],
    dev: &B::Device,
) -> (BT<B, 1, burn::tensor::Int>, BT<B, 2>) {
    let idx: Vec<i32> = toks.iter().map(|&(ti, _)| ti as i32).collect();
    let w: Vec<f32> = toks.iter().map(|&(_, wgt)| wgt).collect();
    let m = toks.len();
    (
        BT::<B, 1, burn::tensor::Int>::from_data(BTD::new(idx, [m]), dev),
        BT::from_data(BTD::new(w, [m, 1]), dev),
    )
}

fn main() -> Result<()> {
    let pile_path = std::env::args().nth(1).map(PathBuf::from).context("usage: <pile> <ids> <out>")?;
    let ids_path = std::env::args().nth(2).map(PathBuf::from).context("usage: <pile> <ids> <out>")?;
    let out_path = std::env::args().nth(3).map(PathBuf::from).context("usage: <pile> <ids> <out>")?;

    // The weights, and everything else the run needs to know about the model.
    // There is no second arm: no `INK_PILE` to opt into, and no checkpoint
    // directory to fall back to.
    let pile_branch = std::env::var("INK_PILE_BRANCH").unwrap_or_else(|_| "inkling".to_string());
    let t_open = Instant::now();
    let mut cp = Weights::open(&pile_path, &pile_branch)
        .with_context(|| format!("opening {} on branch {pile_branch}", pile_path.display()))?;
    let open_secs = t_open.elapsed().as_secs_f64();

    // …and the config comes from the SAME source. It used to come from a
    // checkpoint directory unconditionally, which meant the pile could hold 159
    // GiB and the run still depended on the 40 KB beside it: a pile that cannot
    // answer this is not authoritative, only large. In a pile the config is
    // FACTS (one entity per JSON scalar, `mary::jsonfacts`), so this is a query,
    // not a stored file being read back.
    //
    // `INK_CONFIG=<file>` overrides it, LOUDLY, for one case: a pile written
    // before the sidecars were facts. That pile still holds every weight and is
    // still worth running, and the alternative — falling back to the checkpoint
    // directory when the pile has no config — is exactly the silent dependency
    // this change removes. An override you have to type is a different thing
    // from a fallback you never see.
    let cfg_source = std::env::var("INK_CONFIG").ok();
    let cfg_text = match &cfg_source {
        Some(p) => std::fs::read_to_string(p)
            .with_context(|| format!("INK_CONFIG={p}"))?,
        None => cp
            .document("config.json")
            .context(
                "this pile carries no config.json. Ingest the checkpoint's \
                 sidecars as facts (inkling_meta_gate <ckpt> <pile>), or point \
                 INK_CONFIG at the file to run without them",
            )?
            .to_string(),
    };
    let cfg = InklingConfig::from_json(&cfg_text).context("parsing config.json")?;
    let t = &cfg.text_config;

    // Which layers THIS process runs. REQUIRED, and a strict subrange.
    //
    // There is no default any more. Unset used to mean "the whole stack here",
    // which on a 119 GiB box against 144 GiB of weights means the page cache
    // evicts what the next token needs and every token pays real block-device
    // I/O — and a run that reads the SSD between tokens is not this model
    // running, it is a disk benchmark wearing its name.
    //
    // The bound is `hi - lo < num_hidden_layers`, NOT `nodes == 2`. Two is what
    // this model happens to need; the 66-layer sibling wants five to seven.
    // What is actually required is that no node carries the whole stack, and
    // that is what this says.
    let spec = std::env::var("INK_LAYERS").map_err(|_| {
        anyhow::anyhow!(
            "INK_LAYERS is required: this model does not fit one node ({} GiB of weights), so \
             every process runs a strict subrange and at least two nodes are needed.\n  \
             tail: INK_LAYERS=20:{}  INK_PIPE=tail:0.0.0.0:7654\n  \
             head: INK_LAYERS=0:20   INK_PIPE=head:<tail-host>:7654",
            144,
            t.num_hidden_layers,
        )
    })?;
    let (a, b) = spec.split_once(':').context("INK_LAYERS wants LO:HI")?;
    let (lo, hi) = (a.parse::<usize>()?, b.parse::<usize>()?);
    anyhow::ensure!(lo < hi, "INK_LAYERS wants LO < HI, got {lo}:{hi}");
    anyhow::ensure!(
        hi <= t.num_hidden_layers,
        "INK_LAYERS {lo}:{hi} runs past the {}-layer stack",
        t.num_hidden_layers
    );
    anyhow::ensure!(
        hi - lo < t.num_hidden_layers,
        "INK_LAYERS {lo}:{hi} is the whole {}-layer stack on one node, which does not fit. \
         Split it: no node may run every layer, and two is the MINIMUM rather than the number.",
        t.num_hidden_layers
    );
    // A pipe end is only a pipe end if it is not the whole stack; refusing the
    // contradiction here is cheaper than debugging a head that also unembeds.
    let pipe_spec = std::env::var("INK_PIPE").ok();
    let is_head = pipe_spec.as_deref().map(|s| s.starts_with("head:")).unwrap_or(false);
    let is_tail = pipe_spec.as_deref().map(|s| s.starts_with("tail:")).unwrap_or(false);
    anyhow::ensure!(
        pipe_spec.is_none() || is_head || is_tail,
        "INK_PIPE wants head:HOST:PORT or tail:ADDR:PORT"
    );
    anyhow::ensure!(
        !is_head || hi < t.num_hidden_layers,
        "a head that runs to the last layer has nothing to send; set INK_LAYERS"
    );
    anyhow::ensure!(!is_tail || lo > 0, "a tail that starts at layer 0 has nothing to receive");

    let mut ids: Vec<usize> = std::fs::read(&ids_path)?
        .chunks_exact(8)
        .map(|c| i64::from_le_bytes(c.try_into().unwrap()) as usize)
        .collect();
    let n = ids.len();
    anyhow::ensure!(n > 0, "no tokens — the forward would be vacuous");

    let h = t.hidden_size;
    println!("=== forward ===");
    println!(
        "  config     : {}",
        match &cfg_source {
            Some(p) => format!("INK_CONFIG={p}  (OVERRIDE -- the pile was not asked)"),
            None => format!("config.json from the pile ({})", pile_path.display()),
        }
    );
    println!(
        "  weights    : pile {}  (index built in {open_secs:.1}s)",
        pile_path.display(),
    );
    println!("  tokens     : {n}  {ids:?}");
    println!("  layers     : {}  hidden {h}  experts {}+{} shared",
             t.num_hidden_layers, t.n_routed_experts, t.n_shared_experts);
    println!(
        "  this process: layers {lo}..{hi} ({} of {}){}",
        hi - lo,
        t.num_hidden_layers,
        match (is_head, is_tail) {
            (true, _) => "  PIPE HEAD -- embeds, then sends the stream on",
            (_, true) => "  PIPE TAIL -- receives the stream, unembeds, argmax",
            _ => "  whole stack on one machine",
        }
    );
    // 968 dense leaves + 20 480 expert leaves. The safetensors index named
    // 1 360 tensors for the same model, because its expert entries are STACKS
    // of 256; the pile names each expert on its own, and that is the
    // granularity a layer split can partition and a deduplicating store can
    // address.
    println!("  {}", cp.inventory());
    println!("{}", mary::models::inkling::pile::mem_line("index built"));

    // How many tokens to generate past the prompt. 0 reproduces the original
    // single-forward behaviour exactly.
    let gen_steps: usize = std::env::var("INK_GEN")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(0);


    // Drafting runs on the TAIL, and only there. This used to refuse a pipe
    // outright -- "neither end can draft alone" -- which was true of the head
    // and false of the tail, and once INK_LAYERS became required it left NO
    // configuration in which acceptance could be measured at all. The tail
    // computes the last layer, so `x` at the draft site IS the whole stack's
    // final hidden state; it owns the final norm and the unembedding because it
    // takes the argmax; and it follows the sequence by recomputing that argmax
    // rather than being told, so its `ids` is the head's. The embedding table
    // was the only missing piece, and it is loaded below.
    //
    // `INK_MTP=k` drafts k tokens ahead with the MTP heads and scores them
    // against what the stack actually generates. It measures ACCEPTANCE, which
    // is the only oracle the composition has -- see mary::models::inkling::mtp.
    // Read here rather than beside the other lane switches because it decides
    // WHICH TABLES THIS PROCESS LOADS: drafting is the one configuration in
    // which a tail needs the embedding table, and the switch has to precede its
    // use.
    let mtp_k: usize = std::env::var("INK_MTP").ok().map(|v| v.parse()).transpose()?.unwrap_or(0);
    let mtp_order = match std::env::var("INK_MTP_ORDER") {
        Ok(v) => MtpConcat::parse(&v)
            .with_context(|| format!("INK_MTP_ORDER wants hidden|embed, got {v:?}"))?,
        Err(_) => MtpConcat::HiddenFirst,
    };
    // A head is still refused, and not out of caution: it holds neither the
    // final hidden state nor any way to turn one into a token.
    anyhow::ensure!(
        mtp_k == 0 || !is_head,
        "INK_MTP drafts on the TAIL, not the head: a head owns neither the stack's final hidden \
         state nor the unembedding, so it cannot turn a draft into a token. Set INK_MTP on the \
         tail process -- it loads the embedding table for exactly this -- and leave it unset here."
    );

    let started = Instant::now();
    // Hoisted: re-reading 4.8 GB of embedding tables per generated token would
    // dwarf everything else in the loop.
    //
    // Split by role, and NOT loaded by the end that will never read them. That
    // is not tidiness: the whole reason for two machines is that this model's
    // working set does not fit in one page cache, and 4.8 GB of embedding
    // pinned on the box that only unembeds is 4.8 GB of expert slabs evicted.
    //
    // Drafting is the ONE exception, and only on the tail. An MTP head takes the
    // stack's final hidden state and the embedding of the token one step ahead;
    // the tail already has the first of those (it computes the last layer), the
    // final norm and the unembedding, so the embedding table is the single
    // missing piece. `INK_MTP` on a tail buys exactly that piece back, and
    // nothing else changes about which end holds what. The embedding NORM stays
    // head-only: it belongs to the main stack's input, and every MTP head norms
    // its own embeddings with its own `embed_norm`.
    let want_embed = !is_tail || mtp_k > 0;
    let want_embed_norm = !is_tail;
    let want_head = !is_head;
    // The supported lane copies every weight this process can alias into one
    // anonymous allocation before registration. `INK_STARTUP_COPY=0` exists
    // only for the pressure reproducer: it deliberately restores the unsafe
    // file-backed alias so the gate can prove that pressure was present.
    if std::env::var("INK_STARTUP_COPY").map(|v| v == "0").unwrap_or(false) {
        println!("  startup weight copy: DISABLED (INK_STARTUP_COPY=0) -- UNSAFE diagnostic arm");
    } else {
        let mut globals = Vec::new();
        if want_embed {
            globals.push("model.llm.embed.weight");
        }
        if want_head {
            globals.push("model.llm.unembed.weight");
        }
        let t0 = Instant::now();
        let (experts, dense, bytes) = cp.copy_share(lo..hi, &globals)?;
        println!(
            "  startup weight copy: {experts} expert + {dense} dense views, {:.2} GiB anonymous in {:.1}s",
            bytes as f64 / GIB,
            t0.elapsed().as_secs_f64(),
        );
    }
    // The embedding table, as the BF16 the pile stores. `cp.held` turned 2.40
    // GiB of stored weight into 4.81 GB of host f32 and pinned it for the run,
    // on a box chosen because the working set only just fits -- and every token
    // read one row of it. `stored` hands back a view of the mapping; the
    // widening is now per LOOKUP, 16 KB a token, which is where it belongs.
    let embed_w = if want_embed {
        let leaf = cp.stored("model.llm.embed.weight")?;
        anyhow::ensure!(leaf.elem == Elem::Bf16, "the embedding table is {:?}", leaf.elem);
        Some(leaf.bytes.clone())
    } else {
        None
    };
    // `want_embed_norm`, NOT `want_embed`: a drafting tail takes the embedding
    // table and leaves the norm behind, because every MTP head norms its own
    // embeddings with its own `embed_norm`.
    let embed_n = if want_embed_norm { Some(cp.held("model.llm.embed_norm.weight")?) } else { None };
    let fnorm = if want_head { Some(cp.held("model.llm.norm.weight")?) } else { None };
    // The unembed table is NOT read here any more, and not widened anywhere.
    // It used to be `cp.held(...)`, which turned 1.6 GiB of stored BF16 into
    // 3.3 GB of host f32, uploaded THAT, and dropped the host copy. Now the
    // stored bytes are bound where they lie (see `unembed_w` below) and the
    // 3.3 GB never exists in either place.
    println!(
        "  embedding tables loaded in {:.1}s{}",
        started.elapsed().as_secs_f32(),
        if is_tail && mtp_k > 0 {
            "  (tail + INK_MTP: the embedding table is loaded here too, for drafting)"
        } else {
            ""
        }
    );

    // The MTP heads, only when asked for: eight dense blocks is 4.16 GiB, and a
    // run that is not drafting should not pay for them.
    let mtp_heads: Vec<MtpOwned> = if mtp_k > 0 {
        let t0 = Instant::now();
        let mut v = Vec::with_capacity(mtp_k);
        for i in 0..mtp_k {
            let p = format!("model.mtp.layers.{i}.transformer_block.");
            let g = |n: &str| cp.held(&format!("{p}{n}"));
            let fused = cp.held(&format!("{p}mlp.w13_dn.weight"))?;
            let (gate, up) = split_gate_up(&fused.data, h);
            let local = cfg.mtp_config.attn_kind(i) == AttnKind::Local;
            v.push(MtpOwned {
                embed_norm: cp.held(&format!("model.mtp.layers.{i}.embed_norm.weight"))?,
                hidden_norm: cp.held(&format!("model.mtp.layers.{i}.hidden_norm.weight"))?,
                input_proj: cp.held(&format!("model.mtp.layers.{i}.input_proj.weight"))?,
                attn_norm: g("attn_norm.weight")?,
                mlp_norm: g("mlp_norm.weight")?,
                attn_sconv: g("attn_sconv.weight")?,
                mlp_sconv: g("mlp_sconv.weight")?,
                wq: g("attn.wq_du.weight")?,
                wk: g("attn.wk_dv.weight")?,
                wv: g("attn.wv_dv.weight")?,
                wr: g("attn.wr_du.weight")?,
                wo: g("attn.wo_ud.weight")?,
                q_norm: g("attn.q_norm.weight")?,
                k_norm: g("attn.k_norm.weight")?,
                k_sconv: g("attn.k_sconv.weight")?,
                v_sconv: g("attn.v_sconv.weight")?,
                rel_proj: g("attn.rel_logits_proj.proj")?,
                gate,
                up,
                down: g("mlp.w2_md.weight")?,
                global_scale: g("mlp.global_scale")?.data[0],
                dims: AttnDims {
                    hidden: h,
                    heads: t.num_attention_heads,
                    kv_heads: t.num_key_value_heads,
                    head_dim: t.head_dim,
                    d_rel: t.d_rel,
                    // The same span rule the LLM's layers use, asked of the
                    // config rather than restated -- an MTP block's kind comes
                    // from mtp_config, but what a kind REACHES is one fact.
                    rel_extent: t.rel_span(if local { AttnKind::Local } else { AttnKind::Global }),
                    kernel: t.sconv_kernel_size,
                    rms_eps: t.rms_norm_eps,
                    kind: if local { AttnKind::Local } else { AttnKind::Global },
                },
                local,
            });
        }
        println!(
            "  MTP heads          : {mtp_k} loaded in {:.1}s, concat {} ({})",
            t0.elapsed().as_secs_f32(),
            mtp_order.name(),
            if mtp_order == MtpConcat::HiddenFirst {
                "what mtp_hidden_states_first reads as"
            } else {
                "the alternative reading"
            }
        );
        v
    } else {
        Vec::new()
    };
    // Drafts waiting to be scored: (the step whose token they predict, how many
    // heads deep, the token). Scored by DRAINING on the step they name, so a
    // draft is compared against what the stack actually produced rather than
    // against another draft.
    let mut mtp_pending: Vec<(usize, usize, usize)> = Vec::new();
    let mut mtp_hits = vec![0usize; mtp_k];
    let mut mtp_seen = vec![0usize; mtp_k];
    // Per ISSUING step, which depths turned out right. The per-depth rates are
    // MARGINALS — head d's draft against the true token at step s+d+1 — and an
    // accept-and-skip loop does not get to keep a correct depth-2 draft that sits
    // behind a wrong depth-1 one. What it keeps is the LEADING RUN of correct
    // drafts, so that is measured here rather than inferred from the marginals:
    // inferring it would need the depths to be independent, and they are not,
    // because head d is fed draft d-1.
    let mut mtp_issued: BTreeMap<usize, Vec<Option<bool>>> = BTreeMap::new();
    // The MTP entry hidden state for every position, retained. 16 KB a token
    // at f32, and the reason the cached lane can draft at all -- see the draft
    // block for why a row, once produced, never changes.
    let mut mtp_main: Vec<f32> = Vec::new();
    // Per head: its STABLE hidden rows, which are what the NEXT head reads,
    // and its own K/V cache. Ragged by one row per depth, because head d's
    // stable rows stop at position seq-1-d.
    let mut mtp_stage: Vec<Vec<f32>> = vec![Vec::new(); mtp_k];
    let mut mtp_caches: Vec<Option<MtpCache>> = (0..mtp_k).map(|_| None).collect();

    // Experts are read, applied and dropped. There is no decoded-expert cache:
    // it measured as no speedup, and it existed to paper over a capacity
    // shortfall (160 GB of checkpoint against 119 GB of box) that a second
    // Spark closes. See this file's header.
    //
    // There are no lane switches left. `INK_GPU`, `INK_ATTN`, `INK_DENSE`,
    // `INK_HEAD` and `INK_EXPERTS` each chose between a device implementation
    // and a host one; `INK_RESIDENT` chose between holding a weight and
    // re-reading it off the SSD every token; `INK_DEQUANT=chain` chose the
    // 46-launch Burn decode over the fused kernel. Every one of those choices
    // has a right answer on this hardware and a wrong one that still runs, and
    // a wrong one that still runs is a wrong one you will ship. The right
    // answers are now the only answers.
    //
    // The KV cache is the one thing still switched, and it is not a host/device
    // choice: `INK_KV=0` is the uncached lane, which is the ONLY oracle the
    // cached lane has (`INK_MTP_CHECK` compares the two in one process), so
    // deleting it would delete the check that the cache is right.
    let kv = std::env::var("INK_KV").map(|v| v == "1" || v == "on").unwrap_or(false);
    // `INK_REPEAT=1` keeps the token loop from GROWING the sequence: every step
    // re-runs the identical pass over the identical prompt. It exists because
    // the only pass of a given width a process can otherwise produce is its
    // FIRST one, which is also the one that pays for uploading every resident
    // weight -- 34 s against 1.4 s for the next. Comparing the cost of a 1-row
    // pass with a 2-row one therefore compared two different warm-up states,
    // not two widths. With this set the second and later steps are warm and
    // identical, so the widths can be compared by running the binary twice with
    // two prompts. Measurement only: the generated token is still printed and
    // still thrown away, so the run produces no continuation.
    let repeat = std::env::var("INK_REPEAT").map(|v| v == "1" || v == "on").unwrap_or(false);
    anyhow::ensure!(!repeat || !kv, "INK_REPEAT wants the uncached lane: with a KV cache a \
         repeated pass would append the same position to the cache again");
    // Head d's first STABLE row sits at position seq-1-d, so a prompt shorter
    // than the number of heads leaves a depth with no stable row at all: every
    // position of that head would be a function of drafts and the cache would
    // have nothing to hold. Refuse rather than quietly fall back to recomputing
    // the prefix, which is the failure this whole change exists to remove.
    anyhow::ensure!(
        mtp_k == 0 || !kv || ids.len() >= mtp_k,
        "INK_MTP={mtp_k} with INK_KV needs a prompt of at least {mtp_k} tokens, this one is {}",
        ids.len()
    );
    anyhow::ensure!(
        mtp_k <= cfg.mtp_config.num_nextn_predict_layers,
        "INK_MTP={mtp_k} but the checkpoint ships {} MTP heads",
        cfg.mtp_config.num_nextn_predict_layers
    );
    println!("  attention          : device, weights DEVICE-RESIDENT");
    println!("  shared + dense MLP : device, uploaded once and held");
    println!("  routed experts     : device, NATIVE tensor cores -- NVFP4 where packed, BF16 at layer 2");
    println!("  head (unembed)     : device");
    let router_arm = RouterArm::from_env();
    println!("  router projection  : {}", router_arm.label());
    // OPT-IN, and this run is slower for it: it uploads a second projection per
    // layer and issues a second matmul and a second BLOCKING read per MoE layer
    // per token. A timing run must not have it on, and the report says which
    // runs did.
    let router_diff = std::env::var("INK_ROUTER_DIFF").map(|v| v == "1").unwrap_or(false);
    if router_diff {
        println!("  router diff        : ON -- selection compared against the f32 [rows,hidden] lane, this pass IS slower");
    }
    println!("  kv cache           : {}", if kv { "on" } else { "off (prefix recomputed each step)" });
    // The SHARED experts' w13 is square, so nothing but a forward can tell the
    // two readings apart. INK_SHARED_W13_HALVED=1 selects the other one.
    let shared_halved = mary::models::inkling::load::shared_w13_halved();
    println!(
        "  shared w13 split   : {}",
        if shared_halved { "HALVED (contiguous)" } else { "INTERLEAVED" }
    );
    let dev = burn::backend::cuda::CudaDevice::default();
    let mut ddense = DeviceDense::default();
    // Every attention layer's projections, on the device, for the whole run.
    //
    // Attention was the last lane that still STREAMED, and it streamed because
    // of a reading of `INK_RESIDENT` that stopped one step short: the host copy
    // is indeed worthless after the upload, so the lane declined to hold it --
    // and then held nothing at all, and paid the read, the widen AND the
    // transfer again on every token. Measured on the pile, decode step, 42
    // layers: 2.2 s widening 6.9 GiB of BF16 into f32 and 6.0 s pushing it
    // across, against 0.2 s of attention. The weights do not change between
    // tokens; only where they are held was ever in question, and on a device
    // lane the answer is the device -- which is why there is no longer a flag
    // that can answer it the other way.
    //
    // Keyed by the layer prefix, exactly as [`DeviceDense`] is, and populated
    // on first use rather than eagerly: a lane that is never taken should not
    // pay for the upload.
    let mut layers_dev: std::collections::BTreeMap<String, LayerDev> =
        std::collections::BTreeMap::new();
    let mut dattn_bytes = 0u64;
    // The compute client, taken FROM a Burn tensor rather than constructed
    // beside it. `CudaRuntime::client(&Default::default())` is meant to return
    // the same client Burn is using, and "meant to" is not a thing to bet a
    // device pointer on: `seam::handle_of` hands a Burn allocation to a raw
    // kernel launched on this client, and if they were two clients that would
    // be a wrong answer rather than an error.
    let fp4_client = mary::models::inkling::seam::client_of(
        &BT::<Bk, 2>::zeros([1, 1], &dev),
    );
    println!("{}", mary::models::inkling::pile::mem_line("after CUDA context"));
    // Nine blocking device round trips for the whole run, instead of four per
    // expert. Every later slab is an offset view of one of these.
    //
    // Always an `Aliases`, even when nothing can be aliased. In the supported
    // lane its one registered mapping is the anonymous startup allocation, not
    // the pile file. The unsafe reproducer arm deliberately registers the pile.
    //
    // `Aliases::disabled()` copies exactly as the old `None` arm did but COUNTS
    // it, so a source whose bytes cannot be aliased reports that rather than
    // going quiet. The registration is unconditional — on a unified-memory part
    // a copy of a weight the device can read where it lies is a copy for
    // nothing, and there is no configuration in which we want one.
    #[cfg(feature = "inkling-cuda")]
    let fp4_aliases = {
        let c = &fp4_client;
        // `INK_ZEROCOPY=0` sends every weight through `create_from_slice`
        // instead of aliasing the startup allocation. It remains a diagnostic
        // arm, not a lane: the supported path aliases anonymous memory and
        // does not need a per-bind device copy.
        if std::env::var("INK_ZEROCOPY").map(|v| v == "0").unwrap_or(false) {
            println!("  zero-copy mappings : DISABLED (INK_ZEROCOPY=0) -- every bind copies");
            Some(mary::models::inkling::fp4gemm::Aliases::disabled())
        } else {
            let t = Instant::now();
            let maps = cp.mappings()?;
            let n = maps.len();
            let a = mary::models::inkling::fp4gemm::Aliases::register(c, maps);
            println!(
                "  zero-copy mappings : {} {n} in {:.1} ms",
                if a.is_some() { "registered" } else { "UNSUPPORTED, copying" },
                t.elapsed().as_secs_f64() * 1e3
            );
            Some(a.unwrap_or_else(mary::models::inkling::fp4gemm::Aliases::disabled))
        }
    };

    // The unembed table, as the BF16 the pile stores. Not uploaded: BOUND.
    //
    // This is the single largest weight in the process and it was the single
    // largest widening: `[201024, 4096]` BF16 is 1.61 GiB, and reading it as
    // f32 made it 3.22 GB — on the host first, then again on the device,
    // costing 1.6 GiB of RAM the box does not have to spare and doubling the
    // bytes the biggest matmul in the forward has to read. Measured at 22.5 ms
    // a token, which is 146 GB/s against a 273 GB/s bus: bandwidth-bound, so
    // the bytes ARE the time.
    //
    // Bound through the same `Aliases` seam the experts use, so on this
    // unified-memory part the GPU reads the pile's own mapped pages and the
    // table is never copied at all. `slice_or_copy` reports which happened.
    //
    // The FULL padded vocabulary is bound, not the effective 200058 rows: the
    // MMA tiles n by 8 and 200058 does not divide, while the padded 201024
    // does. The extra rows are the checkpoint's own padding and the argmax
    // slices them off, exactly as the f32 path sliced them off after uploading
    // them.
    let unembed_w = if want_head {
        use mary::models::inkling::bf16gemm::Bf16W;
        use mary::models::inkling::pile::Elem;
        let leaf = cp.stored("model.llm.unembed.weight")?;
        anyhow::ensure!(
            leaf.elem == Elem::Bf16,
            "the unembed table is stored as {:?}, and this lane multiplies BF16 by BF16. \
             Widening it to reuse an f32 path is the thing rule 3 forbids; a pile holding \
             f32 here needs an f32 MMA, not a cast.",
            leaf.elem
        );
        let (rows, cols) = (leaf.dims[0] as usize, leaf.dims[1] as usize);
        anyhow::ensure!(rows == t.vocab_size && cols == h, "unembed is {rows}x{cols}");
        anyhow::ensure!(
            Bf16W::tileable(rows, cols),
            "unembed {rows}x{cols} does not tile as m16n8k16"
        );
        let align = note_align(&leaf.bytes, rows, cols);
        let copy_to_align = align < 16
            && std::env::var("INK_ALIGN_COPY").map(|v| v == "1").unwrap_or(false);
        let hnd = match fp4_aliases.as_ref() {
            Some(al) if !copy_to_align => al.slice_or_copy(&fp4_client, &leaf.bytes),
            _ => fp4_client.create_from_slice(&leaf.bytes),
        };
        println!(
            "  unembed BOUND as BF16, {rows} x {cols} = {:.2} GiB stored (the f32 lane it \
             replaces materialised {:.2} GiB)",
            leaf.bytes.len() as f64 / GIB,
            2.0 * leaf.bytes.len() as f64 / GIB
        );
        Some(Bf16W { h: hnd, n: rows, k: cols, align: if copy_to_align { 16 } else { align } })
    } else {
        None
    };
    // The final norm's gain, uploaded once for the same reason -- it used to be
    // re-uploaded from the host copy on every pass, and on every MTP draft.
    let fnorm_dev = fnorm
        .as_ref()
        .map(|f| up1r::<Bk>(&f.data, h, &dev));

    // Pay the storage layer's BLAKE3 for this node's experts HERE, once, rather
    // than in whichever decode step first routes to each of them.
    //
    // Rule 6 says never touch the SSD once running, and this is the read that
    // was doing it -- 9.4 to 33.6 MB an expert, hashed on first touch, landing
    // wherever the router sent a token first. On the 20-layer head at step 41
    // to 80 of an 80-token generation it was still costing between 0.5 and 274
    // ms a pass, and that one variable explained the entire spread.
    //
    // It also makes an A/B possible at all. Two builds that produce different
    // tokens route to different experts and therefore pay different amounts of
    // a cost that belongs to neither of them; with the warm done, the
    // comparison is between the models rather than between their luck.
    {
        let t0 = Instant::now();
        let mut last = Instant::now();
        let (count, bytes) = cp.warm_experts(lo..hi, |i, total, b| {
            if last.elapsed().as_secs_f64() > 10.0 || i == total {
                last = Instant::now();
                println!(
                    "  warming experts    : {i}/{total}  {:.1} GiB  {:.0}s",
                    b as f64 / GIB,
                    t0.elapsed().as_secs_f64()
                );
            }
        })?;
        println!(
            "  warmed             : {count} expert planes, {:.2} GiB validated in {:.1}s ({:.2} GiB/s)",
            bytes as f64 / GIB,
            t0.elapsed().as_secs_f64(),
            bytes as f64 / GIB / t0.elapsed().as_secs_f64()
        );
    }

    // Everything one layer carries between generated tokens. The attention
    // cache is the headline, but the two layer-level short convolutions have
    // state too: they reach `kernel - 1` positions back, and a cache that
    // remembers K and V while forgetting those is wrong in a way that still
    // produces fluent-looking text.
    struct LayerCache {
        attn: dev_lane::AttnCache<Bk>,
        attn_sconv: BT<Bk, 2>,
        /// `None` until the prefill seeds it, and a device tensor rather than a
        /// `Vec<f32>` for the same reason everything else here is: the
        /// convolution that reads it runs on the device, and a history that
        /// lived on the host would drag the whole MLP half back across.
        mlp_sconv: Option<BT<Bk, 2>>,
    }
    let mut caches: Vec<LayerCache> = Vec::new();

    // The wire, opened AFTER the weights so a connection is never left hanging
    // while the other end spends a minute building its index. The tail binds and
    // waits; the head connects, so the tail must be started first.
    let mut pipe = match pipe_spec.as_deref() {
        Some(s) if is_head => {
            let addr = &s["head:".len()..];
            let t0 = Instant::now();
            let sock = TcpStream::connect(addr)
                .with_context(|| format!("connecting to the tail at {addr}"))?;
            sock.set_nodelay(true)?;
            println!("  pipe: connected to the tail at {addr} in {:.1}s", t0.elapsed().as_secs_f32());
            Some(Pipe::Head(sock))
        }
        Some(s) if is_tail => {
            let addr = &s["tail:".len()..];
            let l = TcpListener::bind(addr).with_context(|| format!("binding {addr}"))?;
            println!("  pipe: listening on {addr}");
            let (sock, peer) = l.accept()?;
            sock.set_nodelay(true)?;
            println!("  pipe: head connected from {peer}");
            Some(Pipe::Tail(sock))
        }
        _ => None,
    };

    // One per layer of the whole stack, indexed by absolute layer number, so
    // the summary can name the layer rather than a position in this node's
    // slice. Declared here and not in the pass because the question is what the
    // RUN did, not what one token did.
    let mut route_diff = vec![RouteDiff::default(); t.num_hidden_layers];

    // `INK_DEV_ROUTE=0` puts the router's DECISION back on the host. It is on
    // by default; the flag exists so the two lanes can be interleaved from one
    // binary, which is the only honest way to price a 15% change against a
    // 2-3 ms pass-to-pass drift.
    let dev_route = std::env::var("INK_DEV_ROUTE").map(|v| v != "0").unwrap_or(true);
    // The gate bias is `[n_routed]` f32 and does not change during a run, so it
    // is uploaded once per layer rather than once per pass. One KiB either way;
    // it is here because a per-pass upload in a lane whose whole subject is
    // host->device round trips would be embarrassing.
    let mut bias_dev: std::collections::HashMap<usize, cubecl::server::Handle> =
        std::collections::HashMap::new();

    let mut top_all: Vec<i64> = Vec::new();
    for step in 0..=gen_steps {
    // A tail's step BEGINS on the wire, and it waits before its own timers
    // start: a tail that charged itself for the head's half would report the
    // pipeline's latency as its own cost, and the per-machine split is the
    // entire question here.
    let incoming = match pipe.as_mut() {
        Some(Pipe::Tail(s)) => match recv_stream(s, h)? {
            Some(v) => Some(v),
            // The head closed. Not an error -- it is how a finished run ends.
            None => break,
        },
        _ => None,
    };

    let pass = Instant::now();
    let io0 = io_read_bytes();
    cp.io_reset();
    // Same scope as the loader counters, for the same reason: the report below
    // says "this ONE pass", and a bind total that accumulated across passes
    // would silently make it say something else.
    #[cfg(feature = "inkling-cuda")]
    if let Some(al) = fp4_aliases.as_ref() {
        al.stats_reset();
    }
    // With a cache, every pass past the prefill feeds exactly the token the
    // previous pass produced; without one, the whole prefix goes through again.
    // `pos0` is that token's ABSOLUTE position, which is what log scaling and
    // the relative bias are functions of -- it is only equal to zero on a pass
    // that starts from the beginning.
    let (feed, pos0): (Vec<usize>, usize) = if kv && step > 0 {
        (vec![*ids.last().expect("a step past the prefill has produced a token")], ids.len() - 1)
    } else {
        (ids.clone(), 0)
    };
    // The tail is handed the stream the head already embedded and ran; it takes
    // `n` and `pos0` from the wire rather than from `ids`, because those are
    // facts about the pass and only the head owns the token loop.
    let t_emb = Instant::now();
    let (n, pos0, x_in) = match incoming {
        Some((n, p, x)) => (n, p, x),
        None => {
            let n = feed.len();
            let e_w = embed_w.as_ref().expect("the head owns the embedding table");
            let e_n = embed_n.as_ref().expect("the head owns the embedding norm");
            (n, pos0, embed_and_norm_bf16(&feed, e_w, &e_n.data, t.rms_norm_eps, t.vocab_size, h))
        }
    };

    let dump_dir = std::env::var("INK_DUMP_DIR").ok();
    if let Some(dir) = dump_dir.as_ref() {
        std::fs::create_dir_all(dir)?;
        let mut bytes = Vec::with_capacity(x_in.len() * 4);
        for v in &x_in {
            bytes.extend_from_slice(&v.to_le_bytes());
        }
        std::fs::write(format!("{dir}/h_embed.bin"), &bytes)?;
    }

    // THE upload. One per pass -- 16 KB a token -- and from here to the end of
    // this node's layers the residual stream does not touch the host again.
    //
    // A head embeds and uploads; a tail uploads what came off the wire. Both
    // are the same 16 KB, which is why the pipe crosses BETWEEN layers and not
    // inside one.
    let mut xd: T2 = up2::<Bk>(x_in, n, h, &dev);
    let t_embed = t_emb.elapsed().as_secs_f64();

    let ls = LogScaling {
        n_floor: t.log_scaling_n_floor as f32,
        alpha: t.log_scaling_alpha as f32,
    };
    let mask_local = causal_mask(n, Some(t.sliding_window_size));
    let mask_global = causal_mask(n, None);
    #[allow(unused_assignments)]
    let mut expert_loads = 0usize;
    let (mut t_attn, mut t_expert, mut t_other, mut t_shared) = (0f64, 0f64, 0f64, 0f64);
    // The attention half at its two host-side seams: reading+widening the ten
    // projections out of the source, and getting them onto the device. What is
    // left over is device work. The sync after the uploads is what makes the
    // second number a TRANSFER rather than an enqueue.
    #[allow(unused_mut)]
    let (mut t_attn_read, mut t_attn_up) = (0f64, 0f64);
    // Reading a tensor out of the mapping and widening BF16 to f32 is host work
    // no lane can move, so it is counted once, separately, rather than being
    // smeared across whichever bucket happened to ask for the weight.
    let t_read = std::cell::Cell::new(0f64);
    // Host-side and therefore honestly attributable, unlike anything downstream
    // of an enqueued device call.
    let mut host_t = HostT::default();
    // The scalar host arithmetic in the block itself, which no timer above
    // covers because it is not "the attention half" or "the expert lane" -- it
    // is the connective tissue between them, and connective tissue that runs on
    // a CPU is a host path in the data plane.
    // `t_h_norm` and `t_h_resid` have no writer any more and that is the point:
    // they are printed, they read zero, and a zero that is measured is worth
    // more than a claim that is asserted.
    let (t_h_norm, mut t_h_route, mut t_h_sconv, t_h_resid) = (0f64, 0f64, 0f64, 0f64);
    // The router bucket, split three ways. It used to be one number, and one
    // number cannot distinguish "the matmul was described" from "the host waited
    // for every kernel this layer had already issued" from "the top-k ran on a
    // CPU". At decode the whole bucket is the middle one, and that is only
    // visible once the other two are subtracted out.
    let (mut t_rt_mm, mut t_rt_read, mut t_rt_host) = (0f64, 0f64, 0f64);
    // `INK_STAGE_SYNC=1` inserts an explicit device sync at each stage boundary
    // in the block and charges the wait to that stage. It is OPT-IN and off by
    // default because it SERIALISES the lane: with it on the pass is slower, and
    // the comparison that matters is probe-on-total against probe-off-total.
    // It changes no arithmetic -- a sync is a wait, not an operation -- so both
    // arms emit the same tokens, which is itself checked below.
    let stage_sync = std::env::var("INK_STAGE_SYNC").map(|v| v == "1").unwrap_or(false);
    let (mut d_attn, mut d_router, mut d_expert, mut d_shared, mut d_tail) =
        (0f64, 0f64, 0f64, 0f64, 0f64);
    let mut stage_syncs = 0usize;

    // Per-layer diagnostics, COLLECTED rather than printed inside the loop.
    //
    // The RMS of the residual stream after each layer used to be computed on
    // the host, from a `Vec<f32>` the loop had anyway. It does not have one any
    // more, and reading `x` back per layer to print a number would reintroduce
    // exactly the round trip this change removes -- forty of them a token, to
    // produce a log line. So each layer enqueues its own reduction and the
    // whole column is read once, after the stack, in a single `cat`.
    //
    // The per-layer WALL TIME went with it, and its absence is the honest
    // report: with nothing synchronising inside the loop, "how long did layer
    // 17 take" is not a question the host can answer. It would have measured
    // enqueueing.
    let mut layer_rms: Vec<T2> = Vec::with_capacity(hi - lo);
    let mut layer_kind: Vec<(usize, bool)> = Vec::with_capacity(hi - lo);
    // Uploaded once per pass rather than once per layer: the mask is a function
    // of `n` and the layer's kind, and there are two kinds.
    let (mask_l_dev, mask_g_dev) = if kv && step > 0 {
        (None, None)
    } else {
        (
            Some(up2::<Bk>(mask_local.clone(), n, n, &dev)),
            Some(up2::<Bk>(mask_global.clone(), n, n, &dev)),
        )
    };

    // One sync, charged to one stage. Written as a macro rather than a closure
    // because every call site names a different accumulator and a closure would
    // have to borrow all five of them mutably at once.
    macro_rules! stage_sync {
        ($acc:expr) => {
            if stage_sync {
                let s = Instant::now();
                <Bk as burn::tensor::backend::Backend>::sync(&dev).expect("stage sync");
                $acc += s.elapsed().as_secs_f64();
                stage_syncs += 1;
            }
        };
    }

    for layer in lo..hi {
        // Cache slot, not layer number. A tail running 20..42 keeps 22 caches
        // and its first layer is its slot 0 — indexing by the absolute layer
        // would walk off the end of a Vec that only ever holds this node's half.
        let slot = layer - lo;
        let kind = t.attn_kind(layer);
        let is_local = kind == AttnKind::Local;
        let (heads, kv_heads, head_dim) = t.heads(kind);
        let p = format!("model.llm.layers.{layer}.");
        // ONE accessor now. `g` used to exist beside it, holding an f32 copy on
        // the host for the host lanes to read; there are no host lanes left in
        // this loop, so every weight is read once, uploaded, and the host copy
        // dropped.
        let gv = |nm: &str| -> Result<Vec<f32>> {
            let s = Instant::now();
            let r = cp.tensor(&format!("{p}{nm}"))?.data;
            t_read.set(t_read.get() + s.elapsed().as_secs_f64());
            Ok(r)
        };

        // ---- this layer's weights, on the device, built once ---------------
        //
        // Everything the block multiplies by: the ten attention projections,
        // BOTH short convolutions, BOTH norm gains and the router matrix. The
        // norms and the mlp short convolution used to be read per token as
        // host `Vec<f32>` because the operations that consumed them were host
        // operations. They are not any more, so they join the rest.
        if !layers_dev.contains_key(&p) {
            let r0 = t_read.get();
            let t_w0 = Instant::now();
            // The five projections bind as the BF16 the pile stores. `gv`
            // widens to f32 on the way out of the mapping and would double
            // every one of them on the device for nothing: `mma.sync…bf16`
            // takes the stored bytes, and where those bytes are inside a
            // registered mapping `bind_bf16` aliases them instead of copying.
            let pw = |nm: &str, rows: usize, cols: usize| -> Result<Bf16W> {
                let s = Instant::now();
                let leaf = cp.stored(&format!("{p}{nm}"))?;
                anyhow::ensure!(
                    leaf.elem == Elem::Bf16,
                    "{p}{nm} is {:?}; this lane multiplies BF16 by BF16",
                    leaf.elem
                );
                t_read.set(t_read.get() + s.elapsed().as_secs_f64());
                Ok(bind_bf16(&fp4_client, fp4_aliases.as_ref(), &leaf.bytes, rows, cols))
            };
            let attn = dev_lane::AttnWeightsDev {
                wq: pw("attn.wq_du.weight", heads * head_dim, h)?,
                wk: pw("attn.wk_dv.weight", kv_heads * head_dim, h)?,
                wv: pw("attn.wv_dv.weight", kv_heads * head_dim, h)?,
                wr: pw("attn.wr_du.weight", heads * t.d_rel, h)?,
                wo: pw("attn.wo_ud.weight", h, heads * head_dim)?,
                k_sconv: up2(gv("attn.k_sconv.weight")?, kv_heads * head_dim, t.sconv_kernel_size, &dev),
                v_sconv: up2(gv("attn.v_sconv.weight")?, kv_heads * head_dim, t.sconv_kernel_size, &dev),
                q_norm: up1(gv("attn.q_norm.weight")?, head_dim, &dev),
                k_norm: up1(gv("attn.k_norm.weight")?, head_dim, &dev),
                rel_proj: up2(gv("attn.rel_logits_proj.proj")?, t.d_rel, t.rel_span(kind), &dev),
            };
            let router = if t.is_dense(layer) {
                None
            } else {
                let rows = t.n_routed_experts + t.n_shared_experts;
                // `gv` widens on the way out of the mapping, so the BF16 arm
                // must not call it for the projection at all -- that read IS the
                // widening. It takes `stored` instead, like the five attention
                // projections beside it.
                let proj = match router_arm {
                    RouterArm::Transpose => {
                        RouterProj::PerCall(up2(gv("mlp.gate.weight")?, rows, h, &dev))
                    }
                    // Transposed HERE, on the host, once per layer for the run,
                    // instead of on the device once per token per layer.
                    RouterArm::Pre => RouterProj::Pre(up2(
                        transpose_rows(&gv("mlp.gate.weight")?, rows, h),
                        h,
                        rows,
                        &dev,
                    )),
                    RouterArm::Bf16 => {
                        use mary::models::inkling::bf16gemm::NTILE;
                        let s = Instant::now();
                        let leaf = cp.stored(&format!("{p}mlp.gate.weight"))?;
                        anyhow::ensure!(
                            leaf.elem == Elem::Bf16,
                            "{p}mlp.gate.weight is {:?}; INK_ROUTER=bf16 multiplies the STORED \
                             BF16 and will not widen to reach an element type",
                            leaf.elem
                        );
                        // Six zero rows so 258 tiles as n8. They produce logits
                        // the host slices off; they are a pad, not a widening.
                        let pad = rows.div_ceil(NTILE) * NTILE;
                        let mut bytes = vec![0u8; pad * h * 2];
                        bytes[..rows * h * 2].copy_from_slice(&leaf.bytes);
                        t_read.set(t_read.get() + s.elapsed().as_secs_f64());
                        RouterProj::Bf16 {
                            w: bind_bf16(&fp4_client, fp4_aliases.as_ref(), &bytes, pad, h),
                            rows,
                        }
                    }
                };
                // Held only under the diff probe, and it is the arm every run
                // before 969bf6f shipped -- the thing the new arm has to be
                // compared AGAINST, not a second opinion invented for the
                // occasion.
                let reference = if router_diff {
                    Some(up2(gv("mlp.gate.weight")?, rows, h, &dev))
                } else {
                    None
                };
                Some(RouterDev {
                    proj,
                    reference,
                    bias: gv("mlp.gate.bias")?,
                    global_scale: gv("mlp.gate.global_scale")?[0],
                })
            };
            let built = LayerDev {
                attn,
                attn_sconv: up2(gv("attn_sconv.weight")?, h, t.sconv_kernel_size, &dev),
                mlp_sconv: up2(gv("mlp_sconv.weight")?, h, t.sconv_kernel_size, &dev),
                attn_norm: up1(gv("attn_norm.weight")?, h, &dev),
                mlp_norm: up1(gv("mlp_norm.weight")?, h, &dev),
                router,
            };
            <Bk as burn::tensor::backend::Backend>::sync(&dev)
                .expect("sync after the layer uploads");
            let span = t_w0.elapsed().as_secs_f64();
            let rd = t_read.get() - r0;
            t_attn_read += rd;
            t_attn_up += span - rd;
            // Two bytes for the projections and four for the rest, because that
            // is now what is on the device. Counting the projections at four
            // would report the widening this commit removed.
            dattn_bytes += (2 * (heads * head_dim * h
                + 2 * kv_heads * head_dim * h
                + heads * t.d_rel * h
                + h * heads * head_dim)
                + 4 * (2 * kv_heads * head_dim * t.sconv_kernel_size
                    + 2 * head_dim
                    + t.d_rel * t.rel_span(kind)
                    + 2 * h * t.sconv_kernel_size
                    + 2 * h)) as u64;
            layers_dev.insert(p.clone(), built);
        }
        let ld = layers_dev.get(&p).expect("inserted directly above");

        // ---- attention ----------------------------------------------------
        let t_a = Instant::now();
        let hn = dev_lane::rms_norm(xd.clone(), ld.attn_norm.clone(), t.rms_norm_eps);
        let dims = AttnDims {
            hidden: h, heads, kv_heads, head_dim,
            d_rel: t.d_rel,
            rel_extent: t.rel_span(kind),
            kernel: t.sconv_kernel_size,
            rms_eps: t.rms_norm_eps,
            kind,
        };
        // The same distinction the mask carries, in the form the cache needs:
        // how far back a query may look, and therefore how much of the cache
        // can never be read again.
        let window = if is_local { Some(t.sliding_window_size) } else { None };
        let mask = || {
            let m = if is_local { &mask_l_dev } else { &mask_g_dev };
            m.clone().expect("a pass that needs a mask uploaded one")
        };
        let a = if kv && step > 0 {
            let y = dev_lane::attention_step(
                hn, &ld.attn, &dims, Some(ls), pos0, window, &mut caches[slot].attn,
            );
            let (out, hist) =
                dev_lane::short_conv_step(caches[slot].attn_sconv.clone(), y, ld.attn_sconv.clone());
            caches[slot].attn_sconv = hist;
            out
        } else if kv {
            let (y, attn) = dev_lane::attention_prefill(
                hn, &ld.attn, &dims, Some(ls), mask(), window,
            );
            let hist = dev_lane::conv_history(y.clone(), t.sconv_kernel_size);
            let out = dev_lane::short_conv(y, ld.attn_sconv.clone());
            caches.push(LayerCache { attn, attn_sconv: hist, mlp_sconv: None });
            out
        } else {
            let y = dev_lane::attention(hn, &ld.attn, &dims, Some(ls), mask());
            dev_lane::short_conv(y, ld.attn_sconv.clone())
        };
        xd = xd + a;

        stage_sync!(d_attn);
        // ---- MLP ----------------------------------------------------------
        t_attn += t_a.elapsed().as_secs_f64();
        let t_o = Instant::now();
        let hn = dev_lane::rms_norm(xd.clone(), ld.mlp_norm.clone(), t.rms_norm_eps);

        let y = if t.is_dense(layer) {
            // Device-resident: uploaded on the first token that reaches this
            // layer and held for the run. The host reference that used to sit
            // beside it (`host_dense`, selected by leaving `INK_DENSE` unset)
            // was a scalar f32 lane over a 537 MB weight; it is not a lane a
            // 276 B model has any use for, and being selectable is how it got
            // run by accident.
            let w = ddense.dense_for(&cp, &fp4_client, fp4_aliases.as_ref(), &p, h)?;
            dense_mlp_bf16(hn, w)
        } else {
            let inter = t.intermediate_size;
            let r = ld.router.as_ref().expect("a MoE layer has a router");
            // The router's PROJECTION is a matmul and runs on the device; its
            // DECISION is control plane and runs here. What crosses is
            // [n, 258] f32 -- 1 KB on a decode step -- against the 14.2 MB of
            // expert weight the decision selects. This read BLOCKS, and it is
            // the only place in the layer that does; it cannot be avoided by
            // scheduling, because which weights to read next is a function of
            // the number that just came back.
            let rows = t.n_routed_experts + t.n_shared_experts;
            let t_rt = Instant::now();
            // `cols` is what comes BACK, which is `rows` except on the BF16 arm,
            // whose weight carries the instruction's n padding.
            let (lg, cols) = match &r.proj {
                RouterProj::PerCall(w) => (dev_lane::linear(hn.clone(), w.clone()), rows),
                RouterProj::Pre(wt) => (dev_lane::linear_pre_t(hn.clone(), wt.clone()), rows),
                RouterProj::Bf16 { w, .. } => (dev_lane::linear_bf16(hn.clone(), w), w.n),
            };
            t_rt_mm += t_rt.elapsed().as_secs_f64();
            stage_sync!(d_router);
            // Two lanes, and the difference is WHERE the top-k runs, not what
            // it decides. The host lane reads `[n, rows]` f32 back and sorts;
            // the device lane runs `routetopk` on the logits where they already
            // are and reads back `[n, 2k + shared + 1]`, which at 512 tokens is
            // 30 KB against 528 KB and, more to the point, is 512 rows of
            // 256-wide selection the host no longer walks.
            //
            // The reference arm below wants the full host logits, so it selects
            // the host lane: a diagnostic that changed the lane it measures
            // would be measuring itself.
            let host_route = !dev_route || r.reference.is_some();
            let routing: Vec<Routing>;
            let mut logits: Vec<f32> = Vec::new();
            if host_route {
                let t_rr = Instant::now();
                logits = drop_pad_cols(down(lg), n, cols, rows);
                t_rt_read += t_rr.elapsed().as_secs_f64();
                let t_rh = Instant::now();
                routing = route_from_logits(
                    &logits, &r.bias, r.global_scale, t.route_scale as f32,
                    n, t.n_routed_experts, t.n_shared_experts, t.num_experts_per_tok,
                );
                t_rt_host += t_rh.elapsed().as_secs_f64();
            } else {
                use mary::models::inkling::routetopk::router_topk_launch;
                use mary::models::inkling::seam::{handle_of, tensor_of};
                let k = t.num_experts_per_tok;
                let ns = t.n_shared_experts;
                let width = 2 * k + ns + 1;
                let bias_h = bias_dev
                    .entry(layer)
                    .or_insert_with(|| fp4_client.create_from_slice(bytes_of(&r.bias)))
                    .clone();
                let t_rt2 = Instant::now();
                let lg_h = handle_of(lg);
                let out_h = router_topk_launch(
                    &fp4_client, &lg_h, &bias_h, n, cols, t.n_routed_experts, ns, k,
                    t.route_scale as f32 * r.global_scale,
                );
                t_rt_mm += t_rt2.elapsed().as_secs_f64();
                let t_rr = Instant::now();
                let flat = down(tensor_of(fp4_client.clone(), dev.clone(), out_h, n, width));
                t_rt_read += t_rr.elapsed().as_secs_f64();
                let t_rh = Instant::now();
                let mut rs = Vec::with_capacity(n);
                for ti in 0..n {
                    let row = &flat[ti * width..(ti + 1) * width];
                    let bad = row[width - 1] as u32;
                    assert!(
                        bad == 0,
                        "router logit is non-finite at token {ti}, row {}",
                        bad - 1
                    );
                    rs.push(Routing {
                        experts: row[..k].iter().map(|&v| v as usize).collect(),
                        weights: row[k..2 * k].to_vec(),
                        shared_gammas: row[2 * k..2 * k + ns].to_vec(),
                    });
                }
                routing = rs;
                // `INK_ROUTE_DBG=1`: the SAME logits through the host rule, and
                // a count of where the two lanes disagree. It reads the logits
                // back and routes twice, so it is slower than either lane and
                // is not a lane -- but it compares the two on ONE input, which
                // is the thing two separate runs cannot do. A device router and
                // a host router started from the same prompt part company after
                // a few layers no matter how right they both are: the routing
                // weights differ in the eighth decimal, the residual stream
                // carries that forward, and a later layer has a near-tie at the
                // top-k boundary that falls the other way. That is chaos, not
                // error, and only a same-input comparison can tell them apart.
                if std::env::var("INK_ROUTE_DBG").is_ok() {
                    let hl = drop_pad_cols(
                        down(tensor_of(fp4_client.clone(), dev.clone(), lg_h.clone(), n, cols)),
                        n, cols, rows,
                    );
                    let hr = route_from_logits(
                        &hl, &r.bias, r.global_scale, t.route_scale as f32,
                        n, t.n_routed_experts, ns, k,
                    );
                    let mut bad = 0usize;
                    let mut worst_w = 0f32;
                    for ti in 0..n {
                        for j in 0..k {
                            let d = (routing[ti].weights[j] - hr[ti].weights[j]).abs();
                            if d > worst_w {
                                worst_w = d;
                            }
                        }
                        for j in 0..ns {
                            let d = (routing[ti].shared_gammas[j] - hr[ti].shared_gammas[j]).abs();
                            if d > worst_w {
                                worst_w = d;
                            }
                        }
                        if hr[ti].experts != routing[ti].experts {
                            bad += 1;
                            if bad <= 4 {
                                println!(
                                    "ROUTEDBG layer {layer} t {ti} host {:?} dev {:?}",
                                    hr[ti].experts, routing[ti].experts
                                );
                            }
                        }
                    }
                    println!(
                        "ROUTEGATE layer {layer}: {n} rows examined, {bad} selections differ, \
                         max |dev-host| weight {worst_w:.3e}"
                    );
                }
                t_rt_host += t_rh.elapsed().as_secs_f64();
            }
            let t_rh = Instant::now();

            // `INK_ROUTER_DIFF=1`: the same activation through the f32 lane,
            // the same selection rule on the result, and a count of where the
            // two disagree. Nothing below reads `ref_routing` -- the run acts on
            // `routing`, whichever arm produced it -- so this measures the arm
            // rather than replacing it.
            if let Some(rw) = r.reference.as_ref() {
                let ref_logits = down(dev_lane::linear(hn.clone(), rw.clone()));
                let ref_routing = route_from_logits(
                    &ref_logits, &r.bias, r.global_scale, t.route_scale as f32,
                    n, t.n_routed_experts, t.n_shared_experts, t.num_experts_per_tok,
                );
                let d = &mut route_diff[layer];
                for ti in 0..n {
                    d.note(
                        &routing[ti],
                        &ref_routing[ti],
                        &logits[ti * rows..(ti + 1) * rows],
                        &ref_logits[ti * rows..(ti + 1) * rows],
                    );
                }
            }

            // Group tokens by expert, so each slab is read once.
            let mut by_expert: BTreeMap<usize, Vec<(usize, f32)>> = BTreeMap::new();
            for (ti, rt) in routing.iter().enumerate() {
                for (slot, &e) in rt.experts.iter().enumerate() {
                    by_expert.entry(e).or_default().push((ti, rt.weights[slot]));
                }
            }
            t_rt_host += t_rh.elapsed().as_secs_f64();
            t_h_route += t_rt.elapsed().as_secs_f64();

            // `INK_ROUTE_LOG=<path>` appends this layer's routing, one line per
            // position, plus the layer's DISTINCT-expert count. The count is
            // written so the file can be checked against `expert slabs
            // decoded` instead of trusted: the sum of the `distinct=` lines of
            // a pass must equal that counter, and a log that disagrees with the
            // number the run acted on is describing a different run.
            //
            // Written from `routing`, the same vector the block then feeds to
            // the experts -- not recomputed -- so the log cannot drift from
            // what was decoded. Positions are ABSOLUTE (`pos0 + ti`), because a
            // cached step's `ti` is always 0 and the sequence position is the
            // whole point of an adjacency measurement.
            if let Ok(rl) = std::env::var("INK_ROUTE_LOG") {
                use std::io::Write as _;
                let mut f = std::fs::OpenOptions::new()
                    .create(true)
                    .append(true)
                    .open(&rl)
                    .with_context(|| format!("INK_ROUTE_LOG={rl}"))?;
                for (ti, r) in routing.iter().enumerate() {
                    writeln!(f, "R {step} {layer} {} {:?}", pos0 + ti, r.experts)?;
                }
                writeln!(f, "D {step} {layer} {}", by_expert.len())?;
            }

            let t_d = Instant::now();
            // Two formats, two instructions, one lane. 41 of the 42 layers
            // carry NVFP4 experts and go through the block-scaled MMA; layer 2
            // carries plain BF16 ones -- no `.scale` sidecar, because nothing
            // quantised them -- and goes through the unscaled BF16 MMA
            // (`mma.sync...bf16`, f32 accumulator, which is the instruction's
            // own output type and not a widening).
            //
            // Which one is decided by the pile, not by a flag.
            let acc = {
                let a = if cp.is_nvfp4(&format!("{p}mlp.experts.w13_weight")) {
                    routed_experts_fp4(
                        &cp, fp4_aliases.as_ref(), &fp4_client, &dev,
                        &p, &by_expert, &hn, n, h, inter, &mut host_t,
                    )?
                } else {
                    routed_experts_bf16(
                        &cp, fp4_aliases.as_ref(), &fp4_client, &dev,
                        &p, &by_expert, &hn, n, h, inter, &mut host_t,
                    )?
                };
                expert_loads += by_expert.len();
                a
            };
            // ENQUEUE time, not work: nothing in this lane synchronises any
            // more. The layer's device time shows up in the one sync after the
            // stack, which is where it belongs and where it cannot be
            // misattributed to whichever bucket happened to hold the readback.
            t_expert += t_d.elapsed().as_secs_f64();
            stage_sync!(d_expert);

            let ns = t.n_shared_experts;
            let gammas: Vec<f32> = routing.iter().flat_map(|rt| rt.shared_gammas.clone()).collect();
            let t_s = Instant::now();
            // Device-resident, uploaded once. `split_shared_w13` is the
            // settled reading — this used to be an open `deinterleave_rows`
            // here and a halved split in the gate, which is the contradiction
            // the INTERLEAVED result closed.
            let sh = {
                let sw = ddense.shared_for(
                    &cp, &fp4_client, fp4_aliases.as_ref(), &p, ns, inter, h, shared_halved,
                )?;
                shared_experts_bf16(&dev, hn, sw, &gammas, ns)
            };
            stage_sync!(d_shared);
            t_shared += t_s.elapsed().as_secs_f64();
            acc + sh
        };

        // The MLP half's own short convolution carries state across generated
        // tokens exactly as attention's do.
        let t_sc = Instant::now();
        let y = if kv {
            if step > 0 {
                let hist = caches[slot]
                    .mlp_sconv
                    .clone()
                    .expect("a step past the prefill has a history");
                let (out, next) = dev_lane::short_conv_step(hist, y, ld.mlp_sconv.clone());
                caches[slot].mlp_sconv = Some(next);
                out
            } else {
                caches[slot].mlp_sconv =
                    Some(dev_lane::conv_history(y.clone(), t.sconv_kernel_size));
                dev_lane::short_conv(y, ld.mlp_sconv.clone())
            }
        } else {
            dev_lane::short_conv(y, ld.mlp_sconv.clone())
        };
        t_h_sconv += t_sc.elapsed().as_secs_f64();
        xd = xd + y;
        stage_sync!(d_tail);

        // A debug dump is a SYNC, and it is the one place left in the loop that
        // costs one. That is the trade: this path exists to compare against a
        // Python capture layer by layer, and it cannot do that without the
        // numbers.
        if let Some(dir) = dump_dir.as_ref() {
            let hx = down(xd.clone());
            let mut bytes = Vec::with_capacity(hx.len() * 4);
            for v in &hx {
                bytes.extend_from_slice(&v.to_le_bytes());
            }
            std::fs::write(format!("{dir}/h_after_{layer:02}.bin"), &bytes)?;
        }
        t_other += t_o.elapsed().as_secs_f64();
        // The same reduction the host loop did, enqueued rather than run:
        // sqrt(mean(x^2)) over the whole [n, h] stream. Read after the stack.
        layer_rms.push(xd.clone().powf_scalar(2.0).mean().reshape([1, 1]));
        layer_kind.push((layer, is_local));
    }

    // ---- the one sync for this node's whole stack --------------------------
    //
    // Everything above is enqueued. THIS is where the device time is, and
    // putting the timer here rather than around each stage is the only honest
    // place for it: with no readbacks inside the loop, a per-stage number would
    // measure how long the host took to describe the work, not how long the
    // work took.
    //
    // The forty-two per-layer RMS reductions come back in one read rather than
    // forty-two, which is the same trick one level up.
    let t_sy = Instant::now();
    // An EXPLICIT sync, not an implicit one. Reading the RMS column would drain
    // the queue anyway -- it depends on every layer's output -- but "would
    // anyway" is how a timer ends up measuring something other than its label.
    // This one costs nothing (the head cannot start before the stack finishes)
    // and it makes `t_stack_sync` the stack's device time and `t_head` the
    // head's, rather than one number smeared over both.
    <Bk as burn::tensor::backend::Backend>::sync(&dev).expect("sync after the stack");
    let rms_col: Vec<f32> = if layer_rms.is_empty() {
        Vec::new()
    } else {
        down(BT::cat(std::mem::take(&mut layer_rms), 0))
    };
    let t_stack_sync = t_sy.elapsed().as_secs_f64();
    for (i, &(layer, is_local)) in layer_kind.iter().enumerate() {
        println!(
            "  layer {layer:2} [{}] rms {:.4}",
            if is_local { "local " } else { "global" },
            rms_col[i].sqrt()
        );
    }

    // The residual stream on the HOST, and only for the readers that genuinely
    // need it there: the wire (16 KB to the tail), the MTP draft path (all
    // scalar host arithmetic), and a debug dump. A whole-stack or tail process
    // that is not drafting never materialises it -- the head reads `xd`.
    let want_host_x = is_head || mtp_k > 0 || dump_dir.is_some();
    let x: Vec<f32> = if want_host_x { down(xd.clone()) } else { Vec::new() };

    // Does anything downstream need a logit row that is NOT the last one?
    //
    // The argmax reads exactly one row -- the last -- and that is the whole of
    // what a forward produces. Every other row of the head exists for the
    // REPORT: the per-position top-5 table and the `INK_DUMP_DIR` capture. On a
    // 512-token prefill those rows are 512 x 200058 f32 = 410 MB read back
    // across the bus, on top of a 4096-wide GEMM over 512 rows instead of 16.
    // So they are computed when a reader has asked for them and not otherwise.
    // `INK_ALL_LOGITS=1` is that ask; a dump implies it.
    let all_logits =
        dump_dir.is_some() || std::env::var("INK_ALL_LOGITS").map(|val| val == "1").unwrap_or(false);

    // ---- head, or the wire in its place ------------------------------------
    let v = t.effective_vocab();
    let t_h = Instant::now();
    // A head has no logits and never will: the rest of the stack and the
    // unembedding both live on the other machine. So it hands the stream over
    // and takes the argmax back, and that blocking call is charged to the same
    // slot the head/unembed occupies on a whole-stack run — which is what makes
    // the two reports read against each other line for line.
    let mut best_wire = None;
    // Which position `logits[0]` is. The head computes `logit_row0..n`, so this
    // is 0 when everything was asked for and `n - 1` when only the argmax's row
    // was. A head computes nothing and the value is unread there.
    let logit_row0 = if all_logits { 0 } else { n - 1 };
    let logits = if let Some(Pipe::Head(s)) = pipe.as_mut() {
        send_stream(s, n, pos0, &x)?;
        let mut back = [0u8; 8];
        s.read_exact(&mut back).context("the tail closed mid-step")?;
        best_wire = Some(i64::from_le_bytes(back) as usize);
        Vec::new()
    } else {
        // 109 x 4096 x 200058 is 89 G multiply-adds — the single largest
        // matmul in the forward, and the one left standing once attention and
        // the MLPs move. There is no host twin: 89 G scalar multiply-adds is
        // not a reference, it is an afternoon. The muP divisor divides BEFORE
        // the projection, matching the reference: doing it after is
        // algebraically equal and numerically not.
        //
        // The rows: `logit_row0..n`, which is the last row alone unless a
        // reporter asked for all of them. The unembedding is the widest matmul
        // in the stack (n x 4096 x 200058) and the only consumer of all but its
        // final row is a print, so the slice happens on the INPUT -- before the
        // GEMM and before the readback -- rather than after both.
        let hx = if all_logits {
            xd.clone()
        } else {
            xd.clone().slice([logit_row0..n, 0..h])
        };
        let hs = dev_lane::rms_norm(
            hx,
            fnorm_dev.clone().expect("the tail owns the final norm"),
            t.rms_norm_eps,
        )
        .div_scalar(t.logits_mup_width_multiplier as f32);
        let uw = unembed_w.as_ref().expect("the tail binds the unembed table");
        down(dev_lane::linear_bf16(hs, uw).slice([0..n - logit_row0, 0..v]))
    };
    let t_head = t_h.elapsed().as_secs_f64();

    // Greedy: the last position's argmax is the next token. A head took it off
    // the wire instead of computing it, and either way it is decided HERE --
    // before the reporting -- so a tail can answer its peer immediately rather
    // than making the head wait on a page of printing.
    let best = match best_wire {
        Some(b) => b,
        None => {
            let last = &logits[(n - 1 - logit_row0) * v..(n - logit_row0) * v];
            let mut best = 0usize;
            for (i, &val) in last.iter().enumerate() {
                if val > last[best] {
                    best = i;
                }
            }
            best
        }
    };
    if let Some(Pipe::Tail(s)) = pipe.as_mut() {
        s.write_all(&(best as i64).to_le_bytes())?;
        s.flush()?;
    }

    // ---- MTP: score the drafts that named this step, then draft afresh -----
    if mtp_k > 0 {
        // SCORE FIRST, against `best` -- the token the full stack just produced.
        // This is the whole experiment: a draft is right or it is not, and the
        // rate over many steps is the only oracle the composition has.
        mtp_pending.retain(|&(target, depth, tok)| {
            if target != step {
                return true;
            }
            mtp_seen[depth] += 1;
            if tok == best {
                mtp_hits[depth] += 1;
            }
            // The step this draft was ISSUED at: head d drafted the token d+1
            // steps ahead, so the issuer is target - depth - 1.
            if let Some(slots) = mtp_issued.get_mut(&(target - depth - 1)) {
                slots[depth] = Some(tok == best);
            }
            println!(
                "  MTP depth {}: drafted {tok}, actual {best} -- {}",
                depth + 1,
                if tok == best { "HIT" } else { "miss" }
            );
            false
        });

        // DRAFT. Head d is fed the previous stage's hidden states and the
        // embeddings of the tokens shifted one further along, so head 0 sees
        // the token the stack just chose and head d sees draft d-1.
        let e_w = embed_w.as_ref().expect("drafting needs the embedding table");
        let fnorm_d = fnorm.as_ref().expect("drafting needs the final norm");
        // Which hidden state head 0 is fed. `x` is the stack's RAW output, which
        // `head` norms on its way to logits. Feeding the FINAL-NORMED one
        // measured twice as well (25% -> 50% on a matched 20-token run), so it
        // is the default; `INK_MTP_RAW=1` is the control that established it.
        //
        // Why it is not merely a scale fix: RMS norm is scale-invariant, so the
        // two differ ONLY by the final norm's learned weight vector applied in
        // between. That the weights help says the heads were trained on
        // post-final-norm hidden states.
        //
        // It also makes `chain_hidden_post_norm: false` coherent, which the raw
        // reading did not. That flag governs the CHAIN — whether a head's output
        // is normed before the next head sees it — and it is off, so stages 1..k
        // stay raw. The ENTRY from the main stack is a separate question and the
        // flag never spoke to it.
        let entry = if std::env::var("INK_MTP_RAW").map(|val| val == "1").unwrap_or(false) {
            x.clone()
        } else {
            rms_norm(&x, &fnorm_d.data, t.rms_norm_eps, n, h)
        };
        // RETAIN it, and that is the whole enabling change. An MTP head's input
        // at position j is (main_hidden[j], embed(token[j+1])), and
        // main_hidden[j] never changes once produced -- attention is causal, so
        // nothing later reaches back and alters it. What stopped the cached lane
        // from drafting was never that the values move; it was that the loop
        // DISCARDED them. 16 KB a token at f32, against a 144 GiB working set.
        //
        // Uncached, every pass recomputes the whole prefix, so this is an
        // assignment there and an append here, and both lanes end holding the
        // same table over the same sequence.
        if kv {
            mtp_main.extend_from_slice(&entry);
        } else {
            mtp_main = entry;
        }
        let seq = ids.len();
        debug_assert_eq!(mtp_main.len(), seq * h, "one retained hidden row per token");

        // One row of logits, argmaxed. The device head returns the FULL vocab
        // width, so index by what came back rather than by whichever constant
        // happens to match, or the argmax silently reads the wrong row.
        //
        // ONE row and not `n` of them: a draft is read off the last position and
        // the head is per-row, so unembedding the prefix was 89 G multiply-adds
        // per position thrown away.
        let draft_argmax = |row: &[f32]| -> usize {
            debug_assert_eq!(row.len(), h, "the draft head unembeds exactly one position");
            let dl = {
                let ud = unembed_w.as_ref().expect("drafting needs the unembed table");
                let hs = dev_lane::rms_norm(
                    up2::<Bk>(row.to_vec(), 1, h, &dev),
                    fnorm_dev.clone().expect("drafting needs the final norm"),
                    t.rms_norm_eps,
                )
                .div_scalar(t.logits_mup_width_multiplier as f32);
                down(dev_lane::linear_bf16(hs, ud).slice([0..1, 0..v]))
            };
            let mut b = 0usize;
            for (i, &val) in dl.iter().take(v).enumerate() {
                if val > dl[b] {
                    b = i;
                }
            }
            b
        };

        // The whole-sequence draft: every head over every position. This is what
        // the uncached lane runs, and what `INK_MTP_CHECK` gates the cached lane
        // against. `hidden` is the ENTRY state for the WHOLE sequence, which is
        // precisely the thing the cache exists so as not to need.
        let draft_whole = |hidden: &[f32], seq: usize, ids: &[usize], best: usize| -> Vec<usize> {
            let mut stage = hidden.to_vec();
            // The token at each position, one step ahead: position j predicts
            // ids[j+1], and the LAST position predicts `best`.
            let mut ahead: Vec<usize> = ids[1..].to_vec();
            ahead.push(best);
            let mut out = Vec::with_capacity(mtp_heads.len());
            for headw in mtp_heads.iter() {
                debug_assert_eq!(ahead.len(), seq, "one shifted token per position");
                let mut embeds = vec![0f32; seq * h];
                for (j, &tok) in ahead.iter().enumerate() {
                    embeds[j * h..(j + 1) * h]
                        .copy_from_slice(&embed_row_bf16(e_w, tok, t.vocab_size, h));
                }
                stage = mtp_block(
                    &stage,
                    &embeds,
                    // DENSE intermediate, not the routed experts' -- every MTP
                    // block is dense regardless of dense_mlp_idx, and the two
                    // sizes differ by 8x (16384 against 2048).
                    &headw.borrow(t.dense_intermediate_size),
                    &headw.dims,
                    Some(ls),
                    headw.window(t.sliding_window_size),
                    seq,
                    mtp_order,
                );
                let b = draft_argmax(&stage[(seq - 1) * h..seq * h]);
                out.push(b);
                ahead.remove(0);
                ahead.push(b);
            }
            out
        };

        let t_mtp = Instant::now();
        let drafts: Vec<usize> = if kv {
            // Head d's newest STABLE row is at position seq-1-d: past that, its
            // input embedding is a token the stack has not produced yet. So a
            // step appends exactly ONE row per head, and the d rows after it --
            // the ones the draft is actually read off -- are functions of drafts
            // and run against a CLONE of the cache that is then dropped.
            // Rollback is the default and there is no commit path to forget,
            // which matters because a speculative K/V left behind does not
            // error: it shows up months later as an acceptance rate that drifts
            // down.
            let mut prev_rows: Vec<Vec<f32>> = Vec::new();
            let mut drafts: Vec<usize> = Vec::with_capacity(mtp_heads.len());
            for (d, headw) in mtp_heads.iter().enumerate() {
                let window = headw.window(t.sliding_window_size);
                let hw = headw.borrow(t.dense_intermediate_size);
                let want = seq - d;
                let have = mtp_stage[d].len() / h;
                let stable: Vec<f32> = if have == 0 {
                    // The prefill: every stable row this head has, in one
                    // whole-sequence pass -- the same call the uncached lane
                    // makes, so the same arithmetic seeds the cache.
                    let mut embeds = vec![0f32; want * h];
                    for j in 0..want {
                        let tok = if j + d + 1 < seq { ids[j + d + 1] } else { best };
                        embeds[j * h..(j + 1) * h]
                            .copy_from_slice(&embed_row_bf16(e_w, tok, t.vocab_size, h));
                    }
                    let hin: &[f32] = if d == 0 {
                        &mtp_main[..want * h]
                    } else {
                        &mtp_stage[d - 1][..want * h]
                    };
                    let (y, cache) = mtp_block_prefill(
                        hin, &embeds, &hw, &headw.dims, Some(ls), window, want, mtp_order,
                    );
                    mtp_caches[d] = Some(cache);
                    y
                } else {
                    // One row, at the position the token just produced made
                    // stable. Its embedding is `best` for EVERY head: head d's
                    // row seq-1-d wants the token at seq, whatever d is.
                    assert_eq!(have + 1, want, "a step makes exactly one row stable");
                    let p = want - 1;
                    let hin: &[f32] = if d == 0 {
                        &mtp_main[p * h..(p + 1) * h]
                    } else {
                        &mtp_stage[d - 1][p * h..(p + 1) * h]
                    };
                    mtp_block_step(
                        hin,
                        &embed_row_bf16(e_w, best, t.vocab_size, h),
                        &hw,
                        &headw.dims,
                        Some(ls),
                        p,
                        window,
                        mtp_caches[d].as_mut().expect("prefilled on the first pass"),
                        mtp_order,
                    )
                };
                mtp_stage[d].extend_from_slice(&stable);
                debug_assert_eq!(mtp_stage[d].len(), want * h);

                // The speculative tail: positions want..seq-1, one per draft
                // already made, each attending to the ones before it. `rows` is
                // what head d+1 reads -- the input to its own stable row, then
                // the inputs to its speculative ones.
                let mut rows: Vec<Vec<f32>> = vec![mtp_stage[d][(want - 1) * h..want * h].to_vec()];
                let mut last = rows[0].clone();
                if d > 0 {
                    let mut scratch = mtp_caches[d].as_ref().expect("prefilled").clone();
                    for i in 0..d {
                        last = mtp_block_step(
                            &prev_rows[i],
                            &embed_row_bf16(e_w, drafts[i], t.vocab_size, h),
                            &hw,
                            &headw.dims,
                            Some(ls),
                            want + i,
                            window,
                            &mut scratch,
                            mtp_order,
                        );
                        rows.push(last.clone());
                    }
                }
                // `scratch` is gone with the block: the speculative K and V
                // never reached the cache, so a wrong draft leaves nothing to
                // undo.
                drafts.push(draft_argmax(&last));
                prev_rows = rows;
            }
            drafts
        } else {
            draft_whole(&mtp_main, seq, &ids, best)
        };
        mtp_issued.insert(step, vec![None; drafts.len()]);
        for (d, &b) in drafts.iter().enumerate() {
            // Head d predicts the token d+1 steps past the one just chosen.
            mtp_pending.push((step + d + 1, d, b));
        }
        println!(
            "  MTP drafted {} token(s) in {:.2}s: {drafts:?}",
            mtp_heads.len(),
            t_mtp.elapsed().as_secs_f32(),
        );
        // The cached lane against the whole-sequence one, on the SAME retained
        // hidden states -- so a disagreement is about the CACHE and nothing
        // else. Both lanes are scalar host arithmetic summed in the same order,
        // so the drafts should agree EXACTLY; anything less is a bug and not
        // rounding, which is what makes this worth asserting rather than
        // reporting.
        if kv && std::env::var("INK_MTP_CHECK").map(|val| val == "1").unwrap_or(false) {
            let t_c = Instant::now();
            let whole = draft_whole(&mtp_main, seq, &ids, best);
            println!(
                "  MTP cache check ({:.2}s): cached {drafts:?} vs whole-sequence {whole:?} -- {}",
                t_c.elapsed().as_secs_f32(),
                if whole == drafts { "agree" } else { "DISAGREE" }
            );
            anyhow::ensure!(
                whole == drafts,
                "the MTP cache drafted {drafts:?} where the whole sequence drafts {whole:?}"
            );
        }
    }
    // A tail follows the sequence by RECOMPUTING it, not by being told: it owns
    // the argmax, so pushing it here keeps its `ids` identical to the head's
    // without a second thing on the wire to get out of step.
    if is_tail && gen_steps > 0 && !repeat {
        ids.push(best);
    }

    println!("\n=== predictions ===");
    println!("  expert slabs decoded: {expert_loads}");
    // t_other covers the whole MLP half, so the expert buckets are inside it.
    // MILLISECONDS. At 0.1 s resolution a decode pass of this stack prints as
    // "0.4 0.4 0.0" and every question worth asking of it is unanswerable.
    let ms = |v: f64| v * 1e3;
    // Two kinds of number, kept apart, because with the residual stream on the
    // device they no longer measure the same thing. Everything above the sync
    // line is the HOST describing work; the device time is the sync line. A
    // per-stage "seconds" column would now report how fast the CPU can enqueue,
    // and it would look wonderful.
    println!("  where the time went, ms:");
    println!("    HOST, enqueue only (nothing in the loop synchronises):");
    println!("      embed + upload  {:9.1}   (the one host->device crossing per pass)", ms(t_embed));
    println!("      attention half  {:9.1}", ms(t_attn));
    println!("      mlp half        {:9.1}   of which:", ms(t_other));
    println!("        routed experts{:9.1}   (slice + bind + issue)", ms(t_expert));
    println!("        shared experts{:9.1}", ms(t_shared));
    println!("        rest of half  {:9.1}   (dense layers, sconv)",
             ms(t_other - t_expert - t_shared - t_h_route - t_h_sconv));
    println!("      router + group  {:9.1}   BLOCKS: [n,{}] logits back, then top-k on the host",
             ms(t_h_route), t.n_routed_experts + t.n_shared_experts);
    println!("        of which: matmul enqueue {:7.1}, BLOCKING read {:7.1}, top-k + group {:7.1}",
             ms(t_rt_mm), ms(t_rt_read), ms(t_rt_host));
    println!("      mlp short_conv  {:9.1}", ms(t_h_sconv));
    println!("      first-touch uploads: read+widen {:9.1}, transfer {:9.1}   (once per layer, not per token)",
             ms(t_attn_read), ms(t_attn_up));
    println!("    DEVICE, one sync for this node's whole stack: {:9.1}", ms(t_stack_sync));
    if stage_sync {
        println!("    DEVICE per stage (INK_STAGE_SYNC=1 -- {stage_syncs} extra syncs, this pass IS slower for them):");
        println!("      attention half  {:9.1}", ms(d_attn));
        println!("      router matmul   {:9.1}", ms(d_router));
        println!("      routed experts  {:9.1}", ms(d_expert));
        println!("      shared experts  {:9.1}", ms(d_shared));
        println!("      sconv + resid   {:9.1}", ms(d_tail));
        println!("      staged total    {:9.1}", ms(d_attn + d_router + d_expert + d_shared + d_tail));
    }
    println!(
        "    {:17} {:9.1}   ({})",
        if best_wire.is_some() { "tail + wire" } else { "head / unembed" },
        ms(t_head),
        if best_wire.is_some() {
            "BLOCKING: the other machine's layers, its head, and the round trip"
        } else {
            "device"
        }
    );
    println!("    of the above, host-only tensor reads (mmap + BF16 widening): {:9.1}", ms(t_read.get()));
    {
        // What the HOST did in the routed-expert lane, one bucket per kind of
        // work. `read (BLOCKS)` is gone from this list because the read is
        // gone: the accumulator is a device tensor and nothing waits for it.
        println!("    of the routed-expert total, what the host did ({expert_loads} loads):");
        println!("      slice from pile {:9.1}   ({:.3} ms/load)   BLAKE3 on first touch",
                 ms(host_t.slice), ms(host_t.slice) / expert_loads.max(1) as f64);
        println!("      gather (select) {:9.1}   ({:.3} ms/load)   enqueue",
                 ms(host_t.gather), ms(host_t.gather) / expert_loads.max(1) as f64);
        println!("      bind + enqueue  {:9.1}   ({:.3} ms/load)",
                 ms(host_t.enqueue), ms(host_t.enqueue) / expert_loads.max(1) as f64);
        println!("      scatter-add     {:9.1}   ({:.3} ms/load)   enqueue",
                 ms(host_t.accum), ms(host_t.accum) / expert_loads.max(1) as f64);
        // WHICH lane ran, counted rather than asserted. The grouped one is a
        // claim about launches per LAYER and the per-expert one about launches
        // per EXPERT; a per-load average over a mixture of the two is a number
        // about neither, so the split has to be visible beside it.
        println!("      lanes: {} layer(s) GROUPED (one launch per stage), {} per-expert",
                 host_t.grouped, host_t.per_expert);
        let named = host_t.slice + host_t.gather + host_t.enqueue + host_t.drain + host_t.accum;
        println!("      remainder       {:9.1}   (whatever the four above did not cover)",
                 ms(t_expert - named));
    }
    // Rule 2, as a number that should be zero and is. Every one of these was
    // scalar f32 arithmetic over the residual stream, on a CPU, between device
    // calls; none of it was control plane. The line stays in the report BECAUSE
    // it reads zero -- a claim of "no host path in the data plane" that nothing
    // measures is a claim that rots.
    println!("    HOST DATA PLANE in the block itself, ms (want: zero):");
    println!("      rms_norm        {:9.1}", ms(t_h_norm));
    println!("      residual adds   {:9.1}", ms(t_h_resid));
    println!("      expert gather   {:9.1}   (a `select` on the device now)", 0.0);
    println!("      expert accum    {:9.1}   (a `select_assign` on the device now)", 0.0);
    println!("      TOTAL           {:9.1}", ms(t_h_norm + t_h_resid));
    let (calls, hits, fileb, hostb, loader_ns) = cp.io_totals();
    let (rb, rn) = cp.resident_bytes();
    println!("  what this ONE pass moved:");
    println!("    loader reads        {calls:8}   answered from RAM {hits:8}");
    println!("    stored bytes        {:8.2} GiB   (what the reads touched, stored precision)",
             fileb as f64 / GIB);
    println!("    host f32 bytes      {:8.2} GiB   (what they became after widening)",
             hostb as f64 / GIB);
    println!("    seconds in loader   {:8.1}", loader_ns as f64 / 1e9);
    println!("    disk read_bytes     {:8.2} GiB   (/proc/self/io -- page-cache hits are free)",
             (io_read_bytes() - io0) as f64 / GIB);
    println!("    resident set        {:8.2} GiB in {rn} weights  (host)", rb as f64 / GIB);
    println!(
        "    device-resident     {:8.2} GiB in {} shared + {} dense layers",
        ddense.bytes as f64 / GIB,
        ddense.shared.len(),
        ddense.dense.len()
    );
    println!(
        "    device-resident     {:8.2} GiB in {} attention layers",
        dattn_bytes as f64 / GIB,
        layers_dev.len()
    );
    // What the zero-copy seam actually achieved on THIS run, at the seam every
    // expert weight passes through. Printed unconditionally: a seam whose hit
    // rate is only visible behind a flag is a seam nobody checks.
    #[cfg(feature = "inkling-cuda")]
    if let Some(al) = fp4_aliases.as_ref() {
        print!("{}", al.stats().report());
    }
    #[cfg(feature = "inkling-cuda")]
    {
        println!(
            "    device-resident     {:8.2} GiB in {} shared + {} dense layers",
            ddense.bytes as f64 / GIB,
            ddense.shared.len(),
            ddense.dense.len()
        );
    }
    #[cfg(feature = "inkling-cuda")]
    if !layers_dev.is_empty() {
        println!(
            "    device-resident     {:8.2} GiB in {} attention layers",
            dattn_bytes as f64 / GIB,
            layers_dev.len()
        );
    }
    if std::env::var("INK_IOSTATS").is_ok() {
        print!("{}", cp.io_table(28));
    }
    report_align();
    println!("  elapsed: {:.1}s", started.elapsed().as_secs_f32());


    // Per-position top-5. Uncached, the final pass has recomputed every
    // position, so it reports all of them and earlier passes report nothing.
    // Cached, each pass computes only the positions it was handed and they
    // accumulate -- prefill contributes the prompt, every step one more -- so
    // the two lanes end with the same table over the same sequence, which is
    // what makes the outputs comparable.
    if !kv {
        top_all.clear();
    }
    // A head has no logits to rank -- the tail owns the table, and writes it.
    if (kv || step == gen_steps) && best_wire.is_none() {
        for ti in logit_row0..n {
            let pos = pos0 + ti;
            let row = &logits[(ti - logit_row0) * v..(ti - logit_row0 + 1) * v];
            let mut idx: Vec<usize> = (0..v).collect();
            idx.sort_unstable_by(|&a, &b| row[b].partial_cmp(&row[a]).unwrap());
            let top: Vec<usize> = idx[..5].to_vec();
            println!("  after token {pos} (id {}): top5 {:?}  logits {:?}",
                     ids[pos], top,
                     top.iter().map(|&i| (row[i] * 100.0).round() / 100.0).collect::<Vec<_>>());
            for &i in &top {
                top_all.push(i as i64);
            }
        }
    }

    if gen_steps > 0 {
        println!("  step {step}: +{best}   [pass {:.1}s, total {:.1}s, ctx {}, pass_ms {:.1}]",
                 pass.elapsed().as_secs_f32(), started.elapsed().as_secs_f32(), ids.len(),
                 pass.elapsed().as_secs_f64() * 1e3);
        // The tail already pushed, when it answered its peer.
        if !is_tail && !repeat {
            ids.push(best);
        }
    }
    if step == gen_steps {
        break;
    }
    }

    // ---- what the router arm changed, if anything -------------------------
    //
    // Printed whether or not it found something, with the examined count on
    // every line. A zero that says how many selections it looked at is a
    // measurement; a zero on its own is a claim.
    if router_diff {
        println!("\n=== router selection: {} vs the f32 [rows,hidden] lane ===", router_arm.label());
        println!("  layer   examined   set!=   order!=   slots!=   max|dlogit|   max|dweight|");
        let (mut ex, mut sd, mut od, mut sl) = (0usize, 0usize, 0usize, 0usize);
        let (mut ml, mut mw) = (0f32, 0f32);
        for (layer, d) in route_diff.iter().enumerate() {
            if d.examined == 0 {
                continue;
            }
            println!(
                "  {layer:5}   {:8}   {:5}   {:7}   {:7}   {:11.3e}   {:12.3e}",
                d.examined, d.set_differs, d.order_differs, d.slots_differ,
                d.max_abs_logit, d.max_abs_weight
            );
            ex += d.examined;
            sd += d.set_differs;
            od += d.order_differs;
            sl += d.slots_differ;
            ml = ml.max(d.max_abs_logit);
            mw = mw.max(d.max_abs_weight);
        }
        if ex == 0 {
            println!("  nothing examined: this node's slice has no MoE layer, so there was no router to compare.");
        } else {
            println!(
                "  TOTAL   {ex:8}   {sd:5}   {od:7}   {sl:7}   {ml:11.3e}   {mw:12.3e}"
            );
            println!(
                "  {:.4}% of {ex} selections chose a different SET of experts; {:.4}% reordered one.",
                100.0 * sd as f64 / ex as f64,
                100.0 * od as f64 / ex as f64,
            );
            println!(
                "  {sl} of {} expert slots named a different expert.",
                ex * t.num_experts_per_tok
            );
        }
    }

    // ---- the MTP experiment's result --------------------------------------
    // Reported whether or not it looks good. A composition that drafts badly is
    // the measurement working, not the run failing, and burying a 0% would make
    // the next window repeat the experiment.
    if mtp_k > 0 {
        println!("\n=== MTP acceptance, concat {} ===", mtp_order.name());
        let scored: usize = mtp_seen.iter().sum();
        if scored == 0 {
            println!("  nothing scored -- every draft named a step past the end of the run.");
            println!("  raise INK_GEN above INK_MTP so drafts have a token to be judged against.");
        } else {
            for d in 0..mtp_k {
                if mtp_seen[d] == 0 {
                    println!("  depth {}: never scored", d + 1);
                    continue;
                }
                let (c_lo, c_hi) = wilson95(mtp_hits[d], mtp_seen[d]);
                println!(
                    "  depth {}: {:5}/{:<5} = {:5.1}%   95% CI [{:5.1}%, {:5.1}%]",
                    d + 1,
                    mtp_hits[d],
                    mtp_seen[d],
                    100.0 * mtp_hits[d] as f64 / mtp_seen[d] as f64,
                    100.0 * c_lo,
                    100.0 * c_hi
                );
            }
            let hits: usize = mtp_hits.iter().sum();
            let (c_lo, c_hi) = wilson95(hits, scored);
            println!(
                "  pooled : {hits:5}/{scored:<5} = {:5.1}%   95% CI [{:5.1}%, {:5.1}%]",
                100.0 * hits as f64 / scored as f64,
                100.0 * c_lo,
                100.0 * c_hi
            );
            // Pooling across depths is a number to read carefully: it averages a
            // depth-1 rate with a depth-4 one, so it moves when INK_MTP moves and
            // is a fact about the configuration as much as about the model. It is
            // kept because a pooled ZERO settles the concat question at a glance;
            // the per-depth rows are what a speculation decision reads.

            // What an accept-and-skip loop would actually keep. Only draft sets
            // every depth of which got scored count — a set whose deeper heads
            // named steps past the end of the run has no prefix length yet, and
            // counting it as "prefix ended here" would bias the mean down by
            // exactly the runs that ran out of tokens.
            let complete: Vec<&Vec<Option<bool>>> =
                mtp_issued.values().filter(|v| v.iter().all(|s| s.is_some())).collect();
            if !complete.is_empty() {
                let sets = complete.len();
                let mut hist = vec![0usize; mtp_k + 1];
                for v in &complete {
                    let mut run = 0usize;
                    while run < v.len() && v[run] == Some(true) {
                        run += 1;
                    }
                    hist[run] += 1;
                }
                let mean: f64 = hist
                    .iter()
                    .enumerate()
                    .map(|(l, &c)| l as f64 * c as f64)
                    .sum::<f64>()
                    / sets as f64;
                println!("  accepted prefix, over {sets} fully-scored draft sets:");
                for (l, &c) in hist.iter().enumerate() {
                    println!("    {l} accepted: {c:5}   ({:5.1}%)", 100.0 * c as f64 / sets as f64);
                }
                println!("    mean {mean:.3} draft tokens accepted per verify pass");
            }
            // What the number MEANS, said here rather than left to the reader,
            // because the whole point of this experiment is that a low rate is
            // evidence about the composition and not about the model.
            println!(
                "  {}",
                if hits * 4 >= scored * 3 {
                    "high -- this reading of mtp_hidden_states_first computes something the \
                     stack agrees with"
                } else if hits == 0 {
                    "zero -- either this concat order is wrong or the wrapper composes some \
                     other way; run INK_MTP_ORDER with the other order before concluding"
                } else {
                    "partial -- better than chance over a 200k vocabulary, so the composition is \
                     close to right; the shortfall is what to explain next"
                }
            );
        }
    }

    // A zero-length batch is the head saying it is done, so the tail's loop
    // ends on a read rather than blocking forever on a peer that has exited.
    if let Some(Pipe::Head(s)) = pipe.as_mut() {
        let _ = send_stream(s, 0, 0, &[]);
    }

    // The head has no top-5 table -- the tail computed the logits and wrote it.
    // Writing an empty file here would look like a run that produced nothing.
    if is_head {
        println!("  head done; the tail wrote the top-5 table. ids {ids:?}");
        return Ok(());
    }
    let mut bytes = Vec::new();
    for i in top_all {
        bytes.extend_from_slice(&i.to_le_bytes());
    }
    std::fs::write(&out_path, &bytes)?;
    println!("  wrote top-5 ids per position to {}", out_path.display());
    Ok(())
}
