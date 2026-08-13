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
//!   INK_LAYERS=20:42 INK_PIPE=tail:0.0.0.0:7654 inkling_forward <ckpt> <ids> <out>
//!   # head, on the first
//!   INK_LAYERS=0:20  INK_PIPE=head:<tail-host>:7654 inkling_forward <ckpt> <ids> <out>
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
//! Mixed precision is not a second lane either. Layer 2's experts are BF16 and
//! the other 41 layers' are NVFP4, so the routed block picks the instruction the
//! stored format calls for — `mma.sync…bf16` or the block-scaled
//! `…mxf4nvf4.block_scale` — and nothing else changes. Both accumulate in f32,
//! which is the MMA's own output type and not a widening; widening the BF16
//! weight to f32 to reuse the f32 path is the exact thing this file does not do.
//!
//! # Where the weights come from
//!
//! `<ckpt>` is a safetensors checkpoint directory. `INK_PILE=<path>` swaps the
//! source for a pile on branch `INK_PILE_BRANCH` (default `inkling`), and it
//! swaps ALL of it: the weights and `config.json` both come from whichever
//! source is named, so a pile-backed run reads nothing from the directory. It
//! used to read the config there regardless, which meant `INK_PILE` moved 159
//! GiB out of the checkpoint and left the run depending on the 40 KB still in
//! it. One environment variable is the whole A/B, which is the point:
//! everything below this line is the same code either way.
//!
//! The argument is still a path so a checkpoint-backed run has somewhere to
//! read from; with `INK_PILE` set it is only a label, and `--` will do.
//!
//!   cargo run --release --features inkling-cuda,cuda-backend,import \
//!       --bin inkling_forward -- <ckpt> <ids.bin> <out.bin>

use std::collections::BTreeMap;
use std::io::{Read as _, Write as _};
use std::net::{TcpListener, TcpStream};
use std::path::PathBuf;
use std::time::Instant;

use anyhow::{Context, Result};

use mary::models::inkling::attn::{causal_mask, AttnDims, AttnWeights, LogScaling};
use mary::models::inkling::block::{
    conv_history, rms_norm, route, short_conv, short_conv_step, Routing,
};
use mary::models::inkling::config::{AttnKind, InklingConfig};
use mary::models::inkling::load::{split_gate_up, Held};
use mary::models::inkling::source::Weights;
use mary::models::inkling::mtp::{
    mtp_block, mtp_block_prefill, mtp_block_step, Concat as MtpConcat, MtpCache, MtpHead,
};
use mary::models::inkling::layer::{LayerMlp, LayerWeights};
use mary::models::inkling::stack::embed_and_norm;

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

/// The same, from a BORROWED slice — for weights that are held on the host.
///
/// The owning [`up2`] exists so a 537 MB dense weight is moved rather than
/// duplicated. A resident weight cannot be moved (the run keeps it), so this
/// copies; the copy is unavoidable and is stated rather than hidden.
fn up2r<B: Backend>(v: &[f32], rows: usize, cols: usize, dev: &B::Device) -> BT<B, 2> {
    assert_eq!(v.len(), rows * cols, "{} values are not [{rows}, {cols}]", v.len());
    BT::from_data(BTD::new(v.to_vec(), [rows, cols]), dev)
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

/// One layer's shared experts, on the device.
struct SharedOnDevice {
    gate: Vec<T2>,
    up: Vec<T2>,
    down: Vec<T2>,
}

/// The dense weights that live in DEVICE-allocated memory for the whole run.
///
/// Distinct from the host-resident set behind `INK_RESIDENT`, and the
/// difference is the point. Host residency stops the re-read and the re-widen;
/// it leaves the arithmetic on the CPU. Device residency moves the weight into
/// the pool the GPU reads fastest and lets the matmul run there, so the token
/// costs a kernel over memory the device already owns rather than an upload.
///
/// These weights are NOT also held on the host: they are read out of the
/// checkpoint once, uploaded, and the host copy is dropped. Holding both would
/// double a budget for no gain, since after the upload nothing on the host ever
/// reads them again.
#[derive(Default)]
struct DeviceDense {
    shared: std::collections::BTreeMap<String, SharedOnDevice>,
    dense: std::collections::BTreeMap<String, (T2, T2, T2, f32)>,
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
            mlp: LayerMlp::Dense {
                gate: &self.gate,
                up: &self.up,
                down: &self.down.data,
                global_scale: self.global_scale,
                inter,
            },
        }
    }
}

impl DeviceDense {
    fn up2(v: Vec<f32>, rows: usize, cols: usize, dev: &burn::backend::cuda::CudaDevice) -> T2 {
        assert_eq!(v.len(), rows * cols, "{} values for [{rows}, {cols}]", v.len());
        burn::tensor::Tensor::from_data(burn::tensor::TensorData::new(v, [rows, cols]), dev)
    }

    /// One layer's shared experts, uploaded on first use.
    #[allow(clippy::too_many_arguments)]
    fn shared_for(
        &mut self,
        cp: &Weights,
        p: &str,
        n_shared: usize,
        inter: usize,
        h: usize,
        halved: bool,
        dev: &burn::backend::cuda::CudaDevice,
    ) -> Result<&SharedOnDevice> {
        if !self.shared.contains_key(p) {
            let fused = cp.tensor(&format!("{p}mlp.shared_experts.shared_w13_weight"))?;
            let (g, u) = mary::models::inkling::load::split_shared_w13(
                &fused.data, n_shared, inter, h, halved,
            );
            drop(fused);
            let d = cp.tensor(&format!("{p}mlp.shared_experts.shared_w2_weight"))?;
            let (per_gu, per_d) = (inter * h, h * inter);
            let mut sd = SharedOnDevice { gate: Vec::new(), up: Vec::new(), down: Vec::new() };
            for e in 0..n_shared {
                sd.gate.push(Self::up2(g[e * per_gu..(e + 1) * per_gu].to_vec(), inter, h, dev));
                sd.up.push(Self::up2(u[e * per_gu..(e + 1) * per_gu].to_vec(), inter, h, dev));
                sd.down
                    .push(Self::up2(d.data[e * per_d..(e + 1) * per_d].to_vec(), h, inter, dev));
            }
            self.bytes += (n_shared * (2 * per_gu + per_d) * 4) as u64;
            self.shared.insert(p.to_string(), sd);
        }
        Ok(&self.shared[p])
    }

    /// One dense layer's MLP, uploaded on first use.
    fn dense_for(
        &mut self,
        cp: &Weights,
        p: &str,
        h: usize,
        dev: &burn::backend::cuda::CudaDevice,
    ) -> Result<&(T2, T2, T2, f32)> {
        if !self.dense.contains_key(p) {
            let fused = cp.tensor(&format!("{p}mlp.w13_dn.weight"))?;
            let (g, u) = split_gate_up(&fused.data, h);
            drop(fused);
            let down = cp.tensor(&format!("{p}mlp.w2_md.weight"))?;
            let gs = cp.tensor(&format!("{p}mlp.global_scale"))?;
            let inter = g.len() / h;
            self.bytes += ((g.len() + u.len() + down.data.len()) * 4) as u64;
            let trip = (
                Self::up2(g, inter, h, dev),
                Self::up2(u, inter, h, dev),
                Self::up2(down.data, h, inter, dev),
                gs.data[0],
            );
            self.dense.insert(p.to_string(), trip);
        }
        Ok(&self.dense[p])
    }
}

/// Every routed expert for one layer, on the NATIVE NVFP4 tensor-core path.
///
/// The only routed lane there is. The packed bytes go straight into
/// `mma.sync…kind::mxf4nvf4…ue4m3`. The lane this replaced (`INK_EXPERTS=gpu`)
/// decoded each expert into a 67.1 + 33.6 MB f32 pair, multiplied THAT, and
/// dropped it — 100 MB of device memory materialised per expert to hold a
/// weight the checkpoint stores in 12.6, four times a token per layer. It is
/// gone rather than kept as a control, because the control was a widening and
/// the whole point of an NVFP4 checkpoint is that the weight is never widened.
///
/// The slab is a BORROW either way — of a checkpoint shard's mapping or of the
/// pile's — and where it came from is not visible here. It used to be: this
/// lane took a second, parallel safetensors reader that existed only because
/// the first one re-parsed a shard header on every accessor call. Both of those
/// facts stopped being true, and the reader went with them.
///
/// Activations are quantised to E2M1 in dynamic per-16 blocks with E4M3
/// scales, which the instruction requires and which is what the checkpoint's
/// own `hf_quant_config.json` specifies for `*input_quantizer`. This lane is
/// therefore CLOSER to the checkpoint's intended numerics than the f32-
/// activation lane it replaces, not further from it.
#[allow(clippy::too_many_arguments)]
fn routed_experts_fp4(
    src: &Weights,
    aliases: Option<&mary::models::inkling::fp4gemm::Aliases>,
    client: &cubecl::prelude::ComputeClient<cubecl::cuda::CudaRuntime>,
    prefix: &str,
    by_expert: &BTreeMap<usize, Vec<(usize, f32)>>,
    hn: &[f32],
    n: usize,
    h: usize,
    inter: usize,
    host: &mut (f64, f64),
) -> Result<Vec<f32>> {
    use cubecl::prelude::CubeElement;
    use mary::models::inkling::fp4gemm::{fp4_linear_launch, gate_up_silu_launch, MTILE};
    use mary::models::inkling::fp4quant::quantize_nvfp4;

    // Zero copy where the hardware allows it: the GPU reads the source's own
    // mapped pages in place. The mappings were registered ONCE at startup, so
    // this is offset arithmetic on a pointer, not a device round trip.
    let bind = |data: &[u8]| match aliases {
        Some(al) => al.slice_or_copy(client, data),
        None => client.create_from_slice(data),
    };

    let n13 = format!("{prefix}mlp.experts.w13_weight");
    let n2 = format!("{prefix}mlp.experts.w2_weight");
    let mut acc = vec![0f32; n * h];
    let mut pending: Vec<(&Vec<(usize, f32)>, cubecl::server::Handle)> =
        Vec::with_capacity(by_expert.len());

    for (&e, toks) in by_expert {
        let t_s = Instant::now();
        let w13 = src.expert_packed(&n13, e)?;
        let w2 = src.expert_packed(&n2, e)?;
        host.0 += t_s.elapsed().as_secs_f64();

        let t_w = Instant::now();
        let m = toks.len();
        let mut x = vec![0f32; m * h];
        for (i, &(ti, _)) in toks.iter().enumerate() {
            x[i * h..(i + 1) * h].copy_from_slice(&hn[ti * h..(ti + 1) * h]);
        }

        // Quantise on the DEVICE, both times. The host lane this replaces had
        // to bring the intermediate activation back across the bus between the
        // two GEMMs purely to requantise it; `act_h` never leaves the device.
        let m_pad = m.div_ceil(MTILE) * MTILE;
        let mut xp = vec![0f32; m_pad * h];
        xp[..m * h].copy_from_slice(&x);
        let x_h = client.create_from_slice(f32::as_bytes(&xp));
        let (a, asc) = quantize_nvfp4(client, &x_h, m_pad, h);

        let (b, bsc) = (bind(w13.codes()), bind(w13.scales()));
        let both = fp4_linear_launch(client, &a, &asc, &b, &bsc, m_pad, h, 2 * inter, w13.scale2());

        let act_h = gate_up_silu_launch(client, &both, m_pad, inter);
        let (a2, asc2) = quantize_nvfp4(client, &act_h, m_pad, inter);

        let (b2, bsc2) = (bind(w2.codes()), bind(w2.scales()));
        let y_h = fp4_linear_launch(client, &a2, &asc2, &b2, &bsc2, m_pad, inter, h, w2.scale2());
        pending.push((toks, y_h));
        host.1 += t_w.elapsed().as_secs_f64();
    }

    // ONE sync for the layer, not one per expert. Reading each expert's result
    // the moment it was enqueued drained the queue 234 times a token and left
    // the device idle across the host's next slab; the queue is the whole point
    // of having one. `pending` is in the `BTreeMap` order the loop walked, so
    // the accumulation is the same sum in the same order and the result is
    // bit-identical -- which is what makes this a scheduling change and not a
    // numerics one.
    let t_r = Instant::now();
    for (toks, y_h) in pending {
        let y = f32::from_bytes(&client.read_one(y_h).expect("read y")).to_vec();
        for (i, &(ti, wgt)) in toks.iter().enumerate() {
            for o in 0..h {
                acc[ti * h + o] += y[i * h + o] * wgt;
            }
        }
    }
    host.1 += t_r.elapsed().as_secs_f64();
    Ok(acc)
}

/// The same lane for layer 2, whose experts are BF16.
///
/// Deliberately the same shape as [`routed_experts_fp4`], line for line: the
/// same grouping by expert, the same pointer-containment binding, the same ONE
/// sync for the layer with the accumulation in `BTreeMap` order. What the
/// format takes away is all that differs — no block scales to bind, no `scale2`
/// to fold in, no activation quantiser. What it puts back is one cast: the MMA
/// takes the same type on both operands, so the f32 residual stream is rounded
/// to BF16 on the device before it enters. That is not a liberty — the
/// reference implementation runs this layer in BF16 throughout, so a BF16
/// activation is what `transformers` multiplies too.
fn routed_experts_bf16(
    src: &Weights,
    aliases: Option<&mary::models::inkling::fp4gemm::Aliases>,
    client: &cubecl::prelude::ComputeClient<cubecl::cuda::CudaRuntime>,
    prefix: &str,
    by_expert: &BTreeMap<usize, Vec<(usize, f32)>>,
    hn: &[f32],
    n: usize,
    h: usize,
    inter: usize,
    host: &mut (f64, f64),
) -> Result<Vec<f32>> {
    use cubecl::prelude::CubeElement;
    use mary::models::inkling::bf16gemm::{bf16_linear_launch, upload_bf16_act};
    use mary::models::inkling::fp4gemm::gate_up_silu_bf16_launch;

    let bind = |data: &[u8]| match aliases {
        Some(al) => al.slice_or_copy(client, data),
        None => client.create_from_slice(data),
    };

    let n13 = format!("{prefix}mlp.experts.w13_weight");
    let n2 = format!("{prefix}mlp.experts.w2_weight");
    let mut acc = vec![0f32; n * h];
    let mut pending: Vec<(&Vec<(usize, f32)>, cubecl::server::Handle)> =
        Vec::with_capacity(by_expert.len());

    for (&e, toks) in by_expert {
        let t_s = Instant::now();
        let w13 = src.expert_bf16(&n13, e)?;
        let w2 = src.expert_bf16(&n2, e)?;
        host.0 += t_s.elapsed().as_secs_f64();

        let t_w = Instant::now();
        let m = toks.len();
        let mut x = vec![0f32; m * h];
        for (i, &(ti, _)) in toks.iter().enumerate() {
            x[i * h..(i + 1) * h].copy_from_slice(&hn[ti * h..(ti + 1) * h]);
        }

        let (a, m_pad) = upload_bf16_act(client, &x, m, h);
        let b = bind(w13.bytes());
        let both = bf16_linear_launch(client, &a, &b, m_pad, h, 2 * inter);

        // The intermediate never leaves the device and never becomes f32 on the
        // host: `gate_up_silu` writes BF16 straight into the second MMA's A
        // operand.
        let act = gate_up_silu_bf16_launch(client, &both, m_pad, inter);
        let b2 = bind(w2.bytes());
        let y_h = bf16_linear_launch(client, &act, &b2, m_pad, inter, h);
        pending.push((toks, y_h));
        host.1 += t_w.elapsed().as_secs_f64();
    }

    let t_r = Instant::now();
    for (toks, y_h) in pending {
        let y = f32::from_bytes(&client.read_one(y_h).expect("read y")).to_vec();
        for (i, &(ti, wgt)) in toks.iter().enumerate() {
            for o in 0..h {
                acc[ti * h + o] += y[i * h + o] * wgt;
            }
        }
    }
    host.1 += t_r.elapsed().as_secs_f64();
    Ok(acc)
}

fn main() -> Result<()> {
    let ckpt = std::env::args().nth(1).map(PathBuf::from).context("usage: <ckpt> <ids> <out>")?;
    let ids_path = std::env::args().nth(2).map(PathBuf::from).context("usage: <ckpt> <ids> <out>")?;
    let out_path = std::env::args().nth(3).map(PathBuf::from).context("usage: <ckpt> <ids> <out>")?;

    // The one line that decides where the weights come from. `INK_PILE` swaps
    // the source; nothing downstream of here asks which it was.
    let pile_path = std::env::var("INK_PILE").ok();
    let pile_branch = std::env::var("INK_PILE_BRANCH").unwrap_or_else(|_| "inkling".to_string());
    let t_open = Instant::now();
    let cp = match &pile_path {
        Some(p) => Weights::open_pile(p, &pile_branch)?,
        None => Weights::open_ckpt(&ckpt)?,
    };
    let open_secs = t_open.elapsed().as_secs_f64();

    // …and the config comes from the SAME source. It used to come from the
    // checkpoint directory unconditionally, which meant `INK_PILE` moved 159 GiB
    // out of the directory and left the run depending on the 40 KB still in it:
    // a pile that cannot answer this is not authoritative, only large. In a pile
    // the config is FACTS (one entity per JSON scalar, `mary::jsonfacts`), so
    // this is a query, not a stored file being read back.
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
                "the weight source carries no config.json. For a pile, ingest \
                 the checkpoint's sidecars as facts (inkling_meta_gate <ckpt> \
                 <pile>), or point INK_CONFIG at the file to run without them",
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
            Some(p) => format!("INK_CONFIG={p}  (OVERRIDE -- the source was not asked)"),
            None => format!(
                "config.json from the {} ({})",
                cp.kind(),
                pile_path.as_deref().unwrap_or(&ckpt.display().to_string())
            ),
        }
    );
    println!(
        "  weights    : {} {}  (index built in {open_secs:.1}s)",
        cp.kind(),
        pile_path.as_deref().unwrap_or(&ckpt.display().to_string()),
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
    // Not the same unit on both sides, and saying so is the point: the
    // checkpoint names 1 360 tensors of which the expert ones are STACKS of
    // 256, the pile names each expert leaf on its own — 968 dense + 20 480
    // experts. Same model, two granularities, and the pile's is the one a
    // layer split can partition.
    println!("  {}", cp.inventory());

    // How many tokens to generate past the prompt. 0 reproduces the original
    // single-forward behaviour exactly.
    let gen_steps: usize = std::env::var("INK_GEN")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(0);

    let started = Instant::now();
    // Hoisted: re-reading 4.8 GB of embedding tables per generated token would
    // dwarf everything else in the loop.
    //
    // Split by role, and NOT loaded by the end that will never read them. That
    // is not tidiness: the whole reason for two machines is that this model's
    // working set does not fit in one page cache, and 4.8 GB of embedding
    // pinned on the box that only unembeds is 4.8 GB of expert slabs evicted.
    let want_embed = !is_tail;
    let want_head = !is_head;
    let embed_w = if want_embed { Some(cp.held("model.llm.embed.weight")?) } else { None };
    let embed_n = if want_embed { Some(cp.held("model.llm.embed_norm.weight")?) } else { None };
    let fnorm = if want_head { Some(cp.held("model.llm.norm.weight")?) } else { None };
    // `Option`, so the tail can DROP its handle on the 3.3 GB host copy once the
    // table is on the device and nothing on the host will read it again.
    let unembed = if want_head { Some(cp.held("model.llm.unembed.weight")?) } else { None };
    println!("  embedding tables loaded in {:.1}s", started.elapsed().as_secs_f32());

    // `INK_MTP=k` drafts k tokens ahead with the MTP heads and scores them
    // against what the stack actually generates. It measures ACCEPTANCE, which
    // is the only oracle the composition has -- see mary::models::inkling::mtp.
    // Read here rather than beside the other lane switches because the heads are
    // loaded with the embedding tables, and the switch has to precede its use.
    let mtp_k: usize = std::env::var("INK_MTP").ok().map(|v| v.parse()).transpose()?.unwrap_or(0);
    let mtp_order = match std::env::var("INK_MTP_ORDER") {
        Ok(v) => MtpConcat::parse(&v)
            .with_context(|| format!("INK_MTP_ORDER wants hidden|embed, got {v:?}"))?,
        Err(_) => MtpConcat::HiddenFirst,
    };

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
        mtp_k == 0 || pipe_spec.is_none(),
        "INK_MTP needs one process to own both ends -- the head owns no unembedding and the tail \
         owns no embedding table, so neither can draft alone. Which means drafting cannot run at \
         all on a stack that no longer fits one node; the heads and the check are kept because \
         the composition they gate is what a cross-node draft would be built from."
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
    let mut dattn: std::collections::BTreeMap<String, (dev_lane::AttnWeightsDev<Bk>, T2)> =
        std::collections::BTreeMap::new();
    let mut dattn_bytes = 0u64;
    // Parsed once for the whole run. The lane it replaces re-parsed a shard
    // header four times per expert slab, ~9950 times over a forward.
    let fp4_client = {
        use cubecl::prelude::Runtime;
        cubecl::cuda::CudaRuntime::client(&Default::default())
    };
    // Nine blocking device round trips for the whole run, instead of four per
    // expert. Every later slab is an offset view of one of these.
    //
    // Always an `Aliases`, even when nothing can be aliased:
    // `Aliases::disabled()` copies exactly as the old `None` arm did but COUNTS
    // it, so a source whose bytes cannot be aliased reports that rather than
    // going quiet. The registration is unconditional — on a unified-memory part
    // a copy of a weight the device can read where it lies is a copy for
    // nothing, and there is no configuration in which we want one.
    #[cfg(feature = "inkling-cuda")]
    let fp4_aliases = {
        let c = &fp4_client;
        {
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

    // The unembed table is 3.3 GB at f32 and does not change between generated
    // tokens, so it is uploaded ONCE here rather than once per step, and the
    // host copy is dropped rather than kept alongside it.
    let unembed_dev = if want_head {
        let v = t.effective_vocab();
        let d = up2r::<Bk>(&unembed.as_ref().expect("unembed held").data, t.vocab_size, h, &dev)
            .slice([0..v, 0..h]);
        println!("  unembed uploaded, {v} x {h}");
        Some(d)
    } else {
        None
    };
    // The host copy has no reader left: the head is on the device and there is
    // no host head lane to fall back to, and the draft head reads the device
    // table too. Residency still holds it, which is what residency IS, not a
    // leak.
    drop(unembed);

    // Everything one layer carries between generated tokens. The attention
    // cache is the headline, but the two layer-level short convolutions have
    // state too: they reach `kernel - 1` positions back, and a cache that
    // remembers K and V while forgetting those is wrong in a way that still
    // produces fluent-looking text.
    struct LayerCache {
        attn: dev_lane::AttnCache<Bk>,
        attn_sconv: BT<Bk, 2>,
        mlp_sconv: Vec<f32>,
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
    let (n, pos0, mut x) = match incoming {
        Some((n, p, x)) => (n, p, x),
        None => {
            let n = feed.len();
            let e_w = embed_w.as_ref().expect("the head owns the embedding table");
            let e_n = embed_n.as_ref().expect("the head owns the embedding norm");
            (n, pos0, embed_and_norm(&feed, &e_w.data, &e_n.data, t.rms_norm_eps, t.vocab_size, h))
        }
    };

    if let Ok(dir) = std::env::var("INK_DUMP_DIR") {
        std::fs::create_dir_all(&dir)?;
        let mut bytes = Vec::with_capacity(x.len() * 4);
        for v in &x {
            bytes.extend_from_slice(&v.to_le_bytes());
        }
        std::fs::write(format!("{dir}/h_embed.bin"), &bytes)?;
    }

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
    // (slice, widen+upload) -- host-side and therefore honestly attributable,
    // unlike anything downstream of an enqueued device call.
    let mut host_t = (0f64, 0f64);

    for layer in lo..hi {
        // Cache slot, not layer number. A tail running 20..42 keeps 22 caches
        // and its first layer is its slot 0 — indexing by the absolute layer
        // would walk off the end of a Vec that only ever holds this node's half.
        let slot = layer - lo;
        let l0 = Instant::now();
        let kind = t.attn_kind(layer);
        let is_local = kind == AttnKind::Local;
        let (heads, kv_heads, head_dim) = t.heads(kind);
        let p = format!("model.llm.layers.{layer}.");
        // Two accessors, because host lanes and device lanes want opposite
        // things from the same read. `g` HOLDS (INK_RESIDENT makes it a pointer
        // copy after the first token) and is what the host lanes read. `gv`
        // hands over an owned buffer that is moved into an upload and dropped —
        // a device lane never reads the host copy again, so holding it would
        // double the budget for no gain. Both charge the same read timer.
        let g = |nm: &str| -> Result<Held> {
            let s = Instant::now();
            let r = cp.held(&format!("{p}{nm}"))?;
            t_read.set(t_read.get() + s.elapsed().as_secs_f64());
            Ok(r)
        };
        let gv = |nm: &str| -> Result<Vec<f32>> {
            let s = Instant::now();
            let r = cp.tensor(&format!("{p}{nm}"))?.data;
            t_read.set(t_read.get() + s.elapsed().as_secs_f64());
            Ok(r)
        };

        // ---- attention ----------------------------------------------------
        let t_a = Instant::now();
        let attn_norm = g("attn_norm.weight")?;
        let hn = rms_norm(&x, &attn_norm.data, t.rms_norm_eps, n, h);
        let dims = AttnDims {
            hidden: h, heads, kv_heads, head_dim,
            d_rel: t.d_rel,
            rel_extent: t.rel_span(kind),
            kernel: t.sconv_kernel_size,
            rms_eps: t.rms_norm_eps,
            kind,
        };
        let mask = if is_local { &mask_local } else { &mask_global };
        // The same distinction the mask carries, in the form the cache needs:
        // how far back a query may look, and therefore how much of the cache
        // can never be read again.
        let window = if is_local { Some(t.sliding_window_size) } else { None };
        let a = {
                // Both projections and the two short convolutions go over, and
                // so does the layer-level `attn_sconv` that follows: leaving one
                // of the five on the host would pay a round trip to save nothing.
                //
                // Built on the FIRST token that reaches this layer and then held
                // (see `dattn`). The timers therefore measure a first-token cost
                // under residency and a per-token cost without it, which is the
                // difference the flag names. The sync is what makes the second
                // one a transfer rather than an enqueue.
                if !dattn.contains_key(&p) {
                    let r0 = t_read.get();
                    let t_w0 = Instant::now();
                    let built = dev_lane::AttnWeightsDev::<Bk> {
                        wq: up2(gv("attn.wq_du.weight")?, heads * head_dim, h, &dev),
                        wk: up2(gv("attn.wk_dv.weight")?, kv_heads * head_dim, h, &dev),
                        wv: up2(gv("attn.wv_dv.weight")?, kv_heads * head_dim, h, &dev),
                        wr: up2(gv("attn.wr_du.weight")?, heads * t.d_rel, h, &dev),
                        wo: up2(gv("attn.wo_ud.weight")?, h, heads * head_dim, &dev),
                        k_sconv: up2(gv("attn.k_sconv.weight")?, kv_heads * head_dim, t.sconv_kernel_size, &dev),
                        v_sconv: up2(gv("attn.v_sconv.weight")?, kv_heads * head_dim, t.sconv_kernel_size, &dev),
                        q_norm: up1(gv("attn.q_norm.weight")?, head_dim, &dev),
                        k_norm: up1(gv("attn.k_norm.weight")?, head_dim, &dev),
                        rel_proj: up2(gv("attn.rel_logits_proj.proj")?, t.d_rel, t.rel_span(kind), &dev),
                    };
                    let sconv = up2(gv("attn_sconv.weight")?, h, t.sconv_kernel_size, &dev);
                    <Bk as burn::tensor::backend::Backend>::sync(&dev)
                        .expect("sync after the attention uploads");
                    let span = t_w0.elapsed().as_secs_f64();
                    let rd = t_read.get() - r0;
                    t_attn_read += rd;
                    t_attn_up += span - rd;
                    {
                        dattn_bytes += 4 * (heads * head_dim * h
                            + 2 * kv_heads * head_dim * h
                            + heads * t.d_rel * h
                            + h * heads * head_dim
                            + 2 * kv_heads * head_dim * t.sconv_kernel_size
                            + 2 * head_dim
                            + t.d_rel * t.rel_span(kind)
                            + h * t.sconv_kernel_size) as u64;
                    }
                    dattn.insert(p.clone(), (built, sconv));
                }
                let (w, sconv_w) = {
                    let e = dattn.get(&p).expect("inserted directly above");
                    // The projections are BORROWED for the call; only the layer
                    // sconv is cloned, and a Burn clone is a handle, not 3.3 MB.
                    (&e.0, e.1.clone())
                };
                let out = if kv && step > 0 {
                    let y = dev_lane::attention_step(
                        up2(hn.clone(), n, h, &dev),
                        &w,
                        &dims,
                        Some(ls),
                        pos0,
                        window,
                        &mut caches[slot].attn,
                    );
                    let (out, hist) = dev_lane::short_conv_step(
                        caches[slot].attn_sconv.clone(),
                        y,
                        sconv_w,
                    );
                    caches[slot].attn_sconv = hist;
                    down(out)
                } else if kv {
                    let (y, attn) = dev_lane::attention_prefill(
                        up2(hn.clone(), n, h, &dev),
                        &w,
                        &dims,
                        Some(ls),
                        up2(mask.clone(), n, n, &dev),
                        window,
                    );
                    let hist = dev_lane::conv_history(y.clone(), t.sconv_kernel_size);
                    let out = down(dev_lane::short_conv(y, sconv_w));
                    caches.push(LayerCache { attn, attn_sconv: hist, mlp_sconv: Vec::new() });
                    out
                } else {
                    let y = dev_lane::attention(
                        up2(hn.clone(), n, h, &dev),
                        &w,
                        &dims,
                        Some(ls),
                        up2(mask.clone(), n, n, &dev),
                    );
                    down(dev_lane::short_conv(y, sconv_w))
                };
                // The map grows to this node's whole share and stays. It used
                // to be emptied per layer when `INK_RESIDENT` was off, so that
                // a box too small for the working set could still stream --
                // which is the configuration this binary now refuses to be.
                out
        };
        for (xi, ai) in x.iter_mut().zip(&a) {
            *xi += ai;
        }

        // ---- MLP ----------------------------------------------------------
        t_attn += t_a.elapsed().as_secs_f64();
        let t_o = Instant::now();
        let mlp_norm = g("mlp_norm.weight")?;
        let hn = rms_norm(&x, &mlp_norm.data, t.rms_norm_eps, n, h);

        let y = if t.is_dense(layer) {
            // Device-resident: uploaded on the first token that reaches this
            // layer and held for the run. The host reference that used to sit
            // beside it (`host_dense`, selected by leaving `INK_DENSE` unset)
            // was a scalar f32 lane over a 537 MB weight; it is not a lane a
            // 276 B model has any use for, and being selectable is how it got
            // run by accident.
            let (dg, du, ddn, dsc) = ddense.dense_for(&cp, &p, h, &dev)?;
            let xd: T2 = burn::tensor::Tensor::from_data(
                burn::tensor::TensorData::new(hn.clone(), [n, h]), &dev);
            let yd = mary::models::inkling::burn::dense_mlp(
                xd, dg.clone(), du.clone(), ddn.clone(), *dsc);
            yd.into_data().convert::<f32>().to_vec::<f32>().expect("dense mlp to host")
        } else {
            let inter = t.intermediate_size;
            let rw = g("mlp.gate.weight")?;
            let rb = g("mlp.gate.bias")?;
            let rg = g("mlp.gate.global_scale")?;
            let routing: Vec<Routing> = route(
                &hn, &rw.data, &rb.data, rg.data[0], t.route_scale as f32,
                n, h, t.n_routed_experts, t.n_shared_experts, t.num_experts_per_tok,
            );

            // Group tokens by expert, so each slab is decoded once.
            let mut by_expert: BTreeMap<usize, Vec<(usize, f32)>> = BTreeMap::new();
            for (ti, r) in routing.iter().enumerate() {
                for (slot, &e) in r.experts.iter().enumerate() {
                    by_expert.entry(e).or_default().push((ti, r.weights[slot]));
                }
            }

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
            // Which one is decided by the checkpoint, not by a flag. What used
            // to sit here was `expert_weight_bf16`, which widened the stored
            // BF16 into f32 on the device and multiplied that -- 604 MB of f32
            // per token for the six experts a token routes to, to hold weights
            // the checkpoint stores in 151 MB -- and then, once that was
            // deleted, an `unimplemented!` that made every 42-layer run
            // impossible. Neither is a lane; this is.
            let acc = {
                let a = if cp.is_nvfp4(&format!("{p}mlp.experts.w13_weight")) {
                    routed_experts_fp4(
                        &cp,
                        fp4_aliases.as_ref(),
                        &fp4_client,
                        &p, &by_expert, &hn, n, h, inter, &mut host_t,
                    )?
                } else {
                    routed_experts_bf16(
                        &cp,
                        fp4_aliases.as_ref(),
                        &fp4_client,
                        &p, &by_expert, &hn, n, h, inter, &mut host_t,
                    )?
                };
                expert_loads += by_expert.len();
                a
            };
            // One number, not two. The device calls are queued, so a
            // decode/arithmetic split would time enqueueing rather than work;
            // `routed_experts_fp4` syncs before returning, so this total is real.
            t_expert += t_d.elapsed().as_secs_f64();

            let ns = t.n_shared_experts;
            let gammas: Vec<f32> = routing.iter().flat_map(|r| r.shared_gammas.clone()).collect();
            let t_s = Instant::now();
            // Device-resident, uploaded once. `split_shared_w13` is the
            // settled reading — this used to be an open `deinterleave_rows`
            // here and a halved split in the gate, which is the contradiction
            // the INTERLEAVED result closed.
            //
            // The host twin (`host_shared`) is gone, and with it the once-per-
            // run device-vs-host comparison it fed. That comparison reported
            // 6.6e-5 relative, every run, and what it actually established was
            // that two f32 transcriptions of the same sums reassociate — which
            // is a fact about f32, not a check on this lane. The lane's real
            // oracle is `inkling_real_gate`, against a bundle Python wrote.
            let sh = {
                let sw = ddense.shared_for(&cp, &p, ns, inter, h, shared_halved, &dev)?;
                let xd: T2 = burn::tensor::Tensor::from_data(
                    burn::tensor::TensorData::new(hn.clone(), [n, h]), &dev);
                let y = mary::models::inkling::burn::shared_experts_dev(
                    xd, &sw.gate, &sw.up, &sw.down, &gammas, ns);
                y.into_data().convert::<f32>().to_vec::<f32>().expect("shared to host")
            };
            t_shared += t_s.elapsed().as_secs_f64();
            acc.iter().zip(&sh).map(|(a, b)| a + b).collect()
        };

        // The MLP half's own short convolution carries state across generated
        // tokens exactly as attention's do.
        let mlp_sconv_w = g("mlp_sconv.weight")?.data.clone();
        let y = if kv {
            if step > 0 {
                short_conv_step(
                    &mut caches[slot].mlp_sconv,
                    &y,
                    &mlp_sconv_w,
                    h,
                    t.sconv_kernel_size,
                )
            } else {
                caches[slot].mlp_sconv = conv_history(&y, n, h, t.sconv_kernel_size);
                short_conv(&y, &mlp_sconv_w, n, h, t.sconv_kernel_size)
            }
        } else {
            short_conv(&y, &mlp_sconv_w, n, h, t.sconv_kernel_size)
        };
        for (xi, yi) in x.iter_mut().zip(&y) {
            *xi += yi;
        }

        if let Ok(dir) = std::env::var("INK_DUMP_DIR") {
            let mut bytes = Vec::with_capacity(x.len() * 4);
            for v in &x {
                bytes.extend_from_slice(&v.to_le_bytes());
            }
            std::fs::write(format!("{dir}/h_after_{layer:02}.bin"), &bytes)?;
        }
        t_other += t_o.elapsed().as_secs_f64();
        let norm: f32 = (x.iter().map(|v| (v * v) as f64).sum::<f64>() / x.len() as f64).sqrt() as f32;
        // Milliseconds, not tenths. A decode pass through an NVFP4 layer is
        // ~50 ms and through layer 2's BF16 experts ~600 ms, and at one decimal
        // place the first of those prints as 0.0s -- which makes the one
        // comparison this line exists for, layer 2 against its neighbours,
        // unanswerable from the log.
        println!("  layer {layer:2} [{}] {:.3}s  rms {norm:.4}",
                 if is_local { "local " } else { "global" }, l0.elapsed().as_secs_f32());
    }

    // ---- head, or the wire in its place ------------------------------------
    let v = t.effective_vocab();
    let t_h = Instant::now();
    // A head has no logits and never will: the rest of the stack and the
    // unembedding both live on the other machine. So it hands the stream over
    // and takes the argmax back, and that blocking call is charged to the same
    // slot the head/unembed occupies on a whole-stack run — which is what makes
    // the two reports read against each other line for line.
    let mut best_wire = None;
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
        let hs = dev_lane::rms_norm(
            up2::<Bk>(x.clone(), n, h, &dev),
            up1r(&fnorm.as_ref().expect("the head owns the final norm").data, h, &dev),
            t.rms_norm_eps,
        )
        .div_scalar(t.logits_mup_width_multiplier as f32);
        down(dev_lane::linear(
            hs,
            unembed_dev.clone().expect("the tail uploads the unembed table"),
        ))
    };
    let t_head = t_h.elapsed().as_secs_f64();

    // Greedy: the last position's argmax is the next token. A head took it off
    // the wire instead of computing it, and either way it is decided HERE --
    // before the reporting -- so a tail can answer its peer immediately rather
    // than making the head wait on a page of printing.
    let best = match best_wire {
        Some(b) => b,
        None => {
            let last = &logits[(n - 1) * v..n * v];
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
                let ud = unembed_dev.as_ref().expect("drafting needs the unembed table");
                let hs = dev_lane::rms_norm(
                    up2::<Bk>(row.to_vec(), 1, h, &dev),
                    up1r(&fnorm_d.data, h, &dev),
                    t.rms_norm_eps,
                )
                .div_scalar(t.logits_mup_width_multiplier as f32);
                down(dev_lane::linear(hs, ud.clone()))
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
                    embeds[j * h..(j + 1) * h].copy_from_slice(&e_w.data[tok * h..(tok + 1) * h]);
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
                            .copy_from_slice(&e_w.data[tok * h..(tok + 1) * h]);
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
                        &e_w.data[best * h..(best + 1) * h],
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
                            &e_w.data[drafts[i] * h..(drafts[i] + 1) * h],
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
    println!("  where the time went, seconds:");
    println!("    attention half      {t_attn:8.3}   (device)");
    {
        println!("      read + widen      {t_attn_read:8.3}   (host: slice the mapping, BF16 -> f32)");
        println!("      upload            {t_attn_up:8.3}   (host -> device, synced)");
        println!("      device            {:8.3}   (projections, scores, sconv)",
                 t_attn - t_attn_read - t_attn_up);
    }
    println!("    mlp half            {t_other:8.3}   of which:");
    println!("      routed experts    {t_expert:8.3}   (slice + bind + native mma, device)");
    println!("      shared experts    {t_shared:8.3}   (device)");
    println!("      rest of the half  {:8.3}   (routing, dense layers, sconv, norms)",
             t_other - t_expert - t_shared);
    println!(
        "    {:19} {t_head:8.3}   ({})",
        if best_wire.is_some() { "tail + wire" } else { "head / unembed" },
        if best_wire.is_some() {
            "BLOCKING: the other machine's layers, its head, and the round trip"
        } else {
            "device"
        }
    );
    println!("    of the above, host-only tensor reads (mmap + BF16 widening): {:8.3}", t_read.get());
    {
        println!("    of the routed-expert total, the host-synchronous parts:");
        println!("      slice from mmap   {:8.3}   ({:.3} ms x {expert_loads} loads)",
                 host_t.0, host_t.0 * 1e3 / expert_loads.max(1) as f64);
        // NOT \"upload\": this bucket is everything after the slice -- binding
        // the weight (an offset when it aliases, a copy when it does not),
        // quantising the activation, four kernel enqueues, and the layer\u0027s one
        // blocking read. Calling it an upload sent a profiling session looking
        // for a transfer that was 4% of it.
        println!("      bind+enqueue+sync {:8.3}   ({:.3} ms x {expert_loads} loads)",
                 host_t.1, host_t.1 * 1e3 / expert_loads.max(1) as f64);
        println!("      remainder         {:8.3}   (whatever the two buckets above did not cover)",
                 t_expert - host_t.0 - host_t.1);
    }
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
        dattn.len()
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
    if !dattn.is_empty() {
        println!(
            "    device-resident     {:8.2} GiB in {} attention layers",
            dattn_bytes as f64 / GIB,
            dattn.len()
        );
    }
    if std::env::var("INK_IOSTATS").is_ok() {
        print!("{}", cp.io_table(28));
    }
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
        for ti in 0..n {
            let pos = pos0 + ti;
            let row = &logits[ti * v..(ti + 1) * v];
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
        println!("  step {step}: +{best}   [pass {:.3}s, total {:.1}s, ctx {}]",
                 pass.elapsed().as_secs_f32(), started.elapsed().as_secs_f32(), ids.len());
        // The tail already pushed, when it answered its peer.
        if !is_tail && !repeat {
            ids.push(best);
        }
    }
    if step == gen_steps {
        break;
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
                println!(
                    "  depth {}: {}/{} = {:.0}%",
                    d + 1,
                    mtp_hits[d],
                    mtp_seen[d],
                    100.0 * mtp_hits[d] as f64 / mtp_seen[d] as f64
                );
            }
            let hits: usize = mtp_hits.iter().sum();
            println!("  overall: {hits}/{scored} = {:.0}%", 100.0 * hits as f64 / scored as f64);
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
