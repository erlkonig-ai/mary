//! A real forward pass of Inkling-Small on one machine, paging experts.
//!
//! Every gate so far ends with the same disclaimer: the checkpoint-name to
//! module mapping is authored on both sides, so a shared misreading would pass.
//! This is the check that can settle it. Coherent continuations cannot come out
//! of a wrong mapping — a transposed projection or a swapped gate/up half
//! produces noise, not English.
//!
//! It fits on one machine because the router is small. The layer's experts are
//! 26 GB at f32 and a short prompt touches a few dozen of 256, but which ones
//! is not known until attention has run — so each layer runs attention, routes,
//! and only then reads the selected expert slabs out of the mapping, applying
//! and dropping each before the next. Peak residency is one layer, not one
//! model.
//!
//! Each distinct expert is decoded ONCE and applied to every token that chose
//! it; decoding per (token, expert) pair would repeat most of the work.
//!
//! # Where the weights come from
//!
//! `<ckpt>` is a safetensors checkpoint directory. `INK_PILE=<path>` swaps the
//! WEIGHT source for a pile on branch `INK_PILE_BRANCH` (default `inkling`) —
//! the directory is then read only for `config.json`, which is not a weight and
//! does not live in the pile. One environment variable is the whole A/B, which
//! is the point: everything below this line is the same code either way.
//!
//!   cargo run --release --features inkling-cuda,cuda-backend --bin inkling_forward \
//!       -- <ckpt> <ids.bin> <out.bin>
//!
//! # Two machines, and why it is a MODE here rather than its own binary
//!
//! One machine cannot hold this model resident: 144 GiB of weights against 121
//! GiB of RAM, so the page cache evicts what the next token needs and every
//! token pays real block-device I/O for its expert slabs. Split by LAYER across
//! two boxes it is ~72 GiB each, which fits with headroom.
//!
//! `INK_LAYERS=LO:HI` runs only that half-open range; `INK_PIPE=head:HOST:PORT`
//! sends the residual stream on when the range ends, and `INK_PIPE=tail:ADDR`
//! receives it, finishes the stack and returns the argmax. Only `[n, 4096]` f32
//! crosses — 16 KB per token per boundary, once — which is why the split is by
//! layer and not within one: splitting a layer needs an all-reduce per layer and
//! 1 GbE cannot carry it.
//!
//! This is a mode and not a second program on purpose. It WAS a second program
//! (`inkling_pipe`), and that program forked: it kept a copy of this file's
//! layer loop from before attention went device-resident, before the fused
//! NVFP4 decode and before the residency cache, so the two lanes computed
//! different things at different speeds and no number from one was comparable
//! to a number from the other. There is one layer loop, so there is nothing
//! left to drift.
//!
//! Byte-balanced at layer 20 rather than the midpoint: layer 2 carries BF16
//! experts (12.7 GiB against 3.55 GiB for an NVFP4 layer), so an even 21/21
//! split is a lopsided 85/71 GiB one. Layers 0..20 are 77.6 GiB and 20..42 are
//! 78.2 GiB.
//!
//!   # tail, on the second box
//!   INK_LAYERS=20:42 INK_PIPE=tail:0.0.0.0:7654 inkling_forward <ckpt> <ids> <out>
//!   # head, on the first
//!   INK_LAYERS=0:20  INK_PIPE=head:<tail-host>:7654 inkling_forward <ckpt> <ids> <out>

use std::collections::BTreeMap;
use std::io::{Read as _, Write as _};
use std::net::{TcpListener, TcpStream};
use std::path::PathBuf;
use std::time::Instant;

use anyhow::{Context, Result};

use mary::models::inkling::attn::{attention, causal_mask, AttnDims, AttnWeights, LogScaling};
use mary::models::inkling::block::{rms_norm, route, short_conv, Routing};
use mary::models::inkling::config::{AttnKind, InklingConfig};
use mary::models::inkling::load::{deinterleave_fused, split_gate_up, Held, Loaded};
use mary::models::inkling::source::Weights;
use mary::models::inkling::mlp::{dense_mlp, shared_experts};
use mary::models::inkling::stack::{embed_and_norm, head};

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

/// `x * sigmoid(x)`.
fn silu(x: f32) -> f32 {
    x / (1.0 + (-x).exp())
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

/// `y = x W^T`, `W` stored `[out, in]`.
///
/// The accumulation is strictly sequential f32, which is the worst-case order
/// for error growth over 4096 terms. `INK_HOST_SUM=reverse` sums the identical
/// products from the far end: same mathematics, different rounding, same lane.
/// That makes the host disagree with itself, which is the only honest way to
/// measure how much of a host-vs-device gap is just f32 reassociation.
fn linear(x: &[f32], w: &[f32], rows: usize, in_dim: usize, out_dim: usize) -> Vec<f32> {
    let rev = std::env::var("INK_HOST_SUM").map(|v| v == "reverse").unwrap_or(false);
    let mut out = vec![0f32; rows * out_dim];
    for r in 0..rows {
        let xr = &x[r * in_dim..(r + 1) * in_dim];
        for o in 0..out_dim {
            let wr = &w[o * in_dim..(o + 1) * in_dim];
            out[r * out_dim + o] = if rev {
                xr.iter().zip(wr).rev().map(|(a, b)| a * b).sum()
            } else {
                xr.iter().zip(wr).map(|(a, b)| a * b).sum()
            };
        }
    }
    out
}

/// Seed a host short convolution's rolling history from a prefill.
///
/// The `kernel - 1` most recent rows, oldest first, left-padded with the same
/// zeros [`short_conv`] assumes for positions before the sequence — so a prompt
/// shorter than the kernel starts correct rather than shifted.
fn conv_history_host(x: &[f32], tokens: usize, dim: usize, kernel: usize) -> Vec<f32> {
    let want = kernel - 1;
    let mut h = vec![0f32; want * dim];
    let take = want.min(tokens);
    h[(want - take) * dim..].copy_from_slice(&x[(tokens - take) * dim..tokens * dim]);
    h
}

/// One position of the host short convolution, advancing `hist` in place.
///
/// `cat(hist, x)` is exactly the window the last row of [`short_conv`] reads,
/// so the tap arithmetic is not restated here — there is one implementation and
/// the cached lane cannot drift from the uncached one.
fn short_conv_step_host(
    hist: &mut Vec<f32>,
    x: &[f32],
    weight: &[f32],
    dim: usize,
    kernel: usize,
) -> Vec<f32> {
    assert_eq!(x.len(), dim, "a decode step convolves exactly one position");
    assert_eq!(hist.len(), (kernel - 1) * dim, "history must be the {} rows before it", kernel - 1);
    let mut win = std::mem::take(hist);
    win.extend_from_slice(x);
    let out = short_conv(&win, weight, kernel, dim, kernel);
    *hist = win[dim..].to_vec();
    out[(kernel - 1) * dim..].to_vec()
}

/// The backend the device lane runs on.
#[cfg(feature = "inkling-cuda")]
type Bk = burn::backend::Cuda<f32>;
#[cfg(feature = "inkling-cuda")]
use burn::prelude::Backend;
#[cfg(feature = "inkling-cuda")]
use burn::tensor::{Tensor as BT, TensorData as BTD};
#[cfg(feature = "inkling-cuda")]
use mary::models::inkling::burn as dev_lane;

/// Move a host `[rows, cols]` matrix to the device, consuming it.
///
/// Takes the `Vec` by value on purpose: the dense `w13` is 537 MB at f32 and a
/// borrowing helper would hold two copies of it at once.
#[cfg(feature = "inkling-cuda")]
fn up2<B: Backend>(v: Vec<f32>, rows: usize, cols: usize, dev: &B::Device) -> BT<B, 2> {
    assert_eq!(v.len(), rows * cols, "{} values are not [{rows}, {cols}]", v.len());
    BT::from_data(BTD::new(v, [rows, cols]), dev)
}

/// The same, from a BORROWED slice — for weights that are held on the host.
///
/// The owning [`up2`] exists so a 537 MB dense weight is moved rather than
/// duplicated. A resident weight cannot be moved (the run keeps it), so this
/// copies; the copy is unavoidable and is stated rather than hidden.
#[cfg(feature = "inkling-cuda")]
fn up2r<B: Backend>(v: &[f32], rows: usize, cols: usize, dev: &B::Device) -> BT<B, 2> {
    assert_eq!(v.len(), rows * cols, "{} values are not [{rows}, {cols}]", v.len());
    BT::from_data(BTD::new(v.to_vec(), [rows, cols]), dev)
}

#[cfg(feature = "inkling-cuda")]
fn up1r<B: Backend>(v: &[f32], len: usize, dev: &B::Device) -> BT<B, 1> {
    assert_eq!(v.len(), len, "{} values are not [{len}]", v.len());
    BT::from_data(BTD::new(v.to_vec(), [len]), dev)
}

#[cfg(feature = "inkling-cuda")]
fn up1<B: Backend>(v: Vec<f32>, len: usize, dev: &B::Device) -> BT<B, 1> {
    assert_eq!(v.len(), len, "{} values are not [{len}]", v.len());
    BT::from_data(BTD::new(v, [len]), dev)
}

/// Read a `[rows, cols]` device tensor back to the host. This is also the sync,
/// so a timer around the call measures work rather than enqueueing.
#[cfg(feature = "inkling-cuda")]
fn down<B: Backend>(t: BT<B, 2>) -> Vec<f32> {
    t.into_data().convert::<f32>().to_vec::<f32>().expect("device readback")
}

/// A device tensor of this run's backend, named once so the residency types
/// below do not have to repeat it.
#[cfg(feature = "inkling-cuda")]
type T2 = burn::tensor::Tensor<Bk, 2>;

/// One layer's shared experts, on the device.
#[cfg(feature = "inkling-cuda")]
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
#[cfg(feature = "inkling-cuda")]
#[derive(Default)]
struct DeviceDense {
    shared: std::collections::BTreeMap<String, SharedOnDevice>,
    dense: std::collections::BTreeMap<String, (T2, T2, T2, f32)>,
    bytes: u64,
}

#[cfg(feature = "inkling-cuda")]
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

/// Every routed expert for one layer, on the device, without a host f32 copy.
///
/// The host never sees a dequantised weight: the checkpoint's own bytes are
/// BORROWED out of the mapping, uploaded as-is, and the E2M1/E4M3 decode
/// happens on the device. That removed the 60.2s of a measured 113.7s forward
/// that was scalar unpacking, and the 67 MB per-expert host allocation with it.
///
/// The decode is one fused kernel per weight
/// ([`mary::models::inkling::dequant_cuda`]), with the gate/up de-interleave
/// folded into where it writes. `INK_DEQUANT=chain` selects the Burn tensor
/// chain instead — 46 launches a weight and 1.4 GB of device traffic to make
/// 100 MB of f32 — because the fused kernel is gated bitwise AGAINST it and a
/// reference you cannot still run is not a reference.
///
/// All of an expert's tokens go through in one matmul rather than one each.
/// That reassociates the sums, so this is NOT bitwise identical to the host
/// lane and must be gated on a tolerance, not on equality.
///
/// Syncs before returning, so the caller's timer measures work and not
/// enqueueing.
#[cfg(feature = "inkling-cuda")]
#[allow(clippy::too_many_arguments)]
fn routed_experts_gpu(
    cp: &Weights,
    prefix: &str,
    by_expert: &BTreeMap<usize, Vec<(usize, f32)>>,
    hn: &[f32],
    n: usize,
    h: usize,
    inter: usize,
    dev: &burn::backend::cuda::CudaDevice,
    chain_decode: bool,
    host: &mut (f64, f64),
) -> Result<Vec<f32>> {
    type B = Bk;
    use burn::tensor::{Int, Tensor, TensorData};
    use mary::models::inkling::burn::{
        deinterleave_rows_device, expert_ffn, expert_weight_from_packed,
    };
    use mary::models::inkling::dequant_cuda::expert_weight_fused;

    let hn_dev: Tensor<B, 2> =
        Tensor::from_data(TensorData::new(hn.to_vec(), [n, h]), dev);
    let mut acc: Tensor<B, 2> = Tensor::zeros([n, h], dev);

    for (&e, toks) in by_expert {
        let n13 = format!("{prefix}mlp.experts.w13_weight");
        let n2 = format!("{prefix}mlp.experts.w2_weight");

        // 39 of 40 MoE layers are packed NVFP4; layer 2 is BF16. The packed
        // branch never makes a host f32 copy. The BF16 branch has to widen
        // somewhere, so it widens on the host — one layer in forty, stated
        // rather than hidden.
        let deint = std::env::var("INK_MUTATE_NO_DEINTERLEAVE").is_err();
        let t_s = Instant::now();
        let gu_dn = if cp.is_nvfp4(&n13) {
            let w13 = cp.expert_packed(&n13, e)?;
            let w2 = cp.expert_packed(&n2, e)?;
            host.0 += t_s.elapsed().as_secs_f64();
            let t_w = Instant::now();
            let r = if chain_decode {
                let gu = expert_weight_from_packed::<B>(
                    w13.codes(), w13.scales(), w13.scale2(), w13.rows(), w13.cols(), dev,
                );
                (
                    if deint { deinterleave_rows_device(gu) } else { gu },
                    expert_weight_from_packed::<B>(
                        w2.codes(), w2.scales(), w2.scale2(), w2.rows(), w2.cols(), dev,
                    ),
                )
            } else {
                (
                    expert_weight_fused(
                        w13.codes(), w13.scales(), w13.scale2(), w13.rows(), w13.cols(), deint, dev,
                    ),
                    expert_weight_fused(
                        w2.codes(), w2.scales(), w2.scale2(), w2.rows(), w2.cols(), false, dev,
                    ),
                )
            };
            host.1 += t_w.elapsed().as_secs_f64();
            r
        } else {
            let a = cp.expert_f32(&n13, e)?;
            let b = cp.expert_f32(&n2, e)?;
            host.0 += t_s.elapsed().as_secs_f64();
            let t_w = Instant::now();
            let fused = Tensor::<B, 2>::from_data(TensorData::new(a.data, [2 * inter, h]), dev);
            let r = (
                if deint { deinterleave_rows_device(fused) } else { fused },
                Tensor::<B, 2>::from_data(TensorData::new(b.data, [h, inter]), dev),
            );
            host.1 += t_w.elapsed().as_secs_f64();
            r
        };
        // INK_MUTATE_NO_DEINTERLEAVE exists so the lane gate can be watched
        // rejecting something, because a gate that has never failed and a gate
        // that cannot fail look identical from outside.
        let (gu, dn) = gu_dn;
        anyhow::ensure!(gu.dims() == [2 * inter, h], "w13 is {:?}, want [{}, {h}]", gu.dims(), 2 * inter);
        anyhow::ensure!(dn.dims() == [h, inter], "w2 is {:?}, want [{h}, {inter}]", dn.dims());

        let rows: Vec<i32> = toks.iter().map(|&(ti, _)| ti as i32).collect();
        let wts: Vec<f32> = toks.iter().map(|&(_, w)| w).collect();
        let k = rows.len();
        let idx: Tensor<B, 1, Int> =
            Tensor::from_data(TensorData::new(rows, [k]), dev);
        let wt: Tensor<B, 2> =
            Tensor::from_data(TensorData::new(wts, [k, 1]), dev);

        let xs = hn_dev.clone().select(0, idx.clone());
        let ys = expert_ffn(xs, gu, dn) * wt;
        acc = acc.select_assign(0, idx, ys, burn::tensor::IndexingUpdateOp::Add);
    }

    // Reading back is the sync. It is also the only host copy in the routine:
    // [n, h] f32, 131 KB at these dimensions, against the 67 MB per expert the
    // host lane allocates.
    Ok(acc.into_data().convert::<f32>().to_vec::<f32>().expect("acc to host"))
}


/// Every routed expert for one layer, on the NATIVE NVFP4 tensor-core path.
///
/// Differs from [`routed_experts_gpu`] in that the packed bytes go straight
/// into `mma.sync…kind::mxf4nvf4…ue4m3` instead of being decoded into a
/// 67.1 + 33.6 MB f32 pair per expert that is read once and dropped.
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
#[cfg(feature = "inkling-cuda")]
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
        let y = f32::from_bytes(&client.read_one(y_h).expect("read y")).to_vec();
        host.1 += t_w.elapsed().as_secs_f64();

        for (i, &(ti, wgt)) in toks.iter().enumerate() {
            for o in 0..h {
                acc[ti * h + o] += y[i * h + o] * wgt;
            }
        }
    }
    Ok(acc)
}

fn main() -> Result<()> {
    let ckpt = std::env::args().nth(1).map(PathBuf::from).context("usage: <ckpt> <ids> <out>")?;
    let ids_path = std::env::args().nth(2).map(PathBuf::from).context("usage: <ckpt> <ids> <out>")?;
    let out_path = std::env::args().nth(3).map(PathBuf::from).context("usage: <ckpt> <ids> <out>")?;

    let cfg_text = std::fs::read_to_string(ckpt.join("config.json"))?;
    let cfg = InklingConfig::from_json(&cfg_text).context("parsing config.json")?;
    let t = &cfg.text_config;
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

    // Which layers THIS process runs. The default is the whole stack, so a
    // single-machine run is the `INK_LAYERS` unset case and not a special one.
    let (lo, hi) = match std::env::var("INK_LAYERS") {
        Ok(s) => {
            let (a, b) = s.split_once(':').context("INK_LAYERS wants LO:HI")?;
            (a.parse::<usize>()?, b.parse::<usize>()?)
        }
        Err(_) => (0, t.num_hidden_layers),
    };
    anyhow::ensure!(lo < hi, "INK_LAYERS wants LO < HI, got {lo}:{hi}");
    anyhow::ensure!(
        hi <= t.num_hidden_layers,
        "INK_LAYERS {lo}:{hi} runs past the {}-layer stack",
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
    println!("  config     : {}", ckpt.display());
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

    // ---- does this node's share FIT? ---------------------------------------
    //
    // The decode loop cannot answer this on its own, and it is worth saying why,
    // because its per-token `read_bytes` looks like the answer. A token routes
    // to ~5 of 256 experts per layer, so nine tokens touch about a tenth of a
    // node's share: most of what they read off disk is a FIRST touch, not a page
    // the kernel evicted. Compulsory misses and capacity misses are different
    // claims about different things, and only the second one is what a layer
    // split is for.
    //
    // So ask directly. Touch the whole share, twice, and report each pass's
    // block-device traffic: a share that fits reads it once and then reads ~zero,
    // and one that does not reads back exactly what had to be evicted to make
    // room for the tail of the first pass. `INK_WARM=1` also leaves the share hot
    // for a decode measurement that follows, which is the other reason to want it.
    // `INK_WARM=N` runs N passes, and more than two is often what it takes to
    // get an answer: the cache a node starts with holds pages from whatever ran
    // before -- on these boxes, the OTHER half of the very same pile, copied in
    // whole -- and those age out only by being outlived. A share that fits shows
    // its disk column falling to zero and staying; one that does not shows a
    // column that will not fall however long you look at it.
    let warm_passes: usize = std::env::var("INK_WARM")
        .ok()
        .map(|v| if v == "1" { 2 } else { v.parse().unwrap_or(0) })
        .unwrap_or(0);
    if warm_passes > 0 {
        println!("  warming layers {lo}..{hi} -- {warm_passes} passes, so eviction is visible:");
        for pass in 1..=warm_passes {
            let t0 = Instant::now();
            let before = io_read_bytes();
            let (bytes, leaves) = cp.warm(lo as i64..=hi as i64 - 1)?;
            let disk = io_read_bytes() - before;
            println!(
                "    pass {pass}: touched {:6.2} GiB in {leaves} leaves in {:5.1}s  \
                 -> {:6.2} GiB off disk ({:.0} MB/s)",
                bytes as f64 / GIB,
                t0.elapsed().as_secs_f32(),
                disk as f64 / GIB,
                disk as f64 / 1e6 / t0.elapsed().as_secs_f64().max(1e-9)
            );
        }
    }

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
    // `Option`, so the head lane can DROP the 3.3 GB host copy after uploading
    // it. `held` only keeps it alive in the resident map when INK_RESIDENT is
    // set, so dropping the handle frees it in the non-resident case — which is
    // what the lane that replaced `std::mem::take` here has to preserve.
    #[allow(unused_mut)]
    let mut unembed = if want_head { Some(cp.held("model.llm.unembed.weight")?) } else { None };
    println!("  embedding tables loaded in {:.1}s", started.elapsed().as_secs_f32());
    println!(
        "  dense weights      : {}",
        if cp.resident_on() {
            "RESIDENT -- read and widened once, then held for the whole run"
        } else {
            "streamed -- re-read and re-widened from the checkpoint every token"
        }
    );

    // Experts are read, applied and dropped. There is no decoded-expert cache:
    // it measured as no speedup, and it existed to paper over a capacity
    // shortfall (160 GB of checkpoint against 119 GB of box) that a second
    // Spark closes. See this file's header.
    // One switch per lane, so a regression can be bisected to the lane that
    // caused it, plus `INK_GPU=all` for the everything-on-device run.
    let all_gpu = std::env::var("INK_GPU").map(|v| v == "all").unwrap_or(false);
    let lane = |k: &str| all_gpu || std::env::var(k).map(|v| v == "gpu").unwrap_or(false);
    // INK_EXPERTS is three-valued, so both branches' spellings keep working:
    //   fp4 -> native NVFP4 tensor cores (spark1's lane)
    //   gpu -> decode to f32, then an f32 matmul (spark2's lane, and what
    //          INK_GPU=all alone selects, exactly as it did before the merge)
    let ink_experts = std::env::var("INK_EXPERTS").unwrap_or_default();
    let experts_fp4 = ink_experts == "fp4";
    let experts_on_gpu = all_gpu || ink_experts == "gpu" || experts_fp4;
    let attn_on_gpu = lane("INK_ATTN");
    // `INK_RESIDENT` reads the same on both lanes -- hold a weight instead of
    // re-reading it every token -- and differs only in WHERE, because a device
    // lane never reads the host copy again. Off, attention streams as before.
    let attn_resident = cp.resident_on();
    // Two names for the shared+dense MLP lane, and they are NOT the same
    // implementation: INK_DENSE=gpu uploads those weights ONCE and keeps them
    // (spark1's device residency), INK_MLP=gpu / INK_GPU=all re-uploads per
    // layer per token but de-interleaves on the device (spark2's). Resident
    // wins where both are asked for; the banner says which one ran.
    let dense_gpu = std::env::var("INK_DENSE").map(|v| v == "gpu").unwrap_or(false);
    let mlp_on_gpu = lane("INK_MLP") || dense_gpu;
    let head_on_gpu = lane("INK_HEAD");
    // The KV cache. Off by default so the uncached lane stays available as the
    // thing to check against: the decisive test of a cache is that it produces
    // the same token sequence as not having one, and that needs both to run.
    let kv = std::env::var("INK_KV").map(|v| v == "1" || v == "on").unwrap_or(false);
    // The NVFP4 decode: one fused kernel per weight, or the Burn tensor chain
    // it is gated against. Default fused; `INK_DEQUANT=chain` is the control.
    let chain_decode = std::env::var("INK_DEQUANT").map(|v| v == "chain").unwrap_or(false);
    anyhow::ensure!(
        !kv || attn_on_gpu,
        "INK_KV needs attention on the device -- set INK_ATTN=gpu or INK_GPU=all"
    );
    let say = |b: bool| if b { "device" } else { "host (f32 oracle)" };
    println!("  attention          : {}", say(attn_on_gpu));
    println!(
        "  shared + dense MLP : {}",
        if dense_gpu {
            "DEVICE-RESIDENT -- uploaded once, matmul on the GPU"
        } else if mlp_on_gpu {
            "device -- uploaded per layer per token"
        } else {
            "host f32 (the reference lane)"
        }
    );
    println!(
        "  routed experts     : {}",
        if experts_fp4 {
            "device, NATIVE NVFP4 tensor cores"
        } else if experts_on_gpu {
            "device, decode to f32 then f32 matmul"
        } else {
            "host (f32 oracle)"
        }
    );
    println!("  head (unembed)     : {}", say(head_on_gpu));
    println!("  kv cache           : {}", if kv { "on" } else { "off (prefix recomputed each step)" });
    println!("  nvfp4 decode       : {}", if chain_decode { "Burn tensor chain (46 launches/weight)" } else { "fused kernel (1 launch/weight)" });
    if std::env::var("INK_HOST_SUM").map(|v| v == "reverse").unwrap_or(false) {
        println!("  host sum order     : REVERSED (reassociation control)");
    }
    if std::env::var("INK_MUTATE_NO_DEINTERLEAVE").is_ok() {
        println!("  !! MUTATION ACTIVE : deinterleave SKIPPED -- this output is expected to be WRONG");
    }
    // The SHARED experts' w13 is square, so nothing but a forward can tell the
    // two readings apart. INK_SHARED_W13_HALVED=1 selects the other one.
    let shared_halved = mary::models::inkling::load::shared_w13_halved();
    // `split_shared_fused` (the per-token device lane) hard-codes the
    // INTERLEAVED reading -- which is the settled one, so the default is right,
    // but it cannot express the control. Refuse rather than run the mutation on
    // three lanes and silently not on the fourth: a control that is quietly
    // ignored on one lane reads as "the mutation made no difference there".
    anyhow::ensure!(
        !(shared_halved && lane("INK_MLP") && !std::env::var("INK_DENSE").map(|v| v == "gpu").unwrap_or(false)),
        "INK_SHARED_W13_HALVED=1 does not reach the INK_MLP=gpu shared-expert lane; \
         use INK_DENSE=gpu (device-resident, honours it) or the host lane"
    );
    println!(
        "  attention weights  : {}",
        if attn_on_gpu && attn_resident {
            "DEVICE-RESIDENT -- read, widened and uploaded once, then held"
        } else if attn_on_gpu {
            "streamed -- re-read, re-widened and re-uploaded every token"
        } else {
            "host"
        }
    );
    println!(
        "  shared w13 split   : {}",
        if shared_halved { "HALVED (contiguous)" } else { "INTERLEAVED" }
    );
    #[cfg(feature = "inkling-cuda")]
    let dev = burn::backend::cuda::CudaDevice::default();
    #[cfg(feature = "inkling-cuda")]
    let mut ddense = if dense_gpu { Some(DeviceDense::default()) } else { None };
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
    // lane the answer is the device.
    //
    // Keyed by the layer prefix, exactly as [`DeviceDense`] is, and populated
    // on first use rather than eagerly: a lane that is never taken should not
    // pay for the upload.
    #[cfg(feature = "inkling-cuda")]
    let mut dattn: std::collections::BTreeMap<String, (dev_lane::AttnWeightsDev<Bk>, T2)> =
        std::collections::BTreeMap::new();
    #[cfg(feature = "inkling-cuda")]
    let mut dattn_bytes = 0u64;
    // Checked by running it, on the first MoE layer of the first pass: the two
    // lanes are two transcriptions of the same arithmetic and a gate that is
    // never run is not a gate.
    let mut dense_checked = false;
    // Parsed once for the whole run. The lane it replaces re-parsed a shard
    // header four times per expert slab, ~9950 times over a forward.
    #[cfg(feature = "inkling-cuda")]
    let fp4_client = if experts_fp4 {
        use cubecl::prelude::Runtime;
        Some(cubecl::cuda::CudaRuntime::client(&Default::default()))
    } else {
        None
    };
    // Nine blocking device round trips for the whole run, instead of four per
    // expert. Every later slab is an offset view of one of these.
    let zerocopy_on = std::env::var("INK_ZEROCOPY").map(|v| v != "0").unwrap_or(true);
    #[cfg(feature = "inkling-cuda")]
    let fp4_aliases = match &fp4_client {
        // INK_ZEROCOPY=0 forces the copying lane, so the seam can be A/B'd
        // against it with the page cache in the same state.
        Some(c) if zerocopy_on => {
            let t = Instant::now();
            let maps = cp.mappings()?;
            let n = maps.len();
            let a = mary::models::inkling::fp4gemm::Aliases::register(c, maps);
            println!(
                "  zero-copy mappings : {} {n} in {:.1} ms",
                if a.is_some() { "registered" } else { "UNSUPPORTED, copying" },
                t.elapsed().as_secs_f64() * 1e3
            );
            a
        }
        _ => None,
    };
    #[cfg(not(feature = "inkling-cuda"))]
    anyhow::ensure!(
        !(experts_on_gpu || attn_on_gpu || mlp_on_gpu || head_on_gpu),
        "a device lane needs --features inkling-cuda"
    );

    // The unembed table is 3.3 GB at f32 and does not change between generated
    // tokens, so it is uploaded ONCE here rather than once per step, and the
    // host copy is dropped rather than kept alongside it.
    #[cfg(feature = "inkling-cuda")]
    let unembed_dev = if head_on_gpu && want_head {
        let v = t.effective_vocab();
        let d = up2r::<Bk>(&unembed.as_ref().expect("unembed held").data, t.vocab_size, h, &dev)
            .slice([0..v, 0..h]);
        // The host copy has no reader left. Dropping the handle frees it when
        // residency is off; with INK_RESIDENT the resident map still holds it,
        // which is that flag's declared cost, not a leak.
        unembed = None;
        println!("  unembed uploaded, {v} x {h}");
        Some(d)
    } else {
        None
    };

    // Everything one layer carries between generated tokens. The attention
    // cache is the headline, but the two layer-level short convolutions have
    // state too: they reach `kernel - 1` positions back, and a cache that
    // remembers K and V while forgetting those is wrong in a way that still
    // produces fluent-looking text.
    #[cfg(feature = "inkling-cuda")]
    struct LayerCache {
        attn: dev_lane::AttnCache<Bk>,
        attn_sconv: BT<Bk, 2>,
        mlp_sconv: Vec<f32>,
    }
    #[cfg(feature = "inkling-cuda")]
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
        #[cfg_attr(not(feature = "inkling-cuda"), allow(unused_variables))]
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
        let a = if attn_on_gpu {
            #[cfg(feature = "inkling-cuda")]
            {
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
                    if attn_resident {
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
                // Residency off means hold NOTHING, not hold everything: the
                // streaming lane is what a box too small for the working set
                // runs, and a map that grew to all 42 layers regardless would
                // have quietly removed that option.
                if !attn_resident {
                    dattn.remove(&p);
                }
                out
            }
            #[cfg(not(feature = "inkling-cuda"))]
            unreachable!("guarded at startup")
        } else {
            let wq = g("attn.wq_du.weight")?;
            let wk = g("attn.wk_dv.weight")?;
            let wv = g("attn.wv_dv.weight")?;
            let wr = g("attn.wr_du.weight")?;
            let wo = g("attn.wo_ud.weight")?;
            let qn = g("attn.q_norm.weight")?;
            let kn = g("attn.k_norm.weight")?;
            let ks = g("attn.k_sconv.weight")?;
            let vs = g("attn.v_sconv.weight")?;
            let rp = g("attn.rel_logits_proj.proj")?;
            let aw = AttnWeights {
                wq: &wq.data, wk: &wk.data, wv: &wv.data, wr: &wr.data, wo: &wo.data,
                k_sconv: &ks.data, v_sconv: &vs.data,
                q_norm: &qn.data, k_norm: &kn.data, rel_proj: &rp.data,
            };
            let a = attention(&hn, &aw, &dims, Some(ls), mask, n);
            drop((wq, wk, wv, wr, wo));
            short_conv(&a, &g("attn_sconv.weight")?.data, n, h, t.sconv_kernel_size)
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
            let di = t.dense_intermediate_size;
            // The host lane as a closure, so a device lane is an ALTERNATIVE to
            // it rather than an addition — running both and discarding one would
            // have measured nothing. The SPLIT is what residency holds, not the
            // fused tensor: caching the fused form would still de-interleave
            // 3.8 GiB of f32 per token and would pin 7.5 GiB to do it.
            let host_dense = |hn: &[f32]| -> Result<Vec<f32>> {
                let wkey = format!("{p}mlp.w13_dn.weight");
                let (gate, up) = cp.derived_pair(&wkey, || {
                    let fused = cp.tensor(&wkey)?;
                    let (a, b) = split_gate_up(&fused.data, h);
                    let shp = vec![a.len() / h, h];
                    Ok((Loaded { data: a, shape: shp.clone() }, Loaded { data: b, shape: shp }))
                })?;
                let dn = cp.held(&format!("{p}mlp.w2_md.weight"))?;
                let gs = cp.held(&format!("{p}mlp.global_scale"))?;
                Ok(dense_mlp(hn, &gate.data, &up.data, &dn.data, gs.data[0], n, h, di))
            };
            #[cfg(feature = "inkling-cuda")]
            {
                // Device-RESIDENT first (INK_DENSE=gpu): uploaded once for the
                // whole run. Then the per-token device lane (INK_MLP=gpu), which
                // de-interleaves on the device. Then the host reference.
                if let Some(dd) = ddense.as_mut() {
                    let (dg, du, ddn, dsc) = dd.dense_for(&cp, &p, h, &dev)?;
                    let xd: T2 = burn::tensor::Tensor::from_data(
                        burn::tensor::TensorData::new(hn.clone(), [n, h]), &dev);
                    let yd = mary::models::inkling::burn::dense_mlp(
                        xd, dg.clone(), du.clone(), ddn.clone(), *dsc);
                    yd.into_data().convert::<f32>().to_vec::<f32>().expect("dense mlp to host")
                } else if mlp_on_gpu {
                    let gs = gv("mlp.global_scale")?[0];
                    // The gate/up de-interleave happens on device too: the fused
                    // dense weight is 134 M elements, and shuffling it in a
                    // scalar loop costs more than the matmul it feeds.
                    let fused = up2::<Bk>(gv("mlp.w13_dn.weight")?, 2 * di, h, &dev);
                    let gu = dev_lane::deinterleave_rows_device(fused);
                    let gate = gu.clone().slice([0..di, 0..h]);
                    let upw = gu.slice([di..2 * di, 0..h]);
                    down(dev_lane::dense_mlp(
                        up2(hn.clone(), n, h, &dev),
                        gate,
                        upw,
                        up2(gv("mlp.w2_md.weight")?, h, di, &dev),
                        gs,
                    ))
                } else {
                    host_dense(&hn)?
                }
            }
            #[cfg(not(feature = "inkling-cuda"))]
            host_dense(&hn)?
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

            let t_d = Instant::now();
            let acc = if experts_on_gpu {
                #[cfg(feature = "inkling-cuda")]
                {
                    // Layer 2 is BF16 and has no `.scale` sidecar, so the FP4
                    // lane cannot take it; that one layer falls back rather
                    // than the whole run refusing.
                    let packed = cp.is_nvfp4(&format!("{p}mlp.experts.w13_weight"));
                    let a = if experts_fp4 && packed {
                        routed_experts_fp4(
                            &cp,
                            fp4_aliases.as_ref(),
                            fp4_client.as_ref().expect("fp4 client"),
                            &p, &by_expert, &hn, n, h, inter, &mut host_t,
                        )?
                    } else {
                        routed_experts_gpu(
                            &cp, &p, &by_expert, &hn, n, h, inter, &dev, chain_decode, &mut host_t,
                        )?
                    };
                    expert_loads += by_expert.len();
                    a
                }
                #[cfg(not(feature = "inkling-cuda"))]
                unreachable!("guarded at startup")
            } else {
                let mut acc = vec![0f32; n * h];
                for (&e, toks) in &by_expert {
                    let gu_raw = cp.expert_f32(&format!("{p}mlp.experts.w13_weight"), e)?.data;
                    let gu = deinterleave_fused(&gu_raw, 2 * inter, h);
                    let dn = cp.expert_f32(&format!("{p}mlp.experts.w2_weight"), e)?.data;
                    expert_loads += 1;
                    for &(ti, wgt) in toks {
                        let xt = &hn[ti * h..(ti + 1) * h];
                        let both = linear(xt, &gu, 1, h, 2 * inter);
                        let act: Vec<f32> =
                            (0..inter).map(|i| silu(both[i]) * both[inter + i]).collect();
                        let contrib = linear(&act, &dn, 1, inter, h);
                        for (o, c) in acc[ti * h..(ti + 1) * h].iter_mut().zip(&contrib) {
                            *o += c * wgt;
                        }
                    }
                    // Dropped here: one expert resident at a time, not 256.
                }
                acc
            };
            // One number, not two. On the device lane the calls are queued, so a
            // decode/arithmetic split would time enqueueing rather than work;
            // `routed_experts_gpu` syncs before returning, so this total is real.
            t_expert += t_d.elapsed().as_secs_f64();

            let ns = t.n_shared_experts;
            let gammas: Vec<f32> = routing.iter().flat_map(|r| r.shared_gammas.clone()).collect();
            let t_s = Instant::now();
            // The host lane. `split_shared_w13` is the settled reading — this
            // used to be an open `deinterleave_rows` here and a halved split in
            // the gate, which is the contradiction the INTERLEAVED result closed.
            let host_shared = |hn: &[f32]| -> Result<Vec<f32>> {
                let skey = format!("{p}mlp.shared_experts.shared_w13_weight");
                let (sg, su) = cp.derived_pair(&skey, || {
                    let sfused = cp.tensor(&skey)?;
                    let (a, b) = mary::models::inkling::load::split_shared_w13(
                        &sfused.data, ns, inter, h, shared_halved,
                    );
                    let shp = vec![ns, inter, h];
                    Ok((Loaded { data: a, shape: shp.clone() }, Loaded { data: b, shape: shp }))
                })?;
                let sd = cp.held(&format!("{p}mlp.shared_experts.shared_w2_weight"))?;
                Ok(shared_experts(hn, &sg.data, &su.data, &sd.data, &gammas, ns, n, h, inter))
            };
            #[cfg(feature = "inkling-cuda")]
            let sh = if ddense.is_some() {
                let dsh = {
                    let dd = ddense.as_mut().expect("device-resident dense");
                    let sw = dd.shared_for(&cp, &p, ns, inter, h, shared_halved, &dev)?;
                    let xd: T2 = burn::tensor::Tensor::from_data(
                        burn::tensor::TensorData::new(hn.clone(), [n, h]), &dev);
                    let y = mary::models::inkling::burn::shared_experts_dev(
                        xd, &sw.gate, &sw.up, &sw.down, &gammas, ns);
                    y.into_data().convert::<f32>().to_vec::<f32>().expect("shared to host")
                };
                // Run the host lane ONCE, on the first MoE layer of the run, and
                // say what the two lanes differ by. A device lane that has never
                // been compared to its reference and one that cannot be told
                // apart from it look identical from outside.
                if !dense_checked {
                    dense_checked = true;
                    let hsh = host_shared(&hn)?;
                    let scale = hsh.iter().fold(0f32, |a, v| a.max(v.abs()));
                    let worst = hsh.iter().zip(&dsh).fold(0f32, |a, (x, y)| a.max((x - y).abs()));
                    println!(
                        "  shared experts, device vs host lane, layer {layer}: worst abs {worst:e} / scale {scale:e} = {:e}",
                        worst / scale.max(f32::MIN_POSITIVE)
                    );
                }
                dsh
            } else if mlp_on_gpu {
                let sfused = gv("mlp.shared_experts.shared_w13_weight")?;
                let (sg, su) = dev_lane::split_shared_fused(
                    up2::<Bk>(sfused, ns * 2 * inter, h, &dev), ns);
                down(dev_lane::shared_experts(
                    up2(hn.clone(), n, h, &dev),
                    sg,
                    su,
                    up2(gv("mlp.shared_experts.shared_w2_weight")?, ns * h, inter, &dev),
                    up2(gammas.clone(), n, ns, &dev),
                    ns,
                ))
            } else {
                host_shared(&hn)?
            };
            #[cfg(not(feature = "inkling-cuda"))]
            let sh = host_shared(&hn)?;
            t_shared += t_s.elapsed().as_secs_f64();
            acc.iter().zip(&sh).map(|(a, b)| a + b).collect()
        };

        // The MLP half's own short convolution carries state across generated
        // tokens exactly as attention's do.
        let mlp_sconv_w = g("mlp_sconv.weight")?.data.clone();
        let y = if kv {
            #[cfg(feature = "inkling-cuda")]
            {
                if step > 0 {
                    short_conv_step_host(
                        &mut caches[slot].mlp_sconv,
                        &y,
                        &mlp_sconv_w,
                        h,
                        t.sconv_kernel_size,
                    )
                } else {
                    caches[slot].mlp_sconv =
                        conv_history_host(&y, n, h, t.sconv_kernel_size);
                    short_conv(&y, &mlp_sconv_w, n, h, t.sconv_kernel_size)
                }
            }
            #[cfg(not(feature = "inkling-cuda"))]
            unreachable!("guarded at startup")
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
        println!("  layer {layer:2} [{}] {:.1}s  rms {norm:.4}",
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
    } else if head_on_gpu {
        #[cfg(feature = "inkling-cuda")]
        {
            // 109 x 4096 x 200058 is 89 G multiply-adds — the single largest
            // matmul in the forward, and the one left standing once attention
            // and the MLPs move. The muP divisor divides BEFORE the projection,
            // matching the reference: doing it after is algebraically equal and
            // numerically not.
            let hs = dev_lane::rms_norm(
                up2::<Bk>(x.clone(), n, h, &dev),
                up1r(&fnorm.as_ref().expect("the head owns the final norm").data, h, &dev),
                t.rms_norm_eps,
            )
            .div_scalar(t.logits_mup_width_multiplier as f32);
            down(dev_lane::linear(
                hs,
                unembed_dev.clone().expect("uploaded when head_on_gpu"),
            ))
        }
        #[cfg(not(feature = "inkling-cuda"))]
        unreachable!("guarded at startup")
    } else {
        head(
            &x,
            &fnorm.as_ref().expect("the head owns the final norm").data,
            &unembed.as_ref().expect("the host head lane needs the unembed table").data,
            t.logits_mup_width_multiplier as f32,
            t.vocab_size, t.effective_vocab(), t.rms_norm_eps, n, h,
        )
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
    // A tail follows the sequence by RECOMPUTING it, not by being told: it owns
    // the argmax, so pushing it here keeps its `ids` identical to the head's
    // without a second thing on the wire to get out of step.
    if is_tail && gen_steps > 0 {
        ids.push(best);
    }

    println!("\n=== predictions ===");
    println!("  expert slabs decoded: {expert_loads}");
    // t_other covers the whole MLP half, so the expert buckets are inside it.
    println!("  where the time went, seconds:");
    println!("    attention half      {t_attn:8.1}   ({})", if attn_on_gpu { "device" } else { "host" });
    #[cfg(feature = "inkling-cuda")]
    if attn_on_gpu {
        println!("      read + widen      {t_attn_read:8.1}   (host: slice the mapping, BF16 -> f32)");
        println!("      upload            {t_attn_up:8.1}   (host -> device, synced)");
        println!("      device            {:8.1}   (projections, scores, sconv)",
                 t_attn - t_attn_read - t_attn_up);
    }
    println!("    mlp half            {t_other:8.1}   of which:");
    println!("      routed experts    {t_expert:8.1}   ({})",
             if experts_on_gpu { "slice + upload + dequant + matmul, device" }
             else { "disk + NVFP4 unpack + matmul, host" });
    println!("      shared experts    {t_shared:8.1}   ({})", if mlp_on_gpu { "device" } else { "host" });
    println!("      rest of the half  {:8.1}   (routing, dense layers, sconv, norms)",
             t_other - t_expert - t_shared);
    println!(
        "    {:19} {t_head:8.1}   ({})",
        if best_wire.is_some() { "tail + wire" } else { "head / unembed" },
        if best_wire.is_some() {
            "BLOCKING: the other machine's layers, its head, and the round trip"
        } else if head_on_gpu {
            "device"
        } else {
            "host"
        }
    );
    println!("    of the above, host-only tensor reads (mmap + BF16 widening): {:8.1}", t_read.get());
    if experts_on_gpu {
        println!("    of the routed-expert total, the host-synchronous parts:");
        println!("      slice from mmap   {:8.1}   ({:.3} ms x {expert_loads} loads)",
                 host_t.0, host_t.0 * 1e3 / expert_loads.max(1) as f64);
        println!("      upload + enqueue  {:8.1}   ({:.3} ms x {expert_loads} loads)",
                 host_t.1, host_t.1 * 1e3 / expert_loads.max(1) as f64);
        println!("      remainder         {:8.1}   (enqueue + the sync, so device work lives here)",
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
    #[cfg(feature = "inkling-cuda")]
    if let Some(dd) = ddense.as_ref() {
        println!(
            "    device-resident     {:8.2} GiB in {} shared + {} dense layers",
            dd.bytes as f64 / GIB,
            dd.shared.len(),
            dd.dense.len()
        );
    }
    #[cfg(feature = "inkling-cuda")]
    if attn_resident && attn_on_gpu {
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
        println!("  step {step}: +{best}   [pass {:.1}s, total {:.1}s, ctx {}]",
                 pass.elapsed().as_secs_f32(), started.elapsed().as_secs_f32(), ids.len());
        // The tail already pushed, when it answered its peer.
        if !is_tail {
            ids.push(best);
        }
    }
    if step == gen_steps {
        break;
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
