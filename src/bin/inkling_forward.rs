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
//!   cargo run --release --features inkling --bin inkling_forward -- <ckpt> <ids.bin> <out.bin>

use std::collections::BTreeMap;
use std::path::PathBuf;
use std::time::Instant;

use anyhow::{Context, Result};

use mary::models::inkling::attn::{attention, causal_mask, AttnDims, AttnWeights, LogScaling};
use mary::models::inkling::block::{rms_norm, route, short_conv, Routing};
use mary::models::inkling::config::{AttnKind, InklingConfig};
use mary::models::inkling::load::{deinterleave_fused, split_gate_up, Checkpoint};
use mary::models::inkling::mlp::{dense_mlp, shared_experts};
use mary::models::inkling::stack::{embed_and_norm, head};

/// `x * sigmoid(x)`.
fn silu(x: f32) -> f32 {
    x / (1.0 + (-x).exp())
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

/// Every routed expert for one layer, on the device, without a host f32 copy.
///
/// The host never sees a dequantised weight: `expert_slice_packed` hands over
/// the checkpoint's own bytes, they are uploaded as-is, and the E2M1/E4M3
/// decode happens on the device through the same gated lookup tables the CPU
/// lane uses. That removes the 60.2s of a measured 113.7s forward that was
/// scalar unpacking, and the 67 MB per-expert host allocation with it.
///
/// All of an expert's tokens go through in one matmul rather than one each.
/// That reassociates the sums, so this is NOT bitwise identical to the host
/// lane and must be gated on a tolerance, not on equality.
///
/// Syncs before returning, so the caller's timer measures work and not
/// enqueueing.
#[cfg(feature = "inkling-cuda")]
#[allow(clippy::too_many_arguments)]
fn routed_experts_gpu<B: Backend>(
    cp: &Checkpoint,
    prefix: &str,
    by_expert: &BTreeMap<usize, Vec<(usize, f32)>>,
    hn: &[f32],
    n: usize,
    h: usize,
    inter: usize,
    dev: &B::Device,
    host: &mut (f64, f64),
) -> Result<Vec<f32>> {
    use burn::tensor::{Int, Tensor, TensorData};
    use mary::models::inkling::burn::{
        deinterleave_rows_device, expert_ffn, expert_weight_from_packed,
    };

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
        let t_s = Instant::now();
        let (fused, dn) = if cp.is_nvfp4(&n13) {
            let w13 = cp.expert_slice_packed(&n13, e)?;
            let w2 = cp.expert_slice_packed(&n2, e)?;
            host.0 += t_s.elapsed().as_secs_f64();
            let t_w = Instant::now();
            let r = (
                expert_weight_from_packed::<B>(
                    &w13.codes, &w13.scales, w13.scale2, w13.rows, w13.cols, dev,
                ),
                expert_weight_from_packed::<B>(
                    &w2.codes, &w2.scales, w2.scale2, w2.rows, w2.cols, dev,
                ),
            );
            host.1 += t_w.elapsed().as_secs_f64();
            r
        } else {
            let a = cp.expert_slice(&n13, e)?;
            let b = cp.expert_slice(&n2, e)?;
            host.0 += t_s.elapsed().as_secs_f64();
            let t_w = Instant::now();
            let r = (
                Tensor::<B, 2>::from_data(TensorData::new(a.data, [2 * inter, h]), dev),
                Tensor::<B, 2>::from_data(TensorData::new(b.data, [h, inter]), dev),
            );
            host.1 += t_w.elapsed().as_secs_f64();
            r
        };
        // One permutation for both branches: the fused tensor arrives in
        // checkpoint interleave either way.
        //
        // INK_MUTATE_NO_DEINTERLEAVE exists so the lane gate can be watched
        // rejecting something, because a gate that has never failed and a gate
        // that cannot fail look identical from outside.
        let gu = if std::env::var("INK_MUTATE_NO_DEINTERLEAVE").is_ok() {
            fused
        } else {
            deinterleave_rows_device(fused)
        };
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

fn main() -> Result<()> {
    let ckpt = std::env::args().nth(1).map(PathBuf::from).context("usage: <ckpt> <ids> <out>")?;
    let ids_path = std::env::args().nth(2).map(PathBuf::from).context("usage: <ckpt> <ids> <out>")?;
    let out_path = std::env::args().nth(3).map(PathBuf::from).context("usage: <ckpt> <ids> <out>")?;

    let cfg_text = std::fs::read_to_string(ckpt.join("config.json"))?;
    let cfg = InklingConfig::from_json(&cfg_text).context("parsing config.json")?;
    let t = &cfg.text_config;
    let cp = Checkpoint::open(&ckpt)?;

    let mut ids: Vec<usize> = std::fs::read(&ids_path)?
        .chunks_exact(8)
        .map(|c| i64::from_le_bytes(c.try_into().unwrap()) as usize)
        .collect();
    let n = ids.len();
    anyhow::ensure!(n > 0, "no tokens — the forward would be vacuous");

    let h = t.hidden_size;
    println!("=== forward ===");
    println!("  checkpoint : {}", ckpt.display());
    println!("  tokens     : {n}  {ids:?}");
    println!("  layers     : {}  hidden {h}  experts {}+{} shared",
             t.num_hidden_layers, t.n_routed_experts, t.n_shared_experts);
    println!("  tensors    : {}", cp.len());

    // How many tokens to generate past the prompt. 0 reproduces the original
    // single-forward behaviour exactly.
    let gen_steps: usize = std::env::var("INK_GEN")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(0);

    let started = Instant::now();
    // Hoisted: re-reading 4.8 GB of embedding tables per generated token would
    // dwarf everything else in the loop.
    let embed_w = cp.tensor("model.llm.embed.weight")?.data;
    let embed_n = cp.tensor("model.llm.embed_norm.weight")?.data;
    let fnorm = cp.tensor("model.llm.norm.weight")?.data;
    #[allow(unused_mut)]
    let mut unembed = cp.tensor("model.llm.unembed.weight")?.data;
    println!("  embedding tables loaded in {:.1}s", started.elapsed().as_secs_f32());

    // Experts are read, applied and dropped. There is no decoded-expert cache:
    // it measured as no speedup, and it existed to paper over a capacity
    // shortfall (160 GB of checkpoint against 119 GB of box) that a second
    // Spark closes. See this file's header.
    // One switch per lane, so a regression can be bisected to the lane that
    // caused it, plus `INK_GPU=all` for the everything-on-device run.
    let all_gpu = std::env::var("INK_GPU").map(|v| v == "all").unwrap_or(false);
    let lane = |k: &str| all_gpu || std::env::var(k).map(|v| v == "gpu").unwrap_or(false);
    let experts_on_gpu = lane("INK_EXPERTS");
    let attn_on_gpu = lane("INK_ATTN");
    let mlp_on_gpu = lane("INK_MLP");
    let head_on_gpu = lane("INK_HEAD");
    // The KV cache. Off by default so the uncached lane stays available as the
    // thing to check against: the decisive test of a cache is that it produces
    // the same token sequence as not having one, and that needs both to run.
    let kv = std::env::var("INK_KV").map(|v| v == "1" || v == "on").unwrap_or(false);
    anyhow::ensure!(
        !kv || attn_on_gpu,
        "INK_KV needs attention on the device -- set INK_ATTN=gpu or INK_GPU=all"
    );
    let say = |b: bool| if b { "device" } else { "host (f32 oracle)" };
    println!("  attention          : {}", say(attn_on_gpu));
    println!("  shared + dense MLP : {}", say(mlp_on_gpu));
    println!("  routed experts     : {}", say(experts_on_gpu));
    println!("  head (unembed)     : {}", say(head_on_gpu));
    println!("  kv cache           : {}", if kv { "on" } else { "off (prefix recomputed each step)" });
    if std::env::var("INK_HOST_SUM").map(|v| v == "reverse").unwrap_or(false) {
        println!("  host sum order     : REVERSED (reassociation control)");
    }
    if std::env::var("INK_MUTATE_NO_DEINTERLEAVE").is_ok() {
        println!("  !! MUTATION ACTIVE : deinterleave SKIPPED -- this output is expected to be WRONG");
    }
    #[cfg(feature = "inkling-cuda")]
    let dev = burn::backend::cuda::CudaDevice::default();
    #[cfg(not(feature = "inkling-cuda"))]
    anyhow::ensure!(
        !(experts_on_gpu || attn_on_gpu || mlp_on_gpu || head_on_gpu),
        "a device lane needs --features inkling-cuda"
    );

    // The unembed table is 3.3 GB at f32 and does not change between generated
    // tokens, so it is uploaded ONCE here rather than once per step, and the
    // host copy is dropped rather than kept alongside it.
    #[cfg(feature = "inkling-cuda")]
    let unembed_dev = if head_on_gpu {
        let v = t.effective_vocab();
        let d = up2::<Bk>(std::mem::take(&mut unembed), t.vocab_size, h, &dev)
            .slice([0..v, 0..h]);
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

    let mut top_all: Vec<i64> = Vec::new();
    for step in 0..=gen_steps {
    let pass = Instant::now();
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
    let n = feed.len();
    let mut x = embed_and_norm(&feed, &embed_w, &embed_n, t.rms_norm_eps, t.vocab_size, h);

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
    // Reading a tensor out of the mapping and widening BF16 to f32 is host work
    // no lane can move, so it is counted once, separately, rather than being
    // smeared across whichever bucket happened to ask for the weight.
    let t_read = std::cell::Cell::new(0f64);
    // (slice, widen+upload) -- host-side and therefore honestly attributable,
    // unlike anything downstream of an enqueued device call.
    let mut host_t = (0f64, 0f64);

    for layer in 0..t.num_hidden_layers {
        let l0 = Instant::now();
        let kind = t.attn_kind(layer);
        let is_local = kind == AttnKind::Local;
        let (heads, kv_heads, head_dim) = t.heads(kind);
        let p = format!("model.llm.layers.{layer}.");
        let g = |nm: &str| -> Result<Vec<f32>> {
            let s = Instant::now();
            let r = cp.tensor(&format!("{p}{nm}"))?.data;
            t_read.set(t_read.get() + s.elapsed().as_secs_f64());
            Ok(r)
        };

        // ---- attention ----------------------------------------------------
        let t_a = Instant::now();
        let attn_norm = g("attn_norm.weight")?;
        let hn = rms_norm(&x, &attn_norm, t.rms_norm_eps, n, h);
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
                let w = dev_lane::AttnWeightsDev::<Bk> {
                    wq: up2(g("attn.wq_du.weight")?, heads * head_dim, h, &dev),
                    wk: up2(g("attn.wk_dv.weight")?, kv_heads * head_dim, h, &dev),
                    wv: up2(g("attn.wv_dv.weight")?, kv_heads * head_dim, h, &dev),
                    wr: up2(g("attn.wr_du.weight")?, heads * t.d_rel, h, &dev),
                    wo: up2(g("attn.wo_ud.weight")?, h, heads * head_dim, &dev),
                    k_sconv: up2(g("attn.k_sconv.weight")?, kv_heads * head_dim, t.sconv_kernel_size, &dev),
                    v_sconv: up2(g("attn.v_sconv.weight")?, kv_heads * head_dim, t.sconv_kernel_size, &dev),
                    q_norm: up1(g("attn.q_norm.weight")?, head_dim, &dev),
                    k_norm: up1(g("attn.k_norm.weight")?, head_dim, &dev),
                    rel_proj: up2(g("attn.rel_logits_proj.proj")?, t.d_rel, t.rel_span(kind), &dev),
                };
                let sconv_w = up2(g("attn_sconv.weight")?, h, t.sconv_kernel_size, &dev);
                if kv && step > 0 {
                    let y = dev_lane::attention_step(
                        up2(hn.clone(), n, h, &dev),
                        &w,
                        &dims,
                        Some(ls),
                        pos0,
                        window,
                        &mut caches[layer].attn,
                    );
                    let (out, hist) = dev_lane::short_conv_step(
                        caches[layer].attn_sconv.clone(),
                        y,
                        sconv_w,
                    );
                    caches[layer].attn_sconv = hist;
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
                }
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
                wq: &wq, wk: &wk, wv: &wv, wr: &wr, wo: &wo,
                k_sconv: &ks, v_sconv: &vs, q_norm: &qn, k_norm: &kn, rel_proj: &rp,
            };
            let a = attention(&hn, &aw, &dims, Some(ls), mask, n);
            drop((wq, wk, wv, wr, wo));
            short_conv(&a, &g("attn_sconv.weight")?, n, h, t.sconv_kernel_size)
        };
        for (xi, ai) in x.iter_mut().zip(&a) {
            *xi += ai;
        }

        // ---- MLP ----------------------------------------------------------
        t_attn += t_a.elapsed().as_secs_f64();
        let t_o = Instant::now();
        let mlp_norm = g("mlp_norm.weight")?;
        let hn = rms_norm(&x, &mlp_norm, t.rms_norm_eps, n, h);

        let y = if t.is_dense(layer) {
            let di = t.dense_intermediate_size;
            let gs = g("mlp.global_scale")?[0];
            if mlp_on_gpu {
                #[cfg(feature = "inkling-cuda")]
                {
                    // The gate/up de-interleave happens on device too: the fused
                    // dense weight is 134 M elements, and shuffling it in a
                    // scalar loop costs more than the matmul it feeds.
                    let fused = up2::<Bk>(g("mlp.w13_dn.weight")?, 2 * di, h, &dev);
                    let gu = dev_lane::deinterleave_rows_device(fused);
                    let gate = gu.clone().slice([0..di, 0..h]);
                    let upw = gu.slice([di..2 * di, 0..h]);
                    down(dev_lane::dense_mlp(
                        up2(hn.clone(), n, h, &dev),
                        gate,
                        upw,
                        up2(g("mlp.w2_md.weight")?, h, di, &dev),
                        gs,
                    ))
                }
                #[cfg(not(feature = "inkling-cuda"))]
                unreachable!("guarded at startup")
            } else {
                let fused = g("mlp.w13_dn.weight")?;
                let (gate, up) = split_gate_up(&fused, h);
                let down = g("mlp.w2_md.weight")?;
                dense_mlp(&hn, &gate, &up, &down, gs, n, h, di)
            }
        } else {
            let inter = t.intermediate_size;
            let rw = g("mlp.gate.weight")?;
            let rb = g("mlp.gate.bias")?;
            let rg = g("mlp.gate.global_scale")?;
            let routing: Vec<Routing> = route(
                &hn, &rw, &rb, rg[0], t.route_scale as f32,
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
                    let a = routed_experts_gpu::<Bk>(&cp, &p, &by_expert, &hn, n, h, inter, &dev, &mut host_t)?;
                    expert_loads += by_expert.len();
                    a
                }
                #[cfg(not(feature = "inkling-cuda"))]
                unreachable!("guarded at startup")
            } else {
                let mut acc = vec![0f32; n * h];
                for (&e, toks) in &by_expert {
                    let gu_raw = cp.expert_slice(&format!("{p}mlp.experts.w13_weight"), e)?.data;
                    let gu = deinterleave_fused(&gu_raw, 2 * inter, h);
                    let dn = cp.expert_slice(&format!("{p}mlp.experts.w2_weight"), e)?.data;
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
            let sh = if mlp_on_gpu {
                #[cfg(feature = "inkling-cuda")]
                {
                    let sfused = g("mlp.shared_experts.shared_w13_weight")?;
                    let (sg, su) = dev_lane::split_shared_fused(
                        up2::<Bk>(sfused, ns * 2 * inter, h, &dev), ns);
                    down(dev_lane::shared_experts(
                        up2(hn.clone(), n, h, &dev),
                        sg,
                        su,
                        up2(g("mlp.shared_experts.shared_w2_weight")?, ns * h, inter, &dev),
                        up2(gammas, n, ns, &dev),
                        ns,
                    ))
                }
                #[cfg(not(feature = "inkling-cuda"))]
                unreachable!("guarded at startup")
            } else {
                let sfused = g("mlp.shared_experts.shared_w13_weight")?;
                let per = sfused.len() / ns;
                let mut sg = Vec::with_capacity(sfused.len() / 2);
                let mut su = Vec::with_capacity(sfused.len() / 2);
                for s in 0..ns {
                    let blk = &sfused[s * per..(s + 1) * per];
                    let (a, b) = mary::models::inkling::load::deinterleave_rows(blk, 2 * inter, h);
                    sg.extend_from_slice(&a);
                    su.extend_from_slice(&b);
                }
                drop(sfused);
                let sd = g("mlp.shared_experts.shared_w2_weight")?;
                shared_experts(&hn, &sg, &su, &sd, &gammas, ns, n, h, inter)
            };
            t_shared += t_s.elapsed().as_secs_f64();
            acc.iter().zip(&sh).map(|(a, b)| a + b).collect()
        };

        // The MLP half's own short convolution carries state across generated
        // tokens exactly as attention's do.
        let mlp_sconv_w = g("mlp_sconv.weight")?;
        let y = if kv {
            #[cfg(feature = "inkling-cuda")]
            {
                if step > 0 {
                    short_conv_step_host(
                        &mut caches[layer].mlp_sconv,
                        &y,
                        &mlp_sconv_w,
                        h,
                        t.sconv_kernel_size,
                    )
                } else {
                    caches[layer].mlp_sconv =
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

    // ---- head --------------------------------------------------------------
    let v = t.effective_vocab();
    let t_h = Instant::now();
    let logits = if head_on_gpu {
        #[cfg(feature = "inkling-cuda")]
        {
            // 109 x 4096 x 200058 is 89 G multiply-adds — the single largest
            // matmul in the forward, and the one left standing once attention
            // and the MLPs move. The muP divisor divides BEFORE the projection,
            // matching the reference: doing it after is algebraically equal and
            // numerically not.
            let hs = dev_lane::rms_norm(
                up2::<Bk>(x.clone(), n, h, &dev),
                up1(fnorm.clone(), h, &dev),
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
            &x, &fnorm, &unembed,
            t.logits_mup_width_multiplier as f32,
            t.vocab_size, t.effective_vocab(), t.rms_norm_eps, n, h,
        )
    };
    let t_head = t_h.elapsed().as_secs_f64();

    println!("\n=== predictions ===");
    println!("  expert slabs decoded: {expert_loads}");
    // t_other covers the whole MLP half, so the expert buckets are inside it.
    println!("  where the time went, seconds:");
    println!("    attention half      {t_attn:8.1}   ({})", if attn_on_gpu { "device" } else { "host" });
    println!("    mlp half            {t_other:8.1}   of which:");
    println!("      routed experts    {t_expert:8.1}   ({})",
             if experts_on_gpu { "slice + upload + dequant + matmul, device" }
             else { "disk + NVFP4 unpack + matmul, host" });
    println!("      shared experts    {t_shared:8.1}   ({})", if mlp_on_gpu { "device" } else { "host" });
    println!("      rest of the half  {:8.1}   (routing, dense layers, sconv, norms)",
             t_other - t_expert - t_shared);
    println!("    head / unembed      {t_head:8.1}   ({})", if head_on_gpu { "device" } else { "host" });
    println!("    of the above, host-only tensor reads (mmap + BF16 widening): {:8.1}", t_read.get());
    if experts_on_gpu {
        println!("    of the routed-expert total, the host-synchronous parts:");
        println!("      slice from mmap   {:8.1}", host_t.0);
        println!("      widen + upload    {:8.1}", host_t.1);
        println!("      remainder         {:8.1}   (enqueue + the sync, so device work lives here)",
                 t_expert - host_t.0 - host_t.1);
    }
    println!("  elapsed: {:.1}s", started.elapsed().as_secs_f32());


    // Greedy: the last position's argmax is the next token.
    let last = &logits[(n - 1) * v..n * v];
    let mut best = 0usize;
    for (i, &val) in last.iter().enumerate() {
        if val > last[best] {
            best = i;
        }
    }

    // Per-position top-5. Uncached, the final pass has recomputed every
    // position, so it reports all of them and earlier passes report nothing.
    // Cached, each pass computes only the positions it was handed and they
    // accumulate -- prefill contributes the prompt, every step one more -- so
    // the two lanes end with the same table over the same sequence, which is
    // what makes the outputs comparable.
    if !kv {
        top_all.clear();
    }
    if kv || step == gen_steps {
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
        ids.push(best);
    }
    if step == gen_steps {
        break;
    }
    }

    let mut bytes = Vec::new();
    for i in top_all {
        bytes.extend_from_slice(&i.to_le_bytes());
    }
    std::fs::write(&out_path, &bytes)?;
    println!("  wrote top-5 ids per position to {}", out_path.display());
    Ok(())
}
