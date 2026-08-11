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

/// The backend the device lane runs on.
#[cfg(feature = "inkling-cuda")]
type Bk = burn::backend::Cuda<f32>;
#[cfg(feature = "inkling-cuda")]
use burn::prelude::Backend;

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


/// Every routed expert for one layer, on the NATIVE NVFP4 tensor-core path.
///
/// Differs from [`routed_experts_gpu`] in two independent ways, which the
/// timers keep apart because they have completely different fixes:
///
/// * the slab arrives from [`ExpertSource`], which parsed every shard header
///   once at startup and hands out a borrow, instead of
///   `Checkpoint::expert_slice_packed`, which re-runs
///   `SafeTensors::deserialize` four times per slab and then copies 12.6 MB;
/// * the packed bytes go straight into `mma.sync…kind::mxf4nvf4…ue4m3`
///   instead of being decoded into a 67.1 + 33.6 MB f32 pair per expert that
///   is read once and dropped.
///
/// Activations are quantised to E2M1 in dynamic per-16 blocks with E4M3
/// scales, which the instruction requires and which is what the checkpoint's
/// own `hf_quant_config.json` specifies for `*input_quantizer`. This lane is
/// therefore CLOSER to the checkpoint's intended numerics than the f32-
/// activation lane it replaces, not further from it.
#[cfg(feature = "inkling-cuda")]
#[allow(clippy::too_many_arguments)]
fn routed_experts_fp4(
    src: &mary::models::inkling::fp4gemm::ExpertSource,
    aliases: Option<&mary::models::inkling::fp4gemm::AliasedShards>,
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
    use mary::models::inkling::fp4gemm::{
        alias_or_copy, fp4_linear_launch, gate_up_silu_launch, MTILE,
    };
    use mary::models::inkling::fp4quant::quantize_nvfp4;

    let n13 = format!("{prefix}mlp.experts.w13_weight");
    let n2 = format!("{prefix}mlp.experts.w2_weight");
    let mut acc = vec![0f32; n * h];

    for (&e, toks) in by_expert {
        let t_s = Instant::now();
        let w13 = src.expert(&n13, e)?;
        let w2 = src.expert(&n2, e)?;
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

        // Zero copy where the hardware allows it: the GPU reads the
        // checkpoint's mmap'd pages in place. The shards were registered ONCE
        // at startup, so this is offset arithmetic, not a device round trip.
        let (b, bsc) = match aliases.and_then(|al| src.expert_aliased(al, &n13, e).ok().flatten()) {
            Some(v) => v,
            None => (
                alias_or_copy(client, w13.codes, w13.codes_keep.clone()),
                alias_or_copy(client, w13.scales, w13.scales_keep.clone()),
            ),
        };
        let both = fp4_linear_launch(client, &a, &asc, &b, &bsc, m_pad, h, 2 * inter, w13.scale2);

        let act_h = gate_up_silu_launch(client, &both, m_pad, inter);
        let (a2, asc2) = quantize_nvfp4(client, &act_h, m_pad, inter);

        let (b2, bsc2) = match aliases.and_then(|al| src.expert_aliased(al, &n2, e).ok().flatten()) {
            Some(v) => v,
            None => (
                alias_or_copy(client, w2.codes, w2.codes_keep.clone()),
                alias_or_copy(client, w2.scales, w2.scales_keep.clone()),
            ),
        };
        let y_h = fp4_linear_launch(client, &a2, &asc2, &b2, &bsc2, m_pad, inter, h, w2.scale2);
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
    let unembed = cp.tensor("model.llm.unembed.weight")?.data;
    println!("  embedding tables loaded in {:.1}s", started.elapsed().as_secs_f32());

    // Experts are read, applied and dropped. There is no decoded-expert cache:
    // it measured as no speedup, and it existed to paper over a capacity
    // shortfall (160 GB of checkpoint against 119 GB of box) that a second
    // Spark closes. See this file's header.
    let ink_experts = std::env::var("INK_EXPERTS").unwrap_or_default();
    let experts_fp4 = ink_experts == "fp4";
    let experts_on_gpu = ink_experts == "gpu" || experts_fp4;
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
    if std::env::var("INK_HOST_SUM").map(|v| v == "reverse").unwrap_or(false) {
        println!("  host sum order     : REVERSED (reassociation control)");
    }
    if std::env::var("INK_MUTATE_NO_DEINTERLEAVE").is_ok() {
        println!("  !! MUTATION ACTIVE : deinterleave SKIPPED -- this output is expected to be WRONG");
    }
    #[cfg(feature = "inkling-cuda")]
    let dev = burn::backend::cuda::CudaDevice::default();
    // Parsed once for the whole run. The lane it replaces re-parsed a shard
    // header four times per expert slab, ~9950 times over a forward.
    #[cfg(feature = "inkling-cuda")]
    let fp4_src = if experts_fp4 {
        let t = Instant::now();
        let s = mary::models::inkling::fp4gemm::ExpertSource::open(&ckpt)?;
        println!("  ExpertSource       : all shard headers parsed in {:.1} ms", t.elapsed().as_secs_f64() * 1e3);
        Some(s)
    } else {
        None
    };
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
    let fp4_aliases = match (&fp4_src, &fp4_client) {
        // INK_ZEROCOPY=0 forces the copying lane, so the seam can be A/B'd
        // against it with the page cache in the same state.
        (Some(s), Some(c)) if zerocopy_on => {
            let t = Instant::now();
            let a = s.alias_shards(c);
            println!(
                "  zero-copy shards   : {} in {:.1} ms",
                if a.is_some() { "registered" } else { "UNSUPPORTED, copying" },
                t.elapsed().as_secs_f64() * 1e3
            );
            a
        }
        _ => None,
    };
    #[cfg(not(feature = "inkling-cuda"))]
    anyhow::ensure!(!experts_on_gpu, "INK_EXPERTS=gpu needs --features inkling-cuda");

    let mut top_all: Vec<i64> = Vec::new();
    for step in 0..=gen_steps {
    let n = ids.len();
    let mut x = embed_and_norm(&ids, &embed_w, &embed_n, t.rms_norm_eps, t.vocab_size, h);

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
    let (mut t_attn, mut t_expert, mut t_other) = (0f64, 0f64, 0f64);
    // (slice, widen+upload) -- host-side and therefore honestly attributable,
    // unlike anything downstream of an enqueued device call.
    let mut host_t = (0f64, 0f64);

    for layer in 0..t.num_hidden_layers {
        let l0 = Instant::now();
        let kind = t.attn_kind(layer);
        let is_local = kind == AttnKind::Local;
        let (heads, kv_heads, head_dim) = t.heads(kind);
        let p = format!("model.llm.layers.{layer}.");
        let g = |nm: &str| -> Result<Vec<f32>> { Ok(cp.tensor(&format!("{p}{nm}"))?.data) };

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
        let mask = if is_local { &mask_local } else { &mask_global };
        let a = attention(&hn, &aw, &dims, Some(ls), mask, n);
        drop((wq, wk, wv, wr, wo));
        let a = short_conv(&a, &g("attn_sconv.weight")?, n, h, t.sconv_kernel_size);
        for (xi, ai) in x.iter_mut().zip(&a) {
            *xi += ai;
        }

        // ---- MLP ----------------------------------------------------------
        t_attn += t_a.elapsed().as_secs_f64();
        let t_o = Instant::now();
        let mlp_norm = g("mlp_norm.weight")?;
        let hn = rms_norm(&x, &mlp_norm, t.rms_norm_eps, n, h);

        let y = if t.is_dense(layer) {
            let fused = g("mlp.w13_dn.weight")?;
            let (gate, up) = split_gate_up(&fused, h);
            let down = g("mlp.w2_md.weight")?;
            let gs = g("mlp.global_scale")?;
            dense_mlp(&hn, &gate, &up, &down, gs[0], n, h, t.dense_intermediate_size)
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
                    // Layer 2 is BF16 and has no `.scale` sidecar, so the FP4
                    // lane cannot take it; that one layer falls back rather
                    // than the whole run refusing.
                    let packed = fp4_src
                        .as_ref()
                        .map(|s| s.is_nvfp4(&format!("{p}mlp.experts.w13_weight")))
                        .unwrap_or(false);
                    let a = if experts_fp4 && packed {
                        routed_experts_fp4(
                            fp4_src.as_ref().unwrap(),
                            fp4_aliases.as_ref(),
                            fp4_client.as_ref().unwrap(),
                            &p, &by_expert, &hn, n, h, inter, &mut host_t,
                        )?
                    } else {
                        routed_experts_gpu::<Bk>(&cp, &p, &by_expert, &hn, n, h, inter, &dev, &mut host_t)?
                    };
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

            let sfused = cp.tensor(&format!("{p}mlp.shared_experts.shared_w13_weight"))?.data;
            let per = sfused.len() / t.n_shared_experts;
            let mut sg = Vec::with_capacity(sfused.len() / 2);
            let mut su = Vec::with_capacity(sfused.len() / 2);
            for s in 0..t.n_shared_experts {
                let blk = &sfused[s * per..(s + 1) * per];
                let (a, b) = mary::models::inkling::load::deinterleave_rows(blk, 2 * inter, h);
                sg.extend_from_slice(&a);
                su.extend_from_slice(&b);
            }
            drop(sfused);
            let sd = cp.tensor(&format!("{p}mlp.shared_experts.shared_w2_weight"))?.data;
            let gammas: Vec<f32> = routing.iter().flat_map(|r| r.shared_gammas.clone()).collect();
            let sh = shared_experts(&hn, &sg, &su, &sd, &gammas, t.n_shared_experts, n, h, inter);
            acc.iter().zip(&sh).map(|(a, b)| a + b).collect()
        };

        let y = short_conv(&y, &g("mlp_sconv.weight")?, n, h, t.sconv_kernel_size);
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
    let logits = head(
        &x, &fnorm, &unembed,
        t.logits_mup_width_multiplier as f32,
        t.vocab_size, t.effective_vocab(), t.rms_norm_eps, n, h,
    );
    let v = t.effective_vocab();

    println!("\n=== predictions ===");
    println!("  expert slabs decoded: {expert_loads}");
    // t_other covers the whole MLP half, so the expert buckets are inside it.
    println!("  where the time went, seconds:");
    println!("    attention half      {t_attn:8.1}");
    println!("    mlp half            {t_other:8.1}   of which:");
    println!("      routed experts    {t_expert:8.1}   ({})",
             if experts_fp4 { "borrow + upload + NVFP4 tensor cores, device" }
             else if experts_on_gpu { "slice + upload + dequant + matmul, device" }
             else { "disk + NVFP4 unpack + matmul, host" });
    println!("      shared + dense    {:8.1}", t_other - t_expert);
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
    if gen_steps > 0 {
        println!("  step {step}: +{best}");
        ids.push(best);
        if step < gen_steps {
            continue;
        }
    }

    top_all.clear();
    for ti in 0..n {
        let row = &logits[ti * v..(ti + 1) * v];
        let mut idx: Vec<usize> = (0..v).collect();
        idx.sort_unstable_by(|&a, &b| row[b].partial_cmp(&row[a]).unwrap());
        let top: Vec<usize> = idx[..5].to_vec();
        println!("  after token {ti} (id {}): top5 {:?}  logits {:?}",
                 ids[ti], top,
                 top.iter().map(|&i| (row[i] * 100.0).round() / 100.0).collect::<Vec<_>>());
        for &i in &top {
            top_all.push(i as i64);
        }
    }

    break;
    }

    let mut bytes = Vec::new();
    for i in top_all {
        bytes.extend_from_slice(&i.to_le_bytes());
    }
    std::fs::write(&out_path, &bytes)?;
    println!("  wrote top-5 ids per position to {}", out_path.display());
    Ok(())
}
