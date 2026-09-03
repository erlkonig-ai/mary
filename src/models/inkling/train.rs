//! The training path's layer transcription (compass 84605490): one Inkling
//! decoder layer as Burn ops under `Autodiff<Cuda<f32>>`, loaded one layer at a
//! time from the safetensors checkpoint, so a training step never holds more
//! than one layer's weights. No host reference, no parity gate (JP, 2026-09-03:
//! no numerical comparison machinery; the gate is the loss on the user turns).
//!
//! Memory discipline, learned the hard way at 02:55 the same night: every
//! allocation point sits behind [`mem_guard`], and a caller must drop one layer
//! before loading the next.
use crate::models::inkling::attn::{AttnDims, LogScaling, causal_mask};
use crate::models::inkling::block::Routing;
use crate::models::inkling::burn::{linear as b_linear, rms_norm as b_rms_norm, short_conv as b_short_conv, silu as b_silu};
use crate::models::inkling::config::{AttnKind, InklingTextConfig};
pub use crate::models::inkling::config::InklingConfig;
use crate::models::inkling::load::{Checkpoint, deinterleave_fused, split_gate_up, split_shared_w13};
use anyhow::Result;
use burn::backend::{Autodiff, Cuda};
use burn::prelude::*;
use burn::tensor::activation;
use std::collections::HashMap;

pub type Bk = Cuda<f32>;
pub type Ad = Autodiff<Bk>;
pub type Dev = burn::backend::cuda::CudaDevice;

/// The settled reading of `shared_w13_weight` (inkling_real_gate): INTERLEAVED.
pub const SHARED_W13_HALVED: bool = false;

// ------------------------------------------------------------------ memory guard
pub fn mem_available_gib() -> f64 {
    let s = std::fs::read_to_string("/proc/meminfo").unwrap_or_default();
    for line in s.lines() {
        if let Some(rest) = line.strip_prefix("MemAvailable:") {
            let kb: f64 = rest.trim().trim_end_matches("kB").trim().parse().unwrap_or(0.0);
            return kb / (1024.0 * 1024.0);
        }
    }
    0.0
}
pub const MEM_FLOOR_GIB: f64 = 28.0;

/// Return the device pool's unused pages to the driver and trim the host heap, then report.
/// Unified memory: pool growth IS host memory loss, so this runs at every layer boundary.
pub fn pool_release(dev: &Dev) -> String {
    use cubecl::Runtime;
    let client = <cubecl::cuda::CudaRuntime as Runtime>::client(dev);
    client.memory_cleanup();
    unsafe { libc::malloc_trim(0); }
    match client.memory_usage() {
        Ok(u) => format!("pool used {:.2} GiB reserved {:.2} GiB ({} allocs)", u.bytes_in_use as f64 / 1073741824.0, u.bytes_reserved as f64 / 1073741824.0, u.number_allocs),
        Err(e) => format!("pool usage unavailable: {e:?}"),
    }
}
/// Abort before the box does. The GB10 is unified memory: host MemAvailable is
/// the number that matters for the device pool as well.
pub fn mem_guard(where_: &str) {
    let a = mem_available_gib();
    if a < MEM_FLOOR_GIB {
        eprintln!("MEMORY GUARD: MemAvailable {a:.1} GiB < floor {MEM_FLOOR_GIB} GiB at {where_} -- aborting before the box does");
        std::process::exit(3);
    }
}

// ------------------------------------------------------------------ host + device weights
pub struct HostDense { pub gate: Vec<f32>, pub up: Vec<f32>, pub down: Vec<f32>, pub global_scale: f32, pub di: usize }

pub struct HostW {
    pub attn_norm: Vec<f32>, pub mlp_norm: Vec<f32>, pub attn_sconv: Vec<f32>, pub mlp_sconv: Vec<f32>,
    pub wq: Vec<f32>, pub wk: Vec<f32>, pub wv: Vec<f32>, pub wr: Vec<f32>, pub wo: Vec<f32>,
    pub qn: Vec<f32>, pub kn: Vec<f32>, pub ks: Vec<f32>, pub vs: Vec<f32>, pub rp: Vec<f32>,
    /// MoE layers only
    pub rw: Vec<f32>, pub rb: Vec<f32>, pub rg: f32,
    pub sgate: Vec<f32>, pub sup: Vec<f32>, pub sdown: Vec<f32>,
    /// routed experts actually touched: e -> (w13 [2mi,h], w2 [h,mi])
    pub experts: HashMap<usize, (Vec<f32>, Vec<f32>)>,
    pub dense: Option<HostDense>,
}

pub struct Geom {
    pub layer: usize, pub n: usize, pub h: usize, pub mi: usize, pub n_routed: usize, pub n_shared: usize, pub top_k: usize,
    pub route_scale: f32, pub kernel: usize, pub eps: f64, pub dims: AttnDims, pub ls: LogScaling, pub mask: Vec<f32>,
    pub heads: usize, pub kv_heads: usize, pub hd: usize, pub d_rel: usize, pub rel_extent: usize, pub is_dense: bool,
}

pub struct DevDense<B: Backend> { pub gate: Tensor<B, 2>, pub up: Tensor<B, 2>, pub down: Tensor<B, 2>, pub global_scale: f32 }

pub struct DevW<B: Backend> {
    pub attn_norm: Tensor<B, 1>, pub mlp_norm: Tensor<B, 1>, pub attn_sconv: Tensor<B, 2>, pub mlp_sconv: Tensor<B, 2>,
    pub wq: Tensor<B, 2>, pub wk: Tensor<B, 2>, pub wv: Tensor<B, 2>, pub wr: Tensor<B, 2>, pub wo: Tensor<B, 2>,
    pub qn: Tensor<B, 1>, pub kn: Tensor<B, 1>, pub ks: Tensor<B, 2>, pub vs: Tensor<B, 2>, pub rp: Tensor<B, 2>,
    pub rw: Tensor<B, 2>, pub sgate: Vec<Tensor<B, 2>>, pub sup: Vec<Tensor<B, 2>>, pub sdown: Vec<Tensor<B, 2>>,
    pub experts: HashMap<usize, (Tensor<B, 2>, Tensor<B, 2>)>,
    pub dense: Option<DevDense<B>>,
}

/// Weight leaf. `require_grad` is a no-op on a backend without autodiff, so the same
/// builders serve the graph-free forward (`Bk`) and the backward (`Ad`).
pub fn t2<B: Backend>(dev: &B::Device, v: &[f32], r: usize, c: usize) -> Tensor<B, 2> {
    assert_eq!(v.len(), r * c, "t2: {} != {r}x{c}", v.len());
    Tensor::<B, 2>::from_data(TensorData::new(v.to_vec(), [r, c]), dev).require_grad()
}
pub fn t1<B: Backend>(dev: &B::Device, v: &[f32]) -> Tensor<B, 1> {
    Tensor::<B, 1>::from_data(TensorData::new(v.to_vec(), [v.len()]), dev).require_grad()
}
pub fn c2<B: Backend>(dev: &B::Device, v: Vec<f32>, r: usize, c: usize) -> Tensor<B, 2> {
    Tensor::<B, 2>::from_data(TensorData::new(v, [r, c]), dev)
}
pub fn ints<B: Backend>(dev: &B::Device, v: Vec<i64>) -> Tensor<B, 1, Int> {
    let n = v.len();
    Tensor::<B, 1, Int>::from_data(TensorData::new(v, [n]), dev)
}

/// Everything a layer needs except its routed experts (which depend on routing).
pub fn load_layer(cp: &Checkpoint, t: &InklingTextConfig, layer: usize, n: usize) -> Result<(HostW, Geom)> {
    let pfx = format!("model.llm.layers.{layer}.");
    let g = |nm: &str| -> Result<Vec<f32>> { Ok(cp.tensor(&format!("{pfx}{nm}"))?.data) };
    let h = t.hidden_size;
    let kind = t.attn_kind(layer);
    let (heads, kv_heads, hd) = t.heads(kind);
    let is_dense = t.is_dense(layer);
    let rp_loaded = cp.tensor(&format!("{pfx}attn.rel_logits_proj.proj"))?;
    anyhow::ensure!(rp_loaded.shape.len() == 2 && rp_loaded.shape[0] == t.d_rel, "layer {layer}: rel_proj shape {:?}", rp_loaded.shape);
    let rel_extent = rp_loaded.shape[1];
    let kernel = t.sconv_kernel_size;
    let window = match kind { AttnKind::Local => Some(t.sliding_window_size), AttnKind::Global => None };
    let mi = t.intermediate_size;
    let (mut rw, mut rb, mut rg, mut sgate, mut sup, mut sdown, mut dense) = (vec![], vec![], 1.0f32, vec![], vec![], vec![], None);
    if is_dense {
        let fused = g("mlp.w13_dn.weight")?;
        let (gate, up) = split_gate_up(&fused, h);
        let di = t.dense_intermediate_size;
        anyhow::ensure!(gate.len() == di * h, "layer {layer}: dense gate {} != {di}x{h}", gate.len());
        dense = Some(HostDense { gate, up, down: g("mlp.w2_md.weight")?, global_scale: g("mlp.global_scale")?[0], di });
    } else {
        rw = g("mlp.gate.weight")?; rb = g("mlp.gate.bias")?; rg = g("mlp.gate.global_scale")?[0];
        let sfused = g("mlp.shared_experts.shared_w13_weight")?;
        let (a, b) = split_shared_w13(&sfused, t.n_shared_experts, mi, h, SHARED_W13_HALVED);
        sgate = a; sup = b; sdown = g("mlp.shared_experts.shared_w2_weight")?;
    }
    let hw = HostW {
        attn_norm: g("attn_norm.weight")?, mlp_norm: g("mlp_norm.weight")?,
        attn_sconv: g("attn_sconv.weight")?, mlp_sconv: g("mlp_sconv.weight")?,
        wq: g("attn.wq_du.weight")?, wk: g("attn.wk_dv.weight")?, wv: g("attn.wv_dv.weight")?,
        wr: g("attn.wr_du.weight")?, wo: g("attn.wo_ud.weight")?,
        qn: g("attn.q_norm.weight")?, kn: g("attn.k_norm.weight")?,
        ks: g("attn.k_sconv.weight")?, vs: g("attn.v_sconv.weight")?, rp: rp_loaded.data,
        rw, rb, rg, sgate, sup, sdown, experts: HashMap::new(), dense,
    };
    let geom = Geom {
        layer, n, h, mi, n_routed: t.n_routed_experts, n_shared: t.n_shared_experts, top_k: t.num_experts_per_tok,
        route_scale: t.route_scale as f32, kernel, eps: t.rms_norm_eps,
        dims: AttnDims { hidden: h, heads, kv_heads, head_dim: hd, d_rel: t.d_rel, rel_extent, kernel, rms_eps: t.rms_norm_eps, kind },
        ls: LogScaling { n_floor: t.log_scaling_n_floor as f32, alpha: t.log_scaling_alpha as f32 },
        mask: causal_mask(n, window), heads, kv_heads, hd, d_rel: t.d_rel, rel_extent, is_dense,
    };
    Ok((hw, geom))
}

/// Fetch (dequantised, host f32) exactly the routed experts named by `routing`.
pub fn load_experts(cp: &Checkpoint, g: &Geom, hw: &mut HostW, routing: &[Routing]) -> Result<usize> {
    let pfx = format!("model.llm.layers.{}.", g.layer);
    let mut touched: Vec<usize> = routing.iter().flat_map(|r| r.experts.clone()).collect();
    touched.sort_unstable(); touched.dedup();
    for &e in &touched {
        if hw.experts.contains_key(&e) { continue; }
        mem_guard(&format!("loading expert {e} of layer {}", g.layer));
        // The checkpoint INTERLEAVES gate and up rows in w13; the layer wants gate rows first,
        // then up rows (the 2026-08-08 "Paris" bug, found again here by a near-uniform user-turn loss).
        let w13 = deinterleave_fused(&cp.expert_slice(&format!("{pfx}mlp.experts.w13_weight"), e)?.data, 2 * g.mi, g.h);
        let w2 = cp.expert_slice(&format!("{pfx}mlp.experts.w2_weight"), e)?.data;
        anyhow::ensure!(w13.len() == 2 * g.mi * g.h && w2.len() == g.h * g.mi, "layer {} expert {e} shape", g.layer);
        hw.experts.insert(e, (w13, w2));
    }
    Ok(touched.len())
}

/// Bind any routed experts present on the host side that the device side lacks.
pub fn bind_experts<B: Backend>(dev: &B::Device, dw: &mut DevW<B>, hw: &HostW, g: &Geom) {
    for (&e, (w13, w2)) in &hw.experts {
        if !dw.experts.contains_key(&e) {
            mem_guard(&format!("binding expert {e} of layer {}", g.layer));
            dw.experts.insert(e, (t2(dev, w13, 2 * g.mi, g.h), t2(dev, w2, g.h, g.mi)));
        }
    }
}

pub fn build_dev<B: Backend>(dev: &B::Device, hw: &HostW, g: &Geom) -> DevW<B> {
    let (h, mi, heads, kvh, hd, kernel) = (g.h, g.mi, g.heads, g.kv_heads, g.hd, g.kernel);
    let mut experts = HashMap::new();
    for (&e, (w13, w2)) in &hw.experts { experts.insert(e, (t2(dev, w13, 2 * mi, h), t2(dev, w2, h, mi))); }
    let dense = hw.dense.as_ref().map(|d| DevDense { gate: t2(dev, &d.gate, d.di, h), up: t2(dev, &d.up, d.di, h), down: t2(dev, &d.down, h, d.di), global_scale: d.global_scale });
    let ns = g.n_shared;
    DevW {
        attn_norm: t1(dev, &hw.attn_norm), mlp_norm: t1(dev, &hw.mlp_norm),
        attn_sconv: t2(dev, &hw.attn_sconv, h, kernel), mlp_sconv: t2(dev, &hw.mlp_sconv, h, kernel),
        wq: t2(dev, &hw.wq, heads * hd, h), wk: t2(dev, &hw.wk, kvh * hd, h), wv: t2(dev, &hw.wv, kvh * hd, h),
        wr: t2(dev, &hw.wr, heads * g.d_rel, h), wo: t2(dev, &hw.wo, h, heads * hd),
        qn: t1(dev, &hw.qn), kn: t1(dev, &hw.kn), ks: t2(dev, &hw.ks, kvh * hd, kernel), vs: t2(dev, &hw.vs, kvh * hd, kernel),
        rp: t2(dev, &hw.rp, g.d_rel, g.rel_extent),
        rw: if g.is_dense { c2(dev, vec![0.0], 1, 1) } else { t2(dev, &hw.rw, g.n_routed + ns, h) },
        sgate: (0..if g.is_dense { 0 } else { ns }).map(|s| t2(dev, &hw.sgate[s * mi * h..(s + 1) * mi * h], mi, h)).collect(),
        sup: (0..if g.is_dense { 0 } else { ns }).map(|s| t2(dev, &hw.sup[s * mi * h..(s + 1) * mi * h], mi, h)).collect(),
        sdown: (0..if g.is_dense { 0 } else { ns }).map(|s| t2(dev, &hw.sdown[s * h * mi..(s + 1) * h * mi], h, mi)).collect(),
        experts, dense,
    }
}

fn head_norm<B: Backend>(t: Tensor<B, 2>, n: usize, heads: usize, hd: usize, gain: &Tensor<B, 1>, eps: f64) -> Tensor<B, 3> {
    let t3 = t.reshape([n, heads, hd]);
    let ms = t3.clone().powf_scalar(2.0).mean_dim(2);
    (t3 / ms.add_scalar(eps).sqrt()) * gain.clone().reshape([1, 1, hd])
}

/// Attention, its short conv, the residual, the MLP norm, and (MoE only) the router
/// logits. Runs with NO experts bound when choosing which experts to fetch.
pub fn pre_moe<B: Backend>(dev: &B::Device, w: &DevW<B>, g: &Geom, x: Tensor<B, 2>) -> (Tensor<B, 2>, Tensor<B, 2>, Option<Tensor<B, 2>>) {
    let (n, h) = (g.n, g.h);
    let d = &g.dims;
    let (heads, kvh, hd) = (d.heads, d.kv_heads, d.head_dim);
    let hn = b_rms_norm(x.clone(), w.attn_norm.clone(), g.eps);
    let q = b_linear(hn.clone(), w.wq.clone());
    let k = b_short_conv(b_linear(hn.clone(), w.wk.clone()), w.ks.clone());
    let v = b_short_conv(b_linear(hn.clone(), w.wv.clone()), w.vs.clone());
    let r = b_linear(hn, w.wr.clone());
    let taus: Vec<f32> = (0..n).map(|t| match d.kind { AttnKind::Global => g.ls.tau(t), _ => 1.0 }).collect();
    let tau_col = c2(dev, taus, n, 1);
    let q3 = head_norm(q, n, heads, hd, &w.qn, g.eps) * tau_col.clone().reshape([n, 1, 1]);
    let k3 = head_norm(k, n, kvh, hd, &w.kn, g.eps);
    let v3 = v.reshape([n, kvh, hd]);
    let rel = r.reshape([n * heads, d.d_rel]).matmul(w.rp.clone()).reshape([n, heads, d.rel_extent]);
    let mut onehot = vec![0f32; n * d.rel_extent * n];
    for qi in 0..n { for ki in 0..n { let dist = qi as isize - ki as isize; if dist >= 0 && (dist as usize) < d.rel_extent { onehot[(qi * d.rel_extent + dist as usize) * n + ki] = 1.0; } } }
    let oh = Tensor::<B, 3>::from_data(TensorData::new(onehot, [n, d.rel_extent, n]), dev);
    let bias = (rel.matmul(oh) * tau_col.reshape([n, 1, 1])).swap_dims(0, 1); // [heads, qi, ki]
    let groups = d.groups();
    let kv_idx: Vec<i64> = (0..heads).map(|hh| (hh / groups) as i64).collect();
    let qh = q3.swap_dims(0, 1);
    let kh = k3.swap_dims(0, 1).select(0, ints(dev, kv_idx.clone()));
    let vh = v3.swap_dims(0, 1).select(0, ints(dev, kv_idx));
    let mask = Tensor::<B, 3>::from_data(TensorData::new(g.mask.clone(), [1, n, n]), dev);
    let scores = qh.matmul(kh.swap_dims(1, 2)).mul_scalar(d.scaling()) + bias + mask;
    let p = activation::softmax(scores, 2);
    let o = p.matmul(vh).swap_dims(0, 1).reshape([n, heads * hd]);
    let a = b_short_conv(b_linear(o, w.wo.clone()), w.attn_sconv.clone());
    let x1 = x + a;
    let hn2 = b_rms_norm(x1.clone(), w.mlp_norm.clone(), g.eps);
    let logits = if g.is_dense { None } else { Some(b_linear(hn2.clone(), w.rw.clone())) };
    (x1, hn2, logits)
}

/// Expert selection from the model's OWN logits: sigmoid + bias, top-k, ties by index.
pub fn select_routing(logits: &[f32], rb: &[f32], g: &Geom) -> Vec<Routing> {
    let rows = g.n_routed + g.n_shared;
    (0..g.n).map(|tk| {
        let lt = &logits[tk * rows..(tk + 1) * rows];
        let score = |e: usize| { let v = lt[e]; (if v >= 0.0 { 1.0 / (1.0 + (-v).exp()) } else { let q = v.exp(); q / (1.0 + q) }) + rb[e] };
        let mut order: Vec<usize> = (0..g.n_routed).collect();
        order.sort_by(|&a, &b| score(b).partial_cmp(&score(a)).unwrap().then(a.cmp(&b)));
        Routing { experts: order[..g.top_k].to_vec(), weights: vec![], shared_gammas: vec![] }
    }).collect()
}

/// The whole layer. `routing` is ignored on dense layers.
pub fn burn_layer<B: Backend>(dev: &B::Device, w: &DevW<B>, g: &Geom, x: Tensor<B, 2>, routing: &[Routing]) -> Tensor<B, 2> {
    let (n, h) = (g.n, g.h);
    let (x1, hn2, logits) = pre_moe(dev, w, g, x);
    let mlp_out: Tensor<B, 2> = if let Some(dd) = &w.dense {
        let gate = b_linear(hn2.clone(), dd.gate.clone());
        let up = b_linear(hn2, dd.up.clone());
        b_linear(b_silu(gate) * up, dd.down.clone()).mul_scalar(dd.global_scale)
    } else {
        let logits = logits.expect("MoE layer has router logits");
        let rows = g.n_routed + g.n_shared;
        let mut outs: Vec<Tensor<B, 2>> = Vec::with_capacity(n);
        for t in 0..n {
            let rt = &routing[t];
            let xt = hn2.clone().slice([t..t + 1, 0..h]);
            let lt = logits.clone().slice([t..t + 1, 0..rows]);
            let mut idx: Vec<i64> = rt.experts.iter().map(|&e| e as i64).collect();
            idx.extend((g.n_routed..rows).map(|e| e as i64));
            let sel = lt.select(1, ints(dev, idx));
            let wts = activation::softmax(activation::log_sigmoid(sel), 1).mul_scalar(g.route_scale * router_global_scale());
            let mut acc: Option<Tensor<B, 2>> = None;
            for (slot, &e) in rt.experts.iter().enumerate() {
                let (w13, w2) = w.experts.get(&e).unwrap_or_else(|| panic!("layer {} expert {e} not bound", g.layer));
                let both = b_linear(xt.clone(), w13.clone());
                let gate = both.clone().slice([0..1, 0..g.mi]);
                let up = both.slice([0..1, g.mi..2 * g.mi]);
                let c = b_linear(b_silu(gate) * up, w2.clone()) * wts.clone().slice([0..1, slot..slot + 1]);
                acc = Some(match acc { None => c, Some(a0) => a0 + c });
            }
            for s in 0..g.n_shared {
                let gs = b_linear(xt.clone(), w.sgate[s].clone());
                let us = b_linear(xt.clone(), w.sup[s].clone());
                let gamma = wts.clone().slice([0..1, g.top_k + s..g.top_k + s + 1]);
                let c = b_linear(b_silu(gs) * us * gamma, w.sdown[s].clone());
                acc = Some(match acc { None => c, Some(a0) => a0 + c });
            }
            outs.push(acc.expect("at least one expert"));
        }
        Tensor::cat(outs, 0)
    };
    x1 + b_short_conv(mlp_out, w.mlp_sconv.clone())
}

// the router's global scale is a host scalar carried on the layer's Geom-side host weights;
// threaded through a thread-local so the graph code stays free of it
thread_local! { static RG: std::cell::Cell<f32> = const { std::cell::Cell::new(1.0) }; }
pub fn set_router_global_scale(v: f32) { RG.with(|c| c.set(v)); }
fn router_global_scale() -> f32 { RG.with(|c| c.get()) }

// ------------------------------------------------------------------ stack ends
pub struct HostHead { pub embed: Vec<f32>, pub embed_norm: Vec<f32>, pub fnorm: Vec<f32>, pub unembed: Vec<f32>, pub vocab: usize, pub unpadded: usize, pub mup: f32 }

pub fn load_head(cp: &Checkpoint, t: &InklingTextConfig) -> Result<HostHead> {
    Ok(HostHead {
        embed: cp.tensor("model.llm.embed.weight")?.data,
        embed_norm: cp.tensor("model.llm.embed_norm.weight")?.data,
        fnorm: cp.tensor("model.llm.norm.weight")?.data,
        unembed: cp.tensor("model.llm.unembed.weight")?.data,
        vocab: t.vocab_size, unpadded: t.unpadded_vocab_size, mup: t.logits_mup_width_multiplier as f32,
    })
}

/// Embedding + embed_norm on the host (the table is frozen in this lane).
pub fn embed_host(hh: &HostHead, ids: &[usize], t: &InklingTextConfig) -> Vec<f32> {
    let h = t.hidden_size;
    let raw = crate::models::inkling::stack::embed(ids, &hh.embed, hh.vocab, h);
    if t.use_embed_norm { crate::models::inkling::block::rms_norm(&raw, &hh.embed_norm, t.rms_norm_eps, ids.len(), h) } else { raw }
}

/// Next-token cross-entropy through the real head, under autodiff. Returns
/// (mean NLL over the T targets, per-token NLLs, dL/d hidden).
pub fn head_loss(dev: &Dev, hh: &HostHead, hidden: &[f32], targets: &[usize], n: usize, h: usize, eps: f64) -> (f32, Vec<f32>, Vec<f32>) {
    let x = c2::<Ad>(dev, hidden.to_vec(), n, h).require_grad();
    let fnorm = Tensor::<Ad, 1>::from_data(TensorData::new(hh.fnorm.clone(), [h]), dev);
    let unembed = c2::<Ad>(dev, hh.unembed[..hh.unpadded * h].to_vec(), hh.unpadded, h);
    let normed = b_rms_norm(x.clone(), fnorm, eps).div_scalar(hh.mup);
    let logits = b_linear(normed, unembed); // [n, unpadded]
    let logp = activation::log_softmax(logits, 1);
    let tgt = Tensor::<Ad, 2, Int>::from_data(TensorData::new(targets.iter().map(|&v| v as i64).collect::<Vec<_>>(), [n, 1]), dev);
    let picked = logp.gather(1, tgt).reshape([n]); // [n]
    let per_token: Vec<f32> = picked.clone().neg().into_data().to_vec::<f32>().unwrap();
    let loss = picked.mean().neg();
    let l = loss.clone().into_data().to_vec::<f32>().unwrap()[0];
    let grads = loss.backward();
    let gx = x.grad(&grads).expect("hidden grad").into_data().to_vec::<f32>().unwrap();
    (l, per_token, gx)
}

pub fn grad_norm<const D: usize>(t: &Tensor<Ad, D>, grads: &<Ad as burn::tensor::backend::AutodiffBackend>::Gradients) -> f32 {
    t.grad(grads).map(|g| g.powf_scalar(2.0).sum().into_data().to_vec::<f32>().unwrap()[0].sqrt()).unwrap_or(0.0)
}
