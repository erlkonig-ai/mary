//! S1 of the Inkling training path (compass 84605490): ONE decoder layer, run
//! differentiably under `Autodiff<Cuda<f32>>` on dequantised weights.
//!
//! There is deliberately NO host reference here and NO forward-parity gate. The
//! host f32 lane was deleted in 694e05b as a moth's flame, and JP caught this
//! probe resurrecting it from git history within a day. The only check a
//! transcription needs is SELF-consistency: perturb one weight, rerun the same
//! Burn forward, and compare central differences against autodiff. That catches
//! the real failure of a transcription -- a detached path, a slice that drops a
//! gradient -- without a second implementation of anything. Whether the layer is
//! numerically the device lane's twin is not the question; whether the STACK
//! still learns and still reads as her is, and that gate lives at stage 2.
//!
//! Routing is DISCRETE and comes from the model's own device logits: sigmoid +
//! bias, top-k on the host over those numbers. The routing WEIGHTS (softmax over
//! log-sigmoid of the selected logits) stay in the graph so gradient reaches the
//! router matrix.
//!
//! Framing: Inkling-Small NVFP4 safetensors checkpoint, one MoE layer, N tokens
//! of N(0,1) residual-stream input, loss = mean squared error to a fixed random
//! target (a placeholder for gradient plumbing only). Central differences on the
//! f32 GPU forward, eps = 1e-2*|w| + 1e-4, on the three largest-|grad|
//! coordinates per weight class, relative budget 5e-2.
//!
//!   INK_CKPT=~/models/thinkingmachines-inkling-small-nvfp4 \
//!   INK_PROBE_LAYER=<first MoE layer> INK_PROBE_TOKENS=8 \
//!   cargo run --release --features inkling-cuda --bin inkling_train_probe
use anyhow::{Context, Result};
use burn::backend::{Autodiff, Cuda};
use burn::prelude::*;
use burn::tensor::activation;
use mary::models::inkling::attn::{AttnDims, LogScaling, causal_mask};
use mary::models::inkling::block::Routing;
use mary::models::inkling::burn::{linear as b_linear, rms_norm as b_rms_norm, short_conv as b_short_conv, silu as b_silu};
use mary::models::inkling::config::{AttnKind, InklingConfig};
use mary::models::inkling::load::{Checkpoint, split_shared_w13};
use std::collections::HashMap;

type Bk = Cuda<f32>;
type Ad = Autodiff<Bk>;
type T2 = Tensor<Ad, 2>;
type T1 = Tensor<Ad, 1>;

/// The settled reading of `shared_w13_weight` (see inkling_real_gate): INTERLEAVED.
const SHARED_W13_HALVED: bool = false;
const FD_BUDGET: f32 = 5e-2;

/// Host-side handles to the f32 weight vectors the leaves were built from, so a
/// finite-difference step can rebuild ONE leaf with one coordinate moved.
struct HostW {
    attn_norm: Vec<f32>, mlp_norm: Vec<f32>, attn_sconv: Vec<f32>, mlp_sconv: Vec<f32>,
    wq: Vec<f32>, wk: Vec<f32>, wv: Vec<f32>, wr: Vec<f32>, wo: Vec<f32>,
    qn: Vec<f32>, kn: Vec<f32>, ks: Vec<f32>, vs: Vec<f32>, rp: Vec<f32>,
    rw: Vec<f32>, rb: Vec<f32>, rg: f32,
    sgate: Vec<f32>, sup: Vec<f32>, sdown: Vec<f32>,
    experts: HashMap<usize, (Vec<f32>, Vec<f32>)>,
}

struct Geom {
    n: usize, h: usize, mi: usize, n_routed: usize, n_shared: usize, top_k: usize,
    route_scale: f32, kernel: usize, eps: f64, dims: AttnDims, ls: LogScaling, mask: Vec<f32>,
}

fn mse(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(p, q)| (p - q) * (p - q)).sum::<f32>() / a.len() as f32
}

// ------------------------------------------------------------------ Burn side
struct DevW {
    attn_norm: T1, mlp_norm: T1, attn_sconv: T2, mlp_sconv: T2,
    wq: T2, wk: T2, wv: T2, wr: T2, wo: T2, qn: T1, kn: T1, ks: T2, vs: T2, rp: T2,
    rw: T2, sgate: Vec<T2>, sup: Vec<T2>, sdown: Vec<T2>,
    experts: HashMap<usize, (T2, T2)>,
}

fn t2(dev: &burn::backend::cuda::CudaDevice, v: &[f32], r: usize, c: usize) -> T2 {
    assert_eq!(v.len(), r * c, "t2: {} != {r}x{c}", v.len());
    Tensor::<Ad, 2>::from_data(TensorData::new(v.to_vec(), [r, c]), dev).require_grad()
}
fn t1(dev: &burn::backend::cuda::CudaDevice, v: &[f32]) -> T1 {
    Tensor::<Ad, 1>::from_data(TensorData::new(v.to_vec(), [v.len()]), dev).require_grad()
}
fn c2(dev: &burn::backend::cuda::CudaDevice, v: Vec<f32>, r: usize, c: usize) -> T2 {
    Tensor::<Ad, 2>::from_data(TensorData::new(v, [r, c]), dev)
}
fn ints(dev: &burn::backend::cuda::CudaDevice, v: Vec<i64>) -> Tensor<Ad, 1, Int> {
    let n = v.len();
    Tensor::<Ad, 1, Int>::from_data(TensorData::new(v, [n]), dev)
}

fn head_norm(t: T2, n: usize, heads: usize, hd: usize, gain: &T1, eps: f64) -> Tensor<Ad, 3> {
    let t3 = t.reshape([n, heads, hd]);
    let ms = t3.clone().powf_scalar(2.0).mean_dim(2);
    let normed = t3 / ms.add_scalar(eps).sqrt();
    normed * gain.clone().reshape([1, 1, hd])
}

/// Everything before the experts: attention, its short conv, the residual, the MLP norm and the
/// router logits. Runs with NO experts bound when choosing which experts to fetch.
fn pre_moe(dev: &burn::backend::cuda::CudaDevice, w: &DevW, g: &Geom, x: T2) -> (T2, T2, T2) {
    let (n, h) = (g.n, g.h);
    let d = &g.dims;
    let (heads, kvh, hd) = (d.heads, d.kv_heads, d.head_dim);
    let hn = b_rms_norm(x.clone(), w.attn_norm.clone(), g.eps);
    // --- attention (mirrors attention_prefill) ---
    let q = b_linear(hn.clone(), w.wq.clone());
    let k = b_short_conv(b_linear(hn.clone(), w.wk.clone()), w.ks.clone());
    let v = b_short_conv(b_linear(hn.clone(), w.wv.clone()), w.vs.clone());
    let r = b_linear(hn.clone(), w.wr.clone());
    let taus: Vec<f32> = (0..n).map(|t| match d.kind { AttnKind::Global => g.ls.tau(t), _ => 1.0 }).collect();
    let tau_col = c2(dev, taus.clone(), n, 1); // [n,1]
    let q3 = head_norm(q, n, heads, hd, &w.qn, g.eps) * tau_col.clone().reshape([n, 1, 1]);
    let k3 = head_norm(k, n, kvh, hd, &w.kn, g.eps);
    let v3 = v.reshape([n, kvh, hd]);
    // relative-position bias: rel[qi,h,e] = (r @ rel_proj); bias[qi,h,ki] = rel[qi,h,qi-ki] * tau[qi]
    let rel = r.reshape([n * heads, d.d_rel]).matmul(w.rp.clone()).reshape([n, heads, d.rel_extent]);
    let mut onehot = vec![0f32; n * d.rel_extent * n];
    for qi in 0..n { for ki in 0..n { let dist = qi as isize - ki as isize; if dist >= 0 && (dist as usize) < d.rel_extent { onehot[(qi * d.rel_extent + dist as usize) * n + ki] = 1.0; } } }
    let oh = Tensor::<Ad, 3>::from_data(TensorData::new(onehot, [n, d.rel_extent, n]), dev);
    let bias = rel.matmul(oh) * tau_col.reshape([n, 1, 1]); // [n(qi), heads, n(ki)]
    let bias = bias.swap_dims(0, 1); // [heads, qi, ki]
    let groups = d.groups();
    let kv_idx: Vec<i64> = (0..heads).map(|hh| (hh / groups) as i64).collect();
    let qh = q3.swap_dims(0, 1); // [heads, n, hd]
    let kh = k3.swap_dims(0, 1).select(0, ints(dev, kv_idx.clone())); // [heads, n, hd]
    let vh = v3.swap_dims(0, 1).select(0, ints(dev, kv_idx)); // [heads, n, hd]
    let mask = Tensor::<Ad, 3>::from_data(TensorData::new(g.mask.clone(), [1, n, n]), dev);
    let scores = qh.matmul(kh.swap_dims(1, 2)).mul_scalar(d.scaling()) + bias + mask;
    let p = activation::softmax(scores, 2);
    let o = p.matmul(vh).swap_dims(0, 1).reshape([n, heads * hd]);
    let a = b_linear(o, w.wo.clone());
    let a = b_short_conv(a, w.attn_sconv.clone());
    let x1 = x + a;
    let hn2 = b_rms_norm(x1.clone(), w.mlp_norm.clone(), g.eps);
    let logits = b_linear(hn2.clone(), w.rw.clone()); // [n, rows]
    (x1, hn2, logits)
}

fn burn_layer(dev: &burn::backend::cuda::CudaDevice, w: &DevW, g: &Geom, x: T2, routing: &[Routing]) -> T2 {
    let n = g.n;
    let h = g.h;
    let (x1, hn2, logits) = pre_moe(dev, w, g, x);
    let rows = g.n_routed + g.n_shared;
    let mut outs: Vec<T2> = Vec::with_capacity(n);
    for t in 0..n {
        let rt = &routing[t];
        let xt = hn2.clone().slice([t..t + 1, 0..h]);
        let lt = logits.clone().slice([t..t + 1, 0..rows]);
        let mut idx: Vec<i64> = rt.experts.iter().map(|&e| e as i64).collect();
        idx.extend((g.n_routed..rows).map(|e| e as i64));
        let sel = lt.select(1, ints(dev, idx)); // [1, k+ns]
        let wts = activation::softmax(activation::log_sigmoid(sel), 1).mul_scalar(g.route_scale * w_rg(g)); // [1, k+ns]
        let mut acc: Option<T2> = None;
        for (slot, &e) in rt.experts.iter().enumerate() {
            let (w13, w2) = w.experts.get(&e).expect("touched expert on device");
            let both = b_linear(xt.clone(), w13.clone()); // [1, 2mi]
            let gate = both.clone().slice([0..1, 0..g.mi]);
            let up = both.slice([0..1, g.mi..2 * g.mi]);
            let act = b_silu(gate) * up;
            let c = b_linear(act, w2.clone()) * wts.clone().slice([0..1, slot..slot + 1]);
            acc = Some(match acc { None => c, Some(a0) => a0 + c });
        }
        for s in 0..g.n_shared {
            let gs = b_linear(xt.clone(), w.sgate[s].clone());
            let us = b_linear(xt.clone(), w.sup[s].clone());
            let gamma = wts.clone().slice([0..1, g.top_k + s..g.top_k + s + 1]);
            let act = b_silu(gs) * us * gamma;
            let c = b_linear(act, w.sdown[s].clone());
            acc = Some(match acc { None => c, Some(a0) => a0 + c });
        }
        outs.push(acc.expect("at least one expert"));
    }
    let moe = Tensor::cat(outs, 0); // [n, h]
    let m = b_short_conv(moe, w.mlp_sconv.clone());
    x1 + m
}
// the router global scale is a host scalar; kept out of the graph like the host does
static mut RG: f32 = 1.0;
fn w_rg(_g: &Geom) -> f32 { unsafe { RG } }


fn slot_mut<'a>(hw: &'a mut HostW, name: &str, e0: usize) -> &'a mut Vec<f32> {
    match name {
        "attn.wq" => &mut hw.wq, "attn.wk" => &mut hw.wk, "attn.wv" => &mut hw.wv, "attn.wo" => &mut hw.wo, "attn.k_sconv" => &mut hw.ks, "attn.rel_proj" => &mut hw.rp,
        "attn_sconv" => &mut hw.attn_sconv, "attn_norm" => &mut hw.attn_norm, "router" => &mut hw.rw,
        "shared.gate[0]" => &mut hw.sgate, "shared.down[0]" => &mut hw.sdown,
        "expert.w13" => &mut hw.experts.get_mut(&e0).unwrap().0, "expert.w2" => &mut hw.experts.get_mut(&e0).unwrap().1,
        _ => unreachable!(),
    }
}

/// Burn tensors are reference-counted: cloning a DevW shares every device buffer.
fn clone_dev(w: &DevW) -> DevW {
    DevW {
        attn_norm: w.attn_norm.clone(), mlp_norm: w.mlp_norm.clone(), attn_sconv: w.attn_sconv.clone(), mlp_sconv: w.mlp_sconv.clone(),
        wq: w.wq.clone(), wk: w.wk.clone(), wv: w.wv.clone(), wr: w.wr.clone(), wo: w.wo.clone(),
        qn: w.qn.clone(), kn: w.kn.clone(), ks: w.ks.clone(), vs: w.vs.clone(), rp: w.rp.clone(), rw: w.rw.clone(),
        sgate: w.sgate.clone(), sup: w.sup.clone(), sdown: w.sdown.clone(),
        experts: w.experts.iter().map(|(k, (a, b))| (*k, (a.clone(), b.clone()))).collect(),
    }
}

/// Replace exactly ONE leaf of a (shared) DevW with a fresh upload of the host vector.
/// This is the whole fix for the 2026-09-03 incident: the finite-difference loop
/// used to rebuild every leaf (~5 GB) per evaluation and pushed the box into the
/// unified-memory thrash that killed sshd. Now an evaluation uploads tens of MB.
fn with_leaf(dev: &burn::backend::cuda::CudaDevice, base: &DevW, hw: &HostW, name: &str, e0: usize, mi: usize, h: usize, heads: usize, kvh: usize, hd: usize, kernel: usize, d_rel: usize, rel_extent: usize) -> DevW {
    let mut d = clone_dev(base);
    match name {
        "attn.wq" => d.wq = t2(dev, &hw.wq, heads * hd, h),
        "attn.wk" => d.wk = t2(dev, &hw.wk, kvh * hd, h),
        "attn.wv" => d.wv = t2(dev, &hw.wv, kvh * hd, h),
        "attn.wo" => d.wo = t2(dev, &hw.wo, h, heads * hd),
        "attn.k_sconv" => d.ks = t2(dev, &hw.ks, kvh * hd, kernel),
        "attn.rel_proj" => d.rp = t2(dev, &hw.rp, d_rel, rel_extent),
        "attn_sconv" => d.attn_sconv = t2(dev, &hw.attn_sconv, h, kernel),
        "attn_norm" => d.attn_norm = t1(dev, &hw.attn_norm),
        "router" => d.rw = t2(dev, &hw.rw, hw.rb.len() + d.sgate.len(), h),
        "shared.gate[0]" => d.sgate[0] = t2(dev, &hw.sgate[0..mi * h], mi, h),
        "shared.down[0]" => d.sdown[0] = t2(dev, &hw.sdown[0..h * mi], h, mi),
        "expert.w13" => { let (w13, _) = hw.experts.get(&e0).unwrap(); d.experts.get_mut(&e0).unwrap().0 = t2(dev, w13, 2 * mi, h); }
        "expert.w2" => { let (_, w2) = hw.experts.get(&e0).unwrap(); d.experts.get_mut(&e0).unwrap().1 = t2(dev, w2, h, mi); }
        _ => unreachable!(),
    }
    d
}

/// Host MemAvailable in GiB, from /proc/meminfo. The GB10 is unified memory: this
/// number is the one that matters for the GPU pool too.
fn mem_available_gib() -> f64 {
    let s = std::fs::read_to_string("/proc/meminfo").unwrap_or_default();
    for line in s.lines() {
        if let Some(rest) = line.strip_prefix("MemAvailable:") {
            let kb: f64 = rest.trim().trim_end_matches("kB").trim().parse().unwrap_or(0.0);
            return kb / (1024.0 * 1024.0);
        }
    }
    0.0
}
const MEM_FLOOR_GIB: f64 = 28.0;
fn mem_guard(where_: &str) {
    let a = mem_available_gib();
    if a < MEM_FLOOR_GIB {
        eprintln!("MEMORY GUARD: MemAvailable {a:.1} GiB < floor {MEM_FLOOR_GIB} GiB at {where_} -- aborting before the box does");
        std::process::exit(3);
    }
}

/// Finite-difference check of a scalar function of ONE small leaf, on the device.
fn fd_check<F>(dev: &burn::backend::cuda::CudaDevice, label: &str, x0: &[f32], shape: [usize; 2], f: F) -> bool
where F: Fn(T2) -> Tensor<Ad, 1> {
    let x = c2(dev, x0.to_vec(), shape[0], shape[1]).require_grad();
    let l = f(x.clone()); let l0 = l.clone().into_data().to_vec::<f32>().unwrap()[0];
    let g = x.grad(&l.backward()).unwrap().into_data().to_vec::<f32>().unwrap();
    let gn: f32 = g.iter().map(|v| v * v).sum::<f32>().sqrt();
    let eps = 0.01 * l0.abs().max(1e-3) / gn.max(1e-12);
    let ev = |sign: f32| { let v: Vec<f32> = x0.iter().zip(&g).map(|(a, b)| a + sign * eps * b / gn).collect(); f(c2(dev, v, shape[0], shape[1])).into_data().to_vec::<f32>().unwrap()[0] };
    let dd = (ev(1.0) - ev(-1.0)) / (2.0 * eps); let rel = (dd - gn).abs() / gn.max(1e-12);
    println!("    unit {label:>28}: |g| {gn:.4e}  dd {dd:.4e}  rel {rel:.2e}  {}", if rel < 2e-2 { "ok" } else { "FAIL" });
    rel < 2e-2
}

fn unit_checks(dev: &burn::backend::cuda::CudaDevice) {
    let mut st = 0xC0FFEEu64; let mut rnd = |n: usize| -> Vec<f32> { (0..n).map(|_| xorshift(&mut st)).collect() };
    let (n, d, k, heads, hd) = (8usize, 32usize, 4usize, 4usize, 8usize);
    let tgt = c2(dev, rnd(n * d), n, d);
    let w_sc = c2(dev, rnd(d * k), d, k);
    let gain = Tensor::<Ad, 1>::from_data(TensorData::new(rnd(d), [d]), dev);
    let ghd = Tensor::<Ad, 1>::from_data(TensorData::new(rnd(hd), [hd]), dev);
    let x0 = rnd(n * d);
    println!("  unit-level input-gradient checks (tiny tensors, same ops as the layer):");
    fd_check(dev, "short_conv input", &x0, [n, d], |x| (b_short_conv(x, w_sc.clone()) - tgt.clone()).powf_scalar(2.0).mean());
    fd_check(dev, "rms_norm input", &x0, [n, d], |x| (b_rms_norm(x, gain.clone(), 1e-6) - tgt.clone()).powf_scalar(2.0).mean());
    fd_check(dev, "head_norm input", &x0, [n, d], |x| (head_norm(x, n, heads, hd, &ghd, 1e-6).reshape([n, d]) - tgt.clone()).powf_scalar(2.0).mean());
    let wq = c2(dev, rnd(d * d), d, d); let wk = c2(dev, rnd(d * d), d, d); let wv = c2(dev, rnd(d * d), d, d);
    let mask = { let mut m = vec![0f32; n * n]; for i in 0..n { for j in (i + 1)..n { m[i * n + j] = f32::NEG_INFINITY; } } Tensor::<Ad, 3>::from_data(TensorData::new(m, [1, n, n]), dev) };
    let attn = |x: T2| -> T2 {
        let q = b_linear(x.clone(), wq.clone()).reshape([n, heads, hd]).swap_dims(0, 1);
        let kk = b_linear(x.clone(), wk.clone()).reshape([n, heads, hd]).swap_dims(0, 1);
        let v = b_linear(x, wv.clone()).reshape([n, heads, hd]).swap_dims(0, 1);
        let sc = q.matmul(kk.swap_dims(1, 2)).mul_scalar(0.125) + mask.clone();
        activation::softmax(sc, 2).matmul(v).swap_dims(0, 1).reshape([n, d])
    };
    fd_check(dev, "attention block input", &x0, [n, d], |x| (attn(x) - tgt.clone()).powf_scalar(2.0).mean());
    let idx = ints(dev, vec![0, 0, 1, 1]);
    let tgt64 = c2(dev, rnd(n * 64), n, 64);
    fd_check(dev, "select(0) repeated idx + swap_dims", &x0, [n, d], |x| (x.reshape([n, 2, 16]).swap_dims(0, 1).select(0, idx.clone()).swap_dims(0, 1).reshape([n, 64]) * tgt64.clone()).powf_scalar(2.0).mean());
    fd_check(dev, "matmul-left input", &x0, [n, d], |x| (x.matmul(wq.clone().transpose()) - tgt.clone()).powf_scalar(2.0).mean());
    fd_check(dev, "matmul-weight 32 (as leaf)", &rnd(d * d), [d, d], |w| (b_linear(c2(dev, x0.clone(), n, d), w) - tgt.clone()).powf_scalar(2.0).mean());
    // the two suspects are 4096x4096 leaves: same op at 512 and 4096, both orientations
    for &big in &[512usize, 4096usize] {
        let xb = c2(dev, rnd(n * big).iter().map(|v| v * 0.05).collect(), n, big);
        let tb = c2(dev, rnd(n * big), n, big);
        let w0: Vec<f32> = rnd(big * big).iter().map(|v| v * 0.02).collect();
        fd_check(dev, &format!("matmul-weight {big} x@w^T"), &w0, [big, big], |w| (b_linear(xb.clone(), w) - tb.clone()).powf_scalar(2.0).mean());
        fd_check(dev, &format!("matmul-weight {big} x@w"), &w0, [big, big], |w| (xb.clone().matmul(w) - tb.clone()).powf_scalar(2.0).mean());
    }
}

fn xorshift(state: &mut u64) -> f32 {
    // Box-Muller on xorshift64* -- deterministic N(0,1)
    let mut next = || { *state ^= *state << 13; *state ^= *state >> 7; *state ^= *state << 17; (*state >> 11) as f64 / (1u64 << 53) as f64 };
    let (u1, u2) = (next().max(1e-12), next());
    ((-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()) as f32
}

fn main() -> Result<()> {
    let ckpt = std::env::var("INK_CKPT").unwrap_or_else(|_| format!("{}/models/thinkingmachines-inkling-small-nvfp4", std::env::var("HOME").unwrap()));
    let cp = Checkpoint::open(&ckpt).with_context(|| format!("open {ckpt}"))?;
    let cfg_text = std::fs::read_to_string(std::path::Path::new(&ckpt).join("config.json"))?;
    let t = InklingConfig::from_json(&cfg_text)?.text_config;
    let layer: usize = std::env::var("INK_PROBE_LAYER").ok().and_then(|s| s.parse().ok()).unwrap_or(t.dense_mlp_idx);
    anyhow::ensure!(!t.is_dense(layer), "layer {layer} is dense; the probe wants a MoE layer (first is {})", t.dense_mlp_idx);
    let n: usize = std::env::var("INK_PROBE_TOKENS").ok().and_then(|s| s.parse().ok()).unwrap_or(8);
    let h = t.hidden_size;
    let kind = t.attn_kind(layer);
    let (heads, kv_heads, hd) = t.heads(kind);
    let mi = t.intermediate_size;
    let (n_routed, n_shared, top_k) = (t.n_routed_experts, t.n_shared_experts, t.num_experts_per_tok);
    let pfx = format!("model.llm.layers.{layer}.");
    let g = |nm: &str| -> Result<Vec<f32>> { Ok(cp.tensor(&format!("{pfx}{nm}"))?.data) };
    println!("=== inkling_train_probe: layer {layer} ({:?}), hidden {h}, heads {heads}/{kv_heads}x{hd}, moe inter {mi}, routed {n_routed} top-{top_k}, shared {n_shared}, tokens {n} ===", kind);

    // ---- host weights (the real gate's recipe) ----
    let tm0 = std::time::Instant::now();
    let rg = g("mlp.gate.global_scale")?[0];
    unsafe { RG = rg; }
    let rp_loaded = cp.tensor(&format!("{pfx}attn.rel_logits_proj.proj"))?;
    anyhow::ensure!(rp_loaded.shape.len() == 2 && rp_loaded.shape[0] == t.d_rel, "rel_proj shape {:?}", rp_loaded.shape);
    let rel_extent = rp_loaded.shape[1];
    println!("  rel_proj is [{}, {rel_extent}] (config rel_extent {}, window {})", t.d_rel, t.rel_extent, t.sliding_window_size);
    let sfused = g("mlp.shared_experts.shared_w13_weight")?;
    let (sgate, sup) = split_shared_w13(&sfused, n_shared, mi, h, SHARED_W13_HALVED);
    drop(sfused);
    let mut hw = HostW {
        attn_norm: g("attn_norm.weight")?, mlp_norm: g("mlp_norm.weight")?,
        attn_sconv: g("attn_sconv.weight")?, mlp_sconv: g("mlp_sconv.weight")?,
        wq: g("attn.wq_du.weight")?, wk: g("attn.wk_dv.weight")?, wv: g("attn.wv_dv.weight")?,
        wr: g("attn.wr_du.weight")?, wo: g("attn.wo_ud.weight")?,
        qn: g("attn.q_norm.weight")?, kn: g("attn.k_norm.weight")?,
        ks: g("attn.k_sconv.weight")?, vs: g("attn.v_sconv.weight")?, rp: rp_loaded.data.clone(),
        rw: g("mlp.gate.weight")?, rb: g("mlp.gate.bias")?, rg,
        sgate, sup, sdown: g("mlp.shared_experts.shared_w2_weight")?,
        experts: HashMap::new(),
    };
    // ---- input, target ----
    let mut st = 0x9E3779B97F4A7C15u64;
    let x: Vec<f32> = (0..n * h).map(|_| xorshift(&mut st)).collect();
    let target: Vec<f32> = (0..n * h).map(|_| xorshift(&mut st)).collect();
    // ---- routing from the model's own device logits, then fetch exactly the touched experts ----
    // (attention half on the device on the ROUTED-EXPERT-FREE weights is not possible before the
    //  experts exist, so this pass runs the pre-MoE half once with no experts bound.)
    let dev = burn::backend::cuda::CudaDevice::default();
    let kernel = t.sconv_kernel_size;
    let window = match kind { AttnKind::Local => Some(t.sliding_window_size), AttnKind::Global => None };
    let geom = Geom {
        n, h, mi, n_routed, n_shared, top_k, route_scale: t.route_scale as f32, kernel, eps: t.rms_norm_eps,
        dims: AttnDims { hidden: h, heads, kv_heads, head_dim: hd, d_rel: t.d_rel, rel_extent, kernel, rms_eps: t.rms_norm_eps, kind },
        ls: LogScaling { n_floor: t.log_scaling_n_floor as f32, alpha: t.log_scaling_alpha as f32 },
        mask: causal_mask(n, window),
    };
    let build_dev = |hw: &HostW| -> DevW {
        let mut experts = HashMap::new();
        for (&e, (w13, w2)) in &hw.experts { experts.insert(e, (t2(&dev, w13, 2 * mi, h), t2(&dev, w2, h, mi))); }
        DevW {
            attn_norm: t1(&dev, &hw.attn_norm), mlp_norm: t1(&dev, &hw.mlp_norm),
            attn_sconv: t2(&dev, &hw.attn_sconv, h, kernel), mlp_sconv: t2(&dev, &hw.mlp_sconv, h, kernel),
            wq: t2(&dev, &hw.wq, heads * hd, h), wk: t2(&dev, &hw.wk, kv_heads * hd, h), wv: t2(&dev, &hw.wv, kv_heads * hd, h),
            wr: t2(&dev, &hw.wr, heads * t.d_rel, h), wo: t2(&dev, &hw.wo, h, heads * hd),
            qn: t1(&dev, &hw.qn), kn: t1(&dev, &hw.kn), ks: t2(&dev, &hw.ks, kv_heads * hd, kernel), vs: t2(&dev, &hw.vs, kv_heads * hd, kernel),
            rp: t2(&dev, &hw.rp, t.d_rel, rel_extent), rw: t2(&dev, &hw.rw, n_routed + n_shared, h),
            sgate: (0..n_shared).map(|s| t2(&dev, &hw.sgate[s * mi * h..(s + 1) * mi * h], mi, h)).collect(),
            sup: (0..n_shared).map(|s| t2(&dev, &hw.sup[s * mi * h..(s + 1) * mi * h], mi, h)).collect(),
            sdown: (0..n_shared).map(|s| t2(&dev, &hw.sdown[s * h * mi..(s + 1) * h * mi], h, mi)).collect(),
            experts,
        }
    };
    let xd = c2(&dev, x.clone(), n, h);
    unit_checks(&dev);
    mem_guard("before routing pass");
    let routing: Vec<Routing> = {
        let dw0 = build_dev(&hw);
        let logits = pre_moe(&dev, &dw0, &geom, xd.clone()).2.into_data().to_vec::<f32>().unwrap();
        let rows = n_routed + n_shared;
        (0..n).map(|tk| {
            let lt = &logits[tk * rows..(tk + 1) * rows];
            let mut order: Vec<usize> = (0..n_routed).collect();
            let score = |e: usize| { let v = lt[e]; (if v >= 0.0 { 1.0 / (1.0 + (-v).exp()) } else { let q = v.exp(); q / (1.0 + q) }) + hw.rb[e] };
            order.sort_by(|&a, &b| score(b).partial_cmp(&score(a)).unwrap().then(a.cmp(&b)));
            Routing { experts: order[..top_k].to_vec(), weights: vec![], shared_gammas: vec![] }
        }).collect()
    };
    let mut touched: Vec<usize> = routing.iter().flat_map(|r| r.experts.clone()).collect();
    touched.sort_unstable(); touched.dedup();
    println!("  touched experts: {} of {n_routed}", touched.len());
    for &e in &touched {
        let w13 = cp.expert_slice(&format!("{pfx}mlp.experts.w13_weight"), e)?.data;
        let w2 = cp.expert_slice(&format!("{pfx}mlp.experts.w2_weight"), e)?.data;
        anyhow::ensure!(w13.len() == 2 * mi * h && w2.len() == h * mi, "expert {e} shape");
        hw.experts.insert(e, (w13, w2));
    }
    println!("  weights loaded in {:.1}s", tm0.elapsed().as_secs_f64());
    mem_guard("before device weights");
    let dw = build_dev(&hw);
    let td = c2(&dev, target.clone(), n, h);
    mem_guard("after device weights");
    let tm2 = std::time::Instant::now();
    let y_dev = burn_layer(&dev, &dw, &geom, xd.clone(), &routing);
    let loss = (y_dev - td.clone()).powf_scalar(2.0).mean();
    let loss0 = loss.clone().into_data().to_vec::<f32>().unwrap()[0];
    println!("  forward + loss {loss0:.6} [{:.2}s]", tm2.elapsed().as_secs_f64());
    let grads = loss.backward();
    println!("  backward done [{:.2}s]", tm2.elapsed().as_secs_f64());
    let e0 = routing[0].experts[0];
    let gradvec = |t: &T2| -> Vec<f32> { t.grad(&grads).expect("grad").into_data().to_vec::<f32>().unwrap() };
    let (ex13, ex2) = dw.experts.get(&e0).unwrap();
    let cases: Vec<(&str, Vec<f32>)> = vec![
        ("attn.wq", gradvec(&dw.wq)), ("attn.wk", gradvec(&dw.wk)), ("attn.wv", gradvec(&dw.wv)), ("attn.wo", gradvec(&dw.wo)), ("attn.k_sconv", gradvec(&dw.ks)),
        ("attn.rel_proj", gradvec(&dw.rp)), ("attn_sconv", gradvec(&dw.attn_sconv)),
        ("attn_norm", dw.attn_norm.grad(&grads).expect("grad").into_data().to_vec::<f32>().unwrap()),
        ("router", gradvec(&dw.rw)), ("shared.gate[0]", gradvec(&dw.sgate[0])), ("shared.down[0]", gradvec(&dw.sdown[0])),
        ("expert.w13", gradvec(ex13)), ("expert.w2", gradvec(ex2)),
    ];
    drop(grads); // release the autodiff graph and its intermediates before the FD loop
    let client = mary::models::inkling::seam::client_of(&dw.wq.clone().inner());
    let pool_gib = |c: &cubecl::prelude::ComputeClient<cubecl::cuda::CudaRuntime>| mary::models::inkling::seam::pool_reserved(c) as f64 / (1u64 << 30) as f64;
    println!("  memory before FD loop: MemAvailable {:.1} GiB, device pool reserved {:.2} GiB (floor {MEM_FLOOR_GIB} GiB)", mem_available_gib(), pool_gib(&client));
    // loss of the SAME Burn forward with ONE leaf rebuilt from the host copy (no grad graph)
    let loss_at = |hw: &HostW, name: &str| -> f32 {
        mem_guard(name);
        let d = with_leaf(&dev, &dw, hw, name, e0, mi, h, heads, kv_heads, hd, kernel, t.d_rel, rel_extent);
        let y = burn_layer(&dev, &d, &geom, xd.clone(), &routing);
        (y - td.clone()).powf_scalar(2.0).mean().into_data().to_vec::<f32>().unwrap()[0]
    };
    let mut fails = 0usize; let mut worst_rel = 0f32; let mut checked = 0usize;
    println!("  gradient self-check (central differences on the same Burn forward, 3 largest-|grad| coordinates per class):");
    for (name, gv) in &cases {
        let mut order: Vec<usize> = (0..gv.len()).collect();
        order.sort_by(|&a, &b| gv[b].abs().partial_cmp(&gv[a].abs()).unwrap());
        for &idx in order.iter().take(3) {
            let gmax = gv[idx];
            let w0 = slot_mut(&mut hw, name, e0)[idx];
            let eps = 1e-2 * w0.abs() + 1e-4;
            slot_mut(&mut hw, name, e0)[idx] = w0 + eps; let lp = loss_at(&hw, name);
            slot_mut(&mut hw, name, e0)[idx] = w0 - eps; let lm = loss_at(&hw, name);
            slot_mut(&mut hw, name, e0)[idx] = w0;
            let fd = (lp - lm) / (2.0 * eps);
            let rel = (fd - gmax).abs() / fd.abs().max(gmax.abs()).max(1e-8);
            worst_rel = worst_rel.max(rel); checked += 1;
            let ok = rel <= FD_BUDGET; if !ok { fails += 1; }
            println!("    {name:>15} idx {idx:>8}  grad {gmax:+.4e}  fd {fd:+.4e}  rel {rel:.2e}  {}   [avail {:.1} GiB, pool {:.2} GiB]", if ok { "ok" } else { "FAIL" }, mem_available_gib(), pool_gib(&client));
        }
    }
    // ---- directional derivative per class: perturb the WHOLE tensor along its gradient.
    // Signal = |g| * eps, chosen so the loss moves by ~0.5% (far above f32 rounding of the
    // loss), then again at eps/4 to check linearity. Autodiff is right iff dd == |g|.
    println!("  directional check per class (whole tensor along its gradient; dd must equal |g|; two step sizes):");
    let mut dfails = 0usize;
    for (name, gv) in &cases {
        let gn: f32 = gv.iter().map(|v| v * v).sum::<f32>().sqrt();
        if gn == 0.0 { println!("    {name:>15} |g| = 0 -- skipped"); continue; }
        let orig: Vec<f32> = slot_mut(&mut hw, name, e0).clone();
        let m = gv.len(); // the grad covers exactly the leaf; the host vec may be wider (shared slots)
        let eps1 = 0.005 * loss0 / gn; let eps2 = eps1 / 4.0;
        let mut dds = Vec::new();
        for &eps in &[eps1, eps2] {
            { let v = slot_mut(&mut hw, name, e0); for i in 0..m { v[i] = orig[i] + eps * gv[i] / gn; } }
            let lp = loss_at(&hw, name);
            { let v = slot_mut(&mut hw, name, e0); for i in 0..m { v[i] = orig[i] - eps * gv[i] / gn; } }
            let lm = loss_at(&hw, name);
            dds.push((lp - lm) / (2.0 * eps));
        }
        { let v = slot_mut(&mut hw, name, e0); v.copy_from_slice(&orig); }
        let rel1 = (dds[0] - gn).abs() / gn; let rel2 = (dds[1] - gn).abs() / gn;
        // SPARSE direction: the 64 largest-|g| coordinates, each moved by 1e-2 of its own magnitude
        // (representable under TF32); predicted dL = sum g_i d_i, measured by central differences.
        let mut order: Vec<usize> = (0..m).collect();
        order.sort_by(|&a, &b| gv[b].abs().partial_cmp(&gv[a].abs()).unwrap());
        let kk: usize = std::env::var("INK_PROBE_SPARSE_K").ok().and_then(|v| v.parse().ok()).unwrap_or(64);
        let top: Vec<usize> = order.iter().take(kk).copied().collect();
        let dvec: Vec<(usize, f32)> = top.iter().map(|&i| (i, 1e-2 * orig[i].abs().max(1e-3) * gv[i].signum())).collect();
        let predicted: f32 = dvec.iter().map(|&(i, di)| gv[i] * di).sum();
        { let v = slot_mut(&mut hw, name, e0); for &(i, di) in &dvec { v[i] = orig[i] + di; } }
        let lp = loss_at(&hw, name);
        { let v = slot_mut(&mut hw, name, e0); for &(i, di) in &dvec { v[i] = orig[i] - di; } }
        let lm = loss_at(&hw, name);
        { let v = slot_mut(&mut hw, name, e0); v.copy_from_slice(&orig); }
        let measured = (lp - lm) / 2.0;
        let rel_s = (measured - predicted).abs() / predicted.abs().max(1e-9);
        let ok = rel_s <= FD_BUDGET; if !ok { dfails += 1; }
        println!("    {name:>15} |g| {gn:.4e}  dense dd {:.3e}/{:.3e} (rel {rel1:.1e}/{rel2:.1e})  SPARSE-{kk} predicted {predicted:+.4e} measured {measured:+.4e} rel {rel_s:.2e}  {}", dds[0], dds[1], if ok { "ok" } else { "FAIL" });
    }
    println!("  directional: {dfails} of {} classes failing", cases.len());
    println!("  worst rel {worst_rel:.2e} over {checked} coordinates, {fails} failing (budget {FD_BUDGET:.0e})");
    println!("\n  NOT CHECKED: routing selection gradient (discrete by construction); v/k projections and k_norm/q_norm gains and mlp_norm/mlp_sconv (same code paths as their checked siblings); numerical agreement beyond f32 central differences.");
    if dfails == 0 { println!("\nPASS: one MoE decoder layer is differentiable under Autodiff<Cuda<f32>>, autodiff agrees with its own finite differences."); Ok(()) }
    else { println!("\nFAIL"); std::process::exit(1) }
}
