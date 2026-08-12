//! Inkling-Small across two machines, split by layer.
//!
//! One machine cannot hold this model. The NVFP4 checkpoint is 159 GiB against
//! 119 GiB of RAM, so `inkling_forward` reads every expert slab it routes to
//! out of the mapping and drops it — and the 40 GiB it cannot cache comes off
//! the disk, every layer, every token. A measured single forward over five
//! tokens moves 17.8 GB through the block layer.
//!
//! Two machines hold it. The split is by LAYER, not by tensor: each node owns a
//! contiguous range, runs it, and hands the residual stream on. That is the
//! shape the arithmetic wants — the only thing crossing the wire is `[n, 4096]`
//! f32, 16 KB per token per boundary, once. Splitting *within* a layer would
//! need an all-reduce per layer and 1 GbE cannot carry it.
//!
//! Byte-balanced at layer 20: layer 2 carries BF16 experts (12.7 GiB against
//! 3.55 GiB for an NVFP4 layer), so an even 21/21 layer split is a lopsided
//! 85/71 GiB split. Layers 0..=19 are 77.6 GiB and 20..=41 are 78.2 GiB;
//! adding the embedding to the first and the unembedding to the second leaves
//! both near 79 GiB, comfortably resident.
//!
//! Residency is the point, so it is measured rather than assumed: every step
//! reports its own `read_bytes` delta from `/proc/self/io`, which counts actual
//! block-device traffic. A warmed node reports zero.
//!
//!   inkling_pipe --ckpt DIR --role tail --layers 20:42 --listen 0.0.0.0:7654
//!   inkling_pipe --ckpt DIR --role head --layers 0:20  --peer HOST:7654 \
//!                --ids ids.bin --gen 8

use std::collections::BTreeMap;
use std::io::{Read, Write};
use std::net::{TcpListener, TcpStream};
use std::path::{Path, PathBuf};
use std::time::Instant;

use anyhow::{Context, Result};
use memmap2::Mmap;
use safetensors::SafeTensors;

use burn::prelude::Backend;
use burn::tensor::{Int, Tensor, TensorData};

use mary::models::inkling::attn::{attention, causal_mask, AttnDims, AttnWeights, LogScaling};
use mary::models::inkling::block::{rms_norm, route, short_conv, Routing};
use mary::models::inkling::burn::{deinterleave_rows_device, expert_ffn, expert_weight_from_packed};
use mary::models::inkling::config::{AttnKind, InklingConfig, InklingTextConfig};
use mary::models::inkling::load::{split_gate_up, Checkpoint};
use mary::models::inkling::mlp::{dense_mlp, shared_experts};
use mary::models::inkling::stack::{embed_and_norm, head};

type Bk = burn::backend::Cuda<f32>;

/// Bytes this process has pulled off the block device.
///
/// `read_bytes` in `/proc/self/io` counts what actually reached storage, so it
/// is blind to a page-cache hit. That is exactly the discrimination this
/// program exists to make: the same mmap read is free when the layer is
/// resident and a disk seek when it is not, and nothing else distinguishes
/// them from inside the process.
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

/// Which layer a checkpoint tensor belongs to, or `None` for the ends.
///
/// The embedding, the final norm and the unembedding carry no layer. Returning
/// `None` rather than 0 keeps them from silently joining the first node's
/// share, which would look like a working split and be a wrong one.
fn layer_of(name: &str) -> Option<usize> {
    let rest = name.split("layers.").nth(1)?;
    rest.split('.').next()?.parse().ok()
}

/// Fault this node's share into the page cache, and report what it cost.
///
/// Touching one byte per 4 KiB page is enough — the kernel's readahead turns a
/// forward-sequential fault pattern into large sequential reads. The checksum
/// exists so the loop is not optimised away.
fn warm(dir: &Path, lo: usize, hi: usize) -> Result<(u64, u64)> {
    let text = std::fs::read_to_string(dir.join("model.safetensors.index.json"))?;
    let v: serde_json::Value = serde_json::from_str(&text)?;
    let map = v.get("weight_map").and_then(|m| m.as_object()).context("weight_map")?;

    let mut by_shard: BTreeMap<String, Vec<String>> = BTreeMap::new();
    for (name, shard) in map {
        let keep = match layer_of(name) {
            Some(l) => l >= lo && l < hi,
            // The ends are loaded eagerly by whoever needs them; warming them
            // here would double-count and, on the wrong node, waste 3 GiB of
            // cache on a table it never reads.
            None => false,
        };
        if keep {
            by_shard.entry(shard.as_str().unwrap_or("").to_string()).or_default().push(name.clone());
        }
    }

    let (mut bytes, mut sum) = (0u64, 0u64);
    for (shard, names) in &by_shard {
        let file = std::fs::File::open(dir.join(shard))?;
        // SAFETY: the checkpoint is read-only and nothing else writes it.
        let mmap = unsafe { Mmap::map(&file) }?;
        let st = SafeTensors::deserialize(&mmap)?;
        for name in names {
            let d = st.tensor(name)?.data();
            bytes += d.len() as u64;
            let mut i = 0usize;
            while i < d.len() {
                sum = sum.wrapping_add(d[i] as u64);
                i += 4096;
            }
        }
    }
    Ok((bytes, sum))
}

/// Every routed expert of one layer, on the device, without a host f32 copy.
///
/// Lifted unchanged in behaviour from `inkling_forward`: the host never sees a
/// dequantised weight, the checkpoint's own bytes are uploaded and decoded on
/// the device, and each distinct expert is decoded once for all of its tokens.
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
) -> Result<Vec<f32>> {
    let hn_dev: Tensor<B, 2> = Tensor::from_data(TensorData::new(hn.to_vec(), [n, h]), dev);
    let mut acc: Tensor<B, 2> = Tensor::zeros([n, h], dev);

    for (&e, toks) in by_expert {
        let n13 = format!("{prefix}mlp.experts.w13_weight");
        let n2 = format!("{prefix}mlp.experts.w2_weight");

        let (fused, dn) = if cp.is_nvfp4(&n13) {
            let w13 = cp.expert_slice_packed(&n13, e)?;
            let w2 = cp.expert_slice_packed(&n2, e)?;
            (
                expert_weight_from_packed::<B>(
                    &w13.codes, &w13.scales, w13.scale2, w13.rows, w13.cols, dev,
                ),
                expert_weight_from_packed::<B>(
                    &w2.codes, &w2.scales, w2.scale2, w2.rows, w2.cols, dev,
                ),
            )
        } else {
            let a = cp.expert_slice(&n13, e)?;
            let b = cp.expert_slice(&n2, e)?;
            (
                Tensor::<B, 2>::from_data(TensorData::new(a.data, [2 * inter, h]), dev),
                Tensor::<B, 2>::from_data(TensorData::new(b.data, [h, inter]), dev),
            )
        };
        let gu = deinterleave_rows_device(fused);
        anyhow::ensure!(gu.dims() == [2 * inter, h], "w13 is {:?}", gu.dims());
        anyhow::ensure!(dn.dims() == [h, inter], "w2 is {:?}", dn.dims());

        let rows: Vec<i32> = toks.iter().map(|&(ti, _)| ti as i32).collect();
        let wts: Vec<f32> = toks.iter().map(|&(_, w)| w).collect();
        let k = rows.len();
        let idx: Tensor<B, 1, Int> = Tensor::from_data(TensorData::new(rows, [k]), dev);
        let wt: Tensor<B, 2> = Tensor::from_data(TensorData::new(wts, [k, 1]), dev);

        let xs = hn_dev.clone().select(0, idx.clone());
        let ys = expert_ffn(xs, gu, dn) * wt;
        acc = acc.select_assign(0, idx, ys, burn::tensor::IndexingUpdateOp::Add);
    }

    Ok(acc.into_data().convert::<f32>().to_vec::<f32>().expect("acc to host"))
}

/// Run layers `lo..hi` over the residual stream in place.
///
/// This is `inkling_forward`'s layer body with the range made a parameter and
/// nothing else changed, so a node running `0..42` reproduces the
/// single-machine forward exactly.
#[allow(clippy::too_many_arguments)]
fn run_layers(
    cp: &Checkpoint,
    t: &InklingTextConfig,
    x: &mut [f32],
    n: usize,
    lo: usize,
    hi: usize,
    dev: &burn::backend::cuda::CudaDevice,
    quiet: bool,
) -> Result<usize> {
    let h = t.hidden_size;
    let ls = LogScaling {
        n_floor: t.log_scaling_n_floor as f32,
        alpha: t.log_scaling_alpha as f32,
    };
    let mask_local = causal_mask(n, Some(t.sliding_window_size));
    let mask_global = causal_mask(n, None);
    let mut expert_loads = 0usize;

    for layer in lo..hi {
        let l0 = Instant::now();
        let kind = t.attn_kind(layer);
        let is_local = kind == AttnKind::Local;
        let (heads, kv_heads, head_dim) = t.heads(kind);
        let p = format!("model.llm.layers.{layer}.");
        let g = |nm: &str| -> Result<Vec<f32>> { Ok(cp.tensor(&format!("{p}{nm}"))?.data) };

        // ---- attention ----------------------------------------------------
        let attn_norm = g("attn_norm.weight")?;
        let hn = rms_norm(x, &attn_norm, t.rms_norm_eps, n, h);
        let dims = AttnDims {
            hidden: h,
            heads,
            kv_heads,
            head_dim,
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
        let mlp_norm = g("mlp_norm.weight")?;
        let hn = rms_norm(x, &mlp_norm, t.rms_norm_eps, n, h);

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

            let mut by_expert: BTreeMap<usize, Vec<(usize, f32)>> = BTreeMap::new();
            for (ti, r) in routing.iter().enumerate() {
                for (slot, &e) in r.experts.iter().enumerate() {
                    by_expert.entry(e).or_default().push((ti, r.weights[slot]));
                }
            }
            expert_loads += by_expert.len();
            let acc = routed_experts_gpu::<Bk>(cp, &p, &by_expert, &hn, n, h, inter, dev)?;

            let sfused = cp.tensor(&format!("{p}mlp.shared_experts.shared_w13_weight"))?.data;
            let (sg, su) = mary::models::inkling::load::split_shared_w13(
                &sfused,
                t.n_shared_experts,
                inter,
                h,
                mary::models::inkling::load::shared_w13_halved(),
            );
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

        if !quiet {
            let norm: f32 =
                (x.iter().map(|v| (v * v) as f64).sum::<f64>() / x.len() as f64).sqrt() as f32;
            println!("  layer {layer:2} [{}] {:.1}s  rms {norm:.4}",
                     if is_local { "local " } else { "global" }, l0.elapsed().as_secs_f32());
        }
    }
    Ok(expert_loads)
}

fn send_all(s: &mut TcpStream, b: &[u8]) -> Result<()> {
    s.write_all(b)?;
    s.flush()?;
    Ok(())
}

fn recv_u32(s: &mut TcpStream) -> Result<u32> {
    let mut b = [0u8; 4];
    s.read_exact(&mut b)?;
    Ok(u32::from_le_bytes(b))
}

fn f32s_to_bytes(v: &[f32]) -> Vec<u8> {
    let mut o = Vec::with_capacity(v.len() * 4);
    for x in v {
        o.extend_from_slice(&x.to_le_bytes());
    }
    o
}

fn bytes_to_f32s(b: &[u8]) -> Vec<f32> {
    b.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect()
}

struct Args {
    ckpt: PathBuf,
    role: String,
    lo: usize,
    hi: usize,
    addr: String,
    ids: Option<PathBuf>,
    gen: usize,
    warm: bool,
    dump: Option<PathBuf>,
}

fn parse() -> Result<Args> {
    let mut a = Args {
        ckpt: PathBuf::new(), role: String::new(), lo: 0, hi: 0,
        addr: String::new(), ids: None, gen: 0, warm: false, dump: None,
    };
    let v: Vec<String> = std::env::args().skip(1).collect();
    if v.is_empty() {
        anyhow::bail!(
            "usage: inkling_pipe --ckpt DIR --role head|tail --layers LO:HI \
             (--peer HOST:PORT | --listen ADDR:PORT) [--ids F] [--gen N] [--warm] [--dump DIR]"
        );
    }
    let mut i = 0;
    while i < v.len() {
        let k = v[i].as_str();
        let mut next = |i: &mut usize| -> Result<String> {
            *i += 1;
            v.get(*i).cloned().with_context(|| format!("{k} wants a value"))
        };
        match k {
            "--ckpt" => a.ckpt = PathBuf::from(next(&mut i)?),
            "--role" => a.role = next(&mut i)?,
            "--layers" => {
                let s = next(&mut i)?;
                let (l, h) = s.split_once(':').context("--layers wants LO:HI")?;
                a.lo = l.parse()?;
                a.hi = h.parse()?;
            }
            "--peer" | "--listen" => a.addr = next(&mut i)?,
            "--ids" => a.ids = Some(PathBuf::from(next(&mut i)?)),
            "--gen" => a.gen = next(&mut i)?.parse()?,
            "--warm" => a.warm = true,
            "--dump" => a.dump = Some(PathBuf::from(next(&mut i)?)),
            other => anyhow::bail!("unknown flag {other}"),
        }
        i += 1;
    }
    anyhow::ensure!(a.lo < a.hi, "--layers LO:HI wants LO < HI");
    Ok(a)
}

fn main() -> Result<()> {
    let a = parse()?;
    let cfg_text = std::fs::read_to_string(a.ckpt.join("config.json"))?;
    let cfg = InklingConfig::from_json(&cfg_text).context("parsing config.json")?;
    let t = &cfg.text_config;
    let h = t.hidden_size;
    let cp = Checkpoint::open(&a.ckpt)?;
    let dev = burn::backend::cuda::CudaDevice::default();

    println!("=== inkling_pipe {} layers {}..{} ===", a.role, a.lo, a.hi);
    println!("  checkpoint : {}", a.ckpt.display());
    println!("  hidden {h}  layers {}  experts {}+{}", t.num_hidden_layers, t.n_routed_experts, t.n_shared_experts);

    if a.warm {
        let t0 = Instant::now();
        let before = io_read_bytes();
        let (bytes, _sum) = warm(&a.ckpt, a.lo, a.hi)?;
        let disk = io_read_bytes() - before;
        println!("  warmed {:.1} GiB of layer share in {:.1}s ({:.0} MB/s), {:.1} GiB off disk",
                 bytes as f64 / (1u64 << 30) as f64, t0.elapsed().as_secs_f32(),
                 bytes as f64 / 1e6 / t0.elapsed().as_secs_f64().max(1e-9),
                 disk as f64 / (1u64 << 30) as f64);
    }

    match a.role.as_str() {
        // ---- warm: does this share FIT? -----------------------------------
        //
        // The question a split has to answer is not how fast a pass is but
        // whether the share stays. So warm twice and report the second pass's
        // block-device traffic: a share that fits reads zero the second time,
        // and one that does not reads back exactly what the kernel had to
        // evict to make room for the tail of the first pass.
        "warm" => {
            for pass in 1..=2 {
                let t0 = Instant::now();
                let before = io_read_bytes();
                let (bytes, _s) = warm(&a.ckpt, a.lo, a.hi)?;
                let disk = io_read_bytes() - before;
                println!("  pass {pass}: touched {:.1} GiB in {:.1}s, {:.2} GiB came off disk",
                         bytes as f64 / (1u64 << 30) as f64,
                         t0.elapsed().as_secs_f32(),
                         disk as f64 / (1u64 << 30) as f64);
            }
        }

        // ---- head: embedding, the first range, then the wire ---------------
        "head" => {
            let ids_path = a.ids.clone().context("--role head wants --ids")?;
            let mut ids: Vec<usize> = std::fs::read(&ids_path)?
                .chunks_exact(8)
                .map(|c| i64::from_le_bytes(c.try_into().unwrap()) as usize)
                .collect();
            anyhow::ensure!(!ids.is_empty(), "no tokens");

            let t0 = Instant::now();
            let embed_w = cp.tensor("model.llm.embed.weight")?.data;
            let embed_n = cp.tensor("model.llm.embed_norm.weight")?.data;
            println!("  embedding table in {:.1}s", t0.elapsed().as_secs_f32());

            let mut sock = TcpStream::connect(&a.addr)
                .with_context(|| format!("connecting to {}", a.addr))?;
            sock.set_nodelay(true)?;
            println!("  connected to {}\n", a.addr);

            let run0 = Instant::now();
            let mut per_step = Vec::new();
            for step in 0..=a.gen {
                let s0 = Instant::now();
                let io0 = io_read_bytes();
                let n = ids.len();
                let mut x = embed_and_norm(&ids, &embed_w, &embed_n, t.rms_norm_eps, t.vocab_size, h);
                let loads = run_layers(&cp, t, &mut x, n, a.lo, a.hi, &dev, true)?;
                let local = s0.elapsed().as_secs_f32();

                if let Some(d) = &a.dump {
                    std::fs::create_dir_all(d)?;
                    std::fs::write(d.join(format!("boundary_{step:02}.bin")), f32s_to_bytes(&x))?;
                }

                let w0 = Instant::now();
                send_all(&mut sock, &(n as u32).to_le_bytes())?;
                send_all(&mut sock, &f32s_to_bytes(&x))?;
                let mut back = [0u8; 8 * 6];
                sock.read_exact(&mut back)?;
                let wire_and_tail = w0.elapsed().as_secs_f32();
                let got: Vec<i64> = back
                    .chunks_exact(8)
                    .map(|c| i64::from_le_bytes(c.try_into().unwrap()))
                    .collect();
                let best = got[0] as usize;
                let disk = io_read_bytes() - io0;
                println!("  step {step:2}: +{best:<7} head {local:5.1}s  tail+wire {wire_and_tail:5.1}s  \
                          total {:5.1}s  experts {loads:4}  disk {:6.2} GiB  top5 {:?}",
                         s0.elapsed().as_secs_f32(), disk as f64 / (1u64 << 30) as f64, &got[1..]);
                per_step.push(s0.elapsed().as_secs_f64());
                ids.push(best);
            }
            // Tell the tail we are done: a zero-length batch.
            send_all(&mut sock, &0u32.to_le_bytes())?;

            let total = run0.elapsed().as_secs_f64();
            let steps = per_step.len() as f64;
            println!("\n  {steps} steps in {total:.1}s = {:.2} s/token, {:.4} tok/s",
                     total / steps, steps / total);
            let last: Vec<i64> = ids.iter().map(|&i| i as i64).collect();
            let mut b = Vec::new();
            for i in &last {
                b.extend_from_slice(&i.to_le_bytes());
            }
            std::fs::write("/tmp/inkling_dist/pipe_ids.bin", &b)?;
            println!("  ids -> /tmp/inkling_dist/pipe_ids.bin  {last:?}");
        }

        // ---- tail: the second range, the head, the argmax ------------------
        "tail" => {
            let t0 = Instant::now();
            let fnorm = cp.tensor("model.llm.norm.weight")?.data;
            let unembed = cp.tensor("model.llm.unembed.weight")?.data;
            println!("  unembedding table in {:.1}s", t0.elapsed().as_secs_f32());

            let l = TcpListener::bind(&a.addr).with_context(|| format!("binding {}", a.addr))?;
            println!("  listening on {}", a.addr);
            let (mut sock, peer) = l.accept()?;
            sock.set_nodelay(true)?;
            println!("  peer {peer}\n");

            let v = t.effective_vocab();
            let mut step = 0usize;
            loop {
                let n = match recv_u32(&mut sock) {
                    Ok(0) => break,
                    Ok(n) => n as usize,
                    Err(_) => break,
                };
                let s0 = Instant::now();
                let io0 = io_read_bytes();
                let mut buf = vec![0u8; n * h * 4];
                sock.read_exact(&mut buf)?;
                let mut x = bytes_to_f32s(&buf);
                let recv = s0.elapsed().as_secs_f32();

                if let Some(d) = &a.dump {
                    std::fs::create_dir_all(d)?;
                    std::fs::write(d.join(format!("boundary_in_{step:02}.bin")), &buf)?;
                }

                let loads = run_layers(&cp, t, &mut x, n, a.lo, a.hi, &dev, true)?;
                let logits = head(
                    &x, &fnorm, &unembed, t.logits_mup_width_multiplier as f32,
                    t.vocab_size, v, t.rms_norm_eps, n, h,
                );
                let last = &logits[(n - 1) * v..n * v];
                let mut idx: Vec<usize> = (0..v).collect();
                idx.sort_unstable_by(|&p, &q| last[q].partial_cmp(&last[p]).unwrap());
                let mut out = Vec::new();
                out.extend_from_slice(&(idx[0] as i64).to_le_bytes());
                for &i in &idx[..5] {
                    out.extend_from_slice(&(i as i64).to_le_bytes());
                }
                send_all(&mut sock, &out)?;
                let disk = io_read_bytes() - io0;
                println!("  step {step:2}: n {n:3}  recv {recv:.2}s  tail {:5.1}s  experts {loads:4}  \
                          disk {:6.2} GiB  -> {} ({:.2})",
                         s0.elapsed().as_secs_f32(), disk as f64 / (1u64 << 30) as f64,
                         idx[0], last[idx[0]]);
                step += 1;
            }
            println!("\n  peer closed after {step} steps");
        }
        other => anyhow::bail!("--role wants head or tail, not {other}"),
    }
    Ok(())
}
