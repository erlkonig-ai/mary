//! Streaming reader for the K3 checkpoint: 96 safetensors shards, 1.5 TB.
//!
//! The model cannot be resident, so nothing here holds a model. Every accessor
//! seeks into a shard, reads exactly one tensor's byte range, and returns it;
//! shard *headers* are cached (there are 96 of them and they are the only thing
//! worth keeping), tensor data never is. A caller that wants a layer asks for a
//! layer and drops it when done — that is how `k3_layer_gate` runs a 7168-wide
//! decoder layer of a 2.78 T-parameter model in a few gigabytes.
//!
//! Two properties are worth stating because they are what makes a wrong load
//! loud rather than plausible:
//!
//! * **Both directions of the index are checked.** `model.safetensors.index.json`
//!   names a shard for every tensor; the shard's own header must then name the
//!   tensor back. A stale index that points at the wrong shard is a panic, not
//!   a silently different weight.
//! * **Dtypes are asserted, never coerced.** [`Ckpt::bf16`] refuses an F32
//!   tensor and vice versa. The K3 checkpoint mixes them deliberately —
//!   `A_log`, `dt_bias`, the three `*_conv1d.weight`s and `o_norm.weight` are
//!   F32 while everything around them is BF16 — so "read it as whatever it is"
//!   would quietly erase the one distinction that matters.

use std::cell::RefCell;
use std::collections::HashMap;
use std::fs::File;
use std::io::{Read, Seek, SeekFrom};
use std::path::{Path, PathBuf};

use burn::prelude::*;

use super::attn_res::AttnResParams;
use super::kda_attn::{KdaAttnConfig, KdaAttnWeights};
use super::mla::{MlaConfig, MlaWeights};
use super::moe::{ExpertWeights, LatentMoeWeights, RouterWeights, SharedExpertWeights};
use crate::nn::mxfp4::decode_mxfp4;

#[derive(serde::Deserialize)]
struct StEntry {
    dtype: String,
    shape: Vec<usize>,
    data_offsets: (u64, u64),
}

/// An opened K3 checkpoint directory.
pub struct Ckpt {
    dir: PathBuf,
    weight_map: HashMap<String, String>,
    headers: RefCell<HashMap<String, (HashMap<String, StEntry>, u64)>>,
}

fn read_at(path: &Path, off: u64, len: usize) -> Vec<u8> {
    let mut f = File::open(path).unwrap_or_else(|e| panic!("open {}: {e}", path.display()));
    f.seek(SeekFrom::Start(off)).expect("seek");
    let mut buf = vec![0u8; len];
    f.read_exact(&mut buf).expect("read");
    buf
}

fn t2<B: Backend>(v: Vec<f32>, shape: [usize; 2], dev: &B::Device) -> Tensor<B, 2> {
    assert_eq!(
        v.len(),
        shape[0] * shape[1],
        "tensor data length vs {shape:?}"
    );
    Tensor::<B, 2>::from_data(TensorData::new(v, shape), dev)
}

fn t1<B: Backend>(v: Vec<f32>, dev: &B::Device) -> Tensor<B, 1> {
    let n = v.len();
    assert!(n > 0, "empty 1-d tensor");
    Tensor::<B, 1>::from_data(TensorData::new(v, [n]), dev)
}

impl Ckpt {
    /// Open a checkpoint directory and read its shard index.
    pub fn open(dir: &Path) -> Ckpt {
        #[derive(serde::Deserialize)]
        struct Index {
            weight_map: HashMap<String, String>,
        }
        let p = dir.join("model.safetensors.index.json");
        let f = File::open(&p).unwrap_or_else(|e| panic!("open {}: {e}", p.display()));
        let idx: Index = serde_json::from_reader(std::io::BufReader::with_capacity(1 << 22, f))
            .expect("index json");
        assert!(!idx.weight_map.is_empty(), "empty weight_map");
        Ckpt {
            dir: dir.to_path_buf(),
            weight_map: idx.weight_map,
            headers: RefCell::new(HashMap::new()),
        }
    }

    /// Whether the checkpoint contains a tensor of this name.
    ///
    /// Read-only evidence about the architecture: which layers carry
    /// `self_attn.A_log` and which carry `self_attn.kv_a_proj_with_mqa` is what
    /// says where the MLA layers are, independently of what the config claims.
    pub fn has(&self, name: &str) -> bool {
        self.weight_map.contains_key(name)
    }

    /// Number of tensors in the index.
    pub fn len(&self) -> usize {
        self.weight_map.len()
    }

    pub fn is_empty(&self) -> bool {
        self.weight_map.is_empty()
    }

    /// One tensor's declared dtype, shape and raw bytes.
    pub fn raw(&self, name: &str) -> (String, Vec<usize>, Vec<u8>) {
        let shard = self
            .weight_map
            .get(name)
            .unwrap_or_else(|| panic!("{name} is not in model.safetensors.index.json"));
        let path = self.dir.join(shard);
        if !self.headers.borrow().contains_key(shard) {
            let mut f =
                File::open(&path).unwrap_or_else(|e| panic!("open {}: {e}", path.display()));
            let mut l = [0u8; 8];
            f.read_exact(&mut l).expect("header len");
            let n = u64::from_le_bytes(l) as usize;
            let mut hb = vec![0u8; n];
            f.read_exact(&mut hb).expect("header");
            let raw_map: HashMap<String, serde_json::Value> =
                serde_json::from_slice(&hb).expect("safetensors header json");
            let map: HashMap<String, StEntry> = raw_map
                .into_iter()
                .filter(|(k, _)| k != "__metadata__")
                .map(|(k, v)| {
                    let e: StEntry = serde_json::from_value(v)
                        .unwrap_or_else(|err| panic!("header entry {k}: {err}"));
                    (k, e)
                })
                .collect();
            self.headers
                .borrow_mut()
                .insert(shard.clone(), (map, 8 + n as u64));
        }
        let hs = self.headers.borrow();
        let (map, base) = hs.get(shard).unwrap();
        // The index names a shard; the shard's OWN header must also name the
        // tensor. Checking both directions is what makes a stale index loud.
        let e = map
            .get(name)
            .unwrap_or_else(|| panic!("{name} is absent from {shard}'s own header"));
        let (s, t) = e.data_offsets;
        assert!(t > s, "{name}: empty tensor in {shard}");
        (
            e.dtype.clone(),
            e.shape.clone(),
            read_at(&path, base + s, (t - s) as usize),
        )
    }

    /// A BF16 tensor, widened to f32 (a shift, not a cast).
    pub fn bf16(&self, name: &str) -> (Vec<usize>, Vec<f32>) {
        let (dt, shape, b) = self.raw(name);
        assert_eq!(dt, "BF16", "{name} is {dt}, expected BF16");
        let v = b
            .chunks_exact(2)
            .map(|c| f32::from_bits((u16::from_le_bytes([c[0], c[1]]) as u32) << 16))
            .collect();
        (shape, v)
    }

    /// An F32 tensor.
    pub fn f32(&self, name: &str) -> (Vec<usize>, Vec<f32>) {
        let (dt, shape, b) = self.raw(name);
        assert_eq!(dt, "F32", "{name} is {dt}, expected F32");
        let v = b
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
            .collect();
        (shape, v)
    }

    /// A U8 tensor — packed MXFP4 nibbles or E8M0 exponents.
    pub fn u8(&self, name: &str) -> (Vec<usize>, Vec<u8>) {
        let (dt, shape, b) = self.raw(name);
        assert_eq!(dt, "U8", "{name} is {dt}, expected U8");
        (shape, b)
    }

    /// Decode one MXFP4 plane: `[rows, cols/2]` packed nibbles + `[rows,
    /// cols/32]` E8M0 scales, both row-major.
    pub fn mxfp4_plane(&self, prefix: &str) -> (usize, usize, Vec<f32>) {
        let (ps, packed) = self.u8(&format!("{prefix}.weight_packed"));
        let (ss, scale) = self.u8(&format!("{prefix}.weight_scale"));
        assert_eq!(ps.len(), 2, "{prefix}.weight_packed rank");
        let rows = ps[0];
        let cols = ps[1] * 2;
        assert_eq!(ss, vec![rows, cols / 32], "{prefix}.weight_scale shape");
        (rows, cols, decode_mxfp4(&packed, &scale, rows, cols))
    }

    /// One routed expert, decoded from its packed nibbles.
    pub fn expert<B: Backend>(&self, layer: usize, id: usize, dev: &B::Device) -> ExpertWeights<B> {
        let p = format!("language_model.model.layers.{layer}.block_sparse_moe.experts.{id}");
        let mk = |suffix: &str| {
            let (r, c, v) = self.mxfp4_plane(&format!("{p}.{suffix}"));
            t2::<B>(v, [r, c], dev)
        };
        ExpertWeights {
            w1: mk("w1"),
            w2: mk("w2"),
            w3: mk("w3"),
        }
    }

    /// Everything in the MoE block except the routed experts.
    ///
    /// `bias_bf16` selects how `e_score_correction_bias` is read: it is stored
    /// F32, and a `dtype=bfloat16` load casts it down. The shipped run this
    /// checkpoint's oracles capture did cast it, so `true` is what reproduces
    /// them; the flag exists because that is a *load-time* choice and not a
    /// property of the arithmetic.
    pub fn moe_block_weights<B: Backend>(
        &self,
        layer: usize,
        bias_bf16: bool,
        dev: &B::Device,
    ) -> LatentMoeWeights<B> {
        let p = format!("language_model.model.layers.{layer}.block_sparse_moe");
        let (gw_s, gw) = self.bf16(&format!("{p}.gate.weight"));
        let (_, bias_f32) = self.f32(&format!("{p}.gate.e_score_correction_bias"));
        let bias: Vec<f32> = if bias_bf16 {
            bias_f32
                .iter()
                .map(|&v| half::bf16::from_f32(v).to_f32())
                .collect()
        } else {
            bias_f32
        };
        let (dp_s, dp) = self.bf16(&format!("{p}.routed_expert_down_proj.weight"));
        let (up_s, up) = self.bf16(&format!("{p}.routed_expert_up_proj.weight"));
        let (_, nw) = self.bf16(&format!("{p}.routed_expert_norm.weight"));
        LatentMoeWeights {
            down_proj: t2::<B>(dp, [dp_s[0], dp_s[1]], dev),
            up_proj: t2::<B>(up, [up_s[0], up_s[1]], dev),
            norm: Some(t1::<B>(nw, dev)),
            router: RouterWeights {
                weight: t2::<B>(gw, [gw_s[0], gw_s[1]], dev),
                bias: t1::<B>(bias, dev),
            },
            shared: Some(self.mlp_weights(&format!("{p}.shared_experts"), dev)),
        }
    }

    /// A `KimiMLP` — the dense layer-0 MLP and the fused shared-expert MLP are
    /// the same module at different widths, so they load the same way.
    pub fn mlp_weights<B: Backend>(&self, prefix: &str, dev: &B::Device) -> SharedExpertWeights<B> {
        let (gp_s, gp) = self.bf16(&format!("{prefix}.gate_proj.weight"));
        let (up_s, up) = self.bf16(&format!("{prefix}.up_proj.weight"));
        let (dn_s, dn) = self.bf16(&format!("{prefix}.down_proj.weight"));
        SharedExpertWeights {
            gate_proj: t2::<B>(gp, [gp_s[0], gp_s[1]], dev),
            up_proj: t2::<B>(up, [up_s[0], up_s[1]], dev),
            down_proj: t2::<B>(dn, [dn_s[0], dn_s[1]], dev),
        }
    }

    /// The eight MLA weights of a full-attention layer.
    pub fn mla_weights<B: Backend>(
        &self,
        layer: usize,
        cfg: &MlaConfig,
        dev: &B::Device,
    ) -> MlaWeights<B> {
        let p = format!("language_model.model.layers.{layer}.self_attn");
        let m2 = |n: &str| {
            let (s, v) = self.bf16(&format!("{p}.{n}.weight"));
            assert_eq!(s.len(), 2, "{p}.{n}.weight rank");
            t2::<B>(v, [s[0], s[1]], dev)
        };
        let m1 = |n: &str| {
            let (_, v) = self.bf16(&format!("{p}.{n}.weight"));
            t1::<B>(v, dev)
        };
        MlaWeights {
            q_a_proj: m2("q_a_proj"),
            q_a_layernorm: m1("q_a_layernorm"),
            q_b_proj: m2("q_b_proj"),
            kv_a_proj_with_mqa: m2("kv_a_proj_with_mqa"),
            kv_a_layernorm: m1("kv_a_layernorm"),
            kv_b_proj: m2("kv_b_proj"),
            o_proj: m2("o_proj"),
            g_proj: cfg.use_output_gate.then(|| m2("g_proj")),
        }
    }

    /// The thirteen KDA weights of a linear-attention layer.
    ///
    /// `A_log` is stored zero-padded to 128 and only the first `num_heads`
    /// entries are live. This truncates to the live ones — using all 128 is a
    /// bug, because `exp(0) = 1` makes every padded head's decay rate 1. The
    /// truncation is asserted, not assumed: every discarded entry must be
    /// exactly zero.
    pub fn kda_weights<B: Backend>(
        &self,
        layer: usize,
        cfg: &KdaAttnConfig,
        dev: &B::Device,
    ) -> KdaAttnWeights<B> {
        let p = format!("language_model.model.layers.{layer}.self_attn");
        let m2 = |n: &str| {
            let (s, v) = self.bf16(&format!("{p}.{n}.weight"));
            assert_eq!(s.len(), 2, "{p}.{n}.weight rank");
            t2::<B>(v, [s[0], s[1]], dev)
        };
        let conv = |n: &str| {
            let (s, v) = self.f32(&format!("{p}.{n}.weight"));
            assert_eq!(
                s,
                vec![cfg.proj_size(), 1, cfg.kda.conv_kernel],
                "{p}.{n}.weight is a depthwise [D, 1, W] conv weight"
            );
            v
        };
        let (a_shape, a_full) = self.f32(&format!("{p}.A_log"));
        assert_eq!(a_shape.len(), 1, "A_log rank");
        let nh = cfg.kda.num_heads;
        assert!(
            a_full.len() >= nh,
            "A_log has {} entries, fewer than the {nh} heads",
            a_full.len()
        );
        for (i, &v) in a_full.iter().enumerate().skip(nh) {
            assert_eq!(
                v, 0.0,
                "A_log[{i}] = {v} is past the {nh} live heads but is not zero — \
                 the padding assumption that makes truncation safe does not hold here"
            );
        }
        let (_, dt_bias) = self.f32(&format!("{p}.dt_bias"));
        let (_, o_norm) = self.f32(&format!("{p}.o_norm.weight"));
        KdaAttnWeights {
            q_proj: m2("q_proj"),
            k_proj: m2("k_proj"),
            v_proj: m2("v_proj"),
            q_conv1d: conv("q_conv1d"),
            k_conv1d: conv("k_conv1d"),
            v_conv1d: conv("v_conv1d"),
            a_log: a_full[..nh].to_vec(),
            dt_bias,
            f_a_proj: m2("f_a_proj"),
            f_b_proj: m2("f_b_proj"),
            b_proj: m2("b_proj"),
            g_proj: m2("g_proj"),
            o_norm,
            o_proj: m2("o_proj"),
        }
    }

    /// `input_layernorm.weight` and `post_attention_layernorm.weight`.
    pub fn layer_norms<B: Backend>(
        &self,
        layer: usize,
        dev: &B::Device,
    ) -> (Tensor<B, 1>, Tensor<B, 1>) {
        let p = format!("language_model.model.layers.{layer}");
        let g = |n: &str| {
            let (_, v) = self.bf16(&format!("{p}.{n}.weight"));
            t1::<B>(v, dev)
        };
        (g("input_layernorm"), g("post_attention_layernorm"))
    }

    /// One AttnRes call site's `(norm gain, projection)` pair, pre-multiplied.
    ///
    /// `prefix` is the site: `…layers.N.self_attention_res`, `…layers.N.mlp_res`
    /// or `language_model.model.output_attn_res`. The two tensors are handed to
    /// [`AttnResParams::new`] at their checkpoint ranks, so a swap is a shape
    /// error — which is the only way it can be caught, the score weight being
    /// their commutative product.
    pub fn attn_res_site<B: Backend>(
        &self,
        prefix: &str,
        eps: f64,
        dev: &B::Device,
    ) -> AttnResParams<B> {
        let (ns, nw) = self.bf16(&format!("{prefix}_norm.weight"));
        let (ps, pw) = self.bf16(&format!("{prefix}_proj.weight"));
        assert_eq!(ns.len(), 1, "{prefix}_norm.weight rank");
        assert_eq!(ps.len(), 2, "{prefix}_proj.weight rank");
        AttnResParams::new(t1::<B>(nw, dev), t2::<B>(pw, [ps[0], ps[1]], dev), eps)
    }
}
