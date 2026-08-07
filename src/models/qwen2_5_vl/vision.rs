//! Qwen2.5-VL vision tower — the vision encoder of `nomic-embed-multimodal-7b`
//! (the adapter carries NO vision LoRA, so the base `visual.*` weights are
//! final). Ported faithfully from `transformers`' `Qwen2_5_VisionTransformer`:
//!
//! - **PatchEmbed**: a Conv3d with kernel == stride == the full patch extent, so
//!   it is exactly a bias-free linear projection of each flattened patch
//!   (`[in*t*ph*pw = 1176] -> [hidden = 1280]`).
//! - **2D RoPE**: per-patch `(h, w)` grid positions index a shared 1D frequency
//!   table; the h- and w-halves are concatenated (40) then applied with the
//!   standard `rotate_half` rule (half-width cos/sin == HF's duplicated form).
//! - **Windowed attention**: blocks attend within `window_size` windows, except
//!   the four `fullatt_block_indexes` ([7,15,23,31]) which attend globally.
//!   Tokens are reordered into window-contiguous, merge-unit-grouped order;
//!   attention runs per window (block-diagonal additive mask); the merged output
//!   is scattered back to raster order.
//! - **PatchMerger**: RMSNorm, 2x2 spatial merge (`1280*4 = 5120`), GELU MLP to
//!   `out_hidden_size = 3584`.
//!
//! NOTE: a single small image whose grid fits one window exercises the
//! single-window path (windowed == full attention, identity reorder). The window
//! partition / scatter logic is implemented generally but is only parity-pinned
//! here in that regime; multi-window / multi-image goldens are a follow-up.

use burn::prelude::*;
use burn::tensor::activation::{gelu, silu, softmax};

use super::config::Qwen2_5VlVisionConfig;
use super::layers::QwenRmsNorm;

/// Weight source for the vision tower. `t2`/`t1` load 2-D/1-D leaves; `patch_proj`
/// loads the (>2-D) conv weight reshaped to `[out, in]`.
pub trait VisionWeights<B: Backend> {
    fn t1(&self, name: &str) -> Tensor<B, 1>;
    fn t2(&self, name: &str) -> Tensor<B, 2>;
    fn patch_proj(&self, name: &str, embed: usize, in_flat: usize) -> Tensor<B, 2>;
}

/// `y = x @ wᵀ (+ b)` against a `[out, in]` weight.
struct Linear<B: Backend> {
    weight: Tensor<B, 2>,
    bias: Option<Tensor<B, 1>>,
}
impl<B: Backend> Linear<B> {
    fn new(weight: Tensor<B, 2>, bias: Option<Tensor<B, 1>>) -> Self {
        Self { weight, bias }
    }
    fn forward<const D: usize>(&self, x: Tensor<B, D>) -> Tensor<B, D> {
        // f32-accumulated matmul (see layers.rs Linear): cubecl Metal's f16
        // accumulation overflows on long vision sequences. Weights stay f16-aliased.
        let dt = x.dtype();
        let xf = x.cast(burn::tensor::FloatDType::F32);
        let wf = self.weight.clone().cast(burn::tensor::FloatDType::F32);
        let out = xf.matmul(wf.transpose().unsqueeze());
        let out = match &self.bias {
            Some(b) => out + b.clone().cast(burn::tensor::FloatDType::F32).unsqueeze(),
            None => out,
        };
        out.cast(dt)
    }
}

/// Vision SwiGLU MLP (all projections have bias).
struct VisionMlp<B: Backend> {
    gate: Linear<B>,
    up: Linear<B>,
    down: Linear<B>,
}
impl<B: Backend> VisionMlp<B> {
    fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        self.down
            .forward(silu(self.gate.forward(x.clone())) * self.up.forward(x))
    }
}

/// Vision attention: fused `qkv` (bias), 2D RoPE, full attention within each
/// window-chunk (additive block-diagonal mask), `proj` (bias).
struct VisionAttention<B: Backend> {
    qkv: Linear<B>,
    proj: Linear<B>,
    n_heads: usize,
    head_dim: usize,
}
impl<B: Backend> VisionAttention<B> {
    fn forward(
        &self,
        x: Tensor<B, 2>,
        cos: Tensor<B, 2>,
        sin: Tensor<B, 2>,
        mask: Tensor<B, 4>,
    ) -> Tensor<B, 2> {
        let [seq, _] = x.dims();
        let (nh, hd) = (self.n_heads, self.head_dim);
        let qkv = self.qkv.forward(x).reshape([seq, 3, nh, hd]);
        let split = |i: usize| {
            qkv.clone()
                .narrow(1, i, 1)
                .reshape([seq, nh, hd])
                .swap_dims(0, 1)
                .reshape([1, nh, seq, hd])
        };
        let q = Self::rope(split(0), cos.clone(), sin.clone());
        let k = Self::rope(split(1), cos, sin);
        let v = split(2);

        let scores = q
            .matmul(k.swap_dims(2, 3))
            .mul_scalar((hd as f64).powf(-0.5))
            + mask;
        // fp32 softmax (HF convention); identity on an f32 backend.
        let dt = scores.dtype();
        let probs = softmax(scores.cast(burn::tensor::FloatDType::F32), 3).cast(dt);
        let out = probs
            .matmul(v)
            .reshape([nh, seq, hd])
            .swap_dims(0, 1)
            .reshape([seq, nh * hd]);
        self.proj.forward(out)
    }

    /// `rotate_half` RoPE with half-width (`hd/2`) cos/sin: `out1 = x1*cos -
    /// x2*sin`, `out2 = x1*sin + x2*cos`.
    fn rope(x: Tensor<B, 4>, cos: Tensor<B, 2>, sin: Tensor<B, 2>) -> Tensor<B, 4> {
        let [b, h, seq, hd] = x.dims();
        let half = hd / 2;
        let cos = cos.reshape([1, 1, seq, half]).expand([b, h, seq, half]);
        let sin = sin.reshape([1, 1, seq, half]).expand([b, h, seq, half]);
        let x1 = x.clone().narrow(3, 0, half);
        let x2 = x.narrow(3, half, half);
        let out1 = x1.clone() * cos.clone() - x2.clone() * sin.clone();
        let out2 = x1 * sin + x2 * cos;
        Tensor::cat(vec![out1, out2], 3)
    }
}

/// One vision block: `x += attn(norm1(x)); x += mlp(norm2(x))`.
struct VisionBlock<B: Backend> {
    norm1: QwenRmsNorm<B>,
    attn: VisionAttention<B>,
    norm2: QwenRmsNorm<B>,
    mlp: VisionMlp<B>,
}
impl<B: Backend> VisionBlock<B> {
    fn forward(
        &self,
        x: Tensor<B, 2>,
        cos: Tensor<B, 2>,
        sin: Tensor<B, 2>,
        mask: Tensor<B, 4>,
    ) -> Tensor<B, 2> {
        let dims = x.dims();
        let n1 = self.norm1.forward(x.clone().unsqueeze::<3>()).reshape(dims);
        let x = x + self.attn.forward(n1, cos, sin, mask);
        let n2 = self.norm2.forward(x.clone().unsqueeze::<3>()).reshape(dims);
        let m = self.mlp.forward(n2.unsqueeze::<3>()).reshape(dims);
        x + m
    }
}

/// PatchMerger: `RMSNorm(context_dim) -> view(-1, context_dim*merge²) -> Linear ->
/// GELU -> Linear(out_hidden)`.
struct PatchMerger<B: Backend> {
    ln_q: QwenRmsNorm<B>,
    fc1: Linear<B>,
    fc2: Linear<B>,
    context_dim: usize,
    merge_unit: usize,
}
impl<B: Backend> PatchMerger<B> {
    fn forward(&self, x: Tensor<B, 2>) -> Tensor<B, 2> {
        let seq = x.dims()[0];
        let normed = self
            .ln_q
            .forward(x.unsqueeze::<3>())
            .reshape([seq, self.context_dim]);
        let grouped = normed.reshape([seq / self.merge_unit, self.context_dim * self.merge_unit]);
        self.fc2.forward(gelu(self.fc1.forward(grouped)))
    }
}

/// The full vision transformer.
pub struct VisionTransformer<B: Backend> {
    patch_proj: Linear<B>,
    blocks: Vec<VisionBlock<B>>,
    merger: PatchMerger<B>,
    cfg: Qwen2_5VlVisionConfig,
    device: B::Device,
}

impl<B: Backend> VisionTransformer<B> {
    pub fn load(
        w: &impl VisionWeights<B>,
        cfg: &Qwen2_5VlVisionConfig,
        device: &B::Device,
    ) -> Self {
        let hidden = cfg.hidden_size;
        let in_flat = cfg.in_channels * cfg.temporal_patch_size * cfg.patch_size * cfg.patch_size;
        let patch_proj = Linear::new(
            w.patch_proj("patch_embed.proj.weight", hidden, in_flat),
            None,
        );
        let lin = |n: &str| {
            Linear::new(
                w.t2(&format!("{n}.weight")),
                Some(w.t1(&format!("{n}.bias"))),
            )
        };
        let blocks = (0..cfg.depth)
            .map(|i| {
                let p = format!("blocks.{i}");
                VisionBlock {
                    norm1: QwenRmsNorm::from_weight(w.t1(&format!("{p}.norm1.weight")), 1e-6),
                    attn: VisionAttention {
                        qkv: lin(&format!("{p}.attn.qkv")),
                        proj: lin(&format!("{p}.attn.proj")),
                        n_heads: cfg.num_heads,
                        head_dim: hidden / cfg.num_heads,
                    },
                    norm2: QwenRmsNorm::from_weight(w.t1(&format!("{p}.norm2.weight")), 1e-6),
                    mlp: VisionMlp {
                        gate: lin(&format!("{p}.mlp.gate_proj")),
                        up: lin(&format!("{p}.mlp.up_proj")),
                        down: lin(&format!("{p}.mlp.down_proj")),
                    },
                }
            })
            .collect();
        let merge_unit = cfg.spatial_merge_size * cfg.spatial_merge_size;
        let merger = PatchMerger {
            ln_q: QwenRmsNorm::from_weight(w.t1("merger.ln_q.weight"), 1e-6),
            fc1: lin("merger.mlp.0"),
            fc2: lin("merger.mlp.2"),
            context_dim: hidden,
            merge_unit,
        };
        Self {
            patch_proj,
            blocks,
            merger,
            cfg: cfg.clone(),
            device: device.clone(),
        }
    }

    /// Run the tower over `pixel_values` `[seq, in_flat]` with `grid` `[(t,h,w)]`.
    /// Returns the merged image tokens `[seq/merge², out_hidden]` in raster order.
    pub fn forward(
        &self,
        pixel_values: Tensor<B, 2>,
        grid: &[(usize, usize, usize)],
    ) -> Tensor<B, 2> {
        let merge_unit = self.cfg.spatial_merge_size * self.cfg.spatial_merge_size;
        let mut x = self.patch_proj.forward(pixel_values); // [seq, hidden]
        let seq = x.dims()[0];

        let (cos, sin) = self.rope_cos_sin(grid); // [seq, head_dim/2], raster order
        let (unit_index, cu_window) = self.window_index(grid); // merge-unit order + token cu

        // expand merge-unit order to per-token gather permutation
        let perm: Vec<i64> = unit_index
            .iter()
            .flat_map(|&u| (0..merge_unit).map(move |k| (u * merge_unit + k) as i64))
            .collect();
        x = gather_rows(x, &perm, &self.device);
        let cos = gather_rows(cos, &perm, &self.device);
        let sin = gather_rows(sin, &perm, &self.device);

        // full-attn chunks: one per image (t*h*w tokens)
        let mut cu_full = vec![0usize];
        for &(t, h, ww) in grid {
            cu_full.push(cu_full.last().unwrap() + t * h * ww);
        }
        let full_mask = block_diag_mask::<B>(seq, &cu_full, &self.device);
        let window_mask = block_diag_mask::<B>(seq, &cu_window, &self.device);

        for (i, blk) in self.blocks.iter().enumerate() {
            let mask = if self.cfg.fullatt_block_indexes.contains(&i) {
                full_mask.clone()
            } else {
                window_mask.clone()
            };
            x = blk.forward(x, cos.clone(), sin.clone(), mask);
        }

        let merged = self.merger.forward(x); // [n_units, out_hidden], window order
                                             // scatter back to raster order: out[raster] = merged[argsort(unit_index)]
        let back: Vec<i64> = argsort(&unit_index);
        gather_rows(merged, &back, &self.device)
    }

    /// Per-token 2D-RoPE `cos`/`sin` `[seq, head_dim/2]` in raster order
    /// (following HF `rot_pos_emb`'s merge permutation of `(h,w)` positions).
    fn rope_cos_sin(&self, grid: &[(usize, usize, usize)]) -> (Tensor<B, 2>, Tensor<B, 2>) {
        let merge = self.cfg.spatial_merge_size;
        let head_dim = self.cfg.hidden_size / self.cfg.num_heads;
        let rdim = head_dim / 2; // 40
        let nfreq = rdim / 2; // 20
        let theta = 10000.0f64;
        let inv: Vec<f64> = (0..nfreq)
            .map(|j| 1.0 / theta.powf((2 * j) as f64 / rdim as f64))
            .collect();
        let freqs = |p: usize| -> Vec<f32> { inv.iter().map(|&f| (p as f64 * f) as f32).collect() };

        let mut cos = Vec::new();
        let mut sin = Vec::new();
        for &(t, h, w) in grid {
            let (hh, ww) = (h / merge, w / merge);
            for _ in 0..t {
                for a in 0..hh {
                    for c in 0..ww {
                        for b in 0..merge {
                            for d in 0..merge {
                                let (hp, wp) = (a * merge + b, c * merge + d);
                                let mut rot = freqs(hp);
                                rot.extend(freqs(wp)); // 40
                                cos.extend(rot.iter().map(|v| v.cos()));
                                sin.extend(rot.iter().map(|v| v.sin()));
                            }
                        }
                    }
                }
            }
        }
        let total = cos.len() / rdim;
        (
            Tensor::<B, 1>::from_floats(&cos[..], &self.device).reshape([total, rdim]),
            Tensor::<B, 1>::from_floats(&sin[..], &self.device).reshape([total, rdim]),
        )
    }

    /// HF `get_window_index`: merge-unit indices in window-contiguous order plus
    /// the per-window cumulative TOKEN counts (`cu_window_seqlens`).
    fn window_index(&self, grid: &[(usize, usize, usize)]) -> (Vec<usize>, Vec<usize>) {
        let merge = self.cfg.spatial_merge_size;
        let merge_unit = merge * merge;
        let win = self.cfg.window_size / merge / self.cfg.patch_size; // vit_merger_window_size
        let mut unit_index: Vec<usize> = Vec::new();
        let mut cu = vec![0usize];
        let mut base = 0usize;
        for &(t, h, w) in grid {
            let (lh, lw) = (h / merge, w / merge);
            let pad_h = (win - lh % win) % win;
            let pad_w = (win - lw % win) % win;
            let nh = (lh + pad_h) / win;
            let nw = (lw + pad_w) / win;
            for _ti in 0..t {
                for wh in 0..nh {
                    for ww_ in 0..nw {
                        let mut count = 0;
                        for ih in 0..win {
                            for iw in 0..win {
                                let (gh, gw) = (wh * win + ih, ww_ * win + iw);
                                if gh < lh && gw < lw {
                                    unit_index.push(base + gh * lw + gw);
                                    count += 1;
                                }
                            }
                        }
                        cu.push(cu.last().unwrap() + count * merge_unit);
                    }
                }
            }
            base += t * lh * lw;
        }
        (unit_index, dedup_consecutive(cu))
    }
}

fn argsort(v: &[usize]) -> Vec<i64> {
    let mut idx: Vec<usize> = (0..v.len()).collect();
    idx.sort_by_key(|&i| v[i]);
    idx.iter().map(|&i| i as i64).collect()
}

fn dedup_consecutive(mut v: Vec<usize>) -> Vec<usize> {
    v.dedup();
    v
}

/// Gather rows of a `[n, d]` tensor by an index vector (`select` on dim 0).
fn gather_rows<B: Backend>(x: Tensor<B, 2>, idx: &[i64], device: &B::Device) -> Tensor<B, 2> {
    let n = idx.len();
    let sel =
        Tensor::<B, 1, Int>::from_data(burn::tensor::TensorData::new(idx.to_vec(), [n]), device);
    x.select(0, sel)
}

/// Additive block-diagonal mask `[1,1,seq,seq]`: 0 within a `cu`-delimited chunk,
/// -inf across chunks.
fn block_diag_mask<B: Backend>(seq: usize, cu: &[usize], device: &B::Device) -> Tensor<B, 4> {
    let mut chunk = vec![0usize; seq];
    for (c, win) in cu.windows(2).enumerate() {
        for p in win[0]..win[1].min(seq) {
            chunk[p] = c;
        }
    }
    let mut data = vec![0.0f32; seq * seq];
    for i in 0..seq {
        for j in 0..seq {
            if chunk[i] != chunk[j] {
                data[i * seq + j] = f32::NEG_INFINITY;
            }
        }
    }
    Tensor::<B, 1>::from_floats(&data[..], device).reshape([1, 1, seq, seq])
}
