//! Reading real Inkling checkpoint tensors as f32.
//!
//! Names come from [`crate::models::inkling::layout::Slot::tensor_name`], so
//! the layout is what actually locates weights rather than a parallel set of
//! string literals that could drift from it.
//!
//! Only the tensors asked for are materialised. A shard is mapped, the tensor
//! copied out and widened, and the mapping dropped — a layer is on the order of
//! a gigabyte at f32 and the whole checkpoint is 159.

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use memmap2::Mmap;
use safetensors::SafeTensors;

use crate::models::inkling::layout::Slot;
use crate::models::inkling::nvfp4::{decode_stacked, GROUP};

/// A checkpoint directory plus its tensor-to-shard index.
pub struct Checkpoint {
    dir: PathBuf,
    shard_of: HashMap<String, String>,
}

/// A tensor read out of the checkpoint and widened to f32.
pub struct Loaded {
    pub data: Vec<f32>,
    pub shape: Vec<usize>,
}

/// A tensor read out of the checkpoint in its OWN dtype.
///
/// `dtype` is the safetensors name ("BF16", "F32", ...) rather than an enum,
/// because the reader's job is to report what the file says and let the caller
/// decide whether it can handle it. An unknown dtype should be a caller-side
/// refusal with the name in the message, not a variant this file has to grow.
pub struct RawTensor {
    pub dtype: String,
    pub shape: Vec<usize>,
    pub bytes: Vec<u8>,
}

/// One expert's packed NVFP4 weight, exactly as it sits in the checkpoint.
///
/// `codes` is `[rows, cols]` bytes, two 4-bit E2M1 codes each, low nibble
/// first. `scales` is `[rows, rows_of_scales]` raw E4M3 bytes, one per
/// [`crate::models::inkling::nvfp4::GROUP`] logical elements. `scale2` is the
/// single F32 factor this expert carries. Logical width is `cols * 2`.
pub struct PackedExpert {
    pub codes: Vec<u8>,
    pub scales: Vec<u8>,
    pub scale2: f32,
    pub rows: usize,
    pub cols: usize,
}

impl Checkpoint {
    /// Open a checkpoint, reading `model.safetensors.index.json`.
    pub fn open(dir: impl AsRef<Path>) -> Result<Self> {
        let dir = dir.as_ref().to_path_buf();
        let idx = dir.join("model.safetensors.index.json");
        let text = std::fs::read_to_string(&idx)
            .with_context(|| format!("reading {}", idx.display()))?;
        let v: serde_json::Value = serde_json::from_str(&text).context("parsing the index")?;
        let map = v
            .get("weight_map")
            .and_then(|m| m.as_object())
            .context("index has no weight_map")?;
        let shard_of = map
            .iter()
            .filter_map(|(k, v)| v.as_str().map(|s| (k.clone(), s.to_string())))
            .collect::<HashMap<_, _>>();
        anyhow::ensure!(!shard_of.is_empty(), "weight_map is empty");
        Ok(Checkpoint { dir, shard_of })
    }

    /// How many tensors the index names — an examined count for callers.
    pub fn len(&self) -> usize {
        self.shard_of.len()
    }

    pub fn is_empty(&self) -> bool {
        self.shard_of.is_empty()
    }

    /// Read one tensor by checkpoint name, widened to f32.
    ///
    /// Handles the dtypes the released checkpoints actually hold: BF16 for
    /// everything dense, F32 for the router bias and scales. Packed NVFP4
    /// expert weights are not read here — they need their sidecars, so they go
    /// through [`Checkpoint::expert_matrix`].
    pub fn tensor(&self, name: &str) -> Result<Loaded> {
        let shard = self
            .shard_of
            .get(name)
            .with_context(|| format!("{name} is not in the index"))?;
        let path = self.dir.join(shard);
        let file = std::fs::File::open(&path).with_context(|| format!("opening {}", path.display()))?;
        // SAFETY: the checkpoint is read-only and nothing else writes it.
        let mmap = unsafe { Mmap::map(&file) }?;
        let st = SafeTensors::deserialize(&mmap)?;
        let view = st.tensor(name)?;
        let shape = view.shape().to_vec();
        let raw = view.data();
        let debug = format!("{:?}", view.dtype());
        let data = match debug.as_str() {
            "BF16" => raw
                .chunks_exact(2)
                .map(|c| f32::from_bits((u16::from_le_bytes([c[0], c[1]]) as u32) << 16))
                .collect(),
            "F32" => raw
                .chunks_exact(4)
                .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                .collect(),
            other => anyhow::bail!("{name} holds {other}, which this reader does not widen"),
        };
        Ok(Loaded { data, shape })
    }

    /// Every tensor name the index holds, sorted.
    pub fn names(&self) -> Vec<String> {
        let mut v: Vec<String> = self.shard_of.keys().cloned().collect();
        v.sort();
        v
    }

    /// Read one tensor WITHOUT widening it — dtype, shape and the bytes as they
    /// sit in the checkpoint.
    ///
    /// The complement to [`Checkpoint::tensor`], which widens everything to f32
    /// so the runtime can compute with it. An importer wants the opposite: a
    /// BF16 weight should land in the pile as BF16, because widening it doubles
    /// the pile and then the loader has to narrow it again to hand it to a GPU
    /// that wanted BF16 all along. Round-tripping through f32 is lossless for
    /// BF16 (every BF16 is an f32 with a truncated mantissa) but it is not
    /// free, and the free version is to not do it.
    ///
    /// Copies once, out of the mmap, because the mapping is local to this call.
    pub fn tensor_raw(&self, name: &str) -> Result<RawTensor> {
        let shard = self
            .shard_of
            .get(name)
            .with_context(|| format!("{name} is not in the index"))?;
        let path = self.dir.join(shard);
        let file =
            std::fs::File::open(&path).with_context(|| format!("opening {}", path.display()))?;
        // SAFETY: the checkpoint is read-only and nothing else writes it.
        let mmap = unsafe { Mmap::map(&file) }?;
        let st = SafeTensors::deserialize(&mmap)?;
        let view = st.tensor(name)?;
        Ok(RawTensor {
            dtype: format!("{:?}", view.dtype()),
            shape: view.shape().to_vec(),
            bytes: view.data().to_vec(),
        })
    }

    /// Read a stacked expert matrix, dequantising when it is NVFP4.
    ///
    /// A layer's experts are either NVFP4 with four sidecars or plain BF16 with
    /// none — the layout gate asserts that all-or-nothing invariant — so the
    /// presence of `.scale` decides which path this takes.
    pub fn expert_matrix(&self, base: &str) -> Result<Loaded> {
        if !self.shard_of.contains_key(&format!("{base}.scale")) {
            return self.tensor(base);
        }
        let codes = self.raw_bytes(base)?;
        let scales = self.raw_bytes(&format!("{base}.scale"))?;
        let scale2 = self.tensor(&format!("{base}.scale2"))?;
        let shape = self.shape_of(base)?;
        anyhow::ensure!(shape.len() == 3, "{base} is rank {}", shape.len());
        let (experts, rows, bytes_per_row) = (shape[0], shape[1], shape[2]);
        let logical = bytes_per_row * 2;
        anyhow::ensure!(
            scales.len() == experts * rows * (logical / GROUP),
            "{base}.scale is {} bytes, expected {}",
            scales.len(),
            experts * rows * (logical / GROUP)
        );
        let mut out = vec![0f32; experts * rows * logical];
        let n = decode_stacked(
            &codes, &scales, &scale2.data, experts, rows, bytes_per_row, &mut out,
        );
        anyhow::ensure!(n == out.len(), "decoded {n} of {}", out.len());
        Ok(Loaded { data: out, shape: vec![experts, rows, logical] })
    }

    /// Read ONE expert's slab from a stacked matrix, dequantising if needed.
    ///
    /// Returns `[rows, logical]` for expert `e`. Reading the whole stack would
    /// be 26 GB at f32 on the 42-layer model, and a short prompt activates a few
    /// dozen experts out of 256.
    pub fn expert_slice(&self, base: &str, e: usize) -> Result<Loaded> {
        let shape = self.shape_of(base)?;
        anyhow::ensure!(shape.len() == 3, "{base} is rank {}", shape.len());
        let (experts, rows, cols) = (shape[0], shape[1], shape[2]);
        anyhow::ensure!(e < experts, "expert {e} of {experts}");

        let quantized = self.shard_of.contains_key(&format!("{base}.scale"));
        if !quantized {
            // BF16, two bytes per element. Slice the mapping, do not copy the
            // whole stack: it is 8 GB and this wants 33 MB of it.
            let per = rows * cols;
            let data = self.with_bytes(base, |raw| {
                anyhow::ensure!(raw.len() == experts * per * 2, "{base} is {} bytes", raw.len());
                Ok(raw[e * per * 2..(e + 1) * per * 2]
                    .chunks_exact(2)
                    .map(|c| f32::from_bits((u16::from_le_bytes([c[0], c[1]]) as u32) << 16))
                    .collect::<Vec<f32>>())
            })?;
            return Ok(Loaded { data, shape: vec![rows, cols] });
        }

        // Slice once, here, and decode on top -- so the device path and this
        // path cannot disagree about where an expert's bytes are.
        let q = self.expert_slice_packed(base, e)?;
        let logical = q.cols * 2;
        let mut out = vec![0f32; q.rows * logical];
        let n = decode_stacked(
            &q.codes, &q.scales, &[q.scale2], 1, q.rows, q.cols, &mut out,
        );
        anyhow::ensure!(n == out.len(), "decoded {n} of {}", out.len());
        Ok(Loaded { data: out, shape: vec![q.rows, logical] })
    }

    /// Map a shard and hand the tensor's raw bytes to `f` without copying them.
    fn with_bytes<R>(&self, name: &str, f: impl FnOnce(&[u8]) -> Result<R>) -> Result<R> {
        let shard = self.shard_of.get(name).with_context(|| format!("{name} not in index"))?;
        let file = std::fs::File::open(self.dir.join(shard))?;
        // SAFETY: the checkpoint is read-only and nothing else writes it.
        let mmap = unsafe { Mmap::map(&file) }?;
        let st = SafeTensors::deserialize(&mmap)?;
        f(st.tensor(name)?.data())
    }

    fn raw_bytes(&self, name: &str) -> Result<Vec<u8>> {
        let shard = self.shard_of.get(name).with_context(|| format!("{name} not in index"))?;
        let file = std::fs::File::open(self.dir.join(shard))?;
        let mmap = unsafe { Mmap::map(&file) }?;
        let st = SafeTensors::deserialize(&mmap)?;
        Ok(st.tensor(name)?.data().to_vec())
    }

    fn shape_of(&self, name: &str) -> Result<Vec<usize>> {
        let shard = self.shard_of.get(name).with_context(|| format!("{name} not in index"))?;
        let file = std::fs::File::open(self.dir.join(shard))?;
        let mmap = unsafe { Mmap::map(&file) }?;
        let st = SafeTensors::deserialize(&mmap)?;
        Ok(st.tensor(name)?.shape().to_vec())
    }

    /// Read the tensor a layout slot names.
    /// Whether an expert stack is packed NVFP4 rather than BF16.
    ///
    /// Inkling-Small is mixed precision: 39 of its 40 MoE layers are NVFP4, and
    /// layer 2 — the first — is BF16 with no `.scale` sidecar. A device lane
    /// that assumes NVFP4 everywhere dies on exactly one layer out of forty,
    /// which is the kind of thing a sampled gate never sees.
    pub fn is_nvfp4(&self, base: &str) -> bool {
        self.shard_of.contains_key(&format!("{base}.scale"))
    }

    /// One expert's NVFP4 bytes, sliced out of the stack and **not decoded**.
    ///
    /// The decode is the expensive part -- 53% of a measured forward -- and on
    /// a GPU it belongs on the device, so the bytes have to be reachable
    /// undecoded. [`Checkpoint::expert_slice`] is this plus
    /// [`crate::models::inkling::nvfp4::decode_stacked`], which keeps the two
    /// lanes reading the same offsets.
    pub fn expert_slice_packed(&self, base: &str, e: usize) -> Result<PackedExpert> {
        let shape = self.shape_of(base)?;
        anyhow::ensure!(shape.len() == 3, "{base} is rank {}", shape.len());
        let (experts, rows, cols) = (shape[0], shape[1], shape[2]);
        anyhow::ensure!(e < experts, "expert {e} of {experts}");
        anyhow::ensure!(
            self.shard_of.contains_key(&format!("{base}.scale")),
            "{base} has no .scale sidecar -- it is not NVFP4",
        );

        let logical = cols * 2;
        anyhow::ensure!(logical % GROUP == 0, "{logical} logical is not a multiple of {GROUP}");
        let scales_per_row = logical / GROUP;
        let scale2 = self.tensor(&format!("{base}.scale2"))?;
        anyhow::ensure!(scale2.data.len() == experts, "scale2 is {}", scale2.data.len());
        let codes = self.with_bytes(base, |raw| {
            anyhow::ensure!(raw.len() == experts * rows * cols, "{base} is {} bytes", raw.len());
            Ok(raw[e * rows * cols..(e + 1) * rows * cols].to_vec())
        })?;
        let scales = self.with_bytes(&format!("{base}.scale"), |raw| {
            let s0 = e * rows * scales_per_row;
            anyhow::ensure!(raw.len() >= s0 + rows * scales_per_row, "{base}.scale is short");
            Ok(raw[s0..s0 + rows * scales_per_row].to_vec())
        })?;
        Ok(PackedExpert { codes, scales, scale2: scale2.data[e], rows, cols })
    }

    pub fn slot(&self, slot: Slot) -> Result<Loaded> {
        self.tensor(&slot.tensor_name())
    }
}

/// De-interleave a fused matrix along its OUTPUT axis.
///
/// The checkpoint stores gate and up rows alternating — `g0, u0, g1, u1, …` —
/// so the even rows are the gate and the odd rows the up projection. Returns
/// them as two contiguous blocks, which is what every consumer here expects.
pub fn deinterleave_rows(fused: &[f32], rows: usize, cols: usize) -> (Vec<f32>, Vec<f32>) {
    assert_eq!(fused.len(), rows * cols);
    assert!(rows % 2 == 0, "a fused gate/up matrix must have an even row count");
    let half = rows / 2;
    let mut a = Vec::with_capacity(half * cols);
    let mut b = Vec::with_capacity(half * cols);
    for r in 0..half {
        a.extend_from_slice(&fused[(2 * r) * cols..(2 * r + 1) * cols]);
        b.extend_from_slice(&fused[(2 * r + 1) * cols..(2 * r + 2) * cols]);
    }
    (a, b)
}

/// Split a fused gate-and-up matrix stored `[2 * inter, hidden]`.
///
/// Authoritative source: `transformers/conversion_mapping.py`, which converts
/// `mlp.w13_dn.weight` with `[Interleave(dim=0), Chunk(dim=0)]`. The interleave
/// comes FIRST, so the rows alternate gate/up rather than sitting in halves.
/// A contiguous split is shape-identical and numerically wrong, and no
/// comparison against a reference that makes the same split can detect it.
pub fn split_gate_up(fused: &[f32], hidden: usize) -> (Vec<f32>, Vec<f32>) {
    assert_eq!(fused.len() % hidden, 0, "fused matrix is not [rows, hidden]");
    deinterleave_rows(fused, fused.len() / hidden, hidden)
}

/// Re-pack an interleaved fused expert matrix into gate-then-up order.
///
/// `mlp.experts.w13_weight` is converted with `Interleave(dim=1)` and left
/// fused, because `InklingExperts` chunks the PRODUCT at run time; so the
/// stored rows must be de-interleaved once at load and then behave as two
/// contiguous halves.
pub fn deinterleave_fused(fused: &[f32], rows: usize, cols: usize) -> Vec<f32> {
    let (a, b) = deinterleave_rows(fused, rows, cols);
    let mut out = a;
    out.extend_from_slice(&b);
    out
}
