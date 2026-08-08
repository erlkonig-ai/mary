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
    pub fn slot(&self, slot: Slot) -> Result<Loaded> {
        self.tensor(&slot.tensor_name())
    }
}

/// Split a fused `[2 * inter, hidden]` gate-and-up matrix into its halves.
///
/// The checkpoint stores `w1` and `w3` concatenated along the OUTPUT dimension:
/// weight row `i` produces output `i`, and the reference chunks the fused
/// output as `(gate, up)`, so the gate occupies the first half of the rows.
///
/// This split is an interpretation of the checkpoint, not something its shapes
/// force — swapping the halves yields the same shapes and a wrong model. It
/// follows the LLaMA convention that `w1` is the gate and `w3` the up
/// projection, which is also what the fused MoE path implies.
pub fn split_gate_up(fused: &[f32], hidden: usize) -> (Vec<f32>, Vec<f32>) {
    assert_eq!(fused.len() % (2 * hidden), 0, "fused matrix is not [2*inter, hidden]");
    let half = fused.len() / 2;
    (fused[..half].to_vec(), fused[half..].to_vec())
}
