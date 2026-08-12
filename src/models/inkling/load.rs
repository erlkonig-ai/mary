//! Reading real Inkling checkpoint tensors as f32.
//!
//! Names come from [`crate::models::inkling::layout::Slot::tensor_name`], so
//! the layout is what actually locates weights rather than a parallel set of
//! string literals that could drift from it.
//!
//! Only the tensors asked for are materialised. A shard is mapped, the tensor
//! copied out and widened, and the mapping dropped — a layer is on the order of
//! a gigabyte at f32 and the whole checkpoint is 159.

use std::collections::{BTreeMap, HashMap};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use std::time::Instant;

use anyhow::{Context, Result};
use memmap2::Mmap;
use safetensors::SafeTensors;

use crate::models::inkling::layout::Slot;
use crate::models::inkling::nvfp4::{decode_stacked, GROUP};

/// A checkpoint directory plus its tensor-to-shard index.
pub struct Checkpoint {
    dir: PathBuf,
    shard_of: HashMap<String, String>,
    /// What the loader has moved, per tensor name. Behind a mutex because
    /// [`Checkpoint::tensor`] takes `&self`; the lock is held for an add, so
    /// it costs nothing against a read that widens a gigabyte.
    io: Mutex<BTreeMap<String, NameIo>>,
    /// Weights kept for the whole run, keyed by tensor name (or by a derived
    /// key for a split half). Empty unless `INK_RESIDENT` is set.
    resident: Mutex<HashMap<String, Held>>,
    resident_on: bool,
}

/// A weight the run HOLDS, instead of re-reading it once per token.
pub type Held = Arc<Loaded>;

/// What the loader moved on account of one tensor name.
///
/// Not a profiler. It counts exactly the two quantities the residency question
/// turns on: how many bytes OF FILE a name costs each time it is asked for, and
/// how many bytes of f32 that becomes on the host. For every BF16 tensor in
/// this checkpoint the second is twice the first, which is why "7.7 GiB of
/// checkpoint per token" and "15.5 GB of f32 per token" are both true
/// statements about the same reads — the widening is [`Checkpoint::tensor`],
/// and it is per call.
#[derive(Default, Clone)]
pub struct NameIo {
    /// Reads that actually touched the file.
    pub calls: u64,
    /// Asks that a resident copy answered.
    pub hits: u64,
    pub file_bytes: u64,
    pub host_bytes: u64,
    pub nanos: u64,
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
        Ok(Checkpoint {
            dir,
            shard_of,
            io: Mutex::new(BTreeMap::new()),
            resident: Mutex::new(HashMap::new()),
            // Off by default: residency pins tens of gigabytes, and this box
            // has 119 of them against a 159 GiB checkpoint. It is a deliberate
            // call, so it is a deliberate flag.
            resident_on: std::env::var("INK_RESIDENT").map(|v| v != "0").unwrap_or(false),
        })
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
        let t0 = Instant::now();
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
        let data: Vec<f32> = match debug.as_str() {
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
        self.note(name, raw.len() as u64, (data.len() * 4) as u64, t0);
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

    /// Slice ONE expert out of a stacked BF16 expert matrix, without widening
    /// and without materialising the stack.
    ///
    /// The BF16 counterpart to [`Checkpoint::expert_slice_packed`]. Needed
    /// because a checkpoint can hold both: Inkling's layer 2 experts are plain
    /// BF16 while the rest are NVFP4, which the presence of a `.scale` sidecar
    /// decides.
    ///
    /// Slicing rather than reading the whole tensor is the point. A stacked
    /// BF16 `[256, 6144, 6144]` is 19 GB; [`Checkpoint::tensor_raw`] would copy
    /// all of it to hand back one 75 MB expert. This maps the shard and copies
    /// only the slice, so importing 256 experts costs 256 slices rather than
    /// 256 whole-stack reads.
    pub fn expert_slice_bf16(&self, base: &str, e: usize) -> Result<RawTensor> {
        let shard = self
            .shard_of
            .get(base)
            .with_context(|| format!("{base} is not in the index"))?;
        let path = self.dir.join(shard);
        let file =
            std::fs::File::open(&path).with_context(|| format!("opening {}", path.display()))?;
        // SAFETY: the checkpoint is read-only and nothing else writes it.
        let mmap = unsafe { Mmap::map(&file) }?;
        let st = SafeTensors::deserialize(&mmap)?;
        let view = st.tensor(base)?;
        let dtype = format!("{:?}", view.dtype());
        anyhow::ensure!(
            dtype == "BF16",
            "{base} holds {dtype}; expert_slice_bf16 is for the unquantised stacks"
        );
        let shape = view.shape();
        anyhow::ensure!(
            shape.len() == 3,
            "{base} has shape {shape:?}; a stacked expert matrix is rank 3"
        );
        let (n, rows, cols) = (shape[0], shape[1], shape[2]);
        anyhow::ensure!(e < n, "{base} stacks {n} experts; {e} is out of range");
        let per = rows * cols * 2;
        let raw = view.data();
        anyhow::ensure!(
            raw.len() == n * per,
            "{base}: {} bytes for {n}x{rows}x{cols} BF16, expected {}",
            raw.len(),
            n * per
        );
        Ok(RawTensor {
            dtype,
            shape: vec![rows, cols],
            bytes: raw[e * per..(e + 1) * per].to_vec(),
        })
    }

    /// How many experts a stacked matrix holds.
    ///
    /// Asked of the checkpoint rather than assumed, so a model with a different
    /// expert count imports fully instead of silently importing a prefix.
    pub fn expert_count(&self, base: &str) -> Result<usize> {
        let shard = self
            .shard_of
            .get(base)
            .with_context(|| format!("{base} is not in the index"))?;
        let path = self.dir.join(shard);
        let file =
            std::fs::File::open(&path).with_context(|| format!("opening {}", path.display()))?;
        // SAFETY: the checkpoint is read-only and nothing else writes it.
        let mmap = unsafe { Mmap::map(&file) }?;
        let st = SafeTensors::deserialize(&mmap)?;
        let shape = st.tensor(base)?.shape().to_vec();
        anyhow::ensure!(
            shape.len() == 3,
            "{base} has shape {shape:?}; a stacked expert matrix is rank 3"
        );
        Ok(shape[0])
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
            let t_slab = Instant::now();
            let data = self.with_bytes(base, |raw| {
                anyhow::ensure!(raw.len() == experts * per * 2, "{base} is {} bytes", raw.len());
                Ok(raw[e * per * 2..(e + 1) * per * 2]
                    .chunks_exact(2)
                    .map(|c| f32::from_bits((u16::from_le_bytes([c[0], c[1]]) as u32) << 16))
                    .collect::<Vec<f32>>())
            })?;
            self.note(base, (per * 2) as u64, (per * 4) as u64, t_slab);
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
        let t_slab = Instant::now();
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
        // Packed: no widening, so host bytes are file bytes. Charged to the
        // stack's own name, which is what makes the routed half of a
        // per-token total separable from the dense half.
        let moved = (codes.len() + scales.len()) as u64;
        self.note(base, moved, moved, t_slab);
        Ok(PackedExpert { codes, scales, scale2: scale2.data[e], rows, cols })
    }

    pub fn slot(&self, slot: Slot) -> Result<Loaded> {
        self.tensor(&slot.tensor_name())
    }

    // ---- accounting -----------------------------------------------------

    /// Charge one read to a tensor name.
    fn note(&self, name: &str, file_bytes: u64, host_bytes: u64, t0: Instant) {
        let mut io = self.io.lock().expect("io stats poisoned");
        let e = io.entry(name.to_string()).or_default();
        e.calls += 1;
        e.file_bytes += file_bytes;
        e.host_bytes += host_bytes;
        e.nanos += t0.elapsed().as_nanos() as u64;
    }

    /// Charge one ask that a resident copy answered.
    fn note_hit(&self, name: &str) {
        let mut io = self.io.lock().expect("io stats poisoned");
        io.entry(name.to_string()).or_default().hits += 1;
    }

    /// `(calls, hits, file bytes, host bytes, nanos)` since the last reset.
    pub fn io_totals(&self) -> (u64, u64, u64, u64, u64) {
        let io = self.io.lock().expect("io stats poisoned");
        io.values().fold((0, 0, 0, 0, 0), |a, e| {
            (a.0 + e.calls, a.1 + e.hits, a.2 + e.file_bytes, a.3 + e.host_bytes, a.4 + e.nanos)
        })
    }

    /// Zero the counters, so a per-token figure is a per-token figure.
    pub fn io_reset(&self) {
        self.io.lock().expect("io stats poisoned").clear();
    }

    /// The `top` heaviest names by bytes, as a table.
    ///
    /// Names are collapsed on the layer index — `layers.17.attn.wq_du.weight`
    /// and `layers.3.attn.wq_du.weight` are the same weight in different
    /// layers, and forty rows of the same shape teach nothing that one row
    /// times forty does not.
    pub fn io_table(&self, top: usize) -> String {
        let io = self.io.lock().expect("io stats poisoned");
        let mut rolled: BTreeMap<String, NameIo> = BTreeMap::new();
        for (name, e) in io.iter() {
            let key = collapse_layer(name);
            let r = rolled.entry(key).or_default();
            r.calls += e.calls;
            r.hits += e.hits;
            r.file_bytes += e.file_bytes;
            r.host_bytes += e.host_bytes;
            r.nanos += e.nanos;
        }
        let mut v: Vec<_> = rolled.into_iter().collect();
        v.sort_by_key(|(_, e)| std::cmp::Reverse(e.file_bytes.max(e.host_bytes)));
        let mut out = String::from("    reads   hits       file MiB       host MiB      s  name\n");
        for (name, e) in v.into_iter().take(top) {
            out.push_str(&format!(
                "  {:7} {:6} {:14.1} {:14.1} {:6.1}  {}\n",
                e.calls,
                e.hits,
                e.file_bytes as f64 / (1u64 << 20) as f64,
                e.host_bytes as f64 / (1u64 << 20) as f64,
                e.nanos as f64 / 1e9,
                name
            ));
        }
        out
    }

    // ---- residency ------------------------------------------------------

    /// Whether weights asked for through [`Checkpoint::held`] stay in RAM.
    pub fn resident_on(&self) -> bool {
        self.resident_on
    }

    /// How many bytes of f32 the resident set holds, and how many weights.
    pub fn resident_bytes(&self) -> (u64, usize) {
        let r = self.resident.lock().expect("resident poisoned");
        (r.values().map(|h| (h.data.len() * 4) as u64).sum(), r.len())
    }

    /// One dense weight — widened once and then KEPT, when residency is on.
    ///
    /// The forward asks for the same few hundred names on every token, and
    /// [`Checkpoint::tensor`] re-maps the shard, re-parses its header and
    /// re-widens the BF16 every single time. Off, this is that call with an
    /// `Arc` around it and nothing survives it. On, the first ask pays the read
    /// and the widen and every later one is a pointer copy.
    ///
    /// The page cache cannot do this job here, which is the non-obvious part:
    /// the routed experts stream ~130 GiB of the same checkpoint per token
    /// through it, so the dense pages are evicted between one token and the
    /// next no matter how warm they were. An owned allocation is not evictable,
    /// and that is the whole difference.
    pub fn held(&self, name: &str) -> Result<Held> {
        if self.resident_on {
            let hit = self
                .resident
                .lock()
                .expect("resident poisoned")
                .get(name)
                .cloned();
            if let Some(h) = hit {
                self.note_hit(name);
                return Ok(h);
            }
        }
        let h = Arc::new(self.tensor(name)?);
        if self.resident_on {
            self.resident
                .lock()
                .expect("resident poisoned")
                .insert(name.to_string(), h.clone());
        }
        Ok(h)
    }

    /// A PAIR of weights DERIVED from checkpoint bytes — the two halves of a
    /// fused gate/up matrix — held under their own key.
    ///
    /// Residency has to cover the derived form or it barely helps. The dense
    /// MLP's `mlp.w13_dn.weight` is 1.9 GiB of BF16 that widens to 3.8 GiB of
    /// f32 and is then de-interleaved into two halves; caching only the fused
    /// tensor would still copy 3.8 GiB per token and would hold 7.5 GiB to do
    /// it. Caching the halves holds 3.8 and copies nothing.
    ///
    /// `make` is not called at all on a hit, so it may be as expensive as it
    /// likes.
    pub fn derived_pair(
        &self,
        key: &str,
        make: impl FnOnce() -> Result<(Loaded, Loaded)>,
    ) -> Result<(Held, Held)> {
        let (k0, k1) = (format!("{key}#0"), format!("{key}#1"));
        if self.resident_on {
            let hit = {
                let r = self.resident.lock().expect("resident poisoned");
                match (r.get(&k0), r.get(&k1)) {
                    (Some(a), Some(b)) => Some((a.clone(), b.clone())),
                    _ => None,
                }
            };
            if let Some(pair) = hit {
                self.note_hit(key);
                return Ok(pair);
            }
        }
        let (a, b) = make()?;
        let (a, b) = (Arc::new(a), Arc::new(b));
        if self.resident_on {
            let mut r = self.resident.lock().expect("resident poisoned");
            r.insert(k0, a.clone());
            r.insert(k1, b.clone());
        }
        Ok((a, b))
    }
}

/// `model.llm.layers.17.attn.wq_du.weight` -> `model.llm.layers.N.attn.wq_du.weight`.
fn collapse_layer(name: &str) -> String {
    let mut out = String::with_capacity(name.len());
    let mut it = name.split('.').peekable();
    while let Some(seg) = it.next() {
        out.push_str(seg);
        if seg == "layers" {
            if let Some(next) = it.peek() {
                if next.chars().all(|c| c.is_ascii_digit()) {
                    it.next();
                    out.push_str(".N");
                }
            }
        }
        if it.peek().is_some() {
            out.push('.');
        }
    }
    out
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

/// Which reading of the SHARED experts' `shared_w13_weight` output axis to use.
///
/// `false` (the default) is INTERLEAVED — `g0, u0, g1, u1, …`, the same
/// convention as [`split_gate_up`]. `INK_SHARED_W13_HALVED=1` selects the
/// contiguous reading, so the question can be settled by running it rather than
/// by argument. See [`split_shared_w13`].
pub fn shared_w13_halved() -> bool {
    std::env::var("INK_SHARED_W13_HALVED").map(|v| v == "1").unwrap_or(false)
}

/// Split the shared experts' `shared_w13_weight`, `[n_shared, 2 * inter, hidden]`,
/// into gate and up, each `[n_shared, inter, hidden]`.
///
/// **This is a hazard, and it is a worse one than the routed experts'.** In both
/// released checkpoints `2 * inter == hidden == 4096`, so the tensor is square
/// on its last two axes and a wrong split loads without complaint and computes
/// nonsense — the same trap [`super::mlp`] documents for the routed matrix. But
/// the routed split has a witness the shared one does not: `w2` is non-square,
/// which pins the `[out, in]` convention, and the routed fused matrix is
/// converted by `transformers/conversion_mapping.py` with an explicit
/// `[Interleave(dim=0), Chunk(dim=0)]`. Nothing in the shared tensor's shape or
/// in a conversion rule distinguishes interleaved from halved, and the two
/// readings have the SAME total sum, so every aggregate fingerprint that does
/// not separate the halves agrees with both.
///
/// This tree contained both readings at once, and the contradiction had never
/// been run: `inkling_forward`/`inkling_pipe` de-interleaved, `inkling_real_gate`
/// halved, and `golden/capture_inkling_real.py` had a docstring and a manifest
/// saying halved above code that calls `_deint`. The shipped
/// `inkling_oracle_fp4` manifest records the shared gate/up sums as
/// `-6.243070e+02 / -6.074423e+02`, which are exactly the HALVED split of layer
/// 3 of `inkling-small-nvfp4`; the interleaved split of the same bytes is
/// `-1.412489e+03 / +1.807401e+02`. So the golden was captured against the other
/// reading than the code that now sits above it — a stale oracle, not a
/// corroboration.
pub fn split_shared_w13(
    fused: &[f32],
    n_shared: usize,
    inter: usize,
    hidden: usize,
    halved: bool,
) -> (Vec<f32>, Vec<f32>) {
    let per = 2 * inter * hidden;
    assert_eq!(
        fused.len(),
        n_shared * per,
        "shared_w13 is not [{n_shared}, {}, {hidden}]",
        2 * inter
    );
    let mut gate = Vec::with_capacity(n_shared * inter * hidden);
    let mut up = Vec::with_capacity(n_shared * inter * hidden);
    for s in 0..n_shared {
        let blk = &fused[s * per..(s + 1) * per];
        if halved {
            gate.extend_from_slice(&blk[..per / 2]);
            up.extend_from_slice(&blk[per / 2..]);
        } else {
            let (g, u) = deinterleave_rows(blk, 2 * inter, hidden);
            gate.extend_from_slice(&g);
            up.extend_from_slice(&u);
        }
    }
    (gate, up)
}
