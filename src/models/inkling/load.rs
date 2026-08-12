//! Reading real Inkling checkpoint tensors as f32.
//!
//! Only the tensors asked for are materialised. A shard stays mapped, the
//! tensor is copied out and widened on demand — a layer is on the order of a
//! gigabyte at f32 and the whole checkpoint is 159.
//!
//! The mapping is opened ONCE per shard and kept. It used to be opened, parsed
//! and dropped per call, and the routed-expert lane calls this eight times per
//! expert: `open` + `mmap` + `SafeTensors::deserialize` (which parses a shard's
//! whole JSON header — 19 GB shards, thousands of tensors) + `munmap`, 6.7 k
//! times in a five-token forward. Measured at **6.2 ms of every 20.9 ms
//! expert**, all of it host time nothing waits on. A shard's tensor spans are
//! parsed once into [`Span`] and every later read is a pointer offset.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

use anyhow::{Context, Result};
use memmap2::Mmap;
use safetensors::SafeTensors;

use crate::models::inkling::nvfp4::{decode_stacked, GROUP};

/// Where one tensor's bytes sit inside its shard's mapping.
///
/// The byte range is derived from the pointer safetensors hands back, so it
/// cannot disagree with what that library would have returned — this is a
/// cache of its answer, not a second parser for the format.
#[derive(Clone)]
pub struct Span {
    pub map: Arc<Mmap>,
    pub off: usize,
    pub len: usize,
    pub dtype: String,
    pub shape: Vec<usize>,
}

impl Span {
    /// The tensor's bytes, borrowed out of the live mapping.
    pub fn bytes(&self) -> &[u8] {
        &self.map[self.off..self.off + self.len]
    }
}

/// A checkpoint directory plus its tensor-to-shard index.
///
/// `maps` and `spans` are caches, hence the `Mutex` behind `&self`: reading a
/// weight is logically a pure lookup and every caller holds a shared reference.
///
/// A pure byte-locator. Residency and accounting used to live here too and now
/// live in [`crate::models::inkling::source::Weights`], because they are
/// properties of how a runtime asks rather than of where the bytes are, and
/// there is more than one where.
pub struct Checkpoint {
    dir: PathBuf,
    shard_of: HashMap<String, String>,
    maps: Mutex<HashMap<String, Arc<Mmap>>>,
    spans: Mutex<HashMap<String, Span>>,
}

/// A weight the run HOLDS, instead of re-reading it once per token.
pub type Held = Arc<Loaded>;

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

/// One expert's BF16 bytes, BORROWED out of the checkpoint mapping.
///
/// The counterpart to [`PackedExpertRef`] for the one layer Inkling leaves
/// unquantised. [`Checkpoint::expert_slice`] widens those bytes to f32 on the
/// host — 25.2 M scalar conversions and a 100 MB allocation per expert — which
/// is the single most expensive thing a decode token does. A device lane wants
/// the two bytes as they sit, and to widen them where the arithmetic is.
pub struct Bf16ExpertRef {
    span: Span,
    off: usize,
    len: usize,
    pub rows: usize,
    pub cols: usize,
}

impl Bf16ExpertRef {
    /// This expert's `[rows, cols]` BF16 bytes, little-endian, two per element.
    pub fn bytes(&self) -> &[u8] {
        let b = self.span.bytes();
        &b[self.off..self.off + self.len]
    }
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

/// The same expert, BORROWED out of the checkpoint mapping.
///
/// [`PackedExpert`] copies 12.6 MB per expert so the caller can own it. A
/// device lane does not want to own it: it hands the bytes to a host-to-device
/// copy and never looks at them again, so the copy into a `Vec` is pure
/// overhead — 843 of them per forward. The `Arc<Mmap>` keeps the mapping alive
/// for exactly as long as the borrow.
pub struct PackedExpertRef {
    codes: Span,
    scales: Span,
    codes_off: usize,
    codes_len: usize,
    scales_off: usize,
    scales_len: usize,
    pub scale2: f32,
    pub rows: usize,
    pub cols: usize,
}

impl PackedExpertRef {
    /// This expert's `[rows, cols]` packed code bytes.
    pub fn codes(&self) -> &[u8] {
        let b = self.codes.bytes();
        &b[self.codes_off..self.codes_off + self.codes_len]
    }

    /// This expert's raw E4M3 block-scale bytes, one per `GROUP` values.
    pub fn scales(&self) -> &[u8] {
        let b = self.scales.bytes();
        &b[self.scales_off..self.scales_off + self.scales_len]
    }

    /// Copy into the owning form, for callers that need to keep the bytes.
    pub fn to_owned_expert(&self) -> PackedExpert {
        PackedExpert {
            codes: self.codes().to_vec(),
            scales: self.scales().to_vec(),
            scale2: self.scale2,
            rows: self.rows,
            cols: self.cols,
        }
    }
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
            maps: Mutex::new(HashMap::new()),
            spans: Mutex::new(HashMap::new()),
        })
    }

    /// Every shard mapping, as `(base, len, keepalive)`.
    ///
    /// What a zero-copy lane registers with the GPU, once. Maps every shard
    /// named by the index — including ones no tensor has been read from yet —
    /// because a registration that arrives after the kernel does is no use.
    pub fn mappings(
        &self,
    ) -> Result<Vec<(usize, usize, Arc<dyn std::any::Any + Send + Sync>)>> {
        let mut shards: Vec<String> = self.shard_of.values().cloned().collect();
        shards.sort();
        shards.dedup();
        let mut out = Vec::with_capacity(shards.len());
        for s in shards {
            let m = self.map_of(&s)?;
            out.push((m.as_ptr() as usize, m.len(), m as Arc<dyn std::any::Any + Send + Sync>));
        }
        Ok(out)
    }

    /// The mapping for one shard file, opened at most once.
    fn map_of(&self, shard: &str) -> Result<Arc<Mmap>> {
        if let Some(m) = self.maps.lock().expect("map cache").get(shard) {
            return Ok(m.clone());
        }
        let path = self.dir.join(shard);
        let file = std::fs::File::open(&path)
            .with_context(|| format!("opening {}", path.display()))?;
        // SAFETY: the checkpoint is read-only and nothing else writes it.
        let map = Arc::new(unsafe { Mmap::map(&file) }?);
        self.maps
            .lock()
            .expect("map cache")
            .insert(shard.to_string(), map.clone());
        Ok(map)
    }

    /// Where one tensor lives — parsing its shard's header at most once.
    ///
    /// A miss inserts EVERY tensor of that shard, because the parse that
    /// answers one name has already answered all of them; paying it per name
    /// would keep the cost this cache exists to remove.
    pub fn span(&self, name: &str) -> Result<Span> {
        if let Some(s) = self.spans.lock().expect("span cache").get(name) {
            return Ok(s.clone());
        }
        let shard = self
            .shard_of
            .get(name)
            .with_context(|| format!("{name} is not in the index"))?
            .clone();
        let map = self.map_of(&shard)?;
        let st = SafeTensors::deserialize(&map)?;
        let base = map.as_ptr() as usize;
        let mut cache = self.spans.lock().expect("span cache");
        for (nm, view) in st.tensors() {
            let data = view.data();
            let off = data.as_ptr() as usize - base;
            cache.insert(
                nm,
                Span {
                    map: map.clone(),
                    off,
                    len: data.len(),
                    dtype: format!("{:?}", view.dtype()),
                    shape: view.shape().to_vec(),
                },
            );
        }
        cache
            .get(name)
            .cloned()
            .with_context(|| format!("{name} is not in shard {shard}"))
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
    /// through [`Checkpoint::expert_slice`].
    pub fn tensor(&self, name: &str) -> Result<Loaded> {
        let span = self.span(name)?;
        let shape = span.shape.clone();
        let raw = span.bytes();
        let debug = span.dtype.clone();
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
        let span = self.span(name)?;
        Ok(RawTensor {
            dtype: span.dtype.clone(),
            shape: span.shape.clone(),
            bytes: span.bytes().to_vec(),
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
        let span = self.span(base)?;
        let dtype = span.dtype.clone();
        anyhow::ensure!(
            dtype == "BF16",
            "{base} holds {dtype}; expert_slice_bf16 is for the unquantised stacks"
        );
        let shape = &span.shape;
        anyhow::ensure!(
            shape.len() == 3,
            "{base} has shape {shape:?}; a stacked expert matrix is rank 3"
        );
        let (n, rows, cols) = (shape[0], shape[1], shape[2]);
        anyhow::ensure!(e < n, "{base} stacks {n} experts; {e} is out of range");
        let per = rows * cols * 2;
        let raw = span.bytes();
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
        let shape = self.span(base)?.shape;
        anyhow::ensure!(
            shape.len() == 3,
            "{base} has shape {shape:?}; a stacked expert matrix is rank 3"
        );
        Ok(shape[0])
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

    /// Hand the tensor's raw bytes to `f` without copying them.
    fn with_bytes<R>(&self, name: &str, f: impl FnOnce(&[u8]) -> Result<R>) -> Result<R> {
        f(self.span(name)?.bytes())
    }

    fn shape_of(&self, name: &str) -> Result<Vec<usize>> {
        Ok(self.span(name)?.shape)
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
        Ok(self.expert_packed_ref(base, e)?.to_owned_expert())
    }

    /// One expert's BF16 bytes, borrowed rather than widened — see
    /// [`Bf16ExpertRef`]. Every bound is checked against the shard's own header.
    pub fn expert_bf16_ref(&self, base: &str, e: usize) -> Result<Bf16ExpertRef> {
        let span = self.span(base)?;
        anyhow::ensure!(span.dtype == "BF16", "{base} holds {}, not BF16", span.dtype);
        let shape = span.shape.clone();
        anyhow::ensure!(shape.len() == 3, "{base} is rank {}", shape.len());
        let (experts, rows, cols) = (shape[0], shape[1], shape[2]);
        anyhow::ensure!(e < experts, "expert {e} of {experts}");
        let per = rows * cols * 2;
        anyhow::ensure!(span.len == experts * per, "{base} is {} bytes", span.len);
        Ok(Bf16ExpertRef { span, off: e * per, len: per, rows, cols })
    }

    /// The same bytes, borrowed rather than copied — see [`PackedExpertRef`].
    ///
    /// Every bound here is checked against the shard's own header, so a wrong
    /// expert index or a short sidecar is an error rather than a slice of some
    /// neighbouring expert's weights.
    pub fn expert_packed_ref(&self, base: &str, e: usize) -> Result<PackedExpertRef> {
        let codes_span = self.span(base)?;
        let shape = codes_span.shape.clone();
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
        anyhow::ensure!(
            codes_span.len == experts * rows * cols,
            "{base} is {} bytes",
            codes_span.len
        );
        let scales_span = self.span(&format!("{base}.scale"))?;
        let s0 = e * rows * scales_per_row;
        anyhow::ensure!(
            scales_span.len >= s0 + rows * scales_per_row,
            "{base}.scale is short"
        );
        Ok(PackedExpertRef {
            codes: codes_span,
            scales: scales_span,
            codes_off: e * rows * cols,
            codes_len: rows * cols,
            scales_off: s0,
            scales_len: rows * scales_per_row,
            scale2: scale2.data[e],
            rows,
            cols,
        })
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
