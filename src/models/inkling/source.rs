//! Where a running Inkling gets its weights.
//!
//! Two backings, one interface: a safetensors checkpoint directory, or a pile.
//! The forward pass does not branch on which — it asks for a tensor by name or
//! an expert by (matrix, index) and gets bytes.
//!
//! # Why the residency and the accounting live HERE
//!
//! They used to live inside [`Checkpoint`], and they cannot stay there once
//! there are two sources, for two different reasons.
//!
//! The accounting has to be ONE instrument or the A/B is not a comparison. "How
//! many host f32 bytes did this token cost" answered by two different counters
//! placed at two different depths is two numbers about two questions. Here both
//! sources are charged at the same seam — the call the forward actually makes —
//! so the totals are commensurable by construction.
//!
//! The residency cache has to be one implementation for the same reason it
//! existed at all: it is a property of the CALLER's access pattern (the forward
//! asks for the same few hundred names every token), not of the storage. What
//! differs between the sources is how much it buys, and that is a measurement,
//! not a code path — over a pile a "read" is a hash-map lookup returning a view
//! of the mapping, so all the cache saves is the widening; over safetensors it
//! also saves re-parsing a shard header and re-copying out of the mmap.

use std::collections::{BTreeMap, HashMap};
use std::sync::{Arc, Mutex};
use std::time::Instant;

use anyhow::Result;

use super::load::{Checkpoint, Held, Loaded, PackedExpertRef};
use super::pile::{PackedSlab, PileSource};

/// What the loader moved on account of one tensor name.
///
/// Not a profiler. It counts exactly the two quantities the residency question
/// turns on: how many bytes OF STORAGE a name costs each time it is asked for,
/// and how many bytes of f32 that becomes on the host.
#[derive(Default, Clone)]
pub struct NameIo {
    /// Reads that actually touched the source.
    pub calls: u64,
    /// Asks that a resident copy answered.
    pub hits: u64,
    pub file_bytes: u64,
    pub host_bytes: u64,
    pub nanos: u64,
}

/// One expert's packed NVFP4 planes, borrowed from whichever source holds them.
///
/// Both arms borrow. The checkpoint arm points into a shard's mmap and the pile
/// arm into the pile's, and each carries the keepalive that makes that sound.
/// Nothing here copies 12.6 MB so a caller can own it — the caller hands the
/// bytes to a device and never reads them again.
pub enum Slab {
    Ckpt(PackedExpertRef),
    Pile(PackedSlab),
}

impl Slab {
    pub fn codes(&self) -> &[u8] {
        match self {
            Slab::Ckpt(r) => r.codes(),
            Slab::Pile(p) => &p.codes,
        }
    }

    pub fn scales(&self) -> &[u8] {
        match self {
            Slab::Ckpt(r) => r.scales(),
            Slab::Pile(p) => &p.scales,
        }
    }

    pub fn scale2(&self) -> f32 {
        match self {
            Slab::Ckpt(r) => r.scale2,
            Slab::Pile(p) => p.scale2,
        }
    }

    /// Output rows of this expert's matrix.
    pub fn rows(&self) -> usize {
        match self {
            Slab::Ckpt(r) => r.rows,
            Slab::Pile(p) => p.rows,
        }
    }

    /// Packed bytes per row; the logical width is `2 * cols`.
    pub fn cols(&self) -> usize {
        match self {
            Slab::Ckpt(r) => r.cols,
            Slab::Pile(p) => p.cols,
        }
    }

    /// Storage bytes this slab spans — codes plus block scales.
    pub fn bytes(&self) -> usize {
        self.codes().len() + self.scales().len()
    }
}

/// The two backings.
pub enum Src {
    Ckpt(Checkpoint),
    Pile(PileSource),
}

/// A model's weights, from either backing, with one residency cache and one set
/// of counters over both.
pub struct Weights {
    src: Src,
    resident_on: bool,
    resident: Mutex<HashMap<String, Held>>,
    io: Mutex<BTreeMap<String, NameIo>>,
}

impl Weights {
    /// Weights out of a safetensors checkpoint directory.
    pub fn open_ckpt(dir: impl AsRef<std::path::Path>) -> Result<Self> {
        Ok(Self::wrap(Src::Ckpt(Checkpoint::open(dir)?)))
    }

    /// Weights out of a pile, on the named branch.
    pub fn open_pile(path: impl AsRef<std::path::Path>, branch: &str) -> Result<Self> {
        Ok(Self::wrap(Src::Pile(PileSource::open(path.as_ref(), branch)?)))
    }

    fn wrap(src: Src) -> Self {
        Weights {
            src,
            // Off by default: residency pins tens of gigabytes, and this box
            // has 119 of them against a 159 GiB model. A deliberate call, so a
            // deliberate flag.
            resident_on: std::env::var("INK_RESIDENT").map(|v| v != "0").unwrap_or(false),
            resident: Mutex::new(HashMap::new()),
            io: Mutex::new(BTreeMap::new()),
        }
    }

    /// Which backing this is, for a banner.
    pub fn kind(&self) -> &'static str {
        match self.src {
            Src::Ckpt(_) => "safetensors checkpoint",
            Src::Pile(_) => "pile",
        }
    }

    /// The checkpoint underneath, when there is one.
    pub fn checkpoint(&self) -> Option<&Checkpoint> {
        match &self.src {
            Src::Ckpt(c) => Some(c),
            Src::Pile(_) => None,
        }
    }

    /// How many tensors this source located.
    pub fn len(&self) -> usize {
        match &self.src {
            Src::Ckpt(c) => c.len(),
            Src::Pile(p) => p.len(),
        }
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// What this source holds, in its own unit — which is NOT the same unit on
    /// both sides, so it says which.
    ///
    /// The checkpoint's index names stacked matrices: one name for 256 experts.
    /// The pile names each expert leaf, because that is the granularity a layer
    /// split partitions and a deduplicating store addresses.
    pub fn inventory(&self) -> String {
        match &self.src {
            Src::Ckpt(c) => format!("tensors    : {} index entries (expert stacks unsplit)", c.len()),
            Src::Pile(p) => format!(
                "tensors    : {} dense leaves + {} expert leaves",
                p.dense_len(),
                p.expert_len()
            ),
        }
    }

    // ---- reading ---------------------------------------------------------

    /// One dense tensor by checkpoint name, widened to f32.
    pub fn tensor(&self, name: &str) -> Result<Loaded> {
        let t0 = Instant::now();
        let out = match &self.src {
            Src::Ckpt(c) => {
                let stored = c.span(name)?.len as u64;
                let l = c.tensor(name)?;
                self.note(name, stored, (l.data.len() * 4) as u64, t0);
                l
            }
            Src::Pile(p) => {
                let leaf = p.leaf(name)?;
                let data = leaf.to_f32();
                self.note(name, leaf.bytes.len() as u64, (data.len() * 4) as u64, t0);
                Loaded { data, shape: leaf.shape() }
            }
        };
        Ok(out)
    }

    /// One dense weight — widened once and then KEPT, when residency is on.
    ///
    /// Off, this is [`Weights::tensor`] with an `Arc` around it and nothing
    /// survives it. On, the first ask pays the read and the widen and every
    /// later one is a pointer copy.
    pub fn held(&self, name: &str) -> Result<Held> {
        if self.resident_on {
            let hit = self.resident.lock().expect("resident").get(name).cloned();
            if let Some(h) = hit {
                self.note_hit(name);
                return Ok(h);
            }
        }
        let h = Arc::new(self.tensor(name)?);
        if self.resident_on {
            self.resident
                .lock()
                .expect("resident")
                .insert(name.to_string(), h.clone());
        }
        Ok(h)
    }

    /// A PAIR of weights DERIVED from stored bytes — the two halves of a fused
    /// gate/up matrix — held under their own key.
    ///
    /// Residency has to cover the derived form or it barely helps: caching only
    /// the fused tensor would still de-interleave 3.8 GiB of f32 per token and
    /// would pin 7.5 GiB to do it. `make` is not called at all on a hit.
    pub fn derived_pair(
        &self,
        key: &str,
        make: impl FnOnce() -> Result<(Loaded, Loaded)>,
    ) -> Result<(Held, Held)> {
        let (k0, k1) = (format!("{key}#0"), format!("{key}#1"));
        if self.resident_on {
            let hit = {
                let r = self.resident.lock().expect("resident");
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
            let mut r = self.resident.lock().expect("resident");
            r.insert(k0, a.clone());
            r.insert(k1, b.clone());
        }
        Ok((a, b))
    }

    /// Whether a stacked expert matrix is packed NVFP4 rather than BF16.
    ///
    /// Inkling-Small is mixed precision: 39 of its 40 MoE layers are NVFP4 and
    /// layer 2 is BF16. A lane that assumes NVFP4 everywhere dies on exactly one
    /// layer in forty, which is the kind of thing a sampled gate never sees.
    pub fn is_nvfp4(&self, base: &str) -> bool {
        match &self.src {
            Src::Ckpt(c) => c.is_nvfp4(base),
            Src::Pile(p) => p.is_nvfp4(base),
        }
    }

    /// One expert's packed planes, borrowed, undecoded.
    pub fn expert_packed(&self, base: &str, e: usize) -> Result<Slab> {
        let t0 = Instant::now();
        let s = match &self.src {
            Src::Ckpt(c) => Slab::Ckpt(c.expert_packed_ref(base, e)?),
            Src::Pile(p) => Slab::Pile(p.expert_packed(base, e)?),
        };
        // Packed: nothing is widened, so host bytes are storage bytes.
        let moved = s.bytes() as u64;
        self.note(base, moved, moved, t0);
        Ok(s)
    }

    /// One expert as f32 — the reference lane, which has to decode.
    pub fn expert_f32(&self, base: &str, e: usize) -> Result<Loaded> {
        let t0 = Instant::now();
        match &self.src {
            Src::Ckpt(c) => {
                let l = c.expert_slice(base, e)?;
                self.note(base, (l.data.len() * 2) as u64, (l.data.len() * 4) as u64, t0);
                Ok(l)
            }
            Src::Pile(p) => {
                let l = if p.is_nvfp4(base) {
                    let q = p.expert_packed(base, e)?;
                    let logical = q.cols * 2;
                    let mut out = vec![0f32; q.rows * logical];
                    let n = super::nvfp4::decode_stacked(
                        &q.codes, &q.scales, &[q.scale2], 1, q.rows, q.cols, &mut out,
                    );
                    anyhow::ensure!(n == out.len(), "decoded {n} of {}", out.len());
                    Loaded { data: out, shape: vec![q.rows, logical] }
                } else {
                    let leaf = p.expert_bf16(base, e)?;
                    Loaded { data: leaf.to_f32(), shape: leaf.shape() }
                };
                self.note(base, (l.data.len() * 2) as u64, (l.data.len() * 4) as u64, t0);
                Ok(l)
            }
        }
    }

    /// Fault a LAYER RANGE's whole share in, and say what it cost.
    ///
    /// The question a layer split has to answer is not how fast a pass is but
    /// whether a node's share STAYS resident, so this is measured against
    /// `/proc/self/io` by the caller rather than assumed. Both arms touch one
    /// byte per 4 KiB page and let the kernel's readahead do the rest.
    ///
    /// The difference between the arms is the whole argument for the pile: over
    /// a checkpoint the share is "every tensor whose NAME parses to a layer in
    /// range", and over a pile it is a query on a stored fact.
    pub fn warm(&self, range: std::ops::RangeInclusive<i64>) -> Result<(u64, usize)> {
        match &self.src {
            Src::Pile(p) => p.warm(*range.start(), *range.end()),
            Src::Ckpt(c) => {
                let (mut bytes, mut leaves, mut sum) = (0u64, 0usize, 0u64);
                for name in c.names() {
                    // Absent rather than zero: the embedding and the head carry
                    // no layer and belong to whoever loads them eagerly, not to
                    // the first node's share.
                    match super::pile::layer_of(&name) {
                        Some(l) if range.contains(&l) => {}
                        _ => continue,
                    }
                    let b = c.span(&name)?;
                    let b = b.bytes();
                    let mut i = 0usize;
                    while i < b.len() {
                        sum = sum.wrapping_add(b[i] as u64);
                        i += 4096;
                    }
                    bytes += b.len() as u64;
                    leaves += 1;
                }
                std::hint::black_box(sum);
                Ok((bytes, leaves))
            }
        }
    }

    /// Every host mapping this source reads through, as `(base, len, keepalive)`.
    ///
    /// What a zero-copy lane registers with the GPU, ONCE. The checkpoint has
    /// nine (its shards); a pile has one, because a pile IS one file — which is
    /// the whole of the difference between the two aliasing implementations this
    /// replaced.
    pub fn mappings(&self) -> Result<Vec<(usize, usize, Arc<dyn std::any::Any + Send + Sync>)>> {
        match &self.src {
            Src::Ckpt(c) => c.mappings(),
            Src::Pile(p) => p.mappings(),
        }
    }

    /// Visit EVERY byte range this source can hand to a device, in place.
    ///
    /// `f` receives `(tensor name, which plane, the borrowed bytes)`. The plane
    /// tag is `"codes"` / `"scales"` for a packed expert, `"expert-bf16"` for one
    /// of layer 2's, and `"dense"` for everything else — because the alignment
    /// question is answered per PLANE, not per tensor: a leaf whose payload
    /// starts aligned still hands the GPU a scale plane at `payload + codes_len`,
    /// and that offset is a different fact.
    ///
    /// A callback rather than a `Vec`, because the ranges borrow a 159 GiB
    /// mapping and collecting them would either copy the model or fight the
    /// borrow checker for nothing.
    ///
    /// Driven by what the SOURCE contains — `expert_keys` for a pile, the
    /// safetensors index for a checkpoint — never by what some layer range
    /// implies. An audit that enumerates the expected set cannot see a leaf that
    /// is there but unreachable, and cannot see one that is reachable but
    /// unexpected.
    pub fn for_each_bindable(
        &self,
        mut f: impl FnMut(&str, &'static str, &[u8]) -> Result<()>,
    ) -> Result<()> {
        match &self.src {
            Src::Pile(p) => {
                for name in p.names() {
                    let leaf = p.leaf(&name)?;
                    f(&name, "dense", &leaf.bytes)?;
                }
                for (name, e) in p.expert_keys() {
                    if p.expert_is_nvfp4(&name, e) == Some(true) {
                        let q = p.expert_packed(&name, e as usize)?;
                        f(&name, "codes", &q.codes)?;
                        f(&name, "scales", &q.scales)?;
                    } else {
                        let l = p.expert_bf16(&name, e as usize)?;
                        f(&name, "expert-bf16", &l.bytes)?;
                    }
                }
            }
            Src::Ckpt(c) => {
                let names = c.names();
                for name in names.iter().filter(|n| !n.contains(".experts.")) {
                    let span = c.span(name)?;
                    f(name, "dense", span.bytes())?;
                }
                let mut bases: Vec<&String> = names
                    .iter()
                    .filter(|n| n.ends_with(".experts.w13_weight") || n.ends_with(".experts.w2_weight"))
                    .collect();
                bases.sort();
                for base in bases {
                    let count = c.expert_count(base)?;
                    for e in 0..count {
                        if c.is_nvfp4(base) {
                            let q = c.expert_packed_ref(base, e)?;
                            f(base, "codes", q.codes())?;
                            f(base, "scales", q.scales())?;
                        } else {
                            // BF16 stacks have no borrowing accessor — the
                            // checkpoint arm copies them out. Reported as such
                            // rather than skipped: a plane the audit cannot see
                            // is not a plane that is fine.
                            let raw = c.expert_slice_bf16(base, e)?;
                            f(base, "expert-bf16-copied", &raw.bytes)?;
                        }
                    }
                }
            }
        }
        Ok(())
    }

    // ---- accounting ------------------------------------------------------

    fn note(&self, name: &str, file_bytes: u64, host_bytes: u64, t0: Instant) {
        let mut io = self.io.lock().expect("io stats");
        let e = io.entry(name.to_string()).or_default();
        e.calls += 1;
        e.file_bytes += file_bytes;
        e.host_bytes += host_bytes;
        e.nanos += t0.elapsed().as_nanos() as u64;
    }

    fn note_hit(&self, name: &str) {
        let mut io = self.io.lock().expect("io stats");
        io.entry(name.to_string()).or_default().hits += 1;
    }

    /// `(calls, hits, storage bytes, host bytes, nanos)` since the last reset.
    pub fn io_totals(&self) -> (u64, u64, u64, u64, u64) {
        let io = self.io.lock().expect("io stats");
        io.values().fold((0, 0, 0, 0, 0), |a, e| {
            (a.0 + e.calls, a.1 + e.hits, a.2 + e.file_bytes, a.3 + e.host_bytes, a.4 + e.nanos)
        })
    }

    /// Zero the counters, so a per-token figure is a per-token figure.
    pub fn io_reset(&self) {
        self.io.lock().expect("io stats").clear();
    }

    /// Whether weights asked for through [`Weights::held`] stay in RAM.
    pub fn resident_on(&self) -> bool {
        self.resident_on
    }

    /// How many bytes of f32 the resident set holds, and how many weights.
    pub fn resident_bytes(&self) -> (u64, usize) {
        let r = self.resident.lock().expect("resident");
        (r.values().map(|h| (h.data.len() * 4) as u64).sum(), r.len())
    }

    /// The `top` heaviest names by bytes, as a table. Layer indices collapse,
    /// because forty rows of the same weight teach nothing one row times forty
    /// does not.
    pub fn io_table(&self, top: usize) -> String {
        let io = self.io.lock().expect("io stats");
        let mut rolled: BTreeMap<String, NameIo> = BTreeMap::new();
        for (name, e) in io.iter() {
            let r = rolled.entry(collapse_layer(name)).or_default();
            r.calls += e.calls;
            r.hits += e.hits;
            r.file_bytes += e.file_bytes;
            r.host_bytes += e.host_bytes;
            r.nanos += e.nanos;
        }
        let mut v: Vec<_> = rolled.into_iter().collect();
        v.sort_by_key(|(_, e)| std::cmp::Reverse(e.file_bytes.max(e.host_bytes)));
        let mut out = String::from("    reads   hits    storage MiB       host MiB      s  name\n");
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
