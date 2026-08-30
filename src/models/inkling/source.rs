//! Where a running Inkling gets its weights: the pile, and nothing else.
//!
//! There used to be two backings behind one interface — a safetensors
//! checkpoint directory or a pile — and the checkpoint arm is gone. Everything
//! comes from the pile: the weights, `config.json`, the chat template, the
//! tokenizer. A source that could also be a directory of shards is a source
//! that will be one by accident, and then half the run's provenance is a path
//! nobody recorded.
//!
//! [`crate::models::inkling::load::Checkpoint`] still exists and still reads
//! safetensors. It is how weights get INTO a pile (`inkling_expert_import`,
//! `inkling_dense_import`, `inkling_pile_import`) and how the parity gates get
//! the original bytes to feed a Python oracle. What it is no longer is a thing
//! a forward pass can be pointed at.
//!
//! # Why the residency and the accounting live HERE
//!
//! They used to live inside `Checkpoint`, and they did not follow it back.
//! Residency is a property of the CALLER's access pattern — the forward asks
//! for the same few hundred names every token — not of the storage, and the
//! byte counters answer "what did this ONE pass move", which is a question
//! about the asking. Over a pile a read is a hash-map lookup returning a view
//! of the mapping, so what the cache saves is the widening and nothing else;
//! that is a measurement, and it is [`Weights::io_table`] that reports it.

use std::collections::{BTreeMap, HashMap};
use std::sync::{Arc, Mutex};
use std::time::Instant;

use anyhow::Result;

use super::load::{Held, Loaded};
use super::pile::{Bf16Slab, PackedSlab, PileSource};

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

/// A model's weights out of a pile, with a residency cache and byte counters.
pub struct Weights {
    src: PileSource,
    resident: Mutex<HashMap<String, Held>>,
    io: Mutex<BTreeMap<String, NameIo>>,
}

impl Weights {
    /// Weights out of a pile's sole model collection. The only constructor
    /// there is.
    ///
    /// Residency is not a choice. `INK_RESIDENT` used to default it OFF,
    /// because pinning tens of gigabytes on a 119 GiB box against a 159 GiB
    /// model was a deliberate call — and that framing is what a second node
    /// retires. A node holds ITS share, which fits; a node that streams is a
    /// node reading the SSD in the middle of a decode step, which is the one
    /// thing this runtime must never do.
    pub fn open(path: impl AsRef<std::path::Path>) -> Result<Self> {
        Ok(Weights {
            src: PileSource::open(path.as_ref())?,
            resident: Mutex::new(HashMap::new()),
            io: Mutex::new(BTreeMap::new()),
        })
    }

    /// How many tensors this source located.
    pub fn len(&self) -> usize {
        self.src.len()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// What the pile holds, in ITS unit — which is not the checkpoint's, and
    /// saying so is the point. The safetensors index named stacked matrices,
    /// one name for 256 experts; the pile names each expert leaf, because that
    /// is the granularity a layer split partitions and a deduplicating store
    /// addresses.
    pub fn inventory(&self) -> String {
        format!(
            "tensors    : {} dense leaves + {} expert leaves",
            self.src.dense_len(),
            self.src.expert_len()
        )
    }

    /// Canonical content identity of every model fact this loader resolved.
    pub fn model_identity(&self) -> [u8; 32] {
        self.src.model_identity()
    }

    // ---- reading ---------------------------------------------------------

    /// One dense tensor by name, widened to f32.
    pub fn tensor(&self, name: &str) -> Result<Loaded> {
        let t0 = Instant::now();
        let leaf = self.src.leaf(name)?;
        let data = leaf.to_f32();
        self.note(name, leaf.bytes.len() as u64, (data.len() * 4) as u64, t0);
        Ok(Loaded {
            data,
            shape: leaf.shape(),
        })
    }

    /// One dense tensor's STORED bytes, in the element type the pile holds them
    /// in, un-widened and un-copied.
    ///
    /// The counterpart of [`Weights::tensor`], and the one rule 3 wants: 924 of
    /// this model's 968 dense leaves are BF16, and `tensor` turns every one of
    /// them into twice as many bytes of f32 on its way to a device that has a
    /// BF16 MMA. A caller that is going to multiply by the weight rather than
    /// read it takes this instead, binds the bytes where they lie, and the
    /// widening never happens.
    ///
    /// Charged through the same counters, with `host_bytes == file_bytes`,
    /// because that identity is exactly what distinguishes this path from the
    /// other one in the report.
    pub fn stored(&self, name: &str) -> Result<&super::pile::Leaf> {
        let t0 = Instant::now();
        let leaf = self.src.leaf(name)?;
        let n = leaf.bytes.len() as u64;
        self.note(name, n, n, t0);
        Ok(leaf)
    }

    /// One dense weight — widened once and then KEPT, when residency is on.
    ///
    /// Off, this is [`Weights::tensor`] with an `Arc` around it and nothing
    /// survives it. On, the first ask pays the read and the widen and every
    /// later one is a pointer copy.
    pub fn held(&self, name: &str) -> Result<Held> {
        let hit = self.resident.lock().expect("resident").get(name).cloned();
        if let Some(h) = hit {
            self.note_hit(name);
            return Ok(h);
        }
        let h = Arc::new(self.tensor(name)?);
        self.resident
            .lock()
            .expect("resident")
            .insert(name.to_string(), h.clone());
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
        let (a, b) = make()?;
        let (a, b) = (Arc::new(a), Arc::new(b));
        {
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
        self.src.is_nvfp4(base)
    }

    /// Whether this source's NVFP4 expert planes are in MMA-FRAGMENT ORDER.
    ///
    /// Threaded to every consumer of [`Weights::expert_packed`] that MULTIPLIES
    /// the bytes rather than merely counting or binding them, because the two
    /// layouts are the same bytes in a different order: a kernel reading the
    /// wrong one produces numbers, not an error.
    pub fn experts_swizzled(&self) -> bool {
        self.src.experts_swizzled()
    }

    /// One expert's packed planes, borrowed, undecoded.
    pub fn expert_packed(&self, base: &str, e: usize) -> Result<PackedSlab> {
        let t0 = Instant::now();
        let s = self.src.expert_packed(base, e)?;
        // Packed: nothing is widened, so host bytes are storage bytes.
        let moved = (s.codes.len() + s.scales.len()) as u64;
        self.note(base, moved, moved, t0);
        Ok(s)
    }

    /// One expert's BF16 plane, borrowed, unwidened.
    ///
    /// The counterpart of [`Weights::expert_packed`] for the one layer that was
    /// never quantised. Charged through the same counters at the same seam, so
    /// a per-token byte total covers both formats in one unit.
    pub fn expert_bf16(&self, base: &str, e: usize) -> Result<Bf16Slab> {
        let t0 = Instant::now();
        let s = self.src.expert_bf16(base, e)?;
        // Stored BF16, handed on as stored BF16: host bytes ARE storage bytes,
        // and that identity is the whole point of this lane.
        let moved = s.bytes.len() as u64;
        self.note(base, moved, moved, t0);
        Ok(s)
    }

    /// Read and VALIDATE every expert plane in `layers`, once, before the token
    /// loop — so that no decode step ever does.
    ///
    /// # What this is actually buying, which is not what it looks like
    ///
    /// It looks like prefetching. It is not; it is paying a HASH.
    /// `PileSnapshot::get` verifies a blob's BLAKE3 against the handle that names
    /// it, because in a content-addressed store the name IS the hash and a read
    /// that skipped the check would be a read of something else. The result is
    /// cached per record for the life of the reader, so each blob costs it
    /// exactly once — but "once" was landing wherever the router happened to
    /// send a token first, and an expert is 9.4 MB (NVFP4 `w13`) to 33.6 MB
    /// (layer 2's BF16), so first touch cost 8-13 ms and a decode step paid it
    /// for however many of its 108 slabs it had not seen.
    ///
    /// Measured on the 20-layer head, steps 41..80 of an 80-token generation —
    /// long past any sensible notion of warm-up — the fastest passes spent 0.5
    /// ms in the loader and the slowest 274 ms, and that ONE variable explains
    /// the whole spread. It is also why an A/B between two builds that generate
    /// different tokens is not a comparison: different tokens route to
    /// different experts, so the two arms pay different amounts of a cost that
    /// has nothing to do with either.
    ///
    /// It is a full host read of every weight byte this node owns, which is a
    /// host path through the data plane by any definition. It cannot be moved
    /// to the device — the check is the storage layer's, and it is the reason
    /// to believe the bytes — so it moves to STARTUP instead, where it is paid
    /// once and where a node that then reads the SSD mid-decode is a node whose
    /// share does not fit.
    ///
    /// The slabs are dropped as they are validated. Holding them would cost a
    /// few MB of handles and change nothing: what persists is the reader's
    /// validation state, and the payload stays exactly where it was, in the
    /// mapping.
    pub fn warm_experts(
        &self,
        layers: std::ops::Range<usize>,
        mut progress: impl FnMut(usize, usize, u64),
    ) -> Result<(usize, u64)> {
        let keys = self.src.expert_keys_in(layers);
        let total = keys.len();
        let mut bytes = 0u64;
        for (i, (name, e)) in keys.iter().enumerate() {
            bytes += match self.src.expert_is_nvfp4(name, *e) {
                Some(true) => {
                    let q = self.src.expert_packed(name, *e as usize)?;
                    (q.codes.len() + q.scales.len()) as u64
                }
                _ => self.src.expert_bf16(name, *e as usize)?.bytes.len() as u64,
            };
            progress(i + 1, total, bytes);
        }
        Ok((total, bytes))
    }

    /// Every host mapping this source reads through, as `(base, len, keepalive)`.
    ///
    /// What a zero-copy lane registers with the GPU, ONCE — and for a pile that
    /// is exactly one registration, because a pile IS one file. The checkpoint
    /// reader had nine, one per shard, and the aliasing seam had to be written
    /// to span them; that generality is what went away with it.
    pub fn mappings(&self) -> Result<Vec<(usize, usize, Arc<dyn std::any::Any + Send + Sync>)>> {
        self.src.mappings()
    }

    /// Move this node's layer share and role-specific global tables into one
    /// anonymous startup allocation before any GPU handle can alias them.
    ///
    /// `shard` is the within-layer split: `Some(tp)` copies only this rank's
    /// half of every routed expert. See
    /// [`super::pile::PileSource::copy_share`] for why the cut belongs in this
    /// copy and nowhere else.
    pub fn copy_share(
        &mut self,
        layers: std::ops::Range<usize>,
        global_dense: &[&str],
        attention_bytes: u64,
        policy: super::budget::AdmissionPolicy,
        shard: Option<super::tp::Tp>,
    ) -> Result<(usize, usize, u64, u64)> {
        self.src
            .copy_share(layers, global_dense, attention_bytes, policy, shard)
    }

    // ---- what the source SAYS about itself --------------------------------

    /// One of the model's JSON sidecars, by the file name it had in the
    /// checkpoint it was imported from.
    ///
    /// The whole reason this is on `Weights` and not on the caller: `config.json`
    /// is not a weight, and for as long as reading it meant reading a checkpoint
    /// DIRECTORY, a pile held the 159 GiB and the run still depended on the 40
    /// KB. A pile that cannot answer this is not authoritative, it is merely
    /// large.
    ///
    /// Answered from FACTS — one entity per JSON scalar, see
    /// [`crate::jsonfacts`] — not from a stored copy of the file, so
    /// `text_config.hidden_size` is reachable as a query and the document
    /// reconstructed here is the same thing read a different way.
    pub fn document(&self, name: &str) -> Result<serde_json::Value> {
        crate::jsonfacts::load_document(self.src.facts(), self.src.reader(), name)
            .map_err(|e| anyhow::anyhow!("{e}"))
    }

    /// A sidecar that is TEXT rather than JSON — the chat template.
    ///
    /// It is a document whose root is a JSON string, so it goes through exactly
    /// the same storage and the same query as the others; there is no second
    /// mechanism for "files that are not JSON".
    pub fn text_document(&self, name: &str) -> Result<String> {
        match self.document(name)? {
            serde_json::Value::String(s) => Ok(s),
            other => anyhow::bail!(
                "{name} is stored as {} rather than a string",
                match other {
                    serde_json::Value::Object(_) => "an object",
                    serde_json::Value::Array(_) => "an array",
                    _ => "a scalar",
                }
            ),
        }
    }

    /// Every document this source can answer for.
    pub fn documents(&self) -> Vec<String> {
        crate::jsonfacts::documents(self.src.facts(), self.src.reader())
            .into_iter()
            .map(|(n, _)| n)
            .collect()
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
    /// Driven by what the SOURCE contains — `expert_keys` — never by what some
    /// layer range implies. An audit that enumerates the expected set cannot see
    /// a leaf that is there but unreachable, and cannot see one that is
    /// reachable but unexpected.
    pub fn for_each_bindable(
        &self,
        mut f: impl FnMut(&str, &'static str, &[u8]) -> Result<()>,
    ) -> Result<()> {
        for name in self.src.names() {
            let leaf = self.src.leaf(&name)?;
            f(&name, "dense", &leaf.bytes)?;
        }
        for (name, e) in self.src.expert_keys() {
            if self.src.expert_is_nvfp4(&name, e) == Some(true) {
                let q = self.src.expert_packed(&name, e as usize)?;
                f(&name, "codes", &q.codes)?;
                f(&name, "scales", &q.scales)?;
            } else {
                let l = self.src.expert_bf16(&name, e as usize)?;
                f(&name, "expert-bf16", &l.bytes)?;
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
            (
                a.0 + e.calls,
                a.1 + e.hits,
                a.2 + e.file_bytes,
                a.3 + e.host_bytes,
                a.4 + e.nanos,
            )
        })
    }

    /// Zero the counters, so a per-token figure is a per-token figure.
    pub fn io_reset(&self) {
        self.io.lock().expect("io stats").clear();
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
