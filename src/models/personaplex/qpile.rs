//! Derived RUNTIME-FORMAT sibling piles for the PersonaPlex realtime lane —
//! the zero-copy seam that deletes the quantize/fold/convert pass from the
//! load path.
//!
//! ## Why
//!
//! The canonical `models/personaplex.pile` stores the checkpoint as exact
//! f32 leaves (bit-exactness gate, `personaplex_persist`). The realtime
//! stages don't RUN f32: the Metal temporal stack streams packed q4/q8
//! words plus f16 scales (or raw f16 rows), and the CPU depformer wants
//! pre-sliced, alpha-folded step-major operands. Producing those at load
//! is a full read-transform pass over ~31 GB — not a copy, a COMPUTE pass
//! ("encoded layer N/32"). Zero-copy therefore means: run that pass ONCE
//! (`personaplex_persist --derive-fmt` / `--derive-depth`), persist the
//! exact bytes the kernels consume as 256-aligned (V3) leaves in a
//! sibling pile, then mmap them — `register_external_aliased` for GPU
//! buffers, direct slices for the CPU gemv operands.
//!
//! ## The siblings (auto-discovered next to the canonical pile)
//!
//! - `<stem>_q4.pile` / `<stem>_q8.pile` / `<stem>_f16.pile` — the temporal
//!   stack per [`WeightFmt`]: entity `personaplex_<fmt>` with leaves
//!   `t.{layer}.{qkv|o|gateup|down}` — the FUSED kernel layouts (qkv
//!   row-concatenated with q/k de-interleaved; gateup pair-interleaved for
//!   the in-kernel SwiGLU epilogue; see `temporal_metal::layer_mats_f32`)
//!   packed per fmt with the logical `[out, in]` shape — plus
//!   `t.{layer}.norm{1,2}` + `t.out_norm` (squeezed `[4096]` f32 alphas),
//!   `t.head_f16` (raw f16 rows) and `t.head_q4` (packed q4) — both logit
//!   heads stay loaded for the A/B.
//! - `<stem>_depth.pile` — the depformer's CPU operands in BOTH storage
//!   widths: entity `personaplex_depth` with `d.f32.{l}.qkv` / `.gate_up`
//!   (fold-applied, step-fused) + `d.f32.dep_in` (the 16 conditioning
//!   projections fused for the one-gemv-per-frame trick), and the full
//!   f16-converted operand set `d.f16.{l}.{qkv|o|gate_up|down}` +
//!   `d.f16.heads` + `d.f16.dep_in`. Operands the f32 mode consumes
//!   UNMODIFIED (o / down / heads / embeddings) are NOT duplicated — the
//!   runtime maps them straight from the canonical pile.
//!
//! ## The format marker IS the kernel ABI version
//!
//! Every derived model entity carries `attrs::format_marker`, a minted id
//! naming the exact packed layout (nibble/byte order, scale grouping, fold
//! conventions, leaf naming). Loaders accept a sibling only when the marker
//! equals their compiled-in constant; any layout change mints a NEW id
//! (`trible genid`), bumps the constant, and re-derives. The siblings are
//! regenerable exhaust — the canonical pile stays the single source of
//! truth and is only ever READ here.
//!
//! ## Identity contract
//!
//! The derive shares the exact transform code with the quantize-at-load
//! path ([`temporal_metal::layer_mats_f32`] + the quantizers,
//! [`depth_fast::fold_qkv_step`]/[`fold_gate_up`] + the f16 conversion), so
//! a sibling-loaded model is BYTE-IDENTICAL to a quantize-at-load one: the
//! rt gates must produce identical numbers with and without the sibling
//! (`MARY_PPLX_MATERIALIZE=1` forces the fallback). Any difference is a
//! derive bug, not acceptable drift.

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use ed25519_dalek::SigningKey;
use triblespace::prelude::*;

use super::config as cfg;
use super::depth_fast::{self, DepthFast};
use super::temporal_metal::{self, TemporalMetal, WeightFmt};
use crate::f16enc::F16Array;
use crate::format::{
    attrs, put_raw, put_raw_f16, put_raw_q4, put_raw_q8, U32Array, U64Array,
};
use crate::ingest::read_shape;
use crate::nn::q4::quantize_q4;
use crate::nn::weight_loader::WeightLoader;

/// The derived sibling of a weights pile for a runtime format tag:
/// `models/personaplex.pile` + `"q8"` → `models/personaplex_q8.pile`.
/// Applies one local naming convention across the whole format axis (`q4` /
/// `q8` / `f16` for the temporal stack, `depth` for depformer operands).
pub fn derived_sibling_path(pile_path: &Path, tag: &str) -> PathBuf {
    let stem = pile_path
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("weights");
    pile_path.with_file_name(format!("{stem}_{tag}.pile"))
}

/// The sibling-file tag and model-entity suffix for a temporal format.
pub fn fmt_tag(fmt: WeightFmt) -> &'static str {
    match fmt {
        WeightFmt::Q4 => "q4",
        WeightFmt::Q8 => "q8",
        WeightFmt::F16 => "f16",
    }
}

/// Layout-version marker for the temporal sibling of `fmt` — one minted id
/// (`trible genid`) per packed layout; bump = new id + re-derive.
///
/// v2 (2026-07-12, current): the FUSED kernel layouts of the frametax lane —
/// `qkv` `[3·DIM, DIM]` row-concatenated (q/k de-interleaved) and `gateup`
/// `[2·FFN, DIM]` pair-interleaved rows (even = gate_j, odd = up_j) for the
/// in-kernel SwiGLU epilogue. v1 (same day, retired unshipped: the 7
/// split-matrix layout) minted `982B18242F107702A9ECC86C6B74502C` /
/// `B8F70AA23223E6F7D111F543D2AB08A3` / `5E916E4D41D72BF7CEA505B9D20B859F`
/// for q4/q8/f16 — recorded so the ids stay burned.
pub fn temporal_marker(fmt: WeightFmt) -> Id {
    match fmt {
        WeightFmt::Q4 => id_hex!("2178B37C4ED9201B73A753C43A991547"),
        WeightFmt::Q8 => id_hex!("EBA20BF70DF03F8704AA1F9BCFB0C837"),
        WeightFmt::F16 => id_hex!("81E9D1853AB8EC327DEFE33C13B1B3B4"),
    }
}

/// Layout-version marker for the depth sibling — minted 2026-07-12.
pub fn depth_marker() -> Id {
    id_hex!("81418AFBE0F880593E44750E1D318C41")
}

/// A leaf of a derived pile.
///
/// The two quantized arms are still THREE handles bound by nothing — packed
/// words, group scales, and a shape stated apart from both — which is the
/// arrangement the dense leaves have left. They keep it until a block-scaled
/// `TensorElement` exists for q4_0/q8_0, at which point the scales travel
/// inside the payload they scale and the shape inside the header, exactly as
/// they do for the dense ones here.
#[derive(Clone)]
pub enum QLeaf {
    /// Packed q4_0 nibble words + f16 group scales + logical shape.
    Q4(
        Inline<inlineencodings::Handle<U32Array>>,
        Inline<inlineencodings::Handle<F16Array>>,
        Inline<inlineencodings::Handle<U64Array>>,
    ),
    /// Packed q8_0 biased-byte words + f16 group scales + logical shape.
    Q8(
        Inline<inlineencodings::Handle<U32Array>>,
        Inline<inlineencodings::Handle<F16Array>>,
        Inline<inlineencodings::Handle<U64Array>>,
    ),
    /// An unquantized leaf, f16 or f32, carrying its own shape.
    Dense(crate::leaf::Leaf),
}

/// An OPENED derived sibling: the leaf index plus the long-lived pile reader
/// whose mmap the zero-copy tensors bind, plus the entity's format marker. No
/// weight is copied — every payload is a view over that mapping.
pub struct QPile {
    pub index: HashMap<String, QLeaf>,
    pub reader: triblespace::core::repo::pile::PileReader,
    pub marker: Option<Id>,
}

impl QPile {
    fn leaf(&self, name: &str) -> anyhow::Result<&QLeaf> {
        self.index
            .get(name)
            .ok_or_else(|| anyhow::anyhow!("derived pile missing leaf {name}"))
    }

    /// An unquantized leaf of the requested width, or a diagnostic.
    fn dense(&self, name: &str, want: crate::leaf::Elem) -> anyhow::Result<&crate::leaf::Leaf> {
        match self.leaf(name)? {
            QLeaf::Dense(leaf) if leaf.elem() == want => Ok(leaf),
            QLeaf::Dense(leaf) => anyhow::bail!(
                "{name}: expected a {want:?} leaf, found {:?}",
                leaf.elem()
            ),
            _ => anyhow::bail!("{name}: not a {want:?} leaf"),
        }
    }

    fn get_bytes<T: blobencodings::ArrayElement>(
        &self,
        h: Inline<inlineencodings::Handle<blobencodings::Array<T>>>,
        what: &str,
    ) -> anyhow::Result<anybytes::Bytes> {
        self.reader
            .get(h)
            .map_err(|e| anyhow::anyhow!("{what}: {e:?}"))
    }

    /// Packed q4 leaf → (nibble-word bytes, scale bytes, logical `[out, in]`).
    pub fn bytes_q4(
        &self,
        name: &str,
    ) -> anyhow::Result<(anybytes::Bytes, anybytes::Bytes, Vec<usize>)> {
        match self.leaf(name)? {
            QLeaf::Q4(d, sc, sh) => {
                let shape = read_shape(&self.reader, *sh);
                anyhow::ensure!(shape.len() == 2, "{name}: q4 leaf is not a matrix");
                Ok((self.get_bytes(*d, name)?, self.get_bytes(*sc, name)?, shape))
            }
            _ => anyhow::bail!("{name}: not a q4 leaf"),
        }
    }

    /// Packed q8 leaf → (biased-byte-word bytes, scale bytes, `[out, in]`).
    pub fn bytes_q8(
        &self,
        name: &str,
    ) -> anyhow::Result<(anybytes::Bytes, anybytes::Bytes, Vec<usize>)> {
        match self.leaf(name)? {
            QLeaf::Q8(d, sc, sh) => {
                let shape = read_shape(&self.reader, *sh);
                anyhow::ensure!(shape.len() == 2, "{name}: q8 leaf is not a matrix");
                Ok((self.get_bytes(*d, name)?, self.get_bytes(*sc, name)?, shape))
            }
            _ => anyhow::bail!("{name}: not a q8 leaf"),
        }
    }

    /// Raw f16 leaf → (bytes, shape).
    pub fn bytes_f16(&self, name: &str) -> anyhow::Result<(anybytes::Bytes, Vec<usize>)> {
        let leaf = self.dense(name, crate::leaf::Elem::F16)?;
        Ok((leaf.payload().clone(), leaf.shape()))
    }

    /// Exact f32 leaf → (bytes, shape).
    pub fn bytes_f32(&self, name: &str) -> anyhow::Result<(anybytes::Bytes, Vec<usize>)> {
        let leaf = self.dense(name, crate::leaf::Elem::F32)?;
        Ok((leaf.payload().clone(), leaf.shape()))
    }

    /// Zero-copy typed view of an f32 leaf (CPU consumption).
    pub fn view_f32(&self, name: &str) -> anyhow::Result<(anybytes::View<[f32]>, Vec<usize>)> {
        let (b, s) = self.bytes_f32(name)?;
        let v = b
            .view::<[f32]>()
            .map_err(|e| anyhow::anyhow!("{name}: f32 view: {e:?}"))?;
        Ok((v, s))
    }

    /// Zero-copy view of an f16 leaf as raw bit patterns (`u16`) — the
    /// storage the NEON `hdot` kernel reads.
    pub fn view_u16(&self, name: &str) -> anyhow::Result<(anybytes::View<[u16]>, Vec<usize>)> {
        let (b, s) = self.bytes_f16(name)?;
        let v = b
            .view::<[u16]>()
            .map_err(|e| anyhow::anyhow!("{name}: u16 view: {e:?}"))?;
        Ok((v, s))
    }

    /// Open a derived sibling pile and index the model entity named
    /// `entity_name` (e.g. `personaplex_q8`). Reads only names/handles — no
    /// tensor data. The reader (and every view/alias resolved through it)
    /// keeps the mmap alive after the repository closes.
    pub fn open(path: &Path, entity_name: &str) -> anyhow::Result<Self> {
        let mut pile =
            Pile::open(path).map_err(|e| anyhow::anyhow!("open pile {path:?}: {e:?}"))?;
        // Read path: non-mutating load, NEVER amputate (see crate::persist).
        pile.refresh().map_err(|e| {
            anyhow::anyhow!(
                "pile {path:?} failed to load ({e:?}); refusing to auto-truncate on a \
                 read path — if the tail is a genuinely torn write, amputate explicitly \
                 with `trible pile amputate`"
            )
        })?;
        let mut repo = Repository::new(
            pile,
            SigningKey::generate(&mut rand::rngs::OsRng),
            TribleSet::new(),
        )
        .map_err(|e| anyhow::anyhow!("repo new: {e:?}"))?;
        let branch_id = repo
            .lookup_branch("main")
            .map_err(|e| anyhow::anyhow!("lookup main: {e:?}"))?
            .ok_or_else(|| anyhow::anyhow!("no 'main' branch in pile {path:?}"))?;
        let mut ws = repo
            .pull(branch_id)
            .map_err(|e| anyhow::anyhow!("pull main: {e:?}"))?;
        let head = ws
            .head()
            .ok_or_else(|| anyhow::anyhow!("'main' has no commits"))?;
        let tribles: TribleSet = ws
            .checkout(ancestors(head))
            .map_err(|e| anyhow::anyhow!("checkout: {e:?}"))?
            .facts()
            .clone();
        let reader = repo
            .storage_mut()
            .reader()
            .map_err(|e| anyhow::anyhow!("pile reader: {e:?}"))?;

        let mut model_id: Option<Id> = None;
        for (m, n) in find!(
            (m: Id, n: Inline<inlineencodings::Handle<blobencodings::LongString>>),
            pattern!(&tribles, [{ ?m @ attrs::model_name: ?n }])
        ) {
            let name: anybytes::View<str> = reader
                .get(n)
                .map_err(|e| anyhow::anyhow!("model name blob: {e:?}"))?;
            if &*name == entity_name {
                model_id = Some(m);
            }
        }
        let model_id = model_id
            .ok_or_else(|| anyhow::anyhow!("no model entity '{entity_name}' in pile {path:?}"))?;

        let marker = find!(
            (mk: Id),
            pattern!(&tribles, [{ model_id @ attrs::format_marker: ?mk }])
        )
        .next()
        .map(|(mk,)| mk);

        let mut index = HashMap::new();
        let members: Vec<_> =
            find!((m: Id), pattern!(&tribles, [{ model_id @ attrs::member: ?m }])).collect();
        for (mid,) in members {
            let (name_h, w_id) = find!(
                (n, w: Id),
                pattern!(&tribles, [{ mid @ attrs::safetensor_path: ?n, attrs::weight: ?w }])
            )
            .next()
            .ok_or_else(|| anyhow::anyhow!("member without name/weight"))?;
            let name: anybytes::View<str> = reader
                .get(name_h)
                .map_err(|e| anyhow::anyhow!("leaf name blob: {e:?}"))?;
            let leaf = if let Some((d, sc, s)) = find!(
                (d, sc, s),
                pattern!(&tribles, [{ w_id @ attrs::data_q4: ?d, attrs::q_scales: ?sc, attrs::shape: ?s }])
            )
            .next()
            {
                QLeaf::Q4(d, sc, s)
            } else if let Some((d, sc, s)) = find!(
                (d, sc, s),
                pattern!(&tribles, [{ w_id @ attrs::data_q8: ?d, attrs::q_scales: ?sc, attrs::shape: ?s }])
            )
            .next()
            {
                QLeaf::Q8(d, sc, s)
            } else if let Some(dense) = crate::leaf::resolve(&tribles, &reader, w_id)? {
                // Typed first, two-blob second — one seam for both, so a
                // derived pile written before the typed encoding and one
                // written after both load.
                QLeaf::Dense(dense)
            } else {
                anyhow::bail!("leaf {} carries no known data attribute", &*name);
            };
            index.insert(name.to_string(), leaf);
        }
        repo.close()
            .map_err(|e| anyhow::anyhow!("close pile: {e:?}"))?;
        Ok(Self {
            index,
            reader,
            marker,
        })
    }
}

// ---------------------------------------------------------------------------
// auto-discovery loaders (the runtime seam)
// ---------------------------------------------------------------------------

/// Load the temporal stack with sibling auto-discovery: the zero-copy mmap
/// path when `<stem>_<fmt>.pile` exists, is marker-current and aliasable;
/// the quantize-at-load path otherwise (and always under
/// `MARY_PPLX_MATERIALIZE=1`, the A/B switch). Both paths produce
/// byte-identical models.
pub fn temporal_auto(pile: &Path, loader: &WeightLoader, fmt: WeightFmt) -> TemporalMetal {
    let tag = fmt_tag(fmt);
    if std::env::var("MARY_PPLX_MATERIALIZE").is_ok() {
        eprintln!("[mary] MARY_PPLX_MATERIALIZE set — temporal quantize-at-load path");
        return TemporalMetal::load(loader, fmt);
    }
    let sib_path = derived_sibling_path(pile, tag);
    if !sib_path.exists() {
        eprintln!(
            "[mary] no derived sibling {sib_path:?} — temporal quantize-at-load \
             (derive it: personaplex_persist --derive-fmt {tag} {pile:?})"
        );
        return TemporalMetal::load(loader, fmt);
    }
    match QPile::open(&sib_path, &format!("personaplex_{tag}"))
        .and_then(|sib| TemporalMetal::load_zero_copy(&sib, loader, fmt))
    {
        Ok(tm) => {
            eprintln!("[mary] temporal {fmt:?}: ZERO-COPY mmap from {sib_path:?}");
            tm
        }
        Err(e) => {
            eprintln!(
                "[mary] derived sibling {sib_path:?} unusable ({e}); temporal \
                 quantize-at-load fallback"
            );
            TemporalMetal::load(loader, fmt)
        }
    }
}

/// Load the depformer with sibling auto-discovery — the depth twin of
/// [`temporal_auto`] (`<stem>_depth.pile`, both storage widths served by the
/// one sibling).
pub fn depth_auto(pile: &Path, loader: &WeightLoader, depth_f16: bool) -> DepthFast {
    if std::env::var("MARY_PPLX_MATERIALIZE").is_ok() {
        eprintln!("[mary] MARY_PPLX_MATERIALIZE set — depformer fold-at-load path");
        return DepthFast::load(loader, depth_f16);
    }
    let sib_path = derived_sibling_path(pile, "depth");
    if !sib_path.exists() {
        eprintln!(
            "[mary] no derived sibling {sib_path:?} — depformer fold-at-load \
             (derive it: personaplex_persist --derive-depth {pile:?})"
        );
        return DepthFast::load(loader, depth_f16);
    }
    match QPile::open(&sib_path, "personaplex_depth")
        .and_then(|sib| DepthFast::load_zero_copy(&sib, loader, depth_f16))
    {
        Ok(df) => {
            eprintln!(
                "[mary] depformer ({}): ZERO-COPY mmap from {sib_path:?}",
                if depth_f16 { "f16" } else { "f32" }
            );
            df
        }
        Err(e) => {
            eprintln!(
                "[mary] derived sibling {sib_path:?} unusable ({e}); depformer \
                 fold-at-load fallback"
            );
            DepthFast::load(loader, depth_f16)
        }
    }
}

// ---------------------------------------------------------------------------
// derive (canonical f32 pile → runtime-format sibling; src strictly READ)
// ---------------------------------------------------------------------------

/// Open the destination sibling for appending (create if absent) and hand
/// back the repo + workspace on `main`.
fn open_dst(
    dst: &Path,
) -> anyhow::Result<(Repository<Pile>, triblespace::core::repo::Workspace<Pile>)> {
    if !dst.exists() {
        eprintln!("[derive] pile {dst:?} does not exist — creating a NEW empty pile");
        std::fs::File::create(dst).map_err(|e| anyhow::anyhow!("create pile {dst:?}: {e}"))?;
    }
    let mut pile = Pile::open(dst).map_err(|e| anyhow::anyhow!("open pile {dst:?}: {e:?}"))?;
    // Non-mutating load; NEVER amputate (see crate::persist).
    pile.refresh().map_err(|e| {
        anyhow::anyhow!(
            "pile {dst:?} failed to load ({e:?}); refusing to auto-truncate — \
             if the tail is a genuinely torn write, amputate explicitly with \
             `trible pile amputate`"
        )
    })?;
    let mut repo = Repository::new(
        pile,
        SigningKey::generate(&mut rand::rngs::OsRng),
        TribleSet::new(),
    )
    .map_err(|e| anyhow::anyhow!("repo new: {e:?}"))?;
    let branch_id = match repo
        .lookup_branch("main")
        .map_err(|e| anyhow::anyhow!("lookup main: {e:?}"))?
    {
        Some(id) => id,
        None => *repo
            .create_branch("main", None)
            .map_err(|e| anyhow::anyhow!("create main: {e:?}"))?,
    };
    let ws = repo
        .pull(branch_id)
        .map_err(|e| anyhow::anyhow!("pull main: {e:?}"))?;
    Ok((repo, ws))
}

/// One derived member: leaf fragment + name → member entity facts.
fn add_member(
    repo: &mut Repository<Pile>,
    facts: &mut TribleSet,
    members: &mut Vec<Id>,
    name: &str,
    leaf: Fragment,
) -> anyhow::Result<()> {
    let leaf_id = leaf.root().expect("leaf root");
    *facts += leaf.into_facts();
    let name_h = repo
        .storage_mut()
        .put::<blobencodings::LongString, _>(name.to_string())
        .map_err(|e| anyhow::anyhow!("{name}: put name blob: {e:?}"))?;
    let m = entity! { _ @ attrs::kind: "matrix", attrs::safetensor_path: name_h, attrs::weight: leaf_id };
    members.push(m.root().expect("module root"));
    *facts += m.into_facts();
    Ok(())
}

/// Derive the TEMPORAL sibling pile for `fmt` from the canonical exact-f32
/// pile: the 32×4 FUSED matvec weights packed exactly as the kernels stream
/// them (shared transform code with [`TemporalMetal::load`] — see the
/// identity contract in the module docs), the squeezed norm alphas, and
/// both logit heads. `src` is only ever read; `dst` is created/appended.
/// Returns `(leaf count, payload bytes)`.
pub fn derive_temporal_pile(
    src: &Path,
    dst: &Path,
    fmt: WeightFmt,
) -> anyhow::Result<(usize, u64)> {
    anyhow::ensure!(
        src.canonicalize()? != dst.canonicalize().unwrap_or_else(|_| dst.to_path_buf()),
        "src and dst are the same pile file {src:?}"
    );
    let loader = crate::persist::personaplex_loader(src)?;
    let (mut repo, mut ws) = open_dst(dst)?;
    let mut facts = TribleSet::new();
    let mut members: Vec<Id> = Vec::new();
    let (mut count, mut bytes) = (0usize, 0u64);
    let tag = fmt_tag(fmt);

    let mut put_mat = |repo: &mut Repository<Pile>,
                       facts: &mut TribleSet,
                       members: &mut Vec<Id>,
                       name: &str,
                       enc: &temporal_metal::Encoded,
                       out: usize,
                       inn: usize|
     -> anyhow::Result<u64> {
        let shape = [out as u64, inn as u64];
        let (leaf, sz) = match (fmt, enc) {
            (WeightFmt::Q4, temporal_metal::Encoded::Packed(wq, sc)) => (
                put_raw_q4(repo.storage_mut(), wq, sc, &shape)
                    .map_err(|e| anyhow::anyhow!("{name}: put q4 leaf: {e}"))?,
                (wq.len() * 4 + sc.len() * 2) as u64,
            ),
            (WeightFmt::Q8, temporal_metal::Encoded::Packed(wq, sc)) => (
                put_raw_q8(repo.storage_mut(), wq, sc, &shape)
                    .map_err(|e| anyhow::anyhow!("{name}: put q8 leaf: {e}"))?,
                (wq.len() * 4 + sc.len() * 2) as u64,
            ),
            (WeightFmt::F16, temporal_metal::Encoded::Half(h)) => {
                // Persist the ALREADY-CONVERTED halves (identical to the
                // upload bytes) — re-encode through put_raw_f16 would
                // double-convert, so store the halves directly.
                let d = repo
                    .storage_mut()
                    .put::<F16Array, _>(h.clone())
                    .map_err(|e| anyhow::anyhow!("{name}: put f16 data: {e:?}"))?;
                let s = repo
                    .storage_mut()
                    .put::<U64Array, _>(shape.to_vec())
                    .map_err(|e| anyhow::anyhow!("{name}: put shape: {e:?}"))?;
                (
                    entity! { _ @ attrs::data_f16: d, attrs::shape: s },
                    (h.len() * 2) as u64,
                )
            }
            _ => unreachable!("encode/persist format mismatch"),
        };
        add_member(repo, facts, members, name, leaf)?;
        Ok(sz)
    };

    for i in 0..cfg::NUM_LAYERS {
        let mats = temporal_metal::layer_mats_f32(&loader, i);
        let meta: Vec<(&'static str, usize, usize)> =
            mats.iter().map(|&(n, _, o, ii)| (n, o, ii)).collect();
        let encoded = temporal_metal::encode_batch(
            mats.into_iter().map(|(_, w, o, ii)| (w, o, ii)).collect(),
            fmt,
        );
        for (enc, (nm, o, ii)) in encoded.iter().zip(meta) {
            bytes += put_mat(
                &mut repo,
                &mut facts,
                &mut members,
                &format!("t.{i}.{nm}"),
                enc,
                o,
                ii,
            )?;
            count += 1;
        }
        for (alpha, nm) in [("norm1", "norm1"), ("norm2", "norm2")] {
            let a = temporal_metal::load_alpha(
                &loader,
                &format!("transformer.layers.{i}.{alpha}.alpha"),
            );
            let leaf = put_raw(repo.storage_mut(), &a, &[cfg::DIM as u64])
                .map_err(|e| anyhow::anyhow!("layer {i} {nm}: put alpha: {e}"))?;
            add_member(
                &mut repo,
                &mut facts,
                &mut members,
                &format!("t.{i}.{nm}"),
                leaf,
            )?;
            bytes += (a.len() * 4) as u64;
            count += 1;
        }
        eprint!(
            "\r[derive] temporal layer {:2}/{} packed ({tag})",
            i + 1,
            cfg::NUM_LAYERS
        );
    }
    eprintln!();

    // globals: out_norm alpha + BOTH logit heads (f16 production + q4 A/B).
    let onw = temporal_metal::load_alpha(&loader, "out_norm.alpha");
    let leaf = put_raw(repo.storage_mut(), &onw, &[cfg::DIM as u64])
        .map_err(|e| anyhow::anyhow!("out_norm: put alpha: {e}"))?;
    add_member(&mut repo, &mut facts, &mut members, "t.out_norm", leaf)?;
    bytes += (onw.len() * 4) as u64;
    count += 1;

    let (head, s) = loader.load_f32("text_linear.weight");
    anyhow::ensure!(s == vec![cfg::TEXT_LOGITS, cfg::DIM], "text_linear shape");
    let head_shape = [cfg::TEXT_LOGITS as u64, cfg::DIM as u64];
    let leaf = put_raw_f16(repo.storage_mut(), &head, &head_shape)
        .map_err(|e| anyhow::anyhow!("head_f16: put leaf: {e}"))?;
    add_member(&mut repo, &mut facts, &mut members, "t.head_f16", leaf)?;
    bytes += (head.len() * 2) as u64;
    count += 1;
    let (hq, hs) = quantize_q4(&head, cfg::TEXT_LOGITS, cfg::DIM);
    let leaf = put_raw_q4(repo.storage_mut(), &hq, &hs, &head_shape)
        .map_err(|e| anyhow::anyhow!("head_q4: put leaf: {e}"))?;
    add_member(&mut repo, &mut facts, &mut members, "t.head_q4", leaf)?;
    bytes += (hq.len() * 4 + hs.len() * 2) as u64;
    count += 1;

    let mn = repo
        .storage_mut()
        .put::<blobencodings::LongString, _>(format!("personaplex_{tag}"))
        .map_err(|e| anyhow::anyhow!("put entity name blob: {e:?}"))?;
    let marker = temporal_marker(fmt);
    let model = entity! { _ @
        attrs::model_name: mn,
        attrs::format_marker: marker,
        attrs::member*: members.iter()
    };
    facts += model.into_facts();

    ws.commit(
        facts,
        &format!("derive personaplex temporal {tag} runtime artifacts (marker {marker:?})"),
    );
    repo.push(&mut ws)
        .map_err(|e| anyhow::anyhow!("push: {e:?}"))?;
    repo.close()
        .map_err(|e| anyhow::anyhow!("close pile: {e:?}"))?;
    Ok((count, bytes))
}

/// Derive the DEPTH sibling pile from the canonical exact-f32 pile: the
/// depformer's transformed CPU operands in both storage widths (see the
/// module docs for exactly which operands are persisted vs mapped from the
/// canonical pile). Returns `(leaf count, payload bytes)`.
pub fn derive_depth_pile(src: &Path, dst: &Path) -> anyhow::Result<(usize, u64)> {
    anyhow::ensure!(
        src.canonicalize()? != dst.canonicalize().unwrap_or_else(|_| dst.to_path_buf()),
        "src and dst are the same pile file {src:?}"
    );
    let loader = crate::persist::personaplex_loader(src)?;
    let (mut repo, mut ws) = open_dst(dst)?;
    let mut facts = TribleSet::new();
    let mut members: Vec<Id> = Vec::new();
    let (mut count, mut bytes) = (0usize, 0u64);

    let n = cfg::WEIGHTS_PER_STEP;
    let (d, fh) = (cfg::DEP_DIM, cfg::DEP_FFN_HIDDEN);

    // Persist one fused operand in f32 and/or f16 width.
    let mut put_both = |repo: &mut Repository<Pile>,
                        facts: &mut TribleSet,
                        members: &mut Vec<Id>,
                        base: &str,
                        w: &[f32],
                        shape: &[u64],
                        f32_too: bool|
     -> anyhow::Result<u64> {
        let mut sz = 0u64;
        if f32_too {
            let leaf = put_raw(repo.storage_mut(), w, shape)
                .map_err(|e| anyhow::anyhow!("d.f32.{base}: put leaf: {e}"))?;
            add_member(repo, facts, members, &format!("d.f32.{base}"), leaf)?;
            sz += (w.len() * 4) as u64;
        }
        let leaf = put_raw_f16(repo.storage_mut(), w, shape)
            .map_err(|e| anyhow::anyhow!("d.f16.{base}: put leaf: {e}"))?;
        add_member(repo, facts, members, &format!("d.f16.{base}"), leaf)?;
        sz += (w.len() * 2) as u64;
        Ok(sz)
    };

    for l in 0..cfg::DEP_LAYERS {
        let src_l = format!("depformer.layers.{l}");
        let (in_proj, s) = loader.load_f32(&format!("{src_l}.self_attn.in_proj_weight"));
        anyhow::ensure!(s == vec![n * 3 * d, d], "{src_l}: in_proj_weight shape");
        let (a1, s) = loader.load_f32(&format!("{src_l}.norm1.alpha"));
        anyhow::ensure!(s == vec![1, 1, d], "{src_l}: norm1.alpha shape");
        let (a2, s) = loader.load_f32(&format!("{src_l}.norm2.alpha"));
        anyhow::ensure!(s == vec![1, 1, d], "{src_l}: norm2.alpha shape");

        // qkv: fold per step (the exact load-path transform), fuse step-major.
        let mut qkv_all = Vec::with_capacity(n * 3 * d * d);
        for t in 0..n {
            qkv_all.extend(depth_fast::fold_qkv_step(&in_proj, &a1, t));
        }
        drop(in_proj);
        bytes += put_both(
            &mut repo,
            &mut facts,
            &mut members,
            &format!("{l}.qkv"),
            &qkv_all,
            &[(n * 3 * d) as u64, d as u64],
            true,
        )?;
        count += 2;
        drop(qkv_all);

        // o: raw rows (f32 mode maps the canonical leaf) — f16 width only.
        let (out_proj, s) = loader.load_f32(&format!("{src_l}.self_attn.out_proj.weight"));
        anyhow::ensure!(s == vec![n * d, d], "{src_l}: out_proj shape");
        bytes += put_both(
            &mut repo,
            &mut facts,
            &mut members,
            &format!("{l}.o"),
            &out_proj,
            &[(n * d) as u64, d as u64],
            false,
        )?;
        count += 1;
        drop(out_proj);

        // gate_up: fold per step, fuse step-major.
        let mut gu_all = Vec::with_capacity(n * 2 * fh * d);
        for t in 0..n {
            let (gu, s) = loader.load_f32(&format!("{src_l}.gating.{t}.linear_in.weight"));
            anyhow::ensure!(s == vec![2 * fh, d], "{src_l}: gating.{t}.linear_in shape");
            gu_all.extend(depth_fast::fold_gate_up(gu, &a2));
        }
        bytes += put_both(
            &mut repo,
            &mut facts,
            &mut members,
            &format!("{l}.gate_up"),
            &gu_all,
            &[(n * 2 * fh) as u64, d as u64],
            true,
        )?;
        count += 2;
        drop(gu_all);

        // down: raw, fused step-major — f16 width only.
        let mut down_all = Vec::with_capacity(n * d * fh);
        for t in 0..n {
            let (dn, s) = loader.load_f32(&format!("{src_l}.gating.{t}.linear_out.weight"));
            anyhow::ensure!(s == vec![d, fh], "{src_l}: gating.{t}.linear_out shape");
            down_all.extend(dn);
        }
        bytes += put_both(
            &mut repo,
            &mut facts,
            &mut members,
            &format!("{l}.down"),
            &down_all,
            &[(n * d) as u64, fh as u64],
            false,
        )?;
        count += 1;
        eprint!(
            "\r[derive] depth layer {}/{} packed",
            l + 1,
            cfg::DEP_LAYERS
        );
    }
    eprintln!();

    // heads: raw, fused step-major — f16 width only.
    let mut heads_all = Vec::with_capacity(n * cfg::CARD * d);
    for t in 0..n {
        let (h, s) = loader.load_f32(&format!("linears.{t}.weight"));
        anyhow::ensure!(s == vec![cfg::CARD, d], "linears.{t} shape");
        heads_all.extend(h);
    }
    bytes += put_both(
        &mut repo,
        &mut facts,
        &mut members,
        "heads",
        &heads_all,
        &[(n * cfg::CARD) as u64, d as u64],
        false,
    )?;
    count += 1;
    drop(heads_all);

    // dep_in: the 16 conditioning projections FUSED (the one-gemv-per-frame
    // trick needs contiguity, which the canonical per-step leaves can't give).
    let mut dep_in_all = Vec::with_capacity(n * d * cfg::DIM);
    for t in 0..n {
        let (w, s) = loader.load_f32(&format!("depformer_in.{t}.weight"));
        anyhow::ensure!(s == vec![d, cfg::DIM], "depformer_in.{t} shape");
        dep_in_all.extend(w);
    }
    bytes += put_both(
        &mut repo,
        &mut facts,
        &mut members,
        "dep_in",
        &dep_in_all,
        &[(n * d) as u64, cfg::DIM as u64],
        true,
    )?;
    count += 2;
    drop(dep_in_all);

    let mn = repo
        .storage_mut()
        .put::<blobencodings::LongString, _>("personaplex_depth".to_string())
        .map_err(|e| anyhow::anyhow!("put entity name blob: {e:?}"))?;
    let marker = depth_marker();
    let model = entity! { _ @
        attrs::model_name: mn,
        attrs::format_marker: marker,
        attrs::member*: members.iter()
    };
    facts += model.into_facts();

    ws.commit(
        facts,
        &format!("derive personaplex depth runtime operands (marker {marker:?})"),
    );
    repo.push(&mut ws)
        .map_err(|e| anyhow::anyhow!("push: {e:?}"))?;
    repo.close()
        .map_err(|e| anyhow::anyhow!("close pile: {e:?}"))?;
    Ok((count, bytes))
}
