//! Totality gate for the Kimi-K3 checkpoint layout.
//!
//! Walks the real safetensors headers of an unmodified K3 checkpoint and
//! asserts that the name-to-slot mapping in `mary::models::k3::layout` is a
//! **bijection**: every tensor in the checkpoint fills exactly one slot, and
//! every slot the config implies is filled by exactly one tensor, with the
//! shape and dtype the config predicts.
//!
//! Nothing but headers is read. The shards are memory-mapped and only the JSON
//! header pages are ever touched, so this never pulls 1.5 TB through the page
//! cache and never writes to the checkpoint.
//!
//! Four things are checked, and the last two are the ones that matter:
//!
//! 1. **Headers against the index.** `model.safetensors.index.json` must name
//!    exactly the tensors the headers hold, in the shards it claims.
//! 2. **Slots against tensors, both ways.** No unfilled slot, no unmapped
//!    tensor, no shape or dtype disagreement.
//! 3. **The two mapping directions against each other.** `Slot::tensor_name`
//!    and `Slot::parse` are written independently; here they must compose to
//!    the identity on every name in the checkpoint. A typo in either one
//!    surfaces as a real failure rather than cancelling out.
//! 4. **The layer-index base, re-derived from the weights.** Which layers carry
//!    `A_log` (KDA-only) and which carry `q_a_proj` (MLA-only) is read off the
//!    checkpoint and compared against the config's answer — and against what a
//!    naive 0-based reading of `kda_layers`/`full_attn_layers` would give, so
//!    the size of the trap is printed rather than assumed.
//!
//!   cargo run --release --features k3,import --bin k3_layout_gate -- [<checkpoint dir>]

use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use memmap2::Mmap;
use safetensors::SafeTensors;

use mary::models::k3::config::AttnKind;
use mary::models::k3::layout::{for_each_slot, Dtype, LayerPart, MoePart, Shape, Slot};
use mary::models::k3::K3Config;

/// Everything the gate needs about one checkpoint tensor.
struct HeaderEntry {
    dtype: Dtype,
    shape: Shape,
    shard: String,
}

fn dtype_of(d: safetensors::Dtype) -> Result<Dtype> {
    Ok(match d {
        safetensors::Dtype::BF16 => Dtype::Bf16,
        safetensors::Dtype::F32 => Dtype::F32,
        safetensors::Dtype::U8 => Dtype::U8,
        other => anyhow::bail!("checkpoint holds an unexpected dtype {other:?}"),
    })
}

/// Read every shard's header. The mmap is dropped as soon as the entries are
/// copied out, so at most one shard is mapped at a time.
fn read_headers(dir: &Path) -> Result<HashMap<String, HeaderEntry>> {
    let mut shards: Vec<PathBuf> = std::fs::read_dir(dir)
        .with_context(|| format!("reading {}", dir.display()))?
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| {
            p.file_name()
                .and_then(|n| n.to_str())
                .is_some_and(|n| n.starts_with("model-") && n.ends_with(".safetensors"))
        })
        .collect();
    shards.sort();
    anyhow::ensure!(!shards.is_empty(), "no model-*.safetensors in {}", dir.display());

    let mut out: HashMap<String, HeaderEntry> = HashMap::new();
    for path in &shards {
        let shard = path.file_name().unwrap().to_string_lossy().to_string();
        let file = std::fs::File::open(path).with_context(|| format!("opening {shard}"))?;
        // SAFETY: the checkpoint is read-only and nothing else writes it.
        let mmap = unsafe { Mmap::map(&file) }.with_context(|| format!("mmapping {shard}"))?;
        let st = SafeTensors::deserialize(&mmap)
            .with_context(|| format!("parsing the safetensors header of {shard}"))?;
        for name in st.names() {
            let view = st.tensor(name)?;
            let entry = HeaderEntry {
                dtype: dtype_of(view.dtype())?,
                shape: Shape::new(view.shape()),
                shard: shard.clone(),
            };
            if let Some(prev) = out.insert(name.to_string(), entry) {
                anyhow::bail!("{name} appears in both {} and {shard}", prev.shard);
            }
        }
    }
    println!("shards read: {}", shards.len());
    Ok(out)
}

/// Cross-check the headers against `model.safetensors.index.json`.
fn check_index(dir: &Path, headers: &HashMap<String, HeaderEntry>) -> Result<Vec<String>> {
    let path = dir.join("model.safetensors.index.json");
    let text = std::fs::read_to_string(&path).with_context(|| format!("reading {}", path.display()))?;
    let index: serde_json::Value = serde_json::from_str(&text)?;
    let map = index["weight_map"]
        .as_object()
        .context("index has no weight_map object")?;

    let mut problems = Vec::new();
    for (name, shard) in map {
        match headers.get(name) {
            None => problems.push(format!("index names {name}, no header has it")),
            Some(e) if e.shard != shard.as_str().unwrap_or_default() => problems.push(format!(
                "index puts {name} in {shard}, header found it in {}",
                e.shard
            )),
            Some(_) => {}
        }
    }
    for name in headers.keys() {
        if !map.contains_key(name) {
            problems.push(format!("{name} is in a header but not in the index"));
        }
    }

    let claimed = index["metadata"]["total_size"].as_u64().unwrap_or(0);
    let actual: u64 = headers
        .values()
        .map(|e| (e.shape.numel() * e.dtype.size()) as u64)
        .sum();
    println!("index weight_map entries: {}", map.len());
    println!("tensor bytes: {actual} (index total_size {claimed})");
    if claimed != 0 && claimed != actual {
        problems.push(format!("index total_size {claimed} != summed tensor bytes {actual}"));
    }
    Ok(problems)
}

/// Re-derive which layers are KDA and which are MLA from the weights alone, and
/// compare against the config. This is the check that the 1-based layer lists
/// were shifted correctly; it does not trust the config's own arithmetic.
fn check_layer_kinds(cfg: &K3Config, headers: &HashMap<String, HeaderEntry>) -> Vec<String> {
    let t = &cfg.text_config;
    let mut from_weights: BTreeMap<usize, AttnKind> = BTreeMap::new();
    for name in headers.keys() {
        let Some(Slot::Layer { layer, part: LayerPart::Attn(p) }) = Slot::parse(name) else {
            continue;
        };
        // Only the kind-exclusive tensors vote; g_proj and o_proj are on both.
        if let Some(kind) = p.kind() {
            from_weights.insert(layer, kind);
        }
    }

    let mut problems = Vec::new();
    let mut kda = 0usize;
    let mut mla = 0usize;
    for layer in 0..t.num_hidden_layers {
        let want = t.attn_kind(layer);
        match from_weights.get(&layer) {
            None => problems.push(format!("layer {layer} carries no kind-exclusive attention tensor")),
            Some(&got) if got != want => {
                problems.push(format!("layer {layer}: config says {want:?}, weights say {got:?}"))
            }
            Some(_) => {}
        }
        match want {
            AttnKind::Kda => kda += 1,
            AttnKind::Mla => mla += 1,
        }
    }

    let mla_layers: Vec<usize> = (0..t.num_hidden_layers)
        .filter(|&l| t.attn_kind(l) == AttnKind::Mla)
        .collect();
    if problems.is_empty() {
        println!(
            "layer kinds: {kda} KDA, {mla} MLA — agreeing with the weights on all {} layers",
            t.num_hidden_layers,
        );
    } else {
        println!(
            "layer kinds: {kda} KDA, {mla} MLA per config — DISAGREEING with the weights on {} of {} layers",
            problems.len(),
            t.num_hidden_layers,
        );
    }
    println!("  MLA layers (0-based): {mla_layers:?}");

    // What the same lists would say if they were read as 0-based. Printed
    // because "off by one" understates it: the wrong reading is wrong about a
    // specific, countable number of layers, and it is not a small number.
    let zero_based: BTreeSet<usize> = t.linear_attn_config.full_attn_layers.iter().copied().collect();
    let one_based: BTreeSet<usize> = mla_layers.iter().copied().collect();
    let wrong = (0..t.num_hidden_layers)
        .filter(|l| zero_based.contains(l) != one_based.contains(l))
        .count();
    let out_of_range = zero_based.iter().filter(|&&l| l >= t.num_hidden_layers).count();
    println!(
        "  a 0-based reading of full_attn_layers would misclassify {wrong} of {} layers \
         and name {out_of_range} layer(s) that do not exist",
        t.num_hidden_layers,
    );

    problems
}

/// Prove the gate is able to fail.
///
/// A totality check that cannot go red is worth nothing. So before reporting a
/// pass, the whole comparison is re-run against a config whose two layer lists
/// have been *exchanged*. That config is still a valid partition of the layers,
/// so it parses and validates cleanly — it is simply wrong about which layers
/// are KDA — and every KDA-only and MLA-only tensor must then land in the wrong
/// place. Returns the number of problems it produced; zero means the checks
/// above are vacuous, and the gate says so rather than passing.
fn negative_control(cfg: &K3Config, headers: &HashMap<String, HeaderEntry>) -> usize {
    let mut bad = cfg.clone();
    let la = &mut bad.text_config.linear_attn_config;
    std::mem::swap(&mut la.kda_layers, &mut la.full_attn_layers);

    let mut expected: HashMap<String, (Shape, Dtype)> = HashMap::new();
    for_each_slot(&bad, |ts| {
        expected.insert(ts.name.clone(), (ts.shape, ts.dtype));
    });

    let mut problems = 0;
    for (name, (shape, dtype)) in &expected {
        match headers.get(name) {
            None => problems += 1,
            Some(h) => {
                if h.shape != *shape || h.dtype != *dtype {
                    problems += 1;
                }
            }
        }
    }
    for name in headers.keys() {
        if !expected.contains_key(name) {
            problems += 1;
        }
    }
    problems
}

fn plural(n: usize) -> &'static str {
    if n == 1 { "" } else { "s" }
}

fn report(label: &str, problems: &[String]) -> usize {
    if problems.is_empty() {
        println!("  {label}: none");
    } else {
        println!("  {label}: {} problem{}", problems.len(), plural(problems.len()));
        for p in problems.iter().take(10) {
            println!("      {p}");
        }
        if problems.len() > 10 {
            println!("      ... and {} more", problems.len() - 10);
        }
    }
    problems.len()
}

fn main() -> Result<()> {
    let dir = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "./kimi-k3".to_string());
    let dir = Path::new(&dir);
    println!("checkpoint: {}", dir.display());

    let cfg_text = std::fs::read_to_string(dir.join("config.json")).context("reading config.json")?;
    let cfg = K3Config::from_json(&cfg_text).map_err(anyhow::Error::msg)?;
    let t = &cfg.text_config;
    println!(
        "config: {} layers, hidden {}, vocab {}, {} experts ({} active) + {} shared, \
         MoE from layer {}, AttnRes period {:?} ({} checkpoints)",
        t.num_hidden_layers,
        t.hidden_size,
        t.vocab_size,
        t.num_experts,
        t.num_experts_per_token,
        t.num_shared_experts.unwrap_or(0),
        t.first_k_dense_replace,
        t.attn_res_block_size,
        t.attn_res_bank_size(),
    );

    let headers = read_headers(dir)?;
    println!("tensors in headers: {}", headers.len());

    let mut expected: HashMap<String, (Slot, Shape, Dtype)> = HashMap::new();
    let mut collisions = Vec::new();
    for_each_slot(&cfg, |ts| {
        if let Some((prev, _, _)) = expected.insert(ts.name.clone(), (ts.slot, ts.shape, ts.dtype)) {
            collisions.push(format!("{} is claimed by both {prev:?} and {:?}", ts.name, ts.slot));
        }
    });
    println!("slots in layout:    {}", expected.len());

    // --- direction 1: every slot filled ---
    let mut unfilled = Vec::new();
    let mut mismatched = Vec::new();
    for (name, (slot, shape, dtype)) in &expected {
        match headers.get(name) {
            None => unfilled.push(format!("{slot:?} wants {name}, checkpoint has no such tensor")),
            Some(h) => {
                if h.shape != *shape || h.dtype != *dtype {
                    mismatched.push(format!(
                        "{name}: layout says {shape} {dtype:?}, checkpoint has {} {:?}",
                        h.shape, h.dtype
                    ));
                }
            }
        }
    }

    // --- direction 2: every tensor mapped, structurally ---
    let mut unmapped = Vec::new();
    let mut roundtrip = Vec::new();
    for name in headers.keys() {
        match Slot::parse(name) {
            None => unmapped.push(format!("{name} parses to no slot")),
            Some(slot) => {
                if mary::models::k3::layout::describe(&cfg, slot).is_none() {
                    unmapped.push(format!("{name} parses to {slot:?}, which this config has no room for"));
                    continue;
                }
                let back = slot.tensor_name();
                if back != *name {
                    roundtrip.push(format!("{name} -> {slot:?} -> {back}"));
                }
                match expected.get(name) {
                    None => unmapped.push(format!("{name} parses to {slot:?} but no slot was enumerated for it")),
                    Some((s, _, _)) if *s != slot => {
                        roundtrip.push(format!("{name}: enumerated as {s:?}, parsed as {slot:?}"))
                    }
                    Some(_) => {}
                }
            }
        }
    }

    let index_problems = check_index(dir, &headers)?;
    let kind_problems = check_layer_kinds(&cfg, &headers);

    // --- census, by where in the model the tensors live ---
    let mut census: BTreeMap<&'static str, (usize, u64)> = BTreeMap::new();
    let mut params: u64 = 0;
    let mut scale_bytes: u64 = 0;
    for (name, (slot, shape, dtype)) in &expected {
        let bucket = match slot {
            Slot::Vision(_) => "vision tower",
            Slot::MmProjector(_) => "mm projector",
            Slot::Layer { part: LayerPart::Attn(p), .. } => match p.kind() {
                Some(AttnKind::Kda) => "decoder: KDA attention",
                Some(AttnKind::Mla) => "decoder: MLA attention",
                None => "decoder: attention out/gate",
            },
            Slot::Layer { part: LayerPart::DenseMlp(_), .. } => "decoder: dense MLP",
            Slot::Layer { part: LayerPart::Moe(m), .. } => {
                if matches!(m, MoePart::Expert { .. }) {
                    "decoder: routed experts"
                } else {
                    "decoder: MoE router/shared/latent"
                }
            }
            Slot::Layer { .. } => "decoder: norms + AttnRes",
            _ => "decoder: embed/head/final",
        };
        let e = census.entry(bucket).or_default();
        e.0 += 1;
        e.1 += (shape.numel() * dtype.size()) as u64;
        // MXFP4 packs two logical parameters per stored byte; the E8M0 scales
        // are metadata, not parameters, and are counted separately.
        if name.ends_with("weight_scale") {
            scale_bytes += shape.numel() as u64;
        } else if name.ends_with("weight_packed") {
            params += 2 * shape.numel() as u64;
        } else {
            params += shape.numel() as u64;
        }
    }
    println!("\ncensus (tensors / bytes):");
    for (bucket, (n, bytes)) in &census {
        println!("  {n:>7}  {:>7.2} GiB  {bucket}", *bytes as f64 / (1u64 << 30) as f64);
    }
    println!(
        "  parameters: {params} ({:.4} T), plus {scale_bytes} MXFP4 scale bytes",
        params as f64 / 1e12,
    );

    let control = negative_control(&cfg, &headers);

    println!("\nGATE:");
    let mut failures = 0;
    if control == 0 {
        println!(
            "  negative control: NO problems — swapping kda_layers and full_attn_layers \
             changed nothing, so every check below is vacuous"
        );
        failures += 1;
    } else {
        println!("  negative control (layer lists swapped): {control} problems, as it must");
    }
    failures += report("slot name collisions", &collisions);
    failures += report("unfilled slots (layout wants, checkpoint lacks)", &unfilled);
    failures += report("unmapped tensors (checkpoint has, layout lacks)", &unmapped);
    failures += report("shape/dtype mismatches", &mismatched);
    failures += report("name<->slot round-trip failures", &roundtrip);
    failures += report("header/index disagreements", &index_problems);
    failures += report("layer-kind disagreements", &kind_problems);

    // Not a failure, but not something to leave for a reader to notice: the
    // checkpoint and the shipped modelling code disagree about A_log's shape.
    // See the AttnPart::ALog docs in mary::models::k3::layout.
    println!("\nANOMALIES (recorded, not gated):");
    let a_log = headers
        .get("language_model.model.layers.0.self_attn.A_log")
        .map(|e| e.shape);
    println!(
        "  self_attn.A_log is {} in the checkpoint; modeling_kimi_linear.py declares \
         torch.empty(num_heads) = [{}], and the fla kernel indexes it per head (A_log + i_hv). \
         b_proj [{}, {}] and dt_bias [{}] both pin the head count at {}. A port that treats \
         A_log as one-entry-per-head reads the first {} of {} values and gets a wrong decay, \
         silently.",
        a_log.map(|s| s.to_string()).unwrap_or_else(|| "absent".into()),
        t.linear_attn_config.num_heads,
        t.linear_attn_config.num_heads,
        t.hidden_size,
        t.linear_attn_config.proj_dim(),
        t.linear_attn_config.num_heads,
        t.linear_attn_config.num_heads,
        t.linear_attn_config.head_dim,
    );

    if failures == 0 {
        println!("\nPASS — mapping is total in both directions.");
        Ok(())
    } else {
        println!("\nFAIL — {failures} problem{}.", plural(failures));
        std::process::exit(1);
    }
}
