//! Totality gate for the Inkling checkpoint layout.
//!
//! Walks the real safetensors headers of an unmodified Inkling checkpoint and
//! asserts that `mary::models::inkling::layout` is a **bijection** onto it:
//! every tensor fills exactly one slot, every slot the config implies is filled
//! by exactly one tensor, with the shape and dtype the config predicts.
//!
//! Only header pages are read. The shards are memory-mapped and the weights are
//! never touched, so this never pulls 159 GB through the page cache.
//!
//! Six things are checked:
//!
//! 1. **Expert quantization is all-or-nothing per layer.** `config.json` does
//!    not say which MoE layers are NVFP4 — the released small checkpoint
//!    quantizes 39 of 40 and leaves layer 2 in BF16. Rather than hardcode that,
//!    the set is read off the checkpoint and the *structural* invariant is
//!    gated: a layer carries all four sidecars or none.
//! 2. **Slots against tensors, both ways.** No unfilled slot, no unmapped
//!    tensor, no shape or dtype disagreement.
//! 3. **The two mapping directions against each other.** `tensor_name` and
//!    `parse` are written independently and must compose to the identity on
//!    every name in the checkpoint.
//! 4. **The attention-kind split, re-derived from the config.** Which layers
//!    are sliding-window and which are global, printed rather than assumed.
//! 5. **The vision pyramid's arithmetic closes.** The last stage's input width
//!    must equal a whole temporal patch, and each grouping factor must be
//!    integral — so a wrong `hidden_dims` default cannot pass silently.
//! 6. **The router is as wide as `shared_expert_sink` implies.** The tensor is
//!    `[258, hidden]` on the small model, and a layout deriving it from
//!    `n_routed_experts` alone would be off by exactly `n_shared_experts`.
//!
//!   cargo run --release --features inkling --bin inkling_layout_gate -- <checkpoint dir>

use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use memmap2::Mmap;
use safetensors::SafeTensors;

use mary::models::inkling::config::{AttnKind, InklingConfig};
use mary::models::inkling::layout::{
    Dtype, ExpertMat, QuantPart, Shape, Slot, describe, for_each_slot,
};

struct HeaderEntry {
    dtype: Dtype,
    shape: Shape,
    shard: String,
}

fn read_headers(dir: &Path) -> Result<HashMap<String, HeaderEntry>> {
    let mut shards: Vec<PathBuf> = std::fs::read_dir(dir)
        .with_context(|| format!("reading {}", dir.display()))?
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| {
            p.file_name()
                .and_then(|n| n.to_str())
                .is_some_and(|n| n.ends_with(".safetensors"))
        })
        .collect();
    shards.sort();
    anyhow::ensure!(!shards.is_empty(), "no *.safetensors in {}", dir.display());

    let mut out: HashMap<String, HeaderEntry> = HashMap::new();
    for path in &shards {
        let shard = path.file_name().unwrap().to_string_lossy().to_string();
        let file = std::fs::File::open(path).with_context(|| format!("opening {shard}"))?;
        // SAFETY: the checkpoint is read-only and nothing else writes it.
        let mmap = unsafe { Mmap::map(&file) }.with_context(|| format!("mmapping {shard}"))?;
        let st = SafeTensors::deserialize(&mmap)
            .with_context(|| format!("parsing the header of {shard}"))?;
        for name in st.names() {
            let view = st.tensor(name)?;
            let debug = format!("{:?}", view.dtype());
            let dtype = Dtype::from_safetensors_debug(&debug)
                .with_context(|| format!("{name} holds an unhandled dtype {debug}"))?;
            let entry = HeaderEntry {
                dtype,
                shape: Shape::new(view.shape()),
                shard: shard.clone(),
            };
            if let Some(prev) = out.insert(name.to_string(), entry) {
                anyhow::bail!("{name} appears in both {} and {shard}", prev.shard);
            }
        }
    }
    Ok(out)
}

/// Read off which MoE layers are NVFP4, and gate the all-or-nothing invariant.
fn quantized_layers(headers: &HashMap<String, HeaderEntry>, fails: &mut usize) -> BTreeSet<usize> {
    // layer -> (weights seen, sidecars seen)
    let mut seen: BTreeMap<usize, (usize, usize)> = BTreeMap::new();
    for name in headers.keys() {
        let Some(rest) = name.strip_prefix("model.llm.layers.") else {
            continue;
        };
        let Some((idx, tail)) = rest.split_once('.') else {
            continue;
        };
        if !tail.starts_with("mlp.experts.") {
            continue;
        }
        let Ok(layer) = idx.parse::<usize>() else {
            continue;
        };
        let e = seen.entry(layer).or_insert((0, 0));
        if tail.ends_with("w13_weight") || tail.ends_with("w2_weight") {
            e.0 += 1;
        } else {
            e.1 += 1;
        }
    }

    let mut quantized = BTreeSet::new();
    let mut bf16 = BTreeSet::new();
    for (&layer, &(weights, sidecars)) in &seen {
        // Two matrices, four sidecars each when quantized.
        match sidecars {
            0 => {
                bf16.insert(layer);
            }
            8 => {
                quantized.insert(layer);
            }
            n => {
                println!(
                    "  FAIL  layer {layer}: {n} expert sidecars, expected 0 (BF16) or 8 (NVFP4) \
                     — quantization is not all-or-nothing here"
                );
                *fails += 1;
            }
        }
        if weights != 2 {
            println!("  FAIL  layer {layer}: {weights} expert weight tensors, expected 2");
            *fails += 1;
        }
    }
    println!("  MoE layers examined : {}", seen.len());
    println!("  NVFP4 layers        : {}", quantized.len());
    println!(
        "  BF16 layers         : {} {:?}",
        bf16.len(),
        bf16.iter().copied().collect::<Vec<_>>()
    );
    quantized
}

fn main() -> Result<()> {
    let dir = std::env::args()
        .nth(1)
        .map(PathBuf::from)
        .context("usage: inkling_layout_gate <checkpoint dir>")?;

    let cfg_text = std::fs::read_to_string(dir.join("config.json"))
        .with_context(|| format!("reading {}/config.json", dir.display()))?;
    let cfg = InklingConfig::from_json(&cfg_text).context("parsing config.json")?;
    let headers = read_headers(&dir)?;

    let t = &cfg.text_config;
    println!("=== checkpoint ===");
    println!("  dir              : {}", dir.display());
    println!("  tensors examined : {}", headers.len());
    println!(
        "  llm layers {} | mtp {} | hidden {} | experts {}+{} shared",
        t.num_hidden_layers,
        cfg.mtp_config.num_nextn_predict_layers,
        t.hidden_size,
        t.n_routed_experts,
        t.n_shared_experts
    );
    anyhow::ensure!(
        !headers.is_empty(),
        "zero tensors examined — the gate would be vacuous"
    );

    let mut fails = 0usize;
    let mut checks = 0usize;

    println!("\n=== 1. expert quantization is all-or-nothing ===");
    let quantized = quantized_layers(&headers, &mut fails);
    checks += 1;

    println!("\n=== 2/3. bijection, and tensor_name . parse == id ===");
    let mut want: BTreeMap<String, (Shape, Dtype, Slot)> = BTreeMap::new();
    for_each_slot(&cfg, &quantized, |ts| {
        let name = ts.slot.tensor_name();
        if let Some((_, _, prev)) = want.insert(name.clone(), (ts.shape, ts.dtype, ts.slot)) {
            println!("  FAIL  two slots claim {name}: {prev:?} and {:?}", ts.slot);
        }
    });
    println!("  slots the config implies : {}", want.len());

    let mut unfilled = 0usize;
    let mut mismatched = 0usize;
    for (name, (shape, dtype, slot)) in &want {
        checks += 1;
        match headers.get(name) {
            None => {
                if unfilled < 8 {
                    println!("  FAIL  slot {slot:?} wants {name}, absent from the checkpoint");
                }
                unfilled += 1;
                fails += 1;
            }
            Some(h) => {
                if h.shape != *shape || h.dtype != *dtype {
                    if mismatched < 8 {
                        println!(
                            "  FAIL  {name}: checkpoint has {} {:?}, config predicts {} {:?}",
                            h.shape, h.dtype, shape, dtype
                        );
                    }
                    mismatched += 1;
                    fails += 1;
                }
            }
        }
    }

    let mut unmapped = 0usize;
    let mut roundtrip_bad = 0usize;
    for name in headers.keys() {
        checks += 1;
        match Slot::parse(name) {
            None => {
                if unmapped < 8 {
                    println!("  FAIL  checkpoint tensor {name} maps to no slot");
                }
                unmapped += 1;
                fails += 1;
            }
            Some(slot) => {
                if slot.tensor_name() != *name {
                    if roundtrip_bad < 8 {
                        println!(
                            "  FAIL  round trip: {name} -> {slot:?} -> {}",
                            slot.tensor_name()
                        );
                    }
                    roundtrip_bad += 1;
                    fails += 1;
                }
                // A parsed slot must also be one the config admits.
                if describe(&cfg, &quantized, slot).is_none() {
                    if unmapped < 8 {
                        println!(
                            "  FAIL  {name} parses to {slot:?}, which the config does not admit"
                        );
                    }
                    unmapped += 1;
                    fails += 1;
                }
            }
        }
    }
    println!("  unfilled slots      : {unfilled}");
    println!("  shape/dtype mismatch: {mismatched}");
    println!("  unmapped tensors    : {unmapped}");
    println!("  round-trip failures : {roundtrip_bad}");

    println!("\n=== 4. attention-kind split ===");
    let local: Vec<usize> = (0..t.num_hidden_layers)
        .filter(|&i| t.attn_kind(i) == AttnKind::Local)
        .collect();
    let global: Vec<usize> = (0..t.num_hidden_layers)
        .filter(|&i| t.attn_kind(i) == AttnKind::Global)
        .collect();
    println!(
        "  local  (window {}) : {} layers",
        t.sliding_window_size,
        local.len()
    );
    println!(
        "  global             : {} layers {:?}",
        global.len(),
        global
    );
    checks += 1;
    if local.len() + global.len() != t.num_hidden_layers {
        println!("  FAIL  the split does not cover every layer");
        fails += 1;
    }

    println!("\n=== 5. vision pyramid arithmetic ===");
    let vc = &cfg.vision_config;
    let per_frame = (vc.patch_size / vc.subpatch_size).pow(2);
    println!(
        "  sub-patch {}x{}x{} = {} elems",
        vc.subpatch_size,
        vc.subpatch_size,
        vc.n_channels,
        vc.subpatch_elems()
    );
    println!("  sub-patches per frame : {per_frame}");
    println!("  whole temporal patch  : {} elems", vc.patch_elems());
    checks += 1;
    if vc.patch_size % vc.subpatch_size != 0 {
        println!(
            "  FAIL  patch {} is not a whole number of sub-patches",
            vc.patch_size
        );
        fails += 1;
    }
    // The last stage must consume exactly one temporal patch's worth.
    let last = Slot::Vision(mary::models::inkling::layout::VisionPart::Linear(
        vc.n_layers - 1,
    ));
    checks += 1;
    match describe(&cfg, &quantized, last) {
        Some((shape, _)) if shape.dims() == [vc.decoder_dmodel, vc.patch_elems()].as_slice() => {
            println!(
                "  final stage {} -> {} : closes",
                vc.patch_elems(),
                vc.decoder_dmodel
            );
        }
        other => {
            println!(
                "  FAIL  final vision stage is {other:?}, expected [{}, {}]",
                vc.decoder_dmodel,
                vc.patch_elems()
            );
            fails += 1;
        }
    }

    println!("\n=== 6. router width follows shared_expert_sink ===");
    println!(
        "  shared_expert_sink={} -> gate rows {} (routed {} + shared {})",
        t.shared_expert_sink,
        t.gate_rows(),
        t.n_routed_experts,
        if t.shared_expert_sink {
            t.n_shared_experts
        } else {
            0
        }
    );
    checks += 1;
    let a_moe = (0..t.num_hidden_layers).find(|&i| !t.is_dense(i));
    if let Some(i) = a_moe {
        let name = Slot::Llm(
            i,
            mary::models::inkling::layout::BlockPart::Mlp(
                mary::models::inkling::layout::MlpPart::Moe(
                    mary::models::inkling::layout::MoePart::GateWeight,
                ),
            ),
        )
        .tensor_name();
        match headers.get(&name) {
            Some(h) if h.shape.dims() == [t.gate_rows(), t.hidden_size].as_slice() => {
                println!("  {name} is {} : matches", h.shape)
            }
            Some(h) => {
                println!(
                    "  FAIL  {name} is {}, config implies [{}, {}]",
                    h.shape,
                    t.gate_rows(),
                    t.hidden_size
                );
                fails += 1;
            }
            None => {
                println!("  FAIL  {name} absent");
                fails += 1;
            }
        }
    }

    // A sanity check on the NVFP4 sidecar family, so the count is never zero.
    let sidecars = headers
        .keys()
        .filter(|n| {
            QuantPart::sidecars()
                .iter()
                .any(|q| n.ends_with(q.suffix()))
        })
        .count();
    let packed = headers
        .keys()
        .filter(|n| {
            [ExpertMat::W13, ExpertMat::W2]
                .iter()
                .any(|m| n.ends_with(&format!("experts.{}", m.suffix())))
        })
        .count();
    println!("\n=== corpus sanity ===");
    println!("  expert weight tensors : {packed}");
    println!("  quantization sidecars : {sidecars}");

    println!("\n=== verdict ===");
    println!("  checks: {checks}");
    if fails == 0 {
        println!("GATE PASSED — {checks} checks, layout is a bijection onto the checkpoint");
        Ok(())
    } else {
        println!("GATE FAILED — {checks} checks, {fails} FAILURES");
        std::process::exit(1);
    }
}
