//! `mxfp4_span_sweep` — how far the bit-exact MXFP4 → NVFP4 transcode
//! actually reaches across a checkpoint.
//!
//! [`mary::nn::mxfp4`] is exact whenever a tensor's E8M0 exponents span at
//! most E4M3FN's 18-octave power-of-two window. `mxfp4_gate` proves the codec
//! on three sampled experts; this answers the *other* question, which no
//! amount of care about the codec can settle: do the other 82,429 stay inside
//! the window too? A single wide tensor anywhere turns "bit-exact transcode"
//! from a property of the checkpoint into a property of three experts.
//!
//! It reads **only** the `weight_scale` planes (344 KB per tensor, 0.06% of
//! the 1.5 TB checkpoint) — no weight byte is touched and nothing is
//! dequantized, because the span question lives entirely in the scales.
//! Reads are grouped by shard and issued in ascending offset order so the
//! sweep is forward-only on a machine other jobs are sharing.
//!
//! ## Gate
//!
//! Before any span is reported, the tool re-derives — from its own
//! safetensors header parsing and its own offset arithmetic — both the
//! absolute byte offsets and the exponent ranges of the 18 tensors the
//! k3oracle measured, and requires exact agreement. Those offsets were
//! themselves cross-checked against the official `safetensors` library, so a
//! header-parsing or `abs = 8 + header_len + rel` mistake (the classic one)
//! stops the run instead of producing a plausible-looking sweep of the wrong
//! bytes. If the gate fails, no span is printed at all.
//!
//! Usage: `mxfp4_span_sweep <MODEL_DIR> [ORACLE_DIR] [EXPERTS_PER_LAYER|all]`
//! (defaults: the k3oracle vectors, 16 experts per layer, chosen on an even
//! stride that always includes the first and last).

use std::collections::BTreeMap;
use std::fs::File;
use std::io::{Read, Seek, SeekFrom};
use std::path::{Path, PathBuf};

use mary::nn::mxfp4::{scale_exponent_range, E4M3_POW2_MAX, E4M3_POW2_MIN, E4M3_POW2_MIN_NORMAL};

/// Octaves E4M3FN can hold exactly, and the narrower budget that also keeps
/// every block scale a *normal* E4M3 (no subnormal-flush hazard).
const EXACT_OCTAVES: i32 = E4M3_POW2_MAX - E4M3_POW2_MIN + 1;
const NORMAL_OCTAVES: i32 = E4M3_POW2_MAX - E4M3_POW2_MIN_NORMAL + 1;

/// One tensor to read: where it lives and what it is.
struct Plane {
    layer: u32,
    expert: u32,
    which: String,
    shard: String,
    abs_start: u64,
    nbytes: usize,
}

/// A safetensors header: `8 || u64 header_len || header_json`, so every
/// tensor's absolute offset is `8 + header_len + data_offsets[0]`. Getting
/// that `8 + header_len` wrong is the classic way to read a file full of
/// perfectly plausible garbage, which is why the gate checks it.
fn read_header(path: &Path) -> (u64, serde_json::Value) {
    let mut f = File::open(path).unwrap_or_else(|e| panic!("open {path:?}: {e}"));
    let mut len = [0u8; 8];
    f.read_exact(&mut len).expect("header length");
    let header_len = u64::from_le_bytes(len);
    let mut buf = vec![0u8; header_len as usize];
    f.read_exact(&mut buf).expect("header bytes");
    (
        header_len,
        serde_json::from_slice(&buf).expect("header json"),
    )
}

/// `language_model.model.layers.{L}.block_sparse_moe.experts.{E}.{w}.weight_scale`
/// → `(layer, expert, which)`.
fn parse_scale_name(name: &str) -> Option<(u32, u32, String)> {
    let rest = name.strip_suffix(".weight_scale")?;
    let (head, which) = rest.rsplit_once('.')?;
    let (head, expert) = head.rsplit_once(".experts.")?;
    let (_, layer) = head
        .strip_suffix(".block_sparse_moe")?
        .rsplit_once(".layers.")?;
    Some((layer.parse().ok()?, expert.parse().ok()?, which.to_string()))
}

/// Read the named planes, one shard at a time in ascending offset order, and
/// return each one's `(e_min, e_max)`.
fn measure(model_dir: &Path, planes: &[Plane]) -> Vec<(i32, i32)> {
    let mut by_shard: BTreeMap<&str, Vec<usize>> = BTreeMap::new();
    for (i, p) in planes.iter().enumerate() {
        by_shard.entry(p.shard.as_str()).or_default().push(i);
    }
    let mut out = vec![(0, 0); planes.len()];
    let mut buf = Vec::new();
    for (shard, mut idxs) in by_shard {
        idxs.sort_by_key(|&i| planes[i].abs_start);
        let mut f = File::open(model_dir.join(shard)).expect("open shard");
        for i in idxs {
            let p = &planes[i];
            buf.resize(p.nbytes, 0);
            f.seek(SeekFrom::Start(p.abs_start)).expect("seek");
            f.read_exact(&mut buf).expect("read scale plane");
            out[i] = scale_exponent_range(&buf).unwrap_or_else(|e| {
                panic!("layer {} expert {} {}: {e}", p.layer, p.expert, p.which)
            });
        }
    }
    out
}

fn main() {
    let mut args = std::env::args().skip(1);
    let model_dir: PathBuf = args
        .next()
        .expect("usage: mxfp4_span_sweep <MODEL_DIR> [ORACLE_DIR] [N|all]")
        .into();
    let oracle_dir = mary::paths::model(args.next().as_deref(), "k3-oracle").unwrap_or_else(|e| {
        eprintln!("{e}");
        std::process::exit(2)
    });
    let per_layer = args.next().unwrap_or_else(|| "16".to_string());

    let index: serde_json::Value = serde_json::from_slice(
        &std::fs::read(model_dir.join("model.safetensors.index.json")).expect("index"),
    )
    .expect("parse index");
    let weight_map = index["weight_map"].as_object().expect("weight_map");

    // Every scale plane in the checkpoint, keyed by (layer, expert, which),
    // with its shard resolved but not yet its offset (that needs the header).
    let mut catalog: BTreeMap<(u32, u32, String), &str> = BTreeMap::new();
    for (name, shard) in weight_map {
        if let Some(key) = parse_scale_name(name) {
            catalog.insert(key, shard.as_str().expect("shard name"));
        }
    }
    println!("{} scale planes in the index", catalog.len());

    // Headers, once per shard — 96 of them at ~820 KB each.
    let shards: std::collections::BTreeSet<&str> = catalog.values().copied().collect();
    let mut headers: BTreeMap<&str, (u64, serde_json::Value)> = BTreeMap::new();
    for s in &shards {
        headers.insert(s, read_header(&model_dir.join(s)));
    }
    println!("{} shard headers parsed", headers.len());

    let plane_of = |layer: u32, expert: u32, which: &str| -> Plane {
        let key = (layer, expert, which.to_string());
        let shard = *catalog
            .get(&key)
            .unwrap_or_else(|| panic!("no {key:?} in index"));
        let name = format!(
            "language_model.model.layers.{layer}.block_sparse_moe.experts.{expert}.{which}.weight_scale"
        );
        let (header_len, header) = &headers[shard];
        let entry = &header[name.as_str()];
        let rel = entry["data_offsets"][0].as_u64().expect("data_offsets");
        let end = entry["data_offsets"][1].as_u64().expect("data_offsets");
        assert_eq!(entry["dtype"].as_str(), Some("U8"), "{name} is not U8");
        Plane {
            layer,
            expert,
            which: which.to_string(),
            shard: shard.to_string(),
            abs_start: 8 + header_len + rel,
            nbytes: (end - rel) as usize,
        }
    };

    // ---------------------------------------------------------------- gate
    let stats: serde_json::Value = serde_json::from_slice(
        &std::fs::read(oracle_dir.join("_decode_stats.json")).expect("decode stats"),
    )
    .expect("parse decode stats");
    let extract: serde_json::Value = serde_json::from_slice(
        &std::fs::read(oracle_dir.join("_extract_meta.json")).expect("extract meta"),
    )
    .expect("parse extract meta");
    // The oracle's abs_start for every tensor it pulled, keyed by name.
    let mut oracle_abs: BTreeMap<String, u64> = BTreeMap::new();
    fn collect_abs(v: &serde_json::Value, out: &mut BTreeMap<String, u64>) {
        match v {
            serde_json::Value::Object(m) => {
                if let (Some(n), Some(a)) = (
                    m.get("name").and_then(|x| x.as_str()),
                    m.get("abs_start").and_then(|x| x.as_u64()),
                ) {
                    out.insert(n.to_string(), a);
                }
                for x in m.values() {
                    collect_abs(x, out);
                }
            }
            serde_json::Value::Array(a) => a.iter().for_each(|x| collect_abs(x, out)),
            _ => {}
        }
    }
    collect_abs(&extract, &mut oracle_abs);

    let mut failures: Vec<String> = Vec::new();
    let mut gate_planes = Vec::new();
    let mut gate_want = Vec::new();
    for (tag, entry) in stats.as_object().expect("stats object") {
        let layer = entry["layer"].as_u64().expect("layer") as u32;
        let expert = entry["expert"].as_u64().expect("expert") as u32;
        for (which, t) in entry["tensors"].as_object().expect("tensors") {
            gate_planes.push(plane_of(layer, expert, which));
            gate_want.push((
                format!("{tag}/{which}"),
                t["e8m0_exp_min"].as_i64().expect("exp min") as i32,
                t["e8m0_exp_max"].as_i64().expect("exp max") as i32,
            ));
        }
    }
    for p in &gate_planes {
        let name = format!(
            "language_model.model.layers.{}.block_sparse_moe.experts.{}.{}.weight_scale",
            p.layer, p.expert, p.which
        );
        match oracle_abs.get(&name) {
            Some(&want) if want == p.abs_start => {}
            Some(&want) => failures.push(format!(
                "{name}: abs_start {} != oracle {want}",
                p.abs_start
            )),
            None => failures.push(format!("{name}: not in the oracle's extract meta")),
        }
    }
    let got = measure(&model_dir, &gate_planes);
    for ((tag, want_min, want_max), (e_min, e_max)) in gate_want.iter().zip(&got) {
        if (e_min, e_max) != (want_min, want_max) {
            failures.push(format!(
                "{tag}: exponents {e_min}..{e_max} != oracle {want_min}..{want_max}"
            ));
        }
    }
    if !failures.is_empty() {
        eprintln!(
            "GATE FAILED — {} problem(s); no span reported:",
            failures.len()
        );
        for f in &failures {
            eprintln!("  {f}");
        }
        std::process::exit(1);
    }
    println!(
        "gate ok — {} oracle tensors: offsets and exponent ranges re-derived and identical\n",
        gate_planes.len()
    );

    // --------------------------------------------------------------- sweep
    let mut layers: BTreeMap<u32, Vec<u32>> = BTreeMap::new();
    for (layer, expert, which) in catalog.keys() {
        if which == "w1" {
            layers.entry(*layer).or_default().push(*expert);
        }
    }
    let sampled: BTreeMap<u32, Vec<u32>> = layers
        .iter()
        .map(|(layer, experts)| {
            let n: usize = if per_layer == "all" {
                experts.len()
            } else {
                per_layer.parse().expect("N")
            };
            let n = n.min(experts.len()).max(1);
            // Even stride across the expert index, endpoints included.
            let picks = (0..n)
                .map(|i| {
                    experts[if n == 1 {
                        0
                    } else {
                        i * (experts.len() - 1) / (n - 1)
                    }]
                })
                .collect::<Vec<_>>();
            (*layer, picks)
        })
        .collect();

    let mut planes = Vec::new();
    for (layer, experts) in &sampled {
        for e in experts {
            for which in ["w1", "w2", "w3"] {
                planes.push(plane_of(*layer, *e, which));
            }
        }
    }
    let gib = planes.iter().map(|p| p.nbytes).sum::<usize>() as f64 / (1 << 30) as f64;
    println!(
        "sweeping {} tensors over {} layers ({gib:.2} GiB of scale planes)",
        planes.len(),
        sampled.len()
    );

    let t0 = std::time::Instant::now();
    let ranges = measure(&model_dir, &planes);
    let secs = t0.elapsed().as_secs_f64();

    let mut worst: BTreeMap<u32, (i32, u32, String, i32, i32)> = BTreeMap::new();
    let mut hist: BTreeMap<i32, usize> = BTreeMap::new();
    let mut over_exact = 0usize;
    let mut over_normal = 0usize;
    let (mut g_min, mut g_max) = (i32::MAX, i32::MIN);
    for (p, &(e_min, e_max)) in planes.iter().zip(&ranges) {
        let span = e_max - e_min + 1;
        *hist.entry(span).or_default() += 1;
        if span > EXACT_OCTAVES {
            over_exact += 1;
        }
        if span > NORMAL_OCTAVES {
            over_normal += 1;
        }
        g_min = g_min.min(e_min);
        g_max = g_max.max(e_max);
        let slot = worst.entry(p.layer).or_insert((0, 0, String::new(), 0, 0));
        if span > slot.0 {
            *slot = (span, p.expert, p.which.clone(), e_min, e_max);
        }
    }

    println!("\nworst per-tensor octave span, by layer:");
    println!(
        "{:<7} {:>6} {:>8} {:>4} {:>12}",
        "layer", "span", "expert", "t", "exponents"
    );
    for (layer, (span, expert, which, e_min, e_max)) in &worst {
        println!(
            "{layer:<7} {span:>6} {expert:>8} {which:>4} {:>12}",
            format!("{e_min}..{e_max}")
        );
    }

    println!("\nspan histogram (tensors per octave span):");
    for (span, n) in &hist {
        println!("  {span:>2} octaves: {n}");
    }
    let worst_span = hist.keys().next_back().copied().unwrap_or(0);
    println!(
        "\nread {gib:.2} GiB in {secs:.1} s. Global exponent range {g_min}..{g_max}. \
         Worst single-tensor span {worst_span} of the {EXACT_OCTAVES} E4M3 holds exactly \
         ({NORMAL_OCTAVES} without subnormal scales)."
    );
    if over_exact == 0 && over_normal == 0 {
        println!(
            "VERDICT: every tensor measured transcodes bit-exactly, with an all-normal E4M3 scale plane."
        );
    } else {
        println!(
            "VERDICT: {over_exact} tensor(s) exceed the exact window and CANNOT be transcoded \
             losslessly; a further {} need subnormal E4M3 block scales.",
            over_normal - over_exact
        );
    }
    let pct = 100.0 * planes.len() as f64 / catalog.len() as f64;
    if planes.len() == catalog.len() {
        println!(
            "Scope: {} of {} scale planes (100%) — every MXFP4 tensor in the checkpoint, not a sample.",
            planes.len(),
            catalog.len()
        );
    } else {
        println!(
            "Scope: {} of {} scale planes ({pct:.2}%) — a sample. Rerun with 'all' for the checkpoint.",
            planes.len(),
            catalog.len()
        );
    }
}
