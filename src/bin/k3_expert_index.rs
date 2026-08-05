//! Build the expert address book for a Kimi-K3 checkpoint, and measure what it
//! costs to actually fetch experts through it.
//!
//! Streaming is not an optimisation for this model, it is the only way it runs:
//! 1.5 TB of weights against 128 GB of RAM. The feasibility work established
//! that one routed expert is exactly 17,547,264 contiguous bytes inside one
//! shard, and that random reads at that granularity cost only ~2% over
//! sequential. Both of those are load-bearing assumptions for every throughput
//! estimate downstream, and both were measured with synthetic reads rather than
//! through the real name->shard->offset path a loader has to walk.
//!
//! This builds that path and re-measures through it. A correctness gate runs
//! first: the address book must reproduce the known invariants, or no bandwidth
//! number is printed.
use std::collections::HashMap;
use std::fs::File;
use std::io::Read;
use std::os::unix::fs::FileExt;
use std::path::{Path, PathBuf};
use std::time::Instant;

/// Where one tensor's bytes live.
#[derive(Clone, Debug)]
struct Located {
    shard: PathBuf,
    /// Absolute file offset. safetensors `data_offsets` are relative to the END
    /// of the header, not the file start — an easy and expensive mistake.
    offset: u64,
    len: u64,
}

fn header_len(path: &Path) -> std::io::Result<u64> {
    let mut f = File::open(path)?;
    let mut n = [0u8; 8];
    f.read_exact(&mut n)?;
    Ok(u64::from_le_bytes(n))
}

/// Parse one shard's JSON header into (name -> (rel_start, rel_end)).
fn shard_entries(path: &Path) -> std::io::Result<(u64, Vec<(String, u64, u64)>)> {
    let hlen = header_len(path)?;
    let mut buf = vec![0u8; hlen as usize];
    File::open(path)?.read_exact_at(&mut buf, 8)?;
    let txt = String::from_utf8_lossy(&buf);
    let mut out = Vec::new();
    // Deliberately not a JSON crate: the shape here is fixed and shallow, and a
    // parser would itself be something to gate. Walk each `"name":{...
    // "data_offsets":[a,b]` record without `?` on Options, so a malformed entry
    // is skipped loudly rather than aborting the whole shard.
    let mut skipped = 0usize;
    for (i, _) in txt.match_indices("\"data_offsets\"") {
        let head = &txt[..i];
        let Some(q_end) = head.rfind("\":{") else { skipped += 1; continue };
        let Some(q_start) = head[..q_end].rfind('"') else { skipped += 1; continue };
        let name = &head[q_start + 1..q_end];
        let rest = &txt[i..];
        let Some(lb) = rest.find('[') else { skipped += 1; continue };
        let Some(rb_rel) = rest[lb..].find(']') else { skipped += 1; continue };
        let mut it = rest[lb + 1..lb + rb_rel].split(',');
        let (Some(a), Some(b)) = (it.next(), it.next()) else { skipped += 1; continue };
        let (Ok(a), Ok(b)) = (a.trim().parse::<u64>(), b.trim().parse::<u64>()) else {
            skipped += 1;
            continue;
        };
        out.push((name.to_string(), a, b));
    }
    if skipped > 0 {
        eprintln!("  warning: {skipped} malformed header record(s) in {}", path.display());
    }
    Ok((8 + hlen, out))
}

fn main() {
    let dir = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "./kimi-k3".into());
    let dir = PathBuf::from(dir);

    let t0 = Instant::now();
    let mut book: HashMap<String, Located> = HashMap::new();
    let mut shards: Vec<PathBuf> = std::fs::read_dir(&dir)
        .expect("read model dir")
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| {
            p.extension().is_some_and(|e| e == "safetensors")
        })
        .collect();
    shards.sort();
    for sp in &shards {
        let (data_start, entries) = match shard_entries(sp) {
            Ok(v) => v,
            Err(e) => {
                eprintln!("header parse failed for {}: {e}", sp.display());
                std::process::exit(1);
            }
        };
        for (name, a, b) in entries {
            book.insert(
                name,
                Located { shard: sp.clone(), offset: data_start + a, len: b - a },
            );
        }
    }
    let build = t0.elapsed();
    println!("address book: {} tensors from {} shards in {:.2}s", book.len(), shards.len(), build.as_secs_f64());

    // ── correctness gate, before any bandwidth number ──
    const EXPERT_BYTES: u64 = 17_547_264;
    let mut experts: Vec<(&String, &Located)> = book
        .iter()
        .filter(|(n, _)| n.contains(".experts.") && n.ends_with(".weight"))
        .collect();
    experts.sort_by(|a, b| a.0.cmp(b.0));

    // Group the three tensors of each expert (w1/w2/w3 packed + scales) by prefix.
    let mut by_expert: HashMap<String, u64> = HashMap::new();
    for (n, l) in book.iter() {
        if let Some(i) = n.find(".experts.") {
            if let Some(j) = n[i + 9..].find('.') {
                let key = n[..i + 9 + j].to_string();
                *by_expert.entry(key).or_insert(0) += l.len;
            }
        }
    }
    let n_experts = by_expert.len();
    let wrong: Vec<(&String, &u64)> = by_expert.iter().filter(|(_, &b)| b != EXPERT_BYTES).collect();
    println!("routed experts found: {n_experts}");
    println!("experts whose total bytes != {EXPERT_BYTES}: {}", wrong.len());
    if let Some((n, b)) = wrong.first() {
        println!("  e.g. {n} = {b}");
    }
    let mut fail = 0;
    if n_experts != 82_432 {
        println!("GATE FAIL: expected 82,432 routed experts (92 MoE layers x 896), found {n_experts}");
        fail += 1;
    }
    if !wrong.is_empty() {
        println!("GATE FAIL: {} expert(s) do not total the known 17,547,264 bytes", wrong.len());
        fail += 1;
    }
    if fail > 0 {
        println!("\nNO BANDWIDTH NUMBER IS REPORTED: the address book did not reproduce the known invariants.");
        std::process::exit(1);
    }
    println!("GATE PASS: {n_experts} experts, every one exactly {EXPERT_BYTES} bytes\n");

    // ── the resident set: everything that is NOT a routed expert ──
    //
    // Routed experts stream; everything else has to be pinned, because it is
    // touched for every token. This is the number that decides whether B=1
    // inference fits at all on a 128 GB machine, and it was previously derived
    // from a separate header pass — recomputing it from the same address book
    // the loader uses removes one place for the two to drift apart.
    let mut resident: HashMap<&str, u64> = HashMap::new();
    let mut resident_total = 0u64;
    for (n, l) in book.iter() {
        if n.contains(".experts.") {
            continue;
        }
        let bucket = if n.contains("self_attn") || n.contains("attn") {
            "attention"
        } else if n.contains("shared_expert") {
            "shared experts"
        } else if n.contains("embed_tokens") {
            "embed_tokens"
        } else if n.contains("lm_head") {
            "lm_head"
        } else if n.contains("vision") || n.contains("vt_") || n.contains("mm_proj") {
            "vision tower"
        } else if n.contains("gate") && n.contains("moe") {
            "router gates"
        } else {
            "other (norms, dense mlp, projections)"
        };
        *resident.entry(bucket).or_insert(0) += l.len;
        resident_total += l.len;
    }
    let mut buckets: Vec<(&&str, &u64)> = resident.iter().collect();
    buckets.sort_by(|a, b| b.1.cmp(a.1));
    let gib = |b: u64| b as f64 / (1u64 << 30) as f64;
    println!("\nresident set (everything that is not a routed expert):");
    for (name, bytes) in &buckets {
        println!("  {:<38} {:>8.2} GiB", name, gib(**bytes));
    }
    println!("  {:<38} {:>8.2} GiB  TOTAL", "", gib(resident_total));
    let streamed: u64 = by_expert.values().sum();
    println!(
        "  streamed (routed experts)              {:>8.2} GiB   {:.2}% of the checkpoint",
        gib(streamed),
        streamed as f64 / (streamed + resident_total) as f64 * 100.0
    );
    // 128 GB machine, minus the OS, the CUDA context and activations.
    const MACHINE_BYTES: f64 = 128.0e9;
    println!(
        "  headroom on a 128 GB box: {:.2} GB before OS/CUDA/activations/KV",
        (MACHINE_BYTES - resident_total as f64) / 1e9
    );

    // ── fetch through the real path ──
    //
    // The member names are DERIVED from the address book, not guessed. A first
    // version hardcoded `.w1.weight` and friends; the real suffixes are
    // `weight_packed` and `weight_scale`, so it silently fetched only the three
    // scale planes — 1 MB per expert instead of 17.5 — while the gate above
    // still passed, because the gate grouped by prefix and the fetch guessed
    // names. Two mechanisms, one checked. Hence the per-expert byte assertion
    // below: the fetch now has to prove it moved what it claims.
    let mut members: HashMap<String, Vec<&Located>> = HashMap::new();
    for (n, l) in book.iter() {
        if let Some(i) = n.find(".experts.") {
            if let Some(j) = n[i + 9..].find('.') {
                members.entry(n[..i + 9 + j].to_string()).or_default().push(l);
            }
        }
    }
    // ONE READ PER EXPERT. Measured at layers 1, 51 and 92: an expert's six
    // members (w1/w2/w3 x packed+scale) are contiguous within a single shard,
    // spanning exactly 17,547,264 bytes with zero internal gaps. So the loader
    // does not need six reads — it needs one, at the minimum offset, for the
    // whole span. The first version issued six reads and re-opened the file for
    // each, and measured 1.02 GB/s against the ~5.95 the medium supports.
    let mut spans: HashMap<String, (PathBuf, u64, u64)> = HashMap::new();
    for (key, ls) in &members {
        let shard = ls[0].shard.clone();
        let one_shard = ls.iter().all(|l| l.shard == shard);
        let lo = ls.iter().map(|l| l.offset).min().unwrap();
        let hi = ls.iter().map(|l| l.offset + l.len).max().unwrap();
        let sum: u64 = ls.iter().map(|l| l.len).sum();
        // Contiguity is an assumption the loader depends on, so it is asserted
        // rather than assumed: span == sum means no foreign bytes in between.
        if one_shard && hi - lo == sum {
            spans.insert(key.clone(), (shard, lo, hi - lo));
        }
    }
    println!(
        "contiguous single-shard experts: {}/{}",
        spans.len(),
        members.len()
    );
    if spans.len() != members.len() {
        println!("GATE FAIL: {} expert(s) are split or interleaved; a single read cannot cover them",
                 members.len() - spans.len());
        std::process::exit(1);
    }

    let keys: Vec<String> = spans.keys().cloned().collect();

    // CONCURRENCY SWEEP, each level on DISJOINT experts.
    //
    // A first version reused the same 128 picks for every thread count and
    // reported 43 GB/s from a drive measured at ~6 — the 1-thread run warmed
    // the page cache and every later run read RAM. With 82,432 experts there is
    // no reason to reuse any, so each level gets its own cold slice and the
    // numbers are about the medium again.
    println!("\nconcurrency sweep (buffered, one read per expert, DISJOINT cold slices):");
    let per_level = 96usize;
    let mut cursor = 0usize;
    for threads in [1usize, 2, 4, 8, 16] {
        // Stride across the whole store so a slice is not one hot shard.
        let stride = keys.len() / (per_level * 6).max(1);
        let picks: Vec<(PathBuf, u64, u64)> = (0..per_level)
            .map(|i| {
                let k = &keys[((cursor + i) * stride.max(1)) % keys.len()];
                spans[k].clone()
            })
            .collect();
        cursor += per_level;

        let t1 = Instant::now();
        let chunk = picks.len().div_ceil(threads);
        std::thread::scope(|sc| {
            for part in picks.chunks(chunk) {
                sc.spawn(move || {
                    let mut buf = vec![0u8; EXPERT_BYTES as usize];
                    let mut handles: HashMap<PathBuf, File> = HashMap::new();
                    for (shard, off, len) in part {
                        let f = handles
                            .entry(shard.clone())
                            .or_insert_with(|| File::open(shard).expect("open shard"));
                        f.read_exact_at(&mut buf[..*len as usize], *off)
                            .expect("read expert");
                    }
                });
            }
        });
        let secs = t1.elapsed().as_secs_f64();
        let bytes = picks.len() as u64 * EXPERT_BYTES;
        println!(
            "  {threads:>2} thread(s): {:.2} GiB in {:.2}s = {:.2} GB/s",
            bytes as f64 / (1u64 << 30) as f64,
            secs,
            bytes as f64 / secs / 1e9
        );
    }
    println!("  (each level reads experts no earlier level touched)");
    println!("(page cache is NOT bypassed here — this is the warm-ish path a loader actually sees)");
}
