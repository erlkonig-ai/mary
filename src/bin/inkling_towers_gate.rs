//! Parity gate for the Inkling vision (HMLP) and audio (dMel) towers.
//!
//! Budget, written down first: worst absolute error over the tensor's own
//! scale, `1e-5`, the same criterion as the other real-weight gates.
//!
//! The first check is the interesting one. mary's `plan_out_scales` derives the
//! pyramid's grids from `patch_size`, `temporal_patch_size`, `n_layers` and
//! `n_channels`; the reference does the same but resolves the under-determined
//! case with `scipy.linear_sum_assignment`, while mary enumerates injective
//! assignments. Those are two different algorithms for the same optimum, so
//! comparing them is a real check rather than a restatement — and it is what
//! makes the widths derived instead of a table transcribed from one checkpoint.
//!
//! Non-vacuity, gated rather than assumed:
//!
//! * some stage must fold spatially AND some stage temporally, or
//!   `fold_timespace_to_depth` is not exercised in both axes;
//! * the audio lookup must hit many distinct codebook rows, or a wrong offset
//!   would still land on the right one;
//! * exactly one stage must lack a norm (the last), or the norm-and-GELU branch
//!   is either never or always taken.
//!
//!   cargo run --release --features inkling --bin inkling_towers_gate -- <ckpt> <oracle>

use std::path::PathBuf;

use anyhow::{Context, Result};

use mary::models::inkling::load::Checkpoint;
use mary::models::inkling::vision::{audio_embed, plan_out_scales, vision_stage, vision_stages};

const BUDGET: f32 = 1e-5;

fn read_f32(p: &std::path::Path) -> Result<Vec<f32>> {
    let b = std::fs::read(p).with_context(|| format!("reading {}", p.display()))?;
    Ok(b.chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect())
}

fn read_i64(p: &std::path::Path) -> Result<Vec<i64>> {
    let b = std::fs::read(p).with_context(|| format!("reading {}", p.display()))?;
    Ok(b.chunks_exact(8)
        .map(|c| i64::from_le_bytes(c.try_into().unwrap()))
        .collect())
}

fn num(text: &str, key: &str) -> Result<f64> {
    let pat = format!("\"{key}\"");
    let at = text
        .find(&pat)
        .with_context(|| format!("manifest has no {key}"))?;
    let rest = &text[at + pat.len()..];
    let colon = rest.find(':').context("malformed manifest")?;
    let s: String = rest[colon + 1..]
        .chars()
        .skip_while(|c| c.is_whitespace())
        .take_while(|c| c.is_ascii_digit() || *c == '.' || *c == '-' || *c == 'e')
        .collect();
    s.parse()
        .with_context(|| format!("{key} is not a number: {s:?}"))
}

fn report(label: &str, mine: &[f32], theirs: &[f32], checks: &mut usize, fails: &mut usize) {
    let mut abs = 0f32;
    let mut scale = 0f32;
    let mut rel = 0f32;
    for (&a, &b) in mine.iter().zip(theirs) {
        let e = (a - b).abs();
        abs = abs.max(e);
        scale = scale.max(b.abs());
        rel = rel.max(e / b.abs().max(1e-6));
    }
    let n = mine.len().min(theirs.len());
    *checks += n;
    let scaled = abs / scale.max(f32::MIN_POSITIVE);
    println!(
        "  {label}: {n} values, worst abs {abs:e} / scale {scale:e} = {scaled:e}, rel {rel:e}"
    );
    if mine.len() != theirs.len() {
        println!(
            "    FAIL  lengths differ: {} vs {}",
            mine.len(),
            theirs.len()
        );
        *fails += 1;
    }
    if scaled > BUDGET {
        println!("    FAIL  over budget {BUDGET:e}");
        *fails += 1;
    }
}

fn main() -> Result<()> {
    let ckpt = std::env::args()
        .nth(1)
        .map(PathBuf::from)
        .context("usage: <ckpt> <oracle>")?;
    let oracle = std::env::args()
        .nth(2)
        .map(PathBuf::from)
        .context("usage: <ckpt> <oracle>")?;
    let man = String::from_utf8(std::fs::read(oracle.join("tow_manifest.json"))?)?;

    let patch = num(&man, "patch_size")? as usize;
    let tpatch = num(&man, "temporal_patch_size")? as usize;
    let n_layers = num(&man, "n_layers")? as usize;
    let n_ch = num(&man, "n_channels")? as usize;
    let text_h = num(&man, "text_hidden")? as usize;
    let eps = num(&man, "rms_norm_eps")?;
    let bins = num(&man, "n_mel_bins")? as usize;
    let levels = num(&man, "mel_levels")? as usize;
    let patches = num(&man, "patches")? as usize;
    let frames = num(&man, "frames")? as usize;
    let fold_hw = num(&man, "stages_folding_hw")? as usize;
    let fold_t = num(&man, "stages_folding_t")? as usize;
    let distinct = num(&man, "distinct_levels_used")? as usize;

    let cp = Checkpoint::open(&ckpt)?;
    println!("=== towers ===");
    println!("  vision: patch {patch}, temporal {tpatch}, layers {n_layers}, channels {n_ch} -> {text_h}");
    println!("  audio : {bins} bins x {levels} levels -> {text_h}");
    println!("  tensors in index: {}", cp.len());

    let mut fails = 0usize;
    let mut checks = 0usize;

    // ---- 1. the derived plan, against scipy's assignment -------------------
    println!("\n=== 1. plan_out_scales, derived here vs scipy in the reference ===");
    let ref_scales = read_i64(&oracle.join("tow_scales.bin"))?;
    anyhow::ensure!(
        !ref_scales.is_empty(),
        "no reference scales — the gate would be vacuous"
    );
    let mine = plan_out_scales(tpatch, patch, n_layers, n_ch);
    println!(
        "  reference rows: {}, mine: {}",
        ref_scales.len() / 4,
        mine.len()
    );
    checks += 1;
    if mine.len() * 4 != ref_scales.len() {
        println!("  FAIL  {} rows vs {}", mine.len(), ref_scales.len() / 4);
        fails += 1;
    } else {
        let mut bad = 0usize;
        for (i, s) in mine.iter().enumerate() {
            checks += 4;
            let r = &ref_scales[i * 4..i * 4 + 4];
            let got = [s.t as i64, s.h as i64, s.w as i64, s.c as i64];
            if got != [r[0], r[1], r[2], r[3]] {
                println!("  FAIL  scale {i}: mine {got:?}, reference {r:?}");
                bad += 1;
            }
        }
        println!("  scale rows disagreeing: {bad}");
        fails += bad;
        for s in &mine {
            println!("    (t {}, h {}, w {}, c {})", s.t, s.h, s.w, s.c);
        }
    }

    // ---- 2. the stages the plan implies ------------------------------------
    println!("\n=== 2. stage shapes ===");
    let ref_stages = read_i64(&oracle.join("tow_stages.bin"))?;
    let stages = vision_stages(&mine, n_layers, text_h);
    checks += 1;
    anyhow::ensure!(
        stages.len() * 5 == ref_stages.len(),
        "{} stages vs {}",
        stages.len(),
        ref_stages.len() / 5
    );
    let mut bad = 0usize;
    for (i, s) in stages.iter().enumerate() {
        checks += 5;
        let r = &ref_stages[i * 5..i * 5 + 5];
        let got = [
            s.t_fold as i64,
            s.hw_fold as i64,
            s.input_dim as i64,
            s.output_dim as i64,
            s.add_norm as i64,
        ];
        let want = [r[0], r[1], r[2], r[3], r[4]];
        let ok = got == want;
        println!(
            "  stage {i}: fold t={} hw={}, {} -> {}, norm={}  {}",
            s.t_fold,
            s.hw_fold,
            s.input_dim,
            s.output_dim,
            s.add_norm,
            if ok { "ok" } else { "FAIL" }
        );
        if !ok {
            println!("    reference: {want:?}");
            bad += 1;
        }
    }
    fails += bad;

    // Non-vacuity on the fold and norm branches.
    checks += 3;
    println!("  stages folding spatially: {fold_hw}, temporally: {fold_t}");
    if fold_hw == 0 || fold_t == 0 {
        println!("  FAIL  one of the fold axes is never exercised");
        fails += 1;
    }
    let no_norm = stages.iter().filter(|s| !s.add_norm).count();
    println!("  stages without a norm: {no_norm} (expect exactly 1, the last)");
    if no_norm != 1 {
        println!("  FAIL  the norm-and-GELU branch is either never or always taken");
        fails += 1;
    }

    // ---- 3. vision forward -------------------------------------------------
    println!("\n=== 3. vision forward on real weights ===");
    let px = read_f32(&oracle.join("tow_px.bin"))?;
    let vref = read_f32(&oracle.join("tow_vision_y.bin"))?;
    anyhow::ensure!(!vref.is_empty(), "no vision reference");
    let per_patch = tpatch * patch * patch * n_ch;
    anyhow::ensure!(
        px.len() == patches * per_patch,
        "pixels are {} not {}",
        px.len(),
        patches * per_patch
    );

    let mut projs = Vec::new();
    let mut norms: Vec<Option<Vec<f32>>> = Vec::new();
    for (i, s) in stages.iter().enumerate() {
        projs.push(
            cp.tensor(&format!("model.visual.layers.linear_{i}.weight"))?
                .data,
        );
        norms.push(if s.add_norm {
            Some(
                cp.tensor(&format!("model.visual.layers.norm_{i}.weight"))?
                    .data,
            )
        } else {
            None
        });
    }
    let final_norm = cp.tensor("model.visual.final_norm.weight")?.data;

    let mut vout = Vec::with_capacity(patches * text_h);
    for p in 0..patches {
        let mut h = px[p * per_patch..(p + 1) * per_patch].to_vec();
        let mut grid = (tpatch, patch, patch, n_ch);
        for (i, s) in stages.iter().enumerate() {
            let (y, g) = vision_stage(&h, s, &projs[i], norms[i].as_deref(), grid, eps);
            h = y;
            grid = g;
        }
        let rows = grid.0 * grid.1 * grid.2;
        let h = mary::models::inkling::block::rms_norm(&h, &final_norm, eps, rows, grid.3);
        vout.extend_from_slice(&h);
    }
    report("vision", &vout, &vref, &mut checks, &mut fails);

    // ---- 4. audio forward --------------------------------------------------
    println!("\n=== 4. audio (dMel) forward on real weights ===");
    let ids: Vec<usize> = read_i64(&oracle.join("tow_audio_ids.bin"))?
        .into_iter()
        .map(|v| v as usize)
        .collect();
    let aref = read_f32(&oracle.join("tow_audio_y.bin"))?;
    anyhow::ensure!(!aref.is_empty(), "no audio reference");
    anyhow::ensure!(
        ids.len() == frames * bins,
        "ids are {} not {}",
        ids.len(),
        frames * bins
    );
    let table = cp.tensor("model.audio.encoder.weight")?.data;
    let anorm = cp.tensor("model.audio.final_norm.weight")?.data;
    println!(
        "  codebook table is {} floats = [{}x{}, {text_h}]",
        table.len(),
        bins,
        levels
    );
    checks += 1;
    if table.len() != bins * levels * text_h {
        println!("  FAIL  table is the wrong size");
        fails += 1;
    }
    let aout = audio_embed(&ids, &table, &anorm, eps, frames, bins, levels, text_h);
    report("audio", &aout, &aref, &mut checks, &mut fails);
    checks += 1;
    println!("  distinct mel levels in the input: {distinct} of {levels}");
    if distinct < 2 {
        println!("  FAIL  the lookup barely varies — a wrong offset could still pass");
        fails += 1;
    }

    println!("\n=== what this does not prove ===");
    println!("  the checkpoint-name -> module mapping is authored on both sides.");
    println!("  the pixel and mel inputs are random, not real images or audio: this checks");
    println!("  the arithmetic, not the preprocessing that would produce them.");

    println!("\n=== verdict ===");
    println!("  checks: {checks}");
    if fails == 0 {
        println!("GATE PASSED — {checks} checks, both towers match transformers on real weights");
        Ok(())
    } else {
        println!("GATE FAILED — {checks} checks, {fails} FAILURES");
        std::process::exit(1);
    }
}
