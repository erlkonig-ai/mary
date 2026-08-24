//! Parity gate for the NVFP4 decode, against `compressed_tensors`.
//!
//! The reference bundle is produced by `nvfp4_emit.py`, which unpacks a real
//! slice of a real expert tensor with
//! `compressed_tensors.compressors.unpack_fp4_from_uint8` — the library that
//! defines this packing for checkpoints. The scales cross the boundary as their
//! **raw E4M3 bytes**, so this gate has to do the FP8 decode itself; handing it
//! float32 scales would test the multiply and skip the harder half.
//!
//! Four things are checked:
//!
//! 1. **The E4M3 decode over the whole domain.** All 256 byte patterns against
//!    torch's own `float8_e4m3fn`, not just the 25 distinct scales this slice
//!    happens to contain. A decode that agrees on 25 values has been checked on
//!    a tenth of the domain, and reporting that as agreement is the same
//!    mistake as a gate that examines nothing and passes.
//! 2. **Nibble order.** Re-derived here by counting how many values match with
//!    the packing reversed; if both orders fit, the check is vacuous and says
//!    so.
//! 3. **Full decode parity** against the reference, elementwise.
//! 4. **The representable bound.** Nothing decoded may exceed
//!    `E4M3_MAX * FP4_MAX * scale2`.
//!
//!   cargo run --release --features inkling --bin inkling_nvfp4_gate -- [<oracle dir>]

use std::path::Path;

use anyhow::{Context, Result};

use mary::models::inkling::nvfp4::{
    decode_stacked, e4m3_to_f32, split_byte, E4M3_MAX, FP4_E2M1, FP4_MAX, GROUP,
};

fn read_bytes(dir: &Path, name: &str) -> Result<Vec<u8>> {
    let p = dir.join(name);
    std::fs::read(&p).with_context(|| format!("reading {}", p.display()))
}

fn read_f32(dir: &Path, name: &str) -> Result<Vec<f32>> {
    let b = read_bytes(dir, name)?;
    anyhow::ensure!(b.len() % 4 == 0, "{name} is not a whole number of f32");
    Ok(b.chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect())
}

/// Pull one integer field out of the manifest without taking a JSON dependency
/// for four numbers.
fn manifest_usize(text: &str, key: &str) -> Result<usize> {
    let pat = format!("\"{key}\"");
    let at = text
        .find(&pat)
        .with_context(|| format!("manifest has no {key}"))?;
    let rest = &text[at + pat.len()..];
    let colon = rest.find(':').context("malformed manifest")?;
    let tail = &rest[colon + 1..];
    let digits: String = tail
        .chars()
        .skip_while(|c| c.is_whitespace())
        .take_while(|c| c.is_ascii_digit())
        .collect();
    digits
        .parse()
        .with_context(|| format!("{key} is not an integer"))
}

fn main() -> Result<()> {
    let dir = mary::paths::model(std::env::args().nth(1).as_deref(), "inkling-oracle")?;

    let manifest = String::from_utf8(read_bytes(&dir, "nvfp4_manifest.json")?)
        .context("manifest is not utf-8")?;
    let experts = manifest_usize(&manifest, "experts")?;
    let rows = manifest_usize(&manifest, "rows")?;
    let bytes_per_row = manifest_usize(&manifest, "bytes_per_row")?;
    let group = manifest_usize(&manifest, "group")?;
    let logical = bytes_per_row * 2;

    println!("=== oracle bundle ===");
    println!("  dir            : {}", dir.display());
    println!("  experts {experts} x rows {rows} x {bytes_per_row} bytes -> {logical} logical");
    println!("  group          : {group} (this build assumes {GROUP})");
    anyhow::ensure!(
        group == GROUP,
        "oracle group {group} != this build's {GROUP}"
    );

    let mut fails = 0usize;
    let mut checks = 0usize;

    // ---- 1. the E4M3 decode, over the whole domain ------------------------
    println!("\n=== 1. E4M3 decode, all 256 byte patterns ===");
    let table = read_f32(&dir, "e4m3_table_f32.bin")?;
    anyhow::ensure!(table.len() == 256, "e4m3 table has {} entries", table.len());
    let mut e4m3_bad = 0usize;
    let mut nan_seen = 0usize;
    for b in 0..=255u8 {
        checks += 1;
        let mine = e4m3_to_f32(b);
        let theirs = table[b as usize];
        if theirs.is_nan() {
            nan_seen += 1;
            if !mine.is_nan() {
                if e4m3_bad < 6 {
                    println!("  FAIL  0x{b:02X}: torch NaN, mine {mine}");
                }
                e4m3_bad += 1;
            }
        } else if mine != theirs {
            if e4m3_bad < 6 {
                println!("  FAIL  0x{b:02X}: torch {theirs}, mine {mine}");
            }
            e4m3_bad += 1;
        }
    }
    println!("  byte patterns examined : 256");
    println!("  NaN patterns           : {nan_seen}");
    println!("  disagreements          : {e4m3_bad}");
    fails += e4m3_bad;

    // ---- load the real slice ----------------------------------------------
    let codes = read_bytes(&dir, "nvfp4_codes.bin")?;
    let scales = read_bytes(&dir, "nvfp4_scale_e4m3.bin")?;
    let scale2 = read_f32(&dir, "nvfp4_scale2_f32.bin")?;
    let expected = read_f32(&dir, "nvfp4_expected_f32.bin")?;
    println!("\n=== corpus ===");
    println!("  code bytes  : {}", codes.len());
    println!("  scale bytes : {}", scales.len());
    println!("  scale2      : {}", scale2.len());
    println!("  expected    : {}", expected.len());
    anyhow::ensure!(
        !expected.is_empty(),
        "zero reference values — the gate would be vacuous"
    );
    let distinct = {
        let mut s: Vec<u8> = scales.clone();
        s.sort_unstable();
        s.dedup();
        s.len()
    };
    println!("  distinct E4M3 scales in this slice: {distinct} of 256");

    // ---- 3. full decode parity --------------------------------------------
    println!("\n=== 2/3. decode parity, elementwise ===");
    let mut out = vec![0f32; experts * rows * logical];
    let written = decode_stacked(
        &codes,
        &scales,
        &scale2,
        experts,
        rows,
        bytes_per_row,
        &mut out,
    );
    println!("  values decoded : {written}");
    anyhow::ensure!(
        written == expected.len(),
        "decoded {written}, reference has {}",
        expected.len()
    );

    let mut worst = 0f32;
    let mut mismatch = 0usize;
    for (i, (&a, &b)) in out.iter().zip(expected.iter()).enumerate() {
        checks += 1;
        // Both sides multiply the same three exact binary values, so this is an
        // exact-equality check, not an approximate one.
        if a != b && !(a == 0.0 && b == 0.0) {
            let d = (a - b).abs();
            if d > worst {
                worst = d;
            }
            if mismatch < 6 {
                println!("  FAIL  [{i}]: mine {a}, reference {b}");
            }
            mismatch += 1;
        }
    }
    println!("  mismatches     : {mismatch}");
    println!("  worst abs diff : {worst:e}");
    fails += mismatch;

    // ---- 2. is the nibble-order check non-vacuous? -------------------------
    // Decode with the pair reversed; if that ALSO matches, the reference cannot
    // distinguish the orders and check 3 proves nothing about packing.
    let mut swapped = vec![0f32; out.len()];
    for e in 0..experts {
        for r in 0..rows {
            let ci = (e * rows + r) * bytes_per_row;
            let si = (e * rows + r) * (logical / GROUP);
            let oi = (e * rows + r) * logical;
            for block in 0..logical / GROUP {
                let s = e4m3_to_f32(scales[si + block]);
                for i in 0..GROUP / 2 {
                    let byte = codes[ci + block * GROUP / 2 + i];
                    let (lo, hi) = split_byte(byte);
                    // reversed on purpose; same association as the real decode
                    swapped[oi + block * GROUP + 2 * i] = FP4_E2M1[hi as usize] * s * scale2[e];
                    swapped[oi + block * GROUP + 2 * i + 1] = FP4_E2M1[lo as usize] * s * scale2[e];
                }
            }
        }
    }
    let swapped_matches = swapped
        .iter()
        .zip(expected.iter())
        .filter(|(a, b)| a == b)
        .count();
    checks += 1;
    println!("\n=== nibble order is actually pinned ===");
    println!(
        "  values where the REVERSED packing also matches: {swapped_matches} of {}",
        expected.len()
    );
    if swapped_matches == expected.len() {
        println!(
            "  FAIL  both orders reproduce the reference — this corpus cannot pin the packing"
        );
        fails += 1;
    } else {
        println!(
            "  reversed packing disagrees on {} values, so the order is pinned",
            expected.len() - swapped_matches
        );
    }

    // ---- 4. representable bound -------------------------------------------
    println!("\n=== 4. representable bound ===");
    let mut over = 0usize;
    for e in 0..experts {
        let bound = E4M3_MAX * FP4_MAX * scale2[e].abs();
        let base = e * rows * logical;
        for &v in &out[base..base + rows * logical] {
            checks += 1;
            if v.abs() > bound {
                over += 1;
            }
        }
        println!("  expert {e}: scale2 {:e}, bound {:e}", scale2[e], bound);
    }
    println!("  values above their bound: {over}");
    fails += over;
    let observed = out.iter().fold(0f32, |m, v| m.max(v.abs()));
    println!("  observed |max|          : {observed:e}");

    println!("\n=== verdict ===");
    println!("  checks: {checks}");
    if fails == 0 {
        println!("GATE PASSED — {checks} checks, NVFP4 decode matches compressed_tensors");
        Ok(())
    } else {
        println!("GATE FAILED — {checks} checks, {fails} FAILURES");
        std::process::exit(1);
    }
}
