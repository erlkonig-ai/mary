//! `mxfp4_gate` — the correctness gate for [`mary::nn::mxfp4`], run against
//! **real Kimi-K3 expert bytes** and the `k3oracle` reference decode.
//!
//! The gate is bit-exactness, not a tolerance, which is what makes it worth
//! running: MXFP4 stores exact powers of two times 3-bit codes, so a correct
//! decoder reproduces the reference float32 *bit patterns*, and a correct
//! MXFP4 → NVFP4 transcode reproduces them again after the relabelling. Any
//! slack in either direction — a collapsed −0.0, a swapped nibble, a
//! renormalized block — shows up as a hash mismatch, so there is nothing to
//! tune.
//!
//! Three checks on the code tables, once:
//!
//! 0a. **E2M1** matches `e2m1_code_table.csv` on all 16 codes, by bit pattern,
//!     so `0x8` staying −0.0 is checked and not assumed.
//! 0b. **E8M0** matches `e8m0_scale_table.csv` on all 256 bytes, including
//!     `0x00`'s subnormal `2^-127` and `0xFF`'s NaN.
//! 0c. **E4M3FN's exact power-of-two window** matches the oracle's *measured*
//!     `e4m3fn_exact_pow2_exponents` (a float32 → E4M3 → float32 round trip
//!     through ml_dtypes). This is the load-bearing one for the transcode: it
//!     is what turns "18 octaves fit" from a reading of the spec into a
//!     witnessed fact, and the E4M3 codec here has no other external witness.
//!
//! Then four checks per tensor, all of which must pass:
//!
//! 1. **Full-scale decode**: sha256 of the complete decoded f32 buffer
//!    (little-endian, C order) equals the oracle's, taken from
//!    `_decode_stats.json` rather than pasted in — 11,010,048 elements each.
//! 2. **Independent file**: the first 64 rows match `*_dec_head64.f32.npy`
//!    bit-for-bit, so a systematic mistake in how the hash is fed can't hide;
//!    for `L46_E447/w1` the oracle stores the *whole* decode, and all
//!    11,010,048 elements are compared against that file instead.
//! 3. **Exponent budget**: the module's own `scale_exponent_range` agrees with
//!    the oracle's histogrammed `e8m0_exp_min/max`.
//! 4. **Transcode**: the NVFP4 relabelling borrows the packed nibbles
//!    unchanged, and decoding it reproduces the same bit patterns and the same
//!    sha256.
//!
//! Plus three **negative controls**, because a bit-exact check that cannot
//! fail is worth nothing. Each is a decode that is wrong in one specific way,
//! and each must be caught:
//!
//! - **A, nibble order.** Decoding high-nibble-first must differ from the
//!    correct decode in exactly the fraction the oracle measured for the wrong
//!    order (`_crosscheck.json`, ~0.910 — not 1.0, because a pair whose two
//!    codes happen to be equal survives the swap). Matching that number to the
//!    last digit also confirms this decoder indexes elements the way the
//!    oracle's does, independently of the hash.
//! - **B, signed zero.** Folding −0.0 to +0.0 — a decode that is *numerically*
//!    perfect — must break the sha256, which is what proves check 1 sees the
//!    sign of zero at all.
//! - **C, scale pairing.** Rolling the NVFP4 block scales by one MXFP4 block
//!    must change the decode, which is what makes "each 32-block's scale is
//!    duplicated into its own two 16-blocks" a load-bearing claim rather than
//!    an untested comment.
//!
//! The oracle also ships a 16,384-element random sample per tensor as `.npz`;
//! it is not read here because check 1 covers every element of the same
//! tensors and is strictly stronger.
//!
//! Usage: `mxfp4_gate [ORACLE_DIR]`; without the argument, `$MARY_MODELS/k3-oracle`.
//! Exits nonzero on any failure and prints no timing at all in that case.

use std::path::Path;
use std::time::Instant;

use mary::nn::mxfp4::{
    decode_mxfp4, decode_mxfp4_f16, decode_nvfp4, e4m3_from_pow2, e4m3_to_f32, e8m0_to_f32,
    scale_exponent_range, transcode_to_nvfp4, Nvfp4, E2M1, E4M3_POW2_MAX, E4M3_POW2_MIN, MX_BLOCK,
    NV_BLOCK,
};
use sha2::{Digest, Sha256};

const EXPERTS: [&str; 3] = ["L01_E000", "L46_E447", "L92_E895"];
const TENSORS: [&str; 3] = ["w1", "w2", "w3"];

/// sha256 of a f32 slice as little-endian C-order bytes — the byte stream
/// numpy's `.tobytes()` produces, written explicitly so the hash doesn't
/// depend on the host's float layout.
fn sha256_f32_le(data: &[f32]) -> String {
    let mut hasher = Sha256::new();
    let mut buf = Vec::with_capacity(1 << 18);
    for chunk in data.chunks(1 << 16) {
        buf.clear();
        for v in chunk {
            buf.extend_from_slice(&v.to_le_bytes());
        }
        hasher.update(&buf);
    }
    format!("{:x}", hasher.finalize())
}

/// First index where two f32 slices differ in *bit pattern* (so +0.0 and −0.0
/// count as different, which is the whole point here).
fn first_bit_diff(a: &[f32], b: &[f32]) -> Option<usize> {
    a.iter().zip(b).position(|(x, y)| x.to_bits() != y.to_bits())
}

/// `2^e` as an f64, built from the exponent field so the reference the E4M3
/// window is checked against is itself exact.
fn pow2_f64(e: i32) -> f64 {
    assert!((-1022..=1023).contains(&e));
    f64::from_bits(((e + 1023) as u64) << 52)
}

/// Checks 0a/0b/0c: the three code tables against the oracle's ml_dtypes-built
/// artifacts. Appends to `failures` rather than panicking so one run reports
/// everything that is wrong.
fn check_code_tables(dir: &Path, failures: &mut Vec<String>) {
    let csv = std::fs::read_to_string(dir.join("e2m1_code_table.csv")).expect("e2m1 table");
    let rows: Vec<&str> = csv.lines().skip(1).filter(|l| !l.trim().is_empty()).collect();
    if rows.len() != 16 {
        failures.push(format!("E2M1 table has {} rows, expected 16", rows.len()));
    }
    for (code, line) in rows.iter().enumerate() {
        let want: f32 = line.rsplit(',').next().expect("value column").trim().parse().expect("f32");
        if want.to_bits() != E2M1[code].to_bits() {
            failures.push(format!(
                "E2M1[{code:#x}] = {} ({:08x}) != oracle {want} ({:08x})",
                E2M1[code],
                E2M1[code].to_bits(),
                want.to_bits()
            ));
        }
    }

    let csv = std::fs::read_to_string(dir.join("e8m0_scale_table.csv")).expect("e8m0 table");
    let mut seen = 0usize;
    for line in csv.lines().skip(1).filter(|l| !l.trim().is_empty()) {
        let f: Vec<&str> = line.split(',').collect();
        let byte: u8 = f[0].trim().parse().expect("byte column");
        let mine = e8m0_to_f32(byte);
        seen += 1;
        if f[3].trim() == "NaN" {
            if !mine.is_nan() {
                failures.push(format!("E8M0[{byte}] = {mine}, oracle says NaN"));
            }
        } else {
            let want: f64 = f[3].trim().parse().expect("f64 value");
            // Every 2^(byte-127) for byte 0..254 is exactly an f32 (2^-127 as
            // a subnormal), so widening to f64 must land on the nose.
            if mine as f64 != want {
                failures.push(format!("E8M0[{byte}] = {mine:e}, oracle {want:e}"));
            }
        }
    }
    if seen != 256 {
        failures.push(format!("E8M0 table has {seen} rows, expected 256"));
    }

    let verification: serde_json::Value =
        serde_json::from_slice(&std::fs::read(dir.join("_verification.json")).expect("read _verification.json"))
            .expect("parse _verification.json");
    let want: Vec<i32> = verification["e4m3fn_exact_pow2_exponents"]
        .as_array()
        .expect("e4m3fn_exact_pow2_exponents")
        .iter()
        .map(|v| v.as_i64().expect("exponent") as i32)
        .collect();
    // Sweep far wider than the answer so a codec that accepted too much would
    // show up as extra entries, not just a missing one.
    let mine: Vec<i32> = (-40..=40)
        .filter(|&e| e4m3_from_pow2(e).is_some_and(|b| e4m3_to_f32(b) as f64 == pow2_f64(e)))
        .collect();
    if mine != want {
        failures.push(format!("E4M3 exact power-of-two exponents {mine:?} != oracle {want:?}"));
    }
    if (E4M3_POW2_MIN, E4M3_POW2_MAX) != (want[0], want[want.len() - 1]) {
        failures.push(format!(
            "E4M3_POW2_MIN/MAX = {E4M3_POW2_MIN}/{E4M3_POW2_MAX} but the measured window is \
             {}/{}",
            want[0],
            want[want.len() - 1]
        ));
    }
}

/// Negative control A — the decode a reader gets by assuming the *other*
/// nibble order. Deliberately a copy of `decode_mxfp4`'s loop with the two
/// stores exchanged, so nothing but the order is being varied.
fn decode_nibble_swapped(packed: &[u8], scale: &[u8], rows: usize, cols: usize) -> Vec<f32> {
    let blocks_per_row = cols / MX_BLOCK;
    let mut out = vec![0f32; rows * cols];
    for r in 0..rows {
        for b in 0..blocks_per_row {
            let s = e8m0_to_f32(scale[r * blocks_per_row + b]);
            let pbase = r * (cols / 2) + b * (MX_BLOCK / 2);
            let obase = r * cols + b * MX_BLOCK;
            for k in 0..MX_BLOCK / 2 {
                let byte = packed[pbase + k];
                out[obase + 2 * k] = E2M1[(byte >> 4) as usize] * s;
                out[obase + 2 * k + 1] = E2M1[(byte & 0x0F) as usize] * s;
            }
        }
    }
    out
}

fn main() {
    let dir = mary::paths::model(std::env::args().nth(1).as_deref(), "k3-oracle")
        .unwrap_or_else(|e| {
            eprintln!("{e}");
            std::process::exit(2)
        });
    let stats: serde_json::Value =
        serde_json::from_slice(&std::fs::read(dir.join("_decode_stats.json")).expect("read _decode_stats.json"))
            .expect("parse _decode_stats.json");

    // The oracle's own wrong-nibble-order fractions, keyed by (expert,
    // tensor) — control A reproduces these rather than inventing a threshold.
    let cross: serde_json::Value =
        serde_json::from_slice(&std::fs::read(dir.join("_crosscheck.json")).expect("read _crosscheck.json"))
            .expect("parse _crosscheck.json");
    let mut control: std::collections::HashMap<(&str, &str), f64> = std::collections::HashMap::new();
    for row in cross["real_byte_agreement"].as_array().expect("real_byte_agreement") {
        let tag = row["tag"].as_str().expect("tag");
        let tensor = row["tensor"].as_str().expect("tensor");
        let expert = EXPERTS.iter().find(|e| **e == tag).expect("known expert");
        let tensor = TENSORS.iter().find(|t| **t == tensor).expect("known tensor");
        control.insert(
            (expert, tensor),
            row["hf_vs_WRONG_order_frac_differing"].as_f64().expect("wrong-order fraction"),
        );
    }

    println!("mxfp4_gate — oracle {}", dir.display());
    println!("E4M3FN exact power-of-two window: 2^{E4M3_POW2_MIN} .. 2^{E4M3_POW2_MAX} ({} octaves)\n",
             E4M3_POW2_MAX - E4M3_POW2_MIN + 1);
    println!("{:<10} {:<3} {:>10} {:>4} {:>6} {:>4} {:>6} {:>9} {:>10} {:>6} {:>9} {:>9}",
             "expert", "t", "elements", "oct", "global", "sub", "decode", "transcode", "vs-npy",
             "f16", "ctlA-val", "ctlA-bit");

    let mut failures: Vec<String> = Vec::new();
    check_code_tables(&dir, &mut failures);
    let mut total_elems: u64 = 0;
    let mut decode_secs = 0f64;
    let mut transcode_secs = 0f64;

    for expert in EXPERTS {
        for t in TENSORS {
            let meta = &stats[expert]["tensors"][t];
            let rows = meta["logical_shape"][0].as_u64().expect("logical_shape") as usize;
            let cols = meta["logical_shape"][1].as_u64().expect("logical_shape") as usize;
            let want_sha = meta["sha256_decoded_f32_full"].as_str().expect("sha256").to_string();
            let want_emin = meta["e8m0_exp_min"].as_i64().expect("e8m0_exp_min") as i32;
            let want_emax = meta["e8m0_exp_max"].as_i64().expect("e8m0_exp_max") as i32;
            let tag = format!("{expert}/{t}");

            let packed = std::fs::read(dir.join(format!("{expert}_{t}_packed.u8.bin")))
                .unwrap_or_else(|e| panic!("read {tag} packed: {e}"));
            let scale = std::fs::read(dir.join(format!("{expert}_{t}_scale.u8.bin")))
                .unwrap_or_else(|e| panic!("read {tag} scale: {e}"));

            // --- check 1/2: decode ------------------------------------------
            let t0 = Instant::now();
            let mx = decode_mxfp4(&packed, &scale, rows, cols);
            decode_secs += t0.elapsed().as_secs_f64();
            total_elems += mx.len() as u64;

            let got_sha = sha256_f32_le(&mx);
            if got_sha != want_sha {
                failures.push(format!("{tag}: decode sha256 {got_sha} != oracle {want_sha}"));
            }

            // The oracle stores one tensor's decode in full; where it exists,
            // compare against that rather than the 64-row head — same check,
            // 172x more of it, and against a file rather than a hash.
            let full_path = dir.join(format!("{expert}_{t}_dec_FULL.f32.npy"));
            let (ref_path, ref_rows) = if full_path.exists() {
                (full_path, rows)
            } else {
                (dir.join(format!("{expert}_{t}_dec_head64.f32.npy")), 64)
            };
            let (reference, ref_shape) = mary::nn::npy::load_npy(&ref_path).expect("load reference npy");
            if ref_shape != vec![ref_rows, cols] {
                failures.push(format!("{tag}: {ref_shape:?} != [{ref_rows}, {cols}] in {ref_path:?}"));
            } else if let Some(i) = first_bit_diff(&reference, &mx[..reference.len()]) {
                failures.push(format!(
                    "{tag}: {ref_path:?} differs at {i}: oracle {:08x} vs mary {:08x}",
                    reference[i].to_bits(),
                    mx[i].to_bits()
                ));
            }
            let ref_elems = reference.len();

            // --- check 3: exponent budget -----------------------------------
            let (e_min, e_max) = match scale_exponent_range(&scale) {
                Ok(r) => r,
                Err(e) => {
                    failures.push(format!("{tag}: scale_exponent_range: {e}"));
                    continue;
                }
            };
            if (e_min, e_max) != (want_emin, want_emax) {
                failures.push(format!(
                    "{tag}: exponent range {e_min}..{e_max} != oracle {want_emin}..{want_emax}"
                ));
            }
            let octaves = e_max - e_min + 1;

            // --- check 4: transcode ------------------------------------------
            let t0 = Instant::now();
            let nv = match transcode_to_nvfp4(&packed, &scale, rows, cols) {
                Ok(nv) => nv,
                Err(e) => {
                    failures.push(format!("{tag}: transcode refused: {e}"));
                    continue;
                }
            };
            transcode_secs += t0.elapsed().as_secs_f64();

            if nv.packed.as_ptr() != packed.as_ptr() || nv.packed.len() != packed.len() {
                failures.push(format!("{tag}: transcode did not borrow the packed nibbles"));
            }
            if nv.block_scale.len() != scale.len() * 2 {
                failures.push(format!(
                    "{tag}: {} block scales, expected 2 x {}",
                    nv.block_scale.len(),
                    scale.len()
                ));
            }

            let back = decode_nvfp4(&nv);
            if let Some(i) = first_bit_diff(&mx, &back) {
                failures.push(format!(
                    "{tag}: NVFP4 decode differs at {i}: mxfp4 {:08x} vs nvfp4 {:08x}",
                    mx[i].to_bits(),
                    back[i].to_bits()
                ));
            }
            let back_sha = sha256_f32_le(&back);
            if back_sha != want_sha {
                failures.push(format!("{tag}: NVFP4 sha256 {back_sha} != oracle {want_sha}"));
            }

            // --- control A: wrong nibble order ------------------------------
            // numpy's `(a != b).mean()`, i.e. VALUE inequality, which is how
            // the oracle's fraction was computed — so ±0.0 pairs count as
            // agreeing in both. The bit-level fraction is carried alongside
            // because it is the stricter statement and they need not be equal.
            let swapped = decode_nibble_swapped(&packed, &scale, rows, cols);
            #[allow(clippy::float_cmp)]
            let swap_val_diff = mx.iter().zip(&swapped).filter(|(a, b)| a != b).count();
            let swap_bit_diff =
                mx.iter().zip(&swapped).filter(|(a, b)| a.to_bits() != b.to_bits()).count();
            let swap_frac = swap_val_diff as f64 / mx.len() as f64;
            // Compare the element COUNT the oracle's fraction encodes, not the
            // fraction. The denominator is 11,010,048 = 21·2^19, so the ratio
            // is not exactly representable in f64 and `count/n` here vs
            // numpy's `mean()` there legitimately differ by 1 ULP on 4 of the
            // 9 tensors — which says nothing about the decode. The count is an
            // integer well inside f64's exact range and recovers cleanly.
            let want_exact = control[&(expert, t)] * mx.len() as f64;
            let want_count = want_exact.round();
            assert!(
                (want_exact - want_count).abs() < 1e-3,
                "{tag}: oracle fraction {want_exact} does not encode an integer count"
            );
            if swap_val_diff as f64 != want_count {
                failures.push(format!(
                    "{tag}: control A — {swap_val_diff} elements differ under the wrong nibble \
                     order, oracle says {want_count} (fractions {swap_frac:.16} vs {:.16})",
                    control[&(expert, t)]
                ));
            }

            // --- control B: signed zero -------------------------------------
            #[allow(clippy::float_cmp)]
            let collapsed: Vec<f32> = mx.iter().map(|&v| if v == 0.0 { 0.0 } else { v }).collect();
            if sha256_f32_le(&collapsed) == want_sha {
                failures.push(format!(
                    "{tag}: control B — folding -0.0 to +0.0 still matched the oracle sha256, so \
                     the gate does not see the sign of zero"
                ));
            }

            // --- control C: NVFP4 scale pairing -----------------------------
            // Roll the block scales by one MXFP4 block (two NVFP4 blocks): the
            // duplication is still 2-for-1, only misaligned.
            let mut rolled_scale = nv.block_scale.clone();
            rolled_scale.rotate_left(MX_BLOCK / NV_BLOCK);
            let rolled = Nvfp4 {
                packed: nv.packed,
                block_scale: rolled_scale,
                global_scale: nv.global_scale,
                global_exp: nv.global_exp,
                subnormal_block_scales: nv.subnormal_block_scales,
                rows,
                cols,
            };
            if first_bit_diff(&mx, &decode_nvfp4(&rolled)).is_none() {
                failures.push(format!(
                    "{tag}: control C — rolling the block scales changed nothing, so the pairing \
                     is not being tested"
                ));
            }

            // Not a gate: how much of this tensor survives a trip through f16,
            // reported so the f16 decode path is never assumed exact.
            let h = decode_mxfp4_f16(&packed, &scale, rows, cols);
            let f16_lossy = h.iter().zip(&mx).filter(|(a, b)| a.to_f32().to_bits() != b.to_bits()).count();

            println!(
                "{:<10} {:<3} {:>10} {:>4} {:>6} {:>4} {:>6} {:>9} {:>10} {:>6} {:>9} {:>9}",
                expert,
                t,
                mx.len(),
                octaves,
                format!("2^{}", nv.global_exp),
                if nv.subnormal_block_scales { "yes" } else { "no" },
                if got_sha == want_sha { "exact" } else { "DIFF" },
                if back_sha == want_sha { "exact" } else { "DIFF" },
                ref_elems,
                if f16_lossy == 0 { "exact".to_string() } else { format!("{f16_lossy}") },
                format!("{swap_frac:.6}"),
                format!("{:.6}", swap_bit_diff as f64 / mx.len() as f64),
            );
        }
    }

    println!();
    if !failures.is_empty() {
        eprintln!("GATE FAILED — {} problem(s):", failures.len());
        for f in &failures {
            eprintln!("  {f}");
        }
        std::process::exit(1);
    }

    println!(
        "GATE PASSED — {total_elems} elements over {} tensors, decoded bit-identically to the \
         oracle, and again after the MXFP4->NVFP4 transcode.\n\
         Controls: the wrong nibble order differs in the oracle's exact fraction, folding -0.0 \
         breaks the hash, and rolling the block scales breaks the transcode — the checks have \
         power.",
        EXPERTS.len() * TENSORS.len()
    );
    // Timing only past the gate, and only as throughput of the scalar CPU
    // reference path — this is the correctness codec, not a loader kernel.
    println!(
        "decode {:.2} s ({:.0} Melem/s), transcode {:.3} s ({:.1} Mscale/s), single-threaded",
        decode_secs,
        total_elems as f64 / decode_secs / 1e6,
        transcode_secs,
        total_elems as f64 / 32.0 / transcode_secs / 1e6,
    );
}
