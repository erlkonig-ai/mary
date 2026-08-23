//! kimi_situ_gate — the correctness gate for `mary::models::k3::situ`.
//!
//! The oracle is `situ_activation.npz`, captured by instantiating and running
//! Kimi K3's *shipped* `SituAndMul` module (torch 2.13, CPU) over a 1868-point
//! sweep, a 34×34 sign-quadrant grid and one MoE-shaped block of 8×6144. Every
//! `*_y_f32` column is that module's verbatim forward pass; every `*_f64`
//! column is an independent float64 transcription of the same formula. Only
//! the arithmetic is independent — `situ` is Moonshot's own activation and no
//! third-party implementation of it exists to execute, so the *mathematics*
//! rests on one reading of one source file. That limit is inherited here and
//! is not something this gate can close; what it does close is whether this
//! Burn port computes the same function as the shipped module, to f32.
//!
//! Four things run per backend, in this order:
//!
//!   1. STRUCTURE — the invariants the formula implies regardless of oracle:
//!      both branches saturate at exactly ±beta / ±linear_beta far out, the
//!      product is bounded by their product, nothing is NaN.
//!   2. PRIMITIVES — the backend's own `tanh` and `sigmoid`, alone, against
//!      float64, reported in f32 ulp. `situ` is nothing but those two plus
//!      three products, so this measures how much of any deviation belongs to
//!      the backend rather than to the port, and it fixes this lane's error
//!      budget *before* the port's own numbers are looked at.
//!   3. ORACLE — every case against `*_y_f32` (the shipped module's own f32
//!      arithmetic) and against the f64 reference.
//!   4. CONTROL — four *wrong* formulas, each a plausible mis-port, through
//!      the identical comparison at the identical budget. They must all FAIL.
//!      A criterion that cannot reject a wrong formula is not evidence that
//!      the right one passed — and re-running the controls at each lane's own
//!      budget is what stops a widened budget from being a free pass.
//!
//! Extreme inputs are the point. The sweep runs to ±3e38 and down to a f64
//! denormal, and includes both knees (±4, ±25) and their multiples, because a
//! soft clip that is subtly wrong — a missing `/beta`, swapped betas, an
//! unclipped up branch — is invisible near zero and only shows in saturation.
//!
//! Run (from the worktree; CUDA lane on top of the CPU lane):
//!   `cargo run --release --features kimi-k3-cuda --bin kimi_situ_gate`
//! CPU + wgpu lanes only:
//!   `cargo run --release --features kimi-k3 --bin kimi_situ_gate`
//! The oracle directory (the `.npz` members unpacked into individual `.npy`
//! files) comes from argv[1], or `SITU_ORACLE_DIR`, or
//! `$MARY_MODELS/k3-situ/oracle_npy`. There is no baked-in path.

use burn::prelude::*;
use burn::tensor::activation::sigmoid;
use mary::models::k3::Situ;
use std::collections::HashMap;
use std::fs;
use std::panic::{self, AssertUnwindSafe};
use std::path::Path;

// ---------------------------------------------------------------------------
// Tolerance
// ---------------------------------------------------------------------------

/// One f32 ulp, relative — the unit every budget below is quoted in.
const F32_EPS: f64 = f32::EPSILON as f64;

/// The fixed relative floor: 1e-6, i.e. ~8.4 × f32 eps.
///
/// It covers the *reference* side of the comparison — torch's own f32
/// `tanh`/`sigmoid` are a couple of ulp off the true value, and that error
/// propagates through three products — plus a lane whose primitives are as
/// accurate as a good libm. No lane is ever held to less than this.
const RTOL_FLOOR: f64 = 1e-6;

/// Absolute tolerance: 1e-9.
///
/// Outputs are bounded by `beta * linear_beta = 100`, and this activation's
/// only consumer is `w2`, a 3072-term f32 dot product whose own accumulation
/// noise is ~1e-5 absolute. 1e-9 is four decades below that — provably
/// invisible downstream. It exists for the handful of sweep points whose true
/// product underflows f32 entirely (`gate = up = 1e-30` → 5e-61, which is 0 in
/// f32), where a relative comparison is meaningless rather than strict.
const ATOL: f64 = 1e-9;

/// Relative error is only *measured* where the reference is large enough for
/// the ratio to carry information in f32; below this it is dominated by the
/// reference's own representation error.
const REL_FLOOR: f64 = 1e-6;

/// Linear error-propagation bound for `situ`, in f32 ulp, given a backend
/// whose `tanh` is `u_tanh` ulp and whose `sigmoid` is `u_sig` ulp.
///
/// `out = (beta·tanh(g/beta) · sigmoid(g)) · (lb·tanh(u/lb))` — two `tanh`
/// calls, one `sigmoid`, and (division, two scalar products, two tensor
/// products) = 5 roundings, each ≤ 0.5 ulp and counted here at 1. Relative
/// errors add across a product, so the bound is the sum. This is a *derived*
/// budget, not a fitted one: it is computed from the lane's measured
/// primitives before the port's own error is compared against it.
fn composed_budget_ulp(u_tanh: f64, u_sig: f64) -> f64 {
    2.0 * u_tanh + u_sig + 5.0
}

// ---------------------------------------------------------------------------
// .npy reading
// ---------------------------------------------------------------------------

/// A loaded array, always widened to f64 (the file's dtypes are `<f8`, `<f4`,
/// `<u2`; the last is kept as raw bits in `bits` for the bf16 columns).
struct Arr {
    shape: Vec<usize>,
    data: Vec<f64>,
    bits: Vec<u16>,
}

impl Arr {
    fn len(&self) -> usize {
        self.data.len().max(self.bits.len())
    }
    fn as_f32(&self) -> Vec<f32> {
        self.data.iter().map(|&v| v as f32).collect()
    }
}

/// Minimal `.npy` v1/v2 reader for exactly the three little-endian C-order
/// dtypes this oracle uses. It lives in the gate rather than in
/// `mary::nn::npy` (which handles f32 and i64) because it is test-fixture
/// plumbing: no model path reads f64 or raw u16 from disk.
fn read_npy(path: &Path) -> Arr {
    let buf = fs::read(path).unwrap_or_else(|e| panic!("read {}: {e}", path.display()));
    assert!(
        buf.len() > 10 && buf[0] == 0x93 && &buf[1..6] == b"NUMPY",
        "{} is not a .npy file",
        path.display()
    );
    let (hlen, hstart) = match buf[6] {
        1 => (u16::from_le_bytes([buf[8], buf[9]]) as usize, 10),
        _ => (
            u32::from_le_bytes([buf[8], buf[9], buf[10], buf[11]]) as usize,
            12,
        ),
    };
    let header = std::str::from_utf8(&buf[hstart..hstart + hlen]).expect("npy header utf8");
    let field = |k: &str| -> String {
        let i = header
            .find(k)
            .unwrap_or_else(|| panic!("no {k} in {header}"))
            + k.len();
        let rest = &header[i..];
        let start = rest.find(|c: char| c != ':' && c != ' ').unwrap();
        let rest = &rest[start..];
        let end = rest.find(',').unwrap_or(rest.len());
        rest[..end].trim().trim_matches('\'').to_string()
    };
    assert_eq!(field("'fortran_order'"), "False", "C-order only");
    let descr = field("'descr'");

    // shape is a tuple, so it has to be sliced out by parens rather than by
    // the comma-terminated `field` above.
    let s0 = header.find("'shape'").expect("no shape") + "'shape'".len();
    let s1 = header[s0..].find('(').expect("shape tuple") + s0 + 1;
    let s2 = header[s1..].find(')').expect("shape tuple") + s1;
    let shape: Vec<usize> = header[s1..s2]
        .split(',')
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .map(|s| s.parse().expect("shape dim"))
        .collect();
    let n: usize = shape.iter().product(); // empty shape (a 0-d scalar) => 1

    let body = &buf[hstart + hlen..];
    let (mut data, mut bits) = (Vec::new(), Vec::new());
    match descr.as_str() {
        "<f8" => {
            assert_eq!(body.len(), n * 8, "{}: f8 body size", path.display());
            data = body
                .chunks_exact(8)
                .map(|c| f64::from_le_bytes(c.try_into().unwrap()))
                .collect();
        }
        "<f4" => {
            assert_eq!(body.len(), n * 4, "{}: f4 body size", path.display());
            data = body
                .chunks_exact(4)
                .map(|c| f32::from_le_bytes(c.try_into().unwrap()) as f64)
                .collect();
        }
        "<u2" => {
            assert_eq!(body.len(), n * 2, "{}: u2 body size", path.display());
            bits = body
                .chunks_exact(2)
                .map(|c| u16::from_le_bytes(c.try_into().unwrap()))
                .collect();
        }
        other => panic!("{}: unsupported dtype {other}", path.display()),
    }
    Arr { shape, data, bits }
}

/// Check the loaded arrays against `index.json`'s declaration.
///
/// The gate read whatever `.npy` files happened to be in the directory and
/// compared against them. `index.json` sits beside those files carrying the
/// source `.npz` path, its sha256, and every array's shape and dtype — and was
/// never opened. So a regenerated, truncated, partially-extracted or simply
/// different oracle would have been compared against silently, and a green
/// result would describe agreement with an unknown reference.
///
/// This does not prove the oracle is *correct*; it proves it is the SAME oracle
/// the gate was written against. Those are different claims and only the second
/// is checkable here.
fn verify_against_index(dir: &Path, loaded: &HashMap<String, Arr>) {
    let idx_path = dir.join("index.json");
    let raw = match fs::read_to_string(&idx_path) {
        Ok(t) => t,
        Err(e) => panic!(
            "{}: {e}. The oracle's manifest is how this gate knows it is reading the \
             intended arrays; refusing to compare against an unidentified reference.",
            idx_path.display()
        ),
    };
    let idx: serde_json::Value =
        serde_json::from_str(&raw).unwrap_or_else(|e| panic!("{}: {e}", idx_path.display()));
    let arrays = idx
        .get("arrays")
        .and_then(|a| a.as_object())
        .unwrap_or_else(|| panic!("{} has no `arrays` object", idx_path.display()));

    println!(
        "oracle identity: {} arrays declared, source sha256 {}",
        arrays.len(),
        idx.get("source_sha256")
            .and_then(|v| v.as_str())
            .unwrap_or("(absent)")
    );

    let mut problems: Vec<String> = Vec::new();
    for (name, spec) in arrays {
        let Some(got) = loaded.get(name) else {
            problems.push(format!("declared but not loaded: {name}"));
            continue;
        };
        let want: Vec<usize> = spec
            .get("shape")
            .and_then(|v| v.as_array())
            .map(|a| {
                a.iter()
                    .filter_map(|v| v.as_u64().map(|n| n as usize))
                    .collect()
            })
            .unwrap_or_default();
        if want != got.shape {
            problems.push(format!(
                "{name}: shape {:?} declared, {:?} loaded",
                want, got.shape
            ));
        }
        let n: usize = want
            .iter()
            .product::<usize>()
            .max(if want.is_empty() { 1 } else { 0 });
        if got.len() != n {
            problems.push(format!(
                "{name}: {n} elements declared, {} loaded",
                got.len()
            ));
        }
    }
    for name in loaded.keys() {
        if !arrays.contains_key(name) {
            problems.push(format!("loaded but not declared: {name}"));
        }
    }
    assert!(
        problems.is_empty(),
        "oracle does not match its own index.json — {} discrepanc(y/ies):\n  {}",
        problems.len(),
        problems.join("\n  ")
    );
    println!("oracle identity: OK — every declared array present with the declared shape\n");
}

fn load_oracle(dir: &Path) -> HashMap<String, Arr> {
    let mut out = HashMap::new();
    for entry in fs::read_dir(dir).unwrap_or_else(|e| panic!("read_dir {}: {e}", dir.display())) {
        let p = entry.expect("dir entry").path();
        if p.extension().map(|e| e == "npy").unwrap_or(false) {
            let name = p.file_stem().unwrap().to_string_lossy().to_string();
            out.insert(name, read_npy(&p));
        }
    }
    assert!(
        out.contains_key("diag_y_f32"),
        "{} holds no situ oracle (expected the .npz members as .npy)",
        dir.display()
    );
    verify_against_index(dir, &out);
    out
}

// ---------------------------------------------------------------------------
// Comparison
// ---------------------------------------------------------------------------

#[derive(Default, Clone)]
struct Cmp {
    n: usize,
    n_exact: usize,
    n_fail: usize,
    /// Same criterion at the fixed `RTOL_FLOOR`, so every report shows exactly
    /// what the lane's derived budget bought it. Nothing is hidden behind the
    /// widening: if these two differ, the difference is printed.
    n_fail_floor: usize,
    n_rel: usize,
    max_abs: f64,
    max_rel: f64,
    rtol: f64,
    worst_fail: Option<(usize, f64, f64)>,
}

/// `|got − want| <= ATOL + rtol·|want|`, per point.
fn compare(got: &[f32], want: &[f64], rtol: f64) -> Cmp {
    assert_eq!(got.len(), want.len(), "length mismatch in comparison");
    // NON-EMPTY, and this is not pedantry. Equal lengths were asserted and
    // non-emptiness was not, so a zero-length oracle array compared to a
    // zero-length result reported `tanh 0.00 ulp` — a green measurement of
    // nothing — AND tightened the derived budget to the floor. On a machine
    // where the arrays failed to load, or a CPU-only box, the whole gate went
    // green having compared no numbers at all.
    assert!(
        !got.is_empty(),
        "empty comparison: {} elements. An empty array compares equal to anything; \
         this is the vacuous-green path, not a pass.",
        got.len()
    );
    let mut c = Cmp {
        n: got.len(),
        rtol,
        ..Default::default()
    };
    for (i, (&g, &w)) in got.iter().zip(want.iter()).enumerate() {
        assert!(!g.is_nan(), "port produced NaN at {i}");
        let g = g as f64;
        if g.to_bits() == (w as f32 as f64).to_bits() {
            c.n_exact += 1;
        }
        let abs = (g - w).abs();
        c.max_abs = c.max_abs.max(abs);
        if w.abs() >= REL_FLOOR {
            c.n_rel += 1;
            c.max_rel = c.max_rel.max(abs / w.abs());
        }
        if abs > ATOL + RTOL_FLOOR * w.abs() {
            c.n_fail_floor += 1;
        }
        if abs > ATOL + rtol * w.abs() {
            c.n_fail += 1;
            if c.worst_fail.map(|(_, _, e)| abs > e).unwrap_or(true) {
                c.worst_fail = Some((i, g, abs));
            }
        }
    }
    c
}

fn report(label: &str, c: &Cmp) -> bool {
    let ok = c.n_fail == 0;
    println!(
        "  {:<32} n={:<6} maxabs={:>9.3e}  maxrel={:>9.3e} = {:>5.2} ulp ({:>5.0}% of budget)  bitexact={:>5}/{:<6} {}",
        label,
        c.n,
        c.max_abs,
        c.max_rel,
        c.max_rel / F32_EPS,
        100.0 * c.max_rel / c.rtol,
        c.n_exact,
        c.n,
        if ok { "PASS" } else { "FAIL" }
    );
    if c.n_fail_floor != c.n_fail {
        println!(
            "      (at the fixed {RTOL_FLOOR:.0e} floor instead of this lane's budget: {} would fail)",
            c.n_fail_floor
        );
    }
    if let Some((i, g, e)) = c.worst_fail {
        println!(
            "      {} failing points; worst at idx {i}: got {g:.9e}, abs err {e:.3e}",
            c.n_fail
        );
    }
    ok
}

// ---------------------------------------------------------------------------
// The lanes
// ---------------------------------------------------------------------------

fn t1<B: Backend>(v: &[f32], dev: &Device<B>) -> Tensor<B, 1> {
    Tensor::from_data(TensorData::new(v.to_vec(), [v.len()]), dev)
}

fn t2<B: Backend>(v: &[f32], r: usize, c: usize, dev: &Device<B>) -> Tensor<B, 2> {
    assert_eq!(v.len(), r * c);
    Tensor::from_data(TensorData::new(v.to_vec(), [r, c]), dev)
}

fn host<B: Backend, const D: usize>(t: Tensor<B, D>) -> Vec<f32> {
    t.into_data().to_vec::<f32>().expect("host readback")
}

/// float64 sigmoid, in the branch-stable form, so the primitive measurement
/// below is limited by the backend and not by the reference.
fn sigmoid64(x: f64) -> f64 {
    if x >= 0.0 {
        1.0 / (1.0 + (-x).exp())
    } else {
        let e = x.exp();
        e / (1.0 + e)
    }
}

/// What this backend's arithmetic does, measured before any of the port's own
/// numbers are looked at:
///
/// * max relative error of `tanh` and `sigmoid`, in f32 ulp, over the same
///   sweep the activation is gated on — against float64 evaluated at the
///   *f32-rounded* input, so it is the function's error and not the input's;
/// * whether the lane flushes f32 subnormals to zero. Two independent probes:
///   a subnormal doubled (`1e-40 * 2`, no transcendental involved), and
///   `sigmoid(-88)`, whose true value 6.05e-39 is subnormal but nonzero. A
///   lane that returns exactly 0 for both is flushing, not merely imprecise.
fn probe_backend<B: Backend>(dev: &Device<B>, x32: &[f32]) -> (f64, f64, bool) {
    let x64: Vec<f64> = x32.iter().map(|&v| v as f64).collect();
    let th = host(t1::<B>(x32, dev).tanh());
    let sg = host(sigmoid(t1::<B>(x32, dev)));
    let (mut ut, mut us) = (0.0f64, 0.0f64);
    for i in 0..x64.len() {
        let rt = x64[i].tanh();
        if rt.abs() >= REL_FLOOR {
            ut = ut.max((th[i] as f64 - rt).abs() / rt.abs() / F32_EPS);
        }
        let rs = sigmoid64(x64[i]);
        if rs.abs() >= REL_FLOOR {
            us = us.max((sg[i] as f64 - rs).abs() / rs.abs() / F32_EPS);
        }
    }
    let doubled = host(t1::<B>(&[1e-40f32], dev).mul_scalar(2.0))[0];
    let sig_sub = host(sigmoid(t1::<B>(&[-88.0f32], dev)))[0];
    debug_assert!(sigmoid64(-88.0) > 0.0 && sigmoid64(-88.0) < f32::MIN_POSITIVE as f64);
    let ftz = doubled == 0.0 && sig_sub == 0.0;
    (ut, us, ftz)
}

/// Everything the gate checks on one backend. Returns `true` on a clean pass.
fn run_lane<B: Backend>(name: &str, dev: &Device<B>, o: &HashMap<String, Arr>) -> bool {
    println!("\n=== lane: {name} ===");
    let situ = Situ::k3();
    let mut ok = true;
    let get = |k: &str| o.get(k).unwrap_or_else(|| panic!("oracle missing {k}"));

    // -- 1. STRUCTURE ------------------------------------------------------
    // The formula's own consequences, checked without the oracle. Saturation
    // must be *exact*: tanh(±7.5e37) is ±1.0 to the last bit in f32, so the
    // branches must land on the betas themselves, not merely near them.
    let probe = [
        3e38f32, 1e30, 1e12, 100.0, 40.0, -40.0, -100.0, -1e30, -3e38,
    ];
    let g = host(situ.gate_branch(t1::<B>(&probe, dev)));
    let u = host(situ.up_branch(t1::<B>(&probe, dev)));
    let sat_ok = g[0] == 4.0 && g[1] == 4.0 && u[0] == 25.0 && *u.last().unwrap() == -25.0;
    let bound = situ.output_bound() as f32;
    let range_ok = g.iter().all(|&v| (-0.27..=4.0).contains(&v))
        && u.iter().all(|&v| v.abs() <= 25.0)
        && g.iter().zip(&u).all(|(&a, &b)| (a * b).abs() <= bound);
    println!(
        "  saturation  gate(+3e38)={} gate(+1e30)={} up(+3e38)={} up(-3e38)={}  {}",
        g[0],
        g[1],
        u[0],
        u.last().unwrap(),
        if sat_ok { "PASS" } else { "FAIL" }
    );
    println!(
        "  ranges      gate in [-0.26977,4], up in [-25,25], |product| <= {bound}  {}",
        if range_ok { "PASS" } else { "FAIL" }
    );
    if !(sat_ok && range_ok) {
        println!("  structure failed; not reporting oracle numbers for this lane");
        return false;
    }

    // -- 2. PRIMITIVES -----------------------------------------------------
    // Fixed before anything about the port is measured.
    let sweep = get("sweep_x_f64").as_f32();
    let (u_tanh, u_sig, ftz) = probe_backend::<B>(dev, &sweep);
    let budget_ulp = composed_budget_ulp(u_tanh, u_sig);
    let rtol = RTOL_FLOOR.max(budget_ulp * F32_EPS);
    println!(
        "  primitives  tanh {u_tanh:.2} ulp, sigmoid {u_sig:.2} ulp  ->  composed budget \
         {budget_ulp:.1} ulp; lane rtol = max({RTOL_FLOOR:.1e}, {:.3e}) = {rtol:.3e}",
        budget_ulp * F32_EPS
    );
    println!(
        "  subnormals  {}",
        if ftz {
            "FLUSHED TO ZERO (1e-40*2 -> 0 and sigmoid(-88) -> 0); outputs below \
             ~1.2e-38 collapse to 0 on this lane"
        } else {
            "preserved (1e-40*2 and sigmoid(-88) both come back nonzero)"
        }
    );

    // -- 3. ORACLE ---------------------------------------------------------
    // (a) each branch alone over the full 1868-point sweep. Only an f64 column
    //     exists for these, so the comparison carries f32's own representation
    //     error too; it is here for coverage of the branches in isolation.
    let gb = host(situ.gate_branch(t1::<B>(&sweep, dev)));
    ok &= report(
        "gate branch vs f64",
        &compare(&gb, &get("situ_gate_branch_f64").data, rtol),
    );
    let ub = host(situ.up_branch(t1::<B>(&sweep, dev)));
    ok &= report(
        "up branch vs f64",
        &compare(&ub, &get("situ_up_branch_f64").data, rtol),
    );

    // (b) the diagonal (gate == up == sweep) through the concatenated
    //     `forward`, which also exercises the last-dim split.
    let dx = get("diag_x_f64");
    let n = dx.shape[0];
    let dy = host(situ.forward(t2::<B>(&dx.as_f32(), n, 2, dev)));
    ok &= report(
        "diag forward vs shipped f32",
        &compare(&dy, &get("diag_y_f32").data, rtol),
    );
    ok &= report(
        "diag forward vs f64",
        &compare(&dy, &get("diag_y_f64").data, rtol),
    );

    // (c) the 34×34 sign-quadrant grid through `forward_pair` — the API a
    //     fused expert path uses, and the only case covering gate<0/up>0.
    let gg = get("grid_gate_f64").as_f32();
    let gu = get("grid_up_f64").as_f32();
    let gy = host(situ.forward_pair(t1::<B>(&gg, dev), t1::<B>(&gu, dev)));
    ok &= report(
        "grid pair vs shipped f32",
        &compare(&gy, &get("grid_y_f32").data, rtol),
    );
    ok &= report(
        "grid pair vs f64",
        &compare(&gy, &get("grid_y_f64").data, rtol),
    );

    // (d) a realistic MoE-shaped block: 8 tokens × 2·3072, sd 6 so most of it
    //     sits past the beta=4 knee.
    let rx = get("rand_x_f64");
    let (rr, rc) = (rx.shape[0], rx.shape[1]);
    let ry = host(situ.forward(t2::<B>(&rx.as_f32(), rr, rc, dev)));
    ok &= report(
        "rand block vs shipped f32",
        &compare(&ry, &get("rand_y_f32").data, rtol),
    );
    ok &= report(
        "rand block vs f64",
        &compare(&ry, &get("rand_y_f64").data, rtol),
    );

    // (e) bf16 in / bf16 out — the storage dtype of a real forward pass. The
    //     shipped module rounds only at the two ends, so this is the f32 path
    //     with rounded inputs and a rounded result. bf16's relative ulp is
    //     2^-8 = 3.9e-3, four orders coarser than any lane's rtol, so a
    //     mismatch can only ever be the rounding of a value that sits on a
    //     boundary: 1 ulp is expected, 2 is not, and would mean the port
    //     rounds somewhere the reference does not.
    for (xk, yk, rows) in [
        ("diag_x_bf16_bits", "diag_y_bf16_bits", n),
        (
            "grid_x_bf16_bits",
            "grid_y_bf16_bits",
            get("grid_x_bf16_bits").shape[0],
        ),
        ("rand_x_bf16_bits", "rand_y_bf16_bits", rr),
    ] {
        let xa = get(xk);
        let cols = xa.len() / rows;
        let xf: Vec<f32> = xa
            .bits
            .iter()
            .map(|&b| f32::from_bits((b as u32) << 16))
            .collect();
        let out = host(situ.forward(t2::<B>(&xf, rows, cols, dev)));
        let want = &get(yk).bits;
        assert_eq!(out.len(), want.len());
        let (mut exact, mut one_ulp, mut worse, mut flushed) = (0usize, 0usize, 0usize, 0usize);
        let mut worst: Option<(usize, u16, u16, f32)> = None;
        for (i, (&f, &w)) in out.iter().zip(want.iter()).enumerate() {
            let b = half::bf16::from_f32(f).to_bits();
            // bf16 patterns are monotone within a sign, so a 1-ulp neighbour
            // differs by 1 in the raw pattern; ±0 differ by the sign bit only.
            let d = (b as i32 - w as i32).unsigned_abs();
            let want_val = half::bf16::from_bits(w).to_f32() as f64;
            if d == 0 {
                exact += 1;
            } else if d == 1 || (b ^ w) == 0x8000 {
                one_ulp += 1;
            } else if ftz && f == 0.0 && want_val.abs() < ATOL {
                // The signature of the flush measured above, and only that:
                // the port produced an exact zero where the reference is a
                // value already below this gate's stated absolute floor. The
                // exemption is conditioned on an independently measured
                // backend property, not on how large the disagreement is.
                flushed += 1;
            } else {
                worse += 1;
                if worst.is_none() {
                    worst = Some((i, b, w, f));
                }
            }
        }
        let pass = worse == 0;
        println!(
            "  {:<32} n={:<6} bitexact={} 1ulp={} flushed={} unexplained={}  {}",
            format!("bf16 {}", xk.trim_end_matches("_x_bf16_bits")),
            out.len(),
            exact,
            one_ulp,
            flushed,
            worse,
            if pass { "PASS" } else { "FAIL" }
        );
        if let Some((i, b, w, f)) = worst {
            println!(
                "      first >1ulp mismatch at idx {i}: got bits 0x{b:04x} ({:e}) want 0x{w:04x} ({:e}); f32 was {f:e}",
                half::bf16::from_bits(b).to_f32(),
                half::bf16::from_bits(w).to_f32()
            );
        }
        ok &= pass;
    }

    // -- 4. CONTROL --------------------------------------------------------
    ok &= run_controls::<B>(dev, o, rtol);
    ok
}

/// Four plausible mis-ports, run through the identical comparison at this
/// lane's budget. Each must be REJECTED; if any slips through, the budget is
/// too loose to be evidence and the lane reports failure.
fn run_controls<B: Backend>(dev: &Device<B>, o: &HashMap<String, Arr>, rtol: f64) -> bool {
    println!("  -- controls (each of these MUST fail) --");
    let get = |k: &str| o.get(k).unwrap_or_else(|| panic!("oracle missing {k}"));
    let gg = get("grid_gate_f64").as_f32();
    let gu = get("grid_up_f64").as_f32();
    let want = &get("grid_y_f32").data;
    let mut all_rejected = true;

    // (1) betas swapped and (2) the up branch left unclipped are both
    //     expressible through the real module, so they run the exact same
    //     Burn path the port does.
    let mut check = |label: &str, y: &[f32]| {
        let c = compare(y, want, rtol);
        let rejected = c.n_fail > 0;
        println!(
            "     {:<31} maxabs={:>9.3e}  rejected {:>5}/{:<6} {}",
            label,
            c.max_abs,
            c.n_fail,
            c.n,
            if rejected { "ok" } else { "SLIPPED THROUGH" }
        );
        all_rejected &= rejected;
    };
    for (label, s) in [
        ("betas swapped (25, 4)", Situ::new(25.0, Some(4.0))),
        ("up branch unclipped", Situ::new(4.0, None)),
    ] {
        let y = host(s.forward_pair(t1::<B>(&gg, dev), t1::<B>(&gu, dev)));
        check(label, &y);
    }

    // (3) the `/beta` dropped inside the tanh, and (4) plain SwiGLU with no
    //     soft clip at all. Neither is expressible through `Situ`, so they are
    //     computed host-side in f32 — the control is about whether the
    //     *criterion* rejects a wrong function, not about which device ran it.
    let missing_div: Vec<f32> = gg
        .iter()
        .zip(&gu)
        .map(|(&g, &u)| (4.0 * g.tanh() * (1.0 / (1.0 + (-g).exp()))) * (25.0 * u.tanh()))
        .collect();
    let swiglu: Vec<f32> = gg
        .iter()
        .zip(&gu)
        .map(|(&g, &u)| g * (1.0 / (1.0 + (-g).exp())) * u)
        .collect();
    check("tanh(gate) without /beta", &missing_div);
    check("plain SwiGLU (no soft clip)", &swiglu);
    all_rejected
}

/// A GPU lane must not turn "no device on this box" into a silent pass, nor
/// into a crash that hides the other lanes. Backend init panics rather than
/// returning, so it is caught and reported as `unavailable`.
thread_local! {
    /// Lanes whose device would not initialise. Kept so their absence is a
    /// reported fact rather than a silent hole in the coverage.
    static DROPPED: std::cell::RefCell<Vec<String>> = const { std::cell::RefCell::new(Vec::new()) };
}

fn try_lane<B: Backend, F>(name: &str, build: F, o: &HashMap<String, Arr>) -> Option<bool>
where
    F: FnOnce() -> Device<B>,
{
    let prev = panic::take_hook();
    panic::set_hook(Box::new(|_| {}));
    let dev = panic::catch_unwind(AssertUnwindSafe(build));
    panic::set_hook(prev);
    match dev {
        Ok(dev) => Some(run_lane::<B>(name, &dev, o)),
        Err(_) => {
            // Recorded, not discarded. A dropped lane used to vanish from
            // `lanes`, and `lanes.iter().all()` is vacuously TRUE over an empty
            // set — so a box where every backend failed to initialise reported
            // GATE PASS having verified nothing.
            println!("\n=== lane: {name} === unavailable (device init panicked)");
            DROPPED.with(|d| d.borrow_mut().push(name.to_string()));
            None
        }
    }
}

fn main() {
    let dir_arg = std::env::args()
        .nth(1)
        .or_else(|| std::env::var("SITU_ORACLE_DIR").ok());
    let dir = mary::paths::model(dir_arg.as_deref(), "k3-situ/oracle_npy").unwrap_or_else(|e| {
        eprintln!("{e}");
        std::process::exit(2)
    });
    println!("kimi_situ_gate — mary::models::k3::situ vs the shipped SituAndMul");
    println!("oracle: {}", dir.display());
    println!("criterion: |got-want| <= {ATOL:e} + rtol·|want|, rtol per lane (>= {RTOL_FLOOR:e})");
    let o = load_oracle(&dir);
    let situ = Situ::k3();
    // The betas are the whole activation; assert the port's constants are the
    // config's before anything else is measured.
    assert_eq!(
        situ.beta, o["beta"].data[0],
        "beta disagrees with the oracle"
    );
    assert_eq!(
        situ.linear_beta.unwrap(),
        o["linear_beta"].data[0],
        "linear_beta disagrees with the oracle"
    );
    println!(
        "betas: beta={} linear_beta={} (match config.json + oracle)",
        situ.beta,
        situ.linear_beta.unwrap()
    );

    let mut lanes: Vec<(&str, bool)> = Vec::new();

    // CPU lane — deterministic, and the closest arithmetic to torch-CPU.
    {
        type Cpu = burn::backend::NdArray;
        let dev = Device::<Cpu>::default();
        lanes.push(("ndarray-cpu", run_lane::<Cpu>("ndarray-cpu", &dev, &o)));
    }

    // GPU lanes — the backends the port will actually run on here.
    #[cfg(feature = "kimi-k3-cuda")]
    {
        type Gpu = burn::backend::Cuda;
        if let Some(r) = try_lane::<Gpu, _>("cuda", Device::<Gpu>::default, &o) {
            lanes.push(("cuda", r));
        }
    }
    // A lane compiled OUT is a coverage hole too, and a quieter one than a
    // dropped lane: it leaves no trace at runtime at all. Say so, so that a
    // green summary cannot be read as "verified on CUDA".
    #[cfg(not(feature = "kimi-k3-cuda"))]
    println!(
        "\n=== lane: cuda === NOT COMPILED IN (build with --features kimi-k3-cuda). \
         Nothing below is evidence about the CUDA backend."
    );
    {
        type Gpu = burn::backend::Wgpu;
        if let Some(r) = try_lane::<Gpu, _>("wgpu-vulkan", Device::<Gpu>::default, &o) {
            lanes.push(("wgpu-vulkan", r));
        }
    }

    println!("\n=== summary ===");
    for (n, r) in &lanes {
        println!("  {:<14} {}", n, if *r { "PASS" } else { "FAIL" });
    }
    let dropped = DROPPED.with(|d| d.borrow().clone());
    for n in &dropped {
        println!("  {:<14} DROPPED (device init failed)", n);
    }
    // `all()` over an empty set is TRUE, so an empty lane list must be an
    // explicit failure rather than an implicit pass.
    let ran_any = !lanes.is_empty();
    if !ran_any {
        println!("\n  NO LANE RAN — nothing was verified.");
    }
    // A dropped lane is a coverage hole. It is only acceptable when the
    // operator says so, which forces the absence to be acknowledged instead of
    // inferred from a green summary.
    let allow_missing = std::env::var("SITU_ALLOW_MISSING_LANES").is_ok();
    if !dropped.is_empty() && !allow_missing {
        println!(
            "\n  {} lane(s) DROPPED and SITU_ALLOW_MISSING_LANES is not set — treating as \
             failure. A backend that did not run is not a backend that passed.",
            dropped.len()
        );
    }
    let pass = ran_any && lanes.iter().all(|(_, r)| *r) && (dropped.is_empty() || allow_missing);
    println!(
        "\nGATE: {}  ({} lane(s) ran, {} dropped)",
        if pass { "PASS" } else { "FAIL" },
        lanes.len(),
        dropped.len()
    );
    if !pass {
        std::process::exit(1);
    }
}
