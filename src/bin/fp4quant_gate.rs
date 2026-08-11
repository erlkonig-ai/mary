//! `fp4quant_gate` — is [`mary::models::inkling::fp4quant`]'s device NVFP4
//! activation quantizer **bit-identical** to an independent host reference, on
//! real Inkling data?
//!
//! ## Why bitwise and not a tolerance
//!
//! Quantization is not an approximation of a real-valued function: given the
//! input f32s, the output codes and scale bytes are *integers*, fixed by the
//! recipe. Two implementations that agree on the recipe must agree on every
//! bit. A tolerance here would only measure how close the two happen to be
//! while hiding a genuine disagreement about a boundary, which is exactly the
//! class of bug that matters — a code chosen one step too high is a 50% error
//! on that element and invisible in an aggregate norm.
//!
//! The only place the two lanes can legitimately part is a *tie*: the device
//! divides in f32, the host in f64, so a quotient that lands within ~6e-8 of
//! an E2M1 decision boundary (or a scale within ~6e-8 of an E4M3 midpoint) can
//! fall either way. This binary counts those near-boundary cases explicitly, so
//! a mismatch can be attributed rather than papered over. It does not loosen
//! the gate.
//!
//! ## The input is real, and deliberately in two shapes
//!
//! Both cases are the same real bytes: `model.llm.layers.10.mlp.experts.w13_weight`,
//! expert 0, rows 0..64, decoded to f32 by [`mary::models::inkling::nvfp4`]'s
//! audited path (real E2M1 codes, real E4M3 block scales, real per-expert F32
//! `scale2`). Their magnitudes — max |w| ~ 0.85, per-block scales 0.015..0.14 —
//! sit squarely inside E4M3's representable band and in the same range as the
//! checkpoint's own `input_amax` (1.3125) for this projection, so this is a
//! fair stand-in for the activations the quantizer will actually see.
//!
//! * **aligned** — `X[i][j] = W[i][j]`. A 16-element quantization block lands
//!   exactly on one of the checkpoint's own blocks. Re-quantizing an already
//!   NVFP4 signal at the same block boundaries is nearly idempotent in the
//!   *codes*; what it is not is free, because this quantizer has no `scale2`
//!   level and must therefore represent `blockscale * scale2` in E4M3 alone.
//!   The error this case reports is precisely that cost.
//! * **mixed** — `X[i][j] = W[(i+j) % 64][j]`. Each 16-element block now draws
//!   its 16 values from 16 *different* weight rows, hence 16 different original
//!   block scales. The block's dynamic range is genuine, codes actually move,
//!   and elements really do round to zero. This is the case whose error number
//!   is worth quoting.
//! * **layout** — `X[i][j] = E2M1[c_ij] * blockscale_ij`, i.e. the same real
//!   rows decoded *without* `scale2`. On any block whose largest code is `±6`,
//!   this input forces the quantizer's own answer to be the checkpoint's:
//!   `amax = 6*bs` is exact in f32, `amax/6` is exactly `bs`, `bs` is already an
//!   E4M3 value, and every `x/bs` is exactly an E2M1 value — no rounding
//!   anywhere. So the emitted bytes must equal the checkpoint's stored bytes.
//!   That is what actually gates the packing claim ("byte-identical on
//!   little-endian to the checkpoint's `e2m1x2` rows"): comparing this kernel's
//!   packer against this gate's unpacker would agree even if both had the
//!   nibble order backwards. Blocks containing the `0x8` (`-0.0`) code are
//!   excluded and counted — `-0.0` is not negative under `< 0.0`, so the
//!   quantizer emits `0x0` for it, which is the same number and a different
//!   byte.
//!
//! Build: `--features cuda-backend,inkling`
//! Run:   `fp4quant_gate [<checkpoint dir>]`

use std::fs::File;
use std::io::{Read, Seek, SeekFrom};
use std::path::{Path, PathBuf};

use anyhow::{bail, Context, Result};

use cubecl::cuda::CudaRuntime;
use cubecl::prelude::*;

use mary::models::inkling::fp4quant::{quantize_nvfp4, E4M3_MAX, FP4_MAX, GROUP};
use mary::models::inkling::nvfp4::{decode_row, e4m3_to_f32, FP4_E2M1};

// ---------------------------------------------------------------------------
// safetensors: header parse + positioned reads (same shape as nvfp4_mma_probe;
// the shards are gigabytes and only the rows this gate touches are read).
// ---------------------------------------------------------------------------

struct Shard {
    file: File,
    data_start: u64,
    header: serde_json::Value,
}

impl Shard {
    fn open(path: &Path) -> Result<Self> {
        let mut file = File::open(path).with_context(|| format!("opening {}", path.display()))?;
        let mut len = [0u8; 8];
        file.read_exact(&mut len).context("reading header length")?;
        let n = u64::from_le_bytes(len);
        let mut buf = vec![0u8; n as usize];
        file.read_exact(&mut buf).context("reading header")?;
        let header: serde_json::Value =
            serde_json::from_slice(&buf).context("parsing safetensors header")?;
        Ok(Shard { file, data_start: 8 + n, header })
    }

    fn info(&self, name: &str) -> Result<(String, Vec<usize>, u64, u64)> {
        let e = self
            .header
            .get(name)
            .with_context(|| format!("shard has no tensor {name}"))?;
        let dtype = e["dtype"].as_str().context("dtype")?.to_string();
        let shape: Vec<usize> = e["shape"]
            .as_array()
            .context("shape")?
            .iter()
            .map(|v| v.as_u64().unwrap_or(0) as usize)
            .collect();
        let off = e["data_offsets"].as_array().context("data_offsets")?;
        Ok((dtype, shape, off[0].as_u64().unwrap(), off[1].as_u64().unwrap()))
    }

    fn read_at(&mut self, name: &str, offset: u64, len: usize) -> Result<Vec<u8>> {
        let (_, _, start, end) = self.info(name)?;
        if offset + len as u64 > end - start {
            bail!("read of {len} at {offset} runs past tensor {name}");
        }
        self.file.seek(SeekFrom::Start(self.data_start + start + offset))?;
        let mut buf = vec![0u8; len];
        self.file.read_exact(&mut buf)?;
        Ok(buf)
    }
}

fn shard_of(dir: &Path, name: &str) -> Result<PathBuf> {
    let idx: serde_json::Value = serde_json::from_slice(
        &std::fs::read(dir.join("model.safetensors.index.json")).context("reading index")?,
    )?;
    let f = idx["weight_map"][name]
        .as_str()
        .with_context(|| format!("index has no {name}"))?;
    Ok(dir.join(f))
}

const WEIGHT: &str = "model.llm.layers.10.mlp.experts.w13_weight";

/// One checkpoint row: the bytes as stored, plus the audited f32 decode.
struct Row {
    /// `k/2` packed E2M1 bytes, low nibble first.
    packed: Vec<u8>,
    /// `k/16` E4M3 block-scale bytes.
    scale_bytes: Vec<u8>,
    /// `k` f32 values, `((E2M1[c] * blockscale) * scale2)`.
    decoded: Vec<f32>,
}

/// Decode `n` consecutive expert rows to f32 through the audited NVFP4 path,
/// keeping the raw bytes alongside.
fn load_decoded_rows(dir: &Path, expert: usize, n: usize, k: usize) -> Result<(Vec<Row>, f32)> {
    let wp = shard_of(dir, WEIGHT)?;
    let sp = shard_of(dir, &format!("{WEIGHT}.scale"))?;
    let s2p = shard_of(dir, &format!("{WEIGHT}.scale2"))?;

    let mut ws = Shard::open(&wp)?;
    let (wd, wshape, _, _) = ws.info(WEIGHT)?;
    if wd != "U8" || wshape.len() != 3 {
        bail!("unexpected weight dtype/shape: {wd} {wshape:?}");
    }
    let rows_per_expert = wshape[1];
    let bytes_per_row = wshape[2];
    if bytes_per_row * 2 < k {
        bail!("requested k={k} exceeds stored row width {}", bytes_per_row * 2);
    }

    let mut ss = Shard::open(&sp)?;
    let (sd, sshape, _, _) = ss.info(&format!("{WEIGHT}.scale"))?;
    if sd != "F8_E4M3" {
        bail!("unexpected scale dtype: {sd}");
    }
    let scales_per_row = sshape[2];

    let mut s2s = Shard::open(&s2p)?;
    let s2b = s2s.read_at(&format!("{WEIGHT}.scale2"), (expert * 4) as u64, 4)?;
    let scale2 = f32::from_le_bytes([s2b[0], s2b[1], s2b[2], s2b[3]]);

    let mut out = Vec::with_capacity(n);
    for r in 0..n {
        let gr = expert * rows_per_expert + r;
        let packed = ws.read_at(WEIGHT, (gr * bytes_per_row) as u64, k / 2)?;
        let scales = ss.read_at(
            &format!("{WEIGHT}.scale"),
            (gr * scales_per_row) as u64,
            k / GROUP,
        )?;
        let mut decoded = vec![0.0f32; k];
        let written = decode_row(&packed, &scales, scale2, &mut decoded);
        if written != k {
            bail!("decode_row wrote {written} of {k}");
        }
        out.push(Row { packed, scale_bytes: scales, decoded });
    }
    Ok((out, scale2))
}

// ---------------------------------------------------------------------------
// host reference — independent of the kernel, f64 throughout
// ---------------------------------------------------------------------------

/// The 127 non-negative finite E4M3 values, ascending, indexed by their byte.
///
/// `0x7F` is the format's only positive NaN and is excluded. The byte order of
/// an IEEE-like format is its value order, so this is monotone by construction
/// — asserted rather than assumed, since the bracketing search below depends
/// on it. The values come from [`e4m3_to_f32`], which is itself gated
/// bit-exactly against torch over all 256 patterns.
fn e4m3_ladder() -> Vec<f64> {
    let v: Vec<f64> = (0u16..=0x7E).map(|b| e4m3_to_f32(b as u8) as f64).collect();
    for w in v.windows(2) {
        assert!(w[1] > w[0], "E4M3 byte order is not value order: {w:?}");
    }
    v
}

/// Round-to-nearest-even encode of a non-negative f64 into an E4M3 byte.
///
/// Bracketing the exhaustive ladder rather than manipulating exponent bits: it
/// cannot get the subnormal boundary wrong (the one place a hand-rolled encoder
/// does), and the tie rule falls out for free — the byte's low bit *is* the
/// significand's low bit, so "ties to even significand" is "ties to even byte".
fn e4m3_encode(v: f64, ladder: &[f64]) -> u8 {
    assert!(v.is_finite() && v >= 0.0, "scale must be finite and non-negative, got {v}");
    let v = v.min(E4M3_MAX as f64);
    let hi = ladder.partition_point(|&t| t < v);
    if hi == 0 {
        return 0; // v == 0
    }
    if hi >= ladder.len() {
        return (ladder.len() - 1) as u8;
    }
    let lo = hi - 1;
    let dl = v - ladder[lo];
    let dh = ladder[hi] - v;
    if dl < dh {
        lo as u8
    } else if dh < dl {
        hi as u8
    } else if lo % 2 == 0 {
        lo as u8
    } else {
        hi as u8
    }
}

/// How close `v` sits to the midpoint of its bracketing E4M3 pair, relatively.
/// `f64::INFINITY` when there is no bracketing pair (v == 0 or v >= max).
fn e4m3_midpoint_proximity(v: f64, ladder: &[f64]) -> f64 {
    let hi = ladder.partition_point(|&t| t < v);
    if hi == 0 || hi >= ladder.len() {
        return f64::INFINITY;
    }
    let mid = 0.5 * (ladder[hi - 1] + ladder[hi]);
    ((v - mid) / mid).abs()
}

/// The seven E2M1 magnitude decision boundaries, ascending. Each is exactly
/// representable in both f32 and f64, so `>=` behaves identically in either.
const E2M1_MIDPOINTS: [f64; 7] = [0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0];

struct HostOut {
    codes: Vec<u32>,
    scales: Vec<u8>,
    /// Blocks whose `amax/6` is within 1e-6 (relative) of an E4M3 midpoint.
    scale_near_ties: usize,
    /// Elements whose `|x/s|` is within 1e-6 (relative) of an E2M1 boundary.
    code_near_ties: usize,
    /// Blocks that quantized to a zero scale byte.
    zero_scale_blocks: usize,
}

/// The same recipe as the kernel, computed independently in f64.
fn host_quantize(x: &[f32], ladder: &[f64]) -> HostOut {
    let n = x.len();
    assert!(n % GROUP == 0);
    let blocks = n / GROUP;
    let mut codes = vec![0u32; n / 8];
    let mut scales = vec![0u8; blocks];
    let mut scale_near_ties = 0usize;
    let mut code_near_ties = 0usize;
    let mut zero_scale_blocks = 0usize;

    for b in 0..blocks {
        let base = b * GROUP;

        let mut amax = 0.0f64;
        for i in 0..GROUP {
            let a = (x[base + i] as f64).abs();
            if a > amax {
                amax = a;
            }
        }

        let mut sf = amax / FP4_MAX as f64;
        if sf > E4M3_MAX as f64 {
            sf = E4M3_MAX as f64;
        }
        if e4m3_midpoint_proximity(sf, ladder) < 1e-6 {
            scale_near_ties += 1;
        }
        let byte = e4m3_encode(sf, ladder);
        scales[b] = byte;
        let s = e4m3_to_f32(byte) as f64;
        if byte == 0 {
            zero_scale_blocks += 1;
        }

        for i in 0..GROUP {
            let v = x[base + i] as f64;
            let mut m = 0u32;
            if s > 0.0 {
                let t = v / s;
                let a = t.abs();
                for (j, &thr) in E2M1_MIDPOINTS.iter().enumerate() {
                    if a >= thr {
                        m = j as u32 + 1;
                    }
                    if ((a - thr) / thr).abs() < 1e-6 {
                        code_near_ties += 1;
                    }
                }
                if t < 0.0 {
                    m += 8;
                }
            }
            let g = base + i;
            codes[g / 8] |= m << (4 * (g % 8));
        }
    }

    HostOut { codes, scales, scale_near_ties, code_near_ties, zero_scale_blocks }
}

#[inline]
fn nibble(codes: &[u32], i: usize) -> u32 {
    (codes[i / 8] >> (4 * (i % 8))) & 0xF
}

// ---------------------------------------------------------------------------

struct Verdict {
    ok: bool,
    /// The device's packed code buffer, as raw little-endian bytes.
    dev_code_bytes: Vec<u8>,
    /// The device's E4M3 scale bytes.
    dev_scales: Vec<u8>,
}

fn run_case(
    name: &str,
    what: &str,
    x: &[f32],
    rows: usize,
    k: usize,
    ladder: &[f64],
    client: &cubecl::client::ComputeClient<CudaRuntime>,
) -> Verdict {
    let n = rows * k;
    assert_eq!(x.len(), n);
    let blocks = n / GROUP;
    let words = n / 8;

    println!("\n=== case `{name}` — {what} ===");
    println!("  shape [{rows}, {k}] = {n} elements, {blocks} blocks, {words} packed words");

    // --- device ---------------------------------------------------------
    let xh = client.create_from_slice(f32::as_bytes(x));
    let (codes_h, scales_h) = quantize_nvfp4(client, &xh, rows, k);
    let codes_bytes = client.read_one(codes_h).expect("read codes");
    let scales_bytes = client.read_one(scales_h).expect("read scales");
    let dev_code_bytes = codes_bytes[..words * 4].to_vec();
    let dev_codes = u32::from_bytes(&codes_bytes)[..words].to_vec();
    let dev_scales = scales_bytes[..blocks].to_vec();

    // --- host -----------------------------------------------------------
    let host = host_quantize(x, ladder);

    // --- bitwise --------------------------------------------------------
    let scale_diff: Vec<usize> = (0..blocks).filter(|&b| dev_scales[b] != host.scales[b]).collect();
    let code_diff: Vec<usize> = (0..n)
        .filter(|&i| nibble(&dev_codes, i) != nibble(&host.codes, i))
        .collect();
    let word_diff = (0..words).filter(|&w| dev_codes[w] != host.codes[w]).count();

    println!(
        "  bitwise: scale bytes differing {} / {blocks}   code nibbles differing {} / {n}   \
         (packed words differing {word_diff} / {words})",
        scale_diff.len(),
        code_diff.len()
    );
    println!(
        "  near-boundary cases (where f32-vs-f64 could legitimately disagree): \
         {} blocks within 1e-6 of an E4M3 midpoint, {} elements within 1e-6 of an E2M1 boundary",
        host.scale_near_ties, host.code_near_ties
    );
    for &b in scale_diff.iter().take(4) {
        println!(
            "    scale block {b}: device 0x{:02X} ({:e})  host 0x{:02X} ({:e})",
            dev_scales[b],
            e4m3_to_f32(dev_scales[b]),
            host.scales[b],
            e4m3_to_f32(host.scales[b])
        );
    }
    for &i in code_diff.iter().take(8) {
        let b = i / GROUP;
        println!(
            "    code {i} (block {b}): x = {:e}  s = {:e}  device {}  host {}",
            x[i],
            e4m3_to_f32(host.scales[b]),
            nibble(&dev_codes, i),
            nibble(&host.codes, i)
        );
    }

    // --- dequantization cost (informational) -----------------------------
    // Judged against the DEVICE output, since that is what a kernel would
    // consume. mean/max relative error skip x == 0 (relative error is not
    // defined there); the RMS ratio is over everything and is the number to
    // quote for "what does 4-bit activation quantization cost".
    let mut sum_rel = 0.0f64;
    let mut cnt_rel = 0usize;
    let mut max_rel = 0.0f64;
    let mut max_at = 0usize;
    let mut num = 0.0f64;
    let mut den = 0.0f64;
    let mut zeroed = 0usize;
    let mut x_zero = 0usize;
    for i in 0..n {
        let b = i / GROUP;
        let deq = FP4_E2M1[nibble(&dev_codes, i) as usize] as f64 * e4m3_to_f32(dev_scales[b]) as f64;
        let xv = x[i] as f64;
        let d = deq - xv;
        num += d * d;
        den += xv * xv;
        if deq == 0.0 {
            zeroed += 1;
        }
        if xv == 0.0 {
            x_zero += 1;
        } else {
            let r = (d / xv).abs();
            sum_rel += r;
            cnt_rel += 1;
            if r > max_rel {
                max_rel = r;
                max_at = i;
            }
        }
    }
    println!(
        "  dequant vs original f32 (informational, not a gate):\n\
         \x20   mean |rel err| = {:.4e}  over {cnt_rel} non-zero elements\n\
         \x20   max  |rel err| = {:.4e}  at element {max_at} (x = {:e})\n\
         \x20   RMS ratio ||deq - x|| / ||x|| = {:.4e}\n\
         \x20   rounds to exactly zero: {} / {n} = {:.3}%   (of which {} were already zero in x)\n\
         \x20   blocks with a zero scale byte: {} / {blocks}",
        sum_rel / cnt_rel.max(1) as f64,
        max_rel,
        x[max_at],
        (num / den.max(f64::MIN_POSITIVE)).sqrt(),
        zeroed,
        100.0 * zeroed as f64 / n as f64,
        x_zero,
        host.zero_scale_blocks
    );

    let ok = scale_diff.is_empty() && code_diff.is_empty();
    println!("  -> device vs host reference: {}", if ok { "bit-identical" } else { "MISMATCH" });
    Verdict { ok, dev_code_bytes, dev_scales }
}

/// Bitwise host-vs-device over a range of shapes, on slices of the same real
/// data.
///
/// `rows` and `k` enter the kernel only through the total block count, so what
/// this varies is the launch geometry — and specifically the partial trailing
/// cube. The main cases are all exact multiples of the 256-thread cube, so
/// without this the `blk < blocks` guard is never the thing that stops a
/// thread, and an out-of-range write would go unnoticed.
fn sweep(
    src: &[f32],
    ladder: &[f64],
    client: &cubecl::client::ComputeClient<CudaRuntime>,
) -> bool {
    println!("\n=== shape sweep — bitwise, exercising the partial-cube tail ===");
    let shapes: [(usize, usize); 8] = [
        (1, 64),
        (1, 192),
        (1, 4096),
        (3, 64),
        (5, 320),
        (7, 128),
        (13, 1024),
        (16, 64),
    ];
    let mut ok = true;
    for (rows, k) in shapes {
        let n = rows * k;
        let x = &src[..n];
        let blocks = n / GROUP;
        let words = n / 8;

        let xh = client.create_from_slice(f32::as_bytes(x));
        let (ch, sh) = quantize_nvfp4(client, &xh, rows, k);
        let cb = client.read_one(ch).expect("read codes");
        let sb = client.read_one(sh).expect("read scales");
        let dev_codes = &u32::from_bytes(&cb)[..words];
        let dev_scales = &sb[..blocks];

        let host = host_quantize(x, ladder);
        let same = dev_codes == &host.codes[..] && dev_scales == &host.scales[..];
        ok &= same;
        println!(
            "  [{rows:>2}, {k:>4}] = {blocks:>4} blocks ({:>2} cube(s), tail {:>3}): {}",
            blocks.div_ceil(256),
            blocks % 256,
            if same { "bit-identical" } else { "MISMATCH" }
        );
    }
    ok
}

/// The packing claim, gated against the checkpoint rather than against this
/// file's own unpacker.
///
/// On the `layout` input every step of the recipe is exact for a block whose
/// largest code is `±6` (`6*bs` fits in f32, `(6*bs)/6` is exactly `bs`, `bs`
/// is already an E4M3 value, `x/bs` is exactly an E2M1 value), so the eight
/// packed bytes the kernel writes must be the eight bytes the checkpoint
/// stores, and the scale byte must be the checkpoint's scale byte. Blocks whose
/// largest code is smaller (the quantizer legitimately rescales those) and
/// blocks containing the `0x8` negative-zero code are skipped, and counted.
fn check_layout_against_checkpoint(w: &[Row], rows: usize, k: usize, v: &Verdict) -> bool {
    let per_row = k / GROUP;
    let mut checked = 0usize;
    let mut skipped_range = 0usize;
    let mut skipped_negzero = 0usize;
    let mut byte_mismatch = 0usize;
    let mut scale_mismatch = 0usize;
    let mut first: Option<String> = None;

    for i in 0..rows {
        for b in 0..per_row {
            let src = &w[i].packed[b * 8..b * 8 + 8];
            let mut maxmag = 0u8;
            let mut has_negzero = false;
            for &byte in src {
                for c in [byte & 0x0F, byte >> 4] {
                    if c == 0x8 {
                        has_negzero = true;
                    }
                    let mag = c & 0x7;
                    if mag > maxmag {
                        maxmag = mag;
                    }
                }
            }
            if maxmag != 7 {
                // 7 is the code for magnitude 6.0 — the top of the E2M1 range.
                skipped_range += 1;
                continue;
            }
            if has_negzero {
                skipped_negzero += 1;
                continue;
            }
            let blk = i * per_row + b;
            let got = &v.dev_code_bytes[blk * 8..blk * 8 + 8];
            if got != src {
                byte_mismatch += 1;
                if first.is_none() {
                    first = Some(format!(
                        "block {blk} (row {i}, block {b}): checkpoint {src:02X?} device {got:02X?}"
                    ));
                }
            }
            if v.dev_scales[blk] != w[i].scale_bytes[b] {
                scale_mismatch += 1;
            }
            checked += 1;
        }
    }

    println!(
        "\n=== packed-layout identity vs the checkpoint's own bytes ===\n  \
         {checked} full-range blocks compared ({skipped_range} skipped: largest code < 6, the \
         quantizer rescales those; {skipped_negzero} skipped: contain the 0x8 negative-zero code)\n  \
         packed 8-byte groups differing: {byte_mismatch}   scale bytes differing: {scale_mismatch}"
    );
    if let Some(f) = first {
        println!("    first: {f}");
    }
    if checked == 0 {
        println!("  -> INCONCLUSIVE: no block qualified");
        return false;
    }
    let ok = byte_mismatch == 0 && scale_mismatch == 0;
    println!(
        "  -> {}",
        if ok {
            "byte-identical to the checkpoint's packed e2m1x2 + E4M3 scale bytes"
        } else {
            "MISMATCH"
        }
    );
    ok
}

fn main() -> Result<()> {
    let dir = std::env::args()
        .nth(1)
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("models/thinkingmachines-inkling-small-nvfp4"));

    const ROWS: usize = 16;
    const K: usize = 4096;
    const SRC: usize = 64;

    let ladder = e4m3_ladder();
    // Self-check the host encoder against the audited decode: every E4M3 value
    // must encode back to its own byte. This is what makes the reference
    // trustworthy enough to be the arbiter of a bitwise gate.
    for b in 0u16..=0x7E {
        let v = e4m3_to_f32(b as u8) as f64;
        let back = e4m3_encode(v, &ladder);
        if back != b as u8 {
            bail!("host E4M3 encoder is not a left inverse of the decode: 0x{b:02X} -> 0x{back:02X}");
        }
    }
    println!("host E4M3 round-trip: all 127 non-negative finite patterns encode back to themselves");

    let (w, scale2) = load_decoded_rows(&dir, 0, SRC, K)?;
    let amax = w
        .iter()
        .flat_map(|r| r.decoded.iter())
        .fold(0.0f32, |m, &v| if v.abs() > m { v.abs() } else { m });
    println!(
        "loaded {SRC}x{K} decoded rows of {WEIGHT} (expert 0), scale2 = {scale2:e}, \
         max |w| = {amax:e}"
    );

    let client = CudaRuntime::client(&Default::default());

    let aligned: Vec<f32> = (0..ROWS).flat_map(|i| w[i].decoded.iter().copied()).collect();
    let mut mixed: Vec<f32> = Vec::with_capacity(ROWS * K);
    for i in 0..ROWS {
        for j in 0..K {
            mixed.push(w[(i + j) % SRC].decoded[j]);
        }
    }
    // Same real rows, decoded WITHOUT scale2: E2M1 code times its own E4M3
    // block scale, which is the input that forces the checkpoint's own bytes
    // back out of the quantizer.
    let mut unscaled: Vec<f32> = Vec::with_capacity(ROWS * K);
    for row in w.iter().take(ROWS) {
        for j in 0..K {
            let byte = row.packed[j / 2];
            let c = if j % 2 == 0 { byte & 0x0F } else { byte >> 4 };
            unscaled.push(FP4_E2M1[c as usize] * e4m3_to_f32(row.scale_bytes[j / GROUP]));
        }
    }

    let mut ok = true;
    ok &= run_case(
        "aligned",
        "X[i][j] = W[i][j] — blocks land on the checkpoint's own 16-element blocks",
        &aligned,
        ROWS,
        K,
        &ladder,
        &client,
    )
    .ok;
    ok &= run_case(
        "mixed",
        "X[i][j] = W[(i+j) % 64][j] — each block draws 16 values from 16 different weight rows",
        &mixed,
        ROWS,
        K,
        &ladder,
        &client,
    )
    .ok;
    let layout = run_case(
        "layout",
        "X[i][j] = E2M1[c] * blockscale — the checkpoint's own rows without scale2",
        &unscaled,
        ROWS,
        K,
        &ladder,
        &client,
    );
    ok &= layout.ok;
    ok &= check_layout_against_checkpoint(&w, ROWS, K, &layout);
    ok &= sweep(&mixed, &ladder, &client);

    if !ok {
        println!(
            "\nFAIL — the device NVFP4 activation quantizer is not bit-identical to the host reference"
        );
        std::process::exit(1);
    }
    println!(
        "\nPASS — quantize_nvfp4 reproduces the host f64 reference bit-for-bit (codes and E4M3 \
         scale bytes) on real Inkling data, in both block alignments"
    );
    Ok(())
}
