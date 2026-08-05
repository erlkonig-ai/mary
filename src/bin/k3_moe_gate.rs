//! `k3_moe_gate` — the correctness gate for [`mary::models::k3::moe`].
//!
//! The oracle is the **whole-layer** capture: a real 13-layer prefix of Kimi K3
//! driven on real token ids through the shipped `KimiLinearModel.forward`, with
//! forward hooks on every sub-module. Nothing in it is a transcription of the
//! Python; every `_in`/`_out` array is the shipped module's own tensor.
//!
//! Two files, both identified by sha256 before a single number is read:
//!
//! * `layer_oracle_prefix13_bf16.npz` — the 32-token bf16 run, layers 0..12.
//! * `layer_oracle_ladder.npz` — the same layer inputs re-run at f32 and f64,
//!   4 tokens, plus the MXFP4 decode cross-check.
//!
//! The weights come from the **checkpoint**, not from the oracle: the gate
//! reads `model.safetensors.index.json`, seeks into the shards, and decodes the
//! routed experts from their packed MXFP4 nibbles with `mary::nn::mxfp4`. So a
//! wrong weight-name → module-slot mapping cannot pass, and the decode is
//! exercised on the same bytes the model runs on.
//!
//! # How this gate is built so it can fail
//!
//! * **Parts, not only the pair.** The single-expert lane checks `w1`, `w3`,
//!   the concatenation, `situ` and `w2` each against their own captured tensor,
//!   each driven from the *captured input* of that sub-module — so a
//!   compensating pair of errors has nowhere to hide. The composition is
//!   checked too, and separately.
//! * **The MXFP4 decode is asserted directly**, against the oracle's float64
//!   decode of the same bytes, bit-for-bit — never through a pack/unpack round
//!   trip, which would be a property of the pair and nothing about either half.
//! * **Every negative control is required to fail**, and where the oracle
//!   stores the wrong answer (`ALT_*`), the deliberately-wrong computation is
//!   additionally required to *reproduce* it. A control that cannot
//!   discriminate is itself a gate failure.
//! * **Comparators are `!(d <= budget)`**, so a NaN scores as a failure rather
//!   than as zero error; every array is asserted non-empty and equal-length
//!   before it is compared; and a lane that is skipped says so in the verdict.
//! * **Budgets are derived before the numbers are looked at** (see
//!   [`bf16_budget`] / [`f32_budget`]), and the bf16 single-step lanes
//!   additionally require a minimum fraction of **bit-identical** elements —
//!   which is the check that actually bites, since a correct port differs from
//!   the shipped bf16 run only where fp32 accumulation order flips one rounding
//!   decision.
//!
//! # Running
//!
//! ```text
//! cargo run --release --features kimi-k3,k3,mxfp4 --bin k3_moe_gate
//! ```
//! Oracle dir: argv[1] or `K3_ORACLE_DIR` (default `./k3-oracle`).
//! Model dir:  argv[2] or `K3_MODEL_DIR`  (default `./kimi-k3`).
//! `K3MOE_FAST=1` runs a subset — it prints a PARTIAL banner, names every
//! skipped lane, and says PARTIAL in the verdict. The authoritative result is a
//! run without it.

use burn::backend::NdArray;
use burn::prelude::*;
use mary::models::k3::{K3Config, K3TextConfig};
use mary::models::k3::moe::{
    ActRound, ExpertWeights, LatentMoe, LatentMoeWeights, MoeDims, RouterWeights, Routing,
    SharedExpertWeights,
};
use mary::nn::mxfp4::decode_mxfp4;
use sha2::{Digest, Sha256};
use std::cell::RefCell;
use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::fs::File;
use std::io::{Read, Seek, SeekFrom};
use std::path::{Path, PathBuf};

type B = NdArray<f32>;
type Dev = Device<B>;

// ===========================================================================
// Budgets — derived, and written down before any number was read
// ===========================================================================

/// Spacing of bfloat16 relative to a value's magnitude.
///
/// bf16 has **8 bits of significand** — 1 implicit plus 7 stored — so
/// consecutive bf16 values differ by `2^-7` relative, not `2^-8`. (Writing
/// `2^-8` here was this gate's first bug: it made every bf16 budget half what
/// the derivation says, and turned four correct results into failures.)
const BF16_ULP_REL: f64 = 1.0 / 128.0;

/// Relative error of one f32 matmul over a few thousand terms, generously:
/// `sqrt(K)·2^-24` is 5e-6 at K = 7168; this allows ~4x for cancellation.
const F32_STEP_REL: f64 = 2e-5;

/// Budget for a bf16-lane comparison whose reference passed through `n` bf16
/// roundings that the port must reproduce.
///
/// Two things fit in `n`. (1) The port computes in f32 and rounds where the
/// reference rounds, so it can differ by one ulp wherever f32 accumulation
/// noise pushes a value across a rounding boundary. (2) The shipped bf16 GEMM
/// is itself not fp32-exact: measured against a float64 recomputation of the
/// same product it lands up to ~0.9 ulp of the tensor maximum away from the
/// true answer (check `X1`). So one wide projection is worth ~1.5 ulp and `n`
/// counts roundings generously.
fn bf16_budget(n_roundings: usize, ref_absmax: f64) -> f64 {
    (n_roundings as f64) * BF16_ULP_REL * ref_absmax
}

/// Budget for an f32-lane comparison over `n` chained matmuls.
fn f32_budget(n_matmuls: usize, ref_absmax: f64) -> f64 {
    (n_matmuls as f64) * F32_STEP_REL * ref_absmax
}

/// Minimum fraction of elements that must be **bit-identical** to the shipped
/// bf16 tensor in a single-step bf16 comparison.
///
/// Derived: the chance that f32 accumulation noise (~`sqrt(K)·2^-24` relative,
/// 5e-6 at K = 7168) straddles a bf16 rounding boundary (spacing `2^-7`
/// relative) is ~6.5e-4, so 99.5% leaves a factor of ~8.
///
/// **This is required only where it is achievable**, i.e. where the shipped
/// tensor is itself within a fraction of an ulp of the exact answer:
/// elementwise ops driven from a captured input, and the narrow single-row
/// expert projections. It is NOT required of the wide 32-row projections,
/// because the shipped bf16 GEMM is measurably not fp32-exact there — see
/// check `X1`, which quantifies that rather than assuming it either way.
const BF16_EXACT_FRAC_MIN: f64 = 0.995;

// ===========================================================================
// .npz (ZIP_STORED) + .npy reading
// ===========================================================================

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Dt {
    F64,
    F32,
    U16,
    U8,
    I64,
}

impl Dt {
    fn parse(descr: &str) -> Dt {
        match descr {
            "<f8" => Dt::F64,
            "<f4" => Dt::F32,
            "<u2" | "|u2" => Dt::U16,
            "|u1" | "<u1" => Dt::U8,
            "<i8" => Dt::I64,
            other => panic!("unhandled npy dtype {other}"),
        }
    }
    fn numpy_name(self) -> &'static str {
        match self {
            Dt::F64 => "float64",
            Dt::F32 => "float32",
            Dt::U16 => "uint16",
            Dt::U8 => "uint8",
            Dt::I64 => "int64",
        }
    }
}

/// One loaded array. Raw bytes are kept so a byte-level comparison against the
/// checkpoint is possible without a lossy detour through floats.
struct Arr {
    name: String,
    shape: Vec<usize>,
    dt: Dt,
    bytes: Vec<u8>,
}

impl Arr {
    /// Element count. An empty shape is a 0-d scalar, i.e. one element — the
    /// empty product, which is what `product()` returns.
    fn len(&self) -> usize {
        self.shape.iter().product()
    }
    fn u8s(&self) -> &[u8] {
        assert_eq!(self.dt, Dt::U8, "{} is not uint8", self.name);
        &self.bytes
    }
    fn u16s(&self) -> Vec<u16> {
        assert_eq!(self.dt, Dt::U16, "{} is not uint16", self.name);
        self.bytes
            .chunks_exact(2)
            .map(|c| u16::from_le_bytes([c[0], c[1]]))
            .collect()
    }
    fn i64s(&self) -> Vec<i64> {
        assert_eq!(self.dt, Dt::I64, "{} is not int64", self.name);
        self.bytes
            .chunks_exact(8)
            .map(|c| i64::from_le_bytes(c.try_into().unwrap()))
            .collect()
    }
    /// bfloat16 bit patterns widened to f32 — exact, no rounding.
    fn bf16_as_f32(&self) -> Vec<f32> {
        self.u16s()
            .into_iter()
            .map(|b| f32::from_bits((b as u32) << 16))
            .collect()
    }
    fn f32s(&self) -> Vec<f32> {
        assert_eq!(self.dt, Dt::F32, "{} is not float32", self.name);
        self.bytes
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
            .collect()
    }
    fn f64s(&self) -> Vec<f64> {
        assert_eq!(self.dt, Dt::F64, "{} is not float64", self.name);
        self.bytes
            .chunks_exact(8)
            .map(|c| f64::from_le_bytes(c.try_into().unwrap()))
            .collect()
    }
    /// Whatever float dtype this column is, widened/narrowed to f32.
    fn as_f32(&self) -> Vec<f32> {
        match self.dt {
            Dt::U16 => self.bf16_as_f32(),
            Dt::F32 => self.f32s(),
            Dt::F64 => self.f64s().into_iter().map(|v| v as f32).collect(),
            other => panic!("{} is {other:?}, not a float column", self.name),
        }
    }
}

fn read_at(path: &Path, off: u64, len: usize) -> Vec<u8> {
    let mut f = File::open(path).unwrap_or_else(|e| panic!("open {}: {e}", path.display()));
    f.seek(SeekFrom::Start(off)).expect("seek");
    let mut buf = vec![0u8; len];
    f.read_exact(&mut buf).expect("read");
    buf
}

/// A `numpy.savez` archive, read in place. `np.savez` writes ZIP_STORED, which
/// is asserted rather than assumed — a deflated member would otherwise be
/// silently misread as raw bytes.
struct Npz {
    path: PathBuf,
    entries: BTreeMap<String, (u64, usize)>,
    sha256: String,
}

impl Npz {
    fn open(path: &Path) -> Npz {
        let mut f = File::open(path).unwrap_or_else(|e| panic!("open {}: {e}", path.display()));
        let size = f.metadata().expect("stat").len();

        let mut hasher = Sha256::new();
        let mut buf = vec![0u8; 8 << 20];
        loop {
            let n = f.read(&mut buf).expect("hash read");
            if n == 0 {
                break;
            }
            hasher.update(&buf[..n]);
        }
        let sha256 = format!("{:x}", hasher.finalize());

        let tail_len = 66_000u64.min(size) as usize;
        let tail = read_at(path, size - tail_len as u64, tail_len);
        let mut eocd = None;
        for i in (0..tail.len().saturating_sub(21)).rev() {
            if tail[i..i + 4] == [0x50, 0x4b, 0x05, 0x06] {
                eocd = Some(i);
                break;
            }
        }
        let e = eocd.unwrap_or_else(|| panic!("{}: no ZIP EOCD", path.display()));
        let n_entries = u16::from_le_bytes([tail[e + 10], tail[e + 11]]) as usize;
        let cd_size = u32::from_le_bytes(tail[e + 12..e + 16].try_into().unwrap()) as usize;
        let cd_off = u32::from_le_bytes(tail[e + 16..e + 20].try_into().unwrap()) as u64;
        assert!(
            n_entries > 0 && cd_size > 0,
            "{}: empty ZIP central directory",
            path.display()
        );

        let cd = read_at(path, cd_off, cd_size);
        let mut entries = BTreeMap::new();
        let mut p = 0usize;
        for _ in 0..n_entries {
            assert_eq!(&cd[p..p + 4], &[0x50, 0x4b, 0x01, 0x02], "central header sig");
            let method = u16::from_le_bytes([cd[p + 10], cd[p + 11]]);
            assert_eq!(
                method, 0,
                "{}: member is compressed (method {method}); np.savez writes ZIP_STORED",
                path.display()
            );
            let csize = u32::from_le_bytes(cd[p + 20..p + 24].try_into().unwrap()) as usize;
            let ucsize = u32::from_le_bytes(cd[p + 24..p + 28].try_into().unwrap()) as usize;
            assert_eq!(csize, ucsize, "stored member sizes must agree");
            let nlen = u16::from_le_bytes([cd[p + 28], cd[p + 29]]) as usize;
            let elen = u16::from_le_bytes([cd[p + 30], cd[p + 31]]) as usize;
            let clen = u16::from_le_bytes([cd[p + 32], cd[p + 33]]) as usize;
            let lho = u32::from_le_bytes(cd[p + 42..p + 46].try_into().unwrap()) as u64;
            let name = String::from_utf8(cd[p + 46..p + 46 + nlen].to_vec()).expect("member name");
            let lh = read_at(path, lho, 30);
            assert_eq!(&lh[0..4], &[0x50, 0x4b, 0x03, 0x04], "local header sig");
            let lnlen = u16::from_le_bytes([lh[26], lh[27]]) as usize;
            let lelen = u16::from_le_bytes([lh[28], lh[29]]) as usize;
            let data_off = lho + 30 + lnlen as u64 + lelen as u64;
            let key = name.strip_suffix(".npy").unwrap_or(&name).to_string();
            entries.insert(key, (data_off, csize));
            p += 46 + nlen + elen + clen;
        }
        assert_eq!(entries.len(), n_entries, "duplicate member names");
        Npz {
            path: path.to_path_buf(),
            entries,
            sha256,
        }
    }

    fn names(&self) -> BTreeSet<String> {
        self.entries.keys().cloned().collect()
    }

    fn has(&self, name: &str) -> bool {
        self.entries.contains_key(name)
    }

    fn get(&self, name: &str) -> Arr {
        let &(off, len) = self
            .entries
            .get(name)
            .unwrap_or_else(|| panic!("{}: no array {name}", self.path.display()));
        let raw = read_at(&self.path, off, len);
        assert!(
            raw.len() > 10 && raw[0] == 0x93 && &raw[1..6] == b"NUMPY",
            "{name} is not a .npy member"
        );
        let (hlen, hstart) = match raw[6] {
            1 => (u16::from_le_bytes([raw[8], raw[9]]) as usize, 10),
            _ => (
                u32::from_le_bytes([raw[8], raw[9], raw[10], raw[11]]) as usize,
                12,
            ),
        };
        let header = std::str::from_utf8(&raw[hstart..hstart + hlen]).expect("npy header");
        let field = |k: &str| -> String {
            let i = header.find(k).unwrap_or_else(|| panic!("no {k} in {header}")) + k.len();
            let rest = &header[i..];
            let s = rest.find(|c: char| c != ':' && c != ' ').unwrap();
            let rest = &rest[s..];
            let e = rest.find(',').unwrap_or(rest.len());
            rest[..e].trim().trim_matches('\'').to_string()
        };
        assert_eq!(field("'fortran_order'"), "False", "{name}: C-order only");
        let dt = Dt::parse(&field("'descr'"));
        let s0 = header.find("'shape'").expect("shape") + "'shape'".len();
        let s1 = header[s0..].find('(').expect("shape tuple") + s0 + 1;
        let s2 = header[s1..].find(')').expect("shape tuple") + s1;
        let shape: Vec<usize> = header[s1..s2]
            .split(',')
            .map(str::trim)
            .filter(|s| !s.is_empty())
            .map(|s| s.parse().expect("shape dim"))
            .collect();
        Arr {
            name: name.to_string(),
            shape,
            dt,
            bytes: raw[hstart + hlen..].to_vec(),
        }
    }
}

/// The inventory side-car: `name -> (shape, dtype)`, plus the file's sha256.
type Inventory = HashMap<String, (Vec<usize>, String)>;

fn load_inventory(path: &Path) -> (Inventory, String) {
    let bytes = std::fs::read(path).unwrap_or_else(|e| panic!("read {}: {e}", path.display()));
    let sha = format!("{:x}", Sha256::digest(&bytes));
    #[derive(serde::Deserialize)]
    struct Decl {
        shape: Vec<usize>,
        dtype: String,
    }
    #[derive(serde::Deserialize)]
    struct Inv {
        arrays: HashMap<String, Decl>,
    }
    let inv: Inv = serde_json::from_slice(&bytes).expect("inventory json");
    assert!(!inv.arrays.is_empty(), "{}: empty inventory", path.display());
    (
        inv.arrays
            .into_iter()
            .map(|(k, d)| (k, (d.shape, d.dtype)))
            .collect(),
        sha,
    )
}

/// Reads an array and checks it against the inventory's declaration — shape,
/// dtype, and non-emptiness — so a regenerated or truncated oracle cannot be
/// compared against silently.
struct Oracle {
    npz: Npz,
    inv: Inventory,
    tag: &'static str,
    seen: RefCell<BTreeSet<String>>,
}

impl Oracle {
    fn get(&self, name: &str) -> Arr {
        let a = self.npz.get(name);
        let d = self
            .inv
            .get(name)
            .unwrap_or_else(|| panic!("{name} is not declared in the {} inventory", self.tag));
        assert_eq!(a.shape, d.0, "{name}: shape {:?} != declared {:?}", a.shape, d.0);
        assert_eq!(a.dt.numpy_name(), d.1, "{name}: dtype != declared {}", d.1);
        assert!(a.len() > 0, "{name} is EMPTY");
        self.seen.borrow_mut().insert(name.to_string());
        a
    }
}

// ===========================================================================
// The checkpoint
// ===========================================================================

#[derive(serde::Deserialize)]
struct StEntry {
    dtype: String,
    shape: Vec<usize>,
    data_offsets: (u64, u64),
}

struct Ckpt {
    dir: PathBuf,
    weight_map: HashMap<String, String>,
    headers: RefCell<HashMap<String, (HashMap<String, StEntry>, u64)>>,
}

impl Ckpt {
    fn open(dir: &Path) -> Ckpt {
        #[derive(serde::Deserialize)]
        struct Index {
            weight_map: HashMap<String, String>,
        }
        let p = dir.join("model.safetensors.index.json");
        let f = File::open(&p).unwrap_or_else(|e| panic!("open {}: {e}", p.display()));
        let idx: Index = serde_json::from_reader(std::io::BufReader::with_capacity(1 << 22, f))
            .expect("index json");
        assert!(!idx.weight_map.is_empty(), "empty weight_map");
        Ckpt {
            dir: dir.to_path_buf(),
            weight_map: idx.weight_map,
            headers: RefCell::new(HashMap::new()),
        }
    }

    fn raw(&self, name: &str) -> (String, Vec<usize>, Vec<u8>) {
        let shard = self
            .weight_map
            .get(name)
            .unwrap_or_else(|| panic!("{name} is not in model.safetensors.index.json"));
        let path = self.dir.join(shard);
        if !self.headers.borrow().contains_key(shard) {
            let mut f =
                File::open(&path).unwrap_or_else(|e| panic!("open {}: {e}", path.display()));
            let mut l = [0u8; 8];
            f.read_exact(&mut l).expect("header len");
            let n = u64::from_le_bytes(l) as usize;
            let mut hb = vec![0u8; n];
            f.read_exact(&mut hb).expect("header");
            let raw_map: HashMap<String, serde_json::Value> =
                serde_json::from_slice(&hb).expect("safetensors header json");
            let map: HashMap<String, StEntry> = raw_map
                .into_iter()
                .filter(|(k, _)| k != "__metadata__")
                .map(|(k, v)| {
                    let e: StEntry = serde_json::from_value(v)
                        .unwrap_or_else(|err| panic!("header entry {k}: {err}"));
                    (k, e)
                })
                .collect();
            self.headers
                .borrow_mut()
                .insert(shard.clone(), (map, 8 + n as u64));
        }
        let hs = self.headers.borrow();
        let (map, base) = hs.get(shard).unwrap();
        // The index names a shard; the shard's OWN header must also name the
        // tensor. Checking both directions is what makes a stale index loud.
        let e = map
            .get(name)
            .unwrap_or_else(|| panic!("{name} is absent from {shard}'s own header"));
        let (s, t) = e.data_offsets;
        assert!(t > s, "{name}: empty tensor in {shard}");
        (
            e.dtype.clone(),
            e.shape.clone(),
            read_at(&path, base + s, (t - s) as usize),
        )
    }

    fn bf16(&self, name: &str) -> (Vec<usize>, Vec<f32>) {
        let (dt, shape, b) = self.raw(name);
        assert_eq!(dt, "BF16", "{name} is {dt}, expected BF16");
        let v = b
            .chunks_exact(2)
            .map(|c| f32::from_bits((u16::from_le_bytes([c[0], c[1]]) as u32) << 16))
            .collect();
        (shape, v)
    }

    fn f32(&self, name: &str) -> (Vec<usize>, Vec<f32>) {
        let (dt, shape, b) = self.raw(name);
        assert_eq!(dt, "F32", "{name} is {dt}, expected F32");
        let v = b
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
            .collect();
        (shape, v)
    }

    fn u8(&self, name: &str) -> (Vec<usize>, Vec<u8>) {
        let (dt, shape, b) = self.raw(name);
        assert_eq!(dt, "U8", "{name} is {dt}, expected U8");
        (shape, b)
    }

    /// Decode one MXFP4 plane: `[rows, cols/2]` packed nibbles + `[rows,
    /// cols/32]` E8M0 scales, both row-major.
    fn mxfp4_plane(&self, prefix: &str) -> (usize, usize, Vec<f32>) {
        let (ps, packed) = self.u8(&format!("{prefix}.weight_packed"));
        let (ss, scale) = self.u8(&format!("{prefix}.weight_scale"));
        assert_eq!(ps.len(), 2, "{prefix}.weight_packed rank");
        let rows = ps[0];
        let cols = ps[1] * 2;
        assert_eq!(ss, vec![rows, cols / 32], "{prefix}.weight_scale shape");
        (rows, cols, decode_mxfp4(&packed, &scale, rows, cols))
    }

    fn expert(&self, layer: usize, id: usize, dev: &Dev) -> ExpertWeights<B> {
        let p = format!("language_model.model.layers.{layer}.block_sparse_moe.experts.{id}");
        let mk = |suffix: &str| {
            let (r, c, v) = self.mxfp4_plane(&format!("{p}.{suffix}"));
            t2(v, [r, c], dev)
        };
        ExpertWeights {
            w1: mk("w1"),
            w2: mk("w2"),
            w3: mk("w3"),
        }
    }

    /// Everything in the block except the routed experts.
    fn block_weights(&self, layer: usize, bias_bf16: bool, dev: &Dev) -> LatentMoeWeights<B> {
        let p = format!("language_model.model.layers.{layer}.block_sparse_moe");
        let (gw_s, gw) = self.bf16(&format!("{p}.gate.weight"));
        let (_, bias_f32) = self.f32(&format!("{p}.gate.e_score_correction_bias"));
        let bias: Vec<f32> = if bias_bf16 {
            bias_f32
                .iter()
                .map(|&v| half::bf16::from_f32(v).to_f32())
                .collect()
        } else {
            bias_f32
        };
        let (dp_s, dp) = self.bf16(&format!("{p}.routed_expert_down_proj.weight"));
        let (up_s, up) = self.bf16(&format!("{p}.routed_expert_up_proj.weight"));
        let (_, nw) = self.bf16(&format!("{p}.routed_expert_norm.weight"));
        let (gp_s, gp) = self.bf16(&format!("{p}.shared_experts.gate_proj.weight"));
        let (upp_s, upp) = self.bf16(&format!("{p}.shared_experts.up_proj.weight"));
        let (dn_s, dn) = self.bf16(&format!("{p}.shared_experts.down_proj.weight"));
        LatentMoeWeights {
            down_proj: t2(dp, [dp_s[0], dp_s[1]], dev),
            up_proj: t2(up, [up_s[0], up_s[1]], dev),
            norm: Some(t1(nw, dev)),
            router: RouterWeights {
                weight: t2(gw, [gw_s[0], gw_s[1]], dev),
                bias: t1(bias, dev),
            },
            shared: Some(SharedExpertWeights {
                gate_proj: t2(gp, [gp_s[0], gp_s[1]], dev),
                up_proj: t2(upp, [upp_s[0], upp_s[1]], dev),
                down_proj: t2(dn, [dn_s[0], dn_s[1]], dev),
            }),
        }
    }
}

fn t2(v: Vec<f32>, shape: [usize; 2], dev: &Dev) -> Tensor<B, 2> {
    assert_eq!(v.len(), shape[0] * shape[1], "tensor data length vs {shape:?}");
    Tensor::<B, 2>::from_data(TensorData::new(v, shape), dev)
}

fn t1(v: Vec<f32>, dev: &Dev) -> Tensor<B, 1> {
    let n = v.len();
    Tensor::<B, 1>::from_data(TensorData::new(v, [n]), dev)
}

fn vec_of<const D: usize>(t: Tensor<B, D>) -> Vec<f32> {
    t.into_data().to_vec().expect("f32 tensor")
}

/// `x[m, k] @ w[cols, k]^T` in float64, for the first `cols` rows of `w`.
///
/// Deliberately the dumbest possible loop: this is the independent reference,
/// so it must not share an implementation with anything it checks.
fn host_f64_matmul(x: &[f32], m: usize, k: usize, w: &[f32], cols: usize) -> Vec<f64> {
    assert!(x.len() >= m * k && w.len() >= cols * k, "host matmul operands");
    let mut out = vec![0f64; m * cols];
    for i in 0..m {
        let xr = &x[i * k..i * k + k];
        for c in 0..cols {
            let wr = &w[c * k..c * k + k];
            let mut acc = 0f64;
            for t in 0..k {
                acc += (xr[t] as f64) * (wr[t] as f64);
            }
            out[i * cols + c] = acc;
        }
    }
    out
}

fn absmax(v: &[f32]) -> f64 {
    v.iter().fold(0f64, |m, &x| m.max((x as f64).abs()))
}

fn sigmoidf(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

// ===========================================================================
// Reporting
// ===========================================================================

struct Check {
    id: String,
    what: String,
    ok: bool,
    detail: String,
}

struct Report {
    checks: Vec<Check>,
    skipped: Vec<String>,
}

struct Cmp {
    n: usize,
    max_abs: f64,
    ref_absmax: f64,
    exact_frac: f64,
    nonfinite: usize,
}

fn compare(got: &[f32], want: &[f32]) -> Cmp {
    assert!(!got.is_empty(), "comparison against an EMPTY result");
    assert!(!want.is_empty(), "comparison against an EMPTY reference");
    assert_eq!(
        got.len(),
        want.len(),
        "length mismatch {} vs {}",
        got.len(),
        want.len()
    );
    let mut max_abs = 0f64;
    let mut ref_absmax = 0f64;
    let mut exact = 0usize;
    let mut nonfinite = 0usize;
    for (&g, &w) in got.iter().zip(want.iter()) {
        if !g.is_finite() || !w.is_finite() {
            nonfinite += 1;
            continue;
        }
        let d = (g as f64 - w as f64).abs();
        if d > max_abs {
            max_abs = d;
        }
        let a = (w as f64).abs();
        if a > ref_absmax {
            ref_absmax = a;
        }
        if g.to_bits() == w.to_bits() {
            exact += 1;
        }
    }
    Cmp {
        n: got.len(),
        max_abs,
        ref_absmax,
        exact_frac: exact as f64 / got.len() as f64,
        nonfinite,
    }
}

impl Report {
    fn new() -> Report {
        Report {
            checks: Vec::new(),
            skipped: Vec::new(),
        }
    }

    fn push(&mut self, id: &str, what: &str, ok: bool, detail: String) {
        println!(
            "  [{}] {id}  {what}\n         {detail}",
            if ok { "PASS" } else { "FAIL" }
        );
        self.checks.push(Check {
            id: id.to_string(),
            what: what.to_string(),
            ok,
            detail,
        });
    }

    /// `got` must match `want` to `budget`, and (when `exact_min` is given) at
    /// least that fraction of elements must be bit-identical.
    ///
    /// The failure test is `!(d <= budget)`, never `d > budget`: the latter is
    /// false for NaN, which would score garbage as zero error.
    fn close(
        &mut self,
        id: &str,
        what: &str,
        got: &[f32],
        want: &[f32],
        budget: f64,
        exact_min: Option<f64>,
    ) {
        let c = compare(got, want);
        let within = !(c.max_abs > budget) && c.max_abs.is_finite();
        let exact_ok = match exact_min {
            Some(m) => !(c.exact_frac < m),
            None => true,
        };
        let ok = within && exact_ok && c.nonfinite == 0;
        let extra = match exact_min {
            Some(m) => format!(
                ", bit-exact {:.4}% (min {:.2}%)",
                c.exact_frac * 100.0,
                m * 100.0
            ),
            None => format!(", bit-exact {:.4}%", c.exact_frac * 100.0),
        };
        self.push(
            id,
            what,
            ok,
            format!(
                "n={} max|d|={:.5e} budget={:.5e} ref|max|={:.5e}{extra}{}",
                c.n,
                c.max_abs,
                budget,
                c.ref_absmax,
                if c.nonfinite > 0 {
                    format!(", NON-FINITE {}", c.nonfinite)
                } else {
                    String::new()
                }
            ),
        );
    }

    fn exact(&mut self, id: &str, what: &str, got: &[f32], want: &[f32]) {
        let c = compare(got, want);
        let ok = c.exact_frac == 1.0 && c.nonfinite == 0;
        self.push(
            id,
            what,
            ok,
            format!(
                "n={} bit-exact {:.6}% max|d|={:.3e}",
                c.n,
                c.exact_frac * 100.0,
                c.max_abs
            ),
        );
    }

    fn bytes_eq(&mut self, id: &str, what: &str, a: &[u8], b: &[u8]) {
        assert!(!a.is_empty() && !b.is_empty(), "{id}: empty byte comparison");
        let ok = a == b;
        let diff = if a.len() == b.len() {
            a.iter().zip(b).filter(|(x, y)| x != y).count()
        } else {
            usize::MAX
        };
        self.push(id, what, ok, format!("{} bytes, {} differ", a.len(), diff));
    }

    fn boolean(&mut self, id: &str, what: &str, ok: bool, detail: String) {
        self.push(id, what, ok, detail);
    }

    /// A negative control: `got` must be MEASURABLY different from `want`.
    fn must_differ(&mut self, id: &str, what: &str, got: &[f32], want: &[f32], floor: f64) {
        let c = compare(got, want);
        let ok = !(c.max_abs <= floor) && c.max_abs.is_finite();
        self.push(
            id,
            what,
            ok,
            format!(
                "n={} max|d|={:.5e} must exceed {:.5e} (bit-exact {:.3}%)",
                c.n,
                c.max_abs,
                floor,
                c.exact_frac * 100.0
            ),
        );
    }

    fn skip(&mut self, lane: &str) {
        println!("  [SKIP] {lane}");
        self.skipped.push(lane.to_string());
    }

    fn failures(&self) -> Vec<&Check> {
        self.checks.iter().filter(|c| !c.ok).collect()
    }
}

// ===========================================================================
// main
// ===========================================================================

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let oracle_dir = PathBuf::from(
        args.get(1)
            .cloned()
            .or_else(|| std::env::var("K3_ORACLE_DIR").ok())
            .unwrap_or_else(|| "./k3-oracle".to_string()),
    );
    let model_dir = PathBuf::from(
        args.get(2)
            .cloned()
            .or_else(|| std::env::var("K3_MODEL_DIR").ok())
            .unwrap_or_else(|| "./kimi-k3".to_string()),
    );
    let fast = std::env::var("K3MOE_FAST").is_ok();

    println!("k3_moe_gate — mary::models::k3::moe against the whole-layer oracle");
    println!("  oracle: {}", oracle_dir.display());
    println!("  model:  {}", model_dir.display());
    if fast {
        println!("\n  ####  RUN MODE: PARTIAL (K3MOE_FAST=1)  ####");
        println!("  ####  skipped lanes are named in the verdict; the authoritative");
        println!("  ####  result is a run WITHOUT this flag.");
    }

    let t0 = std::time::Instant::now();
    let dev: Dev = Default::default();
    let mut r = Report::new();

    // ---------------------------------------------------------------- oracle
    println!("\n== O: the oracle is the oracle ==");
    // sha256 as published in MANIFEST_layer_oracle.md §9.
    const SHA_PREFIX: &str = "fdb3b897f0bb43e8506d27dd283defee87910006dd1038c131687a1b48e61d7c";
    const SHA_LADDER: &str = "83daedc5071e93bcbed3f7bedeaefbc84c309ddc08c43fcaa6346150f958d1e5";
    const SHA_INV_PREFIX: &str = "af853091814e9627cd61c20f1c55a4acfab978c928b304715819a4a0f7d067eb";
    const SHA_INV_LADDER: &str = "67a00cfa32045a5e982e96fa900684f95dfbcffa08ed33224c7e4bb0c7336981";

    let (inv_p, sha_inv_p) =
        load_inventory(&oracle_dir.join("layer_oracle_prefix13_bf16_inventory.json"));
    let (inv_l, sha_inv_l) = load_inventory(&oracle_dir.join("layer_oracle_ladder_inventory.json"));
    let p = Oracle {
        npz: Npz::open(&oracle_dir.join("layer_oracle_prefix13_bf16.npz")),
        inv: inv_p,
        tag: "prefix13",
        seen: RefCell::new(BTreeSet::new()),
    };
    let l = Oracle {
        npz: Npz::open(&oracle_dir.join("layer_oracle_ladder.npz")),
        inv: inv_l,
        tag: "ladder",
        seen: RefCell::new(BTreeSet::new()),
    };

    r.boolean(
        "O1",
        "layer_oracle_prefix13_bf16.npz sha256 == MANIFEST",
        p.npz.sha256 == SHA_PREFIX,
        p.npz.sha256.clone(),
    );
    r.boolean(
        "O2",
        "layer_oracle_ladder.npz sha256 == MANIFEST",
        l.npz.sha256 == SHA_LADDER,
        l.npz.sha256.clone(),
    );
    r.boolean(
        "O3",
        "prefix13 inventory sha256 == MANIFEST",
        sha_inv_p == SHA_INV_PREFIX,
        sha_inv_p.clone(),
    );
    r.boolean(
        "O4",
        "ladder inventory sha256 == MANIFEST",
        sha_inv_l == SHA_INV_LADDER,
        sha_inv_l.clone(),
    );
    for (id, o) in [("O5", &p), ("O6", &l)] {
        let members = o.npz.names();
        let declared: BTreeSet<String> = o.inv.keys().cloned().collect();
        let only_file: Vec<_> = members.difference(&declared).cloned().collect();
        let only_inv: Vec<_> = declared.difference(&members).cloned().collect();
        r.boolean(
            id,
            &format!(
                "{}: npz members == inventory declarations, both directions",
                o.tag
            ),
            only_file.is_empty() && only_inv.is_empty() && !members.is_empty(),
            format!(
                "{} members, {} declared; file-only {:?}; declared-only {:?}",
                members.len(),
                declared.len(),
                &only_file[..only_file.len().min(4)],
                &only_inv[..only_inv.len().min(4)]
            ),
        );
    }

    // -------------------------------------------------------------- checkpoint
    let ckpt = Ckpt::open(&model_dir);
    let cfg_json = std::fs::read_to_string(model_dir.join("config.json")).expect("config.json");
    let cfg = K3Config::from_json(&cfg_json).expect("config.json parses and validates");

    // ---------------------------------------------------------------- config
    println!("\n== C: the config, and the shapes this port refuses ==");
    let dims = MoeDims::from_text_config(&cfg.text_config).expect("MoeDims from the real config");
    r.boolean(
        "C1",
        "MoeDims::from_text_config(real config) == the hard-coded MoeDims::k3()",
        dims == MoeDims::k3(),
        format!("{dims:?}"),
    );
    {
        let cases: Vec<(&str, Box<dyn Fn(&mut K3TextConfig)>)> = vec![
            (
                "moe_router_activation_func=softmax",
                Box::new(|c: &mut K3TextConfig| c.moe_router_activation_func = "softmax".into()),
            ),
            (
                "topk_method=greedy",
                Box::new(|c: &mut K3TextConfig| c.topk_method = "greedy".into()),
            ),
            (
                "num_expert_group=8 (grouped top-k)",
                Box::new(|c: &mut K3TextConfig| c.num_expert_group = 8),
            ),
            (
                "hidden_act=silu",
                Box::new(|c: &mut K3TextConfig| c.hidden_act = "silu".into()),
            ),
            (
                "routed_expert_hidden_size=None",
                Box::new(|c: &mut K3TextConfig| c.routed_expert_hidden_size = None),
            ),
        ];
        let n = cases.len();
        let mut accepted = Vec::new();
        for (name, f) in cases {
            let mut c = cfg.text_config.clone();
            f(&mut c);
            if MoeDims::from_text_config(&c).is_ok() {
                accepted.push(name.to_string());
            }
        }
        r.boolean(
            "C2",
            "every unmodelled config shape is REFUSED, not silently run",
            accepted.is_empty(),
            format!(
                "{}/{n} refused; accepted (= a silent hole): {accepted:?}",
                n - accepted.len()
            ),
        );
    }
    {
        // The config's own layer predicate against the oracle's array set.
        let mut wrong = Vec::new();
        for layer in 0..13usize {
            let declared = cfg.text_config.is_moe_layer(layer);
            let captured = p.npz.has(&format!("L{layer:02}_moe_in_bf16bits"));
            if declared != captured {
                wrong.push(layer);
            }
        }
        r.boolean(
            "C3",
            "config.is_moe_layer agrees with which layers the shipped run actually put a \
             MoE block on (layer 0 is dense: first_k_dense_replace = 1)",
            wrong.is_empty(),
            format!("layers 0..12 checked, disagreements: {wrong:?}"),
        );
    }
    r.boolean(
        "C4",
        "routed_scaling_factor is 1.0 in THIS checkpoint — so NO captured vector can \
         distinguish a port that applies it from one that ignores it",
        dims.routed_scaling_factor == 1.0,
        format!("routed_scaling_factor = {}", dims.routed_scaling_factor),
    );
    {
        // ...so exercise it synthetically instead, on the port's own router.
        let mut d2 = dims.clone();
        d2.routed_scaling_factor = 2.0;
        let mut d3 = dims.clone();
        d3.moe_renormalize = false;
        let hs = t2(
            (0..4 * dims.hidden_size)
                .map(|i| ((i % 97) as f32 - 48.0) / 64.0)
                .collect(),
            [4, dims.hidden_size],
            &dev,
        );
        let rw = RouterWeights {
            weight: t2(
                (0..dims.num_experts * dims.hidden_size)
                    .map(|i| ((i % 31) as f32 - 15.0) / 512.0)
                    .collect(),
                [dims.num_experts, dims.hidden_size],
                &dev,
            ),
            bias: t1(
                (0..dims.num_experts)
                    .map(|i| ((i % 13) as f32 - 6.0) / 100.0)
                    .collect(),
                &dev,
            ),
        };
        let base = LatentMoe::new_f32(dims.clone()).route(hs.clone(), &rw);
        let scaled = LatentMoe::new_f32(d2).route(hs.clone(), &rw);
        let unnorm = LatentMoe::new_f32(d3).route(hs, &rw);
        let ok_scale = base
            .topk_weight
            .iter()
            .zip(scaled.topk_weight.iter())
            .all(|(a, b)| (b - 2.0 * a).abs() <= 1e-6 * a.abs().max(1e-6))
            && base.topk_weight.iter().any(|&w| w > 0.0);
        r.boolean(
            "C5",
            "routed_scaling_factor IS consulted: doubling it doubles every combining weight",
            ok_scale,
            format!(
                "{} weights, first {:.6} -> {:.6}",
                base.topk_weight.len(),
                base.topk_weight[0],
                scaled.topk_weight[0]
            ),
        );
        let row_sum: f32 = base.topk_weight[..dims.top_k].iter().sum();
        let ok_renorm = unnorm
            .topk_weight
            .iter()
            .zip(unnorm.topk_weight_prerenorm.iter())
            .all(|(a, b)| a == b)
            && base.topk_weight != base.topk_weight_prerenorm
            && (row_sum - 1.0).abs() < 1e-5;
        r.boolean(
            "C6",
            "moe_renormalize IS consulted: false leaves the gathered scores alone, true \
             makes each token's 16 weights sum to 1",
            ok_renorm,
            format!("renormalised row 0 sums to {row_sum:.7}"),
        );
    }

    // ---------------------------------------------------------------- MXFP4
    println!("\n== M: the MXFP4 expert decode, asserted directly ==");
    {
        let pk_o = l.get("mxfp4_L04_E000_w1_packed_u8");
        let sc_o = l.get("mxfp4_L04_E000_w1_scale_u8");
        let (_, pk_c) =
            ckpt.u8("language_model.model.layers.4.block_sparse_moe.experts.0.w1.weight_packed");
        let (_, sc_c) =
            ckpt.u8("language_model.model.layers.4.block_sparse_moe.experts.0.w1.weight_scale");
        r.bytes_eq(
            "M1",
            "the checkpoint's packed nibbles for L4/E0/w1 ARE the bytes the oracle decoded",
            &pk_c,
            pk_o.u8s(),
        );
        r.bytes_eq(
            "M2",
            "...and so are its E8M0 scale bytes",
            &sc_c,
            sc_o.u8s(),
        );
        let dec = decode_mxfp4(&pk_c, &sc_c, 3072, 3584);
        let ref64 = l.get("mxfp4_L04_E000_w1_decoded_f64").f64s();
        assert!(!dec.is_empty() && dec.len() == ref64.len(), "decode length");
        let mut bad = 0usize;
        let mut worst = 0f64;
        let mut nz = 0usize;
        let mut neg_zero = 0usize;
        for (i, &v) in dec.iter().enumerate() {
            let d = (v as f64 - ref64[i]).abs();
            if !(d <= 0.0) {
                bad += 1;
                if !(d <= worst) {
                    worst = d;
                }
            }
            if v != 0.0 {
                nz += 1;
            } else if v.is_sign_negative() {
                neg_zero += 1;
            }
        }
        r.boolean(
            "M3",
            "decode_mxfp4(checkpoint bytes) == the oracle's float64 decode, BIT-EXACT",
            bad == 0,
            format!(
                "{} values ({nz} nonzero, {neg_zero} negative zeros), {bad} differ, worst {worst:.3e}",
                dec.len()
            ),
        );
        let alt = l
            .get("mxfp4_ALT_L04_E000_w1_decoded_f64_high_nibble_first")
            .f64s();
        let mut worst_alt = 0f64;
        let mut diff_alt = 0usize;
        for (i, &v) in dec.iter().enumerate() {
            let d = (v as f64 - alt[i]).abs();
            if !(d <= worst_alt) {
                worst_alt = d;
            }
            if d != 0.0 {
                diff_alt += 1;
            }
        }
        r.boolean(
            "M4",
            "NEGATIVE CONTROL: the high-nibble-first decode is NOT what we produce",
            !(worst_alt <= 0.0) && worst_alt.is_finite(),
            format!(
                "{diff_alt} of {} values differ, worst {worst_alt:.3e}",
                dec.len()
            ),
        );
    }

    // ------------------------------------------------- single expert isolated
    println!("\n== E: one routed expert in isolation (layer 4, experts 6 and 12) ==");
    let moe_bf16 = LatentMoe::new(dims.clone());
    for e in [6usize, 12] {
        let tag = format!("L04_expert{e:03}");
        let x_a = p.get(&format!("{tag}_in_bf16bits"));
        let rows = x_a.shape[0];
        let x = x_a.bf16_as_f32();
        let w1_in = p.get(&format!("{tag}_w1_in_bf16bits")).bf16_as_f32();
        let w3_in = p.get(&format!("{tag}_w3_in_bf16bits")).bf16_as_f32();
        let w1_out = p.get(&format!("{tag}_w1_out_bf16bits")).bf16_as_f32();
        let w3_out = p.get(&format!("{tag}_w3_out_bf16bits")).bf16_as_f32();
        let situ_in = p.get(&format!("{tag}_situ_in_bf16bits")).bf16_as_f32();
        let situ_out = p.get(&format!("{tag}_situ_out_bf16bits")).bf16_as_f32();
        let w2_in = p.get(&format!("{tag}_w2_in_bf16bits")).bf16_as_f32();
        let w2_out = p.get(&format!("{tag}_w2_out_bf16bits")).bf16_as_f32();
        let out = p.get(&format!("{tag}_out_bf16bits")).bf16_as_f32();

        // The oracle's own wiring first, before anything is leaned on.
        r.exact(&format!("E1a/{e}"), "oracle wiring: w1_in == expert_in", &w1_in, &x);
        r.exact(&format!("E1b/{e}"), "oracle wiring: w3_in == expert_in", &w3_in, &x);
        r.exact(
            &format!("E1c/{e}"),
            "oracle wiring: w2_in == situ_out",
            &w2_in,
            &situ_out,
        );
        r.exact(
            &format!("E1d/{e}"),
            "oracle wiring: the expert's output IS w2's output",
            &out,
            &w2_out,
        );
        {
            let cat: Vec<f32> = w1_out.iter().chain(w3_out.iter()).copied().collect();
            r.exact(
                &format!("E1e/{e}"),
                "oracle wiring: situ_in == cat(w1_out, w3_out) — w1 IS the gate half",
                &cat,
                &situ_in,
            );
        }

        let ew = ckpt.expert(4, e, &dev);
        let xt = t2(x.clone(), [rows, dims.moe_hidden_size], &dev);
        r.close(
            &format!("E2/{e}"),
            "w1(expert_in) == the shipped w1 output",
            &vec_of(moe_bf16.linear(xt.clone(), ew.w1.clone())),
            &w1_out,
            bf16_budget(1, absmax(&w1_out)),
            Some(BF16_EXACT_FRAC_MIN),
        );
        r.close(
            &format!("E3/{e}"),
            "w3(expert_in) == the shipped w3 output",
            &vec_of(moe_bf16.linear(xt.clone(), ew.w3.clone())),
            &w3_out,
            bf16_budget(1, absmax(&w3_out)),
            Some(BF16_EXACT_FRAC_MIN),
        );
        // situ driven from the CAPTURED situ input, so this is situ alone.
        let si = t2(situ_in.clone(), [rows, 2 * dims.moe_intermediate_size], &dev);
        r.close(
            &format!("E4/{e}"),
            "situ(captured situ_in) == the shipped SituAndMul output",
            &vec_of(ActRound::Bf16.apply(dims.situ.forward(si.clone()))),
            &situ_out,
            bf16_budget(1, absmax(&situ_out)),
            Some(BF16_EXACT_FRAC_MIN),
        );
        let so = t2(situ_out.clone(), [rows, dims.moe_intermediate_size], &dev);
        r.close(
            &format!("E5/{e}"),
            "w2(captured situ_out) == the shipped w2 output",
            &vec_of(moe_bf16.linear(so.clone(), ew.w2.clone())),
            &w2_out,
            bf16_budget(1, absmax(&w2_out)),
            Some(BF16_EXACT_FRAC_MIN),
        );
        let tr = moe_bf16.expert_traced(xt.clone(), &ew);
        r.close(
            &format!("E6/{e}"),
            "our own cat(w1(x), w3(x)) == the captured situ_in",
            &vec_of(tr.situ_in.clone()),
            &situ_in,
            bf16_budget(1, absmax(&situ_in)),
            Some(BF16_EXACT_FRAC_MIN),
        );
        r.close(
            &format!("E7/{e}"),
            "the whole expert, composed, == the shipped expert output",
            &vec_of(tr.out.clone()),
            &out,
            bf16_budget(3, absmax(&out)),
            None,
        );

        // NEGATIVE CONTROLS.
        let swapped = ExpertWeights {
            w1: ew.w3.clone(),
            w2: ew.w2.clone(),
            w3: ew.w1.clone(),
        };
        r.must_differ(
            &format!("E8/{e}"),
            "NEGATIVE: gate/up (w1/w3) swapped must NOT reproduce the expert output",
            &vec_of(moe_bf16.expert_traced(xt.clone(), &swapped).out),
            &out,
            bf16_budget(3, absmax(&out)),
        );
        {
            let g = si.clone().narrow(1, 0, dims.moe_intermediate_size);
            let u = si.narrow(1, dims.moe_intermediate_size, dims.moe_intermediate_size);
            let swiglu =
                ActRound::Bf16.apply(g.clone() * burn::tensor::activation::sigmoid(g) * u);
            r.must_differ(
                &format!("E9/{e}"),
                "NEGATIVE: plain SwiGLU in place of situ must NOT reproduce the activation",
                &vec_of(swiglu),
                &situ_out,
                bf16_budget(1, absmax(&situ_out)),
            );
        }
    }

    // ------------------------------------------ router / latent / shared sweep
    println!("\n== R/L/S: router, latent projections, shared experts — 32 tokens ==");
    let layers: Vec<usize> = if fast { vec![1, 4, 12] } else { (1..=12).collect() };
    if fast {
        r.skip("R/L/S sweep restricted to layers 1, 4, 12 (K3MOE_FAST)");
    }
    let moe_f32 = LatentMoe::new_f32(dims.clone());
    for &layer in &layers {
        let tag = format!("L{layer:02}");
        let pre = format!("language_model.model.layers.{layer}.block_sparse_moe");
        let (_, bias_f32) = ckpt.f32(&format!("{pre}.gate.e_score_correction_bias"));
        let bias_bf16: Vec<f32> = bias_f32
            .iter()
            .map(|&v| half::bf16::from_f32(v).to_f32())
            .collect();
        let bias_oracle = p
            .get(&format!("{tag}_moe_router_e_score_correction_bias_bf16bits"))
            .bf16_as_f32();
        r.exact(
            &format!("R1/{tag}"),
            "the shipped bf16 run's router bias IS bf16(the checkpoint's F32 bias) — the \
             checkpoint stores F32 and `dtype=bfloat16` casts it down",
            &bias_bf16,
            &bias_oracle,
        );

        let w = ckpt.block_weights(layer, true, &dev);
        let hin = p.get(&format!("{tag}_moe_gate_in_bf16bits"));
        let ntok = hin.shape[0] * hin.shape[1];
        let h = t2(hin.bf16_as_f32(), [ntok, dims.hidden_size], &dev);
        let rt = moe_f32.route(h.clone(), &w.router);

        let logits = p.get(&format!("{tag}_moe_router_logits")).f32s();
        let scores = p.get(&format!("{tag}_moe_router_scores")).f32s();
        let sfc = p.get(&format!("{tag}_moe_router_scores_for_choice")).f32s();
        r.close(
            &format!("R2/{tag}"),
            "router logits == F.linear(h.float(), W.float())",
            &vec_of(rt.logits.clone()),
            &logits,
            f32_budget(1, absmax(&logits)),
            None,
        );
        r.close(
            &format!("R3/{tag}"),
            "scores == sigmoid(logits)",
            &vec_of(rt.scores.clone()),
            &scores,
            f32_budget(1, absmax(&scores)),
            None,
        );
        r.close(
            &format!("R4/{tag}"),
            "scores_for_choice == scores + bias (the bias lands on the SCORE)",
            &vec_of(rt.scores_for_choice.clone()),
            &sfc,
            f32_budget(1, absmax(&sfc)),
            None,
        );

        let oi: Vec<usize> = p
            .get(&format!("{tag}_moe_gate_out_topk_idx"))
            .i64s()
            .into_iter()
            .map(|v| v as usize)
            .collect();
        let ow = p.get(&format!("{tag}_moe_gate_out_topk_weight")).f32s();
        let opre = p
            .get(&format!("{tag}_moe_router_topk_weight_prerenorm"))
            .f32s();
        let k = dims.top_k;
        let al = align_topk(
            &rt.topk_idx,
            &rt.topk_weight,
            &rt.topk_weight_prerenorm,
            &oi,
            &ow,
            &opre,
            ntok,
            k,
        );
        r.boolean(
            &format!("R5/{tag}"),
            "the selected expert SET matches the shipped gate for every token (torch's \
             `sorted=False` order is unspecified — measured on the oracle it is neither \
             score- nor id-ordered — so this must be a set test)",
            al.n_set_diff == 0,
            format!("{} of {ntok} tokens differ", al.n_set_diff),
        );
        r.boolean(
            &format!("R6/{tag}"),
            "the top-k boundary is unambiguous: the k-th and (k+1)-th scores_for_choice \
             differ, so the SET is well defined and not a tie-break convention",
            rt.min_topk_margin > 0.0 && rt.min_topk_margin.is_finite(),
            format!("min margin = {:.4e}", rt.min_topk_margin),
        );
        r.close(
            &format!("R7/{tag}"),
            "topk_weight_prerenorm == scores.gather(idx) — the UNBIASED score",
            &al.mine_pre,
            &al.ref_pre,
            f32_budget(1, absmax(&al.ref_pre)),
            None,
        );
        r.close(
            &format!("R8/{tag}"),
            "topk_weight == prerenorm / their sum",
            &al.mine_w,
            &al.ref_w,
            f32_budget(1, absmax(&al.ref_w)),
            None,
        );

        // NEGATIVE CONTROL (C7): the combining weight from the BIASED score.
        {
            let alt = p
                .get(&format!(
                    "{tag}_moe_router_ALT_topk_weight_from_scores_for_choice"
                ))
                .f32s();
            let sfc_v = vec_of(rt.scores_for_choice.clone());
            let ne = dims.num_experts;
            let mut wrong = vec![0f32; ntok * k];
            for t in 0..ntok {
                let mut s = 0f32;
                for j in 0..k {
                    s += sfc_v[t * ne + rt.topk_idx[t * k + j]];
                }
                for j in 0..k {
                    wrong[t * k + j] = sfc_v[t * ne + rt.topk_idx[t * k + j]] / (s + 1e-20);
                }
            }
            let al2 = align_topk(&rt.topk_idx, &wrong, &wrong, &oi, &alt, &alt, ntok, k);
            r.must_differ(
                &format!("R9a/{tag}"),
                "NEGATIVE: a weight taken from scores_for_choice must NOT equal the real weight",
                &al2.mine_w,
                &al.ref_w,
                f32_budget(1, absmax(&al.ref_w)),
            );
            r.close(
                &format!("R9b/{tag}"),
                "...and it must REPRODUCE the oracle's stored wrong answer — a control that \
                 does not discriminate is not a control",
                &al2.mine_w,
                &al2.ref_w,
                f32_budget(1, absmax(&al2.ref_w)),
                None,
            );
        }
        // NEGATIVE CONTROL (C8): the bias added to the LOGIT.
        {
            let alt_idx: Vec<usize> = p
                .get(&format!("{tag}_moe_router_ALT_topk_idx_bias_on_logits"))
                .i64s()
                .into_iter()
                .map(|v| v as usize)
                .collect();
            let same_declared = p
                .get(&format!(
                    "{tag}_moe_router_ALT_bias_on_logits_n_tokens_same_set"
                ))
                .i64s()[0] as usize;
            let lv = vec_of(rt.logits.clone());
            let ne = dims.num_experts;
            let mut wrong_sets: Vec<BTreeSet<usize>> = Vec::with_capacity(ntok);
            for t in 0..ntok {
                let mut row: Vec<(f32, usize)> = (0..ne)
                    .map(|e| (sigmoidf(lv[t * ne + e] + bias_bf16[e]), e))
                    .collect();
                row.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap().then(a.1.cmp(&b.1)));
                wrong_sets.push(row[..k].iter().map(|q| q.1).collect());
            }
            let real_sets: Vec<BTreeSet<usize>> = (0..ntok)
                .map(|t| oi[t * k..(t + 1) * k].iter().copied().collect())
                .collect();
            let alt_sets: Vec<BTreeSet<usize>> = (0..ntok)
                .map(|t| alt_idx[t * k..(t + 1) * k].iter().copied().collect())
                .collect();
            let same_vs_real = (0..ntok).filter(|&t| wrong_sets[t] == real_sets[t]).count();
            let same_vs_alt = (0..ntok).filter(|&t| wrong_sets[t] == alt_sets[t]).count();
            r.boolean(
                &format!("R10a/{tag}"),
                "NEGATIVE: the bias on the LOGIT selects a different set — on exactly the \
                 number of tokens the oracle recorded",
                same_vs_real == same_declared,
                format!("same as real on {same_vs_real}/{ntok}; oracle recorded {same_declared}"),
            );
            r.boolean(
                &format!("R10b/{tag}"),
                "...and it reproduces the oracle's stored wrong SET on every token",
                same_vs_alt == ntok,
                format!("matches the stored ALT set on {same_vs_alt}/{ntok}"),
            );
        }

        // ------------------------------------------------------------ latent
        let ldi = p.get(&format!("{tag}_moe_latent_down_in_bf16bits")).bf16_as_f32();
        let ldo = p.get(&format!("{tag}_moe_latent_down_out_bf16bits")).bf16_as_f32();
        let lni = p.get(&format!("{tag}_moe_latent_norm_in_bf16bits")).bf16_as_f32();
        let lno = p.get(&format!("{tag}_moe_latent_norm_out_bf16bits")).bf16_as_f32();
        let lui = p.get(&format!("{tag}_moe_latent_up_in_bf16bits")).bf16_as_f32();
        let luo = p.get(&format!("{tag}_moe_latent_up_out_bf16bits")).bf16_as_f32();
        let moe_in = p.get(&format!("{tag}_moe_in_bf16bits")).bf16_as_f32();
        let shared_in = p.get(&format!("{tag}_moe_shared_in_bf16bits")).bf16_as_f32();
        r.exact(
            &format!("L1a/{tag}"),
            "oracle wiring: moe_in (flattened) IS routed_expert_down_proj's input",
            &moe_in,
            &ldi,
        );
        r.exact(
            &format!("L1b/{tag}"),
            "oracle wiring: routed_expert_norm's output IS routed_expert_up_proj's input \
             (C9: the norm sits BETWEEN the combination and the up-projection)",
            &lno,
            &lui,
        );
        r.exact(
            &format!("L1c/{tag}"),
            "oracle wiring: the shared experts see the BLOCK's input, not the latent (C10)",
            &moe_in,
            &shared_in,
        );

        r.close(
            &format!("L2/{tag}"),
            "routed_expert_down_proj (K=7168; no bit-exactness required — see X1)",
            &vec_of(moe_bf16.linear(
                t2(ldi.clone(), [ntok, dims.hidden_size], &dev),
                w.down_proj.clone(),
            )),
            &ldo,
            bf16_budget(3, absmax(&ldo)),
            None,
        );
        let nwt = w.norm.clone().unwrap();
        r.close(
            &format!("L3/{tag}"),
            "routed_expert_norm (KimiRMSNorm: f32 inside, cast BEFORE the weight multiply)",
            &vec_of(moe_bf16.rms_norm(
                t2(lni.clone(), [ntok, dims.moe_hidden_size], &dev),
                nwt.clone(),
            )),
            &lno,
            bf16_budget(2, absmax(&lno)),
            Some(BF16_EXACT_FRAC_MIN),
        );
        r.close(
            &format!("L4/{tag}"),
            "routed_expert_up_proj (K=3584 x 32 rows; no bit-exactness required — see X1)",
            &vec_of(moe_bf16.linear(
                t2(lui.clone(), [ntok, dims.moe_hidden_size], &dev),
                w.up_proj.clone(),
            )),
            &luo,
            bf16_budget(3, absmax(&luo)),
            None,
        );
        {
            let mut lo = f64::INFINITY;
            let mut hi = 0f64;
            for t in 0..ntok {
                let row = &lni[t * dims.moe_hidden_size..(t + 1) * dims.moe_hidden_size];
                let s: f64 = row.iter().map(|&v| (v as f64) * (v as f64)).sum();
                let rms = (s / dims.moe_hidden_size as f64).sqrt();
                lo = lo.min(rms);
                hi = hi.max(rms);
            }
            r.boolean(
                &format!("L5/{tag}"),
                "C9 as a MEASUREMENT: the norm's input has a non-constant per-token RMS, so \
                 the norm is doing work — it is not a mis-placed no-op",
                hi / lo > 1.5 && lo.is_finite(),
                format!("per-token RMS {lo:.4}..{hi:.4}, ratio {:.2}", hi / lo),
            );
            let scaled = t2(lni.clone(), [ntok, dims.moe_hidden_size], &dev)
                * nwt.clone().unsqueeze::<2>();
            r.must_differ(
                &format!("L6/{tag}"),
                "NEGATIVE: applying the norm weight BEFORE normalising must not reproduce it",
                &vec_of(moe_bf16.rms_norm(scaled, nwt.clone())),
                &lno,
                bf16_budget(2, absmax(&lno)),
            );
        }

        // ------------------------------------------------------------ shared
        //
        // Each sub-module is driven from ITS OWN captured input, exactly as the
        // single-expert lane is, so a compensating pair of errors cannot hide
        // inside the composition. The composition is then checked separately.
        let sw = w.shared.clone().unwrap();
        let fw = dims.shared_intermediate_size.expect("K3 has shared experts");
        let sgo = p.get(&format!("{tag}_moe_shared_gate_proj_out_bf16bits")).bf16_as_f32();
        let suo = p.get(&format!("{tag}_moe_shared_up_proj_out_bf16bits")).bf16_as_f32();
        let ssi = p.get(&format!("{tag}_moe_shared_situ_in_bf16bits")).bf16_as_f32();
        let sso = p.get(&format!("{tag}_moe_shared_situ_out_bf16bits")).bf16_as_f32();
        let sdo = p.get(&format!("{tag}_moe_shared_down_proj_out_bf16bits")).bf16_as_f32();
        let sout = p.get(&format!("{tag}_moe_shared_out_bf16bits")).bf16_as_f32();
        let sin_t = t2(shared_in.clone(), [ntok, dims.hidden_size], &dev);
        r.close(
            &format!("S1/{tag}"),
            "shared gate_proj, from the captured block input",
            &vec_of(moe_bf16.linear(sin_t.clone(), sw.gate_proj.clone())),
            &sgo,
            bf16_budget(3, absmax(&sgo)),
            None,
        );
        r.close(
            &format!("S2/{tag}"),
            "shared up_proj, from the captured block input",
            &vec_of(moe_bf16.linear(sin_t.clone(), sw.up_proj.clone())),
            &suo,
            bf16_budget(3, absmax(&suo)),
            None,
        );
        {
            let cat: Vec<f32> = (0..ntok)
                .flat_map(|t| {
                    sgo[t * fw..(t + 1) * fw]
                        .iter()
                        .chain(suo[t * fw..(t + 1) * fw].iter())
                        .copied()
                        .collect::<Vec<f32>>()
                })
                .collect();
            r.exact(
                &format!("S3/{tag}"),
                "oracle wiring: the shared situ input is cat([gate_proj, up_proj]) in THAT \
                 order — gate first",
                &cat,
                &ssi,
            );
        }
        r.close(
            &format!("S4/{tag}"),
            "shared situ, from the captured situ input",
            &vec_of(ActRound::Bf16.apply(
                dims.situ.forward(t2(ssi.clone(), [ntok, 2 * fw], &dev)),
            )),
            &sso,
            bf16_budget(1, absmax(&sso)),
            Some(BF16_EXACT_FRAC_MIN),
        );
        r.close(
            &format!("S5/{tag}"),
            "shared down_proj, from the captured activation",
            &vec_of(moe_bf16.linear(t2(sso.clone(), [ntok, fw], &dev), sw.down_proj.clone())),
            &sdo,
            bf16_budget(3, absmax(&sdo)),
            None,
        );
        r.exact(
            &format!("S6/{tag}"),
            "oracle wiring: the shared MLP output IS its down_proj output",
            &sout,
            &sdo,
        );
        r.close(
            &format!("S7/{tag}"),
            "the whole fused shared MLP, composed",
            &vec_of(moe_bf16.shared_traced(sin_t.clone(), &sw).out),
            &sout,
            bf16_budget(6, absmax(&sout)),
            None,
        );
        {
            let swapped = SharedExpertWeights {
                gate_proj: sw.up_proj.clone(),
                up_proj: sw.gate_proj.clone(),
                down_proj: sw.down_proj.clone(),
            };
            r.must_differ(
                &format!("S8/{tag}"),
                "NEGATIVE: shared gate/up swapped must not reproduce the shared output",
                &vec_of(moe_bf16.shared_traced(sin_t.clone(), &swapped).out),
                &sout,
                bf16_budget(6, absmax(&sout)),
            );
        }

        // ---- X1: characterise the REFERENCE, once, on the widest projection.
        //
        // A float64 recomputation of `latent_down_in @ down_projT` from the
        // checkpoint's own bytes. Two things fall out of one measurement:
        // (a) the shipped tensor IS that product, which pins the weight-name to
        // module-slot mapping independently of this port; and (b) how far the
        // shipped bf16 GEMM sits from the exact answer, which is what decides
        // whether bit-exactness is a fair thing to demand of a port. Only the
        // first 256 output columns, to keep a scalar f64 GEMM cheap.
        if layer == 4 {
            let cols = 256usize;
            let (dp_shape, dp_raw) =
                ckpt.bf16(&format!("{pre}.routed_expert_down_proj.weight"));
            assert_eq!(dp_shape, vec![dims.moe_hidden_size, dims.hidden_size]);
            let exact = host_f64_matmul(&ldi, ntok, dims.hidden_size, &dp_raw, cols);
            let shipped: Vec<f32> = (0..ntok)
                .flat_map(|t| {
                    ldo[t * dims.moe_hidden_size..t * dims.moe_hidden_size + cols].to_vec()
                })
                .collect();
            assert_eq!(exact.len(), shipped.len());
            assert!(!shipped.is_empty());
            let ulp = absmax(&shipped) * BF16_ULP_REL;
            let mut worst = 0f64;
            let mut bits = 0usize;
            for i in 0..shipped.len() {
                let d = (exact[i] - shipped[i] as f64).abs();
                if !(d <= worst) {
                    worst = d;
                }
                if half::bf16::from_f64(exact[i]).to_f32().to_bits() == shipped[i].to_bits() {
                    bits += 1;
                }
            }
            r.boolean(
                "X1",
                "the shipped wide projection IS x @ W^T for the checkpoint weight — an \
                 independent float64 recomputation lands within 2 bf16 ulp of it. The \
                 SHORTFALL from bit-exactness here belongs to the reference kernel, not \
                 to any port, and is why the wide bf16 lanes above ask for a budget and \
                 not for bits",
                !(worst > 2.0 * ulp) && worst.is_finite() && ulp > 0.0,
                format!(
                    "{} values: shipped-vs-exact worst {:.4} bf16 ulp (of tensor max {:.4e}); \
                     shipped is bit-exact on only {:.2}% of them",
                    shipped.len(),
                    worst / ulp,
                    absmax(&shipped),
                    100.0 * bits as f64 / shipped.len() as f64
                ),
            );
        }
    }

    // ------------------------------------------ the routed-expert combination
    println!("\n== G/W: the routed-expert combination and the whole block ==");
    if fast {
        r.skip("G/W 32-token layer-4 lane (172 real experts) — K3MOE_FAST");
    } else {
        run_bf16_block(&mut r, &p, &ckpt, &dims, &dev, 4, "L04");
    }
    check_combination_rounds_once(&mut r, &dims, &dev);
    for (lane, layer, tag) in [("kda_L04_f32", 4usize, "L04"), ("mla_L03_f32", 3usize, "L03")] {
        run_f32_lane(&mut r, &l, &ckpt, &dims, &dev, lane, layer, tag);
    }
    run_f64_lane(&mut r, &l, &ckpt, &dims, &dev, "mla_L03_f64pure", 3, "L03");

    // ------------------------------------------------------------------- done
    println!("\n== summary ==");
    println!(
        "  oracle arrays read, shape/dtype-verified against the inventory: {} prefix13 + {} ladder",
        p.seen.borrow().len(),
        l.seen.borrow().len()
    );
    let fails = r.failures();
    println!("  checks: {} run, {} failed", r.checks.len(), fails.len());
    for f in &fails {
        println!("    FAIL {} — {} :: {}", f.id, f.what, f.detail);
    }
    if !r.skipped.is_empty() {
        println!("  PARTIAL RUN — {} lane(s) skipped:", r.skipped.len());
        for s in &r.skipped {
            println!("    SKIPPED: {s}");
        }
    }
    println!("  elapsed {:.1}s", t0.elapsed().as_secs_f64());
    if fails.is_empty() && !r.checks.is_empty() {
        if r.skipped.is_empty() {
            println!("\nGATE: PASS ({} checks)", r.checks.len());
        } else {
            println!(
                "\nGATE: PASS on a PARTIAL run ({} checks, {} lanes skipped) — NOT the \
                 authoritative result",
                r.checks.len(),
                r.skipped.len()
            );
        }
    } else {
        println!("\nGATE: FAIL ({} of {} checks)", fails.len(), r.checks.len());
        std::process::exit(1);
    }
}

/// Two top-k lists aligned by expert id — torch's order is unspecified, so an
/// elementwise comparison would be testing an implementation detail.
struct Aligned {
    n_set_diff: usize,
    mine_w: Vec<f32>,
    mine_pre: Vec<f32>,
    ref_w: Vec<f32>,
    ref_pre: Vec<f32>,
}

#[allow(clippy::too_many_arguments)]
fn align_topk(
    mine_idx: &[usize],
    mine_w: &[f32],
    mine_pre: &[f32],
    ref_idx: &[usize],
    ref_w: &[f32],
    ref_pre: &[f32],
    ntok: usize,
    k: usize,
) -> Aligned {
    assert_eq!(mine_idx.len(), ntok * k, "top-k index length");
    assert_eq!(ref_idx.len(), ntok * k, "reference top-k index length");
    let mut out = Aligned {
        n_set_diff: 0,
        mine_w: vec![0f32; ntok * k],
        mine_pre: vec![0f32; ntok * k],
        ref_w: vec![0f32; ntok * k],
        ref_pre: vec![0f32; ntok * k],
    };
    for t in 0..ntok {
        let mut a: Vec<(usize, f32, f32)> = (0..k)
            .map(|j| (mine_idx[t * k + j], mine_w[t * k + j], mine_pre[t * k + j]))
            .collect();
        let mut b: Vec<(usize, f32, f32)> = (0..k)
            .map(|j| (ref_idx[t * k + j], ref_w[t * k + j], ref_pre[t * k + j]))
            .collect();
        a.sort_by_key(|q| q.0);
        b.sort_by_key(|q| q.0);
        if a.iter().map(|q| q.0).ne(b.iter().map(|q| q.0)) {
            out.n_set_diff += 1;
        }
        for j in 0..k {
            out.mine_w[t * k + j] = a[j].1;
            out.mine_pre[t * k + j] = a[j].2;
            out.ref_w[t * k + j] = b[j].1;
            out.ref_pre[t * k + j] = b[j].2;
        }
    }
    out
}

/// The 32-token bf16 lane at one layer: the routed combination over every
/// expert the router really selected, and the whole block end to end.
fn run_bf16_block(
    r: &mut Report,
    p: &Oracle,
    ckpt: &Ckpt,
    dims: &MoeDims,
    dev: &Dev,
    layer: usize,
    tag: &str,
) {
    let moe = LatentMoe::new(dims.clone());
    let hin = p.get(&format!("{tag}_moe_in_bf16bits"));
    let ntok = hin.shape[0] * hin.shape[1];
    let h = t2(hin.bf16_as_f32(), [ntok, dims.hidden_size], dev);
    let w = ckpt.block_weights(layer, true, dev);

    let mut fetched: Vec<usize> = Vec::new();
    let tr = moe.forward_traced(h.clone(), &w, |id| {
        fetched.push(id);
        ckpt.expert(layer, id, dev)
    });

    let meta: Vec<usize> = p
        .get(&format!("meta_expert_ids_{tag}"))
        .i64s()
        .into_iter()
        .map(|v| v as usize)
        .collect();
    let mine: BTreeSet<usize> = fetched.iter().copied().collect();
    let theirs: BTreeSet<usize> = meta.iter().copied().collect();
    r.boolean(
        "G1",
        "the experts we fetch are EXACTLY the ones the shipped run materialised — no more \
         (a dense port would touch all 896) and no fewer",
        mine == theirs && fetched.len() == mine.len() && !mine.is_empty(),
        format!(
            "fetched {} distinct in {} calls; the shipped run materialised {} of {}",
            mine.len(),
            fetched.len(),
            theirs.len(),
            dims.num_experts
        ),
    );

    let lni = p.get(&format!("{tag}_moe_latent_norm_in_bf16bits")).bf16_as_f32();
    r.close(
        "G2",
        "moe_infer: the top-16 combination over every selected expert == the shipped \
         combination (32 tokens, real routing, real MXFP4 weights)",
        &vec_of(tr.combined.clone()),
        &lni,
        bf16_budget(4, absmax(&lni)),
        None,
    );

    let moe_out = p.get(&format!("{tag}_moe_out_bf16bits")).bf16_as_f32();
    r.close(
        "W1",
        "the WHOLE block, moe_in -> moe_out, 32 tokens",
        &vec_of(tr.out.clone()),
        &moe_out,
        bf16_budget(8, absmax(&moe_out)),
        None,
    );

    // C10 as an identity on the oracle's own arrays: the shared experts are
    // added in the ORIGINAL hidden space, and the bf16 add rounds once.
    let luo = p.get(&format!("{tag}_moe_latent_up_out_bf16bits")).bf16_as_f32();
    let sho = p.get(&format!("{tag}_moe_shared_out_bf16bits")).bf16_as_f32();
    let sum: Vec<f32> = luo
        .iter()
        .zip(sho.iter())
        .map(|(&a, &b)| half::bf16::from_f32(a + b).to_f32())
        .collect();
    r.exact(
        "W2",
        "C10 on the oracle's own arrays: moe_out == round_bf16(latent_up_out + shared_out)",
        &sum,
        &moe_out,
    );

    // W4 pins the LAST line's rounding, driven from two captured tensors so
    // there is no GEMM noise between the claim and the check: the shipped add
    // is `bf16 + bf16 -> bf16`, which rounds once. An unrounded add reproduces
    // W2's arithmetic and fails this by exactly the missing rounding — which is
    // the whole difference between a port that stores bf16 and one that only
    // computes in it.
    let luo_t = t2(luo.clone(), [ntok, dims.hidden_size], dev);
    let sho_t = t2(sho.clone(), [ntok, dims.hidden_size], dev);
    r.exact(
        "W4",
        "the block's final residual add, driven from the two captured tensors, is \
         bit-exactly the shipped moe_out — i.e. the add rounds to bf16",
        &vec_of(moe.combine_with_shared(luo_t, sho_t)),
        &moe_out,
    );

    // NEGATIVE: forget the shared experts.
    r.must_differ(
        "W3",
        "NEGATIVE: dropping the shared experts must not reproduce the block output",
        &vec_of(tr.latent_up_out.clone()),
        &moe_out,
        bf16_budget(8, absmax(&moe_out)),
    );
    // NEGATIVE: uniform combining weights.
    {
        let uniform = Routing::<B> {
            logits: tr.routing.logits.clone(),
            scores: tr.routing.scores.clone(),
            scores_for_choice: tr.routing.scores_for_choice.clone(),
            topk_idx: tr.routing.topk_idx.clone(),
            topk_weight_prerenorm: tr.routing.topk_weight_prerenorm.clone(),
            topk_weight: vec![1.0 / dims.top_k as f32; tr.routing.topk_idx.len()],
            tokens: tr.routing.tokens,
            top_k: tr.routing.top_k,
            min_topk_margin: tr.routing.min_topk_margin,
        };
        r.must_differ(
            "G3",
            "NEGATIVE: uniform 1/k combining weights must not reproduce the combination",
            &vec_of(moe.moe_infer(tr.latent_down_out.clone(), &uniform, |id| {
                ckpt.expert(layer, id, dev)
            })),
            &lni,
            bf16_budget(4, absmax(&lni)),
        );
    }
}

/// G4 — `moe_infer` accumulates the top-k combination in fp32 and rounds to the
/// activation dtype **once, at the end**.
///
/// This is the one claim in the block that the captured vectors cannot settle
/// on their own: rounding after every expert instead of once at the end moves
/// the 32-token result by well under the reference kernel's own GEMM noise, so
/// both survive a comparison against `latent_norm_in`. (Measured: max |d| goes
/// from 6.10e-4 to 9.77e-4 against a budget of 3.39e-3 — the wrong version
/// passes.) It is a READ claim, from `moe_infer`'s
/// `.type(topk_weight.dtype).mul_().sum(dim=1).type(new_x.dtype)`, and this
/// mechanises it: a small block whose per-expert outputs are taken from the
/// port itself, accumulated on the host in f32 in the same order, rounded once,
/// must reproduce `moe_infer` **bit-for-bit**.
///
/// The check carries its own discriminator: the round-after-every-add variant
/// is computed too and is REQUIRED to differ. A criterion that both variants
/// satisfy would prove nothing, so it fails rather than passes.
fn check_combination_rounds_once(r: &mut Report, real: &MoeDims, dev: &Dev) {
    let hm = 64usize;
    let fw = 64usize;
    let tokens = 5usize;
    let k = 3usize;
    let n_experts = 4usize;
    let dims = MoeDims {
        hidden_size: 32,
        moe_hidden_size: hm,
        moe_intermediate_size: fw,
        shared_intermediate_size: None,
        num_experts: n_experts,
        top_k: k,
        moe_renormalize: true,
        routed_scaling_factor: 1.0,
        latent_moe_use_norm: false,
        rms_norm_eps: real.rms_norm_eps,
        situ: real.situ,
    };
    let moe = LatentMoe::new(dims.clone());
    // Deterministic, sign-varied, and scaled so `situ` is in its live region.
    let gen = |n: usize, salt: usize| -> Vec<f32> {
        (0..n)
            .map(|i| {
                let z = ((i * 2654435761 + salt * 40503) % 4093) as f32 / 4093.0;
                (z - 0.5) * 0.5
            })
            .collect()
    };
    let experts: Vec<ExpertWeights<B>> = (0..n_experts)
        .map(|e| ExpertWeights {
            w1: t2(gen(fw * hm, e * 3 + 1), [fw, hm], dev),
            w2: t2(gen(hm * fw, e * 3 + 2), [hm, fw], dev),
            w3: t2(gen(fw * hm, e * 3 + 3), [fw, hm], dev),
        })
        .collect();
    let latent = t2(gen(tokens * hm, 97), [tokens, hm], dev);
    let mut topk_idx = Vec::new();
    let mut topk_weight = Vec::new();
    for t in 0..tokens {
        for j in 0..k {
            topk_idx.push((t + j) % n_experts);
            topk_weight.push(0.1 + 0.3 * ((t * k + j) % 5) as f32);
        }
    }
    let zeros = t2(vec![0f32; tokens * n_experts], [tokens, n_experts], dev);
    let routing = Routing::<B> {
        logits: zeros.clone(),
        scores: zeros.clone(),
        scores_for_choice: zeros,
        topk_idx: topk_idx.clone(),
        topk_weight_prerenorm: topk_weight.clone(),
        topk_weight: topk_weight.clone(),
        tokens,
        top_k: k,
        min_topk_margin: 1.0,
    };
    let got = vec_of(moe.moe_infer(latent.clone(), &routing, |id| experts[id].clone()));

    // The same weighted sum, from the port's OWN per-expert outputs, on the
    // host — once in the shipped order (fp32 all the way, one rounding at the
    // end) and once rounding after every expert.
    let mut once = vec![0f32; tokens * hm];
    let mut each = vec![0f32; tokens * hm];
    for id in routing.touched_experts() {
        let (toks, ws) = routing.tokens_for(id);
        let sel = Tensor::<B, 1, burn::prelude::Int>::from_data(
            TensorData::new(
                toks.iter().map(|&t| t as i64).collect::<Vec<_>>(),
                [toks.len()],
            ),
            dev,
        );
        let block = latent.clone().select(0, sel);
        let out = vec_of(moe.expert_traced(block, &experts[id]).out);
        for (j, &t) in toks.iter().enumerate() {
            for c in 0..hm {
                once[t * hm + c] += out[j * hm + c] * ws[j];
                each[t * hm + c] =
                    half::bf16::from_f32(each[t * hm + c] + out[j * hm + c] * ws[j]).to_f32();
            }
        }
    }
    let once: Vec<f32> = once
        .into_iter()
        .map(|v| half::bf16::from_f32(v).to_f32())
        .collect();
    let differ = once
        .iter()
        .zip(each.iter())
        .filter(|(a, b)| a.to_bits() != b.to_bits())
        .count();
    r.boolean(
        "G4a",
        "the two accumulation rules are DISTINGUISHABLE on this block — round-once and \
         round-every-expert give different bf16 answers, so the check below can fail",
        differ > 0,
        format!("{differ} of {} elements differ between the two rules", once.len()),
    );
    r.exact(
        "G4b",
        "moe_infer accumulates the top-k combination in fp32 and rounds ONCE at the end \
         (the shipped `.mul_().sum(dim=1).type(dtype)`), bit-for-bit",
        &got,
        &once,
    );
}

/// One f32 ladder lane (4 tokens). Every weight is exact in f32 — MXFP4 values
/// are `code · 2^e` and the bf16 tensors widen losslessly — so the only source
/// of error is f32 accumulation. This is the sharpest lane available.
#[allow(clippy::too_many_arguments)]
fn run_f32_lane(
    r: &mut Report,
    l: &Oracle,
    ckpt: &Ckpt,
    dims: &MoeDims,
    dev: &Dev,
    lane: &str,
    layer: usize,
    tag: &str,
) {
    let moe = LatentMoe::new_f32(dims.clone());
    let hin = l.get(&format!("{lane}__{tag}_moe_in"));
    let ntok: usize = hin.len() / dims.hidden_size;
    let h = t2(hin.f32s(), [ntok, dims.hidden_size], dev);

    // In the f32 lane the router bias is the checkpoint's F32, NOT a bf16 cast.
    let pre = format!("language_model.model.layers.{layer}.block_sparse_moe");
    let (_, bias_f32) = ckpt.f32(&format!("{pre}.gate.e_score_correction_bias"));
    let bias_lane = l
        .get(&format!("{lane}__{tag}_moe_router_e_score_correction_bias"))
        .f32s();
    r.exact(
        &format!("R11/{lane}"),
        "the f32 lane's router bias IS the checkpoint's F32 tensor, unrounded — while the \
         bf16 lane's is a bf16 cast of the SAME tensor. Both are measured; a port must \
         pick per model dtype, not per habit",
        &bias_lane,
        &bias_f32,
    );

    let w = ckpt.block_weights(layer, false, dev);
    let tr = moe.forward_traced(h, &w, |id| ckpt.expert(layer, id, dev));

    let ldo = l.get(&format!("{lane}__{tag}_moe_latent_down_out")).f32s();
    let lni = l.get(&format!("{lane}__{tag}_moe_latent_norm_in")).f32s();
    let lno = l.get(&format!("{lane}__{tag}_moe_latent_norm_out")).f32s();
    let luo = l.get(&format!("{lane}__{tag}_moe_latent_up_out")).f32s();
    let sho = l.get(&format!("{lane}__{tag}_moe_shared_out")).f32s();
    let out = l.get(&format!("{lane}__{tag}_moe_out")).f32s();
    r.close(
        &format!("F1/{lane}"),
        "f32 lane: routed_expert_down_proj",
        &vec_of(tr.latent_down_out.clone()),
        &ldo,
        f32_budget(1, absmax(&ldo)),
        None,
    );
    r.close(
        &format!("F2/{lane}"),
        "f32 lane: the routed-expert combination over the real expert set",
        &vec_of(tr.combined.clone()),
        &lni,
        f32_budget(4, absmax(&lni)),
        None,
    );
    r.close(
        &format!("F3/{lane}"),
        "f32 lane: routed_expert_norm",
        &vec_of(tr.normed.clone()),
        &lno,
        f32_budget(4, absmax(&lno)),
        None,
    );
    r.close(
        &format!("F4/{lane}"),
        "f32 lane: routed_expert_up_proj",
        &vec_of(tr.latent_up_out.clone()),
        &luo,
        f32_budget(5, absmax(&luo)),
        None,
    );
    r.close(
        &format!("F5/{lane}"),
        "f32 lane: the shared experts",
        &vec_of(tr.shared_out.clone().unwrap()),
        &sho,
        f32_budget(3, absmax(&sho)),
        None,
    );
    r.close(
        &format!("F6/{lane}"),
        "f32 lane: the WHOLE block",
        &vec_of(tr.out.clone()),
        &out,
        f32_budget(8, absmax(&out)),
        None,
    );
    r.boolean(
        &format!("F7/{lane}"),
        "structural: routed_expert_norm is moe_hidden-wide, so it CANNOT sit after the \
         up-projection — that ordering is forced by the shapes, not chosen",
        dims.moe_hidden_size != dims.hidden_size && tr.latent_up_out.dims()[1] == dims.hidden_size,
        format!(
            "norm width {} vs up_proj output width {}",
            dims.moe_hidden_size,
            tr.latent_up_out.dims()[1]
        ),
    );
}

/// The float64 lane at L03: the reference ran with f64 parameters, f64
/// activations, and the shipped fp32 islands widened to f64. Comparing an f32
/// port against it at an f32-roundoff budget asks whether the port computes the
/// right FUNCTION, not merely the same f32 arithmetic.
#[allow(clippy::too_many_arguments)]
fn run_f64_lane(
    r: &mut Report,
    l: &Oracle,
    ckpt: &Ckpt,
    dims: &MoeDims,
    dev: &Dev,
    lane: &str,
    layer: usize,
    tag: &str,
) {
    let moe = LatentMoe::new_f32(dims.clone());
    let hin = l.get(&format!("{lane}__{tag}_moe_in"));
    let ntok: usize = hin.len() / dims.hidden_size;
    let h = t2(hin.as_f32(), [ntok, dims.hidden_size], dev);
    let w = ckpt.block_weights(layer, false, dev);
    let tr = moe.forward_traced(h, &w, |id| ckpt.expert(layer, id, dev));
    let lni = l.get(&format!("{lane}__{tag}_moe_latent_norm_in")).as_f32();
    let out = l.get(&format!("{lane}__{tag}_moe_out")).as_f32();
    r.close(
        &format!("D1/{lane}"),
        "float64 reference: the routed-expert combination",
        &vec_of(tr.combined.clone()),
        &lni,
        f32_budget(4, absmax(&lni)),
        None,
    );
    r.close(
        &format!("D2/{lane}"),
        "float64 reference: the WHOLE block",
        &vec_of(tr.out.clone()),
        &out,
        f32_budget(8, absmax(&out)),
        None,
    );
}
