//! `nvfp4_mma_probe` — can **CubeCL** emit the NVFP4 (E4M3 scales, one per 16)
//! block-scaled FP4 tensor-core MMA, and does it compute the right answer on
//! **real Inkling expert weights**?
//!
//! ## What was already known before this binary
//!
//! * `mxmma_probe` proved CubeCL reaches the **MXFP4** variant
//!   (`ue8m0`, `scale_vec::2X`, one scale per 32).
//! * `mary::nn::mxfp4`'s module doc records a PTX/SASS-level measurement that
//!   the **NVFP4** variant (`ue4m3`, `scale_vec::4X`) assembles and executes on
//!   `sm_121a`, via hand-written inline PTX.
//!
//! Neither covers the combination this port actually needs: CubeCL emitting the
//! **NVFP4** instruction. CubeCL's own test suite hardcodes `type S = ue8m0` in
//! `test_cmma_scaled_fp4`, so the `e4m3` / `scales_factor = 4` row of
//! `supported_scaled_mma_combinations` is *declared but untested upstream*.
//! This binary tests it, and does so on bytes read out of the real checkpoint
//! rather than synthetic codes, so the E4M3 scale decode is exercised on the
//! scale distribution the model actually contains.
//!
//! ## What it checks
//!
//! A 16x8xK matmul where **both** operands are real NVFP4 expert rows from
//! `model.llm.layers.10.mlp.experts.w13_weight` — packed E2M1 nibbles, real
//! E4M3 per-16 block scales, real per-expert F32 `scale2` — accumulated over
//! the full K of the checkpoint, against a CPU f32 reference that decodes the
//! same bytes with [`mary::models::inkling::nvfp4`]'s audited decode.
//!
//! Using real rows for the A operand as well as B is deliberate: synthetic
//! scales would likely be a handful of round values, and the E4M3 decode is
//! exactly the part most likely to be wrong. Real rows bring real scale bytes.
//!
//! Build: `--features cuda-backend,inkling`
//! Run:   `nvfp4_mma_probe [<checkpoint dir>]`

use std::fs::File;
use std::io::{Read, Seek, SeekFrom};
use std::path::{Path, PathBuf};

use anyhow::{bail, Context, Result};

use cubecl::cuda::CudaRuntime;
use cubecl::ir::MatrixIdent;
use cubecl::prelude::*;
use cubecl::{e2m1x2, e4m3};

use mary::models::inkling::nvfp4::{e4m3_to_f32, FP4_E2M1, GROUP};

/// NVFP4 block-scaled MMA, m16n8k64, accumulating over `k_tiles` tiles of k=64.
///
/// `scales_factor = 4` at k=64 is one scale per 16 elements — NVFP4.
#[cube(launch)]
pub fn nv_mma<AB: Scalar, S: Scalar, NA: Size, NC: Size>(
    a: &Tensor<Vector<AB, NA>>,
    b: &Tensor<Vector<AB, NA>>,
    scales_a: &Tensor<S>,
    scales_b: &Tensor<S>,
    out: &mut Tensor<Vector<f32, NC>>,
    #[comptime] size_k: usize,
    #[comptime] size_n: usize,
    #[comptime] scales_factor: usize,
) {
    let def =
        cmma::MmaDefinition::<AB, AB, f32>::new_scaled::<S>(16usize, 8usize, 64usize, scales_factor);
    let lane = UNIT_POS_PLANE;
    let pack = AB::packing_factor();

    let ec_a = def.elems_per_lane(MatrixIdent::A);
    let vs_a = def.vector_size(MatrixIdent::A);
    let vc_a = comptime!(ec_a / vs_a);
    let ec_b = def.elems_per_lane(MatrixIdent::B);
    let vs_b = def.vector_size(MatrixIdent::B);
    let vc_b = comptime!(ec_b / vs_b);
    let ec_c = def.elems_per_lane(MatrixIdent::Accumulator);
    let vs_c = def.vector_size(MatrixIdent::Accumulator);
    let vc_c = comptime!(ec_c / vs_c);

    let mut reg_a = Array::<Vector<AB, NA>>::new(vc_a);
    let mut reg_b = Array::<Vector<AB, NA>>::new(vc_b);
    let mut acc = Array::<Vector<f32, NC>>::new(vc_c);
    #[unroll]
    for i in 0..vc_c {
        acc[i] = Vector::<f32, NC>::cast_from(0.0f32);
    }

    let scales_count = def.scales_count();
    let size!(NS) = def.scales_vector_size();
    let sia = def.scales_index(lane, MatrixIdent::A) as usize;
    let sib = def.scales_index(lane, MatrixIdent::B) as usize;
    // Scales per row across the whole K, not just this tile.
    let spr = comptime!(size_k / 16);
    let k_tiles = comptime!(size_k / 64);

    for t in 0..k_tiles {
        let kbase = t * 64;
        #[unroll]
        for i in 0..vc_a {
            let (row, col) = def.position_of_nth(lane, (i * vs_a * pack) as u32, MatrixIdent::A);
            let gcol = col as usize + kbase;
            reg_a[i] = a[(row as usize * size_k / 2 + gcol / 2) / a.vector_size()];
        }
        #[unroll]
        for i in 0..vc_b {
            let (row, col) = def.position_of_nth(lane, (i * vs_b * pack) as u32, MatrixIdent::B);
            // B is column-major w.r.t. the tile: `col` indexes n, `row` indexes k.
            let grow = row as usize + kbase;
            reg_b[i] = b[(col as usize * size_k / 2 + grow / 2) / b.vector_size()];
        }

        let mut sa = Vector::<S, NS>::empty();
        let mut sb = Vector::<S, NS>::empty();
        #[unroll]
        for i in 0..scales_count {
            sa[i] = scales_a[sia * spr + t * scales_factor + i];
            sb[i] = scales_b[sib * spr + t * scales_factor + i];
        }

        let d = def.execute_scaled(&reg_a, &reg_b, &acc, sa, sb);
        #[unroll]
        for i in 0..vc_c {
            acc[i] = d[i];
        }
    }

    #[unroll]
    for i in 0..vc_c {
        let (row, col) = def.position_of_nth(lane, (i * vs_c) as u32, MatrixIdent::Accumulator);
        out[(row as usize * size_n + col as usize) / out.vector_size()] = acc[i];
    }
}

// ---------------------------------------------------------------------------
// safetensors: header parse + positioned reads. The shards are gigabytes; only
// the handful of rows this probe touches are ever read.
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

    /// Read `len` bytes at `offset` bytes into the tensor's own data region.
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

/// One expert row, decoded to f32 by the audited CPU path.
struct Row {
    packed: Vec<u8>,
    scales: Vec<u8>,
}

impl Row {
    fn decode(&self, k: usize, scale2: f32) -> Vec<f32> {
        (0..k)
            .map(|j| {
                let byte = self.packed[j / 2];
                // Low nibble first (settled against compressed_tensors).
                let code = if j % 2 == 0 { byte & 0x0F } else { byte >> 4 };
                FP4_E2M1[code as usize] * e4m3_to_f32(self.scales[j / GROUP]) * scale2
            })
            .collect()
    }
}

const WEIGHT: &str = "model.llm.layers.10.mlp.experts.w13_weight";

fn load_rows(dir: &Path, expert: usize, first_row: usize, n: usize, k: usize) -> Result<(Vec<Row>, f32)> {
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
        let gr = expert * rows_per_expert + first_row + r;
        let packed = ws.read_at(WEIGHT, (gr * bytes_per_row) as u64, k / 2)?;
        let scales = ss.read_at(
            &format!("{WEIGHT}.scale"),
            (gr * scales_per_row) as u64,
            k / GROUP,
        )?;
        out.push(Row { packed, scales });
    }
    Ok((out, scale2))
}

fn main() -> Result<()> {
    let dir = std::env::args()
        .nth(1)
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("models/thinkingmachines-inkling-small-nvfp4"));

    let (m, n, sf) = (16usize, 8usize, 4usize);
    // Full K of the checkpoint's w13 expert rows (2048 bytes = 4096 codes).
    let k = 4096usize;

    let client = CudaRuntime::client(&Default::default());

    // --- is the NVFP4 combination even registered? ------------------------
    // Matched by inspection rather than by constructing a ScaledMmaConfig, so
    // this does not depend on the exact public path of that type.
    let props = client.properties();
    let registered = props.features.matmul.scaled_mma.iter().any(|c| {
        c.a_type == e2m1x2::cube_type()
            && c.b_type == e2m1x2::cube_type()
            && c.cd_type == f32::cube_type()
            && c.scales_type == e4m3::cube_type()
            && c.m == m as u32
            && c.n == n as u32
            && c.k == 64
            && c.scales_factor == sf as u32
    });
    println!(
        "CubeCL reports the NVFP4 combination (e2m1x2 x e2m1x2 -> f32, e4m3 scales, \
         m16n8k64, scales_factor 4): {}",
        if registered { "REGISTERED" } else { "NOT registered" }
    );
    if !registered {
        bail!("CubeCL does not advertise the NVFP4 scaled-MMA combination on this device");
    }

    // --- real expert rows -------------------------------------------------
    // A: rows 0..16 of expert 0. B: rows 64..72 of expert 0. Both are genuine
    // NVFP4 rows with genuine E4M3 block scales.
    let (a_rows, scale2) = load_rows(&dir, 0, 0, m, k)?;
    let (b_rows, scale2_b) = load_rows(&dir, 0, 64, n, k)?;
    println!("loaded {m}x{k} A rows and {n}x{k} B rows from {WEIGHT} (expert 0), scale2 = {scale2:e}");

    // Host-side reference in f32, from the audited decode.
    let a_ref: Vec<Vec<f32>> = a_rows.iter().map(|r| r.decode(k, scale2)).collect();
    let b_ref: Vec<Vec<f32>> = b_rows.iter().map(|r| r.decode(k, scale2_b)).collect();

    // --- device buffers ---------------------------------------------------
    // The MMA consumes raw codes and raw E4M3 scale bytes; the two per-expert
    // F32 scale2 factors are scalars and fold in afterwards, exactly as a real
    // kernel would apply them.
    let a_packed: Vec<u8> = a_rows.iter().flat_map(|r| r.packed.clone()).collect();
    let b_packed: Vec<u8> = b_rows.iter().flat_map(|r| r.packed.clone()).collect();
    let a_scales: Vec<u8> = a_rows.iter().flat_map(|r| r.scales.clone()).collect();
    let b_scales: Vec<u8> = b_rows.iter().flat_map(|r| r.scales.clone()).collect();

    let ah = client.create_from_slice(&a_packed);
    let bh = client.create_from_slice(&b_packed);
    let sah = client.create_from_slice(&a_scales);
    let sbh = client.create_from_slice(&b_scales);
    let oh = client.create_from_slice(f32::as_bytes(&vec![0.0f32; m * n]));

    let cd = CubeDim::new_1d(client.properties().hardware.plane_size_max);
    let vs = 32 / e2m1x2::cube_type().size_bits();
    let spr = k / GROUP;

    unsafe {
        nv_mma::launch::<e2m1x2, e4m3, CudaRuntime>(
            &client,
            CubeCount::Static(1, 1, 1),
            cd,
            vs,
            2,
            TensorArg::from_raw_parts(ah.clone(), [k / 2, 1].into(), [m, k / 2].into()),
            TensorArg::from_raw_parts(bh.clone(), [k / 2, 1].into(), [n, k / 2].into()),
            TensorArg::from_raw_parts(sah.clone(), [spr, 1].into(), [m, spr].into()),
            TensorArg::from_raw_parts(sbh.clone(), [spr, 1].into(), [n, spr].into()),
            TensorArg::from_raw_parts(oh.clone(), [n, 1].into(), [m, n].into()),
            k,
            n,
            sf,
        )
    };
    let got = f32::from_bytes(&client.read_one(oh).expect("read")).to_vec();

    // --- correctness ------------------------------------------------------
    // The MMA sees codes and block scales only, so its result must be scaled by
    // the two global scale2 factors to be comparable with the decoded reference.
    let global = scale2 * scale2_b;
    let mut worst = 0.0f32;
    let mut worst_at = (0usize, 0usize);
    for i in 0..m {
        for j in 0..n {
            let mut s = 0.0f32;
            for l in 0..k {
                s += a_ref[i][l] * b_ref[j][l];
            }
            let g = got[i * n + j] * global;
            if !g.is_finite() {
                println!("NON-FINITE at ({i},{j}) — FAIL");
                std::process::exit(1);
            }
            let r = (g - s).abs() / s.abs().max(1e-12);
            if r > worst {
                worst = r;
                worst_at = (i, j);
            }
        }
    }
    println!(
        "max relative error {worst:.3e} at {worst_at:?}  (K = {k}, {} MMA tiles accumulated)",
        k / 64
    );

    // f32 accumulation over 4096 terms in a different order than the CPU sums
    // them; a few ulp of drift per term is expected and is not a wrong answer.
    if worst > 1e-4 {
        println!("FAIL — CubeCL's NVFP4 MMA did not reproduce the reference");
        std::process::exit(1);
    }
    println!(
        "PASS — CubeCL emits and executes the NVFP4 (ue4m3, 1 scale / 16) tensor-core MMA, \
         correct on real Inkling expert weights"
    );
    Ok(())
}
