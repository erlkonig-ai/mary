//! A real-valued EMA teacher for the last routed layer.
//!
//! The student and teacher share frozen trunk weights, NOT sequence state.
//! This bank owns an independent packed NVFP4 arena and F32 shadows of both
//! W13 and W2. Shadows hold decoded weights including E4M3 and scale2, never
//! code indices. Every declared update visits every expert, including experts
//! inactive on the latest step, and refreshes teacher codes against the fixed
//! scales using deterministic nearest rounding (ties toward smaller magnitude).
//! Sub-code updates survive in the F32 shadow rather than disappearing at each
//! requantization. No student or static frozen-control byte is written.
//!
//! Initialization, student learning, teacher reads, and `advance` must use the
//! SAME ordered client stream. Versions describe enqueued state, not a device
//! completion fence. The owner must not retain teacher context histories across
//! a version change without rebuilding them, or run concurrent arena writers.
//! This module does not publish weights or claim resumable EMA persistence.

use anyhow::{Context, Result};
use cubecl::prelude::*;
use cubecl::server::Handle;

use super::{e2m1_code, e4m3_bits_to_f32};
use super::super::devplan::ExpertTable;
use super::super::fp4quant::e2m1_bits;

type Client = ComputeClient<cubecl::cuda::CudaRuntime>;
const CUBE: u32 = 256;

/// Additional device storage, excluding shared student scale2 handles and
/// allocator/workspace overhead. Calculable before any device allocation.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct EmaBytes {
    pub packed: usize,
    pub shadow: usize,
    pub offsets: usize,
    pub total: usize,
}

#[derive(Clone, Copy, Debug)]
struct Layout {
    values13: usize,
    values2: usize,
    expert_bytes: usize,
    bytes: EmaBytes,
}

fn product(a: usize, b: usize) -> Result<usize> {
    a.checked_mul(b).context("EMA byte/shape count overflow")
}

fn sum(a: usize, b: usize) -> Result<usize> {
    a.checked_add(b).context("EMA byte/shape count overflow")
}

impl Layout {
    fn new(n_routed: usize, hidden: usize, local_inter: usize) -> Result<Self> {
        anyhow::ensure!(n_routed > 0 && n_routed <= u16::MAX as usize, "EMA needs 1..=65535 experts");
        anyhow::ensure!(
            hidden > 0 && local_inter > 0 && hidden % 64 == 0 && local_inter % 64 == 0,
            "EMA requires nonzero 64-aligned hidden and rank-local intermediate widths"
        );
        let values2 = product(hidden, local_inter)?;
        let values13 = product(values2, 2)?;
        let values = sum(values13, values2)?;
        let expert_bytes = sum(values / 2, values / 16)?;
        let packed = product(expert_bytes, n_routed)?;
        let shadow = product(product(values, n_routed)?, 4)?;
        let offsets = product(n_routed, 4 * 8)?;
        let total = sum(sum(packed, shadow)?, offsets)?;
        let max_words = values13 / 8;
        anyhow::ensure!(max_words.div_ceil(CUBE as usize) <= i32::MAX as usize, "EMA launch grid is too wide");
        Ok(Self { values13, values2, expert_bytes, bytes: EmaBytes { packed, shadow, offsets, total } })
    }

    fn offsets(self, n_routed: usize) -> (Vec<u64>, Vec<u64>) {
        let mut off13 = Vec::with_capacity(n_routed * 2);
        let mut off2 = Vec::with_capacity(n_routed * 2);
        for e in 0..n_routed {
            let base = e * self.expert_bytes;
            let code2 = base + self.values13 / 2 + self.values13 / 16;
            off13.extend([base as u64, (base + self.values13 / 2) as u64]);
            off2.extend([code2 as u64, (code2 + self.values2 / 2) as u64]);
        }
        (off13, off2)
    }
}

pub fn memory_bytes(n_routed: usize, hidden: usize, local_inter: usize) -> Result<EmaBytes> {
    Ok(Layout::new(n_routed, hidden, local_inter)?.bytes)
}

fn validate_beta(beta: f32) -> Result<()> {
    anyhow::ensure!(beta.is_finite() && (0.0..1.0).contains(&beta), "EMA beta must be finite in [0, 1)");
    Ok(())
}

fn next_version(version: u64, expected: u64, last_student: u64, student: u64) -> Result<u64> {
    anyhow::ensure!(version == expected, "EMA teacher version {version} differs from expected {expected}");
    anyhow::ensure!(student > last_student, "EMA requires a newer student step than {last_student}, got {student}");
    version.checked_add(1).context("EMA teacher version exhausted")
}

fn table_clone(t: &ExpertTable) -> ExpertTable {
    ExpertTable {
        off13: t.off13.clone(), off2: t.off2.clone(), sc13: t.sc13.clone(), sc2: t.sc2.clone(),
        wmap: t.wmap.clone(), wmap_bytes: t.wmap_bytes, expert_bytes: t.expert_bytes,
        n_routed: t.n_routed, stride: t.stride, scaled: t.scaled,
    }
}

fn u64_bytes(values: &[u64]) -> Vec<u8> {
    values.iter().flat_map(|v| v.to_le_bytes()).collect()
}

/// The teacher's owner. Clone only its table for a forward override; table
/// clones alias this one bank and therefore are NOT version snapshots.
pub struct EmaBank {
    teacher: ExpertTable,
    student: ExpertTable,
    shadow13: Handle,
    shadow2: Handle,
    layout: Layout,
    beta: f32,
    version: u64,
    student_step: u64,
    poisoned: bool,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct EmaReport {
    pub version: u64,
    pub student_step: u64,
    pub beta: f32,
    pub bytes: EmaBytes,
    /// An enqueue panicked after entering mutation. This is not recoverable by
    /// retrying at the old version; the owner must discard the bank/session.
    pub poisoned: bool,
}

impl EmaBank {
    /// Copy this rank's complete layer into a separate teacher bank, decoding
    /// its F32 shadows on device. `local_inter` is the TP-sharded intermediate
    /// width; every routed expert is present. The source table must describe
    /// the existing swizzled NVFP4 arena with immutable scales and scale2.
    /// Host shape/beta/table checks all precede allocation or launch.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        client: &Client,
        student: &ExpertTable,
        hidden: usize,
        local_inter: usize,
        swizzled: bool,
        beta: f32,
        initial_student_step: u64,
    ) -> Result<Self> {
        validate_beta(beta)?;
        let layout = Layout::new(student.n_routed, hidden, local_inter)?;
        anyhow::ensure!(swizzled && student.scaled && student.stride == 2, "EMA requires a swizzled NVFP4 expert table");
        anyhow::ensure!(student.expert_bytes == layout.expert_bytes, "EMA expert shape differs from the student table");
        anyhow::ensure!(student.wmap_bytes >= layout.bytes.packed && student.wmap_bytes % 4 == 0, "EMA student arena has an invalid size");
        let (off13, off2) = layout.offsets(student.n_routed);
        let teacher = ExpertTable {
            off13: client.create_from_slice(&u64_bytes(&off13)),
            off2: client.create_from_slice(&u64_bytes(&off2)),
            // Scale2 is immutable under both the learner and EMA; sharing it
            // does not share any mutable teacher/student parameter bytes.
            sc13: student.sc13.clone(), sc2: student.sc2.clone(),
            wmap: client.empty(layout.bytes.packed), wmap_bytes: layout.bytes.packed,
            expert_bytes: layout.expert_bytes, n_routed: student.n_routed, stride: 2, scaled: true,
        };
        let bank = Self {
            shadow13: client.empty(product(product(layout.values13, student.n_routed)?, 4)?),
            shadow2: client.empty(product(product(layout.values2, student.n_routed)?, 4)?),
            teacher, student: table_clone(student), layout, beta, version: 0,
            student_step: initial_student_step, poisoned: false,
        };
        bank.launch(client, true);
        Ok(bank)
    }

    pub fn teacher_table(&self) -> &ExpertTable {
        assert!(!self.poisoned, "an interrupted EMA update poisoned the teacher bank");
        &self.teacher
    }

    pub fn teacher_table_clone(&self) -> ExpertTable { table_clone(self.teacher_table()) }

    pub fn report(&self) -> EmaReport {
        EmaReport { version: self.version, student_step: self.student_step, beta: self.beta, bytes: self.layout.bytes, poisoned: self.poisoned }
    }

    /// Nonmutating host preflight, for agreement before a TP broadcast.
    pub fn validate_advance(&self, expected_version: u64, new_student_step: u64) -> Result<()> {
        anyhow::ensure!(!self.poisoned, "an interrupted EMA update poisoned the teacher bank");
        next_version(self.version, expected_version, self.student_step, new_student_step)?;
        Ok(())
    }

    /// One EMA update per declared boundary, NOT one per skipped student step.
    /// Teacher reads already queued finish first; subsequent reads see the
    /// refreshed bank. A failed version/step preflight enqueues nothing.
    pub fn advance(&mut self, client: &Client, expected_version: u64, new_student_step: u64) -> Result<EmaReport> {
        self.validate_advance(expected_version, new_student_step)?;
        let next = next_version(self.version, expected_version, self.student_step, new_student_step)?;
        self.poisoned = true;
        self.launch(client, false);
        self.version = next;
        self.student_step = new_student_step;
        self.poisoned = false;
        Ok(self.report())
    }

    fn launch(&self, client: &Client, initialize: bool) {
        for (source_off, target_off, scale2, shadow, values) in [
            (&self.student.off13, &self.teacher.off13, &self.student.sc13, &self.shadow13, self.layout.values13),
            (&self.student.off2, &self.teacher.off2, &self.student.sc2, &self.shadow2, self.layout.values2),
        ] {
            // Offsets/lengths come from the admitted ExpertTable producer;
            // dynamic addresses cover the multi-GiB source and F32 shadows.
            unsafe {
                ema_plane::launch::<cubecl::cuda::CudaRuntime>(
                    client,
                    CubeCount::Static((values / 8).div_ceil(CUBE as usize) as u32, self.student.n_routed as u32, 1),
                    CubeDim::new_1d(CUBE), AddressType::U64,
                    ArrayArg::from_raw_parts(self.student.wmap.clone(), self.student.wmap_bytes / 4),
                    ArrayArg::from_raw_parts(source_off.clone(), 2 * self.student.n_routed),
                    ArrayArg::from_raw_parts(scale2.clone(), self.student.n_routed),
                    ArrayArg::from_raw_parts(self.teacher.wmap.clone(), self.teacher.wmap_bytes / 4),
                    ArrayArg::from_raw_parts(target_off.clone(), 2 * self.student.n_routed),
                    ArrayArg::from_raw_parts(shadow.clone(), values * self.student.n_routed),
                    self.beta, values, initialize,
                )
            };
        }
    }
}

/// One lane owns a whole code word (eight weights), so no two writers race on
/// nibbles. Shadows use that same physical word order. The two adjacent code
/// words of a 16-value block use its SAME scale byte; scale copying is instead
/// word-granular and disjoint from code writes.
#[cube(launch, address_type = "dynamic")]
#[allow(clippy::too_many_arguments)]
fn ema_plane(
    source: &Array<u32>,
    source_off: &Array<u64>,
    scale2: &Array<f32>,
    teacher: &mut Array<u32>,
    target_off: &Array<u64>,
    shadow: &mut Array<f32>,
    beta: f32,
    values: usize,
    #[comptime] initialize: bool,
) {
    let e = CUBE_POS_Y as usize;
    let wi = (CUBE_POS_X as usize) * (CUBE as usize) + UNIT_POS as usize;
    let words = values / 8;
    if wi >= words { terminate!(); }
    let src_code = usize::cast_from(source_off[2 * e]) / 4;
    let src_scale = usize::cast_from(source_off[2 * e + 1]);
    let dst_code = usize::cast_from(target_off[2 * e]) / 4;
    let dst_scale = usize::cast_from(target_off[2 * e + 1]) / 4;
    let word = source[src_code + wi];
    let local = wi % 64;
    let row = (local % 32) / 4;
    let kw = (local / 32) * 4 + local % 4;
    let sb = src_scale + (wi / 64) * 32 + row * 4 + kw / 2;
    let scale_code = (source[sb / 4] >> u32::cast_from((sb % 4) * 8)) & 255u32;
    let block_scale = e4m3_bits_to_f32(scale_code);
    let expert_scale = scale2[e];
    let scale = block_scale * expert_scale;
    let mut out = word;
    if comptime![initialize] {
        if wi < values / 64 {
            teacher[dst_scale + wi] = source[src_scale / 4 + wi];
        }
    } else if scale > 0.0 && beta != 0.0 {
        out = 0u32;
    }
    #[unroll]
    for j in 0..8usize {
        let index = (e * words + wi) * 8 + j;
        let code = (word >> u32::cast_from(j * 4)) & 15u32;
        // Match the stored decoder's multiplication order. The product of
        // scales is used only to encode back into the fixed code grid.
        let student = e2m1_bits(code) * block_scale * expert_scale;
        let mut value = student;
        if comptime![initialize] {
            // The initial packed bank is copied exactly, including signed zero.
        } else {
            if beta != 0.0 {
                value = beta * shadow[index] + (1.0f32 - beta) * student;
            }
            if scale > 0.0 && beta != 0.0 {
                let encoded = e2m1_code(value / scale, 0.0f32, false);
                out |= encoded << u32::cast_from(j * 4);
            }
        }
        shadow[index] = value;
    }
    teacher[dst_code + wi] = out;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bytes_price_this_ranks_intermediate_cut_before_allocation() {
        let full = memory_bytes(256, 4096, 2048).unwrap();
        let half = memory_bytes(256, 4096, 1024).unwrap();
        assert_eq!(full.packed, 3_623_878_656);
        assert_eq!(full.shadow, 25_769_803_776);
        assert_eq!(half.packed * 2, full.packed);
        assert_eq!(half.shadow * 2, full.shadow);
        assert_eq!(full.offsets, 8192);
        assert_eq!(full.total, full.packed + full.shadow + full.offsets);
        assert!(memory_bytes(0, 4096, 1024).is_err());
        assert!(memory_bytes(256, 4096, 1025).is_err());
        assert!(memory_bytes(256, usize::MAX - 63, 1024).is_err());
    }

    #[test]
    fn beta_and_epoch_refusals_are_host_preflight() {
        for b in [f32::NAN, f32::INFINITY, -0.1, 1.0] { assert!(validate_beta(b).is_err()); }
        assert!(validate_beta(0.0).is_ok());
        assert!(validate_beta(0.999).is_ok());
        assert_eq!(next_version(3, 3, 100, 104).unwrap(), 4);
        assert!(next_version(3, 2, 100, 101).is_err());
        assert!(next_version(3, 3, 100, 100).is_err());
        assert!(next_version(u64::MAX, u64::MAX, 100, 101).is_err());
    }

    #[test]
    fn compact_offsets_are_aligned_disjoint_and_complete() {
        let l = Layout::new(3, 64, 64).unwrap();
        let (a, b) = l.offsets(3);
        let mut cursor = 0usize;
        for e in 0..3 {
            assert_eq!(a[2 * e] as usize, cursor);
            cursor += l.values13 / 2;
            assert_eq!(a[2 * e + 1] as usize, cursor);
            cursor += l.values13 / 16;
            assert_eq!(b[2 * e] as usize, cursor);
            cursor += l.values2 / 2;
            assert_eq!(b[2 * e + 1] as usize, cursor);
            cursor += l.values2 / 16;
        }
        assert_eq!(cursor, l.bytes.packed);
        assert!(a.iter().chain(&b).all(|offset| offset % 16 == 0));
    }

    fn read(client: &Client, handle: &Handle) -> Vec<u8> {
        client.read_one(handle.clone()).expect("tiny EMA readback").to_vec()
    }

    fn floats(bytes: &[u8]) -> Vec<f32> {
        bytes.chunks_exact(4).map(|b| f32::from_le_bytes(b.try_into().unwrap())).collect()
    }

    fn words(values: &[u32]) -> Vec<u8> {
        values.iter().flat_map(|v| v.to_le_bytes()).collect()
    }

    // Real swizzled weights with changing block scales, signed-zero codes,
    // non-unit scale2 and a source offset skew. No model/pile is involved.
    fn tiny_student(client: &Client) -> (ExpertTable, Vec<u8>) {
        use super::super::super::fp4gemm::{swizzle_b_codes, swizzle_b_scales};
        let layout = Layout::new(2, 64, 64).unwrap();
        let (mut off13, mut off2) = layout.offsets(2);
        for offset in off13.iter_mut().chain(&mut off2) { *offset += 32; }
        let mut arena = vec![0xa5; 32];
        for e in 0..2 {
            for (n, k, seed) in [(128, 64, e + 1), (64, 64, e + 7)] {
                let codes: Vec<u8> = (0..n * k / 2).map(|i| {
                    let a = ((i * 3 + seed) % 16) as u8;
                    let b = ((i * 7 + seed * 2) % 16) as u8;
                    a | (b << 4)
                }).collect();
                let scales: Vec<u8> = (0..n * k / 16).map(|i| {
                    [0, 1, 0x28, 0x31, 0x3b, 0x40, 0x4d][(i + seed) % 7]
                }).collect();
                arena.extend(swizzle_b_codes(&codes, n, k));
                arena.extend(swizzle_b_scales(&scales, n, k));
            }
        }
        let table = ExpertTable {
            off13: client.create_from_slice(&u64_bytes(&off13)),
            off2: client.create_from_slice(&u64_bytes(&off2)),
            sc13: client.create_from_slice(&[0.37f32, 1.25].into_iter().flat_map(f32::to_le_bytes).collect::<Vec<_>>()),
            sc2: client.create_from_slice(&[1.7f32, 0.75].into_iter().flat_map(f32::to_le_bytes).collect::<Vec<_>>()),
            wmap: client.create_from_slice(&arena), wmap_bytes: arena.len(),
            expert_bytes: layout.expert_bytes, n_routed: 2, stride: 2, scaled: true,
        };
        (table, arena)
    }

    // Decode through the existing ROW-MAJOR format reader, then place those
    // real values into the shadow's physical word order. This checks scales
    // and nibble addressing independently of ema_plane's inverse map.
    fn expected_shadow(arena: &[u8], offsets: &[u64], sc2: &[f32], n: usize, k: usize) -> Vec<f32> {
        use super::super::super::{fp4gemm, nvfp4};
        let mut out = vec![0.0; sc2.len() * n * k];
        for e in 0..sc2.len() {
            let c = offsets[2 * e] as usize;
            let s = offsets[2 * e + 1] as usize;
            let mut codes = vec![0; n * k / 2];
            let mut scales = vec![0; n * k / 16];
            fp4gemm::unswizzle_b_codes_into(&arena[c..c + codes.len()], &mut codes, n, k);
            fp4gemm::unswizzle_b_scales_into(&arena[s..s + scales.len()], &mut scales, n, k);
            let mut logical = vec![0.0; n * k];
            nvfp4::decode_row(&codes, &scales, sc2[e], &mut logical);
            for r in 0..n {
                for col in 0..k {
                    let block = (r / 8) * (k / 64) + col / 64;
                    let w = (col % 64) / 8;
                    let word = (w / 4) * 32 + (r % 8) * 4 + w % 4;
                    out[e * n * k + block * 512 + word * 8 + col % 8] = logical[r * k + col];
                }
            }
        }
        out
    }

    #[cube(launch)]
    fn replace_expert_codes(w: &mut Array<u32>, off: &Array<u64>, expert: usize, values: usize, codeword: u32) {
        let i = ABSOLUTE_POS as usize;
        let base = usize::cast_from(off[2 * expert]) / 4;
        if i < values / 8 { w[base + i] = codeword; }
    }

    fn replace(client: &Client, t: &ExpertTable, off: &Handle, expert: usize, values: usize, word: u32) {
        unsafe {
            replace_expert_codes::launch::<cubecl::cuda::CudaRuntime>(
                client, CubeCount::new_1d((values / 8).div_ceil(CUBE as usize) as u32), CubeDim::new_1d(CUBE),
                ArrayArg::from_raw_parts(t.wmap.clone(), t.wmap_bytes / 4),
                ArrayArg::from_raw_parts(off.clone(), 2 * t.n_routed), expert, values, word,
            )
        };
    }

    fn w13_product(client: &Client, t: &ExpertTable) -> Vec<u8> {
        use super::super::super::{fp4quant, moegroup};
        let input: Vec<u8> = (0..32 * 64).flat_map(|i| {
            half::bf16::from_f32((i as f32 * 0.13).sin()).to_le_bytes()
        }).collect();
        let input = client.create_from_slice(&input);
        let (a, asc) = fp4quant::quantize_nvfp4_bf16(client, &input, 32, 64);
        let blk = moegroup::BlockPlanDev {
            slot: client.create_from_slice(&words(&[0, 1])),
            tile0: client.create_from_slice(&words(&[0, 1])),
            cnt: client.create_from_slice(&words(&[1, 1])),
            blocks: 2, planes: 1, rows_real: 32,
        };
        let output = moegroup::fp4_linear_grouped_launch_as::<half::bf16, cubecl::cuda::CudaRuntime>(
            client, true, &a, &asc, &t.wmap, t.wmap_bytes, &blk, &t.off13, &t.sc13,
            2, 32, 64, 128,
        );
        read(client, &output)
    }

    #[test]
    #[ignore = "opt-in tiny CUDA EMA storage/forward test; requires the GPU reservation"]
    fn gpu_teacher_is_separate_and_retains_real_ema_between_code_refreshes() {
        let client = <cubecl::cuda::CudaRuntime as Runtime>::client(&Default::default());
        let (student, original) = tiny_student(&client);
        let mut bank = EmaBank::new(&client, &student, 64, 64, true, 0.5, 9).unwrap();
        assert_eq!(read(&client, &bank.teacher.wmap), original[32..]);
        assert_eq!(w13_product(&client, &student), w13_product(&client, bank.teacher_table()));
        let l = bank.layout;
        let (mut o13, mut o2) = l.offsets(2);
        for offset in o13.iter_mut().chain(&mut o2) { *offset += 32; }
        let before13 = expected_shadow(&original, &o13, &[0.37, 1.25], 128, 64);
        let before2 = expected_shadow(&original, &o2, &[1.7, 0.75], 64, 64);
        assert_eq!(floats(&read(&client, &bank.shadow13)), before13);
        assert_eq!(floats(&read(&client, &bank.shadow2)), before2);
        replace(&client, &student, &student.off13, 1, l.values13, 0x33333333);
        replace(&client, &student, &student.off2, 0, l.values2, 0xeeeeeeee);
        let changed = read(&client, &student.wmap);
        assert_eq!(read(&client, &bank.teacher.wmap), original[32..], "student update leaked into teacher");
        let now13 = expected_shadow(&changed, &o13, &[0.37, 1.25], 128, 64);
        let now2 = expected_shadow(&changed, &o2, &[1.7, 0.75], 64, 64);
        let mut expected13 = before13;
        let mut expected2 = before2;
        for version in 0..2 {
            // Second boundary has no new code changes. EMA must still bring
            // EVERY shadow toward the student rather than skip inactive slabs.
            let report = bank.advance(&client, version, 10 + version).unwrap();
            assert_eq!(report.version, version + 1);
            for (a, b) in expected13.iter_mut().zip(&now13) { *a = 0.5 * *a + 0.5 * b; }
            for (a, b) in expected2.iter_mut().zip(&now2) { *a = 0.5 * *a + 0.5 * b; }
            assert_eq!(floats(&read(&client, &bank.shadow13)), expected13);
            assert_eq!(floats(&read(&client, &bank.shadow2)), expected2);
            assert_eq!(read(&client, &student.wmap), changed, "EMA wrote student storage");
        }
        let teacher = read(&client, &bank.teacher.wmap);
        assert_ne!(teacher, original[32..], "EMA never refreshed teacher codes");
        let (a, b) = l.offsets(2);
        for (off, values) in [(&a, l.values13), (&b, l.values2)] {
            for e in 0..2 {
                let s = off[2 * e + 1] as usize;
                assert_eq!(&teacher[s..s + values / 16], &original[32 + s..32 + s + values / 16]);
            }
        }
        assert!(bank.advance(&client, 1, 12).is_err());
        assert!(bank.advance(&client, 2, 11).is_err());
        assert_eq!(read(&client, &bank.teacher.wmap), teacher, "refused boundary mutated teacher");
    }

    #[test]
    #[ignore = "opt-in tiny CUDA EMA endpoint test; requires the GPU reservation"]
    fn gpu_beta_zero_refreshes_exact_codes_without_touching_student() {
        let client = <cubecl::cuda::CudaRuntime as Runtime>::client(&Default::default());
        let (student, _) = tiny_student(&client);
        let mut bank = EmaBank::new(&client, &student, 64, 64, true, 0.0, 0).unwrap();
        replace(&client, &student, &student.off13, 1, bank.layout.values13, 0x88888888);
        bank.advance(&client, 0, 1).unwrap();
        let source = read(&client, &student.wmap);
        assert_eq!(read(&client, &bank.teacher.wmap), source[32..]);
        assert_eq!(w13_product(&client, &student), w13_product(&client, bank.teacher_table()));
    }
}
