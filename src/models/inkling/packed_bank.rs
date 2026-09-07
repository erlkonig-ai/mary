//! Host-packed rows against an explicit expert bank. No Weights/source lookup.

use std::collections::BTreeMap;
use std::time::Instant;

use anyhow::{Result, ensure};
use cubecl::prelude::*;

use super::{HostT, T2, bytes_of, grouped_experts_core};
use super::super::devplan::ExpertTable;
use super::super::moegroup::{BlockPlanDev, RowPlan};

/// Entries are copied in the SAME slot order as RowPlan's expert iterator.
/// The only uploaded routing identity is one expert id per occupied slot;
/// offsets and scale2 stay on-device and belong to the supplied bank.
#[cube(launch)]
fn gather_table(
    ids: &Array<u32>,
    source13: &Array<u64>,
    source2: &Array<u64>,
    source_sc13: &Array<f32>,
    source_sc2: &Array<f32>,
    off13: &mut Array<u64>,
    off2: &mut Array<u64>,
    sc13: &mut Array<f32>,
    sc2: &mut Array<f32>,
) {
    let slot = ABSOLUTE_POS as usize;
    if slot < ids.len() {
        let expert = ids[slot] as usize;
        off13[2 * slot] = source13[2 * expert];
        off13[2 * slot + 1] = source13[2 * expert + 1];
        off2[2 * slot] = source2[2 * expert];
        off2[2 * slot + 1] = source2[2 * expert + 1];
        sc13[slot] = source_sc13[expert];
        sc2[slot] = source_sc2[expert];
    }
}

fn plan(
    by_expert: &BTreeMap<usize, Vec<(usize, f32)>>,
    n: usize,
    n_routed: usize,
    planes: usize,
) -> Result<(Vec<u32>, RowPlan)> {
    ensure!(n > 0 && n <= i32::MAX as usize, "packed bank needs an addressable token batch");
    ensure!(!by_expert.is_empty(), "packed bank has no active expert rows");
    let mut ids = Vec::with_capacity(by_expert.len());
    for (&expert, rows) in by_expert {
        ensure!(expert < n_routed && expert <= u32::MAX as usize,
            "expert {expert} is outside the supplied bank of {n_routed} experts");
        ensure!(!rows.is_empty(), "active expert {expert} has no token rows");
        for &(token, weight) in rows {
            ensure!(token < n && weight.is_finite() && weight >= 0.0,
                "invalid token/weight for packed expert {expert}: {token}/{weight}");
        }
        ids.push(expert as u32);
    }
    let plan = RowPlan::build(by_expert.values(), n, planes);
    ensure!(plan.m_total() <= u32::MAX as usize, "packed rows exceed device indices");
    Ok((ids, plan))
}

#[allow(clippy::too_many_arguments)]
pub(super) fn run(
    client: &ComputeClient<cubecl::cuda::CudaRuntime>,
    dev: &burn::backend::cuda::CudaDevice,
    prefix: &str,
    table: &ExpertTable,
    by_expert: &BTreeMap<usize, Vec<(usize, f32)>>,
    hn: &T2,
    n: usize,
    h: usize,
    inter: usize,
    swizzled: bool,
    host: &mut HostT,
) -> Result<T2> {
    ensure!(table.scaled && table.stride == 2,
        "explicit packed banks require an NVFP4 offset/scale table");
    ensure!(table.wmap_bytes > 0 && table.wmap_bytes % 4 == 0,
        "explicit bank mapping must contain aligned packed words");
    ensure!(h > 0 && inter > 0 && h % 64 == 0 && inter % 64 == 0,
        "explicit bank projections need complete NVFP4 k64 tiles");
    let expert_bytes = 3u128 * h as u128 * inter as u128 * 9 / 16;
    ensure!(expert_bytes == table.expert_bytes as u128,
        "explicit bank expert width differs from the grouped projection shape");
    ensure!(hn.dims() == [n, h], "packed bank input shape does not match its row plan");
    let started = Instant::now();
    let planes = RowPlan::planes();
    let (ids, plan) = plan(by_expert, n, table.n_routed, planes)?;
    let slots = ids.len();
    let static_start = Instant::now();
    let row_tok = client.create_from_slice(bytes_of(&plan.row_tok));
    let blk = BlockPlanDev {
        slot: client.create_from_slice(bytes_of(&plan.blk_slot)),
        tile0: client.create_from_slice(bytes_of(&plan.blk_tile0)),
        cnt: client.create_from_slice(bytes_of(&plan.blk_cnt)),
        blocks: plan.blk_slot.len(), planes, rows_real: plan.rows_real(),
    };
    let tok_rows = client.create_from_slice(bytes_of(&plan.tok_rows));
    let tok_cnt = client.create_from_slice(bytes_of(&plan.tok_cnt));
    host.plan_up_static += static_start.elapsed().as_secs_f64();
    let routed_start = Instant::now();
    let row_wgt = client.create_from_slice(bytes_of(&plan.row_wgt));
    let expert_ids = client.create_from_slice(bytes_of(&ids));
    let off13 = client.empty(slots * 2 * size_of::<u64>());
    let off2 = client.empty(slots * 2 * size_of::<u64>());
    let sc13 = client.empty(slots * size_of::<f32>());
    let sc2 = client.empty(slots * size_of::<f32>());
    unsafe {
        gather_table::launch::<cubecl::cuda::CudaRuntime>(
            client, CubeCount::Static(slots.div_ceil(128) as u32, 1, 1), CubeDim::new_1d(128),
            ArrayArg::from_raw_parts(expert_ids, slots),
            ArrayArg::from_raw_parts(table.off13.clone(), 2 * table.n_routed),
            ArrayArg::from_raw_parts(table.off2.clone(), 2 * table.n_routed),
            ArrayArg::from_raw_parts(table.sc13.clone(), table.n_routed),
            ArrayArg::from_raw_parts(table.sc2.clone(), table.n_routed),
            ArrayArg::from_raw_parts(off13.clone(), 2 * slots),
            ArrayArg::from_raw_parts(off2.clone(), 2 * slots),
            ArrayArg::from_raw_parts(sc13.clone(), slots),
            ArrayArg::from_raw_parts(sc2.clone(), slots),
        )
    };
    host.plan_up_routed += routed_start.elapsed().as_secs_f64();
    let (input, dtype) = super::super::seam::handle_of_any(hn.clone());
    let output = grouped_experts_core(client, dev, prefix, &table.wmap, table.wmap_bytes,
        &blk, &input, dtype, &row_tok, &row_wgt, &tok_rows, &tok_cnt,
        &off13, &off2, &sc13, &sc2, slots, plan.m_total(), plan.kmax,
        n, h, inter, swizzled, started, host);
    host.grouped += 1;
    host.expert_slots += slots;
    Ok(output)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn expert_ids_and_packed_slots_have_one_canonical_order() {
        let by_expert = BTreeMap::from([
            (9, vec![(0, 0.25), (1, 0.75)]),
            (2, vec![(1, 0.5)]),
        ]);
        let (ids, plan) = plan(&by_expert, 2, 16, 4).unwrap();
        assert_eq!(ids, [2, 9]);
        assert_eq!(plan.blk_slot, [0, 1]);
        assert_eq!(plan.row_tok[0], 1);
        assert_eq!(&plan.row_tok[16..18], &[0, 1]);
        assert_eq!(plan.tok_cnt, [1, 2]);
        assert_eq!(&plan.tok_rows[2..4], &[0, 17]);
        assert_eq!(plan.row_wgt[0], 0.5);
        assert_eq!(&plan.row_wgt[16..18], &[0.25, 0.75]);
        assert_eq!(plan.rows_real(), 3);
    }

    #[test]
    fn shared_experts_pack_real_tokens_across_tile_boundary() {
        for n in [2, 17] {
            let routes = BTreeMap::from([
                (0, (0..n).map(|i| (i, 0.25)).collect()),
                (1, (0..n).map(|i| (i, 0.75)).collect()),
            ]);
            let (_, plan) = plan(&routes, n, 2, 4).unwrap();
            assert_eq!(plan.m_total(), 2 * n.div_ceil(16) * 16);
            assert_eq!(plan.rows_real(), 2 * n);
            assert!(plan.m_total() < n * 2 * 16);
            assert!(plan.tok_cnt.iter().all(|&count| count == 2));
        }
    }

    #[test]
    fn invalid_bank_indices_and_routes_fail_before_launch() {
        for routes in [
            BTreeMap::from([(2, vec![(0, 1.0)])]),
            BTreeMap::from([(0, vec![(2, 1.0)])]),
            BTreeMap::from([(0, vec![(0, f32::NAN)])]),
            BTreeMap::from([(0, vec![(0, -1.0)])]),
            BTreeMap::from([(0, Vec::new())]),
        ] {
            assert!(plan(&routes, 2, 2, 4).is_err());
        }
    }

    /// Small address/row-layout check, not a real-model numerical identity
    /// gate. Both executions consume the same synthetic packed bank; neither
    /// reconstructs a host model. Requires explicit CUDA test admission.
    #[test]
    #[ignore = "requires CUDA with native NVFP4 support; run under the shared GPU lock"]
    fn cuda_explicit_bank_packed_and_device_rows() {
        use super::super::super::{devplan, fp4gemm, seam};
        use super::super::{Bk, devroute_new, down, routed_experts_fp4_dev, up2};

        let dev = burn::backend::cuda::CudaDevice::default();
        let (h, inter) = (128, 128);
        for n in [2, 17] {
            let input: Vec<f32> = (0..n).flat_map(|row| {
                (0..h).map(move |col| 0.001 * (row + 1) as f32 * (col % 3 + 1) as f32)
            }).collect();
            let hn = up2::<Bk>(input, n, h, &dev);
            let client = seam::client_of(&hn);
            let mut bytes = Vec::new();
            let mut off13 = Vec::<u64>::new();
            let mut off2 = Vec::<u64>::new();
            // Distinct expert planes, row-major NVFP4 storage. Scale bytes
            // 0x38 encode 1.0; code nibbles 1 and 2 encode 0.5 and 1.0.
            for code in [0x33u8, 0x11u8, 0x44u8, 0x22u8] {
                off13.push(bytes.len() as u64);
                bytes.extend(std::iter::repeat_n(code, 2 * inter * h / 2));
                off13.push(bytes.len() as u64);
                bytes.extend(std::iter::repeat_n(0x38, 2 * inter * h / 16));
                off2.push(bytes.len() as u64);
                bytes.extend(std::iter::repeat_n(code, inter * h / 2));
                off2.push(bytes.len() as u64);
                bytes.extend(std::iter::repeat_n(0x38, inter * h / 16));
            }
            let table = ExpertTable {
                off13: client.create_from_slice(bytes_of(&off13)),
                off2: client.create_from_slice(bytes_of(&off2)),
                sc13: client.create_from_slice(bytes_of(&[0.25f32, 1.0, 0.5, 2.0])),
                sc2: client.create_from_slice(bytes_of(&[1.5f32, 0.5, 1.0, 1.0])),
                wmap: client.create_from_slice(&bytes), wmap_bytes: bytes.len(),
                expert_bytes: bytes.len() / 4, n_routed: 4, stride: 2, scaled: true,
            };
            let routes = BTreeMap::from([
                (1, (0..n).map(|i| (i, 0.25)).collect()),
                (3, (0..n).map(|i| (i, 0.75)).collect()),
            ]);
            let topk: Vec<f32> = (0..n).flat_map(|_| [1.0, 3.0, 0.25, 0.75, 0.0]).collect();
            let topk = client.create_from_slice(bytes_of(&topk));
            let dr = devroute_new(&client, 2, n);
            let dp = devplan::plan_from_topk_launch(&client, &topk, &table, &dr.fault,
                dr.kmax, fp4gemm::MTILE, 5, n);
            let packed = run(&client, &dev, "synthetic-bank.", &table, &routes,
                &hn, n, h, inter, false, &mut HostT::default()).unwrap();
            let device = routed_experts_fp4_dev(&client, &dev, "synthetic-bank.", &table,
                &dp, &dr, &hn, n, h, inter, false, Instant::now(), &mut HostT::default());
            let packed = down(packed);
            let device = down(device);
            assert_eq!(packed.len(), n * h);
            assert_eq!(device.len(), packed.len());
            let mut worst = 0.0f32;
            for (&a, &b) in packed.iter().zip(&device) {
                assert!(a.is_finite() && b.is_finite());
                worst = worst.max((a - b).abs() / a.abs().max(b.abs()).max(1e-4));
            }
            eprintln!("synthetic explicit bank B={n}: packed/device max relative gap {worst}");
            assert!(worst <= 0.02, "synthetic packed bank row/address gap {worst}");
            assert_ne!(packed[0], packed[(n - 1) * h], "distinct token rows must stay distinct");
        }
    }
}
