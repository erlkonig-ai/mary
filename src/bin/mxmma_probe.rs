//! Does CubeCL emit the MXFP4 (ue8m0, one scale per 32) tensor-core MMA?
//!
//! The PTX-level measurement says the instruction exists and executes on
//! sm_121a. This asks the question that actually matters for the port: can the
//! framework we build in reach it, or would we need inline PTX?
//!
//! Correctness gate before any claim: a 16x8x64 scaled matmul with known
//! operands, checked against a CPU reference computed from the same decoded
//! values.
use cubecl::cuda::CudaRuntime;
use cubecl::prelude::*;
use cubecl::{e2m1, e2m1x2, ue8m0};
use cubecl::ir::MatrixIdent;

#[cube(launch)]
pub fn mx_mma<AB: Scalar, S: Scalar, NA: Size, NC: Size>(
    a: &Tensor<Vector<AB, NA>>,
    b: &Tensor<Vector<AB, NA>>,
    scales_a: &Tensor<S>,
    scales_b: &Tensor<S>,
    out: &mut Tensor<Vector<f32, NC>>,
    #[comptime] size_k: usize,
    #[comptime] size_n: usize,
    #[comptime] scales_factor: usize,
) {
    // scales_factor 2 at k=64 == one scale per 32 elements == MXFP4.
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
    let mut reg_a = Array::<Vector<AB, NA>>::new(vc_a);
    let mut reg_b = Array::<Vector<AB, NA>>::new(vc_b);
    let ec_c = def.elems_per_lane(MatrixIdent::Accumulator);
    let vs_c = def.vector_size(MatrixIdent::Accumulator);
    let vc_c = comptime!(ec_c / vs_c);
    let mut acc = Array::<Vector<f32, NC>>::new(vc_c);

    #[unroll]
    for i in 0..vc_c {
        acc[i] = Vector::<f32, NC>::cast_from(0.0f32);
    }
    #[unroll]
    for i in 0..vc_a {
        let (row, col) = def.position_of_nth(lane, (i * vs_a * pack) as u32, MatrixIdent::A);
        reg_a[i] = a[(row as usize * size_k / 2 + col as usize / 2) / a.vector_size()];
    }
    #[unroll]
    for i in 0..vc_b {
        let (row, col) = def.position_of_nth(lane, (i * vs_b * pack) as u32, MatrixIdent::B);
        reg_b[i] = b[(col as usize * size_k / 2 + row as usize / 2) / b.vector_size()];
    }

    let scales_count = def.scales_count();
    let size!(NS) = def.scales_vector_size();
    let mut sa = Vector::<S, NS>::empty();
    let mut sb = Vector::<S, NS>::empty();
    let sia = def.scales_index(lane, MatrixIdent::A) as usize;
    let sib = def.scales_index(lane, MatrixIdent::B) as usize;
    #[unroll]
    for i in 0..scales_count {
        sa[i] = scales_a[sia * scales_factor + i];
        sb[i] = scales_b[sib * scales_factor + i];
    }

    let d = def.execute_scaled(&reg_a, &reg_b, &acc, sa, sb);
    #[unroll]
    for i in 0..vc_c {
        let (row, col) = def.position_of_nth(lane, (i * vs_c) as u32, MatrixIdent::Accumulator);
        out[(row as usize * size_n + col as usize) / out.vector_size()] = d[i];
    }
}

fn main() {
    let (m, n, k, sf) = (16usize, 8usize, 64usize, 2usize);
    let client = CudaRuntime::client(&Default::default());
    let cd = CubeDim::new_1d(client.properties().hardware.plane_size_max);

    // Known, non-uniform operands so a transpose or index slip cannot pass.
    let a_f: Vec<f32> = (0..m * k)
        .map(|i| e2m1::from_bits(((i % 7) + 1) as u8).to_f32())
        .collect();
    let b_f: Vec<f32> = (0..n * k)
        .map(|i| e2m1::from_bits(((i % 5) + 1) as u8).to_f32())
        .collect();
    let spr = k / (64 / sf);
    let sa: Vec<ue8m0> = (0..m * spr).map(|i| ue8m0::from_bits((127 + (i % 3)) as u8)).collect();
    let sb: Vec<ue8m0> = (0..n * spr).map(|i| ue8m0::from_bits((127 + (i % 2)) as u8)).collect();

    let a_p = e2m1x2::from_f32_slice(&a_f);
    let b_p = e2m1x2::from_f32_slice(&b_f);
    let ah = client.create_from_slice(e2m1x2::as_bytes(&a_p));
    let bh = client.create_from_slice(e2m1x2::as_bytes(&b_p));
    let sah = client.create_from_slice(ue8m0::as_bytes(&sa));
    let sbh = client.create_from_slice(ue8m0::as_bytes(&sb));
    let oh = client.create_from_slice(f32::as_bytes(&vec![0.0f32; m * n]));
    let vs = 32 / e2m1x2::cube_type().size_bits();

    unsafe {
        mx_mma::launch::<e2m1x2, ue8m0, CudaRuntime>(
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

    let mut worst = 0.0f32;
    for i in 0..m {
        for j in 0..n {
            let mut s = 0.0f32;
            for l in 0..k {
                let t = l / (64 / sf);
                s += a_f[i * k + l] * sa[i * spr + t].to_f32() * b_f[j * k + l] * sb[j * spr + t].to_f32();
            }
            let g = got[i * n + j];
            if !g.is_finite() {
                println!("NON-FINITE at ({i},{j}) — FAIL");
                std::process::exit(1);
            }
            let r = (g - s).abs() / s.abs().max(1e-6);
            if !(r <= worst) {
                worst = r;
            }
        }
    }
    println!("max relative error {worst:.3e}");
    if worst > 1e-5 {
        println!("FAIL — CubeCL did not reproduce the reference");
        std::process::exit(1);
    }
    println!("PASS — CubeCL emits and executes the MXFP4 (ue8m0, 1 scale / 32) tensor-core MMA");
}
