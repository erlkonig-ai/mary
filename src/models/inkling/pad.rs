//! Gathering an expert's token rows into the M-padded buffer its MMA wants,
//! in one kernel.
//!
//! Both routed-expert lanes need the same thing: the rows this expert was
//! routed, laid out `[m_pad, hidden]` where `m_pad` is `m` rounded up to the
//! MMA's row tile, with the padding rows zero. That was three Burn ops — a
//! `select`, a `zeros`, and a `cat` that allocates and writes both halves —
//! producing four launches per expert. Six experts a layer, eighteen MoE layers
//! on this node's half, and it came to 432 of a decode step's 4720 kernels for
//! what is a strided copy of one row.
//!
//! The padding rows are not incidental: the MMA tiles M by 16 and a decode step
//! feeds ONE token, so fifteen sixteenths of every expert's A operand is zero.
//! Zeros are what makes that harmless — a zero row produces a zero output row,
//! which the caller slices off — but they are still zeros somebody has to
//! write, and writing them while gathering costs nothing extra.

use cubecl::prelude::*;
use cubecl::server::Handle;

/// Threads per cube.
const CUBE_SIZE: u32 = 256;

/// `out[r, :] = src[idx[r], :]` for `r < m`, zero for `r >= m`.
///
/// `src` is `[_, h]` and `out` is `[m_pad, h]`, both f32 row-major. `idx` holds
/// `m` row numbers into `src` as `i32` — the same buffer the scatter-back uses,
/// so there is one index upload per expert and not two.
#[cube(launch_unchecked)]
fn gather_rows_pad_kernel(
    src: &Array<f32>,
    idx: &Array<i32>,
    out: &mut Array<f32>,
    m: usize,
    h: usize,
    total: usize,
) {
    let p = ABSOLUTE_POS as usize;
    if p < total {
        let r = p / h;
        let mut v = f32::new(0.0f32);
        if r < m {
            let row = u32::cast_from(idx[r]) as usize;
            v = src[row * h + p % h];
        }
        out[p] = v;
    }
}

/// Launch [`gather_rows_pad_kernel`], returning the `[m_pad, h]` buffer.
pub fn gather_rows_pad<R: Runtime>(
    client: &ComputeClient<R>,
    src: &Handle,
    idx: &Handle,
    src_rows: usize,
    m: usize,
    m_pad: usize,
    h: usize,
) -> Handle {
    assert!(m <= m_pad, "{m} rows do not fit in {m_pad}");
    let total = m_pad * h;
    let out = client.empty(total * core::mem::size_of::<f32>());
    let cubes = total.div_ceil(CUBE_SIZE as usize) as u32;
    unsafe {
        gather_rows_pad_kernel::launch_unchecked::<R>(
            client,
            CubeCount::new_1d(cubes),
            CubeDim::new_1d(CUBE_SIZE),
            ArrayArg::from_raw_parts(src.clone(), src_rows * h),
            ArrayArg::from_raw_parts(idx.clone(), m),
            ArrayArg::from_raw_parts(out.clone(), total),
            m,
            h,
            total,
        );
    }
    out
}
