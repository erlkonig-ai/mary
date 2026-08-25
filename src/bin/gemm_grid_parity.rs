//! Digest of the hand 4-bit GEMM lanes' outputs, for cross-revision comparison.
//!
//! The grid-axis swap in [`mary::models::inkling::fp4gemm`] and
//! [`mary::models::inkling::w4a16gemm`] is a pure scheduling change: every cube
//! computes the same output tile from the same inputs, and only the order the
//! cubes are launched in changes. That claim is checkable rather than
//! arguable — run this at both revisions and the digests have to be equal, at
//! every `m_pad`, or the swap moved more than the schedule.
//!
//! Inputs are host-created from a fixed LCG so two processes see the same
//! bytes; `client.empty()` would not be comparable across runs.

use cubecl::prelude::*;
use cubecl::server::Handle;

use mary::models::inkling::fp4gemm::{
    fp4_linear_launch, fp4_linear_swz_launch, swizzle_b_codes, swizzle_b_scales,
};
use mary::models::inkling::w4a16gemm::{w4a16_linear_launch, w4a16_linear_wide_launch};

type Rt = cubecl::cuda::CudaRuntime;

/// FNV-1a over the raw output bytes. Bit-identity, not closeness — the point of
/// the check is that nothing about the arithmetic moved.
fn digest(bytes: &[u8]) -> u64 {
    let mut h = 0xcbf2_9ce4_8422_2325u64;
    for b in bytes {
        h ^= *b as u64;
        h = h.wrapping_mul(0x1000_0000_01b3);
    }
    h
}

/// Deterministic pseudo-random bytes, same sequence in every process.
fn bytes(n: usize, seed: u64) -> Vec<u8> {
    let mut s = seed | 1;
    (0..n)
        .map(|_| {
            s = s
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            (s >> 33) as u8
        })
        .collect()
}

/// E4M3 scale bytes drawn from a few live exponents — 0x38 is 1.0. A random
/// byte here can be a NaN or an enormous exponent, which would make the f32
/// output all-NaN and the digest insensitive to everything else.
fn scales(n: usize) -> Vec<u8> {
    (0..n).map(|i| [0x38u8, 0x40, 0x30, 0x3c][i % 4]).collect()
}

fn read_digest(client: &ComputeClient<Rt>, h: Handle) -> u64 {
    digest(&client.read_one(h).unwrap())
}

fn main() {
    let device = Default::default();
    let client = Rt::client(&device);

    // k is a multiple of 64 (fp4gemm's KTILE) and of 32 (w4a16's KSTEP);
    // n is a multiple of 8. Small on purpose — this checks values, not speed.
    let k = 256usize;
    let n = 512usize;

    println!("k={k} n={n}  (FNV-1a of the f32 output bytes)");
    for m_pad in [16usize, 32, 64, 128] {
        // W4A16: BF16 activation, packed-u32 weight, E4M3 scales.
        let a_bf = bytes(m_pad * k * 2, 0x51ed);
        let wb = bytes(n * (k / 8) * 4, 0xa17c);
        let ws = scales(n * (k / 16));
        let ha = client.create_from_slice(&a_bf);
        let hb = client.create_from_slice(&wb);
        let hs = client.create_from_slice(&ws);
        let d_w4 = read_digest(
            &client,
            w4a16_linear_launch::<Rt>(&client, &ha, &hb, &hs, m_pad, k, n, 0.75),
        );
        let d_wide = read_digest(
            &client,
            w4a16_linear_wide_launch::<Rt>(&client, &ha, &hb, &hs, m_pad, k, n, 0.75),
        );

        // W4A4: both operands packed E2M1 with E4M3 block scales.
        let qa = client.create_from_slice(&bytes(m_pad * k / 2, 0x2b91));
        let qa_sc = client.create_from_slice(&scales(m_pad * (k / 16)));
        let b_raw = bytes(n * k / 2, 0x77c3);
        let b_sc_raw = scales(n * (k / 16));
        let qb = client.create_from_slice(&b_raw);
        let qb_sc = client.create_from_slice(&b_sc_raw);
        let d_fp4 = read_digest(
            &client,
            fp4_linear_launch::<Rt>(&client, &qa, &qa_sc, &qb, &qb_sc, m_pad, k, n, 0.75),
        );

        // W4A4 again with B PRE-PERMUTED into MMA fragment order. The bytes are
        // the same bytes in a different order and the kernel undoes the order,
        // so this digest has to equal `d_fp4` exactly -- and the three columns
        // to its left have to equal what this harness printed at the revision
        // BEFORE the permutation existed, or something other than the layout
        // moved. Those columns are unchanged code; the check is that they are
        // also unchanged output.
        let qbz = client.create_from_slice(&swizzle_b_codes(&b_raw, n, k));
        let qbz_sc = client.create_from_slice(&swizzle_b_scales(&b_sc_raw, n, k));
        let d_swz = read_digest(
            &client,
            fp4_linear_swz_launch::<Rt>(
                &client, &qa, &qa_sc, &qbz, &qbz_sc, m_pad, k, n, 0.75, true,
            ),
        );
        // And with the SCALE plane left in row-major order, which is the
        // half-permuted variant `fp4_linear_swz`'s `swz_sc = false` reads.
        let d_swz_c = read_digest(
            &client,
            fp4_linear_swz_launch::<Rt>(
                &client, &qa, &qa_sc, &qbz, &qb_sc, m_pad, k, n, 0.75, false,
            ),
        );

        println!(
            "m_pad {m_pad:>4}  w4a16 {d_w4:016x}  wide {d_wide:016x}  fp4 {d_fp4:016x}  \
             swz {d_swz:016x}  swz(codes only) {d_swz_c:016x}"
        );
        assert_eq!(
            d_swz, d_fp4,
            "pre-permuted B is NOT bit-identical at m_pad {m_pad}"
        );
        assert_eq!(
            d_swz_c, d_fp4,
            "pre-permuted codes with row-major scales are NOT bit-identical at m_pad {m_pad}"
        );
    }
}
