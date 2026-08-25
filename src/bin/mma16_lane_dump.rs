//! The `m16n8k16` B fragment map, off the device, and what it costs to read.
//!
//! `fp4_frag_b_map` settled `m16n8k64` — one instruction, one operand width,
//! one constructor — and the FP4 weight permutation was derived from it. The
//! head lane and every BF16 lane issue a DIFFERENT instruction:
//! `MmaDefinition::<bf16, bf16, f32>::new(16, 8, 16)`. Nothing about the k64
//! derivation transfers, so this dumps the k16 map on its own before any
//! permutation is built on top of it.
//!
//! Three sections, all cheap, no model load:
//!
//! 1. **The map.** `position_of_nth(lane, i * vs_b, MatrixIdent::B)` for all 32
//!    lanes, as the device answers it. Also A and the accumulator, because a
//!    surprise in either would mean the dump is reading a different definition
//!    than the kernels do.
//! 2. **The access, counted.** Feed the map through the two consumers' OWN
//!    index arithmetic (`w4a16_linear`: `b[gr * k/8 + gc/8]` over `[n, k/8]`
//!    u32; `bf16_linear`: `b[(gr * k + gc) / 2]` over `[n, k]` bf16) and count,
//!    per warp load instruction, the distinct 32-byte sectors and 128-byte
//!    lines touched and the bytes actually consumed out of them. Then the same
//!    over a whole k loop, which is the number that says whether the scatter
//!    costs DRAM traffic or only requests.
//! 3. **The permutation the map forces**, if any, printed as a destination
//!    formula plus a bijection check on the host.
//!
//! `INK_DUMP_K` / `INK_DUMP_N` set the shape the counting uses (default the
//! unembedding's `k = 4096`, and `n = 64` — n only sets the row stride's
//! multiple, and the counts are per warp, so a small n keeps this instant).

use cubecl::future;
use cubecl::prelude::*;
use mary::models::inkling::w4a16gemm::mma16_frag_map_launch;

type Rt = cubecl::cuda::CudaRuntime;

/// Distinct aligned blocks of `g` bytes touched by `addrs`.
fn granules(addrs: &[(usize, usize)], g: usize) -> usize {
    let mut v: Vec<usize> = addrs
        .iter()
        .flat_map(|&(a, len)| (a / g..=(a + len - 1) / g).collect::<Vec<_>>())
        .collect();
    v.sort_unstable();
    v.dedup();
    v.len()
}

fn main() {
    let device = Default::default();
    let client = Rt::client(&device);

    let k: usize = std::env::var("INK_DUMP_K")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(4096);
    let n: usize = std::env::var("INK_DUMP_N")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(64);

    let h = mma16_frag_map_launch::<Rt>(&client);
    let _ = future::block_on(client.sync());
    let raw = client.read_one(h).unwrap();
    let w: Vec<u32> = raw
        .chunks_exact(4)
        .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();

    let (vc_b, vs_b, ec_b) = (w[768] as usize, w[769] as usize, w[770] as usize);
    let (vc_a, vs_a, ec_a) = (w[771] as usize, w[772] as usize, w[773] as usize);
    let (vc_c, vs_c, ec_c) = (w[774] as usize, w[775] as usize, w[776] as usize);
    let pack = w[777] as usize;

    println!("m16n8k16, MmaDefinition::<bf16, bf16, f32>::new(16, 8, 16), on the device");
    println!(
        "  B: elems/lane {ec_b}, vector {vs_b}, loads/lane {vc_b}   \
         A: {ec_a}/{vs_a}/{vc_a}   Acc: {ec_c}/{vs_c}/{vc_c}   packing_factor {pack}"
    );

    // The B map itself.
    println!("\n  B fragment, position_of_nth(lane, i * vs_b, B) -> (k row, n col):");
    for lane in 0..32usize {
        let mut s = String::new();
        for i in 0..vc_b {
            let (r, c) = (w[(lane * 4 + i) * 2], w[(lane * 4 + i) * 2 + 1]);
            s.push_str(&format!(" i{i}=(k{r:>2}, n{c})"));
        }
        println!("    lane {lane:>2}{s}");
    }

    // Does it match the closed form the hypothesis assumes?
    let mut closed_form_holds = true;
    for lane in 0..32usize {
        for i in 0..vc_b {
            let (r, c) = (
                w[(lane * 4 + i) * 2] as usize,
                w[(lane * 4 + i) * 2 + 1] as usize,
            );
            if c != lane >> 2 || r != 2 * (lane & 3) + 8 * i {
                closed_form_holds = false;
            }
        }
    }
    println!(
        "\n  closed form  col = lane>>2,  row = 2*(lane&3) + 8*i : {}",
        if closed_form_holds {
            "HOLDS"
        } else {
            "DOES NOT HOLD -- read the table above"
        }
    );

    // --- Section 2: what one warp's load instruction actually touches. -------
    // Both consumers, same map, their own arithmetic. `n_base = 0`, `kbase` from
    // tile `t`; the counts are per warp and independent of n_base.
    let wpr = k / 8; // u32 words per weight row, W4A16
    println!("\n  per warp LOAD INSTRUCTION (n_tile 0, k_tile 0), k = {k}, n = {n}:");
    for (name, elem_bytes, row_bytes, addr) in [
        (
            "w4a16_linear  b[gr*k/8 + gc/8]  (u32 word)",
            4usize,
            wpr * 4,
            0usize,
        ),
        ("bf16_linear   b[(gr*k+gc)/2]      (2x bf16)", 4, k * 2, 1),
    ] {
        for i in 0..vc_b {
            let mut touched: Vec<(usize, usize)> = Vec::new();
            for lane in 0..32usize {
                let r = w[(lane * 4 + i) * 2] as usize;
                let c = w[(lane * 4 + i) * 2 + 1] as usize;
                let gr = c; // + n_base
                let gc = r; // + kbase, kbase = 0
                let byte = if addr == 0 {
                    gr * row_bytes + (gc / 8) * 4
                } else {
                    gr * row_bytes + gc * 2
                };
                touched.push((byte, elem_bytes));
            }
            let mut distinct: Vec<usize> = touched.iter().map(|t| t.0).collect();
            distinct.sort_unstable();
            distinct.dedup();
            let useful = distinct.len() * elem_bytes;
            let s32 = granules(&touched, 32);
            let l128 = granules(&touched, 128);
            println!(
                "    {name}  i={i}: {:>2} distinct addrs, {useful:>3} useful B, \
                 {s32:>2} x 32B sectors ({:>5.1}% used), {l128} x 128B lines ({:>5.1}% used)",
                distinct.len(),
                100.0 * useful as f64 / (s32 * 32) as f64,
                100.0 * useful as f64 / (l128 * 128) as f64,
            );
        }
    }

    // Over a WHOLE k loop: the same warp, every k tile, so sectors a later tile
    // re-reads are counted once. This separates "costs DRAM bytes" from "costs
    // requests" -- the k loop walks each weight row forward, so a sector a load
    // half-uses is finished by the next tile if it is still in L1.
    println!(
        "\n  per warp over the WHOLE k loop ({} k-tiles, one n-tile):",
        k / 16
    );
    for (name, elem_bytes, row_bytes, addr) in [
        ("w4a16_linear", 4usize, wpr * 4, 0usize),
        ("bf16_linear ", 4, k * 2, 1),
    ] {
        let mut touched: Vec<(usize, usize)> = Vec::new();
        for t in 0..k / 16 {
            for i in 0..vc_b {
                for lane in 0..32usize {
                    let r = w[(lane * 4 + i) * 2] as usize;
                    let c = w[(lane * 4 + i) * 2 + 1] as usize;
                    let gr = c;
                    let gc = r + t * 16;
                    let byte = if addr == 0 {
                        gr * row_bytes + (gc / 8) * 4
                    } else {
                        gr * row_bytes + gc * 2
                    };
                    touched.push((byte, elem_bytes));
                }
            }
        }
        let mut distinct: Vec<usize> = touched.iter().map(|t| t.0).collect();
        distinct.sort_unstable();
        distinct.dedup();
        let useful = distinct.len() * elem_bytes;
        let s32 = granules(&touched, 32);
        let l128 = granules(&touched, 128);
        let loads = (k / 16) * vc_b;
        println!(
            "    {name}: {useful} useful B over {s32} x 32B sectors ({:.1}% used), \
             {l128} x 128B lines ({:.1}% used); {loads} warp loads",
            100.0 * useful as f64 / (s32 * 32) as f64,
            100.0 * useful as f64 / (l128 * 128) as f64,
        );
        println!(
            "      coalesced ideal for the same {useful} B: {} x 32B sectors, {} x 128B lines, \
             {} warp loads",
            useful.div_ceil(32),
            useful.div_ceil(128),
            useful.div_ceil(128),
        );
    }

    // The scale plane, W4A16 only: one E4M3 byte per weight row per k tile.
    println!("\n  W4A16 scale plane, b_sc[gr*k/16 + gc/16], per warp load:");
    {
        let spr = k / 16;
        let mut touched: Vec<(usize, usize)> = Vec::new();
        for lane in 0..32usize {
            let c = w[lane * 8 + 1] as usize;
            touched.push((c * spr, 1));
        }
        let mut distinct: Vec<usize> = touched.iter().map(|t| t.0).collect();
        distinct.sort_unstable();
        distinct.dedup();
        let s32 = granules(&touched, 32);
        println!(
            "    {} distinct addrs, {} useful B, {s32} x 32B sectors ({:.1}% used)",
            distinct.len(),
            distinct.len(),
            100.0 * distinct.len() as f64 / (s32 * 32) as f64
        );
    }

    // --- Section 3: the permutation the map forces. --------------------------
    if closed_form_holds {
        println!(
            "\n  The permutation the map forces (destination inside one (n_tile, k_tile) block):"
        );
        println!(
            "    W4A16 codes: block = NTILE(8) rows x KTILE/2(8) B = 64 B\n\
             \x20     dst_byte(col, word) = word * 32 + col * 4      word = row/8 in 0..2\n\
             \x20     substituting col = lane>>2, word = i  ->  dst_byte = 32*i + 4*(lane>>2)\n\
             \x20     so load i is 32 CONTIGUOUS bytes, 8 distinct words broadcast to 4 lanes each:\n\
             \x20     ONE sector where the row-major form takes eight."
        );
        println!(
            "    W4A16 scales: block = 8 rows x 1 B = 8 B, dst_byte(col) = col\n\
             \x20     load = 8 contiguous bytes, one sector where the row-major form takes eight."
        );
        println!(
            "    BF16 weights: block = 8 rows x 16 elems x 2 B = 256 B\n\
             \x20     dst_byte(col, p) = (p/4) * 128 + col * 16 + (p%4) * 4    p = row/2 in 0..8\n\
             \x20     substituting col = lane>>2, p = (lane&3) + 4*i  ->  dst_byte = 128*i + 4*lane\n\
             \x20     so load i is 128 CONTIGUOUS bytes in lane order: the fully-coalesced case."
        );

        // Bijection, on the host, for each of the three.
        let mut ok = true;
        // codes: (col in 0..8, word in 0..2) -> 0..64 step 4
        let mut seen = vec![false; 64];
        for c in 0..8usize {
            for word in 0..2usize {
                let d = word * 32 + c * 4;
                for b in 0..4 {
                    if seen[d + b] {
                        ok = false;
                    }
                    seen[d + b] = true;
                }
            }
        }
        ok &= seen.iter().all(|v| *v);
        // scales
        let mut seen = vec![false; 8];
        for c in 0..8usize {
            if seen[c] {
                ok = false;
            }
            seen[c] = true;
        }
        ok &= seen.iter().all(|v| *v);
        // bf16
        let mut seen = vec![false; 256];
        for c in 0..8usize {
            for p in 0..8usize {
                let d = (p / 4) * 128 + c * 16 + (p % 4) * 4;
                for b in 0..4 {
                    if seen[d + b] {
                        ok = false;
                    }
                    seen[d + b] = true;
                }
            }
        }
        ok &= seen.iter().all(|v| *v);
        println!(
            "    all three destination formulas are bijections on their block: {}",
            if ok {
                "yes"
            } else {
                "NO -- do not build on them"
            }
        );
    }
    let _ = n;
}
