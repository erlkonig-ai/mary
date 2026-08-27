//! Drive a [`Session`] and print the tokens it produces — the parity harness
//! for the seam, not a server.
//!
//! # What this is for
//!
//! `inkling_forward` is the measurement harness and it produces tokens by
//! running a loop inside `main`. [`mary::models::inkling::session::Session`]
//! produces them by holding a model open. Those are two code paths through the
//! same components, and the only claim worth making about the second one is that
//! it agrees with the first: **same weights, same prompt, same schedule, same
//! tokens.** Nothing here samples, batches, drafts, speculates or measures — it
//! prefills, steps, and prints, so a `diff` against `inkling_forward`'s token
//! line is the whole test.
//!
//! It also demonstrates the property the seam exists for, which a single
//! generation cannot: `--turns` runs several turns against ONE session, feeding
//! each turn's tokens back as the next turn's delta. The KV cache is never
//! rebuilt and the prompt is never re-read. That is the thing a program-per-
//! request could not do at any speed.
//!
//! ```text
//! INK_LAYERS=0:4 inkling_session <pile> <ids.bin> [--gen N] [--turns T]
//! ```

use anyhow::{Context, Result};

use mary::models::inkling::session::{Session, SessionConfig};

fn main() -> Result<()> {
    let mut args = std::env::args().skip(1);
    let pile = args
        .next()
        .context("usage: inkling_session <pile> <ids.bin> [--gen N] [--turns T]")?;
    let ids_path = args
        .next()
        .context("usage: inkling_session <pile> <ids.bin> [--gen N] [--turns T]")?;
    let (mut want, mut turns) = (8usize, 1usize);
    let rest: Vec<String> = args.collect();
    let mut i = 0;
    while i < rest.len() {
        match rest[i].as_str() {
            "--gen" => {
                want = rest[i + 1].parse().context("--gen wants a count")?;
                i += 2;
            }
            "--turns" => {
                turns = rest[i + 1].parse().context("--turns wants a count")?;
                i += 2;
            }
            other => anyhow::bail!("unknown argument {other:?}"),
        }
    }

    // The same i64 little-endian ids file `inkling_forward` reads, so the two
    // are given literally the same bytes.
    let prompt: Vec<usize> = std::fs::read(&ids_path)?
        .chunks_exact(8)
        .map(|c| i64::from_le_bytes(c.try_into().expect("8 bytes")) as usize)
        .collect();
    anyhow::ensure!(
        !prompt.is_empty(),
        "no tokens — the forward would be vacuous"
    );

    let t0 = std::time::Instant::now();
    let mut session = Session::load(SessionConfig::new(&pile))?;
    println!(
        "  session loaded     : layers {:?} in {:.1}s",
        session.layer_range(),
        t0.elapsed().as_secs_f64()
    );

    for turn in 0..turns {
        // Turn 0 prefills the prompt. Every turn after it hands over ONLY what
        // is new — which on this harness is the previous turn's own output —
        // because everything before that is still in the cache. That `extend`
        // and not a re-prefill is what a session buys.
        let t_p = std::time::Instant::now();
        let mut tok = match turn {
            0 => session.prefill(&prompt)?,
            _ => session.step()?,
        };
        let t_prefill = t_p.elapsed().as_secs_f64();

        let mut out = vec![tok];
        let t_d = std::time::Instant::now();
        for _ in 1..want {
            tok = session.step()?;
            out.push(tok);
        }
        let t_decode = t_d.elapsed().as_secs_f64();

        println!("TOKENS turn {turn}: {out:?}");
        println!(
            "  turn {turn}: first token {:.2}s, {} more in {:.2}s ({:.3} tok/s), position {}",
            t_prefill,
            want - 1,
            t_decode,
            (want - 1) as f64 / t_decode.max(f64::EPSILON),
            session.position(),
        );
    }
    Ok(())
}
