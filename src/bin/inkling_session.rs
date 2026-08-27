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
//! `--rewind` runs the other claim, the one about going BACK: that a session
//! rewound to a checkpoint and re-extended with a DIFFERENT tail is the session
//! that was built that way from the start. See [`rewind_gate`].
//!
//! ```text
//! INK_LAYERS=0:4 inkling_session <pile> <ids.bin> [--gen N] [--turns T] [--rewind]
//! ```

use anyhow::{Context, Result};

use mary::models::inkling::session::{Session, SessionConfig};

fn main() -> Result<()> {
    let mut args = std::env::args().skip(1);
    let pile = args
        .next()
        .context("usage: inkling_session <pile> <ids.bin> [--gen N] [--turns T] [--rewind]")?;
    let ids_path = args
        .next()
        .context("usage: inkling_session <pile> <ids.bin> [--gen N] [--turns T] [--rewind]")?;
    let (mut want, mut turns) = (8usize, 1usize);
    let mut rewind = false;
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
            "--rewind" => {
                rewind = true;
                i += 1;
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

    if rewind {
        return rewind_gate(&mut session, &prompt, want);
    }

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

/// The rewind claim, on the real model: a session put back to a checkpoint and
/// extended with a DIFFERENT tail is the session that was built that way.
///
/// # What it runs
///
/// The prompt is split in three. `head` is the settled prefix — the part a
/// caller would keep. `a` and `b` are two different tails over the same
/// positions, which is the shape of a prefix whose last chunk changed.
///
/// ```text
///   REWOUND : prefill(head)  checkpoint  extend(a)  …  rewind  extend(b)  gen
///   FRESH   : reset  prefill(head)                             extend(b)  gen
/// ```
///
/// and the two token streams must agree. That is the whole check, and it is on
/// the TOKENS rather than on `position()` because a cache of the right length
/// holding the wrong keys is precisely what a length assertion cannot see.
///
/// # Why the tokens, and not the logits
///
/// A rewind restores the cache exactly — handles for the windowed layers,
/// counters for the global ones — but the arithmetic AFTER it need not be
/// bit-identical to the run that first passed through: a truncation re-opens
/// the page being written and leaves the settled pages as whatever
/// `Pages::merge_settled` had joined them into, so the flash read can split the
/// same keys into a different set of runs. Same values, different order of
/// summation, and this model's router picks six experts of 256 on a margin that
/// a last-place logit bit can flip. So a disagreement here is read as: WHERE
/// does it start? A stream that agrees for many tokens and then parts company
/// is that; a stream that parts at the first token is a wrong cache.
fn rewind_gate(session: &mut Session, prompt: &[usize], steps: usize) -> Result<()> {
    anyhow::ensure!(
        prompt.len() >= 6,
        "the rewind gate splits the prompt in three; {} tokens is not enough",
        prompt.len()
    );
    // Two thirds settled, the last third the part that "changed". `b` is `a`
    // reversed, so both tails are real tokens of this vocabulary and neither is
    // a constant the model would find degenerate.
    let cut = prompt.len() * 2 / 3;
    let (head, a) = prompt.split_at(cut);
    let b: Vec<usize> = a.iter().rev().copied().collect();

    // Every `run` reports what its two halves cost, because the cost model this
    // gate is here to produce is exactly the comparison between them: a rewind
    // pays for the RE-EXTENDED tail, a reset pays for the whole prompt again.
    let run = |s: &mut Session, tail: &[usize], what: &str| -> Result<Vec<usize>> {
        let t0 = std::time::Instant::now();
        let mut tok = s.extend(tail)?;
        let extended = t0.elapsed().as_secs_f64();
        let mut out = vec![tok];
        let t1 = std::time::Instant::now();
        for _ in 1..steps {
            tok = s.step()?;
            out.push(tok);
        }
        let decoded = t1.elapsed().as_secs_f64();
        println!(
            "  extend {what:<14}: {} tokens in {extended:.3}s ({:.1} ms/token, WALKED), then {} \
             steps in {decoded:.3}s ({:.1} ms/step)",
            tail.len(),
            extended * 1e3 / tail.len().max(1) as f64,
            steps - 1,
            decoded * 1e3 / (steps - 1).max(1) as f64,
        );
        Ok(out)
    };

    // ── the rewound path ────────────────────────────────────────────────────
    // This first prefill is COLD -- it is the pass that binds every layer's
    // weights -- so its seconds are not the number to quote. The one after the
    // reset is.
    session.prefill(head)?;
    let t_cp = std::time::Instant::now();
    let mark = session.checkpoint()?;
    println!(
        "  checkpoint            : position {} in {:.4}s",
        mark.position(),
        t_cp.elapsed().as_secs_f64()
    );
    let discarded = run(session, a, "tail A")?;
    println!(
        "  ran tail A         : position {} (these {} tokens are about to be un-attended to)",
        session.position(),
        discarded.len()
    );
    let t_rw = std::time::Instant::now();
    session.rewind(&mark)?;
    let rewind_s = t_rw.elapsed().as_secs_f64();
    println!("  rewind                : {rewind_s:.4}s");
    anyhow::ensure!(
        session.position() == mark.position(),
        "rewound to {} but the session says {}",
        mark.position(),
        session.position()
    );
    let rewound = run(session, &b, "tail B (rewound)")?;

    // ── the session that was built that way ─────────────────────────────────
    // And this is the ALTERNATIVE a rewind is measured against: throwing the
    // sequence away and re-reading the settled prefix. This prefill is WARM --
    // every layer is bound -- so the seconds it prints are the seconds a rewind
    // does not spend.
    session.reset();
    let t_pf = std::time::Instant::now();
    session.prefill(head)?;
    let prefill_s = t_pf.elapsed().as_secs_f64();
    println!(
        "  reset + prefill       : {} tokens in {prefill_s:.3}s ({:.2} ms/token, BATCHED) -- \
         this is what a rewind does not pay",
        head.len(),
        prefill_s * 1e3 / head.len() as f64,
    );
    let fresh = run(session, &b, "tail B (fresh)")?;

    println!("REWOUND: {rewound:?}");
    println!("FRESH  : {fresh:?}");
    let agree = rewound
        .iter()
        .zip(&fresh)
        .take_while(|(x, y)| x == y)
        .count();
    println!(
        "  agreement          : {agree}/{} tokens{}",
        rewound.len(),
        match agree == rewound.len() {
            true => String::new(),
            false => format!(
                " -- first divergence at {agree} ({} against {})",
                rewound[agree], fresh[agree]
            ),
        }
    );
    // The checkpoint is from the sequence `reset` ended, so it must now be
    // refused. Checked here rather than left to a doc claim: the buffers it
    // holds are still alive, so "refused" is a real behaviour and not the
    // absence of one.
    anyhow::ensure!(
        session.rewind(&mark).is_err(),
        "a checkpoint from before a `reset` was accepted"
    );
    println!("  stale checkpoint   : refused, as it must be");
    anyhow::ensure!(
        agree == rewound.len(),
        "the rewound session diverges from the one built that way at token {agree}"
    );
    println!(
        "  saving                : {prefill_s:.3}s of settled prefix kept, against \
         {rewind_s:.4}s to put the cache back"
    );
    println!("REWIND GATE: PASS");
    Ok(())
}
