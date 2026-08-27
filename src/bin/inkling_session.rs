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
//! `--batched` runs the claim that makes the rewind worth having: that
//! appending a KNOWN multi-token delta in one pass leaves the session the walk
//! leaves, and costs a batched pass rather than one decode step a token. See
//! [`batched_gate`].
//!
//! `--carry` runs the claim a SERVED conversation depends on and that nothing
//! else here checks: that everything a turn said is in the cache when the next
//! turn starts. See [`carry_gate`], which exists because it was not.
//!
//! ```text
//! INK_LAYERS=0:4 inkling_session <pile> <ids.bin> [--gen N] [--turns T] \
//!                                [--rewind] [--batched] [--carry]
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
    let mut batched = false;
    let mut carry = false;
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
            "--batched" => {
                batched = true;
                i += 1;
            }
            "--carry" => {
                carry = true;
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
    if batched {
        return batched_gate(&mut session, &prompt, want);
    }
    if carry {
        return carry_gate(&mut session, &prompt, want);
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
///
/// # What it measured, 2026-08-27, and the framing rule
///
/// One GB10 (`spark`), advisory box lock held, release build, features
/// `inkling-cuda`, `inkling-small-complete.pile`, a 960-token prompt split
/// 640/320 (the head is deliberately longer than the 512 sliding window, so the
/// local layers have genuinely FORGOTTEN keys the checkpoint needs by the time
/// the rewind happens), `--gen 8`, greedy, no `INK_*` switch but the layer
/// range. Every figure below is per TOKEN of the call named, on that box, for
/// that layer range — not per turn and not for the whole 42-layer model.
///
/// | | layers 0..6 | layers 0..21 |
/// |---|---|---|
/// | `checkpoint()` | 0.0000 s | 0.0001 s |
/// | `rewind()` | 0.0000 s | 0.0004 s |
/// | `prefill`, warm, BATCHED | 3.39 ms/token | 11.32 ms/token |
/// | `extend`, warm, WALKED | 19.3 ms/token | 47.4 ms/token |
/// | decode step | 19.4 ms/step | 47.4 ms/step |
///
/// Both ranges agreed 8/8 tokens between the rewound session and the one built
/// that way, so the equivalence held exactly at this length on the real weights.
///
/// A third run pushed past the case where a truncation is trivially exact.
/// `Pages::merge_settled` joins the settled pages once their count passes
/// [`mary::models::inkling::kvpages::MAX_PAGES`] (8), so a 960-token run never
/// reaches it — 960 rows is exactly 8 pages, and truncating back to 640 restores
/// the very page partition that existed there. At 1536 tokens split 1024/512 it
/// DOES: the global stores merge to a single 1024-row page, and the truncation
/// hands the reader ONE key run where the checkpoint's own history had eight.
/// Different split of the same sum, which is the one place a rewind can differ
/// arithmetically from the run that first passed through — and at layers 0..21
/// the two still agreed **12/12 tokens** (prefill 12.32 ms/token, extend 47.3
/// ms/token). That is evidence and not a proof of identity at every length: the
/// same class of partition change is documented at a 3732-token prompt in
/// [`mary::models::inkling::burn::AttnCache::reserve_kv`], where it diverges at
/// step 8. So a rewind at long context should be read as *agrees for a while,
/// then may part company* — never as bit-identical.
///
/// **And the ratio is the finding, not the saving.** Taking and using a
/// checkpoint is free — sub-millisecond, because it is handle clones and
/// counters. What is not free is what comes after it: `Session::extend` walks
/// the delta ONE POSITION AT A TIME (see its own doc for why), so a re-extended
/// token costs a DECODE step where a prefilled one costs a batched pass —
/// 47.4 against 11.32 ms at layers 0..21, a factor of **4.19**. So against the
/// honest alternative (`reset` + one batched `prefill` of the whole new prefix)
/// a rewind is a win only while the CHANGED SUFFIX is under **23.9%** of the
/// prefix, and a loss above it. That crossover is a property of `extend`, not
/// of the rewind: batching the delta would move it to 100%.
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

/// The batched-append claim, on the real model: a session handed a known
/// multi-token delta IN ONE PASS is the session that walked the same delta a
/// position at a time — and it is not the same price.
///
/// # What it runs
///
/// The prompt is split the way [`rewind_gate`] splits it, so the two gates'
/// numbers compose: two thirds settled prefix, one third the delta. Three arms,
/// all against ONE loaded session, each starting from a `reset` and the same
/// warm prefill of the same head:
///
/// ```text
///   WALKED  : prefill(head)  extend_batch=1            extend(tail)  gen
///   BATCHED : prefill(head)  extend_batch=|tail|       extend(tail)  gen
///   CHUNKED : prefill(head)  extend_batch=ceil(|tail|/3)  extend(tail)  gen
/// ```
///
/// and all three token streams must agree. The third arm is not decoration: a
/// delta longer than `extend_batch` is appended in consecutive passes, and the
/// claim that a chunk boundary is a COMMIT POINT AND NOTHING ELSE — which is
/// what makes `extend_batch` a resource knob rather than a semantic one — is
/// only true if a delta split three ways leaves the cache one pass leaves.
///
/// # What it asserts, and why not on `position()`
///
/// The tokens, for [`rewind_gate`]'s reason: a cache of the right length
/// holding the wrong keys is exactly what a length assertion cannot see, and
/// the window bug this path could plausibly have — a batch that carried a local
/// layer past its 512-key window without trimming, or trimmed it by the wrong
/// amount — leaves the position counter perfectly correct. It is also why the
/// head is deliberately longer than the sliding window and the tail is longer
/// than a page: at 640/320 every local layer has genuinely forgotten keys
/// before the delta starts, and the delta itself spans more than two pages.
///
/// Exactness is expected here in a way it is NOT after a rewind. All three arms
/// build their cache forward from the same prefill, so none of them re-opens a
/// settled page the way a truncation does; what differs is only the WIDTH of
/// the passes that append. That width does change the GEMM shape and therefore
/// the accumulation order, so a late divergence is arithmetic and is reported
/// as WHERE it starts rather than as a pass/fail on the first token — the same
/// reading [`rewind_gate`] applies, for the same reason.
///
/// # The cost model this exists to produce
///
/// Three numbers on the same axis, all per TOKEN OF THE DELTA: what a walked
/// append costs, what a batched one costs, and what the settled prefix would
/// cost to re-read through a batched `prefill`. The third is the alternative a
/// rewind is measured against, so the crossover falls straight out of the
/// ratio: a rewind beats `reset` + `prefill` while
///
/// ```text
///   changed suffix / prefix  <  prefill ms/token / extend ms/token
/// ```
///
/// With the walked extend that bound was 11.32 / 47.4 = 23.9%. What it becomes
/// is what this prints.
fn batched_gate(session: &mut Session, prompt: &[usize], steps: usize) -> Result<()> {
    anyhow::ensure!(
        prompt.len() >= 6,
        "the batched gate splits the prompt in two; {} tokens is not enough",
        prompt.len()
    );
    anyhow::ensure!(
        steps >= 2,
        "--gen must be at least 2 to have a stream to compare"
    );
    let cut = prompt.len() * 2 / 3;
    let (head, tail) = prompt.split_at(cut);

    // The COLD pass, which binds every layer's weights. Its seconds are the
    // upload and belong to no arm; every number below is measured after it.
    let t_cold = std::time::Instant::now();
    session.prefill(head)?;
    println!(
        "  cold prefill          : {} tokens in {:.1}s (binds the layers; not a measurement)",
        head.len(),
        t_cold.elapsed().as_secs_f64()
    );

    // One arm: reset, warm prefill of the head, then the delta at `rows` per
    // pass, then `steps - 1` decode steps. Returns the tokens and the two
    // per-token costs, so the caller compares like with like.
    let arm = |s: &mut Session, rows: usize, what: &str| -> Result<(Vec<usize>, f64, f64)> {
        s.reset();
        let t_p = std::time::Instant::now();
        s.prefill(head)?;
        let prefill_s = t_p.elapsed().as_secs_f64();
        s.set_extend_batch(rows)?;
        let t_e = std::time::Instant::now();
        let mut tok = s.extend(tail)?;
        let extend_s = t_e.elapsed().as_secs_f64();
        anyhow::ensure!(
            s.position() == prompt.len(),
            "{what}: appended {} positions but the session stands at {}",
            tail.len(),
            s.position()
        );
        let mut out = vec![tok];
        let t_d = std::time::Instant::now();
        for _ in 1..steps {
            tok = s.step()?;
            out.push(tok);
        }
        let decode_s = t_d.elapsed().as_secs_f64();
        let (pf, ex) = (
            prefill_s * 1e3 / head.len() as f64,
            extend_s * 1e3 / tail.len() as f64,
        );
        println!(
            "  {what:<22}: prefill {} tok in {prefill_s:.3}s ({pf:.2} ms/token, BATCHED) | \
             extend {} tok in {extend_s:.3}s ({ex:.2} ms/token, {rows} rows a pass) | \
             {} steps in {decode_s:.3}s ({:.1} ms/step)",
            head.len(),
            tail.len(),
            steps - 1,
            decode_s * 1e3 / (steps - 1) as f64,
        );
        Ok((out, pf, ex))
    };

    let (walked, _, walked_ms) = arm(session, 1, "WALKED (1 row a pass)")?;
    let (batched, prefill_ms, batched_ms) = arm(session, tail.len(), "BATCHED (one pass)")?;
    let chunk = tail.len().div_ceil(3);
    let (chunked, _, chunked_ms) = arm(session, chunk, "CHUNKED (three passes)")?;

    println!("WALKED : {walked:?}");
    println!("BATCHED: {batched:?}");
    println!("CHUNKED: {chunked:?}");

    let agreement =
        |a: &[usize], b: &[usize]| -> usize { a.iter().zip(b).take_while(|(x, y)| x == y).count() };
    let ab = agreement(&batched, &walked);
    let cb = agreement(&chunked, &batched);
    println!(
        "  batched vs walked     : {ab}/{} tokens{}",
        walked.len(),
        match ab == walked.len() {
            true => String::new(),
            false => format!(
                " -- first divergence at {ab} ({} against {})",
                batched[ab], walked[ab]
            ),
        }
    );
    println!(
        "  chunked vs batched    : {cb}/{} tokens{}",
        batched.len(),
        match cb == batched.len() {
            true => String::new(),
            false => format!(
                " -- first divergence at {cb} ({} against {})",
                chunked[cb], batched[cb]
            ),
        }
    );

    println!(
        "  speedup               : {:.2}x ({walked_ms:.2} -> {batched_ms:.2} ms/token of \
         delta); chunked {chunked_ms:.2}",
        walked_ms / batched_ms.max(f64::EPSILON)
    );
    // The crossover, restated from THIS run's own two numbers rather than
    // carried over: a rewind is worth taking while re-extending the changed
    // suffix costs less than re-prefilling the whole prefix.
    println!(
        "  rewind crossover      : changed suffix < {:.1}% of the prefix (was {:.1}% walked) \
         -- prefill {prefill_ms:.2} ms/token against extend {batched_ms:.2}",
        100.0 * prefill_ms / batched_ms.max(f64::EPSILON),
        100.0 * prefill_ms / walked_ms.max(f64::EPSILON),
    );

    anyhow::ensure!(
        ab == walked.len(),
        "a batched delta diverges from a walked one at token {ab}"
    );
    anyhow::ensure!(
        cb == batched.len(),
        "a delta split into three passes diverges from one pass at token {cb}"
    );
    println!("BATCHED EXTEND GATE: PASS");
    Ok(())
}

/// The CARRY claim, on the real model: **everything a turn said is in the cache
/// when the next turn starts.**
///
/// # The schedule this is about, and the token it used to lose
///
/// A served turn generates by feeding each token back as the next one's input,
/// and it stops one step short of that: the last token is EMITTED and never
/// fed, because a decode step whose argmax nobody will read is 44 ms spent on
/// nothing. That is `inkling_serve`'s `if step + 1 < want`, and it is a good
/// saving — but it leaves the turn's own final token in the STREAM and not in
/// the KV CACHE, which is correct only if the NEXT pass appends it.
///
/// Until 2026-08-27 the next pass did not. `Session::extend` appends exactly the
/// ids it is handed, and the token a pass produced is held in a private field
/// that only `Session::step` ever reads — so a turn that ended and was followed
/// by `extend(new context)` dropped its own last word, permanently, once per
/// turn, in every conversation. Nothing caught it: the cache stayed CONSISTENT,
/// `position()` stayed exactly `prompt + fed`, and every length assertion in
/// this file still passed. It was one token short of the sequence it claimed to
/// hold, and a length cannot see that. This gate is the check that would have.
///
/// # What it runs
///
/// The prompt is split the way [`rewind_gate`] and [`batched_gate`] split it, so
/// the three gates' numbers compose: two thirds the settled prefix a first turn
/// prefills, one third the DELTA a second turn is handed. Three arms against
/// ONE loaded session, each starting from a `reset`:
///
/// ```text
///   turn one, every arm : prefill(head); generate `steps` on the SERVE
///                         SCHEDULE -> said[0..steps], of which said[steps-1]
///                         is emitted and NOT in the cache
///   DROPPED : extend(delta)                     then generate   <- as it was
///   CARRIED : extend([said.last()] ++ delta)    then generate   <- as it is
///   WHOLE   : reset; prefill(head ++ said ++ delta); generate   <- the reference
/// ```
///
/// `WHOLE` is the whole point: a session fed the IDENTICAL TOKEN SEQUENCE IN ONE
/// PASS, which is what a served conversation claims to be. So this gate is
/// behavioural — it never reads a length and never reaches inside a `Session`,
/// it asks the model what comes next and compares three answers.
///
/// The turn-one schedule below is copied from `inkling_serve`'s generation loop
/// rather than approximated, guard and all. If one of the two changes the other
/// must: a gate that generated on a different schedule would be checking a
/// conversation nobody has.
///
/// # What it asserts, and what it only reports
///
/// It ASSERTS `CARRIED == WHOLE`: the served schedule answers what the one-pass
/// session answers. That is the regression guard, and it is the property the
/// defect broke.
///
/// It REPORTS `DROPPED vs WHOLE`, because that comparison is EVIDENCE and not an
/// invariant. Whether one missing token of context flips a greedy argmax is a
/// property of this model and this prompt, not of the defect; a run in which
/// DROPPED also agrees has proven nothing about the bug and says so in those
/// words — it has not shown the bug absent.
///
/// Exactness between `CARRIED` and `WHOLE` is expected for [`batched_gate`]'s
/// reason and carries its caveat. Both build forward and neither re-opens a
/// settled page, so this is not the rewind's situation; but they partition the
/// same tokens into passes differently — one prefill plus `steps - 1` single
/// rows plus one batch, against one prefill — and a pass width changes a GEMM
/// shape and therefore an accumulation order. A LATE divergence is arithmetic,
/// and is reported as where it starts rather than as a verdict on token 0.
fn carry_gate(session: &mut Session, prompt: &[usize], steps: usize) -> Result<()> {
    anyhow::ensure!(
        prompt.len() >= 6,
        "the carry gate splits the prompt in two; {} tokens is not enough",
        prompt.len()
    );
    anyhow::ensure!(
        steps >= 2,
        "--gen must be at least 2: with one token a turn's first and last token are the same \
         one, and the schedule this gate is about has no interior to get right"
    );
    let cut = prompt.len() * 2 / 3;
    let (head, delta) = prompt.split_at(cut);

    // The COLD pass, which binds every layer's weights. Its seconds belong to no
    // arm; every arm starts from a `reset` after it.
    let t_cold = std::time::Instant::now();
    session.prefill(head)?;
    println!(
        "  cold prefill          : {} tokens in {:.1}s (binds the layers; not a measurement)",
        head.len(),
        t_cold.elapsed().as_secs_f64()
    );

    // TURN ONE, on the serving process's own schedule: emit `steps` tokens and
    // feed back `steps - 1` of them.
    let turn_one = |s: &mut Session| -> Result<Vec<usize>> {
        s.reset();
        let mut tok = s.prefill(head)?;
        let mut said = Vec::with_capacity(steps);
        for step in 0..steps {
            said.push(tok);
            // `serve_turn`'s guard, verbatim: the last emitted token is not fed
            // back, because its successor would be generated and thrown away.
            if step + 1 < steps {
                tok = s.step()?;
            }
        }
        Ok(said)
    };
    // TURN TWO: attend to what is new, then generate. Every arm makes the same
    // call; the arms differ only in what "new" means.
    let turn_two = |s: &mut Session, ids: &[usize]| -> Result<Vec<usize>> {
        let mut tok = s.extend(ids)?;
        let mut out = vec![tok];
        for _ in 1..steps {
            tok = s.step()?;
            out.push(tok);
        }
        Ok(out)
    };

    // ── DROPPED: the schedule as it was ─────────────────────────────────────
    let said = turn_one(session)?;
    println!(
        "  turn one              : said {said:?} — {} token(s) emitted, {} of them fed back, \
         so the session stands at {} = {} prompt + {} fed",
        said.len(),
        steps - 1,
        session.position(),
        head.len(),
        steps - 1,
    );
    let t_d = std::time::Instant::now();
    let dropped = turn_two(session, delta)?;
    let dropped_s = t_d.elapsed().as_secs_f64();

    // ── CARRIED: the same conversation, with the pending token at the head of
    // the delta. One extra ROW on a pass the turn was making anyway.
    let again = turn_one(session)?;
    anyhow::ensure!(
        again == said,
        "the same prefill and the same schedule produced two different turns ({again:?} against \
         {said:?}); this gate compares arms by replaying turn one and cannot do that without \
         greedy determinism"
    );
    let with_carry: Vec<usize> =
        std::iter::once(*said.last().expect("a turn emits at least one token"))
            .chain(delta.iter().copied())
            .collect();
    let t_c = std::time::Instant::now();
    let fixed = turn_two(session, &with_carry)?;
    let fixed_s = t_c.elapsed().as_secs_f64();

    // ── WHOLE: the reference, one pass over the identical token sequence ─────
    session.reset();
    let whole_ids: Vec<usize> = head
        .iter()
        .chain(said.iter())
        .chain(delta.iter())
        .copied()
        .collect();
    let mut tok = session.prefill(&whole_ids)?;
    let mut whole = vec![tok];
    for _ in 1..steps {
        tok = session.step()?;
        whole.push(tok);
    }

    // ── the report ──────────────────────────────────────────────────────────
    println!(
        "  the sequence          : {} head + {} said + {} delta = {} tokens",
        head.len(),
        said.len(),
        delta.len(),
        whole_ids.len()
    );
    println!("DROPPED: {dropped:?}");
    println!("CARRIED: {fixed:?}");
    println!("WHOLE  : {whole:?}");

    let agreement =
        |a: &[usize], b: &[usize]| -> usize { a.iter().zip(b).take_while(|(x, y)| x == y).count() };
    let dw = agreement(&dropped, &whole);
    let cw = agreement(&fixed, &whole);
    println!(
        "  DROPPED vs WHOLE      : {dw}/{} tokens{}",
        whole.len(),
        match dw == whole.len() {
            true => " — AGREED. One missing token of context did not flip a greedy argmax at this \
                 prompt and this length, so THIS RUN has proven nothing about the defect. It \
                 has not shown it absent."
                .to_string(),
            false => format!(
                " — diverges at {dw} ({} against {}), which is the missing token showing up in \
                 the only place it can: the answer",
                dropped[dw], whole[dw]
            ),
        }
    );
    println!(
        "  CARRIED vs WHOLE      : {cw}/{} tokens{}",
        whole.len(),
        match cw == whole.len() {
            true => String::new(),
            false => format!(" — diverges at {cw} ({} against {})", fixed[cw], whole[cw]),
        }
    );
    // FRAMING: seconds for one TURN TWO -- one `extend` pass over the delta plus
    // `steps - 1` decode steps -- at the layer range this session runs, on one
    // box, against each other and nothing else. The carried arm's delta is one
    // token wider and is the same single batched pass, which is the claim.
    println!(
        "  what the carry cost   : turn two {dropped_s:.3}s dropped against {fixed_s:.3}s \
         carried ({} vs {} delta tokens, {} decode steps each) — the carried token is a ROW of \
         a batch the turn was making anyway, not a decode step",
        delta.len(),
        with_carry.len(),
        steps - 1,
    );

    anyhow::ensure!(
        cw == whole.len(),
        "a served turn that carries its last token forward diverges at token {cw} from the \
         session fed the identical sequence in one pass — so a served conversation is not the \
         sequence it claims to be"
    );
    println!("CARRY GATE: PASS");
    Ok(())
}
