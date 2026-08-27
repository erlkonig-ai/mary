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
/// The `extend` row is **historical**: it was measured while `Session::extend`
/// walked its delta, and it does not walk any more. What this gate prints for
/// that row now is the batched figure ([`batched_gate`] has the table), so the
/// numbers above are kept as what the argument below was built from rather than
/// as what a re-run reports.
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
/// counters. What was not free is what came after it: `Session::extend` walked
/// the delta ONE POSITION AT A TIME, so a re-extended token cost a DECODE step
/// where a prefilled one costs a batched pass — 47.4 against 11.32 ms at layers
/// 0..21, a factor of **4.19**. So against the honest alternative (`reset` +
/// one batched `prefill` of the whole new prefix) a rewind was a win only while
/// the CHANGED SUFFIX was under **23.9%** of the prefix, and a loss above it.
///
/// That crossover was a property of `extend` rather than of the rewind, and it
/// has since moved: `extend` batches (see its own doc), and [`batched_gate`]
/// measures the same 320-token delta at **10.97 ms/token** against a warm
/// batched prefill's 10.88 at the same layer range. **The crossover is 99.2%**
/// — a rewind is now worth taking for any change short of the whole prompt, and
/// the number above is kept only because it is what this gate measured on the
/// day and what the argument was built from.
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
            "  extend {what:<14}: {} tokens in {extended:.3}s ({:.1} ms/token, BATCHED), then {} \
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
/// each run TWICE. The third arm is not decoration: a delta longer than
/// `extend_batch` is appended in consecutive passes, and the claim that a chunk
/// boundary is a COMMIT POINT AND NOTHING ELSE — which is what makes
/// `extend_batch` a resource knob rather than a semantic one — is only true if
/// a delta split three ways leaves the cache one pass leaves.
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
/// **Exactness across the arms is NOT the bar here, and expecting it would be a
/// mistake about what they are.** The three arms hand the same GEMMs three
/// different `m` — 1, 320 and 107 — so they are three accumulation orders over
/// the same values, and this model's router picks six of 256 experts on a
/// margin a last-place logit bit can flip. That is the same class of difference
/// `AttnCache::reserve_kv` records between a merged page and a split one.
/// [`rewind_gate`] could demand the whole stream because both of ITS arms
/// walked; this one cannot, and asserts instead on the FIRST token — the one
/// computed from the delta's last row against the cache the delta just built,
/// which a short cache or a mis-taken convolution history moves and a rounding
/// difference does not. Each arm must also agree with ITSELF across its two
/// reps, which is what rules out reading a buffer nobody wrote.
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
///
/// # What it measured, 2026-08-27, and the framing rule
///
/// One GB10 (`spark`), advisory box lock held, release build, features
/// `inkling-cuda`, `inkling-small-complete.pile`, the same 960-token prompt
/// [`rewind_gate`] uses split the same 640/320, `--gen 8`, greedy, page cache
/// dropped before the run, no `INK_*` switch but the layer range. **Every
/// figure is per TOKEN OF THE 320-TOKEN DELTA**, on that box, for that layer
/// range — not per turn, not per prefilled token, and not for the whole
/// 42-layer model. "rep 2" is the second run of that arm in the same process:
/// the width's kernels are compiled and `target/autotune/` has an entry.
///
/// | per token of the delta | layers 0..6 | layers 0..21 |
/// |---|---|---|
/// | `extend`, WALKED — 1 row a pass | 19.42 ms | 46.92 ms |
/// | `extend`, BATCHED — 320 rows, one pass | **3.42 ms** | **10.97 ms** |
/// | `extend`, CHUNKED — 107 rows, three passes | 3.55 ms | 11.28 ms |
/// | `prefill`, warm, batched, 640 rows | 3.38 ms | 10.88 ms |
/// | decode step | 19.2 ms/step | 47.1 ms/step |
/// | speedup, walked → batched | **5.68x** | **4.28x** |
/// | rewind crossover | 17.4% → **98.9%** | 23.2% → **99.2%** |
///
/// The finding is the last two rows together. A batched append costs what a
/// PREFILLED token costs — 10.97 against 10.88 at layers 0..21, within the
/// spread — which is the ceiling, because a prefill is the same rows through
/// the same GEMMs with no cache to read. So re-extending a changed suffix is no
/// longer a different price from re-reading the prefix, and the rewind
/// crossover moves from a quarter of the prefix to essentially all of it: a
/// rewind is now worth taking for any change that is not the whole prompt.
///
/// **The first pass at a WIDTH costs extra, and it is a one-off.** cubecl keys
/// compiled kernels on shape and burn's matmul autotune keys its choice the
/// same way, so rep 1 of an arm pays for its width and rep 2 does not: 11.66
/// against 10.97 (one new width) and 14.37 against 11.28 (two, because
/// `div_ceil` splits 320 into 107+107+106) at layers 0..21. An earlier run on a
/// COLD `target/autotune/` measured that chunked arm at 47.91 ms/token — 10.7 s
/// over the delta — against 4.6 s for the same arm once the cache had entries
/// for those widths, and that gap is not further isolated here. It is recorded
/// because the autotune cache is on disk and per worktree, so a fresh checkout
/// pays it once per width and a running process pays it never again — and
/// because it is the reason `extend_batch` defaults wide enough that a
/// conversational delta is ONE width rather than two.
///
/// The token streams agreed 8/8 across all three arms at layers 0..6 and parted
/// after 3 at layers 0..21 — which is the reading above, from the other side:
/// the arms' arithmetic differs in the last bits, six MoE layers do not amplify
/// it into a different expert and twenty-one do.
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

    // One REP of one arm: reset, warm prefill of the head, then the delta at
    // `rows` per pass, then `steps - 1` decode steps. Returns the tokens and
    // the two per-token costs, so the caller compares like with like.
    let rep = |s: &mut Session, rows: usize| -> Result<(Vec<usize>, f64, f64, f64)> {
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
            "appended {} positions but the session stands at {}",
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
        Ok((
            out,
            prefill_s * 1e3 / head.len() as f64,
            extend_s * 1e3 / tail.len() as f64,
            decode_s * 1e3 / (steps - 1) as f64,
        ))
    };

    // TWO REPS PER ARM, and the difference between them is a measurement rather
    // than noise.
    //
    // cubecl keys its compiled kernels on the shapes it is handed, and a pass
    // of `rows` rows hands every GEMM in the stack a shape it has not seen
    // unless a pass of exactly that width has already run. So rep 1 of an arm
    // carries a one-off COMPILATION of that width and rep 2 does not, and the
    // gap between them is what that compilation costs. Reporting only rep 1
    // would charge a serving process a price it pays once; reporting only rep 2
    // would hide a price it pays at all. `AttnCache::attention_step` records
    // the same effect on the KV length axis, where the fix was a shape bucket.
    //
    // Rep 1 and rep 2 must also produce the SAME TOKENS. That is the
    // determinism check, and it is the one this comparison would be worthless
    // without: two arms that disagree could be disagreeing because one of them
    // is reading a buffer nobody wrote, and an arm that does not even agree
    // with itself cannot be evidence about the other.
    let mut arm = |s: &mut Session, rows: usize, what: &str| -> Result<(Vec<usize>, f64, f64)> {
        let (first, pf1, ex1, dec1) = rep(s, rows)?;
        let (second, pf2, ex2, dec2) = rep(s, rows)?;
        println!(
            "  {what:<22}: extend {} tok at {rows} rows a pass -- rep1 {ex1:.2} ms/token (first \
             pass at this width: compiles it), rep2 {ex2:.2} ms/token (warm) | prefill \
             {pf1:.2}/{pf2:.2} ms/token BATCHED | decode {dec1:.1}/{dec2:.1} ms/step",
            tail.len(),
        );
        anyhow::ensure!(
            first == second,
            "{what} is not deterministic: {first:?} then {second:?}"
        );
        Ok((second, pf2, ex2))
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

    // WHAT IS ASSERTED, and why it is not the whole stream.
    //
    // The three arms hand the same GEMMs three different `m`, so they are three
    // different accumulation orders over the same values -- the same class of
    // difference `AttnCache::reserve_kv` documents between a merged page and a
    // split one, and `rewind_to` between a truncated store and the run that
    // first passed through it. This model's router picks six experts of 256 on
    // a margin a last-place logit bit can flip, so a stream that agrees for a
    // few tokens and then parts company is that, and it is EXPECTED here in a
    // way it was not for `rewind_gate` (whose two arms both walked).
    //
    // So the bar is the first token, and it is not a weak one: it is the token
    // computed from the delta's last row against the cache the delta just
    // built, so a layer left short of keys, a convolution history taken from
    // the wrong row, or a batch that skipped a trim moves it. What CANNOT be
    // settled here is settled where it can be -- `walked_vs_batched` and
    // `compare_batched` in `models::inkling::burn`, which hold the batched
    // cache to a walked one and to the uncached lane at a tolerance, on
    // synthetic weights with no router to amplify a last bit.
    anyhow::ensure!(
        ab >= 1,
        "a batched delta and a walked one disagree on the FIRST token ({} against {}): that is \
         the token computed from the delta's last row against the cache it just built, and it \
         does not move for a rounding difference",
        batched[0],
        walked[0]
    );
    anyhow::ensure!(
        cb >= 1,
        "a delta split into three passes and the same delta in one pass disagree on the FIRST \
         token ({} against {}): a chunk boundary is supposed to be a commit point and nothing \
         else",
        chunked[0],
        batched[0]
    );
    println!(
        "BATCHED EXTEND GATE: PASS (first token agrees across all three arms; {ab} and {cb} \
         tokens of agreement before the accumulation orders part)"
    );
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
/// turn, in every conversation that had anything new to say to it. Nothing
/// caught it: the cache stayed CONSISTENT, `position()` stayed exactly
/// `prompt + fed`, and every length assertion in this file still passed. It was
/// one token short of the sequence it claimed to hold, and a length cannot see
/// that. This gate is the check that would have.
///
/// # Why every arm walks, and why that is the whole design
///
/// The obvious reference is a session handed `head ++ said ++ delta` in one
/// batched `prefill`. **It does not work, and the failure is not small.** A
/// prefill of `n` rows and a walk over the same `n` positions are different GEMM
/// shapes and therefore different accumulation orders, and this model's router
/// picks six experts of 256 on margins a last-place logit bit can flip. Measured
/// here at layers 0..21 on 2026-08-27, a one-pass prefill parted company with
/// the walked build of the identical 136 tokens at token 1 of 8 — by itself,
/// with nothing missing. A gate whose reference moves that much cannot attribute
/// a divergence to anything.
///
/// So every arm that is compared runs at `extend_batch = 1`, and that makes the
/// comparison EXACT rather than approximate. [`Session::step`] is
/// `forward(&[last])`, a one-row `extend` is `forward(&[id])` — the same call —
/// so a reference that walks `said ++ delta` through `extend` makes literally
/// the same sequence of forward passes as a served turn that steps through
/// `said` and then extends: same widths, same shapes, same order of summation.
/// The only thing left that can differ is WHICH TOKENS, which is the thing under
/// test. Exact equality is therefore an invariant here and not a hope.
///
/// # What it runs
///
/// The prompt is split the way [`rewind_gate`] and [`batched_gate`] split it, so
/// the three gates' numbers compose: two thirds the settled prefix a first turn
/// prefills, one third the DELTA a second turn is handed. Five arms against ONE
/// loaded session, each from a `reset`; `turn one` is the serving process's own
/// generation loop, guard and all, copied rather than approximated.
///
/// ```text
///   turn one   : prefill(head); generate `steps` -> said[0..steps], of which
///                said[steps-1] is emitted and NOT fed back
///   DROPPED    : turn one; extend(delta)                    ; generate
///   CARRIED    : turn one; extend([said.last()] ++ delta)   ; generate
///   WHOLE      : prefill(head); extend(said ++ delta)       ; generate
///   ONE-PASS   : prefill(head ++ said ++ delta)             ; generate
///   BATCHED    : turn one; extend([said.last()] ++ delta) in ONE pass; generate
/// ```
///
/// `WHOLE` is the reference: a session fed the identical token sequence, one
/// position at a time, which is the same pass partition `CARRIED` uses.
/// `ONE-PASS` is the reference that looks right and is not, kept as a CONTROL so
/// the run says in its own numbers how far repartitioning alone moves the
/// answer. `BATCHED` is the shape that actually ships — the carried token as one
/// extra ROW of a pass the turn was making anyway — and it is reported against
/// `WHOLE` under the same caveat as `ONE-PASS`.
///
/// # What it asserts, and what it only reports
///
/// It ASSERTS `CARRIED == WHOLE`, exactly, for the reason above. That is the
/// regression guard and it is the property the defect broke.
///
/// It REPORTS `DROPPED vs WHOLE`, because that comparison is EVIDENCE and not an
/// invariant: whether one missing token of context flips a greedy argmax is a
/// property of this model and this prompt. A run in which DROPPED also agrees
/// has proven nothing about the defect and says so in those words — it has not
/// shown it absent.
///
/// # What it measured, 2026-08-28, and the framing rule
///
/// One GB10 (`spark2`), advisory box lock held, release build, features
/// `inkling-cuda,cuda-backend,import`, `work-inkling-complete.pile`, layers
/// **0..21 of 42** (a PARTIAL STACK, so the tokens are diagnostic and only their
/// AGREEMENT means anything), greedy, `--gen 8`, one sample an arm, no `INK_*`
/// switch but the layer range. Agreement is counted as the length of the
/// matching PREFIX of the eight generated tokens; seconds are for the APPEND —
/// the `extend` call and nothing else — stated per PASS, not per token and not
/// per turn.
///
/// | prompt | head/delta | CARRIED vs WHOLE | DROPPED vs WHOLE | ONE-PASS vs WHOLE | BATCHED vs WHOLE |
/// |---|---|---|---|---|---|
/// | 10 tok | 6/4 | **8/8** | 0/8 | 7/8 | 1/8 |
/// | 128 tok | 85/43 | **8/8** | 1/8 | 1/8 | 8/8 |
/// | 1024 tok | 682/342 | **8/8** | 2/8 | 2/8 | 2/8 |
///
/// Read the first column as the invariant and the rest as context for it.
/// CARRIED is EXACT at every length, which is what an identical pass partition
/// over identical tokens has to give. DROPPED shares that partition and differs
/// only by the missing token, and it disagreed at every length — earlier the
/// less other context there was to dilute it, which is the dependence that makes
/// it a report. ONE-PASS and BATCHED disagree by REPARTITIONING ALONE, on the
/// same tokens, at the same order of magnitude — which is exactly why neither
/// may be the reference, and why the first version of this gate (which used
/// ONE-PASS) could not have told a dropped token from a wider GEMM.
///
/// The appends, same run, per pass: at 1024 tokens a walked 343-row append cost
/// **16.487 s** and the same rows in ONE pass cost **4.716 s** (13.7 ms a row),
/// so the carried token rides at the head of a batch for a small fraction of the
/// ~47 ms decode step that feeding it through [`Session::step`] would have cost.
/// At 44 rows the batched pass was SLOWER than the walk (2.871 s against
/// 1.962 s, 65.3 against 44.6 ms a row): the width has a fixed cost that only
/// long deltas amortise. That is a property of the batched append and not of the
/// carry — one sample an arm, so it is a flag for [`batched_gate`] and not a
/// measurement of anything.
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
    // call and the arms differ only in what "new" means. The seconds are the
    // APPEND's alone, which is the half of a turn the carry can change.
    let turn_two = |s: &mut Session, ids: &[usize]| -> Result<(Vec<usize>, f64)> {
        let t = std::time::Instant::now();
        let mut tok = s.extend(ids)?;
        let appended = t.elapsed().as_secs_f64();
        let mut out = vec![tok];
        for _ in 1..steps {
            tok = s.step()?;
            out.push(tok);
        }
        Ok((out, appended))
    };

    // One row a pass, so that a `step` and an `extend` are the same call and the
    // reference arm's partition is the served arm's. See this function's doc.
    session.set_extend_batch(1)?;

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
    let (dropped, dropped_s) = turn_two(session, delta)?;

    // ── CARRIED: the same conversation with the pending token at the head of
    // the delta.
    let again = turn_one(session)?;
    anyhow::ensure!(
        again == said,
        "the same prefill and the same schedule produced two different turns ({again:?} against \
         {said:?}); this gate compares arms by replaying turn one and cannot do that without \
         greedy determinism"
    );
    let carry = *said.last().expect("a turn emits at least one token");
    let with_carry: Vec<usize> = std::iter::once(carry)
        .chain(delta.iter().copied())
        .collect();
    let (fixed, fixed_s) = turn_two(session, &with_carry)?;

    // ── WHOLE: the reference. Same prefill, then every token of the sequence
    // walked through `extend` -- which is the same forward call `step` makes,
    // over the same ids, in the same order.
    session.reset();
    session.prefill(head)?;
    let forced: Vec<usize> = said.iter().chain(delta.iter()).copied().collect();
    let (whole, whole_s) = turn_two(session, &forced)?;

    // ── ONE-PASS: the reference that looks right. Kept as a CONTROL: whatever
    // it disagrees with `WHOLE` about is repartitioning and not context.
    session.reset();
    let one_pass_ids: Vec<usize> = head.iter().chain(forced.iter()).copied().collect();
    let mut tok = session.prefill(&one_pass_ids)?;
    let mut one_pass = vec![tok];
    for _ in 1..steps {
        tok = session.step()?;
        one_pass.push(tok);
    }

    // ── BATCHED: the shape that ships. The carried token is row 0 of a pass the
    // turn was making anyway.
    let again = turn_one(session)?;
    anyhow::ensure!(again == said, "turn one drifted between arms");
    session.set_extend_batch(with_carry.len())?;
    let (batched, batched_s) = turn_two(session, &with_carry)?;

    // ── the report ──────────────────────────────────────────────────────────
    println!(
        "  the sequence          : {} head + {} said + {} delta = {} tokens; the carried token \
         is {carry}",
        head.len(),
        said.len(),
        delta.len(),
        one_pass_ids.len()
    );
    println!("DROPPED : {dropped:?}");
    println!("CARRIED : {fixed:?}");
    println!("WHOLE   : {whole:?}");
    println!("ONE-PASS: {one_pass:?}");
    println!("BATCHED : {batched:?}");

    let agreement =
        |a: &[usize], b: &[usize]| -> usize { a.iter().zip(b).take_while(|(x, y)| x == y).count() };
    let line = |what: &str, a: &[usize], b: &[usize], note: &str| {
        let n = agreement(a, b);
        println!(
            "  {what:<22}: {n}/{} tokens{}",
            b.len(),
            match n == b.len() {
                true => String::new(),
                false => format!(" — diverges at {n} ({} against {}){note}", a[n], b[n]),
            }
        );
        n
    };
    let cw = line("CARRIED vs WHOLE", &fixed, &whole, "");
    let dw = line(
        "DROPPED vs WHOLE",
        &dropped,
        &whole,
        ", which is the missing token showing up in the only place it can: the answer",
    );
    if dw == whole.len() {
        println!(
            "      ^ AGREED. One missing token of context did not flip a greedy argmax at this \
             prompt and this length, so THIS RUN has proven nothing about the defect. It has \
             not shown it absent."
        );
    }
    line(
        "ONE-PASS vs WHOLE",
        &one_pass,
        &whole,
        " — REPARTITIONING ALONE, same tokens; this is the scale a DROPPED divergence has to be \
         read against",
    );
    line(
        "BATCHED vs WHOLE",
        &batched,
        &whole,
        " — the served shape against the walked reference, same tokens; repartitioning again",
    );

    // FRAMING: seconds for the APPEND of one turn two -- the `extend` call and
    // nothing else -- at the layer range this session runs, on one box, one
    // sample an arm. Per PASS, not per token, and not per turn.
    println!(
        "  what the append cost  : DROPPED {dropped_s:.3}s / {} tok walked | CARRIED \
         {fixed_s:.3}s / {} tok walked | WHOLE {whole_s:.3}s / {} tok walked | BATCHED \
         {batched_s:.3}s / {} tok in ONE pass",
        delta.len(),
        with_carry.len(),
        forced.len(),
        with_carry.len(),
    );
    println!(
        "  what the carry cost   : one extra ROW of a pass the turn was making anyway — \
         {batched_s:.3}s for {} rows, {:.1} ms a row — against a whole DECODE STEP, which is \
         what feeding it back through `step` would have cost instead",
        with_carry.len(),
        1e3 * batched_s / with_carry.len() as f64,
    );

    anyhow::ensure!(
        cw == whole.len(),
        "a served turn that carries its last token forward diverges at token {cw} from the \
         session fed the identical sequence one position at a time — same partition, same \
         shapes, so this is the tokens and nothing else"
    );
    println!("CARRY GATE: PASS");
    Ok(())
}
