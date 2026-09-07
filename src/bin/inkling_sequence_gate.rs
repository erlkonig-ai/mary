//! Opt-in CUDA gate for independent resident conversations.
//!
//! Run only under the shared GPU reservation:
//!   inkling_sequence_gate MODEL.pile [--config config.json] [--layers 1]
//!   inkling_sequence_gate MODEL.pile --layers 3 --teacher-bank
//!
//! Prerequisites: a readable current Inkling model pile with config.json (or
//! explicit --config), embedding/norm/unembedding, and layers 0..layers. The
//! default loads ONE decoder layer, including attention and all four short
//! convolution histories. --layers 3 also exercises routed expert packing.
//! The hard maximum is three layers: this never loads the full single-box model.
//! CUDA must support the existing BF16/NVFP4 kernels. No tokenizer, network,
//! teacher model, new pile, training, or model-output quality claim is involved.
//! INK_TP and INK_LEARN_LR must be unset; the CLI explicitly bounds INK_LAYERS.
//!
//! Both arms start from the same real cache checkpoints. Serial isolated
//! one-token forwards are compared with a B3 active-plus-two-capsule forward at
//! unequal positions. A common next token makes each row's distinct history,
//! rather than its current embedding, the distinguishing input. The gate prints
//! numerical gaps, permits rounding differences, and requires each batch row to
//! be much nearer its own history's output than another history's output.
//! Reset, checkpoint identity, and foreground restoration are checked exactly.
//!
//! --teacher-bank adds a small synthetic zero routed bank, not an EMA oracle:
//! it checks mixed-bank refusal and foreground isolation across a teacher pass.
//! Its maximum 64 MiB is explicitly reserved before loading weights.

use std::path::PathBuf;

use anyhow::{Context, Result, ensure};
use clap::Parser;
use mary::models::inkling::assembly::{bytes_of, down};
use mary::models::inkling::devplan::ExpertTable;
use mary::models::inkling::seam;
use mary::models::inkling::session::{Session, SessionConfig, Sequence};

const EXTRA_BANK_BYTES: u64 = 64 * 1024 * 1024;
const GATE_WORKSPACE_BYTES: u64 = 64 * 1024 * 1024;

#[derive(Parser)]
#[command(about = "Partial-stack CUDA gate for independent resident sequence state")]
struct Args {
    pile: PathBuf,
    #[arg(long)]
    config: Option<PathBuf>,
    #[arg(long, default_value_t = 1)]
    layers: usize,
    #[arg(long, default_value_t = 4)]
    rounds: usize,
    /// Maximum absolute logit gap divided by the reference's maximum magnitude.
    #[arg(long, default_value_t = 0.02)]
    max_scaled_error: f64,
    /// Also check synthetic explicit-bank isolation; requires --layers 3.
    #[arg(long)]
    teacher_bank: bool,
}

fn ids(vocab: usize, length: usize, seed: usize) -> Vec<usize> {
    (0..length).map(|i| 17 + (seed + i * 37) % (vocab.min(1024) - 17)).collect()
}

fn foreground(session: &Session) -> (usize, Option<usize>) {
    (session.position(), session.next_token())
}

fn seed_capsule(session: &mut Session, sequence: &mut Sequence, prompt: &[usize]) -> Result<()> {
    let before = foreground(session);
    session.with_sequence(sequence, |session| {
        let _ = session.extend_logits(prompt)?;
        session.set_next_token(17)?;
        session.validate_cache_completeness()?;
        Ok(())
    })?;
    ensure!(foreground(session) == before, "prefilling a capsule changed the foreground");
    Ok(())
}

fn isolated(session: &mut Session, sequence: &mut Sequence, token: usize) -> Result<Vec<f32>> {
    let before = foreground(session);
    let logits = session.with_sequence(sequence, |session| {
        Ok(down(session.extend_logits(&[token])?))
    })?;
    ensure!(foreground(session) == before, "running a capsule changed foreground position/last");
    Ok(logits)
}

fn distance(a: &[f32], b: &[f32]) -> f64 {
    let mut sum = 0.0;
    let mut count = 0;
    for (&x, &y) in a.iter().zip(b) {
        if x == f32::NEG_INFINITY && y == f32::NEG_INFINITY { continue; }
        if !x.is_finite() || !y.is_finite() { return f64::INFINITY; }
        sum += (f64::from(x) - f64::from(y)).powi(2);
        count += 1;
    }
    if count == 0 { f64::INFINITY } else { (sum / count as f64).sqrt() }
}

fn compare(label: &str, actual: &[f32], expected: &[f32], limit: f64) -> Result<f64> {
    ensure!(!actual.is_empty() && actual.len() == expected.len(), "{label}: logit shape mismatch");
    ensure!(actual.iter().zip(expected).all(|(&a, &b)|
        (a.is_finite() && b.is_finite()) || (a == f32::NEG_INFINITY && b == f32::NEG_INFINITY)),
        "{label}: nonfinite logits or different forbidden-token masks");
    ensure!(expected.iter().any(|x| x.is_finite()), "{label}: every token is forbidden");
    let scale = expected.iter().filter(|x| x.is_finite()).map(|x| f64::from(x.abs())).fold(0.0f64, f64::max);
    let max_abs = actual.iter().zip(expected)
        .filter(|(_, b)| b.is_finite())
        .map(|(&a, &b)| (f64::from(a) - f64::from(b)).abs()).fold(0.0f64, f64::max);
    let scaled = max_abs / scale.max(1e-12);
    let rms = distance(actual, expected);
    println!("{label}: max_abs={max_abs:.6e} scaled={scaled:.6e} rms={rms:.6e}");
    ensure!(scaled <= limit, "{label}: scaled logit gap {scaled} exceeds {limit}");
    Ok(rms)
}

fn check_positions(session: &mut Session, b: &mut Sequence, c: &mut Sequence, want: [usize; 3]) -> Result<()> {
    ensure!([session.position(), b.position(), c.position()] == want, "sequence positions crossed");
    ensure!(!b.is_poisoned() && !c.is_poisoned(), "a successful batch poisoned a capsule");
    let live = foreground(session);
    session.validate_cache_completeness()?;
    for sequence in [b, c] {
        session.with_sequence(sequence, |session| session.validate_cache_completeness().map(|_| ()))?;
    }
    ensure!(foreground(session) == live, "checking a capsule changed foreground state");
    Ok(())
}

fn round(session: &mut Session, b: &mut Sequence, c: &mut Sequence, token: usize, label: &str, limit: f64) -> Result<()> {
    let positions = [session.position(), b.position(), c.position()];
    let identities = [b.identity(), c.identity()];
    let live = foreground(session);
    let ca = session.checkpoint()?;
    let cb = session.with_sequence(b, |session| session.checkpoint())?;
    let cc = session.with_sequence(c, |session| session.checkpoint())?;
    ensure!(foreground(session) == live, "taking capsule checkpoints changed foreground state");

    let expected = [down(session.extend_logits(&[token])?), isolated(session, b, token)?, isolated(session, c, token)?];
    session.rewind(&ca)?;
    session.with_sequence(b, |session| session.rewind(&cb))?;
    session.with_sequence(c, |session| session.rewind(&cc))?;
    ensure!(foreground(session) == live, "rewind did not restore foreground position/last");
    check_positions(session, b, c, positions)?;

    let actual = down(session.batch_logits(Some(token), &mut [&mut *b, &mut *c], &[token, token])?);
    let width = expected[0].len();
    ensure!(actual.len() == 3 * width, "batch did not return active-first B3 logits");
    for row in 0..3 {
        let rms = compare(&format!("{label}/row{row}"), &actual[row * width..(row + 1) * width], &expected[row], limit)?;
        let separation = (0..3).filter(|&other| other != row)
            .map(|other| distance(&expected[row], &expected[other])).fold(f64::INFINITY, f64::min);
        ensure!(separation > 1e-7, "{label}: histories are indistinguishable at the head; gate is inconclusive");
        println!("{label}/row{row}: nearest other-history rms={separation:.6e}");
        ensure!(rms < 0.25 * separation,
            "{label}/row{row}: own-history error is not small relative to different-history separation");
    }
    ensure!([b.identity(), c.identity()] == identities, "batch replaced a sequence identity");
    check_positions(session, b, c, positions.map(|p| p + 1))?;
    session.set_next_token(token)?;
    session.set_sequence_next_token(b, token)?;
    session.set_sequence_next_token(c, token)?;
    Ok(())
}

fn reset_shadow(session: &mut Session, c: &mut Sequence, vocab: usize) -> Result<()> {
    let live = foreground(session);
    let old_id = c.identity();
    let old = session.with_sequence(c, |session| session.checkpoint())?;
    c.reset();
    ensure!(c.position() == 0 && c.last_token().is_none() && !c.is_poisoned() && c.identity() != old_id,
        "reset did not clear just the selected sequence");
    let error = session.with_sequence(c, |session| session.rewind(&old)).err()
        .context("a reset capsule accepted its old checkpoint")?;
    ensure!(error.to_string().contains("different sequence"), "unexpected old-checkpoint error: {error:#}");
    ensure!(foreground(session) == live, "reset/refused rewind changed the foreground");
    seed_capsule(session, c, &ids(vocab, 3, 701))?;
    ensure!(foreground(session) == live, "short re-prefill changed the foreground");
    println!("reset: shadow identity renewed, old checkpoint refused, foreground restored");
    Ok(())
}

fn zero_bank(
    client: &cubecl::prelude::ComputeClient<cubecl::cuda::CudaRuntime>,
    h: usize, inter: usize, experts: usize,
) -> Result<ExpertTable> {
    let code13 = h.checked_mul(inter).context("teacher shape overflow")?;
    let scale13 = code13 / 8;
    let code2 = code13 / 2;
    let scale2 = code2 / 8;
    let bytes = code13 + scale13 + code2 + scale2;
    ensure!(h % 64 == 0 && inter % 64 == 0 && 2 * bytes as u64 + experts as u64 * 80 <= EXTRA_BANK_BYTES,
        "synthetic bank exceeds its explicit 64 MiB admission or NVFP4 geometry");
    let mut packed = vec![0u8; bytes];
    packed[code13..code13 + scale13].fill(0x38);
    packed[code13 + scale13 + code2..].fill(0x38);
    let off13: Vec<u64> = (0..experts).flat_map(|_| [0, code13 as u64]).collect();
    let off2: Vec<u64> = (0..experts).flat_map(|_| [(code13 + scale13) as u64, (bytes - scale2) as u64]).collect();
    let scales = vec![1.0f32; experts];
    Ok(ExpertTable {
        off13: client.create_from_slice(bytes_of(&off13)), off2: client.create_from_slice(bytes_of(&off2)),
        sc13: client.create_from_slice(bytes_of(&scales)), sc2: client.create_from_slice(bytes_of(&scales)),
        wmap: client.create_from_slice(&packed), wmap_bytes: bytes, expert_bytes: bytes,
        n_routed: experts, stride: 2, scaled: true,
    })
}

fn teacher_isolation(session: &mut Session, teacher: &mut Sequence, vocab: usize, limit: f64) -> Result<()> {
    let token = ids(vocab, 1, 501)[0];
    let checkpoint = session.checkpoint()?;
    let live = foreground(session);
    let reference = session.extend_logits(&[token])?;
    let client = seam::client_of(&reference);
    let expected = down(reference);
    session.rewind(&checkpoint)?;
    let t = &session.config().text_config;
    let layer = session.layer_range().end - 1;
    ensure!(!t.is_dense(layer), "--teacher-bank needs a routed last layer");
    let bank = zero_bank(&client, t.hidden_size, t.intermediate_size, t.n_routed_experts)?;
    teacher.reset();
    session.with_sequence(teacher, |session| {
        session.set_teacher_bank(layer, 1, bank)?;
        let logits = down(session.extend_logits(&ids(vocab, 5, 911))?);
        ensure!(logits.iter().all(|x| x.is_finite() || *x == f32::NEG_INFINITY)
            && logits.iter().any(|x| x.is_finite()), "synthetic teacher produced invalid logits");
        Ok(())
    })?;
    ensure!(foreground(session) == live, "teacher pass changed foreground state");
    let teacher_pos = teacher.position();
    let error = session.batch_logits(Some(token), &mut [&mut *teacher], &[token]).err()
        .context("a batch silently mixed student and teacher banks")?;
    ensure!(error.to_string().contains("cannot mix"), "unexpected mixed-bank refusal: {error:#}");
    ensure!(foreground(session) == live && teacher.position() == teacher_pos && !teacher.is_poisoned(),
        "mixed-bank preflight mutated a sequence");
    let actual = down(session.extend_logits(&[token])?);
    compare("foreground-after-teacher", &actual, &expected, limit)?;
    session.set_next_token(token)?;
    reset_shadow(session, teacher, vocab)?;
    println!("bank isolation: teacher ran separately; mixed batch refused without mutation; reset cleared bank");
    Ok(())
}

fn main() -> Result<()> {
    let args = Args::parse();
    ensure!((1..=3).contains(&args.layers), "this bounded gate only admits 1..=3 layers");
    ensure!((1..=8).contains(&args.rounds), "this bounded gate only admits 1..=8 rounds");
    ensure!(args.max_scaled_error.is_finite() && args.max_scaled_error > 0.0,
        "--max-scaled-error must be finite and positive");
    ensure!(!args.teacher_bank || args.layers == 3, "--teacher-bank requires --layers 3");
    ensure!(std::env::var_os("INK_LEARN_LR").is_none(), "unset INK_LEARN_LR: this gate never learns");
    let mut cfg = SessionConfig::new(args.pile).layers(0..args.layers);
    cfg.config_override = args.config;
    cfg.prefill_budget = 32;
    cfg.extend_batch = 32;
    cfg.target_budget = 0;
    cfg.context_budget = 64;
    cfg.sequence_context_budgets = vec![64, 64];
    // Keep gate checkpoints and observed heads beside the admitted sequences;
    // these are test lifetimes, not a free production cache reservation.
    cfg.extra_reserved_bytes = GATE_WORKSPACE_BYTES
        + if args.teacher_bank { EXTRA_BANK_BYTES } else { 0 };
    let mut session = Session::load(cfg)?;
    ensure!(session.is_partial_stack() && !session.learning(), "gate unexpectedly loaded a full or learning model");
    let t = &session.config().text_config;
    ensure!(t.vocab_size > 64 && t.use_sconv && (2..=7).contains(&t.sconv_kernel_size),
        "gate requires text embeddings and nontrivial short-convolution history");
    ensure!(args.rounds >= t.sconv_kernel_size, "run at least one full convolution-history turnover");
    let vocab = t.vocab_size;
    let _ = session.extend_logits(&ids(vocab, 7, 11))?;
    session.set_next_token(17)?;
    let mut b = session.new_sequence()?;
    let mut c = session.new_sequence()?;
    seed_capsule(&mut session, &mut b, &ids(vocab, 15, 251))?;
    seed_capsule(&mut session, &mut c, &ids(vocab, 31, 541))?;
    let live_checkpoint = session.checkpoint()?;
    ensure!(session.with_sequence(&mut b, |s| s.rewind(&live_checkpoint)).is_err(),
        "a background capsule accepted a foreground checkpoint");
    check_positions(&mut session, &mut b, &mut c, [7, 15, 31])?;
    for i in 0..args.rounds {
        round(&mut session, &mut b, &mut c, ids(vocab, 1, 71 + i)[0],
            &format!("unequal/step{i}"), args.max_scaled_error)?;
    }
    reset_shadow(&mut session, &mut c, vocab)?;
    round(&mut session, &mut b, &mut c, ids(vocab, 1, 301)[0], "after-reset", args.max_scaled_error)?;
    if args.teacher_bank {
        teacher_isolation(&mut session, &mut c, vocab, args.max_scaled_error)?;
        round(&mut session, &mut b, &mut c, ids(vocab, 1, 401)[0], "after-bank-reset", args.max_scaled_error)?;
    }
    println!("PASS: partial layers 0..{}, unequal-position B3, KV/convolution histories and reset isolation", args.layers);
    Ok(())
}
