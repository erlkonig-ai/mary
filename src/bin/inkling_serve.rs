//! `inkling_serve` — the SERVING PROCESS: one `Session`, held open, answering
//! turns over the framed-stream convention.
//!
//! ```text
//! INK_LAYERS=0:4 inkling_serve --pile <model.pile> --tokenizer <tokenizer.json>
//! ```
//!
//! # What this is, against what came before
//!
//! `inkling_forward` is a measurement harness that runs the model inside `main`
//! and exits. `mary::models::inkling::session::Session` made the model a value
//! that survives across calls, and `inkling_session` drives one and prints
//! tokens. Neither is reachable by another PROGRAM: a `Session` lives in one
//! address space, and the program that wants it — `drive` — deliberately does
//! not link `mary`, because drive must keep building GPU-free in seconds.
//!
//! This is the process that closes that gap. It loads once, holds the weights,
//! the KV cache and the position, and serves turns on stdin/stdout. The protocol
//! is `mary::models::inkling::serve`, which is the framed-stream convention with
//! three control content-types on it — not a new format.
//!
//! # It streams, and that is the whole point
//!
//! Every token is written and FLUSHED as it is decoded, one framed record each.
//! A consumer can therefore start speaking on the first word of a sentence
//! instead of waiting for the last. Buffering the turn and sending it at the end
//! would be a legal framed stream and would throw away the only property that
//! makes a pipe better than a function call.
//!
//! # stdout is the PROTOCOL, so nothing else may write to it
//!
//! `Session::load` and everything under it print load diagnostics with
//! `println!`. A single stray line in the middle of a framed stream is not a
//! cosmetic problem: it is a corrupt record, and the reader would report a
//! continuity violation somewhere downstream of the actual cause. So the very
//! first thing this program does is `dup` the real stdout to a private fd and
//! point fd 1 at stderr. After that, every `println!` in every library this
//! links lands on stderr where it belongs, and the protocol owns a descriptor
//! nothing else can reach. This is a guard, not a convention: it holds for code
//! that has never heard of it.
//!
//! # Tokenizing is on THIS side
//!
//! Text on the wire, never ids. Drive's `Mind` seam asks for no tokenizer — the
//! loop never teacher-forces tokens — so the tokenizer belongs to whoever owns
//! the model. The tokenizer is the checkpoint's own `tokenizer.json`, read by
//! the same `tokenizers::Tokenizer::from_file` that `inkling_encode` and
//! `inkling_tokenizer_gate` use, so there is one tokenizer in this tree and not
//! a second transcription of one. (`mary::persist::load_tokenizer_from_pile`
//! reads the same thing out of a pile's facts, which is where this should read
//! it from once a model pile carries the tokenizer graph. `--tokenizer` is an
//! explicit path rather than a silent fallback so it is visible which one ran.)
//!
//! # One process is one RANK
//!
//! Without explicit tensor-parallel arguments, `Session::load` enforces
//! `hi - lo < num_hidden_layers` (144 GiB of weights do not fit a 121 GiB box).
//! Such a SINGLE-BOX serving process necessarily runs a strict subrange,
//! unembeds through layers it did not all run, and produces DIAGNOSTIC tokens
//! rather than the model's. That is said on the wire, in the READY record's
//! `partial` flag, rather than left to be inferred from fluent-looking wrong
//! text.
//!
//! With `--tp-rank`, `--tp-world`, and `--tp-rendezvous`, the serving process
//! instead forms and warms one communicator and gives that exact Group to
//! `Session::load_with_group`. Every rank then runs the full layer range on its
//! within-layer shard. The fan-out proxy starts two such processes in lockstep
//! and speaks this same protocol downstream.

use std::io::Write as _;

use anyhow::{Context, Result};

use mary::models::inkling::serve::{
    CONSULT_TYPE, CONTENT_TYPE, Consult, READY_TYPE, Ready, TURN_TYPE, TurnEnd, UNIT,
};
use mary::models::inkling::session::{Session, SessionConfig};
use mary::models::inkling::tp::Tp;
use mary::models::inkling::tpcomm::{Group, transport_note};
use triblespace::core::blob::IntoBlob;
use triblespace::core::blob::encodings::rawbytes::RawBytes;

fn usage() -> &'static str {
    "\
inkling_serve — one Session, held open, answering turns on stdin/stdout

USAGE:
    inkling_serve --pile <model.pile> --tokenizer <tokenizer.json> [OPTIONS]

OPTIONS:
    --pile <path>        The model collection: weights AND config.json
    --tokenizer <path>   The checkpoint's tokenizer.json
    --layers <lo:hi>     Layers this rank runs (default: $INK_LAYERS)
    --gen <n>            Default tokens per turn when a consult does not say
    --stop-id <id>       Stop on this token id; repeatable (single-rank only)
    --prefill-budget <n> Maximum tokens processed in one prefill pass
    --context-budget <n> Maximum positions retained by the session (default:
                         the effective prefill budget)
    --tp-rank <rank>     This process's tensor-parallel rank (all TP flags together)
    --tp-world <world>   Number of tensor-parallel ranks
    --tp-rendezvous <a>  Rank 0's HOST:PORT on the fast fabric
    -h, --help           This text

The protocol is the framed-stream convention with three control content-types;
see `mary::models::inkling::serve`.
"
}

struct Options {
    pile: std::path::PathBuf,
    tokenizer: std::path::PathBuf,
    layers: Option<std::ops::Range<usize>>,
    tokens: usize,
    stop: Vec<u32>,
    prefill_budget: Option<usize>,
    context_budget: Option<usize>,
    tensor_parallel: Option<TensorParallel>,
}

struct TensorParallel {
    tp: Tp,
    rendezvous: String,
}

fn parse() -> Result<Option<Options>> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let (mut pile, mut tokenizer, mut layers, mut prefill_budget, mut context_budget) =
        (None, None, None, None, None);
    let (mut tp_rank, mut tp_world, mut tp_rendezvous) = (None, None, None);
    let mut tokens = 32usize;
    let mut stop = Vec::new();
    let mut i = 0;
    while i < args.len() {
        let need = |i: usize| -> Result<&String> {
            args.get(i + 1)
                .with_context(|| format!("{} wants a value", args[i]))
        };
        match args[i].as_str() {
            "-h" | "--help" => return Ok(None),
            "--pile" => {
                pile = Some(std::path::PathBuf::from(need(i)?));
                i += 2;
            }
            "--tokenizer" => {
                tokenizer = Some(std::path::PathBuf::from(need(i)?));
                i += 2;
            }
            "--layers" => {
                let value = need(i)?;
                let (lo, hi) = value
                    .split_once(':')
                    .with_context(|| format!("--layers wants LO:HI, got {value:?}"))?;
                layers = Some(lo.parse()?..hi.parse()?);
                i += 2;
            }
            "--gen" => {
                tokens = need(i)?.parse().context("--gen wants a count")?;
                i += 2;
            }
            "--stop-id" => {
                stop.push(need(i)?.parse().context("--stop-id wants a token id")?);
                i += 2;
            }
            "--prefill-budget" => {
                prefill_budget = Some(need(i)?.parse().context("--prefill-budget wants a count")?);
                i += 2;
            }
            "--context-budget" => {
                context_budget = Some(need(i)?.parse().context("--context-budget wants a count")?);
                i += 2;
            }
            "--tp-rank" => {
                tp_rank = Some(need(i)?.parse().context("--tp-rank wants a number")?);
                i += 2;
            }
            "--tp-world" => {
                tp_world = Some(need(i)?.parse().context("--tp-world wants a count")?);
                i += 2;
            }
            "--tp-rendezvous" => {
                tp_rendezvous = Some(need(i)?.clone());
                i += 2;
            }
            other => anyhow::bail!("unknown argument {other:?}\n\n{}", usage()),
        }
    }
    let tensor_parallel = match (tp_rank, tp_world, tp_rendezvous) {
        (None, None, None) => None,
        (Some(rank), Some(world), Some(rendezvous)) => {
            let tp = Tp::new(rank, world)?;
            anyhow::ensure!(
                tp.is_split(),
                "--tp-world must be greater than one; omit all three --tp-* flags for one rank"
            );
            Some(TensorParallel { tp, rendezvous })
        }
        _ => anyhow::bail!(
            "--tp-rank, --tp-world, and --tp-rendezvous are one launch contract; provide all or none"
        ),
    };
    anyhow::ensure!(
        tensor_parallel.is_none() || stop.is_empty(),
        "--stop-id cannot be decided independently by tensor ranks: one rank stopping while its \
         peer enters the next collective would deadlock. The paired serving layer must arbitrate \
         any early stop; use max_tokens for this rank protocol."
    );
    Ok(Some(Options {
        pile: pile.context("--pile is required")?,
        tokenizer: tokenizer.context("--tokenizer is required")?,
        layers,
        tokens,
        stop,
        prefill_budget,
        context_budget,
        tensor_parallel,
    }))
}

/// Take fd 1 for the protocol and point every `println!` at stderr.
///
/// Done before ANYTHING else runs, because the load path prints and a printed
/// line inside a framed stream is a corrupt record. Returns the private
/// descriptor the protocol writes to.
fn claim_stdout() -> Result<std::fs::File> {
    use std::os::fd::FromRawFd as _;
    // Flush whatever Rust has buffered on stdout before the descriptor moves,
    // so nothing written before the swap lands on the protocol's fd.
    let _ = std::io::stdout().flush();
    let raw = unsafe { libc::dup(libc::STDOUT_FILENO) };
    anyhow::ensure!(raw >= 0, "could not dup stdout for the protocol stream");
    let redirected = unsafe { libc::dup2(libc::STDERR_FILENO, libc::STDOUT_FILENO) };
    anyhow::ensure!(redirected >= 0, "could not point stdout at stderr");
    Ok(unsafe { std::fs::File::from_raw_fd(raw) })
}

fn hex_identity(bytes: [u8; 32]) -> String {
    use std::fmt::Write as _;

    let mut text = String::with_capacity(64);
    for byte in bytes {
        write!(&mut text, "{byte:02X}").expect("writing into a String is infallible");
    }
    text
}

fn main() {
    if let Err(error) = run() {
        eprintln!("inkling_serve: {error:#}");
        std::process::exit(1);
    }
}

fn run() -> Result<()> {
    let Some(options) = parse()? else {
        print!("{}", usage());
        return Ok(());
    };
    let protocol = claim_stdout()?;

    // Both preambles are written before either side reads, so the handshake
    // cannot deadlock. Ours goes out FIRST — before the minutes of loading —
    // which is what lets a client distinguish "starting" from "not ours".
    let mut out = framed_stream::FramedWriter::open(protocol, CONTENT_TYPE, UNIT)
        .context("open the protocol's output stream")?;
    let mut input = framed_stream::FramedReader::open(std::io::stdin().lock())
        .context("open the protocol's input stream")?;
    input
        .require_content_type(CONTENT_TYPE)
        .context("this serving process is fed text, and was handed something else")?;

    let tokenizer_bytes = std::fs::read(&options.tokenizer)
        .with_context(|| format!("read {}", options.tokenizer.display()))?;
    let tokenizer_identity = IntoBlob::<RawBytes>::to_blob(tokenizer_bytes.as_slice())
        .get_handle()
        .raw;
    let tokenizer = tokenizers::Tokenizer::from_bytes(&tokenizer_bytes)
        .map_err(|e| anyhow::anyhow!("load {}: {e}", options.tokenizer.display()))?;

    let mut config = SessionConfig::new(&options.pile);
    if let Some(layers) = options.layers.clone() {
        config = config.layers(layers);
    }
    if let Some(budget) = options.prefill_budget {
        config.prefill_budget = budget;
    }
    // Historically there was only one length axis. Preserve that CLI meaning:
    // setting just --prefill-budget admits that many retained positions too,
    // while an explicit --context-budget opts into bounded-width chunking of a
    // longer logical sequence.
    config.context_budget = options.context_budget.unwrap_or(config.prefill_budget);
    let loaded = std::time::Instant::now();
    let mut session = match options.tensor_parallel {
        None => Session::load(config).context("load the model")?,
        Some(tensor_parallel) => {
            eprintln!(
                "inkling_serve: forming tensor rank {} of {} at {}",
                tensor_parallel.tp.rank(),
                tensor_parallel.tp.world(),
                tensor_parallel.rendezvous,
            );
            let group = Group::form_default(tensor_parallel.tp, &tensor_parallel.rendezvous)
                .context("form the tensor-parallel group")?;
            group
                .warm()
                .context("warm and verify the tensor-parallel group")?;
            eprintln!("inkling_serve: tensor group paired ({})", transport_note());
            Session::load_with_group(config, group).context("load this tensor-parallel rank")?
        }
    };
    let load_secs = loaded.elapsed().as_secs_f64();

    // Decoder state belongs to the whole logical token sequence, not to one
    // generated turn. Byte-fallback and spacing decoders both need surrounding
    // ids. New world-context ids advance this stream without being spoken;
    // generated ids advance the same stream and their chunks are emitted. A
    // carried token is never advanced twice: it entered this sequence when it
    // was generated, while `carry` only catches the KV cache up to that fact.
    let mut decode_stream = tokenizer.decode_stream(false);
    let mut decode = |id: u32| {
        decode_stream
            .step(id)
            .map_err(|error| anyhow::anyhow!("streaming decode: {error}"))
    };

    let range = session.layer_range();
    let ready = Ready {
        pile: options.pile.display().to_string(),
        model_identity: hex_identity(session.model_identity()),
        tokenizer_identity: hex_identity(tokenizer_identity),
        layers: [range.start, range.end],
        stack: session.config().text_config.num_hidden_layers,
        partial: session.is_partial_stack(),
        vocab: session.config().text_config.effective_vocab(),
        load_secs,
    };
    eprintln!(
        "inkling_serve: ready in {load_secs:.1}s — layers {}..{} of {}{}",
        ready.layers[0],
        ready.layers[1],
        ready.stack,
        match ready.partial {
            true => " (PARTIAL STACK: diagnostic tokens, not the model's)",
            false => "",
        }
    );
    let payload = serde_json::to_vec(&ready)?;
    let extent = payload.len() as u64;
    out.record_as(READY_TYPE, &payload, extent)?;

    // ── serve ───────────────────────────────────────────────────────────────
    //
    // Context accumulates as text records; a CONSULT record ends the delta and
    // asks for a turn. Nothing here is asynchronous: the client writes, we read
    // until the consult, we generate and write, the client reads. Strict
    // alternation, so the two-pipe deadlock cannot arise.
    let mut delta = String::new();
    let mut turn = 0usize;
    // The token the previous turn EMITTED and never fed back, waiting for the
    // next pass to put it in the cache. `None` is also "no turn has run yet",
    // which is the same fact as "nothing is prefilled": every turn emits at
    // least one token. See `serve_turn`.
    let mut carry: Option<usize> = None;
    loop {
        match input.next_frame()? {
            framed_stream::Frame::Record(record) if record.content_type() == CONSULT_TYPE => {
                let consult: Consult =
                    serde_json::from_slice(&record.payload).unwrap_or(Consult::new(options.tokens));
                let want = consult.max_tokens.max(1);
                let end = serve_turn(
                    &mut session,
                    &tokenizer,
                    &mut decode,
                    &mut out,
                    std::mem::take(&mut delta),
                    want,
                    &options.stop,
                    turn,
                    &mut carry,
                )?;
                let payload = serde_json::to_vec(&end)?;
                let extent = payload.len() as u64;
                out.record_as(TURN_TYPE, &payload, extent)?;
                eprintln!("inkling_serve: {}", end.summary());
                turn += 1;
            }
            framed_stream::Frame::Record(record) if record.content_type() == CONTENT_TYPE => {
                delta.push_str(record.text()?);
            }
            framed_stream::Frame::Record(record) => {
                anyhow::bail!(
                    "this serving process does not understand a {} record",
                    record.content_type()
                )
            }
            framed_stream::Frame::Gap(gap) => {
                // The client declared context it could not deliver. Attending to
                // the rest as if nothing were missing is exactly what a gap
                // exists to prevent, so it is marked in the context itself.
                eprintln!(
                    "inkling_serve: client gap of {} byte(s): {}",
                    gap.extent, gap.reason
                );
                delta.push_str(&format!("\n[{} bytes lost: {}]\n", gap.extent, gap.reason));
            }
            framed_stream::Frame::End(status) => {
                eprintln!("inkling_serve: input stream ended ({status:?}) after {turn} turn(s)");
                break;
            }
        }
    }
    out.finish(framed_stream::EndStatus::Complete)?;
    Ok(())
}

/// One turn: attend to the delta, then generate, emitting each token as it is
/// decoded.
///
/// The two `Session` calls that matter are here and nowhere else: `prefill` for
/// the first sequence, `extend` for every turn after it — which attends ONLY to
/// what is new, because the KV cache is still holding everything before it. That
/// is the property the whole exercise is for, and it is why the second turn is
/// three orders of magnitude cheaper than the first.
///
/// # What is NEW is not only what the client sent
///
/// A turn's last token is emitted and never fed back: the loop below stops one
/// step short, because generating a successor for a token the caller will not
/// read costs a whole decode step (~44 ms at layers 0..21) and produces nothing.
/// That saving is real and it is kept — but it means the turn ends with one
/// token of the sequence in the consumer's stream and NOT in the KV cache.
///
/// So `carry` holds it, and the next turn appends it at the HEAD of its delta.
/// That is the only place it can go and the cheapest place it could have gone:
/// `Session::extend` batches, so the carried token is one extra ROW of a pass
/// the turn was making anyway rather than a decode step of its own.
///
/// **Until 2026-08-27 nothing carried it and every turn lost its own final
/// word, permanently.** The failure was invisible from inside: `position()`
/// stayed exactly `prompt + fed`, no length disagreed with any other, and the
/// cache was perfectly CONSISTENT — one token short of the sequence it stood
/// for. `inkling_session --carry` is the gate that catches it, and it catches it
/// by asking the model what comes next rather than by measuring anything.
fn advance_context_decode(
    decode: &mut impl FnMut(u32) -> Result<Option<String>>,
    ids: &[usize],
) -> Result<()> {
    for &id in ids {
        let _ = decode(id as u32)?;
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn serve_turn(
    session: &mut Session,
    tokenizer: &tokenizers::Tokenizer,
    decode: &mut impl FnMut(u32) -> Result<Option<String>>,
    out: &mut framed_stream::FramedWriter<std::fs::File>,
    delta: String,
    want: usize,
    stop: &[u32],
    turn: usize,
    carry: &mut Option<usize>,
) -> Result<TurnEnd> {
    // `false`: no special tokens. The prompts this model is measured against
    // carry none, and a BOS silently prepended here would make a served turn
    // incomparable with every reference prompt in the tree while every shape
    // check still passed. (Same reasoning, same flag, as `inkling_encode`.)
    let delta_ids: Vec<usize> = match delta.is_empty() {
        true => Vec::new(),
        false => tokenizer
            .encode(delta.as_str(), false)
            .map_err(|e| anyhow::anyhow!("encode the delta: {e}"))?
            .get_ids()
            .iter()
            .map(|&id| id as usize)
            .collect(),
    };
    // Context participates in decoding even though it is not speech. Advancing
    // and discarding here makes the next generated id see its real predecessor
    // without ever echoing a tool result. `carry` is excluded because the
    // decoder already saw that id when the previous turn generated it.
    advance_context_decode(decode, &delta_ids)?;
    anyhow::ensure!(
        carry.is_some() || !delta_ids.is_empty(),
        "the first turn has nothing to attend to: a prefill with no tokens would be vacuous"
    );

    // What this pass appends: the previous turn's unfed last token, then the new
    // context. On turn 0 there is no carry and this IS the delta.
    let ids: Vec<usize> = carry
        .iter()
        .copied()
        .chain(delta_ids.iter().copied())
        .collect();
    let carried = ids.len() - delta_ids.len();

    let started = std::time::Instant::now();
    let first = match carry.is_some() {
        false => session
            .prefill(&ids)
            .context("prefill the first sequence")?,
        // Never empty on a primed session: the carry alone is a token, so a
        // consult with no new context is still a one-row `extend` rather than a
        // bare `step`. Same pass, and it is the pass that closes the gap.
        true => session.extend(&ids).context("extend the sequence")?,
    };
    let first_token_secs = started.elapsed().as_secs_f64();

    // ── the incremental detokenizer ─────────────────────────────────────────
    //
    // `DecodeStream` owns the prefix needed by byte-fallback and spacing
    // decoders. Each generated id therefore yields either one final text chunk
    // or `None` while an incomplete sequence waits for a later logical token.
    // No replacement character is emitted and no spoken prefix is rewritten.
    let mut generated: Vec<u32> = Vec::with_capacity(want);
    let mut stopped = "max_tokens";
    let mut token = first;
    for step in 0..want {
        generated.push(token as u32);
        if let Some(text) = decode(token as u32)?
            && !text.is_empty()
        {
            // Written and FLUSHED here, inside the generation loop. This one
            // call is the difference between a stream and a batch.
            out.text(&text)?;
        }
        if stop.contains(&(token as u32)) {
            stopped = "stop_token";
            break;
        }
        // One step short on purpose: the successor of the last emitted token
        // would cost a full decode step and nobody would read it. The token
        // itself is not lost — it leaves in `carry` below and is appended by the
        // next turn's `extend`. Break that pairing and the model stops hearing
        // its own last word. See this function's doc.
        if step + 1 < want {
            token = session.step().context("advance one token")?;
        }
    }

    // What this turn emitted and did not feed back. Both exits above land here:
    // the `want` exit skipped the final step, and the stop-token exit broke
    // before it. A turn always emits at least one token, so this is always
    // `Some` afterwards — which is also what tells the next turn it is not the
    // first.
    *carry = generated.last().map(|&t| t as usize);

    let tokens = generated.len();
    Ok(TurnEnd {
        turn,
        tokens,
        token_ids: generated,
        delta_tokens: delta_ids.len(),
        carried,
        stopped: stopped.to_string(),
        first_token_secs,
        turn_secs: started.elapsed().as_secs_f64(),
        position: session.position(),
    })
}

#[cfg(test)]
mod tests {
    use tokenizers::decoders::byte_fallback::ByteFallback;
    use tokenizers::models::bpe::BPE;
    use tokenizers::normalizers::unicode::NFC;
    use tokenizers::pre_tokenizers::byte_level::ByteLevel;
    use tokenizers::{Tokenizer, TokenizerBuilder};

    use super::advance_context_decode;

    fn byte_fallback_tokenizer() -> Tokenizer {
        let vocab = [
            ("<0x20>".to_string(), 0),
            ("<0xC3>".to_string(), 1),
            ("<0xA9>".to_string(), 2),
        ];
        let bpe = BPE::builder()
            .vocab_and_merges(vocab, Vec::new())
            .byte_fallback(true)
            .build()
            .unwrap();
        TokenizerBuilder::default()
            .with_model(bpe)
            .with_decoder(Some(ByteFallback::default()))
            .with_normalizer(Some(NFC))
            .with_pre_tokenizer(Some(ByteLevel::default()))
            .with_post_processor(Some(ByteLevel::default()))
            .build()
            .unwrap()
            .into()
    }

    #[test]
    fn incomplete_output_can_finish_on_the_next_turn() {
        let tokenizer = byte_fallback_tokenizer();
        let mut stream = tokenizer.decode_stream(false);
        let mut decode = |id| stream.step(id).map_err(|error| anyhow::anyhow!("{error}"));

        advance_context_decode(&mut decode, &[0]).unwrap();
        assert_eq!(decode(1).unwrap(), None, "the first output byte waits");
        assert_eq!(
            decode(2).unwrap().as_deref(),
            Some("é"),
            "a no-delta next turn completes rather than loses the character"
        );
    }

    #[test]
    fn text_completed_by_hidden_context_stays_hidden() {
        let tokenizer = byte_fallback_tokenizer();
        let mut stream = tokenizer.decode_stream(false);
        let mut decode = |id| stream.step(id).map_err(|error| anyhow::anyhow!("{error}"));

        assert_eq!(decode(1).unwrap(), None, "the output byte is incomplete");
        advance_context_decode(&mut decode, &[2]).unwrap();
        assert_eq!(
            decode(0).unwrap().as_deref(),
            Some(" "),
            "bytes completed partly by world input are consumed, not spoken later"
        );
    }
}
