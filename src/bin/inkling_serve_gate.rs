//! **THE GATE** — `drive`'s real loop, driven by a real Inkling `Session`
//! instead of `StubMind`.
//!
//! ```text
//! inkling_serve_gate drive  --serve <inkling_serve> --serve-arg … \
//!                           --playground <mock_mcp_server> \
//!                           --expect-command <exact-command> --voice <stub_speech>
//! inkling_serve_gate probe  --serve <inkling_serve> --serve-arg …
//! ```
//!
//! # Why this binary is GPU-free, and lives in `mary`
//!
//! The model runs in ANOTHER PROCESS, so nothing here needs CUDA: this links
//! `drive` (which builds GPU-free in seconds and must keep doing so) and the
//! client half of `mary::models::inkling::serve`. It lives in `mary` rather than
//! in `drive` because `drive` deliberately has no `mary` dependency, and the
//! adapter is a `mary` concern — mary knows what an Inkling turn is; drive knows
//! only that a mind is a `&mut self` function from world to turn.
//!
//! # The two modes, and what each one can prove
//!
//! **`drive`** is the gate proper. It is drive's own `Shell` — the same
//! `Shell::open`/`turn`/`finish` `tests/shell_gate.rs` runs — with three real
//! parts and one stand-in:
//!
//!   - the MIND is `InklingMind`, holding a real `Session` over a pipe. Not a
//!     stub.
//!   - the LOOP, the world, the span verification, the audit and the pile are
//!     drive's production code, unmodified.
//!   - the VOICE is `stub_speech`, drive's own speech faculty with the vocoder
//!     left out: text records in, PCM records out, ONE PER WORD AS EACH WORD
//!     ARRIVES. It is what makes "the consumer produced output before the turn
//!     ended" observable without a speaker.
//!   - the SANDBOX is `mock_mcp_server`, drive's protocol mock. It is a
//!     stand-in for containment, not for the loop: when Inkling emits a valid
//!     native `exec` call, Drive really audits it, runs it through this server,
//!     and feeds the typed result back to the retained model session.
//!
//! **`probe`** skips drive entirely and drives the serving process directly.
//! The native Drive gate can now exercise a typed result delta, but only when a
//! sampled response chooses to call a tool. The probe makes an arbitrary raw
//! delta deterministic and isolates `feed` → `Session::extend` from the rest of
//! the loop, then reports what each turn cost.
//!
//! # What the numbers mean
//!
//! Every duration comes from inside the serving process, around the `Session`
//! calls, and carries its framing rule with it: `first_token_secs` is seconds
//! per FIRST TOKEN OF A TURN over the layer range the READY record names — not
//! per token and not per turn. A paired proxy reports the slower rank, i.e. the
//! distributed critical path. Turn 0 pays the prefill; every turn after it pays
//! only what is new. That difference is the whole claim.

use std::path::PathBuf;
use std::sync::atomic::AtomicBool;

use anyhow::{Context, Result};

use mary::models::inkling::serve::{Consult, InklingMind, ServeClient, TurnEnd};

fn usage() -> &'static str {
    "\
inkling_serve_gate — drive's loop, driven by a real Inkling Session

USAGE:
    inkling_serve_gate drive [OPTIONS]     # the gate: drive, with a real mind
    inkling_serve_gate probe [OPTIONS]     # the protocol alone, with a delta

OPTIONS (both):
    --serve <path>       The inkling_serve binary [required]
    --serve-arg <arg>    Argument passed to it; repeatable
    --gen <n>            Generation cap per response/consultation (default 24)
    --turns <n>          Turns to run (default 3)
    --system <text>      First delta: what the model is given to start from

OPTIONS (drive):
    --playground <path>  A `playground mcp`-compatible binary (mock_mcp_server)
                         [required]
    --expect-command <command>
                         The one exact command this gate permits and requires.
                         `drive` refuses to start without it
    --exec-timeout-ms <n>
                         Finite wall-clock limit for that command (default 5000)
    --voice <path>       A streaming speech faculty (stub_speech). Without one
                         there is no consumer, and the streaming claim cannot be
                         measured — only asserted
    --voice-arg <arg>    Argument passed to the voice; repeatable
    --pile <path>        Scratch provenance pile (default: a temp file)
    --memory-pile <path> Durable self pile whose real cover is injected before
                         turn 0 (default: no memory)
    --memory-bin <path>  Memory faculty used to read that cover (default: memory)
    --memory-window-tokens <n>
                         Context window used to budget the cover (default 200000;
                         output reserve and chars/token match Drive defaults)

OPTIONS (probe):
    --feed <text>        Delta fed before a turn; repeatable, one per turn after
                         the first
"
}

const DEFAULT_SYSTEM: &str = "You are in a shell. Think out loud in your own words.";

struct Options {
    serve: PathBuf,
    serve_args: Vec<String>,
    playground: Option<PathBuf>,
    voice: Option<PathBuf>,
    voice_args: Vec<String>,
    pile: Option<PathBuf>,
    memory_pile: Option<PathBuf>,
    memory_bin: PathBuf,
    memory_window_tokens: u64,
    tokens: usize,
    turns: usize,
    system: String,
    feeds: Vec<String>,
    expected_command: Option<String>,
    exec_timeout_ms: u64,
}

fn parse(args: &[String]) -> Result<Options> {
    let mut o = Options {
        serve: PathBuf::new(),
        serve_args: Vec::new(),
        playground: None,
        voice: None,
        voice_args: Vec::new(),
        pile: None,
        memory_pile: None,
        memory_bin: PathBuf::from("memory"),
        memory_window_tokens: 200_000,
        tokens: 24,
        turns: 3,
        system: DEFAULT_SYSTEM.to_string(),
        feeds: Vec::new(),
        expected_command: None,
        exec_timeout_ms: 5_000,
    };
    let mut i = 0;
    while i < args.len() {
        let need = |i: usize| -> Result<String> {
            args.get(i + 1)
                .cloned()
                .with_context(|| format!("{} wants a value", args[i]))
        };
        match args[i].as_str() {
            "--serve" => o.serve = PathBuf::from(need(i)?),
            "--serve-arg" => o.serve_args.push(need(i)?),
            "--playground" => o.playground = Some(PathBuf::from(need(i)?)),
            "--voice" => o.voice = Some(PathBuf::from(need(i)?)),
            "--voice-arg" => o.voice_args.push(need(i)?),
            "--pile" => o.pile = Some(PathBuf::from(need(i)?)),
            "--memory-pile" => o.memory_pile = Some(PathBuf::from(need(i)?)),
            "--memory-bin" => o.memory_bin = PathBuf::from(need(i)?),
            "--memory-window-tokens" => {
                o.memory_window_tokens = need(i)?
                    .parse()
                    .context("--memory-window-tokens wants a count")?
            }
            "--gen" => o.tokens = need(i)?.parse().context("--gen wants a count")?,
            "--turns" => o.turns = need(i)?.parse().context("--turns wants a count")?,
            "--system" => o.system = need(i)?,
            "--feed" => o.feeds.push(need(i)?),
            "--expect-command" => o.expected_command = Some(need(i)?),
            "--exec-timeout-ms" => {
                o.exec_timeout_ms = need(i)?
                    .parse()
                    .context("--exec-timeout-ms wants a millisecond count")?
            }
            other => anyhow::bail!("unknown argument {other:?}\n\n{}", usage()),
        }
        i += 2;
    }
    anyhow::ensure!(
        !o.serve.as_os_str().is_empty(),
        "--serve is required: this gate does not run a model, it drives one"
    );
    Ok(o)
}

fn serve_client(o: &Options) -> Result<ServeClient> {
    let mut command = std::process::Command::new(&o.serve);
    command.args(&o.serve_args);
    eprintln!(
        "gate: starting {} {} — this blocks for the whole model load",
        o.serve.display(),
        o.serve_args.join(" ")
    );
    let client = ServeClient::spawn(&mut command)?;
    let ready = client.ready();
    anyhow::ensure!(
        matches!(
            ready.execution_profile.as_str(),
            "sealed-v1" | "observed-v1"
        ) && ready.execution_identity.len() == 64
            && ready
                .execution_identity
                .bytes()
                .all(|byte| byte.is_ascii_hexdigit()),
        "serve announced invalid execution manifest {:?} {:?}",
        ready.execution_profile,
        ready.execution_identity
    );
    eprintln!(
        "gate: READY — layers {}..{} of {}, vocab {}, prefill chunk {}, context {}, loaded in \
         {:.1}s{}, {} {}",
        ready.layers[0],
        ready.layers[1],
        ready.stack,
        ready.vocab,
        ready.prefill_budget,
        ready.context_budget,
        ready.load_secs,
        match ready.partial {
            true => "  [PARTIAL STACK: the tokens are diagnostic, not the model's]",
            false => "",
        },
        ready.execution_profile,
        ready.execution_identity,
    );
    if !ready.execution_unavailable.is_empty() {
        eprintln!(
            "gate: execution manifest unavailable facts: {}",
            ready.execution_unavailable.join(", ")
        );
    }
    Ok(client)
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let result = match args.first().map(String::as_str) {
        Some("drive") => parse(&args[1..]).and_then(|o| cmd_drive(&o)),
        Some("probe") => parse(&args[1..]).and_then(|o| cmd_probe(&o)),
        Some("-h") | Some("--help") | None => {
            print!("{}", usage());
            Ok(())
        }
        Some(other) => Err(anyhow::anyhow!("unknown command {other:?}\n\n{}", usage())),
    };
    if let Err(error) = result {
        eprintln!("inkling_serve_gate: {error:#}");
        std::process::exit(1);
    }
}

fn cmd_drive(o: &Options) -> Result<()> {
    let expected = o.expected_command.as_deref().context(
        "--expect-command is required for `drive`: the gate refuses to expose local command \
         execution without one exact operator-declared command",
    )?;
    anyhow::ensure!(!expected.is_empty(), "--expect-command must not be empty");
    anyhow::ensure!(
        o.exec_timeout_ms > 0,
        "--exec-timeout-ms must be greater than zero"
    );
    run_drive(
        o,
        o.tokens,
        o.turns,
        "drive",
        ActionPolicy {
            command: expected,
            timeout_ms: o.exec_timeout_ms,
        },
    )
}

#[derive(Clone, Copy)]
struct ActionPolicy<'a> {
    command: &'a str,
    timeout_ms: u64,
}

impl ActionPolicy<'_> {
    fn sandbox_args(self) -> Vec<String> {
        vec!["--allow-command".to_string(), self.command.to_string()]
    }

    fn timeout_ms(self) -> Option<u64> {
        Some(self.timeout_ms)
    }
}

/// THE GATE: drive's loop, with a real mind on the other end of a pipe.
fn run_drive(
    o: &Options,
    tokens: usize,
    turns: usize,
    label: &str,
    action_policy: ActionPolicy<'_>,
) -> Result<()> {
    let playground = o
        .playground
        .clone()
        .context("--playground is required: Shell::open boots a sandbox synchronously")?;
    let pile = o.pile.clone().unwrap_or_else(|| {
        let nanos = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .expect("clock")
            .as_nanos();
        std::env::temp_dir().join(format!("inkling-serve-gate-{nanos}.pile"))
    });

    let client = serve_client(o)?;
    let partial = client.ready().partial;
    let mind = InklingMind::new_gate(client, tokens, Some(o.system.clone()));
    let voice_slot = mind.voice_slot();
    let log = mind.log().expect("the finite gate retains turn evidence");
    let proofs = mind
        .proofs()
        .expect("the finite gate retains streaming evidence");

    let config = drive::shell::ShellConfig {
        pile: drive::config::PileConfig { path: pile.clone() },
        exec: drive::config::ExecConfig {
            playground_bin: playground,
            backend: "mock".to_string(),
            extra_args: action_policy.sandbox_args(),
            tenant: "mary".to_string(),
            timeout_ms: action_policy.timeout_ms(),
        },
        system_prompt: o.system.clone(),
        voice: o.voice.as_ref().map(|path| {
            let mut command = drive::stream::FacultyCommand::new(path);
            for arg in &o.voice_args {
                command = command.arg(arg);
            }
            command
        }),
        // Exhaust has its own gate in drive; this run is about the seam.
        telemetry: None,
        // Optional because the original seam gate still needs a tiny no-memory
        // form. When supplied, this is Drive's production cover path: the same
        // memory faculty, one shell-physical command/result pair per chunk, and
        // the same hard completeness rule. That makes a large-cover run an
        // end-to-end continuity measurement rather than a synthetic prompt.
        memory: o.memory_pile.as_ref().map(|memory_pile| {
            let mut budget = drive::context::ModelBudget::default();
            budget.context_window_tokens = o.memory_window_tokens;
            drive::context::MemoryConfig {
                binary: o.memory_bin.clone(),
                pile: memory_pile.clone(),
                budget,
            }
        }),
    };

    let mut shell = drive::shell::Shell::open(&config, Box::new(mind))
        .context("open drive's shell with the inkling mind")?;
    // CLAIM THE VOICE: from here the backend is the stream's writer and the
    // shell stops speaking the mind's turns for it. One producer per stream, and
    // the finer grain — one record per token, mid-`observe` — is the whole
    // reason this seam exists.
    *voice_slot.lock().expect("voice slot") = shell.claim_voice();
    eprintln!(
        "gate: session {:X} — drive {}, pile {}",
        shell.session(),
        drive::DRIVE_GIT_REV,
        pile.display()
    );

    let outcomes = shell.run(drive::shell::Extent::Turns(turns), &AtomicBool::new(false));
    let monologue = shell.monologue_window().text.to_string();
    let voice_report = shell.voice().map(|voice| voice.report());
    let finish = shell.finish();
    let outcomes = outcomes.context("run drive's loop")?;
    finish.context("finalize the session and close the pile")?;

    // ── the report ──────────────────────────────────────────────────────────
    println!(
        "\n=== {label}: drive ran {} turn(s) against a real Session ===",
        outcomes.len()
    );
    for outcome in &outcomes {
        println!(
            "  turn @{}: {:?}, decision {:X}, said {} byte(s)",
            outcome.coord,
            outcome.disposition,
            outcome.decision_id,
            outcome.said.len()
        );
        if outcome.disposition == drive::mind::Disposition::Fire {
            println!(
                "    action: command {:?}, result {}, isError {:?}, exit {:?}",
                outcome.command,
                outcome.result_id.is_some(),
                outcome.is_error,
                outcome.exit_code
            );
        }
    }
    println!("\n=== what the model cost, per turn ===");
    println!("  FRAMING: seconds per FIRST TOKEN OF A TURN and per TURN, measured inside the");
    println!("  serving process around the Session calls only, over the READY layer range.");
    println!("  A paired proxy reports the slower rank (the critical path). Not per token/step.");
    let turns = log.lock().expect("turn log").clone();
    for end in &turns {
        println!("  {}", end.summary());
    }
    if turns.len() >= 2 {
        let cold = turns[0].first_token_secs;
        let warm: Vec<f64> = turns[1..].iter().map(|t| t.first_token_secs).collect();
        let mean = warm.iter().sum::<f64>() / warm.len() as f64;
        println!(
            "  COLD/WARM: turn 0 first token {cold:.3}s, turns 1..{} mean {mean:.3}s — {:.0}x",
            turns.len() - 1,
            cold / mean.max(f64::EPSILON)
        );
    }

    println!("\n=== did it STREAM, or did it batch? ===");
    match voice_report {
        None => println!("  NOT MEASURED: no --voice, so there was no consumer to observe."),
        Some(report) => {
            println!("  voice: {}", report.summary());
            for proof in proofs.lock().expect("proof log").iter() {
                match proof.tokens_at_first_return {
                    Some(k) => println!(
                        "  turn {}: the faculty had produced output after {k} of {} token(s) — \
                         with the turn still running. STREAMED.",
                        proof.turn, proof.tokens
                    ),
                    None => println!(
                        "  turn {}: the faculty produced nothing before the turn ended \
                         ({} token(s), {} record(s) back). NOT PROVEN.",
                        proof.turn, proof.tokens, proof.records_at_end
                    ),
                }
            }
        }
    }

    println!(
        "\n=== what she said ({} bytes) ===\n{monologue}",
        monologue.len()
    );
    if partial {
        println!(
            "\nNOTE: a PARTIAL STACK produced this. The tokens are diagnostic — they came out \
             of an unembedding applied to layers the process did not all run. What is proven \
             here is the SEAM, not the model's text."
        );
    }
    verify_drive_turns(&turns)?;
    verify_action_policy(&outcomes, action_policy)?;
    println!(
        "\n=== exact native action: PASS ===\n  one Fire for {:?}, with a typed successful \
         result inside the {}ms bound",
        action_policy.command, action_policy.timeout_ms
    );
    Ok(())
}

#[derive(Debug, Clone, Copy)]
struct ActionEvidence<'a> {
    command: Option<&'a str>,
    has_content: bool,
    is_error: Option<bool>,
    exit_code: Option<i32>,
    has_result_id: bool,
}

fn verify_action_policy(
    outcomes: &[drive::shell::TurnOutcome],
    policy: ActionPolicy<'_>,
) -> Result<()> {
    let actions: Vec<_> = outcomes
        .iter()
        .filter(|outcome| outcome.disposition == drive::mind::Disposition::Fire)
        .map(|outcome| ActionEvidence {
            command: outcome.command.as_deref(),
            has_content: outcome.content.is_some(),
            is_error: outcome.is_error,
            exit_code: outcome.exit_code,
            has_result_id: outcome.result_id.is_some(),
        })
        .collect();
    verify_action_evidence(&actions, policy)
}

fn verify_action_evidence(actions: &[ActionEvidence<'_>], policy: ActionPolicy<'_>) -> Result<()> {
    let command = policy.command;
    let [action] = actions else {
        anyhow::bail!(
            "expected exactly one Fire for {command:?}, observed {}",
            actions.len()
        );
    };
    anyhow::ensure!(
        action.command == Some(command),
        "the one Fire named {:?}, expected {command:?}",
        action.command
    );
    anyhow::ensure!(
        action.has_content && action.has_result_id,
        "the Fire for {command:?} did not produce both a typed result and durable result id"
    );
    anyhow::ensure!(
        action.is_error == Some(false),
        "the Fire for {command:?} returned isError {:?}",
        action.is_error
    );
    anyhow::ensure!(
        action.exit_code == Some(0),
        "the Fire for {command:?} returned exit {:?}",
        action.exit_code
    );
    Ok(())
}

fn verify_drive_turns(turns: &[TurnEnd]) -> Result<()> {
    anyhow::ensure!(
        !turns.is_empty(),
        "Drive completed without consulting its mind"
    );
    let mut expected_position = 0usize;
    for (ordinal, end) in turns.iter().enumerate() {
        anyhow::ensure!(
            end.turn == ordinal,
            "Drive turn {ordinal} was labelled as turn {}",
            end.turn
        );
        anyhow::ensure!(end.tokens >= 1, "turn {ordinal} generated no token");
        anyhow::ensure!(
            end.token_ids.len() == end.tokens,
            "turn {ordinal} retained only {} exact token id(s) for {} generated token(s)",
            end.token_ids.len(),
            end.tokens
        );
        // REINITIALIZE stages one complete replacement initialization in a
        // reset Session. Its first turn therefore has fresh delta, no carry,
        // and a position measured from zero again. Every ordinary turn after
        // the first still has to carry the preceding generated token.
        let reinitialized = ordinal > 0 && end.carried == 0 && end.delta_tokens > 0;
        if reinitialized {
            expected_position = 0;
        }
        let expected_carried = usize::from(ordinal > 0 && !reinitialized);
        anyhow::ensure!(
            end.carried == expected_carried,
            "turn {ordinal} carried {} token(s), expected {expected_carried}",
            end.carried
        );
        if ordinal == 0 {
            anyhow::ensure!(
                end.delta_tokens > 0,
                "the first Drive turn did not receive its system prompt and memory cover"
            );
        }
        expected_position = expected_position
            .checked_add(end.delta_tokens)
            .and_then(|n| n.checked_add(end.carried))
            .and_then(|n| n.checked_add(end.tokens - 1))
            .context("expected session position overflow")?;
        anyhow::ensure!(
            end.position == expected_position,
            "turn {ordinal} ended at position {}, but its exact inputs and generated steps imply \
             {expected_position}",
            end.position
        );
    }
    Ok(())
}

/// The protocol alone, with a deterministic raw delta between turns.
fn cmd_probe(o: &Options) -> Result<()> {
    let mut client = serve_client(o)?;
    client.feed(&o.system)?;
    println!("\n=== turns against ONE session, with deltas ===");
    println!("  FRAMING: seconds per FIRST TOKEN OF A TURN inside the serving process, over the");
    println!("  READY layer range. A paired proxy reports the slower rank (the critical path).");
    let mut ends = Vec::new();
    for turn in 0..o.turns {
        if turn > 0
            && let Some(text) = o.feeds.get(turn - 1)
        {
            client.feed(text)?;
        }
        let started = std::time::Instant::now();
        let mut first_token_at = None;
        let mut said = String::new();
        let end = client.consult(&Consult::new(o.tokens), |text| {
            if first_token_at.is_none() {
                first_token_at = Some(started.elapsed().as_secs_f64());
            }
            said.push_str(text);
            Ok(())
        })?;
        println!("  {}", end.summary());
        println!(
            "    first token reached THIS process at {:.3}s; whole turn took {:.3}s here",
            first_token_at.unwrap_or(f64::NAN),
            started.elapsed().as_secs_f64()
        );
        println!("    said: {said:?}");
        ends.push(end);
    }
    if ends.len() >= 2 {
        let cold = ends[0].first_token_secs;
        let warm: Vec<f64> = ends[1..].iter().map(|t| t.first_token_secs).collect();
        let mean = warm.iter().sum::<f64>() / warm.len() as f64;
        println!(
            "  COLD/WARM: turn 0 {cold:.3}s, turns 1..{} mean {mean:.3}s — {:.0}x",
            ends.len() - 1,
            cold / mean.max(f64::EPSILON)
        );
    }

    // THE CARRY, on the real serving process. A turn emits its last token and
    // never feeds it back — that step is deliberately skipped — so the next pass
    // must append it, or the model permanently stops hearing its own last word.
    // Turn 0 has nothing to carry; every turn after it carries exactly one.
    //
    // This is the STRUCTURAL half of the check and it is cheap: it says the
    // carry happened, on the binary that ships. That it is the RIGHT thing to
    // carry is the behavioural half, and it lives in `inkling_session --carry`,
    // which compares a served two-turn conversation against a session fed the
    // identical token sequence in one pass.
    let uncarried: Vec<usize> = ends
        .iter()
        .filter(|e| e.turn > 0 && e.carried == 0)
        .map(|e| e.turn)
        .collect();
    println!(
        "  carry: turn 0 carried {}, turns 1..{} carried {:?}",
        ends.first().map(|e| e.carried).unwrap_or(0),
        ends.len(),
        ends.iter().skip(1).map(|e| e.carried).collect::<Vec<_>>(),
    );
    anyhow::ensure!(
        ends.first().map(|e| e.carried).unwrap_or(0) == 0,
        "turn 0 carried a token, and there was no previous turn for it to come from"
    );
    anyhow::ensure!(
        uncarried.is_empty(),
        "turn(s) {uncarried:?} did not carry the previous turn's last token forward, so the \
         model never attended to its own final word of those turns"
    );

    let status = client.close()?;
    println!("  serving process exited: {status}");
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn turn(
        turn: usize,
        tokens: usize,
        delta_tokens: usize,
        carried: usize,
        position: usize,
    ) -> TurnEnd {
        TurnEnd {
            turn,
            tokens,
            token_ids: (0..tokens as u32).collect(),
            delta_tokens,
            carried,
            stopped: "content_model_end_sampling".to_string(),
            first_token_secs: 0.1,
            turn_secs: 0.2,
            position,
        }
    }

    #[test]
    fn drive_position_is_the_sum_of_inputs_and_internal_decode_steps() {
        let turns = [turn(0, 3, 20, 0, 22), turn(1, 2, 0, 1, 24)];
        verify_drive_turns(&turns).expect("the exact sequence accounts for every KV row");
    }

    #[test]
    fn typed_post_initialization_deltas_participate_in_position_accounting() {
        let turns = [turn(0, 3, 20, 0, 22), turn(1, 2, 7, 1, 31)];
        verify_drive_turns(&turns)
            .expect("a tool result or generation prompt is a legitimate later delta");
    }

    #[test]
    fn a_reinitialized_epoch_restarts_position_without_a_carry() {
        let turns = [
            turn(0, 3, 20, 0, 22),
            turn(1, 2, 0, 1, 24),
            turn(2, 2, 17, 0, 18),
            turn(3, 1, 0, 1, 19),
        ];
        verify_drive_turns(&turns)
            .expect("the replacement initialization starts one fresh Session epoch");
    }

    #[test]
    fn a_fast_turn_cannot_hide_a_forgotten_position() {
        let turns = [turn(0, 1, 20, 0, 20), turn(1, 1, 0, 1, 20)];
        let error = verify_drive_turns(&turns).expect_err("turn 1 forgot its carried token");
        assert!(error.to_string().contains("imply 21"), "{error:#}");
    }

    #[test]
    fn drive_refuses_before_starting_without_an_exact_expected_command() {
        let options = parse(&["--serve".into(), "must-not-start".into()]).expect("valid options");
        let error = cmd_drive(&options).expect_err("an unbounded action gate must refuse");
        assert!(
            error.to_string().contains("--expect-command is required"),
            "{error:#}"
        );

        let options = parse(&[
            "--serve".into(),
            "must-not-start".into(),
            "--expect-command".into(),
            "true".into(),
            "--exec-timeout-ms".into(),
            "0".into(),
        ])
        .expect("valid options");
        let error = cmd_drive(&options).expect_err("an unbounded timeout must refuse");
        assert!(
            error
                .to_string()
                .contains("--exec-timeout-ms must be greater than zero"),
            "{error:#}"
        );
    }

    #[test]
    fn expected_action_requires_one_exact_successful_durable_result() {
        let expected = ActionPolicy {
            command: "printf native-tp2-ok",
            timeout_ms: 5_000,
        };
        let good = ActionEvidence {
            command: Some("printf native-tp2-ok"),
            has_content: true,
            is_error: Some(false),
            exit_code: Some(0),
            has_result_id: true,
        };
        verify_action_evidence(&[good], expected).expect("the exact successful Fire is the gate");

        for bad in [
            ActionEvidence {
                command: Some("printf wrong"),
                ..good
            },
            ActionEvidence {
                is_error: Some(true),
                ..good
            },
            ActionEvidence {
                exit_code: Some(1),
                ..good
            },
            ActionEvidence {
                has_result_id: false,
                ..good
            },
        ] {
            verify_action_evidence(&[bad], expected)
                .expect_err("a mismatched or incomplete action must fail the gate");
        }
        verify_action_evidence(&[], expected).expect_err("no Fire must fail the gate");
        verify_action_evidence(&[good, good], expected)
            .expect_err("a second Fire must fail the gate");
    }
}
