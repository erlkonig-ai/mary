//! `inkling_drive` — **ONE BINARY**: the Drive loop AND the model, in one
//! process, deployed unchanged to both DGX Sparks.
//!
//! ```text
//!   # the SAME command line on BOTH boxes
//!   inkling_drive --model <model.pile> --tokenizer <tokenizer.json> \
//!                 --layers 0:42 --tp-world 2 --tp-rendezvous <rank0-fabric>:29500 \
//!                 --playground <playground> --live
//! ```
//!
//! # What this replaced
//!
//! Until 2026-08-30 a resident run was THREE process kinds and a protocol:
//! `inkling_drive` (GPU-free) spawned `inkling_serve_pair`, which spawned two
//! `inkling_serve` ranks — the second over passwordless `ssh` — and fanned one
//! framed stream out to both, comparing their token text byte for byte.
//!
//! Measured on hardware, per generated token, 42 layers, TP2 across both boxes,
//! inside the serving process: decode within one 32-token consult was
//! 55.8 / 58.7 ms, while decode AS DRIVE ACTUALLY USED IT was p50 82 ms
//! (n = 768, min 58, p25 65, p75 114, p95 186). **About 26 ms a token — a third
//! of resident decode — was the protocol**, because Drive consults one token at
//! a time and every token paid a framed-stream round trip through the proxy, a
//! fan-out to two rank pipes and an `ssh` channel. That is what this deletes.
//!
//! It also deletes the `ssh` trust edge: rank 0 no longer launches rank 1, so
//! the two boxes are independent failure domains and independent security
//! domains. And it deletes version skew as a category — there is one binary, and
//! its exact bytes enter the execution identity both ranks announce.
//!
//! # Which box is which
//!
//! Nothing in the invocation says. `mary::models::inkling::tpcomm::elect_rank`
//! compares this box's own addresses to the `--tp-rendezvous` host: the box that
//! HOLDS that address is rank 0 (it binds, owns the Drive loop, owns the
//! cognition pile and the sandbox); the box that does not is rank 1 and runs as
//! a pure model rank. The rendezvous address already had to name rank 0 —
//! `Group::form` binds it on rank 0 and dials it everywhere else — so this is
//! reading configuration that already existed rather than adding any.
//!
//! # Still not a daemon
//!
//! One model, one Drive sandbox session and one cognition ledger for the whole
//! invocation. `--live` ends cooperatively at the next generated token boundary
//! on the first SIGINT; a second SIGINT is the force-kill escape hatch.

use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};

use anyhow::{Context, Result};
use drive::config::{ExecConfig, PileConfig};
use drive::context::{MemoryConfig, ModelBudget};
use drive::shell::{Extent, Shell, ShellConfig, TurnOutcome};
use drive::stream::FacultyCommand;
use mary::models::inkling::engine::{self, EngineConfig, Loaded, TensorParallel};
use mary::models::inkling::resident::{InklingMind, Ready, StreamProof, TurnEnd};
use mary::models::inkling::tpcomm::elect_rank;

const DEFAULT_SYSTEM: &str = "You are a mind in a shell. Think out loud in your own words, and run \
faculties when you want to know or change something. What you run comes back to you as a result.";

fn usage() -> &'static str {
    "\
inkling_drive — the Drive loop and the model, one process, one binary, both boxes

USAGE:
    inkling_drive --model <model.pile> --tokenizer <tokenizer.json> \\
                  --playground <path> (--turns <n> | --live) [OPTIONS]

MODEL:
    --model <path>          The model collection: weights AND config.json [required]
    --tokenizer <path>      The checkpoint's tokenizer.json [required]
    --layers <lo:hi>        Layers this rank runs (a TP rank runs all of them)
    --prefill-budget <n>    Maximum tokens processed in one prefill pass
    --context-budget <n>    Maximum positions retained (default: the prefill budget)
    --sealed                Reject execution-changing environment overrides
    --system <text>         System prompt

TENSOR PARALLELISM (both flags together, or neither):
    --tp-world <n>          Number of ranks. Address-match election supports 2.
    --tp-rendezvous <a>     RANK 0's HOST:PORT on the fast fabric. The box that
                            holds this address IS rank 0 and owns the Drive
                            loop; the box that does not is a pure model rank.
                            There is no --tp-rank: the two invocations are
                            identical, not merely the two binaries.

LIFETIME:
    --turns <n>             Run exactly this many Drive turns, then report the
                            per-turn and per-token decode evidence
    --live                  Run until SIGINT, stopping at the next token boundary

DRIVE / SANDBOX (rank 0 only; rank 1 parses and ignores them):
    --pile <path>           Scratch COGNITION pile (default: unique /tmp path).
                            Not --model; this is the ledger, not the weights.
    --playground <path>     `playground` or protocol-compatible binary [required]
    --backend <name>        `playground mcp --backend <name>` (default: lima)
    --backend-arg <arg>     Backend/policy argument; repeatable and passed verbatim
    --tenant <name>         Sandbox tenant (default: default)
    --exec-timeout <ms>     Per-command wall-clock timeout

OPTIONAL FACULTIES:
    --voice <path>          Streaming speech faculty
    --voice-arg <arg>       Voice argument; repeatable
    --telemetry-pile <path> Per-turn exhaust pile (default: the cognition pile's
                            `.turns.pile` sibling)
    --no-telemetry          Do not write turn telemetry
    --memory-pile <path>    Durable self pile whose cover is injected at wake
    --memory-bin <path>     Memory faculty (default: memory on PATH)
    --max-output <n>        Hard tokens per response and output reservation
                            (default: 8192)
    --context-margin <n>    Additional reserved tokens (default: 4096)
    --chars-per-token <n>   Cover sizing approximation (default: 4)

The runner adds no command allowlist or host-exec escape hatch. Sandbox policy
is exactly the selected playground backend plus the explicitly supplied
--backend-arg values. The model's own context budget is also the single capacity
used to size Drive's memory cover; there is no second runner window to keep in
sync.
"
}

#[derive(Clone, Debug)]
struct Options {
    model: Option<PathBuf>,
    tokenizer: Option<PathBuf>,
    layers: Option<std::ops::Range<usize>>,
    prefill_budget: Option<usize>,
    context_budget: Option<usize>,
    sealed: bool,
    tp_world: Option<usize>,
    tp_rendezvous: Option<String>,
    system: String,
    turns: Option<usize>,
    live: bool,
    pile: PathBuf,
    playground: Option<PathBuf>,
    backend: String,
    backend_args: Vec<String>,
    tenant: String,
    exec_timeout_ms: Option<u64>,
    voice: Option<PathBuf>,
    voice_args: Vec<String>,
    telemetry: Option<PathBuf>,
    no_telemetry: bool,
    memory_pile: Option<PathBuf>,
    memory_bin: PathBuf,
    budget: ModelBudget,
}

impl Default for Options {
    fn default() -> Self {
        let stamp = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos();
        Self {
            model: None,
            tokenizer: None,
            layers: None,
            prefill_budget: None,
            context_budget: None,
            sealed: false,
            tp_world: None,
            tp_rendezvous: None,
            system: DEFAULT_SYSTEM.to_string(),
            turns: None,
            live: false,
            pile: std::env::temp_dir().join(format!("inkling-drive-{stamp}.pile")),
            playground: None,
            backend: "lima".to_string(),
            backend_args: Vec::new(),
            tenant: "default".to_string(),
            exec_timeout_ms: None,
            voice: None,
            voice_args: Vec::new(),
            telemetry: None,
            no_telemetry: false,
            memory_pile: None,
            memory_bin: PathBuf::from("memory"),
            budget: ModelBudget::default(),
        }
    }
}

impl Options {
    fn parse(args: &[String]) -> Result<Self> {
        let mut options = Self::default();
        let mut index = 0usize;
        let next = |index: &mut usize, flag: &str| -> Result<String> {
            *index += 1;
            args.get(*index)
                .cloned()
                .with_context(|| format!("{flag} needs a value"))
        };

        while index < args.len() {
            match args[index].as_str() {
                "--model" => options.model = Some(PathBuf::from(next(&mut index, "--model")?)),
                "--tokenizer" => {
                    options.tokenizer = Some(PathBuf::from(next(&mut index, "--tokenizer")?));
                }
                "--layers" => {
                    let value = next(&mut index, "--layers")?;
                    let (lo, hi) = value
                        .split_once(':')
                        .with_context(|| format!("--layers wants LO:HI, got {value:?}"))?;
                    options.layers = Some(lo.parse()?..hi.parse()?);
                }
                "--prefill-budget" => {
                    options.prefill_budget = Some(
                        next(&mut index, "--prefill-budget")?
                            .parse()
                            .context("--prefill-budget wants a count")?,
                    );
                }
                "--context-budget" => {
                    options.context_budget = Some(
                        next(&mut index, "--context-budget")?
                            .parse()
                            .context("--context-budget wants a count")?,
                    );
                }
                "--sealed" => options.sealed = true,
                "--tp-world" => {
                    options.tp_world = Some(
                        next(&mut index, "--tp-world")?
                            .parse()
                            .context("--tp-world wants a count")?,
                    );
                }
                "--tp-rendezvous" => {
                    options.tp_rendezvous = Some(next(&mut index, "--tp-rendezvous")?);
                }
                "--system" => options.system = next(&mut index, "--system")?,
                "--turns" => {
                    options.turns = Some(
                        next(&mut index, "--turns")?
                            .parse()
                            .context("--turns takes a count")?,
                    );
                }
                "--live" => options.live = true,
                "--pile" => options.pile = PathBuf::from(next(&mut index, "--pile")?),
                "--playground" => {
                    options.playground = Some(PathBuf::from(next(&mut index, "--playground")?));
                }
                "--backend" => options.backend = next(&mut index, "--backend")?,
                "--backend-arg" => options
                    .backend_args
                    .push(next(&mut index, "--backend-arg")?),
                "--tenant" => options.tenant = next(&mut index, "--tenant")?,
                "--exec-timeout" => {
                    options.exec_timeout_ms = Some(
                        next(&mut index, "--exec-timeout")?
                            .parse()
                            .context("--exec-timeout takes milliseconds")?,
                    );
                }
                "--voice" => options.voice = Some(PathBuf::from(next(&mut index, "--voice")?)),
                "--voice-arg" => options.voice_args.push(next(&mut index, "--voice-arg")?),
                "--telemetry-pile" => {
                    options.telemetry = Some(PathBuf::from(next(&mut index, "--telemetry-pile")?));
                }
                "--no-telemetry" => options.no_telemetry = true,
                "--memory-pile" => {
                    options.memory_pile = Some(PathBuf::from(next(&mut index, "--memory-pile")?));
                }
                "--memory-bin" => {
                    options.memory_bin = PathBuf::from(next(&mut index, "--memory-bin")?);
                }
                "--max-output" => {
                    options.budget.max_output_tokens = next(&mut index, "--max-output")?
                        .parse()
                        .context("--max-output takes tokens")?;
                }
                "--context-margin" => {
                    options.budget.safety_margin_tokens = next(&mut index, "--context-margin")?
                        .parse()
                        .context("--context-margin takes tokens")?;
                }
                "--chars-per-token" => {
                    options.budget.chars_per_token = next(&mut index, "--chars-per-token")?
                        .parse()
                        .context("--chars-per-token takes a number")?;
                }
                "-h" | "--help" => anyhow::bail!("help requested\n\n{}", usage()),
                other => anyhow::bail!("unknown option {other:?}\n\n{}", usage()),
            }
            index += 1;
        }

        anyhow::ensure!(options.model.is_some(), "--model is required");
        anyhow::ensure!(options.tokenizer.is_some(), "--tokenizer is required");
        anyhow::ensure!(options.playground.is_some(), "--playground is required");
        anyhow::ensure!(
            options.budget.max_output_tokens > 0,
            "--max-output must be greater than zero"
        );
        anyhow::ensure!(
            matches!(
                (options.live, options.turns),
                (true, None) | (false, Some(_))
            ),
            "choose exactly one of --turns <n> or --live"
        );
        anyhow::ensure!(
            options.turns != Some(0),
            "--turns must be greater than zero"
        );
        anyhow::ensure!(
            options.exec_timeout_ms != Some(0),
            "--exec-timeout must be greater than zero"
        );
        anyhow::ensure!(
            !(options.no_telemetry && options.telemetry.is_some()),
            "--no-telemetry and --telemetry-pile contradict each other"
        );
        anyhow::ensure!(
            options.voice.is_some() || options.voice_args.is_empty(),
            "--voice-arg without --voice has no recipient"
        );
        anyhow::ensure!(
            options.memory_pile.as_ref() != Some(&options.pile),
            "--memory-pile and --pile must differ: durable memory is read-only and cognition \
             provenance is scratch output"
        );
        anyhow::ensure!(
            options.model.as_ref() != Some(&options.pile),
            "--model and --pile must differ: --model is the weights, --pile is this run's \
             cognition ledger"
        );
        anyhow::ensure!(
            options.tp_world.is_some() == options.tp_rendezvous.is_some(),
            "--tp-world and --tp-rendezvous are one launch contract; provide both or neither. \
             There is deliberately no --tp-rank: which box is rank 0 is decided by which box \
             holds the rendezvous address."
        );
        Ok(options)
    }

    fn extent(&self) -> Extent {
        match (self.live, self.turns) {
            (true, None) => Extent::Live,
            (false, Some(turns)) => Extent::Turns(turns),
            _ => unreachable!("parse enforces one extent"),
        }
    }

    /// How to load this box's rank, once the election has decided which it is.
    fn engine_config(&self) -> Result<EngineConfig> {
        let tensor_parallel = match (&self.tp_rendezvous, self.tp_world) {
            (Some(rendezvous), Some(world)) => Some(TensorParallel {
                tp: elect_rank(rendezvous, world)
                    .context("decide this box's tensor-parallel rank")?,
                rendezvous: rendezvous.clone(),
            }),
            _ => None,
        };
        Ok(EngineConfig {
            pile: self
                .model
                .clone()
                .expect("parse enforces a model collection"),
            tokenizer: self.tokenizer.clone().expect("parse enforces a tokenizer"),
            layers: self.layers.clone(),
            prefill_budget: self.prefill_budget,
            context_budget: self.context_budget,
            tensor_parallel,
            sealed: self.sealed,
        })
    }

    fn shell_config(&self, context_window_tokens: u64) -> ShellConfig {
        let voice = self.voice.as_ref().map(|program| FacultyCommand {
            program: program.clone(),
            args: self.voice_args.clone(),
        });
        let telemetry = match (self.no_telemetry, &self.telemetry) {
            (true, _) => None,
            (false, Some(path)) => Some(path.clone()),
            (false, None) => Some(drive::telemetry::default_turn_pile(&self.pile)),
        };
        let memory = self.memory_pile.as_ref().map(|pile| MemoryConfig {
            binary: self.memory_bin.clone(),
            pile: pile.clone(),
            budget: ModelBudget {
                context_window_tokens,
                ..self.budget
            },
        });
        ShellConfig {
            pile: PileConfig {
                path: self.pile.clone(),
            },
            exec: ExecConfig {
                playground_bin: self
                    .playground
                    .clone()
                    .expect("parse enforces a playground"),
                backend: self.backend.clone(),
                extra_args: self.backend_args.clone(),
                tenant: self.tenant.clone(),
                timeout_ms: self.exec_timeout_ms,
            },
            system_prompt: self.system.clone(),
            voice,
            telemetry,
            memory,
        }
    }
}

static SIGINT_STOP: AtomicBool = AtomicBool::new(false);

extern "C" fn on_sigint(_signal: libc::c_int) {
    SIGINT_STOP.store(true, Ordering::Relaxed);
    // A second SIGINT is the escape hatch if a backend or teardown is wedged.
    unsafe {
        libc::signal(libc::SIGINT, libc::SIG_DFL);
    }
}

fn install_sigint_stop() {
    unsafe {
        libc::signal(libc::SIGINT, on_sigint as *const () as libc::sighandler_t);
    }
}

#[derive(Default)]
struct RunObservations {
    turns: usize,
    fired: usize,
    result_errors: usize,
    said_bytes: u64,
}

impl RunObservations {
    fn observe(&mut self, outcome: TurnOutcome) {
        self.turns += 1;
        self.fired += usize::from(outcome.command.is_some());
        self.result_errors += usize::from(outcome.is_error == Some(true));
        self.said_bytes = self.said_bytes.saturating_add(outcome.said.len() as u64);
        eprintln!(
            "inkling_drive: turn {} at {} — {:?}, {} byte(s), command {}, result error {}",
            self.turns,
            outcome.coord,
            outcome.disposition,
            outcome.said.len(),
            outcome.command.is_some(),
            outcome.is_error == Some(true),
        );
        // `outcome` dies here. A live run retains no per-turn content vector.
    }
}

fn validate_ready(ready: &Ready) -> Result<()> {
    anyhow::ensure!(
        !ready.partial,
        "this rank loaded a partial stack; its tokens are diagnostic, not the model's. A \
         full-stack resident run needs --layers 0:<stack> with --tp-world/--tp-rendezvous, \
         because 144 GiB of weights do not fit one 121 GiB box"
    );
    anyhow::ensure!(
        matches!(
            ready.execution_profile.as_str(),
            "sealed-v1" | "observed-v1"
        ),
        "unsupported execution profile {:?}",
        ready.execution_profile
    );
    anyhow::ensure!(
        ready.execution_identity.len() == 64
            && ready
                .execution_identity
                .bytes()
                .all(|byte| byte.is_ascii_hexdigit()),
        "invalid execution identity {:?}",
        ready.execution_identity
    );
    Ok(())
}

fn panic_message(panic: &(dyn std::any::Any + Send)) -> String {
    if let Some(message) = panic.downcast_ref::<&str>() {
        (*message).to_string()
    } else if let Some(message) = panic.downcast_ref::<String>() {
        message.clone()
    } else {
        "non-string panic payload".to_string()
    }
}

fn caught(result: std::thread::Result<Result<()>>, phase: &str) -> Result<()> {
    match result {
        Ok(result) => result,
        Err(panic) => anyhow::bail!("{phase} panicked: {}", panic_message(&*panic)),
    }
}

fn combine_run_and_finish(run: Result<()>, finish: Result<()>) -> Result<()> {
    match (run, finish) {
        (Ok(()), Ok(())) => Ok(()),
        (Err(run), Ok(())) => Err(run),
        (Ok(()), Err(finish)) => Err(finish).context("finish the resident shell"),
        (Err(run), Err(finish)) => Err(anyhow::anyhow!(
            "resident loop failed: {run:#}; teardown also failed: {finish:#}"
        )),
    }
}

/// One number's worth of order statistics, with the framing rule attached.
///
/// This is the whole point of a finite `--turns` run: it produces the AFTER
/// number for the module header's BEFORE number, at the same granularity and
/// measured in the same place, so the two are comparable without
/// reconstruction.
fn report_decode_distribution(proofs: &[StreamProof]) {
    let mut secs: Vec<f64> = proofs
        .iter()
        .flat_map(|proof| proof.token_secs.iter().copied())
        .collect();
    if secs.is_empty() {
        println!("decode: no tokens generated");
        return;
    }
    secs.sort_by(|left, right| {
        left.partial_cmp(right)
            .expect("a measured duration is never NaN")
    });
    let quantile = |q: f64| -> f64 {
        let index = (((secs.len() - 1) as f64) * q).round() as usize;
        secs[index] * 1_000.0
    };
    println!(
        "decode: n={} min {:.1} p25 {:.1} p50 {:.1} p75 {:.1} p95 {:.1} max {:.1} ms",
        secs.len(),
        secs[0] * 1_000.0,
        quantile(0.25),
        quantile(0.50),
        quantile(0.75),
        quantile(0.95),
        secs[secs.len() - 1] * 1_000.0,
    );
    println!(
        "  framing rule: MILLISECONDS PER GENERATED TOKEN, one one-token consult each — the \
         granularity Drive actually generates at, not a multi-token consult amortised. Measured \
         around this process's own Session calls. The number to compare it against is the \
         framed-stream baseline in this binary's header: p50 82 ms (n=768, min 58, p25 65, \
         p75 114, p95 186), 42 layers, TP2, both Sparks."
    );
}

fn report_turns(log: &[TurnEnd], proofs: &[StreamProof]) {
    for end in log {
        println!("{}", end.summary());
    }
    let streamed = proofs.iter().filter(|proof| proof.streamed()).count();
    println!(
        "streaming: {streamed} of {} turn(s) produced consumer output before the mind stopped \
         generating",
        proofs.len()
    );
    report_decode_distribution(proofs);
}

/// Rank 1: no Drive, no pile, no sandbox, no tokenizer. Replay rank 0's passes.
fn run_follower(mut follower: engine::Follower) -> Result<()> {
    validate_ready(follower.ready())?;
    eprintln!(
        "inkling_drive: FOLLOWER — rank {:?} of {}, layers {}..{}, {} {}. This box owns no \
         cognition pile and makes no decisions; it replays rank 0's passes.",
        follower.ready().tp_rank,
        follower.ready().tp_world,
        follower.ready().layers[0],
        follower.ready().layers[1],
        follower.ready().execution_profile,
        follower.ready().execution_identity,
    );
    follower.follow()
}

fn run(options: Options) -> Result<()> {
    // Armed before the load, because the load is minutes and a run interrupted
    // during it should still be interruptible once rather than twice.
    install_sigint_stop();
    // The election runs BEFORE the model loads, because it decides which of the
    // two `Session::load` paths this box takes, and it must be able to refuse a
    // rendezvous address that names neither box without first spending minutes
    // mapping a 171 GB pile.
    let loaded = engine::load(options.engine_config()?).context("load this box's rank")?;
    let engine = match loaded {
        Loaded::Follower(follower) => return run_follower(follower),
        Loaded::Engine(engine) => engine,
    };

    let max_response_tokens = usize::try_from(options.budget.max_output_tokens)
        .context("--max-output does not fit this platform's token index")?;
    validate_ready(engine.ready())?;
    anyhow::ensure!(
        max_response_tokens.saturating_add(1) <= engine.ready().context_budget,
        "the {max_response_tokens}-token response cap plus its one-token carry/prompt admission \
         exceeds the model's {}-token context budget",
        engine.ready().context_budget
    );
    let context_window_tokens = u64::try_from(engine.ready().context_budget)
        .context("the model's context budget does not fit Drive's token budget")?;
    eprintln!(
        "inkling_drive: LEADER — {} layers, context {}, {}, {}",
        engine.ready().stack,
        engine.ready().context_budget,
        engine.ready().execution_profile,
        engine.ready().execution_identity,
    );

    // A finite run retains per-turn and per-token evidence so it can report the
    // decode distribution afterwards; an unbounded one must not accumulate one
    // `f64` per token forever. The extent already draws that line, so there is
    // no separate flag for it.
    let finite = options.turns.is_some();
    let mind = match finite {
        true => InklingMind::new_gate(
            Box::new(engine),
            max_response_tokens,
            Some(options.system.clone()),
        ),
        false => InklingMind::new(
            Box::new(engine),
            max_response_tokens,
            Some(options.system.clone()),
        ),
    }?
    .with_cancellation(|| SIGINT_STOP.load(Ordering::Relaxed));

    let voice_slot = mind.voice_slot();
    let log = mind.log();
    let proofs = mind.proofs();
    let shell_config = options.shell_config(context_window_tokens);
    eprintln!(
        "inkling_drive: opening scratch ledger {} and sandbox {} mcp --backend {}",
        shell_config.pile.path.display(),
        shell_config.exec.playground_bin.display(),
        shell_config.exec.backend,
    );
    let mut shell = Shell::open(&shell_config, Box::new(mind))
        .context("open Drive's shell around the resident Inkling mind")?;

    let mut observations = RunObservations::default();
    let run = caught(
        std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            *voice_slot.lock().expect("voice slot") = shell.claim_voice();
            eprintln!(
                "inkling_drive: session {:X} — drive {}",
                shell.session(),
                drive::DRIVE_GIT_REV,
            );
            shell.run_each(options.extent(), &SIGINT_STOP, |outcome| {
                observations.observe(outcome);
            })
        })),
        "resident shell loop",
    );
    // `finish` consumes Shell, finalizes/flushes both Drive ledgers, and then
    // drops InklingMind, whose Drop calls `Model::shutdown` — which tells the
    // OTHER BOX the run is over. Without that the peer sits in `follow`
    // forever, holding its whole arena. It is attempted after success, error,
    // or panic.
    let finish = caught(
        std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| shell.finish())),
        "resident shell teardown",
    );
    let result = combine_run_and_finish(run, finish);
    eprintln!(
        "inkling_drive: {} turn(s), {} command(s), {} error result(s), {} spoken byte(s)",
        observations.turns, observations.fired, observations.result_errors, observations.said_bytes,
    );
    if let (Some(log), Some(proofs)) = (log, proofs) {
        let log = log.lock().expect("turn log").clone();
        let proofs = proofs.lock().expect("proof log").clone();
        report_turns(&log, &proofs);
    }
    result
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if matches!(
        args.first().map(String::as_str),
        None | Some("-h" | "--help")
    ) {
        print!("{}", usage());
        return;
    }
    if let Err(error) = Options::parse(&args).and_then(run) {
        eprintln!("inkling_drive: {error:#}");
        std::process::exit(1);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn strings(args: &[&str]) -> Vec<String> {
        args.iter().map(|arg| (*arg).to_string()).collect()
    }

    fn minimum(extra: &[&str]) -> Vec<String> {
        let mut args = strings(&[
            "--model",
            "/models/inkling.pile",
            "--tokenizer",
            "/models/tokenizer.json",
            "--playground",
            "playground",
            "--turns",
            "3",
        ]);
        args.extend(strings(extra));
        args
    }

    #[test]
    fn parser_requires_one_explicit_lifetime() {
        for args in [
            strings(&[
                "--model",
                "/models/inkling.pile",
                "--tokenizer",
                "/models/tokenizer.json",
                "--playground",
                "playground",
            ]),
            minimum(&["--live"]),
            minimum(&["--turns", "0"]),
        ] {
            let error = Options::parse(&args).expect_err("invalid lifetime must refuse");
            assert!(
                error.to_string().contains("--turns") || error.to_string().contains("--live"),
                "{error:#}"
            );
        }

        assert_eq!(
            Options::parse(&minimum(&[])).unwrap().extent(),
            Extent::Turns(3)
        );
        let live = Options::parse(&strings(&[
            "--model",
            "/models/inkling.pile",
            "--tokenizer",
            "/models/tokenizer.json",
            "--playground",
            "playground",
            "--live",
        ]))
        .unwrap();
        assert_eq!(live.extent(), Extent::Live);
    }

    /// There is no `--tp-rank`, and that is the deployment property being
    /// bought: the two boxes run the same COMMAND, not merely the same binary.
    #[test]
    fn tensor_parallelism_is_one_contract_with_no_rank_in_it() {
        let error = Options::parse(&minimum(&["--tp-world", "2"]))
            .expect_err("a world without a rendezvous names no rank 0");
        assert!(
            error.to_string().contains("one launch contract"),
            "{error:#}"
        );

        let error = Options::parse(&minimum(&["--tp-rendezvous", "10.0.0.1:29500"]))
            .expect_err("a rendezvous without a world sizes nothing");
        assert!(
            error.to_string().contains("one launch contract"),
            "{error:#}"
        );

        Options::parse(&minimum(&[
            "--tp-world",
            "2",
            "--tp-rendezvous",
            "10.0.0.1:29500",
        ]))
        .expect("both together are the whole contract");

        Options::parse(&minimum(&["--tp-rank", "0"]))
            .expect_err("--tp-rank must not exist: the invocations are identical");
    }

    /// `--model` is the weights and `--pile` is the cognition ledger. They were
    /// both spelled `--pile` in the two binaries this one replaces, in opposite
    /// senses, so the merged surface has to keep them apart loudly.
    #[test]
    fn the_weights_and_the_ledger_are_different_piles() {
        let options = Options::parse(&minimum(&["--pile", "/tmp/cognition.pile"])).unwrap();
        assert_eq!(options.model, Some(PathBuf::from("/models/inkling.pile")));
        assert_eq!(options.pile, PathBuf::from("/tmp/cognition.pile"));

        let error = Options::parse(&minimum(&["--pile", "/models/inkling.pile"]))
            .expect_err("the ledger must not be written over the weights");
        assert!(
            error.to_string().contains("--model and --pile"),
            "{error:#}"
        );
    }

    #[test]
    fn composition_preserves_real_sandbox_and_optional_faculty_shape() {
        let options = Options::parse(&minimum(&[
            "--layers",
            "0:42",
            "--pile",
            "/tmp/cognition.pile",
            "--backend",
            "jail",
            "--backend-arg",
            "--jail-local",
            "--tenant",
            "jp",
            "--exec-timeout",
            "7000",
            "--voice",
            "voice",
            "--voice-arg",
            "speak",
            "--telemetry-pile",
            "/tmp/turns.pile",
            "--memory-pile",
            "/data/self.pile",
            "--memory-bin",
            "/bin/memory",
            "--max-output",
            "4096",
            "--context-margin",
            "2048",
            "--chars-per-token",
            "3",
        ]))
        .unwrap();

        assert_eq!(options.layers, Some(0..42));
        let config = options.shell_config(131_072);
        assert_eq!(config.exec.playground_bin, PathBuf::from("playground"));
        assert_eq!(config.exec.backend, "jail");
        assert_eq!(config.exec.extra_args, ["--jail-local"]);
        assert_eq!(config.exec.tenant, "jp");
        assert_eq!(config.exec.timeout_ms, Some(7_000));
        assert_eq!(config.pile.path, PathBuf::from("/tmp/cognition.pile"));
        assert_eq!(config.telemetry, Some(PathBuf::from("/tmp/turns.pile")));
        let voice = config.voice.expect("voice");
        assert_eq!(voice.program, PathBuf::from("voice"));
        assert_eq!(voice.args, ["speak"]);
        let memory = config.memory.expect("memory");
        assert_eq!(memory.pile, PathBuf::from("/data/self.pile"));
        assert_eq!(memory.binary, PathBuf::from("/bin/memory"));
        assert_eq!(memory.budget.context_window_tokens, 131_072);
        assert_eq!(memory.budget.max_output_tokens, 4_096);
        assert_eq!(memory.budget.safety_margin_tokens, 2_048);
        assert_eq!(memory.budget.chars_per_token, 3);
    }

    #[test]
    fn default_composition_adds_no_backend_policy_or_permission() {
        let options = Options::parse(&minimum(&[])).unwrap();
        let config = options.shell_config(65_536);
        assert_eq!(config.exec.backend, "lima");
        assert!(config.exec.extra_args.is_empty());
        assert_eq!(config.exec.tenant, "default");
        assert_eq!(config.exec.timeout_ms, None);
        assert!(config.voice.is_none());
        assert!(config.memory.is_none());
        assert_eq!(
            config.telemetry,
            Some(drive::telemetry::default_turn_pile(&config.pile.path))
        );
    }

    #[test]
    fn parser_rejects_dangling_or_contradictory_faculty_options() {
        for extra in [
            vec!["--voice-arg", "orphan"],
            vec!["--no-telemetry", "--telemetry-pile", "/tmp/turns"],
            vec!["--exec-timeout", "0"],
            vec!["--max-output", "0"],
            vec!["--context-window", "131072"],
            vec!["--gen", "64"],
            // The deleted three-process launch surface. A stale command line
            // must refuse rather than be silently reinterpreted.
            vec!["--serve", "inkling_serve_pair"],
            vec!["--serve-arg", "--rank0-program"],
        ] {
            Options::parse(&minimum(&extra)).expect_err("contradiction must refuse");
        }
    }

    #[test]
    fn teardown_errors_never_hide_loop_errors() {
        let both = combine_run_and_finish(
            Err(anyhow::anyhow!("loop failed")),
            Err(anyhow::anyhow!("finish failed")),
        )
        .expect_err("both errors matter")
        .to_string();
        assert!(both.contains("loop failed"), "{both}");
        assert!(both.contains("finish failed"), "{both}");
    }
}
