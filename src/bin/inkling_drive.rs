//! `inkling_drive` — one resident Inkling session in Drive's foreground shell.
//!
//! This is deliberately not a daemon. It owns one serving process (normally an
//! `inkling_serve_pair`), one Drive sandbox session, and one cognition ledger
//! for the whole invocation. `--live` ends cooperatively at the next generated
//! token boundary on the first SIGINT; a second SIGINT is the force-kill escape
//! hatch.

use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};

use anyhow::{Context, Result};
use drive::config::{ExecConfig, PileConfig};
use drive::context::{MemoryConfig, ModelBudget};
use drive::shell::{Extent, Shell, ShellConfig, TurnOutcome};
use drive::stream::FacultyCommand;
use mary::models::inkling::serve::{InklingMind, Ready, ServeClient};

const DEFAULT_SYSTEM: &str = "You are a mind in a shell. Think out loud in your own words, and run \
faculties when you want to know or change something. What you run comes back to you as a result.";

fn usage() -> &'static str {
    "\
inkling_drive — one resident Inkling session in Drive's foreground shell

USAGE:
    inkling_drive --serve <path> --playground <path> (--turns <n> | --live) [OPTIONS]

MODEL:
    --serve <path>          `inkling_serve`-compatible process (normally
                            `inkling_serve_pair`) [required]
    --serve-arg <arg>       Argument passed to the serving process; repeatable
    --system <text>         System prompt

LIFETIME:
    --turns <n>             Run exactly this many Drive turns
    --live                  Run until SIGINT, stopping at the next token boundary

DRIVE / SANDBOX:
    --pile <path>           Scratch cognition pile (default: unique /tmp path)
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
--backend-arg values. The serving process's READY context budget is also the
single capacity used to size Drive's memory cover; there is no second runner
window to keep in sync.
"
}

#[derive(Clone, Debug)]
struct Options {
    serve: Option<PathBuf>,
    serve_args: Vec<String>,
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
            serve: None,
            serve_args: Vec::new(),
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
                "--serve" => options.serve = Some(PathBuf::from(next(&mut index, "--serve")?)),
                "--serve-arg" => options.serve_args.push(next(&mut index, "--serve-arg")?),
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

        anyhow::ensure!(options.serve.is_some(), "--serve is required");
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
        Ok(options)
    }

    fn extent(&self) -> Extent {
        match (self.live, self.turns) {
            (true, None) => Extent::Live,
            (false, Some(turns)) => Extent::Turns(turns),
            _ => unreachable!("parse enforces one extent"),
        }
    }

    fn serve_command(&self) -> std::process::Command {
        let mut command = std::process::Command::new(
            self.serve
                .as_ref()
                .expect("parse enforces a serving process"),
        );
        command.args(&self.serve_args);
        command
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
        "the serving process announced a partial stack; its tokens are diagnostic, not the \
         model's. Use inkling_serve_gate for partial-stack experiments"
    );
    anyhow::ensure!(
        matches!(
            ready.execution_profile.as_str(),
            "sealed-v1" | "observed-v1"
        ),
        "serving process announced unsupported execution profile {:?}",
        ready.execution_profile
    );
    anyhow::ensure!(
        ready.execution_identity.len() == 64
            && ready
                .execution_identity
                .bytes()
                .all(|byte| byte.is_ascii_hexdigit()),
        "serving process announced invalid execution identity {:?}",
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

fn run(options: Options) -> Result<()> {
    install_sigint_stop();

    let max_response_tokens = usize::try_from(options.budget.max_output_tokens)
        .context("--max-output does not fit this platform's token index")?;
    let mut serve = options.serve_command();
    eprintln!(
        "inkling_drive: loading one resident serving process: {} ({} argument(s))",
        serve.get_program().to_string_lossy(),
        options.serve_args.len(),
    );
    let client = ServeClient::spawn(&mut serve).context("start the Inkling serving process")?;
    let mind = InklingMind::new(client, max_response_tokens, Some(options.system.clone()))?
        .with_cancellation(|| SIGINT_STOP.load(Ordering::Relaxed));
    // Validate after the client is owned by InklingMind: every refusal below
    // therefore takes the same bounded shutdown/reap path as a completed run.
    validate_ready(mind.ready())?;
    anyhow::ensure!(
        max_response_tokens.saturating_add(1) <= mind.ready().context_budget,
        "the {max_response_tokens}-token response cap plus its one-token carry/prompt admission exceeds the model's {}-token context budget",
        mind.ready().context_budget
    );
    eprintln!(
        "inkling_drive: READY — {} layers, context {}, {}, {}",
        mind.ready().stack,
        mind.ready().context_budget,
        mind.ready().execution_profile,
        mind.ready().execution_identity,
    );

    let voice_slot = mind.voice_slot();
    let context_window_tokens = u64::try_from(mind.ready().context_budget)
        .context("the serving process context budget does not fit Drive's token budget")?;
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
    // drops InklingMind, whose Drop sends a complete END and boundedly reaps the
    // one serving process/pair. It is attempted after success, error, or panic.
    let finish = caught(
        std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| shell.finish())),
        "resident shell teardown",
    );
    let result = combine_run_and_finish(run, finish);
    eprintln!(
        "inkling_drive: {} turn(s), {} command(s), {} error result(s), {} spoken byte(s)",
        observations.turns, observations.fired, observations.result_errors, observations.said_bytes,
    );
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
            "--serve",
            "pair",
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
            strings(&["--serve", "pair", "--playground", "playground"]),
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
            "--serve",
            "pair",
            "--playground",
            "playground",
            "--live",
        ]))
        .unwrap();
        assert_eq!(live.extent(), Extent::Live);
    }

    #[test]
    fn composition_preserves_real_sandbox_and_optional_faculty_shape() {
        let options = Options::parse(&minimum(&[
            "--serve-arg",
            "--rank0-program",
            "--serve-arg",
            "/rank0",
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

        let serve = options.serve_command();
        assert_eq!(serve.get_program(), "pair");
        assert_eq!(
            serve.get_args().collect::<Vec<_>>(),
            ["--rank0-program", "/rank0"]
        );

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
