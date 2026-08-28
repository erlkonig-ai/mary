//! `inkling_serve_pair` — make two tensor-parallel `inkling_serve` ranks look
//! like one serving process.
//!
//! The downstream wire is exactly `mary::models::inkling::serve`: raw text,
//! typed context, and CONSULT go to both ranks, equal token fragments stream
//! back as soon as the second rank confirms them, and one TURN closes the turn.
//! Rank orchestration is expressed as structured program/argv/environment
//! fields; callers never supply a shell command blob.

use std::io::{Read, Write};

use anyhow::{Context, Result};

use mary::models::inkling::serve::{
    CONSULT_TYPE, CONTENT_TYPE, CONTEXT_PREFLIGHT_TYPE, CONTEXT_PREFLIGHTED_TYPE, CONTEXT_TYPE,
    Consult, ContextPreflight, InklingContext, READY_TYPE, REINITIALIZE_TYPE, REINITIALIZED_TYPE,
    RankCommand, ServePair, TURN_TYPE, UNIT,
};

fn usage() -> &'static str {
    "\
inkling_serve_pair — two full-stack tensor-parallel ranks, one serving stream

USAGE:
    inkling_serve_pair \\
      --rank0-program <path> [--rank0-host <host>] [--rank0-arg <arg>]... \\
      --rank1-program <path> [--rank1-host <host>] [--rank1-arg <arg>]...

RANK OPTIONS (replace N with 0 or 1):
    --rankN-program <path>    The rank executable on that host
    --rankN-host <host>       Launch through OpenSSH; omit for a local rank
    --rankN-ssh <path>        Local ssh executable (default: ssh)
    --rankN-supervisor <path> Remote inkling_serve_pair executable
    --rankN-arg <value>       One argv value; repeat in exact order
    --rankN-env <key=value>   One environment entry; repeat as needed

PAIR OPTIONS:
    --startup-timeout-secs <n>  Deadline for both READY records (default: 900)
    --shutdown-timeout-secs <n> Deadline for both ranks to exit (default: 60)

The rank program is normally inkling_serve, with --sealed, explicit
--tp-rank, --tp-world, --tp-rendezvous, full --layers, pile, and tokenizer
args. Pair startup refuses non-sealed READY records. Nothing here reads INK_TP
or guesses network placement.
"
}

#[derive(Default)]
struct RankOptions {
    host: Option<String>,
    ssh: Option<std::ffi::OsString>,
    supervisor: Option<std::ffi::OsString>,
    program: Option<std::ffi::OsString>,
    args: Vec<std::ffi::OsString>,
    env: Vec<(std::ffi::OsString, std::ffi::OsString)>,
}

impl RankOptions {
    fn finish(
        self,
        rank: usize,
        remote_shutdown_timeout: std::time::Duration,
    ) -> Result<RankCommand> {
        let program = self
            .program
            .with_context(|| format!("--rank{rank}-program is required"))?;
        let mut command = match self.host {
            Some(host) => RankCommand::ssh(host, program),
            None => RankCommand::local(program),
        };
        if let Some(ssh) = self.ssh {
            command = command.ssh_program(ssh);
        }
        if let Some(supervisor) = self.supervisor {
            command = command.remote_supervisor(supervisor);
        }
        command = command.remote_shutdown_timeout(remote_shutdown_timeout);
        for argument in self.args {
            command = command.arg(argument);
        }
        for (key, value) in self.env {
            command = command.env(key, value);
        }
        Ok(command)
    }
}

struct Options {
    commands: [RankCommand; 2],
    startup_timeout: std::time::Duration,
    shutdown_timeout: std::time::Duration,
}

fn parse() -> Result<Option<Options>> {
    let args: Vec<std::ffi::OsString> = std::env::args_os().skip(1).collect();
    if args.len() == 1 && (args[0] == "-h" || args[0] == "--help") {
        return Ok(None);
    }
    let mut ranks = [RankOptions::default(), RankOptions::default()];
    let mut startup_timeout = ServePair::DEFAULT_STARTUP_TIMEOUT;
    let mut shutdown_timeout = ServePair::DEFAULT_SHUTDOWN_TIMEOUT;
    let mut index = 0usize;
    while index < args.len() {
        let flag = args[index]
            .to_str()
            .context("option names must be valid UTF-8")?;
        let value = args
            .get(index + 1)
            .with_context(|| format!("{flag} wants a value"))?
            .clone();
        if flag == "--startup-timeout-secs" {
            let seconds: u64 = value
                .to_str()
                .context("--startup-timeout-secs must be valid UTF-8")?
                .parse()
                .context("--startup-timeout-secs wants an integer")?;
            startup_timeout = std::time::Duration::from_secs(seconds);
            index += 2;
            continue;
        }
        if flag == "--shutdown-timeout-secs" {
            let seconds: u64 = value
                .to_str()
                .context("--shutdown-timeout-secs must be valid UTF-8")?
                .parse()
                .context("--shutdown-timeout-secs wants an integer")?;
            shutdown_timeout = std::time::Duration::from_secs(seconds);
            index += 2;
            continue;
        }
        let (rank, field) =
            rank_flag(flag).with_context(|| format!("unknown argument {flag:?}\n\n{}", usage()))?;
        match field {
            "program" => ranks[rank].program = Some(value),
            "host" => {
                ranks[rank].host = Some(
                    value
                        .into_string()
                        .map_err(|_| anyhow::anyhow!("{flag} must be valid UTF-8"))?,
                )
            }
            "ssh" => ranks[rank].ssh = Some(value),
            "supervisor" => ranks[rank].supervisor = Some(value),
            "arg" => ranks[rank].args.push(value),
            "env" => {
                let value = value
                    .into_string()
                    .map_err(|_| anyhow::anyhow!("{flag} must be valid UTF-8"))?;
                let (key, value) = value
                    .split_once('=')
                    .with_context(|| format!("{flag} wants KEY=VALUE"))?;
                ranks[rank]
                    .env
                    .push((key.to_string().into(), value.to_string().into()));
            }
            _ => unreachable!("rank_flag returns only known fields"),
        }
        index += 2;
    }
    let [rank0, rank1] = ranks;
    anyhow::ensure!(
        shutdown_timeout >= std::time::Duration::from_secs(2),
        "--shutdown-timeout-secs must be at least 2"
    );
    let remote_shutdown_timeout =
        std::time::Duration::from_secs(shutdown_timeout.as_secs().saturating_sub(5).max(1));
    Ok(Some(Options {
        commands: [
            rank0.finish(0, remote_shutdown_timeout)?,
            rank1.finish(1, remote_shutdown_timeout)?,
        ],
        startup_timeout,
        shutdown_timeout,
    }))
}

fn rank_flag(flag: &str) -> Option<(usize, &str)> {
    let rest = flag.strip_prefix("--rank")?;
    let (rank, field) = rest.split_once('-')?;
    let rank = match rank {
        "0" => 0,
        "1" => 1,
        _ => return None,
    };
    matches!(
        field,
        "program" | "host" | "ssh" | "supervisor" | "arg" | "env"
    )
    .then_some((rank, field))
}

struct SupervisorOptions {
    program: std::ffi::OsString,
    args: Vec<std::ffi::OsString>,
    env: Vec<(std::ffi::OsString, std::ffi::OsString)>,
    shutdown_timeout: std::time::Duration,
}

fn parse_supervisor() -> Result<SupervisorOptions> {
    let args: Vec<std::ffi::OsString> = std::env::args_os().skip(2).collect();
    let mut program = None;
    let mut rank_args = Vec::new();
    let mut env = Vec::new();
    let mut shutdown_timeout = ServePair::DEFAULT_SHUTDOWN_TIMEOUT;
    let mut index = 0usize;
    while index < args.len() {
        let flag = args[index]
            .to_str()
            .context("supervisor option names must be valid UTF-8")?;
        let value = args
            .get(index + 1)
            .with_context(|| format!("{flag} wants a value"))?
            .clone();
        match flag {
            "--program" => program = Some(value),
            "--arg" => rank_args.push(value),
            "--env" => {
                let value = value
                    .into_string()
                    .map_err(|_| anyhow::anyhow!("--env must be valid UTF-8"))?;
                let (key, value) = value.split_once('=').context("--env wants KEY=VALUE")?;
                anyhow::ensure!(valid_env_name(key), "invalid environment name {key:?}");
                env.push((key.into(), value.into()));
            }
            "--shutdown-timeout-secs" => {
                let seconds: u64 = value
                    .to_str()
                    .context("--shutdown-timeout-secs must be valid UTF-8")?
                    .parse()
                    .context("--shutdown-timeout-secs wants an integer")?;
                shutdown_timeout = std::time::Duration::from_secs(seconds);
            }
            _ => anyhow::bail!("unknown supervisor argument {flag:?}"),
        }
        index += 2;
    }
    anyhow::ensure!(
        !shutdown_timeout.is_zero(),
        "the supervisor shutdown timeout must be nonzero"
    );
    Ok(SupervisorOptions {
        program: program.context("--program is required")?,
        args: rank_args,
        env,
        shutdown_timeout,
    })
}

fn valid_env_name(name: &str) -> bool {
    let mut chars = name.chars();
    matches!(chars.next(), Some('_' | 'a'..='z' | 'A'..='Z'))
        && chars.all(|c| matches!(c, '_' | 'a'..='z' | 'A'..='Z' | '0'..='9'))
}

enum SupervisorEvent {
    InputClosed(Option<String>),
    OutputClosed(Option<String>),
}

/// Owns a live rank process group. Every early return after `spawn` therefore
/// tears the group down; once `try_wait` reaps the leader, the guard disarms so
/// it never signals a stale numeric process-group id.
struct SupervisedChild {
    child: std::process::Child,
    process_group: u32,
    reaped: bool,
}

impl SupervisedChild {
    fn new(child: std::process::Child) -> Self {
        let process_group = child.id();
        Self {
            child,
            process_group,
            reaped: false,
        }
    }

    fn kill(&mut self) {
        if self.reaped {
            return;
        }
        kill_process_group(self.process_group);
        let _ = self.child.kill();
    }

    fn try_wait(&mut self) -> std::io::Result<Option<std::process::ExitStatus>> {
        let status = self.child.try_wait()?;
        if status.is_some() {
            self.reaped = true;
        }
        Ok(status)
    }

    fn wait(&mut self) -> std::io::Result<std::process::ExitStatus> {
        let status = self.child.wait()?;
        self.reaped = true;
        Ok(status)
    }
}

impl Drop for SupervisedChild {
    fn drop(&mut self) {
        if !self.reaped {
            self.kill();
            let _ = self.wait();
        }
    }
}

/// Relay one rank while making the SSH channel the lifetime authority.
///
/// EOF is allowed a short clean-shutdown window because a complete framed END
/// also closes stdin. If the rank remains alive after that window, its process
/// group is killed. On Linux `PDEATHSIG` additionally covers the supervisor
/// itself being killed before either relay thread can observe the channel. A
/// rank is one process and must not daemonize; once that leader is reaped its
/// numeric process-group id is deliberately never signalled again.
fn supervise_rank<R, W>(
    options: SupervisorOptions,
    mut input: R,
    mut output: W,
    channel_output_fd: Option<libc::c_int>,
) -> Result<()>
where
    R: Read + Send + 'static,
    W: Write + Send + 'static,
{
    let mut command = std::process::Command::new(&options.program);
    command
        .args(&options.args)
        .envs(options.env)
        .stdin(std::process::Stdio::piped())
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::inherit());
    configure_supervised_child(&mut command);
    let child = command.spawn().with_context(|| {
        format!(
            "launch supervised rank {}",
            options.program.to_string_lossy()
        )
    })?;
    let mut child = SupervisedChild::new(child);
    let mut child_input = child
        .child
        .stdin
        .take()
        .context("rank stdin was not piped")?;
    let mut child_output = child
        .child
        .stdout
        .take()
        .context("rank stdout was not piped")?;
    let (send, receive) = std::sync::mpsc::channel();
    let input_send = send.clone();
    std::thread::spawn(move || {
        let error = std::io::copy(&mut input, &mut child_input)
            .and_then(|_| child_input.flush())
            .err()
            .map(|error| error.to_string());
        drop(child_input);
        let _ = input_send.send(SupervisorEvent::InputClosed(error));
    });
    let output_send = send.clone();
    std::thread::spawn(move || {
        let error = std::io::copy(&mut child_output, &mut output)
            .and_then(|_| output.flush())
            .err()
            .map(|error| error.to_string());
        let _ = output_send.send(SupervisorEvent::OutputClosed(error));
    });
    drop(send);

    let mut status = None;
    let mut input_closed = false;
    let mut output_closed = false;
    let mut terminal_error: Option<String> = None;
    let mut deadline = None;
    loop {
        if status.is_none() {
            status = child.try_wait().context("poll supervised rank")?;
            if status.is_some() && !output_closed {
                deadline
                    .get_or_insert_with(|| std::time::Instant::now() + options.shutdown_timeout);
            }
        }
        if let Some(status) = status {
            if output_closed {
                if let Some(error) = terminal_error.take() {
                    anyhow::bail!(error);
                }
                anyhow::ensure!(status.success(), "supervised rank exited with {status}");
                return Ok(());
            }
        }
        if deadline.is_some_and(|deadline| std::time::Instant::now() >= deadline) {
            if status.is_none() {
                child.kill();
                let _ = child.wait();
            }
            anyhow::bail!(
                "supervised rank did not finish after its channel closed within {:?}",
                options.shutdown_timeout
            );
        }

        if channel_output_fd.is_some_and(output_channel_broken) {
            terminal_error
                .get_or_insert_with(|| "the supervising SSH output channel closed".to_string());
            child.kill();
            deadline.get_or_insert_with(|| std::time::Instant::now() + options.shutdown_timeout);
        }

        match receive.recv_timeout(std::time::Duration::from_millis(10)) {
            Ok(SupervisorEvent::InputClosed(error)) => {
                input_closed = true;
                if let Some(error) = error {
                    terminal_error.get_or_insert_with(|| {
                        format!("read the supervising SSH channel: {error}")
                    });
                    child.kill();
                }
                deadline
                    .get_or_insert_with(|| std::time::Instant::now() + options.shutdown_timeout);
            }
            Ok(SupervisorEvent::OutputClosed(error)) => {
                output_closed = true;
                deadline
                    .get_or_insert_with(|| std::time::Instant::now() + options.shutdown_timeout);
                if let Some(error) = error {
                    terminal_error.get_or_insert_with(|| {
                        format!("write the supervising SSH channel: {error}")
                    });
                    child.kill();
                }
            }
            Err(std::sync::mpsc::RecvTimeoutError::Timeout) => {}
            Err(std::sync::mpsc::RecvTimeoutError::Disconnected) => {
                if !(input_closed && output_closed) {
                    terminal_error
                        .get_or_insert_with(|| "the rank relay workers disappeared".to_string());
                    child.kill();
                    deadline.get_or_insert_with(|| {
                        std::time::Instant::now() + options.shutdown_timeout
                    });
                } else {
                    // Both terminal relay events were observed. A disconnected
                    // channel now means only that the rank is still releasing
                    // resources; keep the child poll bounded without spinning.
                    std::thread::sleep(std::time::Duration::from_millis(10));
                }
            }
        }
    }
}

fn configure_supervised_child(command: &mut std::process::Command) {
    #[cfg(unix)]
    {
        use std::os::unix::process::CommandExt as _;

        // SAFETY: only async-signal-safe libc calls run between fork and exec.
        #[cfg(target_os = "linux")]
        let expected_parent = std::process::id() as libc::pid_t;
        unsafe {
            command.pre_exec(move || {
                if libc::setpgid(0, 0) != 0 {
                    return Err(std::io::Error::last_os_error());
                }
                #[cfg(target_os = "linux")]
                {
                    if libc::prctl(libc::PR_SET_PDEATHSIG, libc::SIGKILL) != 0 {
                        return Err(std::io::Error::last_os_error());
                    }
                    if libc::getppid() != expected_parent {
                        return Err(std::io::Error::from_raw_os_error(libc::ECHILD));
                    }
                }
                Ok(())
            });
        }
    }
}

fn kill_process_group(process_group: u32) {
    #[cfg(unix)]
    unsafe {
        let _ = libc::kill(-(process_group as libc::pid_t), libc::SIGKILL);
    }
}

fn output_channel_broken(fd: libc::c_int) -> bool {
    #[cfg(unix)]
    {
        let mut descriptor = libc::pollfd {
            fd,
            events: 0,
            revents: 0,
        };
        // SAFETY: `descriptor` is valid for the one-element poll array.
        let polled = unsafe { libc::poll(&mut descriptor, 1, 0) };
        polled > 0 && descriptor.revents & (libc::POLLHUP | libc::POLLERR | libc::POLLNVAL) != 0
    }
    #[cfg(not(unix))]
    {
        let _ = fd;
        false
    }
}

fn run_supervisor() -> Result<()> {
    let options = parse_supervisor()?;
    #[cfg(unix)]
    {
        use std::os::fd::FromRawFd as _;

        // Raw files make the relay byte-transparent and unbuffered. In
        // particular, token records are not held until a newline or EOF.
        let input_fd = unsafe { libc::dup(libc::STDIN_FILENO) };
        anyhow::ensure!(input_fd >= 0, "could not duplicate supervisor stdin");
        let output_fd = unsafe { libc::dup(libc::STDOUT_FILENO) };
        if output_fd < 0 {
            unsafe { libc::close(input_fd) };
            anyhow::bail!("could not duplicate supervisor stdout");
        }
        let input = unsafe { std::fs::File::from_raw_fd(input_fd) };
        let output = unsafe { std::fs::File::from_raw_fd(output_fd) };
        return supervise_rank(options, input, output, Some(libc::STDOUT_FILENO));
    }
    #[cfg(not(unix))]
    supervise_rank(options, std::io::stdin(), std::io::stdout(), None)
}

/// Take fd 1 for the protocol and point ordinary Rust stdout at stderr.
fn claim_stdout() -> Result<std::fs::File> {
    use std::os::fd::FromRawFd as _;
    let _ = std::io::stdout().flush();
    let raw = unsafe { libc::dup(libc::STDOUT_FILENO) };
    anyhow::ensure!(raw >= 0, "could not dup stdout for the protocol stream");
    let redirected = unsafe { libc::dup2(libc::STDERR_FILENO, libc::STDOUT_FILENO) };
    anyhow::ensure!(redirected >= 0, "could not point stdout at stderr");
    Ok(unsafe { std::fs::File::from_raw_fd(raw) })
}

fn main() {
    let supervise = std::env::args_os()
        .nth(1)
        .is_some_and(|argument| argument == "__supervise");
    let result = if supervise { run_supervisor() } else { run() };
    if let Err(error) = result {
        eprintln!("inkling_serve_pair: {error:#}");
        std::process::exit(1);
    }
}

fn run() -> Result<()> {
    let Some(options) = parse()? else {
        print!("{}", usage());
        return Ok(());
    };
    let protocol = claim_stdout()?;
    // Establish the downstream streams before the potentially long model load,
    // exactly as one `inkling_serve` does.
    let mut output = framed_stream::FramedWriter::open(protocol, CONTENT_TYPE, UNIT)
        .context("open the downstream output stream")?;
    let mut input = framed_stream::FramedReader::open(std::io::stdin().lock())
        .context("open the downstream input stream")?;
    input.require_content_type(CONTENT_TYPE)?;

    let mut pair = ServePair::spawn_with_timeout(options.commands, options.startup_timeout)
        .context("start the tensor-parallel pair")?;
    let ready_payload = serde_json::to_vec(pair.ready())?;
    output.record_as(READY_TYPE, &ready_payload, ready_payload.len() as u64)?;

    loop {
        match input.next_frame()? {
            framed_stream::Frame::Record(record) if record.content_type() == CONTENT_TYPE => {
                pair.feed(record.text()?)?;
            }
            framed_stream::Frame::Record(record) if record.content_type() == CONTEXT_TYPE => {
                let context: InklingContext = serde_json::from_slice(&record.payload)
                    .context("parse the downstream typed context record")?;
                pair.context(&context)?;
            }
            framed_stream::Frame::Record(record)
                if record.content_type() == CONTEXT_PREFLIGHT_TYPE =>
            {
                let request: ContextPreflight = serde_json::from_slice(&record.payload)
                    .context("parse downstream context preflight")?;
                let evidence = pair.preflight_context(&request)?;
                let payload = serde_json::to_vec(&evidence)
                    .context("encode downstream context-preflight evidence")?;
                output.record_as(CONTEXT_PREFLIGHTED_TYPE, &payload, payload.len() as u64)?;
            }
            framed_stream::Frame::Record(record) if record.content_type() == CONSULT_TYPE => {
                let consult: Consult = serde_json::from_slice(&record.payload)
                    .context("parse the downstream CONSULT record")?;
                // A Gap is forward-only skipped extent; it cannot retract equal
                // fragments already emitted. Preserve those as partial speech
                // and let dropping `output` write ABORTED on pair failure.
                let end = pair.consult(&consult, |text| output.text(text))?;
                // The pair already proved both ranks produced these exact ids.
                // Keep that evidence on the downstream record so a caller can
                // compare continuity across independent serving sessions.
                let payload = downstream_turn_payload(end)?;
                output.record_as(TURN_TYPE, &payload, payload.len() as u64)?;
            }
            framed_stream::Frame::Record(record) if record.content_type() == REINITIALIZE_TYPE => {
                let initialization: InklingContext = serde_json::from_slice(&record.payload)
                    .context("parse downstream REINITIALIZE record")?;
                let acknowledgement = pair.reinitialize(&initialization)?;
                let payload = serde_json::to_vec(&acknowledgement)
                    .context("encode downstream REINITIALIZED record")?;
                output.record_as(REINITIALIZED_TYPE, &payload, payload.len() as u64)?;
            }
            framed_stream::Frame::Record(record) => anyhow::bail!(
                "this serving proxy does not understand a {} record",
                record.content_type()
            ),
            framed_stream::Frame::Gap(gap) => {
                let marker = format!("\n[{} bytes lost: {}]\n", gap.extent, gap.reason);
                pair.feed(&marker)?;
            }
            framed_stream::Frame::End(status) => {
                eprintln!("inkling_serve_pair: downstream ended ({status:?})");
                break;
            }
        }
    }

    let statuses = pair.close_with_timeout(options.shutdown_timeout)?;
    anyhow::ensure!(
        statuses.iter().all(std::process::ExitStatus::success),
        "rank shutdown failed: rank 0 {}, rank 1 {}",
        statuses[0],
        statuses[1]
    );
    output.finish(framed_stream::EndStatus::Complete)?;
    Ok(())
}

fn downstream_turn_payload(end: mary::models::inkling::serve::TurnEnd) -> Result<Vec<u8>> {
    serde_json::to_vec(&end).context("encode the downstream TURN record")
}

#[cfg(test)]
mod tests {
    use super::*;
    use mary::models::inkling::serve::TurnEnd;

    #[derive(Clone, Default)]
    struct SharedOutput(std::sync::Arc<std::sync::Mutex<Vec<u8>>>);

    impl Write for SharedOutput {
        fn write(&mut self, bytes: &[u8]) -> std::io::Result<usize> {
            self.0.lock().expect("output lock").extend_from_slice(bytes);
            Ok(bytes.len())
        }

        fn flush(&mut self) -> std::io::Result<()> {
            Ok(())
        }
    }

    struct BrokenOutput;

    impl Write for BrokenOutput {
        fn write(&mut self, _: &[u8]) -> std::io::Result<usize> {
            Err(std::io::Error::new(
                std::io::ErrorKind::BrokenPipe,
                "fixture downstream vanished",
            ))
        }

        fn flush(&mut self) -> std::io::Result<()> {
            Ok(())
        }
    }

    fn supervisor_options(script: &str, timeout: std::time::Duration) -> SupervisorOptions {
        SupervisorOptions {
            program: "sh".into(),
            args: vec!["-c".into(), script.into()],
            env: Vec::new(),
            shutdown_timeout: timeout,
        }
    }

    #[test]
    fn downstream_turn_preserves_arbitrated_token_ids() {
        let end = TurnEnd {
            turn: 0,
            tokens: 2,
            token_ids: vec![41, 42],
            delta_tokens: 3,
            carried: 0,
            stopped: "max_tokens".to_string(),
            first_token_secs: 0.1,
            turn_secs: 0.2,
            position: 5,
        };
        let payload = downstream_turn_payload(end).expect("payload");
        let json: serde_json::Value = serde_json::from_slice(&payload).expect("json");
        assert_eq!(json["token_ids"], serde_json::json!([41, 42]));
        assert_eq!(json["tokens"], 2);
    }

    #[test]
    fn supervisor_relays_arbitrary_bytes_and_allows_clean_eof() {
        let expected = b"\0READY\n\xfftoken\0".to_vec();
        let output = SharedOutput::default();
        supervise_rank(
            supervisor_options("exec cat", std::time::Duration::from_secs(1)),
            std::io::Cursor::new(expected.clone()),
            output.clone(),
            None,
        )
        .expect("cat exits cleanly on channel EOF");
        assert_eq!(*output.0.lock().expect("output lock"), expected);
    }

    #[test]
    fn supervisor_exposes_output_before_a_delayed_clean_exit() {
        use std::sync::atomic::{AtomicBool, Ordering};

        let output = SharedOutput::default();
        let thread_output = output.clone();
        let done = std::sync::Arc::new(AtomicBool::new(false));
        let thread_done = done.clone();
        let worker = std::thread::spawn(move || {
            let result = supervise_rank(
                supervisor_options("printf READY; sleep 0.3", std::time::Duration::from_secs(1)),
                std::io::Cursor::new(Vec::<u8>::new()),
                thread_output,
                None,
            );
            thread_done.store(true, Ordering::Release);
            result
        });
        let deadline = std::time::Instant::now() + std::time::Duration::from_millis(200);
        while output.0.lock().expect("output lock").is_empty()
            && std::time::Instant::now() < deadline
        {
            std::thread::sleep(std::time::Duration::from_millis(1));
        }
        assert_eq!(&*output.0.lock().expect("output lock"), b"READY");
        assert!(!done.load(Ordering::Acquire), "rank must still be sleeping");
        worker
            .join()
            .expect("supervisor worker")
            .expect("clean exit");
    }

    #[test]
    fn supervisor_kills_a_rank_stuck_after_channel_eof() {
        let started = std::time::Instant::now();
        let error = supervise_rank(
            supervisor_options("exec sleep 30", std::time::Duration::from_millis(50)),
            std::io::Cursor::new(Vec::<u8>::new()),
            std::io::sink(),
            None,
        )
        .expect_err("hung rank must hit the supervisor deadline");
        assert!(error.to_string().contains("did not finish"));
        assert!(started.elapsed() < std::time::Duration::from_secs(3));
    }

    #[test]
    fn broken_supervisor_output_kills_a_rank_promptly() {
        let started = std::time::Instant::now();
        let error = supervise_rank(
            supervisor_options(
                "printf x; exec sleep 30",
                std::time::Duration::from_secs(10),
            ),
            std::io::Cursor::new(Vec::<u8>::new()),
            BrokenOutput,
            None,
        )
        .expect_err("broken output must fail the supervisor");
        assert!(
            error
                .to_string()
                .contains("write the supervising SSH channel")
        );
        assert!(started.elapsed() < std::time::Duration::from_secs(3));
    }
}
