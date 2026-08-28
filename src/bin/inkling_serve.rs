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
//! four control content-types on it — not a new format.
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
//! Drive owns no tokenizer. Raw probe text and typed context JSON cross the
//! wire; this process alone turns them into ids, and TURN reports generated
//! exact ids so the client can parse structure. The tokenizer is the
//! checkpoint's own `tokenizer.json`, read by
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
    CONSULT_TYPE, CONTENT_TYPE, CONTEXT_PREFLIGHT_TYPE, CONTEXT_PREFLIGHTED_TYPE, CONTEXT_TYPE,
    Consult, ContextPlacement, ContextPreflight, ExecutionManifest, InklingContext,
    InklingContextCodec, READY_TYPE, REINITIALIZE_TYPE, REINITIALIZED_TYPE, Ready, Reinitialized,
    TURN_TYPE, TurnEnd, UNIT, context_preflight,
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
    --sealed             Reject execution-changing environment overrides and
                         announce a sealed-v1 execution manifest
    --tp-rank <rank>     This process's tensor-parallel rank (all TP flags together)
    --tp-world <world>   Number of tensor-parallel ranks
    --tp-rendezvous <a>  Rank 0's HOST:PORT on the fast fabric
    -h, --help           This text

The protocol is the framed-stream convention with four control content-types;
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
    sealed: bool,
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
    let mut sealed = false;
    let mut i = 0;
    while i < args.len() {
        let need = |i: usize| -> Result<&String> {
            args.get(i + 1)
                .with_context(|| format!("{} wants a value", args[i]))
        };
        match args[i].as_str() {
            "-h" | "--help" => return Ok(None),
            "--sealed" => {
                sealed = true;
                i += 1;
            }
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
        sealed,
    }))
}

/// Environment is inherited ambient authority. Sealed mode refuses namespaces
/// this runtime uses to alter kernels, numerics, scheduling, or allocation
/// before a CUDA client exists. The exact exceptions below are rank-local
/// placement, transport routing, and diagnostics: they must be able to differ
/// across hosts and do not choose model numerics or kernels. In particular,
/// `CUDA_VISIBLE_DEVICES` maps CUDA's logical device 0, whose effective class
/// and compute capability are witnessed in the shared manifest; the NCCL
/// exceptions only route the two-rank transport or report what it selected.
/// Every other CUDA/NCCL variable remains rejected, including algorithm knobs.
///
/// Explicit serving CLI settings remain allowed because they enter the
/// manifest. Library discovery through `LD_LIBRARY_PATH` is the other narrow
/// phase-1 exception: the exact selected library bytes are hashed after load,
/// while rejecting it would make the CUDA deployment layout unusable.
fn sealed_environment_rejections(names: impl IntoIterator<Item = String>) -> Vec<String> {
    const PREFIXES: &[&str] = &[
        "INK_",
        "CUBECL_",
        "CUDA_",
        "NCCL_",
        "NVRTC_",
        "CUBLAS_",
        "CUDNN_",
        "BURN_",
        "OMP_",
        "MKL_",
        "OPENBLAS_",
        "RAYON_",
        "MALLOC_",
    ];
    const EXACT: &[&str] = &["GLIBC_TUNABLES", "LD_AUDIT", "LD_PRELOAD"];
    const RANK_LOCAL_EXACT: &[&str] = &[
        "CUDA_VISIBLE_DEVICES",
        "NCCL_IB_DISABLE",
        "NCCL_SOCKET_IFNAME",
        "NCCL_IB_HCA",
    ];
    let mut rejected = names
        .into_iter()
        .filter(|name| {
            let rank_local = RANK_LOCAL_EXACT.contains(&name.as_str())
                || name == "NCCL_DEBUG"
                || name.starts_with("NCCL_DEBUG_");
            !rank_local
                && (EXACT.contains(&name.as_str())
                    || PREFIXES.iter().any(|prefix| name.starts_with(prefix)))
        })
        .collect::<Vec<_>>();
    rejected.sort();
    rejected
}

fn reject_sealed_environment() -> Result<()> {
    let rejected = sealed_environment_rejections(
        std::env::vars_os().map(|(name, _)| name.to_string_lossy().into_owned()),
    );
    anyhow::ensure!(
        rejected.is_empty(),
        "sealed-v1 refuses execution-changing environment overrides: {}. Express serving shape \
         through explicit CLI arguments; library selection is witnessed by exact mapped-library \
         hashes",
        rejected.join(", ")
    );
    Ok(())
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

struct RuntimeFacts {
    gpu_class: String,
    compute_capability: String,
    cuda_version: String,
    cuda_library_hashes: String,
    nvrtc_version: String,
    nvrtc_library_hashes: String,
    nccl_version: String,
    nccl_library_hashes: String,
    unavailable: Vec<String>,
}

impl RuntimeFacts {
    /// Observe only facts supplied by the loaded process and CUDA APIs. Phase 1
    /// does not invoke nvidia-smi, nvcc, the package manager, or filesystem
    /// heuristics that could describe a toolkit different from the one executing
    /// kernels. `/proc/self/maps` is the exact phase-1 boundary for library
    /// identity: it hashes mapped backing-file bytes after confirming the path
    /// still names the mapped device/inode, not relocated in-memory pages. When
    /// that backing file cannot be named/read, the manifest records unavailable.
    fn observe() -> Self {
        let mut unavailable = Vec::new();
        let (gpu_class, compute_capability) = observe_gpu(&mut unavailable);
        let cuda_library_hashes = observe_library("cuda.library", "libcuda.so", &mut unavailable);
        let nvrtc_library_hashes =
            observe_library("nvrtc.library", "libnvrtc.so", &mut unavailable);
        let nccl_library_hashes = observe_library("nccl.library", "libnccl.so", &mut unavailable);

        let cuda_version = observe_version("cuda.version", &mut unavailable, || {
            use cudarc::driver::result::DriverError;

            let mut version = 0;
            let result: std::result::Result<(), DriverError> =
                unsafe { cudarc::driver::sys::cuDriverGetVersion(&mut version).result() };
            result.map_err(|error| format!("{error}"))?;
            Ok(version.to_string())
        });
        let nvrtc_version = match nvrtc_library_hashes.as_str() {
            "unavailable" => {
                unavailable.push("nvrtc.version".to_string());
                "unavailable".to_string()
            }
            _ => observe_version("nvrtc.version", &mut unavailable, || {
                let (mut major, mut minor) = (0, 0);
                unsafe { cudarc::nvrtc::sys::nvrtcVersion(&mut major, &mut minor).result() }
                    .map_err(|error| format!("{error}"))?;
                Ok(format!("{major}.{minor}"))
            }),
        };
        let nccl_version = match nccl_library_hashes.as_str() {
            "unavailable" => {
                if !unavailable.iter().any(|fact| fact == "nccl.version") {
                    unavailable.push("nccl.version".to_string());
                }
                "unavailable".to_string()
            }
            _ => observe_version("nccl.version", &mut unavailable, || {
                cudarc::nccl::result::get_nccl_version()
                    .map(|version| version.to_string())
                    .map_err(|error| format!("{error:?}"))
            }),
        };
        unavailable.sort();
        unavailable.dedup();
        Self {
            gpu_class,
            compute_capability,
            cuda_version,
            cuda_library_hashes,
            nvrtc_version,
            nvrtc_library_hashes,
            nccl_version,
            nccl_library_hashes,
            unavailable,
        }
    }

    fn add_to(&self, manifest: &mut ExecutionManifest) {
        for (name, value) in [
            ("gpu-class", self.gpu_class.as_str()),
            ("gpu-compute-capability", self.compute_capability.as_str()),
            ("cuda-driver-version", self.cuda_version.as_str()),
            ("cuda-library-blake3", self.cuda_library_hashes.as_str()),
            ("nvrtc-version", self.nvrtc_version.as_str()),
            ("nvrtc-library-blake3", self.nvrtc_library_hashes.as_str()),
            ("nccl-version", self.nccl_version.as_str()),
            ("nccl-library-blake3", self.nccl_library_hashes.as_str()),
        ] {
            manifest.field(name, value.as_bytes());
        }
    }
}

fn observe_gpu(unavailable: &mut Vec<String>) -> (String, String) {
    let result = std::panic::catch_unwind(|| -> Result<(String, String)> {
        cudarc::driver::result::init().context("initialize CUDA driver observation")?;
        let device =
            cudarc::driver::result::device::get(0).context("open CUDA device ordinal 0")?;
        let name =
            cudarc::driver::result::device::get_name(device).context("read CUDA device class")?;
        let major = unsafe {
            cudarc::driver::result::device::get_attribute(
                device,
                cudarc::driver::sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR,
            )
        }
        .context("read CUDA compute-capability major")?;
        let minor = unsafe {
            cudarc::driver::result::device::get_attribute(
                device,
                cudarc::driver::sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR,
            )
        }
        .context("read CUDA compute-capability minor")?;
        Ok((name, format!("{major}.{minor}")))
    });
    match result {
        Ok(Ok(facts)) => facts,
        Ok(Err(error)) => {
            eprintln!("inkling_serve: GPU manifest facts unavailable: {error:#}");
            unavailable.extend([
                "gpu.class".to_string(),
                "gpu.compute-capability".to_string(),
            ]);
            ("unavailable".to_string(), "unavailable".to_string())
        }
        Err(_) => {
            unavailable.extend([
                "gpu.class".to_string(),
                "gpu.compute-capability".to_string(),
            ]);
            ("unavailable".to_string(), "unavailable".to_string())
        }
    }
}

fn observe_version(
    name: &str,
    unavailable: &mut Vec<String>,
    observe: impl FnOnce() -> std::result::Result<String, String>,
) -> String {
    match std::panic::catch_unwind(std::panic::AssertUnwindSafe(observe)) {
        Ok(Ok(version)) => version,
        Ok(Err(error)) => {
            eprintln!("inkling_serve: {name} unavailable: {error}");
            unavailable.push(name.to_string());
            "unavailable".to_string()
        }
        Err(_) => {
            unavailable.push(name.to_string());
            "unavailable".to_string()
        }
    }
}

fn observe_library(name: &str, prefix: &str, unavailable: &mut Vec<String>) -> String {
    match mapped_library_hashes(prefix) {
        Ok(hashes) if !hashes.is_empty() => hashes.join(","),
        Ok(_) => {
            unavailable.push(name.to_string());
            "unavailable".to_string()
        }
        Err(error) => {
            eprintln!("inkling_serve: {name} unavailable: {error:#}");
            unavailable.push(name.to_string());
            "unavailable".to_string()
        }
    }
}

fn mapped_library_hashes(prefix: &str) -> Result<Vec<String>> {
    let maps = std::fs::read_to_string("/proc/self/maps").context("read /proc/self/maps")?;
    let mut paths = std::collections::BTreeMap::new();
    for line in maps.lines() {
        let mut columns = line.split_whitespace();
        let _range = columns.next();
        let _permissions = columns.next();
        let _offset = columns.next();
        let device = columns.next();
        let inode = columns.next().and_then(|value| value.parse::<u64>().ok());
        let Some(path_start) = line.find('/') else {
            continue;
        };
        let path = line[path_start..].trim_end_matches(" (deleted)");
        let matches = std::path::Path::new(path)
            .file_name()
            .and_then(std::ffi::OsStr::to_str)
            .is_some_and(|file| file.starts_with(prefix));
        if matches {
            let (Some(device), Some(inode)) = (device, inode) else {
                anyhow::bail!("mapped library row has no device/inode witness: {line}");
            };
            let path = std::path::PathBuf::from(path);
            if let Some(previous) = paths.insert(path.clone(), (device.to_string(), inode)) {
                anyhow::ensure!(
                    previous == (device.to_string(), inode),
                    "mapped library {} changed identity within /proc/self/maps",
                    path.display()
                );
            }
        }
    }
    let mut hashes = Vec::with_capacity(paths.len());
    for (path, (_mapped_device, _mapped_inode)) in paths {
        let mut file = std::fs::File::open(&path)
            .with_context(|| format!("open mapped library {}", path.display()))?;
        // Hashing the spelling from /proc/self/maps without this check could
        // hash a replacement installed after dlopen rather than the inode the
        // process mapped.
        #[cfg(target_os = "linux")]
        {
            use std::os::unix::fs::MetadataExt as _;

            let metadata = file.metadata()?;
            let actual_device = format!(
                "{:02x}:{:02x}",
                libc::major(metadata.dev()),
                libc::minor(metadata.dev())
            );
            anyhow::ensure!(
                metadata.ino() == _mapped_inode && actual_device == _mapped_device,
                "mapped library {} is device/inode {_mapped_device}/{_mapped_inode}, but its path \
                 now names {actual_device}/{}",
                path.display(),
                metadata.ino()
            );
        }
        let mut hasher = blake3::Hasher::new();
        let mut buffer = [0u8; 64 * 1024];
        loop {
            let count = std::io::Read::read(&mut file, &mut buffer)
                .with_context(|| format!("hash mapped library {}", path.display()))?;
            if count == 0 {
                break;
            }
            hasher.update(&buffer[..count]);
        }
        hashes.push(hex_identity(*hasher.finalize().as_bytes()));
    }
    hashes.sort();
    hashes.dedup();
    Ok(hashes)
}

fn begin_execution_manifest(profile: &str) -> Result<ExecutionManifest> {
    // Linux exposes the inode this process is actually executing, even if the
    // launch path is replaced later. current_exe() cannot make that claim.
    let file = std::fs::File::open("/proc/self/exe")
        .context("open /proc/self/exe for the execution manifest")?;
    let length = file.metadata()?.len();
    let mut manifest = ExecutionManifest::new(profile);
    manifest.reader("executable-bytes", length, file)?;
    Ok(manifest)
}

fn main() {
    if let Err(error) = run() {
        eprintln!("inkling_serve: {error:#}");
        std::process::exit(1);
    }
}

/// A reinitialization replaces a sequence; it must not become a disguised way
/// to discard context already queued for the next turn. The input loop itself
/// is synchronous, so reaching this check proves no CONSULT is in flight.
fn validate_reinitialize_boundary(
    completed_turns: usize,
    pending_delta_tokens: usize,
    has_carry: bool,
) -> Result<()> {
    anyhow::ensure!(
        completed_turns > 0 && has_carry,
        "reinitialization requires a completed turn"
    );
    anyhow::ensure!(
        pending_delta_tokens == 0,
        "reinitialization cannot discard {pending_delta_tokens} pending context token(s)"
    );
    Ok(())
}

fn run() -> Result<()> {
    let Some(options) = parse()? else {
        print!("{}", usage());
        return Ok(());
    };
    if options.sealed {
        reject_sealed_environment()?;
    }
    let execution_profile = match options.sealed {
        true => "sealed-v1",
        false => "observed-v1",
    };
    let mut execution_manifest = begin_execution_manifest(execution_profile)?;
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
    let context_codec = InklingContextCodec::from_json(&tokenizer_bytes)
        .with_context(|| format!("build context codec from {}", options.tokenizer.display()))?;
    let tokenizer = tokenizers::Tokenizer::from_bytes(&tokenizer_bytes)
        .map_err(|e| anyhow::anyhow!("load {}: {e}", options.tokenizer.display()))?;

    let mut config = SessionConfig::new(&options.pile);
    if let Some(layers) = options.layers.clone() {
        config = config.layers(layers);
    }
    if let Some(budget) = options.prefill_budget {
        config.prefill_budget = budget;
        // There is one bounded-width append path, not an independently tuned
        // second strategy. A caller narrowing prefill chunks narrows later
        // multi-row extends to the same admitted width; single-token decode is
        // unchanged. SessionConfig still rejects an explicitly inconsistent
        // library caller instead of silently repairing it.
        config.extend_batch = config.extend_batch.min(budget);
    }
    // Historically there was only one length axis. Preserve that CLI meaning:
    // setting just --prefill-budget admits that many retained positions too,
    // while an explicit --context-budget opts into bounded-width chunking of a
    // longer logical sequence.
    config.context_budget = options.context_budget.unwrap_or(config.prefill_budget);
    let prefill_budget = config.prefill_budget;
    let context_budget = config.context_budget;
    let extend_batch = config.extend_batch;
    // Select once before any CUDA client. Group/Session observe this same
    // process-global value later; sealed-v1 has already refused an ambient
    // CUBECL_MEMORY_CONFIG override, so the effective baseline is fixed.
    let allocator = mary::models::inkling::pool::choose_memory_config();
    let (tp_rank, tp_world) = options
        .tensor_parallel
        .as_ref()
        .map(|parallel| (Some(parallel.tp.rank()), parallel.tp.world()))
        .unwrap_or((None, 1));
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
    let runtime_facts = RuntimeFacts::observe();

    // Decoder state belongs to the whole logical token sequence, not to one
    // generated turn. Byte-fallback and spacing decoders both need surrounding
    // ids. New world-context ids advance this stream without being spoken;
    // generated ids advance the same stream and their chunks are emitted. A
    // carried token is never advanced twice: it entered this sequence when it
    // was generated, while `carry` only catches the KV cache up to that fact.
    let mut decode_stream = tokenizer.decode_stream(false);

    let range = session.layer_range();
    let model_identity = hex_identity(session.model_identity());
    let tokenizer_identity = hex_identity(tokenizer_identity);
    for (name, value) in [
        ("model-identity", model_identity.as_str()),
        ("tokenizer-identity", tokenizer_identity.as_str()),
        ("tp-role-schema", "rank-normalized-v1"),
        ("allocator", allocator.env_value()),
        // INK_POOL_CLEANUP is rejected in sealed-v1, so this is the effective
        // default selected by CleanupPolicy::choose.
        ("allocator-cleanup", "when-stranded"),
        (
            "burn-timing-autotune",
            if cfg!(feature = "inkling-cuda-autotune") {
                "enabled"
            } else {
                "disabled"
            },
        ),
    ] {
        execution_manifest.field(name, value.as_bytes());
    }
    execution_manifest.usize("tp-world", tp_world);
    execution_manifest.usize("layer-lo", range.start);
    execution_manifest.usize("layer-hi", range.end);
    execution_manifest.usize(
        "stack-layers",
        session.config().text_config.num_hidden_layers,
    );
    execution_manifest.usize(
        "effective-vocab",
        session.config().text_config.effective_vocab(),
    );
    execution_manifest.usize("prefill-budget", prefill_budget);
    execution_manifest.usize("context-budget", context_budget);
    execution_manifest.usize("extend-batch", extend_batch);
    runtime_facts.add_to(&mut execution_manifest);
    let execution_identity = execution_manifest.finish_hex();
    let ready = Ready {
        pile: options.pile.display().to_string(),
        model_identity,
        tokenizer_identity,
        special_ids: context_codec.special_ids().clone(),
        execution_profile: execution_profile.to_string(),
        execution_identity,
        execution_unavailable: runtime_facts.unavailable,
        tp_rank,
        tp_world,
        layers: [range.start, range.end],
        stack: session.config().text_config.num_hidden_layers,
        partial: session.is_partial_stack(),
        vocab: session.config().text_config.effective_vocab(),
        prefill_budget,
        context_budget,
        load_secs,
    };
    eprintln!(
        "inkling_serve: ready in {load_secs:.1}s — layers {}..{} of {}{}, {} {}",
        ready.layers[0],
        ready.layers[1],
        ready.stack,
        match ready.partial {
            true => " (PARTIAL STACK: diagnostic tokens, not the model's)",
            false => "",
        },
        ready.execution_profile,
        ready.execution_identity,
    );
    if !ready.execution_unavailable.is_empty() {
        eprintln!(
            "inkling_serve: execution manifest unavailable facts: {}",
            ready.execution_unavailable.join(", ")
        );
    }
    let payload = serde_json::to_vec(&ready)?;
    let extent = payload.len() as u64;
    out.record_as(READY_TYPE, &payload, extent)?;

    // ── serve ───────────────────────────────────────────────────────────────
    //
    // Context accumulates as token ids; a CONSULT record ends the delta and
    // asks for a turn. Typed records contribute structural ids while raw probe
    // text passes through the codec's content-only tokenizer. Nothing here is
    // asynchronous: the client writes, we read until the consult, we generate
    // and write, the client reads. Strict alternation, so the two-pipe deadlock
    // cannot arise.
    let mut delta = Vec::new();
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
                    &mut |id| {
                        decode_stream
                            .step(id)
                            .map_err(|error| anyhow::anyhow!("streaming decode: {error}"))
                    },
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
            framed_stream::Frame::Record(record)
                if record.content_type() == CONTEXT_PREFLIGHT_TYPE =>
            {
                let request: ContextPreflight = serde_json::from_slice(&record.payload)
                    .context("parse typed context preflight")?;
                anyhow::ensure!(
                    delta.is_empty(),
                    "context preflight requires an empty pending delta"
                );
                if request.placement == ContextPlacement::Replace {
                    anyhow::ensure!(
                        matches!(&request.context, InklingContext::Initialize { .. }),
                        "replacement preflight requires one complete Initialize payload"
                    );
                    validate_reinitialize_boundary(turn, delta.len(), carry.is_some())?;
                }
                let encoded = context_codec
                    .encode(&request.context)
                    .context("encode context preflight")?;
                let evidence = context_preflight(
                    request.placement,
                    session.position(),
                    usize::from(carry.is_some()),
                    encoded.len(),
                    request.max_response_tokens,
                    context_budget,
                )?;
                let payload =
                    serde_json::to_vec(&evidence).context("encode context-preflight evidence")?;
                out.record_as(CONTEXT_PREFLIGHTED_TYPE, &payload, payload.len() as u64)?;
            }
            framed_stream::Frame::Record(record) if record.content_type() == REINITIALIZE_TYPE => {
                let initialization: InklingContext = serde_json::from_slice(&record.payload)
                    .context("parse REINITIALIZE initialization")?;
                anyhow::ensure!(
                    matches!(initialization, InklingContext::Initialize { .. }),
                    "REINITIALIZE requires one complete Initialize payload"
                );
                validate_reinitialize_boundary(turn, delta.len(), carry.is_some())?;

                // Every fallible operation that can reject the replacement is
                // above the reset. An invalid or over-wide cover leaves the old
                // sequence byte-for-byte alive.
                let replacement = context_codec
                    .encode(&initialization)
                    .context("encode REINITIALIZE initialization")?;
                anyhow::ensure!(
                    replacement.len() <= context_budget,
                    "the {}-token replacement initialization exceeds this Session's \
                     {context_budget}-token context budget",
                    replacement.len()
                );
                let acknowledgement = Reinitialized {
                    previous_position: session.position(),
                    previous_turns: turn,
                    initialization_tokens: replacement.len(),
                };
                let payload = serde_json::to_vec(&acknowledgement)
                    .context("encode REINITIALIZED acknowledgement")?;

                session.reset();
                decode_stream = tokenizer.decode_stream(false);
                delta = replacement;
                carry = None;
                turn = 0;

                out.record_as(REINITIALIZED_TYPE, &payload, payload.len() as u64)?;
                eprintln!(
                    "inkling_serve: reinitialized after {} turn(s) at position {}; \
                     {} replacement token(s) staged",
                    acknowledgement.previous_turns,
                    acknowledgement.previous_position,
                    acknowledgement.initialization_tokens,
                );
            }
            framed_stream::Frame::Record(record) if record.content_type() == CONTENT_TYPE => {
                delta.extend(
                    context_codec
                        .encode_raw_content(record.text()?)
                        .context("encode raw content record")?,
                );
            }
            framed_stream::Frame::Record(record) if record.content_type() == CONTEXT_TYPE => {
                let context: InklingContext = serde_json::from_slice(&record.payload)
                    .context("parse typed Inkling context record")?;
                delta.extend(
                    context_codec
                        .encode(&context)
                        .context("encode typed Inkling context record")?,
                );
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
                let marker = format!("\n[{} bytes lost: {}]\n", gap.extent, gap.reason);
                delta.extend(
                    context_codec
                        .encode_raw_content(&marker)
                        .context("encode client gap marker")?,
                );
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
    decode: &mut impl FnMut(u32) -> Result<Option<String>>,
    out: &mut framed_stream::FramedWriter<std::fs::File>,
    delta_ids: Vec<usize>,
    want: usize,
    stop: &[u32],
    turn: usize,
    carry: &mut Option<usize>,
) -> Result<TurnEnd> {
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
    let local_layers = session
        .validate_cache_completeness()
        .context("validate every attention cache at the completed-turn seam")?;
    eprintln!(
        "inkling_serve: turn {turn} cache-complete at position {} ({local_layers} local layer(s))",
        session.position()
    );
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

    use super::{
        advance_context_decode, reject_sealed_environment, sealed_environment_rejections,
        validate_reinitialize_boundary,
    };

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
    fn sealed_tp_child_accepts_deployed_rank_local_environment() {
        const CHILD: &str = "MARY_SEALED_ENVIRONMENT_TEST_CHILD";
        if std::env::var_os(CHILD).is_some() {
            reject_sealed_environment()
                .expect("the deployed rank-local environment passes sealed validation");
            return;
        }

        let output = std::process::Command::new(std::env::current_exe().unwrap())
            .arg("tests::sealed_tp_child_accepts_deployed_rank_local_environment")
            .arg("--exact")
            .arg("--nocapture")
            .env_clear()
            .env(CHILD, "1")
            .env("CUDA_VISIBLE_DEVICES", "0")
            .env("NCCL_IB_DISABLE", "0")
            .env("NCCL_SOCKET_IFNAME", "rocep1s0f0")
            .env("NCCL_IB_HCA", "rocep1s0f0")
            .env("NCCL_DEBUG", "INFO")
            .env("NCCL_DEBUG_SUBSYS", "INIT,NET")
            .env("NCCL_DEBUG_FILE", "/tmp/nccl.%h.%p.log")
            .output()
            .expect("launch the sealed-environment fixture child");
        assert!(
            output.status.success(),
            "sealed TP child did not pass environment validation:\nstdout:\n{}\nstderr:\n{}",
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr),
        );
    }

    #[test]
    fn sealed_environment_still_rejects_execution_selection() {
        let rejected = sealed_environment_rejections(
            ["NCCL_ALGO", "CUBECL_MEMORY_CONFIG"]
                .into_iter()
                .map(str::to_owned),
        );
        assert_eq!(rejected, ["CUBECL_MEMORY_CONFIG", "NCCL_ALGO"]);
    }

    #[test]
    fn reinitialization_requires_an_empty_completed_turn_boundary() {
        let before_first = validate_reinitialize_boundary(0, 0, false)
            .expect_err("there is no old sequence to replace");
        assert!(
            before_first.to_string().contains("completed turn"),
            "{before_first:#}"
        );

        let queued = validate_reinitialize_boundary(3, 7, true)
            .expect_err("queued context cannot be discarded");
        assert!(queued.to_string().contains("7 pending"), "{queued:#}");

        validate_reinitialize_boundary(3, 0, true)
            .expect("a completed turn with no pending context is exact");
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
