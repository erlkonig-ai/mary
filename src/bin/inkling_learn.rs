//! Serve JP's real turns through the resident [`Engine`], learning as it goes,
//! and print the one score that counts: the prequential loss of each turn
//! under the weights in force when it arrived.
//!
//! This is the online-learning path on the WHOLE model, which needs both
//! Sparks: run the same command on both boxes with `--tp-rendezvous` naming
//! rank 0's address on the fast fabric, and each box elects its rank by
//! address match (`tpcomm::elect_rank`). Rank 0 feeds the turns and prints;
//! rank 1 follows the passes rank 0 names — including the scored ones, so its
//! cut of every expert learns too. Without `--tp-rendezvous` it is one rank on
//! a partial stack (`INK_LAYERS`), whose numbers are diagnostic.
//!
//! Each corpus line is one of JP's turns. The resident never sees a bare user
//! message; his words reach it as the content of a tool result, so that is how
//! they are rendered here: a `ToolResult` whose command names the faculty that
//! would have carried them and whose content is the line. The score is over
//! every appended id (the wrapper and the words), in nats per delta token.
//!
//! `INK_LEARN_LR=<lr>` on BOTH ranks arms the learner (the last layer's routed
//! experts); unset, this is the no-learning baseline over the same turns.
//! `INK_LEARN_RN=1` is the nearest-rounding control.
//!
//! ```text
//! INK_LEARN_LR=1.0 inkling_learn <pile> <tokenizer.json> <turns.txt> \
//!     [--from LINE] [--turns N] [--gen G] [--tp-rendezvous HOST:PORT] [--layers a:b]
//! ```
use anyhow::{Context, Result};
use mary::models::inkling::engine::{self, EngineConfig, Loaded, TensorParallel};
use mary::models::inkling::resident::{Consult, ExecResultContext, InklingContext, Model};
use mary::models::inkling::tpcomm::elect_rank;

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();
    anyhow::ensure!(
        args.len() >= 4,
        "usage: inkling_learn <pile> <tokenizer.json> <turns.txt> [--from LINE] [--turns N] \
         [--gen G] [--tp-rendezvous HOST:PORT] [--layers a:b] [--export] \
         [--save | --save-commit --signing-key <path>]"
    );
    let (pile, tokenizer, corpus) = (&args[1], &args[2], &args[3]);
    let mut from = 100usize;
    let mut turns = 8usize;
    let mut want = 1usize;
    let mut rendezvous: Option<String> = None;
    let mut layers: Option<std::ops::Range<usize>> = None;
    let mut export = false;
    let mut save = Save::No;
    let mut signing_key: Option<String> = None;
    let mut i = 4;
    while i < args.len() {
        match args[i].as_str() {
            // After the last turn, pull every learned expert out of both
            // ranks' arenas, joined whole, and say what came back. Nothing is
            // written to a pile yet (see `inkling::learned`).
            "--export" => {
                export = true;
                i += 1;
            }
            // After the export, assemble the learned snapshot -- the parent
            // collection's facts with the learned leaves substituted -- as a
            // model collection named after the parent, and say what it holds.
            // `--save` stops there; `--save-commit` commits it, signed with
            // `--signing-key`. See `inkling::learned`.
            "--save" => {
                export = true;
                save = Save::Dry;
                i += 1;
            }
            "--save-commit" => {
                export = true;
                save = Save::Commit;
                i += 1;
            }
            "--signing-key" => {
                signing_key = Some(args[i + 1].clone());
                i += 2;
            }
            "--from" => {
                from = args[i + 1].parse().context("--from wants a line number")?;
                i += 2;
            }
            "--turns" => {
                turns = args[i + 1].parse().context("--turns wants a count")?;
                i += 2;
            }
            "--gen" => {
                want = args[i + 1].parse().context("--gen wants a count")?;
                i += 2;
            }
            "--tp-rendezvous" => {
                rendezvous = Some(args[i + 1].clone());
                i += 2;
            }
            "--layers" => {
                let (a, b) = args[i + 1]
                    .split_once(':')
                    .context("--layers wants a:b")?;
                layers = Some(a.parse()?..b.parse()?);
                i += 2;
            }
            other => anyhow::bail!("unknown argument {other:?}"),
        }
    }

    let lines: Vec<String> = std::fs::read_to_string(corpus)
        .with_context(|| format!("read the turns from {corpus}"))?
        .lines()
        .skip(from)
        .take(turns)
        .map(|l| l.to_string())
        .filter(|l| !l.trim().is_empty())
        .collect();
    anyhow::ensure!(!lines.is_empty(), "no turns from line {from} in {corpus}");

    let tensor_parallel = match &rendezvous {
        Some(addr) => Some(TensorParallel {
            tp: elect_rank(addr, 2)?,
            rendezvous: addr.clone(),
        }),
        None => None,
    };
    let lr = std::env::var("INK_LEARN_LR").ok();
    println!(
        "=== inkling_learn: {} turns from line {from}, gen {want}, learning {}, {} ===",
        lines.len(),
        lr.as_deref().unwrap_or("OFF (baseline)"),
        match &rendezvous {
            Some(a) => format!("tensor-parallel pair via {a}"),
            None => "one rank".to_string(),
        }
    );

    let t0 = std::time::Instant::now();
    let loaded = engine::load(EngineConfig {
        pile: pile.into(),
        tokenizer: tokenizer.into(),
        layers,
        prefill_budget: None,
        // The engine's default is the million-position window the resident is
        // built for. This is the replay bench: a few hundred of his turns at
        // about 53 tokens each, on a pair whose gate (2026-09-03 17:06Z) prices
        // the million at 116.48 GiB against 112.03 available once the rank's
        // own footprint is in place -- it fits the machine by 5.15 GiB and
        // misses the moment by 4.45. The bench asks for what it uses.
        context_budget: Some(16384),
        tensor_parallel,
        sealed: false,
    })?;
    let mut engine = match loaded {
        Loaded::Follower(mut follower) => {
            println!("  rank 1 ready in {:.1}s; following", t0.elapsed().as_secs_f64());
            return follower.follow();
        }
        Loaded::Engine(engine) => engine,
    };
    println!(
        "  ready in {:.1}s: {}",
        t0.elapsed().as_secs_f64(),
        format!("{:?}", engine.ready()).chars().take(200).collect::<String>()
    );

    let mut means = Vec::with_capacity(lines.len());
    let mut frozen_means = Vec::with_capacity(lines.len());
    for (k, line) in lines.iter().enumerate() {
        // His words, the way the resident meets them: as what the message
        // faculty printed.
        engine.context(&InklingContext::ToolResult {
            result: ExecResultContext {
                command: "message poll".to_string(),
                content: line.clone(),
            },
        })?;
        let mut said = String::new();
        let end = engine.consult(&Consult::new(want), &mut |text| {
            said.push_str(text);
            Ok(())
        })?;
        let mean = end.delta_mean_nll().unwrap_or(f64::NAN);
        means.push(mean);
        // The control, when a layer is frozen: the checkpoint's experts over
        // the same rows of the same pass. Its column is the null hypothesis
        // of every turn.
        let frozen = match end.delta_mean_nll_frozen() {
            Some(f) => {
                frozen_means.push(f);
                format!("  frozen {f:.4}")
            }
            None => String::new(),
        };
        println!(
            "turn {k:3} ({:3} scored of {:3} delta): {mean:.4} nats/token{frozen}  first {:.2}s  turn {:.2}s  said {:?}",
            end.delta_nll.len(),
            end.delta_tokens,
            end.first_token_secs,
            end.turn_secs,
            said.chars().take(40).collect::<String>()
        );
    }
    let n = means.len();
    let all = means.iter().sum::<f64>() / n as f64;
    let later = means.iter().skip(1).sum::<f64>() / (n - 1).max(1) as f64;
    println!(
        "=== {n} turns: mean {all:.4} nats/delta token; turns 1.. {later:.4}; per turn {} ===",
        means.iter().map(|m| format!("{m:.3}")).collect::<Vec<_>>().join(" ")
    );
    if frozen_means.len() == n {
        let f_all = frozen_means.iter().sum::<f64>() / n as f64;
        let wins = means.iter().zip(&frozen_means).filter(|(m, f)| m < f).count();
        println!(
            "=== frozen control: mean {f_all:.4} nats/delta token; learned below frozen on {wins}/{n} turns; per turn {} ===",
            frozen_means.iter().map(|m| format!("{m:.3}")).collect::<Vec<_>>().join(" ")
        );
    }
    if export {
        let t = std::time::Instant::now();
        let learned = engine.export_learned()?;
        let secs = t.elapsed().as_secs_f64();
        let mut per_name: std::collections::BTreeMap<&str, (usize, (usize, usize))> =
            Default::default();
        let mut blob_bytes = 0usize;
        for x in &learned {
            let blob = mary::models::inkling::pile::expert_blob(&x.packed)
                .with_context(|| format!("{}[{}] as a pile leaf", x.name, x.expert))?;
            blob_bytes += blob.bytes.len();
            let e = per_name.entry(x.name.as_str()).or_default();
            e.0 += 1;
            e.1 = (x.packed.rows, x.packed.cols * 2);
        }
        println!(
            "=== export: {} learned experts, {:.1} MiB of leaves, in {secs:.1}s ===",
            learned.len(),
            blob_bytes as f64 / (1u64 << 20) as f64
        );
        for (name, (count, (rows, logical))) in &per_name {
            println!("  {name}: {count} experts, each [{rows}, {logical}]");
        }
        if save != Save::No && !learned.is_empty() {
            use mary::models::inkling::learned::{learned_snapshot, publish_learned_snapshot};
            use triblespace::prelude::Pile;
            let parent = mary::model_collection::mary_model_graph_name();
            let t = std::time::Instant::now();
            let mut store = Pile::open(std::path::Path::new(pile))
                .map_err(|e| anyhow::anyhow!("open {pile} to write: {e:?}"))?;
            store
                .refresh()
                .map_err(|e| anyhow::anyhow!("refresh {pile}: {e:?}"))?;
            let snapshot = learned_snapshot(&mut store, parent, &learned)?;
            println!(
                "=== snapshot '{}': {} leaves replaced, {} roots rebuilt, {} facts kept, {} facts in all, assembled in {:.1}s ===",
                snapshot.name,
                snapshot.replaced,
                snapshot.roots,
                snapshot.kept,
                snapshot.facts.len(),
                t.elapsed().as_secs_f64()
            );
            if save == Save::Commit {
                let key_path = signing_key
                    .as_deref()
                    .context("--save-commit needs --signing-key <path>")?;
                let key = triblespace::core::signing_key_file::load_existing(std::path::Path::new(
                    key_path,
                ))
                .with_context(|| format!("load the signing key {key_path}"))?;
                let name = snapshot.name;
                publish_learned_snapshot(&mut store, &key, snapshot)?;
                println!("=== committed '{name}' ===");
            } else {
                println!("  (not committed; --save-commit --signing-key <path> commits it)");
            }
            store
                .close()
                .map_err(|e| anyhow::anyhow!("close {pile}: {e:?}"))?;
        }
    }
    engine.shutdown()?;
    Ok(())
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Save {
    No,
    Dry,
    Commit,
}
