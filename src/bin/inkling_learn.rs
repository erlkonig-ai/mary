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
         [--gen G] [--tp-rendezvous HOST:PORT] [--layers a:b]"
    );
    let (pile, tokenizer, corpus) = (&args[1], &args[2], &args[3]);
    let mut from = 100usize;
    let mut turns = 8usize;
    let mut want = 1usize;
    let mut rendezvous: Option<String> = None;
    let mut layers: Option<std::ops::Range<usize>> = None;
    let mut i = 4;
    while i < args.len() {
        match args[i].as_str() {
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
        context_budget: None,
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
        println!(
            "turn {k:3} ({:3} scored of {:3} delta): {mean:.4} nats/token  first {:.2}s  turn {:.2}s  said {:?}",
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
    engine.shutdown()?;
    Ok(())
}
