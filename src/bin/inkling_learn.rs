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
//! Each corpus line is one of JP's turns. By default it arrives through the
//! shipped command-result path, which renders its text in the user role.
//! `--input-role tool` instead renders the SAME text as a sensed text record,
//! in the tool role after a content-tokenized source label. The report splits
//! the prequential score into literal content tokens and everything framing
//! them (role, source, structural markers and the next-response prompt), while
//! retaining the whole-delta score for continuity with earlier runs.
//!
//! `INK_LEARN_LR=<lr>` on BOTH ranks arms the learner (the last layer's routed
//! experts); unset, this is the no-learning baseline over the same turns.
//! `INK_LEARN_RN=1` is the nearest-rounding control.
//!
//! ```text
//! INK_LEARN_LR=1.0 inkling_learn <pile> <turns.txt> \
//!     [--from LINE] [--turns N] [--gen G] [--input-role user|tool] \
//!     [--source-label TEXT] [--tp-rendezvous HOST:PORT] [--layers a:b]
//! ```
use anyhow::{Context, Result};
use mary::models::inkling::engine::{self, EngineConfig, Loaded, TensorParallel};
use mary::models::inkling::resident::{
    Consult, ExecResultContext, InklingContext, InklingInput, Model, SenseMedia, SenseRecord,
};
use mary::models::inkling::tpcomm::elect_rank;

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();
    anyhow::ensure!(
        args.len() >= 3,
        "usage: inkling_learn <pile> <turns.txt> [--from LINE] [--turns N] \
         [--gen G] [--input-role user|tool] [--source-label TEXT] \
         [--tp-rendezvous HOST:PORT] [--layers a:b] [--model-root ID] [--export] \
         [--save | --save-commit --signing-key <path>]"
    );
    let (pile, corpus) = (&args[1], &args[2]);
    let mut from = 100usize;
    let mut turns = 8usize;
    let mut want = 1usize;
    let mut rendezvous: Option<String> = None;
    let mut layers: Option<std::ops::Range<usize>> = None;
    let mut model_root = None;
    let mut export = false;
    let mut save = Save::No;
    let mut signing_key: Option<String> = None;
    let mut explain: Option<String> = None;
    let mut input_role = InputRole::User;
    let mut source_label = "message poll".to_string();
    let mut i = 3;
    while i < args.len() {
        match args[i].as_str() {
            "--model-root" => {
                model_root = Some(
                    triblespace::prelude::Id::from_hex(
                        args.get(i + 1)
                            .context("--model-root wants a 32-hex model root id")?,
                    )
                    .context("--model-root wants a 32-hex model root id")?,
                );
                i += 2;
            }
            // After the last turn, pull every learned expert out of both
            // ranks' arenas, joined whole, and say what came back. Nothing is
            // written to a pile yet (see `inkling::learned`).
            "--export" => {
                export = true;
                i += 1;
            }
            // After the export, assemble the learned VERSION -- a new root in
            // the model graph whose members are the parent's with the learned
            // leaves substituted, a `parent` edge, and the recipe -- and say
            // what it holds. `--save` stops there; `--save-commit` commits it,
            // signed with `--signing-key`. See `inkling::learned`.
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
            // Why this version exists, in words, onto the version root.
            "--explain" => {
                explain = Some(args[i + 1].clone());
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
            "--input-role" => {
                input_role =
                    InputRole::parse(args.get(i + 1).context("--input-role wants user or tool")?)?;
                i += 2;
            }
            "--source-label" => {
                source_label = args
                    .get(i + 1)
                    .context("--source-label wants text")?
                    .clone();
                i += 2;
            }
            "--tp-rendezvous" => {
                rendezvous = Some(args[i + 1].clone());
                i += 2;
            }
            "--layers" => {
                let (a, b) = args[i + 1].split_once(':').context("--layers wants a:b")?;
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
        "=== inkling_learn: {} turns from line {from}, gen {want}, input {}, source {:?}, learning {}, {} ===",
        lines.len(),
        input_role.as_str(),
        source_label,
        lr.as_deref().unwrap_or("OFF (baseline)"),
        match &rendezvous {
            Some(a) => format!("tensor-parallel pair via {a}"),
            None => "one rank".to_string(),
        }
    );

    let t0 = std::time::Instant::now();
    let loaded = engine::load(EngineConfig {
        cache: None,
        distillation: None,
        pile: pile.into(),
        model_root,
        layers,
        prefill_budget: None,
        // The engine's default is the million-position window the resident is
        // built for. This is the replay bench: a few hundred of his turns at
        // about 53 tokens each, on a pair whose gate (2026-09-03 17:06Z) prices
        // the million at 116.48 GiB against 112.03 available once the rank's
        // own footprint is in place -- it fits the machine by 5.15 GiB and
        // misses the moment by 4.45. The bench asks for what it uses.
        context_budget: Some(16384),
        preallocate_kv: false,
        weight_storage: mary::models::inkling::pile::WeightStorage::Host,
        cached_attention: mary::models::inkling::flash::CachedAttentionPolicy::Legacy,
        tensor_parallel,
        sealed: false,
        // The bench writes a version through its own --save flow, if at all.
        signing_key: None,
    })?;
    let mut engine = match loaded {
        Loaded::Follower(mut follower) => {
            println!(
                "  rank 1 ready in {:.1}s; following",
                t0.elapsed().as_secs_f64()
            );
            return follower.follow();
        }
        Loaded::Engine(engine) => engine,
    };
    println!(
        "  ready in {:.1}s: {}",
        t0.elapsed().as_secs_f64(),
        format!("{:?}", engine.ready())
            .chars()
            .take(200)
            .collect::<String>()
    );

    let mut means = Vec::with_capacity(lines.len());
    let mut frozen_means = Vec::with_capacity(lines.len());
    let mut attributed = AttributedLoss::default();
    let mut attributed_frozen = AttributedLoss::default();
    let mut content_fingerprint = blake3::Hasher::new();
    for (k, line) in lines.iter().enumerate() {
        let context = InklingContext::Observation {
            inputs: vec![input_role.input(&source_label, line)],
        };
        let delta_ids = engine.encode_context_ids(&context)?;
        let special = &engine.ready().special_ids;
        let content_span = text_content_span(
            &delta_ids,
            special.content_text as usize,
            special.end_message as usize,
        )?;
        let content_ids = &delta_ids[content_span.clone()];
        content_fingerprint.update(&(content_ids.len() as u64).to_le_bytes());
        for &id in content_ids {
            content_fingerprint.update(&(id as u64).to_le_bytes());
        }
        engine.context(&context)?;
        let mut said = String::new();
        let end = engine.consult(&Consult::new(want), &mut |text| {
            said.push_str(text);
            Ok(())
        })?;
        let mean = end.delta_mean_nll().unwrap_or(f64::NAN);
        means.push(mean);
        anyhow::ensure!(
            end.delta_tokens == delta_ids.len(),
            "turn {k} encoded {} ids but scored a {}-token delta",
            delta_ids.len(),
            end.delta_tokens
        );
        let turn_attributed = attribute_scores(&delta_ids, content_span.clone(), &end.delta_nll)?;
        attributed += turn_attributed;
        // The control, when a layer is frozen: the checkpoint's experts over
        // the same rows of the same pass. Its column is the null hypothesis
        // of every turn.
        let frozen = match end.delta_mean_nll_frozen() {
            Some(f) => {
                anyhow::ensure!(
                    end.delta_nll_frozen.len() == end.delta_nll.len(),
                    "turn {k} has {} learned scores but {} frozen scores",
                    end.delta_nll.len(),
                    end.delta_nll_frozen.len()
                );
                let turn_frozen =
                    attribute_scores(&delta_ids, content_span, &end.delta_nll_frozen)?;
                attributed_frozen += turn_frozen;
                frozen_means.push(f);
                format!(
                    "  frozen {f:.4} [content {}, frame {}]",
                    turn_frozen.content.describe(),
                    turn_frozen.frame.describe()
                )
            }
            None => String::new(),
        };
        println!(
            "turn {k:3} ({:3} scored of {:3} delta): {mean:.4} nats/token [content {}, frame {}]{frozen}  first {:.2}s  turn {:.2}s  said {:?}",
            end.delta_nll.len(),
            end.delta_tokens,
            turn_attributed.content.describe(),
            turn_attributed.frame.describe(),
            end.first_token_secs,
            end.turn_secs,
            said
        );
    }
    let n = means.len();
    let all = means.iter().sum::<f64>() / n as f64;
    let later = means.iter().skip(1).sum::<f64>() / (n - 1).max(1) as f64;
    println!(
        "=== {n} turns: mean {all:.4} nats/delta token; turns 1.. {later:.4}; per turn {} ===",
        means
            .iter()
            .map(|m| format!("{m:.3}"))
            .collect::<Vec<_>>()
            .join(" ")
    );
    println!(
        "=== attributed {}: content {}; frame+source {}; content-token fingerprint {} ===",
        input_role.as_str(),
        attributed.content.describe(),
        attributed.frame.describe(),
        content_fingerprint.finalize()
    );
    if frozen_means.len() == n {
        let f_all = frozen_means.iter().sum::<f64>() / n as f64;
        let wins = means
            .iter()
            .zip(&frozen_means)
            .filter(|(m, f)| m < f)
            .count();
        println!(
            "=== frozen control: mean {f_all:.4} nats/delta token; learned below frozen on {wins}/{n} turns; per turn {} ===",
            frozen_means
                .iter()
                .map(|m| format!("{m:.3}"))
                .collect::<Vec<_>>()
                .join(" ")
        );
        println!(
            "=== attributed frozen: content {}; frame+source {}; learned-minus-frozen: content {}, frame+source {} ===",
            attributed_frozen.content.describe(),
            attributed_frozen.frame.describe(),
            attributed.content.delta_from(attributed_frozen.content),
            attributed.frame.delta_from(attributed_frozen.frame)
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
            use mary::models::inkling::resident::VersionRecipe;
            use mary::models::inkling::version::{learned_version, publish_version};
            use triblespace::prelude::Pile;
            use triblespace::prelude::inlineencodings::{Blake3, Hash};
            let t = std::time::Instant::now();
            let mut store = Pile::open(std::path::Path::new(pile))
                .map_err(|e| anyhow::anyhow!("open {pile} to write: {e:?}"))?;
            store
                .refresh()
                .map_err(|e| anyhow::anyhow!("refresh {pile}: {e:?}"))?;
            let recipe = VersionRecipe {
                lr: lr.as_deref().and_then(|v| v.parse().ok()).unwrap_or(0.0),
                anchor: std::env::var("INK_LEARN_ANCHOR")
                    .ok()
                    .and_then(|v| v.parse().ok()),
                seed: std::env::var("INK_LEARN_SEED")
                    .ok()
                    .and_then(|v| v.parse().ok())
                    .unwrap_or(0),
                steps: lines.len() as u64,
                span: format!(
                    "{corpus} lines {from}..{}, input {}, source {:?}, {want} generated token(s) a turn",
                    from + lines.len(),
                    input_role.as_str(),
                    source_label
                ),
                explanation: explain.clone().unwrap_or_default(),
                code_revision: std::env::var("INK_CODE_REVISION").unwrap_or_default(),
            };
            let parent = triblespace::prelude::Id::from_hex(&engine.ready().model_root)
                .context("loaded engine has no valid model root")?;
            let collection = Hash::<Blake3>::from_hex(&engine.ready().model_collection)
                .context("loaded engine has no valid model collection")?
                .transmute();
            let version = learned_version(&mut store, collection, &learned, parent, &recipe)?;
            println!(
                "=== version {:X} '{}': parent {:X}, {} leaves replaced, {} members, {} facts to add, assembled in {:.1}s ===",
                version.root,
                version.name,
                version.parent,
                version.replaced,
                version.members,
                version.facts.len(),
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
                let name = version.name.clone();
                publish_version(&mut store, &key, version)?;
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

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum InputRole {
    User,
    Tool,
}

impl InputRole {
    fn parse(value: &str) -> Result<Self> {
        match value {
            "user" => Ok(Self::User),
            "tool" => Ok(Self::Tool),
            other => anyhow::bail!("--input-role wants user or tool, got {other:?}"),
        }
    }

    fn as_str(self) -> &'static str {
        match self {
            Self::User => "user",
            Self::Tool => "tool",
        }
    }

    fn input(self, source: &str, content: &str) -> InklingInput {
        match self {
            Self::User => InklingInput::text_result(ExecResultContext {
                command: source.to_string(),
                content: content.to_string(),
            }),
            Self::Tool => InklingInput::Sensed {
                record: SenseRecord {
                    source: source.to_string(),
                    media: SenseMedia::Text {
                        text: content.to_string(),
                    },
                },
            },
        }
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq)]
struct LossSlice {
    sum: f64,
    scored: usize,
}

impl LossSlice {
    fn push(&mut self, score: f32) {
        if score.is_finite() {
            self.sum += score as f64;
            self.scored += 1;
        }
    }

    fn mean(self) -> Option<f64> {
        (self.scored > 0).then(|| self.sum / self.scored as f64)
    }

    fn describe(self) -> String {
        match self.mean() {
            Some(mean) => format!("{mean:.4} nats/token over {}", self.scored),
            None => "not scored".to_string(),
        }
    }

    fn delta_from(self, control: Self) -> String {
        match (self.mean(), control.mean()) {
            (Some(live), Some(frozen)) => format!("{:+.4} nats/token", live - frozen),
            _ => "not scored".to_string(),
        }
    }
}

impl std::ops::AddAssign for LossSlice {
    fn add_assign(&mut self, rhs: Self) {
        self.sum += rhs.sum;
        self.scored += rhs.scored;
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq)]
struct AttributedLoss {
    content: LossSlice,
    frame: LossSlice,
}

impl std::ops::AddAssign for AttributedLoss {
    fn add_assign(&mut self, rhs: Self) {
        self.content += rhs.content;
        self.frame += rhs.frame;
    }
}

fn text_content_span(
    delta_ids: &[usize],
    content_text: usize,
    end_message: usize,
) -> Result<std::ops::Range<usize>> {
    let content_markers = delta_ids
        .iter()
        .enumerate()
        .filter_map(|(index, id)| (*id == content_text).then_some(index))
        .collect::<Vec<_>>();
    anyhow::ensure!(
        content_markers.len() == 1,
        "text observation has {} content-text markers, expected one",
        content_markers.len()
    );
    let ends = delta_ids
        .iter()
        .enumerate()
        .filter_map(|(index, id)| (*id == end_message).then_some(index))
        .collect::<Vec<_>>();
    anyhow::ensure!(
        ends.len() == 1,
        "text observation has {} end-message markers, expected one",
        ends.len()
    );
    let start = content_markers[0] + 1;
    let end = ends[0];
    anyhow::ensure!(
        start <= end,
        "the content-text marker at {} follows the end-message marker at {end}",
        start - 1
    );
    Ok(start..end)
}

fn attribute_scores(
    delta_ids: &[usize],
    content: std::ops::Range<usize>,
    scores: &[f32],
) -> Result<AttributedLoss> {
    anyhow::ensure!(
        content.start <= content.end && content.end <= delta_ids.len(),
        "content span {:?} is outside a {}-token delta",
        content,
        delta_ids.len()
    );
    anyhow::ensure!(
        scores.len() <= delta_ids.len(),
        "{} scores cannot describe a {}-token delta",
        scores.len(),
        delta_ids.len()
    );
    let first_scored = delta_ids.len() - scores.len();
    if !scores.is_empty() {
        anyhow::ensure!(
            first_scored <= 1,
            "scored delta omitted {first_scored} leading tokens, expected at most one"
        );
    }
    let mut attributed = AttributedLoss::default();
    for (offset, &score) in scores.iter().enumerate() {
        let delta_index = first_scored + offset;
        if content.contains(&delta_index) {
            attributed.content.push(score);
        } else {
            attributed.frame.push(score);
        }
    }
    Ok(attributed)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn locates_the_literal_text_between_structural_markers() {
        let ids = [10, 20, 30, 31, 40, 50];
        assert_eq!(text_content_span(&ids, 20, 40).unwrap(), 2..4);
    }

    #[test]
    fn attributes_a_first_turn_after_its_unscored_role_token() {
        let ids = [10, 20, 30, 31, 40, 50];
        let loss = attribute_scores(&ids, 2..4, &[1.0, 2.0, 3.0, 4.0, 5.0]).unwrap();
        assert_eq!(
            loss.content,
            LossSlice {
                sum: 5.0,
                scored: 2
            }
        );
        assert_eq!(
            loss.frame,
            LossSlice {
                sum: 10.0,
                scored: 3
            }
        );
    }

    #[test]
    fn attributes_a_primed_turn_from_its_first_delta_token() {
        let ids = [10, 20, 30, 31, 40, 50];
        let loss = attribute_scores(&ids, 2..4, &[1.0, 2.0, 3.0, 4.0, 5.0, f32::NAN]).unwrap();
        assert_eq!(
            loss.content,
            LossSlice {
                sum: 7.0,
                scored: 2
            }
        );
        assert_eq!(
            loss.frame,
            LossSlice {
                sum: 8.0,
                scored: 3
            }
        );
    }
}
