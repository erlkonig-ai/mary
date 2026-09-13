//! Per-rank execution of background self-distillation against a resident model.
//!
//! The caller orders the SAME Work on every tensor-parallel rank. This object
//! owns independent student capsules, one privileged-context teacher capsule,
//! soft targets, and the real EMA bank; it does not sample background tokens,
//! execute actions, or write the live transcript. Foreground decode can share
//! a row batch and returns only its own greedy token for the live parser.
//!
//! A cohort rolls out and scores under fixed student/teacher versions. ALL
//! prepared slots must be scored before the first optimizer step. Subsequent
//! per-sequence steps deliberately consume that same rollout cohort rather
//! than claiming freshly on-policy data after each update. Teacher EMA advances
//! once after every prepared slot has learned, then the capsules are reset.

use anyhow::{Context, Result, ensure};

use super::assembly::T2;
use super::learn::{LearnReport, ema::{EmaBank, EmaReport}, validate_soft_targets};
use super::sdft::{Config, PreparedExample, Work, response_inputs};
use super::session::{Sequence, Session};

/// GPU logits stay local to the rank. The scheduler samples only shadow rows;
/// `foreground` is the sole output authorized to enter the live decode path.
pub enum Outcome {
    Logits(T2),
    Decoded { foreground: Option<usize>, logits: T2 },
    Scored { rows: usize, teacher_version: u64 },
    Learned(LearnReport),
    TeacherUpdated(EmaReport),
}

struct Episode {
    example: PreparedExample,
    /// Tokens explicitly consumed by Decode after the complete prompt.
    consumed: Vec<usize>,
    rollout_version: u64,
    scored_teacher_version: Option<u64>,
    learned: bool,
}

struct Targets {
    distribution: T2,
    continuation: Vec<usize>,
    rollout_version: u64,
    teacher_version: u64,
}

/// One resident runtime. Construction consumes the sequence reservations made
/// by SessionConfig.distillation; it does not load another set of trunk weights.
pub struct Runtime {
    config: Config,
    students: Vec<Sequence>,
    teacher: Sequence,
    episodes: Vec<Option<Episode>>,
    targets: Vec<Option<Targets>>,
    bank: Option<EmaBank>,
    version: u64,
    rollout_version: Option<u64>,
    learning: bool,
    poisoned: bool,
}

impl Runtime {
    pub fn new(session: &mut Session, config: Config) -> Result<Self> {
        config.validate()?;
        ensure!(config.sequences < 64 && config.max_rollout <= 64,
            "SDFT runtime admits at most 63 background sequences and 64 response rows");
        let version = session.learning_step().context("SDFT runtime requires an admitted explicit learner")?;
        let mut students = Vec::with_capacity(config.sequences);
        for _ in 0..config.sequences {
            let sequence = session.new_sequence()?;
            ensure!(sequence.context_budget() >= config.context_budget,
                "student capsule was not admitted for the SDFT context budget");
            students.push(sequence);
        }
        let teacher = session.new_sequence()?;
        ensure!(teacher.context_budget() >= config.context_budget,
            "teacher capsule was not admitted for the SDFT context budget");
        Ok(Self {
            episodes: (0..config.sequences).map(|_| None).collect(),
            targets: (0..config.sequences).map(|_| None).collect(),
            config, students, teacher, bank: None, version, rollout_version: None,
            learning: false, poisoned: false,
        })
    }

    /// Latest optimizer step, not the earlier version that generated a cohort.
    pub fn version(&self) -> u64 { self.version }

    /// Before lazy teacher initialization its first version is defined as zero.
    pub fn teacher_version(&self) -> u64 {
        self.bank.as_ref().map(|bank| bank.report().version).unwrap_or(0)
    }

    pub fn is_poisoned(&self) -> bool { self.poisoned }

    /// Host-only preflight for the leader BEFORE broadcasting a Work. Session
    /// performs its own cache/budget/media checks before executing each pass.
    /// Refused work here mutates neither the runtime nor the resident model.
    pub fn validate(&self, session: &Session, work: &Work) -> Result<()> {
        ensure!(!self.poisoned, "an interrupted SDFT operation poisoned this runtime");
        ensure!(session.learning_step() == Some(self.version),
            "student weights changed outside this SDFT runtime");
        if let Some(bank) = &self.bank {
            ensure!(!bank.report().poisoned, "the SDFT teacher bank is poisoned");
        }
        let vocab = session.config().text_config.effective_vocab();
        match work {
            Work::Prepare { slot, example } => {
                self.slot(*slot)?;
                ensure!(!self.learning, "cannot prepare a new rollout during cohort learning");
                ensure!(self.episodes[*slot].is_none(), "SDFT slot {slot} already belongs to this cohort");
                ensure!(self.rollout_version.is_none_or(|v| v == self.version), "rollout student version changed");
                example.validate(&self.config, vocab)?;
            }
            Work::Decode { active, slots, tokens } => {
                validate_decode_slots(slots, tokens.len(), self.students.len())?;
                ensure!(active.is_some() || !slots.is_empty(), "empty SDFT decode batch");
                ensure!(slots.len() + usize::from(active.is_some()) <= 64, "SDFT decode exceeds 64 rows");
                if let Some(token) = active {
                    ensure!(*token < vocab, "foreground decode token is outside the vocabulary");
                }
                for (&slot, &token) in slots.iter().zip(tokens) {
                    ensure!(!self.learning, "background rollout cannot continue after cohort learning started");
                    let episode = self.episode(slot)?;
                    ensure!(episode.scored_teacher_version.is_none(), "SDFT slot {slot} was already scored");
                    ensure!(episode.rollout_version == self.version, "SDFT decode uses stale student context");
                    ensure!(episode.consumed.len() < self.config.max_rollout, "SDFT rollout exceeded its token budget");
                    ensure!(token < vocab, "background decode token is outside the vocabulary");
                    ensure!(!self.students[slot].is_poisoned(), "SDFT student capsule is poisoned");
                }
            }
            Work::Score { slot, continuation, student_version } => {
                let episode = self.episode(*slot)?;
                ensure!(!self.learning, "all teacher scores must precede the first cohort update");
                ensure!(episode.scored_teacher_version.is_none(), "SDFT slot {slot} was already scored");
                ensure!(*student_version == self.version && *student_version == episode.rollout_version,
                    "teacher score does not name the student's rollout version");
                validate_continuation(&episode.consumed, continuation, self.config.max_rollout, vocab)?;
                ensure!(self.bank.is_some(), "teacher has not been initialized by a prepared student");
            }
            Work::Learn { slot, student_version, teacher_version } => {
                let episode = self.episode(*slot)?;
                ensure!(!episode.learned, "SDFT slot {slot} was already learned");
                ensure!(ready_for_learning(&self.episodes), "every prepared slot must be scored before learning");
                let target = self.targets[*slot].as_ref().context("SDFT slot has no teacher targets")?;
                ensure!(*student_version == episode.rollout_version && *student_version == target.rollout_version
                    && self.rollout_version == Some(*student_version), "learning work names a different rollout epoch");
                ensure!(*teacher_version == target.teacher_version && *teacher_version == self.teacher_version()
                    && episode.scored_teacher_version == Some(*teacher_version), "learning work names stale teacher targets");
                ensure!(self.version >= *student_version, "learner precedes the version that generated its data");
            }
            Work::UpdateTeacher { student_version, teacher_version } => {
                ensure!(self.learning && finished_learning(&self.episodes),
                    "teacher update requires every prepared slot to have learned exactly once");
                ensure!(*student_version == self.version, "teacher update names an outdated student step");
                self.bank.as_ref().context("SDFT teacher is missing")?
                    .validate_advance(*teacher_version, *student_version)?;
            }
        }
        Ok(())
    }

    /// Execute one rank's agreed work. Ordinary errors after execution begins
    /// leave this runtime poisoned, never retryable under misleading old epoch
    /// metadata. Capsule helpers restore the foreground on ordinary errors;
    /// panics/device failures require the owning engine's fatal/poison handling.
    pub fn execute(&mut self, session: &mut Session, work: &Work) -> Result<Outcome> {
        self.validate(session, work)?;
        self.poisoned = true;
        let result = self.execute_inner(session, work);
        if result.is_ok() { self.poisoned = false; }
        result
    }

    fn execute_inner(&mut self, session: &mut Session, work: &Work) -> Result<Outcome> {
        match work {
            Work::Prepare { slot, example } => {
                let logits = session.with_sequence(&mut self.students[*slot], |session| {
                    session.reset();
                    session.extend_logits(&example.student)
                })?;
                if self.bank.is_none() {
                    self.bank = Some(session.create_teacher_bank(self.config.ema_decay)?);
                }
                ensure!(session.learning_step() == Some(self.version), "preparing a student unexpectedly learned");
                self.rollout_version = Some(self.version);
                self.episodes[*slot] = Some(Episode {
                    example: example.clone(), consumed: Vec::new(), rollout_version: self.version,
                    scored_teacher_version: None, learned: false,
                });
                Ok(Outcome::Logits(logits))
            }
            Work::Decode { active, slots, tokens } => {
                // One mutable borrow per validated unique slot, reordered into
                // caller order without raw pointers or moving capsules away.
                let logits = {
                    let mut selected: Vec<(usize, &mut Sequence)> = self.students.iter_mut().enumerate()
                        .filter_map(|(index, sequence)| {
                            slots.iter().position(|&slot| slot == index).map(|order| (order, sequence))
                        }).collect();
                    selected.sort_by_key(|(order, _)| *order);
                    let mut sequences: Vec<&mut Sequence> = selected.into_iter().map(|(_, sequence)| sequence).collect();
                    session.batch_logits(*active, &mut sequences, tokens)?
                };
                let foreground = if active.is_some() {
                    // Reduce only the live row, without the INK_TOPB debug
                    // printer or a host read of the full vocabulary.
                    let vocab = logits.dims()[1];
                    let token = logits.clone().slice([0..1, 0..vocab]).argmax(1)
                        .into_data().iter::<i64>().next().context("foreground argmax returned no row")?;
                    let token = usize::try_from(token).context("foreground argmax returned a negative token")?;
                    session.set_next_token(token)?;
                    Some(token)
                } else { None };
                for (&slot, &token) in slots.iter().zip(tokens) {
                    self.episodes[slot].as_mut().expect("validated episode").consumed.push(token);
                }
                Ok(Outcome::Decoded { foreground, logits })
            }
            Work::Score { slot, continuation, student_version } => {
                let episode = self.episodes[*slot].as_ref().expect("validated episode");
                let prompt = &episode.example.teacher;
                let inputs = response_inputs(prompt, continuation)?;
                let version = self.teacher_version();
                let layer = session.layer_range().end - 1;
                let table = self.bank.as_ref().expect("validated bank").teacher_table_clone();
                let logits = session.with_sequence(&mut self.teacher, |session| {
                    session.reset();
                    session.set_teacher_bank(layer, version, table)?;
                    // Explicit-bank host packing keeps teacher prompts dense
                    // without ever resolving student weights from the source.
                    // The session applies its admitted prefill chunk budget.
                    if prompt.len() > 1 {
                        session.extend_logits(&prompt[..prompt.len() - 1])?;
                    }
                    session.observe_logits(&inputs)
                })?;
                let distribution = burn::tensor::activation::softmax(logits.cast(burn::tensor::DType::F32), 1);
                validate_soft_targets(&distribution, 0, inputs.len(), session.config().text_config.effective_vocab(), 1.0)?;
                ensure!(session.learning_step() == Some(self.version), "teacher scoring unexpectedly learned");
                self.targets[*slot] = Some(Targets {
                    distribution, continuation: continuation.clone(), rollout_version: *student_version,
                    teacher_version: version,
                });
                self.episodes[*slot].as_mut().expect("validated episode").scored_teacher_version = Some(version);
                Ok(Outcome::Scored { rows: inputs.len(), teacher_version: version })
            }
            Work::Learn { slot, .. } => {
                self.learning = true;
                let episode = self.episodes[*slot].as_ref().expect("validated episode");
                let target = self.targets[*slot].as_ref().expect("validated targets");
                let prompt = &episode.example.student;
                let t = &session.config().text_config;
                let history = if t.use_sconv { t.sconv_kernel_size.saturating_sub(1) } else { 0 };
                let (prefix, inputs, start) = super::sdft::training_inputs(prompt, &target.continuation, history)?;
                let expected = self.version.checked_add(1).context("SDFT optimizer version overflow")?;
                let report = session.with_sequence(&mut self.students[*slot], |session| {
                    session.reset();
                    if prefix > 0 { session.extend_logits(&prompt[..prefix])?; }
                    session.learn_distribution(&inputs, &target.distribution, start, 1.0)
                })?;
                ensure!(session.learning_step() == Some(expected), "SDFT learning did not advance exactly one optimizer step");
                self.version = expected;
                self.episodes[*slot].as_mut().expect("validated episode").learned = true;
                // Targets are kept until the EMA boundary as explicit evidence
                // of the scored cohort; no slot can accidentally learn twice.
                Ok(Outcome::Learned(report))
            }
            Work::UpdateTeacher { teacher_version, .. } => {
                self.teacher.reset();
                let report = session.advance_teacher_bank(self.bank.as_mut().expect("validated bank"), *teacher_version)?;
                ensure!(report.student_step == self.version && !report.poisoned, "EMA update reports a different student version");
                for sequence in &mut self.students { sequence.reset(); }
                for episode in &mut self.episodes { *episode = None; }
                for targets in &mut self.targets { *targets = None; }
                self.rollout_version = None;
                self.learning = false;
                Ok(Outcome::TeacherUpdated(report))
            }
        }
    }

    fn slot(&self, slot: usize) -> Result<()> {
        ensure!(slot < self.students.len(), "SDFT slot {slot} exceeds {} admitted students", self.students.len());
        Ok(())
    }

    fn episode(&self, slot: usize) -> Result<&Episode> {
        self.slot(slot)?;
        self.episodes[slot].as_ref().with_context(|| format!("SDFT slot {slot} has not been prepared"))
    }
}

fn validate_decode_slots(slots: &[usize], tokens: usize, admitted: usize) -> Result<()> {
    ensure!(slots.len() == tokens, "SDFT decode needs one input token per selected slot");
    for (i, &slot) in slots.iter().enumerate() {
        ensure!(slot < admitted, "SDFT decode slot {slot} is not admitted");
        ensure!(!slots[..i].contains(&slot), "SDFT decode repeats slot {slot}");
    }
    Ok(())
}

fn validate_continuation(consumed: &[usize], continuation: &[usize], limit: usize, vocab: usize) -> Result<()> {
    ensure!(!continuation.is_empty() && continuation.len() <= limit, "SDFT continuation is empty or exceeds its rollout budget");
    ensure!(continuation.iter().all(|&token| token < vocab), "SDFT continuation contains an invalid token");
    // The last sampled token may not have been consumed by another decode
    // because its successor distribution was unnecessary. Both cases are valid.
    ensure!(consumed.len() <= continuation.len() && continuation.len() - consumed.len() <= 1,
        "teacher continuation is not the student-generated decode prefix");
    ensure!(continuation.starts_with(consumed), "teacher continuation differs from the student's consumed tokens");
    Ok(())
}

fn ready_for_learning(episodes: &[Option<Episode>]) -> bool {
    episodes.iter().any(Option::is_some)
        && episodes.iter().flatten().all(|episode| episode.scored_teacher_version.is_some())
}

fn finished_learning(episodes: &[Option<Episode>]) -> bool {
    ready_for_learning(episodes) && episodes.iter().flatten().all(|episode| episode.learned)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn episode(scored: bool, learned: bool) -> Option<Episode> {
        Some(Episode {
            example: PreparedExample { student: vec![1], teacher: vec![2, 1] },
            consumed: vec![3], rollout_version: 7,
            scored_teacher_version: scored.then_some(2), learned,
        })
    }

    #[test]
    fn decode_order_is_explicit_and_no_capsule_can_alias_a_second_row() {
        assert!(validate_decode_slots(&[2, 0], 2, 3).is_ok());
        assert!(validate_decode_slots(&[1, 1], 2, 3).is_err());
        assert!(validate_decode_slots(&[3], 1, 3).is_err());
        assert!(validate_decode_slots(&[1], 0, 3).is_err());
    }

    #[test]
    fn teacher_scores_only_the_consumed_student_prefix_plus_final_sample() {
        assert!(validate_continuation(&[], &[3], 3, 10).is_ok());
        assert!(validate_continuation(&[3], &[3, 4], 3, 10).is_ok());
        assert!(validate_continuation(&[3, 4], &[3, 4], 3, 10).is_ok());
        assert!(validate_continuation(&[], &[3, 4], 3, 10).is_err());
        assert!(validate_continuation(&[3], &[4, 5], 3, 10).is_err());
        assert!(validate_continuation(&[3, 4], &[3], 3, 10).is_err());
        assert!(validate_continuation(&[], &[], 3, 10).is_err());
        assert!(validate_continuation(&[3], &[3, 10], 3, 10).is_err());
    }

    #[test]
    fn all_prepared_slots_score_before_any_learning_and_learn_before_ema() {
        assert!(!ready_for_learning(&[None, None]));
        assert!(!ready_for_learning(&[episode(true, false), episode(false, false)]));
        assert!(ready_for_learning(&[episode(true, false), None, episode(true, false)]));
        assert!(!finished_learning(&[episode(true, true), episode(true, false)]));
        assert!(finished_learning(&[episode(true, true), None, episode(true, true)]));
    }
}
