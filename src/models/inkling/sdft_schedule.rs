//! CPU-only planning for one finite cohort of background SDFT rollouts.
//!
//! Planning is read-only. The owner executes a `Work` on every rank, samples
//! only on rank zero, then acknowledges SUCCESS. An execution error must not
//! call the acknowledgment or silently advance this ledger. Idle work prepares
//! contexts, finishes rollouts, scores every response, learns each response in
//! turn, then updates the teacher. The foreground path only joins ready decode
//! rows; it never prepares a prompt or starts a weight update.

use std::collections::VecDeque;

use anyhow::{Result, ensure};

use super::sdft::{Config, PreparedExample, Work};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Stage {
    Rollout,
    Scored,
    Learned,
}

#[derive(Debug)]
struct Slot {
    continuation: Vec<usize>,
    stage: Stage,
}

#[derive(Clone, Copy, Debug)]
struct Epoch {
    /// The student that sampled EVERY continuation in this cohort.
    rollout_student: u64,
    /// The exact current student, advanced by successful serial learning.
    current_student: u64,
    teacher: u64,
}

/// One bounded cohort, plus examples waiting for a later free slot/cohort.
///
/// Completed slots remain occupied until the teacher update succeeds. New
/// examples can fill unused slots until scoring begins, but can never refill
/// a completed slot indefinitely. All teacher targets precede all updates;
/// sequential learning keeps the original rollout epoch on its `Work` items.
pub struct Scheduler {
    config: Config,
    stop_ids: Vec<usize>,
    queue: VecDeque<PreparedExample>,
    slots: Vec<Option<Slot>>,
    epoch: Option<Epoch>,
}

impl Scheduler {
    pub fn new(config: Config, mut stop_ids: Vec<usize>) -> Result<Self> {
        config.validate()?;
        stop_ids.sort_unstable();
        stop_ids.dedup();
        Ok(Self {
            slots: (0..config.sequences).map(|_| None).collect(),
            config,
            stop_ids,
            queue: VecDeque::new(),
            epoch: None,
        })
    }

    /// Queue already-tokenized model input. Vocabulary validation belongs to
    /// `PreparedExample::validate` at admission, where the model is available;
    /// this scheduler additionally checks its own context/rollout bounds.
    pub fn enqueue(&mut self, example: PreparedExample) -> Result<()> {
        for (name, ids) in [("student", &example.student), ("teacher", &example.teacher)] {
            ensure!(!ids.is_empty(), "SDFT {name} prompt is empty");
            let end = ids.len().checked_add(self.config.max_rollout)
                .ok_or_else(|| anyhow::anyhow!("SDFT {name} context length overflow"))?;
            ensure!(end <= self.config.context_budget,
                "SDFT {name} prompt plus rollout needs {end} positions, budget is {}",
                self.config.context_budget);
        }
        self.queue.push_back(example);
        Ok(())
    }

    /// Foreground latency-sensitive path: only already-prepared, unfinished
    /// student rows can join this token. `None` means ordinary live decode.
    pub fn next_decode(
        &self,
        active: usize,
        student_version: u64,
        teacher_version: u64,
    ) -> Result<Option<Work>> {
        self.check_versions(student_version, teacher_version)?;
        Ok(self.decode_work(Some(active)))
    }

    /// Explicit idle step. Repeated calls return the same work until an
    /// acknowledgment or enqueue changes state; merely inspecting work never
    /// consumes an example, extends a response, or advances a training stage.
    pub fn next_idle(
        &self,
        student_version: u64,
        teacher_version: u64,
    ) -> Result<Option<Work>> {
        self.check_versions(student_version, teacher_version)?;
        if self.accepting() {
            if let (Some(slot), Some(example)) = (self.free_slot(), self.queue.front()) {
                return Ok(Some(Work::Prepare { slot, example: example.clone() }));
            }
        }
        if let Some(work) = self.decode_work(None) {
            return Ok(Some(work));
        }
        let Some(epoch) = self.epoch else { return Ok(None) };
        if let Some(slot) = self.first_at(Stage::Rollout) {
            return Ok(Some(Work::Score {
                slot,
                continuation: self.slots[slot].as_ref().unwrap().continuation.clone(),
                student_version: epoch.rollout_student,
            }));
        }
        if let Some(slot) = self.first_at(Stage::Scored) {
            return Ok(Some(Work::Learn {
                slot,
                student_version: epoch.rollout_student,
                teacher_version: epoch.teacher,
            }));
        }
        Ok(Some(Work::UpdateTeacher {
            student_version: epoch.current_student,
            teacher_version: epoch.teacher,
        }))
    }

    /// A successful prompt prefill has sampled response token ZERO. Even a
    /// stop token belongs to the continuation; it must receive a teacher target.
    pub fn prepared(
        &mut self,
        slot: usize,
        first_token: usize,
        student_version: u64,
        teacher_version: u64,
    ) -> Result<()> {
        self.check_versions(student_version, teacher_version)?;
        ensure!(self.accepting(), "SDFT cohort is already scoring or learning");
        ensure!(self.free_slot() == Some(slot), "SDFT prepare acknowledgment has wrong slot {slot}");
        ensure!(!self.queue.is_empty(), "SDFT prepare acknowledgment has no queued example");
        let _ = self.queue.pop_front();
        self.slots[slot] = Some(Slot { continuation: vec![first_token], stage: Stage::Rollout });
        self.epoch.get_or_insert(Epoch {
            rollout_student: student_version,
            current_student: student_version,
            teacher: teacher_version,
        });
        Ok(())
    }

    /// Acknowledge one sampled successor for each supplied slot, in the exact
    /// returned order. All arguments are checked before ANY response changes.
    /// The supplied slots may be a subset, but cannot repeat or be finished.
    pub fn decoded(&mut self, slots: &[usize], next_tokens: &[usize]) -> Result<()> {
        ensure!(!slots.is_empty(), "SDFT decode acknowledgment has no student slots");
        ensure!(slots.len() == next_tokens.len(), "SDFT decode acknowledgment length mismatch");
        for (position, &id) in slots.iter().enumerate() {
            ensure!(!slots[..position].contains(&id), "SDFT decode repeats slot {id}");
            let slot = self.slots.get(id).and_then(Option::as_ref)
                .ok_or_else(|| anyhow::anyhow!("SDFT decode has no prepared slot {id}"))?;
            ensure!(slot.stage == Stage::Rollout && !self.finished(slot),
                "SDFT decode cannot advance finished slot {id}");
        }
        for (&id, &token) in slots.iter().zip(next_tokens) {
            self.slots[id].as_mut().unwrap().continuation.push(token);
        }
        Ok(())
    }

    /// First successful scoring seals the cohort. Queued examples then wait
    /// for the next cohort, even if this one had unused sequence slots.
    pub fn scored(&mut self, slot: usize) -> Result<()> {
        ensure!(self.slots.iter().flatten().all(|slot| self.finished(slot)),
            "SDFT cannot score until every cohort rollout finishes");
        ensure!(self.first_at(Stage::Rollout) == Some(slot),
            "SDFT score acknowledgment has wrong slot {slot}");
        self.slots[slot].as_mut().unwrap().stage = Stage::Scored;
        Ok(())
    }

    /// Commit a successful serial update, retaining the old rollout label but
    /// tracking the exact new live student version for subsequent planning.
    pub fn learned(&mut self, slot: usize, updated_student_version: u64) -> Result<()> {
        ensure!(self.first_at(Stage::Rollout).is_none(),
            "SDFT cannot learn before every teacher target has been scored");
        ensure!(self.first_at(Stage::Scored) == Some(slot),
            "SDFT learn acknowledgment has wrong slot {slot}");
        let epoch = self.epoch.as_ref().unwrap();
        ensure!(updated_student_version > epoch.current_student,
            "SDFT learn did not advance student version {} to {updated_student_version}",
            epoch.current_student);
        self.slots[slot].as_mut().unwrap().stage = Stage::Learned;
        self.epoch.as_mut().unwrap().current_student = updated_student_version;
        Ok(())
    }

    /// Release a completed cohort ONLY after the runtime confirms EMA success.
    pub fn teacher_updated(&mut self) -> Result<()> {
        ensure!(self.epoch.is_some(), "SDFT teacher update has no cohort");
        ensure!(self.slots.iter().flatten().all(|slot| slot.stage == Stage::Learned),
            "SDFT teacher update precedes completed student learning");
        for slot in &mut self.slots {
            *slot = None;
        }
        self.epoch = None;
        Ok(())
    }

    pub fn pending(&self) -> bool {
        !self.queue.is_empty() || self.epoch.is_some()
    }

    fn check_versions(&self, student: u64, teacher: u64) -> Result<()> {
        if let Some(epoch) = self.epoch {
            ensure!(student == epoch.current_student,
                "SDFT student version changed outside acknowledged learning: expected {}, got {student}",
                epoch.current_student);
            ensure!(teacher == epoch.teacher,
                "SDFT teacher version changed during cohort: expected {}, got {teacher}", epoch.teacher);
        }
        Ok(())
    }

    fn accepting(&self) -> bool {
        self.slots.iter().flatten().all(|slot| slot.stage == Stage::Rollout)
    }

    fn free_slot(&self) -> Option<usize> {
        self.slots.iter().position(Option::is_none)
    }

    fn first_at(&self, stage: Stage) -> Option<usize> {
        self.slots.iter().position(|slot| slot.as_ref().is_some_and(|slot| slot.stage == stage))
    }

    fn finished(&self, slot: &Slot) -> bool {
        slot.continuation.len() >= self.config.max_rollout
            || slot.continuation.last().is_some_and(|id| self.stop_ids.binary_search(id).is_ok())
    }

    fn decode_work(&self, active: Option<usize>) -> Option<Work> {
        let mut slots = Vec::new();
        let mut tokens = Vec::new();
        for (id, slot) in self.slots.iter().enumerate() {
            if let Some(slot) = slot {
                if slot.stage == Stage::Rollout && !self.finished(slot) {
                    slots.push(id);
                    tokens.push(*slot.continuation.last().unwrap());
                }
            }
        }
        (!slots.is_empty()).then_some(Work::Decode { active, slots, tokens })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn config(sequences: usize, max_rollout: usize) -> Config {
        Config { sequences, max_rollout, context_budget: 16, ..Config::default() }
    }

    fn example(token: usize) -> PreparedExample {
        PreparedExample { student: vec![10, token], teacher: vec![20, 30, token] }
    }

    fn prepare(scheduler: &mut Scheduler, first_token: usize, student: u64, teacher: u64) -> usize {
        let Work::Prepare { slot, .. } = scheduler.next_idle(student, teacher).unwrap().unwrap()
            else { panic!("expected a prepare") };
        scheduler.prepared(slot, first_token, student, teacher).unwrap();
        slot
    }

    #[test]
    fn planners_are_read_only_and_never_consume_work() {
        let mut scheduler = Scheduler::new(config(1, 3), vec![99]).unwrap();
        assert!(!scheduler.pending());
        scheduler.enqueue(example(1)).unwrap();
        let expected = Some(Work::Prepare { slot: 0, example: example(1) });
        for _ in 0..3 {
            assert_eq!(scheduler.next_idle(7, 2).unwrap(), expected);
            assert_eq!(scheduler.next_decode(8, 7, 2).unwrap(), None);
            assert!(scheduler.pending());
        }
        scheduler.prepared(0, 40, 7, 2).unwrap();
        let expected = Some(Work::Decode { active: None, slots: vec![0], tokens: vec![40] });
        for _ in 0..3 {
            assert_eq!(scheduler.next_idle(7, 2).unwrap(), expected);
        }
    }

    #[test]
    fn response_alignment_includes_first_sample_and_stop_token() {
        let mut scheduler = Scheduler::new(config(1, 5), vec![99]).unwrap();
        scheduler.enqueue(example(1)).unwrap();
        prepare(&mut scheduler, 70, 4, 2);
        assert_eq!(scheduler.next_decode(8, 4, 2).unwrap(), Some(Work::Decode {
            active: Some(8), slots: vec![0], tokens: vec![70],
        }));
        scheduler.decoded(&[0], &[80]).unwrap();
        assert_eq!(scheduler.next_idle(4, 2).unwrap(), Some(Work::Decode {
            active: None, slots: vec![0], tokens: vec![80],
        }));
        scheduler.decoded(&[0], &[99]).unwrap();
        assert_eq!(scheduler.next_decode(8, 4, 2).unwrap(), None);
        assert_eq!(scheduler.next_idle(4, 2).unwrap(), Some(Work::Score {
            slot: 0, continuation: vec![70, 80, 99], student_version: 4,
        }));
        assert_eq!(super::super::sdft::response_inputs(&example(1).student, &[70, 80, 99])
            .unwrap(), [1, 70, 80]);
    }

    #[test]
    fn different_stop_lengths_keep_rows_independent() {
        let mut scheduler = Scheduler::new(config(3, 4), vec![99]).unwrap();
        for token in 1..=3 { scheduler.enqueue(example(token)).unwrap(); }
        for token in [11, 21, 31] { prepare(&mut scheduler, token, 4, 2); }
        scheduler.decoded(&[0, 1, 2], &[99, 22, 32]).unwrap();
        assert_eq!(scheduler.next_idle(4, 2).unwrap(), Some(Work::Decode {
            active: None, slots: vec![1, 2], tokens: vec![22, 32],
        }));
        // A returned subset/order is explicit; results never shift into a
        // different conversation when another slot finishes.
        scheduler.decoded(&[2, 1], &[33, 99]).unwrap();
        assert_eq!(scheduler.next_idle(4, 2).unwrap(), Some(Work::Decode {
            active: None, slots: vec![2], tokens: vec![33],
        }));
        scheduler.decoded(&[2], &[34]).unwrap();
        for (slot, continuation) in [vec![11, 99], vec![21, 22, 99], vec![31, 32, 33, 34]]
            .into_iter().enumerate()
        {
            assert_eq!(scheduler.next_idle(4, 2).unwrap(), Some(Work::Score {
                slot, continuation, student_version: 4,
            }));
            scheduler.scored(slot).unwrap();
        }
    }

    #[test]
    fn all_scores_precede_learning_and_rollout_epoch_is_not_relabelled() {
        let mut scheduler = Scheduler::new(config(2, 1), vec![99]).unwrap();
        scheduler.enqueue(example(1)).unwrap();
        scheduler.enqueue(example(2)).unwrap();
        prepare(&mut scheduler, 41, 10, 3);
        prepare(&mut scheduler, 42, 10, 3);
        scheduler.scored(0).unwrap();
        assert!(scheduler.learned(0, 11).is_err());
        assert_eq!(scheduler.next_idle(10, 3).unwrap(), Some(Work::Score {
            slot: 1, continuation: vec![42], student_version: 10,
        }));
        scheduler.scored(1).unwrap();
        for (slot, current_student) in [(0, 10), (1, 11)] {
            assert_eq!(scheduler.next_idle(current_student, 3).unwrap(), Some(Work::Learn {
                slot, student_version: 10, teacher_version: 3,
            }));
            assert!(scheduler.teacher_updated().is_err());
            scheduler.learned(slot, current_student + 1).unwrap();
        }
        assert_eq!(scheduler.next_idle(12, 3).unwrap(), Some(Work::UpdateTeacher {
            student_version: 12, teacher_version: 3,
        }));
        assert!(scheduler.pending());
        scheduler.teacher_updated().unwrap();
        assert!(!scheduler.pending());
        assert_eq!(scheduler.next_idle(12, 4).unwrap(), None);
    }

    #[test]
    fn unexpected_versions_and_nonadvancing_updates_are_rejected() {
        let mut scheduler = Scheduler::new(config(1, 1), vec![]).unwrap();
        scheduler.enqueue(example(1)).unwrap();
        prepare(&mut scheduler, 41, 10, 3);
        assert!(scheduler.next_idle(11, 3).is_err());
        assert!(scheduler.next_idle(10, 4).is_err());
        assert!(scheduler.next_decode(9, 11, 3).is_err());
        scheduler.scored(0).unwrap();
        let expected = scheduler.next_idle(10, 3).unwrap();
        assert!(scheduler.learned(0, 10).is_err());
        assert!(scheduler.learned(0, 9).is_err());
        assert_eq!(scheduler.next_idle(10, 3).unwrap(), expected);
        scheduler.learned(0, 11).unwrap();
        assert!(scheduler.next_idle(10, 3).is_err());
        assert!(scheduler.next_idle(12, 3).is_err());
        assert!(scheduler.next_idle(11, 4).is_err());
    }

    #[test]
    fn invalid_acknowledgments_do_not_partially_mutate_state() {
        let mut scheduler = Scheduler::new(config(2, 3), vec![99]).unwrap();
        scheduler.enqueue(example(1)).unwrap();
        scheduler.enqueue(example(2)).unwrap();
        let expected = scheduler.next_idle(10, 3).unwrap();
        assert!(scheduler.prepared(1, 41, 10, 3).is_err());
        assert_eq!(scheduler.next_idle(10, 3).unwrap(), expected);
        prepare(&mut scheduler, 41, 10, 3);
        prepare(&mut scheduler, 42, 10, 3);
        let expected = scheduler.next_idle(10, 3).unwrap();
        assert!(scheduler.decoded(&[0, 5], &[50, 60]).is_err());
        assert!(scheduler.decoded(&[0, 0], &[50, 60]).is_err());
        assert!(scheduler.decoded(&[0, 1], &[50]).is_err());
        assert!(scheduler.decoded(&[], &[]).is_err());
        assert!(scheduler.scored(0).is_err());
        assert!(scheduler.learned(0, 11).is_err());
        assert!(scheduler.teacher_updated().is_err());
        assert_eq!(scheduler.next_idle(10, 3).unwrap(), expected);
        scheduler.decoded(&[0], &[99]).unwrap();
        let expected = scheduler.next_idle(10, 3).unwrap();
        assert!(scheduler.decoded(&[1, 0], &[50, 60]).is_err());
        assert_eq!(scheduler.next_idle(10, 3).unwrap(), expected);
    }

    #[test]
    fn foreground_path_never_prepares_scores_or_updates() {
        let mut scheduler = Scheduler::new(config(2, 2), vec![99]).unwrap();
        scheduler.enqueue(example(1)).unwrap();
        assert_eq!(scheduler.next_decode(8, 10, 3).unwrap(), None);
        prepare(&mut scheduler, 41, 10, 3);
        scheduler.enqueue(example(2)).unwrap();
        assert!(matches!(scheduler.next_idle(10, 3).unwrap(), Some(Work::Prepare { slot: 1, .. })));
        assert_eq!(scheduler.next_decode(8, 10, 3).unwrap(), Some(Work::Decode {
            active: Some(8), slots: vec![0], tokens: vec![41],
        }));
        scheduler.decoded(&[0], &[99]).unwrap();
        assert_eq!(scheduler.next_decode(8, 10, 3).unwrap(), None);
        prepare(&mut scheduler, 99, 10, 3);
        for slot in 0..2 {
            assert_eq!(scheduler.next_decode(8, 10, 3).unwrap(), None);
            scheduler.scored(slot).unwrap();
        }
        for slot in 0..2 {
            assert_eq!(scheduler.next_decode(8, 10 + slot as u64, 3).unwrap(), None);
            scheduler.learned(slot, 11 + slot as u64).unwrap();
        }
        assert_eq!(scheduler.next_decode(8, 12, 3).unwrap(), None);
    }

    #[test]
    fn slots_are_not_refilled_until_the_finite_cohort_finishes() {
        let mut scheduler = Scheduler::new(config(2, 1), vec![]).unwrap();
        for token in 1..=3 { scheduler.enqueue(example(token)).unwrap(); }
        prepare(&mut scheduler, 41, 10, 3);
        prepare(&mut scheduler, 42, 10, 3);
        for slot in 0..2 {
            assert!(matches!(scheduler.next_idle(10, 3).unwrap(), Some(Work::Score { .. })));
            scheduler.scored(slot).unwrap();
        }
        for slot in 0..2 { scheduler.learned(slot, 11 + slot as u64).unwrap(); }
        scheduler.teacher_updated().unwrap();
        assert!(scheduler.pending());
        assert_eq!(scheduler.next_idle(12, 4).unwrap(), Some(Work::Prepare {
            slot: 0, example: example(3),
        }));
        scheduler.prepared(0, 43, 12, 4).unwrap();
        assert_eq!(scheduler.next_idle(12, 4).unwrap(), Some(Work::Score {
            slot: 0, continuation: vec![43], student_version: 12,
        }));
    }

    #[test]
    fn scoring_seals_a_partially_filled_cohort() {
        let mut scheduler = Scheduler::new(config(2, 1), vec![]).unwrap();
        scheduler.enqueue(example(1)).unwrap();
        prepare(&mut scheduler, 41, 10, 3);
        scheduler.scored(0).unwrap();
        scheduler.enqueue(example(2)).unwrap();
        assert!(scheduler.prepared(1, 42, 10, 3).is_err());
        assert_eq!(scheduler.next_idle(10, 3).unwrap(), Some(Work::Learn {
            slot: 0, student_version: 10, teacher_version: 3,
        }));
    }

    #[test]
    fn first_sample_can_finish_a_rollout_without_a_decode() {
        let mut scheduler = Scheduler::new(config(1, 4), vec![99, 99]).unwrap();
        scheduler.enqueue(example(1)).unwrap();
        prepare(&mut scheduler, 99, 10, 3);
        assert_eq!(scheduler.next_decode(8, 10, 3).unwrap(), None);
        assert_eq!(scheduler.next_idle(10, 3).unwrap(), Some(Work::Score {
            slot: 0, continuation: vec![99], student_version: 10,
        }));
    }

    #[test]
    fn context_checks_happen_before_an_example_is_queued() {
        assert!(Scheduler::new(config(0, 3), vec![]).is_err());
        let mut scheduler = Scheduler::new(config(1, 4), vec![]).unwrap();
        for example in [
            PreparedExample { student: vec![], teacher: vec![1] },
            PreparedExample { student: vec![1], teacher: vec![] },
            PreparedExample { student: vec![1; 13], teacher: vec![2] },
            PreparedExample { student: vec![1], teacher: vec![2; 13] },
        ] {
            assert!(scheduler.enqueue(example).is_err());
            assert!(!scheduler.pending());
        }
        assert!(scheduler.prepared(0, 41, 10, 3).is_err());
        assert!(!scheduler.pending());
    }
}
