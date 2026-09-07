//! Background self-distillation, separate from the live conversation.
//!
//! Student continuations are sampled without a demonstration. An EMA teacher
//! sees that demonstration and scores the same continuation. Only response
//! prediction rows carry loss; neither side's prompt is a training target.

use anyhow::{Result, ensure};

/// Explicit opt-in. Ordinary resident inference does not allocate a teacher
/// or background contexts. Limits are priced before loading model weights.
#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct Config {
    pub sequences: usize,
    pub context_budget: usize,
    pub max_rollout: usize,
    pub temperature: f32,
    pub learning_rate: f32,
    pub ema_decay: f32,
    pub seed: u64,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            sequences: 1,
            context_budget: 4096,
            max_rollout: 16,
            temperature: 1.0,
            learning_rate: 0.1,
            ema_decay: 0.99,
            seed: 0,
        }
    }
}

impl Config {
    pub fn validate(&self) -> Result<()> {
        ensure!((1..64).contains(&self.sequences), "SDFT needs 1..=63 background students (plus the live row)");
        ensure!(self.max_rollout > 0, "SDFT rollout length must be positive");
        ensure!(self.max_rollout <= 64, "SDFT rollout exceeds the 64-row observation/head budget");
        ensure!(
            self.context_budget > self.max_rollout,
            "SDFT context budget must leave room for a prompt and the rollout"
        );
        ensure!(
            self.temperature.is_finite() && self.temperature > 0.0,
            "SDFT uses categorical student rollouts: temperature must be finite and positive"
        );
        ensure!(
            self.learning_rate.is_finite() && self.learning_rate > 0.0,
            "SDFT learning rate must be finite and positive"
        );
        ensure!(
            self.ema_decay.is_finite() && (0.0..1.0).contains(&self.ema_decay),
            "SDFT EMA decay must be finite and in [0, 1)"
        );
        self.sequences.checked_add(1).ok_or_else(|| anyhow::anyhow!("SDFT sequence count overflow"))?;
        Ok(())
    }

    /// One context per student and one reusable teacher context. Backward
    /// reuses a student's capsule after its rollout is complete.
    pub fn sequence_budgets(&self) -> Result<Vec<usize>> {
        self.validate()?;
        Ok(vec![self.context_budget; self.sequences + 1])
    }
}

/// One explicitly supplied learning example. The demonstration is privileged
/// teacher information, not an event inserted into the foreground history.
#[derive(Clone, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Example {
    pub prompt: String,
    pub demonstration: String,
}

impl Example {
    pub fn validate(&self) -> Result<()> {
        ensure!(!self.prompt.trim().is_empty(), "SDFT example has an empty prompt");
        ensure!(
            !self.demonstration.trim().is_empty(),
            "SDFT example has no teacher demonstration"
        );
        Ok(())
    }
}

/// Exact model inputs after content-only tokenization. Construction checks
/// both contexts before either rank changes state.
#[derive(Clone, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct PreparedExample {
    pub student: Vec<usize>,
    pub teacher: Vec<usize>,
}

/// One ordered operation on every tensor-parallel rank. Sampling decisions
/// happen only on rank zero; `Decode` carries the chosen input tokens.
#[derive(Clone, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(tag = "op", rename_all = "snake_case", deny_unknown_fields)]
pub enum Work {
    Prepare { slot: usize, example: PreparedExample },
    Decode { active: Option<usize>, slots: Vec<usize>, tokens: Vec<usize> },
    Score { slot: usize, continuation: Vec<usize>, student_version: u64 },
    Learn { slot: usize, student_version: u64, teacher_version: u64 },
    UpdateTeacher { student_version: u64, teacher_version: u64 },
}

impl PreparedExample {
    pub fn validate(&self, config: &Config, vocab: usize) -> Result<()> {
        config.validate()?;
        for (name, ids) in [("student", &self.student), ("teacher", &self.teacher)] {
            ensure!(!ids.is_empty(), "SDFT {name} prompt is empty");
            let end = ids.len().checked_add(config.max_rollout)
                .ok_or_else(|| anyhow::anyhow!("SDFT {name} context length overflow"))?;
            ensure!(end <= config.context_budget,
                "SDFT {name} prompt plus rollout needs {end} positions, budget is {}",
                config.context_budget);
            ensure!(ids.iter().all(|&id| id < vocab), "SDFT {name} prompt contains an invalid token");
        }
        Ok(())
    }
}

/// Shift a sampled continuation into its response-prediction inputs.
///
/// Re-prefill `prompt[..prompt.len()-1]`, then train these rows against the
/// teacher's distributions. Row zero predicts response token zero; no prompt
/// token becomes a label. The final sampled token has no successor to score.
pub fn response_inputs(prompt: &[usize], continuation: &[usize]) -> Result<Vec<usize>> {
    let &last = prompt.last().ok_or_else(|| anyhow::anyhow!("empty SDFT prompt"))?;
    ensure!(!continuation.is_empty(), "empty SDFT continuation");
    let mut inputs = Vec::with_capacity(continuation.len());
    inputs.push(last);
    inputs.extend_from_slice(&continuation[..continuation.len() - 1]);
    Ok(inputs)
}

/// A response loss can reach into the final MLP convolution's prompt history.
/// Capture that short tail too, without giving any prompt row a head target.
/// Earlier layers are frozen, so no other temporal backward history is needed.
pub fn training_inputs(
    prompt: &[usize], continuation: &[usize], history: usize,
) -> Result<(usize, Vec<usize>, usize)> {
    let response = response_inputs(prompt, continuation)?;
    let start = history.min(prompt.len() - 1);
    let prefix = prompt.len() - 1 - start;
    let mut inputs = Vec::with_capacity(start + response.len());
    inputs.extend_from_slice(&prompt[prefix..prompt.len() - 1]);
    inputs.extend(response);
    Ok((prefix, inputs, start))
}

/// Evidence for one optimizer update. Host time includes any required
/// readbacks, but is not a synchronized GPU benchmark of the whole episode.
#[derive(Clone, Debug, Default, serde::Serialize, serde::Deserialize)]
pub struct Report {
    pub student_version: u64,
    pub teacher_version: u64,
    pub updated_student_version: u64,
    pub response_tokens: usize,
    pub update_host_seconds: f64,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn config_prices_a_teacher_separately_from_student_slots() {
        let cfg = Config { sequences: 3, ..Config::default() };
        assert_eq!(cfg.sequence_budgets().unwrap(), vec![4096; 4]);
    }

    #[test]
    fn invalid_learning_configuration_is_rejected_before_allocation() {
        for cfg in [
            Config { sequences: 0, ..Config::default() },
            Config { sequences: 64, ..Config::default() },
            Config { max_rollout: 0, ..Config::default() },
            Config { temperature: 0.0, ..Config::default() },
            Config { temperature: f32::NAN, ..Config::default() },
            Config { learning_rate: f32::INFINITY, ..Config::default() },
            Config { ema_decay: 1.0, ..Config::default() },
            Config { context_budget: 16, ..Config::default() },
        ] {
            assert!(cfg.validate().is_err(), "accepted {cfg:?}");
        }
    }

    #[test]
    fn longer_teacher_context_is_admitted_independently() {
        let cfg = Config { context_budget: 20, max_rollout: 4, ..Config::default() };
        let mut example = PreparedExample { student: vec![1; 3], teacher: vec![2; 16] };
        example.validate(&cfg, 10).unwrap();
        example.teacher.push(2);
        assert!(example.validate(&cfg, 10).is_err());
        example.teacher = vec![10];
        assert!(example.validate(&cfg, 10).is_err());
    }

    #[test]
    fn response_alignment_never_trains_on_a_prompt_token() {
        assert_eq!(response_inputs(&[1, 2, 3], &[7, 8, 9]).unwrap(), [3, 7, 8]);
        assert_eq!(response_inputs(&[1], &[7]).unwrap(), [1]);
        assert!(response_inputs(&[], &[7]).is_err());
        assert!(response_inputs(&[1], &[]).is_err());
    }

    #[test]
    fn training_captures_convolution_antecedents_without_prompt_labels() {
        let (prefix, inputs, start) = training_inputs(&[1, 2, 3, 4, 5], &[7, 8, 9], 2).unwrap();
        assert_eq!(prefix, 2);
        assert_eq!(inputs, [3, 4, 5, 7, 8]);
        assert_eq!(start, 2);
        assert_eq!(&inputs[start..], response_inputs(&[1, 2, 3, 4, 5], &[7, 8, 9]).unwrap());
        assert_eq!(training_inputs(&[1], &[7], 3).unwrap(), (0, vec![1], 0));
        assert_eq!(training_inputs(&[1, 2], &[7], 3).unwrap(), (0, vec![1, 2], 1));
        assert_eq!(training_inputs(&[1, 2], &[7], 0).unwrap(), (1, vec![2], 0));
    }
}
