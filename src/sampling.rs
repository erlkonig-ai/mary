//! Seeded categorical sampling shared by resident model implementations.
//!
//! Each conversation owns its RNG. A full-distribution draw (the default for
//! self-distillation) takes O(vocabulary) time and reusable O(vocabulary)
//! scratch, without sorting or allocating a probability distribution per token.
//! Top-k uses partial selection; only nucleus sampling sorts its candidates.
//!
//! Non-finite logits and explicitly forbidden tokens have zero probability,
//! including in greedy mode. An empty or entirely excluded row is an error,
//! not a reason to invent a token. PersonaPlex's older sampler intentionally
//! retains its different greedy defaults and invalid-row fallback.

use anyhow::{Result, ensure};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

/// Temperature, then top-k, then top-p, then a categorical draw.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SamplingConfig {
    /// Zero selects greedy decoding; otherwise finite and positive.
    pub temperature: f32,
    /// Keep this many highest logits. Zero, or at least the eligible token
    /// count, disables the cut. Equal logits prefer the lower token id.
    pub top_k: usize,
    /// Keep the smallest prefix reaching this fraction of probability mass
    /// AFTER the top-k cut. Must be in (0, 1]; one disables the cut.
    pub top_p: f32,
}

impl SamplingConfig {
    pub const fn greedy() -> Self {
        Self {
            temperature: 0.0,
            top_k: 0,
            top_p: 1.0,
        }
    }

    pub fn is_greedy(&self) -> bool {
        self.temperature == 0.0
    }

    pub fn validate(&self) -> Result<()> {
        ensure!(
            self.temperature.is_finite() && self.temperature >= 0.0,
            "sampling temperature must be finite and nonnegative"
        );
        ensure!(
            self.top_p.is_finite() && self.top_p > 0.0 && self.top_p <= 1.0,
            "sampling top_p must be finite and in (0, 1]"
        );
        Ok(())
    }
}

impl Default for SamplingConfig {
    fn default() -> Self {
        Self {
            temperature: 1.0,
            top_k: 0,
            top_p: 1.0,
        }
    }
}

/// A conversation-local RNG and reusable sampling workspace.
///
/// A seeded sampler reproduces its own stream, not another implementation's
/// RNG or historical releases of `rand`. Greedy draws and rejected input do
/// not consume randomness. Sampling never mutates the caller's logits.
pub struct Sampler {
    cfg: SamplingConfig,
    seed: u64,
    rng: StdRng,
    // Token-indexed logits, then unnormalized probabilities. f64 keeps finite
    // f32 logits/temperatures safe even near their representable extremes.
    weights: Vec<f64>,
    // Allocated only if a top-k or top-p cut actually requires candidates.
    indices: Vec<usize>,
}

impl Sampler {
    pub fn new(cfg: SamplingConfig, seed: u64) -> Result<Self> {
        cfg.validate()?;
        Ok(Self {
            cfg,
            seed,
            rng: StdRng::seed_from_u64(seed),
            weights: Vec::new(),
            indices: Vec::new(),
        })
    }

    pub fn config(&self) -> &SamplingConfig {
        &self.cfg
    }

    pub fn seed(&self) -> u64 {
        self.seed
    }

    /// Restart the exact RNG stream without reallocating the workspace.
    pub fn reseed(&mut self) {
        self.rng = StdRng::seed_from_u64(self.seed);
    }

    pub fn token(&mut self, logits: &[f32]) -> Result<usize> {
        self.token_excluding(logits, &[])
    }

    /// Draw with zero probability for every token id in `forbidden`.
    ///
    /// Duplicate exclusions are harmless; out-of-vocabulary ids are an error.
    /// Exclusion work is O(vocabulary + forbidden), not a membership scan for
    /// every token. The all-excluded case errors even for greedy decoding.
    pub fn token_excluding(&mut self, logits: &[f32], forbidden: &[usize]) -> Result<usize> {
        ensure!(!logits.is_empty(), "cannot sample an empty logit row");
        for &id in forbidden {
            ensure!(
                id < logits.len(),
                "forbidden token {id} is outside vocabulary {}",
                logits.len()
            );
        }

        self.weights.clear();
        self.weights.extend(logits.iter().map(|&value| {
            if value.is_finite() {
                value as f64
            } else {
                f64::NEG_INFINITY
            }
        }));
        for &id in forbidden {
            self.weights[id] = f64::NEG_INFINITY;
        }

        let mut peak = None;
        let mut max = f64::NEG_INFINITY;
        let mut eligible = 0;
        for (id, &value) in self.weights.iter().enumerate() {
            if value.is_finite() {
                eligible += 1;
                if value > max {
                    max = value;
                    peak = Some(id);
                }
            }
        }
        let peak = peak.ok_or_else(|| anyhow::anyhow!("no eligible finite logit to sample"))?;
        if self.cfg.is_greedy() {
            return Ok(peak);
        }

        let temperature = self.cfg.temperature as f64;
        let cut_k = self.cfg.top_k > 0 && self.cfg.top_k < eligible;
        let cut_p = self.cfg.top_p < 1.0;
        self.indices.clear();
        if !cut_k && !cut_p {
            // Common SDFT path: preserve vocabulary order, no candidate list,
            // no sort, and no per-token allocation after the workspace grows.
            let mut sum = 0.0;
            for value in &mut self.weights {
                *value = ((*value - max) / temperature).exp();
                sum += *value;
            }
            return Ok(draw(self.weights.iter().copied().enumerate(), sum, &mut self.rng));
        }

        self.indices.extend(
            self.weights
                .iter()
                .enumerate()
                .filter_map(|(id, value)| value.is_finite().then_some(id)),
        );
        let weights = &self.weights;
        let descending = |&a: &usize, &b: &usize| {
            weights[b].partial_cmp(&weights[a]).unwrap().then(a.cmp(&b))
        };
        if cut_k {
            let k = self.cfg.top_k;
            self.indices.select_nth_unstable_by(k - 1, descending);
            self.indices.truncate(k);
        }
        if cut_p {
            self.indices.sort_unstable_by(descending);
        }

        let mut sum = 0.0;
        for &id in &self.indices {
            let value = &mut self.weights[id];
            *value = ((*value - max) / temperature).exp();
            sum += *value;
        }
        if cut_p {
            let threshold = sum * self.cfg.top_p as f64;
            let mut cumulative = 0.0;
            let mut keep = self.indices.len();
            for (position, &id) in self.indices.iter().enumerate() {
                cumulative += self.weights[id];
                if cumulative >= threshold {
                    keep = position + 1;
                    break;
                }
            }
            self.indices.truncate(keep);
            sum = cumulative;
        }
        Ok(draw(
            self.indices.iter().map(|&id| (id, self.weights[id])),
            sum,
            &mut self.rng,
        ))
    }
}

/// Inverse CDF with unnormalized nonnegative weights and a positive finite
/// sum. Roundoff at the upper endpoint falls back to the last POSITIVE weight,
/// never an excluded token. The maximum logit always contributes weight one.
fn draw(weights: impl Iterator<Item = (usize, f64)>, sum: f64, rng: &mut StdRng) -> usize {
    let mut threshold = rng.r#gen::<f64>() * sum;
    let mut last_positive = None;
    for (id, weight) in weights {
        if weight > 0.0 {
            last_positive = Some(id);
            if threshold < weight {
                return id;
            }
            threshold -= weight;
        }
    }
    last_positive.expect("validated sampling row has positive probability mass")
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sequence(cfg: SamplingConfig, seed: u64, logits: &[f32], count: usize) -> Vec<usize> {
        let mut sampler = Sampler::new(cfg, seed).unwrap();
        (0..count).map(|_| sampler.token(logits).unwrap()).collect()
    }

    #[test]
    fn default_is_full_distribution_not_greedy() {
        let cfg = SamplingConfig::default();
        assert_eq!(cfg.temperature, 1.0);
        assert_eq!(cfg.top_k, 0);
        assert_eq!(cfg.top_p, 1.0);
        assert!(!cfg.is_greedy());
    }

    #[test]
    fn rejects_invalid_configuration() {
        for temperature in [-1.0, f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
            assert!(Sampler::new(SamplingConfig { temperature, ..Default::default() }, 0).is_err());
        }
        for top_p in [0.0, -1.0, 1.1, f32::NAN, f32::INFINITY] {
            assert!(Sampler::new(SamplingConfig { top_p, ..Default::default() }, 0).is_err());
        }
    }

    #[test]
    fn fixed_seed_and_reseed_reproduce_the_stream() {
        let logits = [0.0, 0.5, 1.0, 1.5];
        for cfg in [
            SamplingConfig::default(),
            SamplingConfig { top_k: 3, ..Default::default() },
            SamplingConfig { top_k: 3, top_p: 0.8, ..Default::default() },
        ] {
            let first = sequence(cfg, 17, &logits, 256);
            assert_eq!(first, sequence(cfg, 17, &logits, 256));
            assert_ne!(first, sequence(cfg, 18, &logits, 256));
            let mut sampler = Sampler::new(cfg, 17).unwrap();
            for &expected in &first {
                assert_eq!(sampler.token(&logits).unwrap(), expected);
            }
            sampler.reseed();
            for &expected in &first {
                assert_eq!(sampler.token(&logits).unwrap(), expected);
            }
        }
    }

    #[test]
    fn full_distribution_matches_inverse_cdf_without_ordering_scratch() {
        let mut sampler = Sampler::new(SamplingConfig::default(), 42).unwrap();
        let mut reference = StdRng::seed_from_u64(42);
        let logits = [0.0_f32, 1.0, -1.0, 0.5];
        let weights = logits.map(|value| (value as f64 - 1.0).exp());
        let sum: f64 = weights.iter().sum();
        let mut seen = [false; 4];
        for _ in 0..1024 {
            let threshold = reference.r#gen::<f64>() * sum;
            let mut cumulative = 0.0;
            let expected = weights.iter().position(|weight| {
                cumulative += weight;
                threshold < cumulative
            }).unwrap();
            let actual = sampler.token(&logits).unwrap();
            assert_eq!(actual, expected);
            seen[actual] = true;
        }
        assert!(seen.into_iter().all(|visited| visited));
        assert_eq!(sampler.indices.capacity(), 0);
    }

    #[test]
    fn large_vocabulary_reuses_workspace_without_candidate_allocation() {
        let logits = vec![0.0; 200_000];
        let mut sampler = Sampler::new(SamplingConfig::default(), 4).unwrap();
        assert!(sampler.token(&logits).unwrap() < logits.len());
        let capacity = sampler.weights.capacity();
        for _ in 0..4 {
            assert!(sampler.token(&logits).unwrap() < logits.len());
            assert_eq!(sampler.weights.capacity(), capacity);
            assert_eq!(sampler.indices.capacity(), 0);
        }
    }

    #[test]
    fn greedy_ties_and_nonfinite_exclusions_do_not_consume_rng() {
        let mut sampler = Sampler::new(SamplingConfig::greedy(), 31).unwrap();
        let logits = [f32::NAN, -2.0, -2.0, f32::INFINITY, f32::NEG_INFINITY];
        assert_eq!(sampler.token(&logits).unwrap(), 1);
        assert_eq!(sampler.token_excluding(&logits, &[1]).unwrap(), 2);
        let mut reference = StdRng::seed_from_u64(31);
        assert_eq!(sampler.rng.r#gen::<u64>(), reference.r#gen::<u64>());
    }

    #[test]
    fn forbidden_and_nonfinite_tokens_never_survive_any_policy() {
        let logits = [f32::NAN, 0.0, 100.0, f32::INFINITY, f32::NEG_INFINITY];
        for cfg in [
            SamplingConfig::greedy(),
            SamplingConfig::default(),
            SamplingConfig { top_k: 1, top_p: 0.1, ..Default::default() },
        ] {
            let mut sampler = Sampler::new(cfg, 9).unwrap();
            for _ in 0..128 {
                assert_eq!(sampler.token_excluding(&logits, &[2, 2]).unwrap(), 1);
            }
        }
    }

    #[test]
    fn excluded_peak_cannot_underflow_the_remaining_distribution() {
        let mut sampler = Sampler::new(SamplingConfig::default(), 41).unwrap();
        for _ in 0..128 {
            let token = sampler.token_excluding(&[f32::MAX, -f32::MAX], &[0]).unwrap();
            assert_eq!(token, 1);
        }
    }

    #[test]
    fn bad_rows_are_errors_without_advancing_rng() {
        for cfg in [SamplingConfig::default(), SamplingConfig::greedy()] {
            let mut sampler = Sampler::new(cfg, 5).unwrap();
            assert!(sampler.token(&[]).is_err());
            assert!(sampler.token(&[f32::NAN, f32::INFINITY, f32::NEG_INFINITY]).is_err());
            assert!(sampler.token_excluding(&[1.0, 2.0], &[0, 1]).is_err());
            assert!(sampler.token_excluding(&[1.0, 2.0], &[2]).is_err());
            let mut reference = StdRng::seed_from_u64(5);
            assert_eq!(sampler.rng.r#gen::<u64>(), reference.r#gen::<u64>());
        }
    }

    #[test]
    fn top_k_obeys_token_id_ties_and_oversized_k_is_unrestricted() {
        let logits = [3.0, 3.0, 3.0, 2.0];
        let cfg = SamplingConfig { top_k: 2, ..Default::default() };
        let tokens = sequence(cfg, 13, &logits, 256);
        assert!(tokens.iter().all(|&id| id < 2));
        assert!(tokens.contains(&0) && tokens.contains(&1));
        let unrestricted = sequence(SamplingConfig::default(), 13, &logits, 256);
        for top_k in [logits.len(), usize::MAX] {
            assert_eq!(
                unrestricted,
                sequence(SamplingConfig { top_k, ..Default::default() }, 13, &logits, 256)
            );
        }
    }

    #[test]
    fn nucleus_keeps_minimum_prefix_including_boundary_token() {
        let cfg = SamplingConfig { top_p: 0.5, ..Default::default() };
        let tokens = sequence(cfg, 20, &[0.0; 4], 256);
        assert!(tokens.iter().all(|&id| id < 2));
        assert!(tokens.contains(&0) && tokens.contains(&1));
        let cfg = SamplingConfig { top_p: f32::from_bits(1), ..Default::default() };
        assert!(sequence(cfg, 20, &[0.0, 1.0, -1.0], 64).iter().all(|&id| id == 1));
    }

    #[test]
    fn nucleus_mass_is_normalized_after_top_k() {
        // Weights 4:3:2:1. The first token has < 50% of the full mass, but
        // > 50% after a top-two cut; only the latter must collapse to token 0.
        let logits = [4.0_f32.ln(), 3.0_f32.ln(), 2.0_f32.ln(), 0.0];
        let cfg = SamplingConfig { top_k: 2, top_p: 0.5, ..Default::default() };
        assert!(sequence(cfg, 12, &logits, 256).iter().all(|&id| id == 0));
        let cfg = SamplingConfig { top_p: 0.5, ..Default::default() };
        let tokens = sequence(cfg, 12, &logits, 256);
        assert!(tokens.contains(&0) && tokens.contains(&1));
        assert!(tokens.iter().all(|&id| id < 2));
    }

    #[test]
    fn extreme_finite_logits_and_temperatures_stay_well_defined() {
        let tiny = f32::from_bits(1);
        let logits = [0.0, -tiny, -2.0 * tiny];
        let cfg = SamplingConfig { temperature: tiny, ..Default::default() };
        assert_eq!(
            sequence(cfg, 21, &logits, 256),
            sequence(SamplingConfig::default(), 21, &[0.0, -1.0, -2.0], 256)
        );
        let cfg = SamplingConfig { temperature: f32::MAX, ..Default::default() };
        assert_eq!(
            sequence(cfg, 21, &[f32::MAX, 0.0, -f32::MAX], 256),
            sequence(SamplingConfig::default(), 21, &[0.0, -1.0, -2.0], 256)
        );
        let cfg = SamplingConfig { temperature: tiny, ..Default::default() };
        let tokens = sequence(cfg, 21, &[f32::MAX, f32::MAX, -f32::MAX], 256);
        assert!(tokens.iter().all(|&id| id < 2));
        assert!(tokens.contains(&0) && tokens.contains(&1));
    }
}
