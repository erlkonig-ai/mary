//! Token sampling for the PersonaPlex step machine — temperature / top-k /
//! top-p (nucleus) over a logit row, with a **seedable** RNG so a session is
//! reproducible. Both the text stream (temporal transformer's `[32000]` head)
//! and the 16 audio streams (depformer's per-step `[2048]` heads) sample
//! through the SAME [`sample`] entry point; greedy (argmax) stays available
//! and is the numeric-parity default.
//!
//! Greedy is [`argmax`](super::depth::argmax) — the parity path against the
//! moshi oracle. Sampling is the *quality* path: the CPU-f32 framebench found
//! a near-tie at step 88 where argmax picks the wrong token by a hair, so the
//! realtime loop needs temperature to break out of that basin. This module is
//! the sampler; wiring it into [`super::lmgen`] / [`super::depth`] /
//! [`super::depth_fast`] is where the argmax sites become "sample when a
//! [`SamplingConfig`] is present, else argmax".
//!
//! [`super::depth_gpu`] is the exception that proves the RNG rule below: its
//! argmax lives in a kernel and the token a step draws feeds the next step's
//! embedding on-device, so it cannot call [`Sampler::token`] at all without
//! 16 readbacks a frame. It calls [`Sampler::uniforms`] instead — the RNG
//! still lives here, on the host, and the device only consumes the numbers it
//! draws.
//!
//! **RNG discipline (hard rule):** the RNG is passed IN as `&mut StdRng` seeded
//! by the caller (`StdRng::seed_from_u64(seed)`), never drawn from wall-clock
//! or env here. Two runs with the same seed and the same logit stream produce
//! identical tokens — that reproducibility is what the gates assert (we do NOT
//! try to match the Python oracle's RNG bit-for-bit; different RNG, so only the
//! sampler's own PROPERTIES are gated: temp→0 == argmax, seeded determinism,
//! top-k truncation).
//!
//! **Warper order** (HF `LogitsWarper` convention, same as
//! [`super::super::qwen3tts`]'s talker sampler): temperature scaling →
//! top-k cut → top-p cut → softmax over the survivors → multinomial draw.

use rand::distributions::{Distribution, WeightedIndex};
use rand::rngs::StdRng;

use super::depth::argmax;

/// Sampling knobs for one logit row. `greedy()` (or `temp <= 0`) reproduces
/// argmax exactly; otherwise the row is temperature-scaled, truncated to the
/// top-`k` logits (`0` = no k cut) and the smallest nucleus whose mass ≥
/// `top_p` (`>= 1.0` = no p cut), then sampled.
#[derive(Clone, Copy, Debug)]
pub struct SamplingConfig {
    /// Softmax temperature. `<= 0.0` means greedy (argmax); `1.0` is the
    /// unscaled distribution; smaller sharpens, larger flattens.
    pub temp: f32,
    /// Keep only the `k` highest logits before sampling (`0` = keep all).
    pub top_k: usize,
    /// Keep the smallest set of top logits whose softmax mass ≥ `top_p`
    /// (`>= 1.0` = keep all). Applied after the top-k cut.
    pub top_p: f32,
}

impl SamplingConfig {
    /// The parity default: argmax, no randomness. Equivalent to any config
    /// with `temp <= 0.0` — [`sample`] short-circuits to [`argmax`].
    pub const fn greedy() -> Self {
        Self {
            temp: 0.0,
            top_k: 0,
            top_p: 1.0,
        }
    }

    /// Whether this config is the greedy (argmax) path.
    pub fn is_greedy(&self) -> bool {
        self.temp <= 0.0
    }
}

impl Default for SamplingConfig {
    fn default() -> Self {
        Self::greedy()
    }
}

/// Sample one token id from a logit row under `cfg`, drawing from `rng`.
///
/// Greedy (`cfg.is_greedy()`) returns [`argmax`] — first-index-wins, exactly
/// matching the parity path (so `temp -> 0` is `argmax`, no separate branch to
/// drift). Otherwise: temperature-scale, top-k cut, top-p (nucleus) cut,
/// softmax over the survivors, multinomial draw. Deterministic for a given
/// `rng` state, so a seeded run is reproducible.
///
/// Panics on an empty `logits` slice (a caller bug — every head has ≥ 1 row).
pub fn sample(logits: &[f32], cfg: &SamplingConfig, rng: &mut StdRng) -> usize {
    assert!(!logits.is_empty(), "sample: empty logit row");
    if cfg.is_greedy() {
        return argmax(logits);
    }

    // Sanitize non-finite logits (a rare q4/q8 quantization artifact): a NaN or
    // ±inf would poison the softmax below and panic `WeightedIndex::new`. Map
    // each non-finite value to NEG_INFINITY so it sorts to the bottom, drops out
    // of top-k/top-p, and gets zero probability; if EVERY logit is non-finite,
    // fall back to argmax (a fully-corrupted distribution can't be sampled).
    let sanitized: Vec<f32> = logits
        .iter()
        .map(|&x| if x.is_finite() { x } else { f32::NEG_INFINITY })
        .collect();
    if !sanitized.iter().any(|x| x.is_finite()) {
        return argmax(logits);
    }
    let logits = &sanitized[..];

    // Candidate indices sorted by descending logit (first-index-wins on ties,
    // matching argmax: a stable sort keeps the lower index first among equals).
    let mut idx: Vec<usize> = (0..logits.len()).collect();
    idx.sort_by(|&a, &b| {
        logits[b]
            .partial_cmp(&logits[a])
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    // top-k: keep the k highest logits (0 = keep all).
    if cfg.top_k > 0 && cfg.top_k < idx.len() {
        idx.truncate(cfg.top_k);
    }

    // Temperature-scaled softmax over the survivors (subtract max for
    // numerical stability; f64 accumulation).
    let t = cfg.temp as f64;
    let scaled: Vec<f64> = idx.iter().map(|&i| logits[i] as f64 / t).collect();
    let m = scaled.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let exps: Vec<f64> = scaled.iter().map(|&s| (s - m).exp()).collect();
    let sum: f64 = exps.iter().sum();

    // top-p (nucleus): keep the smallest prefix (already descending) whose
    // cumulative probability mass reaches top_p. Always keep at least one.
    let mut probs: Vec<f64> = exps.iter().map(|&e| e / sum).collect();
    if (cfg.top_p as f64) < 1.0 {
        let p = cfg.top_p as f64;
        let mut cum = 0.0;
        let mut keep = probs.len();
        for (n, &pr) in probs.iter().enumerate() {
            cum += pr;
            if cum >= p {
                keep = n + 1;
                break;
            }
        }
        idx.truncate(keep);
        probs.truncate(keep);
    }

    let dist = WeightedIndex::new(&probs).expect("sampling weights");
    idx[dist.sample(rng)]
}

/// A config + seeded RNG bundled together — the stateful sampler the step
/// machine holds and threads to both the temporal (text) and depth (audio)
/// heads. `Sampler::token` picks one id per logit row and advances the RNG,
/// so a session started from one seed is fully reproducible.
pub struct Sampler {
    cfg: SamplingConfig,
    /// The seed the RNG was created from — retained so [`Sampler::reseed`] can
    /// restart the exact stream (a per-session reset that reproduces run 1).
    seed: u64,
    rng: StdRng,
}

impl Sampler {
    /// A sampler with `cfg`, RNG seeded from `seed`.
    pub fn new(cfg: SamplingConfig, seed: u64) -> Self {
        use rand::SeedableRng;
        Self {
            cfg,
            seed,
            rng: StdRng::seed_from_u64(seed),
        }
    }

    /// Sample one token id from a logit row (greedy iff `cfg.is_greedy()`).
    pub fn token(&mut self, logits: &[f32]) -> usize {
        sample(logits, &self.cfg, &mut self.rng)
    }

    /// Draw one uniform in `[0, 1)` per element of `out`, in order — the
    /// randomness a DEVICE-side sampler needs, without the RNG leaving the
    /// host.
    ///
    /// [`super::depth_gpu`] does its temperature / top-k / top-p draw in a
    /// kernel, because reading 16 logit rows back per frame to sample them on
    /// the host would cost the whole reason for being on the device. This is
    /// the seam that keeps the module's hard rule intact anyway: the device
    /// contributes no entropy, it only consumes these numbers, so a seeded
    /// session is still reproducible and `reseed` still restarts the same
    /// stream. Drawing them all at frame start (rather than per step) is what
    /// keeps the frame at one host->device upload.
    pub fn uniforms(&mut self, out: &mut [f32]) {
        use rand::Rng;
        for u in out.iter_mut() {
            *u = self.rng.r#gen::<f32>();
        }
    }

    /// Restore the RNG to its original seed, so the next session samples the
    /// identical token stream as the first. Called from the step machine's
    /// `reset_session` so a reset is a true "start over" without the caller
    /// having to re-apply the sampling config.
    pub fn reseed(&mut self) {
        use rand::SeedableRng;
        self.rng = StdRng::seed_from_u64(self.seed);
    }

    /// The config this sampler carries.
    pub fn config(&self) -> &SamplingConfig {
        &self.cfg
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;

    /// A reproducible pseudo-random logit row (LCG, no external RNG so the
    /// fixture itself is deterministic).
    fn logits(n: usize, seed: u64) -> Vec<f32> {
        let mut s = seed;
        (0..n)
            .map(|_| {
                s = s
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                ((s >> 33) as f32) / (1u32 << 31) as f32 - 1.0
            })
            .collect()
    }

    /// Greedy config == argmax on a battery of rows (the parity guarantee).
    #[test]
    fn greedy_equals_argmax() {
        let cfg = SamplingConfig::greedy();
        for seed in 0..50 {
            let l = logits(2048, seed);
            let mut rng = StdRng::seed_from_u64(seed);
            assert_eq!(
                sample(&l, &cfg, &mut rng),
                argmax(&l),
                "greedy != argmax seed {seed}"
            );
        }
    }

    /// A vanishingly small temperature reproduces argmax **when the peak has a
    /// real margin** — the softmax concentrates entirely on the max, so the
    /// numeric `temp -> 0` limit agrees with the `is_greedy` short-circuit.
    /// (For rows with two logits closer together than `temp`, the limit is
    /// only approached asymptotically — that's the definition of the limit,
    /// not a bug; the exact-argmax guarantee is the greedy flag, gated above.)
    #[test]
    fn tiny_temp_equals_argmax() {
        let cfg = SamplingConfig {
            temp: 1e-4,
            top_k: 0,
            top_p: 1.0,
        };
        for seed in 0..50u64 {
            let mut l = logits(2048, seed + 999);
            // Give a clear winner (well above the [-1, 1] fixture range) so the
            // 1e-4-temperature softmax mass is ≈ 1.0 on it.
            let peak = (seed as usize * 37 + 5) % l.len();
            l[peak] = 100.0;
            let mut rng = StdRng::seed_from_u64(seed);
            let got = sample(&l, &cfg, &mut rng);
            assert_eq!(got, peak, "tiny-temp margin seed {seed}");
            assert_eq!(got, argmax(&l), "tiny-temp != argmax seed {seed}");
        }
    }

    /// A non-finite logit (NaN or ±inf — a q4/q8 quantization artifact) must not
    /// panic the sampler: the poisoned entries are excluded and a finite index
    /// is drawn. If EVERY logit is non-finite, it falls back to argmax.
    #[test]
    fn non_finite_logits_do_not_panic() {
        let cfg = SamplingConfig {
            temp: 0.8,
            top_k: 50,
            top_p: 0.95,
        };
        for (seed, bad) in [(1u64, f32::NAN), (2, f32::INFINITY), (3, f32::NEG_INFINITY)] {
            let mut l = logits(2048, seed);
            l[7] = bad;
            l[100] = bad;
            l[500] = f32::NAN;
            l[42] = 50.0; // a clear finite winner
            let mut rng = StdRng::seed_from_u64(seed);
            let got = sample(&l, &cfg, &mut rng); // must not panic
            assert!(
                l[got].is_finite(),
                "sampled a non-finite index (seed {seed})"
            );
        }
        // All non-finite -> argmax fallback (no panic).
        let cfg = SamplingConfig {
            temp: 1.0,
            top_k: 0,
            top_p: 1.0,
        };
        let mut rng = StdRng::seed_from_u64(0);
        let _ = sample(&vec![f32::NAN; 16], &cfg, &mut rng); // must not panic
    }

    /// Same seed → same token sequence across two independent runs.
    #[test]
    fn seeded_is_deterministic() {
        let cfg = SamplingConfig {
            temp: 1.0,
            top_k: 64,
            top_p: 0.95,
        };
        let l = logits(2048, 7);
        let draw = |seed: u64| {
            let mut rng = StdRng::seed_from_u64(seed);
            (0..100)
                .map(|_| sample(&l, &cfg, &mut rng))
                .collect::<Vec<_>>()
        };
        assert_eq!(
            draw(42),
            draw(42),
            "same seed must reproduce the token stream"
        );
        // And a different seed generally diverges (guards against a constant).
        assert_ne!(draw(42), draw(43), "different seeds should diverge");
    }

    /// top-k truncation: every sampled token is among the k highest logits.
    #[test]
    fn top_k_truncates() {
        let k = 8;
        let cfg = SamplingConfig {
            temp: 2.0,
            top_k: k,
            top_p: 1.0,
        };
        let l = logits(2048, 3);
        // The k highest logit indices.
        let mut idx: Vec<usize> = (0..l.len()).collect();
        idx.sort_by(|&a, &b| l[b].partial_cmp(&l[a]).unwrap());
        let allowed: std::collections::HashSet<usize> = idx[..k].iter().copied().collect();
        let mut rng = StdRng::seed_from_u64(1);
        for _ in 0..2000 {
            let tok = sample(&l, &cfg, &mut rng);
            assert!(allowed.contains(&tok), "sampled {tok} outside top-{k}");
        }
    }

    /// top-p (nucleus): with a tight p only the very top tokens are eligible.
    #[test]
    fn top_p_truncates() {
        // A row with one dominant logit: nucleus 0.5 must always pick it.
        let mut l = vec![0.0f32; 2048];
        l[123] = 20.0; // softmax mass ≈ 1.0 on index 123
        let cfg = SamplingConfig {
            temp: 1.0,
            top_k: 0,
            top_p: 0.5,
        };
        let mut rng = StdRng::seed_from_u64(5);
        for _ in 0..500 {
            assert_eq!(
                sample(&l, &cfg, &mut rng),
                123,
                "nucleus should collapse to the peak"
            );
        }
    }

    /// Sampling with a wide config visits more than one token (not a
    /// degenerate argmax-in-disguise) — sanity that randomness is live.
    #[test]
    fn wide_sampling_is_nondegenerate() {
        let cfg = SamplingConfig {
            temp: 1.5,
            top_k: 128,
            top_p: 1.0,
        };
        let l = logits(2048, 11);
        let mut rng = StdRng::seed_from_u64(2);
        let seen: std::collections::HashSet<usize> =
            (0..500).map(|_| sample(&l, &cfg, &mut rng)).collect();
        assert!(seen.len() > 1, "wide sampling collapsed to a single token");
    }
}
