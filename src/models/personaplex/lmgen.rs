//! PersonaPlex `LMGen.step` machinery — the delay/undelay token bookkeeping
//! around the temporal + depth transformers (moshi `lm.py`:
//! `LMGen.prepare_step_input` / `process_transformer_output`), ported as a
//! pure integer machine ([`StreamCache`]) plus the model glue ([`LmGen`]).
//!
//! The 17 streams are `[text, agent audio 1..8, user audio 1..8]` with
//! acquisition delays `[0, 0,1×7, 0,1×7]` (semantic codebooks and text
//! undelayed, the 7 acoustic codebooks of each stream lag one frame). The
//! ring cache holds `max_delay + 3 = 4` positions per stream; at stream
//! offset `o` the machine
//!
//! 1. writes caller-provided tokens at position `(o + delay_k) % 4` and
//!    flags them `provided` (user audio → streams 9..17, agent/"moshi"
//!    prompt audio → streams 1..9, text → stream 0),
//! 2. seeds initial tokens (`2048` audio / `32000` text) at `o % 4` for
//!    every stream with `o <= delay_k` (only fires at offsets 0 and 1),
//! 3. at `o == 0` only: fills position 0 with initials and skips the model,
//! 4. feeds position `(o-1) % 4` to the temporal transformer, samples text,
//!    runs the depformer with per-stream forcing from the `provided` flags
//!    at position `o % 4`, writes sampled tokens there (where not provided),
//!    clears the `provided` flags at the input position, and
//! 5. emits the undelayed output frame `out[k] = cache[k][(o - 1 + delay_k)
//!    % 4]` once `o > max_delay`.
//!
//! Everything is greedy (argmax) — the parity path. Sampling knobs are a
//! later increment.

use burn::prelude::*;

use super::config as cfg;
use super::depth::{DepthTransformer, argmax};
use super::sampling::{Sampler, SamplingConfig};
use super::temporal::TemporalTransformer;

/// Ring size (moshi: `max_delay + 3`).
pub const CT: usize = cfg::MAX_DELAY + 3;
/// moshi `ungenerated_token_id` — a slot no one has written yet.
const UNGENERATED: i64 = -2;

fn initial(k: usize) -> i64 {
    if k == 0 {
        cfg::TEXT_CARD as i64 // text_initial_token_id = 32000
    } else {
        cfg::CARD as i64 // audio initial_token_id = 2048
    }
}

/// One prepared model step: the token frame to feed the temporal stack and
/// the forcing view of the target position.
pub struct Prepared {
    /// `cache[:, (offset-1) % CT]` — the 17 input tokens for this step.
    pub input: [i64; cfg::NUM_STREAMS],
    /// `cache[:, offset % CT]` — target tokens (valid where `provided`).
    pub target: [i64; cfg::NUM_STREAMS],
    /// Which targets were provided (teacher-force the depformer's prev-token
    /// chain and survive the post-step cache write).
    pub provided: [bool; cfg::NUM_STREAMS],
    input_pos: usize,
    target_pos: usize,
}

impl Prepared {
    /// The depformer forcing view: `forced[s] = provided target of audio
    /// stream s+1` (moshi `depformer_step`'s `audio_tokens`/`audio_provided`).
    pub fn forced(&self) -> [Option<i64>; cfg::DEP_Q] {
        std::array::from_fn(|s| self.provided[1 + s].then_some(self.target[1 + s]))
    }
}

/// The pure token bookkeeping of `LMGen` — no tensors, fully deterministic,
/// shared by the parity gates (driven from goldens) and [`LmGen`].
pub struct StreamCache {
    cache: [[i64; CT]; cfg::NUM_STREAMS],
    provided: [[bool; CT]; cfg::NUM_STREAMS],
    offset: usize,
}

impl StreamCache {
    #[allow(clippy::new_without_default)]
    pub fn new() -> Self {
        Self {
            cache: [[UNGENERATED; CT]; cfg::NUM_STREAMS],
            provided: [[false; CT]; cfg::NUM_STREAMS],
            offset: 0,
        }
    }

    pub fn offset(&self) -> usize {
        self.offset
    }

    /// Reset the token ring, `provided` flags, and offset to the pristine
    /// state of [`StreamCache::new`] — a new conversation without reallocating.
    /// Equivalent to assigning a fresh `StreamCache`.
    pub fn reset(&mut self) {
        self.cache = [[UNGENERATED; CT]; cfg::NUM_STREAMS];
        self.provided = [[false; CT]; cfg::NUM_STREAMS];
        self.offset = 0;
    }

    /// moshi `prepare_step_input`. Returns `None` only for the very first
    /// call (offset 0 — cache seeding, no model step).
    pub fn prepare(
        &mut self,
        input_tokens: Option<&[i64; 8]>,
        moshi_tokens: Option<&[i64; 8]>,
        text_token: Option<i64>,
    ) -> Option<Prepared> {
        if let Some(it) = input_tokens {
            for (q, &t) in it.iter().enumerate() {
                let k = cfg::AUDIO_TOKENS_PER_STREAM + 1 + q; // user streams 9..17
                let wp = (self.offset + cfg::DELAYS[k]) % CT;
                self.cache[k][wp] = t;
                self.provided[k][wp] = true;
            }
        }
        if let Some(mt) = moshi_tokens {
            for (q, &t) in mt.iter().enumerate() {
                let k = 1 + q; // agent streams 1..9
                let wp = (self.offset + cfg::DELAYS[k]) % CT;
                self.cache[k][wp] = t;
                self.provided[k][wp] = true;
            }
        }
        if let Some(t) = text_token {
            let wp = (self.offset + cfg::DELAYS[0]) % CT;
            self.cache[0][wp] = t;
            self.provided[0][wp] = true;
        }

        // Initial-token seeding for delayed streams at the very beginning.
        for k in 0..cfg::NUM_STREAMS {
            if self.offset <= cfg::DELAYS[k] {
                self.cache[k][self.offset % CT] = initial(k);
                self.provided[k][self.offset % CT] = true;
            }
        }

        if self.offset == 0 {
            for k in 0..cfg::NUM_STREAMS {
                self.cache[k][0] = initial(k);
            }
            self.offset += 1;
            return None;
        }

        let input_pos = (self.offset - 1) % CT;
        let target_pos = self.offset % CT;
        Some(Prepared {
            input: std::array::from_fn(|k| self.cache[k][input_pos]),
            target: std::array::from_fn(|k| self.cache[k][target_pos]),
            provided: std::array::from_fn(|k| self.provided[k][target_pos]),
            input_pos,
            target_pos,
        })
    }

    /// moshi `process_transformer_output`'s bookkeeping tail: clear the
    /// `provided` flags at the consumed input position, write the sampled
    /// tokens at the target where not provided, and emit the undelayed
    /// output frame (all 17 streams) once past the delay horizon.
    pub fn commit(
        &mut self,
        p: &Prepared,
        sampled_text: i64,
        sampled_audio: &[i64; cfg::DEP_Q],
    ) -> Option<[i64; cfg::NUM_STREAMS]> {
        for k in 0..cfg::NUM_STREAMS {
            self.provided[k][p.input_pos] = false;
        }
        if !self.provided[0][p.target_pos] {
            self.cache[0][p.target_pos] = sampled_text;
        }
        for (s, &t) in sampled_audio.iter().enumerate() {
            if !self.provided[1 + s][p.target_pos] {
                self.cache[1 + s][p.target_pos] = t;
            }
        }
        let out = (self.offset > cfg::MAX_DELAY).then(|| {
            std::array::from_fn(|k| {
                self.cache[k][(self.offset - cfg::MAX_DELAY + cfg::DELAYS[k]) % CT]
            })
        });
        self.offset += 1;
        out
    }

    /// moshi's voice-prompt replay tail: `state.cache.copy_(voice_prompt_
    /// cache)` — replace the token ring wholesale (row-major `[17, CT]`);
    /// `provided` flags and offset are deliberately untouched.
    pub fn overwrite(&mut self, snapshot: &[i64]) {
        assert_eq!(
            snapshot.len(),
            cfg::NUM_STREAMS * CT,
            "cache snapshot shape"
        );
        for k in 0..cfg::NUM_STREAMS {
            self.cache[k].copy_from_slice(&snapshot[k * CT..(k + 1) * CT]);
        }
    }
}

/// What one `LmGen` step did — the probe gates every field against the
/// oracle goldens; the production loop only needs `out`.
pub struct StepTrace {
    /// The 17 input tokens fed to the temporal stack (`None` for
    /// embedding-fed voice-prompt steps).
    pub input: Option<[i64; cfg::NUM_STREAMS]>,
    /// `next_text_token` handed to the depformer (forced target or sampled).
    pub next_text: i64,
    /// The depformer's 16 greedy tokens.
    pub dep_tokens: [i64; cfg::DEP_Q],
    /// Undelayed output frame, once past the delay horizon.
    pub out: Option<[i64; cfg::NUM_STREAMS]>,
}

/// The full step machine: temporal transformer + depth transformer + stream
/// bookkeeping. Greedy/parity path (CPU f32) by default; call
/// [`Self::set_sampling`] to switch both the text and audio heads onto the
/// seeded [`Sampler`] (the quality path).
pub struct LmGen<B: Backend> {
    pub temporal: TemporalTransformer<B>,
    pub depth: DepthTransformer<B>,
    pub stream: StreamCache,
    /// `None` = greedy argmax (parity). `Some` = seeded temp/top-k/top-p over
    /// both the text logits and each depformer step's audio logits.
    sampler: Option<Sampler>,
}

impl<B: Backend> LmGen<B> {
    pub fn new(temporal: TemporalTransformer<B>, depth: DepthTransformer<B>) -> Self {
        Self {
            temporal,
            depth,
            stream: StreamCache::new(),
            sampler: None,
        }
    }

    /// Switch the step machine onto seeded sampling (`cfg` + `seed`) for both
    /// the temporal text head and the depformer audio heads. A greedy `cfg`
    /// (`SamplingConfig::greedy()`) still reproduces argmax exactly.
    pub fn set_sampling(&mut self, cfg: SamplingConfig, seed: u64) {
        self.sampler = Some(Sampler::new(cfg, seed));
    }

    /// Drop back to greedy argmax (clears the sampler).
    pub fn set_greedy(&mut self) {
        self.sampler = None;
    }

    /// Reset the streaming state so a new conversation can start WITHOUT
    /// reloading weights: clears the temporal KV cache + offset and the stream
    /// ring/offset. The depth transformer is stateless between frames, and the
    /// sampler (if any) keeps its RNG — call [`Self::set_sampling`] again to
    /// restart the seeded stream. Token-exact vs a fresh [`LmGen::new`] over
    /// the same weights (gated).
    pub fn reset_session(&mut self) {
        self.temporal.reset();
        self.stream.reset();
    }

    /// moshi `LMGen.step`: token-fed step. During prompts the caller passes
    /// `moshi_tokens` (agent audio) + `text_token`; during generation only
    /// `input_tokens` (user audio). Returns `None` until the machine is past
    /// its delay horizon (and for the offset-0 seeding call).
    pub fn step(
        &mut self,
        input_tokens: Option<&[i64; 8]>,
        moshi_tokens: Option<&[i64; 8]>,
        text_token: Option<i64>,
        device: &B::Device,
    ) -> StepTrace {
        let Some(p) = self.stream.prepare(input_tokens, moshi_tokens, text_token) else {
            return StepTrace {
                input: None,
                next_text: -1,
                dep_tokens: [0; cfg::DEP_Q],
                out: None,
            };
        };
        let input = p.input;
        let x = self.temporal.embed_codes(&input, device);
        self.forward(x, p, Some(input))
    }

    /// moshi `LMGen.step_embeddings`: voice-prompt replay step. The cache is
    /// still advanced with dummy initial tokens + PAD text (the oracle's
    /// `_dummy_audio_token` writes), but the temporal stack consumes the
    /// pre-recorded embedding.
    pub fn step_embeddings(&mut self, x: Tensor<B, 3>) -> StepTrace {
        let dummy = [cfg::CARD as i64; 8];
        let p = loop {
            if let Some(p) =
                self.stream
                    .prepare(Some(&dummy), Some(&dummy), Some(cfg::TEXT_PAD_TOKEN as i64))
            {
                break p;
            }
        };
        self.forward(x, p, None)
    }

    fn forward(
        &mut self,
        x: Tensor<B, 3>,
        p: Prepared,
        input: Option<[i64; cfg::NUM_STREAMS]>,
    ) -> StepTrace {
        let device = x.device();
        let (hidden, text_logits) = self.temporal.forward_embeddings(x, &device);
        let tl: Vec<f32> = text_logits.into_data().to_vec::<f32>().unwrap();
        // Text stream: sample (when configured) then audio streams — one RNG
        // threaded text-first-then-audio per frame, so a seed is reproducible.
        let sampled_text = match self.sampler.as_mut() {
            Some(smp) => smp.token(&tl) as i64,
            None => argmax(&tl) as i64,
        };
        let next_text = if p.provided[0] {
            p.target[0]
        } else {
            sampled_text
        };
        let (dep_tokens, _) = self.depth.frame(
            &hidden,
            next_text,
            &p.forced(),
            None,
            self.sampler.as_mut(),
            &device,
        );
        let out = self.stream.commit(&p, sampled_text, &dep_tokens);
        StepTrace {
            input,
            next_text,
            dep_tokens,
            out,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // A deterministic pseudo-token generator for driving the pure integer
    // StreamCache without any model (LCG, in the audio codebook range).
    fn lcg(seed: u64) -> impl FnMut() -> i64 {
        let mut s = seed;
        move || {
            s = s
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((s >> 40) as i64) % (cfg::CARD as i64)
        }
    }

    // Drive `n` steps of the pure integer machine with a fixed pseudo stream
    // (user audio + sampled text/audio via the same LCG), collecting every
    // emitted output frame. Deterministic given `seed`.
    fn run(cache: &mut StreamCache, n: usize, seed: u64) -> Vec<[i64; cfg::NUM_STREAMS]> {
        let mut rnd = lcg(seed);
        let mut outs = Vec::new();
        for _ in 0..n {
            let user: [i64; 8] = std::array::from_fn(|_| rnd());
            match cache.prepare(Some(&user), None, None) {
                None => {} // offset-0 seeding step
                Some(p) => {
                    let text = rnd();
                    let audio: [i64; cfg::DEP_Q] = std::array::from_fn(|_| rnd());
                    if let Some(o) = cache.commit(&p, text, &audio) {
                        outs.push(o);
                    }
                }
            }
        }
        outs
    }

    /// `reset()` returns the ring to the pristine state, so a reset-then-run
    /// reproduces a fresh-then-run token-for-token (the integer core of the
    /// pipeline's `reset_session == reload` guarantee, model-free).
    #[test]
    fn stream_reset_equals_fresh() {
        // A fresh cache's trajectory.
        let mut fresh = StreamCache::new();
        let want = run(&mut fresh, 40, 123);

        // A used cache (different seed, so a different trajectory) then reset.
        let mut reused = StreamCache::new();
        let _ = run(&mut reused, 25, 999);
        assert!(reused.offset() > 0, "precondition: cache was actually used");
        reused.reset();
        assert_eq!(reused.offset(), 0, "reset zeroes the offset");
        let got = run(&mut reused, 40, 123);

        assert_eq!(got, want, "reset-then-run must match fresh-then-run");
        assert!(!want.is_empty(), "sanity: the run emitted frames");
    }

    /// `reset()` is idempotent with `StreamCache::new()` — offset and the first
    /// few emitted frames agree after either.
    #[test]
    fn reset_matches_new() {
        let mut a = StreamCache::new();
        let mut b = StreamCache::new();
        let _ = run(&mut b, 10, 7);
        b.reset();
        assert_eq!(a.offset(), b.offset());
        assert_eq!(run(&mut a, 30, 55), run(&mut b, 30, 55));
    }
}
