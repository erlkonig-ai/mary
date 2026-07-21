//! PersonaPlex end-to-end voice pipeline — LM part 3: the audio-out
//! integration around [`LmGen`], wiring input WAV → Mimi encode (user codes)
//! → temporal + depth free-run → agent codes → Mimi decode → 24 kHz PCM.
//!
//! Stream semantics (moshi `offline.py`): of the 17 undelayed output streams
//! `[text, agent 1..8, user 1..8]`, ONLY the agent audio streams `1..=8` are
//! Mimi-decoded (`decode_tokens_to_pcm` slices `out[:, 1:9]`); stream 0 is
//! text, streams 9..=16 are the model's prediction of the USER's audio and
//! are never decoded.
//!
//! **The skip / partial-frame rule** (moshi `LMGen.step`, read from lm.py —
//! there is no partial frame, only `None`):
//!
//! 1. the very first call (`offset == 0`) seeds the token ring with initial
//!    tokens and returns `None` — no model step at all
//!    (`prepare_step_input`'s `state.offset == 0` branch);
//! 2. while `offset <= max_delay` (= 1 here) the model steps but
//!    `process_transformer_output` still returns `None` — the undelayed read
//!    position of the acoustic (delay-1) streams has not been written yet;
//! 3. from `offset == max_delay + 1` onward EVERY step returns a complete
//!    frame, `out[k] = cache[k][(offset − max_delay + delay_k) % CT]`. By
//!    construction each emitted slot has been written by a sampled or
//!    provided token — initial (2048) and ungenerated (−2) tokens never
//!    surface. [`agent_codes`] asserts that invariant before handing codes
//!    to the decoder.
//!
//! In the production prompt flow (voice prompt → silence → text → silence →
//! user audio) both `None` steps happen inside the voice-prompt phase, so
//! every user-audio step yields a decodable frame; prompt-phase frames are
//! discarded (the oracle's `_step_*` helpers ignore them too).
//!
//! CPU-f32 parity path (`personaplex_probe pipeline` gates the chain against
//! the oracle's `out_audio` golden). The realtime build is
//! [`RealtimePipeline`] (feature `q4`): same step semantics on the fast
//! stages — Metal quantized temporal + Accelerate/NEON depformer + CPU Mimi
//! — gated by `personaplex_rt_probe pipeline`.

use burn::prelude::*;

use super::config as cfg;
use super::depth::DepthTransformer;
use super::lmgen::LmGen;
use super::mimi::config as mimi_cfg;
use super::mimi::{MimiDecoder, MimiEncoder};
use super::temporal::TemporalTransformer;
use crate::nn::weight_loader::WeightLoader;

#[cfg(feature = "q4")]
use super::depth::argmax;
#[cfg(feature = "q4")]
use super::depth_fast::DepthFast;
#[cfg(feature = "q4")]
use super::lmgen::{Prepared, StreamCache};
#[cfg(feature = "q4")]
use super::temporal_metal::{Head, TemporalMetal, WeightFmt};

/// Agent-side spacer frame (`lm.py SILENCE_TOKENS`) as step input.
pub const SILENCE: [i64; 8] = {
    let mut a = [0i64; 8];
    let mut i = 0;
    while i < 8 {
        a[i] = cfg::SILENCE_TOKENS[i] as i64;
        i += 1;
    }
    a
};
/// User-side reference sine frame (`lm.py SINE_TOKENS`) as step input.
pub const SINE: [i64; 8] = {
    let mut a = [0i64; 8];
    let mut i = 0;
    while i < 8 {
        a[i] = cfg::SINE_TOKENS[i] as i64;
        i += 1;
    }
    a
};

/// The full voice pipeline: Mimi encoder (user audio in), the 7B LM step
/// machine, and the Mimi decoder (agent audio out). All three load from the
/// one union weight pile (`models/personaplex.pile`).
pub struct VoicePipeline<B: Backend> {
    pub lm: LmGen<B>,
    pub encoder: MimiEncoder,
    pub decoder: MimiDecoder,
}

impl<B: Backend> VoicePipeline<B> {
    pub fn load(loader: &WeightLoader, device: &B::Device) -> Self {
        Self {
            lm: LmGen::new(
                TemporalTransformer::load(loader, device),
                DepthTransformer::load(loader, device),
            ),
            encoder: MimiEncoder::load(loader),
            decoder: MimiDecoder::load(loader),
        }
    }

    /// Switch the step machine onto seeded sampling (`cfg` + `seed`) for both
    /// the text and audio heads. A greedy `cfg` reproduces argmax exactly.
    pub fn set_sampling(&mut self, cfg: super::sampling::SamplingConfig, seed: u64) {
        self.lm.set_sampling(cfg, seed);
    }

    /// Drop back to greedy argmax.
    pub fn set_greedy(&mut self) {
        self.lm.set_greedy();
    }

    /// Reset the streaming state so a new conversation can start WITHOUT
    /// reloading weights (see [`LmGen::reset_session`]). Token-exact vs a
    /// fresh [`VoicePipeline::load`] of the same weights.
    pub fn reset_session(&mut self) {
        self.lm.reset_session();
    }

    /// Voice-prompt replay (moshi `_step_voice_prompt` on a packaged voice):
    /// feed the pre-recorded embeddings (`[n, 4096]` row-major) one step at a
    /// time, then overwrite the token ring with the packaged cache snapshot
    /// (`[17, CT]` row-major — the oracle's `state.cache.copy_(...)`).
    pub fn prompt_voice(&mut self, embeddings: &[f32], cache_snapshot: &[i64], device: &B::Device) {
        assert_eq!(
            embeddings.len() % cfg::DIM,
            0,
            "vp embeddings not row-aligned"
        );
        for row in embeddings.chunks_exact(cfg::DIM) {
            let x = Tensor::<B, 1>::from_floats(row, device).reshape([1, 1, cfg::DIM]);
            self.lm.step_embeddings(x);
        }
        self.lm.stream.overwrite(cache_snapshot);
    }

    /// Silence spacer (moshi `_step_audio_silence`): `frames` steps of agent
    /// SILENCE + user SINE + text PAD.
    pub fn prompt_silence(&mut self, frames: usize, device: &B::Device) {
        for _ in 0..frames {
            self.lm.step(
                Some(&SINE),
                Some(&SILENCE),
                Some(cfg::TEXT_PAD_TOKEN as i64),
                device,
            );
        }
    }

    /// Text system prompt (moshi `_step_text_prompt`): one token per step,
    /// agent SILENCE + user SINE alongside.
    pub fn prompt_text(&mut self, tokens: &[i64], device: &B::Device) {
        for &t in tokens {
            self.lm.step(Some(&SINE), Some(&SILENCE), Some(t), device);
        }
    }

    /// The full system-prompt flow from an assembled [`super::prompt::Prompt`]
    /// (moshi `step_system_prompts`): voice replay → silence → text → silence.
    pub fn run_prompt(&mut self, p: &super::prompt::Prompt, device: &B::Device) {
        self.prompt_voice(&p.voice.embeddings, &p.voice.cache, device);
        self.prompt_silence(p.silence_frames, device);
        self.prompt_text(&p.text_tokens, device);
        self.prompt_silence(p.silence_frames, device);
    }

    /// One user-audio frame (8 Mimi codes) in → the undelayed 17-stream
    /// output frame, once past the delay horizon (see the skip rule in the
    /// module docs — in the prompt flow this is always `Some` by the time
    /// user audio starts). Feed [`agent_codes`] of it to [`Self::decode`].
    pub fn step_user_frame(
        &mut self,
        codes: &[i64; 8],
        device: &B::Device,
    ) -> Option<[i64; cfg::NUM_STREAMS]> {
        self.lm.step(Some(codes), None, None, device).out
    }

    /// Agent frames → 24 kHz mono PCM (1920 samples per frame).
    pub fn decode(&self, frames: &[[u32; mimi_cfg::NUM_CODEBOOKS]]) -> Vec<f32> {
        self.decoder.decode(frames)
    }
}

/// One [`RealtimePipeline`] step's trace — the [`super::lmgen::StepTrace`]
/// fields plus the read-back text logits (the probe gates them; a production
/// loop can ignore them — they are read back anyway for the host argmax).
#[cfg(feature = "q4")]
pub struct RtStepTrace {
    /// The 17 input tokens fed to the temporal stack (`None` for
    /// embedding-fed voice-prompt steps and the offset-0 seeding call).
    pub input: Option<[i64; cfg::NUM_STREAMS]>,
    /// `next_text_token` handed to the depformer (forced target or sampled).
    pub next_text: i64,
    /// The depformer's 16 greedy tokens.
    pub dep_tokens: [i64; cfg::DEP_Q],
    /// Undelayed output frame, once past the delay horizon.
    pub out: Option<[i64; cfg::NUM_STREAMS]>,
    /// The temporal stack's text logits `[32000]` (empty for the offset-0
    /// seeding call).
    pub text_logits: Vec<f32>,
}

/// The **realtime** voice pipeline: the same step semantics as [`VoicePipeline`]
/// (the CPU-f32 parity oracle) rebuilt on the fast stages — the Metal
/// quantized temporal transformer ([`TemporalMetal`], q4/q8/f16 stack with
/// the f16 logit head as the documented production choice), the
/// Accelerate/NEON CPU depformer predictor ([`DepthFast`]), and the CPU Mimi
/// codec.
///
/// **Numerics honesty:** with a quantized (q4/q8) temporal stack this is a
/// REAL numerics change — free-run token streams are expected to diverge
/// from the f32 oracle at some step (an argmax near-tie flips, then the
/// autoregressive paths separate). `personaplex_rt_probe pipeline` measures
/// agreement %, the first-divergence step, and the logits at the divergence
/// point; the f16 stack is the wiring-exactness ablation. Never claim
/// token-exactness for the quantized formats.
///
/// **Mimi threading:** Mimi runs sequentially in this build (batch encode up
/// front, batch decode at the end in the golden gate flow; ~5 ms/frame if
/// decoded per frame in a live duplex loop). Moving it onto its own thread
/// (as in `qwen3tts_stream`) is deferred until a live session loop exists to
/// consume streaming frames — the LM step machine here is synchronous, so a
/// decode thread would only overlap the ~5 ms Mimi cost, not the LM.
#[cfg(feature = "q4")]
pub struct RealtimePipeline {
    pub temporal: TemporalMetal,
    pub depth: DepthFast,
    pub stream: StreamCache,
    pub encoder: MimiEncoder,
    pub decoder: MimiDecoder,
    /// Logit head variant (f16 is the measured production choice — see
    /// `temporal_metal` module docs).
    pub head: Head,
    /// `None` = greedy argmax (parity). `Some` = seeded temp/top-k/top-p over
    /// the text head and each depformer audio step (the quality path).
    sampler: Option<super::sampling::Sampler>,
}

#[cfg(feature = "q4")]
impl RealtimePipeline {
    /// Load all components from the one union weight pile. `fmt` picks the
    /// temporal stack's weight format (q4/q8/f16); `depth_f16` stores the
    /// depformer weights as f16 (f32 accumulate) instead of f32.
    pub fn load(loader: &WeightLoader, fmt: WeightFmt, depth_f16: bool) -> Self {
        Self {
            temporal: TemporalMetal::load(loader, fmt),
            depth: DepthFast::load(loader, depth_f16),
            stream: StreamCache::new(),
            encoder: MimiEncoder::load(loader),
            decoder: MimiDecoder::load(loader),
            head: Head::F16,
            sampler: None,
        }
    }

    /// [`Self::load`] with derived-sibling AUTO-DISCOVERY (see
    /// [`super::qpile`]): the temporal stack zero-copy-mmaps
    /// `<stem>_<fmt>.pile` and the depformer `<stem>_depth.pile` when they
    /// exist and carry the current format marker; each component falls back to
    /// its transform-at-load path independently otherwise
    /// (`MARY_PPLX_MATERIALIZE=1` forces both fallbacks — the A/B switch).
    /// Identical outputs on every path; only load time / RSS differ.
    #[cfg(target_os = "macos")]
    pub fn load_auto(
        pile_path: &std::path::Path,
        loader: &WeightLoader,
        fmt: WeightFmt,
        depth_f16: bool,
    ) -> Self {
        Self {
            temporal: super::qpile::temporal_auto(pile_path, loader, fmt),
            depth: super::qpile::depth_auto(pile_path, loader, depth_f16),
            stream: StreamCache::new(),
            encoder: MimiEncoder::load(loader),
            decoder: MimiDecoder::load(loader),
            head: Head::F16,
            sampler: None,
        }
    }

    /// Switch the realtime step machine onto seeded sampling (`cfg` + `seed`)
    /// for both the temporal text head and the depformer audio heads. A greedy
    /// `cfg` reproduces argmax exactly.
    pub fn set_sampling(&mut self, cfg: super::sampling::SamplingConfig, seed: u64) {
        self.sampler = Some(super::sampling::Sampler::new(cfg, seed));
    }

    /// Drop back to greedy argmax (clears the sampler).
    pub fn set_greedy(&mut self) {
        self.sampler = None;
    }

    /// Reset the streaming state so a new conversation can start WITHOUT
    /// reloading weights (moshi's per-session `streaming` reset): clears the
    /// temporal KV cache + offset, the stream ring/offset, AND re-seeds an
    /// existing sampler back to its original seed — so a reset session samples
    /// the identical token stream as the first, no `set_sampling` re-call
    /// needed. Token-exact vs a fresh [`RealtimePipeline::load`] of the same
    /// weights (gated).
    pub fn reset_session(&mut self) {
        self.temporal.reset();
        self.stream.reset();
        if let Some(smp) = self.sampler.as_mut() {
            smp.reseed();
        }
    }

    fn forward(
        &mut self,
        x: &[f32],
        p: Prepared,
        input: Option<[i64; cfg::NUM_STREAMS]>,
    ) -> RtStepTrace {
        self.temporal.step_submit(x, self.head);
        let (hidden, text_logits) = self.temporal.read_hidden_logits();
        // Text first, then audio — one RNG threaded per frame (reproducible).
        let sampled_text = match self.sampler.as_mut() {
            Some(smp) => smp.token(&text_logits) as i64,
            None => argmax(&text_logits) as i64,
        };
        let next_text = if p.provided[0] { p.target[0] } else { sampled_text };
        let dep_tokens =
            self.depth.frame(&hidden, next_text, &p.forced(), None, self.sampler.as_mut());
        let out = self.stream.commit(&p, sampled_text, &dep_tokens);
        RtStepTrace { input, next_text, dep_tokens, out, text_logits }
    }

    /// moshi `LMGen.step` on the fast stages (see [`super::lmgen::LmGen::step`]
    /// for the contract).
    pub fn step(
        &mut self,
        input_tokens: Option<&[i64; 8]>,
        moshi_tokens: Option<&[i64; 8]>,
        text_token: Option<i64>,
    ) -> RtStepTrace {
        let Some(p) = self.stream.prepare(input_tokens, moshi_tokens, text_token) else {
            return RtStepTrace {
                input: None,
                next_text: -1,
                dep_tokens: [0; cfg::DEP_Q],
                out: None,
                text_logits: Vec::new(),
            };
        };
        let input = p.input;
        let x = self.temporal.embed_codes(&input);
        self.forward(&x, p, Some(input))
    }

    /// Voice-prompt replay step: the cache advances with dummy initial
    /// tokens + PAD text while the temporal stack consumes the pre-recorded
    /// embedding row `[4096]` (see [`super::lmgen::LmGen::step_embeddings`]).
    pub fn step_embedding(&mut self, x: &[f32]) -> RtStepTrace {
        let dummy = [cfg::CARD as i64; 8];
        let p = loop {
            if let Some(p) =
                self.stream.prepare(Some(&dummy), Some(&dummy), Some(cfg::TEXT_PAD_TOKEN as i64))
            {
                break p;
            }
        };
        self.forward(x, p, None)
    }

    /// Voice-prompt replay (moshi `_step_voice_prompt`): feed the packaged
    /// embeddings (`[n, 4096]` row-major), then overwrite the token ring
    /// with the packaged cache snapshot (`[17, CT]` row-major).
    pub fn prompt_voice(&mut self, embeddings: &[f32], cache_snapshot: &[i64]) {
        assert_eq!(embeddings.len() % cfg::DIM, 0, "vp embeddings not row-aligned");
        for row in embeddings.chunks_exact(cfg::DIM) {
            self.step_embedding(row);
        }
        self.stream.overwrite(cache_snapshot);
    }

    /// Silence spacer (moshi `_step_audio_silence`).
    pub fn prompt_silence(&mut self, frames: usize) {
        for _ in 0..frames {
            self.step(Some(&SINE), Some(&SILENCE), Some(cfg::TEXT_PAD_TOKEN as i64));
        }
    }

    /// Text system prompt (moshi `_step_text_prompt`).
    pub fn prompt_text(&mut self, tokens: &[i64]) {
        for &t in tokens {
            self.step(Some(&SINE), Some(&SILENCE), Some(t));
        }
    }

    /// The full system-prompt flow from an assembled [`super::prompt::Prompt`]
    /// (moshi `step_system_prompts`): voice replay → silence → text → silence.
    pub fn run_prompt(&mut self, p: &super::prompt::Prompt) {
        self.prompt_voice(&p.voice.embeddings, &p.voice.cache);
        self.prompt_silence(p.silence_frames);
        self.prompt_text(&p.text_tokens);
        self.prompt_silence(p.silence_frames);
    }

    /// One user-audio frame (8 Mimi codes) in → the undelayed 17-stream
    /// output frame (always `Some` once the prompt flow consumed the `None`
    /// steps). Feed [`agent_codes`] of it to [`Self::decode`].
    pub fn step_user_frame(&mut self, codes: &[i64; 8]) -> Option<[i64; cfg::NUM_STREAMS]> {
        self.step(Some(codes), None, None).out
    }

    /// Agent frames → 24 kHz mono PCM (1920 samples per frame).
    pub fn decode(&self, frames: &[[u32; mimi_cfg::NUM_CODEBOOKS]]) -> Vec<f32> {
        self.decoder.decode(frames)
    }
}

/// Extract the agent's Mimi codes (streams `1..=8`) from an undelayed output
/// frame. Streams `9..=16` (user prediction) are never decoded; stream 0 is
/// text. Asserts the delay machinery's invariant: emitted frames contain
/// only real codebook entries — an initial (2048) or ungenerated token here
/// means the skip rule was violated upstream.
pub fn agent_codes(out: &[i64; cfg::NUM_STREAMS]) -> [u32; mimi_cfg::NUM_CODEBOOKS] {
    std::array::from_fn(|q| {
        let t = out[1 + q];
        assert!(
            (0..cfg::CARD as i64).contains(&t),
            "agent stream {} token {t} outside the Mimi codebook — an initial/ungenerated \
             token leaked past the delay horizon",
            1 + q
        );
        t as u32
    })
}
