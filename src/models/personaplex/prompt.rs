//! PersonaPlex prompt assembly — Phase 5: mary builds the full system-prompt
//! flow from PRIMARY sources (a packaged voice `.pt` + a raw system-prompt
//! string), replacing the golden-npy feeds the parity gates bootstrapped
//! with. The flow (moshi `offline.py` → `LMGen.step_system_prompts`):
//!
//! 1. **voice prompt** — [`super::voice_prompt::VoicePrompt`] embedding
//!    replay (N steps of pre-recorded input embeddings, bypassing the
//!    embedding tables), then the packaged token-ring snapshot overwrites
//!    the `StreamCache`;
//! 2. **silence** — `int(0.5 s × 12.5 Hz)` = [`SILENCE_FRAMES`] steps of
//!    agent `SILENCE_TOKENS` + user `SINE_TOKENS` + text PAD;
//! 3. **text prompt** — [`wrap_with_system_tags`] → [`super::spm`] tokens,
//!    one per step on stream 0 (SILENCE/SINE alongside);
//! 4. **silence** again.
//!
//! Gated end-to-end in `personaplex_probe prompt`: the assembled 113-step
//! model-input stream must equal `step_tokens`/`step_token_idx` exactly.

use std::path::Path;

use super::spm::SpmTokenizer;
use super::voice_prompt::VoicePrompt;

/// `int(0.5 * frame_rate)` silence-spacer frames (offline.py
/// `audio_silence_frame_cnt` at 12.5 Hz).
pub const SILENCE_FRAMES: usize = 6;

/// moshi `offline.py wrap_with_system_tags`: add `<system>` tags if missing.
/// The tags are NOT special tokens — they tokenize as `▁<`+`system`+`>`.
pub fn wrap_with_system_tags(text: &str) -> String {
    let cleaned = text.trim();
    if cleaned.starts_with("<system>") && cleaned.ends_with("<system>") {
        cleaned.to_string()
    } else {
        format!("<system> {cleaned} <system>")
    }
}

/// A fully assembled system prompt: everything the prompt phases need,
/// built from primary sources.
pub struct Prompt {
    pub voice: VoicePrompt,
    /// SPM tokens of the `<system>`-wrapped prompt text (empty text → no
    /// text-prompt phase, mirroring offline.py's `None`).
    pub text_tokens: Vec<i64>,
    pub silence_frames: usize,
}

impl Prompt {
    pub fn build(voice_pt: &Path, spm: &SpmTokenizer, system_text: &str) -> Self {
        let text_tokens = if system_text.is_empty() {
            Vec::new()
        } else {
            spm.encode(&wrap_with_system_tags(system_text))
        };
        Self { voice: VoicePrompt::load(voice_pt), text_tokens, silence_frames: SILENCE_FRAMES }
    }

    /// Total temporal steps the prompt phases consume (the delay horizon's
    /// two `None` steps fall inside the voice phase).
    pub fn total_steps(&self) -> usize {
        self.voice.n_frames + 2 * self.silence_frames + self.text_tokens.len()
    }
}
