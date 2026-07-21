//! PersonaPlex-7B full-duplex model port. Phase 1: the Mimi neural audio codec
//! ([`mimi`]). Phase 0 of the LM port: verified architecture constants in
//! [`config`] (from the real checkpoint's tensor shapes) and the weight pile
//! (`personaplex_persist`). LM part 1: the 7B [`temporal`] transformer
//! forward. LM part 2: the [`depth`] transformer (depformer) and the
//! [`lmgen`] delay/undelay step machinery (all CPU-f32 parity vs the moshi
//! oracle — `personaplex_probe`). LM part 3: the end-to-end [`pipeline`] —
//! input WAV → Mimi encode → LM free-run → agent streams 1..=8 → Mimi decode
//! → 24 kHz audio out. Realtime lane A: the [`temporal_metal`] q4/Metal
//! decode build (feature `q4`; gated by `personaplex_rt_probe`). Realtime
//! lane B: [`depth_fast`] — the Accelerate/NEON CPU depformer predictor
//! (preloaded per-step weight sets, fixed buffers, optional f16 storage with
//! f32 accumulate; gate+bench `moshi_depth_probe`). Phase 5: the prompt
//! machinery — [`spm`] (pure-Rust SentencePiece unigram text tokenizer, encode
//! + decode), [`voice_prompt`] (packaged voice `.pt` reader) and [`prompt`]
//! (system prompts assembled from primary sources instead of golden npys;
//! gated by `personaplex_probe prompt`). Realtime foundation: [`sampling`]
//! (seedable temperature / top-k / top-p over the text + audio heads; greedy
//! stays the parity default) and the `reset_session` seam on the [`lmgen`] /
//! [`pipeline`] step machines (a new conversation without a weight reload).

pub mod config;
pub mod depth;
pub mod depth_fast;
pub mod lmgen;
pub mod mimi;
pub mod pipeline;
pub mod prompt;
pub mod sampling;
// Derived runtime-format sibling piles (zero-copy load seam): the format
// marker/ABI, the derive step, and the auto-discovery loaders.
#[cfg(all(feature = "q4", target_os = "macos"))]
pub mod qpile;
pub mod spm;
pub mod temporal;
#[cfg(feature = "q4")]
pub mod temporal_metal;
pub mod voice_prompt;
