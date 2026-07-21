//! Voxtral-Mini-4B-Realtime-2602 — delay-conditioned streaming
//! speech-to-text (one AR step per 80 ms frame, text delayed 80 ms–2.4 s
//! behind the audio via ada-RMS-norm conditioning). Architecture + port log:
//! `docs/VOXTRAL_ARCH.md`. Weights: `models/voxtral_mini.pile`.

pub mod config;
pub mod decoder;
pub mod encoder;
pub mod fast;
pub mod layers;
pub mod mel;
pub mod pipeline;
pub mod tokenizer;
