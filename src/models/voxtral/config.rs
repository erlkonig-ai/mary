//! Voxtral-Mini-4B-Realtime-2602 geometry + token constants — all measured
//! from `config.json` / `params.json` / `tekken.json` (see docs/VOXTRAL_ARCH.md).

// ── mel front-end ──────────────────────────────────────────────────────────
pub const SAMPLE_RATE: usize = 16000;
pub const N_FFT: usize = 400; // = win_length (25 ms), hann periodic
pub const HOP: usize = 160; // 10 ms
pub const MEL_BINS: usize = 128;
pub const FMAX: f64 = 8000.0;
pub const GLOBAL_LOG_MEL_MAX: f32 = 1.5;

// ── audio encoder (whisper-shaped, causal, RoPE, RMS, SwiGLU) ─────────────
pub const ENC_LAYERS: usize = 32;
pub const ENC_HIDDEN: usize = 1280;
pub const ENC_HEADS: usize = 32; // MHA (kv == heads)
pub const ENC_HEAD_DIM: usize = 64; // attention dim 2048 != hidden 1280
pub const ENC_MLP: usize = 5120;
pub const ENC_WINDOW: usize = 750; // sliding window (~15 s at 50 Hz)

// ── projector ──────────────────────────────────────────────────────────────
pub const DOWNSAMPLE: usize = 4; // stack 4 × 1280 → 5120 per 80 ms token

// ── decoder (Ministral-3-3B) ───────────────────────────────────────────────
pub const DEC_LAYERS: usize = 26;
pub const DEC_HIDDEN: usize = 3072;
pub const DEC_HEADS: usize = 32;
pub const DEC_KV_HEADS: usize = 8;
pub const DEC_HEAD_DIM: usize = 128;
pub const DEC_MLP: usize = 9216;
pub const DEC_WINDOW: usize = 8192;
pub const VOCAB: usize = 131072;
pub const ADA_DIM: usize = 32; // ada_rms_norm bottleneck
pub const TIME_THETA: f64 = 10000.0; // delay time-embedding theta

pub const EPS: f64 = 1e-5;
pub const ROPE_THETA: f64 = 1_000_000.0;

// ── tokens (tekken: ids 0..999 special, base vocab at id+1000) ────────────
pub const BOS: u32 = 1;
pub const EOS: u32 = 2;
pub const STREAMING_PAD: u32 = 32;

// ── streaming schedule ────────────────────────────────────────────────────
/// One text token per 80 ms of audio.
pub const SAMPLES_PER_TOK: usize = 1280;
pub const MEL_PER_TOK: usize = 8;
/// Silence prepended to every stream ("more compute", from mistral_common).
pub const N_LEFT_PAD_TOKENS: usize = 32;
/// Extra right padding in offline mode beyond (delay + 1) tokens.
pub const OFFLINE_BUFFER_TOKENS: usize = 10;

/// Delay in tokens for a delay in ms (multiples of 80 between 80..=1200, or 2400).
pub fn delay_tokens(delay_ms: usize) -> usize {
    assert!(delay_ms % 80 == 0, "delay must be a multiple of 80 ms");
    delay_ms / 80
}
