//! Measured dimensions of Qwen3-TTS-12Hz-1.7B-Base — from the checkpoint
//! `config.json` + `speech_tokenizer/config.json` (see PORT_NOTES.md), not the
//! paper. Hardcoded house-style: the checkpoint is the config.

/// Talker backbone (the codec-frame LM).
pub const TALKER_LAYERS: usize = 28;
pub const TALKER_HIDDEN: usize = 2048;
pub const TALKER_HEADS: usize = 16;
pub const TALKER_KV_HEADS: usize = 8;
pub const TALKER_HEAD_DIM: usize = 128;
pub const TALKER_EPS: f64 = 1e-6;
pub const TALKER_ROPE_THETA: f64 = 1_000_000.0;
pub const CODEC_VOCAB: usize = 3072;

/// Code predictor (sub-talker MTP head over codebooks 1..15).
pub const PRED_LAYERS: usize = 5;
pub const PRED_HIDDEN: usize = 1024;
pub const PRED_HEADS: usize = 16;
pub const PRED_KV_HEADS: usize = 8;
pub const PRED_HEAD_DIM: usize = 128;
pub const PRED_VOCAB: usize = 2048;

/// Codebooks per 12.5 Hz frame.
pub const NUM_CODE_GROUPS: usize = 16;

/// Codec control ids inside the 3072 codec vocab.
pub const CODEC_PAD: u32 = 2148;
pub const CODEC_BOS: u32 = 2149;
pub const CODEC_EOS: u32 = 2150;
pub const CODEC_THINK: u32 = 2154;
pub const CODEC_NOTHINK: u32 = 2155;
pub const CODEC_THINK_BOS: u32 = 2156;
pub const CODEC_THINK_EOS: u32 = 2157;
/// codec_language_id (config.json); "auto" = no language tag.
pub const LANG_ENGLISH: u32 = 2050;
pub const LANG_GERMAN: u32 = 2053;

/// Text-side special token ids (Qwen2 BPE vocab).
pub const IM_START: u32 = 151644;
pub const IM_END: u32 = 151645;
pub const TTS_PAD: u32 = 151671;
pub const TTS_BOS: u32 = 151672;
pub const TTS_EOS: u32 = 151673;

/// Codec decoder (speech_tokenizer, 12 Hz tokenizer v2).
pub const DEC_CODEBOOK_SIZE: usize = 2048;
pub const DEC_CODE_DIM: usize = 256; // per-quantizer embedding dim
pub const DEC_QUANT_OUT: usize = 512; // codebook_dim (post output_proj)
pub const DEC_LATENT: usize = 1024;
pub const DEC_HIDDEN: usize = 512; // pre_transformer width
pub const DEC_TR_LAYERS: usize = 8;
pub const DEC_TR_HEADS: usize = 16;
pub const DEC_TR_HEAD_DIM: usize = 64;
pub const DEC_TR_WINDOW: usize = 72;
pub const DEC_TR_EPS: f64 = 1e-5;
pub const DEC_TR_ROPE_THETA: f64 = 10_000.0;
pub const DEC_DECODER_DIM: usize = 1536;
pub const DEC_UPSAMPLE_RATES: [usize; 4] = [8, 5, 4, 3];
pub const DEC_UPSAMPLING_RATIOS: [usize; 2] = [2, 2];
/// Samples per frame at 24 kHz: 8·5·4·3·2·2 = 1920 (12.5 Hz frames).
pub const SAMPLES_PER_FRAME: usize = 1920;
pub const SAMPLE_RATE: u32 = 24_000;
