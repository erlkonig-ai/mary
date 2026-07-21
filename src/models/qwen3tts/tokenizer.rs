//! Qwen2 byte-level BPE, built from the checkpoint's `vocab.json` +
//! `merges.txt` (no tokenizer.json ships with the TTS repo). The Qwen2
//! pre-tokenizer regex differs from GPT-2's (case-insensitive contractions,
//! single-digit numerals), so we assemble Split(qwen2-regex) → ByteLevel
//! explicitly. The 33 added tokens (151643..151675, `<|im_start|>`,
//! `<tts_pad>`, …) are appended in id order so ids line up exactly.
//! Parity-gated against the reference tokenizer in `qwen3tts_probe`.

use std::path::Path;

use tokenizers::models::bpe::BPE;
use tokenizers::normalizers::unicode::NFC;
use tokenizers::pre_tokenizers::byte_level::ByteLevel;
use tokenizers::pre_tokenizers::sequence::Sequence;
use tokenizers::pre_tokenizers::split::Split;
use tokenizers::{AddedToken, SplitDelimiterBehavior, Tokenizer};

/// Qwen2 pre-tokenization regex (from tokenization_qwen2.py PRETOKENIZE_REGEX).
const QWEN2_RE: &str = r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+";

const ADDED: [&str; 33] = [
    "<|endoftext|>", "<|im_start|>", "<|im_end|>", "<|object_ref_start|>",
    "<|object_ref_end|>", "<|box_start|>", "<|box_end|>", "<|quad_start|>",
    "<|quad_end|>", "<|vision_start|>", "<|vision_end|>", "<|vision_pad|>",
    "<|image_pad|>", "<|video_pad|>", "<tool_call>", "</tool_call>",
    "<|fim_prefix|>", "<|fim_middle|>", "<|fim_suffix|>", "<|fim_pad|>",
    "<|repo_name|>", "<|file_sep|>", "<tool_response>", "</tool_response>",
    "<think>", "</think>", "<|audio_start|>", "<|audio_end|>", "<tts_pad>",
    "<tts_text_bos>", "<tts_text_eod>", "<tts_text_bos_single>", "<|audio_pad|>",
];

pub struct TextTokenizer {
    tok: Tokenizer,
}

impl TextTokenizer {
    /// `dir` = checkpoint dir holding vocab.json + merges.txt.
    pub fn load(dir: &Path) -> Self {
        let bpe = BPE::from_file(
            dir.join("vocab.json").to_str().unwrap(),
            dir.join("merges.txt").to_str().unwrap(),
        )
        .build()
        .expect("build BPE");
        let mut tok = Tokenizer::new(bpe);
        tok.with_normalizer(Some(NFC));
        let split = Split::new(
            tokenizers::pre_tokenizers::split::SplitPattern::Regex(QWEN2_RE.into()),
            SplitDelimiterBehavior::Isolated,
            false,
        )
        .expect("qwen2 split regex");
        tok.with_pre_tokenizer(Some(Sequence::new(vec![
            split.into(),
            ByteLevel::new(false, false, false).into(),
        ])));
        tok.add_special_tokens(
            &ADDED
                .iter()
                .map(|s| AddedToken::from(s.to_string(), true))
                .collect::<Vec<_>>(),
        );
        Self { tok }
    }

    pub fn encode(&self, text: &str) -> Vec<u32> {
        self.tok
            .encode(text, false)
            .expect("tokenize")
            .get_ids()
            .to_vec()
    }
}
