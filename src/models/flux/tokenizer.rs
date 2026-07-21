use std::path::Path;
use tokenizers::Tokenizer;

/// Wrapper around HuggingFace tokenizers for Qwen-3 text encoding (FLUX.2-klein).
pub struct Qwen2Tokenizer {
    tokenizer: Tokenizer,
    pub max_length: usize,
}

impl Qwen2Tokenizer {
    /// Load tokenizer from a tokenizer.json file.
    pub fn from_file(path: &Path) -> Self {
        let tokenizer = Tokenizer::from_file(path)
            .unwrap_or_else(|e| panic!("Failed to load tokenizer from {}: {}", path.display(), e));
        Self {
            tokenizer,
            max_length: 512,
        }
    }

    /// Encode a prompt using the Qwen-3 chat template.
    /// Returns (input_ids, attention_mask) both of length max_length.
    pub fn encode_prompt(&self, prompt: &str) -> (Vec<u32>, Vec<u32>) {
        // Apply Qwen-3 chat template manually
        let templated = format!(
            "<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n",
            prompt
        );

        let encoding = self
            .tokenizer
            .encode(templated, true)
            .unwrap_or_else(|e| panic!("Tokenization failed: {}", e));

        let mut ids: Vec<u32> = encoding.get_ids().to_vec();
        let token_len = ids.len();

        // Truncate if needed
        if ids.len() > self.max_length {
            ids.truncate(self.max_length);
        }

        // Build attention mask (1 for real tokens, 0 for padding)
        let real_len = ids.len();
        let mut attention_mask = vec![1u32; real_len];

        // Pad to max_length with pad token (151643 = <|endoftext|> for Qwen2)
        let pad_token_id = self
            .tokenizer
            .token_to_id("<|endoftext|>")
            .unwrap_or(151643);
        while ids.len() < self.max_length {
            ids.push(pad_token_id);
            attention_mask.push(0);
        }

        eprintln!(
            "Tokenized prompt: {} tokens (padded to {})",
            token_len, self.max_length
        );

        (ids, attention_mask)
    }

    /// Get the actual number of tokens (before padding) for a prompt.
    pub fn token_count(&self, prompt: &str) -> usize {
        let templated = format!(
            "<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n",
            prompt
        );
        let encoding = self.tokenizer.encode(templated, true).unwrap();
        encoding.get_ids().len().min(self.max_length)
    }
}

/// Wrapper around HuggingFace tokenizers for Mistral3 text encoding (FLUX.2-dev).
///
/// Applies the Mistral3 chat template with system message, matching the diffusers pipeline:
/// `<s>[SYSTEM_PROMPT]{system_msg}[/SYSTEM_PROMPT][INST]{prompt}[/INST]`
///
/// Uses RIGHT padding (pad tokens after real tokens) with pad_token_id=11.
pub struct MistralTokenizer {
    tokenizer: Tokenizer,
    pub max_length: usize,
}

/// System message used by the diffusers Flux2Pipeline for Dev.
const MISTRAL_SYSTEM_MESSAGE: &str = "You are an AI that reasons about image descriptions. You give structured responses focusing on object relationships, object\nattribution and actions without speculation.";

/// Special token IDs for Mistral3 chat template.
const MISTRAL_BOS: u32 = 1;
const MISTRAL_SYSTEM_PROMPT_START: u32 = 17;
const MISTRAL_SYSTEM_PROMPT_END: u32 = 18;
const MISTRAL_INST_START: u32 = 3;
const MISTRAL_INST_END: u32 = 4;
const MISTRAL_PAD: u32 = 11;

impl MistralTokenizer {
    /// Load tokenizer from a tokenizer.json file.
    pub fn from_file(path: &Path) -> Self {
        let tokenizer = Tokenizer::from_file(path)
            .unwrap_or_else(|e| panic!("Failed to load tokenizer from {}: {}", path.display(), e));
        Self {
            tokenizer,
            max_length: 512,
        }
    }

    /// Encode a prompt for FLUX.2-dev text encoding.
    /// Applies the Mistral3 chat template and returns (input_ids, attention_mask) of length max_length.
    pub fn encode_prompt(&self, prompt: &str) -> (Vec<u32>, Vec<u32>) {
        // Tokenize system message and prompt separately (no special tokens — we add them manually)
        let sys_encoding = self
            .tokenizer
            .encode(MISTRAL_SYSTEM_MESSAGE.to_string(), false)
            .unwrap_or_else(|e| panic!("Tokenization failed: {}", e));
        let prompt_encoding = self
            .tokenizer
            .encode(prompt.to_string(), false)
            .unwrap_or_else(|e| panic!("Tokenization failed: {}", e));

        // Build: [BOS, SYSTEM_PROMPT_START] + sys_tokens + [SYSTEM_PROMPT_END, INST_START] + prompt_tokens + [INST_END]
        let mut ids = Vec::new();
        ids.push(MISTRAL_BOS);
        ids.push(MISTRAL_SYSTEM_PROMPT_START);
        ids.extend_from_slice(sys_encoding.get_ids());
        ids.push(MISTRAL_SYSTEM_PROMPT_END);
        ids.push(MISTRAL_INST_START);
        ids.extend_from_slice(prompt_encoding.get_ids());
        ids.push(MISTRAL_INST_END);

        let token_len = ids.len();

        // Truncate if needed
        if ids.len() > self.max_length {
            ids.truncate(self.max_length);
        }

        // RIGHT padding: real tokens first, then pad tokens
        let real_len = ids.len();
        let mut attention_mask = vec![1u32; real_len];

        while ids.len() < self.max_length {
            ids.push(MISTRAL_PAD);
            attention_mask.push(0);
        }

        eprintln!(
            "Tokenized prompt: {} tokens (right-padded to {})",
            token_len, self.max_length
        );

        (ids, attention_mask)
    }

    /// Get the actual number of tokens (before padding) for a prompt.
    pub fn token_count(&self, prompt: &str) -> usize {
        let sys_encoding = self
            .tokenizer
            .encode(MISTRAL_SYSTEM_MESSAGE.to_string(), false)
            .unwrap();
        let prompt_encoding = self.tokenizer.encode(prompt.to_string(), false).unwrap();
        // BOS + SYSTEM_START + sys_tokens + SYSTEM_END + INST_START + prompt_tokens + INST_END
        let total = 2 + sys_encoding.get_ids().len() + 2 + prompt_encoding.get_ids().len() + 1;
        total.min(self.max_length)
    }
}
