//! In-process text-generation API — the seam the playground's
//! `ModelBackend::Local` calls. A warm `GemmaLM` handle: load the weights once,
//! then `generate(prompt) -> String` as a direct Rust call. No ollama, no HTTP,
//! no subprocess — the brain runs in the substrate (shell-is-physics).
//!
//! Backend-generic: instantiate with `mary::nn::backend::B` (f32) or `BHalf`
//! (f16, so the dense 31B fits 128GB). The handle is stateless across calls —
//! each `generate` builds fresh KV caches, so `&self` is enough to serve many
//! requests from one warm model.

use crate::models::gemma::gemma4::config::Gemma4Config;
use crate::models::gemma::gemma4::decoder::Gemma4Model;
use crate::models::gemma::gemma4::weights::load_gemma4_from_keymap;
#[cfg(feature = "import")]
use crate::models::gemma::gemma4::weights::load_gemma4;
use crate::models::gemma::rope::RopeTable;
use burn::prelude::*;
use std::collections::HashMap;
use std::path::Path;
use tokenizers::Tokenizer;

fn argmax(v: &[f32]) -> usize {
    let mut i = 0;
    let mut b = f32::NEG_INFINITY;
    for (k, &x) in v.iter().enumerate() {
        if x > b { b = x; i = k; }
    }
    i
}

/// A loaded, warm Gemma 4 text model ready to generate in-process.
pub struct GemmaLM<B: Backend> {
    model: Gemma4Model<B>,
    rope_s: RopeTable<B>,
    rope_g: RopeTable<B>,
    tokenizer: Tokenizer,
    scale: f32,
    /// end-of-turn token id — generation stops here (besides <eos>).
    eot: u32,
    device: B::Device,
}

impl<B: Backend> GemmaLM<B> {
    /// Load from safetensors shard paths + a `tokenizer.json`. The model is held
    /// warm; call [`generate`](Self::generate) repeatedly.
    #[cfg(feature = "import")]
    pub fn load(
        config: Gemma4Config,
        shard_paths: &[&Path],
        tokenizer_path: &Path,
        device: B::Device,
    ) -> Self {
        let (model, _vision) = load_gemma4::<B>(config.clone(), shard_paths, &device);
        Self::from_model(config, model, tokenizer_path, device)
    }

    /// Load from a pile-derived keymap (`name → (f32 data, shape)`) + a
    /// `tokenizer.json`. The in-substrate counterpart to [`load`](Self::load):
    /// weights come from the pile instead of safetensors shards on disk.
    pub fn from_keymap(
        config: Gemma4Config,
        keymap: HashMap<String, (Vec<f32>, Vec<usize>)>,
        tokenizer_path: &Path,
        device: B::Device,
    ) -> Self {
        let (model, _vision) = load_gemma4_from_keymap::<B>(config.clone(), keymap, &device);
        Self::from_model(config, model, tokenizer_path, device)
    }

    /// Build by STREAMING weights from a pile — index the handles, load each
    /// tensor on demand (peak CPU = one tensor). The path that scales
    /// weights-as-tribles to the dense 31B without the full-keymap OOM.
    #[cfg(feature = "gemma")]
    pub fn from_streaming_pile(
        config: Gemma4Config,
        pile_path: &Path,
        tokenizer_path: &Path,
        device: B::Device,
    ) -> Self {
        let (model, _vision) =
            crate::persist::load_gemma4_streaming_from_pile::<B>(pile_path, config.clone(), &device)
                .unwrap_or_else(|e| panic!("stream gemma4 from pile {pile_path:?}: {e:?}"));
        Self::from_model(config, model, tokenizer_path, device)
    }

    /// Shared assembly: derive rope tables, tokenizer, scale, and eot from a
    /// built model — the part common to every `GemmaLM` constructor.
    pub fn from_model(
        config: Gemma4Config,
        model: Gemma4Model<B>,
        tokenizer_path: &Path,
        device: B::Device,
    ) -> Self {
        let (rope_s, rope_g) = model.rope_tables(&device);
        let tokenizer = Tokenizer::from_file(tokenizer_path)
            .unwrap_or_else(|e| panic!("tokenizer {tokenizer_path:?}: {e}"));
        let eot = tokenizer
            .encode("<turn|>", false)
            .unwrap()
            .get_ids()
            .first()
            .copied()
            .unwrap_or(u32::MAX);
        let scale = (config.text_config.hidden_size as f64).sqrt() as f32;
        Self { model, rope_s, rope_g, tokenizer, scale, eot, device }
    }

    /// Greedy-decode a completion for `prompt` (wrapped in the single-user Gemma
    /// chat template), up to `max_new` tokens. Convenience over [`complete`](Self::complete).
    pub fn generate(&self, prompt: &str, max_new: usize) -> String {
        let chat = format!("<bos><|turn>user\n{prompt}<turn|>\n<|turn>model\n");
        self.complete(&chat, max_new, &[]).0
    }

    /// Lower-level greedy decode from a fully-built chat string (the caller owns
    /// the chat template). Stops on <eos>, end-of-turn, `max_new`, or the first
    /// occurrence of any `stop` string — none of which leak into the text.
    /// Returns `(text, prompt_tokens, completion_tokens)`.
    pub fn complete(&self, chat: &str, max_new: usize, stop: &[String]) -> (String, usize, usize) {
        let ids: Vec<i32> = self
            .tokenizer
            .encode(chat, false)
            .unwrap()
            .get_ids()
            .iter()
            .map(|&x| x as i32)
            .collect();
        let n_chat = ids.len();

        let tokens = Tensor::<B, 1, Int>::from_ints(&ids[..], &self.device).reshape([1, n_chat]);
        let emb = self.model.decoder.embed.forward(tokens.clone()).mul_scalar(self.scale);
        let mut caches = self.model.new_caches();
        let l = self.model.forward_embeds(emb, tokens.clone(), &self.rope_s, &self.rope_g, &mut caches, &[], None);
        let [_, sl, vv] = l.dims();
        let last: Vec<f32> = l
            .slice([0..1, (sl - 1)..sl, 0..vv])
            .reshape([vv])
            .to_data()
            .convert::<f32>()
            .to_vec()
            .unwrap();

        let mut out_ids: Vec<u32> = Vec::new();
        let mut cur = argmax(&last);
        let mut n = 0usize;
        loop {
            if cur as u32 == 1 || cur as u32 == self.eot || n >= max_new {
                break;
            }
            out_ids.push(cur as u32);
            n += 1;
            // Stop-string check on the decoded text so far (the agent loop wants
            // to cut at "\n"). Truncate at the earliest stop and return.
            if !stop.is_empty() {
                let so_far = self.tokenizer.decode(&out_ids, false).unwrap_or_default();
                if let Some(cut) = stop.iter().filter(|s| !s.is_empty()).filter_map(|s| so_far.find(s.as_str())).min() {
                    return (so_far[..cut].to_string(), n_chat, n);
                }
            }
            let inp = Tensor::<B, 1, Int>::from_ints([cur as i32], &self.device).reshape([1, 1]);
            let l = self.model.forward_cached(inp, &self.rope_s, &self.rope_g, &mut caches);
            let v = l.dims()[2];
            let d: Vec<f32> = l.reshape([v]).to_data().convert::<f32>().to_vec().unwrap();
            cur = argmax(&d);
        }
        let text = self.tokenizer.decode(&out_ids, false).unwrap_or_default();
        (text, n_chat, n)
    }

    /// Decode a token-id sequence back to text with this model's tokenizer.
    pub fn decode(&self, ids: &[u32]) -> String {
        self.tokenizer.decode(ids, false).unwrap_or_default()
    }

    /// Complete a prompt that includes vision soft-tokens.
    pub fn complete_with_vision(
        &self,
        pre_text: &str,
        soft_tokens: Tensor<B, 2>, // [num_soft_tokens, hidden_size]
        post_text: &str,
        max_new: usize,
    ) -> String {
        let pre_ids: Vec<i32> = self.tokenizer.encode(pre_text, false).unwrap().get_ids().iter().map(|&x| x as i32).collect();
        let post_ids: Vec<i32> = self.tokenizer.encode(post_text, false).unwrap().get_ids().iter().map(|&x| x as i32).collect();

        let n_pre = pre_ids.len();
        let n_post = post_ids.len();
        let [n_soft, hidden] = soft_tokens.dims();
        let n_chat = n_pre + n_soft + n_post;

        let pre_t = Tensor::<B, 1, Int>::from_ints(&pre_ids[..], &self.device).reshape([1, n_pre]);
        let post_t = Tensor::<B, 1, Int>::from_ints(&post_ids[..], &self.device).reshape([1, n_post]);

        let pre_emb = self.model.decoder.embed.forward(pre_t.clone()).mul_scalar(self.scale);
        let post_emb = self.model.decoder.embed.forward(post_t.clone()).mul_scalar(self.scale);

        // Soft tokens are already in the text hidden space, just reshape to [1, n_soft, hidden]
        let soft_emb = soft_tokens.reshape([1, n_soft, hidden]);

        let emb = Tensor::cat(vec![pre_emb, soft_emb, post_emb], 1);

        // For `tokens` (used by `forward_embeds` for sliding window masking, etc.),
        // we can just fill the vision positions with the image token ID (e.g. 258880 or just 0, it doesn't matter much as long as length is correct).
        let image_token_id = 258880;
        let soft_ids = vec![image_token_id as i32; n_soft];
        let mut all_ids = pre_ids.clone();
        all_ids.extend(soft_ids);
        all_ids.extend(post_ids);

        let tokens = Tensor::<B, 1, Int>::from_ints(&all_ids[..], &self.device).reshape([1, n_chat]);

        let mut caches = self.model.new_caches();
        let l = self.model.forward_embeds(emb, tokens, &self.rope_s, &self.rope_g, &mut caches, &[], None);
        let [_, sl, vv] = l.dims();
        let last: Vec<f32> = l
            .slice([0..1, (sl - 1)..sl, 0..vv])
            .reshape([vv])
            .to_data()
            .convert::<f32>()
            .to_vec()
            .unwrap();

        let mut out_ids: Vec<u32> = Vec::new();
        let mut cur = argmax(&last);
        let mut n = 0usize;
        loop {
            if cur as u32 == 1 || cur as u32 == self.eot || n >= max_new {
                break;
            }
            out_ids.push(cur as u32);
            n += 1;
            let inp = Tensor::<B, 1, Int>::from_ints([cur as i32], &self.device).reshape([1, 1]);
            let l = self.model.forward_cached(inp, &self.rope_s, &self.rope_g, &mut caches);
            let v = l.dims()[2];
            let d: Vec<f32> = l.reshape([v]).to_data().convert::<f32>().to_vec().unwrap();
            cur = argmax(&d);
        }
        self.tokenizer.decode(&out_ids, false).unwrap_or_default()
    }

    /// Greedy-decode `chat` and return the raw generated token-id sequence (no
    /// stop-string handling, no decode). Used by the pile round-trip parity gate
    /// to compare the safetensors and pile load paths token-for-token.
    pub fn complete_ids(&self, chat: &str, max_new: usize) -> Vec<u32> {
        let ids: Vec<i32> = self
            .tokenizer
            .encode(chat, false)
            .unwrap()
            .get_ids()
            .iter()
            .map(|&x| x as i32)
            .collect();
        let n_chat = ids.len();

        let tokens = Tensor::<B, 1, Int>::from_ints(&ids[..], &self.device).reshape([1, n_chat]);
        let emb = self.model.decoder.embed.forward(tokens.clone()).mul_scalar(self.scale);
        let mut caches = self.model.new_caches();
        let l = self.model.forward_embeds(emb, tokens.clone(), &self.rope_s, &self.rope_g, &mut caches, &[], None);
        let [_, sl, vv] = l.dims();
        let last: Vec<f32> = l
            .slice([0..1, (sl - 1)..sl, 0..vv])
            .reshape([vv])
            .to_data()
            .convert::<f32>()
            .to_vec()
            .unwrap();

        let mut out_ids: Vec<u32> = Vec::new();
        let mut cur = argmax(&last);
        let mut n = 0usize;
        loop {
            if cur as u32 == 1 || cur as u32 == self.eot || n >= max_new {
                break;
            }
            out_ids.push(cur as u32);
            n += 1;
            let inp = Tensor::<B, 1, Int>::from_ints([cur as i32], &self.device).reshape([1, 1]);
            let l = self.model.forward_cached(inp, &self.rope_s, &self.rope_g, &mut caches);
            let v = l.dims()[2];
            let d: Vec<f32> = l.reshape([v]).to_data().convert::<f32>().to_vec().unwrap();
            cur = argmax(&d);
        }
        out_ids
    }
}

/// Zero-copy loader — `BHalf`/Metal only, since aliasing the pile's f16 blobs onto
/// the GPU is backend-specific. The weights are never copied to CPU or
/// materialized as f32: each tensor IS its mmap'd pile blob (see
/// [`crate::persist::load_gemma4_aliased_from_pile`]).
#[cfg(all(feature = "gemma", target_os = "macos"))]
impl GemmaLM<crate::nn::backend::BHalf> {
    pub fn from_aliased_pile(
        config: Gemma4Config,
        pile_path: &Path,
        tokenizer_path: &Path,
        device: burn::backend::wgpu::WgpuDevice,
    ) -> Self {
        let model =
            crate::persist::load_gemma4_aliased_from_pile(pile_path, config.clone(), device.clone())
                .unwrap_or_else(|e| panic!("alias gemma4 from pile {pile_path:?}: {e:?}"));
        Self::from_model(config, model, tokenizer_path, device)
    }
}
