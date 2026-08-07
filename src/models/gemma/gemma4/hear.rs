//! The hearing seam: 16 kHz waveform → understanding, in-process.
//!
//! `Hearing` bundles the parity-gated audio path (feature extractor → audio
//! tower → multimodal embedder, see `gemma_audio_parity`) with the text
//! decoder and tokenizer into one warm handle. `gemma_hear` (one-shot file)
//! and `gemma_listen` (live utterance loop) both call [`Hearing::understand`];
//! neither re-implements the merge/prefill/decode dance.

use burn::prelude::*;
use tokenizers::Tokenizer;

use super::audio::{AudioEmbedder, AudioModel};
use super::audio_preprocess::AudioFeatureExtractor;
use super::decoder::Gemma4Model;
use crate::models::gemma::rope::RopeTable;

/// Gemma 4 special token ids for the audio chat frame (E2B/E4B tokenizer).
pub const AUDIO_SOFT_TOKEN_ID: i64 = 258881; // <audio_soft_token>
pub const BOA_TOKEN_ID: i64 = 256000; // <|audio>
pub const EOA_TOKEN_ID: i64 = 258883; // <audio|>
pub const EOS_TOKEN_ID: u32 = 1;

/// A warm hearing stack: decoder + audio tower + embedder + tokenizer + the
/// precomputed RoPE tables. Build once, understand many utterances.
pub struct Hearing<B: Backend> {
    pub model: Gemma4Model<B>,
    pub tower: AudioModel<B>,
    pub embedder: AudioEmbedder<B>,
    pub tokenizer: Tokenizer,
    pub fe: AudioFeatureExtractor,
    rope_sliding: RopeTable<B>,
    rope_global: RopeTable<B>,
    device: B::Device,
}

fn argmax(v: &[f32]) -> usize {
    let mut best = 0;
    let mut bv = f32::NEG_INFINITY;
    for (i, &x) in v.iter().enumerate() {
        if x > bv {
            bv = x;
            best = i;
        }
    }
    best
}

impl<B: Backend> Hearing<B> {
    pub fn new(
        model: Gemma4Model<B>,
        tower: AudioModel<B>,
        embedder: AudioEmbedder<B>,
        tokenizer: Tokenizer,
        device: B::Device,
    ) -> Self {
        let (rope_sliding, rope_global) = model.rope_tables(&device);
        Hearing {
            model,
            tower,
            embedder,
            tokenizer,
            fe: AudioFeatureExtractor::new(),
            rope_sliding,
            rope_global,
            device,
        }
    }

    /// Run one utterance through stt + decoder. `wave` is 16 kHz mono f32
    /// (≤ 30 s — longer input is truncated by the feature extractor). Greedy
    /// decode up to `max_new` tokens; each decoded piece is also streamed to
    /// `on_token` so callers can print as generation happens. Returns the
    /// full response text.
    pub fn understand(
        &self,
        wave: &[f32],
        prompt: &str,
        max_new: usize,
        mut on_token: impl FnMut(&str),
    ) -> String {
        let device = &self.device;

        // --- Transcriber: log-mel → tower → embedder ---
        let (feat, _mask, n_frames) = self.fe.extract(wave);
        let input_features = Tensor::<B, 1>::from_floats(&feat[..], device).reshape([
            1,
            n_frames,
            self.fe.feature_size,
        ]);
        let tower_out = self.tower.forward(input_features);
        let [_, n_audio_tokens, multi_hidden] = tower_out.dims();
        let audio_embeds = self
            .embedder
            .forward(tower_out.reshape([n_audio_tokens, multi_hidden]));

        // --- Chat frame ---
        //   <bos><|turn>user\n<|audio>[audio_soft × N]<audio|>{prompt}<turn|>\n<|turn>model\n
        let pre = self
            .tokenizer
            .encode("<bos><|turn>user\n<|audio>", false)
            .unwrap();
        let post_str = format!("<audio|>{prompt}<turn|>\n<|turn>model\n");
        let post = self.tokenizer.encode(post_str.as_str(), false).unwrap();
        assert_eq!(*pre.get_ids().last().unwrap() as i64, BOA_TOKEN_ID);
        assert_eq!(post.get_ids()[0] as i64, EOA_TOKEN_ID);

        let mut ids: Vec<i64> = Vec::new();
        ids.extend(pre.get_ids().iter().map(|&x| x as i64));
        let audio_start = ids.len();
        ids.extend(std::iter::repeat(AUDIO_SOFT_TOKEN_ID).take(n_audio_tokens));
        let audio_end = ids.len();
        ids.extend(post.get_ids().iter().map(|&x| x as i64));
        let n_chat = ids.len();

        // --- Merge audio soft tokens into the input embeddings ---
        let scale = (self.model.config.hidden_size as f64).sqrt() as f32;
        let tok_i32: Vec<i32> = ids.iter().map(|&x| x as i32).collect();
        let tokens = Tensor::<B, 1, Int>::from_ints(&tok_i32[..], device).reshape([1, n_chat]);
        let mut emb = self
            .model
            .decoder
            .embed
            .forward(tokens.clone())
            .mul_scalar(scale);
        {
            let [_, _, h] = emb.dims();
            let mut d: Vec<f32> = emb.to_data().to_vec().unwrap();
            let v: Vec<f32> = audio_embeds.to_data().to_vec().unwrap();
            for i in 0..n_audio_tokens {
                let off = (audio_start + i) * h;
                d[off..off + h].copy_from_slice(&v[i * h..i * h + h]);
            }
            emb = Tensor::<B, 1>::from_floats(&d[..], device).reshape([1, n_chat, h]);
        }

        // --- Prefill + greedy decode ---
        let mut caches = self.model.new_caches();
        let l = self.model.forward_embeds(
            emb,
            tokens,
            &self.rope_sliding,
            &self.rope_global,
            &mut caches,
            &[(audio_start, audio_end)],
            None,
        );
        let [_, sl, vv] = l.dims();
        let last: Vec<f32> = l
            .slice([0..1, (sl - 1)..sl, 0..vv])
            .reshape([vv])
            .to_data()
            .to_vec()
            .unwrap();

        // Stop at end-of-sequence OR end-of-turn — greedy decode otherwise
        // rambles past the model's own `<turn|>` into a hallucinated next turn.
        let eot: u32 = self
            .tokenizer
            .encode("<turn|>", false)
            .ok()
            .and_then(|e| e.get_ids().first().copied())
            .unwrap_or(EOS_TOKEN_ID);
        let stop = |id: u32| id == EOS_TOKEN_ID || id == eot;

        let mut out = String::new();
        let mut cur = argmax(&last);
        if !stop(cur as u32) {
            let piece = self
                .tokenizer
                .decode(&[cur as u32], false)
                .unwrap_or_default();
            on_token(&piece);
            out.push_str(&piece);
            for _ in 0..max_new {
                let inp = Tensor::<B, 1, Int>::from_ints([cur as i32], device).reshape([1, 1]);
                let l = self.model.forward_cached(
                    inp,
                    &self.rope_sliding,
                    &self.rope_global,
                    &mut caches,
                );
                let [_, _, vv] = l.dims();
                let d: Vec<f32> = l.reshape([vv]).to_data().to_vec().unwrap();
                cur = argmax(&d);
                if stop(cur as u32) {
                    break;
                }
                let piece = self
                    .tokenizer
                    .decode(&[cur as u32], false)
                    .unwrap_or_default();
                on_token(&piece);
                out.push_str(&piece);
            }
        }
        out
    }
}
