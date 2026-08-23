//! The playground ↔ mary in-process text-generation seam (v1).
//!
//! Contract per the playground integration spec:
//! the playground's `ModelBackend::Local` holds a warm [`LocalTextEngine`] and
//! calls [`generate`](LocalTextEngine::generate) every turn — no ollama, no HTTP.
//! mary owns the tokenizer + chat template + decode loop; the playground passes
//! raw roles+content and sampling params.
//!
//! v1 scope: greedy decode (temperature is accepted but v1 is argmax), stateless
//! across turns (full re-encode each call — KV-cache reuse is a later mary-side
//! optimization that doesn't change the trait), no streaming. Stop strings are
//! honoured (the agent loop hints "\n").

use crate::models::gemma::gemma4::config::Gemma4Config;
use crate::models::gemma::gemma4::lm::GemmaLM;
use crate::nn::backend::{BHalf, WgpuDevice, B};
use burn::prelude::Backend;
use std::path::Path;

/// A Metal device with the storage-buffer-binding cap raised past wgpu's 4 GiB
/// default — REQUIRED for the dense 31B (its embedding is 5.6 GB at f32 / 2.8 GB
/// at f16; the default device panics in cubecl dispatch on the large buffers).
/// Pass this as the `device` to the loaders for big models. Pair with the f16
/// constructors so 31B's ~60 GB fits 128 GB (f32 is ~120 GB and will not).
pub use crate::models::gemma::metal_device::{
    init_metal_device_16gb, init_metal_device_with_large_buffers,
};

/// Chat role. Gemma has no native system turn, so System content is folded into
/// the next user turn (the standard Gemma convention).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LocalRole {
    System,
    User,
    Assistant,
}

/// One turn of the conversation the playground hands to mary.
#[derive(Debug, Clone)]
pub struct LocalChatTurn {
    pub role: LocalRole,
    pub content: String,
}

/// Generation parameters. v1 decodes greedily; `temperature`/`top_p`/`seed` are
/// accepted for forward-compat but not yet sampled (temperature 0.0 == greedy,
/// which is what v1 always does).
#[derive(Debug, Clone)]
pub struct LocalGenParams {
    pub max_tokens: usize,
    pub temperature: f32,
    pub top_p: Option<f32>,
    pub stop: Vec<String>,
    pub seed: Option<u64>,
}

impl Default for LocalGenParams {
    fn default() -> Self {
        Self {
            max_tokens: 128,
            temperature: 0.0,
            top_p: None,
            stop: vec![],
            seed: None,
        }
    }
}

/// The result of one generation, mapping onto the playground's `ModelResult`.
#[derive(Debug, Clone)]
pub struct LocalGeneration {
    pub text: String,
    pub reasoning: Option<String>,
    pub prompt_tokens: usize,
    pub completion_tokens: usize,
}

/// A warm, in-process text engine. Constructed once in the model worker and
/// reused across turns. `&mut self` so a later version can reuse KV cache.
pub trait LocalTextEngine: Send {
    fn generate(
        &mut self,
        turns: &[LocalChatTurn],
        params: &LocalGenParams,
    ) -> anyhow::Result<LocalGeneration>;
}

/// Strip Gemma structural/special-token markup from a fragment.
fn strip_markers(s: &str) -> String {
    s.replace("<|channel>", "")
        .replace("<channel|>", "")
        .replace("<|think|>", "")
        .replace("<|turn>", "")
        .replace("<turn|>", "")
        .replace("<bos>", "")
        .replace("<eos>", "")
}

/// Split a thinking-variant completion into (final_text, reasoning).
///
/// gemma-4-31B-it is a thinking model: it emits channels as
/// `<|channel>NAME<channel|>CONTENT`, e.g.
/// `<|channel>thought <channel|>compass move a1b2 done`. The agent loop wants the
/// FINAL channel's content as the command, with the reasoning separated out. We
/// take the channel named "final"/"answer" (or, lacking one, the last channel)
/// as the text, and join the other channels as reasoning. Output with no channel
/// markup passes through unchanged (non-thinking models like E2B/E4B).
fn split_channels(raw: &str) -> (String, Option<String>) {
    if !raw.contains("<|channel>") {
        return (strip_markers(raw).trim().to_string(), None);
    }
    let mut channels: Vec<(String, String)> = Vec::new();
    for seg in raw.split("<|channel>").skip(1) {
        if let Some((name, content)) = seg.split_once("<channel|>") {
            channels.push((
                name.trim().to_lowercase(),
                strip_markers(content).trim().to_string(),
            ));
        } else {
            // Header with no close marker — treat the whole segment as content.
            channels.push((String::new(), strip_markers(seg).trim().to_string()));
        }
    }
    if channels.is_empty() {
        return (strip_markers(raw).trim().to_string(), None);
    }
    let final_idx = channels
        .iter()
        .rposition(|(n, _)| n.contains("final") || n.contains("answer"))
        .unwrap_or(channels.len() - 1);
    let text = channels[final_idx].1.clone();
    let reasoning: Vec<String> = channels
        .iter()
        .enumerate()
        .filter(|(i, (_, c))| *i != final_idx && !c.is_empty())
        .map(|(_, (_, c))| c.clone())
        .collect();
    let reasoning = if reasoning.is_empty() {
        None
    } else {
        Some(reasoning.join("\n"))
    };
    (text, reasoning)
}

/// Apply stop strings: truncate at the earliest occurrence of any stop string.
fn apply_stop(text: &str, stop: &[String]) -> String {
    match stop
        .iter()
        .filter(|s| !s.is_empty())
        .filter_map(|s| text.find(s.as_str()))
        .min()
    {
        Some(cut) => text[..cut].to_string(),
        None => text.to_string(),
    }
}

/// Build the Gemma chat string from roles+content. System turns are folded into
/// the following user turn; the prompt ends primed for a model turn.
fn build_chat(turns: &[LocalChatTurn]) -> String {
    let mut s = String::from("<bos>");
    let mut sys = String::new();
    for t in turns {
        match t.role {
            LocalRole::System => {
                sys.push_str(&t.content);
                sys.push('\n');
            }
            LocalRole::User => {
                s.push_str("<|turn>user\n");
                if !sys.is_empty() {
                    s.push_str(&sys);
                    sys.clear();
                }
                s.push_str(&t.content);
                s.push_str("<turn|>\n");
            }
            LocalRole::Assistant => {
                s.push_str("<|turn>model\n");
                s.push_str(&t.content);
                s.push_str("<turn|>\n");
            }
        }
    }
    // A trailing system with no following user still needs a home.
    if !sys.is_empty() {
        s.push_str("<|turn>user\n");
        s.push_str(&sys);
        s.push_str("<turn|>\n");
    }
    s.push_str("<|turn>model\n");
    s
}

/// A [`LocalTextEngine`] backed by a warm Gemma 4 model. Generic over backend so
/// the same engine runs f32 (`B`) or f16 (`BHalf`).
pub struct GemmaEngine<Bk: Backend> {
    lm: GemmaLM<Bk>,
}

impl<Bk: Backend> LocalTextEngine for GemmaEngine<Bk>
where
    GemmaEngine<Bk>: Send,
{
    fn generate(
        &mut self,
        turns: &[LocalChatTurn],
        params: &LocalGenParams,
    ) -> anyhow::Result<LocalGeneration> {
        let chat = build_chat(turns);
        // Generate fully (no early stop-string) so a thinking model's channels
        // complete; the stop-string applies to the FINAL channel, post-split, so
        // a "\n" cuts the command and not the reasoning.
        let (raw, prompt_tokens, completion_tokens) =
            self.lm.complete(&chat, params.max_tokens, &[]);
        let (final_text, reasoning) = split_channels(&raw);
        let text = apply_stop(&final_text, &params.stop);
        Ok(LocalGeneration {
            text,
            reasoning,
            prompt_tokens,
            completion_tokens,
        })
    }
}

/// Load a warm f32 Gemma engine from JUST a persisted on-disk pile — NO
/// safetensors. The true shell-is-physics endpoint: the weights live as
/// content-addressed tribles on disk, imported once by
/// [`crate::persist::import_model_to_collection`]. STREAMS the build
/// ([`GemmaLM::from_streaming_pile`]): the blob handles are indexed once, then
/// each tensor leaf is read on demand, converted to the f32 backend width, and
/// dropped after upload — peak CPU is one tensor, NOT the whole keymap — so
/// this scales to the dense 31B (the old materialized path would OOM at ~120 GB
/// f32). `config.json` +
/// `tokenizer.json` stay as small files; the WEIGHTS come entirely from the pile.
pub fn load_gemma4_from_persisted_pile(
    pile_path: &Path,
    selector: crate::selection::ModelSelector<'_>,
    config_path: &Path,
    tokenizer_path: &Path,
    device: WgpuDevice,
) -> anyhow::Result<Box<dyn LocalTextEngine>> {
    let cfg = Gemma4Config::load(config_path);
    let lm = GemmaLM::<B>::from_streaming_pile(cfg, pile_path, selector, tokenizer_path, device);
    Ok(Box::new(GemmaEngine { lm }))
}

/// f16 variant of [`load_gemma4_from_persisted_pile`] — halves resident weights
/// so the dense 31B fits a 128GB M4 Max. Native f16 leaves stay f16; f32 leaves
/// are down-cast as each tensor is streamed in.
pub fn load_gemma4_from_persisted_pile_f16(
    pile_path: &Path,
    selector: crate::selection::ModelSelector<'_>,
    config_path: &Path,
    tokenizer_path: &Path,
    device: WgpuDevice,
) -> anyhow::Result<Box<dyn LocalTextEngine>> {
    let cfg = Gemma4Config::load(config_path);
    let lm =
        GemmaLM::<BHalf>::from_streaming_pile(cfg, pile_path, selector, tokenizer_path, device);
    Ok(Box::new(GemmaEngine { lm }))
}

#[cfg(test)]
mod channel_tests {
    use super::{apply_stop, split_channels};

    #[test]
    fn plain_output_passes_through() {
        let (t, r) = split_channels("ls -l");
        assert_eq!(t, "ls -l");
        assert!(r.is_none());
    }

    #[test]
    fn single_thought_channel_yields_command_text() {
        // The real 31B-it example from the playground.
        let (t, r) = split_channels("<|channel>thought <channel|>compass move a1b2 done");
        assert_eq!(t, "compass move a1b2 done");
        assert!(r.is_none()); // only one channel → it is the final/text, nothing left for reasoning
    }

    #[test]
    fn thought_then_final_splits_reasoning_from_text() {
        let (t, r) = split_channels(
            "<|channel>thought<channel|>the goal is done, so mark it<|channel>final<channel|>compass move a1b2 done<turn|>",
        );
        assert_eq!(t, "compass move a1b2 done");
        assert_eq!(r.as_deref(), Some("the goal is done, so mark it"));
    }

    #[test]
    fn exact_31b_framing_from_playground() {
        // Confirmed raw framing (token ids 100/101): <|channel>thought\n<channel|>{command}\n\n
        let (t, r) = split_channels("<|channel>thought\n<channel|>compass move a1b2 done\n\n\n");
        assert_eq!(t, "compass move a1b2 done");
        assert!(r.is_none());
    }

    #[test]
    fn stop_string_cuts_final_text_only() {
        // reasoning may contain newlines; stop must apply to the final text.
        let (t, _r) = split_channels(
            "<|channel>thought<channel|>line one\nline two<|channel>final<channel|>ls -l\nrm x",
        );
        assert_eq!(apply_stop(&t, &["\n".to_string()]), "ls -l");
    }
}
