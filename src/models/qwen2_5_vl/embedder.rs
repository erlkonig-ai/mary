//! `NomicMultimodalEmbedder` — the usable seam over the ported Qwen2.5-VL
//! backbone of `nomic-embed-multimodal-7b` (`BiQwen2_5`, a DENSE 3584-d
//! last-token embedder).
//!
//! Lives here (gemma-gated, beside the model) rather than in `mary::embed`: the
//! backbone is gemma-feature code, and keeping the seam off `mary::embed` avoids
//! a feature-gating tangle with the existing CLIP/nomic-768 embedders there.
//!
//! Text path (verified — see `tests/nomic_mm7b_real_parity.rs`, cos 0.9999999):
//! - `embed_document(text)` = tokenize(text) → backbone → last-token pool + L2.
//! - `embed_query(text)`    = tokenize(text) + 10×`<|endoftext|>` augmentation
//!   tokens (ColQwen query expansion) → backbone → pool + L2.
//!
//! Image path (`embed_image`, verified — see `tests/nomic_mm7b_image_parity.rs`):
//! run the vision tower (`vision.rs`) → merged image tokens, splice them into the
//! text embedding sequence at the `<|image_pad|>` positions, build M-RoPE 3D
//! position-ids (`get_rope_index`), run the backbone, last-token pool + L2.
//! Needs a vision tower attached (`load_with_vision` / `with_vision`).

use burn::prelude::*;
use burn::tensor::TensorData;
use tokenizers::Tokenizer;

use super::config::{Qwen2_5VlTextConfig, Qwen2_5VlVisionConfig};
use super::layers::{get_rope_index, QwenTextModel, QwenWeights};
use super::vision::{VisionTransformer, VisionWeights};

/// The `<|endoftext|>` token, used 10× as the ColQwen query-augmentation suffix.
const QUERY_AUG_TOKEN: i64 = 151643;
const QUERY_AUG_COUNT: usize = 10;
/// `<|image_pad|>` — the placeholder rows the vision tokens are spliced into.
const IMAGE_TOKEN_ID: i64 = 151655;
/// Qwen2.5-VL `spatial_merge_size` (2×2 patch merge in the vision tower).
const SPATIAL_MERGE_SIZE: usize = 2;

pub struct NomicMultimodalEmbedder<B: Backend> {
    model: QwenTextModel<B>,
    vision: Option<VisionTransformer<B>>,
    tokenizer: Tokenizer,
    device: B::Device,
}

impl<B: Backend> NomicMultimodalEmbedder<B> {
    pub fn new(model: QwenTextModel<B>, tokenizer: Tokenizer, device: B::Device) -> Self {
        Self { model, vision: None, tokenizer, device }
    }

    /// Attach a parity-verified vision tower, enabling [`Self::embed_image`].
    pub fn with_vision(mut self, vision: VisionTransformer<B>) -> Self {
        self.vision = Some(vision);
        self
    }

    /// Build from a weight source (e.g. a pile keymap) + a `tokenizer.json` path.
    pub fn load(
        weights: &impl QwenWeights<B>,
        tokenizer_path: &std::path::Path,
        device: B::Device,
    ) -> anyhow::Result<Self> {
        let tokenizer = Tokenizer::from_file(tokenizer_path)
            .map_err(|e| anyhow::anyhow!("load tokenizer {tokenizer_path:?}: {e}"))?;
        let cfg = Qwen2_5VlTextConfig::nomic_mm7b();
        let model = QwenTextModel::load(weights, &cfg, &device);
        Ok(Self::new(model, tokenizer, device))
    }

    /// Build text + vision from a combined weight source (text keys
    /// `embed_tokens/layers/norm` and vision keys `patch_embed/blocks/merger`
    /// coexist with no collision) + a `tokenizer.json` path.
    pub fn load_with_vision<W>(
        weights: &W,
        tokenizer_path: &std::path::Path,
        device: B::Device,
    ) -> anyhow::Result<Self>
    where
        W: QwenWeights<B> + VisionWeights<B>,
    {
        let tokenizer = Tokenizer::from_file(tokenizer_path)
            .map_err(|e| anyhow::anyhow!("load tokenizer {tokenizer_path:?}: {e}"))?;
        let text_cfg = Qwen2_5VlTextConfig::nomic_mm7b();
        let vision_cfg = Qwen2_5VlVisionConfig::nomic_mm7b();
        let model = QwenTextModel::load(weights, &text_cfg, &device);
        let vision = VisionTransformer::load(weights, &vision_cfg, &device);
        Ok(Self::new(model, tokenizer, device).with_vision(vision))
    }

    /// Tokenize `text` into ids (no extra special tokens; literal `<|...|>`
    /// markers in the text are mapped to their added-token ids).
    pub fn tokenize(&self, text: &str) -> anyhow::Result<Vec<i64>> {
        let enc = self
            .tokenizer
            .encode(text, false)
            .map_err(|e| anyhow::anyhow!("tokenize: {e}"))?;
        Ok(enc.get_ids().iter().map(|&u| u as i64).collect())
    }

    /// Document id sequence (plain tokenization) — the input to `embed_document`.
    pub fn embed_document_ids(&self, text: &str) -> anyhow::Result<Vec<i64>> {
        self.tokenize(text)
    }

    /// Query id sequence (tokenization + 10 `<|endoftext|>` augmentation tokens)
    /// — the input to `embed_query`.
    pub fn embed_query_ids(&self, text: &str) -> anyhow::Result<Vec<i64>> {
        let mut ids = self.tokenize(text)?;
        ids.extend(std::iter::repeat(QUERY_AUG_TOKEN).take(QUERY_AUG_COUNT));
        Ok(ids)
    }

    /// Core: dense embedding of an explicit id sequence (last-token pool + L2).
    pub fn embed_ids(&self, ids: &[i64]) -> Vec<f32> {
        let s = ids.len();
        let t = Tensor::<B, 2, Int>::from_data(TensorData::new(ids.to_vec(), [1, s]), &self.device);
        self.model.embed(t).into_data().convert::<f32>().to_vec::<f32>().unwrap()
    }

    /// Document embedding: plain tokenization (ColQwen `process_texts`).
    pub fn embed_document(&self, text: &str) -> anyhow::Result<Vec<f32>> {
        Ok(self.embed_ids(&self.embed_document_ids(text)?))
    }

    /// Query embedding: tokenization + 10 `<|endoftext|>` augmentation tokens
    /// (ColQwen `process_queries`, empty query_prefix).
    pub fn embed_query(&self, text: &str) -> anyhow::Result<Vec<f32>> {
        Ok(self.embed_ids(&self.embed_query_ids(text)?))
    }

    /// Splice already-merged vision tokens `[n_img_tokens, H]` into the text
    /// embedding sequence at the `<|image_pad|>` positions, build M-RoPE 3D
    /// position-ids (`get_rope_index`), run the backbone, last-token pool + L2.
    ///
    /// `input_ids` is the full interleaved prompt (with `n_img_tokens` image-pad
    /// rows in total across `image_grids`); `vision_tokens` are the merged tokens
    /// in raster order, concatenated across images.
    pub fn embed_spliced(
        &self,
        input_ids: &[i64],
        image_grids: &[(usize, usize, usize)],
        vision_tokens: Tensor<B, 2>,
    ) -> Vec<f32> {
        Self::pooled_to_vec(self.run_spliced(input_ids, image_grids, vision_tokens))
    }

    /// As [`Self::embed_spliced`] but returns the pooled `[1, H]` tensor (used by
    /// the parity harness, which also wants the pre-pool anchors).
    pub fn run_spliced(
        &self,
        input_ids: &[i64],
        image_grids: &[(usize, usize, usize)],
        vision_tokens: Tensor<B, 2>,
    ) -> Tensor<B, 2> {
        let position_ids =
            get_rope_index(input_ids, image_grids, IMAGE_TOKEN_ID, SPATIAL_MERGE_SIZE);
        let embeds = self.spliced_embeds(input_ids, vision_tokens);
        self.model.embed_from_embeds(embeds, &position_ids)
    }

    /// Token embeddings `[1, S, H]` with the `<|image_pad|>` rows overwritten, in
    /// order, by the merged vision tokens.
    pub fn spliced_embeds(&self, input_ids: &[i64], vision_tokens: Tensor<B, 2>) -> Tensor<B, 3> {
        let s = input_ids.len();
        let h = vision_tokens.dims()[1];
        let ids = Tensor::<B, 2, Int>::from_data(
            TensorData::new(input_ids.to_vec(), [1, s]),
            &self.device,
        );
        let mut embeds = self.model.embed_tokens(ids); // [1, s, h]
        let mut k = 0usize;
        for (p, &tok) in input_ids.iter().enumerate() {
            if tok == IMAGE_TOKEN_ID {
                let row = vision_tokens.clone().narrow(0, k, 1).reshape([1, 1, h]);
                embeds = embeds.slice_assign([0..1, p..p + 1, 0..h], row);
                k += 1;
            }
        }
        debug_assert_eq!(k, vision_tokens.dims()[0], "spliced {k} of {} vision tokens", vision_tokens.dims()[0]);
        embeds
    }

    fn pooled_to_vec(pooled: Tensor<B, 2>) -> Vec<f32> {
        pooled.into_data().convert::<f32>().to_vec::<f32>().unwrap()
    }

    /// Image embedding from precomputed pixels: run the vision tower over
    /// `pixel_values` `[seq, in_flat]` with `image_grids` `[(t,h,w)]`, splice the
    /// merged tokens into `input_ids` at the image-pad positions, and run the
    /// multimodal backbone (pool + L2). Requires a vision tower (see
    /// [`Self::load_with_vision`] / [`Self::with_vision`]).
    ///
    /// This is the tensor-level seam used by the parity harness; for real files
    /// call [`Self::embed_image`], which preprocesses the bytes first.
    pub fn embed_image_pixels(
        &self,
        pixel_values: Tensor<B, 2>,
        image_grids: &[(usize, usize, usize)],
        input_ids: &[i64],
    ) -> anyhow::Result<Vec<f32>> {
        let vision = self.vision.as_ref().ok_or_else(|| {
            anyhow::anyhow!("embed_image: no vision tower attached (use load_with_vision)")
        })?;
        let vision_tokens = vision.forward(pixel_values, image_grids);
        Ok(self.embed_spliced(input_ids, image_grids, vision_tokens))
    }

    /// Image embedding from raw file bytes (PNG/JPEG), pure Rust end-to-end:
    /// preprocess (smart-resize → normalize → patchify) → assemble the BiQwen2.5
    /// image prompt → vision tower → multimodal splice → backbone → pool + L2.
    /// Requires a vision tower (see [`Self::load_with_vision`]).
    pub fn embed_image(&self, bytes: &[u8]) -> anyhow::Result<Vec<f32>> {
        use super::preprocess::{build_image_prompt, preprocess_image, PATCH_DIM};
        let (pixels, grid) = preprocess_image(bytes)?;
        let (gt, gh, gw) = grid;
        let seq = gt * gh * gw;
        let pixel_values =
            Tensor::<B, 2>::from_data(TensorData::new(pixels, [seq, PATCH_DIM]), &self.device);
        let input_ids = self.tokenize(&build_image_prompt(grid))?;
        self.embed_image_pixels(pixel_values, &[grid], &input_ids)
    }
}
