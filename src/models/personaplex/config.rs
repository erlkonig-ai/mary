//! PersonaPlex-7B LM (Moshi/Helium temporal + depth transformer) — CONFIG
//! TRUTH, established 2026-07-11 from the safetensors header of the gated
//! checkpoint `nvidia/personaplex-7b-v1/model.safetensors`
//! (sha256 db1290db…, 16 742 874 000 bytes, 475 tensors, ALL BF16) plus the
//! NVIDIA `personaplex/moshi` reference (`models/loaders.py::_lm_kwargs`,
//! `modules/gating.py`, `modules/transformer.py`, `models/lm.py`). Every
//! dimension below is verified against a tensor shape, cited inline — not
//! taken from the paper or from `config.json` (which holds no architecture
//! fields at all, just `{"model_type": "personaplex", "version": "7b-v1"}`).
//!
//! Parameter accounting closes exactly: 8 371 437 000 params × 2 bytes (bf16)
//! = 16 742 874 000 bytes = the file size.
//!
//! The Mimi codec constants live in [`super::mimi::config`]; this module is
//! the 7B LM side (temporal transformer + depth transformer + embeddings).

// ─────────────────────────── temporal transformer ───────────────────────────
// moshi names: `transformer.layers.{0..31}.…`

/// Temporal transformer width. Evidence: `transformer.layers.0.self_attn.
/// in_proj_weight` is `[12288, 4096]` and every norm alpha is `[1, 1, 4096]`.
pub const DIM: usize = 4096;
/// Evidence: `transformer.layers.{i}` for i = 0..=31, exactly 32 layers.
pub const NUM_LAYERS: usize = 32;
/// Full MHA — NOT GQA. Evidence: fused `in_proj_weight [3·4096, 4096]`
/// (q|k|v each `[4096, 4096]`, so kv_heads == heads); moshi's
/// `StreamingMultiheadAttention.forward` rearranges `(p h d) → p, h=num_heads`
/// with p=3 — k and v get all 32 heads. `kv_repeat` does not exist in this
/// fork; there is no GQA anywhere.
pub const NUM_HEADS: usize = 32;
pub const HEAD_DIM: usize = DIM / NUM_HEADS; // 128
/// THE FFN ANSWER. The checkpoint's gated-SiLU FFN hidden width is **11264**.
/// Evidence: `transformer.layers.0.gating.linear_in.weight [22528, 4096]`
/// (fused gate|up = 2·11264) and `gating.linear_out.weight [4096, 11264]`.
/// Derivation (gating.py): nominal `dim_feedforward = hidden_scale·dim =
/// 4.125·4096 = 16896`; since 16896 ≠ 4·dim, `hidden = 2·16896/3 = 11264`.
/// So of the three accountings floating around: 16384 (the realtime probe's
/// synthetic guess) is WRONG, 22528 is the fused linear_in row count, and
/// 11264 is the true per-branch hidden dim.
pub const FFN_HIDDEN: usize = 11264;
/// `linear_in` row layout (gating.py `x.view(B,T,2,-1)`): rows `[0, 11264)`
/// are the SiLU (gate) branch, rows `[11264, 22528)` the linear (up) branch:
/// `out = linear_out( silu(x·Wg) * (x·Wu) )`.
pub const FFN_FUSED_IN: usize = 2 * FFN_HIDDEN; // 22528
/// RoPE max period (loaders.py `max_period: 10000`). Convention is
/// INTERLEAVED pairs (`rope.py` views `D//2, 2` — adjacent elements form the
/// complex pair), same convention the Mimi port verified — not split-half.
pub const ROPE_THETA: f64 = 10_000.0;
/// Attention context window in frames (loaders.py `context: 3000` = 4 min at
/// 12.5 Hz). The KV ring-cache capacity.
pub const CONTEXT: usize = 3000;
/// Norm is `rms_norm_f32`: computed in f32, `eps = 1e-8` (transformer.py
/// `create_norm_fn`), `y = x · alpha · rsqrt(eps + mean(x²))`, weight name
/// `norm{1,2}.alpha` shaped `[1, 1, dim]` (also `out_norm.alpha`). NOTE: eps
/// 1e-8, not the usual 1e-5, and alpha carries an explicit `[1,1,D]` shape.
pub const RMS_EPS: f64 = 1e-8;

// ──────────────────────────── depth transformer ─────────────────────────────
// moshi names: `depformer.layers.{0..5}.…`

/// Evidence: `depformer.layers.0.norm1.alpha [1, 1, 1024]`.
pub const DEP_DIM: usize = 1024;
/// Evidence: `depformer.layers.{i}` for i = 0..=5.
pub const DEP_LAYERS: usize = 6;
/// loaders.py `depformer_num_heads: 16` (head_dim 64). Not directly readable
/// from shapes (fused per-step projections), taken from the reference config.
pub const DEP_HEADS: usize = 16;
pub const DEP_HEAD_DIM: usize = DEP_DIM / DEP_HEADS; // 64
/// Depth FFN hidden width **2816** (the "4096 vs 5632" question). Evidence:
/// `depformer.layers.0.gating.{t}.linear_in.weight [5632, 1024]` = 2·2816,
/// `linear_out.weight [1024, 2816]`. Same 4.125 derivation: nominal
/// `int(4.125·1024) = 4224`, `hidden = 2·4224/3 = 2816`.
pub const DEP_FFN_HIDDEN: usize = 2816;
/// Per-step weights (`weights_per_step`): the checkpoint NATIVELY ships 16
/// steps — `gating.{0..15}`, `depformer_in.{0..15}`, `linears.{0..15}`,
/// `depformer_emb.{0..14}` all present, and the fused per-step attention
/// projections are `in_proj_weight [49152, 1024]` = 16·3·1024 and
/// `out_proj.weight [16384, 1024]` = 16·1024. The loaders.py 8→16
/// expand/copy patches (`copy_missing_weights`) are DORMANT for this
/// checkpoint — nothing is missing. Step t uses row-slice
/// `[t·3·1024, (t+1)·3·1024)` of in_proj (resp. `[t·1024, (t+1)·1024)` of
/// out_proj) — moshi's `multi_linear` indexed by the in-frame step offset.
pub const WEIGHTS_PER_STEP: usize = 16;
/// The depth transformer generates all 16 audio codebooks per frame
/// (loaders.py sets `dep_q = 16` at load: 8 agent + 8 user-prediction).
/// Only steps 1..=8 of the OUTPUT (agent) stream are Mimi-decoded to audio.
pub const DEP_Q: usize = 16;
/// Depth transformer has NO positional embedding (`depformer_pos_emb:
/// "none"`) and streams over at most the 16 in-frame steps (context is the
/// per-frame streaming state, reset every frame).
pub const DEP_POS_EMB_NONE: bool = true;

// ───────────────────────── streams, vocab, embeddings ───────────────────────

/// Audio codebook cardinality (Mimi codebook size).
pub const CARD: usize = 2048;
/// Audio embedding tables have one extra row for the initial token:
/// `emb.{0..15}.weight [2049, 4096]`, `depformer_emb.{0..14}.weight
/// [2049, 1024]`. `initial_token_id == CARD == 2048`.
pub const AUDIO_VOCAB: usize = CARD + 1; // 2049
pub const AUDIO_INITIAL_TOKEN: u32 = CARD as u32; // 2048
/// Text vocabulary (SentencePiece `tokenizer_spm_32k_3.model`).
pub const TEXT_CARD: usize = 32000;
/// `text_emb.weight [32001, 4096]`, `depformer_text_emb.weight [32001,
/// 1024]` — one extra row, `text_initial_token_id == 32000` (BOS-of-stream).
pub const TEXT_VOCAB: usize = TEXT_CARD + 1; // 32001
/// `text_linear.weight [32000, 4096]` — NO extra output row, because
/// `existing_text_padding_id = 3` (the tokenizer's PAD is reused; the model
/// never has to emit the initial token).
pub const TEXT_LOGITS: usize = TEXT_CARD; // 32000
/// Text special ids (offline.py token map + loaders.py): 0=EPAD (end of
/// word-padding run), 1=BOS, 2=EOS, 3=PAD.
pub const TEXT_PAD_TOKEN: u32 = 3;
pub const TEXT_EPAD_TOKEN: u32 = 0;

/// Audio streams modeled by the temporal transformer: 16 = 8 agent (output)
/// + 8 user (input). Evidence: `emb.{0..15}`. With the text stream the model
/// consumes 17 token streams per frame (`num_codebooks = n_q + 1`).
pub const N_Q: usize = 16;
pub const NUM_STREAMS: usize = N_Q + 1; // 17: [text, 8 agent audio, 8 user audio]
/// Mimi codebooks per audio stream (semantic + 7 acoustic).
pub const AUDIO_TOKENS_PER_STREAM: usize = 8;

/// Per-stream acquisition delays (loaders.py `_lm_kwargs["delays"]`), index
/// order `[text, agent 1..8, user 1..8]`: semantic codebooks (and text) are
/// undelayed, the 7 acoustic codebooks of each stream lag one frame.
pub const DELAYS: [usize; NUM_STREAMS] = [0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1];
/// max(DELAYS) — the LMGen output lag in frames.
pub const MAX_DELAY: usize = 1;

/// Mimi tokens for one frame of digital silence on the user input channels
/// (lm.py `SILENCE_TOKENS`) — used as spacer frames in the prompt phases.
pub const SILENCE_TOKENS: [u32; 8] = [948, 243, 1178, 546, 1736, 1030, 1978, 2008];
/// Mimi tokens for one frame of the reference sine wave (lm.py `SINE_TOKENS`)
/// — fed on the user input channels while the agent voice prompt plays.
pub const SINE_TOKENS: [u32; 8] = [430, 1268, 381, 1611, 1095, 1495, 56, 472];

/// Total checkpoint tensor count (safetensors header) — the persist gate
/// checks this exact number of leaves round-trips through the pile.
pub const CHECKPOINT_TENSORS: usize = 475;
