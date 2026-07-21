# F5TTS_v1_Base — exact architecture (from the safetensors header, 2026-06-16)

Read directly from `SWivid/F5-TTS :: F5TTS_v1_Base/model_1250000.safetensors`
(366 tensors, key prefix `ema_model.transformer.`). This is the ground truth
the Burn port targets.

## Config
- dim (hidden) = **1024**, depth = **22** transformer blocks, heads = **16**, head_dim = 64
- ff_mult = 2 (ff_dim 2048), activation GELU (plain MLP, not GEGLU)
- text vocab = **2546** (2545 chars + filler), text_dim = **512**
- ConvNeXt-V2 text encoder: **4** blocks, dwconv kernel 7, expand 512→1024, GRN
- mel = 100 bins, 24 kHz, hop 256, win 1024, n_fft 1024, vocoder = Vocos
- RoPE: `rotary_embed.inv_freq[32]` → rotary dim 64 (head_dim); **no q/k norm**
- CFM: Euler ODE, NFE≈32, sway sampling coef −1.0, CFG ~2.0

## Top-level (`ema_model.transformer.`)
- `text_embed.text_embed.weight [2546,512]` — char embedding
- `text_embed.text_blocks.{0..3}.` — ConvNeXt-V2: `dwconv[512,1,7]`, `norm[512]`,
  `pwconv1[1024,512]`, `pwconv2[512,1024]`, `grn.gamma[1,1,1024]`, `grn.beta[1,1,1024]`
- `input_embed.proj [1024,712]` — **712 = mel(100) ⊕ cond_mel(100) ⊕ text(512)** → 1024
- `input_embed.conv_pos_embed.conv1d.{0,2} [1024,64,31]` — conv positional embed
- `time_embed.time_mlp.{0,2}` — [1024,256] then [1024,1024] (sinusoidal-256 → SiLU → 1024)
- `rotary_embed.inv_freq [32]`
- `norm_out.linear [2048,1024]` — final AdaLN (scale+shift = 2×dim)
- `proj_out [100,1024]` — → 100 mel channels

## One DiT block (`transformer_blocks.{0..21}.`) — textbook AdaLN-zero DiT
- `attn.to_q/to_k/to_v [1024,1024]` + bias, `attn.to_out.0 [1024,1024]` + bias — plain MHSA
- `attn_norm.linear [6144,1024]` — AdaLN-zero: time → 6×dim (shift/scale/gate ×2)
- `ff.ff.0.0 [2048,1024]` (in), `ff.ff.2 [1024,2048]` (out) — Linear→GELU→Linear
- Block: `x += gate_msa·attn(mod(LN(x), shift_msa, scale_msa), rope)` then
  `x += gate_mlp·ff(mod(LN(x), shift_mlp, scale_mlp))`

## Reuse / new (Burn)
- reuse `avatar`: RoPE, `layer_norm_no_affine`, `WeightLoader` (safetensors),
  `FlowMatchEuler` scheduler structure; the NN-graph blob schema + shared attr IDs
- new: ConvNeXt-V2 text block, the textbook DiT block (avatar's are Flux-specific),
  conv-pos-embed, **Vocos vocoder** (mel→waveform, separate model)
