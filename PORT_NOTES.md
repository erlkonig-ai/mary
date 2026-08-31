# Qwen3-TTS-12Hz-1.7B-Base → Burn port notes

Port of Qwen/Qwen3-TTS-12Hz-1.7B-Base (Apache-2.0, arXiv 2601.15621) as a
streaming-TTS **candidate A** (candidate B: Voxtral TTS, ported
in a parallel lane). Reference: github.com/QwenLM/Qwen3-TTS @ `qwen_tts`
python package, transformers 4.57.3. Everything below is **measured from the
checkpoint configs + reference code**, not from the paper.

## Components (weights: `model.safetensors`, 480 tensors, bf16)

The HF repo holds two safetensors: the TTS model (`model.safetensors`) and the
codec (`speech_tokenizer/model.safetensors`, 496 tensors, f32).

### 1. Talker (`talker.*`) — the backbone LM
Qwen3-style decoder over **codec frames** (one position = one 12.5 Hz frame):

| dim | value |
|---|---|
| layers | 28 |
| hidden | 2048 |
| heads / kv-heads / head_dim | 16 / 8 / 128 (GQA, **q_norm + k_norm** RMS on head_dim) |
| mlp intermediate | 6144 (SwiGLU) |
| rms_norm_eps | 1e-6 |
| rope_theta | 1,000,000 |
| codec vocab | 3072 (`codec_embedding`, `codec_head`) |
| text vocab | 151936 (`text_embedding`, hidden 2048) |

- `text_projection`: ResizeMLP `2048 → 2048 (silu) → 2048`, **with bias** —
  maps text-embedding space into talker space.
- RoPE is *m-rope* `mrope_section [24,20,20], interleaved=true` — but the
  position ids fed to it are `(3, B, L)` with **all three streams identical**
  (see `get_rope_index`: pure cumsum of the attention mask). With identical
  streams the interleave collapses to plain 1-D RoPE, so the Burn port uses
  standard RoPE (asserted batch=1, no padding).
- Codec control ids (in the 3072 codec vocab): pad 2148, bos 2149, eos 2150,
  think 2154, nothink 2155, think_bos 2156, think_eos 2157, language ids
  (english 2050, …). Sampling suppresses ids `[2048, 3072)` except eos.

### 2. Code predictor (`talker.code_predictor.*`) — sub-talker / MTP head
Predicts codebooks 1..15 for the current frame, conditioned on the talker's
last hidden state and codebook 0:

- 5 layers, hidden 1024, heads 16/kv 8/head_dim 128 (q/k-norm), mlp 3072,
  rope_theta 1e6, full attention, vocab 2048.
- `small_to_mtp_projection`: Linear 2048 → 1024 **with bias**.
- 15 × `codec_embedding[i]`: Embedding(2048, **2048**) — talker-width, because
  the same embeddings are used to build the talker's next-frame input.
- 15 × `lm_head[i]`: Linear 1024 → 2048, one per codebook position.
- Per frame: sequence starts `[talker_hidden, embed(code0)]` (both projected
  2048→1024), then 15 autoregressive steps; step i embeds the last sampled
  code with `codec_embedding[i-1]` and reads logits from `lm_head[i]`
  (i = `generation_steps`, starts at 1 after prefill of length 2). Fresh KV
  cache per frame.

### 3. Speaker encoder (`speaker_encoder.*`) — ECAPA-TDNN x-vector
mel(128) → TDNN(512,k5) → 3× SERes2Net(512, k3, dilation 2/3/4, scale 8,
se 128) → concat(3×512) → MFA TDNN(1536,k1) → AttentiveStatsPooling(att 128)
→ concat(mean,std)=3072 → Conv1d k1 → **2048**-dim embedding.
All convs `padding="same"` with **reflect** padding, ReLU.
Mel: n_fft 1024, hop 256, win 1024, 128 mels, fmin 0, fmax 12000, 24 kHz,
center=False with (n_fft-hop)/2 reflect pre-pad, slaney-norm librosa filters,
log(clamp(x, 1e-5)).

### 4. Codec — Qwen3-TTS-Tokenizer-12Hz (`speech_tokenizer/model.safetensors`)
16 codebooks/frame **used** (`encoder_valid_num_quantizers=16` of the
encoder's 32), codebook_size 2048, 12.5 Hz frames, 1920× upsample to 24 kHz.

**Encoder** (`encoder.*`): a transformers **MimiModel** (SEANet conv encoder +
8-layer transformer + split RVQ, ratios [8,6,5,4]×downsample 4 = 1920).
Used only to turn reference audio into ref codes.

**Decoder** (`decoder.*`), the part the port needs for speech-out —
"lightweight non-DiT", fully causal:
1. **Split RVQ decode**: codes (B,16,T) → semantic rvq_first (code 0) +
   acoustic rvq_rest (codes 1..15). Each: EuclideanCodebook embedding =
   `embedding_sum / clamp(cluster_usage, 1e-5)` (2048×256), lookup, sum over
   quantizers, `output_proj` Conv1d k1 256→512, first+rest added → (B,512,T).
2. `pre_conv`: causal Conv1d 512→1024 k3.
3. **pre_transformer** (8 layers, hidden 512): input_proj Linear 1024→512,
   GQA-less attention 16 heads × head_dim 64 (q/k/v 512→1024, o 1024→512,
   **no q/k norm**), **sliding window 72, causal**, RoPE theta 10000,
   SwiGLU mlp 512→1024, RMSNorm eps 1e-5, **LayerScale** (init 0.01) on both
   residual branches; final norm then output_proj Linear 512→1024.
4. **upsample** ×2: CausalTransConv1d(1024→1024, k=factor=2, stride 2) +
   ConvNeXt block (dwconv k7 causal groups=1024, LayerNorm eps 1e-6,
   pw 1024→4096 GELU →1024, gamma scale, residual). Total ×4.
5. **decoder stack** (SEANet-style): causal Conv1d 1024→1536 k7, then 4 blocks
   with upsample_rates [8,5,4,3]: block i = SnakeBeta + CausalTransConv
   (dim/2^i → dim/2^{i+1}, k=2·rate, stride=rate) + 3 residual units
   (SnakeBeta→causal conv k7 dilation {1,3,9}→SnakeBeta→conv k1, residual);
   dims 1536→768→384→192→96; tail SnakeBeta(96) + causal Conv1d 96→1 k7,
   clamp to [-1,1]. SnakeBeta: `x + sin²(x·e^α)/(e^β+1e-9)` per channel.
- causal conv = left-pad `(k-1)·dilation` (+ right pad to full frames; 0 for
  stride 1); causal transconv = full ConvTranspose1d, trim `k−stride` on the
  right.
- `chunked_decode`: 300-frame chunks with 25-frame left context (a **batch**
  streaming primitive — real streaming = same thing with smaller chunks).

## Generation flow (voice clone, streaming text mode = default)

Prefill embeds, assembled entirely in embedding space (B=1):
```
[ text_proj(text_emb(<|im_start|>assistant\n))                    # 3 tokens
  (tts_pad×k, tts_bos) + codec_emb([nothink|think…, think_bos, (lang), think_eos, pad])
  # ↑ speaker x-vector inserted as an extra "codec" position before pad
  text_proj(text_emb(ref_text_ids + text_ids)) ⊕ codec side:
    [codec_bos, per-frame Σ₁₆ codebook-embeds of ref_code]        # ICL prompt
  … aligned min(text_len, codec_len); leftover text becomes
  trailing_text_hidden, consumed one embed per generated frame, then tts_pad ]
```
- text ids: `<|im_start|>assistant\n{text}<|im_end|>\n<|im_start|>assistant\n`
  (Qwen2 BPE; tts_pad 151671, tts_bos 151672, tts_eos 151673); ref ids
  `<|im_start|>assistant\n{ref}<|im_end|>\n`; slices [3:-5] / [3:-2] drop the
  wrappers.
- Each decode step: talker → hidden h_t → `codec_head` logits → sample
  codebook-0 token (top_k 50, top_p 1.0, temp 0.9, repetition_penalty 1.05,
  suppress [2048,3072)∖{eos}); code predictor generates codes 1..15
  (top_k 50, temp 0.9); next talker input = Σ₁₆ codebook embeds +
  trailing_text_hidden[step] (or tts_pad embed). Stop at eos 2150 in
  codebook 0; max_new_tokens 8192 (generation_config).
- Decode: `ref_code ++ gen_codes` through the codec decoder, then cut
  `ref_len/total_len` of the samples — the ref prefix conditions the decoder
  (it's causal), which is part of why the clone holds the voice.

## Port strategy / scope decisions
- **Oracle**: official package on CPU float32, deterministic greedy run for
  goldens + default-sampling run for the STT parity test (`golden/`).
- **Codec encoder (Mimi) is NOT ported tonight**: ref codes for
  `ref_voice.wav` are captured once from the oracle (`golden/ref_code.npy`)
  and shipped as a prompt artifact. The speaker encoder IS ported (it's in
  the main checkpoint and small). Porting the Mimi encoder is the follow-up
  that makes new-reference-audio cloning self-contained in Rust.
- Burn backend: `Metal` f32 (mary house backend `nn::backend::B`).
- Batch=1 only; no padding ⇒ plain RoPE (see m-rope note above).

## Results (2026-07-03 overnight — first port)

All parity gates green (`qwen3tts_probe`): tokenizer exact-id (3 strings +
both prompts), speaker mel + x-vector cos=1.0, prefill assembly cos=1.0,
talker hidden/logits cos=1.0, first 20 greedy frames **320/320
token-identical** (exercises talker + code predictor jointly), codec
quantizer + full waveform decode cos=1.0.

End-to-end (`qwen3tts_say`, Metal f32 + fusion): test line → 93 frames,
7.4 s audio; whisper transcription of the port sample is word-for-word the
test line; resemblyzer speaker-similarity to `ref_voice.wav` = **0.922**
(oracle sample from the same prompt: 0.932; the MLX spike: 0.939).
Samples: `port_sample.wav` (seed 0), `port_sample_seed7.wav`,
oracle `golden/ref_sample.wav`.

**Realtime factor: 0.26–0.44× observed** (bimodal across runs of the same
binary — GPU power-state/system noise), vs 1.14× for the MLX spike.
- Optimizations landed then: precomputed RoPE tables; fused q‖k‖v and
  gate‖up GEMMs; GPU-side gumbel-max sampling in the predictor (wgpu `topk`
  is a device sort ≈5 ms — replaced by full-vocab gumbel, top-k truncation
  traded away); **codec convs rewritten as im2col GEMMs** (wgpu's direct
  conv1d measured ~3% efficiency: codec 11.7 s → 1.1 s, a 10× win).
- Fusion is enabled via a **local alias** `nn::backend::BFused`
  (burn-fusion + burn-cubecl/fusion deps), NOT burn-wgpu's global `fusion`
  flag — that flag rewrites the `Metal` alias and breaks gemma's zero-copy
  CubeTensor seam in `persist.rs`.

## Results (2026-07-03/04 — faster-than-realtime, `qwen3tts-streaming`)

**Sustained ≥1× reached: 1.15–1.22× generation loop, 1.01–1.05× including
batch codec decode** (f16 talker + f32 CPU predictor + f32 codec, quiet
machine). Identity holds: resemblyzer 0.912–0.946 across seeds (f32 0.922,
oracle 0.932); whisper transcripts stay word-perfect. Parity: all f32 gates
still cos=1.0 with greedy 320/320; f16 gates ≥0.999 (talker hidden
0.999998, logits 0.9999998); f16 greedy diverges at frame 0 by AR cascade
(documented in the probe, informational under `--f16`).

The three structural findings behind the speedup:

1. **The decode loop was CPU-submission-bound, not GPU-bound.** Every burn
   op costs ~15–25 µs of host time (fusion graph capture, buffer mgmt,
   encoding) regardless of kernel size; the predictor drained on the GPU in
   4–7 ms/frame while the host spent ~100–180 ms *submitting* it. Neither
   raw (non-fused) backend nor `CUBECL_WGPU_MAX_TASKS` changed this. So
   per-op count, not kernel efficiency, was the whale.
2. **The code predictor now runs on the CPU** (`cpu.rs`: Accelerate
   `cblas_sgemv` + plain Rust attention/norm/rope, `predictor.rs`). Its 15
   sequential single-token steps are ~1 GB of matvecs with strict
   dependencies — worst case for GPU dispatch, trivial for the CPU:
   **~37 ms/frame, deterministic f32, token-exact vs the oracle**
   (320/320). The codec-head logits gemv (3072×2048) and next-input
   embedding assembly also moved host-side: **one GPU sync per frame**
   (read back the talker's last hidden state) and one small upload.
3. **The talker (the only GPU stage left in the loop) runs f16 under
   `--f16`**: sync+drain 50 ms → ~16 ms/frame. Norm variance chains
   (weightless `rms`, q/k-norm `s`) compute in f32 and cast back — LLM
   activation outliers overflow f16 per-element (x² > 65504) and in the
   mean accumulator; without this the talker collapses (hidden cos 0.85).
   The **codec stays f32** even under `--f16` — its im2col GEMMs measured
   *slower* in f16 (0.5 s → 1.2 s per utterance) and it's cheap in f32.
   `BFusedHalf` alias next to `BFused` in `nn::backend`.

Attention/layer layout reworked for op-count (all folds gated exact,
greedy 320/320 preserved in f32): one wide fused matmul `[q‖k | R(q‖k) | v]`
with rotate_half pre-applied to weight rows (RoPE = short elementwise
chain, no narrow/cat); q‖k share one f32 variance pass; input/post norm
weights folded into qkv/gate_up rows; LayerScale into o_proj/down_proj
columns; 1/√d into the q-section; predictor-norm into the 15 lm_heads;
codec tr_norm into output_proj; cos/sin sliced once per stack forward;
Linear weights pre-transposed+pre-unsqueezed; single-token decode folds GQA
groups onto the query axis ([B,H,1,D] ≅ [B,Hkv,G,D]) — no kv expand, no
mask. Talker submit ~11–13 ms/frame (was ~20).

Per-frame budget at 1.2× (80 ms available): talker submit ~12 ms +
sync+logits ~16 ms + CPU predictor ~37 ms ≈ 66 ms.

**Measurement discipline:** this machine is *shared* — ambient daemons
(mediaanalysisd bursts, the Reachy daemon at ~60% CPU, WebKit tabs) and
other concurrent sessions (a TTS run, cargo builds of the main mary
checkout) swing BOTH CPU and GPU throughput 4–10×. Benchmarks here are
interleaved runs against a fixed control binary; single uncontrolled runs
are meaningless. The 1.15–1.22× figure reproduced across three separate
quiet windows; under heavy contention the loop drops to 0.2–0.7×. The
binaries pin their threads to user-interactive QoS (`cpu::
set_interactive_qos`) which protects against background-class daemons —
but not against same-class competitors (another window's cargo build).

Remaining headroom (not yet taken): multithread the CPU predictor's gemvs
(Accelerate runs them single-threaded at these sizes — ~35 GB/s effective
vs ~150 available); overlap codec decode with generation (GPU is idle
during the 37 ms CPU predictor window — free in the streaming path);
speculative next-frame submit to hide the sync.

## Streaming (`qwen3tts_stream`, landed)

Frames go to a codec thread (own f32 wgpu client) the moment they are
sampled (`pipeline::generate_streaming` per-frame sink); the codec decodes
hop-sized chunks (default 8 frames = 640 ms) with a 25-frame left-context
window — the reference's own `chunked_decode` primitive at a smaller chunk
size — and emits 24 kHz PCM16-LE to stdout (`--pcm`) as each chunk lands.
Codec work overlaps the GPU-idle window while the CPU predictor runs, so
streaming costs ~nothing over batch. The decode path is warmed at the
steady-state chunk shape during load (first chunk 0.8 s cold → ~40 ms).

Measured (f16, moderately loaded machine): **TTFA 1.72 s** (dominated by
prefill-side JIT/autotune; steady-state chunk decode is ~40 ms), zero
underruns with monotonically growing margins (+0.14 s → +1.3 s) at 1.04×
sustained on the standard test line. Streamed-audio identity: resemblyzer
0.910 vs 0.912 for batch decode of the same seed; transcript word-perfect
(the 25-frame chunk context vs the batch path's full-reference first chunk
is an inaudible trade). Long utterances degrade gracefully: margins shrink
when ambient load pushes the loop below 1× — the underrun counter in the
stream's stderr is the honest signal.

## Codec encoder (landed) — self-contained cloning

`encoder.rs`: the Mimi encoder path (SEANet conv stack → 8-layer
transformer → ×2 downsample → split-RVQ encode), all-CPU via Accelerate
sgemm im2col convs (~0.5 s per 10 s clip; runs once per reference voice).
**Gate: 2080/2080 codes exact** vs the oracle's captured `ref_code` (probe
gate 8). `qwen3tts_say --ref clip.wav --ref-text "…"` clones arbitrary
24 kHz references fully in Rust; `golden/ref_code.npy` is now only the
probe's oracle fixture. Two reference subtleties (documented in source):
the `downsample` conv pads **replicate** (everything else zero-pads), and
the encoder transformer's configured sliding_window=250 is **never applied**
by transformers' eager/sdpa path — full causal matches the oracle.

## Megakernel prototype (2026-07-03, `megakernel` feature)

Research lane: can custom cubecl kernels beat the Burn-op decode loop's
host-submission cost? Answer: **yes, by ~an order of magnitude on the talker
component** — `src/models/qwen3tts/megakernel.rs` + `qwen3tts_megakernel_probe`.

**Design.** The talker's single-token decode step, hand-fused from ~18 burn
dispatches per layer to **5** (141/frame total incl. final norm):
qkv-matvec+q/k-norm+RoPE+cache-append · attention (shared-memory softmax,
one cube per q head) · o_proj+residual · rms+gate‖up+SwiGLU · down+residual.
The engine aliases the Burn `Talker`'s GPU weight buffers zero-copy
(`CubeTensor` handle extraction) and inherits all its fold conventions
(norm weights in matmul rows, 1/√d + q/k-norm in the RoPE chain weights,
rotate_half pre-applied to weight rows). KV cache is a **preallocated**
`[max_seq, 1024]` buffer per layer written in place — the Burn path's
per-layer per-frame `cat` (realloc + full-history copy) disappears. Prefill
stays on the Burn path; its caches import once (host roundtrip + copy
kernel). f32 only for now.

**Parity.** Teacher-forced steps vs the Burn path on identical inputs:
hidden-state **cos = 1.000000000** (min 0.999999999 over 16 steps),
codebook-0 argmax 16/16. Same math, same folds, different fp association —
gate is 0.999, measured is exact.

**Measured (M4 Max, min-of-medians across rounds — contention only
inflates; same process, seq≈408):**

| path | submit ms/frame | full ms/frame | conditions |
|---|---|---|---|
| burn-raw (f32, no fusion) | 28.3 | 60.1 | load ~30 |
| burn-fused (f32, `BFused`) | 11.5–23.3 | 100.8–103.4 | load 30–50 |
| fused-f16 (`BFusedHalf`, production) | 11.6 | 86.8 | load 45–68 |
| **engine (f32, 141 dispatches)** | **0.76–2.7** | **23.5–28.4** | load 30–68 |

Engine submit = 141 × ~5.4 µs = the measured per-dispatch encode cost —
submissions stopped being the bottleneck; the frame is now GPU-bandwidth
bound (6.3 GB of f32 weights ≈ 21 ms at ~300 GB/s — full-frame time tracks
that + the ~1.6 ms sync floor).

Two second-order findings:
- **Dispatch-count reduction buys contention immunity, not just host time.**
  Under load 45–68 (browser + UI on the same GPU) the engine's full frame
  held at ~28 ms while the f16 fused path — whose quiet-machine budget is
  ~28 ms — tripled to ~87 ms. Every dispatch boundary is a chance to lose
  the GPU to another process; the engine has 141 of them, the Burn path
  500+. The f32 engine beat the f16 production talker 3× on frame time
  under load despite moving 2× the bytes.
- **Two fusion backends in one process thrash each other**: with `BFused`
  (f32) and `BFusedHalf` (f16) both live, the f16 path's submit degraded
  ~13× (11.6 → ~150 ms stable, low variance — structural, not noise).
  Measure fusion baselines in separate processes.

An f16 kernel variant would halve the weight traffic → ~12–14 ms/frame,
and the engine frees ~10 ms of CPU per frame that currently fights the
Accelerate predictor.

**Microbenchmarks** (quiet-window numbers; the probe re-measures):
- Dispatch cost: **~5.4 µs encode** (quiet; 11–67 µs under load), ~6.9 µs
  encode+drain amortized batched, **1.6–3.9 ms per sync round-trip** (wgpu
  map/poll). One-sync-per-frame is not an optimization, it's the only viable
  shape; and burn's ~15–25 µs/op is ~3–4× cubecl's raw launch cost (graph
  capture + tensor bookkeeping).
- Persistent-kernel experiment (option b): wgpu/Metal has **no grid-wide
  barrier**, so a device-side AR loop is capped at ONE workgroup
  (`sync_cube` only). A 100-step dependent matvec chain (n=1024): quiet
  machine multi-dispatch 82 GB/s vs single-cube best 30 GB/s (dim 256) —
  a ~2.7× bandwidth sacrifice, which is why the 15-step predictor stays on
  the CPU (Accelerate ~35 GB/s but zero launch overhead and exact f32
  parity). Under heavy contention the ranking *flips* (persistent 44 GB/s
  vs multi 29 GB/s): a single dispatch is immune to per-dispatch scheduler
  gaps. For latency-bound small-working-set chains (jerky/GPU-succinct
  rank/select walks) the single-workgroup persistent kernel is therefore
  viable — succinct queries want latency, not bandwidth.

**cubecl 0.10 field notes** (the sharp edges, for the next kernel):
- `SharedMemory` reads expand to `NativeExpand<f32>`; they cannot be
  assigned into literal-initialized `let mut` locals (those are *comptime*
  consts) — the error is a baffling `f32: From<NativeExpand<f32>>` at the
  `#[cube]` attribute's span. Force runtime vars: `f32::new(0.0)`.
- Same trap at *expansion time* (runtime panic "mutable operation on a
  const variable", only when the kernel first launches): accumulators
  initialized from literals and `let mut stride = d / 2` where `d` is
  `#[comptime]`. `u32::new((d / 2) as i64)` fixes the latter.
- No shared→shared or array→shared direct element assigns; launder through
  a `let` binding.
- Weight tensors produced by `transpose().mul(...).reshape(...)` can carry
  non-canonical strides on size-1 dims — `is_contiguous()` false; run
  `burn_cubecl::kernel::into_contiguous` before extracting the handle.
- Burn tensor ↔ raw kernel seam: `into_primitive()` →
  `CubeTensor { handle, client, .. }` → `ArrayArg::from_raw_parts(handle, len)`;
  handles are refcounted, keeping one keeps the buffer alive. Works only on
  the raw backend — `Fusion<CubeBackend>` wraps primitives in fusion
  handles (the engine runs raw and *is* the fusion).
- Keep barrier structure cube-uniform (all cubes execute the same
  `sync_cube` sequence) — v-head cubes compute-and-discard a variance
  reduce so the qk cubes can have theirs.

**Not done / next:** f16 kernel variants (halve the drain; needs the f32
variance-chain trick inside the kernels — trivial since we control them);
wiring the engine into `pipeline::generate_streaming` (needs trailing-text
hiddens host-side + a `logits_from`-compatible readback — mechanical);
prefill-shaped fused kernels (kill the 1.2 s TTFA JIT).

## Zero-copy alias probe (2026-07-04, deleted)

A TRUE zero-copy pile→GPU load for the fused speak backends was built and
probed: mmap'd pile blob → Metal `newBufferWithBytesNoCopy` → cubecl handle
(the gemma raw-backend seam), then imported into the fusion runtime by
registering the existing buffer as an already-initialized tensor
(`InitOperationIr`, the same route fusion's own `float_from_data` takes —
minus the host copy). Verdict, from the `qwen3tts_alias_test` gate:

- **Correct in isolation**: bit-exact on readback; exact in single fused
  ops (matmul / elementwise / select). V3's 256-aligned payloads satisfy
  Metal's buffer-binding alignment; the keepalive chain (blob →
  `MmapRaw` Arc → `WgpuStorage.external_keepalives`) held.
- **Miscompiles at graph scale**: a large fused graph over MANY
  externally-registered Init tensors breaks in `burn-cubecl-fusion`'s
  codegen — `vector_size`/ordering panics, end-to-end talker cos≈0.41.
  The bug is burn 0.21's fusion codegen, not the aliasing.

The path (`src/nn/alias.rs`, `src/bin/qwen3tts_alias_test.rs`, an env-gated
`MARY_SPEAK_ALIAS=1` branch in the aliased loader) was DELETED rather than
kept opt-in — a known-miscompiling path doesn't ship in-tree. Recover it
from git history if burn's fusion learns to handle external Init tensors or
the talker moves to the raw backend. The production load keeps the win that
mattered: direct native-width upload of the f16 leaves (no f32
materialization/cast; cold load 6.7 s → 2.7 s), byte-identical batch output
vs the materialized path.

## Follow-ups
1. Sampling fidelity nit: predictor + talker-side sub-sampling uses
   full-vocab gumbel-max instead of the reference's top-k=50 for the
   sub-talker (CPU top-k is cheap now — trivial to restore if ever audible).
2. Perf headroom (only needed for margin under ambient load, see Results):
   multithread the CPU predictor's gemvs; speculative next-frame submit to
   hide the per-frame sync; TTFA ~1.2 s of prefill JIT could shrink with a
   prefill-shaped warmup.
3. Wire `qwen3tts_stream --pcm` into the `mary::say` consumer seam as
   the streaming voice (the F5 lane is the current `say`; this lane is
   faster-than-realtime and streams).
4. Concurrent sessions share one GPU: two heavy GPU users (e.g. this loop +
   an F5 synthesis) halve each other — coordinate before long voice
   sessions.

## Realtime lane (2026-07-10, `voice-realtime`)

Goal: sustained ≥1× audio-rate for `mary::speak` (wall/audio ≤ 1.0 on a quiet
machine; stretch ≤ 0.8). Method: per-component instrumentation first, then
attack the dominant term. All timing on a 686-char 2-pass LONG fixture,
seed 7, f16 talker, `speak_check` runs interleaved against the pristine-main
control binary (the machine shared two other bench lanes all night — every
number carries its load window; compare within-window only).

### Phase 1 — per-component profile (`QWEN3TTS_BENCH`)

Env-gated and timing-only: the seed-7 wav sha is identical to main with the
instrumentation on and off. Calm window (load ~4-5), main == branch pool-off:

| per frame (80 ms budget) | ms |
|---|---|
| talker submit (host, fusion op registration) | 11–13 |
| sync (GPU drain of the step) | 15.5 |
| logits gemv + sampling | 0.2 |
| **CPU code predictor** | **36** |
| next-input embed + upload | 0.2 |
| total | 63–65 (0.82 wall/audio incl. TTFA 4.1–4.3 s) |

Predictor internals: proj ~1.1 + 5-layer stack ~35 (gemv 33.6, scalar
attn/norm 1.5) + lm_head+sample ~1.8 — the stack's gemvs (~5 GB/frame of
strictly sequential f32 weight traffic through ONE Accelerate stream) are
the frame. Codec: 40–86 ms per 8-frame chunk on its own thread, fully hidden
behind generation. Prefill submit 80–460 ms/pass + frame-0 JIT drain ~45 ms.

### Phase 2a — row-parallel predictor gemvs (`cpu::sgemv_mt`)

> **Correction, 2026-08-18.** "Bit-exactness first" is retired as a design
> priority (JP: kill the bit-perfection rule; it is dead against a previous
> implementation *and* run-to-run — wiki:f5dcc88988bb28e472e50fa030332adb). The
> row-block result below stands and is worth keeping, but read it as a
> *consequence* of splitting along the row axis — which reassociates nothing —
> not as the admission test. In particular the `down` exclusion at the end of
> this paragraph is no longer a correctness argument: a column-blocked
> reduction is the better-conditioned one, so refusing it to preserve the
> serial lane's bytes preserved the worse numerics. Reopen it on measurement.

Bit-exactness first: a row-block `cblas_sgemv` computes each row identically
to the full-matrix call — verified standalone for every routed shape
(qkv [4096×1024], o/proj [1024×2048], gate_up [6144×1024], lm_head
[2048×1024], codec head [3072×2048]) at splits 2..=8 AND chunk grids of
64/128/256/512 rows; end-to-end, EVERY pool variant reproduced the identical
466-frame LONG wav sha. The one exception: `down` [1024×3072] — at n=3072
the full call selects a different (column-blocked) kernel than any row block
(~1e-6 diffs, flip between m=512 and m=1024) — `down` stays a serial full
call, keeping the pool byte-transparent.

Two pool designs, measured predictor ms/frame (LONG, interleaved):

| ways | fixed slices | work-stealing 64-row chunks |
|---|---|---|
| serial | 36 | 36 |
| 2 | 29.3 | 30.0 |
| 3 | — | 33.5 (36–42 in a noisier window) |
| 4 | 42–51 | 31–34 (37–47 noisier) |
| 6 | 53–84 | 35–41 |
| 8 | 117–195 | — |

Findings:
1. **Fixed slices die by straggler on a shared machine** — each gemv waits
   for the slowest of P threads, so width HURTS under ambient load (8 ways =
   5× worse than serial). The work-stealing grid (generation-tagged CAS chunk
   counter; a preempted thread delays one 64-row chunk, not m/P rows) repairs
   4–6 ways but never beats 2.
2. **Two ways is the ceiling and it is structural**: Accelerate's sgemv runs
   on the AMX/SME units — one per P-cluster, two on an M4 Max. Two streams
   saturate them (~165 GB/s effective vs ~135 single); a third only queues.
   Default `MARY_PRED_THREADS=2` (0/1 disables).

Result: predictor 36 → 30 ms; frame 64.7 → 57–59 ms; **0.82 → 0.73–0.74
wall/audio for the whole utterance including load, JIT and prefill;
1.35–1.40× audio-rate steady** (calm window). Same-window A/B with the final
binary in a busier window (load 6–8): pool-off 0.92–0.93, default 0.84.
Target ≤1.0 met with margin; stretch ≤0.8 met on calm windows.

### Phase 2c — `down` joins the pool (2026-08-19, `perf/voice-predictor`)

Phase 2a excluded `down` [1024x3072] from the pool to keep the pool
byte-transparent, and the 08-18 correction above said to reopen it on
measurement. Nobody had. Measured now, the exclusion was the most expensive
thing in the predictor.

Method: `qwen3tts_pred_bench`, a new bin that runs the **real**
`CodePredictor::predict_frame` over synthetic weights at the checkpoint's
shapes — Accelerate's sgemv timing is data-independent, so shapes, traffic and
dispatch pattern are what a bench of this lane has to get right, and those come
from `CodePredictor::load`. It exists because the canonical `qwen3tts.pile` was
on an unmounted volume that day. Both arms (`MARY_PRED_DOWN_SERIAL`, flippable
in process via `predictor::set_down_serial`) interleave round by round, so an
ambient-load change hits both equally. 100 rounds, p10/p90 within +/-1% of p50,
and the harness reproduces Phase 2a's own endpoints (35.6 ms at ways=1 vs its
"serial 36"; 28.8 ms at ways=2 vs its 30.0).

| ms/frame (ways=2) | frame | proj | qkv | o | gate_up | **down** | scalar | head |
|---|---|---|---|---|---|---|---|---|
| `down` serial | 28.36 | 0.6 | 5.67 | 2.89 | 8.38 | **8.36** | 1.3 | 1.1 |
| `down` pooled | 24.39 | 0.6 | 5.70 | 2.90 | 8.42 | **4.38** | 1.3 | 1.1 |

`down` was **29.5% of the whole predictor frame** while carrying 20% of the
weight traffic; pooling it costs -14.0% off the frame, 2.82x -> 3.28x audio-rate
for the predictor alone. The reason it was disproportionate is worth keeping:
the wide-n kernel Phase 2a noticed is not merely *different*, it is *slower* —
~118 GB/s of weight traffic against ~230 GB/s for every pooled gemv, i.e.
exactly one AMX stream's worth. A row block at n=3072 selects the ordinary
kernel and picks up the same 2x as everything else. `down` was never special;
it was just serial.

**Tree reduction: measured, not taken.** A k-split of `down` into column blocks
summed pairwise (the O(log n * eps) shape) was a wash at 2 ways (-0.4%, inside
the noise) and cost +5.5% at 4. The accuracy half of the conjecture did not
survive either: against an f64 reference, serial / pooled / ksplit at 2, 4, 8
all land at 4.5e-8 +/- 0.1e-8 relative L2. At n=3072 in f32 the reduction depth
is not what bounds the error.

**Pool width.** In this harness (predictor alone, nothing else on the machine)
even widths beat odd and 8 ways edged 2 by ~4%. NOT acted on: Phase 2a's ways=2
was chosen in situ, against a live GPU talker submit thread, which is the
condition that decides it. Default stays 2.

**NOT ear-gated.** No local pile could load the cohort (only the 173-tensor f16
talker fold was on this machine), so nothing has been listened to. The change is
a pure row split of one gemv onto the pool that already carries the other three,
but the A/B is owed.

### Phase 2b — 0.6B talker sibling (`MARY_SPEAK_MODEL=0.6b`)

Qwen3-TTS-12Hz-0.6B-Base: hidden 1024 (vs 2048), mlp 3072 (vs 6144), same
28 layers / 16:8 heads / head_dim 128 / vocab; codec safetensors LFS-oid
IDENTICAL to the 1.7B's; and — decisive — the code predictor is the SAME
geometry (5L×1024h×3072i), so the dominant CPU term does not shrink. The
checkpoint has no `small_to_mtp_projection` (talker width == predictor
width; the reference instantiates `nn.Identity`) — the port now reads
talker/predictor/speaker geometry from checkpoint shapes and treats the
projection as optional (`has_weight`; the SingleFile arm now really parses
names instead of returning false). Persisted to `models/qwen3tts_0.6b.pile`
(5.4 GiB, f16 parity gate green); `MARY_SPEAK_MODEL=0.6b` in `mary::speak`
selects the sibling pile next to the caller's.

Measured (load ~8 window, 4 ways): sync 15.5 → 3.4–3.6 ms (the GPU tail
tracks talker size), TTFA 4.1 → 2.8 s, predictor unchanged — the 0.6B buys
GPU headroom and TTFA, not frame budget. With the pool the 1.7B already
clears realtime, so the 0.6B is latency/contention insurance, not a
necessity. Ear-gate kit: three seed-7 A/B pairs + README in `/tmp/voice_ab`
(whisper word-perfect on all six; resemblyzer 0.6B 0.856–0.949 vs 1.7B
0.893–0.933; the 0.6B's s2 rendered 12.9 s for a ~6 s sentence — pacing
stability is the thing to listen for).

### Not taken (and why)

- **Sync tail (15.5 ms) / talker submit (11–13 ms)**: GPU-side and host-side
  dispatch-count bound (~500 fused dispatches/step) — megakernel territory,
  excluded per the review verdict; its measurement methodology (component
  isolation, min-of-medians, contention notes) was reused instead.
- **Prefill JIT off the clock**: no cross-process shader/autotune cache in
  this burn/cubecl; the talker-prewarm race with the codec JIT is documented
  in speak.rs. A resident warm-weight process is the real TTFA fix (the F5-era
  `say_serve` prototype that demonstrated it is git-only as of 2026-08-27).
- **f16/NEON predictor weights**: two AMX f32 streams already outrun hand
  NEON f16, and it would break byte-identity.

## PersonaPlex-7B LM — temporal transformer (2026-07-11, `voice-realtime`)

LM part 1 of the PersonaPlex port: the 7B **temporal transformer** forward
(moshi `transformer.*` + `text_emb`/`emb.{0..15}`/`out_norm`/`text_linear`),
CPU-f32, gated against the moshi CPU-f32 oracle. Files:
`src/models/personaplex/temporal.rs` (the stack), `src/bin/personaplex_probe.rs`
(the gate), `nn::npy::load_npy_i64` (token goldens), `personaplex` Cargo feature
(rides `qwen3tts`, adds `burn-ndarray` + Accelerate BLAS).

**Reused, not rewritten.** The whole stack is `qwen3tts::layers`
(`DecoderLayer`/`Attention`/`RopeTable`/`KvCache`/`Linear`/`RmsNorm`/`Embedding`)
run at moshi's knobs — full MHA (`kv_heads == heads == 32`), RMS eps 1e-8, no
q/k-norm, no LayerScale, no sliding window — all already expressible in
`AttnConfig`. **Zero layer edits**: everything moshi-specific is resolved at
weight-adaptation time (`temporal::adapt_layer`), so the fused decode path runs
untouched.

**The convention trap (there's always one): interleaved vs split-half RoPE.**
moshi rotates INTERLEAVED complex pairs `(x[2i], x[2i+1])` at freq
`θ^(-2i/D)`; `layers::RopeTable` implements split-half rotate_half
`(x[i], x[half+i])` at the *same* freq. The two are conjugate under the
per-head de-interleave permutation `P` (`P(x)[i]=x[2i]`,
`P(x)[half+i]=x[2i+1]`): `rope_split_half(P·x) = P(rope_interleaved(x))`. Since
attention is inner products and a shared permutation of q and k cancels
(`(P·q)ᵀ(P·k)=qᵀk`), applying `P` to the **rows** of the q_proj/k_proj weight
blocks (v untouched) makes the port numerically exact at zero runtime cost and
zero kernel changes — the standard GPT-J→NeoX checkpoint conversion. Caught it
up front from `modules/rope.py` (the `view(D//2, 2)` giveaway); the probe
confirmed cos=1.0, so it was the *only* trap.

Name-adapting loader (`adapt_layer`/`adapt_globals`): fetches each moshi
layer's tensors from the shelf pile, row-splits `in_proj_weight [12288,4096]`
→ q/k/v (q,k de-interleaved), `gating.linear_in [22528,4096]` →
gate/up (rows `[0:11264)`/`[11264:22528)`), squeezes `norm{1,2}.alpha [1,1,D]`
→ `[D]`, renames `out_proj`→`o_proj`, `gating.linear_out`→`down_proj`, and
serves them as a transient `WeightLoader::Pile` in mary's qwen3tts convention,
**one layer at a time** — the ~26 GiB f32 model never fully materializes in the
adaptation buffer (weights come off the pile lazily via the handle index).

**Parity (all 113 oracle steps, greedy/f32):**
per-step hidden **cos = 1.000000000** (min at step 52, max|Δ| 7.3e-5),
per-step text-logits **cos = 1.000000000** (min at step 52, max|Δ| 6.4e-5),
text-argmax **113/113 (100%)**. Both bars (≥0.99999) clear with the whole
margin to spare, across all five prompt/gen phases (voice-prompt embedding
replay 0..49, silence, text prompt, silence, user-audio gen). 3.0 s/step under
NdArray+Accelerate (parity path, not the perf path); load 100 s, run 339 s;
~61 GiB RSS.

**KV-cache note for the decode loop.** `layers::KvCache` grows (cat per step);
moshi's `RingKVCache` is identical *within* the 3000-frame context and only
diverges once a session exceeds it (4 min at 12.5 Hz) — at which point the ring
overwrite + the `delta < context` causal-window mask must be ported. Parity
windows and short sessions never reach it. The temporal `offset` is the RoPE
position and the cache length; `TemporalTransformer::reset()` clears both.

**Next increment: depformer per-step parity.** The depth transformer (6L×1024,
16 per-step weight sets, `depformer_in.{0..15}` / `gating.{0..15}` /
`linears.{0..15}`, fresh streaming state per frame) conditioned on
`transformer_out` + `next_text_token`. Goldens already captured
(`dep_logits [113,16,2048]`, `dep_tokens`, `dep_in_text`); gate its 16
per-codebook logits against them, then wire the full `LMGen.step` cache /
delay / undelay bookkeeping for end-to-end token output vs `out_tokens`.
*(Done — next section.)*

## PersonaPlex-7B LM — depformer + LMGen step machinery (2026-07-11, `voice-realtime`)

LM part 2: the **depth transformer** (moshi `depformer.*` +
`depformer_in.{0..15}` / `depformer_text_emb` / `depformer_emb.{0..14}` /
`linears.{0..15}`) and the **`LMGen.step` delay/undelay bookkeeping**,
CPU-f32, gated against the same 113-step oracle stream. Files:
`src/models/personaplex/depth.rs` (the depformer),
`src/models/personaplex/lmgen.rs` (`StreamCache` — the pure integer token
machine — plus `LmGen`, the temporal+depth+bookkeeping glue),
`personaplex_probe` grew `depth` and `e2e` subcommands (`temporal` keeps the
part-1 gate).

**Per-step vs shared (the answer from code, not assumption).** With
`depformer_weights_per_step = True` (`weights_per_step = dep_q = 16`),
PER-STEP ×16 are: the attention projections — `self_attn.in_proj_weight
[16·3·1024, 1024]` and `out_proj.weight [16·1024, 1024]`, row-sliced by
`multi_linear` at the in-frame step offset (`modules/transformer.py`) — and
the FFN (`gating.{0..15}`, a per-step `ModuleList`), plus at the LM level the
conditioning projections `depformer_in.{0..15}` (each `[1024, 4096]`, applied
to `transformer_out` fresh at EVERY step), the logit heads `linears.{0..15}`
and the prev-token embeddings (`depformer_text_emb` for step 0,
`depformer_emb.{s-1}` for step s). SHARED across the 16 steps: only
`norm{1,2}.alpha` per layer (folding the shared alpha into each per-step
weight set at load is exact). There is **no positional embedding**
(`depformer_pos_emb: "none"`) and **no final norm** — `linears.{s}` reads the
raw residual stream. Reuse story mirrors part 1: the whole stack is
`qwen3tts::layers::DecoderLayer` at moshi knobs, 16×6 instances sharing 6
per-frame KV caches; "no RoPE" is expressed as an identity rope slice
(cos = 1, sin = 0 — exact in f32, the folded 1/√64 is a power of two). No
de-interleave here (nothing to rotate).

**The convention trap (there's always one): the depformer attention window
is 15, not 8 and not 16.** Two stacked facts, found the hard way (first run:
agent codebooks 0..7 cos=1.0, user-pred 8..15 as low as 0.69 — a signature
that points straight at in-frame steps ≥ 8):
1. loaders.py ships `depformer_context: 8`, but `LMModel.__init__` DISCARDS
   it — `kwargs_dep["context"] = None` (lm.py:321). Dead knob. Ring capacity
   is therefore `weights_per_step = 16` and there is no `delta < context`
   term: steps 0..=14 attend fully causally.
2. moshi's `RingKVCache.complete()` has a wrap quirk: the slot at
   `end_index = end_offset % capacity` is labeled with the FUTURE position
   `end_offset` (the `delta <= 0` branch), so the instant the ring fills,
   the OLDEST key is causally masked. At in-frame step 15 the visible set is
   `{1..=15}` — key 0 (the text-conditioned step) is dropped. Verified
   empirically against the oracle class (capacity 8 shows the same off-by-one,
   which is also why fact 1 alone — window 8 — was wrong twice over).
Both together ≡ a sliding window of `capacity − 1 = 15` over mary's growing
cache + mask (softmax-identical to ring overwrite + position mask):
`AttnConfig { window: Some(15) }`. One-line fix, all 16 codebooks green.
Corollary for the temporal stack's eventual ring port: context 3000 means an
effective window of 2999 once wrapped.

**LMGen bookkeeping** (`lm.py prepare_step_input`/`process_transformer_output`
→ `StreamCache`): 17 streams `[text, agent 1..8, user 1..8]`, delays
`[0, 0,1×7, 0,1×7]`, ring of `max_delay + 3 = 4` positions. Caller tokens
land at `(offset + delay) % 4` and set `provided`; initial tokens (2048
audio / 32000 text) seed positions for `offset <= delay` (fires only at
offsets 0/1, offset 0 is a cache-seeding step with no model call); the model
eats position `(offset−1) % 4`; sampled text + the depformer's 16 tokens are
written at `offset % 4` where not provided; provided targets teacher-force
the depformer's prev-token chain (this is how user audio drives the 8
user-prediction codebooks and prompts force the agent side); the undelayed
output frame reads `(offset − 1 + delay_k) % 4` once `offset > max_delay`.
Voice-prompt replay = embedding-fed steps with dummy-initial cache writes,
then the packaged `.pt` cache snapshot OVERWRITES the ring (provided flags
kept) — `StreamCache::overwrite`, fed by the `vp_cache` golden.

**Parity (greedy/f32, oracle stream: 50 vp + 6 silence + 26 text + 6 silence
+ 25 user-audio steps):**
- `depth` gate (teacher-forced, loads only the ~5.5 GiB depformer side, 64 s):
  per-codebook logits **min cos = 1.000000000 for ALL 16 codebooks** over
  113 steps (max|Δ| = 2.7e-5), argmax **1808/1808 (100%)**,
  `next_text_token` 113/113 vs `dep_in_text`, undelayed output frames
  **25/25 × 17 streams exact** vs `out_tokens`.
- `e2e` gate (free-running `LmGen`: real temporal 7B + real depformer + the
  bookkeeping, fed exactly the oracle's inputs — vp embeddings, SILENCE/SINE
  frames, text-prompt tokens, `user_codes`): assembled model inputs
  **63/63 × 17 exact** vs `step_tokens`, depformer tokens **113/113 × 16
  exact** vs `dep_tokens`, output stream **25/25 × 17 exact** vs
  `out_tokens` — the whole token stream integer-exact end to end.

**Next increment:** Mimi decode integration — the out-frame's agent audio
streams 1..=8, undelayed, minus the initial-token frames, into the already
parity-green Mimi decoder (`personaplex::mimi`) for full audio-out; then the
q4/Metal decode build (the 80 ms/frame realtime lane — the temporal 7B is
the whale, the depformer's 16 sequential 1024-wide steps are the
predictor-shaped CPU candidate). *(Done — next section.)*

## PersonaPlex-7B — end-to-end audio out (2026-07-11, `voice-realtime`)

LM part 3: the full chain wired — input WAV → Mimi encode (user codes) →
LmGen free-run → agent audio streams → Mimi decode → 24 kHz PCM. Files:
`src/models/personaplex/pipeline.rs` (`VoicePipeline` — encoder + LmGen +
decoder from the one union pile, plus the prompt-flow helpers and
`agent_codes`), `personaplex_probe` grew a `pipeline` subcommand,
`golden/capture_personaplex.py` grew the oracle audio dump.

**Which streams become audio (offline.py, not assumption).** Of the 17
undelayed output streams `[text, agent 1..8, user 1..8]`, only the AGENT
audio streams `1..=8` are Mimi-decoded — `decode_tokens_to_pcm` slices
`out[:, 1:9]`. Stream 0 is text; streams 9..=16 are the model's prediction
of the USER's audio and are never decoded.

**The skip/partial-frame rule (from lm.py — there is no partial frame, only
`None`).** (1) The very first `LMGen.step` call (`offset == 0`) seeds the
token ring with initials and returns `None` without a model step
(`prepare_step_input`'s offset-0 branch). (2) While `offset <= max_delay`
(= 1) the model steps but `process_transformer_output` returns `None` — the
undelayed read position of the delay-1 acoustic streams hasn't been written
yet. (3) From `offset == max_delay + 1` on, EVERY step emits a complete
frame `out[k] = cache[k][(offset − max_delay + delay_k) % CT]`; each emitted
slot has by construction been written by a sampled or provided token, so
initial (2048) / ungenerated (−2) tokens never surface in an emitted frame
(`agent_codes` asserts this before handing codes to the decoder). In the
production prompt flow both `None` steps fall inside the voice-prompt phase;
prompt-phase frames are discarded (the oracle's `_step_*` helpers ignore
them too), so every user-audio step yields exactly one decodable frame.

**Oracle audio golden.** `capture_personaplex.py` now decodes each out
frame's agent codes through the SAME streaming mimi inside the gen loop
(production semantics; decode state fresh at gen start — `reset_streaming`
runs after the prompts) → `out_audio.npy` [48000], plus a fresh
non-streaming batch decode of the same 25×8 code matrix →
`out_audio_batch.npy` as the fp cross-check. Streaming vs batch: **cos =
1.0, max|Δ| = 3.9e-8** — so mary's batch decoder is a valid comparison
target for the streaming reference. Re-running the extended capture
reproduced all 13 pre-existing goldens **byte-identical** (sha256) — the
oracle is deterministic; nothing shifted under the new dump. The output is
mostly silence for this 2 s input (agent is listening: RMS 2.96e-4,
max|x| 0.0123) — expected; the gate is oracle-match, not interesting audio.

**Sanity note on the input path:** `sphn.read` = pcm16 ÷ 32768 exactly and
same-rate `sphn.resample` is the identity (verified numerically), so
`wav::read_pcm16_mono` feeds bit-identical samples to mary's encoder.

**Parity (`personaplex_probe pipeline`, CPU-f32, greedy):**
- Mimi encode: **200/200 codes integer-exact** vs the oracle's *streaming*
  encode (`user_codes`) — mary's batch encode ≡ oracle streaming encode on
  this window.
- LM free-run out frames: **25/25 × 17 streams integer-exact** vs
  `out_tokens` (the pipeline reproduces the e2e gate through the
  `VoicePipeline` seam).
- Audio: **cos = 0.999999999 vs the streaming oracle decode** (max|Δ| =
  4.4e-8 in sample units), batch cross-check cos = 0.999999999 (max|Δ| =
  5.4e-8) — the ≥ 0.999 gate clears with the whole margin; the residual is
  pure sgemm-association fp noise on a near-silent signal. Output written
  to `/tmp/mary-personaplex/pipeline_out.wav` (24 kHz PCM16).
- Timing (parity path, loaded machine): load 138 s, prompt phases 341 s
  (88 steps), 3.4–3.8 s/step free-run, batch decode of 2 s audio 1.2 s —
  NdArray f32 throughput, not the realtime lane.

**Next increment: the q4/Metal realtime decode build.** The 80 ms/frame
budget, mirroring the qwen3tts split: the temporal 7B on Metal with q4
weight-quant matvecs (the `nn::q4` bandwidth lever — 17.2 GB/step f32 →
~4.8 GB) as the GPU stage, and the depformer as the CPU/Accelerate
predictor (16 strictly-sequential 1024-wide steps — the same
predictor-shape that won on CPU for qwen3tts), with the CPU Mimi codec on
its own thread as in `qwen3tts_stream`.

## PersonaPlex-7B — Metal quantized temporal decode (2026-07-11, `rt-q4-temporal`)

Realtime lane, temporal stage: the 7B temporal transformer rebuilt for the
80 ms/frame budget — `src/models/personaplex/temporal_metal.rs`
(hand-launched cubecl kernels on the raw wgpu/Metal device; burn-fusion
never touches the graph, per the FusedReduceLaunch stride miscompile) +
`personaplex_rt_probe` (`gate` / `bench` / `quantcheck`). Structure:

- **418 dispatches/step, one readback** (megakernel discipline): per layer
  3 quant matvecs (q/k/v) + RoPE-and-cache-write + split-K attention (2) +
  o + fused add-rms + gate + up + swiglu + down + fused add-rms. Every rms
  fuses into the residual add that feeds it (`add_rms_kernel`); the last
  layer's add fuses with `out_norm` directly. Submit ≈ 2.3 ms/step measured
  (≈ 5.5 µs/dispatch, the known cubecl encode cost).
- **Static-slot f16 KV caches**: K dim-major `[4096, 3000]` (score reads
  coalesce across positions), V position-major `[3000, 4096]` (weighted-V
  reads coalesce across dims); written in place at the stream offset — no
  cat/realloc. Hard stop asserted at 3000 frames (4 min); the ring port
  must also apply the effective-2999 window (RingKVCache wrap quirk, see
  the depformer notes above).
- **Split-K attention** (flash-decoding shape): 16 chunks × 32 heads = 512
  partial cubes + a per-head combine. The first build's one-cube-per-head
  attention left ~31/40 GPU cores idle and latency-bound the serial V
  sweep — measured ~51 GB/s effective on the KV read, +61 ms/step at fill
  3000 (format-independent, the discriminator that it wasn't weight
  traffic). Split-K brought the fill-3000 KV term to ~+9–20 ms.
- Interleaved-RoPE q/k row permutation at load BEFORE quantization (same
  conversion as `temporal.rs`); 1/√d folded into q in the RoPE kernel; norm
  alphas explicit f32 (measured a wash for q4 error — `quantcheck`: folded
  8.8e-2 vs raw 8.7e-2 per matvec, even against layer 0's 8·10⁴× in-group
  alpha range). Embeddings host-side f32 (`embed_codes` on CPU, one 16 KB
  upload/step).

**Pipeline exactness (f16 ablation): cos = 1.000000 (6 dp) on all 113
golden steps, hidden AND text logits, text argmax 113/113.** The kernels,
the RoPE permutation, the static f16 KV and the fused norms are exact;
everything below is quantization, cleanly separated from implementation.

**The lane's real finding: GGUF-q4_0 on this checkpoint is ~8.7e-2
relative RMS per matvec — not the ~3e-2 the spike quoted.** That IS the
analytic q4_0 class for Gaussian-ish weights (E[max of 32] ≈ 2.2σ ⇒ step ≈
0.275σ ⇒ err ≈ 0.08σ); 6.3e-2 is the uniform-weights figure and "expect
logits cos ~0.999x" was optimism. Compounded over 64 residual adds, q4_0
lands far from 0.999 (table). Response: a **weight-format axis**
(`WeightFmt` — q4_0 4.5 bpw / q8_0-style biased bytes 8.5 bpw / f16), same
kernels and gates throughout. **q8 is the production recommendation** — it
delivers the 0.999x class inside the frame budget.

Gate (113-step golden stream, f16 logit head, min/mean over steps):

| fmt | hidden cos min/mean | text-logits cos min/mean | text argmax | first miss |
|---|---|---|---|---|
| f16 | 1.000000 / 1.000000 | 1.000000 / 1.000000 | 113/113 | none |
| q8  | 0.999234 / 0.999823 | 0.999602 / 0.999950 | 111/113 (98.2%) | step 72 |
| q4  | 0.791 / 0.952 | 0.675 / 0.976 | 95/113 (84.1%) | step 2 |

(q4's argmax misses start at step 2 — in free-run the text stream would
diverge essentially immediately; q4_0 alone is not deliverable quality for
the text stream. The probe's per-format PASS bars are regression tripwires
calibrated to these measured classes, not quality claims.)

Bench (single-token decode, fill pinned per step, min-of-medians full
ms/step, **best observed across 3 interleaved passes** — the desktop
shared 2+ other active lanes all session, load 10–35; contention only
inflates, so per-cell minima approach the quiet floor; cells that never
caught a quiet window are marked †):

| fmt (f16 head) | fill 256 | fill 1024 | fill 3000 |
|---|---|---|---|
| q4  | 13.4 | 15.7 | 42.5† |
| q8  | 20.7 | 42.8† | 48.9† |
| f16 (1 pass) | 33.0 | 63.8† | 79.9† |

† bandwidth-floor cross-checks from the quietest cells: q8 with the q4
head at fill 3000 measured **30.3 ms** ≈ 10.2 GB (weights 7.0 + KV 3.1 +
head 0.07) at ~335 GB/s — the physics-consistent quiet-window estimate for
production q8+f16-head at full context is **~31 ms**; q4@1024 = 15.7
matches its 4.9 GB at ~310 GB/s.

**Logit head: f16 (measured decision).** Isolated A/B on the f16 stack:
the q4 head ALONE drops text argmax 113/113 → 108/113 (min cos 0.980,
mean 0.9947) to save ~190 MB ≈ 0.6 ms/step. The text stream teacher-forces
the depformer's prev-token chain — fidelity worth 0.6 ms. Both heads stay
loaded; every gate run re-measures the A/B.

**Frame budget (80 ms @ 12.5 Hz), production pick q8 + f16 head:**
temporal ~21 ms (short context) → ~31 ms (full 3000, quiet-window
estimate) + depth ~21.6 + mimi ~5 ⇒ **~48–58 ms → clears by 22–32 ms**;
q4 (13.4 → ~35 ms) clears by more, at the fidelity cost above.

**Next:** wire `TemporalMetal` into an LmGen-shaped realtime loop
(depformer as the CPU/Accelerate stage, mimi on its own thread, one GPU
sync/frame); the 3000-frame ring + effective-2999 window for unbounded
sessions; attention-kernel headroom if ctx-3000 margins ever thin
(vectorized f16 KV loads, 32 chunks); a q6_K-class middle format only if
q8's fidelity ever needs cheapening.

## PersonaPlex-7B — depth_fast, the Accelerate/NEON depformer (2026-07-11, `rt-depth-accel`)

Lane B of the realtime build: `src/models/personaplex/depth_fast.rs` — the
depformer as a fast CPU predictor (same math as the burn `depth.rs`
reference, rebuilt for the frame budget): all 16 per-step weight sets
pre-sliced at load (norm alphas + the exact 2⁻³ attention scale folded into
the matvec rows), fixed scratch buffers (zero per-frame allocation), the 16
`depformer_in` conditioning gemvs collapsed into ONE `[16·1024, 4096]` gemv
per frame, window-15 attention as a visible range over flat KV slabs. Two
storage modes: f32 through the Accelerate sgemv pool (2 AMX streams), f16
storage with f32 accumulate through a hand NEON `fcvtl`+`fmla` kernel on its
own work-stealing pool (`MARY_DEPTH_THREADS`, default 6). Gate + bench:
`moshi_depth_probe` (unit gates: NEON hdot vs scalar-f64, threaded hgemv
bit-identical to serial).

**Gate (teacher-forced 113-step oracle replay, incl. in-process `depth.rs`
cross-check):** f32 — min per-codebook logits cos vs `dep_logits` =
**1.000000000 over all 16 codebooks** (max|Δ| 1.5e-5), argmax **1808/1808**,
next_text 113/113, out frames 25/25, vs-depth.rs tokens 1808/1808 exact.
f16 — min cos **0.999999851** (max|Δ| 6.6e-3 on raw logits), argmax
1808/1808, out frames 25/25, vs-depth.rs 1808/1808. Both PASS (bar 0.99999);
the checkpoint is bf16 so f16 storage is exact for every weight ≥ 2⁻¹⁴ —
same numerics family, not a q4-style relaxation.

## PersonaPlex-7B — the assembled realtime pipeline (2026-07-11, `voice-realtime`)

Integration of the two lanes: `RealtimePipeline`
(`src/models/personaplex/pipeline.rs`, feature `q4`) = Metal quantized
temporal ([`temporal_metal`], f16 logit head) + `depth_fast` (f32 Accelerate
default, `--depth-f16` optional) + CPU Mimi, driving the SAME `StreamCache`
delay/undelay bookkeeping as the parity `VoicePipeline` — one struct, same
step contract, fast stages. Mimi runs sequentially (batch encode up front,
batch decode at the end in the gate flow; measured ~1.2–1.6 s for 25 frames
≈ 50–60 ms/frame batch, the live-loop per-frame cost is the ~5 ms class
projected earlier); the own-thread streaming split waits for a live session
loop to consume it. Gate: `personaplex_rt_probe pipeline [q4|q8|f16]` —
free-runs the oracle's golden input flow (WAV → Mimi encode → prompts → 25
user-audio frames → decode → `/tmp/mary-personaplex/rt_pipeline_out.wav`).

**Free-run results (vs the f32-oracle goldens; all three formats PASS their
per-format bars):**

| | f16 stack | q8 stack | q4 stack |
|---|---|---|---|
| mimi encode vs `user_codes` | 200/200 | 200/200 | 200/200 |
| text-logits cos on shared prefix (min/mean) | 1.000000/1.000000 | 0.999602/0.999946 | 0.674800/0.973433 |
| out frames vs `out_tokens` | **25/25 exact** | 1/25 | 1/25 |
| user-pred streams 9..=16 in out frames | 200/200 | 200/200 | 200/200 |
| first committed divergence | **none** | step 88 (first sampled text) | step 88 (first sampled text) |
| logits cos at divergence | — | 0.999966 | 0.988638 |
| pre-divergence audio cos vs oracle | 0.999999999 (25 frames) | 1.000000000 (1 frame) | 1.000000000 (1 frame) |
| gen-phase ms/step (median, fill ≈ 113) | 60.8 | 61.0 | 48.7 |

**The divergence is one near-tie coin flip, measured.** Step 88 is the
FIRST free-run-sampled text token (every earlier step is teacher-forced by
the prompt flow — prompt phases cannot diverge, only the KV cache carries
quantization noise). At that step the ORACLE's own top-2 logit gap is
**0.0200** (token 3 over token 0) — a knife-edge the f32 pipeline happens
to fall one side of. q8 (logits cos 0.999966 at that step) flips it by
0.0134; q4 (cos 0.988638) flips it by 1.09; in both, the golden token ranks
#2. The f16 stack preserves the tie's winner and reproduces the oracle
token stream exactly (out frames 25/25, audio cos vs the streaming-oracle
decode 0.999999999) — the wiring-exactness ablation that separates
"integration bugs" from "quantization flips a tie". After the flip the
trajectories are legitimately different autoregressive paths; the 1/25 and
post-divergence agreement numbers are path comparisons, not fidelity
measurements. Provided user codes round-trip the ring exactly in all
formats (the bookkeeping invariant).

**Timing (loaded desktop, small-context regime):** assembled step = q4
~48.7 / q8+f16 ~61 ms median at fill ≈ 113, sequential LM→depformer;
depformer in situ 25.8–35.7 ms/frame across runs (contention band around
lane B's ~21.6 ms quiet-window bench). Inside the 80 ms budget at short
context; the full-context margin story stays the lane-A projection
(temporal ~31 ms q8 / ~43 q4 at fill 3000) — the next squeeze if ctx-3000
margins thin is overlapping the depformer's 16 sequential steps with the
NEXT frame's temporal submit, plus the mimi thread.

**Next:** a live session loop (microphone frames in, speaker frames out,
mimi on its own thread, sampling instead of greedy — greedy at a 0.02-logit
tie is the pathology above; real sessions sample), the 3000-frame ring +
effective-2999 window, and the sampling knobs (temp/top-k) that LMGen
stubs out today.

## PersonaPlex-7B — realtime build: full-frame latency (2026-07-11, `voice-realtime`)

The measurement that decides whether the pipeline runs realtime:
`personaplex_rt_probe framebench [q4|q8|f16]` times the wall clock per
EMITTED FRAME — temporal step (host embed + 418-dispatch submit + GPU drain
+ hidden/logits readbacks) + the 16 sequential CPU depformer steps + mimi
decode OF THAT FRAME + LMGen bookkeeping — at temporal-cache fills 256 /
1024 / 2999 (the static KV cap), reached by a genuine 3000-step free-run on
synthetic SINE user frames (timing is value-independent). 80-step windows
ending at each target fill, min-of-medians over 5 × 16-step rounds (the
spike methodology), raw best/worst kept as the contention band.

### The mimi lane was a scalar loop, not a ~5 ms stage

The first run measured mimi decode at **71.5 ms for one frame, ~46 ms/frame
marginal** — not the ~5 ms the spike projection allotted, and alone more
than half the 80 ms budget (this is also where the parity gate's
"1.2–1.6 s / 25 frames batch" came from; it was never overhead, it was
compute). Cause: `CpuTransConv::forward` — the SEANet upsample stages'
transposed convs, ~105 MMACs/frame — was a fully scalar `ic × oc × ti × k`
loop, the ONE conv class in the decoder that never got the im2col-GEMM
treatment (the same 10× class of fix as the qwen3tts codec's im2col
rewrite). Fix: dense (`groups == 1`) transconvs now run as one Accelerate
sgemm over a `[out·k, in]` weight re-layout + a col2im scatter-add; the
depthwise `upsample` keeps the scalar path (it is a few k·L MACs). Gated by
a unit test (GEMM path ≡ scalar reference on a SEANet-shaped case) and by
the f16 `pipeline` audio gate below.

| mimi decode (CPU, stateless batch of t frames) | scalar (before) | sgemm+col2im (after) |
|---|---|---|
| 1 frame | 71.5 ms | 6.6–10.7 ms |
| marginal ms/frame (Δtotal/Δframes, 25→50) | ~46 | **1.7–3.1** |

The in-loop per-frame figure below is the honest live-loop shape today: a
1-frame stateless decode every frame (~5–7 ms/frame of it is per-call
overhead a streaming/stateful decoder would amortize toward the ~2–3 ms
marginal).

### Full-frame latency (ms, min-of-medians; loaded desktop)

Machine state during the runs: load average 10–39, a sibling 12-core
triblespace test suite ran through the whole scalar-mimi baseline and most
of the first q4 run; the resident ambient floor is ~2 P-cores (the Reachy
daemon + an archive job) that never stops on this desktop. Contention only
inflates — per-window raw bests approach the quiet floor.

BASELINE (scalar mimi, heaviest contention window — kept as the honest
"before"): q4 frame = 223.2 / 121.5 / 159.0 ms at fills 256/1024/2999, of
which mimi(1f) 123.8 / 60.5 / 87.8.

With the GEMM transconv fix:

| ms (min-of-medians) | fill 256 | fill 1024 | fill 2999 |
|---|---|---|---|
| **q4 frame** (quietest pass) | **56.5** | **70.8** | **73.5** |
| · temporal (of which host submit) | 24.0 (2.5) | 26.4 (3.9) | 28.6 (4.0) |
| · depformer | 26.0 | 34.9 | 34.8 |
| · mimi (1-frame decode) | 6.3 | 9.5 | 9.8 |
| · other (bookkeeping) | 0.0 | 0.1 | 0.1 |
| · raw best/worst | 53.1 / 63.1 | 66.6 / 123.6 | 70.2 / 376.2 |
| q4 frame, first pass (12-core sibling live) | 73.3 | 81.5 | 88.8 |
| **q8 frame** (one pass, archive-wave ambient) | 81.5 | 80.5 | 81.2 |
| · q8 temporal / depth / mimi | 27.5 / 39.4 / 12.2 | 28.2 / 39.4 / 12.3 | 28.9 / 39.5 / 12.4 |

The fill-256 q4 window is the only one that caught the ambient trough and
shows the real floor: its raw WORST frame was 63.1 ms — the whole window
under budget. The @1024/@2999 windows carried an archive-job wave
(depformer +9 ms vs its own @256 measurement).

### Where the frame goes, vs the spike projection

The spike projected 47.6 / 48.9 / 52.2 ms at ctx 256/1024/3000 (temporal
q4 ~15.6 + depth ~21.6 + mimi ~5 + submission ~5). Decomposition of the
measured frame:

- **temporal**: 24.0 → 28.6 ms from fill 256 → 2999 (q4; q8 27.5 → 28.9,
  strikingly flat). ~4–8 ms above the lane-A step bench at short fill —
  the in-situ step pays the second readback (text logits `[32000]`), the
  per-step input upload, and coexistence with the process's own CPU pools
  — but grows only +4.6 ms to full context: the split-K attention's KV
  term lands milder in situ than the lane's contended fill-3000 cells
  (42.5†) suggested. Host submit is 2.5–5.4 ms of it.
- **depformer**: the variance term. Lane B's quiet standalone bench is
  21.6 ms/frame; in situ the per-window means ranged 26.1 (ambient
  trough) to 60.8 ms (gemv share 23.0–52.6 — the 2 AMX streams losing
  bandwidth to ambient CPU work and, plausibly, the per-frame GPU-phase
  interleave keeping the memory path off its boost state; QoS is already
  pinned in both pools). Even the trough sits ~4.5 ms over the standalone
  bench — that part is structural interleave cost, not contention.
- **mimi(1f)**: 6.3–12.4 ms — the fixed lane, back in the spike's class
  (was 60–124 under the scalar loop). ~5–7 ms of it is per-call overhead
  a streaming decoder would amortize toward the 1.7–3 ms marginal.
- **other** (prepare/argmax/commit bookkeeping): ~0.1 ms — free.

### Verdict

**q4 clears the 80 ms budget at all three fills in the representative
run: 56.5 / 70.8 / 73.5 ms (margins 23.5 / 9.2 / 6.5) — the 12.5 Hz
duplex loop is real on this hardware at q4 fidelity.** The gap to the
spike projection (47.6/48.9/52.2) lives in the depformer's in-situ
contention band (+5–13 ms over its 21.6 quiet floor) and the temporal
step's in-situ overhead (+5–8 ms), not in any single broken component.
**q8 — the fidelity recommendation — is NOT yet measured under budget:**
81.5 / 80.5 / 81.2 in the one pass it got, every window carrying the
~13 ms depformer wave. Its quiet-floor projection (~29 temporal + ~26
depth + ~12 mimi ≈ 67 ms) says it fits, but that is a projection; it
needs either a quiet pass or lever 1 below before claiming realtime.

Numerics validation of the mimi change: `personaplex_rt_probe pipeline
f16` PASS after the rewrite — out frames 25/25 token-exact, audio vs the
streaming-oracle decode cos = 0.999999999 (max|Δ| 4.6e-8, unchanged
class), and the gate's batch decode dropped 1.2–1.6 s → 0.2 s for 25
frames. Unit test `transconv_gemm_matches_scalar` pins GEMM ≡ scalar.

Remaining levers, in value order (none taken tonight — measurement lane):
1. **Overlap the depformer with the next frame's temporal submit** — the
   GPU is idle during the 26–59 ms depformer window and the depformer
   depends only on the already-read hidden state; pipelining hides most of
   the depformer behind the temporal step and turns the frame into
   max(temporal, depth) + mimi + ε. This is the single biggest margin win.
2. **Streaming mimi decoder state** — turns the 1-frame decode's ~5–7 ms
   per-call overhead into the ~2–3 ms marginal.
3. `--depth-f16` (NEON pool, half the weight traffic) if AMX contention
   stays the limiting term on the shared desktop.

## PersonaPlex-7B — Phase 5: the prompt machinery (2026-07-11, `voice-realtime`)

The rung that makes voice-prompt assembly self-contained: mary now assembles the full
system-prompt flow from PRIMARY sources — a packaged voice `.pt` + a raw
system-prompt string — instead of the golden npys the parity gates
bootstrapped with. Files: `src/models/personaplex/spm.rs` (pure-Rust
SentencePiece unigram tokenizer), `voice_prompt.rs` (torch-pickle `.pt`
reader), `prompt.rs` (`wrap_with_system_tags` + `Prompt::build`), both
pipelines grew `run_prompt(&Prompt)`, `personaplex_probe` grew `prompt`
(the gate) and `ownprompt` (model-free assembly smoke for any voice),
`golden/capture_spm_battery.py` (tokenizer battery golden) and
`golden/build_voice_prompt.py` (the upstream WAV→`.pt` flow, driven).

**SPM tokenizer (`spm.rs`) — hand-rolled, zero deps, and exact.** The
checkpoint's `tokenizer_spm_32k_3.model` is the best possible porting case,
all asserted from the ModelProto at load (a different model file fails
loudly): UNIGRAM model, **identity normalizer with an EMPTY
`precompiled_charsmap`** (no NFKC, no double-array trie — normalization is
exactly "prepend one dummy-prefix space, escape ` `→`▁`"),
`remove_extra_whitespaces=false`, `byte_fallback=true` (all 256 `<0xXX>`
pieces). Encode = Viterbi over piece log-probs at UTF-8 char boundaries;
one-char UNK edges (score `min−10`, argmax-irrelevant — UNK never competes
with a known char) decompose into byte pieces post-Viterbi; CONTROL/UNK/BYTE
pieces never match from text; ties strict-`>` first-best. The convention
trap for once was the ABSENCE of one: `<system>` is NOT a special token —
it tokenizes as `▁<`+`system`+`>` through the plain encoder. Gates:
**26/26** on the capture's exact system prompt vs `text_prompt_tokens.npy`,
**25/25 strings** (338 tokens) on the oracle-venv battery
(`spm_battery.json`: whitespace runs, tabs/newlines, German/CJK/
Korean/Cyrillic, emoji→byte-fallback, the `▁` metasymbol itself, URLs).

**Voice-prompt `.pt` reader (`voice_prompt.rs`) — parsed natively, no
python at runtime.** A packaged voice is `torch.save({"embeddings":
[N,1,1,4096], "cache": [1,17,4]})`; torch's `.pt` is an UNCOMPRESSED zip
whose pickle only ever calls `_rebuild_tensor_v2` — a targeted ~150-line
stack machine covers it (anything unexpected panics with the opcode, no
guessing). Stock voices are bf16 with `cuda:0` LongStorage caches — the
upstream `map_location` trap is just a device *string* here, and bf16→f32
is a bit-shift, so NATM0.pt loads **BIT-exact vs `vp_embeddings.npy`
(204800/204800 f32 values) + `vp_cache` 68/68**. f32/f16 storages also
supported (voices built on CPU save f32).

**The assembled-stream gate (`personaplex_probe prompt`) — the strongest
one:** free-runs the real `LmGen` with the prompt assembled from
NATM0.pt + the raw prompt text (vp replay → cache overwrite → 6 silence →
26 text tokens → 6 silence → the golden user codes), gating every surface
integer-exact vs the oracle: model inputs **63/63** vs
`step_tokens`/`step_token_idx`, dep tokens **113/113 × 16**, out frames
**25/25 × 17** — PASS (113 steps in 890.6 s = 7.88 s/step: the NdArray
parity path on a load-16–22 machine, contended). `e2e` (golden-fed) stays
as the ablation separating LM wiring from prompt assembly.

**A custom voice prompt — built and loading.** `golden/build_voice_prompt.py`
drives the upstream WAV→`.pt` flow (`LMGen.load_voice_prompt` — load +
−24 LUFS normalize — then `_step_voice_prompt` with
`save_voice_prompt_embeddings=True`; the exact code path `offline.py run()`
exposes for non-`.pt` voice prompts, so this is NVIDIA's flow, not an
invention — with one honest caveat: `normalize_audio` lazily imports
`pyloudnorm`, an UNDECLARED upstream dep, pip-installed into the venv).
ref_voice.wav (10.4 s, 130 frames) → **`ref_voice.pt`: 129 f32
embedding-replay frames + cache** (the first frame's codes are consumed by
the offset-0 seeding step that returns before the model — same off-by-one
as NATM0's 50, stock voices were built from 51-frame clips). CPU f32 build:
31 s of LM stepping on a quiet machine. Durable copy:
`models/ref_voice_prompt.pt` (next to the piles). mary loads it
through the same parser (`ownprompt` smoke: 129 frames, 0 non-finite,
cache in range, 174-step prompt stream assembled with a custom system
prompt, all 45 token-fed inputs in-vocabulary — tokenization of the custom
prompt cross-checked 33/33 vs the venv).

**Next:** the live duplex loop — mic/speaker streaming around
`RealtimePipeline` (mimi on its own thread, sampling instead of greedy —
LMGen's temp/top-k knobs are still stubbed) with
`Prompt::build(ref_voice_prompt.pt, …)` as the session opener; or first the q8 quiet-window
re-measure that decides the production format's realtime claim.

## Gemma-4-31B — the large background LLM (2026-07-11, `voice-realtime`)

The dense flagship of the Gemma 4 lineup (`google/gemma-4-31B-it`,
text+vision — this checkpoint has `audio_config: null`, so audio stays
on the E4B lane, not here) persisted to the model shelf as
`models/gemma_31b.pile` and gated. **No new architecture** — the gemma4
port (`src/models/gemma/gemma4/`) already covered E2B/E4B and the loader +
persist path were built shape-generic; the 31B is those same paths driven
at flagship dims. The whole codebase already anticipated it (the streaming
loader's doc: "scales weights-as-tribles to the dense 31B"; `gemma_gen`'s
`f16gen` feature: "the 31B fits 128GB only at f16 — streamed from the pile
either way"). So the work was execution + the parity/fit gates, not a port.

**Config (from the checkpoint `config.json`, no minting — the generic
`Gemma4TextConfig`/`Gemma4VisionConfig` serde-parse it; unknown fields are
ignored):** text = hidden 5376, 60 layers, 32 heads / 16 KV heads, head_dim
256, **global_head_dim 512, num_global_key_value_heads 4, `attention_k_eq_v:
true`** (K=V in global layers halves that KV cache), interleaved 5×sliding /
1×full (10k / 1M θ, full uses `partial_rotary_factor 0.25` proportional
RoPE), sliding_window 1024, `tie_word_embeddings: true`, no PLE
(`hidden_size_per_layer_input: 0`), no MoE, `use_bidirectional_attention:
"vision"`, `final_logit_softcapping 30`, vocab 262144. Vision =
`gemma4_vision`, 27 layers, hidden 1152, head_dim 72, patch 16,
pooling_kernel 3, 280 soft-tokens/image, `standardize: true` (uses the
std_bias/std_scale buffers). Tensor shapes are asserted implicitly on
persist (every leaf's shape stored) and on load (each `load_2d`/`load_3d`
reshape must match); a wrong config would panic in the reshape.

**Persist (F16 leaves, `gemma_persist`, unchanged bin):** 1188 tensors from
the 2 bf16 shards → `models/gemma_31b.pile`, **58.25 GiB in 77 s**. Same
model-agnostic `persist_safetensors_to_pile` that made `gemma_e4b.pile`
(bf16 → f32 widen → f16 store; the only lossy hop is bf16→f16, safe since
weights sit in f16's range). q4 secondary entity deferred — recommendation:
skip it; the f16 pile already leaves >50 GiB headroom next to PersonaPlex
(below), so bandwidth, not capacity, would be q4's only motive, and the
31B is a background/reflective model, not the realtime hot path.

**Text parity vs HF-transformers (the house cross-check).** Same chat string
both sides (`<bos><|turn>user\n{prompt}<turn|>\n<|turn>model\n`), tokenized
`add_special_tokens=false` — **21 token ids identical** across mary's
`tokenizers` and HF's `AutoTokenizer`. mary loads f16 streamed from the pile
(`gemma31b_parity` bin); HF runs bf16 on MPS (`Gemma4ForConditionalGeneration`,
transformers 5.6.0.dev0). Final-position logits over the full 262144 vocab:
**cosine = 0.999937** (clears the ≥0.9999 bar), **argmax match: token 100
`<|channel>`** (31B-it is a thinking model — it opens a `<|channel>thought`
span), top-5 identical up to a rank-4/5 swap of two near-tied logits,
max|Δlogit| 0.56 / mean 0.081. The residual gap is exactly the mary-f16 vs
HF-bf16 precision difference (not a port bug) — the argmax and cos confirm
the decoder + lm_head are numerically faithful. Standalone greedy sanity:
"What is the capital of France?" → `<|channel>thought\n<channel|>Paris`
(load 131 s streaming, gen 13.6 s for 12 tokens, f16).

**Vision encoder — loads + forwards; one real bug fixed, one f16 follow-up
found.** All **356 vision tensors** are in the pile (persist ingests every
float tensor); with `vision_config` kept, the streaming loader builds
`Gemma4VisionEncoder` (patch embedder + 27 layers + embedding_projection +
`std_bias`/`std_scale` — the `standardize: true` path E2B/E4B never had).
`gemma31b_vision` runs a forward on a synthetic 12×12 patch grid → soft-token
output **[16, 5376]** (4×4 pooled tokens by `pooling_kernel 3`, projected into
the 5376-dim text space). Two things surfaced because the 31B is the first
Gemma-vision run under the **f16 (`BHalf`) backend** (the E-series only ran
vision at f32 via `gemma_hear`):
- **Fixed:** `vision.rs:382` did a bare `hidden_states.to_data().to_vec::<f32>()`
  in the spatial-pool scatter — TypeMismatch panic under f16. Now
  `.convert::<f32>().to_vec()` (width-agnostic, no-op at f32). This is a
  genuine correctness fix for any f16 vision forward.
- **Follow-up (f16 numerics):** with the panic gone, the forward completes and
  the output is structurally correct, but **4 of the 16 pooled tokens overflow
  to NaN** — the vision tower is bf16-native (the `scale = sqrt(1152) ≈ 33.9`
  pooler multiply + the 10240-wide position-embedding accumulation exceed
  f16's 65504 range). Same class as the nomic-mm7b note ("this bf16-native
  model overflows f16's range… upcasts weights per-op so activations run in
  f32"). The text decoder does NOT hit this (parity is exact above); the fix
  is per-op f32 upcast inside the vision tower, deferred as a follow-up since
  the 31B's job here is text and audio is out of scope. Under the **f32
  backend the same forward is all-finite** (confirming overflow, not a logic
  bug). A full image→text vision *parity* gate (HF pixel preprocessing +
  splicing the 280 soft-tokens into the decoder at `image_token_id 258880`)
  is the larger follow-up — the E-series never had an image-forward harness
  either.

**Concurrent fit — 31B-dense is VIABLE, no MoE fallback needed
(`gemma31b_fit`).** Held PersonaPlex-7B (**15.77 GiB f16**, 8.47 G params
incl. codec+speaker encoder, materialized onto the GPU) AND Gemma-4-31B
(**58.25 GiB**, zero-copy aliased via `load_gemma4_aliased_from_pile`)
resident on the same Metal device, then ran a 31B forward with PersonaPlex
still live (both correct: 31B → "Mars" for "Name one planet"). **Peak RSS =
73.54 GiB on 128 GiB → 54.46 GiB headroom** — ample for KV caches +
activations during concurrent operation. So the dense 31B is the
background model; the 26B-A4B MoE fallback stays unbuilt (flagged only, not
needed).

**Artifacts.** Pile: `models/gemma_31b.pile` (58.25 GiB, f16, `main`
branch). New bins (all `gemma`/`f16gen`): `gemma31b_parity` (logit dump for
the HF cross-check), `gemma31b_fit` (concurrent-fit peak RSS),
`gemma31b_vision` (vision load + sanity forward). The HF reference +
comparison scripts live under `/tmp/gemma31b_parity/` (throwaway).

## Qwen3-TTS — the zero-copy RAW talker lane (2026-07-12, `qwen3tts-zerocopy`)

Design directive: "mary shouldn't have any non-zero-copy implementations —
integrating with the zero-copy nature of triblespace is one of its main
points." The talker was the holdout: it rides the FUSED backends, and burn
0.21's fusion codegen miscompiles graphs over many externally-registered
buffers (the deleted 2026-07-04 probe), so zero-copy and fusion are
mutually exclusive at this burn version. This lane makes the talker
loadable on the RAW f16 backend (`MARY_SPEAK_RAW=1`) with every GPU tensor
a Metal `newBufferWithBytesNoCopy` view of the mmap'd pile pages.

**The fold problem, resolved as derive-time folds (option a).** The
talker's load-time layout work (wide `[qk | R(qk) | v]` matmul with
rotate_half pre-applied to weight ROWS, norm weights folded into matmul
rows, pre-transposed Linears, q/k-norm × 1/√d chain weights) means the hot
weights are TRANSFORMS of checkpoint leaves — they can never be mmap views
of the canonical pile. `qwen3tts_persist --fold-derive` therefore loads
the production fused-f16 talker once, reads back every fold-transformed
tensor, and writes them to a NEW derived sibling pile
(`<stem>_folded_f16.pile`, extends the voxtral `_f16` convention: same
model, derived layout, half width) under `talker.folded.*` names. The
sibling's bytes are BY CONSTRUCTION bit-identical to the weights the
production lane computes at load. The untransformed big tensors (text +
codec embeddings, 634 MB) are NOT duplicated — the folded loader aliases
them straight from the canonical pile's `talker_f16` leaves. CPU stages
(code predictor, codec-head gemv, codec-embedding rows) keep reading the
exact f32 leaves.

**Measured facts (gates green):**
- `--fold-derive`: 173 tensors / 2.97 GiB → `qwen3tts_folded_f16.pile`
  (3.19 GB file); inline gate re-aliases every leaf and compares
  bit-for-bit — PASSED. Canonical pile length AND sha256 verified
  unchanged after the derive.
- `qwen3tts_raw_gate` gate 1: the fold-transformed weights of (a) the
  production fused talker, (b) the raw talker folding at load, and
  (c) the zero-copy folded-alias talker are BIT-IDENTICAL, all 173
  tensors — the folds are exact wherever they run, and fold-location
  (fused GPU / raw GPU / derive-time) does not exist as a numeric axis.
- The gate is one-process-per-lane (dump + compare): mixing the fused
  and raw backends in one process corrupted the fusion op stream on the
  first attempt ("Ordering is bigger than operations" → CallError), and
  the single-backend fused lane then exposed its own codegen bug (see
  below) — the raw lanes never crashed once.
- `qwen3tts_raw_gate` gate 2 (the STRICT identity gate for the zero-copy
  change): the ordinary-loader raw talker and the folded-alias raw talker
  produce BYTE-IDENTICAL hidden states over a 154-token prefill + 32
  teacher-forced decode steps — compared across two separate processes, so
  this simultaneously observed the raw backend reproducing itself for this
  graph (fixed kernel selection, no autotune jitter at these shapes).
  *(2026-08-18: that observation is a diagnostic, not a property the lane owes
  anyone. Run-to-run determinism is not a gate — wiki:f5dcc88988bb28e472e50fa030332adb.
  The gate this bullet describes is a LOAD-PATH gate, and bit equality is right
  there because neither lane computes anything the other does not.)*
  Codebook-0 argmax streams identical. The zero-copy load cannot change
  the voice relative to the raw lane.
- Cross-backend identity (raw vs fused) is NOT bit-level and cannot be:
  the backends compile different kernels (different fp association), the
  seeded sampler amplifies any logit epsilon by AR cascade — the seed-7
  LONG fixture renders 430 frames on fused vs 412 on raw. This is the
  same class of divergence the lane accepted for f32→f16 (documented in
  the probe); the identity gates the lane trusts for BACKEND changes all
  pass: resemblyzer vs reference 0.957 (fused) / 0.950 (raw zero-copy),
  pairwise 0.987 (historical accept band 0.91–0.946); whisper transcripts
  word-equivalent (same fixture mishearing in both). WITHIN the raw
  backend, identity is absolute: the folded-alias, leaf-aliased, and
  fully materialized loads render the BYTE-IDENTICAL 412-frame wav
  (sha e1862afb…) across separate processes.

**Staleness verdict — "fusion buys the talker 2×" is stale.** Interleaved
A/B (LONG fixture, seed 7, per-component `QWEN3TTS_BENCH`), calm window
(1-min load 4.4–6.0), best pass per lane:

| per frame | fused-f16 (production) | raw-f16 zero-copy |
|---|---|---|
| talker submit | 13.0–15.1 ms | 18.6–19.0 ms |
| sync (GPU drain) | 14.6–15.9 ms | 8.4–9.8 ms |
| talker GPU total | **27.6–31.0 ms** | **27.4–28.4 ms** |
| frame total | 60.2–61.7 ms | 60.6–63.1 ms |
| audio-rate steady | 1.30–1.33× | 1.27–1.32× |

The raw lane trades ~+5 ms submit (per-op encoding, no graph capture) for
~−6 ms sync (work reaches the GPU earlier, less left to drain at the
read-back) — a wash. The 2× era predates the wide-matmul folds, the CPU
predictor/logits move, and one-sync-per-frame; with ~500 ops/frame the
loop is host-submission-bound on EITHER backend, and fusion's graph
capture costs about what its dispatch-count savings buy back. (Round-2/3
numbers under load 6–70 stay within the same envelope, increasingly
noisy — contention-labeled in /tmp/qwen3tts_zc_bench logs.)

**Load + memory.** Calm window (interleaved A/B rounds; second round =
warm page cache):

| load path | weight load | max RSS |
|---|---|---|
| fused-f16 (production, native-width upload) | 6.5 s / 4.2 s warm | 7.1 GB |
| raw-f16 folded-alias (zero-copy) | 5.7 s / **2.9 s** warm | 6.5 GB |

And the zero-copy vs materialized split (same-window contended pair,
load 20–90 — labels apply):

| load path | weight load | max RSS | peak footprint |
|---|---|---|---|
| raw talker, folded-alias (zero-copy) | 10.2 s | 6.4 GB | **3.6 GB** |
| raw talker, materialized (A/B switch) | 21.2 s | 16.7 GB | 20.6 GB |

(macOS `ru_maxrss` counts touched mmap pages as resident, so the RSS gap
understates the win; `peak memory footprint` — anonymous/dirty memory —
is the honest axis: 3.6 vs 20.6 GB. First-use page-in is folded into the
zero-copy lane's prefill/TTFA and was not separable from JIT under the
night's contention; the warm-cache load-time drop 5.7→2.9 s bounds it.)

The zero-copied bytes: 2.97 GiB folded weights + 634 MB embeddings (the
canonical `talker_f16` leaves) = ~3.6 GiB of talker never copied — the
GPU buffers ARE the pile pages (first-touch page-in, evictable, shared
across processes). Still copied: RoPE tables (computed, ~4 MB), the ECAPA
speaker encoder (materialize+cast, small), the CPU stages (predictor /
codec-head / codec-embedding rows, exact f32 leaves — a follow-up:
`anybytes` views could make these mmap-backed too), and the codec
(fused backend, direct upload — blocked on the same burn fusion bug).

**Default policy.** `MARY_SPEAK_RAW=1` opts into the raw zero-copy lane;
the fused lane stays the default until review (the raw lane is at
perf parity, loads with ~10 GB less transient footprint, is run-to-run
deterministic, and its load path is the project-native zero-copy one —
the case for flipping the default is strong, but voice-default changes
get a second pair of ears first). Flipping = swapping the branch in
`speak::synthesize_stream`. *(Superseded 2026-07-12: the ear-gate passed
and the fused lane was removed outright — see "raw is the ONLY talker
lane" below.)*

**Gate harness shape (final):** `qwen3tts_raw_gate --lane <l> --out <d>`
one process per lane (fused weights dump always works; the fused FORWARD
panics fusion on every synthetic graph tried — bare select entry, fresh
upload entry, production-shaped codec+text add entry — so it degrades to
weights-only and `--compare` skips the synthetic cross-backend cos,
resting cross-backend identity on the e2e gates). Final run: gate 1
weights 173/173 bit-identical ✓, gate 2 raw≡folded bit-exact ✓, gate 3
documented-skip, PASSED.

**New burn-fusion codegen bug found (gate collateral):** feeding a bare
embedding select — or even a freshly uploaded plain tensor — straight
into the talker stack's first rms-reduce on `BFusedHalf` deterministically
panics `burn-cubecl-fusion` (`FusedReduceLaunch` strides index out of
bounds → fusion stream corrupt, CallError). Production dodges it because
`build_prefill` feeds the stack from cat/add-built graphs. Third distinct
fusion-codegen failure on this model (miscompile over external buffers,
two-backend JIT corruption, now this) — the raw lane exists precisely so
the pile seam never depends on that machinery.

## PersonaPlex-7B — realtime frame tax: dispatch fusion + vectorized kernels (2026-07-12, `personaplex-frametax`)

Shaving the fixed frame tax of the realtime loop, with the PRIORITY FLIP in
force: **f16 ≤ 80 ms at fill 3000 is the primary target** (fall back to q8
only if forced). Every lever is a scheduling/fusion change — the arithmetic
(per-row dot-product order, reduce trees, silu expression, RoPE math) is
untouched, and the whole gate battery reproduces the pre-change numbers TO
THE PRINTED DIGIT per format (bit-identity as a regression invariant, not a
re-calibration).

### The dispatch census (what the 418 were, what the 290 are)

Before: 13 dispatches/layer — q, k, v matvecs (3) + rope/cache (1) + split-K
attention (2) + o (1) + fused add-rms (1) + gate, up matvecs (2) + swiglu
(1) + down (1) + fused add-rms (1) — ×32 + initial norm + head = **418**,
plus TWO sequential blocking readbacks (hidden `[4096]`, then logits
`[32000]`).

After: 9/layer — **qkv fused into ONE `[12288, 4096]` matvec** (the same
rows moshi's `in_proj_weight` ships; q4/q8 groups run along the input dim,
so row concatenation encodes bit-identically), rope/cache reads the single
fused buffer (one binding instead of three); **gate+up+swiglu fused into ONE
`[22528, 4096]` matvec with interleaved rows** (even = gate_j, odd = up_j:
each 8-row cube owns whole pairs, and lane 0 of each even row applies
silu(g)·u in the reduce epilogue — the standalone swiglu kernel and the
gb/ub buffers die); o + down + the two add-rms fusions + split-K pair
unchanged — ×32 + initial norm + head = **290**, ONE combined readback
(hidden + logits in a single staging flush + poll).

### Vectorized loads (bit-order-preserving)

- **f16 matvec**: the eight scalar 2-byte weight loads + eight scalar f32
  activation loads per lane-iteration become two `vec4<f16>` + two
  `vec4<f32>` loads; components are consumed in the original order, so the
  FMA sequence is unchanged. This is the f16-priority lever: the f16 stack
  is pure weight bandwidth (13.2 GB/step) and the scalar build left
  load-issue width on the table.
- **q4/q8 matvecs**: activations likewise vec4-ized (16→4 and 8→2 load
  instructions per iteration); the nibble/byte unpack chains are
  ALU-issue-bound, so load-issue width is the cheap win. Weight words were
  already u32.

### Mimi decode off the LM critical path (framebench = live-loop shape)

Mimi decode is downstream-only — frame t's PCM feeds no later step — so
`framebench` now sends every emitted frame to a decode worker thread
(sync_channel(8)); the frame column is the LM critical path (prepare +
embed + submit + drain + depformer + commit + handoff) and the worker-side
decode cost is reported alongside (it only has to stay under the frame
budget, which it clears ~10×; a live session loop should adopt the same
pattern, and mimi ENCODE of the next frame overlaps the same way).

### Gate battery (all green, all numbers IDENTICAL to the pre-change runs)

- `gate f16`: min/mean hidden AND logits cos 1.000000, argmax **113/113**,
  max|Δ| 1.248e-2 / 8.406e-3; q4-head A/B 0.980130 / 0.994736, 108/113,
  first miss step 61 — every digit identical before/after each lever.
- `gate q8`: 0.999234/0.999823 hidden, 0.999602/0.999950 logits,
  **111/113**, first miss step 72 — identical.
- `gate q4`: 0.791089/0.952289 hidden, 0.674800/0.976378 logits, **95/113**,
  first miss step 2 — identical.
- `pipeline f16`: out frames **25/25 exact**, no committed divergence, audio
  cos vs streaming-oracle decode 0.999999999 (max|Δ| 4.552e-8) — identical.
- `pipeline q8/q4`: the SAME single near-tie divergence signature as
  documented — step 88 (first free-run-sampled text token), q8 flips it by
  0.0134 (logits cos 0.999966), q4 by 1.0904 (cos 0.988638), golden token
  ranks #2 in both — identical.
- `reset f16` (greedy + seeded sampling): PASS after repairing a LATENT GATE
  BUG from a9501cc — the sampling arm's run 1 started on the pipeline still
  dirty from the greedy arm (temporal offset ≈ the schedule length), so it
  attended over the greedy KV prefix and emitted one extra out frame; the
  arm could never pass and had not been run to green against the pile
  before. The gate now resets before each mode's run 1 (a no-op on the
  fresh load), making it test its own claim: fresh == reset == reload,
  token-exact, in both modes.
- `cargo test --features personaplex,q4 --lib`: 20/20; clippy clean of new
  warnings (kernel `%` gets a targeted allow — `is_multiple_of` is not a
  cube intrinsic).

### Measured effect — the definitive quiet pass (loadavg 2.8-4.1 throughout)

Single-token temporal step (`bench`, full ms, f16 head; the BEFORE row is
the rt-q4-temporal session's best-of-3 contended cells — † never caught a
quiet window, so compare the 256 column tightly and the depth columns as
upper bounds):

| fmt (f16 head) | fill 256 | fill 1024 | fill 3000 |
|---|---|---|---|
| q4 BEFORE | 13.4 | 15.7 | 42.5† |
| **q4 AFTER** | **12.4** | **13.3** | **15.4** |
| q8 BEFORE | 20.7 | 42.8† | 48.9† |
| **q8 AFTER** | **18.7** | **19.4** | **22.4** |
| f16 BEFORE | 33.0 | 63.8† | 79.9† |
| **f16 AFTER** | **32.3** | **32.8** | **35.0** |

The fill-256 cells barely moved (those were already near the weight-
bandwidth floor); the depth cells collapsed. The fill-256→3000 tax is now
+3.0/+3.7/+2.7 ms (q4/q8/f16) — bandwidth-consistent with the 1.44 GB of
extra KV traffic (~500+ GB/s effective on the attention pair, up from
~50-90 GB/s). Lever attribution at f16 fill 3000 within one session:
79.9† → 50.3 (dispatch fusion + vec4 matvec + single readback, measured at
loadavg 4.7) → **35.0** (attention unrolls, loadavg 4.1). `submit` stays
flat 1.4-2.4 ms at every fill (the back-pressure that ballooned it to
13 ms behind the slow attention is gone).

Full-frame LM critical path (`framebench`, mimi decode on its own thread,
min-of-medians; every window at loadavg < 4.7):

| frame (ms) | fill 256 | fill 1024 | fill 2999 | raw best/worst @2999 |
|---|---|---|---|---|
| **f16** | **57.3** | **59.1** | **58.5** | 57.3 / 61.7 |
| **q8** | — | — | **46.5** | 45.7 / 50.3 |

Decomposition @2999 (f16): temporal 33.7 (submit 1.7) + depformer 24.6 +
other 0.0; mimi worker-side 8.8 (off the critical path, ~9× under budget).
The frame is now fill-FLAT — the only fill-dependent component is the
temporal step's ~+3 ms.

**Verdict: the f16-exact mind fits conversation depth. 58.5 ms at fill
2999 against the 80 ms budget (worst frame in the clean window 61.7) —
21.5 ms of margin, with the monologue 113/113 by construction.** q8 (the
former fallback recommendation) sits at 46.5 ms — 33.5 ms of margin — and
is no longer needed for the budget on a quiet machine; it remains the
co-load cushion.

**Co-load margin (q8/f16 under contention):** the pre-lever q8 co-load
projection was p50 ~70 / p95 crossing 80. The clean q8 frame is now 46.5
(f16 58.5), and the contaminated windows of this same session (loadavg
37-63, two sibling lanes live) show WHERE co-load lands: the GPU column
held 42-67 ms while the CPU depformer starved (68-152 ms vs its 24.6
clean). So the co-load margin problem has moved off the GPU: the next
co-load lever is the depformer's CPU side (--depth-f16 halves its weight
traffic; QoS/core-pinning against sibling load), not the temporal step.

### Next levers (none required for the budget)

1. Depformer co-load hardening (above) — the only component that folds
   under machine load.
2. RT_FB_SKIP spot-measurement mode landed for cheap deep-window checks.
3. The rope-into-qkv epilogue fusion (−32 dispatches) and a
   CUBECL_WGPU_MAX_TASKS sweep remain unexplored — at 1.7 ms flat submit
   there is little left to win there.

## PersonaPlex-7B — signed source bundles (2026-08-21)

The old runtime path treated a broad union of model facts plus
filename-discovered sibling piles as authority. It has been replaced by one
signed model-bundle token per COMMIT:

`τ = (model_root, metadata::archive, H)`

`H` is a canonical, self-contained archive of the exact model facts. Loading
freezes an exact bundle ticket, validates every one-row token and its canonical
archive independently, selects the sole Source/native PersonaPlex root, and
keeps `(collection descriptor, root, H, τ, cover)` bound to the weight
loader. A snapshot cannot be relabelled under another collection policy, and a
broad graph union
cannot fill missing facts in a candidate bundle.

Legacy qpile derivation, filename/sibling discovery, and model-name runtime
identity are removed. `RealtimePipeline::load_auto` currently recomputes the
runtime transforms from the verified bundle. Any future cached representation
must preserve this authority/loader binding rather than introducing a second,
caller-supplied identity.

## Qwen3-TTS — raw is the ONLY talker lane (2026-07-12, `talker-raw-only`)

Two decisions in one (2026-07-12). (1) The ear-gate passed: the raw
zero-copy talker lane won the A/B — "better tempo" — so raw became the
default. (2) "Just clean up the old fused path — the raw one seems
superior in every way": house style removes unused paths outright, so the
fused talker lane is GONE, not hidden — no `MARY_SPEAK_FUSED`, no env flag
to resurrect it.

**The rationale trail.** The raw lane was already at measured perf parity
(~28 ms/frame talker GPU total — the interleaved A/B in "the zero-copy RAW
talker lane" above), loads with ~10 GB less transient footprint (3.6 vs
20.6 GB peak), is run-to-run deterministic, and its load path is the
project-native zero-copy one. The fused lane's only remaining role was A/B
control — and its machinery is this model's most reliable source of bugs:
three distinct burn-fusion 0.21 codegen failures on record (miscompile
over externally-registered buffers; two-backend-JIT op-stream corruption;
FusedReduceLaunch strides OOB on synthetic entries). The ear-verdict
removed the last reason to keep it.

**Removed (talker-only):**
- `MARY_SPEAK_RAW` — the raw lane needs no opt-in. `MARY_SPEAK_F32` now
  only selects precision within the raw lane (raw-f32 vs raw-f16 talker).
  `MARY_SPEAK_MATERIALIZE` survives — the load-path A/B within the lane
  (folded-alias / leaf-alias vs fully materialized).
- The two fused dispatch arms in `speak::synthesize_stream` (the
  `BFused`/`BFusedHalf` talker instantiations).
- The `fused` lane of `qwen3tts_raw_gate` (weights dump + the
  degrade-to-weights-only forward and its cross-backend cos gate). The
  fused lane's final green comparison — gate 1 weights 173/173
  bit-identical across fused/raw-fold/folded-alias — is on record at
  fa03334, the pre-removal HEAD (the run itself: the section above, merged
  at 5d93a95). The gate now proves raw-fold ≡ folded-alias (weights sha,
  bit-exact hidden, identical argmax) across separate processes; the E2E
  determinism gate rides `speak_check` (seeded render twice,
  byte-identical wav). *(2026-08-18: the E2E "determinism gate" is
  downgraded to a tripwire — a differing wav is something to look at, not a
  failure. See wiki:f5dcc88988bb28e472e50fa030332adb. The load-path gate above
  is unaffected.)*

**Verified shared — kept:**
- `BFused`/`BFusedHalf` backend types: the codec decode loop
  (`CodecDecoder::<BFused>` — launch-bound, fusion still earns its keep
  there, and it never aliases), voxtral's STT lanes (`voxtral_listen`
  fused/fused-f16), the probe bins (`qwen3tts_say`/`_stream`/`_probe`/
  `megakernel_probe`/`moshi_realtime_probe`), and `qwen3tts_persist
  --fold-derive`, which loads the talker on `BFusedHalf` to READ BACK the
  fold-transformed tensors the sibling pile stores — the raw lane's own
  producer.
- `AliasedPile::gpu_tensor`'s fused-upload arms (`want_f16`/`want_f32` in
  `weight_loader.rs`) — same consumers as above.
- The ordinary-loader leaf-alias path in `speak::load_talker` — the
  surviving fallback for piles without the folded sibling (`--fold-derive`
  not yet run): leaf-alias zero-copy + fold math at load, byte-identical
  output (gate 2).

## Code predictor on the GPU (2026-08-21, `predictor_gpu`)

The last host-side stage of the voice lane. Per 80 ms frame the sub-talker
runs 16 strictly sequential positions through a 5x1024 Qwen3 decoder — 1.26 G
weight reads, ~5 GB at f32 — and on the CPU that is the frame's largest
single term. `qwen3tts::predictor_gpu` is the cubecl port; it is the sibling
of `megakernel` (the talker's fused decode step), not a new design.

**Shape.** Weights stay row-major `[out, in]` and ride `nn::q4`'s single-token
matvec thread shape (32 lanes per output row, 8 rows per 256-thread cube,
vec4 loads both sides). Three foldings happen once at load so no dispatch
exists purely to normalize: `input_layernorm` into the qkv columns,
`post_attention_layernorm` into gate‖up, `model.norm` into all 15 `lm_head`s,
leaving each kernel a weightless rms. gate/up rows interleave so SwiGLU is the
matvec's epilogue; `1/sqrt(d)` folds into the RoPE chain.

Unlike the talker's megakernel, the q/k-norm + RoPE chain is its **own**
dispatch rather than folded into a widened qkv matmul. That fold duplicates
the qk columns (`wide_out` 7168 vs a plain qkv's 4096) — +20% on the layer's
weight traffic to save one ~2 us dispatch out of 34 per position. At f32
aliased talker shapes the trade pays; at these it does not.

**The chain stays on the device.** Each of the 15 steps samples a token that
indexes the next step's input embedding; reading it back would cost 15 syncs
per frame. Sampling runs on the GPU (`gumbel_argmax_kernel`) into a device
slot `embed_gather_kernel` consumes, and a frame syncs **once**, on one
`[2048+15]` buffer carrying the embedding sum and the codes. Sampling had to
be handled, not avoided — production runs `subtalker_do_sample: true`. The
gumbel noise is drawn on the host from the caller's rng in the CPU path's
exact order and count (15 x 2048 draws), so the shared rng advances
identically whichever engine runs and the talker's own sampling is untouched.

**Cost model.** Two formats at 541 dispatches/frame give a two-point fit:
~1.2 ms of dispatch overhead (~2.3 us each) and ~152 GB/s effective
bandwidth. So this stage is **bandwidth**-bound, not dispatch-bound — the
opposite of the depth port, and the reason narrower weights buy time here.
152 GB/s is well under the ~400 the same kernel skeleton reaches on Moshi's
4096-wide rows; at 1024-3072 each lane sweeps only 4-12 iterations and the
memory latency is not hidden. That headroom is unclaimed, and is the next
lever if this stage ever matters again.

**Numbers** (`qwen3tts_pred_bench 60 --gpu`, all arms interleaved round by
round in ONE process — the only honest form on a machine this loaded; ms per
predictor frame):

| arm | M4 Max (Metal, load ~18) | GB10 Spark (CUDA, load 1.2) |
|---|---|---|
| cpu down=pooled | 26.85 (p10 25.86, p90 30.57) | 320.14 |
| gpu f16, 32 lanes | **17.83** (p10 15.77, p90 19.87) | **14.08** |
| gpu q8, 32 lanes | 10.06 (p10 9.14, p90 10.65) | 9.31 |
| gpu q8, 16 lanes | 9.11 | 9.67 |
| gpu q8, 8 lanes | 11.64 | 11.11 |

Do NOT read the two columns against each other — the Mac carried other
agents' builds all session and the Spark was idle. What they jointly show is
that the same kernels run on both backends and rank the lane width the same
way (32 > 16 > 8 at f16 shapes), which is why 32 is the default.

The Spark row is worth its own sentence: the CPU predictor costs 320 ms/frame
there (no Accelerate on aarch64 Linux), i.e. 4x slower than audio-rate, so
corpus generation on a Spark was not viable at all before this port and is
~23x faster now.

**In situ** — the number that decides whether it rebuffers. Same quiet window
(load ~4), same 687-char two-pass fixture, seed 7, `MARY_PRED_CPU=1` for the
control, per frame against an 80 ms budget:

| | talker submit | sync | logits | **predictor** | total | wall/audio |
|---|---|---|---|---|---|---|
| cpu | 15.2 | 7.7 | 0.2 | **25.9** | 49.2 | 0.62x |
| gpu f16 | 17.3 | 7.3 | 0.2 | **10.9** | 35.7 | 0.45x |

464 and 450 frames respectively. The predictor is 2.4x faster in place —
better than the 1.5x the synthetic bench shows, because in the real loop the
predictor's dispatches are encoded while the talker's previous GPU work is
still draining, and only the residue lands on the critical path. The frame
now spends 36 ms of its 80, and the talker is once again the largest term.

**Parity.** No bit-exactness gate; judged by per-codebook token agreement
against the CPU f32 oracle, dual-run on the real generation path
(`MARY_PRED_GATE=1`, which runs both engines on every frame's real inputs
with a cloned rng, so a disagreement is numerics and not luck):

- **f16: 1275/1275 = 100.00%**, max |delta embed_sum| exactly 0, over an
  85-frame utterance — and the rendered WAVs are **byte-identical** to the CPU
  path's at the same seed. That is the control the format choice hangs on: it
  says the kernels are right, so any disagreement elsewhere is quantization.
- q8: 1148/1350 = 85.04%.

Across three fixtures f16 agrees 7262/7275 = 99.82%; the residue is near-ties
in a 2048-way gumbel argmax, and because the stage is autoregressive a single
flip re-rolls the rest of the utterance (the 464-vs-450-frame difference in
the table above is exactly that, not a defect). On the shorter fixtures where
no token flips, the rendered WAV is byte-identical.

f16 is therefore the default and the ~8 ms q8 would save is deliberately left
on the table: the talker ahead of this stage costs ~28 ms/frame against an
80 ms budget, so the faster format buys throughput nobody is short of while
giving up the exactness that makes the port trustworthy. `MARY_PRED_W=q8`
opts in where throughput is worth more than agreement.

The negative result is worth carrying: "q8 is free on an M4 because the path
is dispatch-bound" (true on the depth port) does **not** transfer here. This
stage is bandwidth-bound, so q8 really is ~1.8x faster — it just pays for it
in fidelity, because the acoustic codebooks' logits sit close enough together
that 8-bit weights reorder the top of a 2048-way argmax one time in seven.

**Not a CPU stage after all: the ECAPA speaker encoder.** The `speak.rs`
header's "the CPU code predictor and the ECAPA speaker encoder read the exact
f32 leaves" reads like two host stages. It is a statement about which
*weights* they load. `qwen3tts::speaker` is ordinary Burn (`Tensor<B, 3>`,
`conv1d`) and has always run on the GPU, once per voice rather than per frame.
There was no CPU speaker encoder to port. Header corrected.
