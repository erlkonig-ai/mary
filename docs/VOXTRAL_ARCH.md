# Voxtral-Mini-4B-Realtime-2602 → Burn port notes (streaming STT)

Port of `mistralai/Voxtral-Mini-4B-Realtime-2602` (Apache-2.0, arXiv
2602.11298) as a streaming speech-to-text port. Reference: transformers 5.13.0
(`models/voxtral_realtime`, native since 5.2) + mistral_common 1.11.5.
Everything below is **measured from config.json / params.json + the reference
code**, not from the paper. Oracle venv: a local
transformers virtualenv (see `golden/voxtral_capture.py`).

Delay-conditioned streaming STT: audio in at 16 kHz, one autoregressive step
per **80 ms** frame, text tokens delayed behind the audio by a configurable
number of frames (80 ms–1.2 s in 80 ms steps, plus 2.4 s standalone;
480 ms = 6 delay tokens is the recommended sweet spot, German WER 6.19%).

## Components (`model.safetensors`, bf16 — HF-format names; the
## `consolidated.safetensors` duplicate in mistral naming is NOT downloaded)

```
wav 16 kHz → log-mel 128 × 100 Hz → conv stem (÷2) → causal encoder (50 Hz)
  → stack 4 frames (5120) → projector → +tok-embed → Mistral decoder (12.5 Hz)
  → tied lm_head → tokens (1 tok = 80 ms)
```

### 1. Mel front-end (`VoxtralRealtimeFeatureExtractor`)
| param | value |
|---|---|
| sampling_rate | 16000 |
| n_fft = win_length | 400 (25 ms), hann |
| hop | 160 (10 ms) |
| mels | 128, slaney-norm + slaney-scale, fmin 0, fmax **8000** |
| stft | torch.stft center=True (reflect pad 200) for the batch/first chunk; center=False for later streaming chunks |
| frames | `stft[..., :-1]` — the **last time frame is dropped** ⇒ n_frames = samples/hop exactly (padded audio is a multiple of 1280 samples) |
| log | `log10(clamp(mel, 1e-10))`, then `max(log_spec, global_log_mel_max − 8)` with **global_log_mel_max = 1.5** (fixed, NOT per-clip max — streaming needs a global reference), then `(x + 4) / 4` |

### 2. Conv stem (`model.audio_tower.embedder`)
- `conv1`: Conv1d 128→1280, k3 s1, **causal** (left-pad 2, zeros) + bias → GELU
- `conv2`: Conv1d 1280→1280, k3 s2, **causal** (left-pad 1) + bias → GELU
- output transposed to (B, T_enc, 1280); T_enc = T_mel / 2 (50 Hz)
- streaming: per-conv left-pad ring cache (`VoxtralRealtimeConv1dPaddingCache`),
  left_pad = (k−1)·d + 1 − stride.

### 3. Audio encoder (`model.audio_tower.layers.{0..31}`) — ~970M
Whisper-large *shape*, but causal + RoPE + RMSNorm + SwiGLU:

| dim | value |
|---|---|
| layers | 32 |
| hidden | 1280 |
| heads | 32 MHA, head_dim 64 ⇒ **attention dim 2048 ≠ hidden 1280** |
| q_proj / v_proj / o_proj | bias **yes** |
| k_proj | bias **no** (Whisper tradition) |
| mlp | MistralMLP SwiGLU 1280→5120, gate/up **no bias**, down_proj **bias yes** |
| norms | RMSNorm eps 1e-5, pre-norm (self_attn_layer_norm / final_layer_norm) + final `norm` |
| rope | theta 1e6, HF default (NeoX split-half) on head_dim 64 |
| attention | causal, **sliding window 750** (≈15 s of context at 50 Hz), scale 1/√64 |

Layer = `x + attn(norm1(x))`, then `x + mlp(norm2(x))` — standard pre-norm.

### 4. Projector (`model.multi_modal_projector`)
- reshape (B, T_enc, 1280) → (B, T_enc/4, **5120**)  (downsample_factor 4 ⇒ 12.5 Hz)
- `linear_1` 5120→3072 no bias → **GELU** → `linear_2` 3072→3072 no bias

### 5. Decoder (`model.language_model.layers.{0..25}`) — Ministral-3-3B, ~3.4B
| dim | value |
|---|---|
| layers | 26 |
| hidden | 3072 |
| heads / kv / head_dim | 32 / 8 / 128 (GQA 4:1), no q/k-norm |
| mlp | SwiGLU 3072→9216, no biases anywhere |
| norms | RMSNorm eps 1e-5 |
| rope | theta 1e6, HF default (NeoX split-half) |
| attention | causal, sliding window 8192, scale 1/√128 |
| vocab | 131072, **tied** embeddings (lm_head = embed_tokens) |

### 6. Delay conditioning (the whole point — must be numerically exact)
- `model.time_embedding`: sinusoidal, dim 3072, theta **10000**:
  `inv_freq = exp(−ln 10000 · arange(1536)/1536)`; `t_cond = cat(cos(t·inv_freq), sin(t·inv_freq))`
  where `t = float(num_delay_tokens)` (e.g. 6.0 for 480 ms). Shape [3072].
- per decoder layer `ada_rms_norm`: `linear1` 3072→**32** no bias → GELU →
  `linear2` 32→3072 no bias.
- injection point, inside every decoder layer:
  `h = norm_post_attn(h); h = h * (1 + ada(t_cond)); h = h + mlp(...)... `
  i.e. AFTER post_attention_layernorm, BEFORE the MLP. Attention path is
  untouched. t_cond is constant per session ⇒ **precompute the 26 scale
  vectors (1 + ada_i(t_cond)) once and fold into the MLP input**.

### 7. Multimodal fusion + schedule
- `inputs_embeds = tok_embed(input_ids) + audio_embeds` — **position-aligned
  1:1 sum at every position**, prompt and generated alike.
- Tokenizer: tekken (`tekken.json`); BOS 1, EOS 2, `[STREAMING_PAD]` 32;
  ids 0..999 special, base vocab piece i ↔ id i+1000 (base64 bytes). ASR only
  needs **decode** (id → bytes) — no BPE encoder required in Rust.
- Prompt (offline streaming mode, mistral_common `encode_transcription`):
  `[BOS] + [STREAMING_PAD] × (n_left_pad + n_delay)` where n_left_pad = 32,
  n_delay = delay_ms/80. (480 ms ⇒ 39 tokens.)
- Audio padding (mistral_common): left-pad `32 × 1280` zero samples (2.56 s),
  right-pad to a multiple of 1280 samples then `(n_delay + 1 + 10) × 1280`
  more zeros (OFFLINE_STREAMING_BUFFER_TOKENS = 10).
- raw_audio_length_per_tok = 1280 samples; audio_length_per_tok = 8 mel frames.
- Decode loop: max_length = num_audio_tokens = mel_frames/8. Each step feeds
  `tok_embed(prev) + audio_embeds[pos]`; encoder side advances 4 conv-stem
  positions (= 8 mel frames) per step through the cached causal encoder.
  Greedy (temperature 0.0 recommended). Stop at EOS or audio exhaustion.
- online mode = same, without right padding (and audio arriving chunkwise;
  chunk boundary state = encoder KV cache + conv pad cache + mel remainder).

## Port strategy / scope decisions
- Persist HF `model.safetensors` names into `models/voxtral_mini.pile`
  (bf16→f32 exact, existing `persist_safetensors_to_pile`; no new schema ids
  needed — the mary weight schema is model-agnostic).
- Oracle goldens: `golden/voxtral_capture.py` → `golden/voxtral/` (npy,
  regenerable; clips committed). Deep taps on `en_short` @480 ms; greedy
  token streams for 4 clips × {6,12,30} delay tokens.
- Decoder RoPE follows the HF checkpoint layout (NeoX split-half) — the vLLM/
  consolidated interleaved form is a weight permutation we never touch.
- Burn backend: house `nn::backend` (Metal f32; f16 later, measured).
- Batch = 1, no padding mask anywhere (single stream).
- en_long (34.6 s ⇒ T_enc ≈ 1930 > 750) exercises the encoder sliding window
  in goldens deliberately.

## Results (2026-07-10 overnight — first port)

All parity gates green (`voxtral_probe --long`, f32 Metal vs CPU-f32 oracle):
prompt/pad construction exact; mel / conv stem / encoder / projector /
prefill-embeds / decoder-hidden / logits all **cos = 1.00000000**; delay
conditioning (t_cond + 26 ada scales at d ∈ {6,12,30}) max|Δ| < 2e-6;
greedy streams **token-identical, transcripts byte-equal** on all six:

| stream | tokens | note |
|---|---|---|
| en_short d6 / d12 / d30 | 107 / 113 / 131 | delay knob exercised end-to-end |
| de_short d6 | 119 | German |
| denglish d6 | 152 | mixed DE/EN |
| en_long d6 | **482** | 38 s — encoder sliding window (T_enc≈1930 > 750) active |

Incremental encoder (4-position KV steps) vs batch: **bit-identical**
(max|Δ| = 0) — the streaming path is exact, not approximate.

Weights: `models/voxtral_mini.pile` (16.5 GiB, 711 tensors f32,
persist gate: bit-identical to bf16→f32 of the checkpoint, 256-aligned).

**Latency (UNOPTIMIZED parity-first layout, f32, raw Metal backend, shared
machine): 215–345 ms/frame mean vs the 80 ms budget.** Not realtime yet —
expected: this layout deliberately mirrors the oracle op-for-op (plain
q/k/v projections, explicit norms, unfused ops, per-op submission cost).
The qwen3tts playbook applies directly:

1. `BFused` alias (already wired for `voxtral`) — fusion alone bought the
   talker 2×.
2. The fold pass: norm weights into matmul rows, wide fused qkv with
   pre-rotated weight rows (biases ride along as `[b‖R(b)‖b_v]`), gate‖up
   fusion, GQA group-fold single-token decode path. All gated exact.
3. f16 decoder weights (`BFusedHalf` + f32 variance chains) — halves the
   3.4B weight traffic; 13.6 GB→6.8 GB per frame ⇒ ~23 ms at 300 GB/s.
4. One GPU sync per frame (argmax currently syncs; logits gemv could move
   host-side like the qwen3tts codec-head if submission cost dominates).
5. KV caches: preallocate + write-in-place instead of per-step `cat`
   (the megakernel lane proved the win).

Budget sketch: decoder f16 ≈ 23 ms + encoder f16 ≈ 4×(1.9 GB)/300 GB/s
≈ 7 ms + sync floor ≈ 2 ms → ~35 ms/frame ⇒ comfortable realtime, before
any megakernel work.

## Results (2026-07-11 — the perf pass: REALTIME reached)

The playbook above, executed on the `voxtral-port` branch. Five lanes,
selectable per process (`voxtral_listen --lane raw|fused|fold|half|rawhalf`;
the fifth, `rawhalf`, landed in the zero-copy pass below — five-lane table
in the 2026-07-12 section). The four of this pass:

| lane | layout | backend | role |
|---|---|---|---|
| raw | op-for-op parity-first | raw Metal f32 | trust anchor; the full oracle probe suite runs here, unchanged |
| fused | op-for-op | `BFused` (fusion f32) | lever 1 measured alone |
| fold | folded (`fast.rs`) | `BFused` | lever 2; gated TOKEN-identical to the oracle on all four greedy streams |
| half | folded | `BFusedHalf` (fusion f16) | levers 2+3; the production lane |

The fold (`src/models/voxtral/fast.rs`): wide fused qkv with rotate_half
pre-applied to the weight ROWS (encoder biases ride as `[b_q‖0 | R(b_q)‖0 |
b_v]` — RoPE-after-bias preserved because rope is linear), all preceding
RMSNorm weights folded into consuming matmul rows (attn→qkv, enc-mlp→gate‖up,
enc-final→projector rows tiled ×4, dec-final→tied head), the decoder's
post-attention norm weight folded into the per-session ada scales, 1/√d in
the q rows, gate‖up fused, GQA group-fold on single-token steps, masks built
once per stack forward (raw builds per layer), and sliding-window-trimmed KV
caches (`keep = window−1`: bounded memory on long sessions + the l==1 step
provably needs no mask). f16 conversion happens at LOAD (`from_floats` on the
f16 backend) — the f32 pile is untouched.

**Measured** (de_short@480 ms, unpaced `--fast`, compute ms/frame = encoder
submit + decoder step incl. the per-frame argmax sync; 80 frames; M4 Max):

| lane | quiet machine p50 / p95 | under a build storm (load 140–200) p50 / p95 |
|---|---|---|
| raw | 829 / 993 | 1373 / 1770 |
| fused | 791 / 823 | 2323 / 3240 |
| fold | 516 / 563 ¹ | 1367 / 1767 |
| **half** | **55 / 57** | 432 / 579 |

¹ fold's "quiet" window still had a Time Machine backup running; treat as an
upper bound. Fusion on the UNfolded layout (lever 1 alone) is neutral-to-
negative — matches the earlier one-shot A/B; the wins are the fold + f16.

**Headline (half lane, quiet, best-of-3): de_short p50 54.7–55.5, p95
57.0–57.1 ms/frame; en_long (443 frames, 38 s, encoder sliding-window phase
active) p50 62, p95 65 ms/frame — sustained REALTIME with ~25% headroom
against the 80 ms budget.** Paced (real-time-fed) mode is compute-equivalent
to unpaced — a same-window A/B measured p50 157 (paced) vs 161 (unpaced)
under contention, i.e. no sleep-decay/downclock penalty — so the quiet
compute numbers carry over to live pacing; under contention the pipeline
degrades gracefully (latency inflates with compute, queue stays bounded,
transcripts stay word-perfect in every run all night). Prefill ~1.0 s.
Peak RSS with the f16 lanes was ~35 GB (the f32 host keymap dominated during
load; freed after) and load time ~25–60 s (contention-dependent) — fixed the
next night by the derived f16 sibling pile (next section).

**Gates:** raw probe suite ALL GREEN (unchanged); `--lane fold` TOKEN-
identical + transcript-byte-equal to the oracle on en_short d6/d12/d30 +
de_short d6; `--lane half` ALSO token-identical on all four (f32 variance
chains keep greedy stable at f16 — stronger than the required word-exact
gate); streaming `voxtral_listen` word-perfect on de_short@480 (13/13 LCS)
in EVERY lane, transcripts byte-identical across lanes, and en_long 66/66.

**Upstream bug (documented dodge):** burn-cubecl-fusion 0.21 panics in
`ReduceOptimization::execute` → `GlobalArgsLaunch::strides`
(`ir.rs:500`, "index out of bounds: len 1 index 1", then cascading
`ordering.rs:49` "Ordering is bigger than operations") when the f16 fusion
backend runs the BATCH encoder pass (l≈500+ in one forward). f32 passes the
identical graph; every streaming-sized f16 shape (l=4 enc / l=39 prefill /
l=1 decode, any cache length) is fine. The half-lane probe therefore gates
via the incremental encoder (= production schedule); repro = flip `!exact`
to `false` in `voxtral_probe::fast_lane_gates`. Also: torch-style reflect
padding for the batch mel moved to the host sample buffer (identical values,
one less GPU op family) while bisecting this.

Not taken (not needed for the budget): in-place KV `slice_assign` (the
window-trimmed `cat` costs ~0.3 ms/frame), host-side logits gemv, and the
megakernel rewrite — dec is ~47 ms of the 55 and runs ~155 GB/s effective on
gemv-shaped matmuls, so a qwen3tts-style engine (~300 GB/s) could roughly
halve the frame again if the deployment ever needs the headroom (e.g. two
heavy GPU pipelines at once — the storm column above is what contention does).

### File-based streaming gate (voxtral_listen, de_short @480 ms, paced)

Online path (StreamingTranscriber, real-time-paced 80 ms chunks + offline-right-pad
tail): transcript **word-perfect** (13/13 LCS vs the offline oracle,
byte-equal minus leading space). Compute latency at the unoptimized f32
layout: p50 822 ms/token (mean 895, p95 1576) — the queue grows ~220 ms per
token when compute is ~300 ms against the 80 ms arrival rate; realtime
needs the perf pass. A one-shot `BFused` A/B measured p50 1006 ms — worse,
but single uncontrolled runs on this shared machine are meaningless
(qwen3tts measurement discipline: interleave against a fixed control);
fusion stays wired for the proper perf pass.

## Results (2026-07-11 — the derived f16 sibling pile: load fixed)

The load-cost follow-up, executed with sign-off (2026-07-12 authorized
the f16 weights persist; the design landed as a SEPARATE sibling pile rather
than an append — separate-now/merge-later is the cheap direction, piles union
by `cat` + consolidate, and the 8.25 GiB f16 pile can deploy without the
16.5 GiB f32).

**Persist:** `voxtral_persist --f16-derive models/voxtral_mini.pile` reads
every f32 leaf back from the pile (one tensor at a time), casts host-side
(`f16::from_f32` — the same rounding the materializing loader applies on the
f16 backend) and persists 711 tensors / 4.43 G elements under the `ears_f16`
entity in `models/voxtral_mini_f16.pile` (8,860,192,256 B = 8.25 GiB; 16 s
derive + full re-read verify). Gates: every f16 leaf bit-identical to
`f16(f32-leaf)`, full 711-tensor coverage, all 256-aligned, and the source
pile is READ-ONLY — byte-length gated in-bin, sha256 over all 16.5 GiB
verified unchanged externally.

**Load:** `load_loader_with_f16_sibling` auto-discovers `<stem>_f16.pile`
next to the given pile (CLIs unchanged): `BFusedHalf` tensors upload the f16
leaves at native width — no whole-model f32 keymap, no cast loop — `BFused`
uploads the exact f32 leaves, everything else materializes lazily one tensor
at a time. Sibling absent → f16 tensors materialize+cast from the f32 leaves
(bit-identical, gated below); `MARY_SPEAK_MATERIALIZE=1` still forces the old
fully-materialized load (the A/B switch).

**Measured** (voxtral_listen de_short@480 half, same machine, back-to-back):
load **14.4 s → 5.5 s**, peak RSS **35.45 GB → 13.0 GB** (the f32 host
keymap is gone; what remains is the mmap'd f16 pile + the fold/GPU working
set). en_long run: 5.5 s / 12.9 GB.

**Identity gates:** listen stdout BYTE-IDENTICAL old-path↔new-path on
de_short@480 (13/13 words) and en_long (66/66); the sibling-absent fallback
also byte-identical. `voxtral_probe --lane half --long` A/B'd between the
f16-sibling load and `MARY_SPEAK_MATERIALIZE=1`: token counts identical on
all six streams. vs the f32 oracle: en_short d6/d12/d30, de_short, and
en_long all TOKEN-identical (482/482 on en_long) — and denglish diverges at
token 96/152 (20/21 words, "Latencibild" → "Latent-CBD") IDENTICALLY under
both load paths: an inherent f16-lane greedy drift (AR cascade on a made-up
word), first surfaced now because `--lane half --long` had never been run.
Not a load-path effect; the raw f32 lane stays 152/152.

## Results (2026-07-12 — the zero-copy raw-backend lane: `rawhalf`)

The raw-backend reopening condition from the deleted fused-alias attempt is
now exercised. `--lane rawhalf` (voxtral_listen + voxtral_probe) runs the
SAME folded graph (`fast.rs`, unchanged) as `half`, but on the RAW unfused
Metal f16 backend (`RealtimeTranscriber<BHalf>`), and loads it TRUE zero-copy: every
f16 leaf of `models/voxtral_mini_f16.pile` aliases its mmap'd pile pages
straight onto the GPU via the cubecl fork's `register_external_aliased`
seam — the gemma production recipe (`persist.rs`), no host staging at all.
The fused lanes keep their upload path: the fusion-import miscompile
documented at `weight_loader.rs::AliasedPile::gpu_tensor` (history:
`src/nn/alias.rs` / `qwen3tts_alias_test.rs`, removed at bf92171) named "a
burn fix or a raw-backend talker" as the reopening conditions — the raw
half ear is exactly the second one, and no fusion import step exists on
this backend to miscompile.

| lane | layout | backend | load | role |
|---|---|---|---|---|
| raw | op-for-op | raw Metal f32 | materialize | trust anchor (full oracle probe suite) |
| fused | op-for-op | `BFused` | upload f32 | lever 1 measured alone |
| fold | folded | `BFused` | upload f32 | lever 2, token-identical gate |
| half | folded | `BFusedHalf` | upload f16 (one-copy) | production realtime lane |
| **rawhalf** | folded | **`BHalf` (raw f16)** | **zero-copy alias** | zero-copy lane; fusion-independence proof |

**Fold analysis (why the load is hybrid, not 100%-resident-zero-copy):**
the folds TRANSFORM every big matrix at load — `wide_t` = cat(q·1/√d, k) +
rotate_half on weight rows + cat v + transpose + norm-row-fold; gate‖up =
cat + transpose (+ norm fold); even plain `Linear::load` transposes; the
tied head is embed-transpose × final-norm. None of that is expressible as
views over the stored leaves, and the qk pre-rotation DUPLICATES the qk
rows, so the folded weights (~9.3 GiB f16) are larger than the stored pile
(8.25 GiB). What zero-copy buys here: (1) the embed table
(`embed_tokens.weight`, 768 MiB — the one big tensor consumed as-stored by
`select`) stays a file-backed mmap alias for the process life; (2) the
other ~7.5 GiB of leaves are aliased as fold SOURCES — the transform
kernels read the pile's own pages (zero host staging, no upload pass, no
host cast loop) and write the folded results into normal GPU buffers.
A FOLDED sibling pile (`wide_t`/`gate_up_t`/`head_t`/… stored post-fold,
~10 GiB) would make the whole model file-backed with near-instant load —
that is option (a), it needs a new derived pile and maintainer sign-off; the
hybrid is what ships without one. The derive MECHANISM now exists on the
qwen3tts side (`qwen3tts_persist --fold-derive <src-pile>` →
`<stem>_folded_f16.pile`, loaded by `persist::load_qwen3tts_talker_folded`
with a readback identity gate); a voxtral `_folded_f16` derive is the
mechanical follow-up — the derive itself still needs maintainer sign-off.

**Incident (fixed in the cubecl fork, f299aed): in-place elementwise into
read-only mmap.** First aliased run was word-perfect on en_short d6 but
degraded everywhere else (en_long emitted nothing). Cause: burn's
elementwise kernels reuse an "owned" input handle as the output buffer
(`can_mut()` = a handle-strong-count heuristic), so load-time fold ops like
`q_proj.mul_scalar(1/√d)` wrote their result IN PLACE into the aliased
read-only pile pages — the pager drops such writes, the op returns its
input, and q came back unscaled (deterministically wrong weights, NOT pile
corruption: the pile mmap is `map_raw_read_only`, so the file was never at
risk — confirmed by the half lane re-run staying token-identical, i.e. the
sibling's bytes still produce the exact known-good streams). Caught by
the A/B that should have been boring: `MARY_SPEAK_MATERIALIZE=1 --lane
rawhalf` (bit-identical weight values, upload path) was token-identical to
the oracle while the aliased run wasn't. Fix at the source:
`register_external` now pins every externally-registered handle
`can_mut() == false` for the registration's life, so ALL aliased-buffer
consumers (gemma included) allocate their outputs.

**Gates (aliased load, after the fix):** en_short d6/d12/d30, de_short d6,
en_long d6 all TOKEN-identical to the f32 oracle (482/482 on en_long) —
the same set the half lane passes. The probe now prints an FNV-1a token
digest per stream for cross-run identity checks: the four short-stream
digests are bit-equal to the `MARY_SPEAK_MATERIALIZE=1` control (same
BHalf graph, upload load — the A/B that caught the incident), and ALL SIX
digests (denglish and en_long included) are bit-equal to the half lane's —
raw and fused f16 produce identical greedy streams end-to-end, including
the drift: denglish diverges at token 96/152 ("Latencibild" →
"Latent-CBD", 20/21 words), the SAME inherent f16 greedy drift documented
for the half lane, now shown to be backend-independent. Streaming
`voxtral_listen`: de_short@480 13/13 words, en_long 66/66 — transcripts
byte-identical across the two lanes.

**Measured** (2026-07-12, voxtral_listen de_short@480 `--fast`; the machine
carried a CONTENDED load the whole session — 1-min loadavg 65–160 from a
concurrent build storm plus another GPU probe (`personaplex_rt_probe`), so
the ms/frame and load-time columns are upper bounds, not quiet numbers;
peak RSS is contention-robust):

| lane | load | peak RSS | ms/frame p50 / p95 | prefill (first frame) |
|---|---|---|---|---|
| half (one-copy upload) | 32.8 s ¹ | **12.82 GB** | 163.8 / 358.6 ¹ | 10.6 s ¹ |
| rawhalf (zero-copy alias) | 15.6 s ¹ | **9.09 GB** | 170.8 / 281.3 ¹ | 4.2 s ¹ |

¹ contended; the half lane's quiet baselines are load 5.5 s, p50 55 / p95
57 ms/frame (previous section). Across four contended rawhalf loads:
15.0–21.8 s; probe-suite peak RSS 9.08–9.10 GB vs half's 12.82 GB — the
**−3.7 GB** is the structural claim (no host staging, no upload copies;
the folded working set plus a file-backed mmap the pager can evict).
ms/frame p50 tracked the half lane within noise under identical
contention (the fold + f16 carry the speed; fusion is not load-bearing on
this graph), but a quiet-window A/B is still owed for the headline
number, first-frame page-in cost, and whether `rawhalf` should take over
as the ear's default lane.

## Continuation plan

1. ~~**Perf pass** to < 80 ms/frame sustained~~ — DONE 2026-07-11 (see
   Results above: half lane p50 55 / p95 57, en_long p50 62 / p95 65).
2. **`voxtral_listen --mic` live-mic test** (Denglish, live) — bin is
   prepared (cpal input, device-by-name, linear resample to 16 kHz; default
   lane is now `half`). File mode is the tested path.
3. Online-mode oracle capture (mistral_common ONLINE + vLLM semantics) if
   the no-right-pad tail behavior needs its own golden.
4. Wire into the runtime: `hear` seam next to `mary::say`/`speak` (mic →
   tokens → runtime events), VAD/endpointing policy, delay setting as a
   per-context knob (480 ms conversational, 2.4 s dictation).
5. ~~Pile: an f16 weights entity (the qwen3tts `talker_f16` pattern) to cut
   the ~25–60 s load to seconds~~ — DONE 2026-07-11 as the derived sibling
   pile `models/voxtral_mini_f16.pile` (authorized; see Results above:
   load 5.5 s, peak RSS 13 GB, everything token-identical).
6. Upstream: minimal repro + report for the burn 0.21 fused-reduce f16
   batch-shape panic (see Results).
7. The denglish f16-lane greedy drift (96/152 vs the oracle, word 20/21) —
   decide whether the half-lane gate set should adopt it as a known-divergent
   stream or whether the lane wants an fp32 logits head; it is NOT a
   load-path issue (identical under both loaders), and NOT a backend issue
   (the rawhalf lane reproduces the identical divergent stream, 2026-07-12).
8. The FOLDED sibling pile (option (a) of the zero-copy analysis, ~10 GiB:
   `wide_t`/`gate_up_t`/`head_t`/… persisted post-fold) — would make the
   whole folded model file-backed: near-instant load, page-in lazily on
   first frames, minimal anonymous RSS. Needs a new derive
   (`voxtral_persist --fold-derive`?) and maintainer sign-off for the pile write.
