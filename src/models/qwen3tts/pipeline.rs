//! End-to-end voice-clone pipeline: prompt assembly (ICL streaming-text mode,
//! batch 1), the talker+predictor generation loop with HF-matching sampling,
//! and codec decode with the reference-prefix cut. Mirrors
//! `Qwen3TTSForConditionalGeneration.generate` + `generate_voice_clone`.

use burn::prelude::*;
use rand::distributions::{Distribution, WeightedIndex};
use rand::rngs::StdRng;
use std::path::Path;

use super::codec::CodecDecoder;
use super::config::*;
use super::encoder::CodecEncoder;
use super::layers::KvCache;
use super::predictor::CodePredictor;
use super::speaker::{SpeakerEncoder, SpeakerMel};
use super::talker::Talker;
use super::tokenizer::TextTokenizer;
use crate::models::f5::wav;

/// One reference-voice prompt: the encoded reference codes plus the x-vector.
pub struct ClonePrompt<B: Backend> {
    /// Reference codec frames (T × 16).
    pub ref_code: Vec<[u32; NUM_CODE_GROUPS]>,
    /// Reference transcript token ids (`<|im_start|>assistant\n{ref}<|im_end|>\n`).
    pub ref_ids: Vec<u32>,
    /// ECAPA x-vector `[2048]`.
    pub spk_embedding: Tensor<B, 1>,
}

impl<B: Backend> ClonePrompt<B> {
    /// Build a clone prompt from an arbitrary 24 kHz mono reference clip and
    /// its transcript — fully in-process: codec-encoder codes (CPU) +
    /// ECAPA x-vector (GPU).
    pub fn from_reference(
        encoder: &CodecEncoder,
        spk_enc: &SpeakerEncoder<B>,
        tok: &TextTokenizer,
        ref_wav: &Path,
        ref_text: &str,
        device: &B::Device,
    ) -> Self {
        let (samples, sr) = wav::read_pcm16_mono(ref_wav);
        assert_eq!(sr, SAMPLE_RATE, "reference must be 24 kHz mono PCM16");
        let spk_embedding = spk_enc.forward(SpeakerMel::<B>::new(device).forward(&samples, device));
        Self {
            ref_code: encoder.encode(&samples),
            ref_ids: tok.encode(&format!("<|im_start|>assistant\n{ref_text}<|im_end|>\n")),
            spk_embedding,
        }
    }
}

pub struct SamplingParams {
    pub do_sample: bool,
    pub top_k: usize,
    pub temperature: f64,
    pub repetition_penalty: f32,
    pub subtalker_do_sample: bool,
    pub subtalker_temperature: f64,
    pub max_frames: usize,
}

impl Default for SamplingParams {
    /// generation_config.json defaults (top_p 1.0 ⇒ no-op, omitted; the
    /// sub-talker's top-k=50 is traded for full-vocab gumbel-max, see
    /// `predictor.rs`).
    fn default() -> Self {
        Self {
            do_sample: true,
            top_k: 50,
            temperature: 0.9,
            repetition_penalty: 1.05,
            subtalker_do_sample: true,
            subtalker_temperature: 0.9,
            max_frames: 2048,
        }
    }
}

/// Sample from logits with top-k + temperature (HF warper order:
/// temperature → top-k → multinomial). Greedy = argmax.
fn sample_logits(
    logits: &[f32],
    do_sample: bool,
    top_k: usize,
    temp: f64,
    rng: &mut StdRng,
) -> u32 {
    if !do_sample {
        let mut best = 0usize;
        for (i, &v) in logits.iter().enumerate() {
            if v > logits[best] {
                best = i;
            }
        }
        return best as u32;
    }
    let mut idx: Vec<usize> = (0..logits.len()).collect();
    idx.sort_unstable_by(|&a, &b| logits[b].partial_cmp(&logits[a]).unwrap());
    idx.truncate(top_k);
    let scaled: Vec<f64> = idx.iter().map(|&i| logits[i] as f64 / temp).collect();
    let m = scaled.iter().cloned().fold(f64::MIN, f64::max);
    let weights: Vec<f64> = scaled.iter().map(|&s| (s - m).exp()).collect();
    let dist = WeightedIndex::new(&weights).expect("weights");
    idx[dist.sample(rng)] as u32
}

/// Build the talker prefill embeddings + trailing text hiddens for a clone
/// request (streaming text mode, explicit language id, ICL + x-vector).
/// `text_ids` = `<|im_start|>assistant\n{text}<|im_end|>\n<|im_start|>assistant\n`.
pub fn build_prefill<B: Backend>(
    talker: &Talker<B>,
    predictor: &CodePredictor,
    prompt: &ClonePrompt<B>,
    text_ids: &[u32],
    language: Option<u32>,
    device: &B::Device,
) -> (Tensor<B, 3>, Tensor<B, 3>, Tensor<B, 3>) {
    // tts control embeds (already text-projected)
    let ctl = talker.embed_text(&[TTS_BOS, TTS_EOS, TTS_PAD], device);
    let tts_bos = ctl.clone().narrow(1, 0, 1);
    let tts_eos = ctl.clone().narrow(1, 1, 1);
    let tts_pad = ctl.narrow(1, 2, 1);

    // role prefix: <|im_start|>assistant\n
    let role = talker.embed_text(&text_ids[..3], device);

    // codec-side prefix: think tags (+ language), then x-vector, then pad+bos
    let prefill_codes: Vec<u32> = match language {
        Some(lang) => vec![CODEC_THINK, CODEC_THINK_BOS, lang, CODEC_THINK_EOS],
        None => vec![CODEC_NOTHINK, CODEC_THINK_BOS, CODEC_THINK_EOS],
    };
    let hidden = talker.hidden;
    let spk = prompt.spk_embedding.clone().reshape([1, 1, hidden]);
    let codec_side = Tensor::cat(
        vec![
            talker.embed_codec(&prefill_codes, device),
            spk,
            talker.embed_codec(&[CODEC_PAD, CODEC_BOS], device),
        ],
        1,
    );
    let n = codec_side.dims()[1];
    // text side: (n-2)×tts_pad + tts_bos, aligned with codec_side[.. n-1]
    let mut pads = vec![tts_pad.clone(); n - 2];
    pads.push(tts_bos);
    let part2 = Tensor::cat(pads, 1) + codec_side.clone().narrow(1, 0, n - 1);

    // ICL block: text = ref_text[3..-2] ++ text[3..-5] ++ tts_eos
    let mut icl_text_ids: Vec<u32> = prompt.ref_ids[3..prompt.ref_ids.len() - 2].to_vec();
    icl_text_ids.extend_from_slice(&text_ids[3..text_ids.len() - 5]);
    let icl_text = Tensor::cat(vec![talker.embed_text(&icl_text_ids, device), tts_eos], 1);

    // codec = [codec_bos] ++ per-frame Σ₁₆ codebook embeds of ref_code:
    // codebook 0 as one batched GPU lookup, codebooks 1..15 accumulated on
    // the CPU (the predictor holds those tables) and uploaded once.
    let t = prompt.ref_code.len();
    let code0s: Vec<u32> = prompt.ref_code.iter().map(|f| f[0]).collect();
    let mut rest = vec![0f32; t * hidden];
    for (i, frame) in prompt.ref_code.iter().enumerate() {
        predictor.accumulate_frame(frame, &mut rest[i * hidden..][..hidden]);
    }
    let rest = Tensor::<B, 1>::from_floats(rest.as_slice(), device).reshape([1, t, hidden]);
    let icl_codec = Tensor::cat(
        vec![
            talker.embed_codec(&[CODEC_BOS], device),
            talker.embed_codec(&code0s, device) + rest,
        ],
        1,
    );

    let t1 = icl_text.dims()[1];
    let t2 = icl_codec.dims()[1];
    let (icl, trailing) = if t1 > t2 {
        (
            icl_text.clone().narrow(1, 0, t2) + icl_codec,
            icl_text.narrow(1, t2, t1 - t2),
        )
    } else {
        let mut parts = vec![icl_text];
        parts.extend(vec![tts_pad.clone(); t2 - t1]);
        (Tensor::cat(parts, 1) + icl_codec, tts_pad.clone())
    };

    let prefill = Tensor::cat(vec![role, part2, icl], 1);
    (prefill, trailing, tts_pad)
}

/// HF-matching logits processing for the talker head: repetition penalty over
/// previously generated codebook-0 ids, min-new-tokens eos block, and the
/// suppress range `[2048, 3072) ∖ {eos}`.
fn process_talker_logits(logits: &mut [f32], generated: &[u32], step: usize, penalty: f32) {
    let mut seen = vec![false; logits.len()];
    for &g in generated {
        seen[g as usize] = true;
    }
    for (i, l) in logits.iter_mut().enumerate() {
        if seen[i] {
            *l = if *l > 0.0 { *l / penalty } else { *l * penalty };
        }
    }
    if step < 2 {
        logits[CODEC_EOS as usize] = f32::NEG_INFINITY;
    }
    for i in PRED_VOCAB..CODEC_VOCAB {
        if i as u32 != CODEC_EOS {
            logits[i] = f32::NEG_INFINITY;
        }
    }
}

/// Run the full generation loop. Returns the generated frames (T × 16),
/// codebook-0 first.
///
/// Division of labor per frame: the talker step runs on the GPU (28 big
/// matmuls); its last hidden state is read back **once** (the only sync);
/// codebook-0 logits (CPU gemv), sampling, and the entire code predictor run
/// host-side; the next input embedding is assembled on the CPU and uploaded
/// as one tensor.
pub fn generate<B: Backend>(
    talker: &Talker<B>,
    predictor: &CodePredictor,
    prefill: Tensor<B, 3>,
    trailing: Tensor<B, 3>,
    tts_pad: Tensor<B, 3>,
    params: &SamplingParams,
    rng: &mut StdRng,
    device: &B::Device,
) -> Vec<[u32; NUM_CODE_GROUPS]> {
    generate_streaming(
        talker,
        predictor,
        prefill,
        trailing,
        tts_pad,
        params,
        rng,
        device,
        |_| true,
    )
}

/// One talker step per frame, behind whichever engine runs it: the Burn op
/// loop ([`BurnStepper`]) or the fused decode engine
/// (`megakernel::EngineStepper`, the Linux/CUDA build). The loop hands the
/// stepper each frame's talker input as ONE host row — codec embedding sum +
/// codebook-0 row + text side, already added — and reads back one normed
/// hidden state per frame, which is the frame's one sync.
pub trait FrameStepper {
    /// The normed hidden state `[hidden]` of the last submitted position.
    fn hidden(&mut self) -> Vec<f32>;
    /// Submit the next position. `false` means the stepper cannot take another
    /// frame (a bounded cache is full) and the pass ends with the frames so far.
    fn submit(&mut self, x: &[f32]) -> bool;
}

/// The Burn op loop: upload the row, run the stack over the growing KV cache,
/// read the last hidden state back.
pub struct BurnStepper<'a, B: Backend> {
    talker: &'a Talker<B>,
    caches: Vec<KvCache<B>>,
    device: &'a B::Device,
    pending: Option<Tensor<B, 3>>,
}

impl<'a, B: Backend> BurnStepper<'a, B> {
    /// Runs the prefill; the first [`hidden`](FrameStepper::hidden) is its
    /// last position.
    pub fn new(talker: &'a Talker<B>, prefill: Tensor<B, 3>, device: &'a B::Device) -> Self {
        let mut caches = talker.new_caches();
        let pending = Some(talker.forward(prefill, &mut caches, device));
        Self {
            talker,
            caches,
            device,
            pending,
        }
    }
}

impl<B: Backend> FrameStepper for BurnStepper<'_, B> {
    fn hidden(&mut self) -> Vec<f32> {
        let h = self.pending.take().expect("a submitted position to read back");
        self.talker.last_hidden(h)
    }

    fn submit(&mut self, x: &[f32]) -> bool {
        let e = Tensor::<B, 1>::from_floats(x, self.device).reshape([1, 1, self.talker.hidden]);
        self.pending = Some(self.talker.forward(e, &mut self.caches, self.device));
        true
    }
}

/// The text side of each frame's input, read back to the host once per pass:
/// the trailing text hiddens (one row per early frame), then `tts_pad`.
pub struct TextSide {
    rows: Vec<f32>,
    pad: Vec<f32>,
    hidden: usize,
}

impl TextSide {
    pub fn read<B: Backend>(
        talker: &Talker<B>,
        trailing: &Tensor<B, 3>,
        tts_pad: &Tensor<B, 3>,
    ) -> Self {
        let host = |t: &Tensor<B, 3>| -> Vec<f32> {
            t.clone().into_data().convert::<f32>().to_vec::<f32>().unwrap()
        };
        Self {
            rows: host(trailing),
            pad: host(tts_pad),
            hidden: talker.hidden,
        }
    }

    /// The row added at frame `step`.
    pub fn row(&self, step: usize) -> &[f32] {
        let n = self.rows.len() / self.hidden;
        if step < n {
            &self.rows[step * self.hidden..][..self.hidden]
        } else {
            &self.pad
        }
    }
}

/// [`generate`] with a per-frame sink — the streaming path hands each frame
/// to the codec thread the moment it is sampled. Returning `false` from the
/// sink stops generation without retaining the rejected frame. Runs the Burn
/// loop; [`generate_streaming_with`] takes any [`FrameStepper`].
#[allow(clippy::too_many_arguments)]
pub fn generate_streaming<B: Backend>(
    talker: &Talker<B>,
    predictor: &CodePredictor,
    prefill: Tensor<B, 3>,
    trailing: Tensor<B, 3>,
    tts_pad: Tensor<B, 3>,
    params: &SamplingParams,
    rng: &mut StdRng,
    device: &B::Device,
    on_frame: impl FnMut(&[u32; NUM_CODE_GROUPS]) -> bool,
) -> Vec<[u32; NUM_CODE_GROUPS]> {
    let text = TextSide::read(talker, &trailing, &tts_pad);
    let mut stepper = BurnStepper::new(talker, prefill, device);
    generate_streaming_with(talker, predictor, &mut stepper, &text, params, rng, on_frame)
}

/// The generation loop over a [`FrameStepper`] that has already run the
/// prefill. Per frame: one read-back of the normed hidden state (the one
/// sync), codebook-0 logits + sampling on the host, the code predictor, then
/// the next input row (codec embedding sum + text side) submitted.
#[allow(clippy::too_many_arguments)]
pub fn generate_streaming_with<B: Backend>(
    talker: &Talker<B>,
    predictor: &CodePredictor,
    stepper: &mut dyn FrameStepper,
    text: &TextSide,
    params: &SamplingParams,
    rng: &mut StdRng,
    mut on_frame: impl FnMut(&[u32; NUM_CODE_GROUPS]) -> bool,
) -> Vec<[u32; NUM_CODE_GROUPS]> {
    let bench = std::env::var("QWEN3TTS_BENCH").is_ok();
    // Parity gate: run BOTH predictor engines on every frame's real inputs and
    // count per-codebook token agreement. The two draw the same gumbel noise
    // (the rng is cloned, not shared), so a disagreement is numerics, not luck.
    // The GPU's frame is the one that feeds the talker, so what gets measured
    // is the lane that would actually ship.
    let gate = std::env::var("MARY_PRED_GATE").is_ok() && predictor.on_gpu();
    let (mut gate_hit, mut gate_tot) = (0usize, 0usize);
    let mut gate_embed_err = 0f32;

    let mut generated: Vec<u32> = Vec::new();
    let mut frames: Vec<[u32; NUM_CODE_GROUPS]> = Vec::new();

    // Per-frame decomposition (bench only): sync = the one hidden-state
    // read-back, logits = CPU head gemv + processing + sampling, pred = the
    // code predictor, embed = next-input assembly, talker = submission. Frame
    // 0's sync also drains the prefill + first-op JIT — tracked separately so
    // the steady-state numbers stay honest.
    let (mut t_talker, mut t_pred, mut t_sync, mut t_logits, mut t_embed) =
        (0f64, 0f64, 0f64, 0f64, 0f64);
    let mut t_sync0 = 0f64;
    for step in 0..params.max_frames {
        let ts = std::time::Instant::now();
        let h = stepper.hidden(); // the one sync per frame
        let dt_sync = ts.elapsed().as_secs_f64();
        if step == 0 {
            t_sync0 = dt_sync;
        } else {
            t_sync += dt_sync;
        }
        let tl = std::time::Instant::now();
        let mut logits = talker.logits_from(&h);
        process_talker_logits(&mut logits, &generated, step, params.repetition_penalty);
        let code0 = sample_logits(
            &logits,
            params.do_sample,
            params.top_k,
            params.temperature,
            rng,
        );
        t_logits += tl.elapsed().as_secs_f64();
        generated.push(code0);
        if code0 == CODEC_EOS {
            break;
        }

        // fill codebooks 1..15 for this frame
        let tp = std::time::Instant::now();
        let code0_row = talker.codec_row(code0);
        let oracle = gate.then(|| {
            let mut r = rng.clone();
            predictor.predict_frame_cpu(
                &h,
                code0_row,
                params.subtalker_do_sample,
                params.subtalker_temperature,
                &mut r,
            )
        });
        let (rest, mut embed_sum) = predictor.predict_frame(
            &h,
            code0_row,
            params.subtalker_do_sample,
            params.subtalker_temperature,
            rng,
        );
        if let Some((oc, oe)) = oracle {
            gate_hit += oc.iter().zip(&rest).filter(|(a, b)| a == b).count();
            gate_tot += oc.len();
            for (a, b) in oe.iter().zip(&embed_sum) {
                gate_embed_err = gate_embed_err.max((a - b).abs());
            }
        }
        t_pred += tp.elapsed().as_secs_f64();
        let mut frame = [0u32; NUM_CODE_GROUPS];
        frame[0] = code0;
        frame[1..].copy_from_slice(&rest);
        if !on_frame(&frame) {
            break;
        }
        frames.push(frame);

        // next talker input: Σ₁₆ codebook embeds + codebook-0 row + text side,
        // assembled host-side as one row
        let te = std::time::Instant::now();
        let text_row = text.row(step);
        for ((e, &r), &t) in embed_sum.iter_mut().zip(code0_row).zip(text_row) {
            *e += r + t;
        }
        t_embed += te.elapsed().as_secs_f64();
        let tt = std::time::Instant::now();
        let more = stepper.submit(&embed_sum);
        t_talker += tt.elapsed().as_secs_f64();
        if !more {
            break;
        }
    }
    if bench {
        // steady-state = per-frame averages with frame 0's sync excluded
        let n = frames.len().max(1) as f64;
        let ns = (frames.len().saturating_sub(1)).max(1) as f64;
        let per_frame_ms =
            (t_talker / n + t_sync / ns + t_logits / n + t_pred / n + t_embed / n) * 1e3;
        eprintln!(
            "bench: frame0-sync(prefill+JIT) {:.0}ms | per frame: {:.1}ms talker-submit + \
             {:.1}ms sync + {:.1}ms logits+sample + {:.1}ms predictor + {:.1}ms embed = \
             {:.1}ms ({} frames, {:.2}x audio-rate steady)",
            t_sync0 * 1e3,
            t_talker / n * 1e3,
            t_sync / ns * 1e3,
            t_logits / n * 1e3,
            t_pred / n * 1e3,
            t_embed / n * 1e3,
            per_frame_ms,
            frames.len(),
            80.0 / per_frame_ms.max(1e-9)
        );
        if let Some(line) = predictor.take_bench() {
            eprintln!("bench: predictor internals: {line}");
        }
    }
    if gate {
        eprintln!(
            "gate: predictor token agreement {}/{} ({:.2}%) over {} frames; max |Δembed_sum| {:.3e}",
            gate_hit,
            gate_tot,
            gate_hit as f64 / gate_tot.max(1) as f64 * 100.0,
            frames.len(),
            gate_embed_err
        );
    }
    frames
}

/// Decode generated frames with the reference-code prefix, cut the reference
/// span, return samples (24 kHz). Backend-independent of the talker's — the
/// codec runs f32 even when the talker runs f16.
pub fn decode_with_ref<B: Backend>(
    codec: &CodecDecoder<B>,
    ref_code: &[[u32; NUM_CODE_GROUPS]],
    frames: &[[u32; NUM_CODE_GROUPS]],
    device: &B::Device,
) -> Vec<f32> {
    let mut all: Vec<[u32; NUM_CODE_GROUPS]> = ref_code.to_vec();
    all.extend_from_slice(frames);
    let wav = codec.chunked_decode(&all, device);
    let cut = (ref_code.len() as f64 / all.len() as f64 * wav.len() as f64) as usize;
    wav[cut..].to_vec()
}
