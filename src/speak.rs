//! `mary::speak` — self-contained Qwen3-TTS in Burn: speak `gen_text` in the
//! voice of the reference kit, no Python. This is the PRODUCTION voice seam
//! (callers use [`synthesize_stream`] / [`synthesize`] in-process); the F5
//! seam in [`crate::say`] remains as the voice-origin/reference lineage.
//!
//! Weights load from four ordinary roots in one frozen native model-collection
//! snapshot (see the `qwen3tts_persist` bin): exact base, shared exact codec,
//! filtered f16 talker, and versioned folded f16 talker. The talker runs on the
//! RAW (non-fusion) Metal backends — the ONLY
//! talker lane since 2026-07-12: the fused lane was removed after an ear
//! A/B picked raw ("better tempo") at measured perf parity (~28 ms/frame
//! talker GPU — see PORT_NOTES.md, "raw is the ONLY talker lane"). On macOS
//! every talker GPU tensor is a ZERO-COPY alias of the snapshot's mmap'd pile
//! pages: the fold-transformed root supplies final-layout tensors while the
//! filtered root supplies the untransformed embeddings. The f32 codec (still
//! fused — its decode loop is launch-bound and never aliases) uploads
//! straight from its pile blobs; the code predictor and the ECAPA speaker
//! encoder read the exact f32 leaves — "exact f32 *leaves*" is a statement
//! about which weights they load, not about where they run: the speaker
//! encoder has always been Burn/GPU (and runs once per voice, not per frame),
//! and since 2026-08-21 the predictor's frame loop is cubecl too
//! (`qwen3tts::predictor_gpu`, `MARY_PRED_CPU=1` to hold it on the host).
//! `MARY_SPEAK_MATERIALIZE=1`
//! forces the old fully materialized load for A/B measurement.
//!
//! (History: a zero-copy path THROUGH the fusion wrapper was probed and
//! deleted — burn 0.21's fusion codegen miscompiles the full talker graph
//! over many externally-registered buffers, one of three distinct fusion
//! codegen failures this model hit. The raw lane exists precisely so the
//! pile seam never depends on that machinery. See PORT_NOTES.md, "Zero-copy
//! alias probe" + "the zero-copy RAW talker lane".)
//!
//! There is ONE generation path — a streaming one. [`synthesize_stream`]
//! returns an iterator of 24 kHz PCM chunks: frames go to a codec thread the
//! moment they are sampled and are decoded in hop-sized windows with trailing
//! left context (the `qwen3tts_stream` loop, productionized). [`synthesize`]
//! is the batch view of the SAME path: it drains the stream and concatenates.
//!
//! The reference kit is three files: a 24 kHz mono PCM16 clip (the x-vector is
//! computed in-process by the ported ECAPA speaker encoder), its EXACT
//! transcript, and the clip's codec frames as an f32 npy `(T, 16)`. The codec
//! encoder is ported for standalone paths; production Voice deliberately keeps
//! using the precomputed frames until arbitrary-reference selection is part of
//! this API.
//!
//! Long text is split into sentence-sized passes (same splitter as `say`;
//! each pass stays far below the 2048-frame generation cap and re-clones the
//! same reference so the voice stays consistent); a short silence gap is
//! emitted between passes. The Qwen2 BPE builds from the tokenizer files
//! committed under `<mary>/assets/qwen3tts/`.

use std::collections::{HashMap, hash_map::Entry};
use std::path::{Path, PathBuf};
use std::sync::mpsc;
use std::time::Instant;

use crate::model_collection::ModelSnapshot;
use burn::prelude::Backend;
use rand::SeedableRng;
use triblespace::prelude::BlobStoreGet;

use crate::leaf::{Elem, Leaf};
use crate::models::f5::wav;
use crate::models::qwen3tts::codec::CodecDecoder;
use crate::models::qwen3tts::config::{
    LANG_ENGLISH, NUM_CODE_GROUPS, SAMPLE_RATE, SAMPLES_PER_FRAME,
};
use crate::models::qwen3tts::pipeline::{self, ClonePrompt, SamplingParams};
use crate::models::qwen3tts::predictor::CodePredictor;
use crate::models::qwen3tts::speaker::{SpeakerEncoder, SpeakerMel};
use crate::models::qwen3tts::talker::Talker;
use crate::models::qwen3tts::tokenizer::TextTokenizer;
use crate::nn::backend::speak::{Device as WgpuDevice, Fused as BFused};
use crate::nn::npy;
use crate::nn::weight_loader::WeightLoader;
use crate::selection::ModelSelector;

/// Keep each pass comfortably under the 2048-frame (~164 s) generation cap
/// while giving the talker whole paragraphs of prosodic context.
const MAX_CHARS: usize = 500;

/// Streaming decode geometry (the values proven by `qwen3tts_stream`): emit a
/// PCM chunk every `HOP` frames (8 frames = 640 ms of audio), decoding with
/// `CTX` trailing frames of left context (the codec's sliding window is 72;
/// 25 trades a little context for ~3× cheaper hop decodes). Overridable via
/// `MARY_SPEAK_HOP` / `MARY_SPEAK_CTX` for empirical sweeps.
const STREAM_HOP: usize = 8;
const STREAM_CTX: usize = 25;

const BASE: &str = "base";
const TALKER_F16: &str = "talker-f16";
const TALKER_FOLDED_F16: &str = "talker-folded-f16-v1";
const CODEC_SOURCE: &str = "Qwen/Qwen3-TTS-12Hz#codec";

/// Native-width weight label used by the filtered and folded talker roots.
pub const QUANTIZATION_F16: &str = "f16";

/// The two Qwen3-TTS checkpoints supported by the shared runtime graph.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Qwen3TtsVariant {
    Base1_7B,
    Base0_6B,
}

impl Qwen3TtsVariant {
    /// Preserve the public runtime switch while making it select roots in one
    /// frozen graph instead of a sibling file by convention.
    pub fn from_env() -> Self {
        match std::env::var("MARY_SPEAK_MODEL").ok().as_deref() {
            Some("0.6b") => Self::Base0_6B,
            _ => Self::Base1_7B,
        }
    }

    pub const fn source(self) -> &'static str {
        match self {
            Self::Base1_7B => "Qwen/Qwen3-TTS-12Hz-1.7B-Base",
            Self::Base0_6B => "Qwen/Qwen3-TTS-12Hz-0.6B-Base",
        }
    }

    pub fn component_source(self, component: &str) -> String {
        format!("{}#{component}", self.source())
    }

    pub fn base_source(self) -> String {
        self.component_source(BASE)
    }

    pub fn talker_f16_source(self) -> String {
        self.component_source(TALKER_F16)
    }

    pub fn folded_f16_source(self) -> String {
        self.component_source(TALKER_FOLDED_F16)
    }

    pub const fn codec_source() -> &'static str {
        CODEC_SOURCE
    }

    /// Detect the checkpoint size from the talker's codec embedding width.
    #[cfg(feature = "import")]
    pub fn detect(model_dir: &Path) -> anyhow::Result<Self> {
        let path = model_dir.join("model.safetensors");
        let file = std::fs::File::open(&path)?;
        // Mapping makes the safetensors header query constant-memory. Only the
        // header pages are touched; detecting a variant must not reread its
        // multi-gigabyte weight payload before import.
        let bytes = unsafe { memmap2::Mmap::map(&file)? };
        let tensors = safetensors::SafeTensors::deserialize(&bytes)?;
        let codec_embedding = tensors.tensor("talker.model.codec_embedding.weight")?;
        let shape = codec_embedding.shape();
        match shape.get(1).copied() {
            Some(2048) => Ok(Self::Base1_7B),
            Some(1024) => Ok(Self::Base0_6B),
            width => anyhow::bail!(
                "unsupported Qwen3-TTS talker codec-embedding shape {shape:?} (hidden {width:?})"
            ),
        }
    }
}

/// The complete Qwen3-TTS runtime cohort selected from one immutable native
/// model-collection snapshot.
///
/// Four ordinary model roots exhaust the live tensors: the variant's exact
/// base, one codec shared by both variants, a filtered f16 talker, and its
/// pre-folded f16 GPU layout. Only the compact leaf indexes remain resident —
/// each leaf holds its bytes as a view over the pile's mapping and keeps that
/// mapping alive, so no reader is retained. No Repository ancestry or
/// sibling-file naming participates in runtime selection.
pub struct Qwen3TtsWeights {
    variant: Qwen3TtsVariant,
    exact: HashMap<String, Leaf>,
    talker_f16: HashMap<String, Leaf>,
    folded_f16: HashMap<String, Leaf>,
}

impl Qwen3TtsWeights {
    pub fn from_snapshot<R: BlobStoreGet>(
        snapshot: ModelSnapshot<R>,
        variant: Qwen3TtsVariant,
    ) -> anyhow::Result<Self> {
        fn select(
            facts: &triblespace::prelude::TribleSet,
            reader: &impl BlobStoreGet,
            source: &str,
            quantization: &str,
        ) -> anyhow::Result<HashMap<String, Leaf>> {
            let root = crate::selection::select_model_root(
                facts,
                reader,
                ModelSelector::Source {
                    source,
                    quantization,
                },
            )?;
            crate::selection::index_keymap_for_root(facts, reader, root)
        }
        fn require_width(
            component: &str,
            leaves: &HashMap<String, Leaf>,
            f16: bool,
        ) -> anyhow::Result<()> {
            for (name, leaf) in leaves {
                if (leaf.elem() == Elem::F16) != f16 {
                    anyhow::bail!(
                        "Qwen3-TTS {component} tensor {name:?} is {}, expected {}",
                        match leaf.elem() {
                            Elem::F32 => "f32",
                            Elem::F16 => "f16",
                        },
                        if f16 { "f16" } else { "f32" }
                    );
                }
            }
            Ok(())
        }

        let base_source = variant.base_source();
        let talker_source = variant.talker_f16_source();
        let folded_source = variant.folded_f16_source();
        let mut exact = select(
            snapshot.facts(),
            snapshot.store(),
            &base_source,
            crate::persist::QUANTIZATION_NATIVE,
        )?;
        let codec = select(
            snapshot.facts(),
            snapshot.store(),
            Qwen3TtsVariant::codec_source(),
            crate::persist::QUANTIZATION_NATIVE,
        )?;
        require_width("base", &exact, false)?;
        require_width("codec", &codec, false)?;
        for (name, handles) in codec {
            match exact.entry(name) {
                Entry::Vacant(entry) => {
                    entry.insert(handles);
                }
                Entry::Occupied(entry) => {
                    anyhow::bail!(
                        "tensor {:?} occurs in both Qwen3-TTS base and codec roots",
                        entry.key()
                    );
                }
            }
        }
        let talker_f16 = select(
            snapshot.facts(),
            snapshot.store(),
            &talker_source,
            QUANTIZATION_F16,
        )?;
        let folded_f16 = select(
            snapshot.facts(),
            snapshot.store(),
            &folded_source,
            QUANTIZATION_F16,
        )?;
        require_width("talker-f16", &talker_f16, true)?;
        require_width("talker-folded-f16", &folded_f16, true)?;
        drop(snapshot);
        Ok(Self {
            variant,
            exact,
            talker_f16,
            folded_f16,
        })
    }

    pub const fn variant(&self) -> Qwen3TtsVariant {
        self.variant
    }

    pub fn counts(&self) -> (usize, usize, usize) {
        (
            self.exact.len(),
            self.talker_f16.len(),
            self.folded_f16.len(),
        )
    }
}

impl Qwen3TtsWeights {
    /// Exercise every production model constructor without running inference.
    ///
    /// This is intentionally expensive and exists for the native importer:
    /// it proves that the selected folded talker, CPU predictor, speaker
    /// encoder, and codec decoder are all tensor-complete before the importer
    /// reports a cohort as deployable. Ordinary Voice calls construct the same
    /// components lazily on their synthesis threads instead.
    #[cfg(target_os = "macos")]
    pub fn validate_runtime_cohort(self) -> anyhow::Result<()> {
        use crate::nn::backend::BHalf;
        use std::any::Any;
        use std::panic::{AssertUnwindSafe, catch_unwind};

        fn panic_text(payload: &(dyn Any + Send)) -> &str {
            payload
                .downcast_ref::<&str>()
                .copied()
                .or_else(|| payload.downcast_ref::<String>().map(String::as_str))
                .unwrap_or("non-string panic")
        }

        fn construct<T>(component: &str, build: impl FnOnce() -> T) -> anyhow::Result<T> {
            catch_unwind(AssertUnwindSafe(build)).map_err(|payload| {
                anyhow::anyhow!(
                    "Qwen3-TTS {component} construction panicked: {}",
                    panic_text(&*payload)
                )
            })
        }

        let folded = crate::persist::load_qwen3tts_talker_folded_from_indexes(
            &self.talker_f16,
            &self.exact,
            &self.folded_f16,
        )?;
        drop(folded);

        let loader = self.into_loader();
        drop(construct("code predictor", || {
            CodePredictor::load(&loader)
        })?);
        let device = WgpuDevice::default();
        drop(construct("speaker encoder", || {
            SpeakerEncoder::<BHalf>::load(&loader, &device)
        })?);
        drop(construct("codec decoder", || {
            CodecDecoder::<BFused>::load(&loader, &device)
        })?);
        Ok(())
    }

    #[cfg(target_os = "macos")]
    fn folded_talker<B: Backend + 'static>(&self) -> anyhow::Result<Option<Talker<B>>> {
        use crate::nn::backend::BHalf;
        use std::any::TypeId;

        if TypeId::of::<B>() != TypeId::of::<BHalf>()
            || std::env::var("MARY_SPEAK_MATERIALIZE").is_ok()
        {
            return Ok(None);
        }
        let talker = crate::persist::load_qwen3tts_talker_folded_from_indexes(
            &self.talker_f16,
            &self.exact,
            &self.folded_f16,
        )?;
        Ok(Some(crate::nn::weight_loader::same_type::<
            Talker<BHalf>,
            Talker<B>,
        >(talker)))
    }

    #[cfg(not(target_os = "macos"))]
    fn folded_talker<B: Backend + 'static>(&self) -> anyhow::Result<Option<Talker<B>>> {
        Ok(None)
    }

    fn into_loader(self) -> WeightLoader {
        let materialize = std::env::var("MARY_SPEAK_MATERIALIZE").is_ok();
        #[cfg(target_os = "macos")]
        if !materialize {
            return WeightLoader::Aliased(crate::nn::weight_loader::AliasedPile::new(
                self.talker_f16,
                self.exact,
                WgpuDevice::default(),
            ));
        }
        if materialize {
            eprintln!("[mary] MARY_SPEAK_MATERIALIZE set — using the fully materialized load");
        }
        let keymap = self
            .exact
            .into_iter()
            .map(|(name, leaf)| (name, leaf.to_f32_shape()))
            .collect();
        WeightLoader::Pile(keymap)
    }
}

fn env_usize(name: &str, default: usize) -> usize {
    std::env::var(name)
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(default)
}

fn env_f64(name: &str, default: f64) -> f64 {
    std::env::var(name)
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(default)
}

// NOTE on JIT warmup: the codec self-warms on its own thread at the streaming
// chunk shape (overlapped with generation — free). The TALKER is deliberately
// NOT pre-warmed. A pre-warm forward would (a) RACE the codec's first-op JIT on
// the same Metal device — two backends compiling kernels concurrently corrupts
// burn-fusion's op stream ("Ordering is bigger than operations" / codegen
// index panics, verified fatal), and (b) not help TTFA anyway under the
// one-shot process model each `voice say`/`shout` uses: the talker's JIT is
// paid once in the real prefill regardless of whether a warm forward paid it
// first. So the talker JITs lazily in the prefill; the codec's own JIT
// overlaps and is hidden.

/// A live synthesis: iterate it for 24 kHz mono PCM chunks in `[-1, 1]` as
/// they become ready (the first chunk arrives after prefill + `HOP` frames —
/// seconds, not the whole utterance), then call [`finish`](Self::finish) to
/// propagate any synthesis error. Dropping it early cancels the remaining
/// generation of THIS utterance once the codec next tries to emit a chunk;
/// the [`Synthesizer`] it came from is untouched and serves the next one.
pub struct SpeakStream {
    rx: mpsc::Receiver<Vec<f32>>,
    done: mpsc::Receiver<anyhow::Result<()>>,
}

impl Iterator for SpeakStream {
    type Item = Vec<f32>;
    fn next(&mut self) -> Option<Vec<f32>> {
        self.rx.recv().ok()
    }
}

impl SpeakStream {
    /// Sample rate of the emitted PCM.
    pub const SAMPLE_RATE: u32 = SAMPLE_RATE;

    /// Wait for this utterance's generation to end and surface its result.
    /// Also drains any chunks the caller didn't consume.
    pub fn finish(self) -> anyhow::Result<()> {
        while self.rx.recv().is_ok() {}
        match self.done.recv() {
            Ok(result) => result,
            Err(_) => anyhow::bail!("the speak generation thread ended without reporting this utterance"),
        }
    }
}

/// One utterance, as the resident generation thread receives it.
struct Request {
    text: String,
    /// When the caller asked, so TTFA is speak-to-first-audio as the caller
    /// experienced it.
    called: Instant,
    pcm: mpsc::Sender<Vec<f32>>,
    done: mpsc::Sender<anyhow::Result<()>>,
}

/// The resident voice: talker, code predictor, tokenizer, the reference clone
/// prompt and the codec, loaded and warmed ONCE and kept on their own threads
/// across utterances. [`speak`](Self::speak) queues one utterance and returns
/// its [`SpeakStream`] at once; utterances are synthesized in order, each
/// starting from the reference codes again (the codec history is reset at
/// every pass end, exactly as between the passes of one long text).
///
/// This is what a faculty that keeps talking needs and a one-shot `say` does
/// not care about: [`synthesize_stream`] is one session, one utterance, and
/// the session is dropped the moment the stream is handed back — the threads
/// finish that utterance and exit. Dropping a `Synthesizer` never blocks:
/// in-flight utterances complete, later ones are refused.
pub struct Synthesizer {
    tx: mpsc::Sender<Request>,
}

impl Synthesizer {
    /// Load everything on a fresh generation thread and return immediately.
    /// Load errors surface on the FIRST utterance's [`SpeakStream::finish`]
    /// (and every later one), never silently.
    ///
    /// - `weights` — four roots selected from one frozen native model snapshot.
    /// - `ref_wav` — 24 kHz mono PCM16 reference clip (the conditioning identity).
    /// - `ref_text` — the clip's exact transcript.
    /// - `ref_codes` — f32 npy `(T, 16)`: the clip's codec frames.
    pub fn spawn(
        weights: Qwen3TtsWeights,
        ref_wav: &Path,
        ref_text: &str,
        ref_codes: &Path,
    ) -> anyhow::Result<Self> {
        // The talker runs on the RAW (non-fusion) backends — the ONLY talker
        // lane. f16 by default (halves per-step weight traffic — the realtime
        // fast path; identity holds within the resemblyzer gate);
        // MARY_SPEAK_F32=1 selects the full-precision talker. The CPU code
        // predictor and the f32 codec are unaffected by the switch.
        if std::env::var("MARY_SPEAK_F32").is_ok() {
            spawn_impl::<crate::nn::backend::speak::Raw>(weights, ref_wav, ref_text, ref_codes)
        } else {
            spawn_impl::<crate::nn::backend::speak::RawHalf>(weights, ref_wav, ref_text, ref_codes)
        }
    }

    /// Queue `text` and hand back its stream. Chunks arrive while generation
    /// runs; call `finish` on the stream for the utterance's result.
    pub fn speak(&self, text: &str) -> anyhow::Result<SpeakStream> {
        let (tx_pcm, rx_pcm) = mpsc::channel::<Vec<f32>>();
        let (tx_done, rx_done) = mpsc::channel::<anyhow::Result<()>>();
        self.tx
            .send(Request {
                text: text.to_string(),
                called: Instant::now(),
                pcm: tx_pcm,
                done: tx_done,
            })
            .map_err(|_| anyhow::anyhow!("the speak generation thread has exited"))?;
        Ok(SpeakStream {
            rx: rx_pcm,
            done: rx_done,
        })
    }
}

/// Speak `gen_text` in the voice of the reference kit, STREAMING: returns a
/// [`SpeakStream`] whose chunks arrive while generation is still running.
/// Model load runs on the generation thread, so this returns immediately; the
/// codec runs on its own thread (self-warming) and overlaps generation.
///
/// - `weights` — four roots selected from one frozen native model snapshot.
/// - `ref_wav` — 24 kHz mono PCM16 reference clip (the conditioning identity).
/// - `ref_text` — the clip's exact transcript.
/// - `ref_codes` — f32 npy `(T, 16)`: the clip's codec frames.
pub fn synthesize_stream(
    weights: Qwen3TtsWeights,
    ref_wav: &Path,
    ref_text: &str,
    ref_codes: &Path,
    gen_text: &str,
) -> anyhow::Result<SpeakStream> {
    // One session for one utterance: the session is dropped here, so its
    // threads finish this utterance and exit — the one-shot process model
    // `voice say`/`shout` runs under, unchanged.
    let session = Synthesizer::spawn(weights, ref_wav, ref_text, ref_codes)?;
    session.speak(gen_text)
}

/// Speak `gen_text` in the voice of the reference kit. Returns 24 kHz mono
/// audio in `[-1, 1]`. This is the batch view of the ONE streaming generation
/// path: [`synthesize_stream`] drained and concatenated.
pub fn synthesize(
    weights: Qwen3TtsWeights,
    ref_wav: &Path,
    ref_text: &str,
    ref_codes: &Path,
    gen_text: &str,
) -> anyhow::Result<Vec<f32>> {
    let mut stream = synthesize_stream(weights, ref_wav, ref_text, ref_codes, gen_text)?;
    let mut audio: Vec<f32> = Vec::new();
    for chunk in stream.by_ref() {
        audio.extend_from_slice(&chunk);
    }
    stream.finish()?;
    Ok(audio)
}

/// What the generation thread hands the codec thread.
enum CodecMsg {
    /// A new utterance begins: its PCM goes here, its cancellation flag is
    /// raised if the consumer stops taking chunks, and TTFA counts from
    /// `called`.
    Begin {
        pcm: mpsc::Sender<Vec<f32>>,
        cancelled: std::sync::Arc<std::sync::atomic::AtomicBool>,
        called: Instant,
    },
    Frame([u32; NUM_CODE_GROUPS]),
    /// End of a text pass: flush the partial window, reset the decode history
    /// to the reference, and (between passes) emit a short silence gap.
    PassEnd {
        gap: bool,
    },
    /// The utterance is over: close its PCM channel and report what was
    /// emitted, so the generation thread can time it and release the caller.
    End,
}

/// What the codec thread reports back per utterance: samples emitted and the
/// TTFA it measured (0 if nothing was emitted).
type CodecAck = (usize, f64);

fn spawn_impl<B: Backend + 'static>(
    weights: Qwen3TtsWeights,
    ref_wav: &Path,
    ref_text: &str,
    ref_codes: &Path,
) -> anyhow::Result<Synthesizer> {
    let ref_wav = ref_wav.to_path_buf();
    let ref_text = ref_text.to_string();
    let ref_codes = ref_codes.to_path_buf();
    let (tx_req, rx_req) = mpsc::channel::<Request>();

    std::thread::Builder::new()
        .name("speak-gen".into())
        .spawn(move || {
            // Everything up to the first request is the session's load. If it
            // fails, every utterance queued or yet to come is refused with the
            // reason — a session that died loading must not look like one
            // that is merely slow.
            let loaded = (|| -> anyhow::Result<_> {
                crate::models::qwen3tts::cpu::set_interactive_qos();
                let dev: B::Device = Default::default();

                let t_load = Instant::now();
                let folded_talker = weights.folded_talker::<B>()?;
                let loader = weights.into_loader();
                let talker = folded_talker.unwrap_or_else(|| Talker::load(&loader, &dev));
                let mut predictor = CodePredictor::load(&loader);
                // The predictor's frame loop moves to the GPU unless explicitly
                // held back: on the host it cost ~50 ms of an 80 ms frame and
                // synthesis ran below realtime. `MARY_PRED_CPU=1` restores the
                // Accelerate path, which is also what `MARY_PRED_GATE=1` compares
                // against frame by frame.
                #[cfg(feature = "predictor-gpu")]
                if std::env::var("MARY_PRED_CPU").is_err() {
                    let t = Instant::now();
                    predictor.use_gpu();
                    eprintln!(
                        "[timing] predictor → GPU: {:.2}s",
                        t.elapsed().as_secs_f32()
                    );
                }
                let predictor = predictor;
                let spk_enc = SpeakerEncoder::<B>::load(&loader, &dev);
                let tok = TextTokenizer::load(
                    &PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("assets/qwen3tts"),
                );
                eprintln!("[timing] weight load (pile): {:.2}s", t_load.elapsed().as_secs_f32());

                // ── reference kit → clone prompt ──
                let (samples, sr) = wav::read_pcm16_mono(&ref_wav);
                anyhow::ensure!(
                    sr == SAMPLE_RATE,
                    "reference clip must be 24 kHz mono PCM16 (got {sr} Hz): {ref_wav:?}"
                );
                let spk_embedding =
                    spk_enc.forward(SpeakerMel::<B>::new(&dev).forward(&samples, &dev));
                drop(spk_enc);
                let (rc, rcs) = npy::load_npy(&ref_codes)?;
                anyhow::ensure!(
                    rcs.len() == 2 && rcs[1] == NUM_CODE_GROUPS,
                    "ref codes must be (T, {NUM_CODE_GROUPS}), got {rcs:?}: {ref_codes:?}"
                );
                let ref_code: Vec<[u32; NUM_CODE_GROUPS]> = (0..rcs[0])
                    .map(|t| {
                        let mut f = [0u32; NUM_CODE_GROUPS];
                        for q in 0..NUM_CODE_GROUPS {
                            f[q] = rc[t * NUM_CODE_GROUPS + q] as u32;
                        }
                        f
                    })
                    .collect();
                let prompt = ClonePrompt {
                    ref_code,
                    ref_ids: tok.encode(&format!("<|im_start|>assistant\n{ref_text}<|im_end|>\n")),
                    spk_embedding,
                };
                Ok((dev, loader, talker, predictor, tok, prompt))
            })();
            let (dev, loader, talker, predictor, tok, prompt) = match loaded {
                Ok(parts) => parts,
                Err(error) => {
                    let message = format!("{error:#}");
                    eprintln!("speak: session failed to load: {message}");
                    while let Ok(req) = rx_req.recv() {
                        let _ = req.done.send(Err(anyhow::anyhow!("speak session failed to load: {message}")));
                    }
                    return;
                }
            };

            let (hop, ctx) =
                (env_usize("MARY_SPEAK_HOP", STREAM_HOP), env_usize("MARY_SPEAK_CTX", STREAM_CTX));

            // ── codec thread: consumes frames, decodes hop-chunks, emits PCM ──
            // The codec (f32 whatever the talker precision: its im2col conv
            // GEMMs measured slower in f16, and it is cheap in f32) loads AND
            // warms here, overlapping the talker warmup on this thread — and
            // its fused tensors live on the thread that uses them. It outlives
            // every utterance; only the PCM channel changes hands.
            let (tx_c, rx_c) = mpsc::channel::<CodecMsg>();
            let (tx_ack, rx_ack) = mpsc::channel::<CodecAck>();
            let ref_codes_for_codec = prompt.ref_code.clone();
            let codec_thread = match std::thread::Builder::new().name("speak-codec".into()).spawn(
                move || {
                    crate::models::qwen3tts::cpu::set_interactive_qos();
                    let cdev = WgpuDevice::default();
                    let codec = CodecDecoder::<BFused>::load(&loader, &cdev);
                    drop(loader);
                    // warm the decode path at the steady-state chunk shape
                    // (shader compile + autotune land here, not in the first
                    // real chunk — measured ~0.8 s cold vs ~40 ms warm)
                    let _ = codec.decode(&vec![[0u32; NUM_CODE_GROUPS]; ctx + hop], &cdev);

                    // history = ref codes ++ generated; windows slide over it
                    let ref_len = ref_codes_for_codec.len();
                    let mut history = ref_codes_for_codec;
                    let mut decoded_upto = ref_len;
                    let bench = std::env::var("QWEN3TTS_BENCH").is_ok();
                    let codec_stats = std::cell::Cell::new((0f64, 0usize, 0usize)); // (s, chunks, frames)

                    // The utterance in progress: where its PCM goes, how to
                    // tell the generator it is no longer wanted, and its clock.
                    struct Out {
                        pcm: mpsc::Sender<Vec<f32>>,
                        cancelled: std::sync::Arc<std::sync::atomic::AtomicBool>,
                        called: Instant,
                        emitted: usize,
                        ttfa: f64,
                    }
                    let mut out: Option<Out> = None;

                    // Decode `history[from..to]` (with left context) and hand
                    // the PCM to the current utterance. A consumer that has
                    // dropped its stream cancels THAT utterance: its flag is
                    // raised so generation stops early, its output is closed,
                    // and the session goes on.
                    let flush = |history: &[[u32; NUM_CODE_GROUPS]],
                                     from: usize,
                                     to: usize,
                                     out: &mut Option<Out>| {
                        let c = ctx.min(from);
                        let td = Instant::now();
                        let wav = codec.decode(&history[from - c..to], &cdev);
                        if bench {
                            let (s, n, f) = codec_stats.get();
                            codec_stats
                                .set((s + td.elapsed().as_secs_f64(), n + 1, f + (to - from)));
                        }
                        let pcm = wav[c * SAMPLES_PER_FRAME..].to_vec();
                        let Some(o) = out.as_mut() else { return };
                        if o.emitted == 0 {
                            o.ttfa = o.called.elapsed().as_secs_f64();
                            eprintln!(
                                "[timing] TTFA: {:.2}s (call → first PCM chunk, incl. load+warm+prefill)",
                                o.ttfa
                            );
                        }
                        o.emitted += pcm.len();
                        if o.pcm.send(pcm).is_err() {
                            o.cancelled.store(true, std::sync::atomic::Ordering::Relaxed);
                            let finished = out.take().expect("utterance in progress");
                            // Keep the counts for the ack; the channel is gone.
                            *out = Some(Out {
                                pcm: mpsc::channel().0,
                                ..finished
                            });
                        }
                    };

                    while let Ok(msg) = rx_c.recv() {
                        match msg {
                            CodecMsg::Begin { pcm, cancelled, called } => {
                                out = Some(Out {
                                    pcm,
                                    cancelled,
                                    called,
                                    emitted: 0,
                                    ttfa: 0.0,
                                });
                            }
                            CodecMsg::Frame(f) => {
                                history.push(f);
                                if history.len() - decoded_upto >= hop {
                                    let (from, to) = (decoded_upto, history.len());
                                    flush(&history, from, to, &mut out);
                                    decoded_upto = to;
                                }
                            }
                            CodecMsg::PassEnd { gap } => {
                                if history.len() > decoded_upto {
                                    let (from, to) = (decoded_upto, history.len());
                                    flush(&history, from, to, &mut out);
                                }
                                history.truncate(ref_len);
                                decoded_upto = ref_len;
                                if gap {
                                    if let Some(o) = out.as_mut() {
                                        let silence = vec![0.0f32; SAMPLE_RATE as usize / 6]; // ~167 ms
                                        o.emitted += silence.len();
                                        let _ = o.pcm.send(silence);
                                    }
                                }
                            }
                            CodecMsg::End => {
                                if history.len() > decoded_upto {
                                    let (from, to) = (decoded_upto, history.len());
                                    flush(&history, from, to, &mut out);
                                }
                                history.truncate(ref_len);
                                decoded_upto = ref_len;
                                let (emitted, ttfa) = out
                                    .take()
                                    .map(|o| (o.emitted, o.ttfa))
                                    .unwrap_or((0, 0.0));
                                // Dropping `out` closed the utterance's PCM
                                // channel: its iterator ends here.
                                if tx_ack.send((emitted, ttfa)).is_err() {
                                    break;
                                }
                            }
                        }
                    }
                    if bench {
                        let (s, n, f) = codec_stats.get();
                        eprintln!(
                            "bench: codec: {:.1}ms/chunk × {} chunks ({:.1}ms per generated \
                             frame, {} frames; overlapped with generation)",
                            s / n.max(1) as f64 * 1e3,
                            n,
                            s / f.max(1) as f64 * 1e3,
                            f
                        );
                    }
                },
            ) {
                Ok(handle) => handle,
                Err(error) => {
                    let message = format!("spawn the codec thread: {error}");
                    while let Ok(req) = rx_req.recv() {
                        let _ = req.done.send(Err(anyhow::anyhow!("{message}")));
                    }
                    return;
                }
            };

            // ── generation: sentence-packed passes, one shared reference ──
            // max_frames overridable via env for empirical sweeps (mirrors MARY_NFE).
            // MARY_SPEAK_TEMP sets BOTH the talker and sub-talker sampling
            // temperature (one "sampling noise" knob for artifact A/Bs);
            // MARY_SPEAK_TOPK sets the talker top-k. Defaults unchanged.
            let greedy = std::env::var("MARY_SPEAK_GREEDY").is_ok();
            let temp = env_f64("MARY_SPEAK_TEMP", SamplingParams::default().temperature);
            let params = SamplingParams {
                max_frames: env_usize(
                    "MARY_SPEAK_MAX_FRAMES",
                    SamplingParams::default().max_frames,
                ),
                do_sample: !greedy,
                subtalker_do_sample: !greedy,
                temperature: temp,
                subtalker_temperature: temp,
                top_k: env_usize("MARY_SPEAK_TOPK", SamplingParams::default().top_k),
                ..Default::default()
            };
            let mut rng = match std::env::var("MARY_SPEAK_SEED").ok().and_then(|s| s.parse().ok()) {
                Some(seed) => rand::rngs::StdRng::seed_from_u64(seed),
                None => rand::rngs::StdRng::from_entropy(),
            };

            // One utterance at a time, in order, until the session is dropped.
            while let Ok(req) = rx_req.recv() {
                let cancelled = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
                if tx_c
                    .send(CodecMsg::Begin {
                        pcm: req.pcm,
                        cancelled: cancelled.clone(),
                        called: req.called,
                    })
                    .is_err()
                {
                    let _ = req.done.send(Err(anyhow::anyhow!("the speak codec thread has exited")));
                    break;
                }
                let chunks = crate::say::chunk_text(&req.text, MAX_CHARS);
                eprintln!("{} pass(es), ref {} frames", chunks.len(), prompt.ref_code.len());
                let t_synth = Instant::now();
                let mut total_frames = 0usize;
                let mut codec_gone = false;
                for (i, chunk) in chunks.iter().enumerate() {
                    if cancelled.load(std::sync::atomic::Ordering::Relaxed) {
                        break;
                    }
                    let text_ids = tok.encode(&format!(
                        "<|im_start|>assistant\n{chunk}<|im_end|>\n<|im_start|>assistant\n"
                    ));
                    let tb = Instant::now();
                    let (prefill, trailing, tts_pad) = pipeline::build_prefill(
                        &talker,
                        &predictor,
                        &prompt,
                        &text_ids,
                        Some(LANG_ENGLISH),
                        &dev,
                    );
                    if std::env::var("QWEN3TTS_BENCH").is_ok() {
                        eprintln!("bench: build_prefill {:.0}ms", tb.elapsed().as_secs_f32() * 1e3);
                    }
                    let frames = pipeline::generate_streaming(
                        &talker,
                        &predictor,
                        prefill,
                        trailing,
                        tts_pad,
                        &params,
                        &mut rng,
                        &dev,
                        |f| {
                            !cancelled.load(std::sync::atomic::Ordering::Relaxed)
                                && tx_c.send(CodecMsg::Frame(*f)).is_ok()
                        },
                    );
                    total_frames += frames.len();
                    if tx_c.send(CodecMsg::PassEnd { gap: i + 1 < chunks.len() }).is_err() {
                        codec_gone = true;
                        break;
                    }
                    eprintln!(
                        "  pass {}/{}: {} chars → {} frames",
                        i + 1,
                        chunks.len(),
                        chunk.len(),
                        frames.len()
                    );
                }
                if codec_gone || tx_c.send(CodecMsg::End).is_err() {
                    let _ = req.done.send(Err(anyhow::anyhow!("the speak codec thread has exited")));
                    break;
                }
                let Ok((emitted, ttfa)) = rx_ack.recv() else {
                    let _ = req.done.send(Err(anyhow::anyhow!("the speak codec thread has exited")));
                    break;
                };
                let synth_s = t_synth.elapsed().as_secs_f32();
                let audio_s = emitted as f32 / SAMPLE_RATE as f32;
                eprintln!(
                    "[timing] synth: {:.2}s for {:.2}s audio ({:.2}x realtime, {} frames, TTFA {:.2}s{})",
                    synth_s,
                    audio_s,
                    synth_s / audio_s.max(1e-6),
                    total_frames,
                    ttfa,
                    if cancelled.load(std::sync::atomic::Ordering::Relaxed) {
                        ", cancelled by the consumer"
                    } else {
                        ""
                    }
                );
                let _ = req.done.send(Ok(()));
            }
            drop(tx_c);
            if codec_thread.join().is_err() {
                eprintln!("speak: codec thread panicked");
            }
        })
        .map_err(|error| anyhow::anyhow!("spawn the speak generation thread: {error}"))?;

    Ok(Synthesizer { tx: tx_req })
}

/// Convenience: [`synthesize`] then write the result to `out_path` as a 24 kHz
/// mono PCM16 WAV. Returns the number of samples written.
pub fn synthesize_to_wav(
    weights: Qwen3TtsWeights,
    ref_wav: &Path,
    ref_text: &str,
    ref_codes: &Path,
    gen_text: &str,
    out_path: &Path,
) -> anyhow::Result<usize> {
    let audio = synthesize(weights, ref_wav, ref_text, ref_codes, gen_text)?;
    wav::write_pcm16_mono(out_path, &audio, SAMPLE_RATE);
    Ok(audio.len())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::format::{F32Array, U64Array, attrs};
    use ed25519_dalek::SigningKey;
    use std::fs::OpenOptions;
    use std::sync::atomic::{AtomicU64, Ordering};
    use std::time::{SystemTime, UNIX_EPOCH};
    use triblespace::core::repo::pile::Pile;
    use triblespace::prelude::blobencodings::UTF8String;
    use triblespace::prelude::*;

    static NEXT_TEST_PILE: AtomicU64 = AtomicU64::new(0);

    struct TestPile(PathBuf);

    impl TestPile {
        fn new() -> Self {
            let ordinal = NEXT_TEST_PILE.fetch_add(1, Ordering::Relaxed);
            let nanos = SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .expect("system clock after Unix epoch")
                .as_nanos();
            let path = std::env::temp_dir().join(format!(
                "mary-qwen3tts-native-{}-{nanos}-{ordinal}.pile",
                std::process::id()
            ));
            OpenOptions::new()
                .write(true)
                .create_new(true)
                .open(&path)
                .expect("create synthetic Qwen3-TTS pile");
            Self(path)
        }

        fn path(&self) -> &Path {
            &self.0
        }
    }

    impl Drop for TestPile {
        fn drop(&mut self) {
            let _ = std::fs::remove_file(&self.0);
        }
    }

    fn component_fragment(
        source: &str,
        quantization: &str,
        tensor: &str,
        value: f32,
        f16: bool,
        form: crate::leaf::Form,
    ) -> Fragment {
        let mut fragment = Fragment::empty();
        let leaf_id = crate::leaf::fixture_leaf(
            &mut fragment,
            form,
            if f16 { Elem::F16 } else { Elem::F32 },
            &[1_u64],
            &[value],
        );

        let name = fragment.put::<UTF8String, _>(tensor.to_owned());
        let member = entity! { _ @ attrs::safetensor_path: name, attrs::weight: &leaf_id };
        let member_id = member.root().expect("model member root");
        fragment += member;

        let source = fragment.put::<UTF8String, _>(source.to_owned());
        fragment += entity! { _ @
            attrs::source: source,
            attrs::quantization: quantization,
            attrs::member: &member_id,
        };
        fragment
    }

    fn publish(path: &Path, fragments: impl IntoIterator<Item = Fragment>) {
        let mut pile = Pile::open(path).expect("open synthetic Qwen3-TTS pile");
        for fragment in fragments {
            crate::model_collection::publish_model_fragment(
                &mut pile,
                &SigningKey::from_bytes(&[0x51; 32]),
                fragment,
            )
            .expect("publish native Qwen3-TTS component");
        }
        pile.close().expect("close synthetic Qwen3-TTS pile");
    }

    fn cohort(variant: Qwen3TtsVariant, form: crate::leaf::Form) -> [Fragment; 4] {
        [
            component_fragment(
                &variant.component_source(BASE),
                crate::persist::QUANTIZATION_NATIVE,
                "base.weight",
                1.0,
                false,
                form,
            ),
            component_fragment(
                CODEC_SOURCE,
                crate::persist::QUANTIZATION_NATIVE,
                "codec.weight",
                2.0,
                false,
                form,
            ),
            component_fragment(
                &variant.component_source(TALKER_F16),
                QUANTIZATION_F16,
                "talker.weight",
                3.0,
                true,
                form,
            ),
            component_fragment(
                &variant.component_source(TALKER_FOLDED_F16),
                QUANTIZATION_F16,
                "talker.folded.weight",
                4.0,
                true,
                form,
            ),
        ]
    }

    /// The cohort selects, and the four components keep their widths, whether
    /// the leaves are one typed blob or the two-blob pair the piles still
    /// hold. Both, because the migration to typed leaves has not happened on
    /// disk yet: the two-blob arm is what `models/qwen3tts.pile` is TODAY and
    /// the typed arm is what a conversion of it would be, and this seam has to
    /// survive the change rather than be adjusted after it.
    #[test]
    fn qwen3tts_cohort_selects_in_either_leaf_form() {
        for form in crate::leaf::FORMS {
            let file = TestPile::new();
            let variant = Qwen3TtsVariant::Base1_7B;
            let fragments = cohort(variant, form);

            // The arm did what it says. Without this the typed pass would
            // still be green if `fixture_leaf` quietly built two-blob leaves
            // for both forms — a test that checks the reader by way of a
            // writer it never checked.
            let two_blob: usize = fragments
                .iter()
                .map(|f| {
                    find!((e: Id), pattern!(f.facts(), [{ ?e @ attrs::data: _?d }])).count()
                        + find!((e: Id), pattern!(f.facts(), [{ ?e @ attrs::data_f16: _?d }]))
                            .count()
                })
                .sum();
            match form {
                crate::leaf::Form::Typed => {
                    assert_eq!(two_blob, 0, "typed cohort must state no two-blob leaf")
                }
                crate::leaf::Form::TwoBlob => {
                    assert_eq!(two_blob, 4, "two-blob cohort must state four of them")
                }
            }
            publish(file.path(), fragments);

            let snapshot = crate::model_collection::load_model_collection_local_latest(file.path())
                .expect("freeze native Qwen3-TTS snapshot");
            let weights = Qwen3TtsWeights::from_snapshot(snapshot, variant)
                .unwrap_or_else(|e| panic!("{} cohort must select: {e:#}", form.label()));
            assert_eq!(weights.counts(), (2, 1, 1), "{}", form.label());
            assert_eq!(
                weights.exact["base.weight"].elem(),
                Elem::F32,
                "{}",
                form.label()
            );
            assert_eq!(
                weights.talker_f16["talker.weight"].elem(),
                Elem::F16,
                "{}",
                form.label()
            );
            assert_eq!(
                weights.folded_f16["talker.folded.weight"].shape(),
                vec![1],
                "{}",
                form.label()
            );
        }
    }

    #[test]
    fn one_snapshot_owns_four_explicit_qwen3tts_components() {
        let file = TestPile::new();
        let variant = Qwen3TtsVariant::Base1_7B;
        publish(file.path(), cohort(variant, crate::leaf::Form::TwoBlob));

        let snapshot = crate::model_collection::load_model_collection_local_latest(file.path())
            .expect("freeze native Qwen3-TTS snapshot");
        let weights = Qwen3TtsWeights::from_snapshot(snapshot, variant)
            .expect("select complete native Qwen3-TTS cohort");
        assert_eq!(weights.variant(), variant);
        assert_eq!(weights.counts(), (2, 1, 1));
        assert!(weights.exact.contains_key("base.weight"));
        assert!(weights.exact.contains_key("codec.weight"));

        // The owned view stays frozen, while a fresh widened view fails closed
        // when a second root claims the codec coordinate.
        publish(
            file.path(),
            [component_fragment(
                CODEC_SOURCE,
                crate::persist::QUANTIZATION_NATIVE,
                "other-codec.weight",
                9.0,
                false,
                crate::leaf::Form::TwoBlob,
            )],
        );
        assert_eq!(weights.counts(), (2, 1, 1));

        let widened = crate::model_collection::load_model_collection_local_latest(file.path())
            .expect("load widened native Qwen3-TTS snapshot");
        let error = Qwen3TtsWeights::from_snapshot(widened, variant)
            .err()
            .expect("same-coordinate codec roots must fail closed");
        assert!(
            error.to_string().contains("ambiguous model root"),
            "unexpected ambiguity diagnostic: {error:#}"
        );
    }
}
