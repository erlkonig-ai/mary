//! gemma_listen — a continuous LISTEN LOOP: microphone → utterances →
//! Gemma 4 hearing → text, continuously.
//!
//! Pipeline: a NAMED CoreAudio input device (cpal; the Reachy Mini's XVF3800
//! mic array shows up as one, so does any Mac mic) → energy VAD with hangover
//! (endpointing) → each finished utterance runs the parity-gated hear path
//! ([`mary::models::gemma::gemma4::hear::Hearing`]: log-mel → audio tower →
//! embedder → decoder, weights pile-only) → transcript/understanding printed
//! with timestamps and appended as one JSON line per utterance to `--log`.
//!
//! File-based gate (no live audio needed): `--wav a.wav,b.wav` feeds recorded
//! clips through the SAME segmenter + hear code path the mic uses — this is
//! how the loop is tested silently; the live mic run is a ceremony, not a CI
//! step.
//!
//! # The bridge plan (next slice): closing the talking loop
//!
//! This bin is the LISTENING end of a full spoken-conversation loop:
//!
//!   1. LISTEN (this bin): Reachy mic → VAD → `Hearing::understand` → utterance
//!      records appended to the `--log` jsonl (`{utc_ms, start_s, end_s,
//!      dur_s, source, text, ...}`). The jsonl is a *seam*: append-only,
//!      tail-able, crash-safe — the same shape as a pile branch, kept as a
//!      flat file only until the cortex side settles.
//!   2. CORTEX: a playground session (or a thin bridge daemon) tails the
//!      jsonl, wraps each utterance as a user turn, and lets the model reply.
//!      The Hearing seam hands over TEXT for v0; the same seam can hand the
//!      decoder AUDIO EMBEDDINGS instead (a consumer that takes embeddings keeps
//!      tone/hesitation/mood across the seam — the paralinguistic upgrade slots
//!      in without touching capture or VAD).
//!   3. MOUTH: the reply goes out through the TTS path (`mary::speak` /
//!      a speaker daemon) — the bridge just calls it.
//!   4. TURN-TAKING v0: half-duplex — IMPLEMENTED via `--pause-file <path>`:
//!      while that file exists (the bridge creates it before speaking and
//!      removes it after), incoming live audio is dropped and any open
//!      utterance is abandoned as presumed self-echo. The speaking bridge
//!      is the other end of this seam. The XVF3800's on-chip AEC makes
//!      full-duplex feasible later without changing the seam.
//!
//! VAD is energy-based with hangover for v0 (adaptive noise floor × ratio,
//! start debounce, pre-roll). Upgrade path: silero-vad (tiny ONNX) drops into
//! `Segmenter::is_speech` without changing anything downstream.
//!
//! Usage:
//!   # enumerate input devices (no model load):
//!   gemma_listen --list-devices
//!   # file-based gate:
//!   gemma_listen --pile gemma_e4b.pile --wav /tmp/clip1.wav,/tmp/clip2.wav \
//!     --log /tmp/gemma_listen.jsonl
//!   # live (pause-file = the half-duplex seam):
//!   gemma_listen --pile gemma_e4b.pile --device "Reachy Mini Audio" \
//!     --log /tmp/stt.jsonl --pause-file /tmp/stt.pause
//!
//! `--pile` falls back to `GEMMA_PILE`. `--save-segments <dir>` writes each
//! utterance as a 16 kHz PCM16 wav (what the stt actually heard — replay any
//! segment through `gemma_hear` to debug VAD or understanding).

use burn::backend::wgpu::{Wgpu, WgpuDevice};
use mary::models::gemma::gemma4::audio_load::{load_audio_16k_mono, resample_to_16k};
use mary::models::gemma::gemma4::config::Gemma4Config;
use mary::models::gemma::gemma4::hear::Hearing;
use mary::persist::load_gemma4_hearing_from_pile;
use std::io::Write;
use std::path::Path;
use std::process::Command;
use std::time::{Instant, SystemTime, UNIX_EPOCH};
use tokenizers::Tokenizer;

type B = Wgpu;

// ---------------------------------------------------------------------------
// Energy VAD + endpointing
// ---------------------------------------------------------------------------

#[derive(Clone, Debug)]
struct VadConfig {
    /// Analysis frame length, ms (energy granularity).
    frame_ms: usize,
    /// Consecutive speech frames required to open an utterance.
    start_frames: usize,
    /// Trailing silence that closes an utterance, ms.
    hangover_ms: usize,
    /// Utterances shorter than this are dropped (clicks, coughs), ms.
    min_utt_ms: usize,
    /// Force-close after this many seconds (stay inside the 30 s window).
    max_utt_s: f32,
    /// Speech threshold = max(abs_floor, noise_floor * ratio).
    ratio: f32,
    abs_floor: f32,
    /// Audio kept from BEFORE the trigger frame (soft onsets), ms.
    preroll_ms: usize,
}

impl Default for VadConfig {
    fn default() -> Self {
        VadConfig {
            frame_ms: 20,
            start_frames: 3,
            hangover_ms: 700,
            min_utt_ms: 300,
            max_utt_s: 28.0,
            ratio: 3.5,
            abs_floor: 0.008,
            preroll_ms: 240,
        }
    }
}

/// A finished utterance at the segmenter's native sample rate.
struct Utterance {
    samples: Vec<f32>,
    rate: usize,
    /// Seconds since the stream/file started.
    start_s: f64,
    end_s: f64,
}

/// Streaming energy-VAD segmenter. Feed arbitrary-size mono chunks at any
/// fixed rate; complete utterances are handed to the `emit` callback. The
/// SAME code path serves the live mic and recorded files.
struct Segmenter {
    cfg: VadConfig,
    rate: usize,
    frame: usize,
    pending: Vec<f32>,
    preroll: std::collections::VecDeque<f32>,
    preroll_cap: usize,
    noise_floor: f32,
    floor_warm: usize,
    in_speech: bool,
    speech_run: usize,
    silence_run: usize,
    current: Vec<f32>,
    utt_start_sample: u64,
    samples_seen: u64,
}

impl Segmenter {
    fn new(rate: usize, cfg: VadConfig) -> Self {
        let frame = rate * cfg.frame_ms / 1000;
        let preroll_cap = rate * cfg.preroll_ms / 1000;
        Segmenter {
            cfg,
            rate,
            frame,
            pending: Vec::new(),
            preroll: std::collections::VecDeque::with_capacity(preroll_cap),
            preroll_cap,
            noise_floor: 0.0,
            floor_warm: 0,
            in_speech: false,
            speech_run: 0,
            silence_run: 0,
            current: Vec::new(),
            utt_start_sample: 0,
            samples_seen: 0,
        }
    }

    fn push(&mut self, chunk: &[f32], emit: &mut impl FnMut(Utterance)) {
        self.pending.extend_from_slice(chunk);
        while self.pending.len() >= self.frame {
            let frame: Vec<f32> = self.pending.drain(..self.frame).collect();
            self.frame_in(&frame, emit);
        }
    }

    /// End of stream/file: close any open utterance.
    fn flush(&mut self, emit: &mut impl FnMut(Utterance)) {
        if !self.pending.is_empty() {
            let rest = std::mem::take(&mut self.pending);
            if self.in_speech {
                self.current.extend_from_slice(&rest);
                self.samples_seen += rest.len() as u64;
            }
        }
        if self.in_speech {
            self.close_utterance(0, emit);
        }
    }

    fn frame_in(&mut self, frame: &[f32], emit: &mut impl FnMut(Utterance)) {
        let rms = (frame.iter().map(|&x| x * x).sum::<f32>() / frame.len() as f32).sqrt();

        // Adaptive noise floor: fast warm-up on the first ~0.5 s, then a slow
        // EMA that only follows CALM audio (speech must not drag it up).
        let warm_frames = 500 / self.cfg.frame_ms;
        if self.floor_warm < warm_frames {
            self.noise_floor = if self.floor_warm == 0 {
                rms
            } else {
                0.7 * self.noise_floor + 0.3 * rms
            };
            self.floor_warm += 1;
        } else if !self.in_speech && rms < self.noise_floor * 2.0 {
            self.noise_floor = 0.98 * self.noise_floor + 0.02 * rms;
        }

        let threshold = (self.noise_floor * self.cfg.ratio).max(self.cfg.abs_floor);
        let speech = rms > threshold;

        if !self.in_speech {
            // Keep the pre-roll ring current.
            for &s in frame {
                if self.preroll.len() == self.preroll_cap {
                    self.preroll.pop_front();
                }
                self.preroll.push_back(s);
            }
            if speech {
                self.speech_run += 1;
                if self.speech_run >= self.cfg.start_frames {
                    // Trigger: utterance opens at the start of the pre-roll.
                    self.in_speech = true;
                    self.silence_run = 0;
                    self.current = self.preroll.iter().copied().collect();
                    self.utt_start_sample = (self.samples_seen + frame.len() as u64)
                        .saturating_sub(self.current.len() as u64);
                }
            } else {
                self.speech_run = 0;
            }
        } else {
            self.current.extend_from_slice(frame);
            if speech {
                self.silence_run = 0;
            } else {
                self.silence_run += 1;
                let hangover_frames = self.cfg.hangover_ms / self.cfg.frame_ms;
                if self.silence_run >= hangover_frames {
                    // Trim most of the hangover, keep a ~200 ms tail.
                    let keep_tail = self.rate * 200 / 1000;
                    let hang = self.silence_run * self.frame;
                    let cut = hang.saturating_sub(keep_tail).min(self.current.len());
                    let newlen = self.current.len() - cut;
                    self.current.truncate(newlen);
                    self.close_utterance(0, emit);
                }
            }
            if self.in_speech && self.current.len() as f32 >= self.cfg.max_utt_s * self.rate as f32
            {
                self.close_utterance(0, emit);
            }
        }
        self.samples_seen += frame.len() as u64;
    }

    /// Half-duplex pause: the mouth is speaking, so `n` incoming samples are
    /// dropped. Any open utterance is abandoned (it would be self-echo), the
    /// speech state clears, the adaptive noise floor is KEPT (no re-warm-up),
    /// and the stream clock still advances so later timestamps stay
    /// stream-relative.
    fn pause_skip(&mut self, n: u64) {
        self.pending.clear();
        self.preroll.clear();
        self.current.clear();
        self.in_speech = false;
        self.speech_run = 0;
        self.silence_run = 0;
        self.samples_seen += n;
    }

    fn close_utterance(&mut self, _pad: usize, emit: &mut impl FnMut(Utterance)) {
        let samples = std::mem::take(&mut self.current);
        self.in_speech = false;
        self.speech_run = 0;
        self.silence_run = 0;
        self.preroll.clear();
        let min_len = self.rate * self.cfg.min_utt_ms / 1000;
        if samples.len() >= min_len {
            let start_s = self.utt_start_sample as f64 / self.rate as f64;
            let end_s = start_s + samples.len() as f64 / self.rate as f64;
            emit(Utterance {
                samples,
                rate: self.rate,
                start_s,
                end_s,
            });
        }
    }
}

// ---------------------------------------------------------------------------
// Small helpers
// ---------------------------------------------------------------------------

fn arg(args: &[String], k: &str) -> Option<String> {
    args.iter()
        .position(|s| s == k)
        .map(|i| args[i + 1].clone())
}

fn find_hf_file(model_id: &str, filename: &str) -> String {
    let o = Command::new("python3")
        .args([
            "-c",
            &format!(
                "from huggingface_hub import hf_hub_download; print(hf_hub_download('{}', '{}'))",
                model_id, filename
            ),
        ])
        .output()
        .unwrap();
    String::from_utf8(o.stdout).unwrap().trim().to_string()
}

/// Minimal PCM16 mono WAV writer (segment snapshots for replay/debug).
fn write_wav_pcm16(path: &Path, samples: &[f32], rate: u32) -> std::io::Result<()> {
    let n = samples.len() as u32;
    let data_len = n * 2;
    let mut f = std::fs::File::create(path)?;
    f.write_all(b"RIFF")?;
    f.write_all(&(36 + data_len).to_le_bytes())?;
    f.write_all(b"WAVEfmt ")?;
    f.write_all(&16u32.to_le_bytes())?;
    f.write_all(&1u16.to_le_bytes())?; // PCM
    f.write_all(&1u16.to_le_bytes())?; // mono
    f.write_all(&rate.to_le_bytes())?;
    f.write_all(&(rate * 2).to_le_bytes())?;
    f.write_all(&2u16.to_le_bytes())?;
    f.write_all(&16u16.to_le_bytes())?;
    f.write_all(b"data")?;
    f.write_all(&data_len.to_le_bytes())?;
    let mut buf = Vec::with_capacity(samples.len() * 2);
    for &s in samples {
        let v = (s.clamp(-1.0, 1.0) * 32767.0) as i16;
        buf.extend_from_slice(&v.to_le_bytes());
    }
    f.write_all(&buf)
}

fn json_escape(s: &str) -> String {
    let mut out = String::with_capacity(s.len() + 8);
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if (c as u32) < 0x20 => out.push_str(&format!("\\u{:04x}", c as u32)),
            c => out.push(c),
        }
    }
    out
}

fn list_input_devices() {
    use cpal::traits::{DeviceTrait, HostTrait};
    let host = cpal::default_host();
    let default_name = host
        .default_input_device()
        .and_then(|d| d.description().ok().map(|desc| desc.name().to_string()));
    println!("Audio INPUT devices (cpal/CoreAudio):");
    match host.input_devices() {
        Ok(devs) => {
            for dev in devs {
                let Ok(desc) = dev.description() else {
                    continue;
                };
                let name = desc.name().to_string();
                let cfg = dev
                    .default_input_config()
                    .map(|c| {
                        format!(
                            "{} ch @ {} Hz, {:?}",
                            c.channels(),
                            c.sample_rate(),
                            c.sample_format()
                        )
                    })
                    .unwrap_or_else(|e| format!("no default config: {e}"));
                let marker = if Some(&name) == default_name.as_ref() {
                    "  [default]"
                } else {
                    ""
                };
                println!("  {name}{marker}\n      {cfg}");
            }
        }
        Err(e) => println!("  (enumeration failed: {e})"),
    }
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

struct UttLog<'a> {
    log_path: &'a str,
    save_dir: Option<&'a str>,
    prompt: &'a str,
    n: usize,
}

impl UttLog<'_> {
    /// Run one utterance through the stt and record it (print + jsonl +
    /// optional segment wav). `wave16k` must already be 16 kHz mono.
    fn handle(
        &mut self,
        hearing: &Hearing<B>,
        source: &str,
        u: &Utterance,
        wave16k: Vec<f32>,
        max_new: usize,
    ) {
        self.n += 1;
        let utc_ms = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_millis() as u64)
            .unwrap_or(0);
        let dur = u.end_s - u.start_s;
        println!(
            "\n[utt {:03}] {source} {:.2}s..{:.2}s ({dur:.2}s)",
            self.n, u.start_s, u.end_s
        );

        let wav_path = self.save_dir.map(|d| {
            let p = format!("{d}/utt_{:03}_{utc_ms}.wav", self.n);
            if let Err(e) = write_wav_pcm16(Path::new(&p), &wave16k, 16_000) {
                eprintln!("  (segment save failed: {e})");
            }
            p
        });

        print!("  → ");
        let t = Instant::now();
        let text = hearing.understand(&wave16k, self.prompt, max_new, |piece| {
            print!("{piece}");
            std::io::stdout().flush().ok();
        });
        let latency = t.elapsed().as_secs_f64();
        println!("\n  ({latency:.2}s)");

        let wav_field = wav_path
            .map(|p| format!(",\"wav\":\"{}\"", json_escape(&p)))
            .unwrap_or_default();
        let line = format!(
            "{{\"utc_ms\":{utc_ms},\"source\":\"{}\",\"start_s\":{:.3},\"end_s\":{:.3},\"dur_s\":{:.3},\"prompt\":\"{}\",\"text\":\"{}\",\"latency_s\":{:.3}{}}}\n",
            json_escape(source),
            u.start_s,
            u.end_s,
            dur,
            json_escape(self.prompt),
            json_escape(text.trim()),
            latency,
            wav_field,
        );
        if let Err(e) = std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(self.log_path)
            .and_then(|mut f| f.write_all(line.as_bytes()))
        {
            eprintln!("  (log append failed: {e})");
        }
    }
}

fn main() {
    let args: Vec<String> = std::env::args().collect();

    if args.iter().any(|a| a == "--list-devices") {
        list_input_devices();
        return;
    }

    let pile = arg(&args, "--pile")
        .or_else(|| std::env::var("GEMMA_PILE").ok())
        .unwrap_or_else(|| {
            eprintln!("gemma_listen: pass --pile <gemma.pile> or set GEMMA_PILE");
            std::process::exit(2);
        });
    let model_id = arg(&args, "--model").unwrap_or_else(|| "google/gemma-4-E4B-it".into());
    let prompt =
        arg(&args, "--prompt").unwrap_or_else(|| "Transcribe exactly what is being said.".into());
    let max_new = arg(&args, "--tokens")
        .and_then(|s| s.parse().ok())
        .unwrap_or(80);
    let log_path = arg(&args, "--log").unwrap_or_else(|| "/tmp/gemma_listen.jsonl".into());
    let save_dir = arg(&args, "--save-segments");
    if let Some(d) = &save_dir {
        std::fs::create_dir_all(d).expect("create --save-segments dir");
    }
    let wavs = arg(&args, "--wav");
    let device_pat = arg(&args, "--device");
    if wavs.is_none() && device_pat.is_none() {
        eprintln!("gemma_listen: pass --wav <a.wav,b.wav> (file gate) or --device <name> (live)");
        std::process::exit(2);
    }

    let mut vad = VadConfig::default();
    if let Some(v) = arg(&args, "--vad-ratio").and_then(|s| s.parse().ok()) {
        vad.ratio = v;
    }
    if let Some(v) = arg(&args, "--vad-floor").and_then(|s| s.parse().ok()) {
        vad.abs_floor = v;
    }
    if let Some(v) = arg(&args, "--hangover-ms").and_then(|s| s.parse().ok()) {
        vad.hangover_ms = v;
    }
    if let Some(v) = arg(&args, "--min-ms").and_then(|s| s.parse().ok()) {
        vad.min_utt_ms = v;
    }
    if let Some(v) = arg(&args, "--max-s").and_then(|s| s.parse().ok()) {
        vad.max_utt_s = v;
    }

    // --- Warm the hearing stack (pile-only weights) ---
    let device = WgpuDevice::default();
    let config_path = find_hf_file(&model_id, "config.json");
    let config = Gemma4Config::load(Path::new(&config_path));
    let tokenizer_path = find_hf_file(&model_id, "tokenizer.json");
    let tokenizer = Tokenizer::from_file(&tokenizer_path).unwrap();
    eprintln!("Loading hearing stack from pile {pile}...");
    let (model, _vision, tower, embedder) =
        load_gemma4_hearing_from_pile::<B>(Path::new(&pile), config, &device)
            .unwrap_or_else(|e| panic!("pile load: {e}"));
    let hearing = Hearing::new(model, tower, embedder, tokenizer, device);
    eprintln!("Loaded. Logging utterances to {log_path}");

    let mut ulog = UttLog {
        log_path: &log_path,
        save_dir: save_dir.as_deref(),
        prompt: &prompt,
        n: 0,
    };

    if let Some(wavs) = wavs {
        // ------- File-based gate: same segmenter + hear path as live -------
        for wav in wavs.split(',').map(str::trim).filter(|s| !s.is_empty()) {
            eprintln!("\n=== {wav} ===");
            let wave = load_audio_16k_mono(Path::new(wav))
                .unwrap_or_else(|e| panic!("audio load {wav}: {e}"));
            let mut seg = Segmenter::new(16_000, vad.clone());
            let mut utts: Vec<Utterance> = Vec::new();
            let mut emit = |u: Utterance| utts.push(u);
            // Feed in 100 ms chunks like a capture callback would.
            for chunk in wave.chunks(1600) {
                seg.push(chunk, &mut emit);
            }
            seg.flush(&mut emit);
            eprintln!("  segmenter: {} utterance(s)", utts.len());
            for u in utts {
                let wave16k = u.samples.clone();
                ulog.handle(&hearing, wav, &u, wave16k, max_new);
            }
        }
        return;
    }

    // ------------------------- Live capture -------------------------------
    let pat = device_pat.unwrap();
    use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
    let host = cpal::default_host();
    let needle = pat.to_lowercase();
    let dev = host
        .input_devices()
        .expect("enumerate input devices")
        .find(|d| {
            d.description()
                .map(|desc| desc.name().to_lowercase().contains(&needle))
                .unwrap_or(false)
        })
        .unwrap_or_else(|| {
            eprintln!("no input device matching {pat:?}; available:");
            list_input_devices();
            std::process::exit(2);
        });
    let dev_name = dev
        .description()
        .map(|d| d.name().to_string())
        .unwrap_or_else(|_| pat.clone());
    let sup = dev.default_input_config().expect("default input config");
    let rate = sup.sample_rate() as usize;
    let channels = sup.channels() as usize;
    eprintln!(
        "Capturing from {dev_name:?}: {channels} ch @ {rate} Hz, {:?}",
        sup.sample_format()
    );

    let (tx, rx) = std::sync::mpsc::channel::<Vec<f32>>();
    let cfg: cpal::StreamConfig = sup.config();
    let err_cb = |e| eprintln!("stream error: {e}");
    // Downmix interleaved frames to mono in the callback; ship to the main
    // thread (inference must not run on the CoreAudio thread).
    let stream = match sup.sample_format() {
        cpal::SampleFormat::F32 => dev
            .build_input_stream(
                &cfg,
                move |data: &[f32], _: &cpal::InputCallbackInfo| {
                    let mono: Vec<f32> = data
                        .chunks_exact(channels)
                        .map(|fr| fr.iter().sum::<f32>() / channels as f32)
                        .collect();
                    let _ = tx.send(mono);
                },
                err_cb,
                None,
            )
            .expect("build f32 input stream"),
        cpal::SampleFormat::I16 => dev
            .build_input_stream(
                &cfg,
                move |data: &[i16], _: &cpal::InputCallbackInfo| {
                    let mono: Vec<f32> = data
                        .chunks_exact(channels)
                        .map(|fr| {
                            fr.iter().map(|&s| s as f32 / 32768.0).sum::<f32>() / channels as f32
                        })
                        .collect();
                    let _ = tx.send(mono);
                },
                err_cb,
                None,
            )
            .expect("build i16 input stream"),
        other => {
            eprintln!("unsupported input sample format {other:?}");
            std::process::exit(2);
        }
    };
    stream.play().expect("start capture stream");
    eprintln!(
        "Listening. Speak; utterances end after {} ms of silence. Ctrl-C to stop.",
        vad.hangover_ms
    );

    let mut seg = Segmenter::new(rate, vad);
    let source = format!("device:{dev_name}");
    let pause_file = arg(&args, "--pause-file");
    let mut was_paused = false;
    loop {
        let Ok(chunk) = rx.recv() else { break };
        // Half-duplex turn-taking: while the pause file exists (the converse
        // bridge holds it open around each spoken reply), drop the audio —
        // without AEC the mic would hear the mouth. A stat per ~10 ms chunk
        // is noise.
        if let Some(p) = &pause_file {
            let paused = Path::new(p).exists();
            if paused != was_paused {
                eprintln!(
                    "[half-duplex] {}",
                    if paused {
                        "paused (mouth speaking)"
                    } else {
                        "resumed listening"
                    }
                );
                was_paused = paused;
            }
            if paused {
                seg.pause_skip(chunk.len() as u64);
                continue;
            }
        }
        let mut finished: Vec<Utterance> = Vec::new();
        {
            let mut emit = |u: Utterance| finished.push(u);
            seg.push(&chunk, &mut emit);
        }
        for u in finished {
            let wave16k = resample_to_16k(u.samples.clone(), u.rate)
                .unwrap_or_else(|e| panic!("resample: {e}"));
            ulog.handle(&hearing, &source, &u, wave16k, max_new);
        }
    }
}
