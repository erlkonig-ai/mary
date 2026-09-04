//! The dMel front end: 16 kHz samples in, eighty mel levels per 50 ms out.
//!
//! This is a TOKENIZER, not model math: an FFT, a Slaney mel filterbank, a
//! log, a clamp and a nearest-centre quantiser, with no learned parameter in
//! it. It runs on the host where the samples are, exactly like the text
//! tokenizer does; the model's side -- the 1,280-row table, the sum over the
//! eighty rows and the norm -- is the Session's, where text ids become rows.
//!
//! The constants are the shipped processor's (`processor_config.json`: n_fft
//! 1600, hop 800, 80 bins, 16 levels over [-7, 2]) and the steps were read off
//! `InklingFeatureExtractor` and `InklingProcessor._extract_dmel_bins` rather
//! than remembered, because four of them are the kind that get remembered
//! wrong: the log is `log10` of the MAGNITUDE (not the power), the window is a
//! periodic Hann, the STFT is not centred (the left pad of `n_fft - hop` is
//! explicit), and the level is `argmin |mel - centre|` over
//! `linspace(-7, 2, 16)`, ties to the lower centre.
//!
//! Resampling is here too, because the body records at 24 kHz and the ear
//! delivers what the body records (raw, never derived); the mind brings the
//! sound to the front end's 16 kHz with [`resample`] before [`DmelFrontEnd::levels`].

use super::resident::{DMEL_BINS, DMEL_LEVELS};
use rustfft::{num_complex::Complex, Fft, FftPlanner};
use std::sync::Arc;

pub const SAMPLE_RATE: usize = 16_000;

/// Mono samples at `from` Hz to `SAMPLE_RATE`, with the same windowed-sinc
/// resampler the Gemma ear uses (`rubato`, 256-tap, Blackman-Harris),
/// delay-compensated and trimmed to the expected length. Identity at 16 kHz.
pub fn resample(mono: Vec<f32>, from: usize) -> anyhow::Result<Vec<f32>> {
    use rubato::{
        Resampler, SincFixedIn, SincInterpolationParameters, SincInterpolationType, WindowFunction,
    };
    if from == SAMPLE_RATE {
        return Ok(mono);
    }
    if mono.is_empty() {
        return Ok(Vec::new());
    }
    let ratio = SAMPLE_RATE as f64 / from as f64;
    let params = SincInterpolationParameters {
        sinc_len: 256,
        f_cutoff: 0.95,
        interpolation: SincInterpolationType::Linear,
        oversampling_factor: 256,
        window: WindowFunction::BlackmanHarris2,
    };
    let chunk = 4096usize;
    let mut resampler = SincFixedIn::<f32>::new(ratio, 2.0, params, chunk, 1)
        .map_err(|e| anyhow::anyhow!("rubato init: {e}"))?;
    let delay = resampler.output_delay();
    let mut out = Vec::with_capacity((mono.len() as f64 * ratio) as usize + 1024);
    let mut i = 0;
    while i + chunk <= mono.len() {
        let waves_out = resampler
            .process(&[mono[i..i + chunk].to_vec()], None)
            .map_err(|e| anyhow::anyhow!("rubato process: {e}"))?;
        out.extend_from_slice(&waves_out[0]);
        i += chunk;
    }
    if i < mono.len() {
        // The last partial chunk, zero-padded to the fixed size and trimmed.
        let mut tail = vec![0.0f32; chunk];
        tail[..mono.len() - i].copy_from_slice(&mono[i..]);
        let waves_out = resampler
            .process(&[tail], None)
            .map_err(|e| anyhow::anyhow!("rubato process tail: {e}"))?;
        let valid = ((mono.len() - i) as f64 * ratio).ceil() as usize;
        out.extend(waves_out[0].iter().take(valid));
    }
    let expected = (mono.len() as f64 * ratio).round() as usize;
    Ok(out.into_iter().skip(delay).take(expected).collect())
}
pub const N_FFT: usize = 1600;
pub const HOP: usize = 800;
pub const DMEL_MIN: f32 = -7.0;
pub const DMEL_MAX: f32 = 2.0;
/// One frame per `HOP` samples: twenty a second.
pub const FRAMES_PER_SECOND: usize = SAMPLE_RATE / HOP;
const N_FREQS: usize = N_FFT / 2 + 1;

/// The front end, with its window, filterbank and FFT plan built once.
pub struct DmelFrontEnd {
    fft: Arc<dyn Fft<f32>>,
    window: Vec<f32>,
    /// `[DMEL_BINS][N_FREQS]`, row-major: one triangle per mel bin.
    filters: Vec<f32>,
    centers: [f32; DMEL_LEVELS],
}

impl Default for DmelFrontEnd {
    fn default() -> Self {
        Self::new()
    }
}

impl DmelFrontEnd {
    pub fn new() -> Self {
        let fft = FftPlanner::<f32>::new().plan_fft_forward(N_FFT);
        // Periodic Hann: `torch.hann_window(N, periodic=True)`, the division
        // is by N, not N - 1.
        let window = (0..N_FFT)
            .map(|n| 0.5 - 0.5 * (2.0 * std::f64::consts::PI * n as f64 / N_FFT as f64).cos())
            .map(|w| w as f32)
            .collect();
        let filters = mel_filter_bank();
        let mut centers = [0f32; DMEL_LEVELS];
        for (i, c) in centers.iter_mut().enumerate() {
            *c = DMEL_MIN + (DMEL_MAX - DMEL_MIN) * i as f32 / (DMEL_LEVELS - 1) as f32;
        }
        Self {
            fft,
            window,
            filters,
            centers,
        }
    }

    /// How many frames `samples` 16 kHz samples become: `ceil(len / HOP)`.
    pub fn frame_count(samples: usize) -> usize {
        samples.div_ceil(HOP)
    }

    /// Mono 16 kHz samples in, `frame_count(len) * DMEL_BINS` levels out, each
    /// in `0..DMEL_LEVELS`, frame-major. Empty input is zero frames.
    pub fn levels(&self, samples: &[f32]) -> Vec<u8> {
        let frames = Self::frame_count(samples.len());
        if frames == 0 {
            return Vec::new();
        }
        // Left pad `n_fft - hop`, right pad to a whole number of hops, then
        // an uncentred STFT: frame f reads padded[f * HOP .. f * HOP + N_FFT].
        let left = N_FFT - HOP;
        let mut padded = vec![0f32; left + frames * HOP];
        padded[left..left + samples.len()].copy_from_slice(samples);

        let mut buf = vec![Complex::new(0f32, 0f32); N_FFT];
        let mut mag = vec![0f32; N_FREQS];
        let mut out = Vec::with_capacity(frames * DMEL_BINS);
        for f in 0..frames {
            let start = f * HOP;
            for (i, b) in buf.iter_mut().enumerate() {
                *b = Complex::new(padded[start + i] * self.window[i], 0.0);
            }
            self.fft.process(&mut buf);
            // `magnitudes = stft.pow(2).sum(-1).clamp_min(1e-10).sqrt()`.
            for (m, c) in mag.iter_mut().zip(&buf[..N_FREQS]) {
                *m = (c.re * c.re + c.im * c.im).max(1e-10).sqrt();
            }
            for b in 0..DMEL_BINS {
                let row = &self.filters[b * N_FREQS..(b + 1) * N_FREQS];
                let mel: f32 = row.iter().zip(&mag).map(|(w, m)| w * m).sum();
                // `mel_spec.clamp_min(1e-10).log10()`, then the dMel clamp and
                // the nearest centre.
                let v = mel.max(1e-10).log10().clamp(DMEL_MIN, DMEL_MAX);
                let mut best = 0usize;
                let mut best_d = f32::INFINITY;
                for (i, c) in self.centers.iter().enumerate() {
                    let d = (v - c).abs();
                    if d < best_d {
                        best_d = d;
                        best = i;
                    }
                }
                out.push(best as u8);
            }
        }
        out
    }
}

// ── the Slaney filterbank, as `transformers.audio_utils.mel_filter_bank` ────

fn hz_to_mel(freq: f64) -> f64 {
    let min_log_hz = 1000.0;
    let f_sp = 200.0 / 3.0;
    let min_log_mel = min_log_hz / f_sp;
    let logstep = (6.4f64).ln() / 27.0;
    if freq >= min_log_hz {
        min_log_mel + (freq / min_log_hz).ln() / logstep
    } else {
        freq / f_sp
    }
}

fn mel_to_hz(mel: f64) -> f64 {
    let min_log_hz = 1000.0;
    let f_sp = 200.0 / 3.0;
    let min_log_mel = min_log_hz / f_sp;
    let logstep = (6.4f64).ln() / 27.0;
    if mel >= min_log_mel {
        min_log_hz * (logstep * (mel - min_log_mel)).exp()
    } else {
        f_sp * mel
    }
}

/// `[DMEL_BINS][N_FREQS]`: triangles between `DMEL_BINS + 2` Slaney-spaced
/// edges from 0 to 8 kHz, each scaled by `2 / (upper edge - lower edge)`.
fn mel_filter_bank() -> Vec<f32> {
    let f_max = SAMPLE_RATE as f64 / 2.0;
    let fft_freqs: Vec<f64> = (0..N_FREQS)
        .map(|i| f_max * i as f64 / (N_FREQS - 1) as f64)
        .collect();
    let m_lo = hz_to_mel(0.0);
    let m_hi = hz_to_mel(f_max);
    let edges: Vec<f64> = (0..DMEL_BINS + 2)
        .map(|i| mel_to_hz(m_lo + (m_hi - m_lo) * i as f64 / (DMEL_BINS + 1) as f64))
        .collect();
    let mut fb = vec![0f32; DMEL_BINS * N_FREQS];
    for b in 0..DMEL_BINS {
        let (lo, mid, hi) = (edges[b], edges[b + 1], edges[b + 2]);
        let enorm = 2.0 / (hi - lo);
        for (i, &f) in fft_freqs.iter().enumerate() {
            let down = (f - lo) / (mid - lo);
            let up = (hi - f) / (hi - mid);
            let w = down.min(up).max(0.0);
            fb[b * N_FREQS + i] = (w * enorm) as f32;
        }
    }
    fb
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Shape checks only, by the standing rule: no numerical twin. A tone
    /// must light the bins around its frequency and nothing far from it;
    /// silence must sit at the floor everywhere; a second of audio is twenty
    /// frames.
    #[test]
    fn a_tone_lights_its_bins_and_silence_sits_at_the_floor() {
        let fe = DmelFrontEnd::new();
        let n = SAMPLE_RATE;
        let tone: Vec<f32> = (0..n)
            .map(|i| 0.5 * (2.0 * std::f32::consts::PI * 1000.0 * i as f32 / n as f32).sin())
            .collect();
        let levels = fe.levels(&tone);
        assert_eq!(levels.len(), FRAMES_PER_SECOND * DMEL_BINS);
        // The bin whose centre is nearest 1 kHz, from the same edges.
        let m_lo = hz_to_mel(0.0);
        let m_hi = hz_to_mel(SAMPLE_RATE as f64 / 2.0);
        let centre = |b: usize| mel_to_hz(m_lo + (m_hi - m_lo) * (b + 1) as f64 / (DMEL_BINS + 1) as f64);
        let hot = (0..DMEL_BINS)
            .min_by(|&a, &b| (centre(a) - 1000.0).abs().partial_cmp(&(centre(b) - 1000.0).abs()).unwrap())
            .unwrap();
        // A steady frame from the middle of the second.
        let frame = &levels[10 * DMEL_BINS..11 * DMEL_BINS];
        let peak = *frame.iter().max().unwrap();
        assert!(frame[hot] >= peak.saturating_sub(1), "bin {hot} at {} of peak {peak}", frame[hot]);
        assert!(frame[DMEL_BINS - 1] < peak, "the top bin should be well under the peak");
        assert!(frame[hot] as usize > DMEL_LEVELS / 2, "a half-amplitude tone is not near the floor");

        // Silence is not level 0: the reference floors the POWER at 1e-10, so
        // the magnitude floor is 1e-5 and a filter's sum of it lands a level
        // or two above the bottom. Low and flat is the shape.
        let silence = vec![0f32; n];
        let quiet = fe.levels(&silence);
        assert!(quiet.iter().all(|&l| l <= 3), "silence sits within the bottom levels");
        assert!(
            frame[hot] >= quiet[hot] + 5,
            "the tone stands well above silence in its own bin: {} vs {}",
            frame[hot],
            quiet[hot]
        );

        assert!(fe.levels(&[]).is_empty());
        assert_eq!(fe.levels(&tone[..HOP + 1]).len(), 2 * DMEL_BINS);
    }
}
