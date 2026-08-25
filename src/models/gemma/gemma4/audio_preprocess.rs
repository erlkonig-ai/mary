//! Gemma 4 audio feature extractor (log-mel spectrogram).
//!
//! Byte-exact port of `transformers.models.gemma4.feature_extraction_gemma4`
//! for the default Gemma 4 config: 16kHz mono, 20ms frames, 10ms hop, 512-pt
//! FFT, 128 HTK-mel filters, preemphasis=0, mel_floor=1e-3. No per-bin
//! normalization. Produces the same `input_features` + `input_features_mask`
//! numpy arrays Python does.

use rustfft::{FftPlanner, num_complex::Complex32};

/// Fixed Gemma 4 audio feature extractor parameters.
#[derive(Debug, Clone)]
pub struct AudioFeatureExtractor {
    pub sampling_rate: usize,      // 16000
    pub frame_length: usize,       // 320 (20ms)
    pub hop_length: usize,         // 160 (10ms)
    pub fft_length: usize,         // 512
    pub feature_size: usize,       // 128
    pub min_frequency: f32,        // 0.0
    pub max_frequency: f32,        // 8000.0
    pub mel_floor: f32,            // 1e-3
    pub max_length: usize,         // 480000 (30s)
    pub pad_to_multiple_of: usize, // 128
    /// Periodic Hann window, length = frame_length.
    pub window: Vec<f32>,
    /// Triangular HTK mel filterbank: shape [num_freq_bins=257, num_mel=128], row-major.
    pub mel_filters: Vec<f32>,
    pub num_freq_bins: usize,
}

impl Default for AudioFeatureExtractor {
    fn default() -> Self {
        Self::new()
    }
}

impl AudioFeatureExtractor {
    /// Build with Gemma 4 defaults (E2B/E4B).
    pub fn new() -> Self {
        let sampling_rate = 16000usize;
        let frame_length_ms = 20.0f64;
        let hop_length_ms = 10.0f64;
        let frame_length = (sampling_rate as f64 * frame_length_ms / 1000.0).round() as usize;
        let hop_length = (sampling_rate as f64 * hop_length_ms / 1000.0).round() as usize;
        let fft_length = (frame_length as f64).log2().ceil() as u32;
        let fft_length = 1usize << fft_length;
        let feature_size = 128usize;
        let min_frequency = 0.0f32;
        let max_frequency = 8000.0f32;
        let num_freq_bins = fft_length / 2 + 1;

        let window = hann_periodic(frame_length);
        let mel_filters = mel_filter_bank_htk(
            num_freq_bins,
            feature_size,
            min_frequency,
            max_frequency,
            sampling_rate,
        );

        Self {
            sampling_rate,
            frame_length,
            hop_length,
            fft_length,
            feature_size,
            min_frequency,
            max_frequency,
            mel_floor: 1e-3,
            max_length: 480_000,
            pad_to_multiple_of: 128,
            window,
            mel_filters,
            num_freq_bins,
        }
    }

    /// Extract log-mel features from a raw 16kHz mono waveform.
    ///
    /// Returns (features, mask) where `features` has shape [T, feature_size]
    /// (flattened row-major), `mask` has length T (true for valid frames).
    /// Output length matches what Python produces: semicausal padding shifts
    /// the frame grid so the first STFT frame centers at t=0.
    pub fn extract(&self, waveform: &[f32]) -> (Vec<f32>, Vec<bool>, usize) {
        // Truncate to max_length
        let mut wv: Vec<f32> = waveform.iter().take(self.max_length).copied().collect();
        // Sample-level attention mask: 1 where real audio, 0 where padded.
        let real_len = wv.len();
        let target_len = ((real_len + self.pad_to_multiple_of - 1) / self.pad_to_multiple_of)
            * self.pad_to_multiple_of;
        wv.resize(target_len, 0.0);
        let mut mask_samples = vec![true; target_len];
        for i in real_len..target_len {
            mask_samples[i] = false;
        }

        // Semicausal pad: prepend frame_length // 2 zeros.
        let pad_left = self.frame_length / 2;
        let mut padded = vec![0.0f32; pad_left + target_len];
        padded[pad_left..].copy_from_slice(&wv);
        let mut pad_mask = vec![false; pad_left + target_len];
        pad_mask[pad_left..].copy_from_slice(&mask_samples);

        // Unfold into frames of size frame_length + 1, step hop_length.
        let unfold_size = self.frame_length + 1;
        let total = padded.len();
        let num_frames = if total >= unfold_size {
            (total - unfold_size) / self.hop_length + 1
        } else {
            0
        };

        // Preemphasis is 0 → we take the first `frame_length` samples of each
        // unfolded window (Python: frames_to_process[..., :-1]).
        let mut out = vec![0.0f32; num_frames * self.feature_size];
        let mut mask = vec![false; num_frames];

        let mut planner = FftPlanner::<f32>::new();
        let fft = planner.plan_fft_forward(self.fft_length);
        let mut buf: Vec<Complex32> = vec![Complex32::new(0.0, 0.0); self.fft_length];
        let mut mag = vec![0.0f32; self.num_freq_bins];

        for f in 0..num_frames {
            let start = f * self.hop_length;
            // Populate buf with windowed frame, zero-pad to fft_length.
            for i in 0..self.fft_length {
                buf[i] = Complex32::new(0.0, 0.0);
            }
            for i in 0..self.frame_length {
                buf[i] = Complex32::new(padded[start + i] * self.window[i], 0.0);
            }
            fft.process(&mut buf);

            // |STFT| for the first num_freq_bins = fft_length/2 + 1 outputs.
            for k in 0..self.num_freq_bins {
                let c = buf[k];
                mag[k] = (c.re * c.re + c.im * c.im).sqrt();
            }

            // mel = magnitude (1 x num_freq_bins) @ mel_filters (num_freq_bins x feature_size)
            let row_out = f * self.feature_size;
            for m in 0..self.feature_size {
                let mut acc = 0.0f32;
                for k in 0..self.num_freq_bins {
                    acc += mag[k] * self.mel_filters[k * self.feature_size + m];
                }
                out[row_out + m] = (acc + self.mel_floor).ln();
            }

            // Frame mask: valid iff the last sample of this frame's window is real audio.
            let end_idx = start + unfold_size - 1;
            mask[f] = pad_mask.get(end_idx).copied().unwrap_or(false);
        }

        // Python: prepared_speech = prepared_speech * mask[..., None] — zero-out
        // padded-frame rows so downstream sees exactly what Python saves.
        for f in 0..num_frames {
            if !mask[f] {
                let row_out = f * self.feature_size;
                for m in 0..self.feature_size {
                    out[row_out + m] = 0.0;
                }
            }
        }

        (out, mask, num_frames)
    }
}

/// Periodic Hann window: length-N window sampled from `0.5 - 0.5 cos(2π·n/N)`
/// for n = 0..N. Matches `transformers.audio_utils.window_function(..,
/// periodic=True)` which uses `np.hanning(N+1)[:-1]`.
fn hann_periodic(n: usize) -> Vec<f32> {
    let np = (n + 1) as f64;
    // np.hanning(M) returns 0.5 - 0.5 cos(2π i / (M-1)) for i in 0..M.
    (0..n)
        .map(|i| {
            let v = 0.5 - 0.5 * (2.0 * std::f64::consts::PI * i as f64 / (np - 1.0)).cos();
            v as f32
        })
        .collect()
}

fn hz_to_mel_htk(f: f32) -> f32 {
    2595.0 * (1.0 + f / 700.0).log10()
}
fn mel_to_hz_htk(m: f32) -> f32 {
    700.0 * (10f32.powf(m / 2595.0) - 1.0)
}

/// HTK mel filterbank, no normalization. Matches
/// `transformers.audio_utils.mel_filter_bank(norm=None, mel_scale="htk")`.
fn mel_filter_bank_htk(
    num_freq_bins: usize,
    num_mel: usize,
    min_hz: f32,
    max_hz: f32,
    sampling_rate: usize,
) -> Vec<f32> {
    let mel_min = hz_to_mel_htk(min_hz);
    let mel_max = hz_to_mel_htk(max_hz);
    let n_pts = num_mel + 2;
    // linspace(mel_min, mel_max, n_pts)
    let mut mel_pts = vec![0.0f32; n_pts];
    for i in 0..n_pts {
        mel_pts[i] = mel_min + (mel_max - mel_min) * (i as f32) / ((n_pts - 1) as f32);
    }
    // mel_to_hz_htk vectorized
    let filter_freqs: Vec<f32> = mel_pts.iter().map(|&m| mel_to_hz_htk(m)).collect();

    // fft_freqs = linspace(0, sampling_rate/2, num_freq_bins)
    let nyq = (sampling_rate / 2) as f32;
    let fft_freqs: Vec<f32> = (0..num_freq_bins)
        .map(|k| nyq * (k as f32) / ((num_freq_bins - 1) as f32))
        .collect();

    // filter_diff[i] = filter_freqs[i+1] - filter_freqs[i], length n_pts - 1
    let filter_diff: Vec<f32> = filter_freqs.windows(2).map(|w| w[1] - w[0]).collect();

    // slopes[k, i] = filter_freqs[i] - fft_freqs[k] for k in 0..K, i in 0..n_pts
    // down_slopes[k, i] = -slopes[k, i] / filter_diff[i]           for i in 0..n_pts-2
    // up_slopes[k, i]   =  slopes[k, i+2] / filter_diff[i+1]       for i in 0..n_pts-2
    // weight = max(0, min(down_slope, up_slope))
    let mut out = vec![0.0f32; num_freq_bins * num_mel];
    for k in 0..num_freq_bins {
        for i in 0..num_mel {
            // Python: slopes[:, :-2] uses indices 0..n_pts-2 → filter_freqs[0..n_pts-2]
            //         and filter_diff[:-1] → filter_diff[0..n_pts-3]
            // down_slope: -(filter_freqs[i] - fft_freqs[k]) / filter_diff[i]
            // up_slope  :  (filter_freqs[i+2] - fft_freqs[k]) / filter_diff[i+1]
            let down = -(filter_freqs[i] - fft_freqs[k]) / filter_diff[i];
            let up = (filter_freqs[i + 2] - fft_freqs[k]) / filter_diff[i + 1];
            let w = down.min(up).max(0.0);
            out[k * num_mel + i] = w;
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hann_len_and_first_zero() {
        let w = hann_periodic(320);
        assert_eq!(w.len(), 320);
        assert!((w[0] - 0.0).abs() < 1e-6);
        // Periodic Hann peaks at center: w[N/2] ≈ 1.0 only when N is even-odd
        // specific; just sanity-check it's roughly bell-shaped.
        let mid = w[w.len() / 2];
        assert!(mid > 0.9, "mid={mid}");
    }

    #[test]
    fn mel_bank_shape() {
        let m = mel_filter_bank_htk(257, 128, 0.0, 8000.0, 16000);
        assert_eq!(m.len(), 257 * 128);
    }
}
