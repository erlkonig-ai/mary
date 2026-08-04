//! Voxtral log-mel front end: torch.stft(n_fft=400, hop=160, hann-periodic,
//! center=true → reflect pad 200) with the LAST frame dropped, slaney
//! filterbank (128 mels, 0–8 kHz, slaney norm), power spectrum,
//! `log10(clamp(·,1e-10))`, floor at `global_log_mel_max − 8` (fixed 1.5 —
//! a per-clip max would break streaming), then `(x+4)/4`.
//!
//! STFT as conv1d with windowed cos/sin kernels (the house pattern from the
//! qwen3tts speaker mel) — no fft dependency, runs on the model backend.

use burn::prelude::*;
use burn::tensor::module::conv1d;
use burn::tensor::ops::ConvOptions;

use super::config::*;

fn hz_to_mel_slaney(f: f64) -> f64 {
    let f_sp = 200.0 / 3.0;
    if f < 1000.0 {
        f / f_sp
    } else {
        1000.0 / f_sp + (f / 1000.0).ln() / (6.4f64.ln() / 27.0)
    }
}

fn mel_to_hz_slaney(m: f64) -> f64 {
    let f_sp = 200.0 / 3.0;
    let min_log_mel = 1000.0 / f_sp;
    if m < min_log_mel {
        m * f_sp
    } else {
        1000.0 * ((m - min_log_mel) * (6.4f64.ln() / 27.0)).exp()
    }
}

/// torch-style reflect padding, applied on the HOST sample buffer — identical
/// values to the tensor-side narrow/flip/cat (pure data movement, no
/// arithmetic), keeps flip off the GPU graph (burn 0.21's fusion codegen
/// panics on the f16 backend's batch-mel graph), and is cheaper.
fn reflect_pad_host(samples: &[f32], p: usize) -> Vec<f32> {
    let n = samples.len();
    let mut v = Vec::with_capacity(n + 2 * p);
    v.extend((0..p).map(|i| samples[p - i]));
    v.extend_from_slice(samples);
    v.extend((0..p).map(|i| samples[n - 2 - i]));
    v
}

pub struct VoxtralMel<B: Backend> {
    kcos: Tensor<B, 3>, // [n_freq, 1, n_fft]
    ksin: Tensor<B, 3>,
    fb: Tensor<B, 2>, // [n_mels, n_freq]
}

impl<B: Backend> VoxtralMel<B> {
    pub fn new(device: &B::Device) -> Self {
        let n_freq = N_FFT / 2 + 1;
        let nf = N_FFT as f64;

        // hann periodic (torch.hann_window default)
        let win: Vec<f64> = (0..N_FFT)
            .map(|n| 0.5 * (1.0 - (2.0 * std::f64::consts::PI * n as f64 / nf).cos()))
            .collect();
        let mut cos = vec![0f32; n_freq * N_FFT];
        let mut sin = vec![0f32; n_freq * N_FFT];
        for f in 0..n_freq {
            for n in 0..N_FFT {
                let theta = 2.0 * std::f64::consts::PI * f as f64 * n as f64 / nf;
                cos[f * N_FFT + n] = (win[n] * theta.cos()) as f32;
                sin[f * N_FFT + n] = (win[n] * theta.sin()) as f32;
            }
        }
        let kcos = Tensor::<B, 1>::from_floats(cos.as_slice(), device).reshape([n_freq, 1, N_FFT]);
        let ksin = Tensor::<B, 1>::from_floats(sin.as_slice(), device).reshape([n_freq, 1, N_FFT]);

        // slaney filterbank, slaney norm (transformers mel_filter_bank defaults)
        let all_freqs: Vec<f64> = (0..n_freq)
            .map(|k| k as f64 * SAMPLE_RATE as f64 / nf)
            .collect();
        let (m_min, m_max) = (hz_to_mel_slaney(0.0), hz_to_mel_slaney(FMAX));
        let f_pts: Vec<f64> = (0..MEL_BINS + 2)
            .map(|i| mel_to_hz_slaney(m_min + (m_max - m_min) * i as f64 / (MEL_BINS + 1) as f64))
            .collect();
        let mut fb = vec![0f32; MEL_BINS * n_freq];
        for m in 0..MEL_BINS {
            let (fl, fc, fr) = (f_pts[m], f_pts[m + 1], f_pts[m + 2]);
            let enorm = 2.0 / (fr - fl);
            for (k, &f) in all_freqs.iter().enumerate() {
                let up = (f - fl) / (fc - fl);
                let down = (fr - f) / (fr - fc);
                let w = up.min(down).max(0.0);
                fb[m * n_freq + k] = (w * enorm) as f32;
            }
        }
        let fb = Tensor::<B, 1>::from_floats(fb.as_slice(), device).reshape([MEL_BINS, n_freq]);

        Self { kcos, ksin, fb }
    }

    /// samples (16 kHz, [−1,1]) → log-mel `[1, 128, T]` with T = len/hop when
    /// `center` (batch/first chunk; the +1th torch frame is dropped) —
    /// `center=false` is the later-streaming-chunk variant.
    pub fn forward(&self, samples: &[f32], center: bool, device: &B::Device) -> Tensor<B, 3> {
        let padded;
        let samples = if center {
            padded = reflect_pad_host(samples, N_FFT / 2);
            &padded[..]
        } else {
            samples
        };
        let x = Tensor::<B, 1>::from_floats(samples, device).reshape([1, 1, samples.len()]);
        let opts = ConvOptions::new([HOP], [0], [1], 1);
        let re = conv1d(x.clone(), self.kcos.clone(), None, opts.clone());
        let im = conv1d(x, self.ksin.clone(), None, opts);
        let power = re.clone().powf_scalar(2.0) + im.powf_scalar(2.0); // [1, n_freq, T(+1)]
        let t = power.dims()[2] - if center { 1 } else { 0 }; // torch drops stft[..., :-1]
        let power = power.narrow(2, 0, t);
        let [_, nf, _] = power.dims();
        let mel = self.fb.clone().reshape([1, MEL_BINS, nf]).matmul(power);
        let log = mel
            .clamp_min(1e-10)
            .log()
            .div_scalar(std::f64::consts::LN_10);
        let log = log.clamp_min(GLOBAL_LOG_MEL_MAX - 8.0);
        log.add_scalar(4.0).div_scalar(4.0)
    }
}
