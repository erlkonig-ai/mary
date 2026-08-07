//! Log-mel extraction (the vocos/F5 cond mel): torchaudio
//! `MelSpectrogram(sr 24000, n_fft 1024, hop 256, win 1024, n_mels 100, power=1,
//! center=True, norm=None, mel_scale="htk")` then `clamp(1e-5).log()`.
//!
//! The forward STFT is done as two `conv1d`s with windowed-DFT kernels (the dual
//! of vocos's ISTFT): re[f,t] = Σ_n pad[t·hop+n]·win[n]·cos(2πfn/N), likewise sin;
//! magnitude = √(re²+im²). Then an analytic HTK mel filterbank + log.

use burn::prelude::*;
use burn::tensor::module::conv1d;
use burn::tensor::ops::ConvOptions;

fn hz_to_mel(f: f64) -> f64 {
    2595.0 * (1.0 + f / 700.0).log10()
}
fn mel_to_hz(m: f64) -> f64 {
    700.0 * (10f64.powf(m / 2595.0) - 1.0)
}

pub struct MelExtractor<B: Backend> {
    kcos: Tensor<B, 3>, // [n_freq, 1, n_fft] window·cos DFT kernel
    ksin: Tensor<B, 3>, // [n_freq, 1, n_fft] window·sin DFT kernel
    fb: Tensor<B, 2>,   // [n_mels, n_freq] HTK mel filterbank
    n_fft: usize,
    hop: usize,
}

impl<B: Backend> MelExtractor<B> {
    pub fn new(device: &B::Device) -> Self {
        let (n_fft, hop, n_mels, sr) = (1024usize, 256usize, 100usize, 24000.0f64);
        let n_freq = n_fft / 2 + 1;
        let nf = n_fft as f64;

        // periodic Hann window: 0.5·(1 − cos(2πn/N))
        let win: Vec<f64> = (0..n_fft)
            .map(|n| 0.5 * (1.0 - (2.0 * std::f64::consts::PI * n as f64 / nf).cos()))
            .collect();

        // windowed-DFT kernels [n_freq, 1, n_fft]
        let mut cos = vec![0f32; n_freq * n_fft];
        let mut sin = vec![0f32; n_freq * n_fft];
        for f in 0..n_freq {
            for n in 0..n_fft {
                let theta = 2.0 * std::f64::consts::PI * f as f64 * n as f64 / nf;
                cos[f * n_fft + n] = (win[n] * theta.cos()) as f32;
                sin[f * n_fft + n] = (win[n] * theta.sin()) as f32;
            }
        }
        let kcos = Tensor::<B, 1>::from_floats(cos.as_slice(), device).reshape([n_freq, 1, n_fft]);
        let ksin = Tensor::<B, 1>::from_floats(sin.as_slice(), device).reshape([n_freq, 1, n_fft]);

        // HTK mel filterbank [n_mels, n_freq], norm=None
        let f_max = sr / 2.0;
        let all_freqs: Vec<f64> = (0..n_freq).map(|k| k as f64 * sr / nf).collect();
        let (m_min, m_max) = (hz_to_mel(0.0), hz_to_mel(f_max));
        let m_pts: Vec<f64> = (0..n_mels + 2)
            .map(|i| m_min + (m_max - m_min) * i as f64 / (n_mels + 1) as f64)
            .collect();
        let f_pts: Vec<f64> = m_pts.iter().map(|&m| mel_to_hz(m)).collect();
        let mut fb = vec![0f32; n_mels * n_freq];
        for m in 0..n_mels {
            let (lo, ct, up) = (f_pts[m], f_pts[m + 1], f_pts[m + 2]);
            for (k, &fk) in all_freqs.iter().enumerate() {
                let down = (fk - lo) / (ct - lo);
                let up_s = (up - fk) / (up - ct);
                let v = down.min(up_s).max(0.0);
                fb[m * n_freq + k] = v as f32;
            }
        }
        let fb = Tensor::<B, 1>::from_floats(fb.as_slice(), device).reshape([n_mels, n_freq]);

        Self {
            kcos,
            ksin,
            fb,
            n_fft,
            hop,
        }
    }

    /// wav: [1, n_samples] → log-mel [1, n_mels, n_frames] (center padding).
    pub fn forward(&self, wav: Tensor<B, 2>) -> Tensor<B, 3> {
        let [b, l] = wav.dims();
        let p = self.n_fft / 2;
        // reflect pad p each side (mirror without repeating the edge sample)
        let left = wav.clone().slice([0..b, 1..1 + p]).flip([1]);
        let right = wav.clone().slice([0..b, l - 1 - p..l - 1]).flip([1]);
        let padded = Tensor::cat(vec![left, wav, right], 1).reshape([b, 1, l + 2 * p]);

        let opts = ConvOptions::new([self.hop], [0], [1], 1);
        let re = conv1d(padded.clone(), self.kcos.clone(), None, opts.clone()); // [1,n_freq,T]
        let im = conv1d(padded, self.ksin.clone(), None, opts);
        let mag = (re.powf_scalar(2.0) + im.powf_scalar(2.0)).sqrt(); // [1,n_freq,T]

        // mel: [n_mels,n_freq] @ [1,n_freq,T] → [1,n_mels,T]
        let [_, nfreq, t] = mag.dims();
        let fb = self.fb.clone().unsqueeze_dim::<3>(0); // [1,n_mels,n_freq]
        let mel = fb.matmul(mag.reshape([b, nfreq, t])); // [1,n_mels,T]
        mel.clamp_min(1e-5).log()
    }
}
