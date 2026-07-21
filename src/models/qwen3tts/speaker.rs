//! ECAPA-TDNN speaker encoder (x-vector, 2048-dim) + the slaney-mel front end
//! it consumes. Reference: `Qwen3TTSSpeakerEncoder` + `mel_spectrogram` in
//! `modeling_qwen3_tts.py`.
//!
//! Mel: n_fft 1024, hop 256, win 1024 (periodic Hann), 128 mels, fmin 0,
//! fmax 12000, sr 24000, **slaney scale + slaney norm** (librosa defaults),
//! center=False with (n_fft−hop)/2 reflect pre-pad, `log(clamp(·,1e-5))` of
//! `sqrt(re²+im²+1e-9)`. STFT as two windowed-DFT conv1ds (the f5 trick).
//!
//! ECAPA: TDNN(128→512,k5) → 3× SE-Res2Net(512, k3, dil 2/3/4, scale 8,
//! se 128) → cat(3×512) → MFA TDNN(1536,k1) → attentive-stats pooling
//! (att 128) → cat(mean,std)=3072 → Conv1d k1 → 2048. All convs
//! `padding="same"` **reflect**, ReLU activations.

use burn::prelude::*;
use burn::tensor::activation::{relu, sigmoid, softmax, tanh};
use burn::tensor::module::conv1d;
use burn::tensor::ops::ConvOptions;

use crate::nn::weight_loader::WeightLoader;

/// Reflect-pad the last dim of `[B, C, L]` by (left, right).
fn reflect_pad<B: Backend>(x: Tensor<B, 3>, left: usize, right: usize) -> Tensor<B, 3> {
    let l = x.dims()[2];
    let mut parts = Vec::new();
    if left > 0 {
        parts.push(x.clone().narrow(2, 1, left).flip([2]));
    }
    parts.push(x.clone());
    if right > 0 {
        parts.push(x.narrow(2, l - 1 - right, right).flip([2]));
    }
    Tensor::cat(parts, 2)
}

/// Conv1d with "same" reflect padding (odd kernel, stride 1).
struct SameConv<B: Backend> {
    w: Tensor<B, 3>, // [out, in/groups, k]
    b: Tensor<B, 1>,
    dilation: usize,
}

impl<B: Backend> SameConv<B> {
    fn load(loader: &WeightLoader, prefix: &str, dilation: usize, device: &B::Device) -> Self {
        Self {
            w: loader.load_tensor(&format!("{prefix}.weight"), device),
            b: loader.load_tensor(&format!("{prefix}.bias"), device),
            dilation,
        }
    }

    fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        let k = self.w.dims()[2];
        let total = self.dilation * (k - 1);
        let x = if total > 0 {
            reflect_pad(x, total / 2, total - total / 2)
        } else {
            x
        };
        conv1d(
            x,
            self.w.clone(),
            Some(self.b.clone()),
            ConvOptions::new([1], [0], [self.dilation], 1),
        )
    }
}

/// TDNN block = same-conv + ReLU.
struct Tdnn<B: Backend>(SameConv<B>);

impl<B: Backend> Tdnn<B> {
    fn load(loader: &WeightLoader, prefix: &str, dilation: usize, device: &B::Device) -> Self {
        Self(SameConv::load(&loader, &format!("{prefix}.conv"), dilation, device))
    }
    fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        relu(self.0.forward(x))
    }
}

/// SE-Res2Net block (in ECAPA): tdnn1 → res2net → tdnn2 → SE, + residual.
struct SeRes2Net<B: Backend> {
    tdnn1: Tdnn<B>,
    res2: Vec<Tdnn<B>>, // scale-1 blocks over channel chunks
    tdnn2: Tdnn<B>,
    se1: SameConv<B>,
    se2: SameConv<B>,
    scale: usize,
}

impl<B: Backend> SeRes2Net<B> {
    fn load(loader: &WeightLoader, prefix: &str, dilation: usize, device: &B::Device) -> Self {
        Self {
            tdnn1: Tdnn::load(loader, &format!("{prefix}.tdnn1"), 1, device),
            res2: (0..7)
                .map(|i| Tdnn::load(loader, &format!("{prefix}.res2net_block.blocks.{i}"), dilation, device))
                .collect(),
            tdnn2: Tdnn::load(loader, &format!("{prefix}.tdnn2"), 1, device),
            se1: SameConv::load(loader, &format!("{prefix}.se_block.conv1"), 1, device),
            se2: SameConv::load(loader, &format!("{prefix}.se_block.conv2"), 1, device),
            scale: 8,
        }
    }

    fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        let residual = x.clone();
        let h = self.tdnn1.forward(x);
        // res2net: chunk channels; block i>0 processes (chunk_i + prev_out)
        let [b, c, l] = h.dims();
        let cs = c / self.scale;
        let mut outs: Vec<Tensor<B, 3>> = Vec::with_capacity(self.scale);
        for i in 0..self.scale {
            let part = h.clone().narrow(1, i * cs, cs);
            let out = match i {
                0 => part,
                1 => self.res2[0].forward(part),
                _ => self.res2[i - 1].forward(part + outs[i - 1].clone()),
            };
            outs.push(out);
        }
        let h = Tensor::cat(outs, 1).reshape([b, c, l]);
        let h = self.tdnn2.forward(h);
        // squeeze-excitation on the time-mean
        let s = h.clone().mean_dim(2); // [B, C, 1]
        let s = sigmoid(self.se2.forward(relu(self.se1.forward(s))));
        h.mul(s) + residual
    }
}

pub struct SpeakerEncoder<B: Backend> {
    block0: Tdnn<B>,
    blocks: Vec<SeRes2Net<B>>,
    mfa: Tdnn<B>,
    asp_tdnn: Tdnn<B>,
    asp_conv: SameConv<B>,
    fc: SameConv<B>,
}

impl<B: Backend> SpeakerEncoder<B> {
    pub fn load(loader: &WeightLoader, device: &B::Device) -> Self {
        let p = "speaker_encoder";
        Self {
            block0: Tdnn::load(loader, &format!("{p}.blocks.0"), 1, device),
            blocks: (1..4)
                .map(|i| SeRes2Net::load(loader, &format!("{p}.blocks.{i}"), i + 1, device))
                .collect(),
            mfa: Tdnn::load(loader, &format!("{p}.mfa"), 1, device),
            asp_tdnn: Tdnn::load(loader, &format!("{p}.asp.tdnn"), 1, device),
            asp_conv: SameConv::load(loader, &format!("{p}.asp.conv"), 1, device),
            fc: SameConv::load(loader, &format!("{p}.fc"), 1, device),
        }
    }

    /// mel `[1, T, 128]` → x-vector `[enc_dim]` (1.7B: 2048, 0.6B: 1024).
    pub fn forward(&self, mel: Tensor<B, 3>) -> Tensor<B, 1> {
        let x = mel.swap_dims(1, 2); // [1, 128, T]
        let mut hs = Vec::new();
        let mut h = self.block0.forward(x);
        hs.push(h.clone());
        for blk in &self.blocks {
            h = blk.forward(h);
            hs.push(h.clone());
        }
        // MFA over blocks 1.. (skip the initial TDNN, per reference `[1:]`)
        let h = self.mfa.forward(Tensor::cat(hs[1..].to_vec(), 1));

        // attentive statistics pooling (full-length mask ⇒ uniform weights)
        let [b, c, l] = h.dims();
        let mean = h.clone().mean_dim(2); // [B,C,1]
        let var = (h.clone() - mean.clone().expand([b, c, l]))
            .powf_scalar(2.0)
            .mean_dim(2);
        let std = var.clamp_min(1e-12).sqrt();
        let attn_in = Tensor::cat(
            vec![h.clone(), mean.expand([b, c, l]), std.expand([b, c, l])],
            1,
        );
        let a = self.asp_conv.forward(tanh(self.asp_tdnn.forward(attn_in)));
        let a = softmax(a, 2); // [B,C,L]
        let mean = h.clone().mul(a.clone()).sum_dim(2); // [B,C,1]
        let var = (h - mean.clone().expand([b, c, l]))
            .powf_scalar(2.0)
            .mul(a)
            .sum_dim(2);
        let std = var.clamp_min(1e-12).sqrt();
        let pooled = Tensor::cat(vec![mean, std], 1); // [B,2C,1]

        let out = self.fc.forward(pooled); // [B, enc_dim, 1]
        let d = out.dims()[1];
        out.reshape([d])
    }
}

// ---------------------------------------------------------------------------
// slaney mel front end
// ---------------------------------------------------------------------------

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

pub struct SpeakerMel<B: Backend> {
    kcos: Tensor<B, 3>,
    ksin: Tensor<B, 3>,
    fb: Tensor<B, 2>, // [n_mels, n_freq]
    n_fft: usize,
    hop: usize,
}

impl<B: Backend> SpeakerMel<B> {
    pub fn new(device: &B::Device) -> Self {
        let (n_fft, hop, n_mels) = (1024usize, 256usize, 128usize);
        let (sr, fmax) = (24000.0f64, 12000.0f64);
        let n_freq = n_fft / 2 + 1;
        let nf = n_fft as f64;

        let win: Vec<f64> = (0..n_fft)
            .map(|n| 0.5 * (1.0 - (2.0 * std::f64::consts::PI * n as f64 / nf).cos()))
            .collect();
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

        // slaney filterbank with slaney norm (librosa defaults)
        let all_freqs: Vec<f64> = (0..n_freq).map(|k| k as f64 * sr / nf).collect();
        let (m_min, m_max) = (hz_to_mel_slaney(0.0), hz_to_mel_slaney(fmax));
        let f_pts: Vec<f64> = (0..n_mels + 2)
            .map(|i| mel_to_hz_slaney(m_min + (m_max - m_min) * i as f64 / (n_mels + 1) as f64))
            .collect();
        let mut fb = vec![0f32; n_mels * n_freq];
        for m in 0..n_mels {
            let (fl, fc, fr) = (f_pts[m], f_pts[m + 1], f_pts[m + 2]);
            let enorm = 2.0 / (fr - fl);
            for (k, &f) in all_freqs.iter().enumerate() {
                let up = (f - fl) / (fc - fl);
                let down = (fr - f) / (fr - fc);
                let w = up.min(down).max(0.0);
                fb[m * n_freq + k] = (w * enorm) as f32;
            }
        }
        let fb = Tensor::<B, 1>::from_floats(fb.as_slice(), device).reshape([n_mels, n_freq]);

        Self { kcos, ksin, fb, n_fft, hop }
    }

    /// samples (24 kHz, [−1,1]) → log-mel `[1, T, 128]`.
    pub fn forward(&self, samples: &[f32], device: &B::Device) -> Tensor<B, 3> {
        let pad = (self.n_fft - self.hop) / 2;
        let x = Tensor::<B, 1>::from_floats(samples, device).reshape([1, 1, samples.len()]);
        let x = reflect_pad(x, pad, pad);
        let opts = ConvOptions::new([self.hop], [0], [1], 1);
        let re = conv1d(x.clone(), self.kcos.clone(), None, opts.clone());
        let im = conv1d(x, self.ksin.clone(), None, opts);
        let mag = (re.clone().powf_scalar(2.0) + im.powf_scalar(2.0))
            .add_scalar(1e-9)
            .sqrt(); // [1, n_freq, T]
        let [_, nf, t] = mag.dims();
        let n_mels = self.fb.dims()[0];
        let mel = self
            .fb
            .clone()
            .reshape([1, n_mels, nf])
            .matmul(mag);
        mel.clamp_min(1e-5).log().reshape([1, n_mels, t]).swap_dims(1, 2)
    }
}
