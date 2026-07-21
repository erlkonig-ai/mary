//! Vocos vocoder: mel → 24 kHz waveform. VocosBackbone (Conv1d embed + 8
//! ConvNeXt-V1 blocks with LayerScale γ) + ISTFTHead (Linear → log-mag/phase →
//! irfft + windowed overlap-add). The one genuinely-new component of the F5
//! stack — the DiT path is reused everywhere else.
//!
//! `torch.istft(center=True)` is reimplemented as: irfft via a real DFT matmul
//! (Hermitian-folded, norm="backward"), synthesis-window multiply, overlap-add
//! through `conv_transpose1d` with an identity kernel, divide by the window²
//! overlap-add envelope, then trim n_fft/2 from each end.
//!
//! Weights: weights/vocos.safetensors (exported by scripts/probe_vocos.py).

use super::dit::Linear;
use crate::nn::weight_loader::WeightLoader;
use burn::prelude::*;
use burn::tensor::activation::gelu;
use burn::tensor::module::{conv1d, conv_transpose1d};
use burn::tensor::ops::{ConvOptions, ConvTransposeOptions};

/// Affine LayerNorm over the last dim.
fn layer_norm<B: Backend>(x: Tensor<B, 3>, w: &Tensor<B, 1>, b: &Tensor<B, 1>, eps: f64) -> Tensor<B, 3> {
    let c = x.dims()[2];
    let mean = x.clone().mean_dim(2);
    let xc = x - mean;
    let var = xc.clone().powf_scalar(2.0).mean_dim(2);
    xc / (var + eps).sqrt() * w.clone().reshape([1, 1, c]) + b.clone().reshape([1, 1, c])
}

/// One ConvNeXt-V1 block (channels-first I/O [B,C,T]): depthwise conv → LN →
/// pw↑ → GELU → pw↓ → γ·x, residual.
struct ConvNeXt<B: Backend> {
    dw_w: Tensor<B, 3>,
    dw_b: Tensor<B, 1>,
    n_w: Tensor<B, 1>,
    n_b: Tensor<B, 1>,
    pw1: Linear<B>,
    gamma: Tensor<B, 1>,
    pw2: Linear<B>,
    ch: usize,
}

impl<B: Backend> ConvNeXt<B> {
    fn load(loader: &WeightLoader, prefix: &str, ch: usize, device: &B::Device) -> Self {
        Self {
            dw_w: loader.load_tensor(&format!("{prefix}.dwconv.weight"), device),
            dw_b: loader.load_tensor(&format!("{prefix}.dwconv.bias"), device),
            n_w: loader.load_tensor(&format!("{prefix}.norm.weight"), device),
            n_b: loader.load_tensor(&format!("{prefix}.norm.bias"), device),
            pw1: Linear::load(loader, &format!("{prefix}.pwconv1"), true, device),
            gamma: loader.load_tensor(&format!("{prefix}.gamma"), device),
            pw2: Linear::load(loader, &format!("{prefix}.pwconv2"), true, device),
            ch,
        }
    }

    fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        let res = x.clone();
        let h = conv1d(x, self.dw_w.clone(), Some(self.dw_b.clone()), ConvOptions::new([1], [3], [1], self.ch));
        let h = h.swap_dims(1, 2); // [B,T,C]
        let h = layer_norm(h, &self.n_w, &self.n_b, 1e-6);
        let h = gelu(self.pw1.forward(h));
        let h = self.pw2.forward(h);
        let h = h * self.gamma.clone().reshape([1, 1, self.ch]);
        res + h.swap_dims(1, 2)
    }
}

pub struct Vocos<B: Backend> {
    embed_w: Tensor<B, 3>,
    embed_b: Tensor<B, 1>,
    norm_w: Tensor<B, 1>,
    norm_b: Tensor<B, 1>,
    blocks: Vec<ConvNeXt<B>>,
    fln_w: Tensor<B, 1>,
    fln_b: Tensor<B, 1>,
    out: Linear<B>,
    cmat: Tensor<B, 2>,   // [n_freq, n_fft] real DFT (Hermitian-folded)
    smat: Tensor<B, 2>,   // [n_freq, n_fft] imag DFT
    window: Tensor<B, 1>, // [n_fft]
    eye: Tensor<B, 3>,    // [n_fft, 1, n_fft] identity kernel for overlap-add
    n_fft: usize,
    hop: usize,
}

impl<B: Backend> Vocos<B> {
    pub fn load(loader: &WeightLoader, device: &B::Device) -> Self {
        let (n_fft, hop) = (1024usize, 256usize);
        let n_freq = n_fft / 2 + 1;
        let blocks = (0..8)
            .map(|i| ConvNeXt::load(loader, &format!("backbone.convnext.{i}"), 512, device))
            .collect();

        // real DFT matrices for irfft (norm="backward"): x[n] = Σ_k w_k/N
        // (Re·cos − Im·sin); w_0 = w_{N/2} = 1, else 2. sin=0 at DC/Nyquist so
        // their imaginary parts drop out automatically (matches torch.fft.irfft).
        let n = n_fft as f64;
        let mut cm = vec![0f32; n_freq * n_fft];
        let mut sm = vec![0f32; n_freq * n_fft];
        for k in 0..n_freq {
            let wk = if k == 0 || k == n_freq - 1 { 1.0 } else { 2.0 } / n;
            for t in 0..n_fft {
                let theta = 2.0 * std::f64::consts::PI * k as f64 * t as f64 / n;
                cm[k * n_fft + t] = (wk * theta.cos()) as f32;
                sm[k * n_fft + t] = (wk * theta.sin()) as f32;
            }
        }
        let cmat = Tensor::<B, 1>::from_floats(cm.as_slice(), device).reshape([n_freq, n_fft]);
        let smat = Tensor::<B, 1>::from_floats(sm.as_slice(), device).reshape([n_freq, n_fft]);

        // identity kernel [n_fft, 1, n_fft] (eye[c,0,c]=1) → overlap-add via conv_transpose1d
        let mut e = vec![0f32; n_fft * n_fft];
        for c in 0..n_fft {
            e[c * n_fft + c] = 1.0;
        }
        let eye = Tensor::<B, 1>::from_floats(e.as_slice(), device).reshape([n_fft, 1, n_fft]);

        Self {
            embed_w: loader.load_tensor("backbone.embed.weight", device),
            embed_b: loader.load_tensor("backbone.embed.bias", device),
            norm_w: loader.load_tensor("backbone.norm.weight", device),
            norm_b: loader.load_tensor("backbone.norm.bias", device),
            blocks,
            fln_w: loader.load_tensor("backbone.final_layer_norm.weight", device),
            fln_b: loader.load_tensor("backbone.final_layer_norm.bias", device),
            out: Linear::load(loader, "head.out", true, device),
            window: loader.load_tensor("head.istft.window", device),
            cmat,
            smat,
            eye,
            n_fft,
            hop,
        }
    }

    /// mel: [B, 100, T] → waveform [B, (T-1)*hop].
    pub fn forward(&self, mel: Tensor<B, 3>) -> Tensor<B, 2> {
        // backbone
        let x = conv1d(mel, self.embed_w.clone(), Some(self.embed_b.clone()), ConvOptions::new([1], [3], [1], 1));
        let x = layer_norm(x.swap_dims(1, 2), &self.norm_w, &self.norm_b, 1e-6).swap_dims(1, 2); // [B,512,T]
        let mut x = x;
        for blk in &self.blocks {
            x = blk.forward(x);
        }
        let x = layer_norm(x.swap_dims(1, 2), &self.fln_w, &self.fln_b, 1e-6); // [B,T,512]

        // ISTFTHead
        let [b, t, _] = x.dims();
        let o = self.out.forward(x).swap_dims(1, 2); // [B,1026,T]
        let nf = self.n_fft / 2 + 1;
        let mag = o.clone().slice([0..b, 0..nf, 0..t]).exp().clamp_max(1e2);
        let phase = o.slice([0..b, nf..2 * nf, 0..t]);
        let re = (mag.clone() * phase.clone().cos()).swap_dims(1, 2); // [B,T,nf]
        let im = (mag * phase.sin()).swap_dims(1, 2);

        // irfft: [B,T,nf] @ [nf,n_fft] → [B,T,n_fft]
        let cm = self.cmat.clone().unsqueeze_dim::<3>(0);
        let sm = self.smat.clone().unsqueeze_dim::<3>(0);
        let frames = re.matmul(cm) - im.matmul(sm); // [B,T,n_fft]
        let frames = frames * self.window.clone().reshape([1, 1, self.n_fft]);

        // overlap-add (numerator) and window² envelope (denominator)
        let opts = ConvTransposeOptions::new([self.hop], [0], [0], [1], 1);
        let framesp = frames.swap_dims(1, 2); // [B,n_fft,T]
        let num = conv_transpose1d(framesp, self.eye.clone(), None, opts.clone()); // [B,1,Lfull]
        let win2 = self.window.clone().powf_scalar(2.0).reshape([1, self.n_fft, 1]);
        let win2f = win2 * Tensor::<B, 3>::ones([b, self.n_fft, t], &num.device());
        let den = conv_transpose1d(win2f, self.eye.clone(), None, opts);
        let y = num.squeeze_dim::<2>(1) / (den.squeeze_dim::<2>(1) + 1e-11); // [B, Lfull]

        // trim n_fft/2 from each end (center padding)
        let pad = self.n_fft / 2;
        let lfull = y.dims()[1];
        y.slice([0..b, pad..lfull - pad])
    }

    /// `forward` with named intermediate taps for parity probing against the
    /// reference vocos. Tap points mirror `scripts/probe_vocos.py`'s hooks.
    #[allow(clippy::type_complexity)]
    pub fn forward_probed(&self, mel: Tensor<B, 3>) -> (Tensor<B, 2>, Vec<(String, Vec<f32>, Vec<usize>)>) {
        let mut p: Vec<(String, Vec<f32>, Vec<usize>)> = Vec::new();
        fn tap<B: Backend, const D: usize>(p: &mut Vec<(String, Vec<f32>, Vec<usize>)>, name: &str, t: &Tensor<B, D>) {
            let data = t.clone().into_data();
            let shape = data.shape.to_vec();
            p.push((name.to_string(), data.to_vec::<f32>().unwrap(), shape));
        }

        let emb = conv1d(mel, self.embed_w.clone(), Some(self.embed_b.clone()), ConvOptions::new([1], [3], [1], 1));
        tap(&mut p, "embed", &emb);
        let x = layer_norm(emb.swap_dims(1, 2), &self.norm_w, &self.norm_b, 1e-6).swap_dims(1, 2);
        let mut x = x;
        for blk in &self.blocks {
            x = blk.forward(x);
        }
        let x = layer_norm(x.swap_dims(1, 2), &self.fln_w, &self.fln_b, 1e-6);
        tap(&mut p, "backbone", &x);

        let [b, t, _] = x.dims();
        let head_out = self.out.forward(x); // [B,T,1026]
        tap(&mut p, "head_out", &head_out);
        let o = head_out.swap_dims(1, 2);
        let nf = self.n_fft / 2 + 1;
        let mag = o.clone().slice([0..b, 0..nf, 0..t]).exp().clamp_max(1e2);
        let phase = o.slice([0..b, nf..2 * nf, 0..t]);
        let re = (mag.clone() * phase.clone().cos()).swap_dims(1, 2);
        let im = (mag * phase.sin()).swap_dims(1, 2);
        let cm = self.cmat.clone().unsqueeze_dim::<3>(0);
        let sm = self.smat.clone().unsqueeze_dim::<3>(0);
        let frames = re.matmul(cm) - im.matmul(sm);
        let frames = frames * self.window.clone().reshape([1, 1, self.n_fft]);
        let opts = ConvTransposeOptions::new([self.hop], [0], [0], [1], 1);
        let framesp = frames.swap_dims(1, 2);
        let num = conv_transpose1d(framesp, self.eye.clone(), None, opts.clone());
        let win2 = self.window.clone().powf_scalar(2.0).reshape([1, self.n_fft, 1]);
        let win2f = win2 * Tensor::<B, 3>::ones([b, self.n_fft, t], &num.device());
        let den = conv_transpose1d(win2f, self.eye.clone(), None, opts);
        let y = num.squeeze_dim::<2>(1) / (den.squeeze_dim::<2>(1) + 1e-11);
        let pad = self.n_fft / 2;
        let lfull = y.dims()[1];
        let audio = y.slice([0..b, pad..lfull - pad]);
        tap(&mut p, "audio", &audio);
        (audio, p)
    }
}
