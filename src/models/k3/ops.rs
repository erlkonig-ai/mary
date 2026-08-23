//! The two tensor operations every K3 sublayer is built out of, in one place.
//!
//! `nn.Linear(bias=False)` and `KimiRMSNorm` appear in the MLA block, in the
//! KDA block, in the MoE block, and twice per decoder layer as
//! `input_layernorm` / `post_attention_layernorm`. Before this module each
//! consumer carried its own transcription. That is exactly the shape of
//! duplication that ships a negated weight: four copies of a two-line function
//! whose subtleties (where the rounding goes, division versus `recip`) are
//! *not* obvious, gated in four different binaries against four different
//! oracles — so a fix applied to one is invisibly absent from the others.
//!
//! Both subtleties below were settled by measurement, by the jobs that first
//! hit them; the comments are theirs, kept with the code they describe rather
//! than with one of its callers.

use burn::prelude::*;

/// Where a block rounds intermediate activations.
///
/// The shipped module's arithmetic is fp32 at the pinned sites (see
/// `MANIFEST_layer_oracle.md` §7) and the model dtype everywhere else; this
/// selects the "everywhere else".
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ActRound {
    /// Keep f32 — reproduces a `dtype=torch.float32` run of the shipped block.
    None,
    /// Round to bfloat16 at every `nn.Linear` output, every activation output,
    /// every norm output and the residual add — reproduces the shipped
    /// `dtype=torch.bfloat16` run, which is what this checkpoint ships as.
    Bf16,
}

impl ActRound {
    /// Round `t` to this lane's storage precision.
    pub fn apply<B: Backend, const D: usize>(self, t: Tensor<B, D>) -> Tensor<B, D> {
        match self {
            ActRound::None => t,
            ActRound::Bf16 => {
                let device = t.device();
                let dims = t.dims();
                let v: Vec<f32> = t
                    .into_data()
                    .convert::<f32>()
                    .to_vec()
                    .expect("tensor -> f32");
                let v: Vec<f32> = v
                    .into_iter()
                    .map(|x| half::bf16::from_f32(x).to_f32())
                    .collect();
                Tensor::from_data(TensorData::new(v, dims), &device)
            }
        }
    }
}

/// `nn.Linear(bias=False)`: `x @ Wᵀ` for a `[out, in]` weight, rounded to the
/// lane's storage precision exactly as a torch module output is.
///
/// The weight stays at its checkpoint rank and orientation, so a transposition
/// mistake is a shape error rather than a plausible wrong answer.
pub fn linear<B: Backend>(x: Tensor<B, 2>, w: &Tensor<B, 2>, round: ActRound) -> Tensor<B, 2> {
    let [_, k] = x.dims();
    let [_, kw] = w.dims();
    assert_eq!(k, kw, "linear: x is [_, {k}] but the weight is [_, {kw}]");
    round.apply(x.matmul(w.clone().transpose()))
}

/// `KimiRMSNorm`: normalise in f32, cast **before** scaling by the weight.
///
/// Two things here are load-bearing, and both were settled by measurement
/// against the shipped module rather than read off it:
///
/// * **The cast placement.** `return self.weight * x.to(dtype)` rounds the
///   normalised value first and the product second — two roundings, not one.
///   With them in the right places a float64 evaluation of this expression
///   reproduces the shipped bf16 output **bit-for-bit**; with a single rounding
///   it reproduces only ~74% of it.
/// * **`x / sqrt(v)`, never `x * sqrt(v).recip()`.** `Tensor::recip` on the
///   ndarray backend dispatches to a SIMD *approximate* reciprocal
///   (`RecipVec` -> `Vector::recip`), which on aarch64 is accurate to about
///   1.4e-3 relative — roughly ten bits, not twenty-four. Written the obvious
///   way this norm reproduced ~85% of the shipped bits; written as a division
///   it reproduces ~100%. MEASURED, not inferred: `1/sqrt(2.21722e-4)` came
///   back as exactly 67.25 against a true 67.15773. The shipped code uses
///   `torch.rsqrt`, which is correctly rounded, so the division is the faithful
///   choice as well as the accurate one.
///
/// The second point is a trap for every RMSNorm in this crate, not just this
/// one.
pub fn rms_norm<B: Backend>(
    x: Tensor<B, 2>,
    weight: &Tensor<B, 1>,
    eps: f64,
    round: ActRound,
) -> Tensor<B, 2> {
    rms_norm_with(x, weight, eps, |t| round.apply(t))
}

/// [`rms_norm`] with the rounding supplied as a closure.
///
/// The formula and the rounding *policy* are separate concerns and only the
/// first should be shared. [`ActRound::apply`] reads the tensor as f32 and
/// rounds with `half::bf16::from_f32`; the MLA block's float64 reference lane
/// reads it as f64 and rounds with `from_f64`, and *those two are not the same
/// function*. MEASURED: swapping one for the other moved 1 element in 320 of
/// the MoE block's top-k combination by 1.22e-4 and turned a bit-exact check
/// from 100% to 99.6875%. So each caller keeps its own rounding and shares the
/// arithmetic — which is what the duplication that caused the `recip` bug in
/// `mla.rs` was actually about.
pub fn rms_norm_with<B: Backend, F>(
    x: Tensor<B, 2>,
    weight: &Tensor<B, 1>,
    eps: f64,
    round: F,
) -> Tensor<B, 2>
where
    F: Fn(Tensor<B, 2>) -> Tensor<B, 2>,
{
    let [_, w] = x.dims();
    assert_eq!(
        weight.dims()[0],
        w,
        "rms_norm: input is {w} wide but the gain is {}",
        weight.dims()[0]
    );
    assert!(eps > 0.0, "rms_norm: epsilon must be positive, got {eps}");
    let ms = x.clone().powf_scalar(2.0).mean_dim(1);
    let denom = ms.add_scalar(eps).sqrt();
    let normed = round(x / denom);
    round(weight.clone().unsqueeze::<2>() * normed)
}
