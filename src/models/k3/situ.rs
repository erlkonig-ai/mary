//! `situ` — the gated activation Moonshot's Kimi K3 runs in place of SwiGLU,
//! in every dense MLP and every routed/shared MoE expert (`hidden_act: "situ"`
//! in `config.json`; `KimiMLP` and `KimiBlockSparseMLP` both dispatch to it).
//!
//! Shipped definition (`modeling_kimi_linear.py`, `class SituAndMul`):
//!
//! ```text
//! d      = x.shape[-1] // 2
//! gate   = x[..., :d].to(float32)
//! up     = x[..., d:].to(float32)
//! situ_a = beta * tanh(gate / beta) * sigmoid(gate)
//! up     = linear_beta * tanh(up / linear_beta)     # only when linear_beta is set
//! out    = (situ_a * up).to(x.dtype)
//! ```
//!
//! Read against SwiGLU (`silu(gate) * up` = `gate*sigmoid(gate) * up`), `situ`
//! replaces the bare `gate` factor with `beta*tanh(gate/beta)` and the bare
//! `up` with `linear_beta*tanh(up/linear_beta)` — a soft clip on each branch.
//! `b*tanh(x/b)` is `x - x³/(3b²) + …`, i.e. the identity to first order and
//! saturating at `±b`, so both branches are unchanged in the small-signal
//! regime and hard-bounded outside it:
//!
//! * gate branch ∈ `[-0.26977, 4]` — `→ +beta` as `gate → +∞`; `→ 0⁻` as
//!   `gate → −∞`, since `sigmoid` decays faster than `tanh` saturates. The
//!   minimum sits at `gate ≈ −1.2188`.
//! * up branch ∈ `[-25, 25]`.
//! * product ∈ `[-100, 100]` — so the MLP's input to `w2` is bounded, whatever
//!   the projections produce.
//!
//! **The two betas are unexplained.** `beta = 4.0` (`activation_situ_beta`)
//! and `linear_beta = 25.0` (`activation_situ_linear_beta`) arrive as bare
//! numbers in `config.json`; the release carries no comment, citation or
//! derivation for either. What they *do* is the saturation ceiling described
//! above — that much is read off the formula — but *why those values*, and why
//! so asymmetric (the gate is clipped while `silu` is still curving, the up
//! branch only far out in its tail), is not stated anywhere in the shipped
//! code, and this port does not invent a reason for them.
//!
//! Layout: the shipped MLP builds `cat([w1(x), w3(x)], dim=-1)` — gate first,
//! up second — and hands that single tensor to one call, so [`Situ::forward`]
//! splits the last dimension in half. [`Situ::forward_pair`] is the same math
//! on two already-separate tensors, which is what an expert-parallel or fused
//! path wants (and what avoids materialising the concatenation at all).
//!
//! Precision: the shipped module up-casts both halves to f32, does *all* of
//! the arithmetic in f32 whatever the storage dtype is, and rounds the product
//! back at the end. An f32 Burn tensor is therefore already in the reference's
//! arithmetic; a bf16 lane must round only at the two ends, never in between.
//!
//! Gated against `flash-linear-attention`-era oracle vectors captured from the
//! shipped `SituAndMul` module itself — see `src/bin/kimi_situ_gate.rs`.

use burn::prelude::*;
use burn::tensor::activation::sigmoid;

/// `activation_situ_beta` from Kimi K3's `config.json` — the gate branch clip.
pub const K3_BETA: f64 = 4.0;

/// `activation_situ_linear_beta` from the same config — the up branch clip.
pub const K3_LINEAR_BETA: f64 = 25.0;

/// The `situ` activation, parameterised exactly as `SituAndMul.__init__` is.
///
/// `linear_beta = None` reproduces the shipped `if self.linear_beta is not
/// None` fall-through (up passes straight through, unclipped); K3 always sets
/// it, but the module keeps the branch because the config type allows `None`
/// and a silently-applied clip would be the wrong default.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Situ {
    /// Gate-branch soft-clip scale — also the gate branch's supremum.
    pub beta: f64,
    /// Up-branch soft-clip scale; `None` leaves the up branch linear.
    pub linear_beta: Option<f64>,
}

impl Default for Situ {
    fn default() -> Self {
        Self::k3()
    }
}

impl Situ {
    /// Arbitrary betas — for reading a config, or for probing what the two
    /// constants actually buy.
    pub const fn new(beta: f64, linear_beta: Option<f64>) -> Self {
        Self { beta, linear_beta }
    }

    /// The shipped Kimi K3 setting: `beta = 4`, `linear_beta = 25`.
    pub const fn k3() -> Self {
        Self::new(K3_BETA, Some(K3_LINEAR_BETA))
    }

    /// `beta * tanh(gate / beta) * sigmoid(gate)` — a soft-clipped `silu`.
    ///
    /// Op order follows the Python left-to-right exactly (`(beta * tanh(·)) *
    /// sigmoid(·)`, division before multiplication), so the f32 rounding
    /// sequence matches the reference rather than merely the algebra.
    pub fn gate_branch<B: Backend, const D: usize>(&self, gate: Tensor<B, D>) -> Tensor<B, D> {
        let clipped = gate.clone().div_scalar(self.beta).tanh().mul_scalar(self.beta);
        clipped * sigmoid(gate)
    }

    /// `linear_beta * tanh(up / linear_beta)`, or the identity when the config
    /// leaves `linear_beta` unset.
    pub fn up_branch<B: Backend, const D: usize>(&self, up: Tensor<B, D>) -> Tensor<B, D> {
        match self.linear_beta {
            Some(lb) => up.div_scalar(lb).tanh().mul_scalar(lb),
            None => up,
        }
    }

    /// SiTU-GLU on already-split halves: `gate_branch(gate) * up_branch(up)`.
    pub fn forward_pair<B: Backend, const D: usize>(
        &self,
        gate: Tensor<B, D>,
        up: Tensor<B, D>,
    ) -> Tensor<B, D> {
        self.gate_branch(gate) * self.up_branch(up)
    }

    /// SiTU-GLU on the concatenated `[gate | up]` tensor the shipped MLP
    /// builds — the direct analogue of `SituAndMul.forward`. The last
    /// dimension must be even; Python's `// 2` would split an odd width
    /// lopsidedly and only fail later, in the broadcast, so fail here instead.
    pub fn forward<B: Backend, const D: usize>(&self, gate_up: Tensor<B, D>) -> Tensor<B, D> {
        let width = gate_up.dims()[D - 1];
        assert!(
            width % 2 == 0,
            "situ expects a concatenated [gate | up] tensor: last dim {width} is odd"
        );
        let d = width / 2;
        let gate = gate_up.clone().narrow(D - 1, 0, d);
        let up = gate_up.narrow(D - 1, d, d);
        self.forward_pair(gate, up)
    }

    /// Tight bound on `|output|` — `beta * linear_beta`, both branches
    /// saturated. Useful as a cheap invariant on a real forward pass, and as
    /// the reason a `situ` MLP cannot blow up however large its projections
    /// get.
    pub fn output_bound(&self) -> f64 {
        self.beta * self.linear_beta.unwrap_or(f64::INFINITY)
    }
}
