#!/usr/bin/env python3
"""Capture layer 2's BF16 routed-expert computation, for `inkling_bf16_expert_gate`.

Layer 2 is the one layer of Inkling-Small whose experts were never quantised:
`w13_weight [256, 4096, 4096]` and `w2_weight [256, 4096, 2048]` in plain BF16,
no `.scale` sidecar, 12.9 GiB. Unlike the MTP heads — which `transformers` does
not implement at all — this is an ORDINARY BF16 MoE layer that `transformers`
implements fully, so a real Python oracle exists and there is no excuse for a
Rust reference standing in for one.

## Two references, because they answer different questions

**The f64 arbiter.** Both lanes multiply the SAME BF16 bits; a BF16 value is an
f32 with a zeroed mantissa tail, so `x.double() @ w.double().T` is the exact
product of exactly those operands. Any disagreement with it is the f32
accumulator's, and nothing else's. This is what carries the tight budget.

**`transformers.models.inkling.modeling_inkling.InklingExperts`.** The authority
on what the layer MEANS: which half of `w13` is the gate, where the SiLU goes,
what `down_proj` is transposed against. It runs in BF16 end to end, which is a
LOWER precision than the device lane (it rounds the fused gate/up result to BF16
before the SiLU; the device keeps the MMA's f32 accumulator), so its budget is
set by that rounding and not by the kernel. It is a semantic check: every
structural mistake — a transposed operand, the wrong expert, the HALVED reading
of `w13` — is an O(1) error and cannot hide under a 1e-2 budget.

## The rounding points are captured, not assumed

The device rounds f32 to BF16 in exactly two places: the activation on the way
in, and the post-SiLU intermediate on the way into the second MMA. Both are
dumped as raw BF16 bits (`x_bf16`, `act_bf16`) so the gate can compare BITS
rather than tolerate a difference — `cvt.rn.bf16.f32` and
`torch.Tensor.to(torch.bfloat16)` are both round-to-nearest-even and there is no
reason to accept a discrepancy. `y_f64` is then the exact product of the
CAPTURED `act_bf16`, so the second GEMM can be gated in isolation from the
first.

## The activation

Real: the checkpoint's own embedding rows for a real five-token prompt, put
through layer 2's own `mlp_norm` RMSNorm — so the magnitude is set by the same
weight the running model sets it by, and the values are full-mantissa f32 rather
than something already BF16-representable (which would leave the inbound cast
untested). It is NOT the true layer-2 input; producing that means running the
stack, and what the true input settles — that the whole layer is right in
context — is settled instead by the end-to-end continuation, which is compared
against `transformers` generating from the same prompt.

Five tokens is deliberate: the lane pads M up to 16, so a five-row case
exercises the zero-padding in the same run.

  usage: capture_inkling_bf16_experts.py <checkpoint dir> <out dir>
"""
import json
import os
import sys

import numpy as np
import torch
from safetensors import safe_open

CKPT = sys.argv[1] if len(sys.argv) > 1 else \
    "models/thinkingmachines-inkling-small-nvfp4"
OUT = sys.argv[2] if len(sys.argv) > 2 else "inkling_oracle_bf16"
os.makedirs(OUT, exist_ok=True)

from transformers.models.inkling.configuration_inkling import InklingTextConfig
from transformers.models.inkling.modeling_inkling import InklingExperts

torch.manual_seed(20260813)
torch.set_default_dtype(torch.float32)

LAYER = 2
# Arbitrary, and it may be: the gate names the expert explicitly on both sides,
# so this choice tests indexing rather than routing. Which experts a token
# ACTUALLY routes to is the router's business and is gated elsewhere.
EXPERTS = [0, 137]
# "The capital of France is" — the same five ids the forward pass is driven with.
IDS = [976, 9029, 328, 10128, 382]

dev = "cuda" if torch.cuda.is_available() else "cpu"

raw = json.load(open(CKPT + "/config.json"))["text_config"]
raw.setdefault("moe_intermediate_size", raw["intermediate_size"])
cfg = InklingTextConfig(**raw)
H = cfg.hidden_size
I = cfg.moe_intermediate_size
T = len(IDS)

weight_map = json.load(open(CKPT + "/model.safetensors.index.json"))["weight_map"]
assert f"model.llm.layers.{LAYER}.mlp.experts.w13_weight.scale" not in weight_map, \
    "layer %d has a .scale sidecar; it is NVFP4, not the unquantised layer" % LAYER


def get(name):
    with safe_open(CKPT + "/" + weight_map[name], framework="pt") as f:
        return f.get_tensor(name)


def get_expert(name, e):
    """One expert out of a stacked matrix, without materialising the stack.

    The stack is 8.6 GB; this wants 33 MB of it.
    """
    with safe_open(CKPT + "/" + weight_map[name], framework="pt") as f:
        sl = f.get_slice(name)
        assert sl.get_dtype() == "BF16", (name, sl.get_dtype())
        return sl[e]


def dump(name, arr):
    a = np.ascontiguousarray(arr)
    open(os.path.join(OUT, name), "wb").write(a.tobytes())
    return a


def fp(t):
    """A compact fingerprint, so a loading error localises to a tensor."""
    f = t.float()
    return {"shape": list(t.shape), "sum": float(f.sum()),
            "absmax": float(f.abs().max()), "std": float(f.std())}


manifest = {
    "layer": LAYER, "experts": EXPERTS, "ids": IDS, "tokens": T,
    "hidden": H, "inter": I, "device": dev,
    "torch": torch.__version__,
    "transformers": __import__("transformers").__version__,
    "fingerprints": {},
}

# ------------------------------------------------------------ the activation --
embed = get("model.llm.embed.weight")
mlp_norm = get(f"model.llm.layers.{LAYER}.mlp_norm.weight")
manifest["fingerprints"]["mlp_norm.weight"] = fp(mlp_norm)
rows = embed[IDS].float()
# InklingRMSNorm, verbatim: variance in f32, then the weight multiplies.
var = rows.pow(2).mean(-1, keepdim=True)
x_f32 = (mlp_norm.float() * (rows * torch.rsqrt(var + cfg.rms_norm_eps))).contiguous()
x_bf16 = x_f32.to(torch.bfloat16)
manifest["fingerprints"]["x_f32"] = fp(x_f32)
manifest["x_source"] = (
    "model.llm.embed.weight[ids] through layer %d's mlp_norm (InklingRMSNorm, f32)" % LAYER
)
dump("bf16_x_f32.bin", x_f32.numpy().astype("<f4"))
dump("bf16_x_bf16.bin", x_bf16.view(torch.uint16).numpy().astype("<u2"))

x64 = x_bf16.double()

# ---------------------------------------------------------------- per expert --
tf_vs_f64 = 0.0
for e in EXPERTS:
    w13 = get_expert(f"model.llm.layers.{LAYER}.mlp.experts.w13_weight", e)
    w2 = get_expert(f"model.llm.layers.{LAYER}.mlp.experts.w2_weight", e)
    assert list(w13.shape) == [2 * I, H], (w13.shape, 2 * I, H)
    assert list(w2.shape) == [H, I], (w2.shape, H, I)
    manifest["fingerprints"][f"w13[{e}]"] = fp(w13)
    manifest["fingerprints"][f"w2[{e}]"] = fp(w2)

    # ---- the exact product, in the CHECKPOINT's own column order -------------
    # `w13` rows alternate g0, u0, g1, u1, ...; nothing here de-interleaves,
    # because the device kernel multiplies the stored rows as stored and the
    # de-interleave happens after, in `gate_up_silu`. Comparing here therefore
    # also pins that the kernel did not permute anything.
    both64 = x64 @ w13.double().T                       # [T, 2I]
    dump(f"bf16_e{e}_both_f64.bin", both64.numpy().astype("<f8"))

    # ---- the intermediate, at the device's own rounding point ---------------
    g = both64[:, 0::2]
    u = both64[:, 1::2]
    act64 = (g / (1.0 + torch.exp(-g))) * u
    act_bf16 = act64.to(torch.bfloat16)
    dump(f"bf16_e{e}_act_bf16.bin", act_bf16.view(torch.uint16).numpy().astype("<u2"))

    # ---- the exact product of THAT intermediate -----------------------------
    y64 = act_bf16.double() @ w2.double().T             # [T, H]
    dump(f"bf16_e{e}_y_f64.bin", y64.numpy().astype("<f8"))

    # ---- the authority, in its own precision --------------------------------
    # A one-expert InklingExperts: the module's parameter is the DE-INTERLEAVED
    # [all gates; all ups] that `conversion_mapping.py` produces at load time
    # ([Interleave(dim=0), Chunk(dim=0)]), which is why `.chunk(2, dim=-1)` in
    # its forward is the same split the device does column-wise on the raw rows.
    sub = InklingTextConfig(**{**raw, "n_routed_experts": 1})
    mod = InklingExperts(sub).to(dev).to(torch.bfloat16)
    with torch.no_grad():
        mod.gate_up_proj.copy_(
            torch.cat([w13[0::2], w13[1::2]], dim=0).unsqueeze(0).to(dev)
        )
        mod.down_proj.copy_(w2.unsqueeze(0).to(dev))
        y_tf = mod(
            x_bf16.to(dev).to(torch.bfloat16),
            torch.zeros(T, 1, dtype=torch.long, device=dev),
            torch.ones(T, 1, dtype=torch.bfloat16, device=dev),
        )
    y_tf = y_tf.float().cpu()
    dump(f"bf16_e{e}_y_tf_f32.bin", y_tf.numpy().astype("<f4"))

    scale = float(y64.abs().max())
    d = float((y_tf.double() - y64).abs().max()) / scale
    tf_vs_f64 = max(tf_vs_f64, d)
    manifest[f"e{e}"] = {
        "both_absmax": float(both64.abs().max()),
        "act_absmax": float(act64.abs().max()),
        "y_absmax": scale,
        "transformers_vs_f64": d,
    }
    print(f"expert {e:3}: |both| {float(both64.abs().max()):.4e}  "
          f"|act| {float(act64.abs().max()):.4e}  |y| {scale:.4e}  "
          f"transformers(bf16) vs f64 arbiter: {d:.3e}")
    del w13, w2, both64, act64, y64, mod

# --------------------------------------------------------------- the budgets --
# Written down HERE, from the arithmetic, before any comparison is run.
#
#  * f32 has a 24-bit significand, half-ulp eps = 2^-24 = 5.96e-8. A dot product
#    of K terms accumulated in f32 drifts from the exact sum by a random walk of
#    that: eps*sqrt(K) of the sum's own magnitude. Four times it leaves room for
#    the worst of a few thousand dot products.
#  * BF16 has an 8-bit stored significand, half-ulp 2^-9 = 1.95e-3. That is what
#    sets the `transformers` budget, NOT the kernel: its chain rounds to BF16 at
#    three points the device lane does not (the fused gate/up result, and the
#    final output; the post-SiLU intermediate both round). Three independent
#    half-ulps in quadrature is 3.4e-3, and 1e-2 is threefold headroom.
#  * The intermediate is where the two references stop agreeing to the bit, and
#    the first version of this file got the size of that wrong. It said: the
#    device's SiLU consumes an f32 that is eps*sqrt(4096) = 3.8e-6 from exact,
#    so it crosses a BF16 rounding boundary with probability 3.8e-6/1.95e-3 =
#    2.0e-3, about four of the 2048 terms of the second GEMM, for 8.6e-5 on the
#    output — budget 3e-4. That reasoning compares a RELATIVE f32 error against
#    a RELATIVE BF16 half-ulp, and the f32 accumulator's error is not relative:
#    it is ABSOLUTE, set by the whole tensor's scale (eps*sqrt(K)*|both|max =
#    1.3e-4 here). So it is the SMALL intermediate elements that flip, and they
#    flip often. Measured: 0.45% and 0.63% of elements, not 0.20%, with the
#    worst per-element flip 7 to 10 ulp — at |act| = 1.4e-3 against a scale of
#    635, i.e. an absolute error of 3.8e-5. A many-ulp flip of a tiny element is
#    a tiny error, which is why the criterion below is on ABSOLUTE deviation
#    over the tensor scale and not on ulps.
#
#    Redone: the largest possible deviation of one intermediate element is one
#    BF16 ulp at the largest element, 2^-8 = 3.9e-3 of the tensor's scale. Carry
#    that through the second GEMM: a fraction f of the K = 2048 terms perturbed
#    by one ulp gives 2^-8*sqrt(f*K) against a sum of sqrt(K) terms — the K
#    cancels, leaving 2^-8*sqrt(f). At f <= 1e-2 (twice the measured worst) that
#    is 3.9e-4, and doubling it for the worst of 20 480 output elements gives
#    1e-3.
#
# Metric throughout: worst absolute difference over the TENSOR's own scale.
# Per-element relative error is meaningless on dot products that cancel, and
# per-element ulps are meaningless where the perturbation is absolute.
eps32 = 2.0 ** -24
manifest["budgets"] = {
    "metric": "max |got - ref| / max |ref|",
    "cast_x_to_bf16": "bitwise identical, no tolerance",
    "gemm1_vs_f64": 4 * eps32 * (H ** 0.5),
    "act_vs_torch": 2.0 ** -8,
    "act_flip_fraction": 1e-2,
    "gemm2_isolated_vs_f64": 4 * eps32 * (I ** 0.5),
    "chain_vs_f64": 1e-3,
    "chain_vs_transformers": 1e-2,
    "padded_rows": "exactly zero",
    "derivation": (
        "f32 half-ulp 2^-24 random-walked over K products is eps*sqrt(K), and "
        "that error is ABSOLUTE; bf16 half-ulp 2^-9 (ulp 2^-8) is RELATIVE, so "
        "it is the small intermediate elements that reround, harmlessly. The "
        "chain budget is 2*2^-8*sqrt(f) at f = 1e-2 flipped terms. The "
        "transformers budget is set by its chain rounding to bf16 at points "
        "the device lane does not."
    ),
}
manifest["transformers_vs_f64_measured"] = tf_vs_f64
json.dump(manifest, open(OUT + "/bf16_manifest.json", "w"), indent=1)

print("\nwrote to", OUT)
for k, v in manifest["budgets"].items():
    print("  budget %-24s %s" % (k, v))
print("  measured: transformers(bf16) vs the f64 arbiter = %.3e "
      "(the floor for chain_vs_transformers)" % tf_vs_f64)
