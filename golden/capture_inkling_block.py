#!/usr/bin/env python3
"""Capture oracle vectors for the Inkling block primitives.

Runs the REAL `transformers.models.inkling` modules with seeded random weights
and dumps their inputs and outputs as raw little-endian f32 (and i64 for the
router's indices), for `inkling_block_gate` to check the Rust against.

Everything is float32 end to end. The reference computes the short convolution
in fp32 regardless, and RMSNorm's `.to(input_dtype)` is a no-op at f32, so this
first gate is about the arithmetic being right rather than about bf16 rounding
policy — which is a separate question and deserves its own gate rather than
being tangled into this one.

Sequence length is deliberately longer than the convolution kernel so the
causal ramp-in is exercised: the first k-1 positions are the ones that depend
on zero-padding, and a conv implemented with the taps reversed agrees with the
correct one everywhere EXCEPT there and at the edges.

  usage: capture_inkling_block.py <checkpoint dir> <out dir>
"""
import json
import os
import sys

import numpy as np
import torch

CKPT = sys.argv[1] if len(sys.argv) > 1 else "./Inkling-Small-NVFP4"
OUT = sys.argv[2] if len(sys.argv) > 2 else "./inkling-oracle"
os.makedirs(OUT, exist_ok=True)

from transformers.models.inkling.configuration_inkling import InklingTextConfig
from transformers.models.inkling.modeling_inkling import (
    InklingRMSNorm,
    InklingShortConvolution,
    InklingTopkRouter,
)

torch.manual_seed(20260808)
torch.set_default_dtype(torch.float32)

raw = json.load(open(CKPT + "/config.json"))["text_config"]
cfg = InklingTextConfig(**raw)
H = cfg.hidden_size
K = cfg.conv_kernel_size if hasattr(cfg, "conv_kernel_size") else raw["sconv_kernel_size"]
T = 11                      # > K, so the causal ramp-in is covered
manifest = {"hidden_size": H, "kernel": K, "tokens": T,
            "rms_norm_eps": cfg.rms_norm_eps}


def dump(name, arr, dtype="<f4"):
    a = np.ascontiguousarray(arr.detach().cpu().numpy().astype(dtype))
    open(os.path.join(OUT, name), "wb").write(a.tobytes())
    return a


# ---------------------------------------------------------------- RMSNorm ---
norm = InklingRMSNorm(H, eps=cfg.rms_norm_eps)
with torch.no_grad():
    norm.weight.normal_(mean=1.0, std=0.1)
x = torch.randn(T, H)
with torch.no_grad():
    y = norm(x)
dump("blk_rms_x.bin", x)
dump("blk_rms_w.bin", norm.weight)
dump("blk_rms_y.bin", y)
manifest["rms"] = {"in": [T, H], "out": [T, H]}
print("rmsnorm   in %s -> out %s  |y| max %.6g" % (tuple(x.shape), tuple(y.shape), y.abs().max()))

# ------------------------------------------------------- ShortConvolution ---
# layer_idx / conv_idx only matter for cache routing; with no cache they are inert.
sconv = InklingShortConvolution(H, K, layer_idx=0, conv_idx=0)
with torch.no_grad():
    sconv.conv1d.weight.normal_(std=0.5)
xs = torch.randn(1, T, H)
with torch.no_grad():
    ys = sconv(xs, past_key_values=None, conv_mask=None)
dump("blk_sconv_x.bin", xs)
dump("blk_sconv_w.bin", sconv.conv1d.weight)      # [H, 1, K]
dump("blk_sconv_y.bin", ys)
manifest["sconv"] = {"in": [1, T, H], "weight": list(sconv.conv1d.weight.shape), "out": [1, T, H]}
print("sconv     weight %s  in %s -> out %s" %
      (tuple(sconv.conv1d.weight.shape), tuple(xs.shape), tuple(ys.shape)))

# A pure-conv reference too (without the module's internal residual), so the
# gate can report WHICH of the two an implementation matches. Without this an
# off-by-an-identity-term bug just looks like "wrong numbers".
with torch.no_grad():
    w = sconv.conv1d.weight.squeeze(1)
    pure = torch.nn.functional.conv1d(
        xs.transpose(1, 2), w.unsqueeze(1), None, padding=K - 1, groups=H
    )[:, :, :T].transpose(1, 2)
dump("blk_sconv_y_noresid.bin", pure)
print("          residual delta |y - conv| max %.6g" % (ys - pure).abs().max())

# ----------------------------------------------------------------- Router ---
router = InklingTopkRouter(cfg)
with torch.no_grad():
    router.weight.normal_(std=0.02)
    router.e_score_correction_bias.normal_(std=0.05)
    router.global_scale.fill_(1.0)
xr = torch.randn(T, H)
with torch.no_grad():
    routed_logits, topk_weights, topk_indices, shared_gammas = router(xr)
dump("blk_router_x.bin", xr)
dump("blk_router_w.bin", router.weight)
dump("blk_router_bias.bin", router.e_score_correction_bias)
dump("blk_router_gscale.bin", router.global_scale)
dump("blk_router_topk_idx.bin", topk_indices, dtype="<i8")
dump("blk_router_topk_w.bin", topk_weights)
dump("blk_router_gammas.bin", shared_gammas)
manifest["router"] = {
    "tokens": T,
    "gate_rows": int(router.weight.shape[0]),
    "n_routed": int(cfg.n_routed_experts),
    "n_shared": int(cfg.n_shared_experts),
    "top_k": int(cfg.num_experts_per_tok),
    "route_scale": float(cfg.route_scale),
}
print("router    weight %s  top_k %d  gammas %s" %
      (tuple(router.weight.shape), cfg.num_experts_per_tok, tuple(shared_gammas.shape)))
print("          weights sum per token (routed+shared): %s" %
      [round(float(v), 6) for v in (topk_weights.sum(-1) + shared_gammas.sum(-1))[:3]])

# How often the bias actually CHANGES the selection. If it never does, a gate
# that ignores the bias would still pass, and the check would be vacuous.
with torch.no_grad():
    scores = routed_logits.sigmoid()
    without = torch.topk(scores, cfg.num_experts_per_tok, dim=-1, sorted=False)[1]
    a = set(map(tuple, torch.sort(without, -1)[0].tolist()))
    b = set(map(tuple, torch.sort(topk_indices, -1)[0].tolist()))
    differing = sum(1 for i in range(T)
                    if sorted(without[i].tolist()) != sorted(topk_indices[i].tolist()))
manifest["router"]["tokens_where_bias_changes_selection"] = differing
print("          bias changes the chosen set on %d of %d tokens" % (differing, T))

json.dump(manifest, open(os.path.join(OUT, "blk_manifest.json"), "w"), indent=1)
print("\nwrote oracle to", OUT)
