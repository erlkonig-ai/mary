#!/usr/bin/env python3
"""Capture Inkling-Small's INTENDED numerics: 4-bit activations, not just weights.

Every existing golden runs `torch.set_default_dtype(torch.float32)` with no input
quantiser, so the Rust f32 lane has only ever been gated against a Python f32
lane -- two implementations of a quantity the checkpoint does not ask for. This
runs the same real layer twice, once each way, so the difference between them is
a measurement instead of a guess.

WHAT IS QUANTISED, and it is a much smaller set than "the model"
---------------------------------------------------------------
`hf_quant_config.json` lists `exclude_modules`, and taking its complement leaves
only `model.llm.layers.{3..41}.mlp.experts` -- the ROUTED experts. Attention,
both norms, both sconvs, the router, the shared experts, the dense MLPs of
layers 0-1, layer 2's experts, embed/unembed, vision and audio are all excluded.
So the input quantiser fires at exactly two matmuls per sparse layer: the
`gate_up_proj` (w13) and `down_proj` (w2) inside `InklingExperts.forward`.

  * w13's input  = post_attention_layernorm(hidden), the routed tokens' rows
  * w2's input   = act_fn(gate) * up

VALIDATION ORDER, on purpose
----------------------------
The FP4 number is only worth as much as the f32 number it is compared against,
so this script earns the comparison in three steps before making it:

  1. reproduce `real_y.bin` from the shipped golden with the module itself,
  2. reproduce it again with the hand-written expert loop and NO quantisation,
  3. only then turn the quantiser on.

Step 2 is what stops a bug in the hand-written loop from being reported as the
cost of 4-bit activations.

  usage: capture_inkling_fp4act.py <slice dir> <golden dir> <out dir>
"""
import json
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F
from safetensors import safe_open

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from nvfp4_act import (  # noqa: E402
    FP4_E2M1_MAX, FP4_E2M1_VALUES as nvfp4_grid, FP8_E4M3_MAX, GROUP,
    global_scale_from_amax, quantize_nvfp4, pack_nibbles, selftest,
)

SLICE = sys.argv[1]
GOLD = sys.argv[2]
OUT = sys.argv[3]
CFG = sys.argv[4] if len(sys.argv) > 4 else "/private/tmp/inkling_port/ckpt_config/config.json"
LAYER = 3
os.makedirs(OUT, exist_ok=True)

from transformers.masking_utils import create_sliding_window_causal_mask  # noqa: E402
from transformers.models.inkling.configuration_inkling import InklingTextConfig  # noqa: E402
from transformers.models.inkling.modeling_inkling import InklingDecoderLayer  # noqa: E402

torch.manual_seed(20260808)
torch.set_default_dtype(torch.float32)

print("=== reference selftest ===")
selftest()

raw = json.load(open(CFG))["text_config"]
raw.setdefault("moe_intermediate_size", raw["intermediate_size"])
cfg = InklingTextConfig(**raw)
cfg._attn_implementation = "eager"
H = cfg.hidden_size
INTER = cfg.moe_intermediate_size
assert cfg.mlp_layer_types[LAYER] == "sparse", cfg.mlp_layer_types[LAYER]

weight_map = json.load(open(SLICE + "/model.safetensors.index.json"))["weight_map"]
gathered = json.load(open(SLICE + "/gathered_indices.json"))
PFX = f"model.llm.layers.{LAYER}."


def get(name):
    with safe_open(SLICE + "/" + weight_map[name], framework="pt") as f:
        return f.get_tensor(name).float()


def get_raw(name):
    with safe_open(SLICE + "/" + weight_map[name], framework="pt") as f:
        return f.get_tensor(name)


# How the fused w13 splits into gate and up. NOT settled in this port: the
# shipped golden bundle and `capture_inkling_real.py`'s own docstring and
# manifest all say HALVED ([0:inter] gate, [inter:] up), while the current code
# in that file uses `_deint` (even rows gate, odd rows up). They disagree, and
# the sums prove it -- interleaved gives shared gate/up sums of
# (-1.412e3, +1.807e2) against the golden's (-6.243e2, -6.074e2), with the same
# total. Default to the split that reproduces the shipped golden, and measure
# the FP4 cost under both so the answer does not ride on the open question.
SPLIT = os.environ.get("INK_SPLIT", "half")
assert SPLIT in ("half", "interleave"), SPLIT


def deint(t, dim, how=None):
    how = how or SPLIT
    n = t.shape[dim]
    assert n % 2 == 0, (t.shape, dim)
    if how == "half":
        return (t.narrow(dim, 0, n // 2).contiguous(),
                t.narrow(dim, n // 2, n // 2).contiguous())
    return (t.index_select(dim, torch.arange(0, n, 2)).contiguous(),
            t.index_select(dim, torch.arange(1, n, 2)).contiguous())


FP4 = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
                -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0], dtype=np.float32)


def dequant_expert(name, e):
    """Dequantise ONE expert of a stacked NVFP4 tensor, low nibble first."""
    with safe_open(SLICE + "/" + weight_map[name], framework="pt") as f:
        codes = f.get_slice(name)[e:e + 1].numpy().astype(np.uint8)
    with safe_open(SLICE + "/" + weight_map[name + ".scale"], framework="pt") as f:
        scale = f.get_slice(name + ".scale")[e:e + 1].float().numpy()
    scale2 = get(name + ".scale2").numpy()[e]
    lo, hi = codes & 0x0F, (codes >> 4) & 0x0F
    vals = np.empty(codes.shape[:-1] + (codes.shape[-1] * 2,), dtype=np.float32)
    vals[..., 0::2] = FP4[lo]
    vals[..., 1::2] = FP4[hi]
    n_scales = scale.shape[-1]
    group = vals.shape[-1] // n_scales
    assert group == GROUP, group
    deq = vals.reshape(*vals.shape[:-1], n_scales, group) * scale[..., None]
    deq = deq.reshape(*vals.shape) * scale2
    return torch.from_numpy(deq[0])


# ------------------------------------------------------------ build layer ---
print("\n=== building real layer %d ===" % LAYER)
sd = {
    "input_layernorm.weight": get(PFX + "attn_norm.weight"),
    "post_attention_layernorm.weight": get(PFX + "mlp_norm.weight"),
    "attn_sconv.conv1d.weight": get(PFX + "attn_sconv.weight"),
    "mlp_sconv.conv1d.weight": get(PFX + "mlp_sconv.weight"),
    "self_attn.q_proj.weight": get(PFX + "attn.wq_du.weight"),
    "self_attn.k_proj.weight": get(PFX + "attn.wk_dv.weight"),
    "self_attn.v_proj.weight": get(PFX + "attn.wv_dv.weight"),
    "self_attn.r_proj.weight": get(PFX + "attn.wr_du.weight"),
    "self_attn.o_proj.weight": get(PFX + "attn.wo_ud.weight"),
    "self_attn.q_norm.weight": get(PFX + "attn.q_norm.weight"),
    "self_attn.k_norm.weight": get(PFX + "attn.k_norm.weight"),
    "self_attn.k_sconv.conv1d.weight": get(PFX + "attn.k_sconv.weight"),
    "self_attn.v_sconv.conv1d.weight": get(PFX + "attn.v_sconv.weight"),
    "self_attn.rel_logits_proj.proj": get(PFX + "attn.rel_logits_proj.proj"),
    "mlp.gate.weight": get(PFX + "mlp.gate.weight"),
    "mlp.gate.e_score_correction_bias": get(PFX + "mlp.gate.bias"),
    "mlp.gate.global_scale": get(PFX + "mlp.gate.global_scale"),
}
sw13 = get(PFX + "mlp.shared_experts.shared_w13_weight")
assert sw13.shape[1] == 2 * INTER, (sw13.shape, INTER)
_sg, _su = deint(sw13, 1)
sd["mlp.shared_experts.gate_proj"] = _sg
sd["mlp.shared_experts.up_proj"] = _su
sd["mlp.shared_experts.down_proj"] = get(PFX + "mlp.shared_experts.shared_w2_weight")
del sw13, _sg, _su

layer = InklingDecoderLayer(cfg, LAYER)
missing, unexpected = layer.load_state_dict(sd, strict=False)
assert not unexpected, unexpected
assert sorted(missing) == ["mlp.experts.down_proj", "mlp.experts.gate_up_proj"], sorted(missing)

# Only the experts the golden's routing actually reaches were extracted; the
# rest stay zero. If routing ever picked one of them the output would collapse,
# which the reproduction check below would catch immediately.
idx13 = gathered[PFX + "mlp.experts.w13_weight"]
idx2 = gathered[PFX + "mlp.experts.w2_weight"]
assert idx13 == idx2, "the two expert stacks were gathered on different indices"
with torch.no_grad():
    layer.mlp.experts.gate_up_proj.zero_()
    layer.mlp.experts.down_proj.zero_()
    for j, e in enumerate(idx13):
        w13 = dequant_expert(PFX + "mlp.experts.w13_weight", j)     # [2*INTER, H]
        g, u = deint(w13, 0)
        layer.mlp.experts.gate_up_proj[e] = torch.cat([g, u], dim=0)
        layer.mlp.experts.down_proj[e] = dequant_expert(PFX + "mlp.experts.w2_weight", j)
        del w13, g, u
print("  filled %d of %d experts (the ones the golden's routing reaches)"
      % (len(idx13), cfg.n_routed_experts))

# Fingerprints from the shipped golden: a mapping error lands on one line.
fps = json.load(open(GOLD + "/real_manifest.json"))["fingerprints"]
bad = []
for k, v in sd.items():
    f = v.flatten()
    if abs(float(f.sum()) - fps[k]["sum"]) > 1e-2 * max(1.0, abs(fps[k]["sum"])) * 1e-2:
        bad.append((k, float(f.sum()), fps[k]["sum"]))
    assert list(v.shape) == fps[k]["shape"], (k, v.shape, fps[k]["shape"])
print("  fingerprints: %d/%d tensors match the shipped golden" % (len(sd) - len(bad), len(sd)))
for k, a, b in bad:
    print("    MISMATCH %-40s got %+.6e want %+.6e" % (k, a, b))
assert not bad, "weight mapping disagrees with the shipped golden"

# ------------------------------------------------ step 1: reproduce golden ---
T = 8
x = torch.from_numpy(np.fromfile(GOLD + "/real_x.bin", dtype="<f4")).reshape(1, T, H)
y_gold = torch.from_numpy(np.fromfile(GOLD + "/real_y.bin", dtype="<f4")).reshape(1, T, H)
mask = create_sliding_window_causal_mask(
    config=cfg, inputs_embeds=x, attention_mask=None,
    past_key_values=None, position_ids=torch.arange(T).unsqueeze(0))

with torch.no_grad():
    y_module = layer(x, attention_mask=mask, conv_mask=None, past_key_values=None)
d = (y_module - y_gold).abs().max()
rel = float(d / y_gold.abs().max())
print("\n=== step 1: module vs shipped golden ===")
print("  max |d| %.3e   rel %.3e" % (d, rel))
assert rel < 1e-5, "the reconstructed layer does not reproduce the shipped golden"

# --------------------------------------------- the hand-written expert loop ---
CAPT = {}


def experts_forward(experts, hidden_states, top_k_index, top_k_weights,
                    s2_13=None, s2_2=None, two_level=True, capture=False):
    """`InklingExperts.forward`, with the input quantiser the config asks for.

    Transcribed from the module so the quantiser can sit on the two matmul
    inputs. With `s2_13=s2_2=None` it must be bit-identical to the module --
    step 2 checks exactly that.
    """
    final = torch.zeros_like(hidden_states)
    with torch.no_grad():
        expert_mask = F.one_hot(top_k_index, num_classes=experts.num_experts).permute(2, 1, 0)
        expert_hit = torch.greater(expert_mask.sum(dim=(-1, -2)), 0).nonzero()

    for expert_idx in expert_hit:
        expert_idx = expert_idx[0]
        if expert_idx == experts.num_experts:
            continue
        top_k_pos, token_idx = torch.where(expert_mask[expert_idx])
        current_state = hidden_states[token_idx]

        # --- quantised matmul 1: w13 / gate_up_proj ---
        a13 = current_state
        if s2_13 is not None:
            a13, bs13, c13 = quantize_nvfp4(a13, s2_13, two_level=two_level)
            if capture:
                CAPT.setdefault("w13_in", []).append(current_state)
                CAPT.setdefault("w13_in_q", []).append(a13)
                CAPT.setdefault("w13_bs", []).append(bs13)
                CAPT.setdefault("w13_codes", []).append(c13)
        elif capture:
            CAPT.setdefault("w13_in", []).append(current_state)

        gate, up = F.linear(a13, experts.gate_up_proj[expert_idx]).chunk(2, dim=-1)
        h = experts.act_fn(gate) * up

        # --- quantised matmul 2: w2 / down_proj ---
        a2 = h
        if s2_2 is not None:
            a2, bs2, c2 = quantize_nvfp4(a2, s2_2, two_level=two_level)
            if capture:
                CAPT.setdefault("w2_in", []).append(h)
                CAPT.setdefault("w2_in_q", []).append(a2)
                CAPT.setdefault("w2_bs", []).append(bs2)
                CAPT.setdefault("w2_codes", []).append(c2)
        elif capture:
            CAPT.setdefault("w2_in", []).append(h)

        cur = F.linear(a2, experts.down_proj[expert_idx])
        cur = cur * top_k_weights[token_idx, top_k_pos, None]
        final.index_add_(0, token_idx, cur.to(final.dtype))
    return final


def run_layer(s2_13=None, s2_2=None, two_level=True, capture=False):
    """The decoder layer's forward, with the routed-expert call swapped out."""
    orig = layer.mlp.experts.forward

    def patched(hidden_states, top_k_index, top_k_weights):
        return experts_forward(layer.mlp.experts, hidden_states, top_k_index,
                               top_k_weights, s2_13, s2_2, two_level, capture)

    layer.mlp.experts.forward = patched
    try:
        with torch.no_grad():
            return layer(x, attention_mask=mask, conv_mask=None, past_key_values=None)
    finally:
        layer.mlp.experts.forward = orig


# ------------------------------- step 2: the loop reproduces the module too ---
CAPT.clear()
y_loop = run_layer(capture=True)
d2 = float((y_loop - y_module).abs().max())
print("\n=== step 2: hand-written expert loop vs module (no quantiser) ===")
print("  max |d| %.3e  (must be 0: same ops, same order)" % d2)
assert d2 == 0.0, "the hand-written loop is not the module; the FP4 delta would be contaminated"

w13_in = torch.cat(CAPT["w13_in"], dim=0)
w2_in = torch.cat(CAPT["w2_in"], dim=0)

# ------------------------------------------------- the calibrated amaxes ---
amax13 = float(get_raw(PFX + "mlp.experts.w13_weight.input_amax").float()[0])
amax2 = float(get_raw(PFX + "mlp.experts.w2_weight.input_amax").float()[0])
s2_13 = global_scale_from_amax(amax13)
s2_2 = global_scale_from_amax(amax2)
print("\n=== calibrated activation scales (shipped in the checkpoint) ===")
print("  w13 input_amax %.6f -> s2 %.6e   observed |a|max here %.6f  (%.2fx of calib)"
      % (amax13, s2_13, float(w13_in.abs().max()), float(w13_in.abs().max()) / amax13))
print("  w2  input_amax %.6f -> s2 %.6e   observed |a|max here %.6f  (%.2fx of calib)"
      % (amax2, s2_2, float(w2_in.abs().max()), float(w2_in.abs().max()) / amax2))


def report(tag, y):
    dv = (y - y_module)
    rel = float(dv.norm() / y_module.norm())
    mx = float(dv.abs().max())
    cos = float(F.cosine_similarity(y.flatten(), y_module.flatten(), dim=0))
    print("  %-34s rel_l2 %.4e  max|d| %.4e  cos %.9f" % (tag, rel, mx, cos))
    return {"rel_l2": rel, "max_abs": mx, "cos": cos}


print("\n=== step 3: cost of 4-bit activations (layer %d output) ===" % LAYER)
print("  ||x|| %.4f  ||y|| %.4f  (the MoE branch is %.2fx the incoming residual,"
      % (float(x.norm()), float(y_module.norm()),
         float((y_module - x).norm() / x.norm())))
print("   so a relative error on the experts lands nearly undiluted on y)")
res = {"x_norm": float(x.norm()), "y_norm": float(y_module.norm())}
res["two_level_both"] = report("two-level, w13+w2", run_layer(s2_13, s2_2, True))
res["two_level_w13"] = report("two-level, w13 only", run_layer(s2_13, None, True))
res["two_level_w2"] = report("two-level, w2 only", run_layer(None, s2_2, True))
res["single_level_both"] = report("SINGLE-level, w13+w2", run_layer(s2_13, s2_2, False))

# ------------------------------------------- per-matmul activation error ---
print("\n=== activation quantisation error, at the matmul inputs ===")
print("  'dynamic' recomputes s2 from THIS tensor's amax instead of the shipped")
print("  input_amax, which separates the format's own cost from the fact that")
print("  these activations are not the ones the checkpoint was calibrated on.")
act = {}
for tag, a, s2c in (("w13_in", w13_in, s2_13), ("w2_in", w2_in, s2_2)):
    s2d = global_scale_from_amax(float(a.abs().max()))
    for lvl, two, s2 in (("two_level", True, s2c),
                         ("single_level", False, s2c),
                         ("two_level_dynamic", True, s2d)):
        q, bs, codes = quantize_nvfp4(a, s2, two_level=two)
        err = q - a
        eff = (bs * s2 if two else bs).unsqueeze(-1)
        ab = a.reshape(*bs.shape, GROUP)
        clipped = float(((ab.abs() / eff) > FP4_E2M1_MAX).float().mean())
        sat = float((bs >= FP8_E4M3_MAX).float().mean())
        # Which E2M1 code each block PEAKS at. A correct scale puts the block's
        # own maximum on code 7 (value 6.0); a scale that is too large parks it
        # lower and wastes the top of the grid. This is the histogram to compare
        # against the Rust lane's "1657 of 4096 blocks top out at code 4".
        peak = (codes.reshape(*bs.shape, GROUP) & 0x07).amax(dim=-1)
        hist = [int((peak == c).sum()) for c in range(8)]
        act[f"{tag}_{lvl}"] = {
            "rel_l1": float(err.abs().sum() / a.abs().sum()),
            "rel_l2": float(err.norm() / a.norm()),
            "block_scale_max": float(bs.max()),
            "blocks_at_e4m3_max_frac": sat,
            "elements_clipped_frac": clipped,
            "blocks": int(bs.numel()),
            "peak_code_hist": hist,
        }
        print("  %-8s %-18s rel_l1 %.4e rel_l2 %.4e  maxscale %7.2f  clip %.3f%%  sat %.3f%%"
              % (tag, lvl, act[f"{tag}_{lvl}"]["rel_l1"], act[f"{tag}_{lvl}"]["rel_l2"],
                 float(bs.max()), 100 * clipped, 100 * sat))
        print("           peak-code histogram (0..7, 7 == value 6.0): %s" % hist)

# The layer-output cost under a dynamic global scale, i.e. with the calibration
# mismatch taken out.
s2_13d = global_scale_from_amax(float(w13_in.abs().max()))
s2_2d = global_scale_from_amax(float(w2_in.abs().max()))
print("\n=== layer output, dynamic global scale (no calibration mismatch) ===")
res["two_level_dynamic_both"] = report("two-level dynamic, w13+w2",
                                       run_layer(s2_13d, s2_2d, True))

# If the two matmuls' errors are independent they add in quadrature. They do,
# to three digits -- which is the cheapest available evidence that this is
# quantisation noise being measured and not a bug in one of the two paths.
q13, q2 = res["two_level_w13"]["rel_l2"], res["two_level_w2"]["rel_l2"]
both = res["two_level_both"]["rel_l2"]
quad = (q13 ** 2 + q2 ** 2) ** 0.5
print("\n  quadrature check: sqrt(w13^2 + w2^2) = %.4e vs measured both = %.4e (%.2f%% apart)"
      % (quad, both, 100 * abs(quad - both) / both))
res["quadrature_predicted"] = quad

# ---------------------------------------------------------------- goldens ---
print("\n=== writing goldens to %s ===" % OUT)


def w(name, t, dt="<f4"):
    a = np.ascontiguousarray(t.detach().cpu().numpy().astype(dt))
    open(os.path.join(OUT, name), "wb").write(a.tobytes())
    return a


y_fp4 = run_layer(s2_13, s2_2, True)
w("fp4act_x.bin", x)
w("fp4act_y_f32act.bin", y_module)
w("fp4act_y_fp4act.bin", y_fp4)

# The direct oracle for a Rust activation quantiser: one real activation tensor,
# its E4M3 block scales as RAW BYTES (so the FP8 decode is under test, not just
# the multiply), its packed E2M1 codes, and the dequantised result.
assoc = {}
for tag, a, s2 in (("w13in", w13_in, s2_13), ("w2in", w2_in, s2_2)):
    q, bs, codes = quantize_nvfp4(a, s2, two_level=True)
    w("fp4act_%s_f32.bin" % tag, a)
    # FOLDED: q * (block_scale * s2). This is what compressed_tensors computes
    # (`scale / global_scale`, then one multiply) and what `_deq` holds.
    w("fp4act_%s_deq.bin" % tag, q)
    # UNFOLDED: (q * block_scale) * s2, the association `nvfp4.rs::decode_row`
    # already uses for WEIGHTS, whose comment records that folding first
    # "disagrees in the last bit on 7% of values". It disagrees here too, so
    # both are shipped and a gate can check the one it implements instead of
    # failing on an ulp it was never going to reproduce.
    grid = torch.tensor(nvfp4_grid)
    unf = (grid[codes.long()].reshape(*bs.shape, GROUP) * bs.unsqueeze(-1)
           ).reshape(a.shape) * s2
    w("fp4act_%s_deq_unfolded.bin" % tag, unf)
    open(os.path.join(OUT, "fp4act_%s_scale_e4m3.bin" % tag), "wb").write(
        bs.to(torch.float8_e4m3fn).view(torch.uint8).numpy().astype(np.uint8).tobytes())
    open(os.path.join(OUT, "fp4act_%s_codes.bin" % tag), "wb").write(
        pack_nibbles(codes).numpy().astype(np.uint8).tobytes())
    assoc[tag] = {
        "max_abs_fold_vs_unfold": float((q - unf).abs().max()),
        "frac_values_differing": float((q != unf).float().mean()),
    }
    print("  %-6s folded vs unfolded: max |d| %.3e on %.2f%% of values"
          % (tag, assoc[tag]["max_abs_fold_vs_unfold"],
             100 * assoc[tag]["frac_values_differing"]))

manifest = {
    "w13_split": SPLIT,
    "layer": LAYER, "tokens": T, "hidden": H, "moe_intermediate": INTER,
    "group": GROUP, "nibble_order": "low-first",
    "fp4_e2m1_max": FP4_E2M1_MAX, "fp8_e4m3_max": FP8_E4M3_MAX,
    "quantised_modules": "model.llm.layers.{3..41}.mlp.experts only "
                         "(complement of hf_quant_config exclude_modules)",
    "w13_input_amax": amax13, "w13_s2": s2_13,
    "w2_input_amax": amax2, "w2_s2": s2_2,
    "w13_rows": list(w13_in.shape), "w2_rows": list(w2_in.shape),
    "s2_formula": "input_amax / (6 * 448)",
    "block_scale_formula": "round_e4m3(block_amax / 6 / s2)",
    "dequant_formula": "code_value * block_scale_e4m3 * s2",
    "authority": "compressed_tensors 0.18.0 (cast_to_fp4, generate_gparam, "
                 "calculate_qparams, fake_quantize)",
    "experts_present": idx13,
    "dequant_association": assoc,
    "deq_bin_association": "folded: code * (block_scale * s2)",
    "deq_unfolded_bin_association": "unfolded: (code * block_scale) * s2",
    "layer_output": res,
    "activation_error": act,
}
json.dump(manifest, open(os.path.join(OUT, "fp4act_manifest.json"), "w"), indent=1)
for f in sorted(os.listdir(OUT)):
    print("  %-34s %9d B" % (f, os.path.getsize(os.path.join(OUT, f))))
print("\nwrote goldens to", OUT)
