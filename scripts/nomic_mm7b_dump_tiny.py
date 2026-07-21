#!/usr/bin/env python3
"""Deterministic TINY Qwen2.5-VL text-backbone reference dump for Rust parity.

No 16 GB download: instantiates transformers' real Qwen2.5-VL layer classes with
fixed-seed random init, so the goldens pin the EXACT math the ported Rust must
reproduce (RMSNorm, SwiGLU MLP, embedding, full decoder layer incl. attention +
1D-RoPE collapse of M-RoPE for text). Writes .npy to tests/golden/nomic_mm7b/.
"""
import json
from pathlib import Path
import numpy as np
import torch
from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import (
    Qwen2_5_VLRMSNorm, Qwen2_5_VLMLP, Qwen2_5_VLTextModel,
)
from transformers import Qwen2_5_VLTextConfig

OUT = Path(__file__).resolve().parent.parent / "tests" / "golden" / "nomic_mm7b"
OUT.mkdir(parents=True, exist_ok=True)
def save(n, t): np.save(OUT / f"{n}.npy", t.detach().cpu().float().numpy())

torch.manual_seed(0)
H, NH, NKV, HD, I, L, V, EPS, THETA = 64, 4, 2, 16, 128, 1, 100, 1e-6, 1e6

# ---- 1. RMSNorm ----
ln = Qwen2_5_VLRMSNorm(H, eps=EPS)
with torch.no_grad(): ln.weight.copy_(torch.randn(H) * 0.2 + 1.0)
x = torch.randn(2, 5, H)
save("rms_in", x); save("rms_weight", ln.weight); save("rms_out", ln(x))

# ---- 2. SwiGLU MLP ----
cfg = Qwen2_5_VLTextConfig(
    vocab_size=V, hidden_size=H, intermediate_size=I, num_hidden_layers=L,
    num_attention_heads=NH, num_key_value_heads=NKV, head_dim=HD,
    hidden_act="silu", rms_norm_eps=EPS, rope_theta=THETA,
    max_position_embeddings=128,
    rope_scaling={"type": "mrope", "mrope_section": [2, 3, 3]},
    tie_word_embeddings=False,
)
mlp = Qwen2_5_VLMLP(cfg)
xm = torch.randn(2, 5, H)
save("mlp_in", xm)
save("mlp_gate_w", mlp.gate_proj.weight); save("mlp_up_w", mlp.up_proj.weight)
save("mlp_down_w", mlp.down_proj.weight)
with torch.no_grad(): save("mlp_out", mlp(xm))

# ---- 3+4. Full 1-layer text model (embedding + decoder layer + final norm) ----
torch.manual_seed(1)
model = Qwen2_5_VLTextModel(cfg).eval()
ids = torch.tensor([[5, 9, 2, 41, 17, 3, 88]])  # [1,7]
np.save(OUT / "ids.npy", ids.cpu().numpy().astype(np.int64))
with torch.no_grad():
    out = model(input_ids=ids, output_hidden_states=True, use_cache=False, return_dict=True)
save("emb_out", out.hidden_states[0])          # token embeddings (layer-0 input)
save("layer0_out", out.hidden_states[1])        # after decoder layer 0
save("final_out", out.last_hidden_state)        # after model.norm

# dump all weights the Rust text model needs
sd = model.state_dict()
def w(name): save(name.replace(".", "__"), sd[name])
w("embed_tokens.weight"); w("norm.weight")
p = "layers.0"
for n in ["input_layernorm.weight", "post_attention_layernorm.weight",
          "self_attn.q_proj.weight", "self_attn.q_proj.bias",
          "self_attn.k_proj.weight", "self_attn.k_proj.bias",
          "self_attn.v_proj.weight", "self_attn.v_proj.bias",
          "self_attn.o_proj.weight",
          "mlp.gate_proj.weight", "mlp.up_proj.weight", "mlp.down_proj.weight"]:
    w(f"{p}.{n}")

meta = dict(hidden=H, n_heads=NH, n_kv_heads=NKV, head_dim=HD, intermediate=I,
            n_layers=L, vocab=V, eps=EPS, rope_theta=THETA, seq=int(ids.shape[1]))
json.dump(meta, open(OUT / "tiny_meta.json", "w"), indent=2)
print("dumped tiny goldens to", OUT)
print(json.dumps(meta, indent=2))
