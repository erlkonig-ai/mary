#!/usr/bin/env python3
"""Empirical reference probe for nomic-ai/nomic-embed-multimodal-7b (BiQwen2_5).

Answers THE GATE empirically (dense vs multi-vector, dim, pooling, prompt format,
scoring, dtype) AND dumps float32 component goldens for the Rust parity harness.

Run: python3 scripts/nomic_mm7b_probe.py   (loads ~16GB base + adapter; CPU fp32)
Writes goldens to tests/golden/nomic_mm7b/ and a JSON summary to
/tmp/codex_outputs/nomic_mm7b_gate.json
"""
from __future__ import annotations
import json, os
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from colpali_engine.models import BiQwen2_5, BiQwen2_5_Processor

MODEL_ID = "nomic-ai/nomic-embed-multimodal-7b"
OUT = Path(__file__).resolve().parent.parent / "tests" / "golden" / "nomic_mm7b"
OUT.mkdir(parents=True, exist_ok=True)
GATE = Path("/tmp/codex_outputs/nomic_mm7b_gate.json")
GATE.parent.mkdir(parents=True, exist_ok=True)

def save(name, t):
    np.save(OUT / f"{name}.npy", t.detach().cpu().float().numpy())

torch.manual_seed(0)
dev = "cpu"
dtype = torch.float32  # clean ground-truth math for parity goldens

print("loading processor...")
proc = BiQwen2_5_Processor.from_pretrained(MODEL_ID)
print("loading model (base Qwen2.5-VL-7B + LoRA adapter)...")
model = BiQwen2_5.from_pretrained(MODEL_ID, torch_dtype=dtype, device_map=dev).eval()

cfg = model.config
summary = {"model_id": MODEL_ID, "weight_dtype_shipped": "bfloat16",
           "probe_dtype": "float32"}

# --- config dump ---
tc = getattr(cfg, "text_config", cfg)
summary["config"] = {
    "hidden_size": getattr(tc, "hidden_size", None),
    "num_hidden_layers": getattr(tc, "num_hidden_layers", None),
    "num_attention_heads": getattr(tc, "num_attention_heads", None),
    "num_key_value_heads": getattr(tc, "num_key_value_heads", None),
    "intermediate_size": getattr(tc, "intermediate_size", None),
    "rms_norm_eps": getattr(tc, "rms_norm_eps", None),
    "rope_theta": getattr(tc, "rope_theta", None),
    "rope_scaling": getattr(tc, "rope_scaling", None),
    "vocab_size": getattr(tc, "vocab_size", None),
    "head_dim": getattr(tc, "head_dim", None),
    "hidden_act": getattr(tc, "hidden_act", None),
    "tie_word_embeddings": getattr(cfg, "tie_word_embeddings", None),
}
vc = getattr(cfg, "vision_config", None)
if vc is not None:
    summary["vision_config"] = {
        k: getattr(vc, k, None) for k in
        ["hidden_size","depth","num_heads","patch_size","spatial_merge_size",
         "in_chans","out_hidden_size","intermediate_size","fullatt_block_indexes",
         "window_size","temporal_patch_size"]
    }

# --- query / document prompt format (decode the actual input_ids) ---
q_text = "What is the capital of France?"
d_text = "The capital of France is Paris, a major European city."
bq = proc.process_queries([q_text])
bt = proc.process_texts([d_text])
summary["query_input_ids"] = bq["input_ids"][0].tolist()
summary["query_decoded"] = proc.tokenizer.decode(bq["input_ids"][0])
summary["doc_text_input_ids"] = bt["input_ids"][0].tolist()
summary["doc_text_decoded"] = proc.tokenizer.decode(bt["input_ids"][0])
summary["query_prefix"] = getattr(proc, "query_prefix", None)
summary["query_aug_token"] = getattr(proc, "query_augmentation_token", None)
summary["visual_prompt_prefix"] = getattr(proc, "visual_prompt_prefix", None)

# image batch
img = Image.new("RGB", (56, 56), color=(123, 200, 90))
bi = proc.process_images([img])
summary["image_input_ids"] = bi["input_ids"][0].tolist()
summary["image_decoded"] = proc.tokenizer.decode(bi["input_ids"][0])
summary["image_grid_thw"] = bi["image_grid_thw"].tolist()
summary["pixel_values_shape"] = list(bi["pixel_values"].shape)

# --- forward: GATE shapes ---
with torch.no_grad():
    q_emb = model(**bq)          # BiQwen forward -> pooled+normalized
    d_emb = model(**bt)
    i_emb = model(**bi)
summary["query_emb_shape"] = list(q_emb.shape)
summary["doc_emb_shape"] = list(d_emb.shape)
summary["image_emb_shape"] = list(i_emb.shape)
summary["embedding_dim"] = int(q_emb.shape[-1])
summary["is_multivector"] = (q_emb.dim() == 3)  # [B, N, D] would be multi-vector
summary["query_emb_norm"] = float(q_emb.norm(dim=-1).mean())
summary["pooling"] = "last_token (left-padded); L2-normalized"
summary["scoring"] = "score_single_vector = einsum('bd,cd->bc') dot == cosine (L2-normed)"

# scores sanity
qs = list(torch.unbind(q_emb))
ps = list(torch.unbind(d_emb))
scores = proc.score(qs, ps)
summary["score_query_vs_doc"] = scores.tolist()

save("query_emb", q_emb)
save("doc_text_emb", d_emb)
save("image_emb", i_emb)
np.save(OUT / "query_input_ids.npy", bq["input_ids"].cpu().numpy())
np.save(OUT / "doc_text_input_ids.npy", bt["input_ids"].cpu().numpy())

# --- component goldens via the underlying Qwen2_5_VLModel (text-only path) ---
# Run the parent forward directly with output_hidden_states for per-layer goldens.
with torch.no_grad():
    base_out = super(BiQwen2_5, model).forward(
        input_ids=bq["input_ids"], attention_mask=bq["attention_mask"],
        output_hidden_states=True, use_cache=False, return_dict=True,
    )
hs = base_out.hidden_states  # tuple: [emb, layer0, ..., layerN] each [1,S,H]
save("text_hidden_emb_in", hs[0])         # token embeddings (input to layer 0)
save("text_hidden_after_layer0", hs[1])   # after first decoder block
save("text_hidden_after_layer1", hs[2])
save("text_last_hidden_state", base_out.last_hidden_state)
summary["num_hidden_states"] = len(hs)
summary["text_seq_len"] = int(bq["input_ids"].shape[1])

# embedding table row for a couple tokens (for an embedding-lookup golden)
emb_table = model.get_input_embeddings().weight  # [V, H]
tok_sample = bq["input_ids"][0, :4].tolist()
save("emb_table_first_tokens", emb_table[bq["input_ids"][0]])
summary["emb_table_shape"] = list(emb_table.shape)
summary["first_tokens"] = tok_sample

with open(GATE, "w") as f:
    json.dump(summary, f, indent=2)
print("=== GATE SUMMARY ===")
print(json.dumps({k: summary[k] for k in
    ["is_multivector","embedding_dim","query_emb_shape","image_emb_shape",
     "query_emb_norm","pooling","query_decoded","doc_text_decoded","image_decoded",
     "score_query_vs_doc","config"]}, indent=2))
print("goldens written to", OUT)
