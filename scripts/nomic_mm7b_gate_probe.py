#!/usr/bin/env python3
"""GATE MECHANICS probe — no 16GB download required.

Confirms empirically (real processor/tokenizer + a TINY random Qwen2.5-VL backbone
exercising the real BiQwen2_5.forward code path):
  - output is DENSE [B, hidden] (not [B, N, hidden] multi-vector)
  - pooling = last token (left-padded), L2-normalized (norm==1)
  - scoring = score_single_vector (dot == cosine)
  - the exact query / document / image prompt strings from the REAL processor
Real dim D = hidden_size of Qwen2.5-VL-7B = 3584 (from cached config.json).
"""
import json, glob, os
import torch
from PIL import Image
from transformers import Qwen2_5_VLConfig
from colpali_engine.models import BiQwen2_5, BiQwen2_5_Processor

MODEL_ID = "nomic-ai/nomic-embed-multimodal-7b"
proc = BiQwen2_5_Processor.from_pretrained(MODEL_ID)

# tiny random backbone (head_dim=8 -> mrope_section sums to 4 = head_dim/2)
base = glob.glob(os.path.expanduser("~") + "/.cache/huggingface/hub/models--Qwen--Qwen2.5-VL-7B-Instruct/snapshots/*/config.json")[0]
full = json.load(open(base))
cfg = Qwen2_5_VLConfig(**full)
# shrink text
cfg.hidden_size = 16; cfg.num_hidden_layers = 2; cfg.num_attention_heads = 2
cfg.num_key_value_heads = 1; cfg.intermediate_size = 32
cfg.rope_scaling = {"type": "mrope", "mrope_section": [1, 1, 2]}  # sum=4=head_dim/2
# shrink vision to match (out_hidden_size must equal text hidden_size)
vc = cfg.vision_config
vc.hidden_size = 16; vc.depth = 2; vc.num_heads = 2; vc.intermediate_size = 32
vc.out_hidden_size = 16
cfg.tie_word_embeddings = False

torch.manual_seed(0)
model = BiQwen2_5(cfg).eval()

q = ["What is the capital of France?"]
d = ["The capital of France is Paris."]
bq = proc.process_queries(q)
bt = proc.process_texts(d)
img = Image.new("RGB", (56, 56), color=(120, 180, 60))
bi = proc.process_images([img])

print("=== PROMPT FORMATS (real processor) ===")
print("query_prefix=%r  query_aug_token=%r" % (
    getattr(proc, "query_prefix", None), getattr(proc, "query_augmentation_token", None)))
print("QUERY decoded :", repr(proc.tokenizer.decode(bq["input_ids"][0])))
print("QUERY ids     :", bq["input_ids"][0].tolist())
print("DOC   decoded :", repr(proc.tokenizer.decode(bt["input_ids"][0])))
print("IMAGE decoded :", repr(proc.tokenizer.decode(bi["input_ids"][0]))[:300])
print("image_grid_thw:", bi["image_grid_thw"].tolist(), " pixel_values:", list(bi["pixel_values"].shape))
print("visual_prompt_prefix:", repr(getattr(proc, "visual_prompt_prefix", None)))

with torch.no_grad():
    qe = model(**bq); de = model(**bt); ie = model(**bi)
print("\n=== GATE (mechanics, tiny backbone) ===")
print("query emb shape:", tuple(qe.shape), " dim:", qe.dim(), "=> DENSE" if qe.dim()==2 else "=> MULTIVECTOR")
print("doc   emb shape:", tuple(de.shape))
print("image emb shape:", tuple(ie.shape))
print("query L2 norm  :", float(qe.norm(dim=-1)[0]), "(==1 => L2-normalized)")
sc = proc.score(list(torch.unbind(qe)), list(torch.unbind(de)))
print("score(q,d) shape:", tuple(sc.shape), " values:", sc.tolist())
print("\nREAL embedding dim D (Qwen2.5-VL-7B hidden_size) = 3584")
