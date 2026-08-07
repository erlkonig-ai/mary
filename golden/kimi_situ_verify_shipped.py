# Does the situ oracle's f32/bf16 column really come from the SHIPPED class?
#
# gen_situ.py transcribes the SituAndMul body inline (`def situ_torch`) rather
# than importing it, and the module cannot simply be imported: the installed
# transformers no longer exports `OutputRecorder`, which modeling_kimi_linear.py
# imports at module scope.
#
# So: parse the shipped file with `ast`, lift out the SituAndMul ClassDef's
# EXACT source segment, exec that text, instantiate it with config.json's betas
# and compare BITWISE against the stored columns. Nothing is retyped -- the
# class body executed here is the shipped bytes. The checkpoint is only read.
import os
import ast
import hashlib
import json
import sys

import numpy as np
import torch
import torch.nn as nn

MODEL = os.environ.get("K3_MODEL_DIR", "./kimi-k3")
SRC = f"{MODEL}/modeling_kimi_linear.py"
NPZ = os.path.join(os.environ.get("K3_ORACLE_DIR", "./k3-oracle"),
              "situ_activation.npz")

text = open(SRC).read()
print("modeling_kimi_linear.py sha256:", hashlib.sha256(text.encode()).hexdigest())
tree = ast.parse(text)
node = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "SituAndMul")
seg = ast.get_source_segment(text, node)
print("---- shipped source, executed verbatim ----")
print(seg)
print("-------------------------------------------")
ns = {"nn": nn, "torch": torch}
exec(compile(seg, SRC, "exec"), ns)
SituAndMul = ns["SituAndMul"]

cfg = json.load(open(f"{MODEL}/config.json"))
tc = cfg.get("text_config", cfg)
beta, linear_beta = tc["activation_situ_beta"], tc["activation_situ_linear_beta"]
print("config:", tc["hidden_act"], "beta", beta, "linear_beta", linear_beta)
act = SituAndMul(beta=beta, linear_beta=linear_beta)

z = np.load(NPZ)
print("npz sha256:", hashlib.sha256(open(NPZ, "rb").read()).hexdigest())
fails = 0

for name, x, want in [
    ("diag", np.stack([z["sweep_x_f64"], z["sweep_x_f64"]], -1), z["diag_y_f32"]),
    ("grid", np.stack([z["grid_gate_f64"], z["grid_up_f64"]], -1), z["grid_y_f32"]),
    ("rand", z["rand_x_f64"], z["rand_y_f32"]),
]:
    with torch.no_grad():
        got = act(torch.from_numpy(np.ascontiguousarray(x)).to(torch.float32)).numpy()
    same = got.shape == want.shape and bool((got.view(np.uint32) == want.view(np.uint32)).all())
    d = float(np.max(np.abs(got.astype(np.float64) - want.astype(np.float64))))
    print(f"  f32  {name:5s} bitwise={same} maxabs={d:.3e}")
    fails += 0 if same else 1

for name, xb, wb in [
    ("diag", z["diag_x_bf16_bits"], z["diag_y_bf16_bits"]),
    ("grid", z["grid_x_bf16_bits"], z["grid_y_bf16_bits"]),
    ("rand", z["rand_x_bf16_bits"], z["rand_y_bf16_bits"]),
]:
    with torch.no_grad():
        got = act(torch.from_numpy(np.ascontiguousarray(xb)).view(torch.bfloat16))
    got = got.view(torch.uint16).numpy()
    same = got.shape == wb.shape and bool((got == wb).all())
    print(f"  bf16 {name:5s} bitwise={same}")
    fails += 0 if same else 1

print("RESULT:", "stored columns ARE the shipped class's output"
      if fails == 0 else f"{fails} column(s) DIFFER")
sys.exit(1 if fails else 0)
