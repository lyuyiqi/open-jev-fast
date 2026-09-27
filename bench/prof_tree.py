import sys, json
from pathlib import Path
import os; sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
import torch
from torch.profiler import profile, ProfilerActivity
import graph_patches  # noqa
from jev.api import compile_request, candidate_prompts
from jev.model import DecisionModel
from fastmodel import FastQwen35
CKPT = Path(os.environ["OJ_CKPT"])
req = json.loads(Path(os.environ["OPEN_JEV_DIR"], "configs", "example-request.json").read_text())
recs = compile_request(req["state"], req["questions"])
model = DecisionModel.load(CKPT); model.backbone = model.backbone.merge_and_unload(); tok = model.tokenizer
fast = FastQwen35(model.backbone); dev = torch.device("cuda")
ps = [tok.apply_chat_template([{"role": "user", "content": p}], tokenize=False, add_generation_prompt=True, enable_thinking=False) for r in recs for p in candidate_prompts(r)]
seqs = [tok(p)["input_ids"] for p in ps]
Lp, mn = 0, min(map(len, seqs))
while Lp < mn - 1 and all(s[Lp] == seqs[0][Lp] for s in seqs): Lp += 1
S, Ls = len(seqs), max(len(s) - Lp for s in seqs)
suf = torch.full((S, Ls), tok.pad_token_id, dtype=torch.long); m = torch.zeros(S, Ls, dtype=torch.long)
for i, s in enumerate(seqs): suf[i, :len(s) - Lp] = torch.tensor(s[Lp:]); m[i, :len(s) - Lp] = 1
pre, suf, m = torch.tensor(seqs[0][:Lp], device=dev), suf.to(dev), m.to(dev)
lay = fast.tree_layout(pre, suf, m)
with torch.inference_mode():
    for _ in range(3): fast.forward_tree(pre, suf, lay)
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        for _ in range(5): fast.forward_tree(pre, suf, lay)
        torch.cuda.synchronize()
kern = [x for x in prof.key_averages() if "CUDA" in str(x.device_type)]
tot = sum(x.self_device_time_total for x in kern) / 5 / 1e3
print(f"TREE GPU kernel time per request: {tot:.2f} ms  kernels/req {sum(x.count for x in kern)//5}", flush=True)
cats = {"GEMM": 0, "FLA": 0, "mine": 0, "sdpa": 0, "other": 0}
for x in kern:
    k = x.key; t = x.self_device_time_total / 5 / 1e3
    c = "GEMM" if ("nvjet" in k or "gemm" in k.lower() or "cutlass" in k) else "FLA" if any(s in k for s in ("chunk", "recompute", "solve", "cumsum", "l2norm", "merge_16x16")) \
        else "mine" if k.split("(")[0].split("<")[0].strip().split(" ")[-1].endswith("_k") else "sdpa" if ("sdpa" in k or "fmha" in k or "attention" in k.lower()) else "other"
    cats[c] += t
print("CATEGORIES", {k: round(v, 2) for k, v in cats.items()}, flush=True)
for x in sorted(kern, key=lambda x: -x.self_device_time_total)[:22]:
    print(f"KERN {x.key[:80]:80s} {x.self_device_time_total/5/1e3:6.2f} ms n={x.count//5}", flush=True)
