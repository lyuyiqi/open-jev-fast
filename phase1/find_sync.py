"""Locate CPU-sync points inside the backbone forward (these break CUDA-graph capture)."""
import json, os, math, traceback
from pathlib import Path
import torch, torch.nn.functional as F
from jev.api import compile_request, candidate_prompts
from jev.model import DecisionModel
import transformers.models.qwen3_5.modeling_qwen3_5 as mq
import sys; import os; sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src")); import graph_patches
CKPT = Path(os.environ["OJ_CKPT"])
req = json.loads(Path(os.environ["OPEN_JEV_DIR"], "configs", "example-request.json").read_text())
records = compile_request(req["state"], req["questions"])
model = DecisionModel.load(CKPT); model.backbone = model.backbone.merge_and_unload()
core = model.backbone; tok = model.tokenizer
prompts = [tok.apply_chat_template([{"role": "user", "content": p}], tokenize=False, add_generation_prompt=True, enable_thinking=False)
           for r in records for p in candidate_prompts(r)]
e = tok(prompts, padding="max_length", max_length=128, return_tensors="pt")
ids, mask = e["input_ids"].cuda(), e["attention_mask"].cuda()
with torch.inference_mode():
    core(input_ids=ids, attention_mask=mask, use_cache=False); torch.cuda.synchronize()   # warm (autotune)
    for attempt in range(6):
        torch.cuda.set_sync_debug_mode("error")
        try:
            core(input_ids=ids, attention_mask=mask, use_cache=False)
            torch.cuda.set_sync_debug_mode("default"); print("NO SYNC FOUND in forward", flush=True); break
        except Exception as ex:
            torch.cuda.set_sync_debug_mode("default")
            tb = traceback.extract_tb(ex.__traceback__)
            print(f"SYNC #{attempt}: {type(ex).__name__}: {str(ex)[:200]}", flush=True)
            for fr in tb[-6:]:
                print(f"    {fr.filename.split('site-packages/')[-1]}:{fr.lineno} {fr.name}: {fr.line}", flush=True)
            break
print("=== FIND DONE ===", flush=True)
