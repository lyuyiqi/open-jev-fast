"""One eager forward_gtree on the example request, for Nsight Compute kernel selection (-k regex, -s/-c)."""
import json, os, sys
from pathlib import Path
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
import torch
import graph_patches  # noqa: F401
from jev.api import candidate_prompts, compile_request
from jev.model import DecisionModel
from fastmodel import FastQwen35
req = json.loads(Path(os.environ["OPEN_JEV_DIR"], "configs", "example-request.json").read_text())
recs = compile_request(req["state"], req["questions"])
model = DecisionModel.load(Path(os.environ["OJ_CKPT"])); model.backbone = model.backbone.merge_and_unload(); tok = model.tokenizer
fast = FastQwen35(model.backbone)
seqs, groups = [], []
for ri, r in enumerate(recs):
    for p in candidate_prompts(r):
        seqs.append(tok(tok.apply_chat_template([{"role": "user", "content": p}], tokenize=False, add_generation_prompt=True, enable_thinking=False))["input_ids"]); groups.append(ri)
ids, lay = FastQwen35.gtree_build(seqs, groups, tok.pad_token_id)
with torch.inference_mode():
    for _ in range(int(os.environ.get("NCU_ITERS", "2"))): fast.forward_gtree(ids, lay)
    torch.cuda.synchronize()
print("NCU MODEL RUN DONE", flush=True)
