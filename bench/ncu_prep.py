"""Run each linear-attention prep kernel variant a few times on real activations, for Nsight Compute."""
import json, os, sys
from pathlib import Path
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
import torch
import graph_patches  # noqa: F401
from jev.api import candidate_prompts, compile_request
from jev.model import DecisionModel
from fastmodel import FastQwen35, K
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
    x = fast.embed(ids).reshape(-1, fast.H).contiguous()
    h, hn = K.add_rmsnorm(x, None, fast.in_w1[0], lay.rowmask, fast.eps)
    d = fast.L[0]; proj = fast._mm(hn, d.w_in); torch.cuda.synchronize()
    for _ in range(2):
        K.linattn_prep_map(proj, lay.src, d.convw, d.A_log, d.dt_bias, d.KD, d.VD, d.HV, d.HK, d.HD)
        K.linattn_prep_map2(proj, lay.src, d.convw, d.A_log, d.dt_bias, d.KD, d.VD, d.HV, d.HK, d.HD, 1)
        K.linattn_prep_dedup(proj, lay.src, lay.rep_ptr, lay.rep_pos, d.convw, d.A_log, d.dt_bias, d.KD, d.VD, d.HV, d.HK, d.HD)
    torch.cuda.synchronize()
print("NCU PREP RUN DONE", flush=True)
