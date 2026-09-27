"""Host-side cost of building the prefix-tree layout (gtree_build) for the example request, with and without the packed
visibility bits used by the fused attention kernel."""
import json, os, sys, time
from pathlib import Path
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
import torch
from jev.api import candidate_prompts, compile_request
from transformers import AutoTokenizer
import fastmodel
from fastmodel import FastQwen35
cfg = json.loads((Path(os.environ["OJ_CKPT"]) / "model.json").read_text())
tok = AutoTokenizer.from_pretrained(cfg["model_id"], revision=cfg["revision"])
req = json.loads(Path(os.environ["OPEN_JEV_DIR"], "configs", "example-request.json").read_text())
recs = compile_request(req["state"], req["questions"])
seqs, groups = [], []
for ri, r in enumerate(recs):
    for p in candidate_prompts(r):
        seqs.append(tok(tok.apply_chat_template([{"role": "user", "content": p}], tokenize=False, add_generation_prompt=True, enable_thinking=False))["input_ids"]); groups.append(ri)
import importlib.util
# REF_FASTMODEL: fastmodel.py of another checkout (e.g. the previous commit); its gtree_build is the reference layout
old_src = open(os.environ["REF_FASTMODEL"]).read()
ns = {}
import types, fastmodel as fm_new
m_old = types.ModuleType("fm_old"); m_old.__dict__.update({k: v for k, v in fm_new.__dict__.items() if k in ("torch", "np", "SimpleNamespace", "K", "F")})
i0 = old_src.index("    def gtree_build("); i1 = old_src.index("    @torch.no_grad()\n    def forward_gtree", i0)
body = "import numpy as np, torch\nfrom types import SimpleNamespace\nclass _O:\n    @staticmethod\n    def _pack_bits(vis):\n        return fm_new.K.pack_bits(vis.contiguous())\n    @staticmethod\n" + old_src[i0:i1]
g = {"fm_new": fm_new}; exec(body + "\nFastQwen35 = _O\n", g)
ids_o, lay_o = g["_O"].gtree_build(seqs, groups, tok.pad_token_id)
ids_n, lay_n = FastQwen35.gtree_build(seqs, groups, tok.pad_token_id)
same = [torch.equal(ids_o, ids_n)] + [(f, torch.equal(getattr(lay_o, f), getattr(lay_n, f)) and getattr(lay_o, f).dtype == getattr(lay_n, f).dtype) for f in ("pos", "amask", "rowmask", "amask_add", "vbits", "src", "inv", "lastidx", "rep_ptr", "rep_pos", "hist", "canon")]
print("BUILD new layout == old layout:", same, flush=True)
orig = FastQwen35._pack_bits
from fastmodel import K
seqs_t = seqs
ids_, lay_ = FastQwen35.gtree_build(seqs, groups, tok.pad_token_id)
N = lay_.N; vis = torch.rand(N, N, device="cuda") > 0.5
w = (vis.view(N, N // 32, 32).to(torch.int64) << torch.arange(32, device=vis.device)).sum(-1)
ref = (w - (w >= 2 ** 31).to(torch.int64) * 2 ** 32).to(torch.int32)
print("BUILD pack_bits kernel == torch packing:", torch.equal(K.pack_bits(vis), ref), flush=True)
for name, fn in (("old", g["_O"].gtree_build), ("new", FastQwen35.gtree_build)):
    for _ in range(20): fn(seqs, groups, tok.pad_token_id)
    torch.cuda.synchronize(); t0 = time.perf_counter()
    for _ in range(200): fn(seqs, groups, tok.pad_token_id)
    torch.cuda.synchronize(); print(f"BUILD {name}: {(time.perf_counter() - t0) / 200 * 1e3:.3f} ms per gtree_build", flush=True)
import cProfile, pstats, io
pr = cProfile.Profile(); pr.enable()
for _ in range(200): FastQwen35.gtree_build(seqs, groups, tok.pad_token_id)
torch.cuda.synchronize(); pr.disable()
st = io.StringIO(); pstats.Stats(pr, stream=st).sort_stats("tottime").print_stats(14)
for line in st.getvalue().splitlines():
    if line.strip() and ("{" in line or ".py" in line): print("BUILDPROF", line[:150], flush=True)
