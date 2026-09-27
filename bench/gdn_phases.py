"""Per-phase time of gdn6_item from globaltimer stamps (block-level critical path), real layer-0 inputs of the example."""
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
names = ["load+cumsum", "kkt", "solve || scale", "u/w MMA", "store w,u,VG0 + Aqk", "o0 + h1", "pp, qh", "o1"]
with torch.inference_mode():
    x = fast.embed(ids).reshape(-1, fast.H).contiguous()
    h, hn = K.add_rmsnorm(x, None, fast.in_w1[0], lay.rowmask, fast.eps)
    d = fast.L[0]; S, L2 = lay.S, lay.L2
    proj = fast._mm(hn, d.w_in)
    q, k, v, g, beta = K.linattn_prep_dedup2(proj, lay.hist, lay.rep_ptr, lay.rep_pos, lay.L2, d.convw, d.A_log, d.dt_bias, d.KD, d.VD, d.HV, d.HK, d.HD, False)
    a = (q.view(S, L2, d.HK, 128), k.view(S, L2, d.HK, 128), v.view(S, L2, d.HV, 128), g.view(S, L2, d.HV), beta.view(S, L2, d.HV), 128 ** -0.5)
    for _ in range(5): K.gdn_fused6(*a)
    acc = None
    for _ in range(20):
        ts = K.gdn_fused6ts(*a).cpu().double()
        st = torch.stack([ts[:, i] for i in list(range(8)) + [15]], 1)
        dur = (st[:, 1:] - st[:, :-1])
        acc = dur if acc is None else acc + dur
    acc /= 20
    tot = acc.sum(1)
    print(f"PHASE block total mean {tot.mean().item() / 1e3:.2f} us (min {tot.min().item() / 1e3:.2f}, max {tot.max().item() / 1e3:.2f}); kernel span {(ts[:, 15].max() - ts[:, 0].min()).item() / 1e3:.2f} us", flush=True)
    for i, n in enumerate(names):
        print(f"PHASE {n:24s} {acc[:, i].mean().item() / 1e3:6.2f} us  ({acc[:, i].mean().item() / tot.mean().item() * 100:4.1f}%)", flush=True)
    sub = []
    for _ in range(20):
        t2 = K.gdn_fused6ts(*a).cpu().double(); sub.append(t2)
    t2 = torch.stack(sub).mean(0)
    base = t2[:, 2]
    for i, n in ((8, "diag inverse + bf16 (warp 0)"), (9, "bar 1 (warps 0-5)"), (10, "merge L1 (warp 0)"), (11, "merge L2 (warp 0)"), (12, "merge L3 (warp 0)"), (13, "scale+zero done (warp 6)"), (3, "B7 (all)")):
        print(f"PHASE solve sub: {n:30s} at +{(t2[:, i] - base).mean().item() / 1e3:5.2f} us after B2", flush=True)
    st0 = ts[:, 0] - ts[:, 0].min()
    srt = torch.sort(st0 / 1e3).values
    print("PHASE sorted start times at ranks 0,100,140..152,290..300,330,335:", [round(srt[i].item(), 2) for i in [0, 100] + list(range(140, 153)) + list(range(290, 301)) + [330, 335]], flush=True)
    print("PHASE block start times (us) quantiles:", [round(x, 1) for x in torch.quantile(st0 / 1e3, torch.tensor([0, .25, .44, .5, .88, .9, 1.0], dtype=torch.float64)).tolist()], flush=True)
print("=== PHASES DONE ===", flush=True)
