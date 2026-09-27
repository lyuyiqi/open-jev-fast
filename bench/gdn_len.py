"""Fused GDN kernel time vs path length T on real layer-0 inputs of the example (T=96 = chunks of 64 + 32 rows):
how much the second, mostly-padding chunk costs."""
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
print("SEQ lens", [len(s) for s in seqs], "N", lay.N, "Nreal", lay.Nreal, "L2", lay.L2, flush=True)
def gpu_us(fn, per=10, reps=20):
    st = torch.cuda.Stream(); st.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(st):
        for _ in range(3): fn()
    torch.cuda.current_stream().wait_stream(st); torch.cuda.synchronize()
    gr = torch.cuda.CUDAGraph()
    with torch.cuda.graph(gr):
        for _ in range(per): fn()
    for _ in range(3): gr.replay()
    a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    torch.cuda.synchronize(); a.record()
    for _ in range(reps): gr.replay()
    b.record(); torch.cuda.synchronize()
    return a.elapsed_time(b) * 1e3 / (reps * per)
with torch.inference_mode():
    x = fast.embed(ids).reshape(-1, fast.H).contiguous()
    h, hn = K.add_rmsnorm(x, None, fast.in_w1[0], lay.rowmask, fast.eps)
    d = fast.L[0]; S, L2 = lay.S, lay.L2
    proj = fast._mm(hn, d.w_in)
    q, k, v, g, beta = K.linattn_prep_dedup2(proj, lay.hist, lay.rep_ptr, lay.rep_pos, lay.L2, d.convw, d.A_log, d.dt_bias, d.KD, d.VD, d.HV, d.HK, d.HD, False)
    a = [q.view(S, L2, d.HK, 128), k.view(S, L2, d.HK, 128), v.view(S, L2, d.HV, 128), g.view(S, L2, d.HV), beta.view(S, L2, d.HV)]
    def cut(T):
        if T <= L2: return [t[:, :T].contiguous() for t in a]
        reps = (T + L2 - 1) // L2
        return [torch.cat([t] * reps, 1)[:, :T].contiguous() for t in a]
    for T, nb in ((64, 7), (96, 7), (128, 7), (192, 7), (576, 4), (576, 7), (1024, 2), (2048, 1)):
        c = [t_[:nb] for t_ in cut(T)]
        o4 = K.gdn_fused4(*c, 128 ** -0.5); o9 = K.gdn_fused9(*c, 128 ** -0.5)
        print(f"GDNLEN T={T:4d} B={nb}: v4 {gpu_us(lambda: K.gdn_fused4(*c, 128 ** -0.5)):6.1f} us  v9 {gpu_us(lambda: K.gdn_fused9(*c, 128 ** -0.5)):6.1f} us  "
              f"v9 == v4 {torch.equal(o9, o4)}", flush=True)
        os.environ["OJ_GDN9_ONLY"] = "L"; tl = gpu_us(lambda: K.gdn_fused9(*c, 128 ** -0.5))
        os.environ["OJ_GDN9_ONLY"] = "D"; td = gpu_us(lambda: K.gdn_fused9(*c, 128 ** -0.5))
        os.environ["OJ_GDN9_ONLY"] = ""
        print(f"GDNLEN   v9 parts: local kernel {tl:6.1f} us  state kernel {td:6.1f} us", flush=True)
print("=== GDNLEN DONE ===", flush=True)
