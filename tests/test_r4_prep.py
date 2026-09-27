"""Round 4 check: linattn_prep_map2 (GVA, preloaded rows) vs linattn_prep_map on real layer activations of the example
request, plus whole-model scores with OJ_GVA on/off (the flag is read at import, so both paths are called explicitly here)."""
import json, os, sys
from pathlib import Path
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
import torch
import graph_patches  # noqa: F401
from jev.api import candidate_prompts, compile_request
from jev.model import DecisionModel
import fastmodel
from fastmodel import FastQwen35, K
CKPT = Path(os.environ["OJ_CKPT"])
req = json.loads(Path(os.environ["OPEN_JEV_DIR"], "configs", "example-request.json").read_text())
recs = compile_request(req["state"], req["questions"])
model = DecisionModel.load(CKPT); model.backbone = model.backbone.merge_and_unload(); tok = model.tokenizer
fast = FastQwen35(model.backbone)
seqs, groups = [], []
for ri, r in enumerate(recs):
    for p in candidate_prompts(r):
        seqs.append(tok(tok.apply_chat_template([{"role": "user", "content": p}], tokenize=False, add_generation_prompt=True, enable_thinking=False))["input_ids"]); groups.append(ri)
ids, lay = FastQwen35.gtree_build(seqs, groups, tok.pad_token_id)

def gpu_us(fn, per=20, reps=20):
    """GPU time per call: `per` calls captured in one CUDA graph (no CPU launch cost), replayed `reps` times."""
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
    rowmask = lay.rowmask
    h, hn = K.add_rmsnorm(x, None, fast.in_w1[0], rowmask, fast.eps)
    ok_all = True
    for li in [i for i, t in enumerate(fast.layer_types) if t == "linear_attention"][:3]:
        d = fast.L[li]
        proj = fast._mm(hn, d.w_in)
        a = K.linattn_prep_map(proj, lay.src, d.convw, d.A_log, d.dt_bias, d.KD, d.VD, d.HV, d.HK, d.HD)
        b = K.linattn_prep_map2(proj, lay.src, d.convw, d.A_log, d.dt_bias, d.KD, d.VD, d.HV, d.HK, d.HD, 4)
        R = a[0].shape[0]; rep = d.HV // d.HK
        qa = a[0].view(R, d.HV, d.HD)[:, ::rep]; ka = a[1].view(R, d.HV, d.HD)[:, ::rep]
        # every replica of a head must equal the first one (sanity of the old layout)
        same_rep = all(torch.equal(a[0].view(R, d.HV, d.HD)[:, r::rep], qa) for r in range(rep))
        eq = [torch.equal(qa, b[0].view(R, d.HK, d.HD)), torch.equal(ka, b[1].view(R, d.HK, d.HD)),
              torch.equal(a[2], b[2]), torch.equal(a[3], b[3]), torch.equal(a[4], b[4])]
        ok_all &= all(eq) and same_rep
        ta = gpu_us(lambda: K.linattn_prep_map(proj, lay.src, d.convw, d.A_log, d.dt_bias, d.KD, d.VD, d.HV, d.HK, d.HD))
        tn = {t: gpu_us(lambda: K.linattn_prep_map2(proj, lay.src, d.convw, d.A_log, d.dt_bias, d.KD, d.VD, d.HV, d.HK, d.HD, t)) for t in (1, 2, 4, 8)}
        eqt = all(all(torch.equal(x, y) for x, y in zip(b, K.linattn_prep_map2(proj, lay.src, d.convw, d.A_log, d.dt_bias, d.KD, d.VD, d.HV, d.HK, d.HD, t))) for t in (1, 2, 8))
        ok_all &= eqt
        c = K.linattn_prep_dedup(proj, lay.src, lay.rep_ptr, lay.rep_pos, d.convw, d.A_log, d.dt_bias, d.KD, d.VD, d.HV, d.HK, d.HD)
        real = (lay.src.reshape(-1) >= 0)
        eqd = [torch.equal(b[0][real], c[0][real]), torch.equal(b[1][real], c[1][real]), torch.equal(b[2][real], c[2][real]),
               torch.equal(b[3], c[3]), torch.equal(b[4], c[4])]
        ok_all &= all(eqd)
        td = gpu_us(lambda: K.linattn_prep_dedup(proj, lay.src, lay.rep_ptr, lay.rep_pos, d.convw, d.A_log, d.dt_bias, d.KD, d.VD, d.HV, d.HK, d.HD))
        c2 = K.linattn_prep_dedup2(proj, lay.hist, lay.rep_ptr, lay.rep_pos, lay.L2, d.convw, d.A_log, d.dt_bias, d.KD, d.VD, d.HV, d.HK, d.HD, False)
        eqd2 = [torch.equal(x_, y_) for x_, y_ in zip(c, c2)]
        ok_all &= all(eqd2)
        td2 = gpu_us(lambda: K.linattn_prep_dedup2(proj, lay.hist, lay.rep_ptr, lay.rep_pos, lay.L2, d.convw, d.A_log, d.dt_bias, d.KD, d.VD, d.HV, d.HK, d.HD, False))
        print(f"DEDUP layer {li}: real-slot q,k,v and all g,beta equal to map2 {eqd}; dedup2 == dedup {eqd2}  dedup {td:.1f} us  dedup2 {td2:.1f} us  (real slots {int(real.sum())} of {real.numel()}, packed rows {lay.N})", flush=True)
        print(f"PREP layer {li}: q,k,v,g,beta bit-identical {eq} (replicas consistent {same_rep}, all TCH equal {eqt})  old {ta:.1f} us  new " +
              " ".join(f"TCH{t}={v:.1f}" for t, v in tn.items()) + " us", flush=True)
    outs = {}
    for name, prep, gva in (("map", "map", False), ("map2", "map2", True), ("dedup", "dedup", True)):
        fastmodel.PREP, fastmodel.GVA = prep, gva
        outs[name] = model.head(fast.forward_gtree(ids, lay).float()).squeeze(-1)
    same = all(torch.equal(outs["map"], outs[n]) for n in ("map2", "dedup"))
    print(f"MODEL scores map vs map2 vs dedup: bit-identical {same}  " +
          " ".join(f"{n}:max|diff|={(outs['map'] - outs[n]).abs().max().item():.3e}" for n in ("map2", "dedup")), flush=True)
    print("RESULT", "PASS" if ok_all and same else "FAIL", flush=True)
