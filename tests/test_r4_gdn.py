"""Round 4 C6 check: fused Gated DeltaNet kernel vs FLA (gdn_split) on real prep outputs of the example request, GPU time
(CUDA graph), and whole-model probabilities with the fused kernel vs FLA."""
import json, os, sys
from pathlib import Path
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
import torch
import graph_patches  # noqa: F401
from jev.api import candidate_prompts, compile_request
from jev.metrics import softmax
from jev.model import DecisionModel
import fastmodel
from fastmodel import FastQwen35, K, gdn_split
CKPT = Path(os.environ["OJ_CKPT"])
T0 = json.loads((CKPT / "temperature.json").read_text())["temperature"]
req = json.loads(Path(os.environ["OPEN_JEV_DIR"], "configs", "example-request.json").read_text())
recs = compile_request(req["state"], req["questions"])
model = DecisionModel.load(CKPT); model.backbone = model.backbone.merge_and_unload(); tok = model.tokenizer
fast = FastQwen35(model.backbone)
seqs, groups = [], []
for ri, r in enumerate(recs):
    for p in candidate_prompts(r):
        seqs.append(tok(tok.apply_chat_template([{"role": "user", "content": p}], tokenize=False, add_generation_prompt=True, enable_thinking=False))["input_ids"]); groups.append(ri)
ids, lay = FastQwen35.gtree_build(seqs, groups, tok.pad_token_id)

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

def to_probs(scores):
    out, off = [], 0
    for r in recs:
        c = len(candidate_prompts(r)); v = scores[off:off + c]
        if r["kind"] == "noul": v = torch.stack([torch.zeros_like(v[0]), v[0]])
        out.append(softmax(v.float().cpu().tolist(), temperature=T0)); off += c
    return out

with torch.inference_mode():
    x = fast.embed(ids).reshape(-1, fast.H).contiguous()
    h, hn = K.add_rmsnorm(x, None, fast.in_w1[0], lay.rowmask, fast.eps)
    S, L2 = lay.S, lay.L2
    for li in (0, 1, 2):
        d = fast.L[li]
        proj = fast._mm(hn, d.w_in)
        q, k, krep, v, g, beta = K.linattn_prep_dedup3(proj, lay.hist, lay.rep_ptr, lay.rep_pos, lay.L2, d.convw, d.A_log, d.dt_bias, d.KD, d.VD, d.HV, d.HK, d.HD)
        args = (q.view(S, L2, d.HK, d.HD), k.view(S, L2, d.HK, d.HD), krep.view(S, L2, d.HV, d.HD), v.view(S, L2, d.HV, d.HD), g.view(S, L2, d.HV), beta.view(S, L2, d.HV))
        ref = gdn_split(*args, d.HD ** -0.5)
        out = K.gdn_fused(args[0], args[1], args[3], args[4], args[5], d.HD ** -0.5)
        real = (lay.src.reshape(S, L2) >= 0)
        rf, of = ref[real].float(), out[real].float()
        diff = (rf - of).abs()
        rel = (diff.norm() / rf.norm()).item()
        t_ref = gpu_us(lambda: gdn_split(*args, d.HD ** -0.5))
        t_fused = gpu_us(lambda: K.gdn_fused(args[0], args[1], args[3], args[4], args[5], d.HD ** -0.5))
        out2 = K.gdn_fused2(args[0], args[1], args[3], args[4], args[5], d.HD ** -0.5)
        d2 = (out2[real].float() - of).abs().max().item()
        t_fused2 = gpu_us(lambda: K.gdn_fused2(args[0], args[1], args[3], args[4], args[5], d.HD ** -0.5))
        print(f"GDN2 layer {li}: v2 vs v1 max|diff| {d2:.3e}  fused2 {t_fused2:.1f} us", flush=True)
        out3 = K.gdn_fused3(args[0], args[1], args[3], args[4], args[5], d.HD ** -0.5)
        d3 = (out3[real].float() - rf).abs(); rel3 = (d3.norm() / rf.norm()).item()
        t_fused3 = gpu_us(lambda: K.gdn_fused3(args[0], args[1], args[3], args[4], args[5], d.HD ** -0.5))
        print(f"GDN3 layer {li}: v3 vs FLA max|diff| {d3.max().item():.3e} rel-L2 {rel3:.3e} bitexact {(d3 == 0).float().mean().item() * 100:.1f}%  fused3 {t_fused3:.1f} us", flush=True)
        out4 = K.gdn_fused4(args[0], args[1], args[3], args[4], args[5], d.HD ** -0.5)
        t_fused4 = gpu_us(lambda: K.gdn_fused4(args[0], args[1], args[3], args[4], args[5], d.HD ** -0.5))
        print(f"GDN4 layer {li}: v4 == v3 {torch.equal(out4[real], out3[real])}  fused4 {t_fused4:.1f} us", flush=True)
        out6 = K.gdn_fused6(args[0], args[1], args[3], args[4], args[5], d.HD ** -0.5)
        t_fused6 = gpu_us(lambda: K.gdn_fused6(args[0], args[1], args[3], args[4], args[5], d.HD ** -0.5))
        d6 = (out6[real].float() - out4[real].float()).abs().max().item()
        print(f"GDN6 layer {li}: v6 == v4 {torch.equal(out6[real], out4[real])} (max|diff| {d6:.3e})  fused6 {t_fused6:.1f} us  fused4 {t_fused4:.1f} us", flush=True)
        _, sig = K.act_tables(proj); zoff = 2 * d.KD + d.VD; sc = d.HD ** -0.5
        y_ref = K.gated_rmsnorm_inv2(out3, lay.inv, proj, d.normw, sig, d.HV, d.HD, zoff, d.norm_eps)
        y5 = K.gdn_fused5(args[0], args[1], args[3], args[4], args[5], sc, proj, zoff, d.normw, sig, lay.canon, lay.rowmask, d.norm_eps)
        vm = lay.rowmask.bool()
        y6 = K.gated_rmsnorm_inv2(out6, lay.inv, proj, d.normw, sig, d.HV, d.HD, zoff, d.norm_eps)
        y7 = K.gdn_fused7(args[0], args[1], args[3], args[4], args[5], sc, proj, zoff, d.normw, sig, lay.canon, lay.rowmask, d.norm_eps)
        y7b = K.gdn_fused7(args[0], args[1], args[3], args[4], args[5], sc, proj, zoff, d.normw, sig, lay.canon, lay.rowmask, d.norm_eps)
        t_6n = gpu_us(lambda: K.gated_rmsnorm_inv2(K.gdn_fused6(args[0], args[1], args[3], args[4], args[5], sc), lay.inv, proj, d.normw, sig, d.HV, d.HD, zoff, d.norm_eps))
        t_7 = gpu_us(lambda: K.gdn_fused7(args[0], args[1], args[3], args[4], args[5], sc, proj, zoff, d.normw, sig, lay.canon, lay.rowmask, d.norm_eps))
        os.environ["OJ_GDN7_ONLY"] = "1"
        o7 = K.gdn_fused7(args[0], args[1], args[3], args[4], args[5], sc, proj, zoff, d.normw, sig, lay.canon, lay.rowmask, d.norm_eps)
        t_7g = gpu_us(lambda: K.gdn_fused7(args[0], args[1], args[3], args[4], args[5], sc, proj, zoff, d.normw, sig, lay.canon, lay.rowmask, d.norm_eps))
        os.environ["OJ_GDN7_ONLY"] = "0"
        print(f"GDN7 layer {li}: persistent GDN-only == v6 {torch.equal(o7[real], out6[real])}  {t_7g:.1f} us (v6 grid {t_fused6:.1f} us)", flush=True)
        print(f"GDN7 layer {li}: v7 == v6+norm on real rows {torch.equal(y7[vm], y6[vm])}  repeat equal {torch.equal(y7, y7b)}  pad rows zero {bool((y7[~vm] == 0).all())}  v6+norm {t_6n:.1f} us  v7 {t_7:.1f} us", flush=True)
        d5 = (y5[vm].float() - y_ref[vm].float()).abs()
        t_4n = gpu_us(lambda: K.gated_rmsnorm_inv2(K.gdn_fused4(args[0], args[1], args[3], args[4], args[5], sc), lay.inv, proj, d.normw, sig, d.HV, d.HD, zoff, d.norm_eps))
        t_5 = gpu_us(lambda: K.gdn_fused5(args[0], args[1], args[3], args[4], args[5], sc, proj, zoff, d.normw, sig, lay.canon, lay.rowmask, d.norm_eps))
        print(f"GDN5 layer {li}: v5 vs v3+norm max|diff| {d5.max().item():.3e} rel-L2 {(d5.norm() / y_ref[vm].float().norm()).item():.3e} "
              f"bitexact {(d5 == 0).float().mean().item() * 100:.2f}%  pad rows zero {bool((y5[~vm] == 0).all())}  v4+norm {t_4n:.1f} us  v5 {t_5:.1f} us", flush=True)
        print(f"GDN layer {li}: max|diff| {diff.max().item():.3e}  mean|diff| {diff.mean().item():.3e}  rel-L2 {rel:.3e}  "
              f"bitexact {(diff == 0).float().mean().item() * 100:.1f}%  |ref|max {rf.abs().max().item():.3e}  FLA {t_ref:.1f} us  fused {t_fused:.1f} us", flush=True)
    outs = {}
    for name, flag, ver in (("fla", False, 3), ("v3", True, 3), ("v4", True, 4), ("v5", True, 5), ("v6", True, 6), ("v7", True, 7)):
        fastmodel.GDN_FUSED, fastmodel.GDN_VER = flag, ver
        outs[name] = to_probs(model.head(fast.forward_gtree(ids, lay).float()).squeeze(-1))
    am = lambda P: [max(range(len(p)), key=p.__getitem__) for p in P]
    for a_, b_ in (("fla", "v3"), ("v3", "v4"), ("v3", "v5"), ("fla", "v5"), ("v4", "v6"), ("v6", "v7")):
        md = max(abs(x - y) for pa, pb in zip(outs[a_], outs[b_]) for x, y in zip(pa, pb))
        print(f"MODEL probs {b_} vs {a_}: max diff {md:.3e}  argmax same {am(outs[a_]) == am(outs[b_])}", flush=True)
print("=== GDN TEST DONE ===", flush=True)
