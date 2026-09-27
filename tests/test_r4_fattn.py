"""Round 4 C8 check: fused tree attention + gate (fattn_tree) vs SDPA (cuDNN) + gate_mul3 on the real layer-3 activations
of the example request and of a JevBench task with a longer tree, GPU time (CUDA graph), and whole-model probabilities."""
import json, os, sys
from pathlib import Path
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
import torch, torch.nn.functional as F
import graph_patches  # noqa: F401
from jev.api import candidate_prompts, compile_request
from jev.metrics import softmax
from jev.model import DecisionModel
import fastmodel
from fastmodel import FastQwen35, K
CKPT = Path(os.environ["OJ_CKPT"])
T0 = json.loads((CKPT / "temperature.json").read_text())["temperature"]
req = json.loads(Path(os.environ["OPEN_JEV_DIR"], "configs", "example-request.json").read_text())
model = DecisionModel.load(CKPT); model.backbone = model.backbone.merge_and_unload(); tok = model.tokenizer
fast = FastQwen35(model.backbone)
cases = [("example", compile_request(req["state"], req["questions"]))]
sys.path.insert(0, os.environ["JEVBENCH_DIR"])
from jevbench.tasks import load_jsonl
from jevbench.adapters.typesafe import TypeSafeAdapter
ad = TypeSafeAdapter(endpoint="http://x", model="m", key_env=None)
want = [(300, 600), (600, 1024)]
for tier in ("hard", "original", "easy"):
    for t in load_jsonl(os.path.join(os.environ["JEVBENCH_DIR"], f"datasets/public/{tier}.jsonl")):
        if not want: break
        r_ = ad.build_request(t); recs = compile_request(r_["state"], r_["questions"])
        seqs_, groups_ = [], []
        for ri, r in enumerate(recs):
            for p in candidate_prompts(r):
                seqs_.append(tok(tok.apply_chat_template([{"role": "user", "content": p}], tokenize=False, add_generation_prompt=True, enable_thinking=False))["input_ids"]); groups_.append(ri)
        if len(seqs_) < 2: continue
        _, lay_ = FastQwen35.gtree_build(seqs_, groups_, tok.pad_token_id)
        for w_ in list(want):
            if w_[0] < lay_.N <= w_[1]: cases.append((t.id, recs)); want.remove(w_); break

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

am = lambda P: [max(range(len(p)), key=p.__getitem__) for p in P]
with torch.inference_mode():
    for name, recs in cases:
        seqs, groups = [], []
        for ri, r in enumerate(recs):
            for p in candidate_prompts(r):
                seqs.append(tok(tok.apply_chat_template([{"role": "user", "content": p}], tokenize=False, add_generation_prompt=True, enable_thinking=False))["input_ids"]); groups.append(ri)
        ids, lay = FastQwen35.gtree_build(seqs, groups, tok.pad_token_id)
        N = lay.N
        li = [i for i, t_ in enumerate(fast.layer_types) if t_ != "linear_attention"][0]
        d = fast.L[li]
        x = fast.embed(ids).reshape(-1, fast.H).contiguous()
        torch.manual_seed(0); proj = torch.randn(N, d.w_in.shape[0], device="cuda", dtype=torch.bfloat16)   # attention inputs: use the real layer's prep on random proj
        cos, sin = fast.rotary(x.view(1, N, -1), lay.pos.view(1, 1, N).expand(3, 1, N))
        cos = cos.reshape(N, -1).contiguous(); sin = sin.reshape(N, -1).contiguous()
        qt, kt, vt, gate = K.fullattn_prep2(proj, d.qw1, d.kw1, cos, sin, 1, N, d.HQ, d.HKV, d.D, fast.eps)
        pk = os.environ.get("PREP_REF")
        if pk:   # bit-exactness of the restructured prep vs the saved outputs of the previous build
            if os.path.exists(pk + f".{name}.pt"):
                ref_p = torch.load(pk + f".{name}.pt")
                print(f"FATTN {name} prep2 vs previous build: " + " ".join(str(torch.equal(a_, b_)) for a_, b_ in zip((qt, kt, vt, gate), ref_p)), flush=True)
            else:
                torch.save([t_.clone() for t_ in (qt, kt, vt, gate)], pk + f".{name}.pt")
        _, sig = K.act_tables(proj)
        ref = K.gate_mul3(F.scaled_dot_product_attention(qt, kt, vt, attn_mask=lay.amask_add, scale=d.scale, enable_gqa=True), gate, sig, N, d.HQ, d.D)
        vm = lay.rowmask.bool()
        t_ref = gpu_us(lambda: K.gate_mul3(F.scaled_dot_product_attention(qt, kt, vt, attn_mask=lay.amask_add, scale=d.scale, enable_gqa=True), gate, sig, N, d.HQ, d.D))
        y2 = K.fattn_tree2(proj, d.qw1, d.kw1, cos, sin, sig, lay.vbits, d.HQ, d.HKV, d.scale, fast.eps)
        df = (y2[vm].float() - ref[vm].float()).abs()
        t_prep = gpu_us(lambda: K.fullattn_prep2(proj, d.qw1, d.kw1, cos, sin, 1, N, d.HQ, d.HKV, d.D, fast.eps))
        print(f"FATTN {name} N={N} fused prep+attn+gate: max|diff| {df.max().item():.3e} rel-L2 {(df.norm() / ref[vm].float().norm()).item():.3e}  "
              f"{gpu_us(lambda: K.fattn_tree2(proj, d.qw1, d.kw1, cos, sin, sig, lay.vbits, d.HQ, d.HKV, d.scale, fast.eps)):.1f} us  vs prep {t_prep:.1f} + sdpa+gate {t_ref:.1f} us", flush=True)
        tsum = None
        for _ in range(10):
            tsx = K.fattn_tree2_ts(proj, d.qw1, d.kw1, cos, sin, sig, lay.vbits, d.HQ, d.HKV, d.scale, fast.eps).cpu().double()
            tsum = tsx if tsum is None else tsum + tsx
        tsx = tsum / 10
        base = tsx[:, 0:1]
        rel = (tsx - base) / 1e3
        rel[tsx == 0] = float("nan")
        print("FATTN2 stamps (us after CTA start; mean over CTAs): " + " ".join(f"{k}:{torch.nanmean(rel[:, k]).item():.2f}" for k in range(16)), flush=True)
        print(f"FATTN2 CTA start spread {((base.max() - base.min()) / 1e3).item():.2f} us, CTA duration max {torch.nanmean(rel[:, 15]).item():.2f} / {rel[:, 15].max().item():.2f} us", flush=True)
        y3 = K.fattn_tree3(qt, kt, vt, gate, sig, lay.vbits, d.scale)
        df = (y3[vm].float() - ref[vm].float()).abs()
        print(f"FATTN {name} N={N} 2 warps/head (tree3): max|diff| {df.max().item():.3e} rel-L2 {(df.norm() / ref[vm].float().norm()).item():.3e}  {gpu_us(lambda: K.fattn_tree3(qt, kt, vt, gate, sig, lay.vbits, d.scale)):.1f} us", flush=True)
        for G in (3, 6):
            y = K.fattn_tree(qt, kt, vt, gate, sig, lay.vbits, d.scale, G)
            df = (y[vm].float() - ref[vm].float()).abs()
            print(f"FATTN {name} N={N} G={G}: max|diff| {df.max().item():.3e} rel-L2 {(df.norm() / ref[vm].float().norm()).item():.3e} "
                  f"bitexact {(df == 0).float().mean().item() * 100:.1f}%  fused {gpu_us(lambda: K.fattn_tree(qt, kt, vt, gate, sig, lay.vbits, d.scale, G)):.1f} us  sdpa+gate {t_ref:.1f} us", flush=True)
        outs = {}
        for f in (0, 3, 4):
            fastmodel.FATTN = f
            sc = model.head(fast.forward_gtree(ids, lay).float()).squeeze(-1)
            P, off = [], 0
            for r in recs:
                c = len(candidate_prompts(r)); v = sc[off:off + c]
                if r["kind"] == "noul": v = torch.stack([torch.zeros_like(v[0]), v[0]])
                P.append(softmax(v.float().cpu().tolist(), temperature=T0)); off += c
            outs[f] = P
        for f in (3, 4):
            md = max(abs(a - b) for pa, pb in zip(outs[0], outs[f]) for a, b in zip(pa, pb))
            print(f"MODEL {name} probs FATTN={f} vs SDPA: max diff {md:.3e}  argmax same {am(outs[0]) == am(outs[f])}", flush=True)
print("=== FATTN TEST DONE ===", flush=True)
