"""Speed of FastQwen35 (custom CUDA kernels) under a CUDA graph at the exact length; single + batched; kernel breakdown."""
import sys, json, time, statistics
from pathlib import Path
import os; sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
import torch
from torch.profiler import profile, ProfilerActivity
import graph_patches  # noqa
from jev.api import compile_request, candidate_prompts
from jev.metrics import softmax
from jev.model import DecisionModel
from fastmodel import FastQwen35
CKPT = Path(os.environ["OJ_CKPT"])
req = json.loads(Path(os.environ["OPEN_JEV_DIR"], "configs", "example-request.json").read_text())
records = compile_request(req["state"], req["questions"])
T0 = json.loads((CKPT / "temperature.json").read_text())["temperature"]
model = DecisionModel.load(CKPT)
with torch.inference_mode():
    base = [softmax(l.float().cpu().tolist(), temperature=T0) for l in model(records)]   # original, unmerged reference
model.backbone = model.backbone.merge_and_unload()
fast = FastQwen35(model.backbone); tok = model.tokenizer; dev = torch.device("cuda")
sync = torch.cuda.synchronize
def encode(recs):
    ps = [tok.apply_chat_template([{"role": "user", "content": p}], tokenize=False, add_generation_prompt=True, enable_thinking=False)
          for r in recs for p in candidate_prompts(r)]
    return tok(ps, padding=True, return_tensors="pt")
def to_probs(scores, recs):
    out, off = [], 0
    for r in recs:
        c = len(candidate_prompts(r)); v = scores[off:off + c]
        if r["kind"] == "noul": v = torch.stack([torch.zeros_like(v[0]), v[0]])
        out.append(softmax(v.float().cpu().tolist(), temperature=T0)); off += c
    return out
md = lambda a, b: max(abs(x - y) for pa, pb in zip(a, b) for x, y in zip(pa, pb))
def p50(fn, reps=30, warm=5):
    for _ in range(warm): fn()
    ts = []
    for _ in range(reps):
        sync(); t = time.perf_counter(); fn(); sync(); ts.append((time.perf_counter() - t) * 1e3)
    return statistics.median(ts)
class Graphed:
    def __init__(self, e):
        B, L = e["input_ids"].shape
        self.ids = e["input_ids"].to(dev).clone(); self.mask = e["attention_mask"].to(dev).clone()
        self.last = (self.mask.sum(-1) - 1); self.ar = torch.arange(B, device=dev)
        with torch.inference_mode():
            st = torch.cuda.Stream(); st.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(st):
                for _ in range(3): self.body()
            torch.cuda.current_stream().wait_stream(st); sync()
            self.g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(self.g): self.out = self.body()
        sync()
    def body(self):
        h = fast(input_ids=self.ids, attention_mask=self.mask).last_hidden_state
        return model.head(h[self.ar, self.last].float()).squeeze(-1)
    def run(self, e):
        self.ids.copy_(e["input_ids"]); self.mask.copy_(e["attention_mask"]); self.last.copy_(e["attention_mask"].sum(-1) - 1)
        self.g.replay(); return self.out
from fastmodel import K
from fla.modules.l2norm import l2norm_fwd
def cmpk(name, a, b):
    d = (a.float() - b.float()).abs(); print(f"V2CHECK {name:28s} maxabs {d.max():.3e}  bitexact {100*(d==0).float().mean():.2f}%", flush=True)
with torch.inference_mode():
    e1 = encode(records)
    # ---- v2 kernels vs validated v1 kernels on real layer-0 / layer-3 activations ----
    ids0, m0 = e1["input_ids"].to(dev), e1["attention_mask"].to(dev); B0, T0_ = ids0.shape
    core = model.backbone
    x0 = core.layers[0].input_layernorm(core.embed_tokens(ids0)).reshape(B0*T0_, -1).contiguous()
    d0 = fast.L[0]; proj0 = torch.nn.functional.linear(x0, d0.w_in)
    a1 = K.linattn_prep(proj0, d0.convw, d0.A_log, d0.dt_bias, T0_, d0.KD, d0.VD, d0.HV, d0.HK, d0.HD)
    a2 = K.linattn_prep2(proj0, d0.convw, d0.A_log, d0.dt_bias, B0, T0_, d0.KD, d0.VD, d0.HV, d0.HK, d0.HD, True)
    cmpk("prep2 q vs l2norm_fwd(prep1 q)", a2[0], l2norm_fwd(a1[0].view(-1, 128))[0].view_as(a2[0]))
    cmpk("prep2 k vs l2norm_fwd(prep1 k)", a2[1], l2norm_fwd(a1[1].view(-1, 128))[0].view_as(a2[1]))
    for nm, i in (("v", 2), ("g", 3), ("beta", 4)): cmpk("prep2 " + nm, a2[i], a1[i])
    x3 = core.layers[3].input_layernorm(core.embed_tokens(ids0)).reshape(B0*T0_, -1).contiguous()
    d3 = fast.L[3]; proj3 = torch.nn.functional.linear(x3, d3.w_in)
    pos = torch.arange(T0_, device=dev).view(1, 1, -1).expand(3, B0, -1)
    cs, sn = core.rotary_emb(x3.view(B0, T0_, -1), pos); cs = cs.reshape(B0*T0_, -1).contiguous(); sn = sn.reshape(B0*T0_, -1).contiguous()
    f1 = K.fullattn_prep(proj3, d3.qw1, d3.kw1, cs, sn, B0, T0_, 24, 4, 256, fast.eps)
    f2 = K.fullattn_prep2(proj3, d3.qw1, d3.kw1, cs, sn, B0, T0_, 24, 4, 256, fast.eps)
    for nm, i in (("q", 0), ("k", 1), ("v", 2), ("gate", 3)): cmpk("fprep2 " + nm, f2[i], f1[i])
    att = torch.randn(B0, 24, T0_, 256, device=dev).bfloat16()
    cmpk("gate_mul2", K.gate_mul2(att, f1[3], T0_, 24, 256), K.gate_mul(att, f1[3], T0_, 24, 256))
    # eager correctness vs original model
    h = fast(input_ids=e1["input_ids"].to(dev), attention_mask=e1["attention_mask"].to(dev)).last_hidden_state
    B = h.shape[0]; last = e1["attention_mask"].to(dev).sum(-1) - 1
    pe = to_probs(model.head(h[torch.arange(B, device=dev), last].float()).squeeze(-1), records)
    print(f"EAGER fast vs original model: prob maxdiff {md(base, pe):.2e}", flush=True)
    t_eager = p50(lambda: fast(input_ids=e1["input_ids"].to(dev), attention_mask=e1["attention_mask"].to(dev)))
    print(f"EAGER fast forward p50 {t_eager:.2f} ms", flush=True)
    # graph, single request
    g1 = Graphed(e1)
    pr = to_probs(g1.run(e1), records)
    t1 = p50(lambda: g1.run(encode(records)))
    t1r = p50(lambda: g1.run(e1))
    print(f"GRAPH fast single  p50 {t1:7.2f} ms incl. tokenize | {t1r:7.2f} ms replay-only ({1000/t1r:6.2f} req/s)  prob maxdiff vs original {md(base, pr):.2e}", flush=True)
    del g1; torch.cuda.empty_cache()
    for n in (4, 16):
        en = encode(records * n); gn = Graphed(en)
        prn = to_probs(gn.run(en), records * n)
        tn = p50(lambda: gn.run(en), reps=10, warm=3)
        print(f"GRAPH fast batch n={n:2d} p50 {tn:7.2f} ms  {n*1000/tn:6.2f} req/s  maxdiff {md(base, prn[:len(records)]):.2e}  mem_GB {torch.cuda.max_memory_allocated()/1e9:.1f}", flush=True)
        del gn; torch.cuda.empty_cache()
    # kernel breakdown (eager, single request)
    ids, mask = e1["input_ids"].to(dev), e1["attention_mask"].to(dev)
    for _ in range(3): fast(input_ids=ids, attention_mask=mask)
    sync()
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        for _ in range(5): fast(input_ids=ids, attention_mask=mask)
        sync()
    kern = [x for x in prof.key_averages() if "CUDA" in str(x.device_type)]
    tot = sum(x.self_device_time_total for x in kern) / 5 / 1e3
    print(f"GPU kernel time per request: {tot:.2f} ms  (kernels/req {sum(x.count for x in kern)//5})", flush=True)
    for x in sorted(kern, key=lambda x: -x.self_device_time_total)[:16]:
        print(f"KERN {x.key[:72]:72s} {x.self_device_time_total/5/1e3:6.2f} ms  n={x.count//5}", flush=True)
print("=== BENCH FAST DONE ===", flush=True)
