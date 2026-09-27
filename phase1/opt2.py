"""Round 2: bucket length sweep for the CUDA graph, GPU kernel breakdown, batched graph throughput."""
import json, math, sys, time, statistics
from pathlib import Path
import torch, torch.nn.functional as F
from torch.profiler import profile, ProfilerActivity
import os; sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src")); import graph_patches
import transformers.models.qwen3_5.modeling_qwen3_5 as mq
from jev.api import compile_request, candidate_prompts
from jev.metrics import softmax
from jev.model import DecisionModel
CKPT = Path(os.environ["OJ_CKPT"])
req = json.loads(Path(os.environ["OPEN_JEV_DIR"], "configs", "example-request.json").read_text())
records = compile_request(req["state"], req["questions"])
T = json.loads((CKPT / "temperature.json").read_text())["temperature"]
sync = torch.cuda.synchronize
model = DecisionModel.load(CKPT)
with torch.inference_mode():
    base_logits = model(records)
base = [softmax(l.float().cpu().tolist(), temperature=T) for l in base_logits]
model.backbone = model.backbone.merge_and_unload()
def fused_forward(self, x):
    w = getattr(self, "_w1p", None)
    if w is None: w = self._w1p = (1.0 + self.weight.float()).contiguous()
    return F.rms_norm(x.float(), (x.shape[-1],), w, self.eps).type_as(x)
mq.Qwen3_5RMSNorm.forward = fused_forward
core, tok, dev = model.backbone, model.tokenizer, next(model.backbone.parameters()).device
counts = [len(candidate_prompts(r)) for r in records]
def encode(recs):
    ps = [tok.apply_chat_template([{"role": "user", "content": p}], tokenize=False, add_generation_prompt=True,
          enable_thinking=False) for r in recs for p in candidate_prompts(r)]
    return tok(ps, padding=True, truncation=False, return_tensors="pt")
E1 = encode(records); Lreal = E1["input_ids"].shape[1]
print(f"Lreal={Lreal}", flush=True)

class Graphed:
    def __init__(self, B, L):
        self.B, self.L = B, L
        self.ids = torch.full((B, L), tok.pad_token_id, dtype=torch.long, device=dev)
        self.mask = torch.zeros((B, L), dtype=torch.long, device=dev)
        self.last = torch.zeros(B, dtype=torch.long, device=dev)
        self.ar = torch.arange(B, device=dev)
        with torch.inference_mode():
            st = torch.cuda.Stream(); st.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(st):
                for _ in range(3): self.body()
            torch.cuda.current_stream().wait_stream(st); sync()
            self.g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(self.g):
                self.out = self.body()
        sync()
    def body(self):
        h = core(input_ids=self.ids, attention_mask=self.mask, use_cache=False, return_dict=True).last_hidden_state
        return model.head(h[self.ar, self.last].float()).squeeze(-1)
    def run(self, e):
        b, l = e["input_ids"].shape
        self.ids.fill_(tok.pad_token_id); self.mask.zero_()
        self.ids[:b, :l].copy_(e["input_ids"]); self.mask[:b, :l].copy_(e["attention_mask"])
        self.last[:b].copy_(e["attention_mask"].sum(-1) - 1)
        self.g.replay()
        return self.out

def to_probs(scores, recs):
    out, off = [], 0
    for r in recs:
        c = len(candidate_prompts(r)); v = scores[off:off + c]
        if r["kind"] == "noul": v = torch.stack([torch.zeros_like(v[0]), v[0]])
        out.append(softmax(v.float().cpu().tolist(), temperature=T)); off += c
    return out
def p50(fn, reps=30, warm=5):
    for _ in range(warm): fn()
    ts = []
    for _ in range(reps):
        sync(); t = time.perf_counter(); fn(); sync(); ts.append((time.perf_counter() - t) * 1e3)
    return statistics.median(ts)
md = lambda a, b: max(abs(x - y) for pa, pb in zip(a, b) for x, y in zip(pa, pb))

# ---- A. bucket-length sweep (single request, B=7) ----
B1 = E1["input_ids"].shape[0]
for L in sorted({Lreal, 88, 96, 112, 128}):
    if L < Lreal: continue
    g = Graphed(B1, L)
    fn = lambda: g.run(encode(records))
    with torch.inference_mode():
        pr = to_probs(fn(), records); t = p50(fn)
    print(f"BUCKET L={L:4d} p50 {t:6.2f} ms  {1000/t:6.2f} req/s  maxdiff {md(base, pr):.2e}", flush=True)
    del g; torch.cuda.empty_cache()

# ---- B. where GPU time goes (eager, same patches, L=Lreal) ----
with torch.inference_mode():
    e = {k: v.to(dev) for k, v in E1.items()}
    for _ in range(3): core(**e, use_cache=False)
    sync()
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        for _ in range(5): core(**e, use_cache=False)
        sync()
kern = [x for x in prof.key_averages() if x.device_type.name == "CUDA" or "CUDA" in str(x.device_type)]
tot = sum(x.self_device_time_total for x in kern) / 5 / 1e3
print(f"GPU kernel time per request (eager, L={Lreal}): {tot:.1f} ms", flush=True)
for x in sorted(kern, key=lambda x: -x.self_device_time_total)[:12]:
    print(f"KERN {x.key[:70]:70s} {x.self_device_time_total/5/1e3:6.2f} ms  n={x.count//5}", flush=True)

# ---- C. batched graph throughput (n requests per batch, bucket = tightest) ----
L = Lreal
for n in (1, 2, 4, 8, 16):
    recs = records * n
    en = encode(recs)
    g = Graphed(en["input_ids"].shape[0], L)
    fn = lambda: g.run(en)
    with torch.inference_mode():
        pr = to_probs(fn(), recs); t = p50(fn, reps=15, warm=3)
    print(f"BATCH n={n:3d} graph p50 {t:7.2f} ms  {n*1000/t:7.2f} req/s  maxdiff_first {md(base, pr[:len(records)]):.2e}"
          f"  mem_GB {torch.cuda.max_memory_allocated()/1e9:.1f}", flush=True)
    del g; torch.cuda.empty_cache()
print("=== OPT2 DONE ===", flush=True)
