"""Open-Jev-27B single-request latency optimization ladder (outputs checked against baseline at each step).
Steps: base -> merge LoRA -> fused RMSNorm -> manual CUDA-graph capture of the backbone at a fixed (B, L) bucket.
Run inside the allocation with CUDA_VISIBLE_DEVICES set to the granted GPU."""
import json, os, time, statistics, math
from pathlib import Path
import torch, torch.nn.functional as F
from jev.api import compile_request, candidate_prompts
from jev.metrics import softmax
from jev.model import DecisionModel

CKPT = Path(os.environ["OJ_CKPT"])
REQ = Path(os.environ["OPEN_JEV_DIR"], "configs", "example-request.json")
req = json.loads(REQ.read_text()); records = compile_request(req["state"], req["questions"])
T = json.loads((CKPT / "temperature.json").read_text())["temperature"]
sync = torch.cuda.synchronize
model = DecisionModel.load(CKPT)
probs = lambda ls: [softmax(l.float().cpu().tolist(), temperature=T) for l in ls]
maxdiff = lambda a, b: max(abs(x - y) for pa, pb in zip(a, b) for x, y in zip(pa, pb))
RESULTS = {}

def p50(fn, reps=30, warm=5):
    for _ in range(warm): fn()
    ts = []
    for _ in range(reps):
        sync(); t = time.perf_counter(); fn(); sync(); ts.append((time.perf_counter() - t) * 1e3)
    return statistics.median(ts), sorted(ts)[int(.95 * len(ts)) - 1]

def report(name, fn, base=None):
    with torch.inference_mode():
        out = probs(fn())
        a, b = p50(fn)
    d = maxdiff(base, out) if base else 0.0
    RESULTS[name] = {"p50_ms": round(a, 2), "p95_ms": round(b, 2), "maxdiff": d, "probs": out}
    print(f"STEP {name:28s} p50 {a:7.2f} ms  p95 {b:7.2f} ms  {1000/a:6.2f} req/s  maxdiff_vs_base {d:.2e}", flush=True)
    return out

# ---- 0. baseline ----
base = report("0_baseline", lambda: model(records))

# ---- 1. merge LoRA into base weights ----
if hasattr(model.backbone, "merge_and_unload"):
    model.backbone = model.backbone.merge_and_unload()
report("1_merge_lora", lambda: model(records), base)

# ---- 2. fused RMSNorm (same math: fp32 norm * (1+w) then cast) ----
import transformers.models.qwen3_5.modeling_qwen3_5 as mq
def fused_forward(self, x):
    w = getattr(self, "_w1p", None)
    if w is None:
        w = self._w1p = (1.0 + self.weight.float()).contiguous()
    return F.rms_norm(x.float(), (x.shape[-1],), w, self.eps).type_as(x)
mq.Qwen3_5RMSNorm.forward = fused_forward
report("2_fused_rmsnorm", lambda: model(records), base)

# ---- 2b. remove CPU-sync shortcuts (needed for graph capture; also measured in eager) ----
import sys; import os; sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src")); import graph_patches
report("2b_no_sync_patches", lambda: model(records), base)

# ---- 3. manual CUDA graph over the backbone at a fixed bucket ----
core = model.backbone
tok = model.tokenizer
def encode(recs):
    prompts = [tok.apply_chat_template([{"role": "user", "content": p}], tokenize=False, add_generation_prompt=True,
               enable_thinking=False) for r in recs for p in candidate_prompts(r)]
    return tok(prompts, padding=True, truncation=False, return_tensors="pt")
enc = encode(records)
B, Lreal = enc["input_ids"].shape
L = int(2 ** math.ceil(math.log2(max(Lreal, 16))))            # bucket length
print(f"bucket B={B} Lreal={Lreal} -> L={L}", flush=True)
dev = next(core.parameters()).device
s_ids = torch.full((B, L), tok.pad_token_id, dtype=torch.long, device=dev)
s_mask = torch.zeros((B, L), dtype=torch.long, device=dev)
s_last = torch.zeros(B, dtype=torch.long, device=dev)
def load_static(e):
    b, l = e["input_ids"].shape
    s_ids.fill_(tok.pad_token_id); s_mask.zero_()
    s_ids[:b, :l].copy_(e["input_ids"], non_blocking=True); s_mask[:b, :l].copy_(e["attention_mask"], non_blocking=True)
    s_last[:b].copy_(e["attention_mask"].sum(-1) - 1, non_blocking=True)
def body():
    h = core(input_ids=s_ids, attention_mask=s_mask, use_cache=False, return_dict=True).last_hidden_state
    return h[torch.arange(B, device=dev), s_last]
graph_ok = False
try:
    with torch.inference_mode():
        load_static(enc)
        st = torch.cuda.Stream(); st.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(st):
            for _ in range(3): body()
        torch.cuda.current_stream().wait_stream(st); sync()
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            s_hidden = body()
        sync(); graph_ok = True
except Exception as e:
    print(f"STEP 3_cuda_graph CAPTURE FAILED: {type(e).__name__}: {str(e)[:400]}", flush=True)

if graph_ok:
    counts = [len(candidate_prompts(r)) for r in records]
    def graphed():
        e = encode(records); load_static(e); g.replay()
        scores = model.head(s_hidden.float().to(model.head.weight.device)).squeeze(-1)
        out, off = [], 0
        for r, c in zip(records, counts):
            v = scores[off:off + c]
            if r["kind"] == "noul": v = torch.stack([torch.zeros_like(v[0]), v[0]])
            out.append(v); off += c
        return out
    report("3_cuda_graph", graphed, base)

Path(os.path.dirname(os.path.abspath(__file__)), "results.json").write_text(json.dumps(RESULTS, indent=1))
print("=== OPT DONE ===", flush=True)
