"""Per-kernel GPU time for the example request on the serving path (two-level prefix tree, CUDA Graph replay).
Reports kernel time by category, the idle time between kernels (replay wall time minus kernel time), and the top kernels."""
import json, os, statistics, sys, time
from pathlib import Path
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
import torch
from torch.profiler import profile, ProfilerActivity
import graph_patches  # noqa: F401
from jev.api import candidate_prompts, compile_request
from jev.model import DecisionModel
from fastmodel import FastQwen35
CKPT = Path(os.environ["OJ_CKPT"])
req = json.loads(Path(os.environ["OPEN_JEV_DIR"], "configs", "example-request.json").read_text())
recs = compile_request(req["state"], req["questions"])
if os.environ.get("JEV_TASK"):   # profile a JevBench task instead of the example request
    sys.path.insert(0, os.environ["JEVBENCH_DIR"])
    from jevbench.tasks import load_jsonl
    from jevbench.adapters.typesafe import TypeSafeAdapter
    tk = [t for tier in ("original", "easy", "hard") for t in load_jsonl(os.path.join(os.environ["JEVBENCH_DIR"], f"datasets/public/{tier}.jsonl")) if t.id == os.environ["JEV_TASK"]][0]
    rq = TypeSafeAdapter(endpoint="http://x", model="m", key_env=None).build_request(tk)
    recs = compile_request(rq["state"], rq["questions"])
model = DecisionModel.load(CKPT); model.backbone = model.backbone.merge_and_unload(); tok = model.tokenizer
fast = FastQwen35(model.backbone); sync = torch.cuda.synchronize
seqs, groups = [], []
for ri, r in enumerate(recs):
    for p in candidate_prompts(r):
        seqs.append(tok(tok.apply_chat_template([{"role": "user", "content": p}], tokenize=False, add_generation_prompt=True, enable_thinking=False))["input_ids"]); groups.append(ri)
ids, lay = FastQwen35.gtree_build(seqs, groups, tok.pad_token_id)

def cat(k):
    if "nvjet" in k or "gemm" in k.lower() or "cutlass" in k or "cublas" in k.lower() or "bmm" in k.lower(): return "GEMM"
    if any(s in k for s in ("chunk", "recompute", "solve", "cumsum", "l2norm", "merge_16x16", "fwd_kernel", "wy_")): return "FLA"
    if "sdpa" in k or "fmha" in k or "attention" in k.lower() or "flash" in k.lower(): return "attn"
    if k.split("(")[0].split("<")[0].strip().split(" ")[-1].endswith("_k"): return "own"
    return "other"

with torch.inference_mode():
    body = lambda: model.head(fast.forward_gtree(ids, lay).float()).squeeze(-1)
    st = torch.cuda.Stream(); st.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(st):
        for _ in range(3): body()
    torch.cuda.current_stream().wait_stream(st); sync()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g): out = body()
    sync()
    for _ in range(10): g.replay()
    sync(); ts = []
    for _ in range(30):
        sync(); a = time.perf_counter(); g.replay(); sync(); ts.append((time.perf_counter() - a) * 1e3)
    wall = statistics.median(ts)
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        for _ in range(5): g.replay()
        sync()
kern = [x for x in prof.key_averages() if "CUDA" in str(x.device_type)]
tot = sum(x.self_device_time_total for x in kern) / 5 / 1e3
cats = {}
for x in kern: cats[cat(x.key)] = cats.get(cat(x.key), 0) + x.self_device_time_total / 5 / 1e3
print(f"EXAMPLE rows={lay.N} key={lay.key} replay wall p50 {wall:.2f} ms | kernel time {tot:.2f} ms | gaps {wall - tot:.2f} ms | kernels/replay {sum(x.count for x in kern) // 5}", flush=True)
print("CATEGORIES " + " ".join(f"{k}={v:.2f}" for k, v in sorted(cats.items(), key=lambda kv: -kv[1])), flush=True)
for x in sorted(kern, key=lambda x: -x.self_device_time_total)[:30]:
    print(f"KERN {cat(x.key):5s} {x.key[:88]:88s} {x.self_device_time_total / 5 / 1e3:6.3f} ms  n={x.count // 5:4d}  avg={x.self_device_time_total / max(1, x.count) :7.1f} us", flush=True)
print("=== PROF EXAMPLE DONE ===", flush=True)
