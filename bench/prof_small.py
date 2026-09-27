import sys, json, time, statistics
from pathlib import Path
import os; sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src")); sys.path.insert(0, os.environ["JEVBENCH_DIR"])
import torch
from torch.profiler import profile, ProfilerActivity
import graph_patches  # noqa
from jev.api import compile_request, candidate_prompts
from jev.model import DecisionModel
from fastmodel import FastQwen35
from jevbench.tasks import load_jsonl
CKPT = Path(os.environ["OJ_CKPT"])
model = DecisionModel.load(CKPT); tok = model.tokenizer; model.backbone = model.backbone.merge_and_unload(); fast = FastQwen35(model.backbone)
t = [x for x in load_jsonl(os.path.join(os.environ["JEVBENCH_DIR"], "datasets/public/original.jsonl")) if x.question["type"] == "noul"][0]
recs = compile_request(t.state, {"decision": t.question})
seqs = [tok(tok.apply_chat_template([{"role": "user", "content": p}], tokenize=False, add_generation_prompt=True, enable_thinking=False))["input_ids"] for r in recs for p in candidate_prompts(r)]
sync = torch.cuda.synchronize
with torch.inference_mode():
    ids, lay = FastQwen35.gtree_build(seqs, [0] * len(seqs), tok.pad_token_id)
    print("small request key", lay.key, "real rows", lay.Nreal, flush=True)
    body = lambda: model.head(fast.forward_gtree(ids, lay).float()).squeeze(-1)
    for _ in range(3): body()
    sync()
    st = torch.cuda.Stream(); st.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(st):
        for _ in range(2): body()
    torch.cuda.current_stream().wait_stream(st); sync()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g): out = body()
    sync()
    ts = []
    for _ in range(20):
        sync(); t0 = time.perf_counter(); g.replay(); sync(); ts.append((time.perf_counter() - t0) * 1e3)
    print(f"graph replay p50 {statistics.median(ts):.2f} ms", flush=True)
    tb = []
    for _ in range(10):
        sync(); t0 = time.perf_counter(); FastQwen35.gtree_build(seqs, [0] * len(seqs), tok.pad_token_id); sync(); tb.append((time.perf_counter() - t0) * 1e3)
    print(f"gtree_build p50 {statistics.median(tb):.2f} ms", flush=True)
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        for _ in range(5): body()
        sync()
kern = [x for x in prof.key_averages() if "CUDA" in str(x.device_type)]
print(f"GPU kernel time {sum(x.self_device_time_total for x in kern)/5/1e3:.2f} ms  kernels {sum(x.count for x in kern)//5}", flush=True)
for x in sorted(kern, key=lambda x: -x.self_device_time_total)[:12]:
    print(f"KERN {x.key[:78]:78s} {x.self_device_time_total/5/1e3:6.2f} ms n={x.count//5}", flush=True)
