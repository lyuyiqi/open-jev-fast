"""Two-level prefix tree: correctness vs ORIGINAL model + graph speed (example request) + a JevBench long task."""
import sys, json, time, statistics
from pathlib import Path
import os; sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src")); sys.path.insert(0, os.environ["JEVBENCH_DIR"])
import torch
import graph_patches  # noqa
from jev.api import compile_request, candidate_prompts
from jev.metrics import softmax
from jev.model import DecisionModel
from fastmodel import FastQwen35
CKPT = Path(os.environ["OJ_CKPT"])
T0 = json.loads((CKPT / "temperature.json").read_text())["temperature"]
model = DecisionModel.load(CKPT); tok = model.tokenizer; dev = torch.device("cuda"); sync = torch.cuda.synchronize
req = json.loads(Path(os.environ["OPEN_JEV_DIR"], "configs", "example-request.json").read_text())
cases = [("example", compile_request(req["state"], req["questions"]))]
from jevbench.tasks import load_jsonl
hard = sorted(load_jsonl(os.path.join(os.environ["JEVBENCH_DIR"], "datasets/public/hard.jsonl")), key=lambda t: -len(str(t.state)))
cases.append((hard[1].id, compile_request(hard[1].state, {"decision": hard[1].question})))
with torch.inference_mode():
    refs = {n: [softmax(l.float().cpu().tolist(), temperature=T0) for l in model(r)] for n, r in cases}
model.backbone = model.backbone.merge_and_unload(); fast = FastQwen35(model.backbone)
def prep(recs):
    seqs, groups = [], []
    for ri, r in enumerate(recs):
        for p in candidate_prompts(r):
            seqs.append(tok(tok.apply_chat_template([{"role": "user", "content": p}], tokenize=False, add_generation_prompt=True, enable_thinking=False))["input_ids"]); groups.append(ri)
    return seqs, groups
def to_probs(scores, recs):
    out, off = [], 0
    for r in recs:
        c = len(candidate_prompts(r)); v = scores[off:off + c]
        if r["kind"] == "noul": v = torch.stack([torch.zeros_like(v[0]), v[0]])
        out.append(softmax(v.float().cpu().tolist(), temperature=T0)); off += c
    return out
md = lambda a, b: max(abs(x - y) for pa, pb in zip(a, b) for x, y in zip(pa, pb))
am = lambda a: [max(range(len(p)), key=p.__getitem__) for p in a]
def p50(fn, reps=30, warm=5):
    for _ in range(warm): fn()
    ts = []
    for _ in range(reps):
        sync(); t = time.perf_counter(); fn(); sync(); ts.append((time.perf_counter() - t) * 1e3)
    return statistics.median(ts)
with torch.inference_mode():
    for name, recs in cases:
        seqs, groups = prep(recs)
        ids, lay = FastQwen35.gtree_build(seqs, groups, tok.pad_token_id)
        pr = to_probs(model.head(fast.forward_gtree(ids, lay).float()).squeeze(-1), recs)
        print(f"GTREE {name:30s} key {lay.key}  rows {lay.N} (real rows {lay.Nreal}) (real tokens {sum(map(len, seqs))})  prob maxdiff vs original {md(refs[name], pr):.2e}  argmax same {am(refs[name]) == am(pr)}", flush=True)
        s_ids = ids.clone()
        body = lambda: model.head(fast.forward_gtree(s_ids, lay).float()).squeeze(-1)
        st = torch.cuda.Stream(); st.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(st):
            for _ in range(3): body()
        torch.cuda.current_stream().wait_stream(st); sync()
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g): out = body()
        sync(); g.replay(); sync()
        pg = to_probs(out, recs)
        tt = p50(lambda: g.replay(), reps=20 if lay.N < 1000 else 8)
        print(f"GTREE-GRAPH {name:30s} replay p50 {tt:.2f} ms ({1000/tt:.1f} req/s)  prob maxdiff vs original {md(refs[name], pg):.2e}", flush=True)
        del g
print("=== GTREE DONE ===", flush=True)
