"""Prefix-tree forward: correctness vs the ORIGINAL model (probabilities) + speed under CUDA graph."""
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
model = DecisionModel.load(CKPT); tok = model.tokenizer; dev = torch.device("cuda")
sync = torch.cuda.synchronize
req = json.loads(Path(os.environ["OPEN_JEV_DIR"], "configs", "example-request.json").read_text())
cases = [("example", compile_request(req["state"], req["questions"]))]
from jevbench.tasks import load_jsonl
hard = load_jsonl(os.path.join(os.environ["JEVBENCH_DIR"], "datasets/public/hard.jsonl"))
hard = sorted(hard, key=lambda t: -len(str(t.state)))
for t in hard[:2] + hard[60:61]:
    cases.append((t.id, compile_request(t.state, {"decision": t.question})))
with torch.inference_mode():
    refs = {name: [softmax(l.float().cpu().tolist(), temperature=T0) for l in model(recs)] for name, recs in cases}
model.backbone = model.backbone.merge_and_unload()
fast = FastQwen35(model.backbone)
def build(recs):
    ps = [tok.apply_chat_template([{"role": "user", "content": p}], tokenize=False, add_generation_prompt=True, enable_thinking=False)
          for r in recs for p in candidate_prompts(r)]
    seqs = [tok(p, add_special_tokens=False)["input_ids"] if False else tok(p)["input_ids"] for p in ps]
    Lp = 0; mn = min(map(len, seqs))
    while Lp < mn - 1 and all(s[Lp] == seqs[0][Lp] for s in seqs): Lp += 1
    S = len(seqs); Ls = max(len(s) - Lp for s in seqs)
    suf = torch.full((S, Ls), tok.pad_token_id, dtype=torch.long); m = torch.zeros(S, Ls, dtype=torch.long)
    for i, s in enumerate(seqs): suf[i, :len(s) - Lp] = torch.tensor(s[Lp:]); m[i, :len(s) - Lp] = 1
    return torch.tensor(seqs[0][:Lp], device=dev), suf.to(dev), m.to(dev), sum(map(len, seqs)), max(map(len, seqs)) * S
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
        pre, suf, m, ntok, npad = build(recs)
        lay = fast.tree_layout(pre, suf, m)
        pt_state = to_probs(model.head(fast.forward_tree(pre, suf, lay, fla_mode="state").float()).squeeze(-1), recs)
        hid = fast.forward_tree(pre, suf, lay)
        pt = to_probs(model.head(hid.float()).squeeze(-1), recs)
        print(f"TREE[state-handoff] {name:34s} prob maxdiff vs original {md(refs[name], pt_state):.2e}", flush=True)
        print(f"TREE[replicate]     {name:34s} Lp={lay.Lp:5d} S={lay.S} Ls={lay.Ls:4d} rows {lay.N:5d} (vs {npad} padded / {ntok} real)  "
              f"prob maxdiff vs original {md(refs[name], pt):.2e}  argmax same {am(refs[name]) == am(pt)}", flush=True)
    # ---- speed: example request under CUDA graph ----
    name, recs = cases[0]
    pre, suf, m, _, _ = build(recs); lay = fast.tree_layout(pre, suf, m)
    s_pre, s_suf = pre.clone(), suf.clone()
    def body(): return model.head(fast.forward_tree(s_pre, s_suf, lay).float()).squeeze(-1)
    st = torch.cuda.Stream(); st.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(st):
        for _ in range(3): body()
    torch.cuda.current_stream().wait_stream(st); sync()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g): out = body()
    sync(); g.replay(); sync()
    pg = to_probs(out, recs)
    t = p50(lambda: g.replay())
    print(f"GRAPH tree example: replay p50 {t:.2f} ms ({1000/t:.1f} req/s)  prob maxdiff vs original {md(refs[name], pg):.2e}", flush=True)
    t_e = p50(lambda: fast.forward_tree(s_pre, s_suf, lay), reps=15)
    print(f"EAGER tree example: p50 {t_e:.2f} ms", flush=True)
    # long case under graph too
    name, recs = cases[1]
    pre, suf, m, ntok, npad = build(recs); lay = fast.tree_layout(pre, suf, m)
    s_pre, s_suf = pre.clone(), suf.clone()
    st = torch.cuda.Stream(); st.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(st):
        for _ in range(2): body()
    torch.cuda.current_stream().wait_stream(st); sync()
    g2 = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g2): out2 = body()
    sync()
    t2 = p50(lambda: g2.replay(), reps=10)
    enc = tok([tok.apply_chat_template([{"role": "user", "content": p}], tokenize=False, add_generation_prompt=True, enable_thinking=False)
               for r in recs for p in candidate_prompts(r)], padding=True, return_tensors="pt")
    ids, msk = enc["input_ids"].to(dev), enc["attention_mask"].to(dev)
    t2f = p50(lambda: fast(input_ids=ids, attention_mask=msk), reps=5, warm=2)
    print(f"LONG {name}: tree graph {t2:.2f} ms vs non-tree eager fast {t2f:.2f} ms  (rows {lay.N} vs {ids.numel()})", flush=True)
print("=== TREE DONE ===", flush=True)
