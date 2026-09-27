"""Server tokenization: per-prompt chat template + tokenizer call vs a cached template prefix/suffix and one batched
tokenizer call. Checks identical token ids on every JevBench request and the example, and times both."""
import json, os, sys, time
sys.path.insert(0, os.environ["JEVBENCH_DIR"])
from pathlib import Path
from jevbench.tasks import load_jsonl
from jevbench.adapters.typesafe import TypeSafeAdapter
from jev.api import candidate_prompts, compile_request
from transformers import AutoTokenizer
cfg = json.loads((Path(os.environ["OJ_CKPT"]) / "model.json").read_text())
tok = AutoTokenizer.from_pretrained(cfg["model_id"], revision=cfg["revision"])
ad = TypeSafeAdapter(endpoint="http://x", model="m", key_env=None)
reqs = [json.loads(Path(os.environ["OPEN_JEV_DIR"], "configs", "example-request.json").read_text())]
for tier in ("original", "easy", "hard"):
    for t in load_jsonl(os.path.join(os.environ["JEVBENCH_DIR"], f"datasets/public/{tier}.jsonl")):
        r = ad.build_request(t); reqs.append({"state": r["state"], "questions": r["questions"]})
M = "\x00OJ\x00"
tpl = tok.apply_chat_template([{"role": "user", "content": M}], tokenize=False, add_generation_prompt=True, enable_thinking=False)
pre, suf = tpl.split(M)
print("TOK template prefix", repr(pre), "suffix", repr(suf), flush=True)
def old(prompts): return [tok(tok.apply_chat_template([{"role": "user", "content": p}], tokenize=False, add_generation_prompt=True, enable_thinking=False))["input_ids"] for p in prompts]
def new(prompts): return tok([pre + p + suf for p in prompts])["input_ids"]
bad = 0; allp = []
for rq in reqs:
    prompts = [p for r in compile_request(rq["state"], rq["questions"]) for p in candidate_prompts(r)]
    allp.append(prompts)
    if old(prompts) != new(prompts): bad += 1
print(f"TOK requests {len(reqs)}, mismatching {bad}", flush=True)
for name, fn in (("old", old), ("new", new)):
    t0 = time.perf_counter()
    for _ in range(3):
        for prompts in allp: fn(prompts)
    dt = (time.perf_counter() - t0) / (3 * len(allp)) * 1e3
    t0 = time.perf_counter()
    for _ in range(50): fn(allp[0])
    print(f"TOK {name}: {dt:.3f} ms per request (JevBench mix), example {(time.perf_counter() - t0) / 50 * 1e3:.3f} ms", flush=True)
bt = tok.backend_tokenizer
def fast(prompts): return [e.ids for e in bt.encode_batch([pre + p + suf for p in prompts], add_special_tokens=True)]
def fast1(prompts): return [bt.encode(pre + p + suf, add_special_tokens=True).ids for p in prompts]
bad = sum(old(p) != fast(p) for p in allp); bad1 = sum(old(p) != fast1(p) for p in allp)
print(f"TOK encode_batch mismatching {bad}, encode mismatching {bad1}", flush=True)
for name, fn in (("encode_batch", fast), ("encode loop", fast1)):
    t0 = time.perf_counter()
    for _ in range(3):
        for prompts in allp: fn(prompts)
    dt = (time.perf_counter() - t0) / (3 * len(allp)) * 1e3
    t0 = time.perf_counter()
    for _ in range(50): fn(allp[0])
    print(f"TOK {name}: {dt:.3f} ms per request (JevBench mix), example {(time.perf_counter() - t0) / 50 * 1e3:.3f} ms", flush=True)
print("TOK example prompt lengths (chars):", [len(p) for p in allp[0]], flush=True)
def split_enc(prompts):
    texts = [pre + p + suf for p in prompts]
    if len(texts) > 1:
        cp = os.path.commonprefix(texts)
        k = cp.rfind("\n")
        while k >= 0 and not all(len(t) > k + 1 and t[k + 1].isalnum() for t in texts): k = cp.rfind("\n", 0, k)
        if k > 0:
            ea = bt.encode(texts[0][:k + 1], add_special_tokens=False).ids
            return [ea + e.ids for e in bt.encode_batch([t[k + 1:] for t in texts], add_special_tokens=False)]
    return [e.ids for e in bt.encode_batch(texts, add_special_tokens=False)]
bad = sum(old(p) != split_enc(p) for p in allp)
print(f"TOK split encode mismatching {bad} of {len(allp)}", flush=True)
t0 = time.perf_counter()
for _ in range(3):
    for prompts in allp: split_enc(prompts)
dt = (time.perf_counter() - t0) / (3 * len(allp)) * 1e3
t0 = time.perf_counter()
for _ in range(50): split_enc(allp[0])
print(f"TOK split encode: {dt:.3f} ms per request (JevBench mix), example {(time.perf_counter() - t0) / 50 * 1e3:.3f} ms", flush=True)
