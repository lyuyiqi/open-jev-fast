"""Latency of one inference on the example request, same harness for every implementation.

E2E_SCOPE=forward (default): forward pass + scoring head only; prompts are built, tokenized and copied to the GPU
                             once beforehand (for `fast` this is the CUDA Graph replay).
E2E_SCOPE=e2e:               records -> candidate prompts -> tokenization -> forward -> scoring head.
Each call is bracketed by torch.cuda.synchronize(). No HTTP. 5 warm-up calls, median of 30.

E2E_MODE=torch  pure-PyTorch path of HF Transformers (FLA and causal-conv1d reported unavailable)
E2E_MODE=fla    original Open-Jev with flash-linear-attention installed (causal-conv1d reported unavailable)
E2E_MODE=fast   this repository: fused kernels + two-level prefix tree + CUDA Graph
"""
import json, os, statistics, sys, time
from pathlib import Path
MODE = os.environ.get("E2E_MODE", "fast")
SCOPE = os.environ.get("E2E_SCOPE", "forward")
import transformers.utils.import_utils as iu
if MODE in ("torch", "fla"):
    iu.is_causal_conv1d_available = lambda: False
if MODE == "torch":
    iu.is_flash_linear_attention_available = lambda: False
import torch
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
from jev.api import candidate_prompts, compile_request
from jev.metrics import softmax
from jev.model import DecisionModel

CKPT = Path(os.environ["OJ_CKPT"])
T0 = json.loads((CKPT / "temperature.json").read_text())["temperature"]
req = json.loads(Path(os.environ["OPEN_JEV_DIR"], "configs", "example-request.json").read_text())
records = compile_request(req["state"], req["questions"])
sync = torch.cuda.synchronize
model = DecisionModel.load(CKPT); tok = model.tokenizer

def p50(fn, reps=30, warm=5):
    for _ in range(warm): fn()
    ts = []
    for _ in range(reps):
        sync(); t = time.perf_counter(); fn(); sync(); ts.append((time.perf_counter() - t) * 1e3)
    return statistics.median(ts), min(ts), max(ts)

def to_probs(logits):
    return [softmax(l.float().cpu().tolist(), temperature=T0) for l in logits]

with torch.inference_mode():
    if MODE in ("torch", "fla"):
        import transformers.models.qwen3_5.modeling_qwen3_5 as mq
        print("MODE", MODE, "| fla", mq.chunk_gated_delta_rule is not None, "| causal_conv1d", mq.causal_conv1d_fn is not None, flush=True)
        probs = to_probs(model(records))
        replay = None
        if SCOPE == "forward":
            prompts = [tok.apply_chat_template([{"role": "user", "content": p}], tokenize=False, add_generation_prompt=True,
                                               enable_thinking=False) for r in records for p in candidate_prompts(r)]
            enc = tok(prompts, padding=True, truncation=False, return_tensors="pt")
            lengths = enc["attention_mask"].sum(-1)
            enc = {k: v.to(model.device_name) for k, v in enc.items()}
            idx = torch.arange(len(prompts), device=model.device_name); last = lengths.to(model.device_name) - 1
            def fn():  # the compute part of jev.model.DecisionModel.forward
                h = model.backbone(**enc, use_cache=False, return_dict=True).last_hidden_state[idx, last]
                return model.head(h.float()).squeeze(-1)
        else:
            fn = lambda: model(records)
    else:
        import graph_patches  # noqa: F401
        from fastmodel import FastQwen35
        model.backbone = model.backbone.merge_and_unload(); fast = FastQwen35(model.backbone)
        def prep(recs):
            seqs, groups = [], []
            for ri, r in enumerate(recs):
                for p in candidate_prompts(r):
                    seqs.append(tok(tok.apply_chat_template([{"role": "user", "content": p}], tokenize=False,
                                                            add_generation_prompt=True, enable_thinking=False))["input_ids"])
                    groups.append(ri)
            return seqs, groups
        seqs, groups = prep(records)
        ids0, lay = FastQwen35.gtree_build(seqs, groups, tok.pad_token_id)
        LAYF = ("pos", "amask", "rowmask", "src", "inv", "lastidx", "rep_ptr", "rep_pos", "hist", "amask_add", "canon", "vbits")
        s_ids = ids0.clone()
        body = lambda: model.head(fast.forward_gtree(s_ids, lay).float()).squeeze(-1)
        st = torch.cuda.Stream(); st.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(st):
            for _ in range(3): body()
        torch.cuda.current_stream().wait_stream(st); sync()
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g): out = body()
        sync()
        def fn():
            sq, gr = prep(records)
            ids, l2 = FastQwen35.gtree_build(sq, gr, tok.pad_token_id)
            assert l2.key == lay.key
            s_ids.copy_(ids)
            for n in LAYF: getattr(lay, n).copy_(getattr(l2, n))
            g.replay()
            scores, logits, off = out.clone(), [], 0
            for r in records:
                c = len(candidate_prompts(r)); v = scores[off:off + c]
                if r["kind"] == "noul": v = torch.stack([torch.zeros_like(v[0]), v[0]])
                logits.append(v); off += c
            return logits
        probs = to_probs(fn())
        replay = p50(lambda: g.replay(), reps=20)[0]
        if SCOPE == "forward":
            fn = lambda: g.replay()
    med, lo, hi = p50(fn)
res = {"mode": MODE, "scope": SCOPE, "p50_ms": round(med, 2), "min_ms": round(lo, 2), "max_ms": round(hi, 2),
       "replay_only_p50_ms": None if replay is None else round(replay, 2), "probs": probs,
       "max_memory_allocated_gib": round(torch.cuda.max_memory_allocated() / 2**30, 1), "gpu": torch.cuda.get_device_name()}
print("E2E", json.dumps(res), flush=True)
