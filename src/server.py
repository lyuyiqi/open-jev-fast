"""Accelerated Open-Jev-27B server (v3): jev.server HTTP API (/v1/systemone), DecisionModel.forward =
custom CUDA kernels + two-level prefix tree (root / question / candidate) + cuBLASLt-autotuned & split-K GEMMs +
CUDA graphs cached by shape (row bucket, #candidates, path-length bucket). One access-log line per request."""
import sys, os, threading, time, collections
import os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch
import graph_patches  # noqa
from jev import model as jm
from jev.api import candidate_prompts
NB, PB = int(os.environ.get("ROW_BUCKET", "32")), int(os.environ.get("PATH_BUCKET", "16"))
PRE_B, SUF_B, REP_MAX = 32, 16, int(os.environ.get("REP_MAX", "4096"))
MAX_GRAPHS = int(os.environ.get("MAX_GRAPHS", "512")); USE_GRAPH = os.environ.get("USE_GRAPH", "1") == "1"
ACCESS_LOG = os.environ.get("ACCESS_LOG", "access.log")
_LOGF = open(ACCESS_LOG, "a", buffering=1 << 16)           # opened once; per-request writes are in-memory appends
def _flusher():
    while True:
        time.sleep(5)
        try: _LOGF.flush()
        except Exception: pass
threading.Thread(target=_flusher, daemon=True).start()
_orig_load = jm.DecisionModel.load.__func__
def _load(cls, *a, **k):
    m = _orig_load(cls, *a, **k)
    m.backbone = m.backbone.merge_and_unload()
    from fastmodel import FastQwen35
    m._fast = FastQwen35(m.backbone); m._graphs = collections.OrderedDict(); m._lock = threading.Lock()
    m._pool = torch.cuda.graph_pool_handle(); m._nreq = 0
    print(f"GRAPH SERVER v3 ready (row bucket {NB}, path bucket {PB}, graphs={USE_GRAPH})", flush=True)
    return m
jm.DecisionModel.load = classmethod(_load)
LAYF = ("pos", "amask", "rowmask", "src", "inv", "lastidx")
class Entry:
    def __init__(self, mdl, ids, lay):
        f = mdl._fast
        self.ids = ids.clone(); self.lay = lay
        for n in LAYF: setattr(self.lay, n, getattr(lay, n).clone())
        body = lambda: mdl.head(f.forward_gtree(self.ids, self.lay).float()).squeeze(-1)
        st = torch.cuda.Stream(); st.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(st):
            for _ in range(2): body()
        torch.cuda.current_stream().wait_stream(st); torch.cuda.synchronize()
        self.g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.g, pool=mdl._pool): self.out = body()
        torch.cuda.synchronize()
    def run(self, ids, lay):
        self.ids.copy_(ids)
        for n in LAYF: getattr(self.lay, n).copy_(getattr(lay, n))
        self.g.replay()
        return self.out
class EntryState:
    """single-level prefix tree with prefix-state handoff (long shared context, one question)"""
    def __init__(self, mdl, Lpb, S, Lsb):
        dev = torch.device("cuda"); f = mdl._fast
        self.pre = torch.zeros(Lpb, dtype=torch.long, device=dev); self.suf = torch.zeros(S, Lsb, dtype=torch.long, device=dev)
        self.msk = torch.ones(S, Lsb, dtype=torch.long, device=dev)
        self.lay = f.tree_layout(self.pre, self.suf, self.msk, pad=0)
        body = lambda: mdl.head(f.forward_tree(self.pre, self.suf, self.lay, fla_mode="state").float()).squeeze(-1)
        st = torch.cuda.Stream(); st.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(st):
            for _ in range(2): body()
        torch.cuda.current_stream().wait_stream(st); torch.cuda.synchronize()
        self.g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.g, pool=mdl._pool): self.out = body()
        torch.cuda.synchronize()
    def run(self, f, pre, suf, msk, pad):
        self.pre.copy_(pre); self.suf.copy_(suf); self.msk.copy_(msk)
        L = f.tree_layout(self.pre, self.suf, self.msk, pad=pad)
        for name in ("pos", "amask", "rowmask", "lastidx", "repidx"): getattr(self.lay, name).copy_(getattr(L, name))
        self.g.replay()
        return self.out
ceil_ = lambda x, b: ((x + b - 1) // b) * b
def _forward(self, records):
    t_start = time.perf_counter()
    tok = self.tokenizer; f = self._fast
    seqs, groups, counts = [], [], []
    for ri, r in enumerate(records):
        e = candidate_prompts(r); counts.append(len(e))
        for p in e:
            seqs.append(tok(tok.apply_chat_template([{"role": "user", "content": p}], tokenize=False, add_generation_prompt=True,
                                                    enable_thinking=False))["input_ids"]); groups.append(ri)
    self.last_input_tokens = sum(map(len, seqs))
    t_tok = time.perf_counter()
    dev = torch.device("cuda")
    with self._lock, torch.inference_mode():
        captured = False
        Lp, mn = 0, min(map(len, seqs))
        while Lp < mn - 1 and all(sq[Lp] == seqs[0][Lp] for sq in seqs): Lp += 1
        S = len(seqs)
        if len(records) == 1 and (S - 1) * Lp > REP_MAX and USE_GRAPH:
            path = "state"
            Ls = max(len(sq) - Lp for sq in seqs); Lpb, Lsb = ceil_(Lp, PRE_B), ceil_(Ls, SUF_B); pad = Lpb - Lp
            pre = torch.tensor([tok.pad_token_id] * pad + seqs[0][:Lp], dtype=torch.long)
            suf = torch.full((S, Lsb), tok.pad_token_id, dtype=torch.long); msk = torch.zeros(S, Lsb, dtype=torch.long)
            for i, sq in enumerate(seqs): suf[i, :len(sq) - Lp] = torch.tensor(sq[Lp:]); msk[i, :len(sq) - Lp] = 1
            key = ("state", Lpb, S, Lsb); nrows = Lp + sum(len(sq) - Lp for sq in seqs)
            ent = self._graphs.get(key)
            if ent is None:
                ent = EntryState(self, Lpb, S, Lsb); self._graphs[key] = ent; captured = True
            else:
                self._graphs.move_to_end(key)
            t_build = time.perf_counter()
            scores = ent.run(f, pre.to(dev), suf.to(dev), msk.to(dev), pad).clone()
        else:
            path = "tree2"
            ids, lay = f.gtree_build(seqs, groups, tok.pad_token_id, NB=NB if USE_GRAPH else 1, PB=PB if USE_GRAPH else 1)
            key = lay.key; nrows = lay.Nreal
            t_build = time.perf_counter()
            if USE_GRAPH:
                ent = self._graphs.get(key)
                if ent is None:
                    ent = Entry(self, ids, lay); self._graphs[key] = ent; captured = True
                else:
                    self._graphs.move_to_end(key)
                scores = ent.run(ids, lay).clone()
            else:
                scores = self.head(f.forward_gtree(ids, lay).float()).squeeze(-1)
        while len(self._graphs) > MAX_GRAPHS: self._graphs.popitem(last=False)
        torch.cuda.synchronize(); t_gpu = time.perf_counter()
        self._nreq += 1
        if True:
            _LOGF.write(f"{time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())} req={self._nreq} path={path} questions={len(records)} candidates={S} "
                     f"tokens={self.last_input_tokens} rows={nrows} key={key} captured={int(captured)} "
                     f"tok_ms={(t_tok-t_start)*1000:.1f} build_ms={(t_build-t_tok)*1000:.1f} gpu_ms={(t_gpu-t_build)*1000:.1f} ms={(t_gpu-t_start)*1000:.1f}\n")
    logits, off = [], 0
    for r, c in zip(records, counts):
        v = scores[off:off + c]
        if r["kind"] == "noul": v = torch.stack([torch.zeros_like(v[0]), v[0]])
        logits.append(v); off += c
    return logits
jm.DecisionModel.forward = _forward
# --- serving-path fixes / instrumentation ---
import socketserver
socketserver.StreamRequestHandler.disable_nagle_algorithm = True     # TCP_NODELAY: no Nagle/delayed-ACK stalls on small responses
from jev import serving as _sv
_orig_score, _orig_predict = _sv.TorchScorer.score, _sv.Predictor.predict
def _score(self, records):
    t = time.perf_counter(); r = _orig_score(self, records)
    _sv._last_score_ms = (time.perf_counter() - t) * 1000; return r
def _predict(self, request):
    t = time.perf_counter(); r = _orig_predict(self, request)
    if True:
        _LOGF.write(f"  predict_ms={(time.perf_counter()-t)*1000:.1f} score_ms={getattr(_sv, '_last_score_ms', -1):.1f}\n")
    return r
_sv.TorchScorer.score = _score; _sv.Predictor.predict = _predict
from jev.server import main
sys.argv = ["jev.server"] + sys.argv[1:]
main()
