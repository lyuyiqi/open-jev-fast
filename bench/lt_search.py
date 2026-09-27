"""Exhaustive cuBLASLt search for the model's GEMM shapes at M rows (default 288): best-fit heuristic (256 results),
per-algorithm-id heuristic, and an explicit tile/stages/cluster/split-K/reduction/swizzle sweep. Screens every candidate
with a short CUDA-graph timing, re-times the best 12, and compares with the configuration the model uses today.
Writes the winners' raw algorithm descriptors to results/lt_search_M{M}.json."""
import json, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
import torch, torch.nn.functional as F
_ch = os.environ.get("CUDA_HOME")
os.environ["CUDA_HOME"] = os.environ.get("LT_CUDA", _ch)
from lt_ext import load
LT = load()
os.environ["CUDA_HOME"] = _ch

def gpu_us(fn, per=20, reps=20):
    st = torch.cuda.Stream(); st.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(st):
        for _ in range(2): fn()
    torch.cuda.current_stream().wait_stream(st); torch.cuda.synchronize()
    gr = torch.cuda.CUDAGraph()
    with torch.cuda.graph(gr):
        for _ in range(per): fn()
    gr.replay()
    a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    torch.cuda.synchronize(); a.record()
    for _ in range(reps): gr.replay()
    b.record(); torch.cuda.synchronize()
    return a.elapsed_time(b) * 1e3 / (reps * per)

M = int(os.environ.get("M", "288")); MAXC = int(os.environ.get("MAXC", "3000")); MODES = int(os.environ.get("MODES", "7"))
XS = float(os.environ.get("XSCALE", "1.0"))
torch.manual_seed(0)
# name, K, N, current S (1 = plain bf16 GEMM tuned over 32 heuristic algos; >1 = torch.bmm fp32 split-K), plans to search
SHAPES = [("gate_up", 5120, 34816, 1, [(1, 0, 0)]), ("lin_in", 5120, 16480, 1, [(1, 0, 0)]), ("full_in", 5120, 14336, 1, [(1, 0, 0)]),
          ("lin_out", 6144, 5120, 2, [(2, 1, 1), (1, 1, 0), (3, 1, 1), (4, 1, 1)]), ("down", 17408, 5120, 4, [(4, 1, 1), (2, 1, 1), (1, 1, 0), (8, 1, 1)])]
only = os.environ.get("ONLY"); out = {}
for name, Kd, N, S0, plans in SHAPES:
    if only and name not in only.split(","): continue
    x = torch.randn(M, Kd, device="cuda", dtype=torch.bfloat16) * XS
    w = torch.randn(N, Kd, device="cuda", dtype=torch.bfloat16) * 0.02
    ref = (x.float() @ w.float().T)
    tol = ref.abs().max().item() * 1e-2
    # today's configuration
    if S0 == 1:
        best0 = 1e9
        for i in range(LT.lt_setup(M, N, Kd, 32)):
            try:
                if (LT.lt_matmul(x, w, i).float() - ref).abs().max().item() > tol: continue
                best0 = min(best0, gpu_us(lambda: LT.lt_matmul(x, w, i)))
            except Exception: pass
    else:
        wsk = w.view(N, S0, Kd // S0).permute(1, 2, 0).contiguous()
        best0 = gpu_us(lambda: torch.bmm(x.view(M, S0, Kd // S0).transpose(0, 1), wsk, out_dtype=torch.float32))
    print(f"SHAPE {name} K={Kd} N={N} today (S={S0}) {best0:.1f} us  weights {N * Kd * 2 / best0 / 1e6:.2f} TB/s", flush=True)
    res = []
    for S, f32, wl in plans:
        wa = w if wl == 0 else w.view(N, S, Kd // S).permute(1, 2, 0).contiguous()
        h = LT.lt2_plan(M, N, Kd, S, f32, wl)
        n = LT.lt2_search(h, MAXC, MODES)
        ok = 0
        for i in range(n):
            try:
                o = LT.lt2_run(h, x, wa, i)
                o = o.float().sum(0) if S > 1 else o.float()
                if (o - ref).abs().max().item() > tol: continue
                t = gpu_us(lambda: LT.lt2_run(h, x, wa, i), per=5, reps=4)
                res.append((t, S, f32, wl, h, i)); ok += 1
            except Exception:
                pass
        print(f"  plan S={S} f32={f32} wl={wl}: {n} candidates, {ok} valid", flush=True)
    res.sort()
    fin = []
    for t, S, f32, wl, h, i in res[:12]:
        wa = w if wl == 0 else w.view(N, S, Kd // S).permute(1, 2, 0).contiguous()
        t2 = min(gpu_us(lambda: LT.lt2_run(h, x, wa, i)) for _ in range(2))
        fin.append((t2, S, f32, wl, h, i))
    fin.sort()
    for t2, S, f32, wl, h, i in fin[:6]:
        print(f"  BEST {t2:6.1f} us ({N * Kd * 2 / t2 / 1e6:.2f} TB/s) S={S} f32={f32} wl={wl} cfg(id,tile,stages,splitk,red,swz,custom,cluster)={LT.lt2_info(h, i)}", flush=True)
    t2, S, f32, wl, h, i = fin[0]
    out[name] = {"K": Kd, "N": N, "M": M, "today_us": best0, "best_us": t2, "S": S, "f32": f32, "wl": wl, "cfg": LT.lt2_info(h, i), "raw": LT.lt2_raw(h, i)}
    del x, w, ref; torch.cuda.empty_cache()
p = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results", f"lt_search_M{M}.json")
json.dump(out, open(p, "w"), indent=1)
print("=== LT SEARCH DONE ===", flush=True)
