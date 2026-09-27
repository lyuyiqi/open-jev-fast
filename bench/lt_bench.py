import sys, torch, torch.nn.functional as F
import os; sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
from lt_ext import load
L = load(); dev = "cuda"
def t(fn, reps=100):
    for _ in range(10): fn()
    torch.cuda.synchronize(); a, b = torch.cuda.Event(True), torch.cuda.Event(True)
    a.record()
    for _ in range(reps): fn()
    b.record(); torch.cuda.synchronize(); return a.elapsed_time(b) / reps * 1000
M = 364; tot_base = tot_best = 0
for name, K, N, cnt in (("lin_in", 5120, 16480, 48), ("lin_out", 6144, 5120, 48), ("full_in", 5120, 14336, 16), ("full_out", 6144, 5120, 16),
                        ("gate_up", 5120, 34816, 64), ("down", 17408, 5120, 64)):
    x = torch.randn(M, K, device=dev, dtype=torch.bfloat16); w = torch.randn(N, K, device=dev, dtype=torch.bfloat16)
    n = L.lt_setup(M, N, K, 64)
    base = t(lambda: F.linear(x, w)); ref = F.linear(x, w).float()
    best, bi, times = 1e9, -1, []
    for i in range(n):
        try:
            o = L.lt_matmul(x, w, i); err = (o.float() - ref).abs().max().item()
            us = t(lambda: L.lt_matmul(x, w, i), reps=40); times.append(us)
            if err < 1.0 and us < best: best, bi = us, i
        except Exception as e:
            times.append(float("nan"))
    tot_base += base * cnt; tot_best += min(base, best) * cnt
    print(f"LT {name:8s} algos {n:2d}  torch {base:6.1f} us  best-lt {best:6.1f} us (idx {bi})  gain {100*(1-best/base):5.1f}%", flush=True)
print(f"LT TOTAL per forward (M=364): torch {tot_base/1000:.2f} ms -> autotuned {tot_best/1000:.2f} ms", flush=True)
