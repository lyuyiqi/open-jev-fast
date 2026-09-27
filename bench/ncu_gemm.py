"""One cuBLASLt gate_up GEMM at M=288 (the algorithm the model picks) for Nsight Compute."""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
import torch
from lt_ext import load
LT = load()
M, Kd, N = int(os.environ.get("M", "288")), int(os.environ.get("KD", "5120")), int(os.environ.get("N", "34816"))
x = torch.randn(M, Kd, device="cuda", dtype=torch.bfloat16); w = torch.randn(N, Kd, device="cuda", dtype=torch.bfloat16) * 0.02
best, bi = 1e9, 0
for i in range(LT.lt_setup(M, N, Kd, 32)):
    try:
        for _ in range(3): LT.lt_matmul(x, w, i)
        a, b = torch.cuda.Event(True), torch.cuda.Event(True); torch.cuda.synchronize(); a.record()
        for _ in range(20): LT.lt_matmul(x, w, i)
        b.record(); torch.cuda.synchronize(); t = a.elapsed_time(b) / 20
        if t < best: best, bi = t, i
    except Exception: pass
print(f"PICK algo {bi} {best * 1e3:.1f} us", flush=True)
torch.cuda.synchronize()
torch.cuda.cudart().cudaProfilerStart()
LT.lt_matmul(x, w, bi); torch.cuda.synchronize()
torch.cuda.cudart().cudaProfilerStop()
print("=== NCU GEMM DONE ===", flush=True)
