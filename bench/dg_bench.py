"""DeepGEMM sm100 bf16 GEMM vs cuBLASLt (our per-shape autotune) vs torch, for the model's GEMM shapes at M=288.
GPU time per call from CUDA-graph replay; max abs diff vs torch."""
import os, sys
sys.path.insert(0, os.environ["DEEPGEMM_DIR"])   # a DeepGEMM checkout built in place
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
import torch, torch.nn.functional as F
import deep_gemm
_ch = os.environ.get("CUDA_HOME")
os.environ["CUDA_HOME"] = os.environ.get("LT_CUDA", _ch)   # our cuBLASLt wrapper was built against the pip CUDA 13 tree
from lt_ext import load
LT = load()
os.environ["CUDA_HOME"] = _ch

def gpu_us(fn, per=20, reps=20):
    st = torch.cuda.Stream(); st.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(st):
        for _ in range(3): fn()
    torch.cuda.current_stream().wait_stream(st); torch.cuda.synchronize()
    gr = torch.cuda.CUDAGraph()
    with torch.cuda.graph(gr):
        for _ in range(per): fn()
    for _ in range(3): gr.replay()
    a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    torch.cuda.synchronize(); a.record()
    for _ in range(reps): gr.replay()
    b.record(); torch.cuda.synchronize()
    return a.elapsed_time(b) * 1e3 / (reps * per)

M = int(os.environ.get("M", "288"))
tot = {"torch": 0.0, "lt": 0.0, "dg": 0.0}
for name, Kd, N, cnt in (("gate_up", 5120, 34816, 64), ("lin_in", 5120, 16480, 48), ("full_in", 5120, 14336, 16),
                         ("lin_out", 6144, 5120, 48), ("full_out", 6144, 5120, 16), ("down", 17408, 5120, 64)):
    x = torch.randn(M, Kd, device="cuda", dtype=torch.bfloat16) * 0.5
    w = torch.randn(N, Kd, device="cuda", dtype=torch.bfloat16) * 0.02
    ref = F.linear(x, w)
    t_torch = gpu_us(lambda: F.linear(x, w))
    best_lt = 1e9
    for i in range(LT.lt_setup(M, N, Kd, 32)):
        try:
            if (LT.lt_matmul(x, w, i).float() - ref.float()).abs().max().item() > 0.05: continue
            best_lt = min(best_lt, gpu_us(lambda: LT.lt_matmul(x, w, i), per=10, reps=10))
        except Exception:
            pass
    d = torch.empty(M, N, device="cuda", dtype=torch.bfloat16)
    deep_gemm.bf16_gemm_nt(x, w, d)
    err = (d.float() - ref.float()).abs().max().item()
    t_dg = gpu_us(lambda: deep_gemm.bf16_gemm_nt(x, w, d))
    tf = 2 * M * Kd * N / 1e12
    for k, v in (("torch", t_torch), ("lt", best_lt), ("dg", t_dg)): tot[k] += v * cnt
    print(f"SHAPE {name:8s} K={Kd:5d} N={N:5d}  torch {t_torch:6.1f}  cuBLASLt-best {best_lt:6.1f}  DeepGEMM {t_dg:6.1f} us "
          f"({tf / t_dg * 1e6 / 1e3:.2f} PF)  max|diff| vs torch {err:.2e}", flush=True)
print("TOTAL per forward (ms): " + "  ".join(f"{k} {v / 1e3:.2f}" for k, v in tot.items()), flush=True)
print("=== DG BENCH DONE ===", flush=True)
