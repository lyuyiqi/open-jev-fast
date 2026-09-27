"""How far the M=288 GEMMs are from the weight-streaming floor: torch/cuBLAS GPU time vs M for the model's GEMM shapes,
and device-to-device copy bandwidth. GPU time per call from CUDA-graph replay."""
import torch, torch.nn.functional as F

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

src = torch.empty(512 << 20, dtype=torch.bfloat16, device="cuda"); dst = torch.empty_like(src)
t = gpu_us(lambda: dst.copy_(src), per=5, reps=10)
print(f"COPY 1 GiB read + 1 GiB write: {t:.1f} us  {2 * src.numel() * 2 / t / 1e6:.2f} TB/s", flush=True)
del src, dst
for name, Kd, N in (("gate_up", 5120, 34816), ("lin_in", 5120, 16480), ("lin_out", 6144, 5120), ("down", 17408, 5120)):
    w = torch.randn(N, Kd, device="cuda", dtype=torch.bfloat16) * 0.02
    line = []
    for M in (8, 64, 128, 192, 256, 288, 384):
        x = torch.randn(M, Kd, device="cuda", dtype=torch.bfloat16)
        t = gpu_us(lambda: F.linear(x, w))
        line.append(f"M{M}={t:.1f}us({w.numel() * 2 / t / 1e6:.2f}TB/s)")
    print(f"SWEEP {name:8s} " + " ".join(line), flush=True)
print("=== MSWEEP DONE ===", flush=True)
