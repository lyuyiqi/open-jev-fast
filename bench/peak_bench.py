import torch
dev = "cuda"
def t(fn, reps=20):
    for _ in range(5): fn()
    torch.cuda.synchronize(); a, b = torch.cuda.Event(True), torch.cuda.Event(True)
    a.record()
    for _ in range(reps): fn()
    b.record(); torch.cuda.synchronize(); return a.elapsed_time(b) / reps * 1000
n = 8192
x = torch.randn(n, n, device=dev, dtype=torch.bfloat16); w = torch.randn(n, n, device=dev, dtype=torch.bfloat16)
us = t(lambda: x @ w.t()); print(f"PEAK bf16 8192^3: {2*n**3/us/1e6:.0f} TFLOPS", flush=True)
# FP8 (e4m3, per-tensor scales) potential on our shapes, M=384 — measurement only
shapes = {"lin_in": (5120, 16480, 48), "lin_out": (6144, 5120, 48), "full_in": (5120, 14336, 16), "full_out": (6144, 5120, 16),
          "gate_up": (5120, 34816, 64), "down": (17408, 5120, 64)}
tb, tf = 0, 0
for name, (K, N, cnt) in shapes.items():
    M = 384
    xb = torch.randn(M, K, device=dev, dtype=torch.bfloat16); wb = torch.randn(N, K, device=dev, dtype=torch.bfloat16)
    x8 = xb.to(torch.float8_e4m3fn); w8 = wb.to(torch.float8_e4m3fn)
    one = torch.ones((), device=dev)
    ub = t(lambda: torch.nn.functional.linear(xb, wb))
    try:
        uf = t(lambda: torch._scaled_mm(x8, w8.t(), scale_a=one, scale_b=one, out_dtype=torch.bfloat16))
    except Exception as e:
        print("FP8 failed", type(e).__name__, str(e)[:200]); uf = float("nan")
    print(f"SHAPE {name:9s} bf16 {ub:6.1f} us  fp8 {uf:6.1f} us", flush=True)
    tb += ub * cnt; tf += uf * cnt
print(f"TOTAL M=384: bf16 {tb/1000:.2f} ms  fp8 {tf/1000:.2f} ms (GEMM only, excl. activation quantization)", flush=True)
