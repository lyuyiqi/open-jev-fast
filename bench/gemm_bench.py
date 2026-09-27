import torch, time, statistics
import torch.nn.functional as F
dev = "cuda"
shapes = {"lin_in": (5120, 16480, 48), "lin_out": (6144, 5120, 48), "full_in": (5120, 14336, 16), "full_out": (6144, 5120, 16),
          "gate_up": (5120, 34816, 64), "down": (17408, 5120, 64)}
def t(fn, reps=50):
    for _ in range(10): fn()
    torch.cuda.synchronize(); ev = [torch.cuda.Event(enable_timing=True) for _ in range(2)]
    ev[0].record()
    for _ in range(reps): fn()
    ev[1].record(); torch.cuda.synchronize()
    return ev[0].elapsed_time(ev[1]) / reps * 1000  # us
for M in (364, 384, 574, 640):
    tot = {}
    print(f"--- M={M}", flush=True)
    for name, (K, N, cnt) in shapes.items():
        x = torch.randn(M, K, device=dev, dtype=torch.bfloat16); W = torch.randn(N, K, device=dev, dtype=torch.bfloat16)
        a = t(lambda: F.linear(x, W))
        b = t(lambda: torch.mm(W, x.t()))                      # swapped: [N, M]
        flops = 2 * M * K * N
        print(f"GEMM {name:9s} K={K:5d} N={N:5d}  F.linear {a:7.1f} us ({flops/a/1e6:6.0f} TF)  swapped {b:7.1f} us ({flops/b/1e6:6.0f} TF)  x{cnt}", flush=True)
        tot[name] = (a * cnt, b * cnt)
    print(f"TOTAL per forward M={M}: F.linear {sum(v[0] for v in tot.values())/1000:.2f} ms | best-of-two {sum(min(v) for v in tot.values())/1000:.2f} ms", flush=True)
