import torch, torch.nn.functional as F
dev = "cuda"
def t(fn, reps=100):
    for _ in range(10): fn()
    torch.cuda.synchronize(); a, b = torch.cuda.Event(True), torch.cuda.Event(True)
    a.record()
    for _ in range(reps): fn()
    b.record(); torch.cuda.synchronize(); return a.elapsed_time(b) / reps * 1000
M = 364
for name, K, N, cnt in (("lin_out", 6144, 5120, 48), ("full_out", 6144, 5120, 16), ("down", 17408, 5120, 64),
                        ("lin_in", 5120, 16480, 48), ("full_in", 5120, 14336, 16), ("gate_up", 5120, 34816, 64)):
    x = torch.randn(M, K, device=dev, dtype=torch.bfloat16); W = torch.randn(N, K, device=dev, dtype=torch.bfloat16)
    base = t(lambda: F.linear(x, W)); res = [f"base {base:6.1f}"]
    for S in (2, 3, 4, 6, 8):
        if K % S: continue
        ks = K // S
        xs = x.view(M, S, ks).transpose(0, 1).contiguous()                 # [S, M, ks]
        Ws = W.view(N, S, ks).permute(1, 2, 0).contiguous()                # [S, ks, N]
        out32 = torch.empty(S, M, N, device=dev, dtype=torch.float32)
        tb = t(lambda: torch.bmm(xs, Ws))                                  # bf16 partials
        res.append(f"S{S} {tb:6.1f}")
    print(f"SPLITK {name:8s} K={K:5d} N={N:5d} x{cnt}: " + "  ".join(res), flush=True)
