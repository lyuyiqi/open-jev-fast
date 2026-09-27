import torch, torch.nn.functional as F
dev = "cuda"
def t(fn, reps=100):
    for _ in range(10): fn()
    torch.cuda.synchronize(); a, b = torch.cuda.Event(True), torch.cuda.Event(True)
    a.record()
    for _ in range(reps): fn()
    b.record(); torch.cuda.synchronize(); return a.elapsed_time(b) / reps * 1000
M = 364
for name, K, N, S in (("lin_out", 6144, 5120, 4), ("full_out", 6144, 5120, 2), ("down", 17408, 5120, 4)):
    x = torch.randn(M, K, device=dev, dtype=torch.bfloat16); W = torch.randn(N, K, device=dev, dtype=torch.bfloat16)
    ks = K // S; Ws = W.view(N, S, ks).permute(1, 2, 0).contiguous()
    xv = x.view(M, S, ks).transpose(0, 1)                # strided view, no copy
    xc = xv.contiguous()
    r = [f"base {t(lambda: F.linear(x, W)):6.1f}", f"bmm-contig {t(lambda: torch.bmm(xc, Ws)):6.1f}"]
    try: r.append(f"bmm-view {t(lambda: torch.bmm(xv, Ws)):6.1f}")
    except Exception as e: r.append(f"bmm-view ERR {type(e).__name__}")
    try: r.append(f"bmm-view-fp32out {t(lambda: torch.bmm(xv, Ws, out_dtype=torch.float32)):6.1f}")
    except Exception as e: r.append(f"fp32out ERR {type(e).__name__}: {str(e)[:80]}")
    ref = F.linear(x.float(), W.float())
    e_base = (F.linear(x, W).float() - ref).abs().max().item()
    e_sk = (torch.bmm(xv, Ws).float().sum(0) - ref).abs().max().item()
    try: e_sk32 = (torch.bmm(xv, Ws, out_dtype=torch.float32).sum(0) - ref).abs().max().item()
    except Exception: e_sk32 = float("nan")
    print(f"SK2 {name:8s} S={S}: " + "  ".join(r) + f"  | err vs fp32: base {e_base:.3e} splitk-bf16 {e_sk:.3e} splitk-fp32 {e_sk32:.3e}", flush=True)
