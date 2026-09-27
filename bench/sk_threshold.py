"""Split-K (fp32 partials + fused reduce/residual/RMSNorm) vs plain GEMM + fused add/RMSNorm, per row count M.
Uses the real merged weights of layer 3 (full attention: o_proj S=2) and its MLP down projection (S=4)."""
import os, sys, statistics, time, json
from pathlib import Path
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
import torch
import graph_patches  # noqa: F401
from jev.model import DecisionModel
from fastmodel import FastQwen35
model = DecisionModel.load(Path(os.environ["OJ_CKPT"])); model.backbone = model.backbone.merge_and_unload()
fast = FastQwen35(model.backbone); d = fast.L[3]; sync = torch.cuda.synchronize

def t(fn, reps=50, warm=10):
    for _ in range(warm): fn()
    ts = []
    for _ in range(reps):
        sync(); a = time.perf_counter(); fn(); sync(); ts.append((time.perf_counter() - a) * 1e6)
    return statistics.median(ts)

with torch.inference_mode():
    for M in [int(x) for x in os.environ.get("MS", "288,512,768,1024,1536,2048,3072,4096").split(",")]:
        h = torch.randn(M, fast.H, device="cuda", dtype=torch.bfloat16)
        row = []
        for name, Kd, wsk, S, w in (("down", d.w_down.shape[1], d.w_down_sk, d.sk_down, d.w_down), ("o_proj", d.w_out.shape[1], d.w_out_sk, d.sk_out, d.w_out)):
            y = torch.randn(M, Kd, device="cuda", dtype=torch.bfloat16) * 0.05
            sk = t(lambda: fast._res(h, y, wsk, S, fast.in_w1[4], None))
            pl = t(lambda: fast._res(h, y, None, 1, fast.in_w1[4], None, w=w))
            row.append(f"{name} S={S}: split-K {sk:7.1f} us  plain {pl:7.1f} us  ({'split-K' if sk < pl else 'plain'} wins by {abs(sk - pl) / max(sk, pl) * 100:4.1f}%)")
        print(f"M={M:5d}  " + "  |  ".join(row), flush=True)
print("=== SK THRESHOLD DONE ===", flush=True)
