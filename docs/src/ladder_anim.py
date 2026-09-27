"""Animated latency ladder: one bar per optimization step, each starting at the previous latency and shrinking to its own,
until the final 17.3 ms. Values: bench/e2e_bench.py for the two baselines and the final bar (results/e2e_latency.json),
development measurements for the steps in between (same values as docs/ladder.svg). Writes ladder.gif.
Usage: python ladder_anim.py <repo> <font> <outdir>"""
import json, subprocess, sys, io
import numpy as np, matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
from matplotlib.patches import FancyBboxPatch
import imageio_ffmpeg
REPO, FONT, OUT = sys.argv[1], sys.argv[2], sys.argv[3]
fm.fontManager.addfont(FONT); FAM = fm.FontProperties(fname=FONT).get_name(); plt.rcParams["font.family"] = FAM
BOLD = FONT.replace("Regular", "Bold"); fm.fontManager.addfont(BOLD); FAMB = fm.FontProperties(fname=BOLD).get_name()
E = {r["mode"]: r["p50_ms"] for r in json.load(open(f"{REPO}/results/e2e_latency.json")) if r.get("scope") == "forward"}
steps = [("Default PyTorch path (HF Transformers)", E["torch"], 0), ("+ flash-linear-attention kernels", E["fla"], 0),
         ("Merge LoRA + fused RMSNorm + remove CPU syncs", 78.6, 1), ("Whole-model CUDA Graph (exact length)", 42.5, 1),
         ("+ causal-conv1d", 36.5, 1), ("+ torch.compile", 32.4, 1),
         ("Hand-written fused CUDA kernels", 31.0, 2), ("+ Shared prefix computed once", 24.4, 2),
         ("+ split-K / cuBLASLt tuning", 22.8, 2), ("+ Two-level prefix tree", 20.1, 2),
         ("+ Prep rewrite, lookup tables", 19.3, 3), ("+ Hand-written Gated DeltaNet kernel", 17.9, 3),
         ("+ Tree attention kernel (final)", E["fast"], 3)]
PHASE = [("Baselines", "#b9b7b1"), ("Phase 1: PyTorch level", "#9fc2ec"), ("Phase 2: CUDA kernels + prefix tree", "#2a78d6"), ("Phase 3: Gated DeltaNet + attention kernels", "#123f7a")]
W, H, DPI, SS, FPS = 1280, 640, 100, 2, 20
SHRINK, HOLD, FINAL = 0.55, 0.55, 3.5
INK, INK2 = "#0f172a", "#64748b"
X0, X1, VMAX = 470, 1000, 260.0
TOP, ROWH = H - 150, 34
ease = lambda u: 1 - (1 - u) ** 3
def state(t):
    k, rem = divmod(t, SHRINK + HOLD)
    k = int(k)
    if k >= len(steps): return len(steps) - 1, 1.0
    return k, ease(min(1.0, rem / SHRINK))
def frame(t):
    k, u = state(t)
    fig = plt.figure(figsize=(W / DPI, H / DPI), dpi=DPI); fig.patch.set_facecolor("white")
    ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, W); ax.set_ylim(0, H); ax.axis("off")
    ax.text(60, H - 50, "Open-Jev-27B, one inference: step by step", fontsize=23, color=INK, family=FAMB, va="center")
    ax.text(60, H - 86, "example request (3 questions, 7 candidates, 539 tokens) on one NVIDIA B300 · forward pass + scoring head, median latency",
            fontsize=11.5, color=INK2, va="center")
    cur = None
    for i in range(k + 1):
        name, v, ph = steps[i]
        prev = steps[i - 1][1] if i else v
        val = v if i < k else prev + (v - prev) * u
        y = TOP - i * ROWH
        a = 1.0 if i < k else min(1.0, 0.25 + u)
        ax.text(X0 - 14, y, name, fontsize=12.2, color=INK, ha="right", va="center", alpha=a, family=FAMB if i == len(steps) - 1 else FAM)
        wbar = max(6, (X1 - X0) * val / VMAX)
        ax.add_patch(FancyBboxPatch((X0, y - 11), wbar, 22, boxstyle="round,pad=0,rounding_size=6", fc=PHASE[ph][1], ec="none"))
        ax.text(X0 + wbar + 10, y, f"{val:.1f} ms", fontsize=12, color=INK, va="center", alpha=a)
        cur = val
    # big readout
    base = steps[0][1]
    ax.text(W - 60, H - 190, f"{cur:.1f} ms", fontsize=40, color=PHASE[steps[k][2]][1] if steps[k][2] else "#8a8883", family=FAMB, ha="right", va="center")
    ax.text(W - 60, H - 245, f"{base / cur:.1f}× faster than the default path" if k else "starting point", fontsize=15, color=INK2, ha="right", va="center")
    ph = steps[k][2]
    ax.text(W - 60, H - 285, PHASE[ph][0], fontsize=13, color=PHASE[ph][1] if ph else "#8a8883", ha="right", va="center", family=FAMB)
    if k == len(steps) - 1 and u >= 1.0:
        ax.text(W - 60, H - 325, f"{steps[1][1] / cur:.1f}× faster than Open-Jev + FLA", fontsize=15, color="#123f7a", ha="right", va="center", family=FAMB)
    ax.text(W - 60, 22, "github.com/lyuyiqi/open-jev-fast", fontsize=10, color="#94a3b8", ha="right", va="center")
    buf = io.BytesIO(); fig.savefig(buf, format="rgba", dpi=DPI * SS); plt.close(fig)
    return np.frombuffer(buf.getvalue(), np.uint8).reshape(H * SS, W * SS, 4)[:, :, :3]
T_END = len(steps) * (SHRINK + HOLD) + FINAL
frames = [frame(i / FPS) for i in range(int(T_END * FPS))]
ff = imageio_ffmpeg.get_ffmpeg_exe()
p = subprocess.Popen([ff, "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W * SS}x{H * SS}", "-r", str(FPS), "-i", "-",
                      "-vf", "scale=1920:-1:flags=lanczos,split[a][b];[a]palettegen=max_colors=128:stats_mode=diff[p];[b][p]paletteuse=dither=none:diff_mode=rectangle",
                      f"{OUT}/ladder.gif"], stdin=subprocess.PIPE)
for f in frames: p.stdin.write(f.tobytes())
p.stdin.close(); p.wait()
from PIL import Image
Image.fromarray(frames[int(4.2 * FPS)]).save(f"{OUT}/ladder_mid.png"); Image.fromarray(frames[-1]).save(f"{OUT}/ladder_end.png")
print("ladder ok", len(frames), "frames")
