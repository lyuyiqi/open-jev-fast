"""Figures for docs/: latency ladder (ladder.{png,svg}) and JevBench CDF (jevbench_cdf.{png,svg}), from results/.
Run from docs/src/: python charts.py [repo root, default ../..] [font, default ../NotoSansSC-Regular.ttf].
Also writes ladder_en.png and cdf_en.png here for make_public_en.py."""
import json, sys, matplotlib; matplotlib.use("Agg"); matplotlib.rcParams["svg.hashsalt"] = "open-jev-fast"
import matplotlib.pyplot as plt, numpy as np
from matplotlib import font_manager as fm
REPO = sys.argv[1] if len(sys.argv) > 1 else "../.."
FONT = sys.argv[2] if len(sys.argv) > 2 else "../NotoSansSC-Regular.ttf"
fm.fontManager.addfont(FONT); plt.rcParams["font.family"] = fm.FontProperties(fname=FONT).get_name()
INK, INK2, GRID, BLUE, BLUE_L, BLUE_D, ORANGE, GRAY = "#0b0b0b", "#52514e", "#e6e5e1", "#2a78d6", "#a9c8ee", "#123f7a", "#eb6834", "#c9c7c1"
E2E = {r["mode"]: r["p50_ms"] for r in json.load(open(f"{REPO}/results/e2e_latency.json")) if r.get("scope") == "forward"}
# first two and last bars: forward pass + head, one harness (bench/e2e_bench.py); the rest: development measurements
steps = [("Default PyTorch path (HF Transformers)", E2E["torch"], "base"), ("+ flash-linear-attention kernels", E2E["fla"], "base"),
         ("Merge LoRA + fused RMSNorm + remove CPU syncs", 78.6, "a"), ("Whole-model CUDA Graph (exact length)", 42.5, "a"),
         ("+ causal-conv1d", 36.5, "a"), ("+ torch.compile (end of PyTorch path)", 32.4, "a"),
         ("Hand-written fused CUDA kernels (new path)", 31.0, "b"), ("+ Shared prefix computed once (1-level)", 24.4, "b"),
         ("+ split-K / cuBLASLt tuning / no copies", 22.8, "b"), ("+ Two-level prefix tree", 20.1, "b"),
         ("+ Prep rewrite, lookup tables, fewer kernels", 19.3, "c"), ("+ Hand-written Gated DeltaNet kernel", 17.9, "c"),
         ("+ Tree attention kernel, split-K 2 (final)", E2E["fast"], "c")]
fig, ax = plt.subplots(figsize=(7.8, 5.0), dpi=200)
y = list(range(len(steps)))[::-1]
cols = [{"base": GRAY, "a": BLUE_L, "b": BLUE, "c": BLUE_D}[s[2]] for s in steps]
ax.barh(y, [s[1] for s in steps], height=0.62, color=cols, edgecolor="white", linewidth=1.5)
for yi, s in zip(y, steps): ax.text(s[1] + 3, yi, f"{s[1]:.1f} ms", va="center", fontsize=8.3, color=INK)
ax.set_yticks(y); ax.set_yticklabels([s[0] for s in steps], fontsize=8.2, color=INK)
ax.set_xlim(0, 300); ax.set_xlabel("ms", fontsize=8.2, color=INK2)
fig.suptitle("Open-Jev-27B latency per request on one B300: default PyTorch path → open-jev-fast", fontsize=10.5, color=INK, x=0.5)
ax.xaxis.grid(True, color=GRID, linewidth=0.8); ax.set_axisbelow(True)
for sp in ("top", "right", "left"): ax.spines[sp].set_visible(False)
ax.spines["bottom"].set_color(GRID); ax.tick_params(axis="x", colors=INK2, labelsize=8); ax.tick_params(axis="y", length=0)
from matplotlib.patches import Patch
ax.legend(handles=[Patch(color=GRAY, label="Baselines"), Patch(color=BLUE_L, label="Phase 1: PyTorch-level optimization"),
                   Patch(color=BLUE, label="Phase 2: hand-written CUDA kernels + prefix tree"), Patch(color=BLUE_D, label="Phase 3: Gated DeltaNet and attention kernels")],
          loc="lower right", fontsize=7.8, frameon=False)
fig.tight_layout(); fig.savefig(f"{REPO}/docs/ladder.png", facecolor="white"); fig.savefig(f"{REPO}/docs/ladder.svg", facecolor="white")
fig.savefig("ladder_en.png", facecolor="white")
o = [r["latency_ms"] for r in json.load(open(f"{REPO}/results/jevbench_original.json"))]
v = [r["latency_ms"] for r in json.load(open(f"{REPO}/results/jevbench_fast.json"))]
fig, ax = plt.subplots(figsize=(7.2, 3.6), dpi=200)
for data, c, lab in ((o, ORANGE, "Original jev.server (with FLA)"), (v, BLUE, "open-jev-fast")):
    xs = np.sort(data); ys = np.arange(1, len(xs) + 1) / len(xs) * 100
    ax.step(xs, ys, where="post", color=c, linewidth=2, label=lab)
    ax.text(np.median(xs) * 1.08, 52, f"P50 {np.median(xs):.0f} ms", color=INK, fontsize=8)
ax.set_xscale("log"); ax.set_xlabel("ms (log scale)", fontsize=8.2, color=INK2); ax.set_ylabel("% of tasks", fontsize=8.2, color=INK2)
ax.set_title("JevBench per-task latency: original Open-Jev server vs. open-jev-fast (231 tasks)", fontsize=10.5, color=INK, loc="center", fontweight="normal", pad=10)
ax.grid(True, color=GRID, linewidth=0.8); ax.set_axisbelow(True)
for sp in ("top", "right"): ax.spines[sp].set_visible(False)
for sp in ("left", "bottom"): ax.spines[sp].set_color(GRID)
ax.tick_params(colors=INK2, labelsize=8); ax.legend(fontsize=8, frameon=False, loc="lower right")
fig.tight_layout(); fig.savefig(f"{REPO}/docs/jevbench_cdf.png", facecolor="white"); fig.savefig(f"{REPO}/docs/jevbench_cdf.svg", facecolor="white")
fig.savefig("cdf_en.png", facecolor="white")
print("charts ok")
