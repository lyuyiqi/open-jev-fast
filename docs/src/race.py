"""Animated race: one inference of the example request in three implementations, measured latencies replayed slower.
Writes race.gif (web page and README). Usage: python race.py <e2e_latency.json> <font> <outdir>"""
import json, subprocess, sys, io
import numpy as np, matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
from matplotlib.patches import FancyBboxPatch
import imageio_ffmpeg
E2E, FONT, OUT = sys.argv[1], sys.argv[2], sys.argv[3]
fm.fontManager.addfont(FONT); FAM = fm.FontProperties(fname=FONT).get_name(); plt.rcParams["font.family"] = FAM
BOLD = FONT.replace("Regular", "Bold")
try:
    fm.fontManager.addfont(BOLD); FAMB = fm.FontProperties(fname=BOLD).get_name()
except Exception:
    FAMB = FAM
E = {r["mode"]: r["p50_ms"] for r in json.load(open(E2E)) if r.get("scope") == "forward"}
lanes = [("Open-Jev, default PyTorch path", E["torch"], "#b9b7b1"), ("Open-Jev + FLA kernels", E["fla"], "#eb6834"), ("OpenJev-Fast", E["fast"], "#2a78d6")]
SLOW, FPS, HOLD = 40.0, 30, 3.0
T_END = max(l[1] for l in lanes) * SLOW / 1000 + HOLD
W, H, DPI = 1280, 560, 100
INK, INK2, TRACK = "#0f172a", "#64748b", "#eef1f5"
def frame(t):
    fig = plt.figure(figsize=(W / DPI, H / DPI), dpi=DPI); fig.patch.set_facecolor("white")
    ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, W); ax.set_ylim(0, H); ax.axis("off")
    ax.text(60, H - 58, "One inference, same request, same GPU", fontsize=24, color=INK, family=FAMB, va="center")
    ax.text(60, H - 98, f"Open-Jev-27B on one NVIDIA B300 · example request: 3 questions, 7 candidates, 539 tokens · measured latency replayed {SLOW:.0f}× slower",
            fontsize=12.5, color=INK2, va="center")
    real_ms = t * 1000 / SLOW
    x0, x1 = 400, 1000
    for i, (name, lat, col) in enumerate(lanes):
        y = H - 190 - i * 118
        ax.text(60, y, name, fontsize=16, color=INK, va="center", family=FAMB if i == 2 else FAM)
        ax.add_patch(FancyBboxPatch((x0, y - 17), x1 - x0, 34, boxstyle="round,pad=0,rounding_size=17", fc=TRACK, ec="none"))
        p = min(1.0, real_ms / lat)
        if p > 0.02:
            ax.add_patch(FancyBboxPatch((x0, y - 17), (x1 - x0) * p, 34, boxstyle="round,pad=0,rounding_size=17", fc=col, ec="none"))
        if p < 1:
            ax.text(x1 + 24, y, f"{real_ms:6.1f} ms", fontsize=17, color=INK2, va="center", family="monospace")
        else:
            ax.text(x1 + 24, y, f"✓ {lat:.1f} ms", fontsize=19, color=col if i else "#8a8883", va="center", family=FAMB)
    fast = lanes[2][1]
    if real_ms >= fast:
        a = min(1.0, (real_ms - fast) / 8.0)
        yb = H - 190 - 2 * 118 - 44
        ax.text(x0, yb, f"{lanes[1][1] / fast:.1f}× faster than Open-Jev + FLA,  {lanes[0][1] / fast:.1f}× faster than the default path",
                fontsize=14.5, color="#2a78d6", va="center", alpha=a, family=FAMB)
    ax.text(W - 60, 28, "forward pass + scoring head, median of 30 runs; OpenJev-Fast: github.com/lyuyiqi/open-jev-fast", fontsize=10.5, color="#94a3b8", ha="right", va="center")
    buf = io.BytesIO(); fig.savefig(buf, format="rgba", dpi=DPI); plt.close(fig)
    return np.frombuffer(buf.getvalue(), np.uint8).reshape(H, W, 4)[:, :, :3]
ff = imageio_ffmpeg.get_ffmpeg_exe()
n = int(T_END * FPS)
frames = [frame(k / FPS) for k in range(n)]
def encode(args, out):
    p = subprocess.Popen([ff, "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-", *args, out], stdin=subprocess.PIPE)
    for f in frames: p.stdin.write(f.tobytes())
    p.stdin.close(); p.wait()
# a GIF plays everywhere (autoplaying <video> is blocked by some browsers and in-app viewers)
encode(["-vf", "fps=20,scale=1024:-1:flags=lanczos,split[a][b];[a]palettegen=max_colors=96:stats_mode=diff[p];[b][p]paletteuse=dither=none:diff_mode=rectangle"], f"{OUT}/race.gif")
print("race ok", n, "frames")
