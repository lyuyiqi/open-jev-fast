"""Real-time race: the 231 JevBench tasks sent one after another to each server, replayed at 1x speed from the measured
per-task latencies (results/jevbench_original.json, results/jevbench_fast.json). Writes race.gif.
Usage: python race_jev.py <repo> <font> <outdir>"""
import json, subprocess, sys, io, itertools
import numpy as np, matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
from matplotlib.patches import Rectangle
import imageio_ffmpeg
REPO, FONT, OUT = sys.argv[1], sys.argv[2], sys.argv[3]
fm.fontManager.addfont(FONT); FAM = fm.FontProperties(fname=FONT).get_name(); plt.rcParams["font.family"] = FAM
BOLD = FONT.replace("Regular", "Bold"); fm.fontManager.addfont(BOLD); FAMB = fm.FontProperties(fname=BOLD).get_name()
lat = {k: [r["latency_ms"] for r in json.load(open(f"{REPO}/results/jevbench_{k}.json"))] for k in ("original", "fast")}
done = {k: np.array(list(itertools.accumulate(v))) / 1000.0 for k, v in lat.items()}      # completion times, s
T_FAST, T_ORIG = done["fast"][-1], done["original"][-1]
FPS, HOLD = 20, 3.5
T_END = T_FAST + HOLD
W, H, DPI, SS = 1280, 620, 100, 2          # SS: supersampling (rendered at 2x)
INK, INK2, PEND = "#0f172a", "#64748b", "#e8ebf0"
COLS, ROWS = 21, 11                                   # 231 tasks
panels = [("original", "Open-Jev (jev.server with FLA)", "#eb6834", 60), ("fast", "OpenJev-Fast", "#2a78d6", 664)]
def frame(t):
    tc = min(t, T_FAST)
    fig = plt.figure(figsize=(W / DPI, H / DPI), dpi=DPI); fig.patch.set_facecolor("white")
    ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, W); ax.set_ylim(0, H); ax.axis("off")
    ax.text(60, H - 50, "231 JevBench tasks, one after another, in real time", fontsize=23, color=INK, family=FAMB, va="center")
    ax.text(60, H - 88, "Open-Jev-27B on one NVIDIA B300 · measured latency of every task (HTTP, one request at a time, both servers warmed up) · played at 1× speed",
            fontsize=11.5, color=INK2, va="center")
    ax.text(W - 60, H - 50, f"{tc:4.1f} s", fontsize=24, color=INK, family="monospace", ha="right", va="center")
    cell, gap = 22, 4
    for key, name, col, x0 in panels:
        n = int((done[key] <= tc).sum())
        ax.text(x0, H - 150, name, fontsize=17, color=INK, family=FAMB if key == "fast" else FAM, va="center")
        top = H - 185
        for i in range(231):
            r, c = divmod(i, COLS)
            ax.add_patch(Rectangle((x0 + c * (cell + gap), top - (r + 1) * (cell + gap)), cell, cell, fc=col if i < n else PEND, ec="none"))
        yb = top - ROWS * (cell + gap) - 34
        if key == "fast" and tc >= T_FAST:
            ax.text(x0, yb, f"✓ all 231 tasks in {T_FAST:.1f} s", fontsize=18, color=col, family=FAMB, va="center")
        else:
            ax.text(x0, yb, f"{n} / 231 tasks done", fontsize=18, color=INK if n else INK2, va="center")
        if key == "original" and t >= T_FAST:
            a = min(1.0, (t - T_FAST) / 0.8)
            ax.text(x0, yb - 36, f"needs {T_ORIG:.1f} s for all 231 ({T_ORIG / T_FAST:.1f}× longer)", fontsize=14, color=col, alpha=a, va="center")
    ax.text(W - 60, 22, "per-task latencies from results/ in github.com/lyuyiqi/open-jev-fast", fontsize=10, color="#94a3b8", ha="right", va="center")
    buf = io.BytesIO(); fig.savefig(buf, format="rgba", dpi=DPI * SS); plt.close(fig)
    return np.frombuffer(buf.getvalue(), np.uint8).reshape(H * SS, W * SS, 4)[:, :, :3]
ff = imageio_ffmpeg.get_ffmpeg_exe()
frames = [frame(k / FPS) for k in range(int(T_END * FPS))]
p = subprocess.Popen([ff, "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W * SS}x{H * SS}", "-r", str(FPS), "-i", "-",
                      "-vf", "scale=1920:-1:flags=lanczos,split[a][b];[a]palettegen=max_colors=96:stats_mode=diff[p];[b][p]paletteuse=dither=none:diff_mode=rectangle",
                      f"{OUT}/race.gif"], stdin=subprocess.PIPE)
for f in frames: p.stdin.write(f.tobytes())
p.stdin.close(); p.wait()
from PIL import Image
Image.fromarray(frames[len(frames) // 3]).save(f"{OUT}/race_mid.png"); Image.fromarray(frames[-1]).save(f"{OUT}/race_end.png")
print("race ok", len(frames), "frames", round(T_FAST, 2), round(T_ORIG, 2))
