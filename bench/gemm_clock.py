"""SM clock and power while the gate_up GEMM (M=288, cuBLASLt pick) runs back to back for a few seconds."""
import os, subprocess, sys, time, threading
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
import torch
from lt_ext import load
LT = load()
M, Kd, N = 288, 5120, 34816
x = torch.randn(M, Kd, device="cuda", dtype=torch.bfloat16); w = torch.randn(N, Kd, device="cuda", dtype=torch.bfloat16) * 0.02
n = LT.lt_setup(M, N, Kd, 32)
samples = []
def smi():
    for _ in range(12):
        samples.append(subprocess.run(["nvidia-smi", "--query-gpu=clocks.sm,power.draw,clocks_throttle_reasons.active", "--format=csv,noheader"],
                                      capture_output=True, text=True).stdout.strip()); time.sleep(0.25)
for _ in range(20): LT.lt_matmul(x, w, 0)
torch.cuda.synchronize()
th = threading.Thread(target=smi); th.start()
t0 = time.time(); k = 0
a, b = torch.cuda.Event(True), torch.cuda.Event(True); a.record()
while time.time() - t0 < 3.5:
    for _ in range(200): LT.lt_matmul(x, w, 0)
    k += 200
    torch.cuda.synchronize()
b.record(); torch.cuda.synchronize(); th.join()
print(f"CLOCK gate_up avg {a.elapsed_time(b) * 1e3 / k:.1f} us over {k} calls", flush=True)
for s_ in samples: print("CLOCK sample", s_, flush=True)
