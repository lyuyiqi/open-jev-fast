"""FLA chunk_gated_delta_rule: q/k replicated to HV heads (current) vs native GVA (q/k with HK heads). Same values; compare output + time."""
import statistics, time, torch
from fla.ops.gated_delta_rule import chunk_gated_delta_rule
torch.manual_seed(0)
HK, HV, D = 16, 48, 128; rep = HV // HK; sync = torch.cuda.synchronize

def t(fn, reps=30, warm=5):
    for _ in range(warm): fn()
    ts = []
    for _ in range(reps):
        sync(); a = time.perf_counter(); fn(); sync(); ts.append((time.perf_counter() - a) * 1e6)
    return statistics.median(ts)

with torch.inference_mode():
    for B, T in ((1, 288), (7, 96), (1, 3680), (1, 16384)):
        q = torch.nn.functional.normalize(torch.randn(B, T, HK, D, device="cuda"), dim=-1).bfloat16()
        k = torch.nn.functional.normalize(torch.randn(B, T, HK, D, device="cuda"), dim=-1).bfloat16()
        v = (torch.randn(B, T, HV, D, device="cuda") * 0.5).bfloat16()
        g = -torch.rand(B, T, HV, device="cuda") * 0.5
        beta = torch.rand(B, T, HV, device="cuda").bfloat16()
        qr, kr = q.repeat_interleave(rep, dim=2), k.repeat_interleave(rep, dim=2)
        f_rep = lambda: chunk_gated_delta_rule(qr, kr, v, g, beta, scale=D ** -0.5, output_final_state=False, use_qk_l2norm_in_kernel=False)[0]
        f_gva = lambda: chunk_gated_delta_rule(q, k, v, g, beta, scale=D ** -0.5, output_final_state=False, use_qk_l2norm_in_kernel=False)[0]
        o1, o2 = f_rep(), f_gva()
        diff = (o1.float() - o2.float()).abs()
        print(f"B={B} T={T:5d}  replicated {t(f_rep):8.1f} us  GVA {t(f_gva):8.1f} us  | max|diff| {diff.max().item():.3e}  "
              f"bitexact {(diff == 0).float().mean().item() * 100:.2f}%", flush=True)
print("=== GVA DONE ===", flush=True)
