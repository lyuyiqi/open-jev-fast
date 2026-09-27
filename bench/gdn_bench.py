import torch, torch.nn.functional as F
from fla.ops.gated_delta_rule import chunk_gated_delta_rule, fused_recurrent_gated_delta_rule
dev = "cuda"; torch.manual_seed(0)
def t(fn, reps=50):
    for _ in range(5): fn()
    torch.cuda.synchronize(); a, b = torch.cuda.Event(True), torch.cuda.Event(True)
    a.record()
    for _ in range(reps): fn()
    b.record(); torch.cuda.synchronize(); return a.elapsed_time(b) / reps * 1000
for (B, T) in ((7, 82), (7, 96), (5, 3782), (1, 364)):
    H, D = 48, 128
    q = F.normalize(torch.randn(B, T, H, D, device=dev), dim=-1).bfloat16(); k = F.normalize(torch.randn(B, T, H, D, device=dev), dim=-1).bfloat16()
    v = torch.randn(B, T, H, D, device=dev).bfloat16()
    g = -torch.rand(B, T, H, device=dev) * 0.5; beta = torch.rand(B, T, H, device=dev).bfloat16()
    oc, _ = chunk_gated_delta_rule(q, k, v, g=g, beta=beta, output_final_state=False, use_qk_l2norm_in_kernel=False)
    orr, _ = fused_recurrent_gated_delta_rule(q, k, v, g=g, beta=beta, output_final_state=False, use_qk_l2norm_in_kernel=False)
    # fp32 reference via recurrence
    tc = t(lambda: chunk_gated_delta_rule(q, k, v, g=g, beta=beta, output_final_state=False, use_qk_l2norm_in_kernel=False))
    tr = t(lambda: fused_recurrent_gated_delta_rule(q, k, v, g=g, beta=beta, output_final_state=False, use_qk_l2norm_in_kernel=False))
    d = (oc.float() - orr.float()).abs()
    print(f"GDN B={B} T={T}: chunk {tc:8.1f} us  recurrent {tr:8.1f} us  | chunk-vs-recurrent maxabs {d.max():.3e} mean {d.mean():.2e} (|o| max {oc.float().abs().max():.2f})", flush=True)
