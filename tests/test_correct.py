"""Correctness harness: (1) each fused kernel vs PyTorch reference on real activations of layer 0/3,
(2) all 64 layers: fast forward vs HF reference forward (same merged weights), (3) head probabilities."""
import sys, json, time
from pathlib import Path
import os; sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
import torch, torch.nn.functional as F
import graph_patches  # noqa  (no-sync mask shortcuts, same math)
from jev.api import compile_request, candidate_prompts
from jev.metrics import softmax
from jev.model import DecisionModel
CKPT = Path(os.environ["OJ_CKPT"])
req = json.loads(Path(os.environ["OPEN_JEV_DIR"], "configs", "example-request.json").read_text())
records = compile_request(req["state"], req["questions"])
T0 = json.loads((CKPT / "temperature.json").read_text())["temperature"]
model = DecisionModel.load(CKPT); model.backbone = model.backbone.merge_and_unload()
core, tok = model.backbone, model.tokenizer
from fastmodel import FastQwen35, K
fast = FastQwen35(core)
prompts = [tok.apply_chat_template([{"role": "user", "content": p}], tokenize=False, add_generation_prompt=True, enable_thinking=False)
           for r in records for p in candidate_prompts(r)]
enc = tok(prompts, padding=True, return_tensors="pt"); ids, mask = enc["input_ids"].cuda(), enc["attention_mask"].cuda()
B, T = ids.shape; real = mask.bool()
def cmp(name, a, b, sel=None):
    a, b = a.float(), b.float()
    if sel is not None: a, b = a[sel], b[sel]
    d = (a - b).abs(); rel = d.max() / b.abs().max().clamp_min(1e-6)
    exact = (d == 0).float().mean()
    print(f"CMP {name:34s} maxabs {d.max():.3e}  rel {rel:.2e}  mean {d.mean():.2e}  bitexact {100*exact:.1f}%", flush=True)
    return d.max().item()
with torch.inference_mode():
    # ---------- (1) kernel unit tests on layer-0 (linear) and layer-3 (full) real inputs ----------
    emb = core.embed_tokens(ids)
    L0 = core.layers[0]; d0 = fast.L[0]
    ref_n = L0.input_layernorm(emb) * mask[..., None].to(emb.dtype)
    h, hn = K.add_rmsnorm(emb.reshape(B*T, -1).contiguous(), None, fast.in_w1[0], mask.reshape(-1).int().contiguous(), fast.eps)
    cmp("K1 rmsnorm(+mask) layer0", hn.view(B, T, -1), ref_n)
    delta = torch.randn_like(emb) * 0.5
    h2, hn2 = K.add_rmsnorm(emb.reshape(B*T, -1).contiguous(), delta.reshape(B*T, -1).contiguous(), fast.post_w1[0], None, fast.eps)
    refh = emb + delta
    cmp("K1 add(residual)", h2.view(B, T, -1), refh); cmp("K1 add+rmsnorm", hn2.view(B, T, -1), L0.post_attention_layernorm(refh))
    x = ref_n.reshape(B*T, -1)
    gu = F.linear(x, d0.w_gu)
    ref_m = L0.mlp.act_fn(L0.mlp.gate_proj(x)) * L0.mlp.up_proj(x)
    cmp("merged gate+up GEMM (vs 2 GEMMs)", gu[:, :gu.shape[1]//2], L0.mlp.gate_proj(x))
    cmp("K2 silu_mul", K.silu_mul(gu), ref_m)
    la = L0.linear_attn
    proj = F.linear(x, d0.w_in)
    q, k, v, g, beta = K.linattn_prep(proj, d0.convw, d0.A_log, d0.dt_bias, T, d0.KD, d0.VD, d0.HV, d0.HK, d0.HD)
    hs = ref_n
    mq = la.in_proj_qkv(hs).transpose(1, 2)
    mq = la.causal_conv1d_fn(x=mq, weight=la.conv1d.weight.squeeze(1), bias=la.conv1d.bias, activation=la.activation, seq_idx=None).transpose(1, 2)
    rq, rk, rv = torch.split(mq, [la.key_dim, la.key_dim, la.value_dim], dim=-1)
    rq = rq.reshape(B, T, -1, la.head_k_dim).repeat_interleave(3, dim=2); rk = rk.reshape(B, T, -1, la.head_k_dim).repeat_interleave(3, dim=2)
    rbeta = la.in_proj_b(hs).sigmoid(); rg = -la.A_log.float().exp() * F.softplus(la.in_proj_a(hs).float() + la.dt_bias)
    cmp("K3 conv+silu -> q (expanded)", q.view(B, T, 48, 128), rq); cmp("K3 k", k.view(B, T, 48, 128), rk)
    cmp("K3 v", v.view(B, T, -1), rv); cmp("K3 beta", beta.view(B, T, -1), rbeta); cmp("K3 g (fp32)", g.view(B, T, -1), rg)
    o, _ = la.chunk_gated_delta_rule(q.view(B, T, 48, 128), k.view(B, T, 48, 128), v.view(B, T, 48, 128), g=g.view(B, T, 48),
                                     beta=beta.view(B, T, 48), initial_state=None, output_final_state=False, use_qk_l2norm_in_kernel=True)
    z = la.in_proj_z(hs).reshape(-1, 128)
    ref_y = la.norm(o.reshape(-1, 128), z).reshape(B*T, -1)
    cmp("K4 gated_rmsnorm", K.gated_rmsnorm(o.contiguous(), proj, d0.normw, 48, 128, 2*d0.KD + d0.VD, d0.norm_eps), ref_y)
    cmp("whole linear-attn block layer0", F.linear(K.gated_rmsnorm(o.contiguous(), proj, d0.normw, 48, 128, 2*d0.KD + d0.VD, d0.norm_eps), d0.w_out).view(B, T, -1),
        la(hs, attention_mask=mask), real)
    # full attention layer 3
    L3 = core.layers[3]; d3 = fast.L[3]; sa = L3.self_attn
    x3 = L3.input_layernorm(emb)
    pos = torch.arange(T, device=ids.device).view(1, 1, -1).expand(3, B, -1)
    cos, sin = core.rotary_emb(emb, pos)
    proj3 = F.linear(x3.reshape(B*T, -1), d3.w_in)
    qt, kt, vt, gate = K.fullattn_prep(proj3, d3.qw1, d3.kw1, cos.reshape(B*T, -1).contiguous(), sin.reshape(B*T, -1).contiguous(), B, T, 24, 4, 256, fast.eps)
    from transformers.models.qwen3_5.modeling_qwen3_5 import apply_rotary_pos_emb
    rq3, rg3 = torch.chunk(sa.q_proj(x3).view(B, T, -1, 512), 2, dim=-1)
    rq3 = sa.q_norm(rq3).transpose(1, 2); rk3 = sa.k_norm(sa.k_proj(x3).view(B, T, -1, 256)).transpose(1, 2)
    rq3, rk3 = apply_rotary_pos_emb(rq3, rk3, cos, sin)
    cmp("K5 q (norm+rope)", qt, rq3); cmp("K5 k (norm+rope)", kt, rk3)
    cmp("K5 v", vt, sa.v_proj(x3).view(B, T, -1, 256).transpose(1, 2)); cmp("K5 gate", gate.view(B, T, -1), rg3.reshape(B, T, -1))
    causal = torch.ones(T, T, dtype=torch.bool, device=ids.device).tril()
    amask = causal[None, None] & mask.bool()[:, None, None, :]
    att = F.scaled_dot_product_attention(qt, kt, vt, attn_mask=amask, scale=d3.scale, enable_gqa=True)
    ya = F.linear(K.gate_mul(att.contiguous(), gate, T, 24, 256), d3.w_out).view(B, T, -1)
    from transformers.masking_utils import create_causal_mask
    cm = create_causal_mask(config=core.config, inputs_embeds=emb, attention_mask=mask, past_key_values=None, position_ids=pos[0])
    refa, _ = sa(x3, position_embeddings=(cos, sin), attention_mask=cm)
    cmp("K5+SDPA+K6 whole full-attn block L3", ya, refa, real)

    # ---------- (2) all 64 layers + final norm ----------
    ref_layers = []
    hooks = [L.register_forward_hook(lambda m, a, o: ref_layers.append(o[0] if isinstance(o, tuple) else o)) for L in core.layers]
    ref_out = core(input_ids=ids, attention_mask=mask, use_cache=False).last_hidden_state
    for hk in hooks: hk.remove()
    fast_layers = []
    out = fast(input_ids=ids, attention_mask=mask, capture=fast_layers).last_hidden_state
    worst_layer = 0.0
    for li, (a_, b_) in enumerate(zip(fast_layers, ref_layers)):
        dd = (a_.float() - b_.float())[real].abs().max().item(); rr = dd / b_.float()[real].abs().max().item()
        worst_layer = max(worst_layer, rr)
        if li in (0, 1, 2, 3, 7, 15, 31, 47, 63): print(f"LAYER {li:2d} ({fast.layer_types[li][:4]}) maxabs {dd:.3e} rel {rr:.2e}", flush=True)
    print(f"LAYER worst relative diff over 64 layers: {worst_layer:.2e}", flush=True)
    worst = cmp("FINAL last_hidden_state (real tokens)", out, ref_out, real)
    idx = torch.arange(B, device=ids.device); last = mask.sum(-1) - 1
    cmp("FINAL last-token hidden (what the head reads)", out[idx, last], ref_out[idx, last])
    # ---------- (3) probabilities ----------
    def probs(hid):
        s = model.head(hid[idx, last].float()).squeeze(-1); outp, off = [], 0
        for r in records:
            c = len(candidate_prompts(r)); v_ = s[off:off+c]
            if r["kind"] == "noul": v_ = torch.stack([torch.zeros_like(v_[0]), v_[0]])
            outp.append(softmax(v_.float().cpu().tolist(), temperature=T0)); off += c
        return outp
    pr, pf = probs(ref_out), probs(out)
    print("PROBS ref ", [[round(x, 4) for x in p] for p in pr], flush=True)
    print("PROBS fast", [[round(x, 4) for x in p] for p in pf], flush=True)
    print(f"PROB maxdiff {max(abs(a-b) for x, y in zip(pr, pf) for a, b in zip(x, y)):.2e}", flush=True)
print("=== CORRECT DONE ===", flush=True)
