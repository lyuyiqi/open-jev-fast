"""FastQwen35: drop-in replacement for the Qwen3.5/3.8 text backbone forward (inference, no cache).
GEMMs: cuBLAS on merged weight matrices. Everything else: fused CUDA kernels (kernels.cu).
Linear-attention core: FLA chunk_gated_delta_rule (unchanged). Full attention: torch SDPA."""
import torch, torch.nn.functional as F
from types import SimpleNamespace
from fla.ops.gated_delta_rule import chunk_gated_delta_rule
from fla.ops.utils import chunk_local_cumsum
from fla.ops.utils.constant import RCP_LN2
from fla.ops.gated_delta_rule.chunk_fwd import chunk_gated_delta_rule_fwd_intra
from fla.ops.common.chunk_delta_h import chunk_gated_delta_rule_fwd_h
from fla.ops.common.chunk_o import chunk_fwd_o


from fla.ops.gated_delta_rule.chunk_fwd import chunk_gated_delta_rule_fwd_kkt_solve_kernel
from fla.ops.gated_delta_rule.wy_fast import recompute_w_u_fwd
import triton as _triton
_A_BUF = {}


def gdn_intra(k, v, g, beta):
    """FLA's chunk_gated_delta_rule_fwd_intra (chunk 64 branch) with the A buffer zeroed once per shape and reused: the
    fused kkt+solve kernel rewrites the same entries on every call, so the per-call torch.zeros (a fill kernel) is not
    needed. Same kernels, same arguments."""
    B, T, H, Kd = k.shape; HV = beta.shape[2]
    key = (B, T, HV, k.device.index)
    A = _A_BUF.get(key)
    if A is None:
        A = _A_BUF[key] = torch.zeros(B, T, HV, 64, device=k.device, dtype=k.dtype)
    chunk_gated_delta_rule_fwd_kkt_solve_kernel[(_triton.cdiv(T, 64), B * HV)](
        k=k, g=g, beta=beta, A=A, cu_seqlens=None, chunk_indices=None, T=T, H=H, HV=HV, K=Kd, BT=64, BC=16)
    return recompute_w_u_fwd(k=k, v=v, beta=beta, A=A, g=g, cu_seqlens=None, chunk_indices=None)


def gdn_split(q, k, krep, v, g, beta, scale):
    """FLA chunk_gated_delta_rule forward, stage by stage (same functions as FLA's own forward), feeding the intra-chunk
    kkt/solve/w-u stage a key tensor replicated to the value heads (its fast path) and the state/output stages the
    grouped (HK-head) keys and queries (their fast path)."""
    gc = chunk_local_cumsum(g, chunk_size=64, scale=RCP_LN2)
    w, u = gdn_intra(krep, v, gc, beta) if A_CACHE else chunk_gated_delta_rule_fwd_intra(k=krep, v=v, g=gc, beta=beta, chunk_size=64)[:2]
    h, v_new, _ = chunk_gated_delta_rule_fwd_h(k=k, w=w, u=u, g=gc, initial_state=None, output_final_state=False, chunk_size=64)
    return chunk_fwd_o(q=q, k=k, v=v_new, h=h, g=gc, scale=scale, chunk_size=64)
from ext import load
K = load()
import os as _os
SK_MAX_M = int(_os.environ.get("OJ_SK_MAX_M", "1000000000"))
GVA = _os.environ.get("OJ_GVA", "1") == "1"
PREP_TCH = int(_os.environ.get("OJ_PREP_TCH", "1"))
PREP = _os.environ.get("OJ_PREP", "dedup")
SK2 = _os.environ.get("OJ_SK2", "1") == "1"            # split-K reduce + residual + RMSNorm with S unrolled (bit-identical)
ADDMASK = _os.environ.get("OJ_ADDMASK", "1") == "1"    # precomputed additive bf16 mask (SDPA would convert the bool mask per layer)
A_CACHE = _os.environ.get("OJ_A_CACHE", "1") == "1"      # reuse FLA's zeroed A buffer (drops a fill kernel per layer)
GDN_FUSED = _os.environ.get("OJ_GDN_FUSED", "1") == "1"   # C6 hand-written fused Gated DeltaNet kernel (replaces FLA on the gtree path)
GDN_VER = int(_os.environ.get("OJ_GDN_VER", "6"))   # 3: fused GDN; 4: + cp.async prefetch of the next chunk; 5: + gated RMSNorm in the epilogue (rejected: slower); 6: 65..96-row paths with both chunks' state-independent work side by side (else 4)
FLA_SPLIT = _os.environ.get("OJ_FLA_SPLIT", "1") == "1"   # FLA stage by stage: replicated k for kkt, GVA q/k for h and o
SK_LIN_OUT = int(_os.environ.get("OJ_SK_LIN_OUT", "2"))   # linear-attn out_proj split-K (M~288: 2 is best)
SK_FULL_OUT = int(_os.environ.get("OJ_SK_FULL_OUT", "2"))   # full-attn o_proj split-K
SK_DOWN = int(_os.environ.get("OJ_SK_DOWN", "2"))           # MLP down_proj split-K (M~288: 2 beats 4 by ~0.2 ms)
FATTN = int(_os.environ.get("OJ_FATTN", "3"))              # C8 fused tree attention (+ gate) instead of SDPA + gate_mul3: 0 = off, 3 / 6 = q heads per CTA; 4 = 2 warps per head (fattn_tree3); 2 = prep fused too (fattn_tree2; rejected: 36.7 us)
FATTN_MAX = int(_os.environ.get("OJ_FATTN_MAX", "384"))     # packed rows up to which it beats cuDNN SDPA (288: 12.7 vs 21.4 us; 576: 35.2 vs 34.2 us)
LT_SK = _os.environ.get("OJ_LT_SK", "0") == "1"             # split-K partial GEMMs through cuBLASLt (per-shape search) instead of torch.bmm
LUT = _os.environ.get("OJ_LUT", "1") == "1"
PF = int(_os.environ.get("OJ_PF", "0"))            # (rejected in r4: no GEMM gain, +0.18 ms) L2 prefetch of out_proj weights during prep/FLA/SDPA (0 = off)
PF_CHUNK = int(_os.environ.get("OJ_PF_CHUNK", "65536"))      # bit-exact activation lookup tables (bf16 inputs)
REPQK = _os.environ.get("OJ_REPQK", "0") == "1"   # 1: replicated q/k (FLA kkt_solve faster, prep slower; net ~equal)
try:
    from lt_ext import load as _lt_load
    LT = _lt_load() if _os.environ.get("OJ_USE_LT", "1") == "1" else None
except Exception as _e:
    print("cuBLASLt autotune unavailable:", _e); LT = None

class FastQwen35(torch.nn.Module):
    def __init__(self, core, free_original=False, v2=True, fuse_l2=True):
        super().__init__()
        self.v2, self.fuse_l2 = v2, fuse_l2
        self._causal = {}; self._lt_best = {}; self._sk_best = {}
        cfg = core.config
        self.cfg, self.core = cfg, core
        self.H = cfg.hidden_size; self.eps = cfg.rms_norm_eps
        self.layer_types = list(cfg.layer_types)
        self.embed = core.embed_tokens
        self.rotary = core.rotary_emb
        w1 = lambda norm: (1.0 + norm.weight.float()).contiguous()
        self.in_w1 = [w1(L.input_layernorm) for L in core.layers]
        self.post_w1 = [w1(L.post_attention_layernorm) for L in core.layers]
        self.final_w1 = w1(core.norm)
        self.L = []
        with torch.no_grad():
            for i, L in enumerate(core.layers):
                d = SimpleNamespace()
                d.w_gu = torch.cat([L.mlp.gate_proj.weight, L.mlp.up_proj.weight], 0).contiguous()
                d.w_down = L.mlp.down_proj.weight
                d.sk_down = SK_DOWN; d.w_down_sk = self._split(d.w_down, d.sk_down)
                if self.layer_types[i] == "linear_attention":
                    la = L.linear_attn
                    d.w_in = torch.cat([la.in_proj_qkv.weight, la.in_proj_z.weight, la.in_proj_b.weight, la.in_proj_a.weight], 0).contiguous()
                    d.w_out = la.out_proj.weight
                    d.sk_out = SK_LIN_OUT                                 # env OJ_SK_LIN_OUT (1 = plain GEMM)
                    d.w_out_sk = self._split(d.w_out, d.sk_out) if d.sk_out > 1 else None
                    d.convw = la.conv1d.weight.squeeze(1).contiguous()
                    d.A_log, d.dt_bias = la.A_log.contiguous(), la.dt_bias.contiguous()
                    d.normw = la.norm.weight.contiguous()
                    d.norm_eps = getattr(la.norm, "eps", getattr(la.norm, "variance_epsilon", self.eps))
                    d.KD, d.VD, d.HV, d.HK, d.HD = la.key_dim, la.value_dim, la.num_v_heads, la.num_k_heads, la.head_v_dim
                    assert la.head_k_dim == la.head_v_dim
                else:
                    sa = L.self_attn
                    d.w_in = torch.cat([sa.q_proj.weight, sa.k_proj.weight, sa.v_proj.weight], 0).contiguous()
                    d.w_out = sa.o_proj.weight
                    d.sk_out = SK_FULL_OUT; d.w_out_sk = self._split(d.w_out, d.sk_out)
                    d.qw1, d.kw1 = w1(sa.q_norm), w1(sa.k_norm)
                    d.HQ, d.HKV, d.D, d.scale = cfg.num_attention_heads, cfg.num_key_value_heads, sa.head_dim, sa.scaling
                self.L.append(d)

    def _tune(self, x, w):
        # time torch's default GEMM against every cuBLASLt heuristic algorithm for this exact shape; keep the fastest valid one
        M, Kd = x.shape; N = w.shape[0]
        def tm(fn, reps=20):
            for _ in range(3): fn()
            torch.cuda.synchronize(); a, b = torch.cuda.Event(True), torch.cuda.Event(True); a.record()
            for _ in range(reps): fn()
            b.record(); torch.cuda.synchronize(); return a.elapsed_time(b) / reps
        ref = F.linear(x, w); best, bi = tm(lambda: F.linear(x, w)), -1
        tol = ref.float().abs().max().item() * 1e-2 + 1e-3
        for i in range(LT.lt_setup(M, N, Kd, 32)):
            try:
                if (LT.lt_matmul(x, w, i).float() - ref.float()).abs().max().item() > tol: continue
                t = tm(lambda: LT.lt_matmul(x, w, i))
                if t < best: best, bi = t, i
            except Exception:
                pass
        self._lt_best[(M, N, Kd)] = bi
        return bi

    def _tune_sk(self, y, wsk, S):
        # split-K partial GEMM: torch.bmm vs every cuBLASLt candidate of the batched plan; timed with the L2 flushed
        # before each call (the out_proj weights alone would fit in L2 and bias the choice)
        M, Kd = y.shape; N = wsk.shape[2]
        if not hasattr(self, "_flush"): self._flush = torch.empty(256 << 20, dtype=torch.uint8, device=y.device)
        def tm(fn, reps=12):
            for _ in range(2): fn()
            ev = [(torch.cuda.Event(True), torch.cuda.Event(True)) for _ in range(reps)]
            for a, b in ev: self._flush.zero_(); a.record(); fn(); b.record()
            torch.cuda.synchronize(); return sum(a.elapsed_time(b) for a, b in ev) / reps
        base = lambda: torch.bmm(y.view(M, S, Kd // S).transpose(0, 1), wsk, out_dtype=torch.float32)
        ref = base(); best, choice = tm(base), None
        tol = ref.abs().max().item() * 1e-2 + 1e-3
        h = LT.lt2_plan(M, N, Kd, S, 1, 1)
        for i in range(LT.lt2_search(h, 400, 3)):
            try:
                if (LT.lt2_run(h, y, wsk, i) - ref).abs().max().item() > tol: continue
                t = tm(lambda: LT.lt2_run(h, y, wsk, i))
                if t < best: best, choice = t, (h, i)
            except Exception:
                pass
        self._sk_best[(M, N, Kd, S)] = choice
        return choice

    def _skmm(self, y, wsk, S):
        M, Kd = y.shape
        if LT is not None and LT_SK:
            key = (M, wsk.shape[2], Kd, S)
            if key in self._sk_best: c = self._sk_best[key]
            elif torch.cuda.is_current_stream_capturing(): c = None
            else: c = self._tune_sk(y, wsk, S)
            if c is not None: return LT.lt2_run(c[0], y, wsk, c[1])
        return torch.bmm(y.view(M, S, Kd // S).transpose(0, 1), wsk, out_dtype=torch.float32)

    def _mm(self, x, w):
        if LT is None: return F.linear(x, w)
        key = (x.shape[0], w.shape[0], x.shape[1])
        idx = self._lt_best.get(key)
        if idx is None:
            if torch.cuda.is_current_stream_capturing(): return F.linear(x, w)
            idx = self._tune(x, w)
        return F.linear(x, w) if idx < 0 else LT.lt_matmul(x, w, idx)

    @staticmethod
    def _pack_bits(vis):
        # [N, N] bool -> [N, N/32] int32, bit b of word w = key 32w + b visible
        if K is not None: return K.pack_bits(vis.contiguous())
        N = vis.shape[0]
        w = (vis.view(N, N // 32, 32).to(torch.int64) << torch.arange(32, device=vis.device)).sum(-1)
        return (w - (w >= 2 ** 31).to(torch.int64) * 2 ** 32).to(torch.int32).contiguous()

    @staticmethod
    def _split(W, S):
        N, Kd = W.shape
        return W.view(N, S, Kd // S).permute(1, 2, 0).contiguous()          # [S, K/S, N]

    def _silu_mul(self, gu):
        if LUT:
            if not hasattr(self, "_silu_tab"): self._silu_tab, self._sig_tab = K.act_tables(gu)
            return K.silu_mul_lut(gu, self._silu_tab)
        return K.silu_mul(gu)

    def _res(self, h, y, wsk, S, w1, mask, w=None):
        # split-K GEMM with fp32 partials (batched -> fills more SMs; fp32 partial sums are more accurate than a
        # single bf16-output GEMM) + fused partial-sum/residual/RMSNorm. S == 1 -> plain GEMM + fused add/norm.
        M, Kd = y.shape
        if S == 1 or (w is not None and M >= SK_MAX_M):   # large M: enough tiles already; fp32 partials would cost more than they save
            return K.add_rmsnorm(h, self._mm(y, w), w1, mask, self.eps)
        parts = self._skmm(y, wsk, S)
        return (K.add_rmsnorm_sk2 if SK2 else K.add_rmsnorm_sk)(h, parts, w1, mask, self.eps)

    @torch.no_grad()
    def forward(self, input_ids=None, attention_mask=None, capture=None, **kw):
        B, T = input_ids.shape; N = B * T
        x = self.embed(input_ids).reshape(N, self.H).contiguous()
        rowmask = attention_mask.reshape(N).to(torch.int32).contiguous()
        pos = torch.arange(T, device=x.device).view(1, 1, -1).expand(3, B, -1)
        cos, sin = self.rotary(x.view(B, T, -1), pos)
        cos = cos.reshape(N, -1).contiguous(); sin = sin.reshape(N, -1).contiguous()
        keym = attention_mask.bool()
        if T not in self._causal: self._causal[T] = torch.ones(T, T, dtype=torch.bool, device=x.device).tril()
        causal = self._causal[T]
        amask = (causal[None, None] & keym[:, None, None, :])
        lin = [t == "linear_attention" for t in self.layer_types]
        h, hn = K.add_rmsnorm(x, None, self.in_w1[0], rowmask if lin[0] else None, self.eps)
        for i, d in enumerate(self.L):
            proj = self._mm(hn, d.w_in)
            if lin[i]:
                if self.v2:
                    q, k, v, g, beta = K.linattn_prep2(proj, d.convw, d.A_log, d.dt_bias, B, T, d.KD, d.VD, d.HV, d.HK, d.HD, self.fuse_l2)
                else:
                    q, k, v, g, beta = K.linattn_prep(proj, d.convw, d.A_log, d.dt_bias, T, d.KD, d.VD, d.HV, d.HK, d.HD)
                o, _ = chunk_gated_delta_rule(q.view(B, T, d.HV, d.HD), k.view(B, T, d.HV, d.HD), v.view(B, T, d.HV, d.HD),
                                              g=g.view(B, T, d.HV), beta=beta.view(B, T, d.HV), initial_state=None,
                                              output_final_state=False, use_qk_l2norm_in_kernel=not (self.v2 and self.fuse_l2))
                y = K.gated_rmsnorm(o.contiguous(), proj, d.normw, d.HV, d.HD, 2 * d.KD + d.VD, d.norm_eps)
            else:
                prep = K.fullattn_prep2 if self.v2 else K.fullattn_prep
                qt, kt, vt, gate = prep(proj, d.qw1, d.kw1, cos, sin, B, T, d.HQ, d.HKV, d.D, self.eps)
                att = F.scaled_dot_product_attention(qt, kt, vt, attn_mask=amask, scale=d.scale, enable_gqa=True)
                y = (K.gate_mul2 if self.v2 else K.gate_mul)(att.contiguous(), gate, T, d.HQ, d.D)
            h, hn = self._res(h, y, d.w_out_sk, d.sk_out, self.post_w1[i], None, d.w_out)
            m = self._silu_mul(self._mm(hn, d.w_gu))
            last = i == len(self.L) - 1
            h, hn = self._res(h, m, d.w_down_sk, d.sk_down, self.final_w1 if last else self.in_w1[i + 1],
                              None if last or not lin[i + 1] else rowmask, w=d.w_down)
            if capture is not None: capture.append(h.view(B, T, self.H))
        return SimpleNamespace(last_hidden_state=hn.view(B, T, self.H))


    # ------------------------------------------------------------------------------------------
    # Prefix-tree forward: shared prefix computed once, S suffixes branch from its final states.
    # Packed rows: [prefix (Lp) | suffix_0 (Ls) | ... | suffix_{S-1} (Ls)], one weight pass.
    def tree_layout(self, pre_ids, suf_ids, suf_mask, pad=0):
        """static per-structure tensors. The first `pad` prefix rows are left-padding (masked; exact: zero input keeps the
        linear-attention state at zero, keys are masked, positions shifted so real tokens start at 0)."""
        dev = suf_ids.device; Lp = pre_ids.shape[0]; S, Ls = suf_ids.shape; N = Lp + S * Ls
        seg = torch.cat([torch.zeros(Lp, dtype=torch.long, device=dev), torch.arange(1, S + 1, device=dev).repeat_interleave(Ls)])
        tpos = torch.cat([torch.arange(Lp, device=dev), torch.arange(Ls, device=dev).repeat(S)])
        pre_valid = torch.arange(Lp, device=dev) >= pad
        valid = torch.cat([pre_valid, suf_mask.reshape(-1).bool()])
        pos = torch.where(seg == 0, tpos - pad, tpos + (Lp - pad))
        qs, ks = seg[:, None], seg[None, :]
        qt_, kt_ = tpos[:, None], tpos[None, :]
        vis = ((ks == 0) & ((qs != 0) | (kt_ <= qt_))) | ((ks == qs) & (qs != 0) & (kt_ <= qt_))
        vis = (vis & valid[None, :]) | torch.eye(N, dtype=torch.bool, device=dev)   # self-visibility: no all-masked rows (NaN)
        lastidx = Lp + torch.arange(S, device=dev) * Ls + suf_mask.sum(-1) - 1
        pp = torch.arange(Lp + Ls, device=dev)[None, :].expand(S, -1)
        repidx = torch.where(pp < Lp, pp, Lp + torch.arange(S, device=dev)[:, None] * Ls + (pp - Lp)).contiguous()
        return SimpleNamespace(Lp=Lp, S=S, Ls=Ls, N=N, pos=pos, amask=vis[None, None].contiguous(), rowmask=valid.to(torch.int32).contiguous(),
                               lastidx=lastidx, repidx=repidx)

    @torch.no_grad()
    def forward_tree(self, pre_ids, suf_ids, lay, fla_mode="replicate"):
        Lp, S, Ls, N = lay.Lp, lay.S, lay.Ls, lay.N
        ids = torch.cat([pre_ids, suf_ids.reshape(-1)])
        x = self.embed(ids).contiguous()
        cos, sin = self.rotary(x.view(1, N, -1), lay.pos.view(1, 1, N).expand(3, 1, N))
        cos = cos.reshape(N, -1).contiguous(); sin = sin.reshape(N, -1).contiguous()
        lin = [t == "linear_attention" for t in self.layer_types]
        rowmask = lay.rowmask
        h, hn = K.add_rmsnorm(x, None, self.in_w1[0], rowmask if lin[0] else None, self.eps)
        for i, d in enumerate(self.L):
            proj = self._mm(hn, d.w_in)
            if lin[i]:
                HV, HD = d.HV, d.HD
                if fla_mode == "replicate":
                    # linear-attn core on full sequences: prep kernel writes the prefix rows in front of every suffix
                    # (replicated layout), gated norm reads FLA's output back through a row map -> no gathers/copies.
                    L2 = Lp + Ls
                    q, k, v, g, beta = K.linattn_prep_rep(proj, d.convw, d.A_log, d.dt_bias, Lp, S, Ls, d.KD, d.VD, d.HV, d.HK, d.HD)
                    o_r, _ = chunk_gated_delta_rule(q.view(S, L2, HV, HD), k.view(S, L2, HV, HD), v.view(S, L2, HV, HD),
                                                    g=g.view(S, L2, HV), beta=beta.view(S, L2, HV), initial_state=None,
                                                    output_final_state=False, use_qk_l2norm_in_kernel=False)
                    y = K.gated_rmsnorm_map(o_r, proj, d.normw, d.HV, d.HD, 2 * d.KD + d.VD, d.norm_eps, Lp, Ls)
                    h, hn = self._res(h, y, d.w_out_sk, d.sk_out, self.post_w1[i], None, d.w_out)
                    m_ = self._silu_mul(self._mm(hn, d.w_gu))
                    last = i == len(self.L) - 1
                    h, hn = self._res(h, m_, d.w_down_sk, d.sk_down, self.final_w1 if last else self.in_w1[i + 1],
                                      None if last or not lin[i + 1] else rowmask, w=d.w_down)
                    continue
                q, k, v, g, beta = K.linattn_prep_tree(proj, d.convw, d.A_log, d.dt_bias, Lp, S, Ls, d.KD, d.VD, d.HV, d.HK, d.HD)
                o_pre, st = chunk_gated_delta_rule(q[:Lp].view(1, Lp, HV, HD), k[:Lp].view(1, Lp, HV, HD), v[:Lp].view(1, Lp, HV, HD),
                                                   g=g[:Lp].view(1, Lp, HV), beta=beta[:Lp].view(1, Lp, HV), initial_state=None,
                                                   output_final_state=True, use_qk_l2norm_in_kernel=False)
                o_suf, _ = chunk_gated_delta_rule(q[Lp:].view(S, Ls, HV, HD), k[Lp:].view(S, Ls, HV, HD), v[Lp:].view(S, Ls, HV, HD),
                                                  g=g[Lp:].view(S, Ls, HV), beta=beta[Lp:].view(S, Ls, HV),
                                                  initial_state=st.expand(S, -1, -1, -1).contiguous(),
                                                  output_final_state=False, use_qk_l2norm_in_kernel=False)
                o = torch.cat([o_pre.reshape(Lp, HV, HD), o_suf.reshape(S * Ls, HV, HD)], 0)
                y = K.gated_rmsnorm(o, proj, d.normw, d.HV, d.HD, 2 * d.KD + d.VD, d.norm_eps)
            else:
                qt, kt, vt, gate = K.fullattn_prep2(proj, d.qw1, d.kw1, cos, sin, 1, N, d.HQ, d.HKV, d.D, self.eps)
                att = F.scaled_dot_product_attention(qt, kt, vt, attn_mask=lay.amask, scale=d.scale, enable_gqa=True)
                y = K.gate_mul2(att.contiguous(), gate, N, d.HQ, d.D)
            h, hn = self._res(h, y, d.w_out_sk, d.sk_out, self.post_w1[i], None, d.w_out)
            m = self._silu_mul(self._mm(hn, d.w_gu))
            last = i == len(self.L) - 1
            h, hn = self._res(h, m, d.w_down_sk, d.sk_down, self.final_w1 if last else self.in_w1[i + 1],
                              None if last or not lin[i + 1] else rowmask, w=d.w_down)
        return hn[lay.lastidx]

    # ------------------------------------------------------------------------------------------
    # Two-level prefix tree: root (shared by all candidates) -> one node per record/question -> one leaf per candidate.
    # Packed rows: [root | q_0 .. q_{R-1} | leaf_0 .. leaf_{S-1}], each node right-padded to its bucket.
    # Linear attention runs on every candidate's full path (replicated layout via src map, padding only at the end);
    # full attention uses an ancestor mask on the packed rows.
    @staticmethod
    def gtree_build(seqs, groups, pad_id, NB=32, PB=16, dev="cuda"):
        """Nodes packed back to back (no per-node padding): [root | q_0 .. q_{R-1} | leaf_0 .. leaf_{S-1} | end padding].
        Only the total row count N (bucket NB) and the path length L2 (bucket PB) define the tensor shapes, so one CUDA graph
        serves every tree with the same (N_b, S, L2_b); the structure itself lives in the index tensors."""
        import numpy as np
        S = len(seqs); R = max(groups) + 1
        lcp = lambda ss, start, lim: next((i for i in range(start, lim) if any(s[i] != ss[0][i] for s in ss)), lim)
        Lr = lcp(seqs, 0, min(map(len, seqs)) - 1)
        q_len = []
        for r in range(R):
            gs = [seqs[c] for c in range(S) if groups[c] == r]
            q_len.append(lcp(gs, Lr, min(map(len, gs)) - 1) - Lr)
        l_len = [len(seqs[c]) - Lr - q_len[groups[c]] for c in range(S)]
        ceil = lambda x, b: ((x + b - 1) // b) * b
        Nreal = Lr + sum(q_len) + sum(l_len)
        N = ceil(Nreal, NB); L2 = ceil(max(len(sq) for sq in seqs), PB)
        ids = np.full(N, pad_id, np.int64); valid = np.zeros(N, bool); pos = np.zeros(N, np.int64); node = np.full(N, -1, np.int64)
        src = np.full((S, L2), -1, np.int32); inv = np.zeros(N, np.int32); last = np.zeros(S, np.int64)
        qb, off = [], Lr
        for r in range(R): qb.append(off); off += q_len[r]
        lb = []
        for c in range(S): lb.append(off); off += l_len[c]
        ids[:Lr] = seqs[0][:Lr]; valid[:Lr] = True; pos[:Lr] = np.arange(Lr); node[:Lr] = 0; inv[:Lr] = np.arange(Lr)
        first = {}
        for c in range(S):
            r = groups[c]; first.setdefault(r, c); a, b = Lr + q_len[r], Lr + q_len[r] + l_len[c]
            src[c, :Lr] = np.arange(Lr); src[c, Lr:a] = qb[r] + np.arange(q_len[r]); src[c, a:b] = lb[c] + np.arange(l_len[c])
            ids[lb[c]:lb[c] + l_len[c]] = seqs[c][a:]; valid[lb[c]:lb[c] + l_len[c]] = True
            pos[lb[c]:lb[c] + l_len[c]] = a + np.arange(l_len[c]); node[lb[c]:lb[c] + l_len[c]] = 1 + R + c
            inv[lb[c]:lb[c] + l_len[c]] = c * L2 + a + np.arange(l_len[c]); last[c] = lb[c] + l_len[c] - 1
        for r in range(R):
            c = first[r]
            ids[qb[r]:qb[r] + q_len[r]] = seqs[c][Lr:Lr + q_len[r]]; valid[qb[r]:qb[r] + q_len[r]] = True
            pos[qb[r]:qb[r] + q_len[r]] = Lr + np.arange(q_len[r]); node[qb[r]:qb[r] + q_len[r]] = 1 + r
            inv[qb[r]:qb[r] + q_len[r]] = c * L2 + Lr + np.arange(q_len[r])
        nn = 2 + R + S                                                 # last node index = end padding
        node[node < 0] = nn - 1
        anc = np.zeros((nn, nn), bool)
        for r in range(R): anc[1 + r, 0] = True
        for c in range(S): anc[1 + R + c, 0] = True; anc[1 + R + c, 1 + groups[c]] = True
        # packed row -> replicated slots (CSR); group N collects the end-padding slots. Lets the linear-attention prep
        # compute each packed row once and write it to every candidate path that contains it.
        flat = src.reshape(-1); grp = np.where(flat >= 0, flat, N)
        rep_pos = np.argsort(grp, kind="stable").astype(np.int32)
        rep_ptr = np.concatenate([[0], np.cumsum(np.bincount(grp, minlength=N + 1))]).astype(np.int32)
        # conv history of each packed row along its path (rows t-3..t), from one representative slot; -1 = none
        hist = np.full((N + 1, 4), -1, np.int32)
        hn = np.nonzero(rep_ptr[:N] < rep_ptr[1:N + 1])[0]
        sc, tc = np.divmod(rep_pos[rep_ptr[hn]].astype(np.int64), L2)
        for i in range(4):
            tt = tc - 3 + i; ok = tt >= 0
            hist[hn[ok], i] = src[sc[ok], tt[ok]]
        # canonical slot of each real packed row (the fused GDN kernel writes the normed output there); -1 elsewhere
        canon = np.full(S * L2, -1, np.int32); canon[inv[valid]] = np.nonzero(valid)[0].astype(np.int32)
        # visibility on the host (ancestor nodes + causal inside a node + self), bits for the fused attention kernel
        ar = np.arange(N)
        vis = anc[node[:, None], node[None, :]] | ((node[:, None] == node[None, :]) & (ar[None, :] <= ar[:, None]))
        vis = (vis & valid[None, :]) | np.eye(N, dtype=bool)
        vbits = np.packbits(vis, axis=1, bitorder="little").view("<i4").reshape(N, N // 32)
        # two host-to-device copies (int64 and int32 fields) instead of one per field
        n64 = N + (-N) % 4                                    # 32-byte aligned int64 fields
        i64 = np.zeros(2 * n64 + len(last), np.int64); i64[:N] = ids; i64[n64:n64 + N] = pos; i64[2 * n64:] = last
        g64 = torch.as_tensor(i64, device=dev)
        parts32 = [src.reshape(-1), inv, rep_ptr, rep_pos, hist.reshape(-1), canon, valid.astype(np.int32), vbits.reshape(-1)]
        # every field starts on a 32-byte boundary (kernels read hist as int4, vbits/src with vector loads)
        pad = lambda x: np.concatenate([x.astype(np.int32), np.zeros((-len(x)) % 8, np.int32)])
        g32 = torch.as_tensor(np.concatenate([pad(x) for x in parts32]), device=dev)
        o = np.cumsum([0] + [len(x) + (-len(x)) % 8 for x in parts32])
        v32 = lambda k, *shape: g32[o[k]:o[k] + len(parts32[k])].view(*shape) if shape else g32[o[k]:o[k] + len(parts32[k])]
        gvis = torch.as_tensor(vis, device=dev)
        lay = SimpleNamespace(N=N, S=S, R=R, L2=L2, Lr=Lr, Nreal=Nreal, key=(N, S, L2),
                              pos=g64[n64:n64 + N], amask=gvis[None, None], rowmask=v32(6),
                              amask_add=torch.zeros((N, N), dtype=torch.bfloat16, device=dev).masked_fill_(~gvis, float("-inf"))[None, None],
                              vbits=v32(7, N, N // 32),
                              src=v32(0, S, L2), inv=v32(1), lastidx=g64[2 * n64:],
                              rep_ptr=v32(2), rep_pos=v32(3), hist=v32(4, N + 1, 4), canon=v32(5))
        return g64[:N], lay

    @torch.no_grad()
    def forward_gtree(self, ids, lay):
        N, S, L2 = lay.N, lay.S, lay.L2
        x = self.embed(ids).contiguous()
        cos, sin = self.rotary(x.view(1, N, -1), lay.pos.view(1, 1, N).expand(3, 1, N))
        cos = cos.reshape(N, -1).contiguous(); sin = sin.reshape(N, -1).contiguous()
        lin = [t == "linear_attention" for t in self.layer_types]
        rowmask = lay.rowmask
        h, hn = K.add_rmsnorm(x, None, self.in_w1[0], rowmask if lin[0] else None, self.eps)
        cur = torch.cuda.current_stream()
        if PF and not hasattr(self, "_pf_stream"): self._pf_stream = torch.cuda.Stream()
        for i, d in enumerate(self.L):
            proj = self._mm(hn, d.w_in)
            if PF:   # weights of this layer's output projection -> L2, overlapped with prep/FLA or attention
                self._pf_stream.wait_stream(cur)
                with torch.cuda.stream(self._pf_stream):
                    K.l2_prefetch(d.w_out_sk if d.w_out_sk is not None else d.w_out, PF_CHUNK)
            if lin[i]:
                HV, HD = d.HV, d.HD
                if PREP == "dedup" and GDN_FUSED:   # hand-written fused Gated DeltaNet kernel (mma.sync / tf32 merges), GVA q/k
                    q, k, v, g, beta = K.linattn_prep_dedup2(proj, lay.hist, lay.rep_ptr, lay.rep_pos, lay.L2, d.convw, d.A_log, d.dt_bias, d.KD, d.VD, d.HV, d.HK, d.HD, False)
                    ga = (q.view(S, L2, d.HK, HD), k.view(S, L2, d.HK, HD), v.view(S, L2, HV, HD), g.view(S, L2, HV), beta.view(S, L2, HV), HD ** -0.5)
                    if GDN_VER == 7:
                        if not hasattr(self, "_sig_tab"): self._silu_tab, self._sig_tab = K.act_tables(proj)
                        y = K.gdn_fused7(*ga, proj, 2 * d.KD + d.VD, d.normw, self._sig_tab, lay.canon, rowmask, d.norm_eps)
                    elif GDN_VER == 5:
                        if not hasattr(self, "_sig_tab"): self._silu_tab, self._sig_tab = K.act_tables(proj)
                        y = K.gdn_fused5(*ga, proj, 2 * d.KD + d.VD, d.normw, self._sig_tab, lay.canon, rowmask, d.norm_eps)
                    else:
                        o_r = (K.gdn_fused6 if GDN_VER == 6 else K.gdn_fused4 if GDN_VER == 4 else K.gdn_fused3)(*ga)
                elif PREP == "dedup" and FLA_SPLIT:
                    q, k, krep, v, g, beta = K.linattn_prep_dedup3(proj, lay.hist, lay.rep_ptr, lay.rep_pos, lay.L2, d.convw, d.A_log, d.dt_bias, d.KD, d.VD, d.HV, d.HK, d.HD)
                    if False:
                        pass
                    else:
                        o_r = gdn_split(q.view(S, L2, d.HK, HD), k.view(S, L2, d.HK, HD), krep.view(S, L2, HV, HD), v.view(S, L2, HV, HD),
                                        g.view(S, L2, HV), beta.view(S, L2, HV), HD ** -0.5)
                elif PREP == "dedup":   # each packed row computed once, written to every path slot; q/k with HK heads (GVA)
                    q, k, v, g, beta = K.linattn_prep_dedup2(proj, lay.hist, lay.rep_ptr, lay.rep_pos, lay.L2, d.convw, d.A_log, d.dt_bias, d.KD, d.VD, d.HV, d.HK, d.HD, REPQK)
                    HQK = HV if REPQK else d.HK
                elif GVA:   # q/k with HK heads; FLA applies grouped-value attention (bit-identical to replicating q/k)
                    q, k, v, g, beta = K.linattn_prep_map2(proj, lay.src, d.convw, d.A_log, d.dt_bias, d.KD, d.VD, d.HV, d.HK, d.HD, PREP_TCH)
                    HQK = d.HK
                else:
                    q, k, v, g, beta = K.linattn_prep_map(proj, lay.src, d.convw, d.A_log, d.dt_bias, d.KD, d.VD, d.HV, d.HK, d.HD)
                    HQK = HV
                if not (PREP == "dedup" and (FLA_SPLIT or GDN_FUSED)):
                    o_r, _ = chunk_gated_delta_rule(q.view(S, L2, HQK, HD), k.view(S, L2, HQK, HD), v.view(S, L2, HV, HD),
                                                    g=g.view(S, L2, HV), beta=beta.view(S, L2, HV), initial_state=None,
                                                    output_final_state=False, use_qk_l2norm_in_kernel=False)
                if PREP == "dedup" and GDN_FUSED and GDN_VER in (5, 7):
                    pass
                elif LUT:
                    if not hasattr(self, "_sig_tab"): self._silu_tab, self._sig_tab = K.act_tables(proj)
                    y = K.gated_rmsnorm_inv2(o_r, lay.inv, proj, d.normw, self._sig_tab, d.HV, d.HD, 2 * d.KD + d.VD, d.norm_eps)
                else:
                    y = K.gated_rmsnorm_inv(o_r, lay.inv, proj, d.normw, d.HV, d.HD, 2 * d.KD + d.VD, d.norm_eps)
            else:
                if FATTN == 2 and N <= FATTN_MAX and hasattr(lay, "vbits"):   # prep + tree attention + gate in one kernel
                    if not hasattr(self, "_sig_tab"): self._silu_tab, self._sig_tab = K.act_tables(proj)
                    y = K.fattn_tree2(proj, d.qw1, d.kw1, cos, sin, self._sig_tab, lay.vbits, d.HQ, d.HKV, d.scale, self.eps)
                else:
                    qt, kt, vt, gate = K.fullattn_prep2(proj, d.qw1, d.kw1, cos, sin, 1, N, d.HQ, d.HKV, d.D, self.eps)
                    if FATTN in (3, 6) and N <= FATTN_MAX and hasattr(lay, "vbits"):   # fused tree attention + gate (no dense mask)
                        if not hasattr(self, "_sig_tab"): self._silu_tab, self._sig_tab = K.act_tables(proj)
                        y = K.fattn_tree(qt, kt, vt, gate, self._sig_tab, lay.vbits, d.scale, FATTN)
                    elif FATTN == 4 and N <= FATTN_MAX and hasattr(lay, "vbits"):   # two warps per head over alternate key blocks
                        if not hasattr(self, "_sig_tab"): self._silu_tab, self._sig_tab = K.act_tables(proj)
                        y = K.fattn_tree3(qt, kt, vt, gate, self._sig_tab, lay.vbits, d.scale)
                    else:
                        att = F.scaled_dot_product_attention(qt, kt, vt, attn_mask=lay.amask_add if ADDMASK else lay.amask, scale=d.scale, enable_gqa=True)
                        if LUT:   # strided read of the SDPA output (no contiguous copy) + sigmoid table
                            if not hasattr(self, "_sig_tab"): self._silu_tab, self._sig_tab = K.act_tables(proj)
                            y = K.gate_mul3(att, gate, self._sig_tab, N, d.HQ, d.D)
                        else:
                            y = K.gate_mul2(att.contiguous(), gate, N, d.HQ, d.D)
            if PF: cur.wait_stream(self._pf_stream)
            h, hn = self._res(h, y, d.w_out_sk, d.sk_out, self.post_w1[i], None, d.w_out)
            m = self._silu_mul(self._mm(hn, d.w_gu))
            last = i == len(self.L) - 1
            h, hn = self._res(h, m, d.w_down_sk, d.sk_down, self.final_w1 if last else self.in_w1[i + 1],
                              None if last or not lin[i + 1] else rowmask, w=d.w_down)
        return hn[lay.lastidx]
