"""FastQwen35: drop-in replacement for the Qwen3.5/3.8 text backbone forward (inference, no cache).
GEMMs: cuBLAS on merged weight matrices. Everything else: fused CUDA kernels (kernels.cu).
Linear-attention core: FLA chunk_gated_delta_rule (unchanged). Full attention: torch SDPA."""
import torch, torch.nn.functional as F
from types import SimpleNamespace
from fla.ops.gated_delta_rule import chunk_gated_delta_rule
from ext import load
K = load()
import os as _os
try:
    from lt_ext import load as _lt_load
    LT = _lt_load() if _os.environ.get("OJ_USE_LT", "1") == "1" else None
except Exception as _e:
    print("cuBLASLt autotune unavailable:", _e); LT = None

class FastQwen35(torch.nn.Module):
    def __init__(self, core, free_original=False, v2=True, fuse_l2=True):
        super().__init__()
        self.v2, self.fuse_l2 = v2, fuse_l2
        self._causal = {}; self._lt_best = {}
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
                d.sk_down = 4; d.w_down_sk = self._split(d.w_down, d.sk_down)
                if self.layer_types[i] == "linear_attention":
                    la = L.linear_attn
                    d.w_in = torch.cat([la.in_proj_qkv.weight, la.in_proj_z.weight, la.in_proj_b.weight, la.in_proj_a.weight], 0).contiguous()
                    d.w_out = la.out_proj.weight
                    d.sk_out = 1; d.w_out_sk = None                       # split-K does not pay with fp32 partials here
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
                    d.sk_out = 2; d.w_out_sk = self._split(d.w_out, d.sk_out)
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

    def _mm(self, x, w):
        if LT is None: return F.linear(x, w)
        key = (x.shape[0], w.shape[0], x.shape[1])
        idx = self._lt_best.get(key)
        if idx is None:
            if torch.cuda.is_current_stream_capturing(): return F.linear(x, w)
            idx = self._tune(x, w)
        return F.linear(x, w) if idx < 0 else LT.lt_matmul(x, w, idx)

    @staticmethod
    def _split(W, S):
        N, Kd = W.shape
        return W.view(N, S, Kd // S).permute(1, 2, 0).contiguous()          # [S, K/S, N]

    def _res(self, h, y, wsk, S, w1, mask, w=None):
        # split-K GEMM with fp32 partials (batched -> fills more SMs; fp32 partial sums are more accurate than a
        # single bf16-output GEMM) + fused partial-sum/residual/RMSNorm. S == 1 -> plain GEMM + fused add/norm.
        if S == 1:
            return K.add_rmsnorm(h, self._mm(y, w), w1, mask, self.eps)
        M, Kd = y.shape
        parts = torch.bmm(y.view(M, S, Kd // S).transpose(0, 1), wsk, out_dtype=torch.float32)
        return K.add_rmsnorm_sk(h, parts, w1, mask, self.eps)

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
            m = K.silu_mul(self._mm(hn, d.w_gu))
            last = i == len(self.L) - 1
            h, hn = self._res(h, m, d.w_down_sk, d.sk_down, self.final_w1 if last else self.in_w1[i + 1],
                              None if last or not lin[i + 1] else rowmask)
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
                    m_ = K.silu_mul(self._mm(hn, d.w_gu))
                    last = i == len(self.L) - 1
                    h, hn = self._res(h, m_, d.w_down_sk, d.sk_down, self.final_w1 if last else self.in_w1[i + 1],
                                      None if last or not lin[i + 1] else rowmask)
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
            m = K.silu_mul(self._mm(hn, d.w_gu))
            last = i == len(self.L) - 1
            h, hn = self._res(h, m, d.w_down_sk, d.sk_down, self.final_w1 if last else self.in_w1[i + 1],
                              None if last or not lin[i + 1] else rowmask)
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
        t = lambda a: torch.as_tensor(a, device=dev)
        nd = t(node); ar = torch.arange(N, device=dev); vd = t(valid)
        vis = t(anc)[nd[:, None], nd[None, :]] | ((nd[:, None] == nd[None, :]) & (ar[None, :] <= ar[:, None]))
        vis = (vis & vd[None, :]) | torch.eye(N, dtype=torch.bool, device=dev)
        lay = SimpleNamespace(N=N, S=S, R=R, L2=L2, Lr=Lr, Nreal=Nreal, key=(N, S, L2),
                              pos=t(pos), amask=vis[None, None].contiguous(), rowmask=vd.to(torch.int32).contiguous(),
                              src=t(src).contiguous(), inv=t(inv).contiguous(), lastidx=t(last))
        return t(ids), lay

    @torch.no_grad()
    def forward_gtree(self, ids, lay):
        N, S, L2 = lay.N, lay.S, lay.L2
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
                q, k, v, g, beta = K.linattn_prep_map(proj, lay.src, d.convw, d.A_log, d.dt_bias, d.KD, d.VD, d.HV, d.HK, d.HD)
                o_r, _ = chunk_gated_delta_rule(q.view(S, L2, HV, HD), k.view(S, L2, HV, HD), v.view(S, L2, HV, HD),
                                                g=g.view(S, L2, HV), beta=beta.view(S, L2, HV), initial_state=None,
                                                output_final_state=False, use_qk_l2norm_in_kernel=False)
                y = K.gated_rmsnorm_inv(o_r, lay.inv, proj, d.normw, d.HV, d.HD, 2 * d.KD + d.VD, d.norm_eps)
            else:
                qt, kt, vt, gate = K.fullattn_prep2(proj, d.qw1, d.kw1, cos, sin, 1, N, d.HQ, d.HKV, d.D, self.eps)
                att = F.scaled_dot_product_attention(qt, kt, vt, attn_mask=lay.amask, scale=d.scale, enable_gqa=True)
                y = K.gate_mul2(att.contiguous(), gate, N, d.HQ, d.D)
            h, hn = self._res(h, y, d.w_out_sk, d.sk_out, self.post_w1[i], None, d.w_out)
            m = K.silu_mul(self._mm(hn, d.w_gu))
            last = i == len(self.L) - 1
            h, hn = self._res(h, m, d.w_down_sk, d.sk_down, self.final_w1 if last else self.in_w1[i + 1],
                              None if last or not lin[i + 1] else rowmask)
        return hn[lay.lastidx]
