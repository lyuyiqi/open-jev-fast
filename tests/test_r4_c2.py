"""Round 4 C2 check: lookup-table silu_mul and gated RMSNorm vs the previous kernels on real activations (bit-exact),
kernel GPU times (CUDA graph), and whole-model scores with OJ_LUT on/off."""
import json, os, sys
from pathlib import Path
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
import torch
import graph_patches  # noqa: F401
from jev.api import candidate_prompts, compile_request
from jev.model import DecisionModel
import fastmodel
from fastmodel import FastQwen35, K
from fla.ops.gated_delta_rule import chunk_gated_delta_rule
req = json.loads(Path(os.environ["OPEN_JEV_DIR"], "configs", "example-request.json").read_text())
recs = compile_request(req["state"], req["questions"])
model = DecisionModel.load(Path(os.environ["OJ_CKPT"])); model.backbone = model.backbone.merge_and_unload(); tok = model.tokenizer
fast = FastQwen35(model.backbone)
seqs, groups = [], []
for ri, r in enumerate(recs):
    for p in candidate_prompts(r):
        seqs.append(tok(tok.apply_chat_template([{"role": "user", "content": p}], tokenize=False, add_generation_prompt=True, enable_thinking=False))["input_ids"]); groups.append(ri)
ids, lay = FastQwen35.gtree_build(seqs, groups, tok.pad_token_id)

def gpu_us(fn, per=20, reps=20):
    st = torch.cuda.Stream(); st.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(st):
        for _ in range(3): fn()
    torch.cuda.current_stream().wait_stream(st); torch.cuda.synchronize()
    gr = torch.cuda.CUDAGraph()
    with torch.cuda.graph(gr):
        for _ in range(per): fn()
    for _ in range(3): gr.replay()
    a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    torch.cuda.synchronize(); a.record()
    for _ in range(reps): gr.replay()
    b.record(); torch.cuda.synchronize()
    return a.elapsed_time(b) * 1e3 / (reps * per)

ok = True
with torch.inference_mode():
    silu_tab, sig_tab = K.act_tables(ids)
    x = fast.embed(ids).reshape(-1, fast.H).contiguous()
    h, hn = K.add_rmsnorm(x, None, fast.in_w1[0], lay.rowmask, fast.eps)
    for li in (0, 1, 3):
        d = fast.L[li]
        gu = fast._mm(hn, d.w_gu)
        a, b = K.silu_mul(gu), K.silu_mul_lut(gu, silu_tab)
        eq = torch.equal(a, b); ok &= eq
        print(f"SILU layer {li}: bit-identical {eq}  old {gpu_us(lambda: K.silu_mul(gu)):.1f} us  lut {gpu_us(lambda: K.silu_mul_lut(gu, silu_tab)):.1f} us", flush=True)
    for li in (0, 1, 2):
        d = fast.L[li]
        proj = fast._mm(hn, d.w_in)
        q, k, v, g, beta = K.linattn_prep_dedup2(proj, lay.hist, lay.rep_ptr, lay.rep_pos, lay.L2, d.convw, d.A_log, d.dt_bias, d.KD, d.VD, d.HV, d.HK, d.HD, False)
        S, L2 = lay.S, lay.L2
        o_r, _ = chunk_gated_delta_rule(q.view(S, L2, d.HK, d.HD), k.view(S, L2, d.HK, d.HD), v.view(S, L2, d.HV, d.HD), g=g.view(S, L2, d.HV),
                                        beta=beta.view(S, L2, d.HV), initial_state=None, output_final_state=False, use_qk_l2norm_in_kernel=False)
        zoff = 2 * d.KD + d.VD
        a = K.gated_rmsnorm_inv(o_r, lay.inv, proj, d.normw, d.HV, d.HD, zoff, d.norm_eps)
        b = K.gated_rmsnorm_inv2(o_r, lay.inv, proj, d.normw, sig_tab, d.HV, d.HD, zoff, d.norm_eps)
        eq = torch.equal(a, b); ok &= eq
        ta = gpu_us(lambda: K.gated_rmsnorm_inv(o_r, lay.inv, proj, d.normw, d.HV, d.HD, zoff, d.norm_eps))
        tb = gpu_us(lambda: K.gated_rmsnorm_inv2(o_r, lay.inv, proj, d.normw, sig_tab, d.HV, d.HD, zoff, d.norm_eps))
        print(f"GATED layer {li}: bit-identical {eq}  old {ta:.1f} us  new {tb:.1f} us", flush=True)
    outs = {}
    for flag in (False, True):
        fastmodel.LUT = flag
        outs[flag] = model.head(fast.forward_gtree(ids, lay).float()).squeeze(-1)
    same = torch.equal(outs[False], outs[True]); ok &= same
    print(f"MODEL scores LUT off vs on: bit-identical {same}", flush=True)
print("RESULT", "PASS" if ok else "FAIL", flush=True)
