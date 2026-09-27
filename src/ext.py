import os
from torch.utils.cpp_extension import load_inline
_HERE = os.path.dirname(os.path.abspath(__file__))
_CPP = """
#include <torch/extension.h>
std::vector<torch::Tensor> add_rmsnorm(torch::Tensor x, c10::optional<torch::Tensor> delta, torch::Tensor w1, c10::optional<torch::Tensor> rowmask, double eps);
torch::Tensor silu_mul(torch::Tensor gu);
std::vector<torch::Tensor> linattn_prep(torch::Tensor proj, torch::Tensor convw, torch::Tensor A_log, torch::Tensor dt_bias, int64_t T, int64_t KD, int64_t VD, int64_t HV, int64_t HK, int64_t HD);
torch::Tensor gated_rmsnorm(torch::Tensor x, torch::Tensor proj, torch::Tensor w, int64_t HV, int64_t HD, int64_t zoff, double eps);
std::vector<torch::Tensor> fullattn_prep(torch::Tensor proj, torch::Tensor qw1, torch::Tensor kw1, torch::Tensor cosb, torch::Tensor sinb, int64_t B, int64_t T, int64_t HQ, int64_t HKV, int64_t D, double eps);
torch::Tensor gate_mul(torch::Tensor att, torch::Tensor gate, int64_t T, int64_t HQ, int64_t D);
std::vector<torch::Tensor> linattn_prep2(torch::Tensor proj, torch::Tensor convw, torch::Tensor A_log, torch::Tensor dt_bias, int64_t B, int64_t T, int64_t KD, int64_t VD, int64_t HV, int64_t HK, int64_t HD, bool do_l2);
std::vector<torch::Tensor> fullattn_prep2(torch::Tensor proj, torch::Tensor qw1, torch::Tensor kw1, torch::Tensor cosb, torch::Tensor sinb, int64_t B, int64_t T, int64_t HQ, int64_t HKV, int64_t D, double eps);
torch::Tensor gate_mul2(torch::Tensor att, torch::Tensor gate, int64_t T, int64_t HQ, int64_t D);
std::vector<torch::Tensor> add_rmsnorm_sk(torch::Tensor x, torch::Tensor parts, torch::Tensor w1, c10::optional<torch::Tensor> rowmask, double eps);
std::vector<torch::Tensor> linattn_prep_rep(torch::Tensor proj, torch::Tensor convw, torch::Tensor A_log, torch::Tensor dt_bias, int64_t Lp, int64_t S, int64_t Ls, int64_t KD, int64_t VD, int64_t HV, int64_t HK, int64_t HD);
torch::Tensor gated_rmsnorm_map(torch::Tensor x, torch::Tensor proj, torch::Tensor w, int64_t HV, int64_t HD, int64_t zoff, double eps, int64_t Lp, int64_t Ls);
std::vector<torch::Tensor> linattn_prep_map(torch::Tensor proj, torch::Tensor src, torch::Tensor convw, torch::Tensor A_log, torch::Tensor dt_bias, int64_t KD, int64_t VD, int64_t HV, int64_t HK, int64_t HD);
torch::Tensor gated_rmsnorm_inv(torch::Tensor x, torch::Tensor inv, torch::Tensor proj, torch::Tensor w, int64_t HV, int64_t HD, int64_t zoff, double eps);
std::vector<torch::Tensor> linattn_prep_map2(torch::Tensor proj, torch::Tensor src, torch::Tensor convw, torch::Tensor A_log, torch::Tensor dt_bias, int64_t KD, int64_t VD, int64_t HV, int64_t HK, int64_t HD, int64_t tch);
std::vector<torch::Tensor> linattn_prep_dedup(torch::Tensor proj, torch::Tensor src, torch::Tensor rep_ptr, torch::Tensor rep_pos, torch::Tensor convw, torch::Tensor A_log, torch::Tensor dt_bias, int64_t KD, int64_t VD, int64_t HV, int64_t HK, int64_t HD);
std::vector<torch::Tensor> linattn_prep_dedup2(torch::Tensor proj, torch::Tensor hist, torch::Tensor rep_ptr, torch::Tensor rep_pos, int64_t L2, torch::Tensor convw, torch::Tensor A_log, torch::Tensor dt_bias, int64_t KD, int64_t VD, int64_t HV, int64_t HK, int64_t HD, bool rep_qk);
std::vector<torch::Tensor> act_tables(torch::Tensor like);
torch::Tensor silu_mul_lut(torch::Tensor gu, torch::Tensor tab);
torch::Tensor gated_rmsnorm_inv2(torch::Tensor x, torch::Tensor inv, torch::Tensor proj, torch::Tensor w, torch::Tensor sig, int64_t HV, int64_t HD, int64_t zoff, double eps);
void l2_prefetch(torch::Tensor t, int64_t chunk);
std::vector<torch::Tensor> linattn_prep_dedup3(torch::Tensor proj, torch::Tensor hist, torch::Tensor rep_ptr, torch::Tensor rep_pos, int64_t L2, torch::Tensor convw, torch::Tensor A_log, torch::Tensor dt_bias, int64_t KD, int64_t VD, int64_t HV, int64_t HK, int64_t HD);
torch::Tensor gate_mul3(torch::Tensor att, torch::Tensor gate, torch::Tensor sig, int64_t T, int64_t HQ, int64_t D);
std::vector<torch::Tensor> add_rmsnorm_sk2(torch::Tensor x, torch::Tensor parts, torch::Tensor w1, c10::optional<torch::Tensor> rowmask, double eps);
torch::Tensor gdn_fused(torch::Tensor q, torch::Tensor k, torch::Tensor v, torch::Tensor g, torch::Tensor beta, double scale);
torch::Tensor gdn_fused2(torch::Tensor q, torch::Tensor k, torch::Tensor v, torch::Tensor g, torch::Tensor beta, double scale);
torch::Tensor gdn_fused3(torch::Tensor q, torch::Tensor k, torch::Tensor v, torch::Tensor g, torch::Tensor beta, double scale);
torch::Tensor gdn_fused4(torch::Tensor q, torch::Tensor k, torch::Tensor v, torch::Tensor g, torch::Tensor beta, double scale);
torch::Tensor gdn_fused6(torch::Tensor q, torch::Tensor k, torch::Tensor v, torch::Tensor g, torch::Tensor beta, double scale);
torch::Tensor gdn_fused7(torch::Tensor q, torch::Tensor k, torch::Tensor v, torch::Tensor g, torch::Tensor beta, double scale, torch::Tensor proj, int64_t zoff, torch::Tensor normw, torch::Tensor sig, torch::Tensor canon, torch::Tensor rowmask, double neps);
torch::Tensor gdn_fused6ts(torch::Tensor q, torch::Tensor k, torch::Tensor v, torch::Tensor g, torch::Tensor beta, double scale);
torch::Tensor fattn_tree(torch::Tensor qt, torch::Tensor kt, torch::Tensor vt, torch::Tensor gate, torch::Tensor sig, torch::Tensor vbits, double scale, int64_t gq);
torch::Tensor fattn_tree2(torch::Tensor proj, torch::Tensor qw1, torch::Tensor kw1, torch::Tensor cosb, torch::Tensor sinb, torch::Tensor sig, torch::Tensor vbits, int64_t HQ, int64_t HKV, double scale, double eps);
torch::Tensor fattn_tree2_ts(torch::Tensor proj, torch::Tensor qw1, torch::Tensor kw1, torch::Tensor cosb, torch::Tensor sinb, torch::Tensor sig, torch::Tensor vbits, int64_t HQ, int64_t HKV, double scale, double eps);
torch::Tensor fattn_tree3(torch::Tensor qt, torch::Tensor kt, torch::Tensor vt, torch::Tensor gate, torch::Tensor sig, torch::Tensor vbits, double scale);
torch::Tensor pack_bits(torch::Tensor vis);
torch::Tensor gdn_fused8(torch::Tensor q, torch::Tensor k, torch::Tensor v, torch::Tensor g, torch::Tensor beta, double scale);
torch::Tensor gdn_fused9(torch::Tensor q, torch::Tensor k, torch::Tensor v, torch::Tensor g, torch::Tensor beta, double scale);
torch::Tensor gdn_fused5(torch::Tensor q, torch::Tensor k, torch::Tensor v, torch::Tensor g, torch::Tensor beta, double scale, torch::Tensor proj, int64_t zoff, torch::Tensor normw, torch::Tensor sig, torch::Tensor canon, torch::Tensor rowmask, double neps);
std::vector<torch::Tensor> linattn_prep_tree(torch::Tensor proj, torch::Tensor convw, torch::Tensor A_log, torch::Tensor dt_bias, int64_t Lp, int64_t S, int64_t Ls, int64_t KD, int64_t VD, int64_t HV, int64_t HK, int64_t HD);
"""
def load(verbose=False):
    src = open(os.path.join(_HERE, "kernels.cu")).read()
    name = os.environ.get("OJ_EXT_NAME", "ojfast"); bdir = os.path.join(_HERE, "build_" + name if name != "ojfast" else "build")
    os.makedirs(bdir, exist_ok=True)
    return load_inline(name=name, cpp_sources=[_CPP], cuda_sources=[src],
                       functions=["add_rmsnorm", "silu_mul", "linattn_prep", "gated_rmsnorm", "fullattn_prep", "gate_mul", "linattn_prep2", "fullattn_prep2", "gate_mul2", "linattn_prep_tree", "add_rmsnorm_sk", "linattn_prep_rep", "gated_rmsnorm_map", "linattn_prep_map", "gated_rmsnorm_inv", "linattn_prep_map2", "linattn_prep_dedup", "linattn_prep_dedup2", "act_tables", "silu_mul_lut", "gated_rmsnorm_inv2", "l2_prefetch", "linattn_prep_dedup3", "gate_mul3", "add_rmsnorm_sk2", "gdn_fused", "gdn_fused2", "gdn_fused3", "gdn_fused4", "gdn_fused5", "gdn_fused6", "gdn_fused7", "gdn_fused6ts", "fattn_tree", "fattn_tree2", "fattn_tree2_ts", "fattn_tree3", "pack_bits", "gdn_fused8", "gdn_fused9"],
                       extra_cuda_cflags=["-O3", "-gencode=arch=compute_103,code=sm_103", "--expt-relaxed-constexpr", "-lineinfo"],
                       build_directory=bdir, verbose=verbose)
