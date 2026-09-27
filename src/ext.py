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
std::vector<torch::Tensor> linattn_prep_tree(torch::Tensor proj, torch::Tensor convw, torch::Tensor A_log, torch::Tensor dt_bias, int64_t Lp, int64_t S, int64_t Ls, int64_t KD, int64_t VD, int64_t HV, int64_t HK, int64_t HD);
"""
def load(verbose=False):
    src = open(os.path.join(_HERE, "kernels.cu")).read()
    name = os.environ.get("OJ_EXT_NAME", "ojfast"); bdir = os.path.join(_HERE, "build_" + name if name != "ojfast" else "build")
    os.makedirs(bdir, exist_ok=True)
    return load_inline(name=name, cpp_sources=[_CPP], cuda_sources=[src],
                       functions=["add_rmsnorm", "silu_mul", "linattn_prep", "gated_rmsnorm", "fullattn_prep", "gate_mul", "linattn_prep2", "fullattn_prep2", "gate_mul2", "linattn_prep_tree", "add_rmsnorm_sk", "linattn_prep_rep", "gated_rmsnorm_map", "linattn_prep_map", "gated_rmsnorm_inv"],
                       extra_cuda_cflags=["-O3", "-gencode=arch=compute_103,code=sm_103", "--expt-relaxed-constexpr"],
                       build_directory=bdir, verbose=verbose)
