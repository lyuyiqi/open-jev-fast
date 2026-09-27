"""Programmatic dependent launch (PDL) inside a CUDA graph: a chain that mimics the MLP half of 16 layers
(copy kernel -> gate_up GEMM -> copy kernel -> down GEMM -> copy kernel) with DeepGEMM or cuBLAS GEMMs, PDL off / on.
The copy kernels call griddepcontrol.wait + launch_dependents when launched with the PDL attribute."""
import os, sys
sys.path.insert(0, os.environ["DEEPGEMM_DIR"])   # a DeepGEMM checkout built in place
import torch, torch.nn.functional as F
from torch.utils.cpp_extension import load_inline
import deep_gemm
SRC = r'''
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_bf16.h>
__global__ void ew_k(const __nv_bfloat16* in, __nv_bfloat16* out, int R, int Cin, int Cout, int pdl) {
  if (pdl) { asm volatile("griddepcontrol.wait;" ::: "memory"); asm volatile("griddepcontrol.launch_dependents;"); }
  const long n = (long)R * Cout;
  for (long i = (blockIdx.x * (long)blockDim.x + threadIdx.x) * 8; i < n; i += (long)gridDim.x * blockDim.x * 8) {
    const int r = i / Cout, c = i % Cout;
    *reinterpret_cast<uint4*>(out + i) = *reinterpret_cast<const uint4*>(in + (long)r * Cin + c);
  }
}
torch::Tensor ew(torch::Tensor in, int64_t Cout, bool pdl) {
  const int R = in.size(0), Cin = in.size(1);
  auto out = torch::empty({R, Cout}, in.options());
  cudaLaunchConfig_t cfg = {}; cfg.gridDim = dim3(296); cfg.blockDim = dim3(256); cfg.stream = at::cuda::getCurrentCUDAStream();
  cudaLaunchAttribute at[1]; at[0].id = cudaLaunchAttributeProgrammaticStreamSerialization; at[0].val.programmaticStreamSerializationAllowed = 1;
  cfg.attrs = at; cfg.numAttrs = pdl ? 1 : 0;
  TORCH_CHECK(cudaLaunchKernelEx(&cfg, ew_k, (const __nv_bfloat16*)in.data_ptr(), (__nv_bfloat16*)out.data_ptr(), R, Cin, (int)Cout, (int)pdl) == cudaSuccess);
  return out;
}
'''
E = load_inline("pdl_bench_ext", cpp_sources="torch::Tensor ew(torch::Tensor in, int64_t Cout, bool pdl);", cuda_sources=SRC, functions=["ew"],
                extra_cuda_cflags=["-O3", "-gencode=arch=compute_103,code=sm_103"],
                build_directory=os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src", "build_pdl"))
M, H, I, NL = int(os.environ.get("M", "288")), 5120, 17408, 16
torch.manual_seed(0)
wgu = [torch.randn(2 * I, H, device="cuda", dtype=torch.bfloat16) * 0.02 for _ in range(NL)]
wd = [torch.randn(H, I, device="cuda", dtype=torch.bfloat16) * 0.02 for _ in range(NL)]
x0 = torch.randn(M, H, device="cuda", dtype=torch.bfloat16)
gu = torch.empty(M, 2 * I, device="cuda", dtype=torch.bfloat16); dn = torch.empty(M, H, device="cuda", dtype=torch.bfloat16)

def chain(gemm, pdl):
    x = x0
    for l in range(NL):
        a = E.ew(x, H, pdl)
        b = gemm(a, wgu[l], gu)
        c = E.ew(b, I, pdl)
        d = gemm(c, wd[l], dn)
        x = E.ew(d, H, pdl)
    return x

def dg(a, w, out): deep_gemm.bf16_gemm_nt(a, w, out); return out
def cb(a, w, out): return F.linear(a, w)

def replay_ms(fn, reps=30):
    st = torch.cuda.Stream(); st.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(st):
        for _ in range(2): fn()
    torch.cuda.current_stream().wait_stream(st); torch.cuda.synchronize()
    gr = torch.cuda.CUDAGraph()
    with torch.cuda.graph(gr): fn()
    for _ in range(3): gr.replay()
    ts = []
    for _ in range(reps):
        a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        a.record(); gr.replay(); b.record(); torch.cuda.synchronize(); ts.append(a.elapsed_time(b))
    ts.sort(); return ts[len(ts) // 2]

ref = chain(cb, False).float()
for name, gemm in (("cublas", cb), ("deepgemm", dg)):
    for pdl in (False, True):
        if name == "deepgemm": deep_gemm.set_pdl(pdl)
        out = chain(gemm, pdl).float()
        err = ((out - ref).abs().max() / ref.abs().max()).item()
        t = replay_ms(lambda: chain(gemm, pdl))
        print(f"PDL {name:8s} pdl={int(pdl)}: {t:.3f} ms per 16 layers ({t / NL * 1e3:.1f} us/layer, 5 kernels/layer)  rel err vs cuBLAS {err:.1e}", flush=True)
print("=== PDL BENCH DONE ===", flush=True)
