"""Micro-benchmark of the per-row q/k prep (RMSNorm + partial RoPE on a 256-dim bf16 row in shared memory, one warp per
row) as used inside fattn2_k: time per row, and which part costs."""
import os, torch
from torch.utils.cpp_extension import load_inline
SRC = r'''
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_bf16.h>
typedef __nv_bfloat16 bf16;
__device__ __forceinline__ float b2f(bf16 x) { return __bfloat162float(x); }
__device__ __forceinline__ bf16 f2b(float x) { return __float2bfloat16(x); }
__device__ __forceinline__ float rbf(float x) { return b2f(f2b(x)); }
template <int MODE>
__device__ __forceinline__ void prep_row(bf16* row, const float* w1, const bf16* cosr, const bf16* sinr, int RD, float eps) {
  const int lane = threadIdx.x & 31, d0 = lane * 8;
  uint4 a = *reinterpret_cast<const uint4*>(row + d0);
  const bf16* ab = reinterpret_cast<const bf16*>(&a);
  float xv[8], ss = 0.f;
  #pragma unroll
  for (int j = 0; j < 8; j++) { xv[j] = b2f(ab[j]); ss += xv[j] * xv[j]; }
  if (MODE != 1) {
    #pragma unroll
    for (int o = 16; o > 0; o >>= 1) ss += __shfl_xor_sync(0xffffffff, ss, o);
  }
  const float r = rsqrtf(ss / 256.f + eps);
  float nv[8];
  #pragma unroll
  for (int j = 0; j < 8; j++) nv[j] = rbf((xv[j] * r) * w1[d0 + j]);
  uint4 ov; bf16* ob = reinterpret_cast<bf16*>(&ov);
  if (MODE != 2) {
    const int hl = RD / 16;
    float pv[8];
    #pragma unroll
    for (int j = 0; j < 8; j++) pv[j] = __shfl_xor_sync(0xffffffff, nv[j], hl);
    #pragma unroll
    for (int j = 0; j < 8; j++) {
      const int dd = d0 + j;
      float outv = nv[j];
      if (dd < RD) { const float rh = (dd < RD / 2) ? -pv[j] : pv[j]; const float c = b2f(cosr[dd]), s = b2f(sinr[dd]); outv = rbf(rbf(nv[j] * c) + rbf(rh * s)); }
      ob[j] = f2b(outv);
    }
  } else {
    #pragma unroll
    for (int j = 0; j < 8; j++) ob[j] = f2b(nv[j]);
  }
  *reinterpret_cast<uint4*>(row + d0) = ov;
}
template <int MODE>
__global__ void k(const bf16* src, bf16* dst, const float* w1g, int rows, int RD, long long* ts) {
  __shared__ __align__(16) bf16 sR[48][264];
  __shared__ float w1[256];
  __shared__ __align__(16) bf16 cs[48][2][64];
  const int tid = threadIdx.x, w = tid >> 5;
  for (int i = tid; i < 48 * 256; i += blockDim.x) sR[i / 256][i % 256] = src[(long)blockIdx.x * 48 * 256 + i];
  for (int i = tid; i < 256; i += blockDim.x) w1[i] = w1g[i];
  for (int i = tid; i < 48 * 128; i += blockDim.x) cs[i / 128][(i / 64) & 1][i % 64] = src[i];
  __syncthreads();
  long long t0; asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t0));
  for (int rr = w; rr < rows; rr += blockDim.x / 32) prep_row<MODE>(sR[rr], w1, cs[rr][0], cs[rr][1], RD, 1e-6f);
  long long t1; asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t1));
  __syncthreads();
  if ((tid & 31) == 0) ts[blockIdx.x * 8 + w] = t1 - t0;
  for (int i = tid; i < 48 * 256; i += blockDim.x) dst[(long)blockIdx.x * 48 * 256 + i] = sR[i / 256][i % 256];
}
torch::Tensor run(torch::Tensor src, torch::Tensor w1, int64_t mode, int64_t rows, int64_t nthreads) {
  auto dst = torch::empty_like(src);
  auto ts = torch::zeros({144, 8}, src.options().dtype(torch::kInt64));
  auto st = at::cuda::getCurrentCUDAStream();
  if (mode == 0) k<0><<<144, nthreads, 0, st>>>((bf16*)src.data_ptr(), (bf16*)dst.data_ptr(), w1.data_ptr<float>(), rows, 64, (long long*)ts.data_ptr<int64_t>());
  if (mode == 1) k<1><<<144, nthreads, 0, st>>>((bf16*)src.data_ptr(), (bf16*)dst.data_ptr(), w1.data_ptr<float>(), rows, 64, (long long*)ts.data_ptr<int64_t>());
  if (mode == 2) k<2><<<144, nthreads, 0, st>>>((bf16*)src.data_ptr(), (bf16*)dst.data_ptr(), w1.data_ptr<float>(), rows, 64, (long long*)ts.data_ptr<int64_t>());
  return ts;
}
'''
E = load_inline("prep_micro", cpp_sources="torch::Tensor run(torch::Tensor src, torch::Tensor w1, int64_t mode, int64_t rows, int64_t nthreads);",
                cuda_sources=SRC, functions=["run"], extra_cuda_cflags=["-O3", "-gencode=arch=compute_103,code=sm_103"],
                build_directory=os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src", "build_prep"))
src = torch.randn(144 * 48 * 256, device="cuda").to(torch.bfloat16); w1 = torch.rand(256, device="cuda") + 0.5
for mode, name in ((0, "full"), (1, "no norm reduce"), (2, "no rope")):
    for nth in (192, 64):
        for _ in range(3): ts = E.run(src, w1, mode, 48, nth)
        torch.cuda.synchronize()
        t = ts[:, : nth // 32].float()
        print(f"PREP {name:15s} threads {nth}: per warp {t.mean().item() / 1e3:.2f} us for {48 // (nth // 32)} rows -> {t.mean().item() / (48 // (nth // 32)):.0f} ns/row", flush=True)
