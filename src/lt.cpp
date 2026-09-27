// cuBLASLt autotuned GEMM for fixed shapes: out[M,N] = x[M,K] @ w[N,K]^T (bf16 in/out, fp32 accumulate).
// lt_setup enumerates heuristic algorithms for (M,N,K); lt_matmul runs a chosen algorithm (graph-capturable).
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cublasLt.h>
#include <map>
#include <tuple>
#include <vector>
#define LTCHECK(x) do { cublasStatus_t s_ = (x); TORCH_CHECK(s_ == CUBLAS_STATUS_SUCCESS, "cublasLt error ", (int)s_, " at ", __LINE__); } while (0)
struct Plan { cublasLtMatmulDesc_t op; cublasLtMatrixLayout_t a, b, c; std::vector<cublasLtMatmulHeuristicResult_t> algos; };
static cublasLtHandle_t g_lt = nullptr;
static std::map<std::tuple<long, long, long>, Plan> g_plans;
static torch::Tensor g_ws;
static const size_t WS = 64ull << 20;
int64_t lt_setup(int64_t M, int64_t N, int64_t K, int64_t max_algos) {
  if (!g_lt) LTCHECK(cublasLtCreate(&g_lt));
  if (!g_ws.defined()) g_ws = torch::empty({(long)WS}, torch::dtype(torch::kUInt8).device(torch::kCUDA));
  auto key = std::make_tuple(M, N, K);
  auto it = g_plans.find(key);
  if (it != g_plans.end()) return it->second.algos.size();
  Plan p;
  LTCHECK(cublasLtMatmulDescCreate(&p.op, CUBLAS_COMPUTE_32F, CUDA_R_32F));
  cublasOperation_t ta = CUBLAS_OP_T, tb = CUBLAS_OP_N;
  LTCHECK(cublasLtMatmulDescSetAttribute(p.op, CUBLASLT_MATMUL_DESC_TRANSA, &ta, sizeof(ta)));
  LTCHECK(cublasLtMatmulDescSetAttribute(p.op, CUBLASLT_MATMUL_DESC_TRANSB, &tb, sizeof(tb)));
  // column-major view: C[N,M] = W^T (W col-major [K,N], transposed) * X (col-major [K,M])
  LTCHECK(cublasLtMatrixLayoutCreate(&p.a, CUDA_R_16BF, K, N, K));
  LTCHECK(cublasLtMatrixLayoutCreate(&p.b, CUDA_R_16BF, K, M, K));
  LTCHECK(cublasLtMatrixLayoutCreate(&p.c, CUDA_R_16BF, N, M, N));
  cublasLtMatmulPreference_t pref;
  LTCHECK(cublasLtMatmulPreferenceCreate(&pref));
  size_t ws = WS;
  LTCHECK(cublasLtMatmulPreferenceSetAttribute(pref, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &ws, sizeof(ws)));
  std::vector<cublasLtMatmulHeuristicResult_t> res(max_algos);
  int n = 0;
  LTCHECK(cublasLtMatmulAlgoGetHeuristic(g_lt, p.op, p.a, p.b, p.c, p.c, pref, (int)max_algos, res.data(), &n));
  cublasLtMatmulPreferenceDestroy(pref);
  res.resize(n);
  p.algos = res;
  g_plans[key] = p;
  return n;
}
torch::Tensor lt_matmul(torch::Tensor x, torch::Tensor w, int64_t idx) {
  TORCH_CHECK(x.is_contiguous() && w.is_contiguous() && x.scalar_type() == torch::kBFloat16);
  const long M = x.size(0), K = x.size(1), N = w.size(0);
  auto it = g_plans.find(std::make_tuple(M, N, K));
  TORCH_CHECK(it != g_plans.end(), "lt_setup not called for this shape");
  Plan& p = it->second;
  TORCH_CHECK(idx >= 0 && idx < (long)p.algos.size());
  auto out = torch::empty({M, N}, x.options());
  float alpha = 1.f, beta = 0.f;
  LTCHECK(cublasLtMatmul(g_lt, p.op, &alpha, w.data_ptr(), p.a, x.data_ptr(), p.b, &beta, out.data_ptr(), p.c, out.data_ptr(), p.c,
                         &p.algos[idx].algo, g_ws.data_ptr(), WS, at::cuda::getCurrentCUDAStream()));
  return out;
}
