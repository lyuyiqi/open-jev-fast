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

// ---------------------------------------------------------------------------------------------------------------------
// Round 4: exhaustive cuBLASLt search. A plan is (M, N, K, S, fp32 out, weight layout):
//   wl = 0: w [N, K] (K-major), S must be 1;  wl = 1: w [S, K/S, N] (split-K weights, N-major), x split along K.
// out = [S, M, N] (fp32 if f32 else bf16). Candidates: best-fit heuristic, per-algo-id heuristic, and an explicit sweep
// of tile x stages x cluster x split-K x reduction x swizzle x custom option (each checked with cublasLtMatmulAlgoCheck).
struct Plan2 { cublasLtMatmulDesc_t op; cublasLtMatrixLayout_t a, b, c; std::vector<cublasLtMatmulAlgo_t> algos; long M, N, K; int S, f32, wl; };
static std::vector<Plan2> g_p2;
int64_t lt2_plan(int64_t M, int64_t N, int64_t K, int64_t S, int64_t f32, int64_t wl) {
  if (!g_lt) LTCHECK(cublasLtCreate(&g_lt));
  if (!g_ws.defined()) g_ws = torch::empty({(long)WS}, torch::dtype(torch::kUInt8).device(torch::kCUDA));
  TORCH_CHECK(wl == 1 || S == 1);
  Plan2 p; p.M = M; p.N = N; p.K = K; p.S = (int)S; p.f32 = (int)f32; p.wl = (int)wl;
  const long Ks = K / S;
  LTCHECK(cublasLtMatmulDescCreate(&p.op, CUBLAS_COMPUTE_32F, CUDA_R_32F));
  cublasOperation_t ta = wl == 0 ? CUBLAS_OP_T : CUBLAS_OP_N, tb = CUBLAS_OP_N;
  LTCHECK(cublasLtMatmulDescSetAttribute(p.op, CUBLASLT_MATMUL_DESC_TRANSA, &ta, sizeof(ta)));
  LTCHECK(cublasLtMatmulDescSetAttribute(p.op, CUBLASLT_MATMUL_DESC_TRANSB, &tb, sizeof(tb)));
  if (wl == 0) LTCHECK(cublasLtMatrixLayoutCreate(&p.a, CUDA_R_16BF, K, N, K));
  else LTCHECK(cublasLtMatrixLayoutCreate(&p.a, CUDA_R_16BF, N, Ks, N));
  LTCHECK(cublasLtMatrixLayoutCreate(&p.b, CUDA_R_16BF, Ks, M, K));
  LTCHECK(cublasLtMatrixLayoutCreate(&p.c, f32 ? CUDA_R_32F : CUDA_R_16BF, N, M, N));
  if (S > 1) {
    int32_t bc = (int32_t)S; int64_t sa = Ks * N, sb = Ks, sc = M * N;
    for (auto [l, st] : {std::make_pair(p.a, sa), std::make_pair(p.b, sb), std::make_pair(p.c, sc)}) {
      LTCHECK(cublasLtMatrixLayoutSetAttribute(l, CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT, &bc, sizeof(bc)));
      LTCHECK(cublasLtMatrixLayoutSetAttribute(l, CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET, &st, sizeof(st)));
    }
  }
  g_p2.push_back(p);
  return (int64_t)g_p2.size() - 1;
}
static bool same_algo(const cublasLtMatmulAlgo_t& x, const cublasLtMatmulAlgo_t& y) { return memcmp(&x, &y, sizeof(x)) == 0; }
static void push_unique(Plan2& p, const cublasLtMatmulAlgo_t& a) {
  for (auto& b : p.algos) if (same_algo(a, b)) return;
  p.algos.push_back(a);
}
int64_t lt2_search(int64_t h, int64_t max_total, int64_t modes) {
  Plan2& p = g_p2.at(h);
  const cudaDataType_t ct = p.f32 ? CUDA_R_32F : CUDA_R_16BF;
  size_t ws = WS;
  if (modes & 1) {   // best-fit heuristic
    cublasLtMatmulPreference_t pref; LTCHECK(cublasLtMatmulPreferenceCreate(&pref));
    LTCHECK(cublasLtMatmulPreferenceSetAttribute(pref, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &ws, sizeof(ws)));
    std::vector<cublasLtMatmulHeuristicResult_t> res(256); int n = 0;
    if (cublasLtMatmulAlgoGetHeuristic(g_lt, p.op, p.a, p.b, p.c, p.c, pref, 256, res.data(), &n) == CUBLAS_STATUS_SUCCESS)
      for (int i = 0; i < n; i++) if (res[i].state == CUBLAS_STATUS_SUCCESS) push_unique(p, res[i].algo);
    cublasLtMatmulPreferenceDestroy(pref);
  }
  std::vector<int> ids(4096); int nid = 0;
  LTCHECK(cublasLtMatmulAlgoGetIds(g_lt, CUBLAS_COMPUTE_32F, CUDA_R_32F, CUDA_R_16BF, CUDA_R_16BF, ct, ct, 4096, ids.data(), &nid));
  if (modes & 2) {   // heuristic limited to each algo id
    for (int ii = 0; ii < nid && (long)p.algos.size() < max_total; ii++) {
      cublasLtMatmulPreference_t pref; LTCHECK(cublasLtMatmulPreferenceCreate(&pref));
      LTCHECK(cublasLtMatmulPreferenceSetAttribute(pref, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &ws, sizeof(ws)));
      uint32_t sm = CUBLASLT_SEARCH_LIMITED_BY_ALGO_ID;
      if (cublasLtMatmulPreferenceSetAttribute(pref, CUBLASLT_MATMUL_PREF_SEARCH_MODE, &sm, sizeof(sm)) != CUBLAS_STATUS_SUCCESS) { cublasLtMatmulPreferenceDestroy(pref); break; }
      std::vector<cublasLtMatmulHeuristicResult_t> res(64); int n = 0;
      for (auto& r : res) cublasLtMatmulAlgoInit(g_lt, CUBLAS_COMPUTE_32F, CUDA_R_32F, CUDA_R_16BF, CUDA_R_16BF, ct, ct, ids[ii], &r.algo);
      if (cublasLtMatmulAlgoGetHeuristic(g_lt, p.op, p.a, p.b, p.c, p.c, pref, 64, res.data(), &n) == CUBLAS_STATUS_SUCCESS)
        for (int i = 0; i < n; i++) if (res[i].state == CUBLAS_STATUS_SUCCESS) push_unique(p, res[i].algo);
      cublasLtMatmulPreferenceDestroy(pref);
    }
  }
  if (modes & 4) {   // explicit configuration sweep
    const int clusters[] = {CUBLASLT_CLUSTER_SHAPE_AUTO, CUBLASLT_CLUSTER_SHAPE_1x1x1, CUBLASLT_CLUSTER_SHAPE_2x1x1, CUBLASLT_CLUSTER_SHAPE_1x2x1};
    const int splits[] = {1, 2, 3, 4, 6, 8};
    for (int ii = 0; ii < nid && (long)p.algos.size() < max_total; ii++) {
      cublasLtMatmulAlgo_t algo;
      if (cublasLtMatmulAlgoInit(g_lt, CUBLAS_COMPUTE_32F, CUDA_R_32F, CUDA_R_16BF, CUDA_R_16BF, ct, ct, ids[ii], &algo) != CUBLAS_STATUS_SUCCESS) continue;
      size_t sz = 0;
      cublasLtMatmulAlgoCapGetAttribute(&algo, CUBLASLT_ALGO_CAP_TILE_IDS, nullptr, 0, &sz);
      std::vector<uint32_t> tiles(std::max<size_t>(1, sz / 4), CUBLASLT_MATMUL_TILE_UNDEFINED);
      if (sz) cublasLtMatmulAlgoCapGetAttribute(&algo, CUBLASLT_ALGO_CAP_TILE_IDS, tiles.data(), sz, &sz);
      sz = 0; cublasLtMatmulAlgoCapGetAttribute(&algo, CUBLASLT_ALGO_CAP_STAGES_IDS, nullptr, 0, &sz);
      std::vector<uint32_t> stages(std::max<size_t>(1, sz / 4), CUBLASLT_MATMUL_STAGES_UNDEFINED);
      if (sz) cublasLtMatmulAlgoCapGetAttribute(&algo, CUBLASLT_ALGO_CAP_STAGES_IDS, stages.data(), sz, &sz);
      int32_t splitk_ok = 0, swz = 0, cmax = 0; uint32_t redmask = 0;
      cublasLtMatmulAlgoCapGetAttribute(&algo, CUBLASLT_ALGO_CAP_SPLITK_SUPPORT, &splitk_ok, sizeof(splitk_ok), &sz);
      cublasLtMatmulAlgoCapGetAttribute(&algo, CUBLASLT_ALGO_CAP_REDUCTION_SCHEME_MASK, &redmask, sizeof(redmask), &sz);
      cublasLtMatmulAlgoCapGetAttribute(&algo, CUBLASLT_ALGO_CAP_CTA_SWIZZLING_SUPPORT, &swz, sizeof(swz), &sz);
      cublasLtMatmulAlgoCapGetAttribute(&algo, CUBLASLT_ALGO_CAP_CUSTOM_OPTION_MAX, &cmax, sizeof(cmax), &sz);
      for (uint32_t tile : tiles) for (uint32_t stg : stages) for (int cl : clusters) for (int sk : splits) {
        if (sk > 1 && !splitk_ok) continue;
        std::vector<uint32_t> reds = {CUBLASLT_REDUCTION_SCHEME_NONE};
        if (sk > 1) { reds.clear(); for (uint32_t r = 1; r <= 4; r <<= 1) if (redmask & r) reds.push_back(r); }
        for (uint32_t red : reds) for (uint32_t sw = 0; sw <= (uint32_t)(swz ? 1 : 0); sw++) for (uint32_t co = 0; co <= (uint32_t)std::min(cmax, 3); co++) {
          if ((long)p.algos.size() >= max_total) break;
          cublasLtMatmulAlgo_t a = algo;
          cublasLtMatmulAlgoConfigSetAttribute(&a, CUBLASLT_ALGO_CONFIG_TILE_ID, &tile, sizeof(tile));
          cublasLtMatmulAlgoConfigSetAttribute(&a, CUBLASLT_ALGO_CONFIG_STAGES_ID, &stg, sizeof(stg));
          uint32_t clu = (uint32_t)cl; cublasLtMatmulAlgoConfigSetAttribute(&a, CUBLASLT_ALGO_CONFIG_CLUSTER_SHAPE_ID, &clu, sizeof(clu));
          int32_t skn = sk; cublasLtMatmulAlgoConfigSetAttribute(&a, CUBLASLT_ALGO_CONFIG_SPLITK_NUM, &skn, sizeof(skn));
          cublasLtMatmulAlgoConfigSetAttribute(&a, CUBLASLT_ALGO_CONFIG_REDUCTION_SCHEME, &red, sizeof(red));
          cublasLtMatmulAlgoConfigSetAttribute(&a, CUBLASLT_ALGO_CONFIG_CTA_SWIZZLING, &sw, sizeof(sw));
          cublasLtMatmulAlgoConfigSetAttribute(&a, CUBLASLT_ALGO_CONFIG_CUSTOM_OPTION, &co, sizeof(co));
          cublasLtMatmulHeuristicResult_t r;
          if (cublasLtMatmulAlgoCheck(g_lt, p.op, p.a, p.b, p.c, p.c, &a, &r) == CUBLAS_STATUS_SUCCESS && r.workspaceSize <= WS) push_unique(p, a);
        }
      }
    }
  }
  return (int64_t)p.algos.size();
}
torch::Tensor lt2_run(int64_t h, torch::Tensor x, torch::Tensor w, int64_t idx) {
  Plan2& p = g_p2.at(h);
  TORCH_CHECK(x.is_contiguous() && w.is_contiguous() && x.size(0) == p.M && x.size(1) == p.K && w.numel() == p.N * p.K);
  TORCH_CHECK(idx >= 0 && idx < (long)p.algos.size());
  auto out = torch::empty(p.S > 1 ? std::vector<int64_t>{p.S, p.M, p.N} : std::vector<int64_t>{p.M, p.N}, x.options().dtype(p.f32 ? torch::kFloat32 : torch::kBFloat16));
  float alpha = 1.f, beta = 0.f;
  LTCHECK(cublasLtMatmul(g_lt, p.op, &alpha, w.data_ptr(), p.a, x.data_ptr(), p.b, &beta, out.data_ptr(), p.c, out.data_ptr(), p.c,
                         &p.algos[idx], g_ws.data_ptr(), WS, at::cuda::getCurrentCUDAStream()));
  return out;
}
std::vector<int64_t> lt2_info(int64_t h, int64_t idx) {   // algo id, tile, stages, split-K, reduction, swizzle, custom, cluster
  const cublasLtMatmulAlgo_t& a = g_p2.at(h).algos.at(idx);
  const cublasLtMatmulAlgoConfigAttributes_t at[] = {CUBLASLT_ALGO_CONFIG_ID, CUBLASLT_ALGO_CONFIG_TILE_ID, CUBLASLT_ALGO_CONFIG_STAGES_ID,
      CUBLASLT_ALGO_CONFIG_SPLITK_NUM, CUBLASLT_ALGO_CONFIG_REDUCTION_SCHEME, CUBLASLT_ALGO_CONFIG_CTA_SWIZZLING,
      CUBLASLT_ALGO_CONFIG_CUSTOM_OPTION, CUBLASLT_ALGO_CONFIG_CLUSTER_SHAPE_ID};
  std::vector<int64_t> out;
  for (auto x : at) { uint64_t v = 0; size_t sz = 0; cublasLtMatmulAlgoConfigGetAttribute(&a, x, &v, sizeof(v), &sz); out.push_back(sz == 4 ? (int64_t)(uint32_t)v : (int64_t)v); }
  return out;
}
std::vector<int64_t> lt2_raw(int64_t h, int64_t idx) {
  const cublasLtMatmulAlgo_t& a = g_p2.at(h).algos.at(idx);
  return std::vector<int64_t>(reinterpret_cast<const int64_t*>(a.data), reinterpret_cast<const int64_t*>(a.data) + 8);
}
int64_t lt2_add_raw(int64_t h, std::vector<int64_t> raw) {   // re-create a saved configuration (same cuBLASLt version)
  TORCH_CHECK(raw.size() == 8);
  Plan2& p = g_p2.at(h); cublasLtMatmulAlgo_t a; memcpy(a.data, raw.data(), sizeof(a.data));
  cublasLtMatmulHeuristicResult_t r;
  LTCHECK(cublasLtMatmulAlgoCheck(g_lt, p.op, p.a, p.b, p.c, p.c, &a, &r));
  p.algos.push_back(a);
  return (int64_t)p.algos.size() - 1;
}
