// Fused non-GEMM kernels for Qwen3.5/3.8 hybrid decoder (Open-Jev-27B inference).
// Every kernel reproduces the bf16 rounding points of the PyTorch reference so outputs match
// the HF implementation up to reduction order.
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_bf16.h>
#include <type_traits>

typedef __nv_bfloat16 bf16;
#define CHECK(x) TORCH_CHECK((x).is_cuda() && (x).is_contiguous(), #x " must be contiguous CUDA")

__device__ __forceinline__ float b2f(bf16 x) { return __bfloat162float(x); }
__device__ __forceinline__ bf16 f2b(float x) { return __float2bfloat16(x); }
__device__ __forceinline__ float rbf(float x) { return b2f(f2b(x)); }            // round through bf16
__device__ __forceinline__ float silu_f(float x) { return x / (1.0f + expf(-x)); }
__device__ __forceinline__ float sigmoid_f(float x) { return 1.0f / (1.0f + expf(-x)); }

// block-wide sum; `red` must hold >= 32 floats of shared memory
__device__ __forceinline__ float block_sum(float v, float* red) {
  #pragma unroll
  for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffff, v, o);
  const int lane = threadIdx.x & 31, wid = threadIdx.x >> 5, nw = (blockDim.x + 31) >> 5;
  if (lane == 0) red[wid] = v;
  __syncthreads();
  float t = (threadIdx.x < nw) ? red[threadIdx.x] : 0.f;
  if (wid == 0) {
    #pragma unroll
    for (int o = 16; o > 0; o >>= 1) t += __shfl_xor_sync(0xffffffff, t, o);
    if (lane == 0) red[0] = t;
  }
  __syncthreads();
  float r = red[0];
  __syncthreads();
  return r;
}

// ---------------------------------------------------------------------------------------------
// K1: h = bf16(x + delta) ; out = bf16( h * rsqrt(mean(h^2)+eps) * w1 ) [* rowmask]
//     (Qwen3_5RMSNorm with w1 = 1 + weight, fp32 math). delta may be absent.
// one block per row, 8 elements per thread (H % 8 == 0, H/8 <= 1024)
__global__ void add_rmsnorm_k(const bf16* __restrict__ x, const bf16* __restrict__ delta, const float* __restrict__ w1,
                              const int* __restrict__ rowmask, bf16* __restrict__ h_out, bf16* __restrict__ n_out,
                              int H, float eps) {
  __shared__ float red[32];
  const size_t row = blockIdx.x;
  const int i = threadIdx.x * 8;
  float v[8];
  uint4 xa = *reinterpret_cast<const uint4*>(x + row * H + i);
  const bf16* xb = reinterpret_cast<const bf16*>(&xa);
  if (delta) {
    uint4 da = *reinterpret_cast<const uint4*>(delta + row * H + i);
    const bf16* db = reinterpret_cast<const bf16*>(&da);
    #pragma unroll
    for (int j = 0; j < 8; j++) v[j] = rbf(b2f(xb[j]) + b2f(db[j]));
    uint4 ho; bf16* hb = reinterpret_cast<bf16*>(&ho);
    #pragma unroll
    for (int j = 0; j < 8; j++) hb[j] = f2b(v[j]);
    *reinterpret_cast<uint4*>(h_out + row * H + i) = ho;
  } else {
    #pragma unroll
    for (int j = 0; j < 8; j++) v[j] = b2f(xb[j]);
  }
  float ss = 0.f;
  #pragma unroll
  for (int j = 0; j < 8; j++) ss += v[j] * v[j];
  ss = block_sum(ss, red);
  const float r = rsqrtf(ss / (float)H + eps);
  const bool keep = rowmask ? (rowmask[row] != 0) : true;
  uint4 no; bf16* nb = reinterpret_cast<bf16*>(&no);
  #pragma unroll
  for (int j = 0; j < 8; j++) nb[j] = keep ? f2b((v[j] * r) * w1[i + j]) : f2b(0.f);
  *reinterpret_cast<uint4*>(n_out + row * H + i) = no;
}

// K2: gu = [gate | up] (row stride 2I) -> out = bf16( bf16(silu(gate)) * up )
__global__ void silu_mul_k(const bf16* __restrict__ gu, bf16* __restrict__ out, int I, long total8) {
  long idx = (long)blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= total8) return;
  const long e = idx * 8;
  const long row = e / I, col = e % I;
  uint4 ga = *reinterpret_cast<const uint4*>(gu + row * 2 * I + col);
  uint4 ua = *reinterpret_cast<const uint4*>(gu + row * 2 * I + I + col);
  const bf16* gb = reinterpret_cast<const bf16*>(&ga);
  const bf16* ub = reinterpret_cast<const bf16*>(&ua);
  uint4 o; bf16* ob = reinterpret_cast<bf16*>(&o);
  #pragma unroll
  for (int j = 0; j < 8; j++) ob[j] = f2b(rbf(silu_f(b2f(gb[j]))) * b2f(ub[j]));
  *reinterpret_cast<uint4*>(out + e) = o;
}

// K3: linear-attention preprocessing.
// proj row layout: [ qkv (C=2*KD+VD) | z (VD) | b (HV) | a (HV) ]
// conv: depthwise causal, width 4, no bias, then silu (causal_conv1d numerics: fp32 accumulate, bf16 out)
// q,k (HK heads) are expanded to HV heads (repeat_interleave(HV/HK)); v copied.
// beta = bf16(sigmoid(b)); g = -exp(A_log) * softplus(a + dt_bias)  (fp32)
__global__ void linattn_prep_k(const bf16* __restrict__ proj, const bf16* __restrict__ convw,
                               const bf16* __restrict__ A_log, const bf16* __restrict__ dt_bias,
                               bf16* __restrict__ q, bf16* __restrict__ k, bf16* __restrict__ v,
                               float* __restrict__ g, bf16* __restrict__ beta,
                               int T, int KD, int VD, int HV, int HK, int HD, int W) {
  const long n = blockIdx.y;              // token row (b*T + t)
  const int t = n % T;
  const int C = 2 * KD + VD;
  const int P = C + VD + 2 * HV;          // proj row width
  const int c0 = (blockIdx.x * blockDim.x + threadIdx.x) * 8;
  if (blockIdx.x == 0 && threadIdx.x < HV) {
    const int h = threadIdx.x;
    const float bv = b2f(proj[n * P + C + VD + h]);
    const float av = b2f(proj[n * P + C + VD + HV + h]);
    beta[n * HV + h] = f2b(sigmoid_f(bv));
    const float xs = av + b2f(dt_bias[h]);
    const float sp = xs > 20.f ? xs : log1pf(expf(xs));
    g[n * HV + h] = (-expf(b2f(A_log[h]))) * sp;
  }
  if (c0 >= C) return;
  float acc[8];
  #pragma unroll
  for (int j = 0; j < 8; j++) acc[j] = 0.f;
  for (int kk = 0; kk < W; kk++) {
    const int tt = t - (W - 1) + kk;
    if (tt < 0) continue;
    uint4 xa = *reinterpret_cast<const uint4*>(proj + (n - (W - 1) + kk) * P + c0);
    const bf16* xb = reinterpret_cast<const bf16*>(&xa);
    #pragma unroll
    for (int j = 0; j < 8; j++) acc[j] += b2f(convw[(c0 + j) * W + kk]) * b2f(xb[j]);
  }
  uint4 o; bf16* ob = reinterpret_cast<bf16*>(&o);
  #pragma unroll
  for (int j = 0; j < 8; j++) ob[j] = f2b(silu_f(acc[j]));
  const int rep = HV / HK;
  if (c0 < 2 * KD) {                       // q or k
    bf16* dst = (c0 < KD) ? q : k;
    const int cc = (c0 < KD) ? c0 : c0 - KD;
    const int hk = cc / HD, d = cc % HD;
    for (int r = 0; r < rep; r++)
      *reinterpret_cast<uint4*>(dst + n * (long)(HV * HD) + (long)(hk * rep + r) * HD + d) = o;
  } else {
    *reinterpret_cast<uint4*>(v + n * (long)VD + (c0 - 2 * KD)) = o;
  }
}

// K4: gated RMSNorm (FLA FusedRMSNormGated, swish): y = x*rstd*w * z*sigmoid(z); one warp per (row, head)
__global__ void gated_rmsnorm_k(const bf16* __restrict__ x, const bf16* __restrict__ proj, const bf16* __restrict__ w,
                                bf16* __restrict__ out, long rows, int HV, int HD, int zoff, int P, float eps) {
  const long r = (long)blockIdx.x * (blockDim.x / 32) + (threadIdx.x / 32);
  if (r >= rows) return;
  const int lane = threadIdx.x & 31;
  const long n = r / HV; const int h = r % HV;
  const int per = HD / 32;                 // 4 for HD=128
  float xv[8], ss = 0.f;
  for (int j = 0; j < per; j++) { xv[j] = b2f(x[r * HD + lane * per + j]); ss += xv[j] * xv[j]; }
  #pragma unroll
  for (int o = 16; o > 0; o >>= 1) ss += __shfl_xor_sync(0xffffffff, ss, o);
  const float rstd = 1.0f / sqrtf(ss / (float)HD + eps);
  for (int j = 0; j < per; j++) {
    const int d = lane * per + j;
    const float zg = b2f(proj[n * P + zoff + h * HD + d]);
    const float y = ((xv[j] * rstd) * b2f(w[d])) * zg * sigmoid_f(zg);
    out[n * (long)(HV * HD) + h * HD + d] = f2b(y);
  }
}

// K5: full-attention preprocessing. proj row: [ q+gate (HQ*2*D) | k (HKV*D) | v (HKV*D) ]
// q/k: Qwen3_5RMSNorm over D with (1+w), then partial RoPE on first RD dims (bf16 rounding per op).
// outputs: qt [B,HQ,T,D], kt/vt [B,HKV,T,D], gate [N, HQ*D]
__global__ void fullattn_prep_k(const bf16* __restrict__ proj, const float* __restrict__ qw1, const float* __restrict__ kw1,
                                const bf16* __restrict__ cosb, const bf16* __restrict__ sinb,
                                bf16* __restrict__ qt, bf16* __restrict__ kt, bf16* __restrict__ vt, bf16* __restrict__ gate,
                                int T, int HQ, int HKV, int D, int RD, float eps) {
  __shared__ float red[32];
  __shared__ float sh[512];
  const long n = blockIdx.y; const int b = n / T, t = n % T;
  const int task = blockIdx.x, d = threadIdx.x;         // blockDim.x == D
  const int P = HQ * 2 * D + 2 * HKV * D;
  if (task >= HQ + HKV) {                               // v copy
    const int kh = task - HQ - HKV;
    vt[(((long)b * HKV + kh) * T + t) * D + d] = proj[n * P + HQ * 2 * D + HKV * D + kh * D + d];
    return;
  }
  const bool isq = task < HQ;
  const int hh = isq ? task : task - HQ;
  const long src = isq ? (n * P + (long)hh * 2 * D + d) : (n * P + HQ * 2 * D + (long)hh * D + d);
  if (isq) gate[n * (long)(HQ * D) + hh * D + d] = proj[n * P + (long)hh * 2 * D + D + d];
  const float xv = b2f(proj[src]);
  const float ss = block_sum(xv * xv, red);
  const float r = rsqrtf(ss / (float)D + eps);
  const float nv = rbf((xv * r) * (isq ? qw1[d] : kw1[d]));   // RMSNorm output (bf16)
  sh[d] = nv;
  __syncthreads();
  float outv = nv;
  if (d < RD) {
    const int half = RD / 2;
    const float rh = (d < half) ? -sh[d + half] : sh[d - half];
    const float c = b2f(cosb[(long)n * RD + d]), s = b2f(sinb[(long)n * RD + d]);
    outv = rbf(rbf(nv * c) + rbf(rh * s));
  }
  bf16* dst = isq ? qt : kt;
  const int HH = isq ? HQ : HKV;
  dst[(((long)b * HH + hh) * T + t) * D + d] = f2b(outv);
}

// K6: out[n, h*D+d] = bf16( att[b,h,t,d] * bf16(sigmoid(gate[n,h*D+d])) )
__global__ void gate_mul_k(const bf16* __restrict__ att, const bf16* __restrict__ gate, bf16* __restrict__ out,
                           int T, int HQ, int D, long total) {
  long e = (long)blockIdx.x * blockDim.x + threadIdx.x;
  if (e >= total) return;
  const int d = e % D; const long hn = e / D; const int h = hn % HQ; const long n = hn / HQ;
  const int b = n / T, t = n % T;
  const float a = b2f(att[(((long)b * HQ + h) * T + t) * D + d]);
  out[e] = f2b(a * rbf(sigmoid_f(b2f(gate[e]))));
}


// ============================== v2 kernels (optimized) ======================================
// K3v2: linear-attention prep, sliding window over TCH timesteps per thread (each input row read once),
// vectorized conv weights, optional fused L2 norm of q/k heads (FLA l2norm_fwd numerics: y = bf16(x / sqrt(sum x^2 + 1e-6))).
template <int TCH>
__global__ void linattn_prep2_k(const bf16* __restrict__ proj, const bf16* __restrict__ convw,
                                const bf16* __restrict__ A_log, const bf16* __restrict__ dt_bias,
                                bf16* __restrict__ q, bf16* __restrict__ k, bf16* __restrict__ v,
                                float* __restrict__ g, bf16* __restrict__ beta,
                                int T, int KD, int VD, int HV, int HK, int HD, int do_l2) {
  const int C = 2 * KD + VD, P = C + VD + 2 * HV;
  const int b = blockIdx.y, t0 = blockIdx.z * TCH;
  const int tend = min(t0 + TCH, T);
  const int c0 = (blockIdx.x * blockDim.x + threadIdx.x) * 8;
  const long rowbase = (long)b * T;
  if (blockIdx.x == 0 && threadIdx.x < HV) {
    const int h = threadIdx.x;
    const float dtb = b2f(dt_bias[h]), ea = -expf(b2f(A_log[h]));
    for (int t = t0; t < tend; t++) {
      const long n = rowbase + t;
      beta[n * HV + h] = f2b(sigmoid_f(b2f(proj[n * P + C + VD + h])));
      const float xs = b2f(proj[n * P + C + VD + HV + h]) + dtb;
      g[n * HV + h] = ea * (xs > 20.f ? xs : log1pf(expf(xs)));
    }
  }
  if (c0 >= C) return;
  float w[8][4];
  {
    uint4 wv[4];
    #pragma unroll
    for (int u = 0; u < 4; u++) wv[u] = *reinterpret_cast<const uint4*>(convw + (long)c0 * 4 + u * 8);
    const bf16* wb = reinterpret_cast<const bf16*>(wv);
    #pragma unroll
    for (int j = 0; j < 8; j++)
      #pragma unroll
      for (int kk = 0; kk < 4; kk++) w[j][kk] = b2f(wb[j * 4 + kk]);
  }
  float x3[8], x2[8], x1[8];
  auto loadrow = [&](int t, float* dst) {
    if (t < 0) {
      #pragma unroll
      for (int j = 0; j < 8; j++) dst[j] = 0.f;
      return;
    }
    uint4 a = *reinterpret_cast<const uint4*>(proj + (rowbase + t) * P + c0);
    const bf16* ab = reinterpret_cast<const bf16*>(&a);
    #pragma unroll
    for (int j = 0; j < 8; j++) dst[j] = b2f(ab[j]);
  };
  loadrow(t0 - 3, x3); loadrow(t0 - 2, x2); loadrow(t0 - 1, x1);
  const bool isqk = c0 < 2 * KD;
  const int rep = HV / HK;
  bf16* dst = (c0 < KD) ? q : k;
  const int cc = (c0 < KD) ? c0 : c0 - KD;
  const int hk = cc / HD, d = cc % HD;
  for (int t = t0; t < tend; t++) {
    float x0[8]; loadrow(t, x0);
    float o[8];
    #pragma unroll
    for (int j = 0; j < 8; j++) {
      float acc = 0.f;
      acc += w[j][0] * x3[j]; acc += w[j][1] * x2[j]; acc += w[j][2] * x1[j]; acc += w[j][3] * x0[j];
      o[j] = rbf(silu_f(acc));
    }
    const long n = rowbase + t;
    if (isqk && do_l2) {                                   // head = HD/8 = 16 consecutive lanes
      float ss = 0.f;
      #pragma unroll
      for (int j = 0; j < 8; j++) ss += o[j] * o[j];
      #pragma unroll
      for (int off = 8; off > 0; off >>= 1) ss += __shfl_xor_sync(0xffffffff, ss, off);
      const float rstd = 1.0f / sqrtf(ss + 1e-6f);
      #pragma unroll
      for (int j = 0; j < 8; j++) o[j] = o[j] * rstd;
    }
    uint4 ov; bf16* ob = reinterpret_cast<bf16*>(&ov);
    #pragma unroll
    for (int j = 0; j < 8; j++) ob[j] = f2b(o[j]);
    if (isqk) {
      for (int r = 0; r < rep; r++)
        *reinterpret_cast<uint4*>(dst + n * (long)(HV * HD) + (long)(hk * rep + r) * HD + d) = ov;
    } else {
      *reinterpret_cast<uint4*>(v + n * (long)VD + (c0 - 2 * KD)) = ov;
    }
    #pragma unroll
    for (int j = 0; j < 8; j++) { x3[j] = x2[j]; x2[j] = x1[j]; x1[j] = x0[j]; }
  }
}

// K5v2: full-attn prep, one warp per (row, task), 8 dims per lane (D = 256), RoPE partner via shuffle.
__global__ void fullattn_prep2_k(const bf16* __restrict__ proj, const float* __restrict__ qw1, const float* __restrict__ kw1,
                                 const bf16* __restrict__ cosb, const bf16* __restrict__ sinb,
                                 bf16* __restrict__ qt, bf16* __restrict__ kt, bf16* __restrict__ vt, bf16* __restrict__ gate,
                                 long N, int T, int HQ, int HKV, int D, int RD, float eps) {
  const int lane = threadIdx.x & 31;
  const int ntask = HQ + 2 * HKV;
  const long wid = (long)blockIdx.x * (blockDim.x / 32) + threadIdx.x / 32;
  if (wid >= N * ntask) return;
  const long n = wid / ntask; const int task = wid % ntask;
  const int b = n / T, t = n % T;
  const int P = HQ * 2 * D + 2 * HKV * D;
  const int d0 = lane * 8;
  if (task >= HQ + HKV) {
    const int kh = task - HQ - HKV;
    *reinterpret_cast<uint4*>(vt + (((long)b * HKV + kh) * T + t) * D + d0) =
        *reinterpret_cast<const uint4*>(proj + n * P + HQ * 2 * D + HKV * D + kh * D + d0);
    return;
  }
  const bool isq = task < HQ;
  const int hh = isq ? task : task - HQ;
  const long src = isq ? (n * P + (long)hh * 2 * D + d0) : (n * P + HQ * 2 * D + (long)hh * D + d0);
  if (isq) *reinterpret_cast<uint4*>(gate + n * (long)(HQ * D) + hh * D + d0) =
               *reinterpret_cast<const uint4*>(proj + n * P + (long)hh * 2 * D + D + d0);
  uint4 a = *reinterpret_cast<const uint4*>(proj + src);
  const bf16* ab = reinterpret_cast<const bf16*>(&a);
  float xv[8], ss = 0.f;
  #pragma unroll
  for (int j = 0; j < 8; j++) { xv[j] = b2f(ab[j]); ss += xv[j] * xv[j]; }
  #pragma unroll
  for (int o = 16; o > 0; o >>= 1) ss += __shfl_xor_sync(0xffffffff, ss, o);
  const float r = rsqrtf(ss / (float)D + eps);
  const float* w1 = isq ? qw1 : kw1;
  float nv[8];
  #pragma unroll
  for (int j = 0; j < 8; j++) nv[j] = rbf((xv[j] * r) * w1[d0 + j]);
  // rotate_half partner: lanes [0, RD/16) <-> [RD/16, RD/8) (for RD=64: lanes 0-3 <-> 4-7)
  const int hl = RD / 16;
  float pv[8];
  #pragma unroll
  for (int j = 0; j < 8; j++) pv[j] = __shfl_xor_sync(0xffffffff, nv[j], hl);
  float outv[8];
  const bool rope = d0 < RD;                                 // RD % 8 == 0 (checked by the wrapper); loads hoisted out of the branch
  const uint4 cv = rope ? *reinterpret_cast<const uint4*>(cosb + n * RD + d0) : make_uint4(0u, 0u, 0u, 0u);
  const uint4 sv = rope ? *reinterpret_cast<const uint4*>(sinb + n * RD + d0) : make_uint4(0u, 0u, 0u, 0u);
  const bf16* cb = reinterpret_cast<const bf16*>(&cv); const bf16* sb = reinterpret_cast<const bf16*>(&sv);
  #pragma unroll
  for (int j = 0; j < 8; j++) {
    const float rh = (d0 + j < RD / 2) ? -pv[j] : pv[j];
    outv[j] = rope ? rbf(rbf(nv[j] * b2f(cb[j])) + rbf(rh * b2f(sb[j]))) : nv[j];
  }
  uint4 ov; bf16* ob = reinterpret_cast<bf16*>(&ov);
  #pragma unroll
  for (int j = 0; j < 8; j++) ob[j] = f2b(outv[j]);
  bf16* dstp = isq ? qt : kt;
  const int HH = isq ? HQ : HKV;
  *reinterpret_cast<uint4*>(dstp + (((long)b * HH + hh) * T + t) * D + d0) = ov;
}

// K6v2: vectorized gate multiply
__global__ void gate_mul2_k(const bf16* __restrict__ att, const bf16* __restrict__ gate, bf16* __restrict__ out,
                            int T, int HQ, int D, long total8) {
  long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= total8) return;
  const long e = i * 8;
  const int d = e % D; const long hn = e / D; const int h = hn % HQ; const long n = hn / HQ;
  const int b = n / T, t = n % T;
  uint4 aa = *reinterpret_cast<const uint4*>(att + (((long)b * HQ + h) * T + t) * D + d);
  uint4 ga = *reinterpret_cast<const uint4*>(gate + e);
  const bf16* ab = reinterpret_cast<const bf16*>(&aa); const bf16* gb = reinterpret_cast<const bf16*>(&ga);
  uint4 o; bf16* ob = reinterpret_cast<bf16*>(&o);
  #pragma unroll
  for (int j = 0; j < 8; j++) ob[j] = f2b(b2f(ab[j]) * rbf(sigmoid_f(b2f(gb[j]))));
  *reinterpret_cast<uint4*>(out + e) = o;
}


// K3tree: linear-attn prep over a packed prefix-tree layout: rows [0,Lp) = shared prefix, then S suffix segments of Ls rows.
// Conv history for suffix position t<3 comes from the prefix tail (rows Lp-3..Lp-1). Fused L2 norm of q/k (FLA numerics).
template <int TCH>
__global__ void linattn_prep_tree_k(const bf16* __restrict__ proj, const bf16* __restrict__ convw,
                                    const bf16* __restrict__ A_log, const bf16* __restrict__ dt_bias,
                                    bf16* __restrict__ q, bf16* __restrict__ k, bf16* __restrict__ v,
                                    float* __restrict__ g, bf16* __restrict__ beta,
                                    int Lp, int Ls, int KD, int VD, int HV, int HK, int HD) {
  const int C = 2 * KD + VD, P = C + VD + 2 * HV;
  const int seg = blockIdx.y;
  const int segLen = seg == 0 ? Lp : Ls;
  const long base = seg == 0 ? 0 : (long)Lp + (long)(seg - 1) * Ls;
  const int t0 = blockIdx.z * TCH;
  if (t0 >= segLen) return;
  const int tend = min(t0 + TCH, segLen);
  const int c0 = (blockIdx.x * blockDim.x + threadIdx.x) * 8;
  if (blockIdx.x == 0 && threadIdx.x < HV) {
    const int h = threadIdx.x;
    const float dtb = b2f(dt_bias[h]), ea = -expf(b2f(A_log[h]));
    for (int t = t0; t < tend; t++) {
      const long n = base + t;
      beta[n * HV + h] = f2b(sigmoid_f(b2f(proj[n * P + C + VD + h])));
      const float xs = b2f(proj[n * P + C + VD + HV + h]) + dtb;
      g[n * HV + h] = ea * (xs > 20.f ? xs : log1pf(expf(xs)));
    }
  }
  if (c0 >= C) return;
  float w[8][4];
  {
    uint4 wv[4];
    #pragma unroll
    for (int u = 0; u < 4; u++) wv[u] = *reinterpret_cast<const uint4*>(convw + (long)c0 * 4 + u * 8);
    const bf16* wb = reinterpret_cast<const bf16*>(wv);
    #pragma unroll
    for (int j = 0; j < 8; j++)
      #pragma unroll
      for (int kk = 0; kk < 4; kk++) w[j][kk] = b2f(wb[j * 4 + kk]);
  }
  auto srcrow = [&](int t) -> long {           // -1 = zero padding
    if (t >= 0) return base + t;
    if (seg == 0) return -1;
    return (Lp + t >= 0) ? (long)(Lp + t) : -1;
  };
  float x3[8], x2[8], x1[8];
  auto loadrow = [&](long r, float* dst) {
    if (r < 0) {
      #pragma unroll
      for (int j = 0; j < 8; j++) dst[j] = 0.f;
      return;
    }
    uint4 a = *reinterpret_cast<const uint4*>(proj + r * P + c0);
    const bf16* ab = reinterpret_cast<const bf16*>(&a);
    #pragma unroll
    for (int j = 0; j < 8; j++) dst[j] = b2f(ab[j]);
  };
  loadrow(srcrow(t0 - 3), x3); loadrow(srcrow(t0 - 2), x2); loadrow(srcrow(t0 - 1), x1);
  const bool isqk = c0 < 2 * KD;
  const int rep = HV / HK;
  bf16* dst = (c0 < KD) ? q : k;
  const int cc = (c0 < KD) ? c0 : c0 - KD;
  const int hk = cc / HD, d = cc % HD;
  for (int t = t0; t < tend; t++) {
    float x0[8]; loadrow(base + t, x0);
    float o[8];
    #pragma unroll
    for (int j = 0; j < 8; j++) {
      float acc = 0.f;
      acc += w[j][0] * x3[j]; acc += w[j][1] * x2[j]; acc += w[j][2] * x1[j]; acc += w[j][3] * x0[j];
      o[j] = rbf(silu_f(acc));
    }
    const long n = base + t;
    if (isqk) {
      float ss = 0.f;
      #pragma unroll
      for (int j = 0; j < 8; j++) ss += o[j] * o[j];
      #pragma unroll
      for (int off = 8; off > 0; off >>= 1) ss += __shfl_xor_sync(0xffffffff, ss, off);
      const float rstd = 1.0f / sqrtf(ss + 1e-6f);
      #pragma unroll
      for (int j = 0; j < 8; j++) o[j] = o[j] * rstd;
    }
    uint4 ov; bf16* ob = reinterpret_cast<bf16*>(&ov);
    #pragma unroll
    for (int j = 0; j < 8; j++) ob[j] = f2b(o[j]);
    if (isqk) {
      for (int r = 0; r < rep; r++)
        *reinterpret_cast<uint4*>(dst + n * (long)(HV * HD) + (long)(hk * rep + r) * HD + d) = ov;
    } else {
      *reinterpret_cast<uint4*>(v + n * (long)VD + (c0 - 2 * KD)) = ov;
    }
    #pragma unroll
    for (int j = 0; j < 8; j++) { x3[j] = x2[j]; x2[j] = x1[j]; x1[j] = x0[j]; }
  }
}

// K1sk: residual add of a split-K GEMM: delta = bf16( sum_s parts[s] ) (fp32 sum), h = bf16(x + delta), then RMSNorm.
template <typename PT>
__global__ void add_rmsnorm_sk_k(const bf16* __restrict__ x, const PT* __restrict__ parts, int S, long MH,
                                 const float* __restrict__ w1, const int* __restrict__ rowmask,
                                 bf16* __restrict__ h_out, bf16* __restrict__ n_out, int H, float eps) {
  __shared__ float red[32];
  const size_t row = blockIdx.x;
  const int i = threadIdx.x * 8;
  float dsum[8];
  #pragma unroll
  for (int j = 0; j < 8; j++) dsum[j] = 0.f;
  for (int sidx = 0; sidx < S; sidx++) {
    const PT* pp = parts + sidx * MH + row * H + i;
    if constexpr (sizeof(PT) == 4) {
      float4 a0 = *reinterpret_cast<const float4*>(pp), a1 = *reinterpret_cast<const float4*>(pp + 4);
      dsum[0] += a0.x; dsum[1] += a0.y; dsum[2] += a0.z; dsum[3] += a0.w; dsum[4] += a1.x; dsum[5] += a1.y; dsum[6] += a1.z; dsum[7] += a1.w;
    } else {
      uint4 pa = *reinterpret_cast<const uint4*>(pp);
      const bf16* pb = reinterpret_cast<const bf16*>(&pa);
      #pragma unroll
      for (int j = 0; j < 8; j++) dsum[j] += b2f(pb[j]);
    }
  }
  uint4 xa = *reinterpret_cast<const uint4*>(x + row * H + i);
  const bf16* xb = reinterpret_cast<const bf16*>(&xa);
  float v[8];
  #pragma unroll
  for (int j = 0; j < 8; j++) v[j] = rbf(b2f(xb[j]) + rbf(dsum[j]));
  uint4 ho; bf16* hb = reinterpret_cast<bf16*>(&ho);
  #pragma unroll
  for (int j = 0; j < 8; j++) hb[j] = f2b(v[j]);
  *reinterpret_cast<uint4*>(h_out + row * H + i) = ho;
  float ss = 0.f;
  #pragma unroll
  for (int j = 0; j < 8; j++) ss += v[j] * v[j];
  ss = block_sum(ss, red);
  const float r = rsqrtf(ss / (float)H + eps);
  const bool keep = rowmask ? (rowmask[row] != 0) : true;
  uint4 no; bf16* nb = reinterpret_cast<bf16*>(&no);
  #pragma unroll
  for (int j = 0; j < 8; j++) nb[j] = keep ? f2b((v[j] * r) * w1[i + j]) : f2b(0.f);
  *reinterpret_cast<uint4*>(n_out + row * H + i) = no;
}
template <int TCH>
__global__ void linattn_prep_rep_k(const bf16* __restrict__ proj, const bf16* __restrict__ convw,
                                    const bf16* __restrict__ A_log, const bf16* __restrict__ dt_bias,
                                    bf16* __restrict__ q, bf16* __restrict__ k, bf16* __restrict__ v,
                                    float* __restrict__ g, bf16* __restrict__ beta,
                                    int Lp, int Ls, int NS, int KD, int VD, int HV, int HK, int HD) {
  const int C = 2 * KD + VD, P = C + VD + 2 * HV;
  const int seg = blockIdx.y;
  const int segLen = seg == 0 ? Lp : Ls;
  const long base = seg == 0 ? 0 : (long)Lp + (long)(seg - 1) * Ls;
  const int t0 = blockIdx.z * TCH;
  if (t0 >= segLen) return;
  const int tend = min(t0 + TCH, segLen);
  const int c0 = (blockIdx.x * blockDim.x + threadIdx.x) * 8;
  if (blockIdx.x == 0 && threadIdx.x < HV) {
    const int h = threadIdx.x;
    const float dtb = b2f(dt_bias[h]), ea = -expf(b2f(A_log[h]));
    for (int t = t0; t < tend; t++) {
      const long n = base + t;
      const bf16 bv = f2b(sigmoid_f(b2f(proj[n * P + C + VD + h])));
      const float xs = b2f(proj[n * P + C + VD + HV + h]) + dtb;
      const float gv = ea * (xs > 20.f ? xs : log1pf(expf(xs)));
      const int L2 = Lp + Ls;
      const int s0 = seg == 0 ? 0 : seg - 1, s1 = seg == 0 ? NS : seg;
      for (int ss = s0; ss < s1; ss++) {
        const long on = (long)ss * L2 + (seg == 0 ? t : Lp + t);
        beta[on * HV + h] = bv; g[on * HV + h] = gv;
      }
    }
  }
  if (c0 >= C) return;
  float w[8][4];
  {
    uint4 wv[4];
    #pragma unroll
    for (int u = 0; u < 4; u++) wv[u] = *reinterpret_cast<const uint4*>(convw + (long)c0 * 4 + u * 8);
    const bf16* wb = reinterpret_cast<const bf16*>(wv);
    #pragma unroll
    for (int j = 0; j < 8; j++)
      #pragma unroll
      for (int kk = 0; kk < 4; kk++) w[j][kk] = b2f(wb[j * 4 + kk]);
  }
  auto srcrow = [&](int t) -> long {           // -1 = zero padding
    if (t >= 0) return base + t;
    if (seg == 0) return -1;
    return (Lp + t >= 0) ? (long)(Lp + t) : -1;
  };
  float x3[8], x2[8], x1[8];
  auto loadrow = [&](long r, float* dst) {
    if (r < 0) {
      #pragma unroll
      for (int j = 0; j < 8; j++) dst[j] = 0.f;
      return;
    }
    uint4 a = *reinterpret_cast<const uint4*>(proj + r * P + c0);
    const bf16* ab = reinterpret_cast<const bf16*>(&a);
    #pragma unroll
    for (int j = 0; j < 8; j++) dst[j] = b2f(ab[j]);
  };
  loadrow(srcrow(t0 - 3), x3); loadrow(srcrow(t0 - 2), x2); loadrow(srcrow(t0 - 1), x1);
  const bool isqk = c0 < 2 * KD;
  const int rep = HV / HK;
  bf16* dst = (c0 < KD) ? q : k;
  const int cc = (c0 < KD) ? c0 : c0 - KD;
  const int hk = cc / HD, d = cc % HD;
  for (int t = t0; t < tend; t++) {
    float x0[8]; loadrow(base + t, x0);
    float o[8];
    #pragma unroll
    for (int j = 0; j < 8; j++) {
      float acc = 0.f;
      acc += w[j][0] * x3[j]; acc += w[j][1] * x2[j]; acc += w[j][2] * x1[j]; acc += w[j][3] * x0[j];
      o[j] = rbf(silu_f(acc));
    }
    const long n = base + t;
    if (isqk) {
      float ss = 0.f;
      #pragma unroll
      for (int j = 0; j < 8; j++) ss += o[j] * o[j];
      #pragma unroll
      for (int off = 8; off > 0; off >>= 1) ss += __shfl_xor_sync(0xffffffff, ss, off);
      const float rstd = 1.0f / sqrtf(ss + 1e-6f);
      #pragma unroll
      for (int j = 0; j < 8; j++) o[j] = o[j] * rstd;
    }
    uint4 ov; bf16* ob = reinterpret_cast<bf16*>(&ov);
    #pragma unroll
    for (int j = 0; j < 8; j++) ob[j] = f2b(o[j]);
    const int L2 = Lp + Ls;
    const int s0 = seg == 0 ? 0 : seg - 1, s1 = seg == 0 ? NS : seg;
    for (int ss = s0; ss < s1; ss++) {
      const long on = (long)ss * L2 + (seg == 0 ? t : Lp + t);
      if (isqk) {
        for (int r = 0; r < rep; r++)
          *reinterpret_cast<uint4*>(dst + on * (long)(HV * HD) + (long)(hk * rep + r) * HD + d) = ov;
      } else {
        *reinterpret_cast<uint4*>(v + on * (long)VD + (c0 - 2 * KD)) = ov;
      }
    }
    #pragma unroll
    for (int j = 0; j < 8; j++) { x3[j] = x2[j]; x2[j] = x1[j]; x1[j] = x0[j]; }
  }
}


// K4map: gated RMSNorm where x is FLA output in replicated layout [NS, L2, HV, HD]; output row n in packed tree layout.
__global__ void gated_rmsnorm_map_k(const bf16* __restrict__ x, const bf16* __restrict__ proj, const bf16* __restrict__ w,
                                    bf16* __restrict__ out, long rows, int HV, int HD, int zoff, int P, float eps, int Lp, int Ls) {
  const long r = (long)blockIdx.x * (blockDim.x / 32) + (threadIdx.x / 32);
  if (r >= rows) return;
  const int lane = threadIdx.x & 31;
  const long n = r / HV; const int h = r % HV;
  const long src = n < Lp ? n : ((n - Lp) / Ls) * (long)(Lp + Ls) + Lp + (n - Lp) % Ls;
  const int per = HD / 32;
  float xv[8], ss = 0.f;
  for (int j = 0; j < per; j++) { xv[j] = b2f(x[(src * HV + h) * HD + lane * per + j]); ss += xv[j] * xv[j]; }
  #pragma unroll
  for (int o = 16; o > 0; o >>= 1) ss += __shfl_xor_sync(0xffffffff, ss, o);
  const float rstd = 1.0f / sqrtf(ss / (float)HD + eps);
  for (int j = 0; j < per; j++) {
    const int d = lane * per + j;
    const float zg = b2f(proj[n * P + zoff + h * HD + d]);
    out[n * (long)(HV * HD) + h * HD + d] = f2b(((xv[j] * rstd) * b2f(w[d])) * zg * sigmoid_f(zg));
  }
}

// K3map: general prefix-tree prep. Replicated layout [S, L2]: candidate s, position p reads packed row src[s*L2+p]
// (-1 = end padding -> zero input, g = 0, beta = 0). Conv window runs along each candidate's own path.
template <int TCH>
__global__ void linattn_prep_map_k(const bf16* __restrict__ proj, const int* __restrict__ src, const bf16* __restrict__ convw,
                                   const bf16* __restrict__ A_log, const bf16* __restrict__ dt_bias,
                                   bf16* __restrict__ q, bf16* __restrict__ k, bf16* __restrict__ v,
                                   float* __restrict__ g, bf16* __restrict__ beta,
                                   int L2, int KD, int VD, int HV, int HK, int HD) {
  const int C = 2 * KD + VD, P = C + VD + 2 * HV;
  const int sidx = blockIdx.y;
  const int t0 = blockIdx.z * TCH;
  if (t0 >= L2) return;
  const int tend = min(t0 + TCH, L2);
  const int* sm = src + (long)sidx * L2;
  const int c0 = (blockIdx.x * blockDim.x + threadIdx.x) * 8;
  if (blockIdx.x == 0 && threadIdx.x < HV) {
    const int h = threadIdx.x;
    const float dtb = b2f(dt_bias[h]), ea = -expf(b2f(A_log[h]));
    for (int t = t0; t < tend; t++) {
      const long on = (long)sidx * L2 + t; const int r = sm[t];
      if (r < 0) { beta[on * HV + h] = f2b(0.f); g[on * HV + h] = 0.f; continue; }
      beta[on * HV + h] = f2b(sigmoid_f(b2f(proj[(long)r * P + C + VD + h])));
      const float xs = b2f(proj[(long)r * P + C + VD + HV + h]) + dtb;
      g[on * HV + h] = ea * (xs > 20.f ? xs : log1pf(expf(xs)));
    }
  }
  if (c0 >= C) return;
  float w[8][4];
  {
    uint4 wv[4];
    #pragma unroll
    for (int u = 0; u < 4; u++) wv[u] = *reinterpret_cast<const uint4*>(convw + (long)c0 * 4 + u * 8);
    const bf16* wb = reinterpret_cast<const bf16*>(wv);
    #pragma unroll
    for (int j = 0; j < 8; j++)
      #pragma unroll
      for (int kk = 0; kk < 4; kk++) w[j][kk] = b2f(wb[j * 4 + kk]);
  }
  auto loadpos = [&](int t, float* dst) {
    const int r = (t >= 0) ? sm[t] : -1;
    if (r < 0) {
      #pragma unroll
      for (int j = 0; j < 8; j++) dst[j] = 0.f;
      return;
    }
    uint4 a = *reinterpret_cast<const uint4*>(proj + (long)r * P + c0);
    const bf16* ab = reinterpret_cast<const bf16*>(&a);
    #pragma unroll
    for (int j = 0; j < 8; j++) dst[j] = b2f(ab[j]);
  };
  float x3[8], x2[8], x1[8];
  loadpos(t0 - 3, x3); loadpos(t0 - 2, x2); loadpos(t0 - 1, x1);
  const bool isqk = c0 < 2 * KD;
  const int rep = HV / HK;
  bf16* dst = (c0 < KD) ? q : k;
  const int cc = (c0 < KD) ? c0 : c0 - KD;
  const int hk = cc / HD, d = cc % HD;
  for (int t = t0; t < tend; t++) {
    float x0[8]; loadpos(t, x0);
    float o[8];
    #pragma unroll
    for (int j = 0; j < 8; j++) {
      float acc = 0.f;
      acc += w[j][0] * x3[j]; acc += w[j][1] * x2[j]; acc += w[j][2] * x1[j]; acc += w[j][3] * x0[j];
      o[j] = rbf(silu_f(acc));
    }
    if (isqk) {
      float ss = 0.f;
      #pragma unroll
      for (int j = 0; j < 8; j++) ss += o[j] * o[j];
      #pragma unroll
      for (int off = 8; off > 0; off >>= 1) ss += __shfl_xor_sync(0xffffffff, ss, off);
      const float rstd = 1.0f / sqrtf(ss + 1e-6f);
      #pragma unroll
      for (int j = 0; j < 8; j++) o[j] = o[j] * rstd;
    }
    uint4 ov; bf16* ob = reinterpret_cast<bf16*>(&ov);
    #pragma unroll
    for (int j = 0; j < 8; j++) ob[j] = f2b(o[j]);
    const long on = (long)sidx * L2 + t;
    if (isqk) {
      for (int r = 0; r < rep; r++)
        *reinterpret_cast<uint4*>(dst + on * (long)(HV * HD) + (long)(hk * rep + r) * HD + d) = ov;
    } else {
      *reinterpret_cast<uint4*>(v + on * (long)VD + (c0 - 2 * KD)) = ov;
    }
    #pragma unroll
    for (int j = 0; j < 8; j++) { x3[j] = x2[j]; x2[j] = x1[j]; x1[j] = x0[j]; }
  }
}

// K4inv: gated RMSNorm reading FLA's replicated output at inv[n] (replicated row index) for each packed row n.
__global__ void gated_rmsnorm_inv_k(const bf16* __restrict__ x, const int* __restrict__ inv, const bf16* __restrict__ proj,
                                    const bf16* __restrict__ w, bf16* __restrict__ out, long rows, int HV, int HD, int zoff, int P, float eps) {
  const long r = (long)blockIdx.x * (blockDim.x / 32) + (threadIdx.x / 32);
  if (r >= rows) return;
  const int lane = threadIdx.x & 31;
  const long n = r / HV; const int h = r % HV;
  const long srow = inv[n];
  const int per = HD / 32;
  float xv[8], ss = 0.f;
  for (int j = 0; j < per; j++) { xv[j] = b2f(x[(srow * HV + h) * HD + lane * per + j]); ss += xv[j] * xv[j]; }
  #pragma unroll
  for (int o = 16; o > 0; o >>= 1) ss += __shfl_xor_sync(0xffffffff, ss, o);
  const float rstd = 1.0f / sqrtf(ss / (float)HD + eps);
  for (int j = 0; j < per; j++) {
    const int d = lane * per + j;
    const float zg = b2f(proj[n * P + zoff + h * HD + d]);
    out[n * (long)(HV * HD) + h * HD + d] = f2b(((xv[j] * rstd) * b2f(w[d])) * zg * sigmoid_f(zg));
  }
}
// ------------------------------------ host wrappers ------------------------------------------
std::vector<torch::Tensor> add_rmsnorm(torch::Tensor x, c10::optional<torch::Tensor> delta, torch::Tensor w1,
                                       c10::optional<torch::Tensor> rowmask, double eps) {
  CHECK(x); CHECK(w1);
  const long N = x.size(0); const int H = x.size(1);
  TORCH_CHECK(H % 8 == 0 && H / 8 <= 1024);
  auto n_out = torch::empty_like(x);
  torch::Tensor h_out = delta.has_value() ? torch::empty_like(x) : x;
  auto st = at::cuda::getCurrentCUDAStream();
  add_rmsnorm_k<<<N, H / 8, 0, st>>>((bf16*)x.data_ptr(), delta.has_value() ? (bf16*)delta->data_ptr() : nullptr,
      w1.data_ptr<float>(), rowmask.has_value() ? rowmask->data_ptr<int>() : nullptr,
      (bf16*)h_out.data_ptr(), (bf16*)n_out.data_ptr(), H, (float)eps);
  return {h_out, n_out};
}

torch::Tensor silu_mul(torch::Tensor gu) {
  CHECK(gu);
  const long N = gu.size(0); const int I = gu.size(1) / 2;
  TORCH_CHECK(I % 8 == 0);
  auto out = torch::empty({N, I}, gu.options());
  const long total8 = N * (long)I / 8;
  silu_mul_k<<<(total8 + 255) / 256, 256, 0, at::cuda::getCurrentCUDAStream()>>>((bf16*)gu.data_ptr(), (bf16*)out.data_ptr(), I, total8);
  return out;
}

std::vector<torch::Tensor> linattn_prep(torch::Tensor proj, torch::Tensor convw, torch::Tensor A_log, torch::Tensor dt_bias,
                                        int64_t T, int64_t KD, int64_t VD, int64_t HV, int64_t HK, int64_t HD) {
  CHECK(proj); CHECK(convw); CHECK(A_log); CHECK(dt_bias);
  const long N = proj.size(0); const int W = convw.size(1);
  auto o = proj.options();
  auto q = torch::empty({N, HV * HD}, o), k = torch::empty({N, HV * HD}, o), v = torch::empty({N, VD}, o);
  auto g = torch::empty({N, HV}, o.dtype(torch::kFloat32)), beta = torch::empty({N, HV}, o);
  const int C = 2 * KD + VD;
  dim3 grid((C / 8 + 255) / 256, N);
  linattn_prep_k<<<grid, 256, 0, at::cuda::getCurrentCUDAStream()>>>((bf16*)proj.data_ptr(), (bf16*)convw.data_ptr(),
      (bf16*)A_log.data_ptr(), (bf16*)dt_bias.data_ptr(), (bf16*)q.data_ptr(), (bf16*)k.data_ptr(), (bf16*)v.data_ptr(),
      g.data_ptr<float>(), (bf16*)beta.data_ptr(), T, KD, VD, HV, HK, HD, W);
  return {q, k, v, g, beta};
}

torch::Tensor gated_rmsnorm(torch::Tensor x, torch::Tensor proj, torch::Tensor w, int64_t HV, int64_t HD, int64_t zoff, double eps) {
  CHECK(x); CHECK(proj); CHECK(w);
  const long rows = x.numel() / HD; const long N = rows / HV;
  auto out = torch::empty({N, HV * HD}, x.options());
  const int wpb = 8;
  gated_rmsnorm_k<<<(rows + wpb - 1) / wpb, wpb * 32, 0, at::cuda::getCurrentCUDAStream()>>>((bf16*)x.data_ptr(), (bf16*)proj.data_ptr(),
      (bf16*)w.data_ptr(), (bf16*)out.data_ptr(), rows, HV, HD, zoff, proj.size(1), (float)eps);
  return out;
}

std::vector<torch::Tensor> fullattn_prep(torch::Tensor proj, torch::Tensor qw1, torch::Tensor kw1, torch::Tensor cosb, torch::Tensor sinb,
                                         int64_t B, int64_t T, int64_t HQ, int64_t HKV, int64_t D, double eps) {
  CHECK(proj); CHECK(qw1); CHECK(kw1); CHECK(cosb); CHECK(sinb);
  const long N = proj.size(0); const int RD = cosb.size(-1);
  auto o = proj.options();
  auto qt = torch::empty({B, HQ, T, D}, o), kt = torch::empty({B, HKV, T, D}, o), vt = torch::empty({B, HKV, T, D}, o);
  auto gate = torch::empty({N, HQ * D}, o);
  dim3 grid(HQ + 2 * HKV, N);
  fullattn_prep_k<<<grid, D, 0, at::cuda::getCurrentCUDAStream()>>>((bf16*)proj.data_ptr(), qw1.data_ptr<float>(), kw1.data_ptr<float>(),
      (bf16*)cosb.data_ptr(), (bf16*)sinb.data_ptr(), (bf16*)qt.data_ptr(), (bf16*)kt.data_ptr(), (bf16*)vt.data_ptr(),
      (bf16*)gate.data_ptr(), T, HQ, HKV, D, RD, (float)eps);
  return {qt, kt, vt, gate};
}

torch::Tensor gate_mul(torch::Tensor att, torch::Tensor gate, int64_t T, int64_t HQ, int64_t D) {
  CHECK(att); CHECK(gate);
  auto out = torch::empty_like(gate);
  const long total = gate.numel();
  gate_mul_k<<<(total + 255) / 256, 256, 0, at::cuda::getCurrentCUDAStream()>>>((bf16*)att.data_ptr(), (bf16*)gate.data_ptr(),
      (bf16*)out.data_ptr(), T, HQ, D, total);
  return out;
}

std::vector<torch::Tensor> linattn_prep2(torch::Tensor proj, torch::Tensor convw, torch::Tensor A_log, torch::Tensor dt_bias,
                                         int64_t B, int64_t T, int64_t KD, int64_t VD, int64_t HV, int64_t HK, int64_t HD, bool do_l2) {
  CHECK(proj); CHECK(convw); CHECK(A_log); CHECK(dt_bias);
  TORCH_CHECK(convw.size(1) == 4 && (KD % 256) == 0 && (HD / 8) == 16);
  const long N = proj.size(0);
  auto o = proj.options();
  auto q = torch::empty({N, HV * HD}, o), k = torch::empty({N, HV * HD}, o), v = torch::empty({N, VD}, o);
  auto g = torch::empty({N, HV}, o.dtype(torch::kFloat32)), beta = torch::empty({N, HV}, o);
  const int C = 2 * KD + VD; constexpr int TCH = 4;
  dim3 grid((C / 8 + 255) / 256, B, (T + TCH - 1) / TCH);
  linattn_prep2_k<TCH><<<grid, 256, 0, at::cuda::getCurrentCUDAStream()>>>((bf16*)proj.data_ptr(), (bf16*)convw.data_ptr(),
      (bf16*)A_log.data_ptr(), (bf16*)dt_bias.data_ptr(), (bf16*)q.data_ptr(), (bf16*)k.data_ptr(), (bf16*)v.data_ptr(),
      g.data_ptr<float>(), (bf16*)beta.data_ptr(), T, KD, VD, HV, HK, HD, do_l2 ? 1 : 0);
  return {q, k, v, g, beta};
}

std::vector<torch::Tensor> fullattn_prep2(torch::Tensor proj, torch::Tensor qw1, torch::Tensor kw1, torch::Tensor cosb, torch::Tensor sinb,
                                          int64_t B, int64_t T, int64_t HQ, int64_t HKV, int64_t D, double eps) {
  CHECK(proj); CHECK(qw1); CHECK(kw1); CHECK(cosb); CHECK(sinb);
  TORCH_CHECK(D == 256);
  const long N = proj.size(0); const int RD = cosb.size(-1);
  TORCH_CHECK(RD % 8 == 0 && RD <= 256);
  auto o = proj.options();
  auto qt = torch::empty({B, HQ, T, D}, o), kt = torch::empty({B, HKV, T, D}, o), vt = torch::empty({B, HKV, T, D}, o);
  auto gate = torch::empty({N, HQ * D}, o);
  const long warps = N * (HQ + 2 * HKV);
  fullattn_prep2_k<<<(warps + 7) / 8, 256, 0, at::cuda::getCurrentCUDAStream()>>>((bf16*)proj.data_ptr(), qw1.data_ptr<float>(), kw1.data_ptr<float>(),
      (bf16*)cosb.data_ptr(), (bf16*)sinb.data_ptr(), (bf16*)qt.data_ptr(), (bf16*)kt.data_ptr(), (bf16*)vt.data_ptr(),
      (bf16*)gate.data_ptr(), N, T, HQ, HKV, D, RD, (float)eps);
  return {qt, kt, vt, gate};
}

torch::Tensor gate_mul2(torch::Tensor att, torch::Tensor gate, int64_t T, int64_t HQ, int64_t D) {
  CHECK(att); CHECK(gate);
  auto out = torch::empty_like(gate);
  const long total8 = gate.numel() / 8;
  gate_mul2_k<<<(total8 + 255) / 256, 256, 0, at::cuda::getCurrentCUDAStream()>>>((bf16*)att.data_ptr(), (bf16*)gate.data_ptr(),
      (bf16*)out.data_ptr(), T, HQ, D, total8);
  return out;
}

std::vector<torch::Tensor> linattn_prep_tree(torch::Tensor proj, torch::Tensor convw, torch::Tensor A_log, torch::Tensor dt_bias,
                                             int64_t Lp, int64_t S, int64_t Ls, int64_t KD, int64_t VD, int64_t HV, int64_t HK, int64_t HD) {
  CHECK(proj); CHECK(convw); CHECK(A_log); CHECK(dt_bias);
  TORCH_CHECK(convw.size(1) == 4 && (KD % 256) == 0 && (HD / 8) == 16);
  const long N = proj.size(0);
  TORCH_CHECK(N == Lp + S * Ls);
  auto o = proj.options();
  auto q = torch::empty({N, HV * HD}, o), k = torch::empty({N, HV * HD}, o), v = torch::empty({N, VD}, o);
  auto g = torch::empty({N, HV}, o.dtype(torch::kFloat32)), beta = torch::empty({N, HV}, o);
  const int C = 2 * KD + VD; constexpr int TCH = 4;
  const long maxlen = Lp > Ls ? Lp : Ls;
  dim3 grid((C / 8 + 255) / 256, 1 + S, (maxlen + TCH - 1) / TCH);
  linattn_prep_tree_k<TCH><<<grid, 256, 0, at::cuda::getCurrentCUDAStream()>>>((bf16*)proj.data_ptr(), (bf16*)convw.data_ptr(),
      (bf16*)A_log.data_ptr(), (bf16*)dt_bias.data_ptr(), (bf16*)q.data_ptr(), (bf16*)k.data_ptr(), (bf16*)v.data_ptr(),
      g.data_ptr<float>(), (bf16*)beta.data_ptr(), Lp, Ls, KD, VD, HV, HK, HD);
  return {q, k, v, g, beta};
}

std::vector<torch::Tensor> add_rmsnorm_sk(torch::Tensor x, torch::Tensor parts, torch::Tensor w1, c10::optional<torch::Tensor> rowmask, double eps) {
  CHECK(x); CHECK(parts); CHECK(w1);
  const long N = x.size(0); const int H = x.size(1); const int S = parts.size(0);
  TORCH_CHECK(parts.size(1) == N && parts.size(2) == H && H % 8 == 0 && H / 8 <= 1024);
  auto n_out = torch::empty_like(x); auto h_out = torch::empty_like(x);
  auto st = at::cuda::getCurrentCUDAStream();
  const int* rm = rowmask.has_value() ? rowmask->data_ptr<int>() : nullptr;
  if (parts.scalar_type() == torch::kFloat32)
    add_rmsnorm_sk_k<float><<<N, H / 8, 0, st>>>((bf16*)x.data_ptr(), parts.data_ptr<float>(), S, N * (long)H,
        w1.data_ptr<float>(), rm, (bf16*)h_out.data_ptr(), (bf16*)n_out.data_ptr(), H, (float)eps);
  else
    add_rmsnorm_sk_k<bf16><<<N, H / 8, 0, st>>>((bf16*)x.data_ptr(), (bf16*)parts.data_ptr(), S, N * (long)H,
        w1.data_ptr<float>(), rm, (bf16*)h_out.data_ptr(), (bf16*)n_out.data_ptr(), H, (float)eps);
  return {h_out, n_out};
}

std::vector<torch::Tensor> linattn_prep_rep(torch::Tensor proj, torch::Tensor convw, torch::Tensor A_log, torch::Tensor dt_bias,
                                            int64_t Lp, int64_t S, int64_t Ls, int64_t KD, int64_t VD, int64_t HV, int64_t HK, int64_t HD) {
  CHECK(proj); CHECK(convw); CHECK(A_log); CHECK(dt_bias);
  TORCH_CHECK(convw.size(1) == 4 && (KD % 256) == 0 && (HD / 8) == 16 && proj.size(0) == Lp + S * Ls);
  const long R = S * (Lp + Ls);
  auto o = proj.options();
  auto q = torch::empty({R, HV * HD}, o), k = torch::empty({R, HV * HD}, o), v = torch::empty({R, VD}, o);
  auto g = torch::empty({R, HV}, o.dtype(torch::kFloat32)), beta = torch::empty({R, HV}, o);
  const int C = 2 * KD + VD; constexpr int TCH = 4;
  const long maxlen = Lp > Ls ? Lp : Ls;
  dim3 grid((C / 8 + 255) / 256, 1 + S, (maxlen + TCH - 1) / TCH);
  linattn_prep_rep_k<TCH><<<grid, 256, 0, at::cuda::getCurrentCUDAStream()>>>((bf16*)proj.data_ptr(), (bf16*)convw.data_ptr(),
      (bf16*)A_log.data_ptr(), (bf16*)dt_bias.data_ptr(), (bf16*)q.data_ptr(), (bf16*)k.data_ptr(), (bf16*)v.data_ptr(),
      g.data_ptr<float>(), (bf16*)beta.data_ptr(), Lp, Ls, S, KD, VD, HV, HK, HD);
  return {q, k, v, g, beta};
}

torch::Tensor gated_rmsnorm_map(torch::Tensor x, torch::Tensor proj, torch::Tensor w, int64_t HV, int64_t HD, int64_t zoff, double eps,
                                int64_t Lp, int64_t Ls) {
  CHECK(x); CHECK(proj); CHECK(w);
  const long N = proj.size(0); const long rows = N * HV;
  auto out = torch::empty({N, HV * HD}, proj.options());
  gated_rmsnorm_map_k<<<(rows + 7) / 8, 256, 0, at::cuda::getCurrentCUDAStream()>>>((bf16*)x.data_ptr(), (bf16*)proj.data_ptr(),
      (bf16*)w.data_ptr(), (bf16*)out.data_ptr(), rows, HV, HD, zoff, proj.size(1), (float)eps, Lp, Ls);
  return out;
}

std::vector<torch::Tensor> linattn_prep_map(torch::Tensor proj, torch::Tensor src, torch::Tensor convw, torch::Tensor A_log, torch::Tensor dt_bias,
                                            int64_t KD, int64_t VD, int64_t HV, int64_t HK, int64_t HD) {
  CHECK(proj); CHECK(src); CHECK(convw); CHECK(A_log); CHECK(dt_bias);
  TORCH_CHECK(convw.size(1) == 4 && (KD % 256) == 0 && (HD / 8) == 16 && src.dim() == 2 && src.scalar_type() == torch::kInt32);
  const long S = src.size(0), L2 = src.size(1), R = S * L2;
  auto o = proj.options();
  auto q = torch::empty({R, HV * HD}, o), k = torch::empty({R, HV * HD}, o), v = torch::empty({R, VD}, o);
  auto g = torch::empty({R, HV}, o.dtype(torch::kFloat32)), beta = torch::empty({R, HV}, o);
  const int C = 2 * KD + VD; constexpr int TCH = 4;
  dim3 grid((C / 8 + 255) / 256, S, (L2 + TCH - 1) / TCH);
  linattn_prep_map_k<TCH><<<grid, 256, 0, at::cuda::getCurrentCUDAStream()>>>((bf16*)proj.data_ptr(), src.data_ptr<int>(), (bf16*)convw.data_ptr(),
      (bf16*)A_log.data_ptr(), (bf16*)dt_bias.data_ptr(), (bf16*)q.data_ptr(), (bf16*)k.data_ptr(), (bf16*)v.data_ptr(),
      g.data_ptr<float>(), (bf16*)beta.data_ptr(), L2, KD, VD, HV, HK, HD);
  return {q, k, v, g, beta};
}

torch::Tensor gated_rmsnorm_inv(torch::Tensor x, torch::Tensor inv, torch::Tensor proj, torch::Tensor w, int64_t HV, int64_t HD, int64_t zoff, double eps) {
  CHECK(x); CHECK(inv); CHECK(proj); CHECK(w);
  const long N = proj.size(0); const long rows = N * HV;
  auto out = torch::empty({N, HV * HD}, proj.options());
  gated_rmsnorm_inv_k<<<(rows + 7) / 8, 256, 0, at::cuda::getCurrentCUDAStream()>>>((bf16*)x.data_ptr(), inv.data_ptr<int>(), (bf16*)proj.data_ptr(),
      (bf16*)w.data_ptr(), (bf16*)out.data_ptr(), rows, HV, HD, zoff, proj.size(1), (float)eps);
  return out;
}

// ---------------------------------------------------------------------------------------------------------------------
// Round 4.
// ld.global.nc with L1::no_allocate: read-only rows streamed once per block (reuse happens in L2, not L1).
__device__ __forceinline__ uint4 ld_nc_na(const void* p) {
  uint4 r;
  asm("ld.global.nc.L1::no_allocate.v4.u32 {%0, %1, %2, %3}, [%4];" : "=r"(r.x), "=r"(r.y), "=r"(r.z), "=r"(r.w) : "l"(p));
  return r;
}

// K3map2: same math and rounding points as linattn_prep_map_k, but
//  (1) q/k are written once with HK heads (FLA applies grouped-value attention itself, bit-identical output), and
//  (2) the row indices for the whole time chunk go to shared memory first and all TCH+3 input rows are loaded up front,
//      so the dependent index->row load chain is paid once per block instead of once per time step.
template <int TCH>
__global__ void __launch_bounds__(256) linattn_prep_map2_k(const bf16* __restrict__ proj, const int* __restrict__ src, const bf16* __restrict__ convw,
                                    const bf16* __restrict__ A_log, const bf16* __restrict__ dt_bias,
                                    bf16* __restrict__ q, bf16* __restrict__ k, bf16* __restrict__ v,
                                    float* __restrict__ g, bf16* __restrict__ beta,
                                    int L2, int KD, int VD, int HV, int HK, int HD) {
  __shared__ int srow[TCH + 3];
  const int C = 2 * KD + VD, P = C + VD + 2 * HV;
  const int sidx = blockIdx.y;
  const int t0 = blockIdx.z * TCH;
  if (t0 >= L2) return;
  const int tend = min(t0 + TCH, L2);
  const int* sm = src + (long)sidx * L2;
  if (threadIdx.x < TCH + 3) {
    const int t = t0 - 3 + (int)threadIdx.x;
    srow[threadIdx.x] = (t >= 0 && t < tend) ? sm[t] : -1;
  }
  __syncthreads();
  if (blockIdx.x == 0 && threadIdx.x < HV) {
    const int h = threadIdx.x;
    const float dtb = b2f(dt_bias[h]), ea = -expf(b2f(A_log[h]));
    for (int t = t0; t < tend; t++) {
      const long on = (long)sidx * L2 + t; const int r = srow[t - t0 + 3];
      if (r < 0) { beta[on * HV + h] = f2b(0.f); g[on * HV + h] = 0.f; continue; }
      beta[on * HV + h] = f2b(sigmoid_f(b2f(proj[(long)r * P + C + VD + h])));
      const float xs = b2f(proj[(long)r * P + C + VD + HV + h]) + dtb;
      g[on * HV + h] = ea * (xs > 20.f ? xs : log1pf(expf(xs)));
    }
  }
  const int c0 = (blockIdx.x * blockDim.x + threadIdx.x) * 8;
  if (c0 >= C) return;
  uint4 raw[TCH + 3];
  #pragma unroll
  for (int i = 0; i < TCH + 3; i++) {
    const int r = srow[i];
    raw[i] = (r >= 0) ? ld_nc_na(proj + (long)r * P + c0) : make_uint4(0u, 0u, 0u, 0u);
  }
  float w[8][4];
  {
    uint4 wv[4];
    #pragma unroll
    for (int u = 0; u < 4; u++) wv[u] = *reinterpret_cast<const uint4*>(convw + (long)c0 * 4 + u * 8);
    const bf16* wb = reinterpret_cast<const bf16*>(wv);
    #pragma unroll
    for (int j = 0; j < 8; j++)
      #pragma unroll
      for (int kk = 0; kk < 4; kk++) w[j][kk] = b2f(wb[j * 4 + kk]);
  }
  const bool isqk = c0 < 2 * KD;
  bf16* dst = (c0 < KD) ? q : k;
  const int cc = (c0 < KD) ? c0 : c0 - KD;
  #pragma unroll
  for (int tt = 0; tt < TCH; tt++) {
    const int t = t0 + tt;
    if (t >= tend) break;
    const bf16* b3 = reinterpret_cast<const bf16*>(&raw[tt]);
    const bf16* b2 = reinterpret_cast<const bf16*>(&raw[tt + 1]);
    const bf16* b1 = reinterpret_cast<const bf16*>(&raw[tt + 2]);
    const bf16* b0 = reinterpret_cast<const bf16*>(&raw[tt + 3]);
    float o[8];
    #pragma unroll
    for (int j = 0; j < 8; j++) {
      float acc = 0.f;
      acc += w[j][0] * b2f(b3[j]); acc += w[j][1] * b2f(b2[j]); acc += w[j][2] * b2f(b1[j]); acc += w[j][3] * b2f(b0[j]);
      o[j] = rbf(silu_f(acc));
    }
    if (isqk) {
      float ss = 0.f;
      #pragma unroll
      for (int j = 0; j < 8; j++) ss += o[j] * o[j];
      #pragma unroll
      for (int off = 8; off > 0; off >>= 1) ss += __shfl_xor_sync(0xffffffff, ss, off);
      const float rstd = 1.0f / sqrtf(ss + 1e-6f);
      #pragma unroll
      for (int j = 0; j < 8; j++) o[j] = o[j] * rstd;
    }
    uint4 ov; bf16* ob = reinterpret_cast<bf16*>(&ov);
    #pragma unroll
    for (int j = 0; j < 8; j++) ob[j] = f2b(o[j]);
    const long on = (long)sidx * L2 + t;
    if (isqk) *reinterpret_cast<uint4*>(dst + on * (long)(HK * HD) + cc) = ov;
    else      *reinterpret_cast<uint4*>(v + on * (long)VD + (c0 - 2 * KD)) = ov;
  }
}

std::vector<torch::Tensor> linattn_prep_map2(torch::Tensor proj, torch::Tensor src, torch::Tensor convw, torch::Tensor A_log, torch::Tensor dt_bias,
                                             int64_t KD, int64_t VD, int64_t HV, int64_t HK, int64_t HD, int64_t tch) {
  CHECK(proj); CHECK(src); CHECK(convw); CHECK(A_log); CHECK(dt_bias);
  TORCH_CHECK(convw.size(1) == 4 && (KD % 256) == 0 && (HD / 8) == 16 && src.dim() == 2 && src.scalar_type() == torch::kInt32 && HK * HD == KD);
  const long S = src.size(0), L2 = src.size(1), R = S * L2;
  auto o = proj.options();
  auto q = torch::empty({R, HK * HD}, o), k = torch::empty({R, HK * HD}, o), v = torch::empty({R, VD}, o);
  auto g = torch::empty({R, HV}, o.dtype(torch::kFloat32)), beta = torch::empty({R, HV}, o);
  const int C = 2 * KD + VD;
  auto launch = [&](auto tag) {
    constexpr int TCH = decltype(tag)::value;
    dim3 grid((C / 8 + 255) / 256, S, (L2 + TCH - 1) / TCH);
    linattn_prep_map2_k<TCH><<<grid, 256, 0, at::cuda::getCurrentCUDAStream()>>>((bf16*)proj.data_ptr(), src.data_ptr<int>(), (bf16*)convw.data_ptr(),
        (bf16*)A_log.data_ptr(), (bf16*)dt_bias.data_ptr(), (bf16*)q.data_ptr(), (bf16*)k.data_ptr(), (bf16*)v.data_ptr(),
        g.data_ptr<float>(), (bf16*)beta.data_ptr(), L2, KD, VD, HV, HK, HD);
  };
  if (tch == 1) launch(std::integral_constant<int, 1>{});
  else if (tch == 2) launch(std::integral_constant<int, 2>{});
  else if (tch == 8) launch(std::integral_constant<int, 8>{});
  else launch(std::integral_constant<int, 4>{});
  return {q, k, v, g, beta};
}

// K3dedup: linear-attention prep computed once per packed row and written to every replicated slot that holds it
// (rep_ptr/rep_pos: CSR from packed row to slots; group N = end-padding slots, written as zeros with g = beta = 0).
// Same math and rounding points as linattn_prep_map_k for every real slot; q/k written with HK heads (GVA).
__global__ void __launch_bounds__(256) linattn_prep_dedup_k(const bf16* __restrict__ proj, const int* __restrict__ src,
                                    const int* __restrict__ rep_ptr, const int* __restrict__ rep_pos, const bf16* __restrict__ convw,
                                    const bf16* __restrict__ A_log, const bf16* __restrict__ dt_bias,
                                    bf16* __restrict__ q, bf16* __restrict__ k, bf16* __restrict__ v,
                                    float* __restrict__ g, bf16* __restrict__ beta,
                                    int N, int L2, int KD, int VD, int HV, int HK, int HD) {
  const int C = 2 * KD + VD, P = C + VD + 2 * HV;
  const int n = blockIdx.y;
  const bool pad = n >= N;                              // blocks >= N: 16 end-padding slots each (spread, no long tail)
  int p0, p1;
  if (!pad) { p0 = rep_ptr[n]; p1 = rep_ptr[n + 1]; }
  else { p0 = rep_ptr[N] + (n - N) * 16; p1 = min(rep_ptr[N + 1], p0 + 16); }
  if (p0 >= p1) return;
  int rows[4] = {-1, -1, -1, -1};                       // t-3, t-2, t-1, t
  if (!pad) {
    const int slot = rep_pos[p0]; const int s = slot / L2, t = slot - s * L2;
    #pragma unroll
    for (int i = 0; i < 4; i++) { const int tt = t - 3 + i; rows[i] = tt >= 0 ? src[(long)s * L2 + tt] : -1; }
  }
  if (blockIdx.x == 0 && threadIdx.x < HV) {
    const int h = threadIdx.x;
    float bv = 0.f, gv = 0.f;
    if (!pad) {
      const long r = rows[3];
      bv = sigmoid_f(b2f(proj[r * P + C + VD + h]));
      const float xs = b2f(proj[r * P + C + VD + HV + h]) + b2f(dt_bias[h]);
      gv = -expf(b2f(A_log[h])) * (xs > 20.f ? xs : log1pf(expf(xs)));
    }
    const bf16 bb = f2b(bv);
    for (int p = p0; p < p1; p++) { const long on = rep_pos[p]; beta[on * HV + h] = bb; g[on * HV + h] = gv; }
  }
  const int c0 = (blockIdx.x * blockDim.x + threadIdx.x) * 8;
  if (c0 >= C) return;
  uint4 ov = make_uint4(0u, 0u, 0u, 0u);
  const bool isqk = c0 < 2 * KD;
  if (!pad) {
    uint4 raw[4];
    #pragma unroll
    for (int i = 0; i < 4; i++) raw[i] = rows[i] >= 0 ? ld_nc_na(proj + (long)rows[i] * P + c0) : make_uint4(0u, 0u, 0u, 0u);
    uint4 wv[4];
    #pragma unroll
    for (int u = 0; u < 4; u++) wv[u] = *reinterpret_cast<const uint4*>(convw + (long)c0 * 4 + u * 8);
    const bf16* wb = reinterpret_cast<const bf16*>(wv);
    const bf16* b3 = reinterpret_cast<const bf16*>(&raw[0]); const bf16* b2 = reinterpret_cast<const bf16*>(&raw[1]);
    const bf16* b1 = reinterpret_cast<const bf16*>(&raw[2]); const bf16* b0 = reinterpret_cast<const bf16*>(&raw[3]);
    float o[8];
    #pragma unroll
    for (int j = 0; j < 8; j++) {
      float acc = 0.f;
      acc += b2f(wb[j * 4 + 0]) * b2f(b3[j]); acc += b2f(wb[j * 4 + 1]) * b2f(b2[j]);
      acc += b2f(wb[j * 4 + 2]) * b2f(b1[j]); acc += b2f(wb[j * 4 + 3]) * b2f(b0[j]);
      o[j] = rbf(silu_f(acc));
    }
    if (isqk) {
      float ss = 0.f;
      #pragma unroll
      for (int j = 0; j < 8; j++) ss += o[j] * o[j];
      #pragma unroll
      for (int off = 8; off > 0; off >>= 1) ss += __shfl_xor_sync(0xffffffff, ss, off);
      const float rstd = 1.0f / sqrtf(ss + 1e-6f);
      #pragma unroll
      for (int j = 0; j < 8; j++) o[j] = o[j] * rstd;
    }
    bf16* ob = reinterpret_cast<bf16*>(&ov);
    #pragma unroll
    for (int j = 0; j < 8; j++) ob[j] = f2b(o[j]);
  }
  bf16* dst = (c0 < KD) ? q : k;
  const int cc = (c0 < KD) ? c0 : c0 - KD;
  for (int p = p0; p < p1; p++) {
    const long on = rep_pos[p];
    if (isqk) *reinterpret_cast<uint4*>(dst + on * (long)(HK * HD) + cc) = ov;
    else      *reinterpret_cast<uint4*>(v + on * (long)VD + (c0 - 2 * KD)) = ov;
  }
}

std::vector<torch::Tensor> linattn_prep_dedup(torch::Tensor proj, torch::Tensor src, torch::Tensor rep_ptr, torch::Tensor rep_pos,
                                              torch::Tensor convw, torch::Tensor A_log, torch::Tensor dt_bias,
                                              int64_t KD, int64_t VD, int64_t HV, int64_t HK, int64_t HD) {
  CHECK(proj); CHECK(src); CHECK(rep_ptr); CHECK(rep_pos); CHECK(convw); CHECK(A_log); CHECK(dt_bias);
  TORCH_CHECK(convw.size(1) == 4 && (HD / 8) == 16 && HK * HD == KD && src.dim() == 2 && src.scalar_type() == torch::kInt32);
  const long N = proj.size(0), S = src.size(0), L2 = src.size(1), R = S * L2;
  TORCH_CHECK(rep_ptr.numel() == N + 2 && rep_pos.numel() == R);
  auto o = proj.options();
  auto q = torch::empty({R, HK * HD}, o), k = torch::empty({R, HK * HD}, o), v = torch::empty({R, VD}, o);
  auto g = torch::empty({R, HV}, o.dtype(torch::kFloat32)), beta = torch::empty({R, HV}, o);
  const int C = 2 * KD + VD;
  dim3 grid((C / 8 + 255) / 256, N + (R + 15) / 16);
  linattn_prep_dedup_k<<<grid, 256, 0, at::cuda::getCurrentCUDAStream()>>>((bf16*)proj.data_ptr(), src.data_ptr<int>(),
      rep_ptr.data_ptr<int>(), rep_pos.data_ptr<int>(), (bf16*)convw.data_ptr(), (bf16*)A_log.data_ptr(), (bf16*)dt_bias.data_ptr(),
      (bf16*)q.data_ptr(), (bf16*)k.data_ptr(), (bf16*)v.data_ptr(), g.data_ptr<float>(), (bf16*)beta.data_ptr(),
      (int)N, (int)L2, KD, VD, HV, HK, HD);
  return {q, k, v, g, beta};
}

// K3dedup2: K3dedup with the per-row conv history precomputed on the host (hist[n] = rows t-3..t, one 16-byte load),
// so each thread's first global loads (hist, rep_ptr) are independent and the proj loads follow one round trip later.
template <int REPQK>
__global__ void __launch_bounds__(256) linattn_prep_dedup2_k(const bf16* __restrict__ proj, const int4* __restrict__ hist,
                                    const int* __restrict__ rep_ptr, const int* __restrict__ rep_pos, const bf16* __restrict__ convw,
                                    const bf16* __restrict__ A_log, const bf16* __restrict__ dt_bias,
                                    bf16* __restrict__ q, bf16* __restrict__ k, bf16* __restrict__ v,
                                    float* __restrict__ g, bf16* __restrict__ beta,
                                    int N, int KD, int VD, int HV, int HK, int HD, bf16* __restrict__ krep = nullptr) {
  const int C = 2 * KD + VD, P = C + VD + 2 * HV;
  const int n = blockIdx.y;
  const bool pad = n >= N;
  const int4 hr = __ldg(hist + min(n, N));
  int p0, p1;
  if (!pad) { p0 = __ldg(rep_ptr + n); p1 = __ldg(rep_ptr + n + 1); }
  else { p0 = __ldg(rep_ptr + N) + (n - N) * 16; p1 = min(__ldg(rep_ptr + N + 1), p0 + 16); }
  if (p0 >= p1) return;
  if (blockIdx.x == 0 && threadIdx.x < HV) {
    const int h = threadIdx.x;
    float bv = 0.f, gv = 0.f;
    if (!pad) {
      const long r = hr.w;
      bv = sigmoid_f(b2f(proj[r * P + C + VD + h]));
      const float xs = b2f(proj[r * P + C + VD + HV + h]) + b2f(dt_bias[h]);
      gv = -expf(b2f(A_log[h])) * (xs > 20.f ? xs : log1pf(expf(xs)));
    }
    const bf16 bb = f2b(bv);
    for (int p = p0; p < p1; p++) { const long on = __ldg(rep_pos + p); beta[on * HV + h] = bb; g[on * HV + h] = gv; }
  }
  const int c0 = (blockIdx.x * blockDim.x + threadIdx.x) * 8;
  if (c0 >= C) return;
  uint4 ov = make_uint4(0u, 0u, 0u, 0u);
  const bool isqk = c0 < 2 * KD;
  if (!pad) {
    const uint4 zero = make_uint4(0u, 0u, 0u, 0u);
    const uint4 r0 = hr.x >= 0 ? ld_nc_na(proj + (long)hr.x * P + c0) : zero;
    const uint4 r1 = hr.y >= 0 ? ld_nc_na(proj + (long)hr.y * P + c0) : zero;
    const uint4 r2 = hr.z >= 0 ? ld_nc_na(proj + (long)hr.z * P + c0) : zero;
    const uint4 r3 = ld_nc_na(proj + (long)hr.w * P + c0);
    uint4 wv[4];
    #pragma unroll
    for (int u = 0; u < 4; u++) wv[u] = __ldg(reinterpret_cast<const uint4*>(convw + (long)c0 * 4 + u * 8));
    const bf16* wb = reinterpret_cast<const bf16*>(wv);
    const bf16* b3 = reinterpret_cast<const bf16*>(&r0); const bf16* b2 = reinterpret_cast<const bf16*>(&r1);
    const bf16* b1 = reinterpret_cast<const bf16*>(&r2); const bf16* b0 = reinterpret_cast<const bf16*>(&r3);
    float o[8];
    #pragma unroll
    for (int j = 0; j < 8; j++) {
      float acc = 0.f;
      acc += b2f(wb[j * 4 + 0]) * b2f(b3[j]); acc += b2f(wb[j * 4 + 1]) * b2f(b2[j]);
      acc += b2f(wb[j * 4 + 2]) * b2f(b1[j]); acc += b2f(wb[j * 4 + 3]) * b2f(b0[j]);
      o[j] = rbf(silu_f(acc));
    }
    if (isqk) {
      float ss = 0.f;
      #pragma unroll
      for (int j = 0; j < 8; j++) ss += o[j] * o[j];
      #pragma unroll
      for (int off = 8; off > 0; off >>= 1) ss += __shfl_xor_sync(0xffffffff, ss, off);
      const float rstd = 1.0f / sqrtf(ss + 1e-6f);
      #pragma unroll
      for (int j = 0; j < 8; j++) o[j] = o[j] * rstd;
    }
    bf16* ob = reinterpret_cast<bf16*>(&ov);
    #pragma unroll
    for (int j = 0; j < 8; j++) ob[j] = f2b(o[j]);
  }
  bf16* dst = (c0 < KD) ? q : k;
  const int cc = (c0 < KD) ? c0 : c0 - KD;
  const int hk = cc / HD, dd = cc - hk * HD;
  #pragma unroll 4
  for (int p = p0; p < p1; p++) {
    const long on = __ldg(rep_pos + p);
    if (isqk) {
      constexpr int RW = REPQK == 2 ? 1 : REPQK;          // REPQK 2: GVA q/k plus a replicated copy of k in krep
      #pragma unroll
      for (int r = 0; r < RW; r++)
        *reinterpret_cast<uint4*>(dst + on * (long)(RW * HK * HD) + (long)(hk * RW + r) * HD + dd) = ov;
      if (REPQK == 2 && c0 >= KD) {
        const int rp = HV / HK;
        for (int r = 0; r < rp; r++)
          *reinterpret_cast<uint4*>(krep + on * (long)(HV * HD) + (long)(hk * rp + r) * HD + dd) = ov;
      }
    } else *reinterpret_cast<uint4*>(v + on * (long)VD + (c0 - 2 * KD)) = ov;
  }
}

std::vector<torch::Tensor> linattn_prep_dedup2(torch::Tensor proj, torch::Tensor hist, torch::Tensor rep_ptr, torch::Tensor rep_pos, int64_t L2,
                                               torch::Tensor convw, torch::Tensor A_log, torch::Tensor dt_bias,
                                               int64_t KD, int64_t VD, int64_t HV, int64_t HK, int64_t HD, bool rep_qk) {
  CHECK(proj); CHECK(hist); CHECK(rep_ptr); CHECK(rep_pos); CHECK(convw); CHECK(A_log); CHECK(dt_bias);
  TORCH_CHECK(convw.size(1) == 4 && (HD / 8) == 16 && HK * HD == KD && hist.scalar_type() == torch::kInt32 && hist.size(1) == 4);
  const long N = proj.size(0), R = rep_pos.numel();
  TORCH_CHECK(rep_ptr.numel() == N + 2 && hist.size(0) == N + 1 && R % L2 == 0);
  auto o = proj.options();
  const int rq = rep_qk ? (int)(HV / HK) : 1;
  TORCH_CHECK(rq == 1 || rq == 3);
  auto q = torch::empty({R, rq * HK * HD}, o), k = torch::empty({R, rq * HK * HD}, o), v = torch::empty({R, VD}, o);
  auto g = torch::empty({R, HV}, o.dtype(torch::kFloat32)), beta = torch::empty({R, HV}, o);
  const int C = 2 * KD + VD;
  dim3 grid((C / 8 + 255) / 256, N + (R + 15) / 16);
  if (rq == 3)
    linattn_prep_dedup2_k<3><<<grid, 256, 0, at::cuda::getCurrentCUDAStream()>>>((bf16*)proj.data_ptr(), reinterpret_cast<const int4*>(hist.data_ptr<int>()),
        rep_ptr.data_ptr<int>(), rep_pos.data_ptr<int>(), (bf16*)convw.data_ptr(), (bf16*)A_log.data_ptr(), (bf16*)dt_bias.data_ptr(),
        (bf16*)q.data_ptr(), (bf16*)k.data_ptr(), (bf16*)v.data_ptr(), g.data_ptr<float>(), (bf16*)beta.data_ptr(), (int)N, KD, VD, HV, HK, HD);
  else
    linattn_prep_dedup2_k<1><<<grid, 256, 0, at::cuda::getCurrentCUDAStream()>>>((bf16*)proj.data_ptr(), reinterpret_cast<const int4*>(hist.data_ptr<int>()),
        rep_ptr.data_ptr<int>(), rep_pos.data_ptr<int>(), (bf16*)convw.data_ptr(), (bf16*)A_log.data_ptr(), (bf16*)dt_bias.data_ptr(),
        (bf16*)q.data_ptr(), (bf16*)k.data_ptr(), (bf16*)v.data_ptr(), g.data_ptr<float>(), (bf16*)beta.data_ptr(), (int)N, KD, VD, HV, HK, HD);
  return {q, k, v, g, beta};
}

// ---------------------------------------------------------------------------------------------------------------------
// Round 4, C2: activation lookup tables. Inputs are bf16, so every possible input has one entry (65536). Each entry is
// computed with exactly the device functions used before, so table lookups are bit-identical to recomputation.
__global__ void act_tables_k(bf16* __restrict__ silu_b, float* __restrict__ sig_f) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= 65536) return;
  const unsigned short u = (unsigned short)i;
  const float x = b2f(*reinterpret_cast<const bf16*>(&u));
  silu_b[i] = f2b(silu_f(x));       // silu_mul: rbf(silu_f(g)) == b2f(silu_b[g])
  sig_f[i] = sigmoid_f(x);          // gated RMSNorm: z * sigmoid_f(z)
}
std::vector<torch::Tensor> act_tables(torch::Tensor like) {
  auto silu_b = torch::empty({65536}, like.options().dtype(torch::kBFloat16));
  auto sig_f = torch::empty({65536}, like.options().dtype(torch::kFloat32));
  act_tables_k<<<256, 256, 0, at::cuda::getCurrentCUDAStream()>>>((bf16*)silu_b.data_ptr(), sig_f.data_ptr<float>());
  return {silu_b, sig_f};
}

// silu_mul with the silu table: out = bf16( b2f(silu_b[g]) * u ), identical to bf16( rbf(silu_f(g)) * u ).
__global__ void __launch_bounds__(256) silu_mul_lut_k(const bf16* __restrict__ gu, const unsigned short* __restrict__ tab,
                                                      bf16* __restrict__ out, int I, long total8) {
  const long idx = (long)blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= total8) return;
  const long e = idx * 8;
  const long row = e / I, col = e - row * I;
  const uint4 ga = ld_nc_na(gu + row * 2 * I + col);
  const uint4 ua = ld_nc_na(gu + row * 2 * I + I + col);
  const unsigned short* gs = reinterpret_cast<const unsigned short*>(&ga);
  const bf16* ub = reinterpret_cast<const bf16*>(&ua);
  unsigned short sv[8];
  #pragma unroll
  for (int j = 0; j < 8; j++) sv[j] = __ldg(tab + gs[j]);
  uint4 o; bf16* ob = reinterpret_cast<bf16*>(&o);
  #pragma unroll
  for (int j = 0; j < 8; j++) ob[j] = f2b(b2f(*reinterpret_cast<const bf16*>(&sv[j])) * b2f(ub[j]));
  *reinterpret_cast<uint4*>(out + e) = o;
}
torch::Tensor silu_mul_lut(torch::Tensor gu, torch::Tensor tab) {
  CHECK(gu); CHECK(tab);
  const long N = gu.size(0); const int I = gu.size(1) / 2;
  TORCH_CHECK(I % 8 == 0);
  auto out = torch::empty({N, I}, gu.options());
  const long total8 = N * (long)I / 8;
  silu_mul_lut_k<<<(total8 + 255) / 256, 256, 0, at::cuda::getCurrentCUDAStream()>>>((bf16*)gu.data_ptr(),
      reinterpret_cast<const unsigned short*>(tab.data_ptr()), (bf16*)out.data_ptr(), I, total8);
  return out;
}

// gated RMSNorm (FLA FusedRMSNormGated, swish) reading FLA's replicated output at inv[n]; one warp per (row, head),
// 4 contiguous elements per lane with 8-byte loads/stores; sigmoid from the table. Same expression order as before:
// out = bf16( ((x*rstd)*w) * z * sigmoid(z) ).
__global__ void __launch_bounds__(256) gated_rmsnorm_inv2_k(const bf16* __restrict__ x, const int* __restrict__ inv, const bf16* __restrict__ proj,
                                    const bf16* __restrict__ w, const float* __restrict__ sig, bf16* __restrict__ out,
                                    long rows, int HV, int HD, int zoff, int P, float eps) {
  const long r = (long)blockIdx.x * (blockDim.x / 32) + (threadIdx.x / 32);
  if (r >= rows) return;
  const int lane = threadIdx.x & 31;
  const long n = r / HV; const int h = (int)(r - n * HV);
  const long srow = __ldg(inv + n);
  const int d0 = lane * 4;                                   // HD == 128
  const uint2 xa = *reinterpret_cast<const uint2*>(x + (srow * HV + h) * HD + d0);
  const uint2 za = *reinterpret_cast<const uint2*>(proj + n * P + zoff + h * HD + d0);
  const uint2 wa = __ldg(reinterpret_cast<const uint2*>(w + d0));
  const bf16* xb = reinterpret_cast<const bf16*>(&xa); const bf16* zb = reinterpret_cast<const bf16*>(&za);
  const bf16* wb = reinterpret_cast<const bf16*>(&wa); const unsigned short* zs = reinterpret_cast<const unsigned short*>(&za);
  float xv[4], ss = 0.f;
  #pragma unroll
  for (int j = 0; j < 4; j++) { xv[j] = b2f(xb[j]); ss += xv[j] * xv[j]; }
  #pragma unroll
  for (int o = 16; o > 0; o >>= 1) ss += __shfl_xor_sync(0xffffffff, ss, o);
  const float rstd = 1.0f / sqrtf(ss / (float)HD + eps);
  uint2 oa; bf16* ob = reinterpret_cast<bf16*>(&oa);
  #pragma unroll
  for (int j = 0; j < 4; j++) {
    const float zg = b2f(zb[j]);
    ob[j] = f2b(((xv[j] * rstd) * b2f(wb[j])) * zg * __ldg(sig + zs[j]));
  }
  *reinterpret_cast<uint2*>(out + n * (long)(HV * HD) + h * HD + d0) = oa;
}
torch::Tensor gated_rmsnorm_inv2(torch::Tensor x, torch::Tensor inv, torch::Tensor proj, torch::Tensor w, torch::Tensor sig,
                                 int64_t HV, int64_t HD, int64_t zoff, double eps) {
  CHECK(x); CHECK(inv); CHECK(proj); CHECK(w); CHECK(sig);
  TORCH_CHECK(HD == 128);
  const long N = proj.size(0); const long rows = N * HV;
  auto out = torch::empty({N, HV * HD}, proj.options());
  gated_rmsnorm_inv2_k<<<(rows + 7) / 8, 256, 0, at::cuda::getCurrentCUDAStream()>>>((bf16*)x.data_ptr(), inv.data_ptr<int>(),
      (bf16*)proj.data_ptr(), (bf16*)w.data_ptr(), sig.data_ptr<float>(), (bf16*)out.data_ptr(), rows, (int)HV, (int)HD, (int)zoff,
      (int)proj.size(1), (float)eps);
  return out;
}

// ---------------------------------------------------------------------------------------------------------------------
// Round 4: L2 prefetch of a weight tensor with the bulk-async prefetch instruction (sm_90+). Each thread issues one
// `cp.async.bulk.prefetch.L2.global` for a CHUNK-byte slice; no shared memory, no registers held, returns immediately.
// Launched on a side stream while the linear-attention prep + FLA kernels leave HBM mostly idle, so the next GEMM
// (the output projection) finds its weights in L2.
__global__ void l2_prefetch_k(const char* __restrict__ base, long nbytes, int chunk) {
  const long off = ((long)blockIdx.x * blockDim.x + threadIdx.x) * (long)chunk;
  if (off >= nbytes) return;
  const unsigned sz = (unsigned)min((long)chunk, nbytes - off);
  asm volatile("cp.async.bulk.prefetch.L2.global [%0], %1;" :: "l"(base + off), "r"(sz) : "memory");
}
void l2_prefetch(torch::Tensor t, int64_t chunk) {
  TORCH_CHECK(t.is_cuda() && t.is_contiguous() && chunk % 16 == 0);
  const long nbytes = (t.numel() * t.element_size()) & ~15L;
  const long n = (nbytes + chunk - 1) / chunk;
  l2_prefetch_k<<<(n + 127) / 128, 128, 0, at::cuda::getCurrentCUDAStream()>>>((const char*)t.data_ptr(), nbytes, (int)chunk);
}

// dedup2 with GVA q/k plus a replicated k (for FLA's intra-chunk kkt/solve stage, which is faster with HV key heads).
std::vector<torch::Tensor> linattn_prep_dedup3(torch::Tensor proj, torch::Tensor hist, torch::Tensor rep_ptr, torch::Tensor rep_pos, int64_t L2,
                                               torch::Tensor convw, torch::Tensor A_log, torch::Tensor dt_bias,
                                               int64_t KD, int64_t VD, int64_t HV, int64_t HK, int64_t HD) {
  CHECK(proj); CHECK(hist); CHECK(rep_ptr); CHECK(rep_pos); CHECK(convw); CHECK(A_log); CHECK(dt_bias);
  TORCH_CHECK(convw.size(1) == 4 && (HD / 8) == 16 && HK * HD == KD && hist.scalar_type() == torch::kInt32 && hist.size(1) == 4);
  const long N = proj.size(0), R = rep_pos.numel();
  TORCH_CHECK(rep_ptr.numel() == N + 2 && hist.size(0) == N + 1 && R % L2 == 0);
  auto o = proj.options();
  auto q = torch::empty({R, HK * HD}, o), k = torch::empty({R, HK * HD}, o), krep = torch::empty({R, HV * HD}, o), v = torch::empty({R, VD}, o);
  auto g = torch::empty({R, HV}, o.dtype(torch::kFloat32)), beta = torch::empty({R, HV}, o);
  const int C = 2 * KD + VD;
  dim3 grid((C / 8 + 255) / 256, N + (R + 15) / 16);
  linattn_prep_dedup2_k<2><<<grid, 256, 0, at::cuda::getCurrentCUDAStream()>>>((bf16*)proj.data_ptr(), reinterpret_cast<const int4*>(hist.data_ptr<int>()),
      rep_ptr.data_ptr<int>(), rep_pos.data_ptr<int>(), (bf16*)convw.data_ptr(), (bf16*)A_log.data_ptr(), (bf16*)dt_bias.data_ptr(),
      (bf16*)q.data_ptr(), (bf16*)k.data_ptr(), (bf16*)v.data_ptr(), g.data_ptr<float>(), (bf16*)beta.data_ptr(), (int)N, KD, VD, HV, HK, HD,
      (bf16*)krep.data_ptr());
  return {q, k, krep, v, g, beta};
}

// gate_mul reading the attention output through its own strides (no .contiguous() copy of the SDPA output) and the
// sigmoid from the table: out[n, h*D+d] = bf16( att[b,h,t,d] * rbf(sigmoid(gate)) ), same numerics as gate_mul2_k.
__global__ void __launch_bounds__(256) gate_mul3_k(const bf16* __restrict__ att, const bf16* __restrict__ gate, const float* __restrict__ sig,
                                                   bf16* __restrict__ out, int T, int HQ, int D, long sB, long sH, long sT, long total8) {
  const long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= total8) return;
  const long e = i * 8;
  const int d = e % D; const long hn = e / D; const int h = hn % HQ; const long n = hn / HQ;
  const int b = n / T, t = n % T;
  const uint4 aa = ld_nc_na(att + (long)b * sB + (long)h * sH + (long)t * sT + d);
  const uint4 ga = ld_nc_na(gate + e);
  const bf16* ab = reinterpret_cast<const bf16*>(&aa); const unsigned short* gs = reinterpret_cast<const unsigned short*>(&ga);
  uint4 o; bf16* ob = reinterpret_cast<bf16*>(&o);
  #pragma unroll
  for (int j = 0; j < 8; j++) ob[j] = f2b(b2f(ab[j]) * rbf(__ldg(sig + gs[j])));
  *reinterpret_cast<uint4*>(out + e) = o;
}
torch::Tensor gate_mul3(torch::Tensor att, torch::Tensor gate, torch::Tensor sig, int64_t T, int64_t HQ, int64_t D) {
  CHECK(gate); CHECK(sig);
  TORCH_CHECK(att.is_cuda() && att.dim() == 4 && att.stride(3) == 1 && att.size(1) == HQ && att.size(2) == T && att.size(3) == D && D % 8 == 0);
  TORCH_CHECK(att.stride(0) % 8 == 0 && att.stride(1) % 8 == 0 && att.stride(2) % 8 == 0);
  auto out = torch::empty_like(gate);
  const long total8 = gate.numel() / 8;
  gate_mul3_k<<<(total8 + 255) / 256, 256, 0, at::cuda::getCurrentCUDAStream()>>>((bf16*)att.data_ptr(), (bf16*)gate.data_ptr(),
      sig.data_ptr<float>(), (bf16*)out.data_ptr(), (int)T, (int)HQ, (int)D, att.stride(0), att.stride(1), att.stride(2), total8);
  return out;
}

// add_rmsnorm_sk_k with S as a compile-time constant (all partial loads in flight at once), weights and row mask loaded
// before the block reduction, streaming loads for the partials. Same arithmetic and reduction order (bit-identical).
template <int S>
__global__ void __launch_bounds__(640) add_rmsnorm_sk2_k(const bf16* __restrict__ x, const float* __restrict__ parts, long MH,
                                  const float* __restrict__ w1, const int* __restrict__ rowmask,
                                  bf16* __restrict__ h_out, bf16* __restrict__ n_out, int H, float eps) {
  __shared__ float red[32];
  const size_t row = blockIdx.x;
  const int i = threadIdx.x * 8;
  float4 pa[S][2];
  #pragma unroll
  for (int s = 0; s < S; s++) {
    const float* pp = parts + s * MH + row * H + i;
    const uint4 u0 = ld_nc_na(pp), u1 = ld_nc_na(pp + 4);
    pa[s][0] = *reinterpret_cast<const float4*>(&u0); pa[s][1] = *reinterpret_cast<const float4*>(&u1);
  }
  const uint4 xa = ld_nc_na(x + row * H + i);
  const float4 wa0 = __ldg(reinterpret_cast<const float4*>(w1 + i)), wa1 = __ldg(reinterpret_cast<const float4*>(w1 + i + 4));
  const bool keep = rowmask ? (__ldg(rowmask + row) != 0) : true;
  float dsum[8];
  #pragma unroll
  for (int j = 0; j < 8; j++) dsum[j] = 0.f;
  #pragma unroll
  for (int s = 0; s < S; s++) {
    dsum[0] += pa[s][0].x; dsum[1] += pa[s][0].y; dsum[2] += pa[s][0].z; dsum[3] += pa[s][0].w;
    dsum[4] += pa[s][1].x; dsum[5] += pa[s][1].y; dsum[6] += pa[s][1].z; dsum[7] += pa[s][1].w;
  }
  const bf16* xb = reinterpret_cast<const bf16*>(&xa);
  float v[8];
  #pragma unroll
  for (int j = 0; j < 8; j++) v[j] = rbf(b2f(xb[j]) + rbf(dsum[j]));
  uint4 ho; bf16* hb = reinterpret_cast<bf16*>(&ho);
  #pragma unroll
  for (int j = 0; j < 8; j++) hb[j] = f2b(v[j]);
  *reinterpret_cast<uint4*>(h_out + row * H + i) = ho;
  float ss = 0.f;
  #pragma unroll
  for (int j = 0; j < 8; j++) ss += v[j] * v[j];
  ss = block_sum(ss, red);
  const float r = rsqrtf(ss / (float)H + eps);
  const float wv[8] = {wa0.x, wa0.y, wa0.z, wa0.w, wa1.x, wa1.y, wa1.z, wa1.w};
  uint4 no; bf16* nb = reinterpret_cast<bf16*>(&no);
  #pragma unroll
  for (int j = 0; j < 8; j++) nb[j] = keep ? f2b((v[j] * r) * wv[j]) : f2b(0.f);
  *reinterpret_cast<uint4*>(n_out + row * H + i) = no;
}
std::vector<torch::Tensor> add_rmsnorm_sk2(torch::Tensor x, torch::Tensor parts, torch::Tensor w1, c10::optional<torch::Tensor> rowmask, double eps) {
  CHECK(x); CHECK(parts); CHECK(w1);
  const long N = x.size(0); const int H = x.size(1); const int S = parts.size(0);
  TORCH_CHECK(parts.size(1) == N && parts.size(2) == H && H % 8 == 0 && H / 8 <= 1024 && parts.scalar_type() == torch::kFloat32);
  auto n_out = torch::empty_like(x); auto h_out = torch::empty_like(x);
  auto st = at::cuda::getCurrentCUDAStream();
  const int* rm = rowmask.has_value() ? rowmask->data_ptr<int>() : nullptr;
  const long MH = N * (long)H;
#define SK2(SS) add_rmsnorm_sk2_k<SS><<<N, H / 8, 0, st>>>((bf16*)x.data_ptr(), parts.data_ptr<float>(), MH, w1.data_ptr<float>(), rm, \
      (bf16*)h_out.data_ptr(), (bf16*)n_out.data_ptr(), H, (float)eps)
  if (S == 2) SK2(2); else if (S == 3) SK2(3); else if (S == 4) SK2(4); else TORCH_CHECK(false, "S must be 2..4");
#undef SK2
  return {h_out, n_out};
}

// =====================================================================================================================
// Round 4, C6: fused Gated DeltaNet chunk forward (FLA chunk_gated_delta_rule, chunk 64), one CTA per (sequence, value head).
// Mirrors FLA 0.5.2's stage formulas and bf16 cast points (r4/docs/fla_forward_spec.md); matrix products use
// mma.sync.m16n8k16 (bf16 in, fp32 accumulate) with ldmatrix, the 64x64 unit-lower-triangular inverse is done blockwise
// in fp32. Not bit-identical to FLA (different MMA families / reduction trees), validated by probability deltas.
namespace gdn {
constexpr int BT = 64, D = 128, LDS = D + 8, LDA = BT + 8, NTH = 256;
__device__ __forceinline__ float ex2(float x) { float y; asm("ex2.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x)); return y; }
__device__ __forceinline__ unsigned smem_u32(const void* p) { return (unsigned)__cvta_generic_to_shared(p); }
__device__ __forceinline__ void ldsm_x4(unsigned& r0, unsigned& r1, unsigned& r2, unsigned& r3, const void* p) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];" : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3) : "r"(smem_u32(p)));
}
__device__ __forceinline__ void ldsm_x4_t(unsigned& r0, unsigned& r1, unsigned& r2, unsigned& r3, const void* p) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];" : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3) : "r"(smem_u32(p)));
}
__device__ __forceinline__ void ldsm_x2(unsigned& r0, unsigned& r1, const void* p) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0,%1}, [%2];" : "=r"(r0), "=r"(r1) : "r"(smem_u32(p)));
}
__device__ __forceinline__ void ldsm_x2_t(unsigned& r0, unsigned& r1, const void* p) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16 {%0,%1}, [%2];" : "=r"(r0), "=r"(r1) : "r"(smem_u32(p)));
}
__device__ __forceinline__ void mma16816(float* c, unsigned a0, unsigned a1, unsigned a2, unsigned a3, unsigned b0, unsigned b1) {
  asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
               : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3]) : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}
// A fragment (16x16) from row-major smem A[m][k] at (m0, k0)
__device__ __forceinline__ void fragA(unsigned* a, const bf16* A, int lda, int m0, int k0) {
  const int l = threadIdx.x & 31;
  ldsm_x4(a[0], a[1], a[2], a[3], A + (m0 + (l & 15)) * lda + k0 + (l >> 4) * 8);
}
// A fragment where smem holds A^T row-major (S[k][m]); fragment of A = S^T at (m0, k0)
__device__ __forceinline__ void fragA_T(unsigned* a, const bf16* S, int lds, int m0, int k0) {
  const int l = threadIdx.x & 31, q = l >> 3, i = l & 7;
  // matrices: m0:(k0..7, m0..7) m1:(k0..7, m0+8..) m2:(k0+8.., m0..) m3:(k0+8.., m0+8..)
  const int kr = k0 + ((q >> 1) << 3) + i, mc = m0 + ((q & 1) << 3);
  ldsm_x4_t(a[0], a[1], a[2], a[3], S + kr * lds + mc);
}
// B fragment (16x8) when smem holds B^T row-major (N x K): rows n0..n0+7, k0..k0+15
__device__ __forceinline__ void fragB_nk(unsigned* b, const bf16* BT_, int ld, int n0, int k0) {
  const int l = threadIdx.x & 31, i = l & 7, h = (l >> 3) & 1;
  ldsm_x2(b[0], b[1], BT_ + (n0 + i) * ld + k0 + h * 8);
}
// B fragment (16x8) when smem holds B row-major (K x N): rows k0..k0+15, cols n0..n0+7
__device__ __forceinline__ void fragB_kn(unsigned* b, const bf16* B, int ld, int n0, int k0) {
  const int l = threadIdx.x & 31, i = l & 7, h = (l >> 3) & 1;
  ldsm_x2_t(b[0], b[1], B + (k0 + h * 8 + i) * ld + n0);
}
// two adjacent 16x8 B fragments (n0 and n0+8) with one ldmatrix.x4: b[0..1] = tile n0, b[2..3] = tile n0+8
__device__ __forceinline__ void fragB_kn2(unsigned* b, const bf16* B, int ld, int n0, int k0) {
  const int l = threadIdx.x & 31;
  ldsm_x4_t(b[0], b[1], b[2], b[3], B + (k0 + ((l >> 3) & 1) * 8 + (l & 7)) * ld + n0 + (l >> 4) * 8);
}
__device__ __forceinline__ void fragB_nk2(unsigned* b, const bf16* BT_, int ld, int n0, int k0) {
  const int l = threadIdx.x & 31;
  ldsm_x4(b[0], b[1], b[2], b[3], BT_ + (n0 + (l >> 4) * 8 + (l & 7)) * ld + k0 + ((l >> 3) & 1) * 8);
}
}  // namespace gdn

namespace gdn {
__global__ void __launch_bounds__(NTH, 1) gdn_fused_k(const bf16* __restrict__ q, const bf16* __restrict__ k, const bf16* __restrict__ v,
                                                       const float* __restrict__ g, const bf16* __restrict__ beta, bf16* __restrict__ o,
                                                       int T, int HK, int HV, float scale) {
  extern __shared__ __align__(16) unsigned char smem_raw[];
  bf16* sK = reinterpret_cast<bf16*>(smem_raw);
  bf16* sQ = sK + BT * LDS;
  bf16* sV = sQ + BT * LDS;            // V tile, later reused for Vg
  bf16* sX1 = sV + BT * LDS;           // v*beta, later v_new
  bf16* sX2 = sX1 + BT * LDS;          // k*beta*2^g, later w
  bf16* sH = sX2 + BT * LDS;           // bf16 state (D x D)
  bf16* sA = sH + D * LDS;             // A (bf16), later the output-stage attention
  float* sAt = reinterpret_cast<float*>(sA + BT * LDA);   // fp32 A~ (BT x BT+1)
  float* sInv = sAt + BT * (BT + 1);                     // fp32 inverse
  float* sTmp = sInv + BT * (BT + 1);                    // 3 x 16 x 17
  float* sG = sTmp + 3 * 16 * 17;                         // gate cumsum (log2 domain)
  float* sGr = sG + BT;                                   // raw gate
  float* sBt = sGr + BT;                                  // beta (fp32 of bf16)
  const int b = blockIdx.y, hv = blockIdx.x, hk = hv / (HV / HK);
  const int tid = threadIdx.x, w = tid >> 5, l = tid & 31, gq = l >> 2, tq = l & 3;
  const int NT = (T + BT - 1) / BT;
  float st[2][8][4];
  #pragma unroll
  for (int mi = 0; mi < 2; mi++)
    #pragma unroll
    for (int ni = 0; ni < 8; ni++)
      #pragma unroll
      for (int e = 0; e < 4; e++) st[mi][ni][e] = 0.f;
  const int mt = w & 3, nh = w >> 2;          // output tiles: rows 16*mt, cols 64*nh (+8*j)
  for (int c = 0; c < NT; c++) {
    const int t0 = c * BT, L = min(BT, T - t0);
    // ---- load tiles ----
    for (int idx = tid; idx < BT * 16; idx += NTH) {
      const int r = idx >> 4, c8 = (idx & 15) * 8;
      uint4 zk = make_uint4(0u, 0u, 0u, 0u), zq = zk, zv = zk;
      if (r < L) {
        const long tr = (long)b * T + t0 + r;
        zk = ld_nc_na(k + (tr * HK + hk) * D + c8);
        zq = ld_nc_na(q + (tr * HK + hk) * D + c8);
        zv = ld_nc_na(v + (tr * HV + hv) * D + c8);
      }
      *reinterpret_cast<uint4*>(sK + r * LDS + c8) = zk;
      *reinterpret_cast<uint4*>(sQ + r * LDS + c8) = zq;
      *reinterpret_cast<uint4*>(sV + r * LDS + c8) = zv;
    }
    if (tid < BT) {
      const long tr = (long)b * T + t0 + tid;
      sGr[tid] = tid < L ? g[tr * HV + hv] : 0.f;
      sBt[tid] = tid < L ? b2f(beta[tr * HV + hv]) : 0.f;
    }
    __syncthreads();
    // ---- gate cumsum per chunk (Hillis-Steele per 32-half, second half + first-half total), * RCP_LN2 ----
    if (w == 0) {
      float x0 = sGr[l], x1 = sGr[32 + l];
      #pragma unroll
      for (int d = 1; d < 32; d <<= 1) {
        const float y0 = __shfl_up_sync(0xffffffff, x0, d), y1 = __shfl_up_sync(0xffffffff, x1, d);
        if (l >= d) { x0 = x0 + y0; x1 = x1 + y1; }
      }
      const float tot = __shfl_sync(0xffffffff, x0, 31);
      x1 = x1 + tot;
      sG[l] = x0 * 1.4426950216f; sG[32 + l] = x1 * 1.4426950216f;
    }
    __syncthreads();
    // ---- Gram K K^T and A~ = strict-lower( (M * 2^(g_r-g_s)) * beta_r ) ----
    {
      float acc[4][4] = {};
      #pragma unroll
      for (int kk = 0; kk < D; kk += 16) {
        unsigned a[4]; fragA(a, sK, LDS, 16 * mt, kk);
        #pragma unroll
        for (int j = 0; j < 4; j++) { unsigned bb[2]; fragB_nk(bb, sK, LDS, 32 * nh + 8 * j, kk); mma16816(acc[j], a[0], a[1], a[2], a[3], bb[0], bb[1]); }
      }
      #pragma unroll
      for (int j = 0; j < 4; j++)
        #pragma unroll
        for (int e = 0; e < 4; e++) {
          const int r = 16 * mt + gq + (e >> 1) * 8, s = 32 * nh + 8 * j + 2 * tq + (e & 1);
          float val = 0.f;
          if (r > s && r < L) val = (acc[j][e] * ex2(sG[r] - sG[s])) * sBt[r];
          sAt[r * (BT + 1) + s] = val;
        }
    }
    __syncthreads();
    // ---- (I + A~)^{-1}: diagonal 16x16 blocks by forward substitution (fp32) ----
    if (w < 4 && l < 16) {
      const int base = 16 * w, j = l;
      float x[16];
      #pragma unroll
      for (int i = 0; i < 16; i++) {
        float val;
        if (i < j) val = 0.f;
        else if (i == j) val = 1.f;
        else {
          float sacc = 0.f;
          #pragma unroll
          for (int m = 0; m < 16; m++) if (m >= j && m < i) sacc += sAt[(base + i) * (BT + 1) + base + m] * x[m];
          val = -sacc;
        }
        x[i] = val;
        sInv[(base + i) * (BT + 1) + base + j] = val;
      }
    }
    __syncthreads();
    // off-diagonal blocks by distance: D_ij = -D_ii * sum_{m=j}^{i-1} A~_im D_mj
    #pragma unroll
    for (int dist = 1; dist < 4; dist++) {
      const int nb = 4 - dist;
      for (int e = tid; e < nb * 256; e += NTH) {
        const int bi_ = e >> 8, rc = e & 255, r = rc >> 4, cc = rc & 15;
        const int bj = bi_, bi = bj + dist;
        float sacc = 0.f;
        for (int m = bj; m < bi; m++)
          #pragma unroll
          for (int x = 0; x < 16; x++) sacc += sAt[(16 * bi + r) * (BT + 1) + 16 * m + x] * sInv[(16 * m + x) * (BT + 1) + 16 * bj + cc];
        sTmp[bi_ * 272 + r * 17 + cc] = sacc;
      }
      __syncthreads();
      for (int e = tid; e < nb * 256; e += NTH) {
        const int bi_ = e >> 8, rc = e & 255, r = rc >> 4, cc = rc & 15;
        const int bj = bi_, bi = bj + dist;
        float sacc = 0.f;
        #pragma unroll
        for (int x = 0; x < 16; x++) sacc += sInv[(16 * bi + r) * (BT + 1) + 16 * bi + x] * sTmp[bi_ * 272 + x * 17 + cc];
        sInv[(16 * bi + r) * (BT + 1) + 16 * bj + cc] = -sacc;
      }
      __syncthreads();
    }
    // A (bf16, lower incl. diagonal), v*beta and k*beta*2^g operands
    for (int e = tid; e < BT * BT; e += NTH) {
      const int r = e >> 6, s = e & 63;
      sA[r * LDA + s] = s <= r ? f2b(sInv[r * (BT + 1) + s]) : f2b(0.f);
    }
    for (int e = tid; e < BT * D; e += NTH) {
      const int r = e >> 7, cc = e & 127;
      const bf16 bt = f2b(sBt[r]);
      sX1[r * LDS + cc] = f2b(b2f(sV[r * LDS + cc]) * b2f(bt));
      const float kb = b2f(f2b(b2f(sK[r * LDS + cc]) * b2f(bt)));
      sX2[r * LDS + cc] = f2b(kb * ex2(sG[r]));
    }
    // bf16 state for this chunk
    #pragma unroll
    for (int mi = 0; mi < 2; mi++)
      #pragma unroll
      for (int ni = 0; ni < 8; ni++)
        #pragma unroll
        for (int e = 0; e < 4; e++) {
          const int r = 32 * mt + 16 * mi + gq + (e >> 1) * 8, cc = 64 * nh + 8 * ni + 2 * tq + (e & 1);
          sH[r * LDS + cc] = f2b(st[mi][ni][e]);
        }
    __syncthreads();
    // ---- u = A (v*beta), w = A (k*beta*2^g) ----
    float uu[8][4] = {}, ww[8][4] = {};
    #pragma unroll
    for (int kk = 0; kk < BT; kk += 16) {
      unsigned a[4]; fragA(a, sA, LDA, 16 * mt, kk);
      #pragma unroll
      for (int j = 0; j < 8; j++) {
        unsigned bb[2];
        fragB_kn(bb, sX1, LDS, 64 * nh + 8 * j, kk); mma16816(uu[j], a[0], a[1], a[2], a[3], bb[0], bb[1]);
        fragB_kn(bb, sX2, LDS, 64 * nh + 8 * j, kk); mma16816(ww[j], a[0], a[1], a[2], a[3], bb[0], bb[1]);
      }
    }
    __syncthreads();
    #pragma unroll
    for (int j = 0; j < 8; j++)
      #pragma unroll
      for (int e = 0; e < 4; e++) {
        const int r = 16 * mt + gq + (e >> 1) * 8, cc = 64 * nh + 8 * j + 2 * tq + (e & 1);
        sX2[r * LDS + cc] = f2b(ww[j][e]);                   // w (bf16)
        uu[j][e] = b2f(f2b(uu[j][e]));                       // u rounded to bf16
      }
    __syncthreads();
    // ---- P = W h; v_new = bf16(u - P); Vg = bf16((u - P) * 2^(gL - g_r)) ----
    {
      float pp[8][4] = {};
      #pragma unroll
      for (int kk = 0; kk < D; kk += 16) {
        unsigned a[4]; fragA(a, sX2, LDS, 16 * mt, kk);
        #pragma unroll
        for (int j = 0; j < 8; j++) { unsigned bb[2]; fragB_kn(bb, sH, LDS, 64 * nh + 8 * j, kk); mma16816(pp[j], a[0], a[1], a[2], a[3], bb[0], bb[1]); }
      }
      const float gL = sG[L - 1];
      #pragma unroll
      for (int j = 0; j < 8; j++)
        #pragma unroll
        for (int e = 0; e < 4; e++) {
          const int r = 16 * mt + gq + (e >> 1) * 8, cc = 64 * nh + 8 * j + 2 * tq + (e & 1);
          const float vn = uu[j][e] - pp[j][e];
          sX1[r * LDS + cc] = f2b(vn);
          sV[r * LDS + cc] = r < L ? f2b(vn * ex2(gL - sG[r])) : f2b(0.f);
        }
    }
    // ---- output-stage intra-chunk attention A_o = bf16( (Q K^T) * 2^(g_r - g_s) ), s <= r ----
    {
      float acc[4][4] = {};
      #pragma unroll
      for (int kk = 0; kk < D; kk += 16) {
        unsigned a[4]; fragA(a, sQ, LDS, 16 * mt, kk);
        #pragma unroll
        for (int j = 0; j < 4; j++) { unsigned bb[2]; fragB_nk(bb, sK, LDS, 32 * nh + 8 * j, kk); mma16816(acc[j], a[0], a[1], a[2], a[3], bb[0], bb[1]); }
      }
      #pragma unroll
      for (int j = 0; j < 4; j++)
        #pragma unroll
        for (int e = 0; e < 4; e++) {
          const int r = 16 * mt + gq + (e >> 1) * 8, s = 32 * nh + 8 * j + 2 * tq + (e & 1);
          sA[r * LDA + s] = (s <= r && r < L && s < L) ? f2b(acc[j][e] * ex2(sG[r] - sG[s])) : f2b(0.f);
        }
    }
    __syncthreads();
    // ---- o = bf16( fma(scale, 2^g_r * (Q h), scale * (A_o v_new)) ) ----
    {
      float qh[8][4] = {}, pv[8][4] = {};
      #pragma unroll
      for (int kk = 0; kk < D; kk += 16) {
        unsigned a[4]; fragA(a, sQ, LDS, 16 * mt, kk);
        #pragma unroll
        for (int j = 0; j < 8; j++) { unsigned bb[2]; fragB_kn(bb, sH, LDS, 64 * nh + 8 * j, kk); mma16816(qh[j], a[0], a[1], a[2], a[3], bb[0], bb[1]); }
      }
      #pragma unroll
      for (int kk = 0; kk < BT; kk += 16) {
        unsigned a[4]; fragA(a, sA, LDA, 16 * mt, kk);
        #pragma unroll
        for (int j = 0; j < 8; j++) { unsigned bb[2]; fragB_kn(bb, sX1, LDS, 64 * nh + 8 * j, kk); mma16816(pv[j], a[0], a[1], a[2], a[3], bb[0], bb[1]); }
      }
      #pragma unroll
      for (int j = 0; j < 8; j++)
        #pragma unroll
        for (int hh = 0; hh < 2; hh++) {
          const int r = 16 * mt + gq + hh * 8, cc = 64 * nh + 8 * j + 2 * tq;
          if (r < L) {
            const float eg = ex2(sG[r]);
            __nv_bfloat162 ov;
            ov.x = f2b(fmaf(scale, eg * qh[j][2 * hh], scale * pv[j][2 * hh]));
            ov.y = f2b(fmaf(scale, eg * qh[j][2 * hh + 1], scale * pv[j][2 * hh + 1]));
            *reinterpret_cast<__nv_bfloat162*>(o + (((long)b * T + t0 + r) * HV + hv) * D + cc) = ov;
          }
        }
    }
    // ---- state update S = S * 2^gL + K^T Vg (skipped after the last chunk) ----
    if (c + 1 < NT) {
      const float dL = ex2(sG[L - 1]);
      #pragma unroll
      for (int mi = 0; mi < 2; mi++)
        #pragma unroll
        for (int ni = 0; ni < 8; ni++)
          #pragma unroll
          for (int e = 0; e < 4; e++) st[mi][ni][e] = st[mi][ni][e] * dL;
      #pragma unroll
      for (int kk = 0; kk < BT; kk += 16) {
        #pragma unroll
        for (int mi = 0; mi < 2; mi++) {
          unsigned a[4]; fragA_T(a, sK, LDS, 32 * mt + 16 * mi, kk);
          #pragma unroll
          for (int ni = 0; ni < 8; ni++) { unsigned bb[2]; fragB_kn(bb, sV, LDS, 64 * nh + 8 * ni, kk); mma16816(st[mi][ni], a[0], a[1], a[2], a[3], bb[0], bb[1]); }
        }
      }
    }
    __syncthreads();
  }
}
}  // namespace gdn

torch::Tensor gdn_fused(torch::Tensor q, torch::Tensor k, torch::Tensor v, torch::Tensor g, torch::Tensor beta, double scale) {
  CHECK(q); CHECK(k); CHECK(v); CHECK(g); CHECK(beta);
  TORCH_CHECK(q.dim() == 4 && q.size(3) == 128 && v.size(3) == 128 && g.scalar_type() == torch::kFloat32);
  const int B = q.size(0), T = q.size(1), HK = q.size(2), HV = v.size(2);
  auto o = torch::empty_like(v);
  const size_t smem = (5 * gdn::BT * gdn::LDS + gdn::D * gdn::LDS + gdn::BT * gdn::LDA) * sizeof(bf16)
                    + (2 * gdn::BT * (gdn::BT + 1) + 3 * 16 * 17 + 3 * gdn::BT) * sizeof(float);
  static bool attr = false;
  if (!attr) { cudaFuncSetAttribute(gdn::gdn_fused_k, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem); attr = true; }
  gdn::gdn_fused_k<<<dim3(HV, B), gdn::NTH, smem, at::cuda::getCurrentCUDAStream()>>>((bf16*)q.data_ptr(), (bf16*)k.data_ptr(),
      (bf16*)v.data_ptr(), g.data_ptr<float>(), (bf16*)beta.data_ptr(), (bf16*)o.data_ptr(), T, HK, HV, (float)scale);
  return o;
}

// C6 v2: 16 warps per CTA (512 threads, <=128 regs), fp32 state split over 16 warps (32 regs), v*beta and k*beta*2^g
// built while loading the tiles (after an early gate cumsum), bf16 state written as packed pairs.
namespace gdn {
constexpr int NTH2 = 512;
__global__ void __launch_bounds__(NTH2, 1) gdn_fused2_k(const bf16* __restrict__ q, const bf16* __restrict__ k, const bf16* __restrict__ v,
                                                         const float* __restrict__ g, const bf16* __restrict__ beta, bf16* __restrict__ o,
                                                         int T, int HK, int HV, float scale) {
  extern __shared__ __align__(16) unsigned char smem_raw[];
  bf16* sK = reinterpret_cast<bf16*>(smem_raw);
  bf16* sQ = sK + BT * LDS;
  bf16* sVG = sQ + BT * LDS;           // gated v_new (state update operand)
  bf16* sX1 = sVG + BT * LDS;          // v*beta, later v_new
  bf16* sX2 = sX1 + BT * LDS;          // k*beta*2^g, later w
  bf16* sH = sX2 + BT * LDS;           // bf16 state (D x D)
  bf16* sA = sH + D * LDS;             // A (bf16), later output-stage attention
  float* sAt = reinterpret_cast<float*>(sA + BT * LDA);
  float* sInv = sAt + BT * (BT + 1);
  float* sTmp = sInv + BT * (BT + 1);
  float* sG = sTmp + 3 * 16 * 17;
  float* sGr = sG + BT;
  float* sBt = sGr + BT;
  const int b = blockIdx.y, hv = blockIdx.x, hk = hv / (HV / HK);
  const int tid = threadIdx.x, w = tid >> 5, l = tid & 31, gq = l >> 2, tq = l & 3;
  const int NT = (T + BT - 1) / BT;
  const int mt = w & 3, nq = w >> 2;           // 64-row tiles: rows 16*mt; 64-col: 16*nq (+8j, j<2); 128-col: 32*nq (+8j, j<4)
  float st[2][4][4];
  #pragma unroll
  for (int mi = 0; mi < 2; mi++)
    #pragma unroll
    for (int ni = 0; ni < 4; ni++)
      #pragma unroll
      for (int e = 0; e < 4; e++) st[mi][ni][e] = 0.f;
  for (int c = 0; c < NT; c++) {
    const int t0 = c * BT, L = min(BT, T - t0);
    if (tid < BT) {
      const long tr = (long)b * T + t0 + tid;
      sGr[tid] = tid < L ? g[tr * HV + hv] : 0.f;
      sBt[tid] = tid < L ? b2f(beta[tr * HV + hv]) : 0.f;
    }
    __syncthreads();
    if (w == 0) {
      float x0 = sGr[l], x1 = sGr[32 + l];
      #pragma unroll
      for (int d = 1; d < 32; d <<= 1) {
        const float y0 = __shfl_up_sync(0xffffffff, x0, d), y1 = __shfl_up_sync(0xffffffff, x1, d);
        if (l >= d) { x0 = x0 + y0; x1 = x1 + y1; }
      }
      const float tot = __shfl_sync(0xffffffff, x0, 31);
      x1 = x1 + tot;
      sG[l] = x0 * 1.4426950216f; sG[32 + l] = x1 * 1.4426950216f;
    }
    __syncthreads();
    // tiles: K, Q raw; v*beta and k*beta*2^g built on the fly
    for (int idx = tid; idx < BT * 16; idx += NTH2) {
      const int r = idx >> 4, c8 = (idx & 15) * 8;
      uint4 zk = make_uint4(0u, 0u, 0u, 0u), zq = zk, zv = zk;
      if (r < L) {
        const long tr = (long)b * T + t0 + r;
        zk = ld_nc_na(k + (tr * HK + hk) * D + c8);
        zq = ld_nc_na(q + (tr * HK + hk) * D + c8);
        zv = ld_nc_na(v + (tr * HV + hv) * D + c8);
      }
      *reinterpret_cast<uint4*>(sK + r * LDS + c8) = zk;
      *reinterpret_cast<uint4*>(sQ + r * LDS + c8) = zq;
      const float bt = b2f(f2b(sBt[r])), eg = ex2(sG[r]);
      const bf16* kb = reinterpret_cast<const bf16*>(&zk); const bf16* vb = reinterpret_cast<const bf16*>(&zv);
      uint4 ovb, okb; bf16* pvb = reinterpret_cast<bf16*>(&ovb); bf16* pkb = reinterpret_cast<bf16*>(&okb);
      #pragma unroll
      for (int j = 0; j < 8; j++) {
        pvb[j] = f2b(b2f(vb[j]) * bt);
        pkb[j] = f2b(b2f(f2b(b2f(kb[j]) * bt)) * eg);
      }
      *reinterpret_cast<uint4*>(sX1 + r * LDS + c8) = ovb;
      *reinterpret_cast<uint4*>(sX2 + r * LDS + c8) = okb;
    }
    // bf16 state for this chunk (packed pairs)
    #pragma unroll
    for (int mi = 0; mi < 2; mi++)
      #pragma unroll
      for (int ni = 0; ni < 4; ni++)
        #pragma unroll
        for (int hh = 0; hh < 2; hh++) {
          const int r = 32 * mt + 16 * mi + gq + hh * 8, cc = 32 * nq + 8 * ni + 2 * tq;
          __nv_bfloat162 pr; pr.x = f2b(st[mi][ni][2 * hh]); pr.y = f2b(st[mi][ni][2 * hh + 1]);
          *reinterpret_cast<__nv_bfloat162*>(sH + r * LDS + cc) = pr;
        }
    __syncthreads();
    // Gram K K^T -> A~
    {
      float acc[2][4] = {};
      #pragma unroll
      for (int kk = 0; kk < D; kk += 16) {
        unsigned a[4]; fragA(a, sK, LDS, 16 * mt, kk);
        #pragma unroll
        for (int j = 0; j < 2; j++) { unsigned bb[2]; fragB_nk(bb, sK, LDS, 16 * nq + 8 * j, kk); mma16816(acc[j], a[0], a[1], a[2], a[3], bb[0], bb[1]); }
      }
      #pragma unroll
      for (int j = 0; j < 2; j++)
        #pragma unroll
        for (int e = 0; e < 4; e++) {
          const int r = 16 * mt + gq + (e >> 1) * 8, s = 16 * nq + 8 * j + 2 * tq + (e & 1);
          float val = 0.f;
          if (r > s && r < L) val = (acc[j][e] * ex2(sG[r] - sG[s])) * sBt[r];
          sAt[r * (BT + 1) + s] = val;
        }
    }
    __syncthreads();
    if (w < 4 && l < 16) {
      const int base = 16 * w, j = l;
      float x[16];
      #pragma unroll
      for (int i = 0; i < 16; i++) {
        float val;
        if (i < j) val = 0.f;
        else if (i == j) val = 1.f;
        else {
          float sacc = 0.f;
          #pragma unroll
          for (int m = 0; m < 16; m++) if (m >= j && m < i) sacc += sAt[(base + i) * (BT + 1) + base + m] * x[m];
          val = -sacc;
        }
        x[i] = val;
        sInv[(base + i) * (BT + 1) + base + j] = val;
      }
    }
    __syncthreads();
    #pragma unroll
    for (int dist = 1; dist < 4; dist++) {
      const int nb = 4 - dist;
      for (int e = tid; e < nb * 256; e += NTH2) {
        const int bi_ = e >> 8, rc = e & 255, r = rc >> 4, cc = rc & 15;
        const int bj = bi_, bi = bj + dist;
        float sacc = 0.f;
        for (int m = bj; m < bi; m++)
          #pragma unroll
          for (int x = 0; x < 16; x++) sacc += sAt[(16 * bi + r) * (BT + 1) + 16 * m + x] * sInv[(16 * m + x) * (BT + 1) + 16 * bj + cc];
        sTmp[bi_ * 272 + r * 17 + cc] = sacc;
      }
      __syncthreads();
      for (int e = tid; e < nb * 256; e += NTH2) {
        const int bi_ = e >> 8, rc = e & 255, r = rc >> 4, cc = rc & 15;
        const int bj = bi_, bi = bj + dist;
        float sacc = 0.f;
        #pragma unroll
        for (int x = 0; x < 16; x++) sacc += sInv[(16 * bi + r) * (BT + 1) + 16 * bi + x] * sTmp[bi_ * 272 + x * 17 + cc];
        sInv[(16 * bi + r) * (BT + 1) + 16 * bj + cc] = -sacc;
      }
      __syncthreads();
    }
    for (int e = tid; e < BT * BT / 2; e += NTH2) {
      const int r = e >> 5, s = (e & 31) * 2;
      __nv_bfloat162 pr;
      pr.x = s <= r ? f2b(sInv[r * (BT + 1) + s]) : f2b(0.f);
      pr.y = s + 1 <= r ? f2b(sInv[r * (BT + 1) + s + 1]) : f2b(0.f);
      *reinterpret_cast<__nv_bfloat162*>(sA + r * LDA + s) = pr;
    }
    __syncthreads();
    // u = A (v*beta), w = A (k*beta*2^g)
    float uu[4][4] = {}, ww[4][4] = {};
    #pragma unroll
    for (int kk = 0; kk < BT; kk += 16) {
      unsigned a[4]; fragA(a, sA, LDA, 16 * mt, kk);
      #pragma unroll
      for (int j = 0; j < 4; j++) {
        unsigned bb[2];
        fragB_kn(bb, sX1, LDS, 32 * nq + 8 * j, kk); mma16816(uu[j], a[0], a[1], a[2], a[3], bb[0], bb[1]);
        fragB_kn(bb, sX2, LDS, 32 * nq + 8 * j, kk); mma16816(ww[j], a[0], a[1], a[2], a[3], bb[0], bb[1]);
      }
    }
    __syncthreads();
    #pragma unroll
    for (int j = 0; j < 4; j++)
      #pragma unroll
      for (int hh = 0; hh < 2; hh++) {
        const int r = 16 * mt + gq + hh * 8, cc = 32 * nq + 8 * j + 2 * tq;
        __nv_bfloat162 pr; pr.x = f2b(ww[j][2 * hh]); pr.y = f2b(ww[j][2 * hh + 1]);
        *reinterpret_cast<__nv_bfloat162*>(sX2 + r * LDS + cc) = pr;
        uu[j][2 * hh] = b2f(f2b(uu[j][2 * hh])); uu[j][2 * hh + 1] = b2f(f2b(uu[j][2 * hh + 1]));
      }
    __syncthreads();
    // P = W h; v_new; Vg
    {
      float pp[4][4] = {};
      #pragma unroll
      for (int kk = 0; kk < D; kk += 16) {
        unsigned a[4]; fragA(a, sX2, LDS, 16 * mt, kk);
        #pragma unroll
        for (int j = 0; j < 4; j++) { unsigned bb[2]; fragB_kn(bb, sH, LDS, 32 * nq + 8 * j, kk); mma16816(pp[j], a[0], a[1], a[2], a[3], bb[0], bb[1]); }
      }
      const float gL = sG[L - 1];
      #pragma unroll
      for (int j = 0; j < 4; j++)
        #pragma unroll
        for (int hh = 0; hh < 2; hh++) {
          const int r = 16 * mt + gq + hh * 8, cc = 32 * nq + 8 * j + 2 * tq;
          const float v0 = uu[j][2 * hh] - pp[j][2 * hh], v1 = uu[j][2 * hh + 1] - pp[j][2 * hh + 1];
          __nv_bfloat162 pn; pn.x = f2b(v0); pn.y = f2b(v1);
          *reinterpret_cast<__nv_bfloat162*>(sX1 + r * LDS + cc) = pn;
          __nv_bfloat162 pg;
          if (r < L) { const float dg = ex2(gL - sG[r]); pg.x = f2b(v0 * dg); pg.y = f2b(v1 * dg); }
          else { pg.x = f2b(0.f); pg.y = f2b(0.f); }
          *reinterpret_cast<__nv_bfloat162*>(sVG + r * LDS + cc) = pg;
        }
    }
    // output-stage attention
    {
      float acc[2][4] = {};
      #pragma unroll
      for (int kk = 0; kk < D; kk += 16) {
        unsigned a[4]; fragA(a, sQ, LDS, 16 * mt, kk);
        #pragma unroll
        for (int j = 0; j < 2; j++) { unsigned bb[2]; fragB_nk(bb, sK, LDS, 16 * nq + 8 * j, kk); mma16816(acc[j], a[0], a[1], a[2], a[3], bb[0], bb[1]); }
      }
      #pragma unroll
      for (int j = 0; j < 2; j++)
        #pragma unroll
        for (int hh = 0; hh < 2; hh++) {
          const int r = 16 * mt + gq + hh * 8, s = 16 * nq + 8 * j + 2 * tq;
          __nv_bfloat162 pr;
          pr.x = (s <= r && r < L && s < L) ? f2b(acc[j][2 * hh] * ex2(sG[r] - sG[s])) : f2b(0.f);
          pr.y = (s + 1 <= r && r < L && s + 1 < L) ? f2b(acc[j][2 * hh + 1] * ex2(sG[r] - sG[s + 1])) : f2b(0.f);
          *reinterpret_cast<__nv_bfloat162*>(sA + r * LDA + s) = pr;
        }
    }
    __syncthreads();
    {
      float qh[4][4] = {}, pv[4][4] = {};
      #pragma unroll
      for (int kk = 0; kk < D; kk += 16) {
        unsigned a[4]; fragA(a, sQ, LDS, 16 * mt, kk);
        #pragma unroll
        for (int j = 0; j < 4; j++) { unsigned bb[2]; fragB_kn(bb, sH, LDS, 32 * nq + 8 * j, kk); mma16816(qh[j], a[0], a[1], a[2], a[3], bb[0], bb[1]); }
      }
      #pragma unroll
      for (int kk = 0; kk < BT; kk += 16) {
        unsigned a[4]; fragA(a, sA, LDA, 16 * mt, kk);
        #pragma unroll
        for (int j = 0; j < 4; j++) { unsigned bb[2]; fragB_kn(bb, sX1, LDS, 32 * nq + 8 * j, kk); mma16816(pv[j], a[0], a[1], a[2], a[3], bb[0], bb[1]); }
      }
      #pragma unroll
      for (int j = 0; j < 4; j++)
        #pragma unroll
        for (int hh = 0; hh < 2; hh++) {
          const int r = 16 * mt + gq + hh * 8, cc = 32 * nq + 8 * j + 2 * tq;
          if (r < L) {
            const float eg = ex2(sG[r]);
            __nv_bfloat162 ov;
            ov.x = f2b(fmaf(scale, eg * qh[j][2 * hh], scale * pv[j][2 * hh]));
            ov.y = f2b(fmaf(scale, eg * qh[j][2 * hh + 1], scale * pv[j][2 * hh + 1]));
            *reinterpret_cast<__nv_bfloat162*>(o + (((long)b * T + t0 + r) * HV + hv) * D + cc) = ov;
          }
        }
    }
    if (c + 1 < NT) {
      const float dL = ex2(sG[L - 1]);
      #pragma unroll
      for (int mi = 0; mi < 2; mi++)
        #pragma unroll
        for (int ni = 0; ni < 4; ni++)
          #pragma unroll
          for (int e = 0; e < 4; e++) st[mi][ni][e] = st[mi][ni][e] * dL;
      #pragma unroll
      for (int kk = 0; kk < BT; kk += 16) {
        #pragma unroll
        for (int mi = 0; mi < 2; mi++) {
          unsigned a[4]; fragA_T(a, sK, LDS, 32 * mt + 16 * mi, kk);
          #pragma unroll
          for (int ni = 0; ni < 4; ni++) { unsigned bb[2]; fragB_kn(bb, sVG, LDS, 32 * nq + 8 * ni, kk); mma16816(st[mi][ni], a[0], a[1], a[2], a[3], bb[0], bb[1]); }
        }
      }
    }
    __syncthreads();
  }
}
}  // namespace gdn

torch::Tensor gdn_fused2(torch::Tensor q, torch::Tensor k, torch::Tensor v, torch::Tensor g, torch::Tensor beta, double scale) {
  CHECK(q); CHECK(k); CHECK(v); CHECK(g); CHECK(beta);
  TORCH_CHECK(q.dim() == 4 && q.size(3) == 128 && v.size(3) == 128 && g.scalar_type() == torch::kFloat32);
  const int B = q.size(0), T = q.size(1), HK = q.size(2), HV = v.size(2);
  auto o = torch::empty_like(v);
  const size_t smem = (5 * gdn::BT * gdn::LDS + gdn::D * gdn::LDS + gdn::BT * gdn::LDA) * sizeof(bf16)
                    + (2 * gdn::BT * (gdn::BT + 1) + 3 * 16 * 17 + 3 * gdn::BT) * sizeof(float);
  static bool attr = false;
  if (!attr) { cudaFuncSetAttribute(gdn::gdn_fused2_k, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem); attr = true; }
  gdn::gdn_fused2_k<<<dim3(HV, B), gdn::NTH2, smem, at::cuda::getCurrentCUDAStream()>>>((bf16*)q.data_ptr(), (bf16*)k.data_ptr(),
      (bf16*)v.data_ptr(), g.data_ptr<float>(), (bf16*)beta.data_ptr(), (bf16*)o.data_ptr(), T, HK, HV, (float)scale);
  return o;
}

// C6 v3 helpers: tf32 mma (FLA uses tf32 dots for the solve_tril block merges) on fp32 smem tiles.
namespace gdn {
__device__ __forceinline__ unsigned f2tf32(float x) { unsigned r; asm("cvt.rna.tf32.f32 %0, %1;" : "=r"(r) : "f"(x)); return r; }
__device__ __forceinline__ void mma1688tf32(float* c, unsigned a0, unsigned a1, unsigned a2, unsigned a3, unsigned b0, unsigned b1) {
  asm volatile("mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
               : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3]) : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}
// C[16x16] (+)= A[16x16] * B[16x16], fp32 smem tiles with leading dimension ld (elements); one warp. acc: [2 n-tiles][4]
__device__ __forceinline__ void mm16_tf32(float (*acc)[4], const float* A, const float* B, int ld) {
  const int l = threadIdx.x & 31, gq = l >> 2, tq = l & 3;
  #pragma unroll
  for (int kk = 0; kk < 16; kk += 8) {
    const unsigned a0 = f2tf32(A[gq * ld + kk + tq]), a1 = f2tf32(A[(gq + 8) * ld + kk + tq]);
    const unsigned a2 = f2tf32(A[gq * ld + kk + tq + 4]), a3 = f2tf32(A[(gq + 8) * ld + kk + tq + 4]);
    #pragma unroll
    for (int nt = 0; nt < 2; nt++) {
      const unsigned b0 = f2tf32(B[(kk + tq) * ld + nt * 8 + gq]), b1 = f2tf32(B[(kk + tq + 4) * ld + nt * 8 + gq]);
      mma1688tf32(acc[nt], a0, a1, a2, a3, b0, b1);
    }
  }
}
__device__ __forceinline__ void st16(float* C, int ld, const float (*acc)[4], float sgn) {
  const int l = threadIdx.x & 31, gq = l >> 2, tq = l & 3;
  #pragma unroll
  for (int nt = 0; nt < 2; nt++) {
    C[gq * ld + nt * 8 + 2 * tq] = sgn * acc[nt][0]; C[gq * ld + nt * 8 + 2 * tq + 1] = sgn * acc[nt][1];
    C[(gq + 8) * ld + nt * 8 + 2 * tq] = sgn * acc[nt][2]; C[(gq + 8) * ld + nt * 8 + 2 * tq + 1] = sgn * acc[nt][3];
  }
}
// block pointer helpers for a 64x64 fp32 matrix with ld = BT+1
__device__ __forceinline__ float* blk(float* M, int bi, int bj) { return M + (16 * bi) * (BT + 1) + 16 * bj; }
// Off-diagonal blocks of (I + A~)^{-1} from the diagonal inverses (FLA's merge order). One warp per product chain.
// sAt: A~ (fp32), sInv: diagonal blocks filled; scratch: 3 x 16x(BT+1) fp32 rows region (uses ld = BT+1).
__device__ __forceinline__ void merge_blocks(float* sAt, float* sInv, float* scr) {
  const int w = threadIdx.x >> 5;
  const int ld = BT + 1;
  // level 1: D10, D21, D32  (D_{i,i-1} = -(D_i A~_{i,i-1}) D_{i-1})
  if (w < 3) {
    const int i = w + 1;
    float t[2][4] = {}, r[2][4] = {};
    mm16_tf32(t, blk(sInv, i, i), blk(sAt, i, i - 1), ld);
    float* T = scr + w * 16 * ld; st16(T, ld, t, 1.f);
    __syncwarp();
    mm16_tf32(r, T, blk(sInv, i - 1, i - 1), ld);
    st16(blk(sInv, i, i - 1), ld, r, -1.f);
  }
  __syncthreads();
  // level 2: D20 = -D2 (A~20 D0 + A~21 D10), D31 = -D3 (A~31 D1 + A~32 D21)
  if (w < 2) {
    const int i = w + 2, j = w;
    float t[2][4] = {}, r[2][4] = {};
    mm16_tf32(t, blk(sAt, i, j), blk(sInv, j, j), ld);
    mm16_tf32(t, blk(sAt, i, j + 1), blk(sInv, j + 1, j), ld);
    float* T = scr + w * 16 * ld; st16(T, ld, t, 1.f);
    __syncwarp();
    mm16_tf32(r, blk(sInv, i, i), T, ld);
    st16(blk(sInv, i, j), ld, r, -1.f);
  }
  __syncthreads();
  // level 3: D30 = -D3 (A~30 D0 + A~31 D10 + A~32 D20)
  if (w == 0) {
    float t[2][4] = {}, r[2][4] = {};
    mm16_tf32(t, blk(sAt, 3, 0), blk(sInv, 0, 0), ld);
    mm16_tf32(t, blk(sAt, 3, 1), blk(sInv, 1, 0), ld);
    mm16_tf32(t, blk(sAt, 3, 2), blk(sInv, 2, 0), ld);
    float* T = scr; st16(T, ld, t, 1.f);
    __syncwarp();
    mm16_tf32(r, blk(sInv, 3, 3), T, ld);
    st16(blk(sInv, 3, 0), ld, r, -1.f);
  }
  __syncthreads();
}
}  // namespace gdn

namespace gdn {
__global__ void __launch_bounds__(NTH2, 1) gdn_fused3_k(const bf16* __restrict__ q, const bf16* __restrict__ k, const bf16* __restrict__ v,
                                                         const float* __restrict__ g, const bf16* __restrict__ beta, bf16* __restrict__ o,
                                                         int T, int HK, int HV, float scale) {
  extern __shared__ __align__(16) unsigned char smem_raw[];
  bf16* sK = reinterpret_cast<bf16*>(smem_raw);
  bf16* sQ = sK + BT * LDS;
  bf16* sVG = sQ + BT * LDS;           // gated v_new (state update operand)
  bf16* sX1 = sVG + BT * LDS;          // v*beta, later v_new
  bf16* sX2 = sX1 + BT * LDS;          // k*beta*2^g, later w
  bf16* sH = sX2 + BT * LDS;           // bf16 state (D x D)
  bf16* sA = sH + D * LDS;             // A (bf16), later output-stage attention
  float* sAt = reinterpret_cast<float*>(sA + BT * LDA);
  float* sInv = sAt + BT * (BT + 1);
  float* sTmp = sInv + BT * (BT + 1);
  float* sG = sTmp + 3 * 16 * 17;
  float* sGr = sG + BT;
  float* sBt = sGr + BT;
  const int b = blockIdx.y, hv = blockIdx.x, hk = hv / (HV / HK);
  const int tid = threadIdx.x, w = tid >> 5, l = tid & 31, gq = l >> 2, tq = l & 3;
  const int NT = (T + BT - 1) / BT;
  const int mt = w & 3, nq = w >> 2;           // 64-row tiles: rows 16*mt; 64-col: 16*nq (+8j, j<2); 128-col: 32*nq (+8j, j<4)
  float st[2][4][4];
  #pragma unroll
  for (int mi = 0; mi < 2; mi++)
    #pragma unroll
    for (int ni = 0; ni < 4; ni++)
      #pragma unroll
      for (int e = 0; e < 4; e++) st[mi][ni][e] = 0.f;
  for (int c = 0; c < NT; c++) {
    const int t0 = c * BT, L = min(BT, T - t0);
    if (tid < BT) {
      const long tr = (long)b * T + t0 + tid;
      sGr[tid] = tid < L ? g[tr * HV + hv] : 0.f;
      sBt[tid] = tid < L ? b2f(beta[tr * HV + hv]) : 0.f;
    }
    __syncthreads();
    if (w == 0) {
      float x0 = sGr[l], x1 = sGr[32 + l];
      #pragma unroll
      for (int d = 1; d < 32; d <<= 1) {
        const float y0 = __shfl_up_sync(0xffffffff, x0, d), y1 = __shfl_up_sync(0xffffffff, x1, d);
        if (l >= d) { x0 = x0 + y0; x1 = x1 + y1; }
      }
      const float tot = __shfl_sync(0xffffffff, x0, 31);
      x1 = x1 + tot;
      sG[l] = x0 * 1.4426950216f; sG[32 + l] = x1 * 1.4426950216f;
    }
    __syncthreads();
    // tiles: K, Q raw; v*beta and k*beta*2^g built on the fly
    for (int idx = tid; idx < BT * 16; idx += NTH2) {
      const int r = idx >> 4, c8 = (idx & 15) * 8;
      uint4 zk = make_uint4(0u, 0u, 0u, 0u), zq = zk, zv = zk;
      if (r < L) {
        const long tr = (long)b * T + t0 + r;
        zk = ld_nc_na(k + (tr * HK + hk) * D + c8);
        zq = ld_nc_na(q + (tr * HK + hk) * D + c8);
        zv = ld_nc_na(v + (tr * HV + hv) * D + c8);
      }
      *reinterpret_cast<uint4*>(sK + r * LDS + c8) = zk;
      *reinterpret_cast<uint4*>(sQ + r * LDS + c8) = zq;
      const float bt = b2f(f2b(sBt[r])), eg = ex2(sG[r]);
      const bf16* kb = reinterpret_cast<const bf16*>(&zk); const bf16* vb = reinterpret_cast<const bf16*>(&zv);
      uint4 ovb, okb; bf16* pvb = reinterpret_cast<bf16*>(&ovb); bf16* pkb = reinterpret_cast<bf16*>(&okb);
      #pragma unroll
      for (int j = 0; j < 8; j++) {
        pvb[j] = f2b(b2f(vb[j]) * bt);
        pkb[j] = f2b(b2f(f2b(b2f(kb[j]) * bt)) * eg);
      }
      *reinterpret_cast<uint4*>(sX1 + r * LDS + c8) = ovb;
      *reinterpret_cast<uint4*>(sX2 + r * LDS + c8) = okb;
    }
    // bf16 state for this chunk (packed pairs)
    #pragma unroll
    for (int mi = 0; mi < 2; mi++)
      #pragma unroll
      for (int ni = 0; ni < 4; ni++)
        #pragma unroll
        for (int hh = 0; hh < 2; hh++) {
          const int r = 32 * mt + 16 * mi + gq + hh * 8, cc = 32 * nq + 8 * ni + 2 * tq;
          __nv_bfloat162 pr; pr.x = f2b(st[mi][ni][2 * hh]); pr.y = f2b(st[mi][ni][2 * hh + 1]);
          *reinterpret_cast<__nv_bfloat162*>(sH + r * LDS + cc) = pr;
        }
    __syncthreads();
    // Gram K K^T -> A~
    {
      float acc[2][4] = {};
      #pragma unroll
      for (int kk = 0; kk < D; kk += 16) {
        unsigned a[4]; fragA(a, sK, LDS, 16 * mt, kk);
        #pragma unroll
        for (int j = 0; j < 2; j++) { unsigned bb[2]; fragB_nk(bb, sK, LDS, 16 * nq + 8 * j, kk); mma16816(acc[j], a[0], a[1], a[2], a[3], bb[0], bb[1]); }
      }
      #pragma unroll
      for (int j = 0; j < 2; j++)
        #pragma unroll
        for (int e = 0; e < 4; e++) {
          const int r = 16 * mt + gq + (e >> 1) * 8, s = 16 * nq + 8 * j + 2 * tq + (e & 1);
          float val = 0.f;
          if (r > s && r < L) val = (acc[j][e] * ex2(sG[r] - sG[s])) * sBt[r];
          sAt[r * (BT + 1) + s] = val;
        }
    }
    __syncthreads();
    if (w < 4 && l < 16) {       // diagonal 16x16 inverses: column j per lane, branch-free forward substitution
      const int base = 16 * w, j = l;
      float x[16];
      #pragma unroll
      for (int i = 0; i < 16; i++) {
        float sacc = 0.f;
        #pragma unroll
        for (int m = 0; m < i; m++) sacc += sAt[(base + i) * (BT + 1) + base + m] * x[m];   // x[m] == 0 for m < j
        const float val = (i < j) ? 0.f : ((i == j) ? 1.f : -sacc);
        x[i] = val;
        sInv[(base + i) * (BT + 1) + base + j] = val;
      }
    }
    __syncthreads();
    merge_blocks(sAt, sInv, reinterpret_cast<float*>(sVG));   // off-diagonal blocks with tf32 MMAs (FLA's merge order)
    for (int e = tid; e < BT * BT / 2; e += NTH2) {
      const int r = e >> 5, s = (e & 31) * 2;
      __nv_bfloat162 pr;
      pr.x = s <= r ? f2b(sInv[r * (BT + 1) + s]) : f2b(0.f);
      pr.y = s + 1 <= r ? f2b(sInv[r * (BT + 1) + s + 1]) : f2b(0.f);
      *reinterpret_cast<__nv_bfloat162*>(sA + r * LDA + s) = pr;
    }
    __syncthreads();
    // u = A (v*beta), w = A (k*beta*2^g)
    float uu[4][4] = {}, ww[4][4] = {};
    #pragma unroll
    for (int kk = 0; kk < BT; kk += 16) {
      unsigned a[4]; fragA(a, sA, LDA, 16 * mt, kk);
      #pragma unroll
      for (int j = 0; j < 4; j++) {
        unsigned bb[2];
        fragB_kn(bb, sX1, LDS, 32 * nq + 8 * j, kk); mma16816(uu[j], a[0], a[1], a[2], a[3], bb[0], bb[1]);
        fragB_kn(bb, sX2, LDS, 32 * nq + 8 * j, kk); mma16816(ww[j], a[0], a[1], a[2], a[3], bb[0], bb[1]);
      }
    }
    __syncthreads();
    #pragma unroll
    for (int j = 0; j < 4; j++)
      #pragma unroll
      for (int hh = 0; hh < 2; hh++) {
        const int r = 16 * mt + gq + hh * 8, cc = 32 * nq + 8 * j + 2 * tq;
        __nv_bfloat162 pr; pr.x = f2b(ww[j][2 * hh]); pr.y = f2b(ww[j][2 * hh + 1]);
        *reinterpret_cast<__nv_bfloat162*>(sX2 + r * LDS + cc) = pr;
        uu[j][2 * hh] = b2f(f2b(uu[j][2 * hh])); uu[j][2 * hh + 1] = b2f(f2b(uu[j][2 * hh + 1]));
      }
    __syncthreads();
    // P = W h; v_new; Vg
    {
      float pp[4][4] = {};
      #pragma unroll
      for (int kk = 0; kk < D; kk += 16) {
        unsigned a[4]; fragA(a, sX2, LDS, 16 * mt, kk);
        #pragma unroll
        for (int j = 0; j < 4; j++) { unsigned bb[2]; fragB_kn(bb, sH, LDS, 32 * nq + 8 * j, kk); mma16816(pp[j], a[0], a[1], a[2], a[3], bb[0], bb[1]); }
      }
      const float gL = sG[L - 1];
      #pragma unroll
      for (int j = 0; j < 4; j++)
        #pragma unroll
        for (int hh = 0; hh < 2; hh++) {
          const int r = 16 * mt + gq + hh * 8, cc = 32 * nq + 8 * j + 2 * tq;
          const float v0 = uu[j][2 * hh] - pp[j][2 * hh], v1 = uu[j][2 * hh + 1] - pp[j][2 * hh + 1];
          __nv_bfloat162 pn; pn.x = f2b(v0); pn.y = f2b(v1);
          *reinterpret_cast<__nv_bfloat162*>(sX1 + r * LDS + cc) = pn;
          __nv_bfloat162 pg;
          if (r < L) { const float dg = ex2(gL - sG[r]); pg.x = f2b(v0 * dg); pg.y = f2b(v1 * dg); }
          else { pg.x = f2b(0.f); pg.y = f2b(0.f); }
          *reinterpret_cast<__nv_bfloat162*>(sVG + r * LDS + cc) = pg;
        }
    }
    // output-stage attention
    {
      float acc[2][4] = {};
      #pragma unroll
      for (int kk = 0; kk < D; kk += 16) {
        unsigned a[4]; fragA(a, sQ, LDS, 16 * mt, kk);
        #pragma unroll
        for (int j = 0; j < 2; j++) { unsigned bb[2]; fragB_nk(bb, sK, LDS, 16 * nq + 8 * j, kk); mma16816(acc[j], a[0], a[1], a[2], a[3], bb[0], bb[1]); }
      }
      #pragma unroll
      for (int j = 0; j < 2; j++)
        #pragma unroll
        for (int hh = 0; hh < 2; hh++) {
          const int r = 16 * mt + gq + hh * 8, s = 16 * nq + 8 * j + 2 * tq;
          __nv_bfloat162 pr;
          pr.x = (s <= r && r < L && s < L) ? f2b(acc[j][2 * hh] * ex2(sG[r] - sG[s])) : f2b(0.f);
          pr.y = (s + 1 <= r && r < L && s + 1 < L) ? f2b(acc[j][2 * hh + 1] * ex2(sG[r] - sG[s + 1])) : f2b(0.f);
          *reinterpret_cast<__nv_bfloat162*>(sA + r * LDA + s) = pr;
        }
    }
    __syncthreads();
    {
      float qh[4][4] = {}, pv[4][4] = {};
      #pragma unroll
      for (int kk = 0; kk < D; kk += 16) {
        unsigned a[4]; fragA(a, sQ, LDS, 16 * mt, kk);
        #pragma unroll
        for (int j = 0; j < 4; j++) { unsigned bb[2]; fragB_kn(bb, sH, LDS, 32 * nq + 8 * j, kk); mma16816(qh[j], a[0], a[1], a[2], a[3], bb[0], bb[1]); }
      }
      #pragma unroll
      for (int kk = 0; kk < BT; kk += 16) {
        unsigned a[4]; fragA(a, sA, LDA, 16 * mt, kk);
        #pragma unroll
        for (int j = 0; j < 4; j++) { unsigned bb[2]; fragB_kn(bb, sX1, LDS, 32 * nq + 8 * j, kk); mma16816(pv[j], a[0], a[1], a[2], a[3], bb[0], bb[1]); }
      }
      #pragma unroll
      for (int j = 0; j < 4; j++)
        #pragma unroll
        for (int hh = 0; hh < 2; hh++) {
          const int r = 16 * mt + gq + hh * 8, cc = 32 * nq + 8 * j + 2 * tq;
          if (r < L) {
            const float eg = ex2(sG[r]);
            __nv_bfloat162 ov;
            ov.x = f2b(fmaf(scale, eg * qh[j][2 * hh], scale * pv[j][2 * hh]));
            ov.y = f2b(fmaf(scale, eg * qh[j][2 * hh + 1], scale * pv[j][2 * hh + 1]));
            *reinterpret_cast<__nv_bfloat162*>(o + (((long)b * T + t0 + r) * HV + hv) * D + cc) = ov;
          }
        }
    }
    if (c + 1 < NT) {
      const float dL = ex2(sG[L - 1]);
      #pragma unroll
      for (int mi = 0; mi < 2; mi++)
        #pragma unroll
        for (int ni = 0; ni < 4; ni++)
          #pragma unroll
          for (int e = 0; e < 4; e++) st[mi][ni][e] = st[mi][ni][e] * dL;
      #pragma unroll
      for (int kk = 0; kk < BT; kk += 16) {
        #pragma unroll
        for (int mi = 0; mi < 2; mi++) {
          unsigned a[4]; fragA_T(a, sK, LDS, 32 * mt + 16 * mi, kk);
          #pragma unroll
          for (int ni = 0; ni < 4; ni++) { unsigned bb[2]; fragB_kn(bb, sVG, LDS, 32 * nq + 8 * ni, kk); mma16816(st[mi][ni], a[0], a[1], a[2], a[3], bb[0], bb[1]); }
        }
      }
    }
    __syncthreads();
  }
}
}  // namespace gdn

namespace gdn {
__device__ __forceinline__ void cp_async16(void* dst, const void* src, bool pred) {
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;" :: "r"(smem_u32(dst)), "l"(src), "r"(pred ? 16 : 0));
}
__device__ __forceinline__ void cp_async_commit() { asm volatile("cp.async.commit_group;"); }
__device__ __forceinline__ void cp_async_wait_all() { asm volatile("cp.async.wait_group 0;" ::: "memory"); }
__device__ __forceinline__ void cp_async_wait1() { asm volatile("cp.async.wait_group 1;" ::: "memory"); }

// C6 v4: v3 + double-buffered K/Q/V tiles filled by cp.async (next chunk's loads overlap this chunk's compute), next
// chunk's gate/beta prefetched into registers, fp32 inverse workspace aliased onto the bf16 state buffer.
// one (sequence b, value head hv) work item; also used by the persistent gdn_fused7_k
__device__ __forceinline__ void gdn4_item(const bf16* __restrict__ q, const bf16* __restrict__ k, const bf16* __restrict__ v,
                                              const float* __restrict__ g, const bf16* __restrict__ beta, bf16* __restrict__ o,
                                              int T, int HK, int HV, float scale, const int b, const int hv) {
  extern __shared__ __align__(16) unsigned char smem_raw[];
  bf16* sKb = reinterpret_cast<bf16*>(smem_raw);           // [2][BT][LDS]
  bf16* sQb = sKb + 2 * BT * LDS;                           // [2][BT][LDS]
  bf16* sVr = sQb + 2 * BT * LDS;                           // [2][BT][LDS] raw v
  bf16* sVG = sVr + 2 * BT * LDS;
  bf16* sX1 = sVG + BT * LDS;
  bf16* sX2 = sX1 + BT * LDS;
  bf16* sH = sX2 + BT * LDS;
  bf16* sA = sH + D * LDS;
  float* sAt = reinterpret_cast<float*>(sH);                 // aliased: only live before sH is written
  float* sInv = sAt + BT * (BT + 1);
  float* sTmp = reinterpret_cast<float*>(sA + BT * LDA);
  float* sG = sTmp + 3 * 16 * 17;
  float* sGr = sG + BT;
  float* sBt = sGr + BT;
  const int hk = hv / (HV / HK);
  const int tid = threadIdx.x, w = tid >> 5, l = tid & 31, gq = l >> 2, tq = l & 3;
  const int NT = (T + BT - 1) / BT;
  const int mt = w & 3, nq = w >> 2;
  auto issue_tiles = [&](int cc, int buf) {
    const int t0 = cc * BT, L = min(BT, T - t0);
    bf16* dK = sKb + buf * BT * LDS; bf16* dQ = sQb + buf * BT * LDS; bf16* dV = sVr + buf * BT * LDS;
    for (int idx = tid; idx < BT * 16; idx += NTH2) {
      const int r = idx >> 4, c8 = (idx & 15) * 8;
      const bool ok = r < L;
      const long tr = (long)b * T + t0 + (ok ? r : 0);
      cp_async16(dK + r * LDS + c8, k + (tr * HK + hk) * D + c8, ok);
      cp_async16(dQ + r * LDS + c8, q + (tr * HK + hk) * D + c8, ok);
      cp_async16(dV + r * LDS + c8, v + (tr * HV + hv) * D + c8, ok);
    }
    cp_async_commit();
  };
  float st[2][4][4];
  #pragma unroll
  for (int mi = 0; mi < 2; mi++)
    #pragma unroll
    for (int ni = 0; ni < 4; ni++)
      #pragma unroll
      for (int e = 0; e < 4; e++) st[mi][ni][e] = 0.f;
  float pgn = 0.f, pbn = 0.f;
  issue_tiles(0, 0);
  if (tid < BT) {
    const long tr = (long)b * T + tid;
    pgn = tid < min(BT, T) ? g[tr * HV + hv] : 0.f;
    pbn = tid < min(BT, T) ? b2f(beta[tr * HV + hv]) : 0.f;
  }
  for (int c = 0; c < NT; c++) {
    const int t0 = c * BT, L = min(BT, T - t0), cur = c & 1;
    const bf16* sK = sKb + cur * BT * LDS; const bf16* sQ = sQb + cur * BT * LDS; const bf16* sV = sVr + cur * BT * LDS;
    if (tid < BT) { sGr[tid] = pgn; sBt[tid] = pbn; }
    cp_async_wait_all();
    __syncthreads();
    if (c + 1 < NT) {                       // next chunk: tiles via cp.async, gate/beta into registers
      issue_tiles(c + 1, cur ^ 1);
      if (tid < BT) {
        const int L1 = min(BT, T - t0 - BT);
        const long tr = (long)b * T + t0 + BT + tid;
        pgn = tid < L1 ? g[tr * HV + hv] : 0.f;
        pbn = tid < L1 ? b2f(beta[tr * HV + hv]) : 0.f;
      }
    }
    if (w == 0) {
      float x0 = sGr[l], x1 = sGr[32 + l];
      #pragma unroll
      for (int d = 1; d < 32; d <<= 1) {
        const float y0 = __shfl_up_sync(0xffffffff, x0, d), y1 = __shfl_up_sync(0xffffffff, x1, d);
        if (l >= d) { x0 = x0 + y0; x1 = x1 + y1; }
      }
      const float tot = __shfl_sync(0xffffffff, x0, 31);
      x1 = x1 + tot;
      sG[l] = x0 * 1.4426950216f; sG[32 + l] = x1 * 1.4426950216f;
    }
    __syncthreads();
    // v*beta, k*beta*2^g from the smem tiles
    for (int idx = tid; idx < BT * 16; idx += NTH2) {
      const int r = idx >> 4, c8 = (idx & 15) * 8;
      const uint4 zk = *reinterpret_cast<const uint4*>(sK + r * LDS + c8), zv = *reinterpret_cast<const uint4*>(sV + r * LDS + c8);
      const float bt = b2f(f2b(sBt[r])), eg = ex2(sG[r]);
      const bf16* kb = reinterpret_cast<const bf16*>(&zk); const bf16* vb = reinterpret_cast<const bf16*>(&zv);
      uint4 ovb, okb; bf16* pvb = reinterpret_cast<bf16*>(&ovb); bf16* pkb = reinterpret_cast<bf16*>(&okb);
      #pragma unroll
      for (int j = 0; j < 8; j++) {
        pvb[j] = f2b(b2f(vb[j]) * bt);
        pkb[j] = f2b(b2f(f2b(b2f(kb[j]) * bt)) * eg);
      }
      *reinterpret_cast<uint4*>(sX1 + r * LDS + c8) = ovb;
      *reinterpret_cast<uint4*>(sX2 + r * LDS + c8) = okb;
    }
    {
      float acc[2][4] = {};
      #pragma unroll
      for (int kk = 0; kk < D; kk += 16) {
        unsigned a[4]; fragA(a, sK, LDS, 16 * mt, kk);
        #pragma unroll
        for (int j = 0; j < 2; j++) { unsigned bb[2]; fragB_nk(bb, sK, LDS, 16 * nq + 8 * j, kk); mma16816(acc[j], a[0], a[1], a[2], a[3], bb[0], bb[1]); }
      }
      #pragma unroll
      for (int j = 0; j < 2; j++)
        #pragma unroll
        for (int e = 0; e < 4; e++) {
          const int r = 16 * mt + gq + (e >> 1) * 8, s = 16 * nq + 8 * j + 2 * tq + (e & 1);
          float val = 0.f;
          if (r > s && r < L) val = (acc[j][e] * ex2(sG[r] - sG[s])) * sBt[r];
          sAt[r * (BT + 1) + s] = val;
        }
    }
    __syncthreads();
    if (w < 4 && l < 16) {
      const int base = 16 * w, j = l;
      // right-looking forward substitution: once x[m] is known every later row adds its term, so each row still sums
      // m = 0, 1, ... in order (same results as the row-by-row loop) but the 16-step chain is one FMA per step
      float x[16], sacc[16];
      #pragma unroll
      for (int i = 0; i < 16; i++) sacc[i] = 0.f;
      #pragma unroll
      for (int m = 0; m < 16; m++) {
        x[m] = (m < j) ? 0.f : ((m == j) ? 1.f : -sacc[m]);
        #pragma unroll
        for (int i = m + 1; i < 16; i++) sacc[i] += sAt[(base + i) * (BT + 1) + base + m] * x[m];
      }
      #pragma unroll
      for (int i = 0; i < 16; i++) sInv[(base + i) * (BT + 1) + base + j] = x[i];
    }
    __syncthreads();
    merge_blocks(sAt, sInv, reinterpret_cast<float*>(sVG));
    for (int e = tid; e < BT * BT / 2; e += NTH2) {
      const int r = e >> 5, s = (e & 31) * 2;
      __nv_bfloat162 pr;
      pr.x = s <= r ? f2b(sInv[r * (BT + 1) + s]) : f2b(0.f);
      pr.y = s + 1 <= r ? f2b(sInv[r * (BT + 1) + s + 1]) : f2b(0.f);
      *reinterpret_cast<__nv_bfloat162*>(sA + r * LDA + s) = pr;
    }
    __syncthreads();
    // bf16 state (sInv region is dead now)
    #pragma unroll
    for (int mi = 0; mi < 2; mi++)
      #pragma unroll
      for (int ni = 0; ni < 4; ni++)
        #pragma unroll
        for (int hh = 0; hh < 2; hh++) {
          const int r = 32 * mt + 16 * mi + gq + hh * 8, cc = 32 * nq + 8 * ni + 2 * tq;
          __nv_bfloat162 pr; pr.x = f2b(st[mi][ni][2 * hh]); pr.y = f2b(st[mi][ni][2 * hh + 1]);
          *reinterpret_cast<__nv_bfloat162*>(sH + r * LDS + cc) = pr;
        }
    // u = A (v*beta), w = A (k*beta*2^g)
    float uu[4][4] = {}, ww[4][4] = {};
    #pragma unroll
    for (int kk = 0; kk < BT; kk += 16) {
      unsigned a[4]; fragA(a, sA, LDA, 16 * mt, kk);
      #pragma unroll
      for (int j = 0; j < 4; j++) {
        unsigned bb[2];
        fragB_kn(bb, sX1, LDS, 32 * nq + 8 * j, kk); mma16816(uu[j], a[0], a[1], a[2], a[3], bb[0], bb[1]);
        fragB_kn(bb, sX2, LDS, 32 * nq + 8 * j, kk); mma16816(ww[j], a[0], a[1], a[2], a[3], bb[0], bb[1]);
      }
    }
    __syncthreads();
    #pragma unroll
    for (int j = 0; j < 4; j++)
      #pragma unroll
      for (int hh = 0; hh < 2; hh++) {
        const int r = 16 * mt + gq + hh * 8, cc = 32 * nq + 8 * j + 2 * tq;
        __nv_bfloat162 pr; pr.x = f2b(ww[j][2 * hh]); pr.y = f2b(ww[j][2 * hh + 1]);
        *reinterpret_cast<__nv_bfloat162*>(sX2 + r * LDS + cc) = pr;
        uu[j][2 * hh] = b2f(f2b(uu[j][2 * hh])); uu[j][2 * hh + 1] = b2f(f2b(uu[j][2 * hh + 1]));
      }
    __syncthreads();
    // P = W h; v_new; Vg
    {
      float pp[4][4] = {};
      #pragma unroll
      for (int kk = 0; kk < D; kk += 16) {
        unsigned a[4]; fragA(a, sX2, LDS, 16 * mt, kk);
        #pragma unroll
        for (int j = 0; j < 4; j++) { unsigned bb[2]; fragB_kn(bb, sH, LDS, 32 * nq + 8 * j, kk); mma16816(pp[j], a[0], a[1], a[2], a[3], bb[0], bb[1]); }
      }
      const float gL = sG[L - 1];
      #pragma unroll
      for (int j = 0; j < 4; j++)
        #pragma unroll
        for (int hh = 0; hh < 2; hh++) {
          const int r = 16 * mt + gq + hh * 8, cc = 32 * nq + 8 * j + 2 * tq;
          const float v0 = uu[j][2 * hh] - pp[j][2 * hh], v1 = uu[j][2 * hh + 1] - pp[j][2 * hh + 1];
          __nv_bfloat162 pn; pn.x = f2b(v0); pn.y = f2b(v1);
          *reinterpret_cast<__nv_bfloat162*>(sX1 + r * LDS + cc) = pn;
          __nv_bfloat162 pg;
          if (r < L) { const float dg = ex2(gL - sG[r]); pg.x = f2b(v0 * dg); pg.y = f2b(v1 * dg); }
          else { pg.x = f2b(0.f); pg.y = f2b(0.f); }
          *reinterpret_cast<__nv_bfloat162*>(sVG + r * LDS + cc) = pg;
        }
    }
    // output-stage attention
    {
      float acc[2][4] = {};
      #pragma unroll
      for (int kk = 0; kk < D; kk += 16) {
        unsigned a[4]; fragA(a, sQ, LDS, 16 * mt, kk);
        #pragma unroll
        for (int j = 0; j < 2; j++) { unsigned bb[2]; fragB_nk(bb, sK, LDS, 16 * nq + 8 * j, kk); mma16816(acc[j], a[0], a[1], a[2], a[3], bb[0], bb[1]); }
      }
      #pragma unroll
      for (int j = 0; j < 2; j++)
        #pragma unroll
        for (int hh = 0; hh < 2; hh++) {
          const int r = 16 * mt + gq + hh * 8, s = 16 * nq + 8 * j + 2 * tq;
          __nv_bfloat162 pr;
          pr.x = (s <= r && r < L && s < L) ? f2b(acc[j][2 * hh] * ex2(sG[r] - sG[s])) : f2b(0.f);
          pr.y = (s + 1 <= r && r < L && s + 1 < L) ? f2b(acc[j][2 * hh + 1] * ex2(sG[r] - sG[s + 1])) : f2b(0.f);
          *reinterpret_cast<__nv_bfloat162*>(sA + r * LDA + s) = pr;
        }
    }
    __syncthreads();
    {
      float qh[4][4] = {}, pv[4][4] = {};
      #pragma unroll
      for (int kk = 0; kk < D; kk += 16) {
        unsigned a[4]; fragA(a, sQ, LDS, 16 * mt, kk);
        #pragma unroll
        for (int j = 0; j < 4; j++) { unsigned bb[2]; fragB_kn(bb, sH, LDS, 32 * nq + 8 * j, kk); mma16816(qh[j], a[0], a[1], a[2], a[3], bb[0], bb[1]); }
      }
      #pragma unroll
      for (int kk = 0; kk < BT; kk += 16) {
        unsigned a[4]; fragA(a, sA, LDA, 16 * mt, kk);
        #pragma unroll
        for (int j = 0; j < 4; j++) { unsigned bb[2]; fragB_kn(bb, sX1, LDS, 32 * nq + 8 * j, kk); mma16816(pv[j], a[0], a[1], a[2], a[3], bb[0], bb[1]); }
      }
      #pragma unroll
      for (int j = 0; j < 4; j++)
        #pragma unroll
        for (int hh = 0; hh < 2; hh++) {
          const int r = 16 * mt + gq + hh * 8, cc = 32 * nq + 8 * j + 2 * tq;
          if (r < L) {
            const float eg = ex2(sG[r]);
            __nv_bfloat162 ov;
            ov.x = f2b(fmaf(scale, eg * qh[j][2 * hh], scale * pv[j][2 * hh]));
            ov.y = f2b(fmaf(scale, eg * qh[j][2 * hh + 1], scale * pv[j][2 * hh + 1]));
            *reinterpret_cast<__nv_bfloat162*>(o + (((long)b * T + t0 + r) * HV + hv) * D + cc) = ov;
          }
        }
    }
    if (c + 1 < NT) {
      const float dL = ex2(sG[L - 1]);
      #pragma unroll
      for (int mi = 0; mi < 2; mi++)
        #pragma unroll
        for (int ni = 0; ni < 4; ni++)
          #pragma unroll
          for (int e = 0; e < 4; e++) st[mi][ni][e] = st[mi][ni][e] * dL;
      #pragma unroll
      for (int kk = 0; kk < BT; kk += 16) {
        #pragma unroll
        for (int mi = 0; mi < 2; mi++) {
          unsigned a[4]; fragA_T(a, sK, LDS, 32 * mt + 16 * mi, kk);
          #pragma unroll
          for (int ni = 0; ni < 4; ni++) { unsigned bb[2]; fragB_kn(bb, sVG, LDS, 32 * nq + 8 * ni, kk); mma16816(st[mi][ni], a[0], a[1], a[2], a[3], bb[0], bb[1]); }
        }
      }
    }
    __syncthreads();
  }
}
// Round 4, C6g: gdn4_item with gdn6's warp specialization for any number of chunks: the (I+A~)^-1 solve runs on warps 0..3
// with named barriers while warps 4..15 scale v*beta / k*beta*2^g and compute the output-stage attention matrix into its
// own buffer (sQK), so that product leaves the state-dependent part. Same arithmetic as gdn4_item (identical output).
__device__ __forceinline__ void gdn8_item(const bf16* __restrict__ q, const bf16* __restrict__ k, const bf16* __restrict__ v,
                                              const float* __restrict__ g, const bf16* __restrict__ beta, bf16* __restrict__ o,
                                              int T, int HK, int HV, float scale, const int b, const int hv) {
  extern __shared__ __align__(16) unsigned char smem_raw[];
  bf16* sKb = reinterpret_cast<bf16*>(smem_raw);           // [2][BT][LDS]
  bf16* sQb = sKb + 2 * BT * LDS;                           // [2][BT][LDS]
  bf16* sVr = sQb + 2 * BT * LDS;                           // [2][BT][LDS] raw v
  bf16* sVG = sVr + 2 * BT * LDS;
  bf16* sX1 = sVG + BT * LDS;
  bf16* sX2 = sX1 + BT * LDS;
  bf16* sH = sX2 + BT * LDS;
  bf16* sA = sH + D * LDS;
  float* sAt = reinterpret_cast<float*>(sH);                 // aliased: only live before sH is written
  float* sInv = sAt + BT * (BT + 1);
  float* sTmp = reinterpret_cast<float*>(sA + BT * LDA);
  float* sG = sTmp + 3 * 16 * 17;
  float* sGr = sG + BT;
  float* sBt = sGr + BT;
  bf16* sQK = reinterpret_cast<bf16*>(sBt + BT);             // [BT][LDA] output-stage attention matrix of this chunk
  const int hk = hv / (HV / HK);
  const int tid = threadIdx.x, w = tid >> 5, l = tid & 31, gq = l >> 2, tq = l & 3;
  const int NT = (T + BT - 1) / BT;
  const int mt = w & 3, nq = w >> 2;
  auto issue_tiles = [&](int cc, int buf) {
    const int t0 = cc * BT, L = min(BT, T - t0);
    bf16* dK = sKb + buf * BT * LDS; bf16* dQ = sQb + buf * BT * LDS; bf16* dV = sVr + buf * BT * LDS;
    for (int idx = tid; idx < BT * 16; idx += NTH2) {
      const int r = idx >> 4, c8 = (idx & 15) * 8;
      const bool ok = r < L;
      const long tr = (long)b * T + t0 + (ok ? r : 0);
      cp_async16(dK + r * LDS + c8, k + (tr * HK + hk) * D + c8, ok);
      cp_async16(dQ + r * LDS + c8, q + (tr * HK + hk) * D + c8, ok);
      cp_async16(dV + r * LDS + c8, v + (tr * HV + hv) * D + c8, ok);
    }
    cp_async_commit();
  };
  float st[2][4][4];
  #pragma unroll
  for (int mi = 0; mi < 2; mi++)
    #pragma unroll
    for (int ni = 0; ni < 4; ni++)
      #pragma unroll
      for (int e = 0; e < 4; e++) st[mi][ni][e] = 0.f;
  float pgn = 0.f, pbn = 0.f;
  issue_tiles(0, 0);
  if (tid < BT) {
    const long tr = (long)b * T + tid;
    pgn = tid < min(BT, T) ? g[tr * HV + hv] : 0.f;
    pbn = tid < min(BT, T) ? b2f(beta[tr * HV + hv]) : 0.f;
  }
  for (int c = 0; c < NT; c++) {
    const int t0 = c * BT, L = min(BT, T - t0), cur = c & 1;
    const bf16* sK = sKb + cur * BT * LDS; const bf16* sQ = sQb + cur * BT * LDS; const bf16* sV = sVr + cur * BT * LDS;
    if (tid < BT) { sGr[tid] = pgn; sBt[tid] = pbn; }
    cp_async_wait_all();
    __syncthreads();
    if (c + 1 < NT) {                       // next chunk: tiles via cp.async, gate/beta into registers
      issue_tiles(c + 1, cur ^ 1);
      if (tid < BT) {
        const int L1 = min(BT, T - t0 - BT);
        const long tr = (long)b * T + t0 + BT + tid;
        pgn = tid < L1 ? g[tr * HV + hv] : 0.f;
        pbn = tid < L1 ? b2f(beta[tr * HV + hv]) : 0.f;
      }
    }
    if (w == 0) {
      float x0 = sGr[l], x1 = sGr[32 + l];
      #pragma unroll
      for (int d = 1; d < 32; d <<= 1) {
        const float y0 = __shfl_up_sync(0xffffffff, x0, d), y1 = __shfl_up_sync(0xffffffff, x1, d);
        if (l >= d) { x0 = x0 + y0; x1 = x1 + y1; }
      }
      const float tot = __shfl_sync(0xffffffff, x0, 31);
      x1 = x1 + tot;
      sG[l] = x0 * 1.4426950216f; sG[32 + l] = x1 * 1.4426950216f;
    }
    __syncthreads();
    {
      float acc[2][4] = {};
      #pragma unroll
      for (int kk = 0; kk < D; kk += 16) {
        unsigned a[4]; fragA(a, sK, LDS, 16 * mt, kk);
        #pragma unroll
        for (int j = 0; j < 2; j++) { unsigned bb[2]; fragB_nk(bb, sK, LDS, 16 * nq + 8 * j, kk); mma16816(acc[j], a[0], a[1], a[2], a[3], bb[0], bb[1]); }
      }
      #pragma unroll
      for (int j = 0; j < 2; j++)
        #pragma unroll
        for (int e = 0; e < 4; e++) {
          const int r = 16 * mt + gq + (e >> 1) * 8, s = 16 * nq + 8 * j + 2 * tq + (e & 1);
          float val = 0.f;
          if (r > s && r < L) val = (acc[j][e] * ex2(sG[r] - sG[s])) * sBt[r];
          sAt[r * (BT + 1) + s] = val;
        }
    }
    __syncthreads();
    if (w < 4) {                                            // (I + A~)^-1: warps 0..3, named barriers
      auto to_bf16 = [&](int bi_, int bj_) {
        for (int e = l; e < 128; e += 32) {
          const int r = 16 * bi_ + (e >> 3), c = 16 * bj_ + (e & 7) * 2;
          __nv_bfloat162 pr;
          pr.x = c <= r ? f2b(sInv[r * (BT + 1) + c]) : f2b(0.f);
          pr.y = c + 1 <= r ? f2b(sInv[r * (BT + 1) + c + 1]) : f2b(0.f);
          *reinterpret_cast<__nv_bfloat162*>(sA + r * LDA + c) = pr;
        }
      };
      if (l < 16) {
        const int base = 16 * w, j = l;
        float x[16], sacc[16];
        #pragma unroll
        for (int i = 0; i < 16; i++) sacc[i] = 0.f;
        #pragma unroll
        for (int m = 0; m < 16; m++) {
          x[m] = (m < j) ? 0.f : ((m == j) ? 1.f : -sacc[m]);
          #pragma unroll
          for (int i = m + 1; i < 16; i++) sacc[i] += sAt[(base + i) * (BT + 1) + base + m] * x[m];
        }
        #pragma unroll
        for (int i = 0; i < 16; i++) sInv[(base + i) * (BT + 1) + base + j] = x[i];
      }
      __syncwarp();
      to_bf16(w, w);
      asm volatile("bar.sync 1, 128;" ::: "memory");
      float* scr = reinterpret_cast<float*>(sVG);
      const int ld = BT + 1;
      if (w < 3) {
        const int i = w + 1;
        float t[2][4] = {}, r[2][4] = {};
        mm16_tf32(t, blk(sInv, i, i), blk(sAt, i, i - 1), ld);
        float* Tm = scr + w * 16 * ld; st16(Tm, ld, t, 1.f);
        __syncwarp();
        mm16_tf32(r, Tm, blk(sInv, i - 1, i - 1), ld);
        st16(blk(sInv, i, i - 1), ld, r, -1.f);
        __syncwarp();
        to_bf16(i, i - 1);
      }
      asm volatile("bar.sync 1, 128;" ::: "memory");
      if (w < 2) {
        const int i = w + 2, j = w;
        float t[2][4] = {}, r[2][4] = {};
        mm16_tf32(t, blk(sAt, i, j), blk(sInv, j, j), ld);
        mm16_tf32(t, blk(sAt, i, j + 1), blk(sInv, j + 1, j), ld);
        float* Tm = scr + w * 16 * ld; st16(Tm, ld, t, 1.f);
        __syncwarp();
        mm16_tf32(r, blk(sInv, i, i), Tm, ld);
        st16(blk(sInv, i, j), ld, r, -1.f);
        __syncwarp();
        to_bf16(i, j);
        asm volatile("bar.sync 2, 64;" ::: "memory");
        if (w == 0) {
          float t3[2][4] = {}, r3[2][4] = {};
          mm16_tf32(t3, blk(sAt, 3, 0), blk(sInv, 0, 0), ld);
          mm16_tf32(t3, blk(sAt, 3, 1), blk(sInv, 1, 0), ld);
          mm16_tf32(t3, blk(sAt, 3, 2), blk(sInv, 2, 0), ld);
          st16(scr, ld, t3, 1.f);
          __syncwarp();
          mm16_tf32(r3, blk(sInv, 3, 3), scr, ld);
          st16(blk(sInv, 3, 0), ld, r3, -1.f);
          __syncwarp();
          to_bf16(3, 0);
        }
      }
    } else {                                                // warps 4..15
    for (int idx = tid - 128; idx < BT * 16; idx += NTH2 - 128) {
        const int r = idx >> 4, c8 = (idx & 15) * 8;
        const uint4 zk = *reinterpret_cast<const uint4*>(sK + r * LDS + c8), zv = *reinterpret_cast<const uint4*>(sV + r * LDS + c8);
        const float bt = b2f(f2b(sBt[r])), eg = ex2(sG[r]);
        const bf16* kb = reinterpret_cast<const bf16*>(&zk); const bf16* vb = reinterpret_cast<const bf16*>(&zv);
        uint4 ovb, okb; bf16* pvb = reinterpret_cast<bf16*>(&ovb); bf16* pkb = reinterpret_cast<bf16*>(&okb);
        #pragma unroll
        for (int j = 0; j < 8; j++) {
          pvb[j] = f2b(b2f(vb[j]) * bt);
          pkb[j] = f2b(b2f(f2b(b2f(kb[j]) * bt)) * eg);
        }
        *reinterpret_cast<uint4*>(sX1 + r * LDS + c8) = ovb;
        *reinterpret_cast<uint4*>(sX2 + r * LDS + c8) = okb;
      }
      for (int e = tid - 128; e < 6 * 128; e += NTH2 - 128) {        // strictly upper 16x16 blocks of the inverse -> 0
        const int bk = e >> 7, r = (e & 127) >> 3, c = (e & 7) * 2;
        const int bi = bk < 3 ? 0 : (bk < 5 ? 1 : 2), bj = bk < 3 ? bk + 1 : (bk < 5 ? bk - 1 : 3);
        *reinterpret_cast<__nv_bfloat162*>(sA + (16 * bi + r) * LDA + 16 * bj + c) = __floats2bfloat162_rn(0.f, 0.f);
      }
      for (int tt = w - 4; tt < 16; tt += 12) {              // output-stage attention (q k^T, decay, causal mask) -> sQK
        const int tm = tt & 3, tn = tt >> 2;
        float acc[2][4] = {};
        #pragma unroll
        for (int kk = 0; kk < D; kk += 16) {
          unsigned a[4]; fragA(a, sQ, LDS, 16 * tm, kk);
          #pragma unroll
          for (int j = 0; j < 2; j++) { unsigned bb[2]; fragB_nk(bb, sK, LDS, 16 * tn + 8 * j, kk); mma16816(acc[j], a[0], a[1], a[2], a[3], bb[0], bb[1]); }
        }
        #pragma unroll
        for (int j = 0; j < 2; j++)
          #pragma unroll
          for (int hh = 0; hh < 2; hh++) {
            const int r = 16 * tm + gq + hh * 8, s_ = 16 * tn + 8 * j + 2 * tq;
            __nv_bfloat162 pr;
            pr.x = (s_ <= r && r < L && s_ < L) ? f2b(acc[j][2 * hh] * ex2(sG[r] - sG[s_])) : f2b(0.f);
            pr.y = (s_ + 1 <= r && r < L && s_ + 1 < L) ? f2b(acc[j][2 * hh + 1] * ex2(sG[r] - sG[s_ + 1])) : f2b(0.f);
            *reinterpret_cast<__nv_bfloat162*>(sQK + r * LDA + s_) = pr;
          }
      }
    }
    __syncthreads();
    // bf16 state (sInv region is dead now)
    #pragma unroll
    for (int mi = 0; mi < 2; mi++)
      #pragma unroll
      for (int ni = 0; ni < 4; ni++)
        #pragma unroll
        for (int hh = 0; hh < 2; hh++) {
          const int r = 32 * mt + 16 * mi + gq + hh * 8, cc = 32 * nq + 8 * ni + 2 * tq;
          __nv_bfloat162 pr; pr.x = f2b(st[mi][ni][2 * hh]); pr.y = f2b(st[mi][ni][2 * hh + 1]);
          *reinterpret_cast<__nv_bfloat162*>(sH + r * LDS + cc) = pr;
        }
    // u = A (v*beta), w = A (k*beta*2^g)
    float uu[4][4] = {}, ww[4][4] = {};
    #pragma unroll
    for (int kk = 0; kk < BT; kk += 16) {
      unsigned a[4]; fragA(a, sA, LDA, 16 * mt, kk);
      #pragma unroll
      for (int j = 0; j < 4; j++) {
        unsigned bb[2];
        fragB_kn(bb, sX1, LDS, 32 * nq + 8 * j, kk); mma16816(uu[j], a[0], a[1], a[2], a[3], bb[0], bb[1]);
        fragB_kn(bb, sX2, LDS, 32 * nq + 8 * j, kk); mma16816(ww[j], a[0], a[1], a[2], a[3], bb[0], bb[1]);
      }
    }
    __syncthreads();
    #pragma unroll
    for (int j = 0; j < 4; j++)
      #pragma unroll
      for (int hh = 0; hh < 2; hh++) {
        const int r = 16 * mt + gq + hh * 8, cc = 32 * nq + 8 * j + 2 * tq;
        __nv_bfloat162 pr; pr.x = f2b(ww[j][2 * hh]); pr.y = f2b(ww[j][2 * hh + 1]);
        *reinterpret_cast<__nv_bfloat162*>(sX2 + r * LDS + cc) = pr;
        uu[j][2 * hh] = b2f(f2b(uu[j][2 * hh])); uu[j][2 * hh + 1] = b2f(f2b(uu[j][2 * hh + 1]));
      }
    __syncthreads();
    // P = W h; v_new; Vg
    {
      float pp[4][4] = {};
      #pragma unroll
      for (int kk = 0; kk < D; kk += 16) {
        unsigned a[4]; fragA(a, sX2, LDS, 16 * mt, kk);
        #pragma unroll
        for (int j = 0; j < 4; j++) { unsigned bb[2]; fragB_kn(bb, sH, LDS, 32 * nq + 8 * j, kk); mma16816(pp[j], a[0], a[1], a[2], a[3], bb[0], bb[1]); }
      }
      const float gL = sG[L - 1];
      #pragma unroll
      for (int j = 0; j < 4; j++)
        #pragma unroll
        for (int hh = 0; hh < 2; hh++) {
          const int r = 16 * mt + gq + hh * 8, cc = 32 * nq + 8 * j + 2 * tq;
          const float v0 = uu[j][2 * hh] - pp[j][2 * hh], v1 = uu[j][2 * hh + 1] - pp[j][2 * hh + 1];
          __nv_bfloat162 pn; pn.x = f2b(v0); pn.y = f2b(v1);
          *reinterpret_cast<__nv_bfloat162*>(sX1 + r * LDS + cc) = pn;
          __nv_bfloat162 pg;
          if (r < L) { const float dg = ex2(gL - sG[r]); pg.x = f2b(v0 * dg); pg.y = f2b(v1 * dg); }
          else { pg.x = f2b(0.f); pg.y = f2b(0.f); }
          *reinterpret_cast<__nv_bfloat162*>(sVG + r * LDS + cc) = pg;
        }
    }
    __syncthreads();
    {
      float qh[4][4] = {}, pv[4][4] = {};
      #pragma unroll
      for (int kk = 0; kk < D; kk += 16) {
        unsigned a[4]; fragA(a, sQ, LDS, 16 * mt, kk);
        #pragma unroll
        for (int j = 0; j < 4; j++) { unsigned bb[2]; fragB_kn(bb, sH, LDS, 32 * nq + 8 * j, kk); mma16816(qh[j], a[0], a[1], a[2], a[3], bb[0], bb[1]); }
      }
      #pragma unroll
      for (int kk = 0; kk < BT; kk += 16) {
        unsigned a[4]; fragA(a, sQK, LDA, 16 * mt, kk);
        #pragma unroll
        for (int j = 0; j < 4; j++) { unsigned bb[2]; fragB_kn(bb, sX1, LDS, 32 * nq + 8 * j, kk); mma16816(pv[j], a[0], a[1], a[2], a[3], bb[0], bb[1]); }
      }
      #pragma unroll
      for (int j = 0; j < 4; j++)
        #pragma unroll
        for (int hh = 0; hh < 2; hh++) {
          const int r = 16 * mt + gq + hh * 8, cc = 32 * nq + 8 * j + 2 * tq;
          if (r < L) {
            const float eg = ex2(sG[r]);
            __nv_bfloat162 ov;
            ov.x = f2b(fmaf(scale, eg * qh[j][2 * hh], scale * pv[j][2 * hh]));
            ov.y = f2b(fmaf(scale, eg * qh[j][2 * hh + 1], scale * pv[j][2 * hh + 1]));
            *reinterpret_cast<__nv_bfloat162*>(o + (((long)b * T + t0 + r) * HV + hv) * D + cc) = ov;
          }
        }
    }
    if (c + 1 < NT) {
      const float dL = ex2(sG[L - 1]);
      #pragma unroll
      for (int mi = 0; mi < 2; mi++)
        #pragma unroll
        for (int ni = 0; ni < 4; ni++)
          #pragma unroll
          for (int e = 0; e < 4; e++) st[mi][ni][e] = st[mi][ni][e] * dL;
      #pragma unroll
      for (int kk = 0; kk < BT; kk += 16) {
        #pragma unroll
        for (int mi = 0; mi < 2; mi++) {
          unsigned a[4]; fragA_T(a, sK, LDS, 32 * mt + 16 * mi, kk);
          #pragma unroll
          for (int ni = 0; ni < 4; ni++) { unsigned bb[2]; fragB_kn(bb, sVG, LDS, 32 * nq + 8 * ni, kk); mma16816(st[mi][ni], a[0], a[1], a[2], a[3], bb[0], bb[1]); }
        }
      }
    }
    __syncthreads();
  }
}
// Round 4, C6h: long paths as two kernels. gdnL_k: every chunk of every (sequence, value head) at once (grid HV x B x NT):
// gate cumsum, kkt, (I+A~)^-1, u = A v*beta, w = A k*beta*2^g and the output-stage attention matrix, written to global
// scratch (bf16 w, u, A_qk; fp32 cumsum). gdnD_k: one CTA per (sequence, value head) walks the chunks and does only the
// state-dependent part (P = W h, v_new, o, state update) with the next chunk's tiles in flight. Same arithmetic and
// cast points as gdn4_item (identical output); the state-independent work no longer sits on the sequential path.
__global__ void __launch_bounds__(NTH2, 1) gdnL_k(const bf16* __restrict__ q, const bf16* __restrict__ k, const bf16* __restrict__ v,
                                                  const float* __restrict__ g, const bf16* __restrict__ beta, int T, int HK, int HV,
                                                  bf16* __restrict__ Wg, bf16* __restrict__ Ug, bf16* __restrict__ QKg, float* __restrict__ Gg) {
  extern __shared__ __align__(16) unsigned char smem_raw[];
  bf16* sK = reinterpret_cast<bf16*>(smem_raw);
  bf16* sQ = sK + BT * LDS;
  bf16* sV = sQ + BT * LDS;
  bf16* sX1 = sV + BT * LDS;
  bf16* sX2 = sX1 + BT * LDS;
  bf16* sA = sX2 + BT * LDS;                                // [BT][LDA]
  bf16* sQK = sA + BT * LDA;                                // [BT][LDA]
  float* sAt = reinterpret_cast<float*>(sQK + BT * LDA);    // [BT][BT+1]
  float* sInv = sAt + BT * (BT + 1);
  float* scr = sInv + BT * (BT + 1);                        // merge scratch 3 x 16 x (BT+1)
  float* sG = scr + 3 * 16 * (BT + 1);
  float* sGr = sG + BT;
  float* sBt = sGr + BT;
  const int b = blockIdx.y, hv = blockIdx.x, c = blockIdx.z, hk = hv / (HV / HK);
  const int NT = gridDim.z;
  const int tid = threadIdx.x, w = tid >> 5, l = tid & 31, gq = l >> 2, tq = l & 3;
  const int mt = w & 3, nq = w >> 2;
  const int t0 = c * BT, L = min(BT, T - t0);
  const long item = ((long)b * HV + hv) * NT + c;
  for (int idx = tid; idx < BT * 16; idx += NTH2) {
    const int r = idx >> 4, c8 = (idx & 15) * 8;
    const bool ok = r < L;
    const long tr = (long)b * T + t0 + (ok ? r : 0);
    cp_async16(sK + r * LDS + c8, k + (tr * HK + hk) * D + c8, ok);
    cp_async16(sQ + r * LDS + c8, q + (tr * HK + hk) * D + c8, ok);
    cp_async16(sV + r * LDS + c8, v + (tr * HV + hv) * D + c8, ok);
  }
  cp_async_commit();
  if (tid < BT) {
    const long tr = (long)b * T + t0 + tid;
    sGr[tid] = tid < L ? g[tr * HV + hv] : 0.f;
    sBt[tid] = tid < L ? b2f(beta[tr * HV + hv]) : 0.f;
  }
  cp_async_wait_all();
  __syncthreads();
  if (w == 0) {
    float x0 = sGr[l], x1 = sGr[32 + l];
    #pragma unroll
    for (int d = 1; d < 32; d <<= 1) {
      const float y0 = __shfl_up_sync(0xffffffff, x0, d), y1 = __shfl_up_sync(0xffffffff, x1, d);
      if (l >= d) { x0 = x0 + y0; x1 = x1 + y1; }
    }
    const float tot = __shfl_sync(0xffffffff, x0, 31);
    x1 = x1 + tot;
    sG[l] = x0 * 1.4426950216f; sG[32 + l] = x1 * 1.4426950216f;
  }
  __syncthreads();
  for (int idx = tid; idx < BT * 16; idx += NTH2) {
    const int r = idx >> 4, c8 = (idx & 15) * 8;
    const uint4 zk = *reinterpret_cast<const uint4*>(sK + r * LDS + c8), zv = *reinterpret_cast<const uint4*>(sV + r * LDS + c8);
    const float bt = b2f(f2b(sBt[r])), eg = ex2(sG[r]);
    const bf16* kb = reinterpret_cast<const bf16*>(&zk); const bf16* vb = reinterpret_cast<const bf16*>(&zv);
    uint4 ovb, okb; bf16* pvb = reinterpret_cast<bf16*>(&ovb); bf16* pkb = reinterpret_cast<bf16*>(&okb);
    #pragma unroll
    for (int j = 0; j < 8; j++) {
      pvb[j] = f2b(b2f(vb[j]) * bt);
      pkb[j] = f2b(b2f(f2b(b2f(kb[j]) * bt)) * eg);
    }
    *reinterpret_cast<uint4*>(sX1 + r * LDS + c8) = ovb;
    *reinterpret_cast<uint4*>(sX2 + r * LDS + c8) = okb;
  }
  {
    float acc[2][4] = {}, aq[2][4] = {};
    #pragma unroll
    for (int kk = 0; kk < D; kk += 16) {
      unsigned a[4], a2[4]; fragA(a, sK, LDS, 16 * mt, kk); fragA(a2, sQ, LDS, 16 * mt, kk);
      #pragma unroll
      for (int j = 0; j < 2; j++) {
        unsigned bb[2]; fragB_nk(bb, sK, LDS, 16 * nq + 8 * j, kk);
        mma16816(acc[j], a[0], a[1], a[2], a[3], bb[0], bb[1]);
        mma16816(aq[j], a2[0], a2[1], a2[2], a2[3], bb[0], bb[1]);
      }
    }
    #pragma unroll
    for (int j = 0; j < 2; j++)
      #pragma unroll
      for (int e = 0; e < 4; e++) {
        const int r = 16 * mt + gq + (e >> 1) * 8, s = 16 * nq + 8 * j + 2 * tq + (e & 1);
        float val = 0.f;
        if (r > s && r < L) val = (acc[j][e] * ex2(sG[r] - sG[s])) * sBt[r];
        sAt[r * (BT + 1) + s] = val;
      }
    #pragma unroll
    for (int j = 0; j < 2; j++)
      #pragma unroll
      for (int hh = 0; hh < 2; hh++) {
        const int r = 16 * mt + gq + hh * 8, s = 16 * nq + 8 * j + 2 * tq;
        __nv_bfloat162 pr;
        pr.x = (s <= r && r < L && s < L) ? f2b(aq[j][2 * hh] * ex2(sG[r] - sG[s])) : f2b(0.f);
        pr.y = (s + 1 <= r && r < L && s + 1 < L) ? f2b(aq[j][2 * hh + 1] * ex2(sG[r] - sG[s + 1])) : f2b(0.f);
        *reinterpret_cast<__nv_bfloat162*>(sQK + r * LDA + s) = pr;
      }
  }
  __syncthreads();
  if (w < 4 && l < 16) {
    const int base = 16 * w, j = l;
    float x[16], sacc[16];
    #pragma unroll
    for (int i = 0; i < 16; i++) sacc[i] = 0.f;
    #pragma unroll
    for (int m = 0; m < 16; m++) {
      x[m] = (m < j) ? 0.f : ((m == j) ? 1.f : -sacc[m]);
      #pragma unroll
      for (int i = m + 1; i < 16; i++) sacc[i] += sAt[(base + i) * (BT + 1) + base + m] * x[m];
    }
    #pragma unroll
    for (int i = 0; i < 16; i++) sInv[(base + i) * (BT + 1) + base + j] = x[i];
  }
  __syncthreads();
  merge_blocks(sAt, sInv, scr);
  for (int e = tid; e < BT * BT / 2; e += NTH2) {
    const int r = e >> 5, s = (e & 31) * 2;
    __nv_bfloat162 pr;
    pr.x = s <= r ? f2b(sInv[r * (BT + 1) + s]) : f2b(0.f);
    pr.y = s + 1 <= r ? f2b(sInv[r * (BT + 1) + s + 1]) : f2b(0.f);
    *reinterpret_cast<__nv_bfloat162*>(sA + r * LDA + s) = pr;
  }
  __syncthreads();
  float uu[4][4] = {}, ww[4][4] = {};
  #pragma unroll
  for (int kk = 0; kk < BT; kk += 16) {
    unsigned a[4]; fragA(a, sA, LDA, 16 * mt, kk);
    #pragma unroll
    for (int j = 0; j < 4; j++) {
      unsigned bb[2];
      fragB_kn(bb, sX1, LDS, 32 * nq + 8 * j, kk); mma16816(uu[j], a[0], a[1], a[2], a[3], bb[0], bb[1]);
      fragB_kn(bb, sX2, LDS, 32 * nq + 8 * j, kk); mma16816(ww[j], a[0], a[1], a[2], a[3], bb[0], bb[1]);
    }
  }
  bf16* Wo = Wg + item * BT * D; bf16* Uo = Ug + item * BT * D;
  #pragma unroll
  for (int j = 0; j < 4; j++)
    #pragma unroll
    for (int hh = 0; hh < 2; hh++) {
      const int r = 16 * mt + gq + hh * 8, cc = 32 * nq + 8 * j + 2 * tq;
      __nv_bfloat162 pw; pw.x = f2b(ww[j][2 * hh]); pw.y = f2b(ww[j][2 * hh + 1]);
      __nv_bfloat162 pu; pu.x = f2b(uu[j][2 * hh]); pu.y = f2b(uu[j][2 * hh + 1]);
      *reinterpret_cast<__nv_bfloat162*>(Wo + r * D + cc) = pw;
      *reinterpret_cast<__nv_bfloat162*>(Uo + r * D + cc) = pu;
    }
  {
    const int r = tid >> 3, c8 = (tid & 7) * 8;           // 64 x 64 bf16 = 512 x 16 B
    *reinterpret_cast<uint4*>(QKg + item * BT * BT + r * BT + c8) = *reinterpret_cast<const uint4*>(sQK + r * LDA + c8);
  }
  if (tid < BT) Gg[item * BT + tid] = sG[tid];
}
__global__ void __launch_bounds__(NTH2, 1) gdnD_k(const bf16* __restrict__ q, const bf16* __restrict__ k,
                                                  const bf16* __restrict__ Wg, const bf16* __restrict__ Ug, const bf16* __restrict__ QKg,
                                                  const float* __restrict__ Gg, bf16* __restrict__ o, int T, int HK, int HV, float scale) {
  extern __shared__ __align__(16) unsigned char smem_raw[];
  bf16* sKb = reinterpret_cast<bf16*>(smem_raw);           // [2][BT][LDS]
  bf16* sQb = sKb + 2 * BT * LDS;
  bf16* sWb = sQb + 2 * BT * LDS;
  bf16* sQKb = sWb + 2 * BT * LDS;                          // [2][BT][LDA]
  bf16* sH = sQKb + 2 * BT * LDA;                           // [D][LDS]
  bf16* sX1 = sH + D * LDS;
  bf16* sVG = sX1 + BT * LDS;
  float* sGb = reinterpret_cast<float*>(sVG + BT * LDS);    // [2][BT]
  const int b = blockIdx.y, hv = blockIdx.x, hk = hv / (HV / HK);
  const int tid = threadIdx.x, w = tid >> 5, l = tid & 31, gq = l >> 2, tq = l & 3;
  const int mt = w & 3, nq = w >> 2;
  const int NT = (T + BT - 1) / BT;
  const long item0 = ((long)b * HV + hv) * NT;
  auto issue = [&](int cc, int buf) {
    const int t0 = cc * BT, L = min(BT, T - t0);
    bf16* dK = sKb + buf * BT * LDS; bf16* dQ = sQb + buf * BT * LDS; bf16* dW = sWb + buf * BT * LDS; bf16* dA = sQKb + buf * BT * LDA;
    const bf16* Wi = Wg + (item0 + cc) * BT * D; const bf16* Ai = QKg + (item0 + cc) * BT * BT;
    for (int idx = tid; idx < BT * 16; idx += NTH2) {
      const int r = idx >> 4, c8 = (idx & 15) * 8;
      const bool ok = r < L;
      const long tr = (long)b * T + t0 + (ok ? r : 0);
      cp_async16(dK + r * LDS + c8, k + (tr * HK + hk) * D + c8, ok);
      cp_async16(dQ + r * LDS + c8, q + (tr * HK + hk) * D + c8, ok);
      cp_async16(dW + r * LDS + c8, Wi + r * D + c8, true);
    }
    { const int r = tid >> 3, c8 = (tid & 7) * 8; cp_async16(dA + r * LDA + c8, Ai + r * BT + c8, true); }
    if (tid < BT / 4) cp_async16(sGb + buf * BT + tid * 4, Gg + (item0 + cc) * BT + tid * 4, true);
    cp_async_commit();
  };
  for (int idx = tid; idx < D * (D / 8); idx += NTH2)
    *reinterpret_cast<uint4*>(sH + (idx / (D / 8)) * LDS + (idx % (D / 8)) * 8) = make_uint4(0u, 0u, 0u, 0u);
  float st[2][4][4];
  #pragma unroll
  for (int mi = 0; mi < 2; mi++)
    #pragma unroll
    for (int ni = 0; ni < 4; ni++)
      #pragma unroll
      for (int e = 0; e < 4; e++) st[mi][ni][e] = 0.f;
  issue(0, 0);
  for (int c = 0; c < NT; c++) {
    const int t0 = c * BT, L = min(BT, T - t0), cur = c & 1;
    const bf16* sK = sKb + cur * BT * LDS; const bf16* sQ = sQb + cur * BT * LDS; const bf16* sW = sWb + cur * BT * LDS;
    const bf16* sQK = sQKb + cur * BT * LDA; const float* sG = sGb + cur * BT;
    if (c + 1 < NT) { issue(c + 1, cur ^ 1); cp_async_wait1(); } else cp_async_wait_all();
    __syncthreads();
    const bf16* Ui = Ug + (item0 + c) * BT * D;
    float uu[4][4];
    #pragma unroll
    for (int j = 0; j < 4; j++)
      #pragma unroll
      for (int hh = 0; hh < 2; hh++) {
        const __nv_bfloat162 u2 = *reinterpret_cast<const __nv_bfloat162*>(Ui + (16 * mt + gq + hh * 8) * D + 32 * nq + 8 * j + 2 * tq);
        uu[j][2 * hh] = b2f(u2.x); uu[j][2 * hh + 1] = b2f(u2.y);
      }
    float qh[4][4] = {};
    {
      float pp[4][4] = {};
      #pragma unroll
      for (int kk = 0; kk < D; kk += 16) {
        unsigned a[4], aq[4]; fragA(a, sW, LDS, 16 * mt, kk); fragA(aq, sQ, LDS, 16 * mt, kk);
        #pragma unroll
        for (int j = 0; j < 4; j++) {
          unsigned bb[2]; fragB_kn(bb, sH, LDS, 32 * nq + 8 * j, kk);
          mma16816(pp[j], a[0], a[1], a[2], a[3], bb[0], bb[1]);
          mma16816(qh[j], aq[0], aq[1], aq[2], aq[3], bb[0], bb[1]);
        }
      }
      const float gL = sG[L - 1];
      #pragma unroll
      for (int j = 0; j < 4; j++)
        #pragma unroll
        for (int hh = 0; hh < 2; hh++) {
          const int r = 16 * mt + gq + hh * 8, cc = 32 * nq + 8 * j + 2 * tq;
          const float v0 = uu[j][2 * hh] - pp[j][2 * hh], v1 = uu[j][2 * hh + 1] - pp[j][2 * hh + 1];
          __nv_bfloat162 pn; pn.x = f2b(v0); pn.y = f2b(v1);
          *reinterpret_cast<__nv_bfloat162*>(sX1 + r * LDS + cc) = pn;
          __nv_bfloat162 pg;
          if (r < L) { const float dg = ex2(gL - sG[r]); pg.x = f2b(v0 * dg); pg.y = f2b(v1 * dg); }
          else { pg.x = f2b(0.f); pg.y = f2b(0.f); }
          *reinterpret_cast<__nv_bfloat162*>(sVG + r * LDS + cc) = pg;
        }
    }
    __syncthreads();
    {
      float pv[4][4] = {};
      #pragma unroll
      for (int kk = 0; kk < BT; kk += 16) {
        unsigned a[4]; fragA(a, sQK, LDA, 16 * mt, kk);
        #pragma unroll
        for (int j = 0; j < 4; j++) { unsigned bb[2]; fragB_kn(bb, sX1, LDS, 32 * nq + 8 * j, kk); mma16816(pv[j], a[0], a[1], a[2], a[3], bb[0], bb[1]); }
      }
      #pragma unroll
      for (int j = 0; j < 4; j++)
        #pragma unroll
        for (int hh = 0; hh < 2; hh++) {
          const int r = 16 * mt + gq + hh * 8, cc = 32 * nq + 8 * j + 2 * tq;
          if (r < L) {
            const float eg = ex2(sG[r]);
            __nv_bfloat162 ov;
            ov.x = f2b(fmaf(scale, eg * qh[j][2 * hh], scale * pv[j][2 * hh]));
            ov.y = f2b(fmaf(scale, eg * qh[j][2 * hh + 1], scale * pv[j][2 * hh + 1]));
            *reinterpret_cast<__nv_bfloat162*>(o + (((long)b * T + t0 + r) * HV + hv) * D + cc) = ov;
          }
        }
    }
    if (c + 1 < NT) {
      const float dL = ex2(sG[L - 1]);
      #pragma unroll
      for (int mi = 0; mi < 2; mi++)
        #pragma unroll
        for (int ni = 0; ni < 4; ni++)
          #pragma unroll
          for (int e = 0; e < 4; e++) st[mi][ni][e] = st[mi][ni][e] * dL;
      #pragma unroll
      for (int kk = 0; kk < BT; kk += 16) {
        #pragma unroll
        for (int mi = 0; mi < 2; mi++) {
          unsigned a[4]; fragA_T(a, sK, LDS, 32 * mt + 16 * mi, kk);
          #pragma unroll
          for (int ni = 0; ni < 4; ni++) { unsigned bb[2]; fragB_kn(bb, sVG, LDS, 32 * nq + 8 * ni, kk); mma16816(st[mi][ni], a[0], a[1], a[2], a[3], bb[0], bb[1]); }
        }
      }
      #pragma unroll
      for (int mi = 0; mi < 2; mi++)
        #pragma unroll
        for (int ni = 0; ni < 4; ni++)
          #pragma unroll
          for (int hh = 0; hh < 2; hh++) {
            const int r = 32 * mt + 16 * mi + gq + hh * 8, cc = 32 * nq + 8 * ni + 2 * tq;
            __nv_bfloat162 pr; pr.x = f2b(st[mi][ni][2 * hh]); pr.y = f2b(st[mi][ni][2 * hh + 1]);
            *reinterpret_cast<__nv_bfloat162*>(sH + r * LDS + cc) = pr;
          }
    }
    __syncthreads();
  }
}
__global__ void __launch_bounds__(NTH2, 1) gdn_fused8_k(const bf16* __restrict__ q, const bf16* __restrict__ k, const bf16* __restrict__ v,
                                                         const float* __restrict__ g, const bf16* __restrict__ beta, bf16* __restrict__ o,
                                                         int T, int HK, int HV, float scale) {
  gdn8_item(q, k, v, g, beta, o, T, HK, HV, scale, blockIdx.y, blockIdx.x);
}
__global__ void __launch_bounds__(NTH2, 1) gdn_fused4_k(const bf16* __restrict__ q, const bf16* __restrict__ k, const bf16* __restrict__ v,
                                                         const float* __restrict__ g, const bf16* __restrict__ beta, bf16* __restrict__ o,
                                                         int T, int HK, int HV, float scale) {
  gdn4_item(q, k, v, g, beta, o, T, HK, HV, scale, blockIdx.y, blockIdx.x);
}
__global__ void __launch_bounds__(NTH2, 1) gdn_fused5_k(const bf16* __restrict__ q, const bf16* __restrict__ k, const bf16* __restrict__ v,
                                                         const float* __restrict__ g, const bf16* __restrict__ beta,
                                                         int T, int HK, int HV, float scale,
                                                         const bf16* __restrict__ proj, int P, int zoff, const bf16* __restrict__ normw,
                                                         const float* __restrict__ sig, const int* __restrict__ canon,
                                                         const int* __restrict__ rowmask, int N, float neps, bf16* __restrict__ y) {
  extern __shared__ __align__(16) unsigned char smem_raw[];
  bf16* sKb = reinterpret_cast<bf16*>(smem_raw);           // [2][BT][LDS]
  bf16* sQb = sKb + 2 * BT * LDS;                           // [2][BT][LDS]
  bf16* sVr = sQb + 2 * BT * LDS;                           // [2][BT][LDS] raw v
  bf16* sVG = sVr + 2 * BT * LDS;
  bf16* sX1 = sVG + BT * LDS;
  bf16* sX2 = sX1 + BT * LDS;
  bf16* sH = sX2 + BT * LDS;
  bf16* sA = sH + D * LDS;
  float* sAt = reinterpret_cast<float*>(sH);                 // aliased: only live before sH is written
  float* sInv = sAt + BT * (BT + 1);
  float* sTmp = reinterpret_cast<float*>(sA + BT * LDA);
  float* sG = sTmp + 3 * 16 * 17;
  float* sGr = sG + BT;
  float* sBt = sGr + BT;
  bf16* sZ = reinterpret_cast<bf16*>(sBt + BT);               // [BT][LDS] output gate z of this chunk (by canonical row)
  int* sCn = reinterpret_cast<int*>(sZ + BT * LDS);           // [BT] packed row of each slot of this chunk, -1 if not canonical
  const int b = blockIdx.y, hv = blockIdx.x, hk = hv / (HV / HK);
  const int tid = threadIdx.x, w = tid >> 5, l = tid & 31, gq = l >> 2, tq = l & 3;
  const int NT = (T + BT - 1) / BT;
  const int mt = w & 3, nq = w >> 2;
  auto issue_tiles = [&](int cc, int buf) {
    const int t0 = cc * BT, L = min(BT, T - t0);
    bf16* dK = sKb + buf * BT * LDS; bf16* dQ = sQb + buf * BT * LDS; bf16* dV = sVr + buf * BT * LDS;
    for (int idx = tid; idx < BT * 16; idx += NTH2) {
      const int r = idx >> 4, c8 = (idx & 15) * 8;
      const bool ok = r < L;
      const long tr = (long)b * T + t0 + (ok ? r : 0);
      cp_async16(dK + r * LDS + c8, k + (tr * HK + hk) * D + c8, ok);
      cp_async16(dQ + r * LDS + c8, q + (tr * HK + hk) * D + c8, ok);
      cp_async16(dV + r * LDS + c8, v + (tr * HV + hv) * D + c8, ok);
    }
    cp_async_commit();
  };
  float st[2][4][4];
  #pragma unroll
  for (int mi = 0; mi < 2; mi++)
    #pragma unroll
    for (int ni = 0; ni < 4; ni++)
      #pragma unroll
      for (int e = 0; e < 4; e++) st[mi][ni][e] = 0.f;
  // z rows of chunk cc: thread handles slots tid>>4 and 32 + (tid>>4); cn0/cn1 = their packed rows (loaded a chunk ahead)
  auto ld_cn = [&](int cc, int& a0, int& a1) {
    const int t0 = cc * BT, L = min(BT, T - t0), r0 = tid >> 4;
    a0 = r0 < L ? __ldg(canon + (long)b * T + t0 + r0) : -1;
    a1 = r0 + 32 < L ? __ldg(canon + (long)b * T + t0 + r0 + 32) : -1;
  };
  int cn0, cn1;
  float pgn = 0.f, pbn = 0.f;
  ld_cn(0, cn0, cn1);
  issue_tiles(0, 0);
  {   // padded packed rows get zeros (they feed the out_proj GEMM and attention as masked keys: must be finite)
    const int per = (N + gridDim.y - 1) / gridDim.y, n0 = b * per, n1 = min(N, n0 + per);
    for (int idx = tid; idx < (n1 - n0) * (D / 8); idx += NTH2) {
      const int n = n0 + idx / (D / 8), c8 = (idx % (D / 8)) * 8;
      if (__ldg(rowmask + n) == 0) *reinterpret_cast<uint4*>(y + (long)n * (HV * D) + hv * D + c8) = make_uint4(0u, 0u, 0u, 0u);
    }
  }
  if (tid < BT) {
    const long tr = (long)b * T + tid;
    pgn = tid < min(BT, T) ? g[tr * HV + hv] : 0.f;
    pbn = tid < min(BT, T) ? b2f(beta[tr * HV + hv]) : 0.f;
  }
  for (int c = 0; c < NT; c++) {
    const int t0 = c * BT, L = min(BT, T - t0), cur = c & 1;
    const bf16* sK = sKb + cur * BT * LDS; const bf16* sQ = sQb + cur * BT * LDS; const bf16* sV = sVr + cur * BT * LDS;
    if (tid < BT) { sGr[tid] = pgn; sBt[tid] = pbn; }
    cp_async_wait_all();
    __syncthreads();
    {   // this chunk's z rows (own cp.async group, committed before the next chunk's tiles so wait_group 1 covers it)
      if ((tid & 15) == 0) { sCn[tid >> 4] = cn0; sCn[32 + (tid >> 4)] = cn1; }
      const int c8 = (tid & 15) * 8;
      cp_async16(sZ + (tid >> 4) * LDS + c8, proj + (long)(cn0 >= 0 ? cn0 : 0) * P + zoff + hv * D + c8, cn0 >= 0);
      cp_async16(sZ + (32 + (tid >> 4)) * LDS + c8, proj + (long)(cn1 >= 0 ? cn1 : 0) * P + zoff + hv * D + c8, cn1 >= 0);
      cp_async_commit();
      if (c + 1 < NT) ld_cn(c + 1, cn0, cn1);
    }
    if (c + 1 < NT) {                       // next chunk: tiles via cp.async, gate/beta into registers
      issue_tiles(c + 1, cur ^ 1);
      if (tid < BT) {
        const int L1 = min(BT, T - t0 - BT);
        const long tr = (long)b * T + t0 + BT + tid;
        pgn = tid < L1 ? g[tr * HV + hv] : 0.f;
        pbn = tid < L1 ? b2f(beta[tr * HV + hv]) : 0.f;
      }
    }
    if (w == 0) {
      float x0 = sGr[l], x1 = sGr[32 + l];
      #pragma unroll
      for (int d = 1; d < 32; d <<= 1) {
        const float y0 = __shfl_up_sync(0xffffffff, x0, d), y1 = __shfl_up_sync(0xffffffff, x1, d);
        if (l >= d) { x0 = x0 + y0; x1 = x1 + y1; }
      }
      const float tot = __shfl_sync(0xffffffff, x0, 31);
      x1 = x1 + tot;
      sG[l] = x0 * 1.4426950216f; sG[32 + l] = x1 * 1.4426950216f;
    }
    __syncthreads();
    // v*beta, k*beta*2^g from the smem tiles
    for (int idx = tid; idx < BT * 16; idx += NTH2) {
      const int r = idx >> 4, c8 = (idx & 15) * 8;
      const uint4 zk = *reinterpret_cast<const uint4*>(sK + r * LDS + c8), zv = *reinterpret_cast<const uint4*>(sV + r * LDS + c8);
      const float bt = b2f(f2b(sBt[r])), eg = ex2(sG[r]);
      const bf16* kb = reinterpret_cast<const bf16*>(&zk); const bf16* vb = reinterpret_cast<const bf16*>(&zv);
      uint4 ovb, okb; bf16* pvb = reinterpret_cast<bf16*>(&ovb); bf16* pkb = reinterpret_cast<bf16*>(&okb);
      #pragma unroll
      for (int j = 0; j < 8; j++) {
        pvb[j] = f2b(b2f(vb[j]) * bt);
        pkb[j] = f2b(b2f(f2b(b2f(kb[j]) * bt)) * eg);
      }
      *reinterpret_cast<uint4*>(sX1 + r * LDS + c8) = ovb;
      *reinterpret_cast<uint4*>(sX2 + r * LDS + c8) = okb;
    }
    {
      float acc[2][4] = {};
      #pragma unroll
      for (int kk = 0; kk < D; kk += 16) {
        unsigned a[4]; fragA(a, sK, LDS, 16 * mt, kk);
        #pragma unroll
        for (int j = 0; j < 2; j++) { unsigned bb[2]; fragB_nk(bb, sK, LDS, 16 * nq + 8 * j, kk); mma16816(acc[j], a[0], a[1], a[2], a[3], bb[0], bb[1]); }
      }
      #pragma unroll
      for (int j = 0; j < 2; j++)
        #pragma unroll
        for (int e = 0; e < 4; e++) {
          const int r = 16 * mt + gq + (e >> 1) * 8, s = 16 * nq + 8 * j + 2 * tq + (e & 1);
          float val = 0.f;
          if (r > s && r < L) val = (acc[j][e] * ex2(sG[r] - sG[s])) * sBt[r];
          sAt[r * (BT + 1) + s] = val;
        }
    }
    __syncthreads();
    if (w < 4 && l < 16) {
      const int base = 16 * w, j = l;
      float x[16];
      #pragma unroll
      for (int i = 0; i < 16; i++) {
        float sacc = 0.f;
        #pragma unroll
        for (int m = 0; m < i; m++) sacc += sAt[(base + i) * (BT + 1) + base + m] * x[m];
        const float val = (i < j) ? 0.f : ((i == j) ? 1.f : -sacc);
        x[i] = val;
        sInv[(base + i) * (BT + 1) + base + j] = val;
      }
    }
    __syncthreads();
    merge_blocks(sAt, sInv, reinterpret_cast<float*>(sVG));
    for (int e = tid; e < BT * BT / 2; e += NTH2) {
      const int r = e >> 5, s = (e & 31) * 2;
      __nv_bfloat162 pr;
      pr.x = s <= r ? f2b(sInv[r * (BT + 1) + s]) : f2b(0.f);
      pr.y = s + 1 <= r ? f2b(sInv[r * (BT + 1) + s + 1]) : f2b(0.f);
      *reinterpret_cast<__nv_bfloat162*>(sA + r * LDA + s) = pr;
    }
    __syncthreads();
    // bf16 state (sInv region is dead now)
    #pragma unroll
    for (int mi = 0; mi < 2; mi++)
      #pragma unroll
      for (int ni = 0; ni < 4; ni++)
        #pragma unroll
        for (int hh = 0; hh < 2; hh++) {
          const int r = 32 * mt + 16 * mi + gq + hh * 8, cc = 32 * nq + 8 * ni + 2 * tq;
          __nv_bfloat162 pr; pr.x = f2b(st[mi][ni][2 * hh]); pr.y = f2b(st[mi][ni][2 * hh + 1]);
          *reinterpret_cast<__nv_bfloat162*>(sH + r * LDS + cc) = pr;
        }
    // u = A (v*beta), w = A (k*beta*2^g)
    float uu[4][4] = {}, ww[4][4] = {};
    #pragma unroll
    for (int kk = 0; kk < BT; kk += 16) {
      unsigned a[4]; fragA(a, sA, LDA, 16 * mt, kk);
      #pragma unroll
      for (int j = 0; j < 4; j++) {
        unsigned bb[2];
        fragB_kn(bb, sX1, LDS, 32 * nq + 8 * j, kk); mma16816(uu[j], a[0], a[1], a[2], a[3], bb[0], bb[1]);
        fragB_kn(bb, sX2, LDS, 32 * nq + 8 * j, kk); mma16816(ww[j], a[0], a[1], a[2], a[3], bb[0], bb[1]);
      }
    }
    __syncthreads();
    #pragma unroll
    for (int j = 0; j < 4; j++)
      #pragma unroll
      for (int hh = 0; hh < 2; hh++) {
        const int r = 16 * mt + gq + hh * 8, cc = 32 * nq + 8 * j + 2 * tq;
        __nv_bfloat162 pr; pr.x = f2b(ww[j][2 * hh]); pr.y = f2b(ww[j][2 * hh + 1]);
        *reinterpret_cast<__nv_bfloat162*>(sX2 + r * LDS + cc) = pr;
        uu[j][2 * hh] = b2f(f2b(uu[j][2 * hh])); uu[j][2 * hh + 1] = b2f(f2b(uu[j][2 * hh + 1]));
      }
    __syncthreads();
    // P = W h; v_new; Vg
    {
      float pp[4][4] = {};
      #pragma unroll
      for (int kk = 0; kk < D; kk += 16) {
        unsigned a[4]; fragA(a, sX2, LDS, 16 * mt, kk);
        #pragma unroll
        for (int j = 0; j < 4; j++) { unsigned bb[2]; fragB_kn(bb, sH, LDS, 32 * nq + 8 * j, kk); mma16816(pp[j], a[0], a[1], a[2], a[3], bb[0], bb[1]); }
      }
      const float gL = sG[L - 1];
      #pragma unroll
      for (int j = 0; j < 4; j++)
        #pragma unroll
        for (int hh = 0; hh < 2; hh++) {
          const int r = 16 * mt + gq + hh * 8, cc = 32 * nq + 8 * j + 2 * tq;
          const float v0 = uu[j][2 * hh] - pp[j][2 * hh], v1 = uu[j][2 * hh + 1] - pp[j][2 * hh + 1];
          __nv_bfloat162 pn; pn.x = f2b(v0); pn.y = f2b(v1);
          *reinterpret_cast<__nv_bfloat162*>(sX1 + r * LDS + cc) = pn;
          __nv_bfloat162 pg;
          if (r < L) { const float dg = ex2(gL - sG[r]); pg.x = f2b(v0 * dg); pg.y = f2b(v1 * dg); }
          else { pg.x = f2b(0.f); pg.y = f2b(0.f); }
          *reinterpret_cast<__nv_bfloat162*>(sVG + r * LDS + cc) = pg;
        }
    }
    // output-stage attention
    {
      float acc[2][4] = {};
      #pragma unroll
      for (int kk = 0; kk < D; kk += 16) {
        unsigned a[4]; fragA(a, sQ, LDS, 16 * mt, kk);
        #pragma unroll
        for (int j = 0; j < 2; j++) { unsigned bb[2]; fragB_nk(bb, sK, LDS, 16 * nq + 8 * j, kk); mma16816(acc[j], a[0], a[1], a[2], a[3], bb[0], bb[1]); }
      }
      #pragma unroll
      for (int j = 0; j < 2; j++)
        #pragma unroll
        for (int hh = 0; hh < 2; hh++) {
          const int r = 16 * mt + gq + hh * 8, s = 16 * nq + 8 * j + 2 * tq;
          __nv_bfloat162 pr;
          pr.x = (s <= r && r < L && s < L) ? f2b(acc[j][2 * hh] * ex2(sG[r] - sG[s])) : f2b(0.f);
          pr.y = (s + 1 <= r && r < L && s + 1 < L) ? f2b(acc[j][2 * hh + 1] * ex2(sG[r] - sG[s + 1])) : f2b(0.f);
          *reinterpret_cast<__nv_bfloat162*>(sA + r * LDA + s) = pr;
        }
    }
    if (c + 1 < NT) cp_async_wait1(); else cp_async_wait_all();
    __syncthreads();
    {
      float qh[4][4] = {}, pv[4][4] = {};
      #pragma unroll
      for (int kk = 0; kk < D; kk += 16) {
        unsigned a[4]; fragA(a, sQ, LDS, 16 * mt, kk);
        #pragma unroll
        for (int j = 0; j < 4; j++) { unsigned bb[2]; fragB_kn(bb, sH, LDS, 32 * nq + 8 * j, kk); mma16816(qh[j], a[0], a[1], a[2], a[3], bb[0], bb[1]); }
      }
      #pragma unroll
      for (int kk = 0; kk < BT; kk += 16) {
        unsigned a[4]; fragA(a, sA, LDA, 16 * mt, kk);
        #pragma unroll
        for (int j = 0; j < 4; j++) { unsigned bb[2]; fragB_kn(bb, sX1, LDS, 32 * nq + 8 * j, kk); mma16816(pv[j], a[0], a[1], a[2], a[3], bb[0], bb[1]); }
      }
      // fused gated RMSNorm (FLA FusedRMSNormGated, swish) on the bf16 GDN output, written straight to the packed row
      float ov[4][4], ssq[2] = {0.f, 0.f};
      #pragma unroll
      for (int j = 0; j < 4; j++)
        #pragma unroll
        for (int e = 0; e < 4; e++) {
          const int r = 16 * mt + gq + (e >> 1) * 8;
          const float val = b2f(f2b(fmaf(scale, ex2(sG[r]) * qh[j][e], scale * pv[j][e])));
          ov[j][e] = val; ssq[e >> 1] += val * val;
        }
      #pragma unroll
      for (int hh = 0; hh < 2; hh++) {
        ssq[hh] += __shfl_xor_sync(0xffffffff, ssq[hh], 1);
        ssq[hh] += __shfl_xor_sync(0xffffffff, ssq[hh], 2);
        if (tq == 0) sTmp[(16 * mt + gq + hh * 8) * 4 + nq] = ssq[hh];
      }
      __syncthreads();
      #pragma unroll
      for (int hh = 0; hh < 2; hh++) {
        const int r = 16 * mt + gq + hh * 8;
        const int n = sCn[r];
        if (n >= 0) {
          const float ss = sTmp[r * 4 + 0] + sTmp[r * 4 + 1] + sTmp[r * 4 + 2] + sTmp[r * 4 + 3];
          const float rstd = 1.0f / sqrtf(ss / (float)D + neps);
          #pragma unroll
          for (int j = 0; j < 4; j++) {
            const int cc = 32 * nq + 8 * j + 2 * tq;
            const __nv_bfloat162 zz = *reinterpret_cast<const __nv_bfloat162*>(sZ + r * LDS + cc);
            const __nv_bfloat162 wv = __ldg(reinterpret_cast<const __nv_bfloat162*>(normw + cc));
            const float z0 = b2f(zz.x), z1 = b2f(zz.y);
            __nv_bfloat162 out;   // sigmoid_f inline: identical to the table entry
            out.x = f2b(((ov[j][2 * hh] * rstd) * b2f(wv.x)) * z0 * sigmoid_f(z0));
            out.y = f2b(((ov[j][2 * hh + 1] * rstd) * b2f(wv.y)) * z1 * sigmoid_f(z1));
            *reinterpret_cast<__nv_bfloat162*>(y + (long)n * (HV * D) + hv * D + cc) = out;
          }
        }
      }
    }
    if (c + 1 < NT) {
      const float dL = ex2(sG[L - 1]);
      #pragma unroll
      for (int mi = 0; mi < 2; mi++)
        #pragma unroll
        for (int ni = 0; ni < 4; ni++)
          #pragma unroll
          for (int e = 0; e < 4; e++) st[mi][ni][e] = st[mi][ni][e] * dL;
      #pragma unroll
      for (int kk = 0; kk < BT; kk += 16) {
        #pragma unroll
        for (int mi = 0; mi < 2; mi++) {
          unsigned a[4]; fragA_T(a, sK, LDS, 32 * mt + 16 * mi, kk);
          #pragma unroll
          for (int ni = 0; ni < 4; ni++) { unsigned bb[2]; fragB_kn(bb, sVG, LDS, 32 * nq + 8 * ni, kk); mma16816(st[mi][ni], a[0], a[1], a[2], a[3], bb[0], bb[1]); }
        }
      }
    }
    __syncthreads();
  }
}
// Round 4, C6d: paths of 65..96 rows (chunk 0 = 64 rows, chunk 1 = 1..32 rows). Every step that does not depend on the
// recurrent state (gate cumsum, k*beta / v*beta, kkt, the (I+A~)^-1 solve, u/w, the output-stage attention matrix) runs
// for both chunks in the same barrier interval (the solve's serial 1-4 warp chains of the two chunks run side by side);
// only state passing is sequential: chunk 0 has h = 0 (v_new0 = u0, o0 = A_qk v_new0), h1 = K0^T (v_new0 * 2^(gL-g)),
// then v_new1 = u1 - w1 h1 and o1. 11 barriers instead of 24; same arithmetic and cast points as gdn_fused4_k
// (skipped MMA k-steps only ever added exact zeros), so the output is identical.
// one (sequence b, value head hv) work item; also used by the persistent gdn_fused7_k
template <bool TS = false>
__device__ __forceinline__ void gdn6_item(const bf16* __restrict__ q, const bf16* __restrict__ k, const bf16* __restrict__ v,
                                              const float* __restrict__ g, const bf16* __restrict__ beta, bf16* __restrict__ o,
                                              int T, int HK, int HV, float scale, const int b, const int hv,
                                              unsigned long long* ts = nullptr) {
  auto stamp = [&](int kk) { if constexpr (TS) { if (threadIdx.x == 0) { unsigned long long t_; asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t_)); ts[(b * HV + hv) * 16 + kk] = t_; } } };
  stamp(0);
  constexpr int R = 96, L1D = 33;
  extern __shared__ __align__(16) unsigned char smem_raw[];
  bf16* sK = reinterpret_cast<bf16*>(smem_raw);             // [R][LDS]
  bf16* sQ = sK + R * LDS;
  bf16* sV = sQ + R * LDS;
  bf16* sX1 = sV + R * LDS;                                 // v*beta -> u (bf16) -> v_new
  bf16* sX2 = sX1 + R * LDS;                                // k*beta*2^g -> w (bf16)
  bf16* sVG = sX2 + R * LDS;                                // chunk-0 merge scratch (fp32), then v_new0 * 2^(gL-g)
  bf16* sH = sVG + R * LDS;                                 // [D][LDS] bf16 h1; fp32 A~ / inverse of chunk 0 before that
  bf16* sA = sH + D * LDS;                                  // [R][LDA] inverse (bf16), then the output-stage attention matrix
  float* sAt0 = reinterpret_cast<float*>(sH);
  float* sInv0 = sAt0 + BT * (BT + 1);
  float* sAt1 = reinterpret_cast<float*>(sA + R * LDA);     // [32][L1D] chunk-1 A~ / inverse / merge scratch (fp32)
  float* sInv1 = sAt1 + 32 * L1D;
  float* sScr1 = sInv1 + 32 * L1D;
  float* sG = sScr1 + 16 * L1D;                             // [R] per-chunk cumulative gate * log2(e)
  float* sBt = sG + R;                                      // [R] beta
  bf16* sQK = reinterpret_cast<bf16*>(sBt + R);              // [R][LDA] output-stage attention matrix (computed with kkt)
  const int hk = hv / (HV / HK);
  const int tid = threadIdx.x, w = tid >> 5, l = tid & 31, gq = l >> 2, tq = l & 3;
  const int mt = w & 3, nq = w >> 2;                        // chunk 0: 16-row x 32-col (16x16 for square products) warp tiles
  const int m1 = w & 1, n1 = w >> 1;                        // chunk 1: 16-row x 16-col warp tiles
  const int L1 = T - BT;
  const long rb = (long)b * T;
  for (int idx = tid; idx < R * 16; idx += NTH2) {
    const int r = idx >> 4, c8 = (idx & 15) * 8;
    const bool ok = r < T;
    const long tr = rb + (ok ? r : 0);
    cp_async16(sK + r * LDS + c8, k + (tr * HK + hk) * D + c8, ok);
    cp_async16(sQ + r * LDS + c8, q + (tr * HK + hk) * D + c8, ok);
    cp_async16(sV + r * LDS + c8, v + (tr * HV + hv) * D + c8, ok);
  }
  cp_async_commit();
  if (tid < R) sBt[tid] = tid < T ? b2f(beta[(rb + tid) * HV + hv]) : 0.f;
  if (w == 0) {                                             // chunk-0 cumsum (as gdn_fused4_k)
    float x0 = g[(rb + l) * HV + hv], x1 = g[(rb + 32 + l) * HV + hv];
    #pragma unroll
    for (int d = 1; d < 32; d <<= 1) {
      const float y0 = __shfl_up_sync(0xffffffff, x0, d), y1 = __shfl_up_sync(0xffffffff, x1, d);
      if (l >= d) { x0 = x0 + y0; x1 = x1 + y1; }
    }
    const float tot = __shfl_sync(0xffffffff, x0, 31);
    x1 = x1 + tot;
    sG[l] = x0 * 1.4426950216f; sG[32 + l] = x1 * 1.4426950216f;
  } else if (w == 1) {                                      // chunk-1 cumsum (rows 64..95; rows >= T add 0)
    float x0 = BT + l < T ? g[(rb + BT + l) * HV + hv] : 0.f, x1 = 0.f;
    #pragma unroll
    for (int d = 1; d < 32; d <<= 1) {
      const float y0 = __shfl_up_sync(0xffffffff, x0, d), y1 = __shfl_up_sync(0xffffffff, x1, d);
      if (l >= d) { x0 = x0 + y0; x1 = x1 + y1; }
    }
    sG[BT + l] = x0 * 1.4426950216f;
  }
  cp_async_wait_all();
  __syncthreads(); stamp(1);                                                                                            // B1
  // kkt of both chunks (the v*beta / k*beta*2^g scaling moved off the critical path, next to the solve)
  {
    float acc[2][4] = {};
    #pragma unroll
    for (int kk = 0; kk < D; kk += 16) {
      unsigned a[4]; fragA(a, sK, LDS, 16 * mt, kk);
      { unsigned bb[4]; fragB_nk2(bb, sK, LDS, 16 * nq, kk); mma16816(acc[0], a[0], a[1], a[2], a[3], bb[0], bb[1]); mma16816(acc[1], a[0], a[1], a[2], a[3], bb[2], bb[3]); }
    }
    #pragma unroll
    for (int j = 0; j < 2; j++)
      #pragma unroll
      for (int e = 0; e < 4; e++) {
        const int r = 16 * mt + gq + (e >> 1) * 8, s = 16 * nq + 8 * j + 2 * tq + (e & 1);
        float val = 0.f;
        if (r > s) val = (acc[j][e] * ex2(sG[r] - sG[s])) * sBt[r];
        sAt0[r * (BT + 1) + s] = val;
      }
    if (w < 4) {                                            // chunk 1: 32x32, warps 0..3 (16x16 each)
      const bf16* K1 = sK + BT * LDS;
      float ac1[2][4] = {};
      #pragma unroll
      for (int kk = 0; kk < D; kk += 16) {
        unsigned a[4]; fragA(a, K1, LDS, 16 * m1, kk);
        { unsigned bb[4]; fragB_nk2(bb, K1, LDS, 16 * (w >> 1), kk); mma16816(ac1[0], a[0], a[1], a[2], a[3], bb[0], bb[1]); mma16816(ac1[1], a[0], a[1], a[2], a[3], bb[2], bb[3]); }
      }
      #pragma unroll
      for (int j = 0; j < 2; j++)
        #pragma unroll
        for (int e = 0; e < 4; e++) {
          const int r = 16 * m1 + gq + (e >> 1) * 8, s = 16 * (w >> 1) + 8 * j + 2 * tq + (e & 1);
          float val = 0.f;
          if (r > s && r < L1) val = (ac1[j][e] * ex2(sG[BT + r] - sG[BT + s])) * sBt[BT + r];
          sAt1[r * L1D + s] = val;
        }
    }
  }
  __syncthreads(); stamp(2);                                                                                            // B2
  // (I + A~)^-1 on warps 0..5 with named barriers (only the warps of each level wait); blocks go to bf16 as soon as
  // they are final. Warps 6..15 meanwhile scale v*beta, k*beta*2^g and zero the upper blocks of sA.
  if (w < 6) {
    const bool c1 = w >= 4;
    const int bi = c1 ? w - 4 : w, ld = c1 ? L1D : BT + 1;
    const float* At = c1 ? sAt1 : sAt0; float* Iv = c1 ? sInv1 : sInv0;
    bf16* dA = sA + (c1 ? BT : 0) * LDA;
    if (l < 16) {
      const int base = 16 * bi, j = l;
      float x[16], sacc[16];                          // right-looking forward substitution (see gdn4_item)
      #pragma unroll
      for (int i = 0; i < 16; i++) sacc[i] = 0.f;
      #pragma unroll
      for (int m = 0; m < 16; m++) {
        x[m] = (m < j) ? 0.f : ((m == j) ? 1.f : -sacc[m]);
        #pragma unroll
        for (int i = m + 1; i < 16; i++) sacc[i] += At[(base + i) * ld + base + m] * x[m];
      }
      #pragma unroll
      for (int i = 0; i < 16; i++) Iv[(base + i) * ld + base + j] = x[i];
    }
    __syncwarp();
    // bf16 copy of block (bi, bj) of Iv (lower triangle incl. diagonal inside diagonal blocks)
    auto to_bf16 = [&](const float* M, int ldm, bf16* dst, int bi_, int bj_) {
      for (int e = l; e < 128; e += 32) {
        const int r = 16 * bi_ + (e >> 3), c = 16 * bj_ + (e & 7) * 2;
        __nv_bfloat162 pr;
        pr.x = c <= r ? f2b(M[r * ldm + c]) : f2b(0.f);
        pr.y = c + 1 <= r ? f2b(M[r * ldm + c + 1]) : f2b(0.f);
        *reinterpret_cast<__nv_bfloat162*>(dst + r * LDA + c) = pr;
      }
    };
    to_bf16(Iv, ld, dA, bi, bi);
    if (w == 0) stamp(8);
    asm volatile("bar.sync 1, 192;" ::: "memory");
    if (w == 0) stamp(9);
    if (w < 4) {
      float* scr = reinterpret_cast<float*>(sVG);
      const int ldm = BT + 1;
      if (w < 3) {                                    // level 1, chunk 0: D_{i,i-1} = -(D_i A~_{i,i-1}) D_{i-1}
        const int i = w + 1;
        float t[2][4] = {}, r[2][4] = {};
        mm16_tf32(t, blk(sInv0, i, i), blk(sAt0, i, i - 1), ldm);
        float* Tm = scr + w * 16 * ldm; st16(Tm, ldm, t, 1.f);
        __syncwarp();
        mm16_tf32(r, Tm, blk(sInv0, i - 1, i - 1), ldm);
        st16(blk(sInv0, i, i - 1), ldm, r, -1.f);
        __syncwarp();
        to_bf16(sInv0, ldm, sA, i, i - 1);
      } else {                                        // level 1, chunk 1: D10
        float t[2][4] = {}, r[2][4] = {};
        mm16_tf32(t, sInv1 + 16 * L1D + 16, sAt1 + 16 * L1D, L1D);
        st16(sScr1, L1D, t, 1.f);
        __syncwarp();
        mm16_tf32(r, sScr1, sInv1, L1D);
        st16(sInv1 + 16 * L1D, L1D, r, -1.f);
        __syncwarp();
        to_bf16(sInv1, L1D, sA + BT * LDA, 1, 0);
      }
      if (w == 0) stamp(10);
      asm volatile("bar.sync 2, 128;" ::: "memory");
      if (w < 2) {                                    // level 2: D20, D31
        const int i = w + 2, j = w;
        float t[2][4] = {}, r[2][4] = {};
        mm16_tf32(t, blk(sAt0, i, j), blk(sInv0, j, j), ldm);
        mm16_tf32(t, blk(sAt0, i, j + 1), blk(sInv0, j + 1, j), ldm);
        float* Tm = scr + w * 16 * ldm; st16(Tm, ldm, t, 1.f);
        __syncwarp();
        mm16_tf32(r, blk(sInv0, i, i), Tm, ldm);
        st16(blk(sInv0, i, j), ldm, r, -1.f);
        __syncwarp();
        to_bf16(sInv0, ldm, sA, i, j);
        if (w == 0) stamp(11);
        asm volatile("bar.sync 3, 64;" ::: "memory");
        if (w == 0) {                                 // level 3: D30
          float t3[2][4] = {}, r3[2][4] = {};
          mm16_tf32(t3, blk(sAt0, 3, 0), blk(sInv0, 0, 0), ldm);
          mm16_tf32(t3, blk(sAt0, 3, 1), blk(sInv0, 1, 0), ldm);
          mm16_tf32(t3, blk(sAt0, 3, 2), blk(sInv0, 2, 0), ldm);
          st16(scr, ldm, t3, 1.f);
          __syncwarp();
          mm16_tf32(r3, blk(sInv0, 3, 3), scr, ldm);
          st16(blk(sInv0, 3, 0), ldm, r3, -1.f);
          __syncwarp();
          to_bf16(sInv0, ldm, sA, 3, 0);
          stamp(12);
        }
      }
    }
  } else {
    for (int idx = tid - 192; idx < R * 16; idx += NTH2 - 192) {
      const int r = idx >> 4, c8 = (idx & 15) * 8;
      const uint4 zk = *reinterpret_cast<const uint4*>(sK + r * LDS + c8), zv = *reinterpret_cast<const uint4*>(sV + r * LDS + c8);
      const float bt = b2f(f2b(sBt[r])), eg = ex2(sG[r]);
      const bf16* kb = reinterpret_cast<const bf16*>(&zk); const bf16* vb = reinterpret_cast<const bf16*>(&zv);
      uint4 ovb, okb; bf16* pvb = reinterpret_cast<bf16*>(&ovb); bf16* pkb = reinterpret_cast<bf16*>(&okb);
      #pragma unroll
      for (int j = 0; j < 8; j++) {
        pvb[j] = f2b(b2f(vb[j]) * bt);
        pkb[j] = f2b(b2f(f2b(b2f(kb[j]) * bt)) * eg);
      }
      *reinterpret_cast<uint4*>(sX1 + r * LDS + c8) = ovb;
      *reinterpret_cast<uint4*>(sX2 + r * LDS + c8) = okb;
    }
    for (int e = tid - 192; e < 7 * 128; e += NTH2 - 192) {      // strictly upper 16x16 blocks of both inverses -> 0
      const int bk = e >> 7, r = (e & 127) >> 3, c = (e & 7) * 2;
      const int bi = bk < 6 ? (bk < 3 ? 0 : (bk < 5 ? 1 : 2)) : 0, bj = bk < 6 ? (bk < 3 ? bk + 1 : (bk < 5 ? bk - 1 : 3)) : 1;
      bf16* dst = sA + (bk < 6 ? 0 : BT) * LDA;
      *reinterpret_cast<__nv_bfloat162*>(dst + (16 * bi + r) * LDA + 16 * bj + c) = __floats2bfloat162_rn(0.f, 0.f);
    }
    // output-stage attention matrices of both chunks (state-independent): 16 + 4 warp tiles of 16x16 on warps 6..15,
    // hidden under the solve's serial chain
    for (int tt = w - 6; tt < 20; tt += 10) {
      const bool c1 = tt >= 16;
      const int tm = c1 ? (tt - 16) & 1 : tt & 3, tn = c1 ? (tt - 16) >> 1 : tt >> 2, r0 = c1 ? BT : 0;
      const bf16* Qc = sQ + r0 * LDS; const bf16* Kc = sK + r0 * LDS;
      const int Lc = c1 ? L1 : BT;
      float acc[2][4] = {};
      #pragma unroll
      for (int kk = 0; kk < D; kk += 16) {
        unsigned a[4]; fragA(a, Qc, LDS, 16 * tm, kk);
        { unsigned bb[4]; fragB_nk2(bb, Kc, LDS, 16 * tn, kk); mma16816(acc[0], a[0], a[1], a[2], a[3], bb[0], bb[1]); mma16816(acc[1], a[0], a[1], a[2], a[3], bb[2], bb[3]); }
      }
      #pragma unroll
      for (int j = 0; j < 2; j++)
        #pragma unroll
        for (int hh = 0; hh < 2; hh++) {
          const int r = 16 * tm + gq + hh * 8, s_ = 16 * tn + 8 * j + 2 * tq;
          __nv_bfloat162 pr;
          pr.x = (s_ <= r && r < Lc && s_ < Lc) ? f2b(acc[j][2 * hh] * ex2(sG[r0 + r] - sG[r0 + s_])) : f2b(0.f);
          pr.y = (s_ + 1 <= r && r < Lc && s_ + 1 < Lc) ? f2b(acc[j][2 * hh + 1] * ex2(sG[r0 + r] - sG[r0 + s_ + 1])) : f2b(0.f);
          *reinterpret_cast<__nv_bfloat162*>(sQK + (r0 + r) * LDA + s_) = pr;
        }
    }
    if (tid == 192) { if constexpr (TS) { unsigned long long t_; asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t_)); ts[(b * HV + hv) * 16 + 13] = t_; } }
  }
  __syncthreads(); stamp(3);                                                                                            // B7
  // u = A (v*beta), w = A (k*beta*2^g) for both chunks
  float uu[4][4] = {}, ww[4][4] = {}, u1[2][4] = {}, w1[2][4] = {};
  #pragma unroll
  for (int kk = 0; kk < BT; kk += 16) {
    unsigned a[4]; fragA(a, sA, LDA, 16 * mt, kk);
    #pragma unroll
    for (int jj = 0; jj < 2; jj++) {
      unsigned bb[4];
      fragB_kn2(bb, sX1, LDS, 32 * nq + 16 * jj, kk);
      mma16816(uu[2 * jj], a[0], a[1], a[2], a[3], bb[0], bb[1]); mma16816(uu[2 * jj + 1], a[0], a[1], a[2], a[3], bb[2], bb[3]);
      fragB_kn2(bb, sX2, LDS, 32 * nq + 16 * jj, kk);
      mma16816(ww[2 * jj], a[0], a[1], a[2], a[3], bb[0], bb[1]); mma16816(ww[2 * jj + 1], a[0], a[1], a[2], a[3], bb[2], bb[3]);
    }
  }
  #pragma unroll
  for (int kk = 0; kk < 32; kk += 16) {
    unsigned a[4]; fragA(a, sA + BT * LDA, LDA, 16 * m1, kk);
    {
      unsigned bb[4];
      fragB_kn2(bb, sX1 + BT * LDS, LDS, 16 * n1, kk);
      mma16816(u1[0], a[0], a[1], a[2], a[3], bb[0], bb[1]); mma16816(u1[1], a[0], a[1], a[2], a[3], bb[2], bb[3]);
      fragB_kn2(bb, sX2 + BT * LDS, LDS, 16 * n1, kk);
      mma16816(w1[0], a[0], a[1], a[2], a[3], bb[0], bb[1]); mma16816(w1[1], a[0], a[1], a[2], a[3], bb[2], bb[3]);
    }
  }
  __syncthreads(); stamp(4);                                                                                            // B8
  // w -> sX2, u -> sX1 (bf16); chunk 0: v_new0 = u0, Vg0 = v_new0 * 2^(gL-g) -> sVG; output-stage attention of both chunks -> sA
  {
    const float gL = sG[BT - 1];
    #pragma unroll
    for (int j = 0; j < 4; j++)
      #pragma unroll
      for (int hh = 0; hh < 2; hh++) {
        const int r = 16 * mt + gq + hh * 8, cc = 32 * nq + 8 * j + 2 * tq;
        __nv_bfloat162 pw; pw.x = f2b(ww[j][2 * hh]); pw.y = f2b(ww[j][2 * hh + 1]);
        *reinterpret_cast<__nv_bfloat162*>(sX2 + r * LDS + cc) = pw;
        const float v0 = b2f(f2b(uu[j][2 * hh])), v1 = b2f(f2b(uu[j][2 * hh + 1]));
        __nv_bfloat162 pn; pn.x = f2b(v0); pn.y = f2b(v1);
        *reinterpret_cast<__nv_bfloat162*>(sX1 + r * LDS + cc) = pn;
        const float dg = ex2(gL - sG[r]);
        __nv_bfloat162 pg; pg.x = f2b(v0 * dg); pg.y = f2b(v1 * dg);
        *reinterpret_cast<__nv_bfloat162*>(sVG + r * LDS + cc) = pg;
      }
    #pragma unroll
    for (int j = 0; j < 2; j++)
      #pragma unroll
      for (int hh = 0; hh < 2; hh++) {
        const int r = 16 * m1 + gq + hh * 8, cc = 16 * n1 + 8 * j + 2 * tq;
        __nv_bfloat162 pw; pw.x = f2b(w1[j][2 * hh]); pw.y = f2b(w1[j][2 * hh + 1]);
        *reinterpret_cast<__nv_bfloat162*>(sX2 + (BT + r) * LDS + cc) = pw;
        u1[j][2 * hh] = b2f(f2b(u1[j][2 * hh])); u1[j][2 * hh + 1] = b2f(f2b(u1[j][2 * hh + 1]));
      }
  }
  __syncthreads(); stamp(5);                                                                                            // B9
  // chunk 0 output (h = 0: o0 = scale * A_qk v_new0) and h1 = K0^T Vg0
  {
    float pv[4][4] = {};
    #pragma unroll
    for (int kk = 0; kk < BT; kk += 16) {
      unsigned a[4]; fragA(a, sQK, LDA, 16 * mt, kk);
      #pragma unroll
      for (int jj = 0; jj < 2; jj++) { unsigned bb[4]; fragB_kn2(bb, sX1, LDS, 32 * nq + 16 * jj, kk); mma16816(pv[2 * jj], a[0], a[1], a[2], a[3], bb[0], bb[1]); mma16816(pv[2 * jj + 1], a[0], a[1], a[2], a[3], bb[2], bb[3]); }
    }
    #pragma unroll
    for (int j = 0; j < 4; j++)
      #pragma unroll
      for (int hh = 0; hh < 2; hh++) {
        const int r = 16 * mt + gq + hh * 8, cc = 32 * nq + 8 * j + 2 * tq;
        __nv_bfloat162 ov;
        ov.x = f2b(fmaf(scale, 0.f, scale * pv[j][2 * hh]));
        ov.y = f2b(fmaf(scale, 0.f, scale * pv[j][2 * hh + 1]));
        *reinterpret_cast<__nv_bfloat162*>(o + ((rb + r) * HV + hv) * D + cc) = ov;
      }
    float st[2][4][4] = {};
    #pragma unroll
    for (int kk = 0; kk < BT; kk += 16) {
      #pragma unroll
      for (int mi = 0; mi < 2; mi++) {
        unsigned a[4]; fragA_T(a, sK, LDS, 32 * mt + 16 * mi, kk);
        #pragma unroll
        for (int nj = 0; nj < 2; nj++) { unsigned bb[4]; fragB_kn2(bb, sVG, LDS, 32 * nq + 16 * nj, kk); mma16816(st[mi][2 * nj], a[0], a[1], a[2], a[3], bb[0], bb[1]); mma16816(st[mi][2 * nj + 1], a[0], a[1], a[2], a[3], bb[2], bb[3]); }
      }
    }
    #pragma unroll
    for (int mi = 0; mi < 2; mi++)
      #pragma unroll
      for (int ni = 0; ni < 4; ni++)
        #pragma unroll
        for (int hh = 0; hh < 2; hh++) {
          const int r = 32 * mt + 16 * mi + gq + hh * 8, cc = 32 * nq + 8 * ni + 2 * tq;
          __nv_bfloat162 pr; pr.x = f2b(st[mi][ni][2 * hh]); pr.y = f2b(st[mi][ni][2 * hh + 1]);
          *reinterpret_cast<__nv_bfloat162*>(sH + r * LDS + cc) = pr;
        }
  }
  __syncthreads(); stamp(6);                                                                                            // B10
  // chunk 1: P = W1 h1, v_new1 = u1 - P -> sX1; Q1 h1
  float qh[2][4] = {};
  {
    float pp[2][4] = {};
    const bf16* W1 = sX2 + BT * LDS; const bf16* Q1 = sQ + BT * LDS;
    #pragma unroll
    for (int kk = 0; kk < D; kk += 16) {
      unsigned a[4], aq[4]; fragA(a, W1, LDS, 16 * m1, kk); fragA(aq, Q1, LDS, 16 * m1, kk);
      {
        unsigned bb[4]; fragB_kn2(bb, sH, LDS, 16 * n1, kk);
        mma16816(pp[0], a[0], a[1], a[2], a[3], bb[0], bb[1]); mma16816(qh[0], aq[0], aq[1], aq[2], aq[3], bb[0], bb[1]);
        mma16816(pp[1], a[0], a[1], a[2], a[3], bb[2], bb[3]); mma16816(qh[1], aq[0], aq[1], aq[2], aq[3], bb[2], bb[3]);
      }
    }
    #pragma unroll
    for (int j = 0; j < 2; j++)
      #pragma unroll
      for (int hh = 0; hh < 2; hh++) {
        const int r = 16 * m1 + gq + hh * 8, cc = 16 * n1 + 8 * j + 2 * tq;
        __nv_bfloat162 pn; pn.x = f2b(u1[j][2 * hh] - pp[j][2 * hh]); pn.y = f2b(u1[j][2 * hh + 1] - pp[j][2 * hh + 1]);
        *reinterpret_cast<__nv_bfloat162*>(sX1 + (BT + r) * LDS + cc) = pn;
      }
  }
  __syncthreads(); stamp(7);                                                                                            // B11
  {
    float pv[2][4] = {};
    #pragma unroll
    for (int kk = 0; kk < 32; kk += 16) {
      unsigned a[4]; fragA(a, sQK + BT * LDA, LDA, 16 * m1, kk);
      { unsigned bb[4]; fragB_kn2(bb, sX1 + BT * LDS, LDS, 16 * n1, kk); mma16816(pv[0], a[0], a[1], a[2], a[3], bb[0], bb[1]); mma16816(pv[1], a[0], a[1], a[2], a[3], bb[2], bb[3]); }
    }
    #pragma unroll
    for (int j = 0; j < 2; j++)
      #pragma unroll
      for (int hh = 0; hh < 2; hh++) {
        const int r = 16 * m1 + gq + hh * 8, cc = 16 * n1 + 8 * j + 2 * tq;
        if (r < L1) {
          const float eg = ex2(sG[BT + r]);
          __nv_bfloat162 ov;
          ov.x = f2b(fmaf(scale, eg * qh[j][2 * hh], scale * pv[j][2 * hh]));
          ov.y = f2b(fmaf(scale, eg * qh[j][2 * hh + 1], scale * pv[j][2 * hh + 1]));
          *reinterpret_cast<__nv_bfloat162*>(o + ((rb + BT + r) * HV + hv) * D + cc) = ov;
        }
      }
  }
  stamp(15);
}
__global__ void __launch_bounds__(NTH2, 1) gdn_fused6_k(const bf16* __restrict__ q, const bf16* __restrict__ k, const bf16* __restrict__ v,
                                                         const float* __restrict__ g, const bf16* __restrict__ beta, bf16* __restrict__ o,
                                                         int T, int HK, int HV, float scale) {
  gdn6_item(q, k, v, g, beta, o, T, HK, HV, scale, blockIdx.y, blockIdx.x);
}
__global__ void __launch_bounds__(NTH2, 1) gdn_fused6ts_k(const bf16* __restrict__ q, const bf16* __restrict__ k, const bf16* __restrict__ v,
                                                           const float* __restrict__ g, const bf16* __restrict__ beta, bf16* __restrict__ o,
                                                           int T, int HK, int HV, float scale, unsigned long long* ts) {
  gdn6_item<true>(q, k, v, g, beta, o, T, HK, HV, scale, blockIdx.y, blockIdx.x, ts);
}
// Round 4, C6e: persistent linear-attention core. 148 blocks draw tickets in order: first every (sequence, value head)
// Gated DeltaNet item (gdn6_item for 65..96-row paths, else gdn4_item), then the matching gated-RMSNorm items. A norm
// item waits for its GDN item's flag, so the norms run on SMs that the GDN tail (336 items = 2.27 waves) leaves idle
// and the separate norm launch disappears. All GDN tickets are handed out before any norm ticket, so a waiting block
// never holds back a GDN item. Real rows are bit-identical to gdn_fused6 + gated_rmsnorm_inv2; padded rows get zeros.
// ctl: [0] ticket counter, [1] finished blocks, [2 + i] done flag of GDN item i (reset by its consumer).
__global__ void __launch_bounds__(NTH2, 1) gdn_fused7_k(const bf16* __restrict__ q, const bf16* __restrict__ k, const bf16* __restrict__ v,
                                                         const float* __restrict__ g, const bf16* __restrict__ beta, bf16* __restrict__ o,
                                                         int T, int HK, int HV, float scale, int B,
                                                         const bf16* __restrict__ proj, int P, int zoff, const bf16* __restrict__ normw,
                                                         const float* __restrict__ sig, const int* __restrict__ canon,
                                                         const int* __restrict__ rowmask, int N, float neps, bf16* __restrict__ y, int* ctl, int gdn_only) {
  __shared__ int s_item;
  const int tid = threadIdx.x, w = tid >> 5, l = tid & 31;
  const int nitem = B * HV;
  // GDN items: static round-robin (same placement as a grid launch, no ticket latency). Norm items: dynamic tickets,
  // each drawn one item ahead so the atomic's round trip overlaps work.
  int ticket = 0;
  for (int it = blockIdx.x; it < nitem; it += gridDim.x) {
    if (tid == 0 && !gdn_only && it + (int)gridDim.x >= nitem) ticket = atomicAdd(ctl, 1);
    const int b = it / HV, hv = it - b * HV;
    if (T > BT && T <= 96) gdn6_item(q, k, v, g, beta, o, T, HK, HV, scale, b, hv);
    else gdn4_item(q, k, v, g, beta, o, T, HK, HV, scale, b, hv);
    __threadfence();
    __syncthreads();
    if (tid == 0 && !gdn_only) atomicExch(ctl + 2 + it, 1);
  }
  if (!gdn_only && blockIdx.x >= nitem && tid == 0) ticket = atomicAdd(ctl, 1);
  while (!gdn_only) {
    if (tid == 0) { s_item = ticket; if (ticket < nitem) ticket = atomicAdd(ctl, 1); }
    __syncthreads();
    const int it = s_item;
    __syncthreads();
    if (it >= nitem) break;
    {
      // norm item: canon / z / weights do not depend on the GDN item, so they are loaded before waiting for its flag;
      // after the flag only the o loads (all rows of the item in flight together) and the arithmetic remain
      const int j = it, b = j / HV, hv = j - b * HV;
      const int d0 = l * 4;
      const uint2 wa = __ldg(reinterpret_cast<const uint2*>(normw + d0));
      const bf16* wb = reinterpret_cast<const bf16*>(&wa);
      constexpr int RPW = 6;                           // rows per warp per round (16 warps x 6 = 96 rows)
      for (int tb = 0; tb < T; tb += RPW * (NTH2 / 32)) {
        int n[RPW]; uint2 za[RPW], xa[RPW];
        #pragma unroll
        for (int i = 0; i < RPW; i++) { const int t = tb + w + i * (NTH2 / 32); n[i] = t < T ? __ldg(canon + (long)b * T + t) : -1; }
        #pragma unroll
        for (int i = 0; i < RPW; i++) if (n[i] >= 0) za[i] = *reinterpret_cast<const uint2*>(proj + (long)n[i] * P + zoff + hv * D + d0);
        if (tb == 0) {
          if (tid == 0) {
            while (atomicAdd(ctl + 2 + j, 0) == 0) __nanosleep(32);
            ctl[2 + j] = 0;
            __threadfence();
          }
          __syncthreads();
        }
        #pragma unroll
        for (int i = 0; i < RPW; i++)
          if (n[i] >= 0) xa[i] = __ldcg(reinterpret_cast<const uint2*>(o + (((long)b * T + tb + w + i * (NTH2 / 32)) * HV + hv) * D + d0));
        #pragma unroll
        for (int i = 0; i < RPW; i++) {              // as gated_rmsnorm_inv2_k: one warp per row, 4 channels per lane
          if (n[i] < 0) continue;
          const bf16* xb = reinterpret_cast<const bf16*>(&xa[i]); const bf16* zb = reinterpret_cast<const bf16*>(&za[i]);
          const unsigned short* zs = reinterpret_cast<const unsigned short*>(&za[i]);
          float xv[4], ss = 0.f;
          #pragma unroll
          for (int jj = 0; jj < 4; jj++) { xv[jj] = b2f(xb[jj]); ss += xv[jj] * xv[jj]; }
          #pragma unroll
          for (int of = 16; of > 0; of >>= 1) ss += __shfl_xor_sync(0xffffffff, ss, of);
          const float rstd = 1.0f / sqrtf(ss / (float)D + neps);
          uint2 oa; bf16* ob = reinterpret_cast<bf16*>(&oa);
          #pragma unroll
          for (int jj = 0; jj < 4; jj++) {
            const float zg = b2f(zb[jj]);
            ob[jj] = f2b(((xv[jj] * rstd) * b2f(wb[jj])) * zg * __ldg(sig + zs[jj]));
          }
          *reinterpret_cast<uint2*>(y + (long)n[i] * (HV * D) + hv * D + d0) = oa;
        }
      }
      if (b == 0)                                      // padded packed rows: zeros (finite for the out_proj GEMM / attention)
        for (int nb = 0; nb < N; nb += 4 * (NTH2 / 32)) {
          int m[4];
          #pragma unroll
          for (int i = 0; i < 4; i++) { const int n = nb + w + i * (NTH2 / 32); m[i] = n < N ? __ldg(rowmask + n) : 1; }
          #pragma unroll
          for (int i = 0; i < 4; i++)
            if (m[i] == 0) *reinterpret_cast<uint2*>(y + (long)(nb + w + i * (NTH2 / 32)) * (HV * D) + hv * D + d0) = make_uint2(0u, 0u);
        }
      __syncthreads();
    }
  }
  if (tid == 0) {
    __threadfence();
    if (atomicAdd(ctl + 1, 1) == (int)gridDim.x - 1) { ctl[0] = 0; ctl[1] = 0; __threadfence(); }
  }
}
}  // namespace gdn

torch::Tensor gdn_fused7(torch::Tensor q, torch::Tensor k, torch::Tensor v, torch::Tensor g, torch::Tensor beta, double scale,
                         torch::Tensor proj, int64_t zoff, torch::Tensor normw, torch::Tensor sig, torch::Tensor canon, torch::Tensor rowmask, double neps) {
  CHECK(q); CHECK(k); CHECK(v); CHECK(g); CHECK(beta); CHECK(proj); CHECK(normw); CHECK(sig); CHECK(canon); CHECK(rowmask);
  TORCH_CHECK(q.dim() == 4 && q.size(3) == 128 && v.size(3) == 128 && g.scalar_type() == torch::kFloat32);
  const int B = q.size(0), T = q.size(1), HK = q.size(2), HV = v.size(2);
  const long N = proj.size(0);
  TORCH_CHECK(canon.numel() == (long)B * T && rowmask.numel() == N);
  static torch::Tensor ctl;
  if (!ctl.defined() || ctl.numel() < 2 + (long)B * HV) ctl = torch::zeros({std::max<long>(4096, 2 + (long)B * HV)}, proj.options().dtype(torch::kInt32));
  auto o = torch::empty_like(v);
  auto y = torch::empty({N, (long)HV * 128}, proj.options());
  const size_t smem = (6 * 96 * gdn::LDS + gdn::D * gdn::LDS + 2 * 96 * gdn::LDA) * sizeof(bf16) + (2 * 32 * 33 + 16 * 33 + 2 * 96) * sizeof(float);
  static bool attr = false;
  if (!attr) { cudaFuncSetAttribute(gdn::gdn_fused7_k, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem); attr = true; }
  static int nsm = 0;
  if (!nsm) cudaDeviceGetAttribute(&nsm, cudaDevAttrMultiProcessorCount, q.get_device());
  const char* go = getenv("OJ_GDN7_ONLY"); const int gdn_only = go && go[0] == '1';   // diagnostics: GDN items only, returns o
  const int grid = std::min(nsm, 2 * B * HV);
  gdn::gdn_fused7_k<<<grid, gdn::NTH2, smem, at::cuda::getCurrentCUDAStream()>>>((bf16*)q.data_ptr(), (bf16*)k.data_ptr(),
      (bf16*)v.data_ptr(), g.data_ptr<float>(), (bf16*)beta.data_ptr(), (bf16*)o.data_ptr(), T, HK, HV, (float)scale, B,
      (bf16*)proj.data_ptr(), (int)proj.size(1), (int)zoff, (bf16*)normw.data_ptr(), sig.data_ptr<float>(), canon.data_ptr<int>(),
      rowmask.data_ptr<int>(), (int)N, (float)neps, (bf16*)y.data_ptr(), ctl.data_ptr<int>(), gdn_only);
  return gdn_only ? o : y;
}
torch::Tensor gdn_fused6ts(torch::Tensor q, torch::Tensor k, torch::Tensor v, torch::Tensor g, torch::Tensor beta, double scale) {
  // diagnostics: per-block globaltimer stamps after each barrier of gdn6_item -> int64 [B*HV, 16]
  const int B = q.size(0), T = q.size(1), HK = q.size(2), HV = v.size(2);
  auto o = torch::empty_like(v);
  auto ts = torch::zeros({(long)B * HV, 16}, q.options().dtype(torch::kInt64));
  const size_t smem = (6 * 96 * gdn::LDS + gdn::D * gdn::LDS + 2 * 96 * gdn::LDA) * sizeof(bf16) + (2 * 32 * 33 + 16 * 33 + 2 * 96) * sizeof(float);
  cudaFuncSetAttribute(gdn::gdn_fused6ts_k, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem);
  gdn::gdn_fused6ts_k<<<dim3(HV, B), gdn::NTH2, smem, at::cuda::getCurrentCUDAStream()>>>((bf16*)q.data_ptr(), (bf16*)k.data_ptr(),
      (bf16*)v.data_ptr(), g.data_ptr<float>(), (bf16*)beta.data_ptr(), (bf16*)o.data_ptr(), T, HK, HV, (float)scale,
      (unsigned long long*)ts.data_ptr<int64_t>());
  return ts;
}
torch::Tensor gdn_fused9(torch::Tensor q, torch::Tensor k, torch::Tensor v, torch::Tensor g, torch::Tensor beta, double scale) {
  CHECK(q); CHECK(k); CHECK(v); CHECK(g); CHECK(beta);
  TORCH_CHECK(q.dim() == 4 && q.size(3) == 128 && v.size(3) == 128 && g.scalar_type() == torch::kFloat32);
  const int B = q.size(0), T = q.size(1), HK = q.size(2), HV = v.size(2), BT = gdn::BT, D = gdn::D;
  const int NT = (T + BT - 1) / BT;
  const long items = (long)B * HV * NT;
  auto o = torch::empty_like(v);
  auto W = torch::empty({items * BT * D}, q.options()), U = torch::empty({items * BT * D}, q.options());
  auto QK = torch::empty({items * BT * BT}, q.options()), G = torch::empty({items * BT}, g.options());
  const size_t smemL = (5 * BT * gdn::LDS + 2 * BT * gdn::LDA) * sizeof(bf16) + (2 * BT * (BT + 1) + 3 * 16 * (BT + 1) + 3 * BT) * sizeof(float);
  const size_t smemD = (6 * BT * gdn::LDS + 2 * BT * gdn::LDA + D * gdn::LDS + 2 * BT * gdn::LDS) * sizeof(bf16) + 2 * BT * sizeof(float);
  static bool attr = false;
  if (!attr) {
    cudaFuncSetAttribute(gdn::gdnL_k, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smemL);
    cudaFuncSetAttribute(gdn::gdnD_k, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smemD);
    attr = true;
  }
  auto st = at::cuda::getCurrentCUDAStream();
  const char* only = getenv("OJ_GDN9_ONLY");                 // diagnostics: "L" or "D" launches one of the two kernels
  if (!(only && only[0] == 'D')) gdn::gdnL_k<<<dim3(HV, B, NT), gdn::NTH2, smemL, st>>>((bf16*)q.data_ptr(), (bf16*)k.data_ptr(), (bf16*)v.data_ptr(), g.data_ptr<float>(),
      (bf16*)beta.data_ptr(), T, HK, HV, (bf16*)W.data_ptr(), (bf16*)U.data_ptr(), (bf16*)QK.data_ptr(), G.data_ptr<float>());
  if (!(only && only[0] == 'L')) gdn::gdnD_k<<<dim3(HV, B), gdn::NTH2, smemD, st>>>((bf16*)q.data_ptr(), (bf16*)k.data_ptr(), (bf16*)W.data_ptr(), (bf16*)U.data_ptr(),
      (bf16*)QK.data_ptr(), G.data_ptr<float>(), (bf16*)o.data_ptr(), T, HK, HV, (float)scale);
  return o;
}
torch::Tensor gdn_fused8(torch::Tensor q, torch::Tensor k, torch::Tensor v, torch::Tensor g, torch::Tensor beta, double scale) {
  CHECK(q); CHECK(k); CHECK(v); CHECK(g); CHECK(beta);
  TORCH_CHECK(q.dim() == 4 && q.size(3) == 128 && v.size(3) == 128 && g.scalar_type() == torch::kFloat32);
  const int B = q.size(0), T = q.size(1), HK = q.size(2), HV = v.size(2);
  auto o = torch::empty_like(v);
  const size_t smem = (9 * gdn::BT * gdn::LDS + gdn::D * gdn::LDS + 2 * gdn::BT * gdn::LDA) * sizeof(bf16) + (3 * 16 * 17 + 3 * gdn::BT) * sizeof(float);
  static bool attr = false;
  if (!attr) { cudaFuncSetAttribute(gdn::gdn_fused8_k, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem); attr = true; }
  gdn::gdn_fused8_k<<<dim3(HV, B), gdn::NTH2, smem, at::cuda::getCurrentCUDAStream()>>>((bf16*)q.data_ptr(), (bf16*)k.data_ptr(),
      (bf16*)v.data_ptr(), g.data_ptr<float>(), (bf16*)beta.data_ptr(), (bf16*)o.data_ptr(), T, HK, HV, (float)scale);
  return o;
}
torch::Tensor gdn_fused4(torch::Tensor q, torch::Tensor k, torch::Tensor v, torch::Tensor g, torch::Tensor beta, double scale);
torch::Tensor gdn_fused6(torch::Tensor q, torch::Tensor k, torch::Tensor v, torch::Tensor g, torch::Tensor beta, double scale) {
  // paths of 65..96 rows: both chunks' state-independent work side by side; otherwise gdn_fused4
  const int T = q.size(1);
  if (T <= gdn::BT || T > 96) return gdn_fused4(q, k, v, g, beta, scale);
  CHECK(q); CHECK(k); CHECK(v); CHECK(g); CHECK(beta);
  TORCH_CHECK(q.dim() == 4 && q.size(3) == 128 && v.size(3) == 128 && g.scalar_type() == torch::kFloat32);
  const int B = q.size(0), HK = q.size(2), HV = v.size(2);
  auto o = torch::empty_like(v);
  const size_t smem = (6 * 96 * gdn::LDS + gdn::D * gdn::LDS + 2 * 96 * gdn::LDA) * sizeof(bf16) + (2 * 32 * 33 + 16 * 33 + 2 * 96) * sizeof(float);
  static bool attr = false;
  if (!attr) { cudaFuncSetAttribute(gdn::gdn_fused6_k, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem); attr = true; }
  gdn::gdn_fused6_k<<<dim3(HV, B), gdn::NTH2, smem, at::cuda::getCurrentCUDAStream()>>>((bf16*)q.data_ptr(), (bf16*)k.data_ptr(),
      (bf16*)v.data_ptr(), g.data_ptr<float>(), (bf16*)beta.data_ptr(), (bf16*)o.data_ptr(), T, HK, HV, (float)scale);
  return o;
}

torch::Tensor gdn_fused5(torch::Tensor q, torch::Tensor k, torch::Tensor v, torch::Tensor g, torch::Tensor beta, double scale,
                         torch::Tensor proj, int64_t zoff, torch::Tensor normw, torch::Tensor sig, torch::Tensor canon, torch::Tensor rowmask, double neps) {
  CHECK(q); CHECK(k); CHECK(v); CHECK(g); CHECK(beta); CHECK(proj); CHECK(normw); CHECK(sig); CHECK(canon); CHECK(rowmask);
  TORCH_CHECK(q.dim() == 4 && q.size(3) == 128 && v.size(3) == 128 && g.scalar_type() == torch::kFloat32);
  const int B = q.size(0), T = q.size(1), HK = q.size(2), HV = v.size(2);
  const long N = proj.size(0);
  TORCH_CHECK(canon.numel() == (long)B * T && rowmask.numel() == N);
  auto y = torch::empty({N, (long)HV * 128}, proj.options());
  const size_t smem = (10 * gdn::BT * gdn::LDS + gdn::D * gdn::LDS + gdn::BT * gdn::LDA) * sizeof(bf16) + (3 * 16 * 17 + 3 * gdn::BT) * sizeof(float)
                      + gdn::BT * sizeof(int);
  static bool attr = false;
  if (!attr) { cudaFuncSetAttribute(gdn::gdn_fused5_k, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem); attr = true; }
  gdn::gdn_fused5_k<<<dim3(HV, B), gdn::NTH2, smem, at::cuda::getCurrentCUDAStream()>>>((bf16*)q.data_ptr(), (bf16*)k.data_ptr(),
      (bf16*)v.data_ptr(), g.data_ptr<float>(), (bf16*)beta.data_ptr(), T, HK, HV, (float)scale,
      (bf16*)proj.data_ptr(), (int)proj.size(1), (int)zoff, (bf16*)normw.data_ptr(), sig.data_ptr<float>(), canon.data_ptr<int>(),
      rowmask.data_ptr<int>(), (int)N, (float)neps, (bf16*)y.data_ptr());
  return y;
}

torch::Tensor gdn_fused4(torch::Tensor q, torch::Tensor k, torch::Tensor v, torch::Tensor g, torch::Tensor beta, double scale) {
  CHECK(q); CHECK(k); CHECK(v); CHECK(g); CHECK(beta);
  TORCH_CHECK(q.dim() == 4 && q.size(3) == 128 && v.size(3) == 128 && g.scalar_type() == torch::kFloat32);
  const int B = q.size(0), T = q.size(1), HK = q.size(2), HV = v.size(2);
  auto o = torch::empty_like(v);
  const size_t smem = (9 * gdn::BT * gdn::LDS + gdn::D * gdn::LDS + gdn::BT * gdn::LDA) * sizeof(bf16) + (3 * 16 * 17 + 3 * gdn::BT) * sizeof(float);
  static bool attr = false;
  if (!attr) { cudaFuncSetAttribute(gdn::gdn_fused4_k, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem); attr = true; }
  gdn::gdn_fused4_k<<<dim3(HV, B), gdn::NTH2, smem, at::cuda::getCurrentCUDAStream()>>>((bf16*)q.data_ptr(), (bf16*)k.data_ptr(),
      (bf16*)v.data_ptr(), g.data_ptr<float>(), (bf16*)beta.data_ptr(), (bf16*)o.data_ptr(), T, HK, HV, (float)scale);
  return o;
}

torch::Tensor gdn_fused3(torch::Tensor q, torch::Tensor k, torch::Tensor v, torch::Tensor g, torch::Tensor beta, double scale) {
  CHECK(q); CHECK(k); CHECK(v); CHECK(g); CHECK(beta);
  TORCH_CHECK(q.dim() == 4 && q.size(3) == 128 && v.size(3) == 128 && g.scalar_type() == torch::kFloat32);
  const int B = q.size(0), T = q.size(1), HK = q.size(2), HV = v.size(2);
  auto o = torch::empty_like(v);
  const size_t smem = (5 * gdn::BT * gdn::LDS + gdn::D * gdn::LDS + gdn::BT * gdn::LDA) * sizeof(bf16)
                    + (2 * gdn::BT * (gdn::BT + 1) + 3 * 16 * 17 + 3 * gdn::BT) * sizeof(float);
  static bool attr = false;
  if (!attr) { cudaFuncSetAttribute(gdn::gdn_fused3_k, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem); attr = true; }
  gdn::gdn_fused3_k<<<dim3(HV, B), gdn::NTH2, smem, at::cuda::getCurrentCUDAStream()>>>((bf16*)q.data_ptr(), (bf16*)k.data_ptr(),
      (bf16*)v.data_ptr(), g.data_ptr<float>(), (bf16*)beta.data_ptr(), (bf16*)o.data_ptr(), T, HK, HV, (float)scale);
  return o;
}

// =====================================================================================================================
// Round 4, C8: full-attention core for the prefix-tree layout, replacing SDPA (cuDNN) + gate_mul3. One CTA per
// (kv head, 16 query rows) and GQ query heads of that kv head (one warp each), so every K/V tile is loaded once for GQ
// heads. The tree mask arrives as bits (vbits[n][w]: key 32w+b visible to row n); 32-key blocks that no row of the CTA
// can see are skipped. FlashAttention-2 style online softmax in fp32 (exp2 domain), P in bf16 for P.V, output
// bf16(O / l) (as SDPA), then the sigmoid gate as gate_mul3: bf16(att * bf16(sigmoid(gate))).
namespace fattn {
using gdn::ldsm_x4; using gdn::mma16816; using gdn::fragA; using gdn::fragB_nk2; using gdn::fragB_kn2;
using gdn::cp_async16; using gdn::cp_async_commit; using gdn::cp_async_wait_all; using gdn::cp_async_wait1;
constexpr int HD = 256, QT = 16, KB = 32, LD = HD + 8;
__device__ __forceinline__ float ex2f(float x) { float y; asm("ex2.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x)); return y; }
__device__ __forceinline__ unsigned pack2(float a, float b) {
  __nv_bfloat162 t = __floats2bfloat162_rn(a, b); return *reinterpret_cast<unsigned*>(&t);
}
template <int GQ>
__global__ void __launch_bounds__(GQ * 32, 1) fattn_k(const bf16* __restrict__ qt, const bf16* __restrict__ kt, const bf16* __restrict__ vt,
                                                      const bf16* __restrict__ gate, const float* __restrict__ sig,
                                                      const int* __restrict__ vbits, bf16* __restrict__ y,
                                                      int N, int HQ, int HKV, float scale_log2) {
  extern __shared__ __align__(16) unsigned char smem_raw[];
  const int W = N / 32;
  bf16* sQ = reinterpret_cast<bf16*>(smem_raw);            // [GQ][QT][LD]
  bf16* sK = sQ + GQ * QT * LD;                             // [2][KB][LD]
  bf16* sV = sK + 2 * KB * LD;                              // [2][KB][LD]
  unsigned* sBits = reinterpret_cast<unsigned*>(sV + 2 * KB * LD);   // [QT][W]
  int* sList = reinterpret_cast<int*>(sBits + QT * W);      // [W] visible 32-key blocks
  __shared__ int sCnt;
  const int tid = threadIdx.x, w = tid >> 5, l = tid & 31, gq = l >> 2, tq = l & 3;
  const int NTH = GQ * 32;
  const int g = blockIdx.x / (HQ / HKV / GQ), hsub = blockIdx.x % (HQ / HKV / GQ);   // kv head, which GQ-slice of its q heads
  const int r0 = blockIdx.y * QT;
  const int h0 = g * (HQ / HKV) + hsub * GQ;               // first q head of this CTA
  for (int idx = tid; idx < GQ * QT * (HD / 8); idx += NTH) {
    const int i = idx / (QT * (HD / 8)), rem = idx % (QT * (HD / 8)), r = rem / (HD / 8), c8 = (rem % (HD / 8)) * 8;
    cp_async16(sQ + (i * QT + r) * LD + c8, qt + ((long)(h0 + i) * N + r0 + r) * HD + c8, true);
  }
  cp_async_commit();
  for (int idx = tid; idx < QT * W; idx += NTH) sBits[idx] = (unsigned)__ldg(vbits + (long)(r0 + idx / W) * W + idx % W);
  __syncthreads();
  if (w == 0) {                                             // list of 32-key blocks visible to any of the QT rows
    int cnt = 0;
    for (int b0 = 0; b0 < W; b0 += 32) {
      const int b = b0 + l;
      unsigned any = 0;
      if (b < W)
        #pragma unroll
        for (int r = 0; r < QT; r++) any |= sBits[r * W + b];
      const unsigned m = __ballot_sync(0xffffffff, any != 0);
      if (any) sList[cnt + __popc(m & ((1u << l) - 1))] = b;
      cnt += __popc(m);
    }
    if (l == 0) sCnt = cnt;
  }
  __syncthreads();
  const int nblk = sCnt;
  auto issue_kv = [&](int bi, int buf) {
    const int kb = sList[bi] * KB;
    for (int idx = tid; idx < KB * (HD / 8); idx += NTH) {
      const int r = idx / (HD / 8), c8 = (idx % (HD / 8)) * 8;
      const long src = ((long)g * N + kb + r) * HD + c8;
      cp_async16(sK + (buf * KB + r) * LD + c8, kt + src, true);
      cp_async16(sV + (buf * KB + r) * LD + c8, vt + src, true);
    }
    cp_async_commit();
  };
  issue_kv(0, 0);
  float O[HD / 8][4];
  #pragma unroll
  for (int n = 0; n < HD / 8; n++) O[n][0] = O[n][1] = O[n][2] = O[n][3] = 0.f;
  float mrow[2] = {-INFINITY, -INFINITY}, lrow[2] = {0.f, 0.f};
  const bf16* Qw = sQ + w * QT * LD;
  for (int bi = 0; bi < nblk; bi++) {
    const int buf = bi & 1;
    if (bi + 1 < nblk) { issue_kv(bi + 1, buf ^ 1); cp_async_wait1(); } else cp_async_wait_all();
    __syncthreads();
    const bf16* Kb = sK + buf * KB * LD; const bf16* Vb = sV + buf * KB * LD;
    float s[4][4];
    #pragma unroll
    for (int j = 0; j < 4; j++) s[j][0] = s[j][1] = s[j][2] = s[j][3] = 0.f;
    #pragma unroll
    for (int kk = 0; kk < HD; kk += 16) {
      unsigned a[4]; fragA(a, Qw, LD, 0, kk);
      #pragma unroll
      for (int jj = 0; jj < 2; jj++) {
        unsigned bb[4]; fragB_nk2(bb, Kb, LD, 16 * jj, kk);
        mma16816(s[2 * jj], a[0], a[1], a[2], a[3], bb[0], bb[1]);
        mma16816(s[2 * jj + 1], a[0], a[1], a[2], a[3], bb[2], bb[3]);
      }
    }
    const int blk = sList[bi];
    const unsigned vb0 = sBits[gq * W + blk], vb1 = sBits[(gq + 8) * W + blk];
    float mx[2] = {-INFINITY, -INFINITY};
    #pragma unroll
    for (int j = 0; j < 4; j++)
      #pragma unroll
      for (int e = 0; e < 4; e++) {
        const int kbit = 8 * j + 2 * tq + (e & 1);
        const unsigned vb = (e >> 1) ? vb1 : vb0;
        const float v = ((vb >> kbit) & 1u) ? s[j][e] * scale_log2 : -INFINITY;
        s[j][e] = v; mx[e >> 1] = fmaxf(mx[e >> 1], v);
      }
    float corr[2], muse[2];
    #pragma unroll
    for (int hh = 0; hh < 2; hh++) {
      mx[hh] = fmaxf(mx[hh], __shfl_xor_sync(0xffffffff, mx[hh], 1));
      mx[hh] = fmaxf(mx[hh], __shfl_xor_sync(0xffffffff, mx[hh], 2));
      const float mnew = fmaxf(mrow[hh], mx[hh]);
      muse[hh] = mnew == -INFINITY ? 0.f : mnew;
      corr[hh] = ex2f(mrow[hh] - muse[hh]);
      mrow[hh] = mnew;
      lrow[hh] *= corr[hh];
    }
    #pragma unroll
    for (int j = 0; j < 4; j++)
      #pragma unroll
      for (int e = 0; e < 4; e++) { const float p = ex2f(s[j][e] - muse[e >> 1]); s[j][e] = p; lrow[e >> 1] += p; }
    #pragma unroll
    for (int n = 0; n < HD / 8; n++) { O[n][0] *= corr[0]; O[n][1] *= corr[0]; O[n][2] *= corr[1]; O[n][3] *= corr[1]; }
    #pragma unroll
    for (int ks = 0; ks < 2; ks++) {
      const unsigned a0 = pack2(s[2 * ks][0], s[2 * ks][1]), a1 = pack2(s[2 * ks][2], s[2 * ks][3]);
      const unsigned a2 = pack2(s[2 * ks + 1][0], s[2 * ks + 1][1]), a3 = pack2(s[2 * ks + 1][2], s[2 * ks + 1][3]);
      #pragma unroll
      for (int nn = 0; nn < HD / 16; nn++) {
        unsigned bb[4]; fragB_kn2(bb, Vb, LD, 16 * nn, 16 * ks);
        mma16816(O[2 * nn], a0, a1, a2, a3, bb[0], bb[1]);
        mma16816(O[2 * nn + 1], a0, a1, a2, a3, bb[2], bb[3]);
      }
    }
    __syncthreads();
  }
  #pragma unroll
  for (int hh = 0; hh < 2; hh++) {
    lrow[hh] += __shfl_xor_sync(0xffffffff, lrow[hh], 1);
    lrow[hh] += __shfl_xor_sync(0xffffffff, lrow[hh], 2);
    lrow[hh] = 1.f / lrow[hh];
  }
  const int h = h0 + w;
  #pragma unroll
  for (int n = 0; n < HD / 8; n++)
    #pragma unroll
    for (int hh = 0; hh < 2; hh++) {
      const int row = r0 + gq + hh * 8, col = 8 * n + 2 * tq;
      const long off = (long)row * HQ * HD + h * HD + col;
      const __nv_bfloat162 gg = *reinterpret_cast<const __nv_bfloat162*>(gate + off);
      const unsigned short* gs = reinterpret_cast<const unsigned short*>(&gg);
      const float a0 = b2f(f2b(O[n][2 * hh] * lrow[hh])), a1 = b2f(f2b(O[n][2 * hh + 1] * lrow[hh]));
      __nv_bfloat162 ov;
      ov.x = f2b(a0 * rbf(__ldg(sig + gs[0]))); ov.y = f2b(a1 * rbf(__ldg(sig + gs[1])));
      *reinterpret_cast<__nv_bfloat162*>(y + off) = ov;
    }
}
}  // namespace fattn

torch::Tensor fattn_tree(torch::Tensor qt, torch::Tensor kt, torch::Tensor vt, torch::Tensor gate, torch::Tensor sig, torch::Tensor vbits,
                         double scale, int64_t gq) {
  CHECK(qt); CHECK(kt); CHECK(vt); CHECK(gate); CHECK(sig); CHECK(vbits);
  const int HQ = qt.size(1), N = qt.size(2), HKV = kt.size(1);
  TORCH_CHECK(qt.size(0) == 1 && qt.size(3) == 256 && N % 32 == 0 && vbits.size(0) == N && vbits.size(1) == N / 32);
  TORCH_CHECK((HQ / HKV) % gq == 0 && (gq == 3 || gq == 6));
  auto y = torch::empty({(long)N, (long)HQ * 256}, qt.options());
  const float sl2 = (float)(scale * 1.4426950408889634);
  const size_t smem = (gq * fattn::QT * fattn::LD + 4 * fattn::KB * fattn::LD) * sizeof(bf16) + (fattn::QT * (N / 32) + N / 32) * 4;
  dim3 grid(HKV * (HQ / HKV / gq), N / fattn::QT);
  auto st = at::cuda::getCurrentCUDAStream();
#define FA(G_) { static size_t attr = 0; if (attr < smem) { TORCH_CHECK(cudaFuncSetAttribute(fattn::fattn_k<G_>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem) == cudaSuccess); attr = smem; } \
  fattn::fattn_k<G_><<<grid, G_ * 32, smem, st>>>((bf16*)qt.data_ptr(), (bf16*)kt.data_ptr(), (bf16*)vt.data_ptr(), (bf16*)gate.data_ptr(), \
      sig.data_ptr<float>(), vbits.data_ptr<int>(), (bf16*)y.data_ptr(), N, HQ, HKV, sl2); }
  if (gq == 6) FA(6) else FA(3)
#undef FA
  return y;
}

// C8b: the same attention with the full-attention prep fused in (q/k RMSNorm + partial RoPE exactly as fullattn_prep2_k,
// read straight from the in_proj output; v and the gate are plain slices of it) and two warps per query head that
// take alternate 32-key blocks (own online-softmax state each, merged at the end), so a CTA runs 2*GQ warps.
namespace fattn {
__device__ __forceinline__ void prep_row(bf16* row, const float* w1, const bf16* cosr, const bf16* sinr, int RD, float eps) {
  const int lane = threadIdx.x & 31, d0 = lane * 8;
  uint4 a = *reinterpret_cast<const uint4*>(row + d0);
  const bf16* ab = reinterpret_cast<const bf16*>(&a);
  float xv[8], ss = 0.f;
  #pragma unroll
  for (int j = 0; j < 8; j++) { xv[j] = b2f(ab[j]); ss += xv[j] * xv[j]; }
  #pragma unroll
  for (int o = 16; o > 0; o >>= 1) ss += __shfl_xor_sync(0xffffffff, ss, o);
  const float r = rsqrtf(ss / (float)HD + eps);
  float nv[8];
  #pragma unroll
  for (int j = 0; j < 8; j++) nv[j] = rbf((xv[j] * r) * w1[d0 + j]);
  const int hl = RD / 16;
  float pv[8];
  #pragma unroll
  for (int j = 0; j < 8; j++) pv[j] = __shfl_xor_sync(0xffffffff, nv[j], hl);
  // RD % 8 == 0: a lane's 8 dims are either all rotary or none. cos/sin as one 16-byte load each, no divergent branch
  // (loads inside a per-element branch were serialized: ~0.6 us per row)
  const bool rope = d0 < RD;
  const uint4 cv = *reinterpret_cast<const uint4*>(cosr + (rope ? d0 : 0)), sv = *reinterpret_cast<const uint4*>(sinr + (rope ? d0 : 0));
  const bf16* cb = reinterpret_cast<const bf16*>(&cv); const bf16* sb = reinterpret_cast<const bf16*>(&sv);
  uint4 ov; bf16* ob = reinterpret_cast<bf16*>(&ov);
  #pragma unroll
  for (int j = 0; j < 8; j++) {
    const float rh = (d0 + j < RD / 2) ? -pv[j] : pv[j];
    const float ro = rbf(rbf(nv[j] * b2f(cb[j])) + rbf(rh * b2f(sb[j])));
    ob[j] = f2b(rope ? ro : nv[j]);
  }
  *reinterpret_cast<uint4*>(row + d0) = ov;
}
template <int GQ, bool PREP>
__global__ void __launch_bounds__(2 * GQ * 32, 1) fattn2_k(const bf16* __restrict__ proj, const float* __restrict__ qw1, const float* __restrict__ kw1,
                                                          const bf16* __restrict__ cosb, const bf16* __restrict__ sinb,
                                                          const float* __restrict__ sig, const int* __restrict__ vbits, bf16* __restrict__ y,
                                                          int N, int HQ, int HKV, int RD, float scale_log2, float eps,
                                                          const bf16* __restrict__ qt, const bf16* __restrict__ kt, const bf16* __restrict__ vt, const bf16* __restrict__ gate,
                                                          unsigned long long* ts = nullptr) {
  auto stamp = [&](int k) { if (ts && threadIdx.x == 0 && k < 16) { unsigned long long t_; asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t_)); ts[(blockIdx.y * gridDim.x + blockIdx.x) * 16 + k] = t_; } };
  stamp(0);
  extern __shared__ __align__(16) unsigned char smem_raw[];
  constexpr int NW = 2 * GQ, NTH = NW * 32;
  const int W = N / 32, P = HQ * 2 * HD + 2 * HKV * HD;
  bf16* sQ = reinterpret_cast<bf16*>(smem_raw);            // [GQ][QT][LD]
  bf16* sK = sQ + GQ * QT * LD;                             // [2 buf][2 half][KB][LD]
  bf16* sV = sK + 4 * KB * LD;                              // [2 buf][2 half][KB][LD]
  unsigned* sBits = reinterpret_cast<unsigned*>(sV + 4 * KB * LD);   // [QT][W]
  int* sList = reinterpret_cast<int*>(sBits + QT * W);      // [W]
  __shared__ int sCnt;
  __shared__ float sW1q[HD], sW1k[HD];
  __shared__ __align__(16) bf16 sCSq[QT][2][64];            // cos / sin of the query rows (RD <= 64)
  __shared__ __align__(16) bf16 sCSk[2][2][KB][2][64];      // [buf][half][row][cos|sin][RD] of the key rows
  const int tid = threadIdx.x, w = tid >> 5, l = tid & 31, gq = l >> 2, tq = l & 3;
  const int hi = w % GQ, part = w / GQ;
  for (int i = tid; i < HD; i += NTH) { sW1q[i] = __ldg(qw1 + i); sW1k[i] = __ldg(kw1 + i); }
  const int g = blockIdx.x / (HQ / HKV / GQ), hsub = blockIdx.x % (HQ / HKV / GQ);
  const int r0 = blockIdx.y * QT, h0 = g * (HQ / HKV) + hsub * GQ;
  const long kcol = (long)HQ * 2 * HD + g * HD, vcol = kcol + (long)HKV * HD;
  for (int idx = tid; idx < GQ * QT * (HD / 8); idx += NTH) {
    const int i = idx / (QT * (HD / 8)), rem = idx % (QT * (HD / 8)), r = rem / (HD / 8), c8 = (rem % (HD / 8)) * 8;
    cp_async16(sQ + (i * QT + r) * LD + c8, PREP ? proj + (long)(r0 + r) * P + (h0 + i) * 2 * HD + c8
                                                 : qt + ((long)(h0 + i) * N + r0 + r) * HD + c8, true);
  }
  if (PREP) for (int idx = tid; idx < QT * 2 * (RD / 8); idx += NTH) {
    const int r = idx / (2 * (RD / 8)), cs = (idx / (RD / 8)) & 1, c8 = (idx % (RD / 8)) * 8;
    cp_async16(&sCSq[r][cs][c8], (cs ? sinb : cosb) + (long)(r0 + r) * RD + c8, true);
  }
  cp_async_commit();
  for (int idx = tid; idx < QT * W; idx += NTH) sBits[idx] = (unsigned)__ldg(vbits + (long)(r0 + idx / W) * W + idx % W);
  __syncthreads();
  if (w == 0) {
    int cnt = 0;
    for (int b0 = 0; b0 < W; b0 += 32) {
      const int b = b0 + l;
      unsigned any = 0;
      if (b < W)
        #pragma unroll
        for (int r = 0; r < QT; r++) any |= sBits[r * W + b];
      const unsigned m = __ballot_sync(0xffffffff, any != 0);
      if (any) sList[cnt + __popc(m & ((1u << l) - 1))] = b;
      cnt += __popc(m);
    }
    if (l == 0) sCnt = cnt;
  }
  __syncthreads();
  const int nblk = sCnt, npair = (nblk + 1) / 2;
  auto issue_pair = [&](int t, int buf) {
    for (int hf = 0; hf < 2; hf++) {
      if (2 * t + hf >= nblk) break;
      const int kb = sList[2 * t + hf] * KB;
      bf16* dK = sK + ((buf * 2 + hf) * KB) * LD; bf16* dV = sV + ((buf * 2 + hf) * KB) * LD;
      for (int idx = tid; idx < KB * (HD / 8); idx += NTH) {
        const int r = idx / (HD / 8), c8 = (idx % (HD / 8)) * 8;
        if (PREP) {
          const long row = (long)(kb + r) * P;
          cp_async16(dK + r * LD + c8, proj + row + kcol + c8, true);
          cp_async16(dV + r * LD + c8, proj + row + vcol + c8, true);
        } else {
          const long row = ((long)g * N + kb + r) * HD;
          cp_async16(dK + r * LD + c8, kt + row + c8, true);
          cp_async16(dV + r * LD + c8, vt + row + c8, true);
        }
      }
      if (PREP) for (int idx = tid; idx < KB * 2 * (RD / 8); idx += NTH) {
        const int r = idx / (2 * (RD / 8)), cs = (idx / (RD / 8)) & 1, c8 = (idx % (RD / 8)) * 8;
        cp_async16(&sCSk[buf][hf][r][cs][c8], (cs ? sinb : cosb) + (long)(kb + r) * RD + c8, true);
      }
    }
    cp_async_commit();
  };
  issue_pair(0, 0);
  stamp(1);
  cp_async_wait1();                                         // q rows landed
  __syncthreads();
  stamp(2);
  if (PREP) for (int rr = w; rr < GQ * QT; rr += NW) {      // q: RMSNorm + RoPE in place
    const int r = rr % QT;
    prep_row(sQ + rr * LD, sW1q, sCSq[r][0], sCSq[r][1], RD, eps);
  }
  float O[HD / 8][4];
  #pragma unroll
  for (int n = 0; n < HD / 8; n++) O[n][0] = O[n][1] = O[n][2] = O[n][3] = 0.f;
  float mrow[2] = {-INFINITY, -INFINITY}, lrow[2] = {0.f, 0.f};
  const bf16* Qw = sQ + hi * QT * LD;
  stamp(3);
  for (int t = 0; t < npair; t++) {
    const int buf = t & 1;
    if (t + 1 < npair) { issue_pair(t + 1, buf ^ 1); cp_async_wait1(); } else cp_async_wait_all();
    __syncthreads();
    stamp(4 + 3 * t);
    if (PREP) for (int rr = w; rr < 2 * KB; rr += NW) {     // k: RMSNorm + RoPE in place for both blocks of the pair
      const int hf = rr / KB, kr = rr % KB;
      if (2 * t + hf >= nblk) break;
      prep_row(sK + ((buf * 2 + hf) * KB + kr) * LD, sW1k, sCSk[buf][hf][kr][0], sCSk[buf][hf][kr][1], RD, eps);
    }
    if (PREP) __syncthreads();
    stamp(5 + 3 * t);
    if (2 * t + part < nblk) {
      const bf16* Kb = sK + ((buf * 2 + part) * KB) * LD; const bf16* Vb = sV + ((buf * 2 + part) * KB) * LD;
      float s[4][4];
      #pragma unroll
      for (int j = 0; j < 4; j++) s[j][0] = s[j][1] = s[j][2] = s[j][3] = 0.f;
      #pragma unroll
      for (int kk = 0; kk < HD; kk += 16) {
        unsigned a[4]; fragA(a, Qw, LD, 0, kk);
        #pragma unroll
        for (int jj = 0; jj < 2; jj++) {
          unsigned bb[4]; fragB_nk2(bb, Kb, LD, 16 * jj, kk);
          mma16816(s[2 * jj], a[0], a[1], a[2], a[3], bb[0], bb[1]);
          mma16816(s[2 * jj + 1], a[0], a[1], a[2], a[3], bb[2], bb[3]);
        }
      }
      const int blk = sList[2 * t + part];
      const unsigned vb0 = sBits[gq * W + blk], vb1 = sBits[(gq + 8) * W + blk];
      float mx[2] = {-INFINITY, -INFINITY};
      #pragma unroll
      for (int j = 0; j < 4; j++)
        #pragma unroll
        for (int e = 0; e < 4; e++) {
          const int kbit = 8 * j + 2 * tq + (e & 1);
          const unsigned vb = (e >> 1) ? vb1 : vb0;
          const float v = ((vb >> kbit) & 1u) ? s[j][e] * scale_log2 : -INFINITY;
          s[j][e] = v; mx[e >> 1] = fmaxf(mx[e >> 1], v);
        }
      float corr[2], muse[2];
      #pragma unroll
      for (int hh = 0; hh < 2; hh++) {
        mx[hh] = fmaxf(mx[hh], __shfl_xor_sync(0xffffffff, mx[hh], 1));
        mx[hh] = fmaxf(mx[hh], __shfl_xor_sync(0xffffffff, mx[hh], 2));
        const float mnew = fmaxf(mrow[hh], mx[hh]);
        muse[hh] = mnew == -INFINITY ? 0.f : mnew;
        corr[hh] = ex2f(mrow[hh] - muse[hh]);
        mrow[hh] = mnew;
        lrow[hh] *= corr[hh];
      }
      #pragma unroll
      for (int j = 0; j < 4; j++)
        #pragma unroll
        for (int e = 0; e < 4; e++) { const float p = ex2f(s[j][e] - muse[e >> 1]); s[j][e] = p; lrow[e >> 1] += p; }
      #pragma unroll
      for (int n = 0; n < HD / 8; n++) { O[n][0] *= corr[0]; O[n][1] *= corr[0]; O[n][2] *= corr[1]; O[n][3] *= corr[1]; }
      #pragma unroll
      for (int ks = 0; ks < 2; ks++) {
        const unsigned a0 = pack2(s[2 * ks][0], s[2 * ks][1]), a1 = pack2(s[2 * ks][2], s[2 * ks][3]);
        const unsigned a2 = pack2(s[2 * ks + 1][0], s[2 * ks + 1][1]), a3 = pack2(s[2 * ks + 1][2], s[2 * ks + 1][3]);
        #pragma unroll
        for (int nn = 0; nn < HD / 16; nn++) {
          unsigned bb[4]; fragB_kn2(bb, Vb, LD, 16 * nn, 16 * ks);
          mma16816(O[2 * nn], a0, a1, a2, a3, bb[0], bb[1]);
          mma16816(O[2 * nn + 1], a0, a1, a2, a3, bb[2], bb[3]);
        }
      }
    }
    __syncthreads();
  }
  #pragma unroll
  for (int hh = 0; hh < 2; hh++) {
    lrow[hh] += __shfl_xor_sync(0xffffffff, lrow[hh], 1);
    lrow[hh] += __shfl_xor_sync(0xffffffff, lrow[hh], 2);
  }
  stamp(13);
  // gate rows of this CTA -> smem (sV region, free now), in flight while the key halves are merged
  bf16* sG = sV;                                            // [GQ][QT][HD]
  for (int idx = tid; idx < GQ * QT * (HD / 8); idx += NTH) {
    const int i = idx / (QT * (HD / 8)), rem = idx % (QT * (HD / 8)), r = rem / (HD / 8), c8 = (rem % (HD / 8)) * 8;
    cp_async16(sG + (i * QT + r) * HD + c8, PREP ? proj + (long)(r0 + r) * P + (h0 + i) * 2 * HD + HD + c8
                                                 : gate + (long)(r0 + r) * HQ * HD + (h0 + i) * HD + c8, true);
  }
  cp_async_commit();
  // merge the two key halves of each head: part 1 publishes (m, l, O), part 0 combines and writes
  float* scr = reinterpret_cast<float*>(sK);               // [GQ][32 lanes][HD/8 * 4 + 4]
  constexpr int SL = HD / 8 * 4 + 4;
  if (part == 1) {
    float* d = scr + (hi * 32 + l) * SL;
    #pragma unroll
    for (int n = 0; n < HD / 8; n++) { d[4 * n] = O[n][0]; d[4 * n + 1] = O[n][1]; d[4 * n + 2] = O[n][2]; d[4 * n + 3] = O[n][3]; }
    d[HD / 2] = mrow[0]; d[HD / 2 + 1] = mrow[1]; d[HD / 2 + 2] = lrow[0]; d[HD / 2 + 3] = lrow[1];
  }
  cp_async_wait_all();
  __syncthreads();
  stamp(14);
  if (part == 0) {
    const float* d = scr + (hi * 32 + l) * SL;
    float a0[2], a1[2], linv[2];
    #pragma unroll
    for (int hh = 0; hh < 2; hh++) {
      const float m1 = d[HD / 2 + hh], l1 = d[HD / 2 + 2 + hh];
      const float m = fmaxf(mrow[hh], m1), mu = m == -INFINITY ? 0.f : m;
      a0[hh] = ex2f(mrow[hh] - mu); a1[hh] = ex2f(m1 - mu);
      linv[hh] = 1.f / (lrow[hh] * a0[hh] + l1 * a1[hh]);
    }
    const int h = h0 + hi;
    #pragma unroll
    for (int n = 0; n < HD / 8; n++)
      #pragma unroll
      for (int hh = 0; hh < 2; hh++) {
        const int row = r0 + gq + hh * 8, col = 8 * n + 2 * tq;
        const float o0 = (O[n][2 * hh] * a0[hh] + d[4 * n + 2 * hh] * a1[hh]) * linv[hh];
        const float o1 = (O[n][2 * hh + 1] * a0[hh] + d[4 * n + 2 * hh + 1] * a1[hh]) * linv[hh];
        const __nv_bfloat162 gg = *reinterpret_cast<const __nv_bfloat162*>(sG + (hi * QT + gq + hh * 8) * HD + col);
        const unsigned short* gs = reinterpret_cast<const unsigned short*>(&gg);
        __nv_bfloat162 ov;
        ov.x = f2b(b2f(f2b(o0)) * rbf(__ldg(sig + gs[0]))); ov.y = f2b(b2f(f2b(o1)) * rbf(__ldg(sig + gs[1])));
        *reinterpret_cast<__nv_bfloat162*>(y + (long)row * HQ * HD + h * HD + col) = ov;
      }
  }
  stamp(15);
}
}  // namespace fattn

static unsigned long long* g_fattn_ts = nullptr;
torch::Tensor fattn_tree2(torch::Tensor proj, torch::Tensor qw1, torch::Tensor kw1, torch::Tensor cosb, torch::Tensor sinb, torch::Tensor sig,
                          torch::Tensor vbits, int64_t HQ, int64_t HKV, double scale, double eps) {
  CHECK(proj); CHECK(qw1); CHECK(kw1); CHECK(cosb); CHECK(sinb); CHECK(sig); CHECK(vbits);
  const int N = proj.size(0), RD = cosb.size(-1);
  TORCH_CHECK(RD <= 64 && RD % 8 == 0 && proj.size(1) == HQ * 2 * 256 + 2 * HKV * 256 && N % 32 == 0 && vbits.size(0) == N && vbits.size(1) == N / 32 && (HQ / HKV) % 3 == 0);
  auto y = torch::empty({(long)N, HQ * 256}, proj.options());
  const float sl2 = (float)(scale * 1.4426950408889634);
  const size_t smem = (3 * fattn::QT * fattn::LD + 8 * fattn::KB * fattn::LD) * sizeof(bf16) + (fattn::QT * (N / 32) + N / 32) * 4;
  static size_t attr = 0;
  if (attr < smem) { TORCH_CHECK(cudaFuncSetAttribute(fattn::fattn2_k<3, true>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem) == cudaSuccess); attr = smem; }
  dim3 grid(HKV * (HQ / HKV / 3), N / fattn::QT);
  fattn::fattn2_k<3, true><<<grid, 6 * 32, smem, at::cuda::getCurrentCUDAStream()>>>((bf16*)proj.data_ptr(), qw1.data_ptr<float>(), kw1.data_ptr<float>(),
      (bf16*)cosb.data_ptr(), (bf16*)sinb.data_ptr(), sig.data_ptr<float>(), vbits.data_ptr<int>(), (bf16*)y.data_ptr(),
      N, (int)HQ, (int)HKV, RD, sl2, (float)eps, nullptr, nullptr, nullptr, nullptr, g_fattn_ts);
  return y;
}
torch::Tensor fattn_tree3(torch::Tensor qt, torch::Tensor kt, torch::Tensor vt, torch::Tensor gate, torch::Tensor sig, torch::Tensor vbits, double scale) {
  // fattn2_k without the prep (inputs from fullattn_prep2): two warps per query head over alternate key blocks
  CHECK(qt); CHECK(kt); CHECK(vt); CHECK(gate); CHECK(sig); CHECK(vbits);
  const int HQ = qt.size(1), N = qt.size(2), HKV = kt.size(1);
  TORCH_CHECK(qt.size(0) == 1 && qt.size(3) == 256 && N % 32 == 0 && vbits.size(0) == N && vbits.size(1) == N / 32 && (HQ / HKV) % 3 == 0);
  auto y = torch::empty({(long)N, (long)HQ * 256}, qt.options());
  const float sl2 = (float)(scale * 1.4426950408889634);
  const size_t smem = (3 * fattn::QT * fattn::LD + 8 * fattn::KB * fattn::LD) * sizeof(bf16) + (fattn::QT * (N / 32) + N / 32) * 4;
  static size_t attr = 0;
  if (attr < smem) { TORCH_CHECK(cudaFuncSetAttribute(fattn::fattn2_k<3, false>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem) == cudaSuccess); attr = smem; }
  dim3 grid(HKV * (HQ / HKV / 3), N / fattn::QT);
  fattn::fattn2_k<3, false><<<grid, 6 * 32, smem, at::cuda::getCurrentCUDAStream()>>>(nullptr, nullptr, nullptr, nullptr, nullptr,
      sig.data_ptr<float>(), vbits.data_ptr<int>(), (bf16*)y.data_ptr(), N, HQ, HKV, 64, sl2, 0.f,
      (bf16*)qt.data_ptr(), (bf16*)kt.data_ptr(), (bf16*)vt.data_ptr(), (bf16*)gate.data_ptr(), nullptr);
  return y;
}
torch::Tensor fattn_tree2_ts(torch::Tensor proj, torch::Tensor qw1, torch::Tensor kw1, torch::Tensor cosb, torch::Tensor sinb, torch::Tensor sig,
                             torch::Tensor vbits, int64_t HQ, int64_t HKV, double scale, double eps) {
  // diagnostics: globaltimer stamps per CTA -> int64 [CTAs, 16]
  auto ts = torch::zeros({(long)HKV * (HQ / HKV / 3) * (proj.size(0) / 16), 16}, proj.options().dtype(torch::kInt64));
  g_fattn_ts = (unsigned long long*)ts.data_ptr<int64_t>();
  fattn_tree2(proj, qw1, kw1, cosb, sinb, sig, vbits, HQ, HKV, scale, eps);
  g_fattn_ts = nullptr;
  return ts;
}

// Tree visibility [N, N] (bool) -> bits [N, N/32] (int32; bit b of word w = key 32w + b), one warp ballot per word:
// one launch instead of the int64 shift/sum chain (host cost of the layout build 0.90 -> ~0.77 ms).
__global__ void pack_bits_k(const bool* __restrict__ vis, int* __restrict__ out, int N) {
  const long wid = ((long)blockIdx.x * blockDim.x + threadIdx.x) >> 5;
  const int lane = threadIdx.x & 31, W = N / 32;
  if (wid >= (long)N * W) return;
  const long n = wid / W; const int w = wid % W;
  const unsigned m = __ballot_sync(0xffffffff, vis[n * N + 32 * w + lane]);
  if (lane == 0) out[wid] = (int)m;
}
torch::Tensor pack_bits(torch::Tensor vis) {
  CHECK(vis);
  TORCH_CHECK(vis.scalar_type() == torch::kBool && vis.dim() == 2 && vis.size(0) == vis.size(1) && vis.size(0) % 32 == 0);
  const int N = vis.size(0);
  auto out = torch::empty({N, N / 32}, vis.options().dtype(torch::kInt32));
  const long threads = (long)N * (N / 32) * 32;
  pack_bits_k<<<(threads + 255) / 256, 256, 0, at::cuda::getCurrentCUDAStream()>>>(vis.data_ptr<bool>(), out.data_ptr<int>(), N);
  return out;
}
