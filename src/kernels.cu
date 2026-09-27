// Fused non-GEMM kernels for Qwen3.5/3.8 hybrid decoder (Open-Jev-27B inference).
// Every kernel reproduces the bf16 rounding points of the PyTorch reference so outputs match
// the HF implementation up to reduction order.
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_bf16.h>

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
  #pragma unroll
  for (int j = 0; j < 8; j++) {
    const int dd = d0 + j;
    if (dd < RD) {
      const float rh = (dd < RD / 2) ? -pv[j] : pv[j];
      const float c = b2f(cosb[n * RD + dd]), s = b2f(sinb[n * RD + dd]);
      outv[j] = rbf(rbf(nv[j] * c) + rbf(rh * s));
    } else outv[j] = nv[j];
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
