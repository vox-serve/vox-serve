// Graph-capturable sampling kernels for the Rust runtime.
//
// Every per-request parameter (temperature, top-k, top-p, min-p, RNG seed / step) is read from
// device memory, so a captured CUDA graph can be replayed by the Rust runtime without any host
// involvement. Randomness comes from Philox keyed by (request seed, request step, codebook), which
// makes sampling independent of batch composition and of how many times a graph was replayed.

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <curand_kernel.h>

#include <cub/block/block_reduce.cuh>
#include <cub/block/block_scan.cuh>

namespace {

constexpr int kThreads = 1024;
constexpr int kBins = 1024;
constexpr float kZRange = 32.f;  // tokens more than 32 nats below the max are never sampled
constexpr float kBinScale = kBins / kZRange;
constexpr int kListCap = 2048;

__device__ __forceinline__ float to_float(float v) { return v; }
__device__ __forceinline__ float to_float(__nv_bfloat16 v) { return __bfloat162float(v); }
__device__ __forceinline__ float to_float(__half v) { return __half2float(v); }

template <typename T>
__device__ __forceinline__ T from_float(float v);
template <>
__device__ __forceinline__ float from_float<float>(float v) { return v; }
template <>
__device__ __forceinline__ __nv_bfloat16 from_float<__nv_bfloat16>(float v) { return __float2bfloat16(v); }
template <>
__device__ __forceinline__ __half from_float<__half>(float v) { return __float2half(v); }

struct MaxIdx {
  float v;
  int i;
};

struct MaxIdxOp {
  __device__ __forceinline__ MaxIdx operator()(const MaxIdx& a, const MaxIdx& b) const {
    if (a.v > b.v) return a;
    if (b.v > a.v) return b;
    return a.i <= b.i ? a : b;
  }
};

// Sort (z, idx) pairs ascending by z (i.e. descending probability) in shared memory.
__device__ void bitonic_sort(float* key, int* val, int n_pow2) {
  for (int k = 2; k <= n_pow2; k <<= 1) {
    for (int j = k >> 1; j > 0; j >>= 1) {
      for (int t = threadIdx.x; t < n_pow2; t += blockDim.x) {
        int ixj = t ^ j;
        if (ixj > t) {
          bool up = (t & k) == 0;
          float a = key[t], b = key[ixj];
          if ((a > b) == up) {
            key[t] = b;
            key[ixj] = a;
            int tv = val[t];
            val[t] = val[ixj];
            val[ixj] = tv;
          }
        }
      }
      __syncthreads();
    }
  }
}

struct SampleSmem {
  float hmass[kBins];
  int hcnt[kBins];
  float list_z[kListCap];
  int list_i[kListCap];
  int list_n;
  int cut_bin_k;
  int cut_bin_p;
  float zthr;
  float zk_mass;
  int found;
  int result;
  float bcast;
  MaxIdx best;
};

template <typename T>
__device__ void collect_bin(const T* x, int V, float xmax, float inv_t, int bin, SampleSmem& s) {
  if (threadIdx.x == 0) s.list_n = 0;
  __syncthreads();
  for (int i = threadIdx.x; i < V; i += blockDim.x) {
    float z = (xmax - to_float(x[i])) * inv_t;
    if (z < kZRange && min(kBins - 1, int(z * kBinScale)) == bin) {
      int pos = atomicAdd(&s.list_n, 1);
      if (pos < kListCap) {
        s.list_z[pos] = z;
        s.list_i[pos] = i;
      }
    }
  }
  __syncthreads();
  int n = min(s.list_n, kListCap);
  int n_pow2 = 1;
  while (n_pow2 < n) n_pow2 <<= 1;
  for (int t = n + threadIdx.x; t < n_pow2; t += blockDim.x) {
    s.list_z[t] = INFINITY;
    s.list_i[t] = INT_MAX;
  }
  __syncthreads();
  bitonic_sort(s.list_z, s.list_i, n_pow2);
}

// logits: [n_rows, V] with `row_stride` elements between rows. Row r belongs to request r / rows_per_req.
template <typename T>
__global__ void __launch_bounds__(kThreads) sample_kernel(
    const T* __restrict__ logits, int64_t row_stride, int V, int rows_per_req,
    const int* __restrict__ active, const float* __restrict__ temperature, const int* __restrict__ top_k,
    const float* __restrict__ top_p, const float* __restrict__ min_p, const int64_t* __restrict__ seed,
    const int* __restrict__ step, int rng_salt, int* __restrict__ out, int64_t out_stride) {
  using BlockReduceMax = cub::BlockReduce<MaxIdx, kThreads>;
  using BlockReduceSum = cub::BlockReduce<float, kThreads>;
  using BlockScanInt = cub::BlockScan<int, kThreads>;
  using BlockScanFloat = cub::BlockScan<float, kThreads>;
  __shared__ union {
    typename BlockReduceMax::TempStorage max;
    typename BlockReduceSum::TempStorage sum;
    typename BlockScanInt::TempStorage scan_i;
    typename BlockScanFloat::TempStorage scan_f;
  } tmp;
  __shared__ SampleSmem s;

  const int row = blockIdx.x;
  const int req = row / rows_per_req;
  const int sub = row % rows_per_req;
  const T* x = logits + row * row_stride;
  int* o = out + req * out_stride + sub;

  if (!active[req]) {
    if (threadIdx.x == 0) *o = 0;
    return;
  }

  // Pass 1: max / argmax (first index wins ties, like torch.argmax).
  MaxIdx mi{-INFINITY, INT_MAX};
  for (int i = threadIdx.x; i < V; i += blockDim.x) {
    float v = to_float(x[i]);
    if (v > mi.v) mi = MaxIdx{v, i};
  }
  MaxIdx best = BlockReduceMax(tmp.max).Reduce(mi, MaxIdxOp());
  if (threadIdx.x == 0) s.best = best;
  __syncthreads();
  best = s.best;

  const float t = temperature[req];
  const int k = top_k[req];
  if (t <= 0.f || k == 1) {
    if (threadIdx.x == 0) *o = best.i;
    return;
  }
  const float inv_t = 1.f / t;
  const float xmax = best.v;

  // Pass 2: histogram of z = (xmax - x) / t (probability = exp(-z) / Z).
  for (int b = threadIdx.x; b < kBins; b += blockDim.x) {
    s.hmass[b] = 0.f;
    s.hcnt[b] = 0;
  }
  __syncthreads();
  for (int i = threadIdx.x; i < V; i += blockDim.x) {
    float z = (xmax - to_float(x[i])) * inv_t;
    if (z < kZRange) {
      int bin = min(kBins - 1, int(z * kBinScale));
      atomicAdd(&s.hmass[bin], __expf(-z));
      atomicAdd(&s.hcnt[bin], 1);
    }
  }
  __syncthreads();

  // Inclusive scans over bins (one bin per thread).
  int my_cnt = s.hcnt[threadIdx.x];
  float my_mass = s.hmass[threadIdx.x];
  int cum_cnt;
  BlockScanInt(tmp.scan_i).InclusiveSum(my_cnt, cum_cnt);
  __syncthreads();
  float cum_mass;
  BlockScanFloat(tmp.scan_f).InclusiveSum(my_mass, cum_mass);
  __syncthreads();

  const int k_eff = (k > 0 && k < V) ? k : INT_MAX;
  if (threadIdx.x == 0) {
    s.cut_bin_k = kBins;  // no top-k cut inside the histogram range
    s.cut_bin_p = kBins;
    s.zthr = kZRange;
  }
  __syncthreads();
  if (cum_cnt >= k_eff && cum_cnt - my_cnt < k_eff) s.cut_bin_k = threadIdx.x;
  __syncthreads();

  // Mass kept after top-k (relative to max), Zk.
  const int bk = s.cut_bin_k;
  if (bk < kBins) collect_bin(x, V, xmax, inv_t, bk, s);
  // Every thread knows its own bin's cumulative values; the thread owning bk publishes them.
  __shared__ int cnt_before_k;
  __shared__ float mass_before_k;
  if (threadIdx.x == bk) {
    cnt_before_k = cum_cnt - my_cnt;
    mass_before_k = cum_mass - my_mass;
  }
  __syncthreads();
  float zk_mass;
  if (bk < kBins) {
    if (threadIdx.x == 0) {
      int n = min(s.list_n, kListCap);
      int need = k_eff - cnt_before_k;
      float m = mass_before_k;
      float zcut = kZRange;
      if (s.list_n > kListCap) {
        // Pathologically flat bin: keep the whole bin.
        m = mass_before_k + s.hmass[bk];
        zcut = (bk + 1) / kBinScale;
      } else {
        for (int j = 0; j < n && j < need; ++j) {
          m += __expf(-s.list_z[j]);
          zcut = s.list_z[j];
        }
      }
      s.zk_mass = m;
      s.zthr = zcut;
    }
    __syncthreads();
    zk_mass = s.zk_mass;
  } else {
    // Total in-range mass.
    if (threadIdx.x == kBins - 1) s.zk_mass = cum_mass;
    __syncthreads();
    zk_mass = s.zk_mass;
  }

  // Top-p on the (top-k restricted) distribution.
  const float p = top_p[req];
  if (p > 0.f && p < 1.f) {
    const float target = p * zk_mass;
    // restricted cumulative mass per bin
    float rc = (int)threadIdx.x < bk ? cum_mass : (threadIdx.x == bk ? zk_mass : INFINITY);
    float rprev = (int)threadIdx.x <= bk ? cum_mass - my_mass : INFINITY;
    if (rc >= target && rprev < target) s.cut_bin_p = threadIdx.x;
    __syncthreads();
    const int bp = s.cut_bin_p;
    __shared__ float mass_before_p;
    if (threadIdx.x == bp) mass_before_p = cum_mass - my_mass;
    __syncthreads();
    if (bp < kBins) {
      if (bp != bk) collect_bin(x, V, xmax, inv_t, bp, s);
      if (threadIdx.x == 0) {
        int n = min(s.list_n, kListCap);
        if (s.list_n > kListCap) {
          s.zthr = fminf(s.zthr, (bp + 1) / kBinScale);
        } else {
          float m = mass_before_p;
          float zcut = s.list_z[0];
          for (int j = 0; j < n; ++j) {
            m += __expf(-s.list_z[j]);
            zcut = s.list_z[j];
            if (m >= target) break;
          }
          s.zthr = fminf(s.zthr, zcut);
        }
      }
      __syncthreads();
    }
  }

  // Min-p: keep p_i >= min_p * p_max  <=>  z_i <= -ln(min_p).
  const float mp = min_p[req];
  if (threadIdx.x == 0 && mp > 0.f) s.zthr = fminf(s.zthr, -__logf(mp));
  __syncthreads();
  const float zthr = s.zthr;

  // Pass 3: inverse-CDF sampling over the kept set, contiguous chunk per thread.
  const int chunk = (V + kThreads - 1) / kThreads;
  const int lo = threadIdx.x * chunk;
  const int hi = min(V, lo + chunk);
  float local = 0.f;
  for (int i = lo; i < hi; ++i) {
    float z = (xmax - to_float(x[i])) * inv_t;
    if (z <= zthr) local += __expf(-z);
  }
  float excl, total;
  BlockScanFloat(tmp.scan_f).ExclusiveSum(local, excl, total);
  __syncthreads();
  if (threadIdx.x == 0) {
    curandStatePhilox4_32_10_t st;
    curand_init((unsigned long long)seed[req], (unsigned long long)step[req] * 64ull + (unsigned long long)sub,
                (unsigned long long)rng_salt * 4ull, &st);
    float u = curand_uniform(&st);  // (0, 1]
    s.bcast = u * total;
    s.found = 0;
    s.result = best.i;
  }
  __syncthreads();
  const float target = s.bcast;
  if (local > 0.f && excl <= target && (target < excl + local)) {
    float c = excl;
    int last = -1;
    for (int i = lo; i < hi; ++i) {
      float z = (xmax - to_float(x[i])) * inv_t;
      if (z <= zthr) {
        c += __expf(-z);
        last = i;
        if (c > target) break;
      }
    }
    if (last >= 0 && atomicExch(&s.found, 1) == 0) s.result = last;
  }
  __syncthreads();
  if (threadIdx.x == 0) *o = s.result;
}

// Repetition penalty with a "seen tokens" bitmask per (slot, sub-row): bits[slot, rows_per_req, W32].
template <typename T>
__global__ void rep_penalty_apply_kernel(T* __restrict__ logits, int64_t row_stride, int V, int rows_per_req,
                                         const uint32_t* __restrict__ bits, int words, const int* __restrict__ slot,
                                         const int* __restrict__ active, const int* __restrict__ first,
                                         const float* __restrict__ penalty) {
  const int row = blockIdx.y;
  const int req = row / rows_per_req;
  const int sub = row % rows_per_req;
  if (!active[req] || first[req]) return;
  const float pen = penalty[req];
  if (pen == 1.f || pen <= 0.f) return;
  const uint32_t* b = bits + ((int64_t)slot[req] * rows_per_req + sub) * words;
  T* x = logits + row * row_stride;
  for (int w = blockIdx.x * blockDim.x + threadIdx.x; w < words; w += gridDim.x * blockDim.x) {
    uint32_t m = b[w];
    while (m) {
      int bit = __ffs(m) - 1;
      m &= m - 1;
      int i = w * 32 + bit;
      if (i < V) {
        float v = to_float(x[i]);
        v = v > 0.f ? v / pen : v * pen;
        x[i] = from_float<T>(v);
      }
    }
  }
}

__global__ void rep_penalty_reset_kernel(uint32_t* __restrict__ bits, int words, int rows_per_req,
                                         const int* __restrict__ slot, const int* __restrict__ active,
                                         const int* __restrict__ first) {
  const int req = blockIdx.y;
  if (!active[req] || !first[req]) return;
  uint32_t* b = bits + (int64_t)slot[req] * rows_per_req * words;
  for (int w = blockIdx.x * blockDim.x + threadIdx.x; w < words * rows_per_req; w += gridDim.x * blockDim.x)
    b[w] = 0u;
}

__global__ void rep_penalty_update_kernel(uint32_t* __restrict__ bits, int words, int rows_per_req, int V,
                                          const int* __restrict__ slot, const int* __restrict__ active,
                                          const int* __restrict__ tokens, int64_t tok_stride) {
  const int req = blockIdx.x;
  const int sub = threadIdx.x;
  if (sub >= rows_per_req || !active[req]) return;
  int tkn = tokens[req * tok_stride + sub];
  if (tkn < 0 || tkn >= V) return;
  uint32_t* b = bits + ((int64_t)slot[req] * rows_per_req + sub) * words;
  atomicOr(&b[tkn >> 5], 1u << (tkn & 31));
}

// Standard normal noise [N, K] keyed by (seed, step, salt); same contract as `sample`'s RNG.
__global__ void randn_kernel(float* __restrict__ out, int n, int K, const int64_t* __restrict__ seed,
                             const int* __restrict__ step, int64_t rng_salt) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  curandStatePhilox4_32_10_t st;
  curand_init((unsigned long long)seed[i], (unsigned long long)step[i] * 64ull + 63ull,
              (unsigned long long)rng_salt * 4ull, &st);
  for (int k = 0; k < K; ++k) out[(int64_t)i * K + k] = curand_normal(&st);
}

}  // namespace

// Host side: C launchers called from the `vox_serve._rt` extension (rust/src/pymodule.rs), which takes the
// pointers, strides, device and stream from torch tensors (vox_serve/rt/kernels.py). Each launches on
// `stream` with `device` current and returns the cudaError_t of the launch.

namespace {

// Logits dtype codes shared with vox_serve/rt/kernels.py.
constexpr int kFloat32 = 0;
constexpr int kFloat16 = 1;
constexpr int kBFloat16 = 2;

#define DISPATCH_LOGITS(dtype, ...)        \
  [&] {                                    \
    if (dtype == kBFloat16) {              \
      using scalar_t = __nv_bfloat16;      \
      __VA_ARGS__();                       \
    } else if (dtype == kFloat16) {        \
      using scalar_t = __half;             \
      __VA_ARGS__();                       \
    } else if (dtype == kFloat32) {        \
      using scalar_t = float;              \
      __VA_ARGS__();                       \
    } else {                               \
      return cudaErrorInvalidValue;        \
    }                                      \
    return cudaGetLastError();             \
  }()

// Makes `device` current for the launch, like c10::cuda::OptionalCUDAGuard.
struct DeviceGuard {
  int prev = -1;
  explicit DeviceGuard(int device) {
    int cur = -1;
    if (cudaGetDevice(&cur) == cudaSuccess && cur != device && cudaSetDevice(device) == cudaSuccess) prev = cur;
  }
  ~DeviceGuard() {
    if (prev >= 0) cudaSetDevice(prev);
  }
};

}  // namespace

// logits: n_rows rows of V (row_stride apart); out: [B, out_stride] int32.
extern "C" int vox_sample(int device, cudaStream_t stream, int dtype, const void* logits, int64_t row_stride, int V,
                          int64_t n_rows, int64_t rows_per_req, const int* active, const float* temperature,
                          const int* top_k, const float* top_p, const float* min_p, const int64_t* seed,
                          const int* step, int64_t rng_salt, int* out, int64_t out_stride) {
  DeviceGuard guard(device);
  return DISPATCH_LOGITS(dtype, [&] {
    sample_kernel<scalar_t><<<n_rows, kThreads, 0, stream>>>(reinterpret_cast<const scalar_t*>(logits), row_stride,
                                                             V, rows_per_req, active, temperature, top_k, top_p,
                                                             min_p, seed, step, rng_salt, out, out_stride);
  });
}

extern "C" int vox_rep_penalty_apply(int device, cudaStream_t stream, int dtype, void* logits, int64_t row_stride,
                                     int V, int64_t n_rows, int64_t rows_per_req, const uint32_t* bits, int words,
                                     const int* slot, const int* active, const int* first, const float* penalty) {
  DeviceGuard guard(device);
  dim3 grid((words + 255) / 256, n_rows);
  return DISPATCH_LOGITS(dtype, [&] {
    rep_penalty_apply_kernel<scalar_t><<<grid, 256, 0, stream>>>(reinterpret_cast<scalar_t*>(logits), row_stride, V,
                                                                 rows_per_req, bits, words, slot, active, first,
                                                                 penalty);
  });
}

extern "C" int vox_rep_penalty_reset(int device, cudaStream_t stream, uint32_t* bits, int words, int64_t rows_per_req,
                                     int n_req, const int* slot, const int* active, const int* first) {
  DeviceGuard guard(device);
  dim3 grid((words * rows_per_req + 255) / 256, n_req);
  rep_penalty_reset_kernel<<<grid, 256, 0, stream>>>(bits, words, rows_per_req, slot, active, first);
  return cudaGetLastError();
}

extern "C" int vox_rep_penalty_update(int device, cudaStream_t stream, uint32_t* bits, int words,
                                      int64_t rows_per_req, int64_t vocab, int n_req, const int* slot,
                                      const int* active, const int* tokens, int64_t tok_stride) {
  DeviceGuard guard(device);
  rep_penalty_update_kernel<<<n_req, 32, 0, stream>>>(bits, words, rows_per_req, vocab, slot, active, tokens,
                                                      tok_stride);
  return cudaGetLastError();
}

// out: [n, K] float32 contiguous.
extern "C" int vox_randn(int device, cudaStream_t stream, float* out, int n, int K, const int64_t* seed,
                         const int* step, int64_t rng_salt) {
  DeviceGuard guard(device);
  randn_kernel<<<(n + 127) / 128, 128, 0, stream>>>(out, n, K, seed, step, rng_salt);
  return cudaGetLastError();
}
