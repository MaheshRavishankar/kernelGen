// =============================================================================
// Low-K streaming GEMM kernel for AMD RDNA3 (gfx1100)
//
// Optimized for memory-bound BF16/F16 GEMMs with K <= 32 and non-skinny M.
// The WMMA boundary kernel is a poor balance point for these shapes because it
// pads K to 32, stages a 128x128 tile, and stores C through the WMMA fragment
// lane layout. This kernel instead stages one 16x256 output tile, reuses B from
// LDS across 16 rows, and has each thread produce adjacent column vectors so C
// stores are contiguous.
//
// See native_hip_gemm.hip and kernelgen_gemm.hip for host dispatch logic.
// =============================================================================

#include "kernelgen_wmma_utils.h"

static constexpr int LOWK_BM = 16;
static constexpr int LOWK_BN = 256;
static constexpr int LOWK_MAX_K = 32;
static constexpr int LOWK_VEC_N = 16;
static constexpr int LOWK_COL_GROUPS = LOWK_BN / LOWK_VEC_N;
static constexpr int LOWK_BLOCK_SIZE = LOWK_BM * LOWK_COL_GROUPS;

static_assert(LOWK_BLOCK_SIZE == 256, "low-K kernel expects 256 threads");
static_assert(LOWK_VEC_N % 8 == 0, "low-K vectors must use 128-bit chunks");

template <typename ElemT>
__device__ __forceinline__ void load_vec(ElemT *dst, const ElemT *src) {
#pragma unroll
  for (int i = 0; i < LOWK_VEC_N; i += 8) {
    load_128(dst + i, src + i);
  }
}

template <typename ElemT>
__device__ __forceinline__ void store_vec(ElemT *dst, const float *values) {
  uint4 packed[LOWK_VEC_N / 8];
  ElemT *out = reinterpret_cast<ElemT *>(packed);
#pragma unroll
  for (int i = 0; i < LOWK_VEC_N; ++i) {
    out[i] = ElemT(values[i]);
  }
#pragma unroll
  for (int i = 0; i < LOWK_VEC_N; i += 8) {
    *reinterpret_cast<uint4 *>(dst + i) = packed[i / 8];
  }
}

template <typename ElemT>
__global__ __launch_bounds__(LOWK_BLOCK_SIZE, 4)
void gemm_low_k_stream(
    const ElemT *__restrict__ A, // [M, K] row-major
    const ElemT *__restrict__ B, // [K, N] row-major
    ElemT *__restrict__ C,       // [M, N] row-major
    const int M, const int N, const int K,
    const float alpha) {

  const int block_m = blockIdx.x * LOWK_BM;
  const int block_n = blockIdx.y * LOWK_BN;

  const int tid = threadIdx.x;
  const int row_local = tid / LOWK_COL_GROUPS;
  const int col_group = tid % LOWK_COL_GROUPS;
  const int col_local = col_group * LOWK_VEC_N;
  const bool aligned_n = (N % 8) == 0;

  __shared__ ElemT smem_A[LOWK_BM * LOWK_MAX_K];
  __shared__ ElemT smem_B[LOWK_MAX_K * LOWK_BN];

  // Stage A. K is runtime-dynamic but dispatch keeps it <= LOWK_MAX_K.
  const int a_elems = LOWK_BM * K;
  for (int idx = tid; idx < a_elems; idx += LOWK_BLOCK_SIZE) {
    const int r = idx / K;
    const int k = idx - r * K;
    const int global_row = block_m + r;
    smem_A[r * LOWK_MAX_K + k] =
        (global_row < M) ? A[(int64_t)global_row * K + k] : ElemT(0.0f);
  }

  // Stage B in contiguous column vectors. The scalar tail keeps the kernel
  // correct for edge tiles.
  const int b_vecs = K * LOWK_COL_GROUPS;
  for (int idx = tid; idx < b_vecs; idx += LOWK_BLOCK_SIZE) {
    const int k = idx / LOWK_COL_GROUPS;
    const int group = idx - k * LOWK_COL_GROUPS;
    const int col = group * LOWK_VEC_N;
    const int global_col = block_n + col;
    ElemT *dst = smem_B + k * LOWK_BN + col;

    if (aligned_n && global_col + LOWK_VEC_N <= N) {
      const ElemT *src = B + (int64_t)k * N + global_col;
      load_vec(dst, src);
    } else {
#pragma unroll
      for (int i = 0; i < LOWK_VEC_N; ++i) {
        const int c = global_col + i;
        dst[i] = (c < N) ? B[(int64_t)k * N + c] : ElemT(0.0f);
      }
    }
  }

  __syncthreads();

  const int row = block_m + row_local;
  const int col = block_n + col_local;
  float acc[LOWK_VEC_N] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};

  if (row < M) {
    for (int k = 0; k < K; ++k) {
      const float a = static_cast<float>(smem_A[row_local * LOWK_MAX_K + k]);
      const ElemT *b = smem_B + k * LOWK_BN + col_local;
#pragma unroll
      for (int i = 0; i < LOWK_VEC_N; ++i) {
        acc[i] += a * static_cast<float>(b[i]);
      }
    }
  }

  if (aligned_n && row < M && col + LOWK_VEC_N <= N) {
    float scaled[LOWK_VEC_N];
#pragma unroll
    for (int i = 0; i < LOWK_VEC_N; ++i) {
      scaled[i] = alpha * acc[i];
    }
    store_vec(C + (int64_t)row * N + col, scaled);
  } else if (row < M) {
#pragma unroll
    for (int i = 0; i < LOWK_VEC_N; ++i) {
      const int c = col + i;
      if (c < N) {
        C[(int64_t)row * N + c] = ElemT(alpha * acc[i]);
      }
    }
  }
}

template __global__ void gemm_low_k_stream<__hip_bfloat16>(
    const __hip_bfloat16 *, const __hip_bfloat16 *, __hip_bfloat16 *,
    int, int, int, float);
template __global__ void gemm_low_k_stream<_Float16>(
    const _Float16 *, const _Float16 *, _Float16 *,
    int, int, int, float);
