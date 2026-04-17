// =============================================================================
// Small/skinny WMMA GEMM balance point for AMD RDNA3 (gfx1100)
//
// Computes a 64x16 output tile per CTA with four 16x16 WMMA warps and
// 128 threads. This is intended for BF16/f32 small-N and skinny-M shapes where
// the 128x128 boundary kernel underfills the GPU and the older skinny kernel
// spends too many warps/threads on rows outside M.
//
// The tile keeps BN=16 to preserve useful N-splitting, uses a smaller M tile
// to avoid padded 128-row work, and gates compute/A loading by the valid number
// of 16-row M tiles in the CTA. It handles arbitrary M/N/K boundaries.
//
// See native_hip_gemm.hip and kernelgen_gemm.hip for host dispatch logic.
// =============================================================================

#include "kernelgen_wmma_utils.h"

// =============================================================================
// Kernel configuration (local to this translation unit)
// =============================================================================

static constexpr int BM = 64;
static constexpr int BN = 16;
static constexpr int BK = 32;

static constexpr int SMEM_PAD = 8;
static constexpr int SMEM_STRIDE = BK + SMEM_PAD; // 40

static constexpr int WARP_SIZE = 32;

static constexpr int WM = 16;
static constexpr int WN = 16;

static constexpr int WARPS_M = BM / WM; // 4
static constexpr int WARPS_N = BN / WN; // 1
static constexpr int NUM_WARPS = WARPS_M * WARPS_N; // 4
static constexpr int BLOCK_SIZE = NUM_WARPS * WARP_SIZE; // 128

// =============================================================================
// Small/skinny GEMM kernel
// =============================================================================

template <typename ElemT>
__global__ __launch_bounds__(BLOCK_SIZE, 4)
void gemm_wmma_64x16_small_skinny(
    const ElemT *__restrict__ A, // [M, K] row-major
    const ElemT *__restrict__ B, // [K, N] row-major
    ElemT *__restrict__ C,       // [M, N] row-major
    const int M, const int N, const int K,
    const float alpha) {

  using Traits = WmmaTraits<ElemT>;
  using frag_t = typename Traits::frag_type;

  const int block_m = blockIdx.x * BM;
  const int block_n = blockIdx.y * BN;

  const int tid = threadIdx.x;
  const int warp_id = tid / WARP_SIZE;
  const int lane_id = tid % WARP_SIZE;

  const int warp_m = warp_id;
  const int frag_row = lane_id % 16;

  const int remaining_m = M - block_m;
  const int active_warps =
      (remaining_m >= BM) ? WARPS_M
                          : ((remaining_m + WM - 1) / WM);
  const int active_a_rows = active_warps * WM;

  __shared__ ElemT smem_A[BM * SMEM_STRIDE];
  __shared__ ElemT smem_BT[BN * SMEM_STRIDE];

  f32x8 acc = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};

  // A tile: 64x32. Two threads per row, each loading a 16-element half-row.
  const int a_load_row = tid / 2;  // 0..63
  const int a_load_half = tid % 2; // 0 or 1

  // B tile: 32x16, transposed to BT[16][40]. Eight K groups of four rows.
  const int b_load_col = tid % BN;          // 0..15
  const int b_load_k_group = tid / BN;      // 0..7
  const int b_load_k_start = b_load_k_group * 4;

  for (int k_base = 0; k_base < K; k_base += BK) {
    // -----------------------------------------------------------------------
    // Load A tile. Rows outside the active 16-row WMMA tiles are not consumed,
    // so skip their global work instead of zero-filling unused rows.
    // -----------------------------------------------------------------------
    if (a_load_row < active_a_rows) {
      const int global_row = block_m + a_load_row;
      const int col_start = k_base + a_load_half * 16;
      ElemT *a_dst =
          smem_A + a_load_row * SMEM_STRIDE + a_load_half * 16;

      if (global_row < M && col_start + 16 <= K) {
        const ElemT *a_src = A + (int64_t)global_row * K + col_start;
        load_128(a_dst, a_src);
        load_128(a_dst + 8, a_src + 8);
      } else if (global_row < M) {
        #pragma unroll
        for (int c = 0; c < 16; c++) {
          const int gc = col_start + c;
          a_dst[c] = (gc < K) ? A[(int64_t)global_row * K + gc]
                               : ElemT(0.0f);
        }
      }
    }

    // -----------------------------------------------------------------------
    // Load B tile with N/K bounds checking.
    // -----------------------------------------------------------------------
    {
      const int global_col = block_n + b_load_col;
      ElemT *bt_dst =
          smem_BT + b_load_col * SMEM_STRIDE + b_load_k_start;

      #pragma unroll
      for (int p = 0; p < 4; p += 2) {
        if (global_col < N) {
          const int row0 = k_base + b_load_k_start + p;
          const int row1 = row0 + 1;
          ElemT v0 = (row0 < K)
              ? B[(int64_t)row0 * N + global_col]
              : ElemT(0.0f);
          ElemT v1 = (row1 < K)
              ? B[(int64_t)row1 * N + global_col]
              : ElemT(0.0f);
          uint packed;
          reinterpret_cast<ElemT *>(&packed)[0] = v0;
          reinterpret_cast<ElemT *>(&packed)[1] = v1;
          *reinterpret_cast<uint *>(bt_dst + p) = packed;
        } else {
          *reinterpret_cast<uint *>(bt_dst + p) = 0;
        }
      }
    }

    __syncthreads();

    // -----------------------------------------------------------------------
    // WMMA compute: one 16x16 output tile per active warp.
    // -----------------------------------------------------------------------
    if (warp_id < active_warps) {
      const int k_remaining = K - k_base;
      const int k_compute_limit = (k_remaining < BK) ? k_remaining : BK;
      #pragma unroll
      for (int ki = 0; ki < BK; ki += WMMA_K) {
        if (ki < k_compute_limit) {
          const ElemT *a_ptr =
              smem_A + (warp_m * WM + frag_row) * SMEM_STRIDE + ki;
          frag_t a_frag = *reinterpret_cast<const frag_t *>(a_ptr);

          const ElemT *b_ptr = smem_BT + frag_row * SMEM_STRIDE + ki;
          frag_t b_frag = *reinterpret_cast<const frag_t *>(b_ptr);

          acc = Traits::wmma(a_frag, b_frag, acc);
        }
      }
    }

    __syncthreads();
  }

  // =========================================================================
  // Store results with M/N bounds checking.
  // =========================================================================
  if (warp_id < active_warps) {
    const int row_in_frag = lane_id % 16;
    const int col_offset = (lane_id < 16) ? 0 : 1;
    const int row = block_m + warp_m * WM + row_in_frag;

    if (row < M) {
      #pragma unroll
      for (int j = 0; j < 8; j++) {
        const int col = block_n + 2 * j + col_offset;
        if (col < N) {
          const float val = alpha * acc[j];
          C[(int64_t)row * N + col] = ElemT(val);
        }
      }
    }
  }
}

// Explicit instantiations.
template __global__ void gemm_wmma_64x16_small_skinny<__hip_bfloat16>(
    const __hip_bfloat16 *, const __hip_bfloat16 *, __hip_bfloat16 *,
    int, int, int, float);
template __global__ void gemm_wmma_64x16_small_skinny<_Float16>(
    const _Float16 *, const _Float16 *, _Float16 *,
    int, int, int, float);
