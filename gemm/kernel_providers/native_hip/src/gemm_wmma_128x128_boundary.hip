// =============================================================================
// Boundary-aware 128x128 WMMA GEMM kernel for AMD RDNA3 (gfx1100)
//
// Handles arbitrary M, N, K (not just multiples of 128/128/32).
// Edge tiles load zeros for OOB positions; C stores skip OOB positions.
// See native_hip_gemm.hip for host dispatch logic.
// =============================================================================

#include "kernelgen_wmma_utils.h"

// =============================================================================
// Kernel configuration (local to this translation unit)
// =============================================================================

static constexpr int BM = 128;
static constexpr int BN = 128;
static constexpr int BK = 32;

static constexpr int SMEM_PAD = 8;
static constexpr int SMEM_STRIDE = BK + SMEM_PAD;

static constexpr int WARP_SIZE = 32;

static constexpr int WM = 32;
static constexpr int WN = 64;

static constexpr int WARPS_M = BM / WM;
static constexpr int WARPS_N = BN / WN;
static constexpr int NUM_WARPS = WARPS_M * WARPS_N;
static constexpr int BLOCK_SIZE = NUM_WARPS * WARP_SIZE;

static constexpr int WARP_TILES_M = WM / WMMA_M;
static constexpr int WARP_TILES_N = WN / WMMA_N;

// =============================================================================
// Boundary-aware GEMM kernel
// =============================================================================

__global__ __launch_bounds__(BLOCK_SIZE, 3)
void gemm_wmma_128x128_boundary(
    const __hip_bfloat16 *__restrict__ A, // [M, K] row-major
    const __hip_bfloat16 *__restrict__ B, // [K, N] row-major
    __hip_bfloat16 *__restrict__ C,       // [M, N] row-major
    const int M, const int N, const int K,
    const float alpha) {

  const int block_m = blockIdx.x * BM;
  const int block_n = blockIdx.y * BN;

  const int tid = threadIdx.x;
  const int warp_id = tid / WARP_SIZE;
  const int lane_id = tid % WARP_SIZE;

  const int warp_m = warp_id / WARPS_N;
  const int warp_n = warp_id % WARPS_N;

  __shared__ __hip_bfloat16 smem_A[BM * SMEM_STRIDE];
  __shared__ __hip_bfloat16 smem_BT[BN * SMEM_STRIDE];

  f32x8 acc[WARP_TILES_M][WARP_TILES_N];
  #pragma unroll
  for (int tm = 0; tm < WARP_TILES_M; tm++) {
    #pragma unroll
    for (int tn = 0; tn < WARP_TILES_N; tn++) {
      acc[tm][tn] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
    }
  }

  const int frag_row = lane_id % 16;

  const int a_load_row = tid / 2;
  const int a_load_half = tid % 2;
  const int b_load_col = tid % 128;
  const int b_load_group = tid / 128;

  for (int k_base = 0; k_base < K; k_base += BK) {

    // -----------------------------------------------------------------------
    // Load A tile with M and K bounds checking.
    // Use vectorized loads when the full 16-element span is in-bounds.
    // -----------------------------------------------------------------------
    {
      const int global_row = block_m + a_load_row;
      const int col_start = k_base + a_load_half * 16;
      __hip_bfloat16 *a_dst =
          smem_A + a_load_row * SMEM_STRIDE + a_load_half * 16;

      if (global_row < M && col_start + 16 <= K) {
        const __hip_bfloat16 *a_src =
            A + (int64_t)global_row * K + col_start;
        load_128(a_dst, a_src);
        load_128(a_dst + 8, a_src + 8);
      } else if (global_row < M) {
        #pragma unroll
        for (int c = 0; c < 16; c++) {
          int gc = col_start + c;
          a_dst[c] = (gc < K) ? A[(int64_t)global_row * K + gc]
                               : __hip_bfloat16(0.0f);
        }
      } else {
        zero_128(a_dst);
        zero_128(a_dst + 8);
      }
    }

    // -----------------------------------------------------------------------
    // Load B tile with N and K bounds checking (transposed to [BN][BK]).
    // -----------------------------------------------------------------------
    {
      const int global_col = block_n + b_load_col;
      __hip_bfloat16 *bt_dst =
          smem_BT + b_load_col * SMEM_STRIDE + b_load_group * 16;

      if (global_col < N) {
        #pragma unroll
        for (int k = 0; k < 16; k += 2) {
          int row0 = k_base + b_load_group * 16 + k;
          int row1 = row0 + 1;
          __hip_bfloat16 v0 = (row0 < K)
              ? B[(int64_t)row0 * N + global_col]
              : __hip_bfloat16(0.0f);
          __hip_bfloat16 v1 = (row1 < K)
              ? B[(int64_t)row1 * N + global_col]
              : __hip_bfloat16(0.0f);
          uint packed;
          reinterpret_cast<__hip_bfloat16 *>(&packed)[0] = v0;
          reinterpret_cast<__hip_bfloat16 *>(&packed)[1] = v1;
          *reinterpret_cast<uint *>(bt_dst + k) = packed;
        }
      } else {
        #pragma unroll
        for (int k = 0; k < 16; k += 2) {
          *reinterpret_cast<uint *>(bt_dst + k) = 0;
        }
      }
    }

    __syncthreads();

    // -----------------------------------------------------------------------
    // WMMA compute — identical to the aligned kernel.
    // Shared memory has been properly zero-filled for OOB positions.
    // -----------------------------------------------------------------------
    #pragma unroll
    for (int ki = 0; ki < BK; ki += WMMA_K) {
      bf16x16 a_frag[WARP_TILES_M];
      #pragma unroll
      for (int tm = 0; tm < WARP_TILES_M; tm++) {
        int smem_row = warp_m * WM + tm * WMMA_M + frag_row;
        const __hip_bfloat16 *frag_ptr = smem_A + smem_row * SMEM_STRIDE + ki;
        a_frag[tm] = *reinterpret_cast<const bf16x16 *>(frag_ptr);
      }

      #pragma unroll
      for (int tn = 0; tn < WARP_TILES_N; tn++) {
        int smem_row = warp_n * WN + tn * WMMA_N + frag_row;
        const __hip_bfloat16 *frag_ptr = smem_BT + smem_row * SMEM_STRIDE + ki;
        bf16x16 b_frag = *reinterpret_cast<const bf16x16 *>(frag_ptr);

        #pragma unroll
        for (int tm = 0; tm < WARP_TILES_M; tm++) {
          acc[tm][tn] = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32(
              a_frag[tm], b_frag, acc[tm][tn]);
        }
      }
    }

    __syncthreads();
  }

  // =========================================================================
  // Store results with M and N bounds checking.
  // =========================================================================
  {
    const int row_in_frag = lane_id % 16;
    const int col_offset = (lane_id < 16) ? 0 : 1;

    #pragma unroll
    for (int tm = 0; tm < WARP_TILES_M; tm++) {
      #pragma unroll
      for (int tn = 0; tn < WARP_TILES_N; tn++) {
        int base_m = block_m + warp_m * WM + tm * WMMA_M + row_in_frag;
        int base_n = block_n + warp_n * WN + tn * WMMA_N;

        if (base_m < M) {
          #pragma unroll
          for (int j = 0; j < 8; j++) {
            int col = base_n + 2 * j + col_offset;
            if (col < N) {
              float val = alpha * acc[tm][tn][j];
              C[(int64_t)base_m * N + col] = __hip_bfloat16(val);
            }
          }
        }
      }
    }
  }
}
