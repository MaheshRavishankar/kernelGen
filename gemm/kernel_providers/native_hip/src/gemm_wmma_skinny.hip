// =============================================================================
// Skinny-M WMMA GEMM kernel for AMD RDNA3 (gfx1100)
//
// Optimized for small-M shapes (M <= 32). Uses BM=128, BN=16 tiles:
//   - BN=16 maximizes grid parallelism along N (32 blocks for N=512)
//   - BM=128 with 8 warps gives 256 waves total for latency hiding
//   - For M=16, warps 1-7 compute OOB rows (waste compute, not bandwidth)
//   - Boundary-aware A loading zero-fills OOB rows (no wasted global reads)
//
// Tile design informed by hipblaslt profiling: hipblaslt uses MT64x16 with
// 128 threads on these shapes, achieving 32 blocks and ~2 TFLOPS. Our version
// uses BM=128 (vs 64) with 256 threads for 2x more waves to better hide
// memory latency on gfx1100.
//
// See native_hip_gemm.hip for host dispatch logic.
// =============================================================================

#include "kernelgen_wmma_utils.h"

// =============================================================================
// Kernel configuration (local to this translation unit)
// =============================================================================

static constexpr int BM = 128;
static constexpr int BN = 16;
static constexpr int BK = 32;

static constexpr int SMEM_PAD = 8;
static constexpr int SMEM_STRIDE = BK + SMEM_PAD; // 40

static constexpr int WARP_SIZE = 32;

static constexpr int WM = 16;
static constexpr int WN = 16;

static constexpr int WARPS_M = BM / WM; // 8
static constexpr int WARPS_N = BN / WN; // 1
static constexpr int NUM_WARPS = WARPS_M * WARPS_N; // 8
static constexpr int BLOCK_SIZE = NUM_WARPS * WARP_SIZE; // 256

// =============================================================================
// Skinny-M GEMM kernel
// =============================================================================

__global__ __launch_bounds__(BLOCK_SIZE, 3)
void gemm_wmma_skinny(
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

  // 8 warps stacked in M direction.
  const int warp_m = warp_id; // 0..7

  __shared__ __hip_bfloat16 smem_A[BM * SMEM_STRIDE];
  __shared__ __hip_bfloat16 smem_BT[BN * SMEM_STRIDE];

  f32x8 acc = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};

  const int frag_row = lane_id % 16;

  // -------------------------------------------------------------------------
  // Loading strategy for 256 threads:
  //
  // A tile: BM x BK = 128 x 32 bf16.
  //   256 threads: row = tid/2 (0..127), half = tid%2 (0 or 1)
  //   Each thread loads 16 bf16 (2 x uint4) for one half of one row.
  //   OOB rows (row >= M) zero-fill via boundary check.
  //
  // B tile: BK x BN = 32 x 16, transposed to BT[16][40].
  //   256 threads: col = tid%16 (0..15), k_group = tid/16 (0..15)
  //   16 groups of 2 K-rows each. Each thread does 1 paired load.
  // -------------------------------------------------------------------------
  const int a_load_row = tid / 2;       // 0..127
  const int a_load_half = tid % 2;      // 0 or 1
  const int b_load_col = tid % 16;      // 0..15
  const int b_load_k_group = tid / 16;  // 0..15

  for (int k_base = 0; k_base < K; k_base += BK) {

    // -----------------------------------------------------------------------
    // Load A tile with M and K bounds checking.
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
    // 16 groups of 2 K-rows each, 1 paired load per thread.
    // -----------------------------------------------------------------------
    {
      const int global_col = block_n + b_load_col;
      const int k_start = b_load_k_group * 2;
      __hip_bfloat16 *bt_dst =
          smem_BT + b_load_col * SMEM_STRIDE + k_start;

      if (global_col < N) {
        int row0 = k_base + k_start;
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
        *reinterpret_cast<uint *>(bt_dst) = packed;
      } else {
        *reinterpret_cast<uint *>(bt_dst) = 0;
      }
    }

    __syncthreads();

    // -----------------------------------------------------------------------
    // WMMA compute: each warp computes 1x1 tile (16x16).
    // -----------------------------------------------------------------------
    #pragma unroll
    for (int ki = 0; ki < BK; ki += WMMA_K) {
      bf16x16 a_frag;
      int smem_row_a = warp_m * WM + frag_row;
      const __hip_bfloat16 *a_ptr = smem_A + smem_row_a * SMEM_STRIDE + ki;
      a_frag = *reinterpret_cast<const bf16x16 *>(a_ptr);

      bf16x16 b_frag;
      const __hip_bfloat16 *b_ptr = smem_BT + frag_row * SMEM_STRIDE + ki;
      b_frag = *reinterpret_cast<const bf16x16 *>(b_ptr);

      acc = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32(a_frag, b_frag, acc);
    }

    __syncthreads();
  }

  // =========================================================================
  // Store results with M and N bounds checking.
  // =========================================================================
  {
    const int row_in_frag = lane_id % 16;
    const int col_offset = (lane_id < 16) ? 0 : 1;

    int base_m = block_m + warp_m * WM + row_in_frag;
    int base_n = block_n;

    if (base_m < M) {
      #pragma unroll
      for (int j = 0; j < 8; j++) {
        int col = base_n + 2 * j + col_offset;
        if (col < N) {
          float val = alpha * acc[j];
          C[(int64_t)base_m * N + col] = __hip_bfloat16(val);
        }
      }
    }
  }
}
