// =============================================================================
// Skinny-M WMMA GEMM kernel for AMD RDNA3 (gfx1100)
//
// Optimized for small-M shapes (M <= 32). Uses BM=128 staging, BN=16 tiles:
//   - BN=16 maximizes grid parallelism along N (32 blocks for N=512)
//   - For K > 32, only the M warps needed for real rows compute and stage A
//   - For K <= 32, all 8 warps compute because the tiny target is launch/load
//     dominated and the branch/imbalance cost is not repaid
//   - Boundary-aware A loading zero-fills OOB rows (no wasted global reads)
//
// Tile design informed by hipblaslt profiling: hipblaslt uses MT64x16 with
// 128 threads on these shapes. Keeping 256 threads preserves B-load coverage
// and wave supply while the compute guard cuts large-K OOB WMMA work.
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
static constexpr int TINY_COMPUTE_WARPS = NUM_WARPS;
static constexpr int BLOCK_SIZE = NUM_WARPS * WARP_SIZE; // 256

// =============================================================================
// Skinny-M GEMM kernel
// =============================================================================

template <typename ElemT>
__global__ __launch_bounds__(BLOCK_SIZE, 3)
void gemm_wmma_skinny(
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

  // Eight warps are stacked in M direction; the K-dependent guard below
  // decides how many of them perform WMMA.
  const int warp_m = warp_id; // 0..7

  __shared__ ElemT smem_A[BM * SMEM_STRIDE];
  __shared__ ElemT smem_BT[BN * SMEM_STRIDE];

  f32x8 acc = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};

  const int frag_row = lane_id % 16;

  // -------------------------------------------------------------------------
  // Loading strategy for 256 threads:
  //
  // A tile: BM x BK = 128 x 32 elements.
  //   256 threads: row = tid/2 (0..127), half = tid%2 (0 or 1)
  //   Each thread loads 16 elements (2 x uint4) for one half of one row.
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
  const int large_k_compute_warps = (M + WM - 1) / WM;
  const int compute_warps =
      (K <= BK) ? TINY_COMPUTE_WARPS : large_k_compute_warps;
  const int active_a_rows = compute_warps * WM;

  for (int k_base = 0; k_base < K; k_base += BK) {

    // -----------------------------------------------------------------------
    // Load A tile with M and K bounds checking.
    // -----------------------------------------------------------------------
    {
      if (a_load_row < active_a_rows) {
        const int global_row = block_m + a_load_row;
        const int col_start = k_base + a_load_half * 16;
        ElemT *a_dst =
            smem_A + a_load_row * SMEM_STRIDE + a_load_half * 16;

        if (global_row < M && col_start + 16 <= K) {
          const ElemT *a_src =
              A + (int64_t)global_row * K + col_start;
          load_128(a_dst, a_src);
          load_128(a_dst + 8, a_src + 8);
        } else if (global_row < M) {
          #pragma unroll
          for (int c = 0; c < 16; c++) {
            int gc = col_start + c;
            a_dst[c] = (gc < K) ? A[(int64_t)global_row * K + gc]
                                 : ElemT(0.0f);
          }
        } else {
          zero_128(a_dst);
          zero_128(a_dst + 8);
        }
      }
    }

    // -----------------------------------------------------------------------
    // Load B tile with N and K bounds checking (transposed to [BN][BK]).
    // 16 groups of 2 K-rows each, 1 paired load per thread.
    // -----------------------------------------------------------------------
    {
      const int global_col = block_n + b_load_col;
      const int k_start = b_load_k_group * 2;
      ElemT *bt_dst =
          smem_BT + b_load_col * SMEM_STRIDE + k_start;

      if (global_col < N) {
        int row0 = k_base + k_start;
        int row1 = row0 + 1;
        ElemT v0 = (row0 < K)
            ? B[(int64_t)row0 * N + global_col]
            : ElemT(0.0f);
        ElemT v1 = (row1 < K)
            ? B[(int64_t)row1 * N + global_col]
            : ElemT(0.0f);
        uint packed;
        reinterpret_cast<ElemT *>(&packed)[0] = v0;
        reinterpret_cast<ElemT *>(&packed)[1] = v1;
        *reinterpret_cast<uint *>(bt_dst) = packed;
      } else {
        *reinterpret_cast<uint *>(bt_dst) = 0;
      }
    }

    __syncthreads();

    // -----------------------------------------------------------------------
    // WMMA compute: each warp computes 1x1 tile (16x16).
    // -----------------------------------------------------------------------
    if (warp_id < compute_warps) {
      #pragma unroll
      for (int ki = 0; ki < BK; ki += WMMA_K) {
        frag_t a_frag;
        int smem_row_a = warp_m * WM + frag_row;
        const ElemT *a_ptr = smem_A + smem_row_a * SMEM_STRIDE + ki;
        a_frag = *reinterpret_cast<const frag_t *>(a_ptr);

        frag_t b_frag;
        const ElemT *b_ptr = smem_BT + frag_row * SMEM_STRIDE + ki;
        b_frag = *reinterpret_cast<const frag_t *>(b_ptr);

        acc = Traits::wmma(a_frag, b_frag, acc);
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

    int base_m = block_m + warp_m * WM + row_in_frag;
    int base_n = block_n;

    if (warp_id < compute_warps && base_m < M) {
      #pragma unroll
      for (int j = 0; j < 8; j++) {
        int col = base_n + 2 * j + col_offset;
        if (col < N) {
          float val = alpha * acc[j];
          C[(int64_t)base_m * N + col] = ElemT(val);
        }
      }
    }
  }
}

// Explicit instantiations.
template __global__ void gemm_wmma_skinny<__hip_bfloat16>(
    const __hip_bfloat16 *, const __hip_bfloat16 *, __hip_bfloat16 *,
    int, int, int, float);
template __global__ void gemm_wmma_skinny<_Float16>(
    const _Float16 *, const _Float16 *, _Float16 *,
    int, int, int, float);
