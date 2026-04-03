// =============================================================================
// Aligned 128x128 WMMA GEMM kernel for AMD RDNA3 (gfx1100)
//
// Requires M % 128 == 0, N % 128 == 0, K % 32 == 0.
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
// GEMM kernel
// =============================================================================

// 8 warps with smaller warp tiles: 120 VGPRs, 0 scratch, high occupancy.
// Hint 3 WGs/CU (max from LDS limit: 3*20480=61440 < 65536).
template <typename ElemT>
__global__ __launch_bounds__(BLOCK_SIZE, 3)
void gemm_wmma_128x128(
    const ElemT *__restrict__ A, // [M, K] row-major
    const ElemT *__restrict__ B, // [K, N] row-major
    ElemT *__restrict__ C,       // [M, N] row-major
    const int M, const int N, const int K,
    const float alpha) {

  using Traits = WmmaTraits<ElemT>;
  using frag_t = typename Traits::frag_type;

  // Block coordinates: each block computes a BM x BN output tile.
  const int block_m = blockIdx.x * BM;
  const int block_n = blockIdx.y * BN;

  // Thread identity within the block.
  const int tid = threadIdx.x;
  const int warp_id = tid / WARP_SIZE;
  const int lane_id = tid % WARP_SIZE;

  // Warp position in the 4x2 arrangement.
  const int warp_m = warp_id / WARPS_N; // 0..3
  const int warp_n = warp_id % WARPS_N; // 0..1

  // -------------------------------------------------------------------------
  // Shared memory: A tile [BM][SMEM_STRIDE] and B transposed tile [BN][SMEM_STRIDE]
  // -------------------------------------------------------------------------
  __shared__ ElemT smem_A[BM * SMEM_STRIDE];
  __shared__ ElemT smem_BT[BN * SMEM_STRIDE];

  // -------------------------------------------------------------------------
  // Accumulators: 2x4 grid of 16x16 F32 tiles per warp = 8 tiles total.
  // -------------------------------------------------------------------------
  f32x8 acc[WARP_TILES_M][WARP_TILES_N];
  #pragma unroll
  for (int tm = 0; tm < WARP_TILES_M; tm++) {
    #pragma unroll
    for (int tn = 0; tn < WARP_TILES_N; tn++) {
      acc[tm][tn] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
    }
  }

  const int frag_row = lane_id % 16;

  // -------------------------------------------------------------------------
  // Loading strategy for 256 threads:
  //
  // A tile: BM x BK = 128 x 32 elements. 256 threads, 2 per row.
  //   Thread tid: row = tid/2, half = tid%2
  //   Half 0: loads columns 0-15 (2 uint4), Half 1: columns 16-31 (2 uint4)
  //
  // B tile: BK x BN = 32 x 128 elements, transposed to [BN][BK] = [128][32].
  //   Thread tid: col = tid%128, group = tid/128
  //   Group 0: loads rows 0-15, Group 1: rows 16-31
  //   Each group: 8 paired loads per thread
  // -------------------------------------------------------------------------
  const int a_load_row = tid / 2;
  const int a_load_half = tid % 2;
  const int b_load_col = tid % 128;
  const int b_load_group = tid / 128;

  // =========================================================================
  // Main K-loop: iterate over K in steps of BK=32
  // =========================================================================
  for (int k_base = 0; k_base < K; k_base += BK) {

    // -----------------------------------------------------------------------
    // Stage 1: Load A tile from global -> shared memory
    //
    // 256 threads, 2 per row. Each loads 16 elements (32 bytes) via two uint4.
    // -----------------------------------------------------------------------
    {
      const ElemT *a_src =
          A + (int64_t)(block_m + a_load_row) * K + k_base + a_load_half * 16;
      ElemT *a_dst =
          smem_A + a_load_row * SMEM_STRIDE + a_load_half * 16;

      load_128(a_dst, a_src);
      load_128(a_dst + 8, a_src + 8);
    }

    // -----------------------------------------------------------------------
    // Stage 2: Load B tile from global -> shared memory (transposed)
    //
    // B tile: BK x BN = 32 x 128, stored transposed as [BN][BK] = [128][32].
    // 256 threads split into 2 groups of 128. Each group loads 16 rows of B
    // for all 128 columns.
    // -----------------------------------------------------------------------
    {
      ElemT *bt_dst =
          smem_BT + b_load_col * SMEM_STRIDE + b_load_group * 16;

      #pragma unroll
      for (int k = 0; k < 16; k += 2) {
        int row = b_load_group * 16 + k;
        ElemT v0 =
            B[(int64_t)(k_base + row) * N + (block_n + b_load_col)];
        ElemT v1 =
            B[(int64_t)(k_base + row + 1) * N + (block_n + b_load_col)];
        uint packed;
        reinterpret_cast<ElemT *>(&packed)[0] = v0;
        reinterpret_cast<ElemT *>(&packed)[1] = v1;
        *reinterpret_cast<uint *>(bt_dst + k) = packed;
      }
    }

    __syncthreads();

    // -----------------------------------------------------------------------
    // Stage 3: WMMA compute — 2 passes of WMMA_K=16 within BK=32
    // -----------------------------------------------------------------------
    #pragma unroll
    for (int ki = 0; ki < BK; ki += WMMA_K) {
      frag_t a_frag[WARP_TILES_M];
      #pragma unroll
      for (int tm = 0; tm < WARP_TILES_M; tm++) {
        int smem_row = warp_m * WM + tm * WMMA_M + frag_row;
        const ElemT *frag_ptr = smem_A + smem_row * SMEM_STRIDE + ki;
        a_frag[tm] = *reinterpret_cast<const frag_t *>(frag_ptr);
      }

      #pragma unroll
      for (int tn = 0; tn < WARP_TILES_N; tn++) {
        int smem_row = warp_n * WN + tn * WMMA_N + frag_row;
        const ElemT *frag_ptr = smem_BT + smem_row * SMEM_STRIDE + ki;
        frag_t b_frag = *reinterpret_cast<const frag_t *>(frag_ptr);

        #pragma unroll
        for (int tm = 0; tm < WARP_TILES_M; tm++) {
          acc[tm][tn] = Traits::wmma(a_frag[tm], b_frag, acc[tm][tn]);
        }
      }
    }

    __syncthreads();
  }

  // =========================================================================
  // Store results: convert F32 accumulators and write to global C
  //
  // WMMA C layout (gfx1100, wave32):
  //   Lane l < 16:  holds C[l][0], C[l][2], ..., C[l][14] (even columns)
  //   Lane l >= 16: holds C[l-16][1], C[l-16][3], ..., C[l-16][15] (odd cols)
  //   i.e., acc[j] = C[lane%16][2*j + (lane >= 16 ? 1 : 0)]
  //
  // Each lane writes its 8 values per WMMA tile directly to global C.
  // Adjacent lanes write to stride-2 addresses in the same row, which the
  // hardware can partially coalesce (32-byte cache lines cover 16 elements).
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

        #pragma unroll
        for (int j = 0; j < 8; j++) {
          int col = base_n + 2 * j + col_offset;
          float val = alpha * acc[tm][tn][j];
          C[(int64_t)base_m * N + col] = ElemT(val);
        }
      }
    }
  }
}

// Explicit instantiations.
template __global__ void gemm_wmma_128x128<__hip_bfloat16>(
    const __hip_bfloat16 *, const __hip_bfloat16 *, __hip_bfloat16 *,
    int, int, int, float);
template __global__ void gemm_wmma_128x128<_Float16>(
    const _Float16 *, const _Float16 *, _Float16 *,
    int, int, int, float);
