// =============================================================================
// Native HIP GEMM kernel using WMMA intrinsics for AMD RDNA3 (gfx1100)
//
// Computes: C = alpha * A @ B    (beta=0 assumed)
//   A: [M, K] row-major BF16
//   B: [K, N] row-major BF16
//   C: [M, N] row-major BF16
//   Accumulation in F32
//
// Kernel: gemm_wmma_128x128
//   Block tile:  128 x 128 (BM x BN)
//   K tile:      16 (matches WMMA_K)
//   Warps/block: 4 (arranged 2x2), 128 threads total
//   Warp tile:   64 x 64 (4x4 WMMA tiles per warp)
//   WMMA tile:   16 x 16 x 16 (native hardware instruction)
//
// Memory strategy:
//   - A loaded to shared memory in row-major layout [BM][BK]
//   - B loaded to shared memory transposed as [BN][BK] for coalesced
//     fragment loads (each lane reads a contiguous row of B^T)
//   - No double-buffering in v1 (single shared memory buffer per tile)
//
// WMMA register layout (gfx1100, wave32):
//   A fragment: lane l holds row (l%16) of the 16x16 tile, all 16 columns
//   B fragment: lane l holds column (l%16) of the 16x16 tile, all 16 rows
//     (equivalent to row l%16 of B^T)
//   C accumulator: lane l < 16 holds even columns of row l
//                  lane l >= 16 holds odd columns of row (l-16)
//     Specifically: acc[j] = C[l%16][2*j + (l>=16 ? 1 : 0)]
// =============================================================================

#include "native_hip_gemm.h"

#include <hip/hip_bf16.h>
#include <hip/hip_runtime.h>

#include <cstdio>

// =============================================================================
// Vector types for WMMA intrinsics
// =============================================================================

// BF16 fragments are passed as int16 vectors to the WMMA built-in.
// Each lane holds 16 BF16 values (one row of a 16x16 tile).
using bf16x16 = short __attribute__((ext_vector_type(16)));

// F32 accumulator: each lane holds 8 floats of the 16x16 output tile.
using f32x8 = float __attribute__((ext_vector_type(8)));

// =============================================================================
// Kernel configuration
// =============================================================================

// Block tile dimensions. The output tile computed by one thread block.
static constexpr int BM = 128;
static constexpr int BN = 128;
static constexpr int BK = 16; // Must equal WMMA_K

// WMMA hardware tile dimensions (fixed by the ISA).
static constexpr int WMMA_M = 16;
static constexpr int WMMA_N = 16;
static constexpr int WMMA_K = 16;

static constexpr int WARP_SIZE = 32;

// Warp tile: portion of the block tile computed by one warp.
// With 2x2 warp arrangement in a 128x128 block, each warp computes 64x64.
static constexpr int WM = 64;
static constexpr int WN = 64;

// Warp arrangement within the block tile.
static constexpr int WARPS_M = BM / WM; // 2
static constexpr int WARPS_N = BN / WN; // 2
static constexpr int NUM_WARPS = WARPS_M * WARPS_N; // 4
static constexpr int BLOCK_SIZE = NUM_WARPS * WARP_SIZE; // 128 threads

// Number of 16x16 WMMA tiles per warp tile.
static constexpr int WARP_TILES_M = WM / WMMA_M; // 4
static constexpr int WARP_TILES_N = WN / WMMA_N; // 4

// =============================================================================
// GEMM kernel
// =============================================================================

__global__ __launch_bounds__(BLOCK_SIZE)
void gemm_wmma_128x128(
    const __hip_bfloat16 *__restrict__ A, // [M, K] row-major
    const __hip_bfloat16 *__restrict__ B, // [K, N] row-major
    __hip_bfloat16 *__restrict__ C,       // [M, N] row-major
    const int M, const int N, const int K,
    const float alpha) {

  // Block coordinates: each block computes a BM x BN output tile.
  const int block_m = blockIdx.x * BM;
  const int block_n = blockIdx.y * BN;

  // Thread identity within the block.
  const int tid = threadIdx.x;
  const int warp_id = tid / WARP_SIZE;
  const int lane_id = tid % WARP_SIZE;

  // Warp position in the 2x2 arrangement.
  const int warp_m = warp_id / WARPS_N; // 0 or 1
  const int warp_n = warp_id % WARPS_N; // 0 or 1

  // -------------------------------------------------------------------------
  // Shared memory: A tile [BM][BK] and B transposed tile [BN][BK]
  //
  // Layout: row-major with BK stride (no padding in v1).
  // A: smem_A[row][k] for row in [0, BM), k in [0, BK)
  // BT: smem_BT[col][k] for col in [0, BN), k in [0, BK)
  //     where smem_BT[col][k] = B[k_base + k][block_n + col]
  // -------------------------------------------------------------------------
  __shared__ __hip_bfloat16 smem_A[BM * BK];
  __shared__ __hip_bfloat16 smem_BT[BN * BK];

  // -------------------------------------------------------------------------
  // Accumulators: 4x4 grid of 16x16 F32 tiles per warp = 16 tiles total.
  // Each tile stored as f32x8 (8 floats per lane).
  // Total per lane: 16 * 8 = 128 VGPRs for accumulators.
  // -------------------------------------------------------------------------
  f32x8 acc[WARP_TILES_M][WARP_TILES_N];
  #pragma unroll
  for (int tm = 0; tm < WARP_TILES_M; tm++) {
    #pragma unroll
    for (int tn = 0; tn < WARP_TILES_N; tn++) {
      acc[tm][tn] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
    }
  }

  // =========================================================================
  // Main K-loop: iterate over K in steps of BK=16
  // =========================================================================
  for (int k_base = 0; k_base < K; k_base += BK) {

    // -----------------------------------------------------------------------
    // Stage 1: Load A tile from global → shared memory
    //
    // A tile: BM x BK = 128 x 16 = 2048 bf16 elements.
    // With 128 threads, each thread loads 16 elements (one row of the tile).
    // Thread tid loads row tid: A[block_m + tid][k_base .. k_base + BK - 1]
    //
    // Global access: each thread reads 16 consecutive BF16 values along K.
    // This is coalesced within each warp (adjacent threads read adjacent
    // rows, which are BK=16 elements apart — a stride access pattern, but
    // the 16-element row read itself is contiguous).
    // -----------------------------------------------------------------------
    {
      const int row = tid; // Thread 0..127 → row 0..127
      const __hip_bfloat16 *a_src = A + (int64_t)(block_m + row) * K + k_base;
      __hip_bfloat16 *a_dst = smem_A + row * BK;

      // Vectorized 128-bit loads: 8 bf16 = 16 bytes = one uint4.
      *reinterpret_cast<uint4 *>(a_dst) =
          *reinterpret_cast<const uint4 *>(a_src);
      *reinterpret_cast<uint4 *>(a_dst + 8) =
          *reinterpret_cast<const uint4 *>(a_src + 8);
    }

    // -----------------------------------------------------------------------
    // Stage 2: Load B tile from global → shared memory (transposed)
    //
    // B tile: BK x BN = 16 x 128 elements in global memory.
    // Stored transposed in shared memory as BN x BK = 128 x 16.
    //
    // Thread tid loads column tid of B (i.e., row tid of B^T):
    //   smem_BT[tid][k] = B[k_base + k][block_n + tid]  for k = 0..15
    //
    // Each individual B[k][block_n + tid] load is coalesced across threads
    // (adjacent threads read adjacent N addresses for the same K row).
    // We issue BK=16 such coalesced loads.
    // -----------------------------------------------------------------------
    {
      const int col = tid; // Thread 0..127 → column 0..127 of B
      __hip_bfloat16 *bt_dst = smem_BT + col * BK;

      #pragma unroll
      for (int k = 0; k < BK; k++) {
        bt_dst[k] = B[(int64_t)(k_base + k) * N + (block_n + col)];
      }
    }

    __syncthreads();

    // -----------------------------------------------------------------------
    // Stage 3: WMMA compute
    //
    // Each warp loads fragments from shared memory and issues WMMA ops.
    // Fragment loading uses lane_id % 16 to select the row within each
    // 16x16 tile (lanes 16-31 duplicate lanes 0-15 as required by WMMA).
    //
    // For each WMMA tile (tm, tn) in the warp's 4x4 grid:
    //   A_frag[tm] = shared_A row [warp_m*64 + tm*16 + lane%16], all 16 cols
    //   B_frag[tn] = shared_BT row [warp_n*64 + tn*16 + lane%16], all 16 cols
    //   acc[tm][tn] += WMMA(A_frag[tm], B_frag[tn])
    //
    // A fragments are loaded once and reused across all tn (4x reuse).
    // B fragments are loaded once per tn and reused across all tm (4x reuse).
    // -----------------------------------------------------------------------
    const int frag_row = lane_id % 16;

    // Load all 4 A fragments for this warp's M tiles.
    bf16x16 a_frag[WARP_TILES_M];
    #pragma unroll
    for (int tm = 0; tm < WARP_TILES_M; tm++) {
      int smem_row = warp_m * WM + tm * WMMA_M + frag_row;
      const __hip_bfloat16 *frag_ptr = smem_A + smem_row * BK;
      a_frag[tm] = *reinterpret_cast<const bf16x16 *>(frag_ptr);
    }

    // Load B fragments and compute WMMA for each tile pair.
    #pragma unroll
    for (int tn = 0; tn < WARP_TILES_N; tn++) {
      int smem_row = warp_n * WN + tn * WMMA_N + frag_row;
      const __hip_bfloat16 *frag_ptr = smem_BT + smem_row * BK;
      bf16x16 b_frag = *reinterpret_cast<const bf16x16 *>(frag_ptr);

      #pragma unroll
      for (int tm = 0; tm < WARP_TILES_M; tm++) {
        acc[tm][tn] = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32(
            a_frag[tm], b_frag, acc[tm][tn]);
      }
    }

    __syncthreads();
  }

  // =========================================================================
  // Store results: convert F32 accumulators to BF16 and write to global C
  //
  // WMMA C layout (gfx1100, wave32):
  //   Lane l < 16:  holds C[l][0], C[l][2], ..., C[l][14] (even columns)
  //   Lane l >= 16: holds C[l-16][1], C[l-16][3], ..., C[l-16][15] (odd cols)
  //   i.e., acc[j] = C[lane%16][2*j + (lane >= 16 ? 1 : 0)]
  //
  // We write each accumulator element individually (scattered stores).
  // This is not coalesced but correct; future versions can use shared memory
  // to reorder into coalesced stores.
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
          C[(int64_t)base_m * N + col] = __hip_bfloat16(val);
        }
      }
    }
  }
}

// =============================================================================
// Host launch
// =============================================================================

namespace kernelgen {
namespace gemm {
namespace native_hip {

GemmResult run(const GemmConfig &config, void *A, void *B, void *C,
               int warmup_runs, int timed_runs) {
  GemmResult result;

  // Validate constraints for this kernel.
  if (config.transA || config.transB) {
    result.error = "native_hip: only NN layout (transA=false, transB=false) supported";
    return result;
  }
  if (config.dtype_A != "bf16" || config.dtype_B != "bf16" || config.dtype_C != "bf16") {
    result.error = "native_hip: only BF16 data types supported";
    return result;
  }
  if (config.M % BM != 0 || config.N % BN != 0) {
    result.error = "native_hip: M must be multiple of " + std::to_string(BM) +
                   ", N must be multiple of " + std::to_string(BN);
    return result;
  }
  if (config.K % BK != 0) {
    result.error = "native_hip: K must be multiple of " + std::to_string(BK);
    return result;
  }

  const int M = config.M;
  const int N = config.N;
  const int K = config.K;
  const float alpha = config.alpha;

  // Grid: one block per BM x BN output tile.
  dim3 grid(M / BM, N / BN);
  dim3 block(BLOCK_SIZE);

  // Create stream and events for timing.
  hipStream_t stream;
  hipStreamCreate(&stream);
  hipEvent_t start, stop;
  hipEventCreate(&start);
  hipEventCreate(&stop);

  auto *dA = static_cast<const __hip_bfloat16 *>(A);
  auto *dB = static_cast<const __hip_bfloat16 *>(B);
  auto *dC = static_cast<__hip_bfloat16 *>(C);

  // Warmup runs.
  for (int i = 0; i < warmup_runs; i++) {
    gemm_wmma_128x128<<<grid, block, 0, stream>>>(dA, dB, dC, M, N, K, alpha);
  }
  hipStreamSynchronize(stream);

  // Check for launch errors.
  hipError_t err = hipGetLastError();
  if (err != hipSuccess) {
    result.error = std::string("kernel launch failed: ") + hipGetErrorString(err);
    hipEventDestroy(start);
    hipEventDestroy(stop);
    hipStreamDestroy(stream);
    return result;
  }

  // Timed runs.
  hipEventRecord(start, stream);
  for (int i = 0; i < timed_runs; i++) {
    gemm_wmma_128x128<<<grid, block, 0, stream>>>(dA, dB, dC, M, N, K, alpha);
  }
  hipEventRecord(stop, stream);
  hipEventSynchronize(stop);

  float elapsed_ms = 0;
  hipEventElapsedTime(&elapsed_ms, start, stop);
  result.kernel_time_us = (elapsed_ms * 1000.0) / timed_runs;
  result.success = true;

  hipEventDestroy(start);
  hipEventDestroy(stop);
  hipStreamDestroy(stream);

  return result;
}

} // namespace native_hip
} // namespace gemm
} // namespace kernelgen
