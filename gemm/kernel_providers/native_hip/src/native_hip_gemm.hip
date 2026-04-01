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
//   K tile:      32 (2x WMMA_K — two WMMA passes per shared memory fill)
//   Warps/block: 4 (arranged 2x2), 128 threads total
//   Warp tile:   64 x 64 (4x4 WMMA tiles per warp)
//   WMMA tile:   16 x 16 x 16 (native hardware instruction)
//
// Memory strategy:
//   - A loaded to shared memory in row-major layout [BM][SMEM_STRIDE]
//   - B loaded to shared memory transposed as [BN][SMEM_STRIDE] for coalesced
//     fragment loads (each lane reads a contiguous row of B^T)
//   - Single shared memory buffer with padding for bank conflict avoidance
//
// Optimizations applied:
//   - Shared memory padding (SMEM_STRIDE = BK+8) to reduce LDS bank conflicts
//   - BK=32 (2x WMMA_K): halves syncthreads count by doing 2 WMMA passes per
//     shared memory tile, doubling compute-to-sync ratio
//   - VGPR limit hint (amdgpu_num_vgpr(192)) reduces 238->214 VGPRs, improving
//     occupancy from 6 to 7 waves per SIMD for better latency hiding
//   - Vectorized B transpose loads (paired 32-bit stores to LDS)
//   - Vectorized A loads (128-bit uint4 loads from global memory)
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
static constexpr int BK = 32; // 2x WMMA_K — two WMMA passes per tile load

// Shared memory padding to avoid bank conflicts.
// Stride BK+8=40 bf16 = 20 words = 80 bytes per row.
// (row*20)%32 has period 8, providing good bank diversity.
static constexpr int SMEM_PAD = 8;
static constexpr int SMEM_STRIDE = BK + SMEM_PAD; // 40 bf16 per row

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

// Limit VGPRs to improve occupancy. Without limit: 238 VGPRs -> 6 waves/SIMD.
// With limit 192: compiler achieves 214 VGPRs -> 7 waves/SIMD, better latency hiding.
__global__ __launch_bounds__(BLOCK_SIZE) __attribute__((amdgpu_num_vgpr(192)))
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
  // Shared memory: A tile [BM][SMEM_STRIDE] and B transposed tile [BN][SMEM_STRIDE]
  // -------------------------------------------------------------------------
  __shared__ __hip_bfloat16 smem_A[BM * SMEM_STRIDE];
  __shared__ __hip_bfloat16 smem_BT[BN * SMEM_STRIDE];

  // -------------------------------------------------------------------------
  // Accumulators: 4x4 grid of 16x16 F32 tiles per warp = 16 tiles total.
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

  // =========================================================================
  // Main K-loop: iterate over K in steps of BK=32
  // =========================================================================
  for (int k_base = 0; k_base < K; k_base += BK) {

    // -----------------------------------------------------------------------
    // Stage 1: Load A tile from global -> shared memory
    //
    // 128 threads, each loads one row of 32 bf16 (64 bytes) via four uint4 loads.
    // -----------------------------------------------------------------------
    {
      const int row = tid;
      const __hip_bfloat16 *a_src = A + (int64_t)(block_m + row) * K + k_base;
      __hip_bfloat16 *a_dst = smem_A + row * SMEM_STRIDE;

      *reinterpret_cast<uint4 *>(a_dst) =
          *reinterpret_cast<const uint4 *>(a_src);
      *reinterpret_cast<uint4 *>(a_dst + 8) =
          *reinterpret_cast<const uint4 *>(a_src + 8);
      *reinterpret_cast<uint4 *>(a_dst + 16) =
          *reinterpret_cast<const uint4 *>(a_src + 16);
      *reinterpret_cast<uint4 *>(a_dst + 24) =
          *reinterpret_cast<const uint4 *>(a_src + 24);
    }

    // -----------------------------------------------------------------------
    // Stage 2: Load B tile from global -> shared memory (transposed)
    //
    // B tile: BK x BN = 32 x 128 in global, stored as BN x BK = 128 x 32.
    // Thread tid loads column tid of B (row tid of B^T).
    // -----------------------------------------------------------------------
    {
      const int col = tid;
      __hip_bfloat16 *bt_dst = smem_BT + col * SMEM_STRIDE;

      #pragma unroll
      for (int k = 0; k < BK; k += 2) {
        __hip_bfloat16 v0 = B[(int64_t)(k_base + k) * N + (block_n + col)];
        __hip_bfloat16 v1 = B[(int64_t)(k_base + k + 1) * N + (block_n + col)];
        uint packed;
        reinterpret_cast<__hip_bfloat16 *>(&packed)[0] = v0;
        reinterpret_cast<__hip_bfloat16 *>(&packed)[1] = v1;
        *reinterpret_cast<uint *>(bt_dst + k) = packed;
      }
    }

    __syncthreads();

    // -----------------------------------------------------------------------
    // Stage 3: WMMA compute — 2 passes of WMMA_K=16 within BK=32
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
  // Store results: convert F32 accumulators to BF16 and write to global C
  //
  // WMMA C layout (gfx1100, wave32):
  //   Lane l < 16:  holds C[l][0], C[l][2], ..., C[l][14] (even columns)
  //   Lane l >= 16: holds C[l-16][1], C[l-16][3], ..., C[l-16][15] (odd cols)
  //   i.e., acc[j] = C[lane%16][2*j + (lane >= 16 ? 1 : 0)]
  //
  // Each lane writes its 8 bf16 values per WMMA tile directly to global C.
  // Adjacent lanes write to stride-2 addresses in the same row, which the
  // hardware can partially coalesce (32-byte cache lines cover 16 bf16).
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
