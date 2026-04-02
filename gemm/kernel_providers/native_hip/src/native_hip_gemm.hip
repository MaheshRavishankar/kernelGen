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
//   Warps/block: 8 (arranged 4x2), 256 threads total
//   Warp tile:   32 x 64 (2x4 WMMA tiles per warp)
//   WMMA tile:   16 x 16 x 16 (native hardware instruction)
//
// Memory strategy:
//   - A loaded to shared memory in row-major layout [BM][SMEM_STRIDE]
//   - B loaded to shared memory transposed as [BN][SMEM_STRIDE] for coalesced
//     fragment loads (each lane reads a contiguous row of B^T)
//   - Single shared memory buffer with padding for bank conflict avoidance
//
// Optimizations applied:
//   - 8 warps with 32x64 warp tiles: halves accumulators per warp (64 vs 128
//     VGPRs), achieving 120 VGPRs / 0 scratch for high occupancy
//   - Shared memory padding (SMEM_STRIDE = BK+8) to reduce LDS bank conflicts
//   - BK=32 (2x WMMA_K): two WMMA passes per shared memory tile
//   - 256 threads: halves per-thread global load work vs 128-thread version
//   - Vectorized A loads (128-bit uint4 loads from global memory)
//   - Vectorized B transpose loads (paired 32-bit stores to LDS)
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

// Load/store 128 bits in a single global_load_b128 instruction.
template <typename T>
__device__ __forceinline__ void load_128(T *dst, const T *src) {
  *reinterpret_cast<uint4 *>(dst) = *reinterpret_cast<const uint4 *>(src);
}

// Zero-fill 128 bits in a single instruction.
template <typename T>
__device__ __forceinline__ void zero_128(T *dst) {
  *reinterpret_cast<uint4 *>(dst) = make_uint4(0, 0, 0, 0);
}

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
// With 4x2 warp arrangement in a 128x128 block, each warp computes 32x64.
static constexpr int WM = 32;
static constexpr int WN = 64;

// Warp arrangement within the block tile.
static constexpr int WARPS_M = BM / WM; // 4
static constexpr int WARPS_N = BN / WN; // 2
static constexpr int NUM_WARPS = WARPS_M * WARPS_N; // 8
static constexpr int BLOCK_SIZE = NUM_WARPS * WARP_SIZE; // 256 threads

// Number of 16x16 WMMA tiles per warp tile.
static constexpr int WARP_TILES_M = WM / WMMA_M; // 2
static constexpr int WARP_TILES_N = WN / WMMA_N; // 4

// =============================================================================
// GEMM kernel
// =============================================================================

// 8 warps with smaller warp tiles: 120 VGPRs, 0 scratch, high occupancy.
// Hint 3 WGs/CU (max from LDS limit: 3*20480=61440 < 65536).
__global__ __launch_bounds__(BLOCK_SIZE, 3)
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

  // Warp position in the 4x2 arrangement.
  const int warp_m = warp_id / WARPS_N; // 0..3
  const int warp_n = warp_id % WARPS_N; // 0..1

  // -------------------------------------------------------------------------
  // Shared memory: A tile [BM][SMEM_STRIDE] and B transposed tile [BN][SMEM_STRIDE]
  // -------------------------------------------------------------------------
  __shared__ __hip_bfloat16 smem_A[BM * SMEM_STRIDE];
  __shared__ __hip_bfloat16 smem_BT[BN * SMEM_STRIDE];

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
  // A tile: BM x BK = 128 x 32 bf16. 256 threads, 2 per row.
  //   Thread tid: row = tid/2, half = tid%2
  //   Half 0: loads columns 0-15 (2 uint4), Half 1: columns 16-31 (2 uint4)
  //
  // B tile: BK x BN = 32 x 128 bf16, transposed to [BN][BK] = [128][32].
  //   Thread tid: col = tid%128, group = tid/128
  //   Group 0: loads rows 0-15, Group 1: rows 16-31
  //   Each group: 8 paired bf16 loads per thread
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
    // 256 threads, 2 per row. Each loads 16 bf16 (32 bytes) via two uint4.
    // -----------------------------------------------------------------------
    {
      const __hip_bfloat16 *a_src =
          A + (int64_t)(block_m + a_load_row) * K + k_base + a_load_half * 16;
      __hip_bfloat16 *a_dst =
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
      __hip_bfloat16 *bt_dst =
          smem_BT + b_load_col * SMEM_STRIDE + b_load_group * 16;

      #pragma unroll
      for (int k = 0; k < 16; k += 2) {
        int row = b_load_group * 16 + k;
        __hip_bfloat16 v0 =
            B[(int64_t)(k_base + row) * N + (block_n + b_load_col)];
        __hip_bfloat16 v1 =
            B[(int64_t)(k_base + row + 1) * N + (block_n + b_load_col)];
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
// Boundary-aware GEMM kernel
//
// Handles arbitrary M, N, K (not just multiples of 128/128/32).
// Edge tiles load zeros for OOB positions; C stores skip OOB positions.
// The original gemm_wmma_128x128 above is completely untouched so its ISA
// is identical — this kernel is only dispatched when alignment doesn't hold.
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

// =============================================================================
// Host launch
// =============================================================================

namespace kernelgen {
namespace gemm {
namespace native_hip {

static inline int ceil_div(int a, int b) { return (a + b - 1) / b; }

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

  const int M = config.M;
  const int N = config.N;
  const int K = config.K;
  const float alpha = config.alpha;

  // Select kernel: use the fast aligned kernel when M, N, K are all aligned,
  // otherwise use the boundary kernel that handles arbitrary dimensions.
  const bool aligned = (M % BM == 0) && (N % BN == 0) && (K % BK == 0);

  dim3 grid(ceil_div(M, BM), ceil_div(N, BN));
  dim3 block(BLOCK_SIZE);

  auto launch = [&](hipStream_t stream) {
    if (aligned) {
      gemm_wmma_128x128<<<grid, block, 0, stream>>>(
          static_cast<const __hip_bfloat16 *>(A),
          static_cast<const __hip_bfloat16 *>(B),
          static_cast<__hip_bfloat16 *>(C), M, N, K, alpha);
    } else {
      gemm_wmma_128x128_boundary<<<grid, block, 0, stream>>>(
          static_cast<const __hip_bfloat16 *>(A),
          static_cast<const __hip_bfloat16 *>(B),
          static_cast<__hip_bfloat16 *>(C), M, N, K, alpha);
    }
  };

  // Create stream and events for timing.
  hipStream_t stream;
  hipStreamCreate(&stream);
  hipEvent_t start, stop;
  hipEventCreate(&start);
  hipEventCreate(&stop);

  // Warmup runs.
  for (int i = 0; i < warmup_runs; i++) {
    launch(stream);
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
    launch(stream);
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
