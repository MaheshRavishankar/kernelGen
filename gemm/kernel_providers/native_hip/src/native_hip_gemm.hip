// =============================================================================
// Native HIP GEMM — host dispatch logic
//
// Kernel implementations are in separate files:
//   gemm_wmma_128x128.hip          — aligned kernel (M%128==0, N%128==0, K%32==0)
//   gemm_wmma_128x128_boundary.hip — boundary kernel (arbitrary M, N, K)
// =============================================================================

#include "native_hip_gemm.h"

#include <hip/hip_bf16.h>
#include <hip/hip_runtime.h>

#include <string>

// Forward declarations of kernel functions defined in separate TUs.
__global__ void gemm_wmma_128x128(
    const __hip_bfloat16 *__restrict__ A,
    const __hip_bfloat16 *__restrict__ B,
    __hip_bfloat16 *__restrict__ C,
    const int M, const int N, const int K,
    const float alpha);

__global__ void gemm_wmma_128x128_boundary(
    const __hip_bfloat16 *__restrict__ A,
    const __hip_bfloat16 *__restrict__ B,
    __hip_bfloat16 *__restrict__ C,
    const int M, const int N, const int K,
    const float alpha);

// =============================================================================
// Host launch
// =============================================================================

namespace kernelgen {
namespace gemm {
namespace native_hip {

// Tile dimensions must match the kernel files.
static constexpr int BM = 128;
static constexpr int BN = 128;
static constexpr int BK = 32;
static constexpr int BLOCK_SIZE = 256;

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
