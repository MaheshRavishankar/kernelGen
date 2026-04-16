// =============================================================================
// Native HIP GEMM — host dispatch logic
//
// Kernel implementations are in separate files:
//   gemm_wmma_skinny.hip            — skinny-M kernel (M <= 32, BM=128 BN=16)
//   gemm_low_k_stream.hip           — low-K streaming kernel (K <= 32)
//   gemm_wmma_128x128.hip           — aligned kernel (M%128==0, N%128==0, K%32==0)
//   gemm_wmma_128x128_boundary.hip  — boundary kernel (arbitrary M, N, K)
// =============================================================================

#include "native_hip_gemm.h"

#include <hip/hip_bf16.h>
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include <string>
#include <type_traits>

// Forward declarations of templated kernel functions defined in separate TUs.
template <typename T>
__global__ void gemm_wmma_skinny(
    const T *__restrict__ A, const T *__restrict__ B, T *__restrict__ C,
    const int M, const int N, const int K, const float alpha);

template <typename T>
__global__ void gemm_wmma_128x128(
    const T *__restrict__ A, const T *__restrict__ B, T *__restrict__ C,
    const int M, const int N, const int K, const float alpha);

template <typename T>
__global__ void gemm_low_k_stream(
    const T *__restrict__ A, const T *__restrict__ B, T *__restrict__ C,
    const int M, const int N, const int K, const float alpha);

template <typename T>
__global__ void gemm_wmma_128x128_boundary(
    const T *__restrict__ A, const T *__restrict__ B, T *__restrict__ C,
    const int M, const int N, const int K, const float alpha);

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
  const bool is_bf16 = (config.dtype_A == "bf16" && config.dtype_B == "bf16" && config.dtype_C == "bf16");
  const bool is_f16 = (config.dtype_A == "f16" && config.dtype_B == "f16" && config.dtype_C == "f16");
  if (!is_bf16 && !is_f16) {
    result.error = "native_hip: only BF16 or F16 data types supported";
    return result;
  }

  const int M = config.M;
  const int N = config.N;
  const int K = config.K;
  const float alpha = config.alpha;

  // Select kernel by M dimension, then by low-K/memory-bound balance point,
  // then by alignment.
  // Skinny-M kernel uses BM=128 staging and BN=16 tiles. Narrow BN maximizes
  // grid parallelism along N; the kernel limits active compute warps for the
  // larger skinny shapes while retaining the full-width tiny-K balance point.
  // Low-K streaming keeps K<=32 shapes off the boundary WMMA path; K around
  // 128+ stays on WMMA because scalar dot work becomes compute-heavy.
  dim3 grid;
  dim3 block;

  enum KernelChoice { SKINNY, LOW_K_STREAM, ALIGNED_128, BOUNDARY_128 };
  KernelChoice choice;

  if (M <= 32) {
    // Skinny kernel: BM=128, BN=16, 256 threads.
    grid = dim3(ceil_div(M, 128), ceil_div(N, 16));
    block = dim3(256);
    choice = SKINNY;
  } else if (K <= 32) {
    grid = dim3(ceil_div(M, 16), ceil_div(N, 256));
    block = dim3(256);
    choice = LOW_K_STREAM;
  } else if ((M % 128 == 0) && (N % 128 == 0) && (K % 32 == 0)) {
    grid = dim3(ceil_div(M, 128), ceil_div(N, 128));
    block = dim3(256);
    choice = ALIGNED_128;
  } else {
    grid = dim3(ceil_div(M, 128), ceil_div(N, 128));
    block = dim3(256);
    choice = BOUNDARY_128;
  }

  // Templated launch helper: dispatches to the correct kernel instantiation.
  auto launch_typed = [&](hipStream_t stream, auto *typed_null) {
    using T = std::remove_pointer_t<decltype(typed_null)>;
    auto *dA = static_cast<const T *>(A);
    auto *dB = static_cast<const T *>(B);
    auto *dC = static_cast<T *>(C);
    switch (choice) {
    case SKINNY:
      gemm_wmma_skinny<T><<<grid, block, 0, stream>>>(dA, dB, dC, M, N, K, alpha);
      break;
    case LOW_K_STREAM:
      gemm_low_k_stream<T><<<grid, block, 0, stream>>>(dA, dB, dC, M, N, K, alpha);
      break;
    case ALIGNED_128:
      gemm_wmma_128x128<T><<<grid, block, 0, stream>>>(dA, dB, dC, M, N, K, alpha);
      break;
    case BOUNDARY_128:
      gemm_wmma_128x128_boundary<T><<<grid, block, 0, stream>>>(dA, dB, dC, M, N, K, alpha);
      break;
    }
  };

  auto launch = [&](hipStream_t stream) {
    if (is_f16) {
      launch_typed(stream, static_cast<_Float16 *>(nullptr));
    } else {
      launch_typed(stream, static_cast<__hip_bfloat16 *>(nullptr));
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
