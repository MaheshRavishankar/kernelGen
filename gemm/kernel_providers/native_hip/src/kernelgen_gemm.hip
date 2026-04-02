// =============================================================================
// kernelgen_gemm — Public C dispatch API for native HIP GEMM kernels
//
// Thin wrapper: validate config, select kernel, compute grid, launch.
// No timing, no warmup, no allocation.
// =============================================================================

#include "kernelgen_gemm.h"

#include <hip/hip_bf16.h>
#include <hip/hip_runtime.h>

// Forward declarations of kernel functions defined in separate TUs.
__global__ void gemm_wmma_skinny(
    const __hip_bfloat16 *__restrict__ A,
    const __hip_bfloat16 *__restrict__ B,
    __hip_bfloat16 *__restrict__ C,
    const int M, const int N, const int K,
    const float alpha);

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

static inline int ceil_div(int a, int b) { return (a + b - 1) / b; }

static kernelgen_status_t validate_config(const kernelgen_gemm_config_t *config) {
    if (config->dtype_A != KERNELGEN_BF16 ||
        config->dtype_B != KERNELGEN_BF16 ||
        config->dtype_C != KERNELGEN_BF16) {
        return KERNELGEN_ERROR_UNSUPPORTED_DTYPE;
    }
    if (config->transA != 0 || config->transB != 0) {
        return KERNELGEN_ERROR_UNSUPPORTED_LAYOUT;
    }
    if (config->beta != 0.0f) {
        return KERNELGEN_ERROR_UNSUPPORTED_LAYOUT;
    }
    if (config->M <= 0 || config->N <= 0 || config->K <= 0) {
        return KERNELGEN_ERROR_INVALID_DIMENSIONS;
    }
    return KERNELGEN_SUCCESS;
}

extern "C" kernelgen_status_t kernelgen_gemm_supported(
    const kernelgen_gemm_config_t *config) {
    return validate_config(config);
}

extern "C" kernelgen_status_t kernelgen_gemm(
    const kernelgen_gemm_config_t *config,
    const void *A,
    const void *B,
    void *C,
    hipStream_t stream) {

    kernelgen_status_t status = validate_config(config);
    if (status != KERNELGEN_SUCCESS) {
        return status;
    }

    const int M = static_cast<int>(config->M);
    const int N = static_cast<int>(config->N);
    const int K = static_cast<int>(config->K);
    const float alpha = config->alpha;

    const auto *dA = static_cast<const __hip_bfloat16 *>(A);
    const auto *dB = static_cast<const __hip_bfloat16 *>(B);
    auto *dC = static_cast<__hip_bfloat16 *>(C);

    dim3 grid, block(256);

    if (M <= 32) {
        // Skinny kernel: BM=128, BN=16, 256 threads.
        grid = dim3(ceil_div(M, 128), ceil_div(N, 16));
        gemm_wmma_skinny<<<grid, block, 0, stream>>>(dA, dB, dC, M, N, K, alpha);
    } else if ((M % 128 == 0) && (N % 128 == 0) && (K % 32 == 0)) {
        // Aligned 128x128 kernel.
        grid = dim3(ceil_div(M, 128), ceil_div(N, 128));
        gemm_wmma_128x128<<<grid, block, 0, stream>>>(dA, dB, dC, M, N, K, alpha);
    } else {
        // Boundary 128x128 kernel.
        grid = dim3(ceil_div(M, 128), ceil_div(N, 128));
        gemm_wmma_128x128_boundary<<<grid, block, 0, stream>>>(dA, dB, dC, M, N, K, alpha);
    }

    hipError_t err = hipGetLastError();
    if (err != hipSuccess) {
        return KERNELGEN_ERROR_LAUNCH_FAILED;
    }

    return KERNELGEN_SUCCESS;
}
