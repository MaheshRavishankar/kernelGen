// =============================================================================
// kernelgen_gemm — Public C dispatch API for native HIP GEMM kernels
//
// Thin wrapper: validate config, select kernel, compute grid, launch.
// No timing, no warmup, no allocation.
// =============================================================================

#include "kernelgen_gemm.h"

#include <hip/hip_bf16.h>
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include <type_traits>

// Forward declarations of templated kernel functions defined in separate TUs.
template <typename T>
__global__ void gemm_wmma_skinny(
    const T *__restrict__ A, const T *__restrict__ B, T *__restrict__ C,
    const int M, const int N, const int K, const float alpha);

template <typename T>
__global__ void gemm_wmma_128x128(
    const T *__restrict__ A, const T *__restrict__ B, T *__restrict__ C,
    const int M, const int N, const int K,
    const int ldb, const int ldc, const float alpha);

template <typename T>
__global__ void gemm_wmma_128x128_k_boundary(
    const T *__restrict__ A, const T *__restrict__ B, T *__restrict__ C,
    const int M, const int N, const int K,
    const int ldb, const int ldc, const float alpha);

template <typename T>
__global__ void gemm_low_k_stream(
    const T *__restrict__ A, const T *__restrict__ B, T *__restrict__ C,
    const int M, const int N, const int K, const float alpha);

template <typename T>
__global__ void gemm_wmma_128x128_boundary(
    const T *__restrict__ A, const T *__restrict__ B, T *__restrict__ C,
    const int M, const int N, const int K,
    const int lda, const int ldb, const int ldc, const float alpha);

static inline int ceil_div(int a, int b) { return (a + b - 1) / b; }

static kernelgen_status_t validate_config(const kernelgen_gemm_config_t *config) {
    const bool is_bf16 = (config->dtype_A == KERNELGEN_BF16 &&
                          config->dtype_B == KERNELGEN_BF16 &&
                          config->dtype_C == KERNELGEN_BF16);
    const bool is_f16 = (config->dtype_A == KERNELGEN_F16 &&
                         config->dtype_B == KERNELGEN_F16 &&
                         config->dtype_C == KERNELGEN_F16);
    if (!is_bf16 && !is_f16) {
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
    const bool is_f16 = (config->dtype_A == KERNELGEN_F16);
    const bool m_aligned = (M % 128 == 0);
    const bool n_aligned = (N % 128 == 0);
    const bool k_aligned = (K % 32 == 0);
    const int full_m = (M / 128) * 128;
    const int tail_m = M - full_m;
    const bool split_m_tail = (full_m > 0) && (tail_m > 0) && n_aligned;

    // Templated dispatch helper.
    auto dispatch = [&](auto *typed_null) {
        using T = std::remove_pointer_t<decltype(typed_null)>;
        const auto *dA = static_cast<const T *>(A);
        const auto *dB = static_cast<const T *>(B);
        auto *dC = static_cast<T *>(C);

        dim3 grid, block(256);

        if (M <= 32) {
            grid = dim3(ceil_div(M, 128), ceil_div(N, 16));
            gemm_wmma_skinny<T><<<grid, block, 0, stream>>>(dA, dB, dC, M, N, K, alpha);
        } else if (K <= 32) {
            grid = dim3(ceil_div(M, 16), ceil_div(N, 256));
            gemm_low_k_stream<T><<<grid, block, 0, stream>>>(dA, dB, dC, M, N, K, alpha);
        } else if (m_aligned && n_aligned && k_aligned) {
            grid = dim3(ceil_div(M, 128), ceil_div(N, 128));
            gemm_wmma_128x128<T><<<grid, block, 0, stream>>>(dA, dB, dC, M, N, K, N, N, alpha);
        } else if (m_aligned && n_aligned) {
            grid = dim3(ceil_div(M, 128), ceil_div(N, 128));
            gemm_wmma_128x128_k_boundary<T><<<grid, block, 0, stream>>>(dA, dB, dC, M, N, K, N, N, alpha);
        } else if (split_m_tail) {
            grid = dim3(full_m / 128, N / 128);
            if (k_aligned) {
                gemm_wmma_128x128<T><<<grid, block, 0, stream>>>(dA, dB, dC, full_m, N, K, N, N, alpha);
            } else {
                gemm_wmma_128x128_k_boundary<T><<<grid, block, 0, stream>>>(dA, dB, dC, full_m, N, K, N, N, alpha);
            }

            grid = dim3(1, N / 128);
            const T *bottomA = dA + (int64_t)full_m * K;
            T *bottomC = dC + (int64_t)full_m * N;
            gemm_wmma_128x128_boundary<T><<<grid, block, 0, stream>>>(bottomA, dB, bottomC, tail_m, N, K, K, N, N, alpha);
        } else {
            grid = dim3(ceil_div(M, 128), ceil_div(N, 128));
            gemm_wmma_128x128_boundary<T><<<grid, block, 0, stream>>>(dA, dB, dC, M, N, K, K, N, N, alpha);
        }
    };

    if (is_f16) {
        dispatch(static_cast<_Float16 *>(nullptr));
    } else {
        dispatch(static_cast<__hip_bfloat16 *>(nullptr));
    }

    hipError_t err = hipGetLastError();
    if (err != hipSuccess) {
        return KERNELGEN_ERROR_LAUNCH_FAILED;
    }

    return KERNELGEN_SUCCESS;
}
