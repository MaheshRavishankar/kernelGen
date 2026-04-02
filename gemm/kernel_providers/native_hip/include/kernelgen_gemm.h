#ifndef KERNELGEN_GEMM_H
#define KERNELGEN_GEMM_H

#include <hip/hip_runtime.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef enum {
  KERNELGEN_BF16 = 0,
  KERNELGEN_F16 = 1,
  KERNELGEN_F32 = 2,
} kernelgen_dtype_t;

typedef enum {
  KERNELGEN_SUCCESS = 0,
  KERNELGEN_ERROR_UNSUPPORTED_DTYPE = 1,
  KERNELGEN_ERROR_UNSUPPORTED_LAYOUT = 2,
  KERNELGEN_ERROR_INVALID_DIMENSIONS = 3,
  KERNELGEN_ERROR_LAUNCH_FAILED = 4,
} kernelgen_status_t;

typedef struct {
  int64_t M, N, K;
  int transA, transB; // 0=no, 1=yes
  float alpha, beta;
  kernelgen_dtype_t dtype_A, dtype_B, dtype_C;
  kernelgen_dtype_t compute_type;
} kernelgen_gemm_config_t;

// Launch a GEMM on the given stream. Stateless — no handle needed.
kernelgen_status_t kernelgen_gemm(const kernelgen_gemm_config_t *config,
                                  const void *A, // device pointer
                                  const void *B, // device pointer
                                  void *C,       // device pointer
                                  hipStream_t stream);

// Query whether a config is supported without launching.
kernelgen_status_t
kernelgen_gemm_supported(const kernelgen_gemm_config_t *config);

#ifdef __cplusplus
}
#endif

#endif // KERNELGEN_GEMM_H
