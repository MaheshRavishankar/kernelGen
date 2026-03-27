#ifndef KERNELGEN_GEMM_HIPBLASLT_GEMM_H
#define KERNELGEN_GEMM_HIPBLASLT_GEMM_H

#include <cstdint>
#include <string>

namespace kernelgen {
namespace gemm {
namespace hipblaslt {

struct GemmConfig {
  int64_t M;
  int64_t N;
  int64_t K;
  bool transA = false;
  bool transB = false;
  float alpha = 1.0f;
  float beta = 0.0f;
  std::string dtype_A = "f16";
  std::string dtype_B = "f16";
  std::string dtype_C = "f16";
  std::string compute_type = "f32";
};

struct GemmResult {
  double kernel_time_us = 0.0;
  bool success = false;
  std::string error;
};

/// Run a GEMM using hipBLAS-LT.
///   A, B, C are device pointers, already allocated and populated.
GemmResult run(const GemmConfig &config, void *A, void *B, void *C,
               int warmup_runs = 5, int timed_runs = 20);

} // namespace hipblaslt
} // namespace gemm
} // namespace kernelgen

#endif // KERNELGEN_GEMM_HIPBLASLT_GEMM_H
