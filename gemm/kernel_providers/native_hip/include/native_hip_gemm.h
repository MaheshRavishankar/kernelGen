#ifndef KERNELGEN_GEMM_NATIVE_HIP_GEMM_H
#define KERNELGEN_GEMM_NATIVE_HIP_GEMM_H

#include <cstdint>
#include <string>

namespace kernelgen {
namespace gemm {
namespace native_hip {

struct GemmConfig {
  int64_t M;
  int64_t N;
  int64_t K;
  bool transA = false;
  bool transB = false;
  float alpha = 1.0f;
  float beta = 0.0f;
  std::string dtype_A = "bf16";
  std::string dtype_B = "bf16";
  std::string dtype_C = "bf16";
  std::string compute_type = "f32";
};

struct GemmResult {
  double kernel_time_us = 0.0;
  bool success = false;
  std::string error;
};

/// Run a GEMM using hand-written HIP kernels with WMMA intrinsics.
///   A, B, C are device pointers, already allocated and populated.
///   Currently supports: BF16 inputs, F32 accumulation, BF16 output.
///   Requires: transA=false, transB=false (NN layout).
///   M and N must be multiples of 128, K must be a multiple of 16.
GemmResult run(const GemmConfig &config, void *A, void *B, void *C,
               int warmup_runs = 5, int timed_runs = 20);

} // namespace native_hip
} // namespace gemm
} // namespace kernelgen

#endif // KERNELGEN_GEMM_NATIVE_HIP_GEMM_H
