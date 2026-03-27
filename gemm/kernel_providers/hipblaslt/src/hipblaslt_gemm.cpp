#include "hipblaslt_gemm.h"

#include <hip/hip_runtime.h>
#include <hipblaslt/hipblaslt.h>

#include <stdexcept>
#include <string>
#include <vector>

#define HIP_CHECK(expr)                                                        \
  do {                                                                         \
    hipError_t _err = (expr);                                                  \
    if (_err != hipSuccess) {                                                  \
      return GemmResult{0, false,                                              \
                        std::string("HIP error: ") + hipGetErrorString(_err)}; \
    }                                                                          \
  } while (0)

#define HIPBLASLT_CHECK(expr)                                                  \
  do {                                                                         \
    hipblasStatus_t _s = (expr);                                               \
    if (_s != HIPBLAS_STATUS_SUCCESS) {                                        \
      return GemmResult{0, false,                                              \
                        "hipBLAS-LT error: status " + std::to_string(_s)};     \
    }                                                                          \
  } while (0)

namespace kernelgen {
namespace gemm {
namespace hipblaslt {

namespace {

hipDataType toHipDataType(const std::string &dtype) {
  if (dtype == "f16")
    return HIP_R_16F;
  if (dtype == "bf16")
    return HIP_R_16BF;
  if (dtype == "f32")
    return HIP_R_32F;
  throw std::runtime_error("Unsupported dtype: " + dtype);
}

hipblasComputeType_t toComputeType(const std::string &dtype) {
  if (dtype == "f16")
    return HIPBLAS_COMPUTE_16F;
  return HIPBLAS_COMPUTE_32F;
}

size_t dtypeSize(const std::string &dtype) {
  if (dtype == "f16" || dtype == "bf16")
    return 2;
  if (dtype == "f32")
    return 4;
  throw std::runtime_error("Unsupported dtype: " + dtype);
}

} // namespace

GemmResult run(const GemmConfig &config, void *A, void *B, void *C,
               int warmup_runs, int timed_runs) {
  hipDataType typeA = toHipDataType(config.dtype_A);
  hipDataType typeB = toHipDataType(config.dtype_B);
  hipDataType typeC = toHipDataType(config.dtype_C);
  hipDataType scaleType = toHipDataType(config.compute_type);
  hipblasComputeType_t computeType = toComputeType(config.compute_type);

  hipblasOperation_t opA = config.transA ? HIPBLAS_OP_T : HIPBLAS_OP_N;
  hipblasOperation_t opB = config.transB ? HIPBLAS_OP_T : HIPBLAS_OP_N;

  int64_t lda = config.transA ? config.K : config.M;
  int64_t ldb = config.transB ? config.N : config.K;
  int64_t ldc = config.M;

  hipblasLtHandle_t handle;
  HIPBLASLT_CHECK(hipblasLtCreate(&handle));

  // Matrix layouts.
  hipblasLtMatrixLayout_t layoutA, layoutB, layoutC;
  HIPBLASLT_CHECK(hipblasLtMatrixLayoutCreate(
      &layoutA, typeA, config.transA ? config.K : config.M,
      config.transA ? config.M : config.K, lda));
  HIPBLASLT_CHECK(hipblasLtMatrixLayoutCreate(
      &layoutB, typeB, config.transB ? config.N : config.K,
      config.transB ? config.K : config.N, ldb));
  HIPBLASLT_CHECK(
      hipblasLtMatrixLayoutCreate(&layoutC, typeC, config.M, config.N, ldc));

  // Matmul descriptor.
  hipblasLtMatmulDesc_t matmulDesc;
  HIPBLASLT_CHECK(
      hipblasLtMatmulDescCreate(&matmulDesc, computeType, scaleType));
  HIPBLASLT_CHECK(hipblasLtMatmulDescSetAttribute(
      matmulDesc, HIPBLASLT_MATMUL_DESC_TRANSA, &opA, sizeof(opA)));
  HIPBLASLT_CHECK(hipblasLtMatmulDescSetAttribute(
      matmulDesc, HIPBLASLT_MATMUL_DESC_TRANSB, &opB, sizeof(opB)));

  // Algorithm heuristic.
  hipblasLtMatmulPreference_t pref;
  HIPBLASLT_CHECK(hipblasLtMatmulPreferenceCreate(&pref));
  size_t maxWorkspace = 256 * 1024 * 1024;
  HIPBLASLT_CHECK(hipblasLtMatmulPreferenceSetAttribute(
      pref, HIPBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &maxWorkspace,
      sizeof(maxWorkspace)));

  const int requestedAlgos = 8;
  std::vector<hipblasLtMatmulHeuristicResult_t> heuristicResults(
      requestedAlgos);
  int returnedAlgos = 0;
  HIPBLASLT_CHECK(hipblasLtMatmulAlgoGetHeuristic(
      handle, matmulDesc, layoutA, layoutB, layoutC, layoutC, pref,
      requestedAlgos, heuristicResults.data(), &returnedAlgos));

  if (returnedAlgos == 0) {
    return GemmResult{0, false, "No algorithms found for this problem"};
  }

  // Workspace.
  void *workspace = nullptr;
  size_t workspaceSize = heuristicResults[0].workspaceSize;
  if (workspaceSize > 0) {
    HIP_CHECK(hipMalloc(&workspace, workspaceSize));
  }

  hipStream_t stream;
  HIP_CHECK(hipStreamCreate(&stream));

  float alpha = config.alpha;
  float beta = config.beta;

  // Warmup.
  for (int i = 0; i < warmup_runs; ++i) {
    hipblasLtMatmul(handle, matmulDesc, &alpha, A, layoutA, B, layoutB, &beta,
                    C, layoutC, C, layoutC, &heuristicResults[0].algo,
                    workspace, workspaceSize, stream);
  }
  HIP_CHECK(hipStreamSynchronize(stream));

  // Timed runs.
  hipEvent_t start, stop;
  HIP_CHECK(hipEventCreate(&start));
  HIP_CHECK(hipEventCreate(&stop));

  HIP_CHECK(hipEventRecord(start, stream));
  for (int i = 0; i < timed_runs; ++i) {
    hipblasLtMatmul(handle, matmulDesc, &alpha, A, layoutA, B, layoutB, &beta,
                    C, layoutC, C, layoutC, &heuristicResults[0].algo,
                    workspace, workspaceSize, stream);
  }
  HIP_CHECK(hipEventRecord(stop, stream));
  HIP_CHECK(hipEventSynchronize(stop));

  float elapsed_ms = 0;
  HIP_CHECK(hipEventElapsedTime(&elapsed_ms, start, stop));
  double avg_us = (elapsed_ms * 1000.0) / timed_runs;

  // Cleanup.
  hipEventDestroy(start);
  hipEventDestroy(stop);
  hipStreamDestroy(stream);
  if (workspace)
    hipFree(workspace);
  hipblasLtMatmulPreferenceDestroy(pref);
  hipblasLtMatmulDescDestroy(matmulDesc);
  hipblasLtMatrixLayoutDestroy(layoutA);
  hipblasLtMatrixLayoutDestroy(layoutB);
  hipblasLtMatrixLayoutDestroy(layoutC);
  hipblasLtDestroy(handle);

  return GemmResult{avg_us, true, ""};
}

} // namespace hipblaslt
} // namespace gemm
} // namespace kernelgen
