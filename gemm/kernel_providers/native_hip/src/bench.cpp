#include "bench_utils.h"
#include "hip_utils.h"
#include "kernelgen_gemm.h"
#include "native_hip_gemm.h"

#include <hip/hip_runtime.h>

#include <iostream>
#include <string>
#include <vector>

using namespace kernelgen::gemm::native_hip;
using namespace kernelgen::gemm::utils;

namespace {

void printUsage(const char *argv0) {
  std::cerr << "Usage: " << argv0
            << " --config <config.json> [--input-a <a.npy> --input-b <b.npy>]"
            << " [--warmup N] [--timed N] [--reference <c.npy>]"
            << " [--use-dispatch]\n"
            << "  If --input-a/--input-b are omitted, random data is "
               "generated on device.\n"
            << "  --use-dispatch  Benchmark through the C dispatch API "
               "instead of calling run() directly.\n";
}

GemmResult runViaDispatch(const GemmConfig &config, void *A, void *B, void *C,
                          int warmup_runs, int timed_runs) {
  GemmResult result;

  kernelgen_gemm_config_t dc;
  dc.M = config.M;
  dc.N = config.N;
  dc.K = config.K;
  dc.transA = config.transA ? 1 : 0;
  dc.transB = config.transB ? 1 : 0;
  dc.alpha = config.alpha;
  dc.beta = config.beta;
  dc.dtype_A = KERNELGEN_BF16;
  dc.dtype_B = KERNELGEN_BF16;
  dc.dtype_C = KERNELGEN_BF16;
  dc.compute_type = KERNELGEN_F32;

  kernelgen_status_t status = kernelgen_gemm_supported(&dc);
  if (status != KERNELGEN_SUCCESS) {
    result.error =
        "dispatch: config not supported (status=" + std::to_string(status) +
        ")";
    return result;
  }

  hipStream_t stream;
  hipStreamCreate(&stream);
  hipEvent_t start, stop;
  hipEventCreate(&start);
  hipEventCreate(&stop);

  // Warmup.
  for (int i = 0; i < warmup_runs; i++) {
    kernelgen_gemm(&dc, A, B, C, stream);
  }
  hipStreamSynchronize(stream);

  hipError_t err = hipGetLastError();
  if (err != hipSuccess) {
    result.error =
        std::string("dispatch launch failed: ") + hipGetErrorString(err);
    hipEventDestroy(start);
    hipEventDestroy(stop);
    hipStreamDestroy(stream);
    return result;
  }

  // Timed runs.
  hipEventRecord(start, stream);
  for (int i = 0; i < timed_runs; i++) {
    kernelgen_gemm(&dc, A, B, C, stream);
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

} // namespace

int main(int argc, char **argv) {
  std::string configPath, inputAPath, inputBPath, refPath;
  int warmup = 5, timed = 20;
  bool useDispatch = false;

  for (int i = 1; i < argc; ++i) {
    std::string arg = argv[i];
    if (arg == "--config" && i + 1 < argc)
      configPath = argv[++i];
    else if (arg == "--input-a" && i + 1 < argc)
      inputAPath = argv[++i];
    else if (arg == "--input-b" && i + 1 < argc)
      inputBPath = argv[++i];
    else if (arg == "--reference" && i + 1 < argc)
      refPath = argv[++i];
    else if (arg == "--warmup" && i + 1 < argc)
      warmup = std::stoi(argv[++i]);
    else if (arg == "--timed" && i + 1 < argc)
      timed = std::stoi(argv[++i]);
    else if (arg == "--use-dispatch")
      useDispatch = true;
    else {
      printUsage(argv[0]);
      return 1;
    }
  }

  if (configPath.empty()) {
    printUsage(argv[0]);
    return 1;
  }
  bool randomInit = inputAPath.empty() || inputBPath.empty();

  auto bc = parseGemmConfig(configPath);

  GemmConfig config;
  config.M = bc.M;
  config.N = bc.N;
  config.K = bc.K;
  config.transA = bc.transA;
  config.transB = bc.transB;
  config.alpha = bc.alpha;
  config.beta = bc.beta;
  config.dtype_A = bc.dtype_A;
  config.dtype_B = bc.dtype_B;
  config.dtype_C = bc.dtype_C;
  config.compute_type = bc.compute_type;

  size_t sizeA = config.M * config.K * dtypeSize(config.dtype_A);
  size_t sizeB = config.K * config.N * dtypeSize(config.dtype_B);
  size_t sizeC = config.M * config.N * dtypeSize(config.dtype_C);

  // Allocate and initialize device memory.
  void *dA = randomInit ? allocDeviceDeterministic(sizeA)
                        : allocDeviceFromNpy(inputAPath, sizeA);
  void *dB = randomInit ? allocDeviceDeterministic(sizeB)
                        : allocDeviceFromNpy(inputBPath, sizeB);
  void *dC;
  hipMalloc(&dC, sizeC);
  hipMemset(dC, 0, sizeC);

  GemmResult result = useDispatch
                          ? runViaDispatch(config, dA, dB, dC, warmup, timed)
                          : run(config, dA, dB, dC, warmup, timed);

  // Output JSON to stdout.
  std::cout << "{\"provider\": \"native_hip\""
            << ", \"kernel_time_us\": " << result.kernel_time_us
            << ", \"success\": " << (result.success ? "true" : "false");
  if (!result.error.empty())
    std::cout << ", \"error\": \"" << result.error << "\"";

  // Verify against reference.
  if (!refPath.empty() && result.success) {
    auto refData = loadNpy(refPath);
    std::vector<char> hostC(sizeC);
    hipMemcpy(hostC.data(), dC, sizeC, hipMemcpyDeviceToHost);

    auto v = verify(hostC.data(), refData.data(), sizeC, config.dtype_C);
    printVerifyJson(v);
  }

  std::cout << "}" << std::endl;

  hipFree(dA);
  hipFree(dB);
  hipFree(dC);

  return result.success ? 0 : 1;
}
