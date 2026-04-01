#include "bench_utils.h"
#include "hip_utils.h"
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
            << " [--warmup N] [--timed N] [--reference <c.npy>]\n"
            << "  If --input-a/--input-b are omitted, random data is "
               "generated on device.\n";
}

} // namespace

int main(int argc, char **argv) {
  std::string configPath, inputAPath, inputBPath, refPath;
  int warmup = 5, timed = 20;

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

  GemmResult result = run(config, dA, dB, dC, warmup, timed);

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
