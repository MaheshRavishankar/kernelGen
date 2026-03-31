#include "hip_utils.h"
#include "bench_utils.h"

#include <hip/hip_runtime.h>

#include <stdexcept>
#include <vector>

namespace kernelgen {
namespace gemm {
namespace utils {

void *allocDeviceDeterministic(size_t bytes) {
  void *dPtr = nullptr;
  if (hipMalloc(&dPtr, bytes) != hipSuccess)
    throw std::runtime_error("hipMalloc failed");
  std::vector<char> hostBuf(bytes);
  fillDeterministic(hostBuf.data(), bytes);
  hipMemcpy(dPtr, hostBuf.data(), bytes, hipMemcpyHostToDevice);
  return dPtr;
}

void *allocDeviceFromNpy(const std::string &npyPath, size_t bytes) {
  auto data = loadNpy(npyPath);
  void *dPtr = nullptr;
  if (hipMalloc(&dPtr, bytes) != hipSuccess)
    throw std::runtime_error("hipMalloc failed");
  hipMemcpy(dPtr, data.data(), bytes, hipMemcpyHostToDevice);
  return dPtr;
}

} // namespace utils
} // namespace gemm
} // namespace kernelgen
