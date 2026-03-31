#ifndef KERNELGEN_GEMM_HIP_UTILS_H
#define KERNELGEN_GEMM_HIP_UTILS_H

#include <cstddef>
#include <string>

namespace kernelgen {
namespace gemm {
namespace utils {

/// Allocate device memory and fill with deterministic pattern.
/// Caller must hipFree the returned pointer.
void *allocDeviceDeterministic(size_t bytes);

/// Allocate device memory and copy .npy file contents into it.
/// Caller must hipFree the returned pointer.
void *allocDeviceFromNpy(const std::string &npyPath, size_t bytes);

} // namespace utils
} // namespace gemm
} // namespace kernelgen

#endif // KERNELGEN_GEMM_HIP_UTILS_H
