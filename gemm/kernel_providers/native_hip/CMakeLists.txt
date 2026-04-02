# =============================================================================
# Native HIP kernel provider for GEMM Builds: - kernelgen_gemm_native_hip
# (static library with WMMA kernels) - native_hip_gemm_bench     (benchmark
# executable)
# =============================================================================

find_package(HIP REQUIRED)

set(CMAKE_CXX_COMPILER "${HIP_HIPCC_EXECUTABLE}")

# Static library containing the kernel(s) and host launch code. The .hip
# extension is compiled as HIP language via -x hip flag.
add_library(
  kernelgen_gemm_native_hip STATIC
  src/native_hip_gemm.hip src/gemm_wmma_128x128.hip
  src/gemm_wmma_128x128_boundary.hip)
set_source_files_properties(
  src/native_hip_gemm.hip src/gemm_wmma_128x128.hip
  src/gemm_wmma_128x128_boundary.hip PROPERTIES LANGUAGE CXX)

target_include_directories(
  kernelgen_gemm_native_hip
  PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>)

target_link_libraries(kernelgen_gemm_native_hip PUBLIC hip::amdhip64)

target_compile_options(kernelgen_gemm_native_hip
                       PRIVATE -x hip --offload-arch=${GPU_TARGETS})

# Benchmark executable.
add_executable(native_hip_gemm_bench src/bench.cpp)

target_link_libraries(native_hip_gemm_bench PRIVATE kernelgen_gemm_native_hip
                                                    kernelgen_gemm_hip_utils)

target_compile_options(native_hip_gemm_bench
                       PRIVATE --offload-arch=${GPU_TARGETS})
