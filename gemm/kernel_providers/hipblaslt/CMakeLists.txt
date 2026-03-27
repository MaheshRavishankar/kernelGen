# =============================================================================
# hipBLAS-LT kernel provider for GEMM Builds: - kernelgen_gemm_hipblaslt (static
# library) - hipblaslt_gemm_bench           (benchmark executable)
# =============================================================================

find_package(HIP REQUIRED)
find_package(HipBlasLt REQUIRED)

set(CMAKE_CXX_COMPILER "${HIP_HIPCC_EXECUTABLE}")

# Static library.
add_library(kernelgen_gemm_hipblaslt STATIC src/hipblaslt_gemm.cpp)

target_include_directories(
  kernelgen_gemm_hipblaslt
  PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>)

target_link_libraries(kernelgen_gemm_hipblaslt PUBLIC hip::amdhip64
                                                      hipblaslt::hipblaslt)

target_compile_options(kernelgen_gemm_hipblaslt
                       PRIVATE --offload-arch=${GPU_TARGETS})

# Benchmark executable.
add_executable(hipblaslt_gemm_bench src/bench.cpp)

target_link_libraries(hipblaslt_gemm_bench PRIVATE kernelgen_gemm_hipblaslt)

target_compile_options(hipblaslt_gemm_bench
                       PRIVATE --offload-arch=${GPU_TARGETS})
