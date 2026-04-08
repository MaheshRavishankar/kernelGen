# Fusilli GEMM kernel provider.
#
# Fusilli is represented as a first-class provider but currently benchmarks via
# the same IREE runtime path so results can be compared and tracked separately.

include(KernelGenIreeRuntime)

kernelgen_configure_iree_runtime()

add_executable(fusilli_gemm_bench
               ${CMAKE_CURRENT_SOURCE_DIR}/../iree/src/bench.cpp)

target_include_directories(
  fusilli_gemm_bench BEFORE PRIVATE ${HIP_INCLUDE_DIRS}
                                    ${IREE_SOURCE_DIR}/runtime/src)

target_link_libraries(
  fusilli_gemm_bench PRIVATE iree_runtime_unified hip::amdhip64
                             kernelgen_gemm_hip_utils)

# Host-only benchmark launcher; IREE emits and dispatches the GPU kernels.
