# IREE GEMM kernel provider.
#
# Builds the IREE runtime from source (runtime only, no compiler) and links the
# benchmark executable against it.
#
# Set IREE_SOURCE_DIR to the IREE source tree.  Defaults to
# ~/kernelGen/iree/iree.

include(KernelGenIreeRuntime)

kernelgen_configure_iree_runtime()

# ---------------------------------------------------------------------------
# Benchmark executable
# ---------------------------------------------------------------------------
add_executable(iree_gemm_bench src/bench.cpp)

target_include_directories(
  iree_gemm_bench BEFORE PRIVATE ${HIP_INCLUDE_DIRS}
                                 ${IREE_SOURCE_DIR}/runtime/src)

target_link_libraries(
  iree_gemm_bench PRIVATE iree_runtime_unified hip::amdhip64
                          kernelgen_gemm_hip_utils)

# No --offload-arch needed: bench.cpp is host-only code (no GPU kernels). The
# IREE runtime dispatches GPU work via the HAL driver.
