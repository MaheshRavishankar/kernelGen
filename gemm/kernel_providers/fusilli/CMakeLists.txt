# Fusilli GEMM kernel provider.
#
# Uses the Fusilli C++ graph frontend to build and JIT-compile GEMM graphs
# against the shared IREE runtime/tooling stack used elsewhere in kernelGen.

include(KernelGenIreeRuntime)

kernelgen_configure_iree_runtime()

set(FUSILLI_SOURCE_DIR
    "$ENV{HOME}/iree/fusilli"
    CACHE PATH "Path to the Fusilli source tree")
set(IREE_COMPILER_LIB
    ""
    CACHE FILEPATH "Path to libIREECompiler.so for Fusilli compiler C API")

if(NOT EXISTS "${FUSILLI_SOURCE_DIR}/include/fusilli.h")
  message(FATAL_ERROR "Fusilli source not found at ${FUSILLI_SOURCE_DIR}.\n"
                      "Set -DFUSILLI_SOURCE_DIR=<path> to a Fusilli checkout.")
endif()

message(STATUS "Fusilli source: ${FUSILLI_SOURCE_DIR}")

add_executable(fusilli_gemm_bench src/bench.cpp)
set_target_properties(fusilli_gemm_bench PROPERTIES CXX_STANDARD 20
                                                    CXX_STANDARD_REQUIRED YES)

target_include_directories(
  fusilli_gemm_bench BEFORE
  PRIVATE ${HIP_INCLUDE_DIRS} ${IREE_SOURCE_DIR}/runtime/src
          ${FUSILLI_SOURCE_DIR}/include)

target_compile_definitions(fusilli_gemm_bench PRIVATE FUSILLI_ENABLE_AMDGPU)
if(IREE_COMPILER_LIB)
  target_compile_definitions(
    fusilli_gemm_bench
    PRIVATE FUSILLI_DEFAULT_IREE_COMPILER_LIB=\"${IREE_COMPILER_LIB}\")
endif()

target_link_libraries(
  fusilli_gemm_bench PRIVATE iree_runtime_unified hip::amdhip64
                             kernelgen_gemm_hip_utils ${CMAKE_DL_LIBS})

# Host-side benchmark launcher; Fusilli emits and dispatches the GPU kernels.
