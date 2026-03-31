# IREE GEMM kernel provider.
#
# Builds the IREE runtime from source (runtime only, no compiler) and links the
# benchmark executable against it.
#
# Set IREE_SOURCE_DIR to the IREE source tree.  Defaults to
# ~/kernelGen/iree/iree.

include(FetchContent)

set(IREE_SOURCE_DIR
    "$ENV{HOME}/kernelGen/iree/iree"
    CACHE PATH "Path to the IREE source tree")

if(NOT EXISTS "${IREE_SOURCE_DIR}/CMakeLists.txt")
  message(FATAL_ERROR "IREE source not found at ${IREE_SOURCE_DIR}.\n"
                      "Set -DIREE_SOURCE_DIR=<path> to the IREE source tree.")
endif()

message(STATUS "IREE source: ${IREE_SOURCE_DIR}")

# ---------------------------------------------------------------------------
# Fetch missing submodules that the source tarball doesn't include.
# ---------------------------------------------------------------------------
macro(fetch_iree_dep name repo tag)
  set(_dir "${IREE_SOURCE_DIR}/third_party/${name}")
  if(NOT EXISTS "${_dir}/CMakeLists.txt"
     AND NOT EXISTS "${_dir}/BUILD"
     AND NOT EXISTS "${_dir}/BUILD.bazel")
    message(STATUS "Fetching IREE dependency: ${name}")
    FetchContent_Declare(
      iree_${name}
      GIT_REPOSITORY ${repo}
      GIT_TAG ${tag}
      GIT_SHALLOW ON
      SOURCE_DIR "${_dir}")
    FetchContent_Populate(iree_${name})
  endif()
endmacro()

fetch_iree_dep(benchmark https://github.com/google/benchmark.git main)
fetch_iree_dep(flatcc https://github.com/dvidelabs/flatcc.git master)
# ---------------------------------------------------------------------------
# Configure IREE – runtime only, HIP driver.
# ---------------------------------------------------------------------------

# Point IREE at real HIP headers from TheRock instead of hip-build-deps stubs.
find_package(HIP REQUIRED)
set(HIP_API_HEADERS_ROOT
    "${HIP_INCLUDE_DIRS}"
    CACHE STRING "" FORCE)
set(IREE_BUILD_COMPILER
    OFF
    CACHE BOOL "" FORCE)
set(IREE_BUILD_TESTS
    OFF
    CACHE BOOL "" FORCE)
set(IREE_BUILD_SAMPLES
    OFF
    CACHE BOOL "" FORCE)
set(IREE_BUILD_BINDINGS_TFLITE
    OFF
    CACHE BOOL "" FORCE)
set(IREE_ERROR_ON_MISSING_SUBMODULES
    OFF
    CACHE BOOL "" FORCE)

set(IREE_HAL_DRIVER_DEFAULTS
    OFF
    CACHE BOOL "" FORCE)
set(IREE_HAL_DRIVER_HIP
    ON
    CACHE BOOL "" FORCE)
set(IREE_HAL_DRIVER_LOCAL_SYNC
    ON
    CACHE BOOL "" FORCE)

add_subdirectory(${IREE_SOURCE_DIR} ${CMAKE_BINARY_DIR}/iree EXCLUDE_FROM_ALL)

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
