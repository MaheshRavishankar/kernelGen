# Shared IREE runtime setup for providers built on top of IREE.

include(FetchContent)

function(kernelgen_configure_iree_runtime)
  set(IREE_SOURCE_DIR
      "$ENV{HOME}/kernelGen/iree/iree"
      CACHE PATH "Path to the IREE source tree")

  if(NOT EXISTS "${IREE_SOURCE_DIR}/CMakeLists.txt")
    message(FATAL_ERROR "IREE source not found at ${IREE_SOURCE_DIR}.\n"
                        "Set -DIREE_SOURCE_DIR=<path> to the IREE source tree.")
  endif()

  message(STATUS "IREE source: ${IREE_SOURCE_DIR}")

  find_package(HIP REQUIRED)
  set(HIP_INCLUDE_DIRS
      "${HIP_INCLUDE_DIRS}"
      PARENT_SCOPE)
  set(IREE_SOURCE_DIR
      "${IREE_SOURCE_DIR}"
      PARENT_SCOPE)

  if(TARGET iree_runtime_unified)
    return()
  endif()

  # Fetch missing submodules that source snapshots may omit.
  macro(kernelgen_fetch_iree_dep name repo tag)
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

  kernelgen_fetch_iree_dep(benchmark https://github.com/google/benchmark.git
                           main)
  kernelgen_fetch_iree_dep(flatcc https://github.com/dvidelabs/flatcc.git
                           master)

  # Point IREE at real HIP headers from TheRock instead of hip-build-deps stubs.
  set(HIP_API_HEADERS_ROOT
      "${HIP_INCLUDE_DIRS}"
      CACHE STRING "" FORCE)
  set(IREE_BUILD_COMPILER
      OFF
      CACHE BOOL "" FORCE)
  set(IREE_BUILD_TESTS
      OFF
      CACHE BOOL "" FORCE)
  set(IREE_ROCM_TEST_TARGET_CHIP
      ""
      CACHE STRING "" FORCE)
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
endfunction()
