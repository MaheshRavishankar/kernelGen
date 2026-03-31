cmake_minimum_required(VERSION 3.21)
project(kernelGen LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# =============================================================================
# TheRock / ROCm configuration
# =============================================================================
if(NOT DEFINED THEROCK_PATH)
  set(THEROCK_PATH "$ENV{HOME}/kernelGen/TheRock")
endif()
set(THEROCK_PATH
    "${THEROCK_PATH}"
    CACHE PATH "Path to TheRock (ROCm) installation")

if(NOT EXISTS "${THEROCK_PATH}")
  message(
    FATAL_ERROR
      "TheRock path does not exist: ${THEROCK_PATH}\n"
      "Set -DTHEROCK_PATH=<path> to your TheRock or ROCm installation.")
endif()

# Make TheRock findable by find_package / find_library
list(PREPEND CMAKE_PREFIX_PATH "${THEROCK_PATH}")
list(PREPEND CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake")

# =============================================================================
# GPU target configuration
# =============================================================================
set(GPU_TARGETS
    "gfx1100"
    CACHE STRING
          "Semicolon-separated list of GPU targets (e.g. gfx1100;gfx942)")

# =============================================================================
# Operation toggles
# =============================================================================
option(KERNELGEN_ENABLE_GEMM "Enable GEMM operation" ON)

# =============================================================================
# Provider toggles (each operation guards its own providers too)
# =============================================================================
option(KERNELGEN_ENABLE_HIPBLASLT "Enable hipBLAS-LT kernel provider" ON)
option(KERNELGEN_ENABLE_IREE "Enable IREE kernel provider" ON)

# =============================================================================
# Operations
# =============================================================================
if(KERNELGEN_ENABLE_GEMM)
  add_subdirectory(gemm)
endif()
