# =============================================================================
# GEMM operation
# =============================================================================

# Shared utilities (npy loader, JSON parsing, verification).
add_subdirectory(utils)

# Provider toggles (guarded by top-level + local options)
if(KERNELGEN_ENABLE_HIPBLASLT)
  add_subdirectory(kernel_providers/hipblaslt)
endif()

if(KERNELGEN_ENABLE_FUSILLI)
  add_subdirectory(kernel_providers/fusilli)
endif()

if(KERNELGEN_ENABLE_IREE)
  add_subdirectory(kernel_providers/iree)
endif()

if(KERNELGEN_ENABLE_NATIVE_HIP)
  add_subdirectory(kernel_providers/native_hip)
endif()
