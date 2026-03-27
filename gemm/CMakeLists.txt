# =============================================================================
# GEMM operation
# =============================================================================

# Provider toggles (guarded by top-level + local options)
if(KERNELGEN_ENABLE_HIPBLASLT)
  add_subdirectory(kernel_providers/hipblaslt)
endif()
