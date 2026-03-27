# FindHipBlasLt.cmake Finds hipBLAS-LT within a TheRock / ROCm installation.
#
# Sets: HipBlasLt_FOUND HipBlasLt_INCLUDE_DIRS HipBlasLt_LIBRARIES

find_path(
  HipBlasLt_INCLUDE_DIR
  NAMES hipblaslt/hipblaslt.h
  HINTS "${THEROCK_PATH}/include"
  NO_DEFAULT_PATH)

find_library(
  HipBlasLt_LIBRARY
  NAMES hipblaslt
  HINTS "${THEROCK_PATH}/lib"
  NO_DEFAULT_PATH)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(HipBlasLt REQUIRED_VARS HipBlasLt_INCLUDE_DIR
                                                          HipBlasLt_LIBRARY)

if(HipBlasLt_FOUND)
  set(HipBlasLt_INCLUDE_DIRS "${HipBlasLt_INCLUDE_DIR}")
  set(HipBlasLt_LIBRARIES "${HipBlasLt_LIBRARY}")

  if(NOT TARGET hipblaslt::hipblaslt)
    add_library(hipblaslt::hipblaslt SHARED IMPORTED)
    set_target_properties(
      hipblaslt::hipblaslt
      PROPERTIES IMPORTED_LOCATION "${HipBlasLt_LIBRARY}"
                 INTERFACE_INCLUDE_DIRECTORIES "${HipBlasLt_INCLUDE_DIR}")
  endif()
endif()
