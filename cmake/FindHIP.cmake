# FindHIP.cmake Finds the HIP runtime within a TheRock / ROCm installation.
#
# Sets: HIP_FOUND HIP_INCLUDE_DIRS HIP_LIBRARIES HIP_HIPCC_EXECUTABLE

find_path(
  HIP_INCLUDE_DIR
  NAMES hip/hip_runtime.h
  HINTS "${THEROCK_PATH}/include"
  NO_DEFAULT_PATH)

find_library(
  HIP_LIBRARY
  NAMES amdhip64
  HINTS "${THEROCK_PATH}/lib"
  NO_DEFAULT_PATH)

find_program(
  HIP_HIPCC_EXECUTABLE
  NAMES hipcc
  HINTS "${THEROCK_PATH}/bin"
  NO_DEFAULT_PATH)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(HIP REQUIRED_VARS HIP_INCLUDE_DIR HIP_LIBRARY
                                                    HIP_HIPCC_EXECUTABLE)

if(HIP_FOUND)
  set(HIP_INCLUDE_DIRS "${HIP_INCLUDE_DIR}")
  set(HIP_LIBRARIES "${HIP_LIBRARY}")

  if(NOT TARGET hip::amdhip64)
    add_library(hip::amdhip64 SHARED IMPORTED)
    set_target_properties(
      hip::amdhip64
      PROPERTIES IMPORTED_LOCATION "${HIP_LIBRARY}"
                 INTERFACE_INCLUDE_DIRECTORIES "${HIP_INCLUDE_DIR}")
  endif()
endif()
