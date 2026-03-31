# Pure C++ utilities (no HIP dependency).
add_library(kernelgen_gemm_utils STATIC src/bench_utils.cpp)
target_include_directories(kernelgen_gemm_utils
                           PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)

# HIP-dependent utilities (device allocation helpers).
find_package(HIP REQUIRED)
set(CMAKE_CXX_COMPILER "${HIP_HIPCC_EXECUTABLE}")

add_library(kernelgen_gemm_hip_utils STATIC src/hip_utils.cpp)
target_include_directories(kernelgen_gemm_hip_utils
                           PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(kernelgen_gemm_hip_utils PUBLIC kernelgen_gemm_utils
                                                      hip::amdhip64)
target_compile_options(kernelgen_gemm_hip_utils
                       PRIVATE --offload-arch=${GPU_TARGETS})
