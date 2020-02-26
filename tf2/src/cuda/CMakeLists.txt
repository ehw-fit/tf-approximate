##========== Copyright (c) 2019, Filip Vaverka, All rights reserved. =========##
##
## Purpose:     Approximate TF operations with GPU support - CUDA sources
##
## $NoKeywords: $ApproxGPUOpsTF $CMakeLists.txt
## $Date:       $2019-05-30
##============================================================================##

set(SRCS_LIST )
set(CUDA_LINK_LIBRARIES_KEYWORD PRIVATE)

find_package(CUDA REQUIRED)
include_directories(${CUDA_TOOLKIT_ROOT_DIR}/..)
include_directories(${TF_INCLUDE_DIR})
include_directories("../")

set(SUB_SRCS
    approx_nn_conv_ops.cu
    approx_nn_conv_ops_gemm.cu
    approx_nn_conv_kernels.cu
    approx_ops_types.cu
)

PREPEND(SUB_SRCS ${CMAKE_CURRENT_SOURCE_DIR} ${SUB_SRCS})
set(SRCS_LIST ${SRCS_LIST} ${SUB_SRCS})

file(GLOB_RECURSE HEADERS_LIST "*.h" "*.hpp" "*.cuh")

set(TFAPPROX_CUDA_ARCHS "50" CACHE STRING "Semicolon separated list of NVIDIA sm_XY for which code should be compiled.")

foreach(CUDA_SM ${TFAPPROX_CUDA_ARCHS})
    set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} --generate-code arch=compute_${CUDA_SM},code=sm_${CUDA_SM}")
endforeach()

add_definitions(-DGOOGLE_CUDA=1 -DNDEBUG)
set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} --expt-relaxed-constexpr ${TF_CXXFLAGS} -Xcompiler -fPIC")

set(CMAKE_CXX_STANDARD 11)
cuda_add_library(cuda_gpu_backend ${SRCS_LIST} ${HEADERS_LIST})
