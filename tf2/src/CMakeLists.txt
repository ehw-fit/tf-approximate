##========== Copyright (c) 2019, Filip Vaverka, All rights reserved. =========##
##
## Purpose:     Approximate TF operations with GPU support - sources
##
## $NoKeywords: $ApproxGPUOpsTF $CMakeLists.txt
## $Date:       $2019-05-30
##============================================================================##

if(TFAPPROX_ALLOW_GPU_CONV)
    add_subdirectory(cuda)
endif()

set(INCLUDE_DIRECTORIES ${CMAKE_CURRENT_SOURCE_DIR} PARENT_SCOPE)

set(SUB_SRCS
    approx_nn_ops.cpp
    approx_nn_conv_ops.cpp
    approx_nn_conv_ops_gemm.cpp
    approx_nn_conv_kernels.cpp
    approx_nn_conv_ops_ref.cpp
)

PREPEND(SUB_SRCS ${CMAKE_CURRENT_SOURCE_DIR} ${SUB_SRCS})
set(SRCS_LIST ${SRCS_LIST} ${SUB_SRCS} PARENT_SCOPE)
