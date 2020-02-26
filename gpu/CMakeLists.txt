##========== Copyright (c) 2019, Filip Vaverka, All rights reserved. =========##
##
## Purpose:     Approximate TF operations with GPU support
##
## $NoKeywords: $ApproxGPUOpsTF $CMakeLists.txt
## $Date:       $2019-05-29
##============================================================================##

cmake_minimum_required(VERSION 3.1)
project(ApproxGPUOpsTF)

# Appends the cmake/modules path to MAKE_MODULE_PATH variable.
set(CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake/modules ${CMAKE_MODULE_PATH})

function(PREPEND var prefix)
    set(listVar "")
    foreach(f ${ARGN})
        list(APPEND listVar "${prefix}/${f}")
    endforeach(f)
    set(${var} "${listVar}" PARENT_SCOPE)
endfunction(PREPEND)

option(TFAPPROX_FORCE_REF_CONV_CPU "Force use of reference Conv2D implementation." OFF)
option(TFAPPROX_ALLOW_GPU_CONV "Allow use of GPU for Conv2D" ON)
option(TFAPPROX_ALLOW_CPU_CONV "Allow use of GPU for Conv2D" ON)

if(TFAPPROX_FORCE_REF_CONV_CPU)
    set(TFAPPROX_ALLOW_GPU_CONV OFF)
    set(TFAPPROX_ALLOW_CPU_CONV OFF)
    add_definitions(-DFORCE_REF_CPU_APPROX_CONV)
endif()

set(TFAPPROX_BACKEND_LIBRARIES )

if(TFAPPROX_ALLOW_CPU_CONV)
    add_definitions(-DALLOW_CPU_FOR_APPROX_CONV)
endif()

if(TFAPPROX_ALLOW_GPU_CONV)
    set(TFAPPROX_BACKEND_LIBRARIES cuda_gpu_backend)
    add_definitions(-DALLOW_GPU_FOR_APPROX_CONV)
    add_definitions(-DGOOGLE_CUDA=1)
endif()

find_package(Tensorflow REQUIRED)
include_directories(${TF_INCLUDE_DIR})

file(GLOB_RECURSE HEADERS_LIST "src/*.h" "src/*.hpp")

set(SRCS_LIST)

add_subdirectory(src)
include_directories(${INCLUDE_DIRECTORIES})

add_subdirectory(containers)

set(CMAKE_CXX_STANDARD 11)

add_library(${PROJECT_NAME} SHARED ${SRCS_LIST} ${HEADERS_LIST})
target_compile_options(${PROJECT_NAME} PRIVATE ${TF_CXXFLAGS})
target_link_libraries(${PROJECT_NAME} PRIVATE ${TFAPPROX_BACKEND_LIBRARIES} ${TF_LDFLAGS})
set_property(TARGET ${PROJECT_NAME} PROPERTY POSITION_INDEPENDENT_CODE 1)
