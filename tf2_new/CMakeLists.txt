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
set(CMAKE_EXPORT_COMPILE_COMMANDS 1)
find_package(Protobuf REQUIRED)
message(" include dirs: ${Protobuf_INCLUDE_DIRS}")
include_directories(${Protobuf_INCLUDE_DIRS})
function(PREPEND var prefix)
    set(listVar "")
    foreach(f ${ARGN})
        list(APPEND listVar "${prefix}/${f}")
    endforeach(f)
    set(${var} "${listVar}" PARENT_SCOPE)
endfunction(PREPEND)

option(TFAPPROX_FORCE_REF_CONV_CPU "Force use of reference Conv2D implementation." OFF)
option(TFAPPROX_ALLOW_GPU_CONV "Allow use of GPU for Conv2D" ON)
option(TFAPPROX_ALLOW_CPU_CONV "Allow use of CPU for Conv2D" OFF)
option(TFAPPROX_CUDA_GEN_PTX "Include PTX code in the binary to allow JIT for future GPUs" ON)

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

set(TF_CFLAGS "-I/home/michalpinos/miniforge3/envs/tfapprox211/lib/python3.9/site-packages/tensorflow/include -D_GLIBCXX_USE_CXX11_ABI=1 --std=c++17 -DEIGEN_MAX_ALIGN_BYTES=64") 
set(TF_LFLAGS "-L/home/michalpinos/miniforge3/envs/tfapprox211/lib/python3.9/site-packages/tensorflow -l:libtensorflow_framework.so.2") 
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O2 -fPIC -ffast-math -march=native -mno-avx -g ${TF_CFLAGS} ${TF_LFLAGS} -D_GLIBCXX_USE_CXX11_ABI=0" )


add_library(${PROJECT_NAME} SHARED ${SRCS_LIST} ${HEADERS_LIST})
target_compile_options(${PROJECT_NAME} PRIVATE ${TF_CXXFLAGS})
target_link_libraries(${PROJECT_NAME} PRIVATE ${TFAPPROX_BACKEND_LIBRARIES} ${TF_LDFLAGS} "-Wl,--no-undefined -D_GLIBCXX_USE_CXX11_ABI=0")
set_property(TARGET ${PROJECT_NAME} PROPERTY POSITION_INDEPENDENT_CODE 1)

