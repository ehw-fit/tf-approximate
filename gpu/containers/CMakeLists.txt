##========== Copyright (c) 2020, Filip Vaverka, All rights reserved. =========##
##
## Purpose:     Simple CMake Module to integrate Singularity container builds
##
## $NoKeywords: $ApproxTF $CMakeLists.cmake
## $Date:       $2020-02-25
##============================================================================##

find_package(SingularityBuilder)

BuildSingularityContainer(tf-approximate-gpu tensorflow/tensorflow:latest-gpu-py3)
