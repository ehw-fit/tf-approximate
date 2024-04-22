##========== Copyright (c) 2020, Filip Vaverka, All rights reserved. =========##
##
## Purpose:     Simple CMake Module to integrate Singularity container builds
##
## $NoKeywords: $ApproxTF $CMakeLists.cmake
## $Date:       $2020-02-25
##============================================================================##

find_package(SingularityBuilder)

set(CONTAINER_BUILD_IMAGE tensorflow/tensorflow:latest-gpu-py3)
set(CONTAINER_DIST_IMAGE  tensorflow/tensorflow:latest-gpu-py3)
AddSingularityContainer(tf-approximate-gpu)

AddTargetToContainer(tf-approximate-gpu ApproxGPUOpsTF /opt/tf-approximate-gpu/)

AddFileToContainer(tf-approximate-gpu python/keras/layers/fake_approx_convolutional.py /opt/tf-approximate-gpu/python/keras/layers/ SOURCE)
AddFileToContainer(tf-approximate-gpu test/test_table_approx_conv_2d.py /opt/tf-approximate-gpu/test/ SOURCE)
AddFileToContainer(tf-approximate-gpu test/test_mul_table.bin /opt/tf-approximate-gpu/test/ SOURCE)

AddEnvVarToContainer(tf-approximate-gpu PYTHONPATH /opt/tf-approximate-gpu/python)
AddEnvVarToContainer(tf-approximate-gpu LD_LIBRARY_PATH /opt/tf-approximate-gpu)
