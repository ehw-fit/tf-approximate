##========== Copyright (c) 2019, Filip Vaverka, All rights reserved. =========##
##
## Purpose:     Simple CMake Module to find build flags for user TF ops
##
## $NoKeywords: $ApproxTF $FindTensorflow.cmake
## $Date:       $2019-05-29
##============================================================================##

include(FindPackageHandleStandardArgs)

function (QueryTF var query)
    if(${var})
        return()
    endif()

    execute_process(COMMAND ${PYTHON_EXECUTABLE} -c "import tensorflow as tf; res = tf.sysconfig.${query}(); print(res if isinstance(res, str) else ' '.join(res))"
        RESULT_VARIABLE result
        OUTPUT_VARIABLE output
        ERROR_QUIET
        OUTPUT_STRIP_TRAILING_WHITESPACE)
    
    if(result EQUAL 0)
        separate_arguments(output UNIX_COMMAND ${output})
        set(${var} "${output}" CACHE INTERNAL "Tensorflow provided sysconfig.${query}()")
    endif()
endfunction()

find_program(PYTHON_EXECUTABLE NAMES python3 python)
mark_as_advanced(PYTHON_EXECUTABLE)

if(PYTHON_EXECUTABLE)
    QueryTF(TF_CXXFLAGS "get_compile_flags")
    QueryTF(TF_LDFLAGS "get_link_flags")
    QueryTF(TF_INCLUDE_DIR "get_include")
endif()

find_package_handle_standard_args(Tensorflow REQUIRED_VARS TF_INCLUDE_DIR TF_CXXFLAGS TF_LDFLAGS)
