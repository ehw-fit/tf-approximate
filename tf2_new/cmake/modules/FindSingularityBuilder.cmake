##========== Copyright (c) 2020, Filip Vaverka, All rights reserved. =========##
##
## Purpose:     Simple CMake Module to integrate Singularity container builds
##
## $NoKeywords: $ApproxTF $FindSingularityBuilder.cmake
## $Date:       $2020-02-25
##============================================================================##

function(AddFileToContainer container src_file dst_file type)
    if(${type} STREQUAL BUILD)
        set_property(TARGET ${container} APPEND_STRING PROPERTY CONTAINER_INSTALL_FILE_LIST "    $<TARGET_PROPERTY:${container},CONTAINER_BUILD_ROOT>/${src_file} ${dst_file}\n")
        set_property(TARGET ${container} APPEND PROPERTY CONTAINER_PACKAGE_FILES $<TARGET_PROPERTY:${container},CONTAINER_SOURCE_ROOT>/${src_file})
    elseif(${type} STREQUAL SOURCE)
        set_property(TARGET ${container} APPEND_STRING PROPERTY CONTAINER_INSTALL_FILE_LIST "    $<TARGET_PROPERTY:${container},CONTAINER_SOURCE_ROOT>/${src_file} ${dst_file}\n")
        set_property(TARGET ${container} APPEND PROPERTY CONTAINER_PACKAGE_FILES $<TARGET_PROPERTY:${container},CONTAINER_SOURCE_ROOT>/${src_file})
    endif()
endfunction()

function(AddTargetToContainer container target dst_file)
    set_property(TARGET ${container} APPEND_STRING PROPERTY CONTAINER_INSTALL_FILE_LIST "    $<TARGET_PROPERTY:${container},CONTAINER_BUILD_ROOT>/$<TARGET_FILE_NAME:${target}> ${dst_file}\n")
    set_property(TARGET ${container} APPEND PROPERTY CONTAINER_BUILD_FILES $<TARGET_PROPERTY:${target},SOURCES>)
endfunction()

function(AddEnvVarToContainer container name path)
    set_property(TARGET ${container} APPEND_STRING PROPERTY CONTAINER_ENV_VARS "    ${name}=\$${name}:${path}\n")
endfunction()

function(AddSingularityContainer container)
    set(CONTAINER_OUTPUT_FILE ${PROJECT_BINARY_DIR}/${container}.sif)
    set(CONTAINER_SOURCE_ROOT ${PROJECT_SOURCE_DIR})
    set(CONTAINER_BUILD_ROOT  ${CMAKE_CURRENT_BINARY_DIR}/${container})
    set(CONTAINER_DEF_FILE    ${CMAKE_CURRENT_BINARY_DIR}/${container}.def)
    
    execute_process(COMMAND id -u OUTPUT_VARIABLE CONTAINER_USER_ID OUTPUT_STRIP_TRAILING_WHITESPACE)
    execute_process(COMMAND id -g OUTPUT_VARIABLE CONTAINER_GROUP_ID OUTPUT_STRIP_TRAILING_WHITESPACE)
    
    add_custom_target(${container} DEPENDS ${CONTAINER_OUTPUT_FILE})
    set_property(TARGET ${container} PROPERTY CONTAINER_OUTPUT_FILE ${CONTAINER_OUTPUT_FILE})
    set_property(TARGET ${container} PROPERTY CONTAINER_SOURCE_ROOT ${CONTAINER_SOURCE_ROOT})
    set_property(TARGET ${container} PROPERTY CONTAINER_BUILD_ROOT ${CONTAINER_BUILD_ROOT})
    set_property(TARGET ${container} PROPERTY CONTAINER_USER_ID ${CONTAINER_USER_ID})
    set_property(TARGET ${container} PROPERTY CONTAINER_GROUP_ID ${CONTAINER_GROUP_ID})
    
    add_custom_command(
        OUTPUT ${CONTAINER_OUTPUT_FILE}
        COMMAND test -e ${CONTAINER_BUILD_ROOT}/.build_done
        COMMAND sudo singularity build -F ${CONTAINER_OUTPUT_FILE} ${CONTAINER_DEF_FILE}
        MAIN_DEPENDENCY ${CONTAINER_DEF_FILE}
        DEPENDS ${CONTAINER_BUILD_ROOT}/.build_done $<TARGET_GENEX_EVAL:${container},$<TARGET_PROPERTY:${container},CONTAINER_PACKAGE_FILES>>
        USES_TERMINAL)
    
    add_custom_command(
        OUTPUT ${CONTAINER_BUILD_ROOT}/.build_done
        BYPRODUCTS ${CONTAINER_BUILD_ROOT}
        COMMAND ${CMAKE_COMMAND} -E remove ${CONTAINER_BUILD_ROOT}/.build_done
        COMMAND ${CMAKE_COMMAND} -E make_directory ${CONTAINER_BUILD_ROOT}
        COMMAND docker run -it --rm --mount type=bind,source=${CONTAINER_SOURCE_ROOT},target=/opt/${container} --mount type=bind,source=${CMAKE_CURRENT_BINARY_DIR},target=/opt/${container}-build ${CONTAINER_BUILD_IMAGE} bash -e /opt/${container}-build/${container}.sh
        DEPENDS $<TARGET_GENEX_EVAL:${container},$<TARGET_PROPERTY:${container},CONTAINER_BUILD_FILES>>
        USES_TERMINAL)
    
    configure_file(${CMAKE_CURRENT_SOURCE_DIR}/${container}.sh.in ${CMAKE_CURRENT_BINARY_DIR}/${container}.sh.gen @ONLY)
    file(GENERATE OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/${container}.sh INPUT ${CMAKE_CURRENT_BINARY_DIR}/${container}.sh.gen)
    
    set(CONTAINER_ENV_LIST $<TARGET_PROPERTY:${container},CONTAINER_ENV_VARS>)
    set(CONTAINER_FILE_LIST $<TARGET_PROPERTY:${container},CONTAINER_INSTALL_FILE_LIST>)
    configure_file(${CMAKE_CURRENT_SOURCE_DIR}/${container}.def.in ${CMAKE_CURRENT_BINARY_DIR}/${container}.def.conf @ONLY)
    file(GENERATE OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/${container}.def.gen INPUT ${CMAKE_CURRENT_BINARY_DIR}/${container}.def.conf)
    file(GENERATE OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/${container}.def INPUT ${CMAKE_CURRENT_BINARY_DIR}/${container}.def.gen)
endfunction()
