##========== Copyright (c) 2020, Filip Vaverka, All rights reserved. =========##
##
## Purpose:     Simple CMake Module to integrate Singularity container builds
##
## $NoKeywords: $ApproxTF $FindSingularityBuilder.cmake
## $Date:       $2020-02-25
##============================================================================##

function(BuildSingularityContainer container_name image_name)
    set(CONTAINER_SRC_ROOT ${PROJECT_SOURCE_DIR})
    set(CONTAINER_BUILD_ROOT ${CMAKE_CURRENT_BINARY_DIR}/${container_name})
    
    execute_process(COMMAND id -u OUTPUT_VARIABLE CONTAINER_USER_ID OUTPUT_STRIP_TRAILING_WHITESPACE)
    execute_process(COMMAND id -g OUTPUT_VARIABLE CONTAINER_GROUP_ID OUTPUT_STRIP_TRAILING_WHITESPACE)
    
    configure_file(${CMAKE_CURRENT_SOURCE_DIR}/${container_name}.sh.in ${CMAKE_CURRENT_BINARY_DIR}/${container_name}.sh @ONLY)
    configure_file(${CMAKE_CURRENT_SOURCE_DIR}/${container_name}.def.in ${CMAKE_CURRENT_BINARY_DIR}/${container_name}.def @ONLY)
    
    add_custom_command(
        OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/${container_name}
        COMMAND ${CMAKE_COMMAND} -E make_directory ${CMAKE_CURRENT_BINARY_DIR}/${container_name}
        COMMAND docker run -it --rm --mount type=bind,source=${PROJECT_SOURCE_DIR},target=/opt/${container_name} --mount type=bind,source=${CMAKE_CURRENT_BINARY_DIR},target=/opt/${container_name}_build ${image_name} bash -e /opt/${container_name}_build/${container_name}.sh
        MAIN_DEPENDENCY ${CMAKE_CURRENT_BINARY_DIR}/${container_name}.def
        USES_TERMINAL)
    add_custom_target(${container_name}-build
        SOURCES ${CMAKE_CURRENT_BINARY_DIR}/${container_name})
        
    add_custom_command(
        OUTPUT ${PROJECT_BINARY_DIR}/${container_name}.sif
        COMMAND sudo singularity build ${PROJECT_BINARY_DIR}/${container_name}.sif ${CMAKE_CURRENT_BINARY_DIR}/${container_name}.def
        MAIN_DEPENDENCY ${CMAKE_CURRENT_BINARY_DIR}/${container_name}.def
        DEPENDS ${container_name}-build
        USES_TERMINAL)
    
    add_custom_target(${container_name}-container SOURCES ${PROJECT_BINARY_DIR}/${container_name}.sif)
    add_custom_target(containers-clean COMMAND rm -rf ${CMAKE_CURRENT_BINARY_DIR}/${container_name} ${PROJECT_BINARY_DIR}/${container_name}.sif)
endfunction()
