cmake_minimum_required(VERSION 3.28)
project(gpe)

set(CMAKE_CXX_STANDARD 20)
add_subdirectory(fmt)

find_package(deal.II 9.6.0 REQUIRED
        HINTS ${DEAL_II_DIR} ../ ../../ $ENV{DEAL_II_DIR}
)
#find_package(fmt CONFIG REQUIRED)
find_package(Boost CONFIG COMPONENTS program_options REQUIRED)
deal_ii_initialize_cached_variables()

# https://github.com/dealii/dealii/issues/18960
# macOS only: deal.II's Trilinos/Kokkos interface targets carry SDK include
# paths (.../<sdk>.sdk/usr/include) that shadow the real toolchain headers and
# break the build. Strip them. On other platforms these paths do not exist, so
# the workaround is unnecessary.
function(gpe_strip_sdk_includes _target)
    if(NOT TARGET ${_target})
        return()
    endif()
    foreach(_prop INTERFACE_INCLUDE_DIRECTORIES INTERFACE_SYSTEM_INCLUDE_DIRECTORIES)
        get_target_property(_dirs ${_target} ${_prop})
        if(_dirs)
            list(FILTER _dirs EXCLUDE REGEX "\\.sdk/usr/include/?$")
            set_target_properties(${_target} PROPERTIES ${_prop} "${_dirs}")
        endif()
    endforeach()
endfunction()

if(APPLE)
    foreach(_dealii_tgt dealii::interface_trilinos dealii::interface_kokkos)
        gpe_strip_sdk_includes(${_dealii_tgt})
    endforeach()
endif()

# Find all .h (header) files in the include/ directory
file(GLOB_RECURSE HEADER_ALL "include/*.h")

function(add_gpe_executable TARGET_NAME)
    # Parse the arguments
    set(options "")
    set(oneValueArgs "")
    set(multiValueArgs LINK_LIBS EXTRA_SOURCES)
    cmake_parse_arguments(ARG "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})

    # Construct the path to the source file
    set(SRC_FILE "${CMAKE_CURRENT_SOURCE_DIR}/src/${TARGET_NAME}.cc")

    if(NOT EXISTS "${SRC_FILE}")
        message(FATAL_ERROR "Cannot create target '${TARGET_NAME}': Source file ${SRC_FILE} not found.")
    endif()

    # Create an executable for this specific source file
    add_executable(${TARGET_NAME} ${SRC_FILE} ${ARG_EXTRA_SOURCES})
    target_sources(${TARGET_NAME} PRIVATE ${HEADER_ALL})

    # Apply the include directories to this target
    target_include_directories(${TARGET_NAME} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)

    # Apply deal.II configurations
    deal_ii_setup_target(${TARGET_NAME})

    # Build up additional libraries
#    set(LIBRARIES_TO_LINK fmt::fmt)
#    if(ARG_BOOST_PO)
#        list(APPEND LIBRARIES_TO_LINK Boost::program_options)
#    endif()

    target_link_libraries(${TARGET_NAME} ${ARG_LINK_LIBS})
endfunction()

#add_compile_definitions(CPU_TIME)

# --- Create targets for executables
add_gpe_executable(main LINK_LIBS Boost::program_options fmt::fmt)
add_gpe_executable(main_bench LINK_LIBS Boost::program_options fmt::fmt)
add_gpe_executable(main_coarse LINK_LIBS Boost::program_options fmt::fmt)
add_gpe_executable(main_sparsity LINK_LIBS Boost::program_options fmt::fmt)
add_gpe_executable(main_operator)

# --- Create targets for tests
enable_testing()

add_gpe_executable(test_descent LINK_LIBS fmt::fmt)
add_gpe_executable(test_gradient LINK_LIBS fmt::fmt)
#add_gpe_executable(test_matrix_free LINK_LIBS fmt::fmt)

add_test(descent test_descent COMMAND test_descent)
add_test(gradient test_gradient COMMAND test_gradient)
#add_test(matrix_free test_matrix_free COMMAND test_matrix_free)
