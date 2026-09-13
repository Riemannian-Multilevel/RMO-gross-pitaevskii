cmake_minimum_required(VERSION 3.28)
project(rmo)

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
function(rmo_strip_sdk_includes _target)
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
        rmo_strip_sdk_includes(${_dealii_tgt})
    endforeach()
endif()

# Find all .h (header) files in the include/ directory
file(GLOB_RECURSE HEADER_ALL "include/*.h")

function(add_rmo_executable TARGET_NAME)
    # Parse the arguments
    set(options "")
    set(oneValueArgs SRC_DIR)
    set(multiValueArgs LINK_LIBS EXTRA_SOURCES)
    cmake_parse_arguments(ARG "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})

    # Executables (main*) live under src/ by default; tests pass SRC_DIR to
    # point at their test/<subdir> location instead (see below).
    if(NOT ARG_SRC_DIR)
        set(ARG_SRC_DIR "src")
    endif()

    # Construct the path to the source file
    set(SRC_FILE "${CMAKE_CURRENT_SOURCE_DIR}/${ARG_SRC_DIR}/${TARGET_NAME}.cc")

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
add_rmo_executable(main LINK_LIBS Boost::program_options fmt::fmt)
add_rmo_executable(main_bench LINK_LIBS Boost::program_options fmt::fmt)
add_rmo_executable(main_cc)
add_rmo_executable(main_cc_coarse)
add_rmo_executable(main_cc_bench)
add_rmo_executable(main_cc_cow)
add_rmo_executable(main_cc_cow_levels)
add_rmo_executable(main_coarse LINK_LIBS Boost::program_options fmt::fmt)
add_rmo_executable(main_sparsity LINK_LIBS Boost::program_options fmt::fmt)

# --- Create targets for tests
enable_testing()

add_rmo_executable(test_arpack SRC_DIR test/gpe LINK_LIBS Boost::program_options fmt::fmt)
add_rmo_executable(test_coherence SRC_DIR test/gpe)
add_rmo_executable(test_descent SRC_DIR test/gpe LINK_LIBS fmt::fmt)
add_rmo_executable(test_gradient SRC_DIR test/gpe LINK_LIBS Boost::program_options fmt::fmt)
add_rmo_executable(test_matrix_free SRC_DIR test/gpe LINK_LIBS Boost::program_options fmt::fmt)

add_rmo_executable(test_operator SRC_DIR test/fe)

add_rmo_executable(test_interpolate SRC_DIR test/cc)
add_rmo_executable(test_manifold SRC_DIR test/cc)
add_rmo_executable(test_oracle SRC_DIR test/cc)
add_rmo_executable(test_oracle_coarse SRC_DIR test/cc)

# ARPACK is optional in deal.II; the test reports 77 (skip) when it is unavailable
add_test(NAME arpack COMMAND test_arpack --dimension 2 --level 5 --radius 8 --tol-residual 1e-10)
set_tests_properties(arpack PROPERTIES SKIP_RETURN_CODE 77)

add_test(NAME cc_interpolate COMMAND test_interpolate)
add_test(NAME cc_manifold COMMAND test_manifold)
add_test(NAME cc_oracle COMMAND test_oracle)
add_test(NAME cc_oracle_coarse COMMAND test_oracle_coarse)
add_test(NAME coherence COMMAND test_coherence)
add_test(NAME descent COMMAND test_descent)
add_test(NAME gradient COMMAND test_gradient --level 6 --trials 10
         --boundary dirichlet --order cuthill_mckee)
add_test(NAME matrix_free COMMAND test_matrix_free --min-level 4 --max-level 6 --trials 20)
add_test(NAME operator COMMAND test_operator)
