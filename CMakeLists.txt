cmake_minimum_required(VERSION 3.28)
project(gpe)

set(CMAKE_CXX_STANDARD 20)
add_subdirectory(fmt)

find_package(deal.II 9.7.0 REQUIRED
        HINTS ${DEAL_II_DIR} ../ ../../ $ENV{DEAL_II_DIR}
)
#find_package(fmt CONFIG REQUIRED)
find_package(Boost CONFIG COMPONENTS program_options REQUIRED)
deal_ii_initialize_cached_variables()

# Find all .h (header) files in the include/ directory
file(GLOB_RECURSE HEADER_ALL CONFIGURE_DEPENDS "include/*.h")

# Loop through each discovered source file
function(add_gpe_executable TARGET_NAME)
    # Construct the path to the source file
    set(SRC_FILE "${CMAKE_CURRENT_SOURCE_DIR}/src/${TARGET_NAME}.cc")

    # Optional but recommended: Check if the file exists to catch typos early
    if(NOT EXISTS "${SRC_FILE}")
        message(FATAL_ERROR "Cannot create target '${TARGET_NAME}': Source file ${SRC_FILE} not found.")
    endif()

    # Create an executable for this specific source file
    add_executable(${TARGET_NAME} ${SRC_FILE})
    target_sources(${TARGET_NAME} PRIVATE ${HEADER_ALL})

    # Apply the include directories to this target
    target_include_directories(${TARGET_NAME} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)

    # Apply deal.II configurations and link external libraries
    deal_ii_setup_target(${TARGET_NAME})
    target_link_libraries(${TARGET_NAME} Boost::program_options fmt::fmt)
endfunction()

add_gpe_executable(main)
add_gpe_executable(main_bench)
add_gpe_executable(main_coarse)
add_gpe_executable(main_sparsity)

enable_testing()

add_gpe_executable(test_descent)
add_gpe_executable(test_gradient)
add_test(descent test_descent COMMAND test_descent)
add_test(gradient test_gradient COMMAND test_gradient)
