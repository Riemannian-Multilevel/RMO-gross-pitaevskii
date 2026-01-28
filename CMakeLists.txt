cmake_minimum_required(VERSION 3.28)
project(gpe)

set(CMAKE_CXX_STANDARD 20)
add_subdirectory(fmt)

find_package(deal.II 9.7.0 REQUIRED
        HINTS ${DEAL_II_DIR} ../ ../../ $ENV{DEAL_II_DIR}
)
#find_package(fmt CONFIG REQUIRED)
find_package(Boost CONFIG COMPONENTS program_options REQUIRED)

set(COMMON_HEADERS
        fe_space.h
        fe_space_mg.h
        grid.h
        assemble.h
        function.h
        lac.h
        util.h
        descent.h
        option.h
        option_util.h
        option_types.h
        sparsity.h
        main.h
)
deal_ii_initialize_cached_variables()

# Evaluate sparsity patterns for different orderings
add_executable(main_sparsity main_sparsity.cc)
target_sources(main_sparsity PRIVATE ${COMMON_HEADERS})
deal_ii_setup_target(main_sparsity)
target_link_libraries(main_sparsity Boost::program_options fmt::fmt)

# Main GPE program
add_executable(main main.cc)
target_sources(main PRIVATE ${COMMON_HEADERS})
deal_ii_setup_target(main)
target_link_libraries(main Boost::program_options fmt::fmt)

# Benchmarking
add_executable(main_bench main_bench.cc)
target_sources(main_bench PRIVATE ${COMMON_HEADERS})
deal_ii_setup_target(main_bench)
target_link_libraries(main_bench fmt::fmt)
