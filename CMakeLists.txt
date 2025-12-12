cmake_minimum_required(VERSION 3.28)
project(gpe)

set(CMAKE_CXX_STANDARD 20)
add_subdirectory(fmt)

find_package(deal.II 9.6.0 REQUIRED
        HINTS ${DEAL_II_DIR} ../ ../../ $ENV{DEAL_II_DIR}
)
#find_package(fmt CONFIG REQUIRED)
find_package(Boost CONFIG COMPONENTS program_options REQUIRED)

set(COMMON_HEADERS
        mesh.h
        dofs.h
        assemble.h
        function.h
        operators.h
        lac.h
        util.h
        descent.h
        gpe.h
        options.h
)
deal_ii_initialize_cached_variables()

# Evaluate sparsity patterns for different orderings
add_executable(sparsity sparsity.cc)
target_sources(sparsity PRIVATE ${COMMON_HEADERS})
deal_ii_setup_target(sparsity)
target_link_libraries(sparsity Boost::program_options fmt::fmt)

# Main GPE program
add_executable(main main.cc)
target_sources(main PRIVATE ${COMMON_HEADERS})
deal_ii_setup_target(main)
target_link_libraries(main Boost::program_options fmt::fmt)

# Multigrid GPE program
add_executable(main_mg main_mg.cc)
target_sources(main_mg PRIVATE ${COMMON_HEADERS})
deal_ii_setup_target(main_mg)
target_link_libraries(main_mg Boost::program_options fmt::fmt)
