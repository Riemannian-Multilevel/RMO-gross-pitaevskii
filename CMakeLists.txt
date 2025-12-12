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
        operators.h
        lac.h
        util.h
        gpe_solve.h
        gpe.h
)
deal_ii_initialize_cached_variables()

# deal.II example programs with modifications
add_executable(step-1 steps/step-1.cc)
deal_ii_setup_target(step-1)

add_executable(step-2 steps/step-2.cc)
deal_ii_setup_target(step-2)
target_link_libraries(step-2 Boost::program_options)

add_executable(step-3 steps/step-3.cc)
deal_ii_setup_target(step-3)

add_executable(step-4 steps/step-4.cc)
deal_ii_setup_target(step-4)

add_executable(sparsity sparsity.cc)
target_sources(sparsity PRIVATE ${COMMON_HEADERS})
deal_ii_setup_target(sparsity)
target_link_libraries(sparsity Boost::program_options fmt::fmt)

# Main GPE program with library and tests
add_executable(main main.cc)
target_sources(main PRIVATE ${COMMON_HEADERS})
deal_ii_setup_target(main)
target_link_libraries(main Boost::program_options fmt::fmt)
