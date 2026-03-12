//
// Created by Ferdinand Vanmaele on 01.10.25.
//
#include "main.h"
#include "option.h"
#include "util.h"

#include <iostream>
#include <fmt/format.h>

using namespace gpe;
using namespace dealii;

int main(int argc, char* argv[])
{
    GPE_Options options{};
    DescentOptions options_rgd{};
    SolverOptions options_slv{};
    MG_Options options_mg{};

    // TODO: add configuration file (cf. boost tutorial)
    try {
        po::options_description all("Allowed options");
        all.add_options()("help", "produce help message");
        all.add(gpe_cli_options());
        all.add(descent_cli_options());
        all.add(mg_cli_options());
        all.add(inner_cli_options());

        po::variables_map vm;
        po::store(po::parse_command_line(argc, argv, all), vm);
        po::notify(vm);

        if (vm.contains("help")) {
            std::cout << all << "\n";
            return 0;
        }
        apply_gpe_options(vm, options);
        apply_descent_options(vm, options_rgd);
        apply_mg_options(vm, options_mg);
        apply_inner_options(vm, options_slv);

        with_dimension(options.dimension, [&](auto D)
        {
            constexpr int dim = decltype(D)::value;
            unsigned int min_level = options_mg.multilevel ? options_mg.min_level : options_mg.max_level-1;
            unsigned int max_level = options_mg.max_level;

            for (unsigned int level = min_level; level < max_level; ++level) {
                // Initialize the orchestrator (Simulator)
                // This sets up the mesh (Package) and Finite Element space
                EnergySimulator<dim> simulator(Square<dim>(), options, level + 1);

                // Set starting value, sufficiently far from an optimal solution
                Vector<double> x0(simulator.n_dofs());
                x0 = 1.0;

                // Run the Riemannian Gradient Descent pipeline
                // Square<dim>() is passed as the Potential V
                // options_rgd contains the solver tolerances and step size
                // options.beta is the non-linear coupling constant
                simulator.distribute(x0);
                auto x = simulator.run(x0, options.beta, options_slv, options_rgd, std::cout);

                // Plot solution
                std::string filename = fmt::format("solution_{}d_lvl{}.vtk", dim, level);
                output_results(x, simulator.get_package().get_dofs(), DataOutBase::OutputFormat::vtk, filename);
            }
        });
    }
    catch (std::exception& e) {
        std::cerr << "error: " << e.what() << "\n";
        return 1;
    }
    catch (...) {
        std::cerr << "Exception of unknown type!\n";
        return 1;
    }
}
