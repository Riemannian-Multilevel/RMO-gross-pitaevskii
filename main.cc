//
// Created by Ferdinand Vanmaele on 01.10.25.
//
#include "gpe.h"
#include "manifold.h"
#include "option.h"
#include "util.h"

#include <iostream>
#include <fmt/format.h>

using namespace gpe;
using namespace dealii;


int main(int argc, char* argv[])
{
    GPE_Options options{};
    GdOptions   options_rgd{};
    MG_Options  options_mg{};

    // TODO: add configuration file (cf. boost tutorial)
    try {
        po::options_description all("Allowed options");
        all.add_options()("help", "produce help message");
        all.add(gpe_cli_options());
        all.add(gd_cli_options());
        all.add(mg_cli_options());

        po::variables_map vm;
        po::store(po::parse_command_line(argc, argv, all), vm);
        po::notify(vm);

        if (vm.contains("help")) {
            std::cout << all << "\n";
            return 0;
        }
        apply_gpe_options(vm, options);
        apply_gd_options(vm, options_rgd);
        apply_mg_options(vm, options_mg);

        with_dimension(options.dimension, [&](auto D)
        {
            constexpr int dim = decltype(D)::value;
            unsigned int min_level = options_mg.multilevel ? options_mg.min_level : options_mg.max_level-1;
            unsigned int max_level = options_mg.max_level;

            for (unsigned int level = min_level; level < max_level; ++level) {
                GrossPitaevskiiPackage<dim> GS(options, level+1);

                // Run gradient descent
                auto x = GS.run(Square<dim>(), 1.0, options.beta, options_rgd, std::cout);

                // Plot solution
                std::string filename = fmt::format("solution_{}d_lvl{}.vtk", dim, level);
                output_results(x, GS.fe_space().get_dofs(), DataOutBase::OutputFormat::vtk, filename);
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
