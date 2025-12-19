//
// Created by Ferdinand Vanmaele on 01.10.25.
//
#include "main.h"
#include "function.h"
#include "options.h"

#include <iostream>
#include <fmt/format.h>

using namespace gpe;
using namespace dealii;

template <int dim, typename ExecutionPolicy>
void package(const GPE_Options& options, const GdOptions& options_rgd)
{
    Square<dim> V;
    GPE_Solve<dim, ExecutionPolicy> GS(options);
    GS.setup();

    // Run gradient descent
    auto x = GS.run(V, 1.0, options.beta, options_rgd);

    // Plot solution
    using dealii::DataOutBase::OutputFormat::vtu;
    output_results(x, GS.get_dofs(), vtu, fmt::format("solution_{}d.vtu", dim));
}

template <int dim>
void run_package(bool parallel, const GPE_Options& options, const GdOptions& options_rgd)
{
    if (parallel) {
        package<dim, execution::par_t>(options, options_rgd);
    } else {
        package<dim, execution::seq_t>(options, options_rgd);
    }
}

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

        if (vm.count("help")) {
            std::cout << all << "\n";
            return 0;
        }
        apply_gpe_options(vm, options);
        apply_gd_options(vm, options_rgd);
        apply_mg_options(vm, options_mg);

        if (options_mg.multigrid) {
            throw std::logic_error("Multigrid is not supported in this program!");
        }

        with_dimension(options.dimension, [&](auto D)
        {
            constexpr int dim = decltype(D)::value;

            run_package<dim>(options_mg.parallel, options, options_rgd);
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
