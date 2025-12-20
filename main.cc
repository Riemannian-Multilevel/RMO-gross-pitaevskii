//
// Created by Ferdinand Vanmaele on 01.10.25.
//
#include "main.h"
#include "main_mg.h"
#include "function.h"
#include "options.h"

#include <iostream>
#include <fmt/format.h>
#include <deal.II/numerics/data_out.h>

#define N_CHECK_RES 1

using namespace gpe;
using namespace dealii;


//!
//! @tparam dim Problem dimension
//! @param solution
//! @param dof_handler
//! @param format
//! @param filename
template <int dim>
void output_results(const Vector<double>& solution, const dealii::DoFHandler<dim>& dof_handler,
    const DataOutBase::OutputFormat format, const std::string& filename)
{
    DataOut<dim> data_out;
    data_out.attach_dof_handler(dof_handler);
    // TODO: add_mg_data_vector()
    data_out.add_data_vector(solution, "psi");
    data_out.build_patches(dof_handler.get_fe().degree);

    std::ofstream output(filename);
    data_out.write(output, format);
}

template <int dim>
void output_hdf5(const Vector<double>& solution, const dealii::DoFHandler<dim>& dof_handler,
    const std::string& filename_h5)
{
    DataOut<dim> data_out;
    data_out.attach_dof_handler(dof_handler);
    data_out.add_data_vector(solution, "psi");
    data_out.build_patches(dof_handler.get_fe().degree);

    DataOutBase::DataOutFilterFlags flags(true, true);
    DataOutBase::DataOutFilter data_filter(flags);
    data_out.write_filtered_data(data_filter);
    data_out.write_hdf5_parallel(data_filter, filename_h5, MPI_COMM_WORLD);
}

template <int dim, typename Solver>
void package(Solver& GS, const double beta, const GdOptions& options_rgd)
{
    Square<dim> V;
    GS.setup();
    GS.assemble_matrix(V);

    // Run gradient descent
    auto x = GS.run(1.0, beta, options_rgd, N_CHECK_RES);

    // Plot solution
    if constexpr (gpe::is_solver_kind_v<Solver, gpe::plain_solver_tag>) {
        using DataOutBase::OutputFormat::vtu;
        output_results(x, GS.get_dofs(), vtu, fmt::format("solution_{}d.vtu", dim));
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

        with_dimension(options.dimension, [&](auto D)
        {
            constexpr int dim = decltype(D)::value;

            if (options_mg.multigrid && options_mg.parallel) {
                GPE_Solve_MG<dim, execution::par_t> GS(options,
                    options_mg.n_levels, options_mg.min_level, options_mg.max_level);
                package<dim>(GS, options.beta, options_rgd);
            }
            else if (options_mg.multigrid && !options_mg.parallel) {
                GPE_Solve_MG<dim, execution::seq_t> GS(options,
                    options_mg.n_levels, options_mg.min_level, options_mg.max_level);
                package<dim>(GS, options.beta, options_rgd);
            }
            else if (!options_mg.multigrid && options_mg.parallel) {
                GPE_Solve<dim, execution::par_t> GS(options, options_mg.n_levels);
                package<dim>(GS, options.beta, options_rgd);
            }
            else if (!options_mg.multigrid && !options_mg.parallel) {
                GPE_Solve<dim, execution::seq_t> GS(options, options_mg.n_levels);
                package<dim>(GS, options.beta, options_rgd);
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
