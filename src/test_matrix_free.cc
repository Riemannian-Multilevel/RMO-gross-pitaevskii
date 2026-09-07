//
// Created by Ferdinand Vanmaele on 09.04.26.
//
#include <rmo/gpe/gpe.h>
#include <rmo/gpe/oracle.h>
#include <rmo/option.h>
#include <rmo/option_types.h>
#include <rmo/util/util.h>
#include <rmo/gpe/model.h>

#include <rmo/ropt/manifold.h>
#include <rmo/util/random.h>
#include <fmt/format.h>

#include <iostream>

constexpr double MEAN   = 0.0;
constexpr double STDDEV = 1.0;
constexpr double MARGIN = 1e-10;

using namespace rmo;
using namespace rmo::gpe;
using namespace dealii;

template <typename Range>
double mean(Range&& x)
{
    const unsigned n = x.size();
    Assert(n > 0, dealii::ExcInternalError());
    long double accum = 0.0;
    for (auto i : x)
        accum += i;
    return accum / static_cast<long double>(n);
}

// Matrix-free evaluation of the Gross Pitaevskii energy. This function does not require calling
// assemble_nonlinear_term() beforehand.
// Conceptually this is similar to rmo::fe::assemble, but iterates over the finite element grid
// instead of assembling a potentially large sparse matrix.
template <int dim>
double get_energy_nonlinear(const DoFHandler<dim>& dof_handler, const Vector<double>& x)
{
    // Non-linear term: (beta / 2) * \int |x|^4 dx
    double energy_nonlinear = 0.0;
    // Use a quadrature formula appropriate for the element degree
    dealii::QGauss<dim> quadrature_formula(dof_handler.get_fe().degree + 1);
    // We only need to know the function values and the quadrature weights (JxW)
    dealii::FEValues<dim> fe_values(dof_handler.get_fe(), quadrature_formula,
                                    dealii::update_values | dealii::update_JxW_values);

    const unsigned int n_q_points = quadrature_formula.size();
    std::vector<double> x_values(n_q_points);

    // Loop over all active cells
    // TODO: dof_handler.mg_cell_iterators_on_level(level)
    for (const auto& cell : dof_handler.active_cell_iterators()) {
        if (cell->is_locally_owned()) {
            fe_values.reinit(cell);
            fe_values.get_function_values(x, x_values);

            // Integrate |x|^4 over the cell
            for (unsigned int q = 0; q < n_q_points; ++q) {
                const double val = x_values[q];
                const double val_sq = val * val;

                energy_nonlinear += (val_sq * val_sq) * fe_values.JxW(q);
            }
        }
    }
    return energy_nonlinear;
}

// TODO: can larger benefits be obtained by also assembling A0?
template <int dim, typename MatrixType>
double get_energy(const DoFHandler<dim>& dof_handler, const Vector<double>& x,
                  const MatrixType& A0, const double beta)
{
    // Linear term: 0.5 * x^T * A0 * x
    // A0 is fixed between operations
    Vector<double> A0_x(x.size());
    A0.vmult(A0_x, x);

    double energy = 0.0;
    energy += x * A0_x;  // linear part
    energy += 0.5*beta*get_energy_nonlinear(dof_handler, x);  // non-linear part

    return 0.5*energy;
}


int main(int argc, char* argv[])
{
    GPE_Options options{};
    unsigned min_level = 0, max_level = 0, n_trials = 0;

    try {
        po::options_description all("Cell-loop vs. sparse matrix-vector evaluation of the GP energy");
        all.add(gpe_cli_options());
        all.add_options()
            ("help", "print this message")
            ("min-level", po::value<unsigned>()->default_value(8), "coarsest refinement level")
            ("max-level", po::value<unsigned>()->default_value(11), "finest refinement level")
            ("trials", po::value<unsigned>()->default_value(200), "random points evaluated per level");

        po::variables_map vm;
        po::store(po::parse_command_line(argc, argv, all), vm);
        po::notify(vm);

        if (vm.count("help")) {
            std::cout << all << std::endl;
            return 0;
        }

        apply_gpe_options(vm, options);
        min_level = vm["min-level"].as<unsigned>();
        max_level = vm["max-level"].as<unsigned>();
        n_trials  = vm["trials"].as<unsigned>();
        AssertThrow(min_level <= max_level,
            dealii::ExcMessage("--min-level must not exceed --max-level"));
    }
    catch (std::exception& e) {
        std::cerr << "error: " << e.what() << "\n";
        return 1;
    }

    try {
        with_dimension(options.dimension, [&]<typename T0>(T0)
        {
        constexpr int dim = T0::value;

        MGLevelObject<std::vector<double>> time_value(min_level, max_level);
        MGLevelObject<std::vector<double>> time_value_cell_loop(min_level, max_level);
        MGLevelObject<std::vector<double>> value_error(min_level, max_level);

        for (unsigned level = min_level; level <= max_level; level++) {
            dealii::Timer timer;
            ModelBuilder<dim> model(potential::Square<dim>(), options, level);

            auto& system = model.get_system();
            const auto& eval = model.get_eval(options.beta, SolverOptions{});
            const unsigned n_dofs = model.n_dofs();

            // Average time over trials for value() + assembly, and the cell-loop reference
            time_value[level].reserve(n_trials);
            time_value_cell_loop[level].reserve(n_trials);
            value_error[level].reserve(n_trials);

            for (unsigned int trial = 0; trial < n_trials; trial++) {
                double value, value_cell_loop;
                Vector<double> x(n_dofs);
                ellipsoid::random_point(x, model.get_M(), MEAN, STDDEV);

                { // Reference value, assembled on the fly in a cell loop
                    auto begin_t = timer.cpu_time();
                    value_cell_loop = get_energy(model.get_dofs(), x, system.get_A0(), options.beta);

                    auto end_t = timer.cpu_time();
                    time_value_cell_loop[level].push_back(end_t - begin_t);
                }

                { // Value through sparse matrix-vector products (LinearCombination)
                    auto begin_t = timer.cpu_time();
                    system.assemble_nonlinear_term(x);
                    value = eval.value(x);

                    auto end_t = timer.cpu_time();
                    time_value[level].push_back(end_t - begin_t);
                }

                // Verify both match within a given margin (done in extended precision to reduce cancellation)
                // TODO: mean/standard deviation of errors
                const long double error = std::abs(static_cast<long double>(value) - value_cell_loop);
                value_error[level].push_back(static_cast<double>(error));

                AssertThrow(error < MARGIN, dealii::ExcInternalError(fmt::format(
                "mismatch between value: {} and value_cell_loop: {} (level: {}, trial: {})",
                    value, value_cell_loop, level, trial)));
            }
            // TODO: write time_value / time_value_cell_loop to file for plotting
            std::cerr << fmt::format("Average time on level {}, spmv: {}s\n", level, mean(time_value[level]))
                      << fmt::format("Average time on level {}, cell loop: {}s\n", level, mean(time_value_cell_loop[level]))
                      << fmt::format("Average error on level {}: {}\n", level, mean(value_error[level]));
        }
        });
    }
    catch (std::exception& e) {
        std::cerr << "error: " << e.what() << "\n";
        return 1;
    }
    return 0;
}
