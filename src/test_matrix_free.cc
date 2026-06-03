//
// Created by Ferdinand Vanmaele on 09.04.26.
//
#include <gpe/problem/gpe.h>
#include <gpe/problem/oracle.h>
#include <gpe/option_types.h>
#include <gpe/main/model.h>

#include "test_gradient.h"
#include <fmt/format.h>

#define NUM_TRIALS 200
#define MIN_LEVEL  8
#define MAX_LEVEL  11
#define DIM        2
#define FE_DEGREE  1
#define RADIUS     10
#define BETA       100
#define MEAN       0.0
#define STDDEV     1.0
#define MARGIN     1e-10

using namespace gpe;
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
// Conceptually this is similar to gpe::assemble, but iterates over the finite element grid
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


int main()
{
    GPE_Options options = {.dimension=DIM, .degree=FE_DEGREE, .radius=RADIUS, .beta=BETA};
    MGLevelObject<std::vector<double>> time_value(MIN_LEVEL, MAX_LEVEL);
    MGLevelObject<std::vector<double>> time_value_matrix_free(MIN_LEVEL, MAX_LEVEL);
    MGLevelObject<std::vector<double>> value_error(MIN_LEVEL, MAX_LEVEL);

    for (unsigned level = MIN_LEVEL; level <= MAX_LEVEL; level++) {
        dealii::Timer timer;
        ModelBuilder<DIM> model(Square<DIM>(), options, level);

        auto& system = model.get_system();
        const auto& eval = model.get_eval(options.beta, SolverOptions{});
        const unsigned n_dofs = model.n_dofs();

        // Average time over trials for value() + assembly, and value_matrix_free()
        time_value[level].reserve(NUM_TRIALS);
        time_value_matrix_free[level].reserve(NUM_TRIALS);
        value_error[level].reserve(NUM_TRIALS);

        for (unsigned int trial = 0; trial < NUM_TRIALS; trial++) {
            double value, value_matrix_free;
            Vector<double> x(n_dofs);
            ellipsoid::random_point(x, model.get_M(), MEAN, STDDEV);

            { // Value through sparse matrix-vector products
                auto begin_t = timer.cpu_time();
                value_matrix_free = get_energy(model.get_dofs(), x, system.get_A0(), options.beta);

                auto end_t = timer.cpu_time();
                time_value_matrix_free[level].push_back(end_t - begin_t);
            }

            { // Value through matrix-free implementation
                auto begin_t = timer.cpu_time();
                system.assemble_nonlinear_term(x);
                value = eval.value(x);

                auto end_t = timer.cpu_time();
                time_value[level].push_back(end_t - begin_t);
            }

            // Verify both match within a given margin (done in extended precision to reduce cancellation)
            // TODO: mean/standard deviation of errors
            const long double error = std::abs(static_cast<long double>(value) - value_matrix_free);
            value_error[level].push_back(static_cast<double>(error));

            AssertThrow(error < MARGIN, dealii::ExcInternalError(fmt::format(
            "mismatch between value: {} and value_matrix_free: {} (level: {}, trial: {})",
                value, value_matrix_free, level, trial)));
        }
        // TODO: write time_value / time_value_matrix_free to file for plotting
        std::cerr << fmt::format("Average time on level {}, spmv: {}s\n", level, mean(time_value[level]))
                  << fmt::format("Average time on level {}, matrix-free: {}s\n", level, mean(time_value_matrix_free[level]))
                  << fmt::format("Average error on level {}: {}\n", level, mean(value_error[level]));
    }
}
