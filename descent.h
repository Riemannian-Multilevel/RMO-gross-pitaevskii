#ifndef GPE_ENERGY_H
#define GPE_ENERGY_H

#include "lac.h"
#include "option_types.h"

#include <cmath>
#include <deal.II/base/convergence_table.h>
#include <deal.II/lac/sparse_ilu.h>

namespace gpe
{
using dealii::ConvergenceTable::RateMode::reduction_rate;
using dealii::ConvergenceTable::RateMode::reduction_rate_log2;

// enum class SolverStatus {
//     CONVERGED,          // iterative method, diverged for given tolerance
//     NOT_CONVERGED,      // iterative method, converged for given tolerance
//     SOLUTION,           // non-iterative method
//     ERROR               // solver error
// };
//
// struct SolverInfo {
//     SolverStatus status;
//     size_t num_iter;
//     Vector solution;
//     size_t elapsed_time;
// };
//
// enum class SolverNorm {
//     L1,
//     L2,
//     LINF,
//     UNKNOWN
// };

template <typename MatrixType, typename VectorType>
struct EmptyUpdateStrategy {
    void operator()(MatrixType& M, const VectorType& v) const
    {}
};


// TODO: Armijo line search (cf. fcvx/descent.h)

//! Riemannian gradient descent for the GPE energy minimization
//! @param O Oracle for function value and (Riemannian) gradient
//! @param x0 Starting value
//! @param options Termination criteria
//! @param os Output stream for diagnostics
//! @return
template <typename Oracle, typename PostSmoothing = EmptyUpdateStrategy<SparseMatrix<double>, Vector<double>>>
Vector<double>
gradient_descent(Oracle&& O, const Vector<double>& x0,
                 const GdOptions& options, std::ostream& os)
{
    Assert(options.step_size > 0, dealii::ExcInternalError("Step size must be positive"));
    Assert(options.max_iter  > 0, dealii::ExcInternalError("At least one iteration required"));

    Vector x(x0);
    dealii::ConvergenceTable convergence_table;

    bool break_on_next = false;
    unsigned int iter;
    unsigned int lac_iter = 0;  // number of iterations in inner solver taken
    // TODO: turn debug printing into logger/verbosity flag in options
    std::cerr << "Iteration: ";

    // Begin RGD iteration
    Vector<double> g(x.size());

    for (iter = 0; iter < options.max_iter; iter++) {
        // Apply constraints and assemble iteration matrices
        O.initialize(x);

        // Compute termination criteria
        auto ctrl = O.residual();
        // TODO: check_every, ConvergenceTable == true -> check_every = 1
        std::cerr << iter << "..";

        // TODO: generalize to oracle method (different objectives have different properties)
        convergence_table.add_value("iter", iter);
        convergence_table.add_value("lac_iter", lac_iter);
        convergence_table.add_value("mass", ctrl.mass);
        convergence_table.add_value("lambda", ctrl.lambda);
        convergence_table.add_value("residual", ctrl.residual);
        convergence_table.add_value("energy", O.value(x));

        // --- riemannian gradient descent
        if (break_on_next) {
            break;
        }
        if (O.is_optimal(options)) {
            // trick so that convergence_table is updated for last step
            // n iterations + starting solution -> n+1 table entries
            break_on_next = true;
            continue;
        }
        // --- pre-smoothing (multigrid)
        // TODO

        // Riemannian gradient: g <- x - A^{-1}x / (x' A^{-1}x)
        // TODO: generic return type (computation of gradient does not necessarily involve a linear system)
        lac_iter = O.gradient(x, g, options);
        // Retraction: x <- (x - h g) / ||x - h g||_M
        O.retract(g, x, -options.step_size);

        // --- post-smoothing (multigrid)
        // TODO
    }
    std::cerr << std::endl << std::endl;
    convergence_table.set_precision("mass", 4);
    convergence_table.set_precision("lambda", 4);
    convergence_table.set_precision("residual", 4);
    convergence_table.set_precision("energy", 4);

    convergence_table.set_scientific("lambda", true);
    convergence_table.set_scientific("residual", true);
    convergence_table.set_scientific("energy", true);

    convergence_table.evaluate_convergence_rates("residual", reduction_rate);
    convergence_table.evaluate_convergence_rates("residual", reduction_rate_log2);
    convergence_table.write_text(os, dealii::TableHandler::TextOutputFormat::table_with_headers);

    //constraints.distribute(x);
    return x;
}

} // namespace gpe
#endif //GPE_ENERGY_H