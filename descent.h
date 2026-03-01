#ifndef GPE_DESCENT_H
#define GPE_DESCENT_H

#include "manifold.h"
#include "lac.h"
#include "option_types.h"

#include <deal.II/base/convergence_table.h>

namespace gpe
{
using dealii::ConvergenceTable::RateMode::reduction_rate;
using dealii::ConvergenceTable::RateMode::reduction_rate_log2;

// TODO: use in gradient_descent()
enum class SolverStatus {
    CONVERGED,          // iterative method, diverged for given tolerance
    NOT_CONVERGED,      // iterative method, converged for given tolerance
    SOLUTION,           // non-iterative method
    ERROR               // solver error
};

// TODO: use in gradient_descent()
struct SolverInfo {
    SolverStatus status;
    size_t num_iter;
    Vector<double> solution;
    size_t elapsed_time;
};

// enum class SolverNorm {
//     L1,
//     L2,
//     LINF,
//     UNKNOWN
// };

// TODO: Armijo line search (cf. fcvx/descent.h)

//! Riemannian gradient descent for the GPE energy minimization
//! @param O Oracle for function value and (Riemannian) gradient
//! @param x0 Starting value
//! @param options Termination criteria
//! @param os Output stream for diagnostics
//! @return
template <typename Oracle, typename UpdateStrategy>
Vector<double>
gradient_descent(Oracle&& O, const Vector<double>& x0, UpdateStrategy&& update_iterate,
                 const GdOptions& options, std::ostream& os)
{
    Assert(options.step_size > 0, dealii::ExcInternalError("Step size must be positive"));
    Assert(options.max_iter  > 0, dealii::ExcInternalError("At least one iteration required"));

    Vector x(x0);
    dealii::ConvergenceTable convergence_table;

    bool break_on_next = false;
    unsigned int lac_iter = 0;  // number of iterations in inner solver taken
    // TODO: turn debug printing into logger/verbosity flag in options
    std::cerr << "Iteration: ";

    // Begin RGD iteration
    Vector<double> g(x.size());

    // Keep track of states locally
    iteration::State current_state;
    iteration::State previous_state;

    for (unsigned int iter = 0; iter < options.max_iter; iter++) {
        // Apply constraints and assemble iteration matrices
        update_iterate(x);
        current_state = O.residual(x);

        // TODO: check_every, ConvergenceTable == true -> check_every = 1
        std::cerr << iter << "..";

        // TODO: move to function.h
        convergence_table.add_value("iter", iter);
        convergence_table.add_value("lac_iter", lac_iter);
        convergence_table.add_value("mass", current_state.mass);
        convergence_table.add_value("lambda", current_state.lambda);
        convergence_table.add_value("residual", current_state.residual);
        convergence_table.add_value("energy", O.value(x));

        if (break_on_next) {
            break;
        }
        if (O.check_convergence(current_state, previous_state, options)) {
            // trick so that convergence_table is updated for last step
            // n iterations + starting solution -> n+1 table entries
            break_on_next = true;
            continue;
        }

        // Riemannian gradient: g <- x - A^{-1}x / (x' A^{-1}x)
        // TODO: generic return type (computation of gradient does not necessarily involve a linear system)
        lac_iter = O.gradient(x, g, options);
        // Retraction: x <- (x - h g) / ||x - h g||_M
        O.retract(g, x, -options.step_size);

        // Store current state for the next iteration's delta check
        previous_state = current_state;
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

#endif //GPE_DESCENT_H