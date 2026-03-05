#ifndef GPE_DESCENT_H
#define GPE_DESCENT_H

#include "option_types.h"
#include "manifold.h"
#include "lac.h"
#include "grid_operators.h"

#include <deal.II/base/convergence_table.h>

#include "gpe.h"

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

// TODO: use in Oracle::run() (package of value(), update() and gradient())
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
/**
 * @brief Performs the Armijo backtracking line search on the manifold.
 * @param oracle The manifold and objective function interface.
 * @param x Current base point.
 * @param eta Search direction (must be a descent direction).
 * @param fx Function value at current point f(x).
 * @param dir_deriv Directional derivative <grad f(x), eta>_x.
 * @param options Line search parameters.
 * @param x_new [out] The accepted new point.
 * @return The accepted step size alpha (returns 0 if failed to converge).
 */
template <typename VectorType, typename OracleType>
double armijo_line_search(const OracleType& oracle,
                          const VectorType& x,
                          const VectorType& eta,
                          const double fx,
                          const double dir_deriv,
                          const DescentOptions& options,
                          VectorType& x_new)
{
    double alpha = options.ls_alpha;

    for (unsigned int ls_iter = 0; ls_iter < options.max_search; ++ls_iter) {
        // Compute tentative step alpha*eta_x and retract
        VectorType step(eta);
        step *= alpha;

        oracle.retract(x, step, x_new);

        // Evaluate function at the new point
        // TODO: this requires a new assembly of A_x - use matrix-free evaluation
        oracle.update(x_new);
        double fx_new = oracle.value(x_new);

        // Armijo condition:
        //   f(Ret_x(alpha * eta)) <= f(x) + sigma * alpha * <grad, eta>_x
        if (fx_new - fx <= options.ls_sigma * alpha * dir_deriv) {
            return alpha; // step accepted
        }
        if (alpha < 1e-8) {
            return  1e-8;
        }
        oracle.update(x);  // step discarded, restore reference point

        // Backtrack
        alpha *= options.ls_beta;
    }
    std::cerr << "Warning: Armijo line search hit max iterations ("
              << options.max_search << ")." << std::endl;
    return 0.0; // Line search failed
}

//! Riemannian gradient descent for the GPE energy minimization
//! @param O Oracle for function value and (Riemannian) gradient
//! @param x0 Starting value
//! @param options Termination criteria
//! @param os Output stream for diagnostics
//! @return
template <typename Oracle>
Vector<double>
gradient_descent(Oracle&& O, const Vector<double>& x0, DescentOptions options, std::ostream& os)
{
    Assert(options.step_size > 0, dealii::ExcInternalError("Step size must be positive"));
    Assert(options.max_iter  > 0, dealii::ExcInternalError("At least one iteration required"));

    Vector x(x0);
    dealii::ConvergenceTable convergence_table;
    O.update(x);

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
        // Assemble iteration matrices for new iterate
        current_state = O.residual(x);
        double Ex = O.value(x);
        current_state.energy = Ex;

        // TODO: check_every, ConvergenceTable == true -> check_every = 1
        std::cerr << iter << "..";

        // TODO: move to function.h
        convergence_table.add_value("iter", iter);
        convergence_table.add_value("lac_iter", lac_iter);
        convergence_table.add_value("mass", current_state.mass);
        convergence_table.add_value("lambda", current_state.lambda);
        convergence_table.add_value("residual", current_state.residual);
        convergence_table.add_value("energy", current_state.energy);

        // if (break_on_next) {
        //     break;
        // }
        // if (O.check_convergence(current_state, previous_state, options)) {
        //     // trick so that convergence_table is updated for last step
        //     // n iterations + starting solution -> n+1 table entries
        //     break_on_next = true;
        //     continue;
        // }

        // Riemannian gradient: g <- x - A^{-1}x / (x' A^{-1}x)
        // TODO: generic return type (computation of gradient does not necessarily involve a linear system)
        //options.tol_inner = std::min(options.tol_inner, 0.1 * current_state.residual);
        lac_iter = O.gradient(x, g, options);
        // Retraction: x <- (x - h g) / ||x - h g||_M
        if (options.line_search) {
            // TODO: support other descent directions (i.e. coarse descent)
            Vector eta(g);  // search direction
            eta *= -1.0;
            double dd = O.metric(g, eta); // <grad f(x), -grad f(x)>_x
            Vector x_new(x);
            // runs O.retract, O.update
            double h = armijo_line_search(O, x, eta, Ex, dd, options, x_new);
            if (h > 0) x = x_new;
            if (h == 0) throw std::runtime_error("line search failed");  // TODO: alternative: non-monotone line search
            convergence_table.add_value("step",h);
        }
        else {
            O.retract(g, x, -options.step_size);
            O.update(x);
            convergence_table.add_value("step",options.step_size);
        }

        // Store current state for the next iteration's delta check
        previous_state = current_state;
    }

    std::cerr << std::endl << std::endl;
    convergence_table.set_precision("mass", 4);
    convergence_table.set_precision("lambda", 4);
    convergence_table.set_precision("residual", 4);
    convergence_table.set_precision("energy", 4);
    convergence_table.set_precision("step", 4);

    convergence_table.set_scientific("lambda", true);
    convergence_table.set_scientific("residual", true);
    convergence_table.set_scientific("energy", true);
    convergence_table.set_scientific("step", true);

    convergence_table.evaluate_convergence_rates("residual", reduction_rate);
    convergence_table.evaluate_convergence_rates("residual", reduction_rate_log2);
    convergence_table.write_text(os, dealii::TableHandler::TextOutputFormat::table_with_headers);

    //constraints.distribute(x);
    return x;
}

} // namespace gpe

#endif //GPE_DESCENT_H