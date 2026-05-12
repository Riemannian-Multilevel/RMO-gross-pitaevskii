#ifndef GPE_DESCENT_H
#define GPE_DESCENT_H

#include <gpe/ropt/manifold.h>
#include <gpe/lac.h>
#include <gpe/option_types.h>

#include <deal.II/base/convergence_table.h>
#include <deal.II/base/timer.h>

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

/**
 * @brief Performs the Armijo backtracking line search on the manifold.
 * @param oracle The manifold and objective function interface.
 * @param manifold
 * @param x [in-out] Current base point.
 * @param eta Search direction (must be a descent direction).
 * @param fx Function value at current point f(x).
 * @param dir_deriv Directional derivative <grad f(x), eta>_x.
 * @param options Line search parameters.
 * @return The accepted step size alpha (returns 0 if failed to converge).
 */
// TODO: vector x is updated in place, even for tentative steps, since no copy
//       of the problem state (large sparse matrix term Mpp) is made
template <typename VectorType, typename OracleType>
double armijo_line_search(OracleType& oracle,
                          const ManifoldBase& manifold,
                          VectorType& x,
                          const VectorType& eta,
                          const double fx,
                          const double dir_deriv,
                          const DescentOptions& options)
{
    if (dir_deriv >= 0) {
        std::cerr << "warning: not a descent direction (" << dir_deriv
                  << std::setprecision(12) << ")" << std::endl;
    }
    double alpha = options.ls.alpha;
    Vector<double> x_trial(x);

    // Avoid numerical issues when close to the solution
    const double eps = std::numeric_limits<double>::epsilon();
    const double noise_tol = 10.0 * eps * std::max(1.0, std::abs(fx));

    for (unsigned int ls_iter = 0; ls_iter < options.ls.max_iter; ++ls_iter) {
        // Compute tentative step alpha*eta_x and retract
        VectorType step(eta);
        step *= alpha;
        manifold.retract(step, x, x_trial);

        // Evaluate function at the new point
        // TODO: this requires a new assembly of A_x - use matrix-free evaluation
        oracle.update(x_trial);
        double fx_new = oracle.value(x_trial);

        // Armijo condition:
        //   f(Ret_x(alpha * eta)) <= f(x) + sigma * alpha * <grad, eta>_x
        if (fx_new <= fx + options.ls.sigma * alpha * dir_deriv + noise_tol) {
            x = x_trial;    // step accepted, write x
            return alpha;
        }
        if (alpha < options.ls.min) {
            x = x_trial;    // step accepted, write x
            return options.ls.min;
        }
        oracle.update(x);   // step discarded, restore original state

        // Backtrack
        alpha *= options.ls.beta;
    }
    std::cerr << "Warning: Armijo line search hit max iterations ("
              << options.ls.max_iter << ")." << std::endl;
    return options.ls.min;  // TODO: throw exception that can be caught
}

struct EmptyCheck
{
    bool operator()(iteration::State, iteration::State)
    {
        return false;
    }
};

//! Riemannian gradient descent for the GPE energy minimization
//! @param oracle Oracle for function value and (Riemannian) gradient.
//! @param manifold
//! @param x0 Starting value.
//! @param options Parameters for gradient descent, such as step-size.
//! @param os Output stream for diagnostics.
//! @param check_convergence Strategy for verifying if gradient descent converged.
//! @return
// TODO: split into cycle() and eval(), compare FullApproximationScheme
template <typename OracleType, typename CheckType = EmptyCheck>
Vector<double>
gradient_descent(OracleType& oracle,
                 const ManifoldBase& manifold,
                 const Vector<double>& x0,
                 DescentOptions options, std::ostream& os,
                 CheckType check_convergence = {})
{
    Assert(options.step_size > 0, dealii::ExcInternalError("Step size must be positive"));
    Assert(options.max_iter  > 0, dealii::ExcInternalError("At least one iteration required"));

    // Keep track of states locally
    iteration::State current_state;
    iteration::State previous_state;

    // Define the timer
    dealii::Timer timer;
    timer.reset();

    Vector x(x0);
    dealii::ConvergenceTable convergence_table;
    oracle.update(x);

    current_state = oracle.residual(x);
    double Ex = oracle.value(x);
    current_state.energy = Ex;
    unsigned int lac_iter = 0;  // number of iterations in inner solver taken

    // TODO: move to function.h
    convergence_table.add_value("iter", 0);
    convergence_table.add_value("lac_iter", lac_iter);
    convergence_table.add_value("mass", current_state.mass);
    convergence_table.add_value("lambda", current_state.lambda);
    convergence_table.add_value("residual", current_state.residual);
    convergence_table.add_value("energy", current_state.energy);
    convergence_table.add_value("step",0);
    convergence_table.add_value("elapsed", 0);  // does not include setup time

    // TODO: turn debug printing into logger/verbosity flag in options
    std::cerr << "Iteration: ";

    // Begin RGD iteration
    Vector<double> g(x.size());

    for (unsigned int iter = 1; iter <= options.max_iter; iter++) {
        // TODO: check_every, ConvergenceTable == true -> check_every = 1
        std::cerr << iter << "..";

        if (check_convergence(current_state, previous_state)) {
            // trick so that convergence_table is updated for last step
            // n iterations + starting solution -> n+1 table entries
            break;
        }

        // ---- Timed section
        timer.start();
        // Riemannian gradient: g <- x - A^{-1}x / (x' A^{-1}x)
        // TODO: generic return type (computation of gradient does not necessarily involve a linear system)
        lac_iter = oracle.gradient(x, g);
        double step_size = options.step_size;
        // Retraction: x <- (x - h g) / ||x - h g||_M

        if (options.line_search) {
            // TODO: support other descent directions (i.e. coarse descent)
            Vector eta(g);  // search direction
            eta *= -1.0;
            double dd = oracle.directional_derivative(x, eta);  // <grad f(x), eta>_x = Df(x)[eta]
            // runs O.retract(), O.update()
            double h = armijo_line_search(oracle, manifold, x, eta, Ex, dd, options);
            //if (h > 0) x = x_new;
            if (h == 0) throw std::runtime_error("line search failed");  // TODO: alternative: non-monotone line search
            step_size = h;
        }
        else {
            manifold.retract(g, x, -options.step_size);
            oracle.update(x);
        }
        // ---- End timed section
        timer.stop();

        current_state = oracle.residual(x);
        Ex = oracle.value(x);
        current_state.energy = Ex;
        convergence_table.add_value("iter", iter);
        convergence_table.add_value("lac_iter", lac_iter);
        convergence_table.add_value("mass", current_state.mass);
        convergence_table.add_value("lambda", current_state.lambda);
        convergence_table.add_value("residual", current_state.residual);
        convergence_table.add_value("energy", current_state.energy);
        convergence_table.add_value("step",step_size);
        convergence_table.add_value("elapsed",timer.cpu_time());

        // Store current state for the next iteration's delta check
        previous_state = current_state;
    }

    std::cerr << std::endl << std::endl;
    convergence_table.set_precision("mass", 2);
    convergence_table.set_precision("lambda", 2);
    convergence_table.set_precision("residual", 4);
    convergence_table.set_precision("energy", 16);
    convergence_table.set_precision("step", 2);
    convergence_table.set_precision("elapsed", 2);

    //convergence_table.set_scientific("mass",true);
    convergence_table.set_scientific("lambda", true);
    convergence_table.set_scientific("residual", true);
    convergence_table.set_scientific("energy", true);
    convergence_table.set_scientific("step", true);
    convergence_table.set_scientific("elapsed", true);

    convergence_table.evaluate_convergence_rates("residual", reduction_rate);
    convergence_table.evaluate_convergence_rates("residual", reduction_rate_log2);
    convergence_table.write_text(os, dealii::TableHandler::TextOutputFormat::org_mode_table);

    //constraints.distribute(x);
    return x;
}

} // namespace gpe

#endif //GPE_DESCENT_H