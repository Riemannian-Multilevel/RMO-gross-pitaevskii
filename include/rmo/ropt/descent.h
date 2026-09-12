#ifndef RMO_ROPT_DESCENT_H
#define RMO_ROPT_DESCENT_H

#include <rmo/ropt/oracle_base.h>
#include <rmo/ropt/manifold.h>
#include <rmo/lac.h>
#include <rmo/option_types.h>

#include <deal.II/base/convergence_table.h>
#include <deal.II/base/timer.h>

namespace rmo
{
using dealii::ConvergenceTable::RateMode::reduction_rate;
using dealii::ConvergenceTable::RateMode::reduction_rate_log2;

// TODO: use in GradientDescent::cycle()
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
        oracle.update(x);   // step discarded, restore original state

        if (alpha < options.ls.min) {
            // Backtracking has shrunk alpha below the configured floor without ever
            // satisfying the sufficient-decrease condition -- there is no step to
            // report. Previously this branch accepted x_trial anyway and returned
            // options.ls.min as if a (nonexistent) floor step had succeeded, which
            // made "Step rejected by line search" (solver.h) a lie: x had in fact
            // been overwritten with a step that failed Armijo. Report failure the
            // same way as hitting max_iter below, so callers can rely on a 0 return
            // meaning x is unchanged.
            return 0.0;
        }

        // Backtrack
        alpha *= options.ls.beta;
    }
    std::cerr << "Warning: Armijo line search hit max iterations ("
              << options.ls.max_iter << ")." << std::endl;
    // x was never assigned x_trial above, so it (and the oracle's cached
    // state, restored after each rejected trial) are unchanged -- signal
    // that no step was taken, per this function's documented contract,
    // instead of returning options.ls.min as if a (nonexistent) floor step
    // had succeeded. Silently doing the latter previously caused callers to
    // treat a stalled search as progress and retry it forever on an
    // identical, still-non-descent direction (see cycle_smooth()/GradientDescent::cycle()).
    return 0.0;
}


} // namespace rmo

#endif //RMO_ROPT_DESCENT_H