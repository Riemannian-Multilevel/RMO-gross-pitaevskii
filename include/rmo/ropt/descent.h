#ifndef RMO_ROPT_DESCENT_H
#define RMO_ROPT_DESCENT_H

#include <rmo/ropt/oracle.h>
#include <rmo/ropt/manifold.h>
#include <rmo/lac.h>
#include <rmo/option_types.h>

#include <deal.II/base/convergence_table.h>
#include <deal.II/base/timer.h>

/**
 * @file
 * @brief Armijo line search on a manifold, and solver status types.
 */
namespace rmo
{
using dealii::ConvergenceTable::RateMode::reduction_rate;
using dealii::ConvergenceTable::RateMode::reduction_rate_log2;

// TODO: use in GradientDescent::cycle()
enum class SolverStatus {
    CONVERGED,          // iterative method, converged for given tolerance
    NOT_CONVERGED,      // iterative method, not converged for given tolerance
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
 * @brief Armijo backtracking line search for @p oracle along the descent direction @p eta at @p x.
 *
 * Starting from options.ls.alpha and reducing by options.ls.beta, accepts the first step size \f$ \alpha \f$ with
 * \f$ f(R_x(\alpha \eta)) \le f(x) + \sigma \alpha \langle \grad f(x), \eta \rangle_x \f$ (up to rounding). On
 * acceptance, @p x and the oracle are at the new point. @p fx is \f$ f(x) \f$ and @p dir_deriv is
 * \f$ \langle \grad f(x), \eta \rangle_x \f$.
 *
 * @return The accepted step size, or 0 if no step was accepted before the step size fell below options.ls.min or
 * within options.ls.max_iter trials; then @p x and the oracle are unchanged.
 */
// TODO: the oracle state (e.g. M_phiphi) is reassembled for every trial point and restored after a rejection,
//       since it is not copied
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

        // No sufficient decrease down to the smallest step size: reject, as for max_iter below
        if (alpha < options.ls.min) {
            return 0.0;
        }

        // Backtrack
        alpha *= options.ls.beta;
    }
    std::cerr << "Warning: Armijo line search hit max iterations ("
              << options.ls.max_iter << ")." << std::endl;
    // No step accepted: x and the oracle state are unchanged. Returning 0 (not options.ls.min) lets callers
    // detect the stalled search, see cycle_smooth() and GradientDescent::cycle().
    return 0.0;
}


} // namespace rmo

#endif //RMO_ROPT_DESCENT_H