#ifndef RMO_ROPT_DESCENT_H
#define RMO_ROPT_DESCENT_H

#include <rmo/ropt/manifold.h>
#include <rmo/ropt/metric.h>
#include <rmo/lac.h>
#include <rmo/option_types.h>

#include <algorithm>
#include <cmath>
#include <deque>

/**
 * @file
 * @brief Armijo line search on a manifold, with the reference values of nonmonotone line searches and
 * Barzilai-Borwein step sizes.
 */

namespace rmo
{

/**
 * @brief Step size along @p eta from @p x: @p step, capped at options.boundary_fraction times
 * ManifoldBase::max_step() (fraction-to-the-boundary rule) where the retraction is not defined for all step sizes.
 * Unchanged on manifolds without such a limit.
 */
inline double capped_step(const ManifoldBase& manifold, const Vector<double>& x, const Vector<double>& eta,
                          const double step, const DescentOptions& options)
{
    const double h_max = manifold.max_step(x, eta);
    if (!std::isfinite(h_max)) {
        return step;
    }
    Assert(options.boundary_fraction > 0 && options.boundary_fraction < 1,
           dealii::ExcMessage("boundary_fraction must be in (0, 1)"));
    return std::min(step, options.boundary_fraction * h_max);
}

/**
 * @brief Armijo backtracking line search for @p oracle along the descent direction @p eta at @p x.
 *
 * Starting from options.ls.alpha, capped by capped_step() so that every trial point is defined, and reducing by
 * options.ls.beta, accepts the first step size \f$ \alpha \f$ with
 * \f$ f(R_x(\alpha \eta)) \le f_{\mathrm{ref}} + \sigma \alpha \langle \grad f(x), \eta \rangle_x \f$ (up to
 * rounding). On acceptance, @p x and the oracle are at the new point. @p fx is the reference value
 * \f$ f_{\mathrm{ref}} \f$: \f$ f(x) \f$ for a monotone search, at least \f$ f(x) \f$ for a nonmonotone one
 * (NonmonotoneReference); @p dir_deriv is \f$ \langle \grad f(x), \eta \rangle_x \f$.
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
    double alpha = capped_step(manifold, x, eta, options.ls.alpha, options);
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


/**
 * @brief Reference value \f$ f_{\mathrm{ref}} \f$ of a nonmonotone Armijo condition
 * \f$ f(R_x(\alpha\eta)) \le f_{\mathrm{ref}} + \sigma \alpha \langle \grad f(x), \eta \rangle_x \f$
 * (options.reference):
 * - LineSearchReference::MONOTONE: \f$ f(x_k) \f$;
 * - LineSearchReference::MAX: \f$ \max_{0 \le j < m} f(x_{k-j}) \f$ with \f$ m \f$ = options.memory (Grippo,
 *   Lampariello and Lucidi);
 * - LineSearchReference::AVERAGE: \f$ C_k \f$ with \f$ C_0 = f(x_0) \f$, \f$ Q_0 = 1 \f$,
 *   \f$ Q_{k+1} = \eta Q_k + 1 \f$, \f$ C_{k+1} = (\eta Q_k C_k + f(x_{k+1})) / Q_{k+1} \f$, \f$ \eta \f$ =
 *   options.eta (Zhang and Hager); \f$ \eta = 0 \f$ is monotone.
 *
 * In all cases \f$ f_{\mathrm{ref}} \ge f(x_k) \f$.
 */
class NonmonotoneReference
{
public:
    explicit NonmonotoneReference(const DescentOptions::LineSearchOptions& options)
        : options(options)
    {
        Assert(options.memory > 0, dealii::ExcMessage("memory must be positive"));
        Assert(options.eta >= 0 && options.eta <= 1, dealii::ExcMessage("eta must be in [0, 1]"));
    }

    //! Appends \f$ f(x_k) \f$ of the next iterate
    void add(const double f)
    {
        values.push_back(f);
        if (values.size() > options.memory) {
            values.pop_front();
        }
        if (Q == 0) {
            C = f;
            Q = 1;
        } else {
            const double Q_new = options.eta * Q + 1;
            C = (options.eta * Q * C + f) / Q_new;
            Q = Q_new;
        }
    }

    [[nodiscard]] double value() const
    {
        Assert(!values.empty(), dealii::ExcMessage("no value added"));
        switch (options.reference) {
            case LineSearchReference::MAX:     return *std::ranges::max_element(values);
            case LineSearchReference::AVERAGE: return std::max(C, values.back());  // C >= f in exact arithmetic
            case LineSearchReference::MONOTONE: break;
        }
        return values.back();
    }

private:
    DescentOptions::LineSearchOptions options;
    std::deque<double> values;
    double C = 0, Q = 0;
};


/**
 * @brief Barzilai-Borwein step sizes from the differences \f$ s = x_k - x_{k-1} \f$ and
 * \f$ y = g_k - g_{k-1} \f$ of the iterates and Riemannian gradients, both taken in the ambient space (no vector
 * transport, as in Wen and Yin), with the inner product of the metric at \f$ x_k \f$:
 * \f$ \alpha^{BB1} = \langle s, s \rangle / \langle s, y \rangle \f$, \f$ \alpha^{BB2} = \langle s, y \rangle /
 * \langle y, y \rangle \f$, or both in turn (options.initial), clamped to [options.bb_min, options.bb_max].
 *
 * Returns @p fallback in the first iteration and if \f$ \langle s, y \rangle \le 0 \f$.
 */
class BarzilaiBorwein
{
public:
    explicit BarzilaiBorwein(const DescentOptions::LineSearchOptions& options)
        : options(options)
    {}

    //! Step size at @p x with gradient @p grad; @p metric is the metric at @p x
    double step(const Vector<double>& x, const Vector<double>& grad, const MetricBase& metric, const double fallback)
    {
        double alpha = fallback;
        if (k > 0) {
            Vector<double> s(x), y(grad);
            s -= x_prev;
            y -= grad_prev;
            const double sy = metric.inner(s, y);
            if (sy > 0) {
                const bool bb1 = options.initial == InitialStep::BB1
                              || (options.initial == InitialStep::ALTERNATING && k % 2 == 1);
                alpha = bb1 ? metric.inner(s, s) / sy : sy / metric.inner(y, y);
                alpha = std::clamp(alpha, options.bb_min, options.bb_max);
            }
        }
        x_prev = x;
        grad_prev = grad;
        ++k;
        return alpha;
    }

private:
    DescentOptions::LineSearchOptions options;
    Vector<double> x_prev, grad_prev;
    unsigned k = 0;
};


} // namespace rmo

#endif //RMO_ROPT_DESCENT_H
