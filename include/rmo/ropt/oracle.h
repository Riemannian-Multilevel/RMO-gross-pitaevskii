#ifndef RMO_ROPT_ORACLE_H
#define RMO_ROPT_ORACLE_H

#include <rmo/lac.h>
#include <rmo/option_types.h>

#include <rmo/ropt/metric.h>

#include <concepts>

/**
 * @file
 * @brief OracleBase: a smooth objective on a manifold for the Riemannian solvers, and the TiltOracle concept.
 */
namespace rmo
{

// Fields for gradient computation with inner solver
// TODO: fields for gradient evaluation of general problems
struct GradInfo
{
    double   residual;
    unsigned num_iter;
    double   tolerance;
    double   elapsed_time;
};

/**
 * @brief Objective evaluated at the point of the last update(): value, directional derivative, and Riemannian
 * gradient in the oracle's metric().
 */
class OracleBase
{
public:
    [[nodiscard]] virtual const char* id() const { return ""; }
    virtual ~OracleBase() = default;

    // TODO: this interface is generic and independent of the used discretization, yet it depends on deal.ii
    //       types, dealii::Vector in particular. Templatize or implement VectorBase, similar to MetricBase?
    //       -> VectorBase (points on the manifold), TangentVectorBase (points on the tangent space of a given base point)
    virtual void update(const Vector<double>& x) = 0;

    // TODO: leave `x` argument in update() exclusively, to avoid mismatches
    //       check marker `needs_assembly`
    [[nodiscard]] virtual double value(const Vector<double>&) const = 0;

    // TODO: Compute directional derivative and Riemannian gradient successively
    [[nodiscard]] virtual double directional_derivative(const Vector<double>&, const Vector<double>&) const = 0;

    // TODO: leave `x` argument in update() exclusively, to avoid mismatches
    //       check marker `needs_gradient
    virtual GradInfo gradient(const Vector<double>&, Vector<double>&) const = 0;  // Riemannian gradient - metric-dependent
    // Riemannian gradient, given the residual at x (debug builds check it, see assert_residual_at()):
    // oracles with inner solvers use it for the tolerance, others ignore it
    virtual GradInfo gradient(const Vector<double>&, Vector<double>&, double) const = 0;

    //! Metric of the Riemannian gradient
    [[nodiscard]] virtual const MetricBase& metric() const = 0;

    [[nodiscard]] virtual unsigned n_dofs() const = 0;
};


// Contract for the level ("tilt") oracle handed to FullApproximationScheme::cycle():
// an OracleBase evaluating a functional of type Functional in the metric used for the coarse correction term.
template <typename T, typename Functional>
concept TiltOracle = std::derived_from<T, OracleBase>
                  && std::constructible_from<T, Functional&, SolverOptions>;


/** @brief Oracle interface with the evaluation point fixed at construction (shared ownership). */
// TODO: the only difference with oracle is the missing update() method, and shared pointer (implementation detail?)
//       consolidate constructors and make update() a separate interface
class IterationBase
{
public:
    virtual ~IterationBase() = default;

    // Shared pointer to ensure lifetime of evaluation point, when IterationBase object relies on it
    IterationBase(std::shared_ptr<const Vector<double>> x_ptr)
        : x_ptr(std::move(x_ptr))
    {}

    [[nodiscard]] virtual double value()    const = 0;
    [[nodiscard]] virtual double residual() const = 0;
    [[nodiscard]] virtual double directional_derivative(const Vector<double> &z) const = 0;

    virtual GradInfo gradient(Vector<double>& dst) const = 0;

    [[nodiscard]] unsigned n_dofs() const { return x_ptr->size(); }

protected:
    std::shared_ptr<const Vector<double>> x_ptr;
};

} // namespace rmo

#endif //RMO_ROPT_ORACLE_H
