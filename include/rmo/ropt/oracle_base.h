#ifndef RMO_ROPT_ORACLE_BASE_H
#define RMO_ROPT_ORACLE_BASE_H

#include <rmo/lac.h>
#include <rmo/option_types.h>

#include <concepts>

namespace rmo
{

// Basic oracle interface
class OracleBase
{
public:
    virtual const char* id() const { return ""; }
    virtual ~OracleBase() = default;

    virtual void update(const Vector<double>& x) = 0;

    // TODO: leave `x` argument in update() exclusively, to avoid mismatches
    //       check marker `needs_assembly`
    virtual double value(const Vector<double>&) const = 0;

    // TODO: Compute directional derivative and Riemannian gradient successively
    virtual double directional_derivative(const Vector<double>&, const Vector<double>&) const = 0;

    // TODO: leave `x` argument in update() exclusively, to avoid mismatches
    //       check marker `needs_gradient
    virtual GradInfo gradient(const Vector<double>&, Vector<double>&) const = 0;  // Riemannian gradient - metric-dependent
    virtual GradInfo gradient(const Vector<double>&, Vector<double>&, double) const = 0;  // method for setting tolerance

    // TODO: move this to a separate interface?
    //       (-> class Metric - arguments may differ from evaluation point)
    virtual double norm(const Vector<double>&) const = 0;  // for (coarse) condition evaluation - metric-dependent
    virtual double metric(const Vector<double>&, const Vector<double>&) const = 0;
    // TODO: improve name
    virtual void apply_metric(const Vector<double>&, Vector<double>&) const = 0;
    virtual MetricKind get_metric() const { return MetricKind::NONE; }
    virtual unsigned n_dofs() const = 0;

    // TODO: move this to a separate interface?
    //       (-> class Residual or GrossPitaevskiiFunctional - matches evaluation point)
    virtual double residual(const Vector<double>&) const = 0;
};


// Contract for the level ("tilt") oracle handed to FullApproximationScheme::cycle():
// an OracleBase evaluating a functional of type Functional in the metric used for the coarse correction term.
template <typename T, typename Functional>
concept TiltOracle = std::derived_from<T, OracleBase>
                  && std::constructible_from<T, Functional&, SolverOptions>;

} // namespace rmo

#endif //RMO_ROPT_ORACLE_BASE_H
