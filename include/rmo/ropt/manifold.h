#ifndef RMO_ROPT_MANIFOLD_H
#define RMO_ROPT_MANIFOLD_H
#ifndef ZERO_ROUNDOFF
#define ZERO_ROUNDOFF 1e-15
#endif

#include <rmo/lac.h>
#include <rmo/util/random.h>

#include <deal.II/base/function.h>

#include <limits>

/**
 * @file
 * @brief ManifoldBase: the operations of the Riemannian solvers on a manifold embedded in \f$ \mathbb{R}^n \f$, a
 * retraction, its inverse and their differentials.
 */
namespace rmo
{

//! Placeholder for a tangent vector type (not used yet).
// TODO: type separation between manifold (point) and tangent vectors
struct TangentVector
{
    Vector<double> data;
};


//! Retraction \f$ R_x \f$ at the point \f$ x \f$, its inverse and their differentials.
class ManifoldBase
{
public:
    virtual ~ManifoldBase() = default;

    //! \f$ x \leftarrow R_x(h z) \f$ with \f$ h \f$ = @p factor
    virtual void retract(const Vector<double>& z, Vector<double>& x, double factor = 1.0) const = 0;
    //! \f$ \mathrm{output} = R_x(h z) \f$
    virtual void retract(const Vector<double>& z, const Vector<double>& x, Vector<double>& output, double factor = 1.0) const = 0;

    //! Supremum of the step sizes \f$ h \f$ for which \f$ R_x(h v) \f$ is defined; infinity if it is defined for all.
    //! The solvers take at most a fraction of it, see DescentOptions::boundary_fraction.
    [[nodiscard]] virtual double max_step(const Vector<double>& /*x*/, const Vector<double>& /*v*/) const
    {
        return std::numeric_limits<double>::infinity();
    }

    //! Point \f$ y = R_x(t v) \f$ and velocity \f$ \dot y = \frac{d}{dt} R_x(t v) \f$, as computed by retract();
    //! used by the exact line search. Not implemented by default.
    virtual void retract_velocity(const Vector<double>& /*x*/, const Vector<double>& /*v*/, double /*t*/,
                                  Vector<double>& /*y*/, Vector<double>& /*y_dot*/) const
    {
        throw dealii::ExcNotImplemented("retract_velocity");
    }

    //! \f$ y^\top M z \f$ for a manifold \f$ \{ y : y^\top M y = 1 \} \f$, 0 for a manifold without such a constraint.
    //! The exact line search subtracts \f$ \lambda(y)\, y^\top M \dot y \f$ from the directional derivative, which
    //! removes the rounding error of the normal component of \f$ \dot y \f$.
    [[nodiscard]] virtual double constraint_inner(const Vector<double>& /*y*/, const Vector<double>& /*z*/) const
    {
        return 0.0;
    }

    //! Whether retract() is the normalization \f$ (x + t v) / \|x + t v\|_M \f$; then OracleBase::ray_step() may give
    //! the exact step in closed form.
    [[nodiscard]] virtual bool retracts_by_normalization() const { return false; }

    //! \f$ v \leftarrow R_x^{-1}(v) \f$
    virtual void retract_inv(Vector<double>& v, const Vector<double>& x) const = 0;

    //! \f$ \mathrm{output} = \mathrm{D} R_x(v)[w] \f$
    virtual void retract_diff(const Vector<double>& x, const Vector<double>& v,
        const Vector<double>& w, Vector<double>& output) const = 0;

    //! \f$ \mathrm{output} = \mathrm{D} R_x^{-1}(\zeta)[u] \f$
    virtual void retract_inv_diff(const Vector<double>& x, const Vector<double>& zeta,
        const Vector<double>& u, Vector<double>& output) const = 0;

    //! Adjoint of retract_inv_diff() in the metric of the manifold
    virtual void retract_inv_diff_adjoint(const Vector<double>& x, const Vector<double>& zeta,
        const Vector<double>& u, Vector<double>& output) const = 0;
};

} // namespace rmo

#endif //RMO_ROPT_MANIFOLD_H