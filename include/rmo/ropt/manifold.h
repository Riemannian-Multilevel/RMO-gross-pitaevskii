#ifndef RMO_ROPT_MANIFOLD_H
#define RMO_ROPT_MANIFOLD_H
#ifndef ZERO_ROUNDOFF
#define ZERO_ROUNDOFF 1e-15
#endif

#include <rmo/lac.h>
#include <rmo/util/random.h>

#include <deal.II/base/function.h>

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