#ifndef RMO_ROPT_MANIFOLD_H
#define RMO_ROPT_MANIFOLD_H
#ifndef ZERO_ROUNDOFF
#define ZERO_ROUNDOFF 1e-15
#endif

#include <rmo/lac.h>
#include <rmo/util/random.h>

#include <deal.II/base/function.h>

namespace rmo
{

// TODO: type separation between manifold (point) and tangent vectors
struct TangentVector
{
    Vector<double> data;
};


class ManifoldBase
{
public:
    virtual ~ManifoldBase() = default;

    virtual void retract(const Vector<double>& z, Vector<double>& x, double factor = 1.0) const = 0;
    virtual void retract(const Vector<double>& z, const Vector<double>& x, Vector<double>& output, double factor = 1.0) const = 0;

    virtual void retract_inv(Vector<double>& v, const Vector<double>& x) const = 0;

    virtual void retract_diff(const Vector<double>& x, const Vector<double>& v,
        const Vector<double>& w, Vector<double>& output) const = 0;

    virtual void retract_inv_diff(const Vector<double>& x, const Vector<double>& zeta,
        const Vector<double>& u, Vector<double>& output) const = 0;

    virtual void retract_inv_diff_adjoint(const Vector<double>& x, const Vector<double>& zeta,
        const Vector<double>& u, Vector<double>& output) const = 0;
};

} // namespace rmo

#endif //RMO_ROPT_MANIFOLD_H