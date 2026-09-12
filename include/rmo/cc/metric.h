//
// Created by Ferdinand Vanmaele.
//

#ifndef RMO_CC_METRIC_H
#define RMO_CC_METRIC_H

#include <rmo/lac.h>

#include <algorithm>
#include <cmath>

namespace rmo::cc::metric
{

/** @brief Fisher-Rao metric on the product Bernoulli manifold @f$ \mathcal{B} = (0,1)^n @f$ (Table 9). */
namespace fisher_rao
{

/**
 * @brief Diagonal entry of the inverse metric tensor, @f$ \phi(1-\phi) @f$, floored at
 * @ref MIN_WEIGHT. Every division by this weight (the retraction, the metric itself) can
 * otherwise blow up as @f$ \phi_i \f$ approaches 0 or 1 -- in particular an oversized
 * line-search step can retract to a @f$ \phi_i @f$ so close to the boundary that
 * @f$ \phi_i(1-\phi_i) @f$ underflows to exactly 0 in double precision, which would make
 * the *next* retraction divide by zero. The reference implementation guards the same
 * division the same way (@c apply_G clamps to @c min=1e-8).
 */
inline constexpr double MIN_WEIGHT = 1e-12;

inline double weight(double phi_i)
{
    return std::max(phi_i * (1.0 - phi_i), MIN_WEIGHT);
}

/** @brief Elementwise @ref weight. */
inline void inverse_metric_weights(const Vector<double>& phi, Vector<double>& dst)
{
    dst.reinit(phi.size());
    for (unsigned int i = 0; i < phi.size(); i++) {
        dst[i] = weight(phi[i]);
    }
}

/** @brief @f$ \langle u,v \rangle_\phi = \mathbb{1}^\top (uv / (\phi(1-\phi))) @f$. */
inline double inner(const Vector<double>& u, const Vector<double>& v, const Vector<double>& phi)
{
    AssertDimension(u.size(), phi.size());
    AssertDimension(v.size(), phi.size());

    double s = 0.0;
    for (unsigned int i = 0; i < phi.size(); i++) {
        const double w = weight(phi[i]);
        AssertThrow(w > 0.0, dealii::ExcMessage("phi must lie componentwise in the open interval (0,1)"));
        s += u[i] * v[i] / w;
    }
    return s;
}

/** @brief @f$ \|v\|_\phi = \sqrt{\langle v,v \rangle_\phi} @f$. */
inline double norm(const Vector<double>& v, const Vector<double>& phi)
{
    return std::sqrt(inner(v, v, phi));
}

} // namespace fisher_rao

} // namespace rmo::cc::metric

#endif //RMO_CC_METRIC_H
