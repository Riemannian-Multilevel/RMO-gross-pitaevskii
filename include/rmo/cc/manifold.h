//
// Created by Ferdinand Vanmaele.
//

#ifndef RMO_CC_MANIFOLD_H
#define RMO_CC_MANIFOLD_H

#include <rmo/cc/metric.h>
#include <rmo/ropt/manifold.h>

#include <algorithm>
#include <cmath>

namespace rmo::cc
{

/**
 * @brief Retraction and lifting map for the product Bernoulli manifold
 * @f$ \mathcal{B} = (0,1)^n @f$ (Table 9).
 *
 * Both reduce to logit-space identities:
 * @f[
 *   R_\phi(v)          = \sigma\big( \mathrm{logit}(\phi) + G(\phi) v \big), \qquad
 *   \mathcal{L}_\phi(z) = G(\phi)^{-1}\big( \mathrm{logit}(z) - \mathrm{logit}(\phi) \big),
 * @f]
 * with @f$ \sigma @f$ the logistic sigmoid and @f$ G(\phi) @f$ the Fisher-Rao metric tensor.
 * Implementing them through the stable @ref sigmoid / @ref logit primitives below, rather
 * than the raw exponential form, avoids overflow as components of @f$ \phi @f$ approach
 * 0 or 1.
 */
namespace bernoulli
{

/**
 * @brief Numerically stable logistic sigmoid @f$ \sigma(t) = 1/(1+e^{-t}) @f$, clamped away
 * from its exact endpoints. For large @f$ |t| @f$ (e.g.\ from an oversized line-search step
 * retracting a point close to the boundary) the unclamped formula below rounds to exactly
 * 0 or 1 in double precision; without the clamp that point would then fail @ref logit on
 * the very next call, even though @f$ \B=(0,1)^n @f$ is mathematically open and the
 * e-retraction never actually reaches its boundary.
 */
inline double sigmoid(double t)
{
    double s;
    if (t >= 0.0) {
        s = 1.0 / (1.0 + std::exp(-t));
    } else {
        const double e = std::exp(t);
        s = e / (1.0 + e);
    }
    return std::clamp(s, metric::fisher_rao::MIN_WEIGHT, 1.0 - metric::fisher_rao::MIN_WEIGHT);
}

/** @brief Numerically stable logit @f$ \mathrm{logit}(s) = \log(s) - \log(1-s) @f$, for @f$ s \in (0,1) @f$. */
inline double logit(double s)
{
    AssertThrow(s > 0.0 && s < 1.0, dealii::ExcMessage("logit argument must lie in the open interval (0,1)"));
    return std::log(s) - std::log1p(-s);
}

/** @brief Elementwise @ref sigmoid. */
inline void sigmoid(const Vector<double>& t, Vector<double>& dst)
{
    dst.reinit(t.size());
    for (unsigned int i = 0; i < t.size(); i++) {
        dst[i] = sigmoid(t[i]);
    }
}

/** @brief Elementwise @ref logit. */
inline void logit(const Vector<double>& s, Vector<double>& dst)
{
    dst.reinit(s.size());
    for (unsigned int i = 0; i < s.size(); i++) {
        dst[i] = logit(s[i]);
    }
}

/** @brief e-retraction (Table 9), out-of-place. */
inline void retract(const Vector<double>& v, const Vector<double>& phi, Vector<double>& dst, double factor = 1.0)
{
    AssertDimension(v.size(), phi.size());
    dst.reinit(phi.size());

    for (unsigned int i = 0; i < phi.size(); i++) {
        const double w = metric::fisher_rao::weight(phi[i]);
        AssertThrow(w > 0.0, dealii::ExcMessage("phi must lie componentwise in the open interval (0,1)"));

        dst[i] = sigmoid(logit(phi[i]) + factor * v[i] / w);
    }
}

/** @brief e-retraction, in-place: @f$ \phi \leftarrow R_\phi(\mathrm{factor} \cdot v) @f$. */
inline void retract(const Vector<double>& v, Vector<double>& phi, double factor = 1.0)
{
    Vector<double> tmp;
    retract(v, phi, tmp, factor);
    phi = tmp;
}

/**
 * @brief Differentiated retraction (not tabulated explicitly; needed for
 * @ref ManifoldBase::retract_diff), obtained by differentiating @f$ \sigma @f$
 * in the logit-coordinate form of the retraction (@f$ \sigma'(t) = \sigma(t)(1-\sigma(t)) @f$):
 * @f[ \mathrm{D}R_\phi(v)[w] = z(1-z) \cdot \dfrac{w}{\phi(1-\phi)}, \qquad z = R_\phi(v). @f]
 */
inline void retract_diff(const Vector<double>& phi, const Vector<double>& v, const Vector<double>& w, Vector<double>& dst)
{
    Vector<double> z;
    retract(v, phi, z, 1.0);

    dst.reinit(phi.size());
    for (unsigned int i = 0; i < phi.size(); i++) {
        dst[i] = metric::fisher_rao::weight(z[i]) * w[i] / metric::fisher_rao::weight(phi[i]);
    }
}

/**
 * @brief Lifting map (Table 9), the local inverse of the e-retraction, in-place.
 * On input @p z is a point of @f$ \mathcal{B} @f$; on output it is the tangent vector
 * @f$ \mathcal{L}_\phi(z) \in \mathrm{T}_\phi\mathcal{B} @f$.
 */
inline void retract_inv(Vector<double>& z, const Vector<double>& phi)
{
    AssertDimension(z.size(), phi.size());

    for (unsigned int i = 0; i < phi.size(); i++) {
        z[i] = metric::fisher_rao::weight(phi[i]) * (logit(z[i]) - logit(phi[i]));
    }
}

/** @brief Differentiated lifting map (Table 9): @f$ \mathrm{D}\mathcal{L}_\phi(z)[u] = \dfrac{\phi(1-\phi)}{z(1-z)}\, u @f$. */
inline void retract_inv_diff(const Vector<double>& phi, const Vector<double>& z, const Vector<double>& u, Vector<double>& dst)
{
    AssertDimension(phi.size(), z.size());
    AssertDimension(phi.size(), u.size());
    dst.reinit(phi.size());

    for (unsigned int i = 0; i < phi.size(); i++) {
        const double w_z = metric::fisher_rao::weight(z[i]);
        AssertThrow(w_z > 0.0, dealii::ExcMessage("z must lie componentwise in the open interval (0,1)"));

        dst[i] = metric::fisher_rao::weight(phi[i]) / w_z * u[i];
    }
}

/**
 * @brief Adjoint of the differentiated lifting map w.r.t. the Euclidean pairing.
 * @ref retract_inv_diff is a diagonal linear map, hence self-adjoint under the Euclidean
 * pairing, so this function is identical to it.
 */
inline void retract_inv_diff_adjoint(const Vector<double>& phi, const Vector<double>& z, const Vector<double>& u, Vector<double>& dst)
{
    retract_inv_diff(phi, z, u, dst);
}

} // namespace bernoulli


/** @brief The product Bernoulli manifold @f$ \mathcal{B} = (0,1)^n @f$ (Table 9), as a @ref ManifoldBase. */
class BernoulliManifold : public ManifoldBase
{
public:
    void retract(const Vector<double>& z, Vector<double>& x, double factor = 1.0) const override
    {
        bernoulli::retract(z, x, factor);
    }

    void retract(const Vector<double>& z, const Vector<double>& x, Vector<double>& output, double factor = 1.0) const override
    {
        bernoulli::retract(z, x, output, factor);
    }

    void retract_inv(Vector<double>& v, const Vector<double>& x) const override
    {
        bernoulli::retract_inv(v, x);
    }

    void retract_diff(const Vector<double>& x, const Vector<double>& v, const Vector<double>& w,
                      Vector<double>& output) const override
    {
        bernoulli::retract_diff(x, v, w, output);
    }

    void retract_inv_diff(const Vector<double>& x, const Vector<double>& zeta, const Vector<double>& u,
                          Vector<double>& output) const override
    {
        bernoulli::retract_inv_diff(x, zeta, u, output);
    }

    void retract_inv_diff_adjoint(const Vector<double>& x, const Vector<double>& zeta, const Vector<double>& u,
                                  Vector<double>& output) const override
    {
        bernoulli::retract_inv_diff_adjoint(x, zeta, u, output);
    }
};

} // namespace rmo::cc

#endif //RMO_CC_MANIFOLD_H
