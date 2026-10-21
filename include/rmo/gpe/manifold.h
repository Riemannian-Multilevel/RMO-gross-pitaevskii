#ifndef RMO_GPE_MANIFOLD_H
#define RMO_GPE_MANIFOLD_H

#include <deal.II/base/exception_macros.h>
#include <deal.II/base/exceptions.h>

#include <rmo/lac_traits.h>
#include <rmo/ropt/manifold.h>
#include <rmo/util/random.h>

#include <cmath>
#include <limits>
#include <algorithm>

/**
 * @file
 * @brief The unit-mass sphere \f$ \mathcal{S} = \{ x : \|x\|_M = 1 \} \f$, \f$ \|x\|_M^2 = x^\top M x \f$:
 * retractions, their inverses and differentials (namespace ellipsoid), and the manifold UnitMassSphere; its
 * intersection with the positive orthant for a lumped mass matrix (namespace ellipsoid_positive, PosUnitMassSphere),
 * with the density retraction (m-connection) and the e-retraction (e-connection).
 *
 * Arguments in ellipsoid: the mass matrix `M` (any type with `vmult`), the base point `x` on the sphere, the
 * vector `v` to retract or lift, and the step factor \f$ h \f$ (`factor`). Retractions overwrite `x`, inverse
 * retractions overwrite `v`.
 */
namespace rmo::gpe
{

/** @brief Retractions on the unit-mass sphere, their inverses and differentials. */
namespace ellipsoid
{

/** @brief Random point on the sphere: entries drawn from @p dist, normalized in the M-norm. */
template <typename MatrixType>
void random_point(Vector<double>& x, const MatrixType& M,
                  DistributionBase<double>& dist)
{
    dist.fill(x);

    Vector<double> Mx(x.size());
    M.vmult(Mx, x);

    const double factor = x*Mx;
    x /= std::sqrt(factor);
}

/** @brief Retraction by normalization, \f$ x \leftarrow (x + h v) / \|x + h v\|_M \f$. */
template <typename MatrixType>
void retract_by_norm(const MatrixType& M, const Vector<double>& v, Vector<double>& x,
                     const double factor = 1.0)
{
    AssertThrow(factor != 0.0, dealii::ExcMessage("factor must be nonzero"));
    x.add(factor, v);           // x' <- x + h z

    Vector<double> Mx(x.size());
    M.vmult(Mx, x);
    x /= std::sqrt(x * Mx);     // x' <- x' / ||x'||_M
}

/** @brief Normalization \f$ x \leftarrow x / \|x\|_M \f$. */
template <typename MatrixType>
void retract_by_norm(const MatrixType& M, Vector<double>& x)
{
    Vector<double> Mx(x.size());
    M.vmult(Mx, x);
    x /= std::sqrt(x * Mx);
}

/** @brief \f$ y^\top M z \f$ for any matrix type with `vmult`. */
template <typename MatrixType>
double mass_inner(const MatrixType& M, const Vector<double>& y, const Vector<double>& z)
{
    Vector<double> Mz(z.size());
    M.vmult(Mz, z);
    return y * Mz;
}

/**
 * @brief Normalization of a curve with its velocity: on input \f$ y = \hat y(t) \f$ and \f$ \dot y = \hat y'(t) \f$, on
 * output \f$ y = \hat y / n \f$ and \f$ \dot y = \hat y' / n - (y^\top M \hat y' / n)\, y \f$ with
 * \f$ n = \|\hat y\|_M \f$, the derivative of \f$ \hat y(t) / \|\hat y(t)\|_M \f$.
 */
template <typename MatrixType>
void normalize_with_velocity(const MatrixType& M, Vector<double>& y, Vector<double>& y_dot)
{
    Vector<double> My(y.size());
    M.vmult(My, y);
    const double n = std::sqrt(y * My);
    y /= n;
    y_dot /= n;
    My /= n;
    y_dot.add(-(My * y_dot), y);
}

/** @brief retract_by_norm() of \f$ t v \f$ with its velocity: \f$ \hat y = x + t v \f$, \f$ \hat y' = v \f$, normalized. */
template <typename MatrixType>
void retract_velocity_by_norm(const MatrixType& M, const Vector<double>& x, const Vector<double>& v, const double t,
                              Vector<double>& y, Vector<double>& y_dot)
{
    y = x;
    y.add(t, v);
    y_dot = v;
    normalize_with_velocity(M, y, y_dot);
}

/** @brief Inverse of retract_by_norm() for \f$ v \in \mathcal{S} \f$: \f$ v \leftarrow v / (x^\top M v) - x \f$. */
template <typename MatrixType>
void retract_inv_by_norm(const MatrixType& M, Vector<double>& v, const Vector<double>& x)
{
    Vector<double> Mv(v.size());
    M.vmult(Mv, v);

    const double xMv = x*Mv;
    AssertThrow(std::abs(xMv) > 0, dealii::ExcInternalError("x'Mv must be nonzero"));

    v /= xMv;
    v.add(-1.0, x);
}

// TODO: use x as output vector as with other functions
/**
 * @brief Differential of retract_by_norm() at \f$ v \f$ in the direction \f$ w \in T_x \mathcal{S} \f$, with
 * \f$ y = x + v \f$ and using \f$ x^\top M w = 0 \f$:
 * \f[ \mathrm{dst} = \mathrm{D} R_x(v)[w] = \frac{1}{\|y\|_M} \Big( w - \frac{v^\top M w}{\|y\|_M^2} \, y \Big) \f]
 */
template <typename MatrixType>
void retract_diff_by_norm(const MatrixType& M,
                          const Vector<double>& x,
                          const Vector<double>& v,
                          const Vector<double>& w,
                          Vector<double>& dst)
{
    // 1. Compute y = phi + v
    Vector<double> y(x);
    y += v;

    // 2. Compute My = M * y
    Vector<double> My(y.size());
    M.vmult(My, y);

    // 3. Compute norm_sq = ||phi+v||_M^2 = y^T M y
    const double norm_sq = y * My;
    AssertThrow(std::abs(norm_sq) > 0, dealii::ExcMessage("Norm of (phi+v) must be non-zero"));
    const double norm = std::sqrt(norm_sq);

    // 4. Compute Mw = M * w
    // Needed for the v^T M w term.
    // Note: Since M is symmetric, v^T M w = (M v)^T w = v^T (M w)
    Vector<double> Mw(w.size());
    M.vmult(Mw, w);

    // 5. Compute scalar alpha = (v^T M w) / norm_sq
    // Note: ((phi+v) v^T M w) / N^2
    const double vMw = v * Mw;
    const double alpha = vMw / norm_sq;

    // 6. Compute dst = (w - alpha * y) / norm
    dst = w;
    dst.add(-alpha, y); // dst <- dst - alpha * y
    dst /= norm;
}

/**
 * @brief Differential of retract_inv_by_norm() at \f$ \zeta \f$ in the direction \f$ u \f$:
 * \f[ \mathrm{dst} = \mathrm{D} R_x^{-1}(\zeta)[u]
 *     = \frac{1}{x^\top M \zeta} \Big( u - \frac{x^\top M u}{x^\top M \zeta} \, \zeta \Big) \f]
 */
template <typename MatrixType>
void retract_inv_diff_by_norm(const MatrixType& M,
                              const Vector<double>& x,
                              const Vector<double>& zeta,
                              const Vector<double>& u,
                              Vector<double>& dst)
{
    // 1. Compute Mzeta = M * zeta
    Vector<double> Mzeta(zeta.size());
    M.vmult(Mzeta, zeta);

    // 2. Compute beta = phi^T M zeta
    const double beta = x * Mzeta;
    AssertThrow(std::abs(beta) > 0, dealii::ExcMessage("phi^T M zeta must be non-zero"));

    // 3. Compute Mu = M * u
    Vector<double> Mu(u.size());
    M.vmult(Mu, u);

    // 4. Compute gamma = phi^T M u
    const double gamma = x * Mu;

    // 5. Compute dst = (1/beta) * (u - (gamma/beta) * zeta)
    dst = u;
    dst.add(-gamma / beta, zeta); // dst <- dst - (gamma/beta) * zeta
    dst /= beta;
}

// TODO: verify adjoint property
/**
 * @brief M-adjoint of retract_inv_diff_by_norm(), taken as that differential with \f$ x \f$ and \f$ \zeta \f$
 * swapped (not yet verified, see the TODO).
 */
template <typename MatrixType>
void retract_inv_diff_by_norm_adjoint(const MatrixType& M,
                                      const Vector<double>& x,
                                      const Vector<double>& zeta,
                                      const Vector<double>& u,
                                      Vector<double>& dst)
{
    retract_inv_diff_by_norm(M, zeta, x, u, dst);
}

// TODO: Adjoint for A-metric
inline void retract_inv_diff_by_norm_adjoint()
{
    throw dealii::ExcNotImplemented("retract_inv_diff_by_norm_adjoint");
}

/**
 * @brief Orthographic retraction \f$ x \leftarrow \sqrt{1 - \|h v\|_M^2} \, x + h v \f$; requires
 * \f$ \|h v\|_M < 1 \f$.
 */
template <typename MatrixType>
void retract_by_ortho(const MatrixType& M, const Vector<double>& v,
                      Vector<double>& x, const double factor = 1.0)
{
    AssertThrow(std::abs(factor) > 0, dealii::ExcMessage("factor must be non-zero"));
    Vector<double> Mv(x.size());
    M.vmult(Mv, v);

    double vMv = v*Mv;
    vMv *= factor;
    vMv *= factor;  // (hz) * M(hZ) = h^2 zMz
    AssertThrow(vMv < 1.0, dealii::ExcInternalError("z'Mz required < 1"));

    x *= std::sqrt(1-vMv);
    x.add(factor, v);
}

/** @brief Inverse of retract_by_ortho(): \f$ v \leftarrow v - (x^\top M v) \, x \f$. */
template <typename MatrixType>
void retract_inv_by_ortho(const MatrixType& M, Vector<double>& v, const Vector<double>& x)
{
    Vector<double> Mv(v.size());
    M.vmult(Mv, v);

    const double xMv = x*Mv;
    v.add(-xMv, x);
}

/**
 * @brief Exponential map \f$ x \leftarrow \cos(h \|v\|_M) \, x + \sin(h \|v\|_M) \, v / \|v\|_M \f$, i.e. along
 * the geodesic.
 */
template <typename MatrixType>
void retract_by_exp(const MatrixType& M, const Vector<double>& v, Vector<double>& x,
                    const double factor = 1.0)
{
    AssertThrow(std::abs(factor) > 0, dealii::ExcMessage("factor must be non-zero"));
    Vector<double> Mv(x.size());
    M.vmult(Mv, v);

    const double vMv = v*Mv;
    const double v_Mnorm = std::sqrt(vMv);
    AssertThrow(v_Mnorm > 0.0, dealii::ExcInternalError("|z|_M must be positive"));

    // Derivation:
    //                  |hz|_M  =  h |z|_M
    //             hz / |hz|_M  =  z / |z|_M
    // sin(|hz|_M) hz / |hz|_M  =  sin(h|z|_M) z / |z|_M
    x *= std::cos(factor*v_Mnorm);
    x.add(std::sin(factor*v_Mnorm) / v_Mnorm, v);
}

/**
 * @brief Logarithmic map, the inverse of retract_by_exp(), with \f$ c = x^\top M v \f$:
 * \f$ v \leftarrow \frac{\arccos c}{\sqrt{1 - c^2}} (v - c \, x) \f$ (factor 1 in the limit \f$ v \to x \f$).
 */
template <typename MatrixType>
void retract_inv_by_exp(const MatrixType& M, Vector<double>& v, const Vector<double>& x)
{
    Vector<double> Mv(v.size());
    M.vmult(Mv, v);

    // Clamp to [-1.0, 1.0] to prevent NaN in acos due to numerical errors
    double xMv = std::clamp(x * Mv, -1.0, 1.0);

    v.add(-xMv, x); // v <- Pi_x(v)

    // Handle the limit case where x and v are identical (xMv -> 1.0)
    // The limit of arccos(z)/sqrt(1-z^2) as z->1 is 1.0
    if (std::abs(1.0 - xMv) < 1e-14) {
        // v is already Pi_x(v), and the scalar multiplier is 1.0.
        return;
    }
    const double nom   = std::acos(xMv);
    const double denom = std::sin(nom); // sin(arccos(x)) = sqrt(1-x^2)

    v *= (nom / denom);
}

} // namespace ellipsoid


/**
 * @brief The positive unit-mass sphere \f$ \mathcal{S}^+ = \{ x > 0 : x^\top D x = 1 \} \f$ of a diagonal (lumped)
 * mass matrix \f$ D \f$: density map, density retraction and its inverse (the lifting).
 *
 * For \f$ x^\top D v = 0 \f$, the density retraction stays on the sphere, since
 * \f$ \sum_i d_i (x_i^2 + 2 x_i v_i) = x^\top D x + 2 x^\top D v = 1 \f$; this requires \f$ D \f$ to be diagonal.
 * Arguments as in ellipsoid.
 */
namespace ellipsoid_positive
{

//! Default lower bound of threshold()
inline constexpr double eps_def = 1e-9;

/** @brief Density map \f$ p = \pi(x) \f$, \f$ p_i = d_i x_i^2 \f$ (@p D diagonal). */
template <typename MatrixType>
void density_map(const MatrixType& D, const Vector<double>& x, Vector<double>& p)
{
    Vector<double> x_sq(x);
    x_sq.scale(x);

    p.reinit(x.size(), true);
    D.vmult(p, x_sq);
}

/** @brief Differential of density_map(), \f$ \mathrm{dst} = J_x v = 2 D \operatorname{diag}(x) v \f$. */
template <typename MatrixType>
void density_map_diff(const MatrixType& D, const Vector<double>& x, const Vector<double>& v, Vector<double>& dst)
{
    AssertDimension(x.size(), v.size());
    Vector<double> xv(x);
    xv.scale(v);
    xv *= 2.0;

    dst.reinit(x.size(), true);
    D.vmult(dst, xv);
}

/** @brief Componentwise \f$ x_i \leftarrow \max(\varepsilon, x_i) \f$; changes the M-norm of @p x. */
// TODO: using this function requires specifying the threshold in every kernel.
// This can be remedied by using a manifold class later on (such as PosUnitMassSphere).
inline void threshold(Vector<double>& x, const double eps = eps_def)
{
    for (double& a: x) {
        a = std::max(eps, a);
    }
}

/** @brief Random point on the positive sphere: entries drawn from @p dist, thresholded and normalized in the D-norm. */
template <typename MatrixType>
void random_point(Vector<double>& x, const MatrixType& D, DistributionBase<double>& dist, const double eps = eps_def)
{
    dist.fill(x);
    threshold(x, eps);

    Vector<double> Dx(x.size());
    D.vmult(Dx, x);
    x /= std::sqrt(x * Dx);
}

/**
 * @brief Density retraction \f$ x_i \leftarrow \sqrt{x_i^2 + 2 h x_i v_i} \f$, defined for
 * \f$ x_i^2 + 2 h x_i v_i > 0 \f$; @p x is unchanged if it is not.
 */
// TODO: investigate cases in which ret**2 is strictly positive, i.e. x_i + 2v_i > 0
//       use destination vector like diff/inverse retraction
inline void retract_by_density(const Vector<double>& v, Vector<double>& x, const double factor = 1.0)
{
    // TODO: tangent vector debug check
    Assert(std::all_of(x.begin(), x.end(), [](double a) {
        return a > 0;
    }), dealii::ExcMessage("Point has non-positive component"));

    const unsigned n = x.size();
    AssertDimension(n, v.size());

    Vector<double> tmp(n);

    for (unsigned i = 0; i < n; i++) {
        tmp[i] = x[i]*x[i] + 2*factor*x[i]*v[i];
    }

    AssertThrow(std::all_of(tmp.begin(), tmp.end(), [](double a) {
        return a > 0;
    }), dealii::ExcMessage("Retraction is undefined"));

    for (unsigned i = 0; i < n; i++) {
        x[i] = std::sqrt(tmp[i]);
    }
}

/**
 * @brief Lifting, the inverse of retract_by_density(): \f$ v_i \leftarrow (v_i^2 - x_i^2) / (2 x_i) \f$ for a point
 * \f$ v \f$ on the positive sphere, which is overwritten with a tangent vector at @p x.
 */
inline void retract_inv_by_density(Vector<double>& v, const Vector<double>& x)
{
    // TODO: tangent vector debug check
    Assert(std::all_of(x.begin(), x.end(), [](double a) {
        return a > 0;
    }), dealii::ExcMessage("Point has non-positive component"));

    const unsigned n = x.size();
    AssertDimension(n, v.size());

    // Factored form: exact zero for v_i = x_i, no cancellation of v_i^2 - x_i^2 (which a contracted FMA evaluates to
    // the rounding error of x_i^2)
    for (unsigned i = 0; i < n; i++) {
        v[i] = (v[i] - x[i]) * (v[i] + x[i]) / (2*x[i]);
    }
}

/**
 * @brief Differential of retract_inv_by_density() at the point \f$ \zeta \f$ in the direction \f$ u \f$:
 * \f$ \mathrm{D} \mathcal{L}_x(\zeta)[u] = (\zeta / x) \circ u \f$. It is diagonal, hence self-adjoint in the
 * metric of a diagonal mass matrix; at \f$ \zeta = x \f$ it is the identity.
 */
inline void retract_inv_diff_by_density(const Vector<double>& x, const Vector<double>& zeta, const Vector<double>& u,
                                        Vector<double>& dst)
{
    const unsigned n = x.size();
    AssertDimension(n, zeta.size());
    AssertDimension(n, u.size());
    dst.reinit(n, true);
    for (unsigned i = 0; i < n; i++) {
        dst[i] = zeta[i] * u[i] / x[i];
    }
}

/**
 * @brief Supremum of the step sizes \f$ h \f$ for which retract_by_density() of \f$ h v \f$ at @p x is defined:
 * \f$ h_{\max} = \min_{v_i < 0} x_i / (2 |v_i|) \f$, the step at which the first density
 * \f$ x_i^2 + 2 h x_i v_i \f$ reaches zero; infinity if no component decreases.
 */
inline double max_step(const Vector<double>& x, const Vector<double>& v)
{
    AssertDimension(x.size(), v.size());
    double h_max = std::numeric_limits<double>::infinity();

    for (unsigned i = 0; i < x.size(); i++) {
        if (v[i] < 0) {
            h_max = std::min(h_max, x[i] / (-2 * v[i]));
        }
    }
    return h_max;
}

/**
 * @brief Exponential retraction (e-retraction) \f$ x_i \leftarrow c \, x_i \exp(h v_i / x_i) \f$, with \f$ c > 0 \f$
 * such that \f$ x^\top D x = 1 \f$.
 *
 * Under the density map it is the e-geodesic \f$ p_i(h) \propto p_i \exp(h u_i) \f$, \f$ u_i = (J_x v)_i / p_i =
 * 2 v_i / x_i \f$, of the exponential connection of information geometry; the density retraction is the m-geodesic
 * \f$ p + h J_x v \f$ of the mixture connection. As retract_by_ortho(), it steps along @p v and rescales; unlike the
 * density retraction, it is defined and positive for every step. For tangent @p v, \f$ \mathrm D R_x(0) = \mathrm{id}
 * \f$. Evaluated in logarithms, shifted by their maximum against overflow; components that underflow are set to the
 * smallest normal number.
 */
template <typename MatrixType>
void retract_by_e(const MatrixType& D, const Vector<double>& v, Vector<double>& x, const double factor = 1.0)
{
    Assert(std::all_of(x.begin(), x.end(), [](double a) {
        return a > 0;
    }), dealii::ExcMessage("Point has non-positive component"));

    const unsigned n = x.size();
    AssertDimension(n, v.size());

    // s_i = log(x_i exp(h v_i / x_i))
    Vector<double> s(n);
    for (unsigned i = 0; i < n; i++) {
        s[i] = std::log(x[i]) + factor * v[i] / x[i];
    }
    AssertThrow(std::all_of(s.begin(), s.end(), [](double a) {
        return std::isfinite(a);
    }), dealii::ExcMessage("e-retraction: exponent not finite"));

    const double s_max = *std::max_element(s.begin(), s.end());
    for (unsigned i = 0; i < n; i++) {
        x[i] = std::exp(s[i] - s_max);
    }
    ellipsoid::retract_by_norm(D, x);
    threshold(x, std::numeric_limits<double>::min());
}

/**
 * @brief Inverse of retract_by_e() for a point \f$ v \f$ on the positive sphere, overwritten with a tangent vector at
 * @p x: \f$ v_i \leftarrow x_i \big( \log(v_i / x_i) - \sum_j p_j \log(v_j / x_j) \big) \f$, \f$ p = \pi(x) \f$.
 */
template <typename MatrixType>
void retract_inv_by_e(const MatrixType& D, Vector<double>& v, const Vector<double>& x)
{
    const unsigned n = x.size();
    AssertDimension(n, v.size());

    Vector<double> p(n);
    density_map(D, x, p);

    double mean = 0, mass = 0;
    for (unsigned i = 0; i < n; i++) {
        v[i] = std::log(v[i] / x[i]);
        mean += p[i] * v[i];
        mass += p[i];
    }
    mean /= mass;
    for (unsigned i = 0; i < n; i++) {
        v[i] = x[i] * (v[i] - mean);
    }
}

/**
 * @brief Density retraction of \f$ t v \f$ followed by the normalization (as PosUnitMassSphere::retract()), with its
 * velocity: \f$ \hat y_i = \sqrt{x_i^2 + 2 t x_i v_i} \f$, \f$ \hat y_i' = x_i v_i / \hat y_i \f$.
 */
template <typename MatrixType>
void retract_velocity_by_density(const MatrixType& D, const Vector<double>& x, const Vector<double>& v, const double t,
                                 Vector<double>& y, Vector<double>& y_dot)
{
    const unsigned n = x.size();
    AssertDimension(n, v.size());
    y.reinit(n, true);
    y_dot.reinit(n, true);
    for (unsigned i = 0; i < n; i++) {
        const double q = x[i] * x[i] + 2 * t * x[i] * v[i];
        AssertThrow(q > 0, dealii::ExcMessage("Retraction is undefined"));
        y[i] = std::sqrt(q);
        y_dot[i] = x[i] * v[i] / y[i];
    }
    ellipsoid::normalize_with_velocity(D, y, y_dot);
}

/**
 * @brief retract_by_e() of \f$ t v \f$ with its velocity: \f$ \hat y = x e^{t w} \f$, \f$ w = v / x \f$, scaled by
 * \f$ e^{-\max_i s_i} \f$ (\f$ s_i = \log x_i + t w_i \f$), \f$ \hat y' = \hat y w \f$, normalized; components raised to
 * the smallest normal number have velocity 0.
 */
template <typename MatrixType>
void retract_velocity_by_e(const MatrixType& D, const Vector<double>& x, const Vector<double>& v, const double t,
                           Vector<double>& y, Vector<double>& y_dot)
{
    const unsigned n = x.size();
    AssertDimension(n, v.size());
    y.reinit(n, true);
    y_dot.reinit(n, true);
    double s_max = -std::numeric_limits<double>::infinity();
    for (unsigned i = 0; i < n; i++) {
        y[i] = std::log(x[i]) + t * v[i] / x[i];
        s_max = std::max(s_max, y[i]);
    }
    AssertThrow(std::isfinite(s_max), dealii::ExcMessage("e-retraction: exponent not finite"));
    for (unsigned i = 0; i < n; i++) {
        y[i] = std::exp(y[i] - s_max);
        y_dot[i] = y[i] * v[i] / x[i];
    }
    ellipsoid::normalize_with_velocity(D, y, y_dot);
    for (unsigned i = 0; i < n; i++) {
        if (y[i] < std::numeric_limits<double>::min()) {
            y[i] = std::numeric_limits<double>::min();
            y_dot[i] = 0.0;
        }
    }
}

} // namespace ellipsoid_positive


/** @brief The unit-mass sphere as ManifoldBase, with retraction by normalization (see ellipsoid). */
template <typename MatrixType>
class UnitMassSphere : public ManifoldBase
{
public:
    explicit UnitMassSphere(const MatrixType& M) : M(M) {}

    /** @brief Retraction by normalization, see ellipsoid::retract_by_norm(). */
    void retract(const Vector<double>& z, Vector<double>& x, double factor) const override
    {
        ellipsoid::retract_by_norm(M, z, x, factor);
    }

    void retract(const Vector<double>& z, const Vector<double>& x, Vector<double>& output, double factor) const override
    {
        output = x;
        retract(z, output, factor);
    }

    void retract_diff(const Vector<double>& x, const Vector<double>& v, const Vector<double>& w,
                      Vector<double>& output) const override
    {
        ellipsoid::retract_diff_by_norm(M, x, v, w, output);
    }

    void retract_velocity(const Vector<double>& x, const Vector<double>& v, double t,
                          Vector<double>& y, Vector<double>& y_dot) const override
    {
        ellipsoid::retract_velocity_by_norm(M, x, v, t, y, y_dot);
    }

    [[nodiscard]] double constraint_inner(const Vector<double>& y, const Vector<double>& z) const override
    {
        return ellipsoid::mass_inner(M, y, z);
    }

    [[nodiscard]] bool retracts_by_normalization() const override { return true; }

    void retract_inv(Vector<double>& v, const Vector<double>& x) const override
    {
        ellipsoid::retract_inv_by_norm(M, v, x);
    }

    void retract_inv_diff(const Vector<double>& x, const Vector<double>& zeta, const Vector<double>& u,
                          Vector<double>& output) const override
    {
        ellipsoid::retract_inv_diff_by_norm(M, x, zeta, u, output);
    }

    void retract_inv_diff_adjoint(const Vector<double>& x, const Vector<double>& zeta, const Vector<double>& u,
                                  Vector<double>& output) const override
    {
        ellipsoid::retract_inv_diff_by_norm_adjoint(M, x, zeta, u, output);
    }

    // Accessors
    const auto& get_M() const { return M; }

private:
    const MatrixType& M;
};


/**
 * @brief The positive unit-mass sphere as ManifoldBase, with the density retraction (see ellipsoid_positive). The
 * mass matrix must be diagonal (lumped), see ellipsoid_positive.
 */
template <typename MatrixType>
class PosUnitMassSphere : public ManifoldBase
{
    static_assert(is_diagonal_matrix_v<MatrixType>,
                  "the density retraction stays on the sphere only for a diagonal (lumped) mass matrix");

public:
    explicit PosUnitMassSphere(const MatrixType& D) : D(D) {}

    /**
     * @brief Density retraction, see ellipsoid_positive::retract_by_density(), followed by a normalization in the
     * D-norm. For a tangent @p z, the normalization is the identity in exact arithmetic. It removes the drift of
     * rounding errors: the density retraction preserves \f$ x^\top D x \f$ but does not restore it, and the M-gradient
     * is tangent only on the sphere, so that \f$ x^\top D x - 1 \f$ would grow by \f$ 1 + 2h \, x^\top A(x) x \f$ per
     * step.
     */
    void retract(const Vector<double>& z, Vector<double>& x, double factor) const override
    {
        ellipsoid_positive::retract_by_density(z, x, factor);
        ellipsoid::retract_by_norm(D, x);
    }

    void retract(const Vector<double>& z, const Vector<double>& x,
                 Vector<double>& output, double factor) const override
    {
        output = x;
        retract(z, output, factor);
    }

    // TODO: differential of the density retraction
    void retract_diff(const Vector<double>& /*x*/, const Vector<double>& /*v*/, const Vector<double>& /*w*/,
                      Vector<double>& /*output*/) const override
    {
        throw dealii::ExcNotImplemented();
    }

    /** @brief Step at which the density retraction becomes undefined, see ellipsoid_positive::max_step(). */
    [[nodiscard]] double max_step(const Vector<double>& x, const Vector<double>& v) const override
    {
        return ellipsoid_positive::max_step(x, v);
    }

    /** @brief retract() of \f$ t v \f$ with its velocity, see ellipsoid_positive::retract_velocity_by_density(). */
    void retract_velocity(const Vector<double>& x, const Vector<double>& v, double t,
                          Vector<double>& y, Vector<double>& y_dot) const override
    {
        ellipsoid_positive::retract_velocity_by_density(D, x, v, t, y, y_dot);
    }

    [[nodiscard]] double constraint_inner(const Vector<double>& y, const Vector<double>& z) const override
    {
        return ellipsoid::mass_inner(D, y, z);
    }

    /** @brief Lifting, see ellipsoid_positive::retract_inv_by_density(). */
    void retract_inv(Vector<double>& v, const Vector<double>& x) const override
    {
        ellipsoid_positive::retract_inv_by_density(v, x);
    }

    /** @brief Differential of the lifting, see ellipsoid_positive::retract_inv_diff_by_density(). */
    void retract_inv_diff(const Vector<double>& x, const Vector<double>& zeta, const Vector<double>& u,
                          Vector<double>& output) const override
    {
        ellipsoid_positive::retract_inv_diff_by_density(x, zeta, u, output);
    }

    /** @brief D-adjoint of retract_inv_diff(): the differential is diagonal, hence self-adjoint. */
    void retract_inv_diff_adjoint(const Vector<double>& x, const Vector<double>& zeta, const Vector<double>& u,
                                  Vector<double>& output) const override
    {
        ellipsoid_positive::retract_inv_diff_by_density(x, zeta, u, output);
    }

    // Accessors
    const auto& get_D() const { return D; }

private:
    const MatrixType& D;
};


} // namespace rmo::gpe

#endif //RMO_GPE_MANIFOLD_H
