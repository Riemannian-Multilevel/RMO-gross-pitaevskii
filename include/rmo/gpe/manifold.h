#ifndef RMO_GPE_MANIFOLD_H
#define RMO_GPE_MANIFOLD_H

#include <rmo/ropt/manifold.h>
#include <rmo/util/random.h>

#include <cmath>
#include <algorithm>

/**
 * @file
 * @brief The unit-mass sphere \f$ \mathcal{S} = \{ x : \|x\|_M = 1 \} \f$, \f$ \|x\|_M^2 = x^\top M x \f$:
 * retractions, their inverses and differentials (namespace ellipsoid), and the manifold UnitMassSphere.
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

/** @brief Random point on the sphere: normal entries (@p mean, @p stddev), normalized in the M-norm. */
template <typename MatrixType>
void random_point(Vector<double>& x, const MatrixType& M,
                  double mean = 0.0, double stddev = 1.0)
{
    normrnd(mean, stddev, x);
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

    double vMv = v*Mv;
    double v_Mnorm = std::sqrt(vMv);
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

} // namespace rmo::gpe

#endif //RMO_GPE_MANIFOLD_H
