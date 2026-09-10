#ifndef RMO_GPE_MANIFOLD_H
#define RMO_GPE_MANIFOLD_H

#include <rmo/ropt/manifold.h>
#include <rmo/util/random.h>

#include <cmath>
#include <algorithm>

namespace rmo::gpe
{

namespace ellipsoid
{

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

/**
 * @brief Computes the retraction by normalization.
 *
 * Maps a tangent vector @p z at @p x onto the manifold by adding the update and
 * normalizing the result:
 * \f[
 * R_x(h z) = \frac{x + h z}{\|x + h z\|_M}
 * \f]
 *
 * @note This function modifies @p x in-place.
 *
 * @tparam MatrixType A matrix class type providing a `vmult` method.
 * @param[in] M The mass matrix defining the metric.
 * @param[in] v The tangent vector (update direction).
 * @param[in,out] x On input, the base point. On output, the retracted point.
 * @param[in] factor Scaling factor \f$ h \f$.
 */
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

template <typename MatrixType>
void retract_by_norm(const MatrixType& M, Vector<double>& x)
{
    Vector<double> Mx(x.size());
    M.vmult(Mx, x);
    x /= std::sqrt(x * Mx);
}

/**
 * @brief Computes the inverse retraction by normalization.
 *
 * Lifts a point @p v back to the tangent space of @p x by reversing the normalization
 * projection.
 *
 * Formula:
 * \f[
 * v \leftarrow \frac{v}{\langle x, v \rangle_M} - x
 * \f]
 *
 * @note This function modifies @p v in-place.
 *
 * @tparam MatrixType A matrix class type providing a `vmult` method.
 * @param[in] M The mass matrix defining the metric.
 * @param[in,out] v On input, the point on the manifold. On output, the tangent vector.
 * @param[in] x The base point on the manifold.
 */
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
 * @brief Computes the differentiated retraction by normalization.
 * * Evaluates the differential of the retraction map at $v$ in the direction $w$:
 * $$ \mathrm{D} R_\phi(v)[w] = \frac{1}{\|\phi+v\|_M} \left( I - \frac{(\phi+v)(\phi+v)^\top M}{\|\phi+v\|_M^2} \right) w $$
 *
 * @note Because $w \in T_\phi \mathcal{M}$, we have $\phi^\top M w = 0$.
 * Thus, the numerator $(\phi+v)^\top M w$ simplifies exactly to $v^\top M w$.
 *
 * @tparam MatrixType A matrix class type providing a `vmult` method.
 * @param[in] M Mass matrix defining the metric.
 * @param[in] x Base point $\phi$ on the manifold.
 * @param[in] v Tangent vector (argument of the retraction).
 * @param[in] w Direction of differentiation.
 * @param[out] dst Resulting vector.
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
 * @brief Computes the differentiated inverse retraction by normalization.
 * D_invRet_phi(zeta)[u] = (1 / (phi^T M zeta)) * ( I - ( zeta phi^T M ) / (phi^T M zeta) ) * u
 *
 * @tparam MatrixType
 * @param M Mass matrix
 * @param x Base point
 * @param zeta Argument of inverse retraction
 * @param u Direction of differentiation
 * @param dst Resulting vector
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
 * @brief Computes the adjoint of the differentiated inverse retraction.
 * In the mass-weighted metric on the sphere, the adjoint of the differential of
 * the inverse retraction evaluates exactly to the forward differential with the
 * base point and the target point swapped.
 *
 * @tparam MatrixType A matrix class type providing a `vmult` method.
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
 * @brief Computes the orthographic retraction.
 *
 * Maps a tangent vector @p z at base point @p x onto the manifold using the
 * orthographic projection:
 * \f[
 * R_x(h z) = \sqrt{1 - \|h z\|_M^2} x + h z
 * \f]
 *
 * @note Requires that \f$ \|h z\|_M < 1 \f$.
 *
 * @tparam MatrixType A matrix class type providing a `vmult` method.
 * @param[in] M The mass matrix defining the metric.
 * @param[in] v The tangent vector.
 * @param[in,out] x On input, the base point. On output, the retracted point.
 * @param[in] factor Scaling factor \f$ h \f$.
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

/**
 * @brief Computes the inverse orthographic retraction.
 *
 * Lifts a point @p v from the manifold to the tangent space at @p x using
 * simple orthogonal projection. This is the inverse of the orthographic retraction.
 *
 * Operation:
 * \f[
 * v \leftarrow v - \langle x, v \rangle_M x
 * \f]
 *
 * @note This function modifies @p v in-place.
 *
 * @tparam MatrixType A matrix class type providing a `vmult` method.
 * @param[in] M The mass matrix defining the metric.
 * @param[in,out] v On input, the point on the manifold. On output, the tangent vector.
 * @param[in] x The base point on the manifold.
 */
template <typename MatrixType>
void retract_inv_by_ortho(const MatrixType& M, Vector<double>& v, const Vector<double>& x)
{
    Vector<double> Mv(v.size());
    M.vmult(Mv, v);

    const double xMv = x*Mv;
    v.add(-xMv, x);
}

/**
 * @brief Computes the exponential map retraction on the sphere.
 *
 * This function maps a tangent vector @p z at base point @p x back onto the manifold
 * along a geodesic.
 *
 * The formula corresponds to the Riemannian exponential map on the sphere:
 * \f[
 * \mathrm{Exp}_x(h z) = \cos(h \|z\|_M) x + \sin(h \|z\|_M) \frac{z}{\|z\|_M}
 * \f]
 *
 * @note This function modifies @p x in-place to store the result.
 *
 * @tparam MatrixType A matrix class type providing a `vmult` method.
 * @param[in] M The mass matrix defining the metric.
 * @param[in] v The tangent vector (direction).
 * @param[in,out] x On input, the base point. On output, the retracted point on the manifold.
 * @param[in] factor A scaling factor \f$ h \f$ applied to the tangent vector @p z. Defaults to 1.0.
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
 * @brief Computes the inverse exponential map (logarithmic map) on the sphere.
 *
 * This function lifts a point @p v from the manifold to the tangent space at @p x.
 * Specifically, it computes \f$ \log_x(v) \f$ where both @p x and @p v are on the sphere.
 *
 * The formula used is:
 * \f[
 * \log_x(v) = \frac{\arccos(\langle x, v \rangle_M)}{\sqrt{1 - \langle x, v \rangle_M^2}}
 * \Pi_x(v)
 * \f]
 * where the projection component is computed via orthogonalization.
 *
 * @note This function modifies @p v in-place to store the result.
 *
 * @tparam MatrixType A matrix class type providing a `vmult` method.
 * @param[in] M The mass matrix defining the metric.
 * @param[in,out] v On input, the point on the manifold. On output, the tangent vector.
 * @param[in] x The base point on the manifold.
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


template <typename MatrixType>
class UnitMassSphere : public ManifoldBase
{
public:
    explicit UnitMassSphere(const MatrixType& M) : M(M) {}

    /**
     * @brief Retracts a tangent vector back to the unit-mass manifold.
     * $$ R_x(z) = \frac{x + z}{\|x + z\|_M} $$
     */
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
