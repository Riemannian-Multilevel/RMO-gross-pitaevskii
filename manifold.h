#ifndef GPE_FUNCTIONS_H
#define GPE_FUNCTIONS_H
#ifndef ZERO_ROUNDOFF
#define ZERO_ROUNDOFF 1e-15
#endif

#include "lac.h"
#include <deal.II/base/function.h>

namespace gpe
{

// TODO
struct TangentVector
{
    Vector<double> data;
};

// TODO
struct ManifoldPoint
{
    Vector<double> data;
};

namespace iteration
{
// Termination criteria for energy function minimization
// TODO: make this generic?
struct State
{
    double energy{0};
    double mass{0};
    double lambda{0};
    double residual{0};
};


// TODO: x*Mx is only for debugging/diagnostic purposes
template <typename MatrixType>
State residual(const Vector<double>& x, const MatrixType& A, const MatrixType& M,
               bool use_m_norm = true)
{
    State prop;
    Vector<double> Mx(x.size());
    M.vmult(Mx, x);
    prop.mass = x * Mx;             // should be ~ 1 (energy constraint)

    Vector<double> Ax(x.size()); // A x
    A.vmult(Ax, x);
    prop.lambda = x * Ax;          // Rayleigh quotient (x'Ax / x'Mx)

    Vector<double> r(Ax);
    r.add(-prop.lambda, Mx);        // r = A x - lambda M x

    // TODO: use enum for setting norm at runtime
    prop.residual = 0.0;
    if (use_m_norm) {
        Vector<double> Mr(r.size());
        M.vmult(Mr, r);
        prop.residual = std::sqrt(r * Mr);
    }
    else {
        prop.residual = r.l2_norm();
    }
    return prop;
}

} // namespace iteration


// Functions for GPE minimization
// Note: due to compartmentalizing as functions, some values are necessarily computed anew.
namespace ellipsoid
{

// Directional derivative of the GP functional
template <typename MatrixType>
double directional_derivative(const Vector<double>& x, const Vector<double>& z, const MatrixType& A)
{
    Vector<double> Ax(x.size());
    A.vmult(Ax, x);
    return Ax * z;
}

/**
 * @brief Evaluates the energy functional for the Gross-Pitaevskii equation.
 *
 * Computes the energy value:
 * \f[
 * E(x) = \frac{1}{2} x^T A_0 x + \frac{beta}{4} x^T M_{\phi\phi}(x) x
 * \f]
 *
 * @tparam MatrixType A matrix class type providing a `vmult` method.
 * @param[in] x The state vector (solution coefficients).
 * @param[in] A0 The linear part of the stiffness/Hamiltonian matrix.
 * @param[in] Mpp The nonlinear part of the matrix (\f$ M_{\phi\phi} \f$) evaluated at @p x.
 * @return The computed energy value.
 */
// TODO: different functions can be defined on the ellipsoid, move to separate module
template <typename MatrixType>
double function_value(const Vector<double>& x, const MatrixType& A0, const MatrixType& Mpp, double beta)
{
    Vector<double> Bx(x.size());
    A0.vmult(Bx, x);
    Bx *= 0.5;

    Vector<double> Mpp_x(x.size());
    Mpp.vmult(Mpp_x, x);

    Bx.add(0.25*beta, Mpp_x);
    return x * Bx;
}


namespace energy
{
/**
 * @brief Projects a vector @p v onto the tangent space at @p x using an energy-based metric.
 *
 * This computes the projection orthogonal to the gradient of the energy functional.
 *
 * Formula:
 * \f[
 * \Pi_x(v) = v - \frac{\langle x, v \rangle_M}{\langle x, M A^{-1} M x \rangle} A^{-1} M x
 * \f]
 *
 * @tparam MatrixType A matrix class type providing a `vmult` method.
 * @tparam InverseMatrixType A solver or operator class representing \f$ A^{-1} \f$.
 * @param[in] A_inv The inverse operator \f$ A^{-1} \f$.
 * @param[in] x The base point.
 * @param[in] M The mass matrix.
 * @param[in] v The vector to be projected.
 * @param[out] output The resulting projected vector.
 */
template <typename MatrixType, typename InverseMatrixType>
void project_onto_tangent_space(const InverseMatrixType& A_inv,
                                const Vector<double>& x, const MatrixType& M, const Vector<double>& v,
                                Vector<double>& output,
                                bool ignore_positivity_constraint = false)
{
    AssertDimension(x.size(), v.size());
    Vector<double> Mx(x.size());
    M.vmult(Mx, x);

    Vector<double> Ainv_Mx(x.size());
    A_inv.vmult(Ainv_Mx, Mx);

    Vector<double> My(x.size());
    M.vmult(My, Ainv_Mx);  // M A_x^{-1} M x

    double denom = x * My;
    // Requirement for M, A positive definite on the tangent space of x
    if (!ignore_positivity_constraint) {
        AssertThrow(denom > 0, dealii::ExcInternalError("x' M A^{-1} M x <= 0"));
    }

    Vector<double> Mv(v.size());
    M.vmult(Mv, v);
    const double nom = x*Mv;

    output = v;
    output.add(-nom / denom, Ainv_Mx);
}


/**
 * @brief Projects the base point @p x onto its own tangent space using an energy-based metric.
 *
 * This overload handles the specific case where the vector to be projected is the base point
 * itself (i.e., \f$ v = x \f$).
 *
 * It computes the projection \f$ \Pi_x(x) \f$ relative to the metric induced by
 * the operator \f$ A^{-1} \f$ (specifically \f$ x^T M A^{-1} M x \f$).
 *
 * Formula:
 * \f[
 * \mathrm{output} = x - \frac{1}{\langle x, M A^{-1} M x \rangle} A^{-1} M x
 * \f]
 *
 * @tparam MatrixType A matrix class type providing a `vmult` method.
 * @tparam InverseMatrixType A solver or operator class representing \f$ A^{-1} \f$.
 * @param[in] A_inv The inverse operator \f$ A^{-1} \f$.
 * @param[in] x The base point.
 * @param[in] M The mass matrix.
 * @param[out] output The resulting projected vector.
 */
template <typename MatrixType, typename InverseMatrixType>
void project_onto_tangent_space(const InverseMatrixType& A_inv, const Vector<double>& x, const MatrixType& M,
                                Vector<double>& output)
{
    Vector<double> Mx(x.size());
    M.vmult(Mx, x);

    Vector<double> Ainv_Mx(x.size());
    A_inv.vmult(Ainv_Mx, Mx);

    Vector<double> My(x.size());
    M.vmult(My, Ainv_Mx);  // M A_x^{-1} M x

    const double denom = x * My;
    AssertThrow(denom > 0, dealii::ExcInternalError("x' M A^{-1} M x <= 0"));

    output = x;
    output.add(-1.0/denom, Ainv_Mx);
}


/**
 * @brief Computes the Riemannian gradient for the Gross-Pitaevskii energy on the unit-mass manifold.
 *
 * This function calculates the gradient of the energy functional $E^{GP}(\phi)$
 * restricted to the sphere $S^{n-1}$ with an energy-adaptive metric $A_\phi$. The mathematical
 * formulation for the Riemannian gradient is:
 * $$ \grad_{A} E^{GP}(\phi) = \phi-\frac{1}{\phi^\top MA_\phi^{-1} M\phi} A_\phi^{-1}M\phi $$
 *
 * @tparam MatrixType A matrix-free operator or sparse matrix type providing a `vmult(dst, src)` method.
 * @tparam InverseMatrixType A solver wrapper or inverse operator type providing a `vmult(dst, src)` method.
 *
 * @param A_inv The inverse linear operator ($A_\phi^{-1}$).
 * @param M The mass matrix ($M$).
 * @param x The current state vector ($\phi$).
 * @param output The vector where the computed Riemannian gradient will be stored.
 */
template <typename MatrixType, typename InverseMatrixType>
void gradient(const InverseMatrixType& A_inv, const MatrixType& M,
              const Vector<double>& x, Vector<double>& output)
{
    // \Pi_x(x): R^n -> T_x S^{n-1}
    project_onto_tangent_space(A_inv, x, M, output);
}

} // namespace energy


namespace mass
{

/**
 * @brief Projects a vector @p v onto the tangent space at @p x with respect to the
 * mass-weighted inner product.
 *
 * This function computes the projection
 * \f[
 * \Pi_x(v) = v - \frac{\langle x, v \rangle_M}{\|x\|_M^2} x,
 * \f]
 * assuming the manifold is the sphere defined by the mass matrix @p M. If @p x is
 * already normalized with respect to @p M (i.e., \f$\|x\|_M = 1\f$), this simplifies
 * to \f$ \Pi_x(v) = v - (x^T M v) x \f$.
 *
 * @tparam MatrixType A matrix class type providing a `vmult` method (e.g., SparseMatrix).
 * @param[in] x The base point on the manifold (assumed to be normalized in the M-metric).
 * @param[in] M The mass matrix defining the inner product \f$ \langle u, w \rangle_M = u^T M w \f$.
 * @param[in] v The vector to be projected.
 * @param[out] output The resulting projected vector in the tangent space \f$ T_x \mathcal{M} \f$.
 */
template <typename MatrixType>
void project_onto_tangent_space(const Vector<double>& x, const MatrixType& M, const Vector<double>& v,
                                Vector<double>& output)
{
    Vector<double> Mv(x.size());
    M.vmult(Mv, v);
    const double xMv = x*Mv;

    output = v;
    output.add(-xMv, x);
}


/**
 * @brief Computes the Riemannian gradient for the Gross-Pitaevskii energy on the unit-mass manifold.
 *
 * This function calculates the gradient of the energy functional $E^{GP}(\phi)$
 * restricted to the sphere $S^{n-1}$ with a mass metric $M$. The mathematical
 * formulation for the Riemannian gradient is:
 * $$ \nabla_M E^{GP}(\phi) = M^{-1}\big(A_\phi\,\phi - (\phi^\top A_\phi\,\phi)M\phi\big) $$
 *
 * @tparam MatrixType A matrix-free operator or sparse matrix type providing a `vmult(dst, src)` method.
 * @tparam InverseMatrixType A solver wrapper or inverse operator type providing a `vmult(dst, src)` method.
 *
 * @param Minv The inverse mass operator ($M^{-1}$).
 * @param A The state-dependent total linear operator ($A_\phi$).
 * @param M The mass matrix ($M$).
 * @param x The current state vector ($\phi$).
 * @param output The vector where the computed Riemannian gradient will be stored.
 */
template <typename MatrixType, typename InverseMatrixType>
void gradient(const InverseMatrixType& Minv, const MatrixType& A, const MatrixType& M,
              const Vector<double>& x, Vector<double>& output)
{
    Vector<double> Ax(x.size());
    A.vmult(Ax, x);

    Vector<double> Mx(x.size());
    M.vmult(Mx, x);

    Ax.add(-(x*Ax), Mx);
    Minv.vmult(output, Ax);
}

} // namespace mass


namespace frobenius
{

template <typename MatrixType>
void project_onto_tangent_space(const Vector<double>& x, const MatrixType& M, const Vector<double>& v,
                                Vector<double>& output)
{
    AssertDimension(x.size(), v.size());

    // Compute M * phi
    Vector<double> Mx(x.size());
    M.vmult(Mx, x);

    // Denominator: phi^T M^2 phi = (M * phi)^T (M * phi) by symmetry of M
    const double denom = Mx * Mx;

    AssertThrow(denom > 0.0, dealii::ExcInternalError("x' M^2 x <= 0"));

    // Numerator: phi^T M xi = (M * phi)^T xi by symmetry of M
    const double nom = Mx * v;

    // Output: xi - (nom / denom) * (M * phi)
    output = v;
    output.add(-nom / denom, Mx);
}


/**
 * @brief Computes the Riemannian gradient in the F-metric.
 * \grad_{\rm F} E^{\rm GP}(\phi) = A_{\phi}\phi - \frac{\phi^\top M A_{\phi}\phi}{\phi^\top M^2 \phi} M \phi
*/
template <typename MatrixType>
void gradient(const MatrixType& A, const MatrixType& M,
              const Vector<double>& x, Vector<double>& output)
{

    const unsigned int n_dofs = x.size();

    Vector<double> Ax(n_dofs);
    A.vmult(Ax, x);

    Vector<double> Mx(n_dofs);
    M.vmult(Mx, x);

    const double Mx_sq = Mx * Mx;     // x^T M^2 x
    const double num   = Mx * Ax;     // x^T M A x

    output = Ax;
    output.add(-num / Mx_sq, Mx);
}

} // namespace frobenius


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
    AssertThrow(xMv > 0, dealii::ExcInternalError("x'Mv must be nonzero"));

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
template <typename MatrixType>
void retract_inv_diff_by_norm_adjoint()
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

    // Clamp to [-1.0, 1.0] to prevent NaN in acos due to floating-point drift
    double xMv = std::clamp(x * Mv, -1.0, 1.0);

    v.add(-xMv, x); // v <- Pi_x(v)

    // Handle the limit case where x and v are identical (xMv -> 1.0)
    // The limit of arccos(z)/sqrt(1-z^2) as z->1 is 1.0.
    if (std::abs(1.0 - xMv) < 1e-14) {
        // v is already Pi_x(v), and the scalar multiplier is 1.0.
        return;
    }
    const double nom   = std::acos(xMv);
    const double denom = std::sin(nom); // sin(arccos(x)) = sqrt(1-x^2)

    v *= (nom / denom);
}

} // namespace energy


namespace coarse
{
double first_order_coherence()  // TODO
{
    throw dealii::ExcNotImplemented(__PRETTY_FUNCTION__);
}

} // namespace coarse

// TODO: the coarse model is DEFINED in the M-metric (-> metric independence for certain vector transports.)
//       it can be SOLVED in either the M- or the A-metric (as implemented by gradient() overloads.)
namespace coarse::mass
{

/**
 * @brief Computes the coarse model function value using the mass-weighted metric.
 *
 * Psi(zeta) = E(zeta) - <w, invRet_phi(zeta)>_M
 * = E(zeta) - w^T * M * invRet_phi(zeta)
 *
 * @param[in] zeta The coarse variable (argument of the function).
 * @param[in] phi The base point (fine grid restriction).
 * @param[in] w The restricted gradient/residual.
 * @param[in] M The mass matrix (coarse level).
 * @param[in] A0 The linear part of the stiffness matrix (coarse level).
 * @param[in] Mpp The nonlinear part of the matrix evaluated at zeta.
 * @param beta
 * @return The scalar value of the coarse model.
 */
template <typename MatrixType>
double function_value(const Vector<double>& zeta,
                      const Vector<double>& phi,
                      const Vector<double>& w,
                      const MatrixType& M,
                      const MatrixType& A0,
                      const MatrixType& Mpp, double beta)
{
    // Using the existing function_value helper
    const double energy = ellipsoid::function_value(zeta, A0, Mpp, beta);

    // We use a temporary vector since invRet modifies the argument in-place
    Vector<double> v(zeta);
    ellipsoid::retract_inv_by_norm(M, v, phi);

    Vector<double> Mv(zeta.size());
    M.vmult(Mv, v);

    const double correction_term = w * Mv;
    return energy - correction_term;
}

template <typename MatrixTypeA, typename MatrixTypeM>
double directional_derivative(const dealii::Vector<double>& zeta,
                              const dealii::Vector<double>& phi,
                              const dealii::Vector<double>& w,
                              const dealii::Vector<double>& z,
                              const MatrixTypeM& M,
                              const MatrixTypeA& A)
{
    // Differential of the standard GP energy: (A * zeta)^T z
    dealii::Vector<double> Az(zeta.size());
    A.vmult(Az, zeta);

    // Adjoint pullback of the correction vector w: u = (D_invRet_phi)^*[w]
    dealii::Vector<double> u(zeta.size());
    ellipsoid::retract_inv_diff_by_norm_adjoint(M, phi, zeta, w, u);

    dealii::Vector<double> Mu(zeta.size());
    M.vmult(Mu, u);

    dealii::Vector<double> dual(Az);
    dual.add(-1.0, Mu);

    return dual * z;
}


/**
 * Computes the coarse gradient update step in the mass-weighted metric.
 * @param M Mass matrix
 * @param M_inv Operator representing M^-1 (must support vmult)
 * @param A Operator representing A_zeta (must support vmult)
 * @param zeta Coarse approximation (base point)
 * @param phi Fine grid restriction
 * @param w Restricted residual
 * @param dst Output vector
 */
// TODO: output directional derivative and riemannian gradient separately ("metric-free" line search)
template <typename MatrixType, typename InverseMatrixType>
void gradient(const MatrixType& M,
              const InverseMatrixType& M_inv,
              const MatrixType& A,
              const Vector<double>& zeta,
              const Vector<double>& phi,
              const Vector<double>& w,
              Vector<double>& dst)
{
    Vector<double> Mz(zeta.size());
    M.vmult(Mz, zeta);

    Vector<double> Az(zeta.size());
    A.vmult(Az, zeta);
    Az.add(-(zeta*Az), Mz);
    M_inv.vmult(dst, Az);

    Vector<double> invRet(zeta.size());
    ellipsoid::retract_inv_diff_by_norm_adjoint(M, phi, zeta, w, invRet);

    dst.add(-1.0, invRet);
}


/**
 * Computes the coarse gradient update step in the energy metric.
 * @param M Mass matrix (M_coarse)
 * @param A_inv Linear operator or InverseMatrix wrapper representing A_zeta^-1
 * @param zeta The coarse approximation (y)
 * @param phi The restricted fine grid approximation (base point)
 * @param w The restricted residual/gradient
 * @param dst Output vector
 */
// TODO: output directional derivative and riemannian gradient separately ("metric-free" line search)
//       tag- or class-based metric selection
template <typename MatrixType, typename InverseMatrixType>
void energy_adaptive_gradient(const MatrixType& M, const InverseMatrixType& A_inv,
                              const Vector<double>& zeta, const Vector<double>& phi,
                              const Vector<double>& w,
                              Vector<double>& dst)
{
    Vector<double> invRet(zeta.size());
    ellipsoid::retract_inv_diff_by_norm_adjoint(M, phi, zeta, w, invRet);
    M.vmult(dst, invRet);

    Vector<double> invAz(zeta.size());
    A_inv.vmult(invAz, dst);
    invAz *= -1.0;
    invAz.add(1.0, zeta);

    // TODO: does <grad_A f(x), v>_A = Df(x)[v] hold after F-projection?
    ellipsoid::frobenius::project_onto_tangent_space(zeta, M, invAz, dst);
}

} // namespace coarse::mass


namespace coarse::energy
{

template <typename MatrixType>
double function_value()
{
    throw dealii::ExcNotImplemented();
}

template <typename MatrixType, typename InverseMatrixType>
void gradient()
{
    throw dealii::ExcNotImplemented();
}

} // namespace coarse::energy


namespace coarse::frobenius
{

// Function value of the Frobenius coarse model
template <typename MatrixType>
double function_value(const Vector<double>& zeta, const Vector<double>& phi,
                      const Vector<double>& w,
                      const MatrixType& M, const MatrixType& A0, const MatrixType& Mpp,
                      double beta)
{
    // 1. Compute the standard Gross-Pitaevskii energy on the coarse grid: E_H(zeta)
    double energy = ellipsoid::function_value(zeta, A0, Mpp, beta);

    // 2. Compute the inverse retraction: invRet_phi(zeta)
    Vector<double> inv_ret(zeta);
    ellipsoid::retract_inv_by_norm(M, inv_ret, phi);

    // 3. Subtract the linear tilt: <w, invRet_phi(zeta)>_F
    const double tilt = w * inv_ret;

    return energy - tilt;
}


template <typename MatrixTypeA, typename MatrixTypeM>
double directional_derivative(const dealii::Vector<double>& zeta,
                              const dealii::Vector<double>& phi,
                              const dealii::Vector<double>& w,
                              const dealii::Vector<double>& z,
                              const MatrixTypeM& M,
                              const MatrixTypeA& A)
{
    const unsigned int n_dofs = zeta.size();

    // Differential of the standard GP energy: (A * zeta)^T z
    dealii::Vector<double> Az(n_dofs);
    A.vmult(Az, zeta);

    dealii::Vector<double> M_zeta(n_dofs);
    M.vmult(M_zeta, zeta);
    const double phi_M_zeta = phi * M_zeta;
    const double w_zeta     = w * zeta;

    // Differential of the F-tilt
    // D T(zeta)[z] = (w^T z) / (phi^T M zeta) - (w^T zeta * phi^T M z) / (phi^T M zeta)^2
    dealii::Vector<double> M_phi(n_dofs);
    M.vmult(M_phi, phi);

    dealii::Vector<double> dual(Az);
    dual.add(-1.0 / phi_M_zeta, w);
    dual.add(w_zeta / (phi_M_zeta * phi_M_zeta), M_phi);

    // Natural pairing with the tangent vector z
    return dual * z;
}


// Frobenius gradient of the Frobenius coarse model
// TODO: output directional derivative and riemannian gradient separately ("metric-free" line search)
template <typename MatrixType>
void gradient(const MatrixType& M, const MatrixType& A,
              const Vector<double>& zeta, const Vector<double>& phi,
              const Vector<double>& w,
              Vector<double>& dst)
{
    const unsigned int n_dofs = zeta.size();

    // 1. Compute phi^T M zeta for the tilt scaling
    Vector<double> M_zeta(n_dofs);
    M.vmult(M_zeta, zeta);
    const double phi_M_zeta = phi * M_zeta;

    // 2. Compute the unprojected vector: v = A * zeta - (1 / phi^T M zeta) * w
    Vector<double> v(n_dofs);
    A.vmult(v, zeta);
    v.add(-1.0 / phi_M_zeta, w);

    // 3. Apply the F-orthogonal projection reusing the geometry namespace
    ellipsoid::frobenius::project_onto_tangent_space(zeta, M, v, dst);
}


// Energy-adaptive gradient of the Frobenius coarse model
// TODO: output directional derivative and riemannian gradient separately ("metric-free" line search)
//       tag- or class-based metric selection
template <typename MatrixType, typename InverseMatrixType>
void energy_adaptive_gradient(const MatrixType& M,
                              const InverseMatrixType& A_inv,
                              const MatrixType& A,
                              const Vector<double>& zeta,
                              const Vector<double>& phi,
                              const Vector<double>& w,
                              Vector<double>& dst)
{
    const unsigned int n_dofs = zeta.size();

    // 1. Compute the standard F-gradient
    Vector<double> grad_F(zeta.size());
    gradient(M, A, zeta, phi, w, grad_F);

    // 2. Apply the restricted Hamiltonian inverse: grad_A = A_inv * grad_F
    Vector<double> v(n_dofs);
    A_inv.vmult(v, grad_F);

    // Note: If A_inv strictly represents \tilde{A}^{-1} (projected),
    // the output should naturally lie in the tangent space.
    ellipsoid::frobenius::project_onto_tangent_space(zeta, M, v, dst);
}

} // namespace coarse::frobenius


namespace box
{

} // namespace box

} // namespace gpe

#endif //GPE_FUNCTIONS_H