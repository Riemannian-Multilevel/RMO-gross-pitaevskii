#ifndef GPE_FUNCTIONS_H
#define GPE_FUNCTIONS_H
#ifndef M_NORM_RESIDUAL
#define M_NORM_RESIDUAL 1
#endif
#ifndef ZERO_ROUNDOFF
#define ZERO_ROUNDOFF 1e-15
#endif

#include "lac.h"
#include <deal.II/base/function.h>

namespace gpe
{

/**
 * @brief Functor computing the square of the Euclidean norm of a point.
 *
 * Computes \f$ f(p) = \sum_{d=0}^{dim-1} p_d^2 \f$.
 * Used primarily for initializing test cases or potentials.
 *
 * @tparam dim The spatial dimension of the point.
 */
template <int dim>
class Square
{
public:
    double operator()(const Point<dim>& p) const {
        typename Point<dim>::value_type out = 0.0;
        for (unsigned d = 0; d < dim; d++) {
            out += p[d]*p[d];
        }
        return out;
    }
};

// Functions for GPE minimization
namespace energy
{

/**
 * @brief Evaluates the energy functional for the Gross-Pitaevskii equation.
 *
 * Computes the energy value:
 * \f[
 * E(x) = \frac{1}{2} x^T A_0 x + \frac{1}{4} x^T M_{\phi\phi}(x) x
 * \f]
 *
 * @tparam MatrixType A matrix class type providing a `vmult` method.
 * @param[in] x The state vector (solution coefficients).
 * @param[in] A0 The linear part of the stiffness/Hamiltonian matrix.
 * @param[in] Mpp The nonlinear part of the matrix (\f$ M_{\phi\phi} \f$) evaluated at @p x.
 * @return The computed energy value.
 */
template <typename MatrixType>
double function_value(const Vector<double>& x, const MatrixType& A0, const MatrixType& Mpp)
{
    Vector<double> Bx(x.size());
    A0.vmult(Bx, x);
    Bx *= 0.5;

    Vector<double> Mpp_x(x.size());
    Mpp.vmult(Mpp_x, x);

    Bx.add(0.25, Mpp_x);
    return x * Bx;
}

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
                                const Vector<double>& x,
                                const MatrixType& M,
                                const Vector<double>& v,
                                Vector<double>& output)
{
    AssertDimension(x.size(), v.size());
    Vector<double> Mx(x.size());
    M.vmult(Mx, x);

    Vector<double> Ainv_Mx(x.size());
    A_inv.vmult(Ainv_Mx, Mx);

    Vector<double> My(x.size());
    M.vmult(My, Ainv_Mx);  // M A_x^{-1} M x

    double denom = x * My;
    AssertThrow(denom > 0, dealii::ExcInternalError("x' M A^{-1} M x <= 0"));

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
void project_onto_tangent_space(const InverseMatrixType& A_inv,
                                const Vector<double>& x,
                                const MatrixType& M,
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

// Riemannian gradient in S^{n-1} with energy metric
template <typename MatrixType, typename InverseMatrixType>
void gradient(const InverseMatrixType& A_inv, const MatrixType& M, const Vector<double>& x, Vector<double>& output)
{
    // \Pi_x(x): R^n -> T_x S^{n-1}
    project_onto_tangent_space(A_inv, x, M, output);
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
 * @param[in] z The tangent vector (update direction).
 * @param[in,out] x On input, the base point. On output, the retracted point.
 * @param[in] factor Scaling factor \f$ h \f$.
 */
template <typename MatrixType>
void retract_by_norm(const MatrixType& M, const Vector<double>& z, Vector<double>& x,
                     const double factor = 1.0)
{
    AssertThrow(factor != 0.0, dealii::ExcMessage("factor must be nonzero"));
    x.add(factor, z);           // x' <- x + h z

    Vector<double> Mx(x.size());
    M.vmult(Mx, x);
    x /= std::sqrt(x * Mx);     // x' <- x' / ||x'||_M
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

// TODO: verify implementation, write tests (cf. manopt)
/** Differentiated retraction
 * D_Ret_phi(v)[w] = (1 / ||phi+v||_M) * (I - ( (phi+v) v^T M ) / ||phi+v||_M^2 ) * w
 *
 * @tparam MatrixType
 * @param M Mass matrix
 * @param phi Base point
 * @param v Tangent vector (argument of the retraction)
 * @param w Direction of differentiation (argument of the differential)
 * @param dst Resulting vector (resized if necessary)
*/
template <typename MatrixType>
void retract_diff_by_norm(const MatrixType& M,
                          const Vector<double>& phi,
                          const Vector<double>& v,
                          const Vector<double>& w,
                          Vector<double>& dst)
{
    // 1. Compute y = phi + v
    Vector<double> y(phi);
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

// TODO: verify implementation, write tests (cf. manopt)
/** Differentiated inverse retraction
 * D_invRet_phi(zeta)[u] = (1 / (phi^T M zeta)) * ( I - ( zeta phi^T M ) / (phi^T M zeta) ) * u
 *
 * @tparam MatrixType
 * @param M Mass matrix
 * @param phi Base point
 * @param zeta Argument of inverse retraction
 * @param u Direction of differentiation
 * @param dst Resulting vector
*/
template <typename MatrixType>
void retract_inv_diff_by_norm(const MatrixType& M,
                              const Vector<double>& phi,
                              const Vector<double>& zeta,
                              const Vector<double>& u,
                              Vector<double>& dst)
{
    // 1. Compute Mzeta = M * zeta
    Vector<double> Mzeta(zeta.size());
    M.vmult(Mzeta, zeta);

    // 2. Compute beta = phi^T M zeta
    const double beta = phi * Mzeta;
    AssertThrow(std::abs(beta) > 0, dealii::ExcMessage("phi^T M zeta must be non-zero"));

    // 3. Compute Mu = M * u
    Vector<double> Mu(u.size());
    M.vmult(Mu, u);

    // 4. Compute gamma = phi^T M u
    const double gamma = phi * Mu;

    // 5. Compute dst = (1/beta) * (u - (gamma/beta) * zeta)
    dst = u;
    dst.add(-gamma / beta, zeta); // dst <- dst - (gamma/beta) * zeta
    dst /= beta;
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
 * @param[in] z The tangent vector.
 * @param[in,out] x On input, the base point. On output, the retracted point.
 * @param[in] factor Scaling factor \f$ h \f$.
 */
template <typename MatrixType>
void retract_by_ortho(const MatrixType& M, const Vector<double>& z,
                      Vector<double>& x, const double factor = 1.0)
{
    AssertThrow(std::abs(factor) > 0, dealii::ExcMessage("factor must be non-zero"));
    Vector<double> Mz(x.size());
    M.vmult(Mz, z);

    double zMz = z*Mz;
    zMz *= factor;
    zMz *= factor;  // (hz) * M(hZ) = h^2 zMz
    AssertThrow(zMz < 1.0, dealii::ExcInternalError("z'Mz required < 1"));

    x *= std::sqrt(1-zMz);
    x.add(factor, z);
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

    double xMv = x*Mv;
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
 * @param[in] z The tangent vector (direction).
 * @param[in,out] x On input, the base point. On output, the retracted point on the manifold.
 * @param[in] factor A scaling factor \f$ h \f$ applied to the tangent vector @p z. Defaults to 1.0.
 */
template <typename MatrixType>
void retract_by_exp(const MatrixType& M, const Vector<double>& z, Vector<double>& x,
                    const double factor = 1.0)
{
    AssertThrow(std::abs(factor) > 0, dealii::ExcMessage("factor must be non-zero"));
    Vector<double> Mz(x.size());
    M.vmult(Mz, z);

    double zMz = z*Mz;
    double z_Mnorm = std::sqrt(zMz);
    AssertThrow(z_Mnorm > 0.0, dealii::ExcInternalError("|z|_M must be positive"));

    // Derivation:
    //                  |hz|_M  =  h |z|_M
    //             hz / |hz|_M  =  z / |z|_M
    // sin(|hz|_M) hz / |hz|_M  =  sin(h|z|_M) z / |z|_M
    x *= std::cos(factor*z_Mnorm);
    x.add(std::sin(factor*z_Mnorm) / z_Mnorm, z);
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

    double xMv = x*Mv;
    v.add(-xMv, x);

    double nom   = std::acos(xMv);
    double denom = std::sin(nom);

    AssertThrow(std::abs(denom) > 0, dealii::ExcInternalError("sin(arccos(x' M v)) must be non-zero"));
    v *= (nom/denom);
}

/**
 * @brief Computes the coarse model function value using the Energy-weighted metric.
 *
 * Psi(zeta) = E(zeta) - <w, invRet_phi(zeta)>_{A_zeta}
 * = E(zeta) - w^T * A_zeta * invRet_phi(zeta)
 *
 * @param[in] zeta The coarse variable.
 * @param[in] phi The base point.
 * @param[in] w The restricted gradient.
 * @param[in] M The mass matrix (required for retraction).
 * @param[in] A0 The linear part of stiffness.
 * @param[in] Mpp The nonlinear part of stiffness.
 * @param[in] A_zeta The full operator A at zeta (used for the metric).
 * @return The scalar value.
 */
template <typename MatrixType, typename OperatorType>
double coarse_function_value(const Vector<double>& zeta,
                             const Vector<double>& phi,
                             const Vector<double>& w,
                             const MatrixType& M,
                             const MatrixType& A0,
                             const MatrixType& Mpp,
                             const OperatorType& A_zeta)
{
    // 1. Compute Energy E(zeta)
    const double energy = function_value(zeta, A0, Mpp);

    // 2. Compute Inverse Retraction: v = invRet_phi(zeta)
    // Note: Retraction is geometric, so it still uses Mass matrix M usually
    Vector<double> v(zeta);
    retract_inv_by_norm(M, v, phi);

    // 3. Compute Inner Product <w, v>_A = w^T * A_zeta * v
    Vector<double> Av(zeta.size());
    A_zeta.vmult(Av, v);

    const double correction_term = w * Av;

    // 4. Result
    return energy - correction_term;
}

// TODO: verify implementation, write tests (cf. manopt)
/**
 * Computes the coarse gradient update step:
 * v_unproj = zeta + A_zeta^-1 * ( (1/s)*w - (<Mw, Mphi>/s^2)*zeta )
 * dst = Proj_zeta( v_unproj )
 *
 * where s = <phi, M*zeta>
 *
 * @param M Mass matrix (M_coarse)
 * @param A_inv Linear operator or InverseMatrix wrapper representing A_zeta^-1
 * @param zeta The coarse approximation (y)
 * @param phi The restricted fine grid approximation (base point)
 * @param w The restricted residual/gradient
 * @param dst Output vector
 */
template <typename MatrixType, typename InverseOperatorType>
void coarse_gradient(const MatrixType& M,
                     const InverseOperatorType& A_inv,
                     const Vector<double>& zeta,
                     const Vector<double>& phi,
                     const Vector<double>& w,
                     Vector<double>& dst)
{
    const unsigned int n = zeta.size();

    // 1. Compute auxiliary vectors
    Vector<double> Mzeta(n);
    M.vmult(Mzeta, zeta);

    Vector<double> Mphi(n);
    M.vmult(Mphi, phi);

    Vector<double> Mw(n);
    M.vmult(Mw, w);

    // 2. Compute scalars
    // s = <phi, M*zeta> = phi * Mzeta
    const double s = phi * Mzeta;
    Assert(std::abs(s) > 0, dealii::ExcMessage("scalar s (phi^T M zeta) must be positive"));

    // K = <Mw, Mphi>
    const double K = Mw * Mphi;

    // 3. Construct RHS for the linear solve
    // rhs = (1/s)*w - (K/s^2)*zeta
    Vector<double> rhs(w);
    rhs *= (1.0 / s);
    rhs.add(-K / (s * s), zeta);

    // 4. Apply inverse operator: u = A^-1 * rhs
    Vector<double> u(n);
    A_inv.vmult(u, rhs);

    // 5. Construct unprojected update: v = zeta + u
    Vector<double> v(zeta);
    v += u;

    // 6. Project onto tangent space
    project_onto_tangent_space(A_inv, zeta, M, v, dst);
}

// Termination criteria for energy function minimization
// TODO: make this generic?
struct State
{
    double mass{0};
    double lambda{0};
    double residual{0};
};

// TODO: x*Mx is only for debugging/diagnostic purposes
template <typename MatrixType>
State residual(const Vector<double>& x,
                  const MatrixType& A0,
                  const MatrixType& Mpp,
                  const MatrixType& M, double beta)
{
    State prop;
    Vector<double> Mx(x.size());
    M.vmult(Mx, x);
    prop.mass = x * Mx;             // should be ~ 1 (energy constraint)

    Vector<double> Ax1(x.size()); // A0 x
    A0.vmult(Ax1, x);

    Vector<double> Ax2(x.size()); // Mpp x
    Mpp.vmult(Ax2, x);

    Ax1.add(beta, Ax2);             // (A0 + beta Mpp) x
    prop.lambda = x * Ax1;          // Rayleigh quotient (x'Ax / x'Mx)

    Vector<double> r(Ax1);
    r.add(-prop.lambda, Mx);        // r = A x - lambda M x

    // TODO: use enum for setting norm at runtime
    prop.residual = 0.0;
    if constexpr (M_NORM_RESIDUAL) {
        Vector<double> Mr(r.size());
        M.vmult(Mr, r);
        prop.residual = std::sqrt(r * Mr);
    }
    else {
        prop.residual = r.l2_norm();
    }
    return prop;
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
 * @return The scalar value of the coarse model.
 */
template <typename MatrixType>
double coarse_function_value(const Vector<double>& zeta,
                             const Vector<double>& phi,
                             const Vector<double>& w,
                             const MatrixType& M,
                             const MatrixType& A0,
                             const MatrixType& Mpp)
{
    // 1. Compute Energy E(zeta)
    // using the existing function_value helper
    const double energy = function_value(zeta, A0, Mpp);

    // 2. Compute Inverse Retraction: v = invRet_phi(zeta)
    // We use a temporary vector since invRet modifies the argument in-place
    Vector<double> v(zeta);
    retract_inv_by_norm(M, v, phi);

    // 3. Compute Inner Product <w, v>_M = w^T * M * v
    Vector<double> Mv(zeta.size());
    M.vmult(Mv, v);

    const double correction_term = w * Mv;
    return energy - correction_term;
}

/**
 * Computes the coarse gradient update step in the mass-weighted metric:
 *
 * grad_M = Proj_zeta( M^-1 * ( A*zeta - (K/s^2)*zeta ) + (1/s)*w )
 *
 * where:
 * s = <phi, M*zeta>
 * K = <Mw, M*phi>
 *
 * @param M Mass matrix
 * @param M_inv Operator representing M^-1 (must support vmult)
 * @param A Operator representing A_zeta (must support vmult)
 * @param zeta Coarse approximation (base point)
 * @param phi Fine grid restriction
 * @param w Restricted residual
 * @param dst Output vector
 */
template <typename MatrixType, typename InverseMatrixType>
void coarse_gradient(const MatrixType& M,
                     const InverseMatrixType& M_inv,
                     const MatrixType& A,
                     const Vector<double>& zeta,
                     const Vector<double>& phi,
                     const Vector<double>& w,
                     Vector<double>& dst)
{
    const unsigned int n = zeta.size();

    // 1. Compute M*zeta (needed for scalar 's')
    Vector<double> Mzeta(n);
    M.vmult(Mzeta, zeta);

    // 2. Compute scalar s = <phi, M*zeta>
    const double s = phi * Mzeta;
    Assert(std::abs(s) > 0, dealii::ExcMessage("scalar s must be positive"));

    // 3. Compute M*w and M*phi (needed for scalar 'K')
    Vector<double> Mw(n);
    M.vmult(Mw, w);

    Vector<double> Mphi(n);
    M.vmult(Mphi, phi);

    // 4. Compute scalar K = <Mw, M*phi>
    const double K = Mw * Mphi;

    // 5. Construct the RHS for the mass inverse application
    // Inside M^-1(...), we have: A*zeta - (K/s^2)*zeta
    Vector<double> Azeta(n);
    A.vmult(Azeta, zeta);

    Vector<double> rhs_M(Azeta);
    rhs_M.add(-K / (s * s), zeta);

    // 6. Apply inverse mass matrix: u = M^-1 * rhs_M
    Vector<double> term1(n);
    M_inv.vmult(term1, rhs_M);

    // 7. Add the w term: v_unproj = term1 + (1/s)*w
    Vector<double> v(term1);
    v.add(1.0 / s, w);

    // 8. Project onto tangent space using the provided mass-based projection
    project_onto_tangent_space(zeta, M, v, dst);
}

} // namespace mass

} // namespace gpe

#endif //GPE_FUNCTIONS_H