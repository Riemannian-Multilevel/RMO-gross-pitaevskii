#ifndef RMO_GPE_KERNELS_H
#define RMO_GPE_KERNELS_H

#include <rmo/lac.h>
#include <rmo/gpe/manifold.h>
#include <rmo/gpe/metric.h>

/**
 * @file
 * @brief Numerical kernels of the Gross-Pitaevskii oracles: Riemannian gradients on the unit-mass
 * manifold, and value, directional derivative and gradient of the coarse models.
 *
 * Shared by the oracles (oracle.h, oracle_coarse.h), the iterations (iteration.h) and the tests.
 */
namespace rmo::gpe::kernels
{

// -------------------------------------------------------------------------
// Riemannian gradients (fine level)
// -------------------------------------------------------------------------
/**
 * @brief Computes the Riemannian gradient for the Gross-Pitaevskii energy on the unit-mass manifold.
 *
 * This function calculates the gradient of the energy functional \f$E^{GP}(\phi)\f$
 * restricted to the sphere \f$S^{n-1}\f$ with an energy-adaptive metric \f$A_\phi\f$. The mathematical
 * formulation for the Riemannian gradient is:
 * \f[ \grad_{A} E^{GP}(\phi) = \phi-\frac{1}{\phi^\top MA_\phi^{-1} M\phi} A_\phi^{-1}M\phi \f]
 *
 * @tparam MatrixType A matrix-free operator or sparse matrix type providing a `vmult(dst, src)` method.
 * @tparam InverseMatrixType A solver wrapper or inverse operator type providing a `vmult(dst, src)` method.
 *
 * @param A_inv The inverse linear operator (\f$A_\phi^{-1}\f$).
 * @param M The mass matrix (\f$M\f$).
 * @param x The current state vector (\f$\phi\f$).
 * @param output The vector where the computed Riemannian gradient will be stored.
 */
template <typename MatrixType, typename InverseMatrixType>
void grad_energy_adaptive(const InverseMatrixType& A_inv, const MatrixType& M,
                          const Vector<double>& x, Vector<double>& output)
{
    // \Pi_x(x): R^n -> T_x S^{n-1}
    metric::energy::project_onto_tangent_space(A_inv, x, M, output);
}

/**
 * @brief Computes the Riemannian gradient for the Gross-Pitaevskii energy on the unit-mass manifold.
 *
 * This function calculates the gradient of the energy functional \f$E^{GP}(\phi)\f$
 * restricted to the sphere \f$S^{n-1}\f$ with a mass metric \f$M\f$. The mathematical
 * formulation for the Riemannian gradient is:
 * \f[ \nabla_M E^{GP}(\phi) = M^{-1}\big(A_\phi\,\phi - (\phi^\top A_\phi\,\phi)M\phi\big) \f]
 *
 * @tparam MatrixType A matrix-free operator or sparse matrix type providing a `vmult(dst, src)` method.
 * @tparam InverseMatrixType A solver wrapper or inverse operator type providing a `vmult(dst, src)` method.
 *
 * @param Minv The inverse mass operator (\f$M^{-1}\f$).
 * @param A The state-dependent total linear operator (\f$A_\phi\f$).
 * @param M The mass matrix (\f$M\f$).
 * @param x The current state vector (\f$\phi\f$).
 * @param output The vector where the computed Riemannian gradient will be stored.
 */
template <typename MatrixType, typename InverseMatrixType>
void grad_mass(const InverseMatrixType& Minv, const MatrixType& A, const MatrixType& M,
               const Vector<double>& x, Vector<double>& output)
{
    Vector<double> Ax(x.size());
    A.vmult(Ax, x);

    Vector<double> Mx(x.size());
    M.vmult(Mx, x);

    Ax.add(-(x * Ax), Mx);
    Minv.vmult(output, Ax);
}


/**
 * @brief Computes the Riemannian gradient in the F-metric.
 * \f[ \grad_{\rm F} E^{\rm GP}(\phi) = A_{\phi}\phi - \frac{\phi^\top M A_{\phi}\phi}{\phi^\top M^2 \phi} M \phi \f]
*/
template <typename MatrixType>
void grad_frobenius(const MatrixType& A, const MatrixType& M,
                    const Vector<double>& x, Vector<double>& output)
{
    const unsigned int n_dofs = x.size();

    Vector<double> Ax(n_dofs);
    A.vmult(Ax, x);

    Vector<double> Mx(n_dofs);
    M.vmult(Mx, x);

    const double Mx_sq = Mx * Mx; // x^T M^2 x
    const double num = Mx * Ax; // x^T M A x

    output = Ax;
    output.add(-num / Mx_sq, Mx);
}


// -------------------------------------------------------------------------
// Coarse models
// -------------------------------------------------------------------------
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
 * @param[in] energy The energy \f$ E(\zeta) \f$, evaluated by the caller.
 * @return The scalar value of the coarse model.
 */
template <typename MatrixType>
double coarse_mass_value(const Vector<double>& zeta,
                         const Vector<double>& phi,
                         const Vector<double>& w,
                         const MatrixType& M,
                         const double energy)
{
    // We use a temporary vector since invRet modifies the argument in-place
    Vector<double> v(zeta);
    ellipsoid::retract_inv_by_norm(M, v, phi);

    Vector<double> Mv(zeta.size());
    M.vmult(Mv, v);

    const double correction_term = w * Mv;
    return energy - correction_term;
}

// Function value of the Frobenius coarse model
template <typename MatrixType>
double coarse_frobenius_value(const Vector<double>& zeta, const Vector<double>& phi,
                              const Vector<double>& w,
                              const MatrixType& M,
                              const double energy)
{
    // Compute the inverse retraction: invRet_phi(zeta)
    Vector<double> inv_ret(zeta);
    ellipsoid::retract_inv_by_norm(M, inv_ret, phi);

    // Subtract the linear tilt: <w, invRet_phi(zeta)>_F
    const double correction_term = w * inv_ret;
    return energy - correction_term;
}

template <typename MatrixTypeA, typename MatrixTypeM>
double coarse_mass_dir_deriv(const Vector<double>& zeta,
                             const Vector<double>& phi,
                             const Vector<double>& w,
                             const Vector<double>& z,
                             const MatrixTypeM& M,
                             const MatrixTypeA& A)
{
    // Differential of the standard GP energy: (A * zeta)^T z
    Vector<double> Az(zeta.size());
    A.vmult(Az, zeta);

    // Adjoint pullback of the correction vector w: u = (D_invRet_phi)^*[w]
    Vector<double> u(zeta.size());
    ellipsoid::retract_inv_diff_by_norm_adjoint(M, phi, zeta, w, u);

    Vector<double> Mu(zeta.size());
    M.vmult(Mu, u);

    Vector<double> grad(Az);
    grad.add(-1.0, Mu);

    return grad * z;
}

template <typename MatrixTypeA, typename MatrixTypeM>
double coarse_frobenius_dir_deriv(const Vector<double>& zeta,
                                  const Vector<double>& phi,
                                  const Vector<double>& w,
                                  const Vector<double>& z,
                                  const MatrixTypeM& M,
                                  const MatrixTypeA& A)
{
    const unsigned int n_dofs = zeta.size();

    // Differential of the GP energy: (A * zeta)^T z
    Vector<double> Az(n_dofs);
    A.vmult(Az, zeta);

    Vector<double> M_zeta(n_dofs);
    M.vmult(M_zeta, zeta);
    const double phi_M_zeta = phi * M_zeta;
    const double w_zeta = w * zeta;

    // Differential of the Frobenius tilt
    // D T(zeta)[z] = (w^T z) / (phi^T M zeta) - (w^T zeta * phi^T M z) / (phi^T M zeta)^2
    Vector<double> M_phi(n_dofs);
    M.vmult(M_phi, phi);

    Vector<double> grad(Az);
    grad.add(-1.0 / phi_M_zeta, w);
    grad.add(w_zeta / (phi_M_zeta * phi_M_zeta), M_phi);

    // Natural pairing with the tangent vector z
    return grad * z;
}

/**
 * Computes the coarse gradient update step in the mass-weighted metric.
 * @param M Mass matrix
 * @param M_inv Operator representing M^-1 (must support vmult)
 * @param A Operator representing A_zeta (must support vmult)
 * @param zeta Coarse variable (argument of the function)
 * @param phi Fine grid restriction (base point)
 * @param w Restricted residual
 * @param dst Output vector
 */
// TODO: output directional derivative and riemannian gradient separately ("metric-free" line search)
template <typename MatrixType, typename InverseMatrixType>
void coarse_mass_grad(const MatrixType& M,
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
    Az.add(-(zeta * Az), Mz);
    M_inv.vmult(dst, Az);

    Vector<double> invRet(zeta.size());
    ellipsoid::retract_inv_diff_by_norm_adjoint(M, phi, zeta, w, invRet);

    dst.add(-1.0, invRet);
}

/**
 * Computes the (M-)coarse gradient update step in the energy metric.
 *
 * @param M Mass matrix (M_coarse)
 * @param A_inv Linear operator or InverseMatrix wrapper representing A_zeta^-1
 * @param zeta The coarse approximation (y)
 * @param phi Fine grid restriction (base point)
 * @param w The restricted residual/gradient
 * @param dst Output vector
 */
// TODO: output directional derivative and riemannian gradient separately ("metric-free" line search)
//       tag- or class-based metric selection
template <typename MatrixType, typename InverseMatrixType>
void coarse_mass_grad_energy_adaptive(const MatrixType& M, const InverseMatrixType& A_inv,
                                      const Vector<double>& zeta,
                                      const Vector<double>& phi,
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

    metric::energy::project_onto_tangent_space(A_inv, zeta, M, invAz, dst);
}

// Frobenius gradient of the Frobenius coarse model
// TODO: output directional derivative and riemannian gradient separately ("metric-free" line search)
template <typename MatrixType>
void coarse_frobenius_grad(const MatrixType& M, const MatrixType& A,
                           const Vector<double>& zeta, const Vector<double>& phi,
                           const Vector<double>& w,
                           Vector<double>& dst)
{
    // phi represents \psi_k, zeta represents \zeta

    // 1. Compute M_H * \zeta
    Vector<double> Mzeta(zeta.size());
    M.vmult(Mzeta, zeta);

    // 2. Compute \beta = \psi_k^\top M_H \zeta
    const double beta = phi * Mzeta;
    AssertThrow(std::abs(beta) > ZERO_ROUNDOFF, dealii::ExcMessage("phi^T M zeta must be non-zero"));

    // 3. Compute M_H * \psi_k
    Vector<double> Mphi(zeta.size());
    M.vmult(Mphi, phi);

    // 4. Compute scalar \zeta^\top w_{k,e}
    const double zeta_w = zeta * w;

    // 5. Construct the inner vector: w_{k,e} - (\zeta^\top w_{k,e} / \beta) M_H \psi_k
    Vector<double> inner(w);
    inner.add(-zeta_w / beta, Mphi);

    // 6. Compute A_\zeta \zeta
    Vector<double> Azeta(zeta.size());
    A.vmult(Azeta, zeta);

    // 7. Assemble the pre-projection vector: u = A_\zeta \zeta - (1 / \beta) * inner
    Vector<double> u(Azeta);
    u.add(-1.0 / beta, inner);

    // 8. Project onto the tangent space using the Frobenius metric
    metric::frobenius::project_onto_tangent_space(zeta, M, u, dst);
}

// Energy-adaptive gradient of the Frobenius coarse model
template <typename MatrixType, typename InverseMatrixType>
void coarse_frobenius_grad_energy_adaptive(const MatrixType& M,
                                           const InverseMatrixType& A_inv,
                                           const MatrixType& A,
                                           const Vector<double>& zeta,
                                           const Vector<double>& phi,
                                           const Vector<double>& w,
                                           Vector<double>& dst)
{
    // 1. Compute the coarse gradient in the Frobenius metric: \grad_{\rm e} q_k^{\rm GP}(\zeta)
    Vector<double> grad_e(zeta.size());
    coarse_frobenius_grad(M, A, zeta, phi, w, grad_e);

    // 2. Apply the inverse operator: v = A_\zeta^{-1}(\grad_{\rm e} q_k^{\rm GP}(\zeta))
    Vector<double> v(zeta.size());
    A_inv.vmult(v, grad_e);

    // 3. Compute M_H \zeta
    Vector<double> Mzeta(zeta.size());
    M.vmult(Mzeta, zeta);

    // 4. Compute A_\zeta^{-1} M_H \zeta
    Vector<double> Ainv_Mzeta(zeta.size());
    A_inv.vmult(Ainv_Mzeta, Mzeta);

    // 5. Compute Numerator: \zeta^\top M_H A_\zeta^{-1}(\grad_{\rm e} q_k^{\rm GP}(\zeta))
    // By grouping, this is equivalent to the dot product of (M_H \zeta) and v
    const double num = Mzeta * v;

    // 6. Compute Denominator: \zeta^\top M_H A_\zeta^{-1} M_H \zeta
    // By grouping, this is equivalent to the dot product of (M_H \zeta) and (A_\zeta^{-1} M_H \zeta)
    const double denom = Mzeta * Ainv_Mzeta;
    AssertThrow(denom > ZERO_ROUNDOFF, dealii::ExcInternalError("zeta^T M_H A^{-1} M_H zeta <= 0"));

    // 7. Assemble the final formulation
    dst = v;
    dst.add(-num / denom, Ainv_Mzeta);
}

} // namespace rmo::gpe::kernels

#endif //RMO_GPE_KERNELS_H
