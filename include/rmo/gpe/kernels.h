#ifndef RMO_GPE_KERNELS_H
#define RMO_GPE_KERNELS_H

#include <rmo/lac.h>
#include <rmo/gpe/manifold.h>
#include <rmo/gpe/metric.h>

/**
 * @file
 * @brief Numerical kernels of the Gross-Pitaevskii oracles: Riemannian gradients on the unit-mass sphere, and value,
 * directional derivative and gradient of the coarse models. The eigenvalue residuals are in residual.h.
 *
 * Arguments: the state `x` (coarse models: the variable `zeta`), the mass matrix `M`, the operator
 * \f$ A = A(x) \f$ and the inverse operators of \f$ M \f$ and \f$ A \f$ (any types with `vmult`), and the result.
 *
 * Coarse models: \f$ \Psi(\zeta) = E(\zeta) - \langle w, R_\phi^{-1}(\zeta) \rangle \f$ with the base point
 * \f$ \phi \f$ (`phi`), the correction \f$ w \f$, the inverse retraction \f$ R_\phi^{-1} \f$ of
 * ellipsoid::retract_inv_by_norm(), and the inner product of the M-metric (`coarse_mass_*`) or the F-metric
 * (`coarse_frobenius_*`).
 *
 * Shared by the oracles (oracle.h, oracle_coarse.h), the iterations (iteration.h) and the tests.
 */
namespace rmo::gpe::kernels
{

// -------------------------------------------------------------------------
// Riemannian gradients (fine level)
// -------------------------------------------------------------------------
/** @brief Energy-adaptive Riemannian gradient \f$ \grad_A E(x) = x - \frac{A^{-1} M x}{x^\top M A^{-1} M x} \f$. */
template <typename MatrixType, typename InverseMatrixType>
void grad_energy_adaptive(const InverseMatrixType& A_inv, const MatrixType& M,
                          const Vector<double>& x, Vector<double>& output)
{
    // \Pi_x(x): R^n -> T_x S^{n-1}
    metric::energy::project_onto_tangent_space(A_inv, x, M, output);
}

/** @brief Riemannian gradient in the M-metric, \f$ \grad_M E(x) = M^{-1} \big( A x - (x^\top A x) M x \big) \f$. */
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


/** @brief Riemannian gradient in the F-metric, \f$ \grad_F E(x) = A x - \frac{x^\top M A x}{x^\top M^2 x} M x \f$. */
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
/** @brief \f$ \Psi(\zeta) \f$ of the M-metric coarse model; @p energy is \f$ E(\zeta) \f$, evaluated by the caller. */
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

/** @brief \f$ \Psi(\zeta) \f$ of the F-metric coarse model; @p energy is \f$ E(\zeta) \f$, evaluated by the caller. */
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

/** @brief \f$ \mathrm{D}\Psi(\zeta)[z] \f$ of the M-metric coarse model, with \f$ A = A(\zeta) \f$. */
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

/** @brief \f$ \mathrm{D}\Psi(\zeta)[z] \f$ of the F-metric coarse model, with \f$ A = A(\zeta) \f$. */
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

// TODO: output directional derivative and riemannian gradient separately ("metric-free" line search)
/** @brief Riemannian gradient of the M-metric coarse model in the M-metric. */
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

// TODO: output directional derivative and riemannian gradient separately ("metric-free" line search)
//       tag- or class-based metric selection
/** @brief Riemannian gradient of the M-metric coarse model in the energy-adaptive metric. */
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
/** @brief Riemannian gradient of the F-metric coarse model in the F-metric. */
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
/** @brief Riemannian gradient of the F-metric coarse model in the energy-adaptive metric. */
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
