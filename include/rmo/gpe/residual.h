#ifndef RMO_GPE_RESIDUAL_H
#define RMO_GPE_RESIDUAL_H

#include <rmo/lac.h>
#include <rmo/gpe/gpe.h>
#include <rmo/gpe/manifold.h>

#include <rmo/ropt/metric.h>
#include <rmo/ropt/oracle_coarse_base.h>

/**
 * @file
 * @brief Eigenvalue residuals of the Gross-Pitaevskii problem on the fine level (GrossPitaevskiiResidual)
 * and of the tilted coarse models (GrossPitaevskiiCoarseResidual). The oracles use them to set the
 * tolerance of the inner solvers in gradient().
 */
namespace rmo::gpe
{

namespace kernels
{

/**
 * @brief Residual of the tilted Gross-Pitaevskii eigenvalue problem.
 *
 * \f[ r = A x - g - \lambda M x, \qquad \lambda = \frac{x^\top (A x - g)}{x^\top M x} \f]
 *
 * The tilt @p g is the linear term of a coarse model (the pulled-back coarse correction).
 *
 * @param A The state-dependent total linear operator (\f$A_x\f$).
 * @param M The mass matrix (\f$M\f$).
 * @param x The current state vector.
 * @param g The linear (tilt) term.
 */
template <typename MatrixType>
Vector<double> eigen_residual(const MatrixType& A, const MatrixType& M, const Vector<double>& x,
                              const Vector<double>& g)
{
    Vector<double> Mx(x.size());
    M.vmult(Mx, x);
    const double mass = x * Mx;

    Vector<double> Ax(x.size());
    A.vmult(Ax, x);
    const double lambda = (x * Ax - x * g) / mass;  // (tilted) Rayleigh quotient

    Vector<double> r(Ax);
    r.add(-1.0, g);
    r.add(-lambda, Mx);

    return r;
}

/**
 * @brief Residual of the Gross-Pitaevskii eigenvalue problem, \f$ r = A x - \lambda M x \f$ with the
 * Rayleigh quotient \f$ \lambda = x^\top A x / x^\top M x \f$.
 */
template <typename MatrixType>
Vector<double> eigen_residual(const MatrixType& A, const MatrixType& M, const Vector<double>& x)
{
    Vector<double> Mx(x.size());
    M.vmult(Mx, x);
    const double mass = x * Mx;

    Vector<double> Ax(x.size());
    A.vmult(Ax, x);
    const double lambda = x * Ax / mass;  // Rayleigh quotient

    Vector<double> r(Ax);
    r.add(-lambda, Mx);

    return r;
}

} // namespace kernels


template <typename System>
class GrossPitaevskiiResidual
{
public:
    using Functional = GrossPitaevskiiFunctional<System>;
    using Operator   = typename Functional::Operator;

    explicit GrossPitaevskiiResidual(const Functional& func)
        : m_func(func)
          , m_metric(func.get_M(), MetricKind::MASS)
    {
    }

    [[nodiscard]] double residual(const Vector<double>& x) const
    {
        return m_metric.norm(residual_vector(x));
    }

protected:
    Vector<double> residual_vector(const Vector<double>& x) const
    {
        // r = A x - lambda M x, with the Rayleigh quotient lambda = x^T A x / x^T M x
        return kernels::eigen_residual(m_func.get_A(), m_func.get_M(), x);
    }

private:
    const Functional& m_func;

    OperatorMetric<Operator> m_metric; // M-metric
};


template <typename System>
class GrossPitaevskiiCoarseResidual
{
public:
    using Operator = typename GrossPitaevskiiFunctional<System>::Operator;

    // M, A: operators of the (uncorrected) coarse objective E_GP
    // model: coarse model, providing the state (y, w) and the metric of the coarse correction term <w,L(z)>
    GrossPitaevskiiCoarseResidual(const CoarseOracleBase& model, const Operator& M, const Operator& A)
        : m_model(model)
          , M(M)
          , A(A)
          , m_metric(M, MetricKind::MASS)
    {
    }

    [[nodiscard]] double residual(const Vector<double>& x) const
    {
        // This is fixed for different coarse models
        return m_metric.norm(residual_vector(x));
    }

protected:
    Vector<double> residual_vector(const Vector<double>& x) const
    {
        const auto& state = m_model.get_state();

        // 1. Compute the pullback of the tilt (u)
        Vector<double> u(x.size());
        ellipsoid::retract_inv_diff_by_norm_adjoint(M, state.y, x, state.w, u);

        // This varies for different coarse models
        Vector<double> grad_tilt(x.size());
        m_model.metric().apply(u, grad_tilt);

        // 2. Residual of the tilted problem: r = (Ax - grad_tilt) - lambda_tilde * Mx
        return kernels::eigen_residual(A, M, x, grad_tilt);
    }

private:
    const CoarseOracleBase& m_model;
    const Operator &M, &A;

    OperatorMetric<Operator> m_metric; // M-metric
};

} // namespace rmo::gpe

#endif //RMO_GPE_RESIDUAL_H
