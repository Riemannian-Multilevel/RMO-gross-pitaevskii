//
// Created by Ferdinand Vanmaele on 12.05.26.
//

#ifndef RMO_GPE_ORACLE_COARSE_H
#define RMO_GPE_ORACLE_COARSE_H

#include <rmo/gpe/oracle.h>
#include <rmo/gpe/manifold.h>
#include <rmo/gpe/metric.h>

#include <rmo/ropt/oracle_coarse_base.h>

namespace rmo::gpe
{

namespace detail
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

} // namespace detail


template <int dim>
class GrossPitaevskiiCoarseResidual
{
public:
    // M, A: matrices for computing residual of (uncorrected) objective E_GP
    // M_tilt: matrix for computing residual of coarse correction term <w,L(z)>
    explicit GrossPitaevskiiCoarseResidual(const CoarseOracleBase& model)
        : m_model(model)
          // Assume CoarseOracleBase<> was constructed from GrossPitaevskiiOracle<>
          , gp_coarse(dynamic_cast<const GrossPitaevskiiOracle<dim>&>(model.coarse()))
          , m_norm(gp_coarse.get_M())
    {
    }

    [[nodiscard]] double residual(const Vector<double>& x) const
    {
        // This is fixed for different coarse models
        return m_norm(residual_vector(x));
    }

protected:
    Vector<double> residual_vector(const Vector<double>& x) const
    {
        const auto& state = m_model.get_state();
        const auto& M = gp_coarse.get_M();
        const auto& A = gp_coarse.get_A();

        Vector<double> Mx(x.size());
        M.vmult(Mx, x);

        const double mass = x * Mx;
        //AssertThrow(std::abs(mass - 1) < 1e-12, dealii::ExcInternalError("mass constraint not fulfilled"));

        // 1. Compute the pullback of the tilt (u)
        Vector<double> u(x.size());
        ellipsoid::retract_inv_diff_by_norm_adjoint(M, state.y, x, state.w, u);

        // This varies for different coarse models
        Vector<double> grad_tilt(x.size());
        m_model.apply_metric(u, grad_tilt);

        // 2. Compute modified lambda: lambda_tilde = x^T A x - x^T M u
        Vector<double> Ax(x.size());
        A.vmult(Ax, x);
        const double lambda = (x * Ax - x * grad_tilt) / mass;

        // 3. Form the modified residual vector: r = (Ax - Mu) - lambda_tilde * Mx
        Vector<double> r(Ax);
        r.add(-1.0, grad_tilt);
        r.add(-lambda, Mx);

        return r;
    }

private:
    const CoarseOracleBase& m_model;
    const GrossPitaevskiiOracle<dim>& gp_coarse;

    SpdNorm<OperatorType> m_norm; // M-norm
};


// O_coarse: oracle for evaluating \grad E_c(y) in correction term w = \grad E_c(y) - R \grad E_f(x)
//           assumed to be consistent with metric in oracle for evaluating <w, .>_y

// O_fine:   oracle for evaluating \grad E_f(x) in correction term w = \grad E_c(y) - R \grad E_f(x)
//           assumed to be consistent with metric in oracle for evaluating <w, .>_y
//           independent of oracle used for gradient descent on the fine level
//           can be either a descent oracle, or a coarse oracle for a recursive implementation

// this:     oracle for evaluating coarse model q_k(y) = E_c(y) + <w, .>_y
//           oracle for evaluating gradient of coarse model \grad q_k(y)
//           metric for \grad q_k(y) can differ from gradient of w and <w, .>_y


// =========================================================================
// Mass Coarse Family
// =========================================================================
template <int dim>
class MassCoarseOracle : public OracleBase
{
public:
    const char* id() const override { return "MC"; }
    static constexpr auto model_t = MetricKind::MASS; // coarse model evaluated in M-metric
    static constexpr auto metric_t = MetricKind::MASS; // gradient evaluated in M-metric

    MassCoarseOracle(CoarseOracleBase& model, SolverOptions options)
        : m_model(model)
          , m_coarse_res(model)
          , options(options)
          // Assume CoarseOracleBase<> was constructed from GrossPitaevskiiOracle<>
          , gp_coarse(dynamic_cast<GrossPitaevskiiOracle<dim>&>(model.coarse()))
          , M_coarse(gp_coarse.get_M())
          , A_coarse(gp_coarse.get_A())
          , M_inv_coarse(gp_coarse.get_M_inv())
          , m_norm(M_coarse)
    {
        AssertThrow(model.coarse().get_metric() == model_t, dealii::ExcInternalError("mass metric expected"));
    }

    // Update for _evaluation_ of the coarse model
    // Distinguish from Base::update_model(x), which updates the coarse parameters w_k
    void update(const Vector<double>& x) override
    {
        gp_coarse.update(x);
    }

    [[nodiscard]] double value(const Vector<double>& x) const override
    {
        const auto& coarse_step = m_model.get_state();

        return detail::coarse_mass_value(x, coarse_step.y, coarse_step.w, M_coarse, gp_coarse.value(x));
    }

    [[nodiscard]] double directional_derivative(const Vector<double>& x, const Vector<double>& z) const override
    {
        const auto& coarse_step = m_model.get_state();

        return detail::coarse_mass_dir_deriv(x, coarse_step.y, coarse_step.w, z, M_coarse, A_coarse);
    }

    [[nodiscard]] double residual(const Vector<double>& x) const override
    {
        return m_coarse_res.residual(x);
    }

    // Wrapper method for providing residual*TOL to matrix solver
    GradInfo gradient(const Vector<double>& x, Vector<double>& output, const double residual) const override
    {
        dealii::Timer timer;
        GradInfo info{};

        if (residual > 0)
        {
            M_inv_coarse.set_tol(residual * options.tol_inner_res);
        }

        timer.start();
        const auto& coarse_step = this->m_model.get_state();

        detail::coarse_mass_grad(M_coarse, M_inv_coarse, A_coarse, x, coarse_step.y, coarse_step.w, output);
        timer.stop();

        info.num_iter = M_inv_coarse.control().last_step();
        info.tolerance = M_inv_coarse.control().tolerance();
        info.elapsed_time = timer.cpu_time();

        return info;
    }

    // TODO: use options.tol_inner_res by default with opt-in residual?
    GradInfo gradient(const Vector<double>& x, Vector<double>& output) const override
    {
        // TODO: include residual in CPU time evaluation
        const double coarse_residual = this->residual(x);
        Assert(coarse_residual >= 0, dealii::ExcInternalError("residual must be positive"));

        auto info = gradient(x, output, coarse_residual);
        info.residual = coarse_residual;

        return info;
    }

    [[nodiscard]] unsigned n_dofs() const override
    {
        return gp_coarse.n_dofs();
    }

    const auto& get_M() const { return M_coarse; }
    const auto& get_A() const { return A_coarse; }

    [[nodiscard]] double norm(const Vector<double>& v) const override
    {
        return m_norm(v);
    }

    [[nodiscard]] double inner(const Vector<double>& x, const Vector<double>& z) const override
    {
        return m_norm(x, z);
    }

    void apply_metric(const Vector<double>& src, Vector<double>& dst) const override
    {
        M_coarse.vmult(dst, src);
    }

    MetricKind get_metric() const override { return metric_t; }

private:
    // TODO: the coarse model is const, but we require a non-const reference for updating the state of the coarse oracle
    // Note: if Base::update_model(x) is called, this will be reflected in MassCoarseOracle
    // TODO: wrap Base::update_model to simplify the calling interface?
    CoarseOracleBase& m_model;
    GrossPitaevskiiCoarseResidual<dim> m_coarse_res;
    SolverOptions options;

    // TODO: dynamic_cast to const? (M, A const methods)
    GrossPitaevskiiOracle<dim>& gp_coarse;
    const OperatorType &M_coarse, &A_coarse;
    InverseOpType& M_inv_coarse;

    SpdNorm<OperatorType> m_norm;
};


template <int dim>
class MassCoarseOracleEnergyAdaptive : public OracleBase
{
public:
    const char* id() const override { return "MCA"; }
    static constexpr auto model_t = MetricKind::MASS; // coarse model evaluated in M-metric
    static constexpr auto metric_t = MetricKind::ENERGY_ADAPTIVE; // gradient evaluated in A-metric

    MassCoarseOracleEnergyAdaptive(CoarseOracleBase& model, SolverOptions options)
        : m_model(model)
          , m_coarse_res(model)
          , options(options)
          // Assume CoarseOracleBase<> was constructed from GrossPitaevskiiOracle<>
          , gp_coarse(dynamic_cast<GrossPitaevskiiOracle<dim>&>(model.coarse()))
          , M_coarse(gp_coarse.get_M())
          , A_coarse(gp_coarse.get_A())
          , A_inv_coarse(gp_coarse.get_A_inv())
          //, m_norm(M_coarse)
          , m_norm(A_coarse)
    {
        AssertThrow(model.coarse().get_metric() == model_t, dealii::ExcInternalError("mass metric expected"));
    }

    void update(const Vector<double>& x) override
    {
        gp_coarse.update(x);
    }

    [[nodiscard]] double value(const Vector<double>& x) const override
    {
        const auto& coarse_step = m_model.get_state();

        return detail::coarse_mass_value(x, coarse_step.y, coarse_step.w, M_coarse, gp_coarse.value(x));
    }

    [[nodiscard]] double directional_derivative(const Vector<double>& x, const Vector<double>& z) const override
    {
        const auto& coarse_step = m_model.get_state();

        return detail::coarse_mass_dir_deriv(x, coarse_step.y, coarse_step.w, z, M_coarse, A_coarse);
    }

    [[nodiscard]] double residual(const Vector<double>& x) const override
    {
        return m_coarse_res.residual(x);
    }

    // Wrapper method for providing residual*TOL to matrix solver
    GradInfo gradient(const Vector<double>& x, Vector<double>& output, const double residual) const override
    {
        dealii::Timer timer;
        GradInfo info{};

        if (residual > 0)
        {
            A_inv_coarse.set_tol(residual * options.tol_inner_res);
        }

        timer.start();
        const auto& coarse_step = this->m_model.get_state();

        detail::coarse_mass_grad_energy_adaptive(M_coarse, A_inv_coarse, x, coarse_step.y, coarse_step.w, output);
        timer.stop();

        info.num_iter = A_inv_coarse.control().last_step();
        info.tolerance = A_inv_coarse.control().tolerance();
        info.elapsed_time = timer.cpu_time();

        return info;
    }

    // TODO: use options.tol_inner_res by default with opt-in residual?
    GradInfo gradient(const Vector<double>& x, Vector<double>& output) const override
    {
        // TODO: include residual in CPU time evaluation
        const double coarse_residual = this->residual(x);
        Assert(coarse_residual >= 0, dealii::ExcInternalError("residual must be positive"));

        auto info = gradient(x, output, coarse_residual);
        info.residual = coarse_residual;

        return info;
    }

    [[nodiscard]] unsigned n_dofs() const override
    {
        return gp_coarse.n_dofs();
    }

    const auto& get_M() const { return M_coarse; }
    const auto& get_A() const { return A_coarse; }

    [[nodiscard]] double norm(const Vector<double>& v) const override
    {
        return m_norm(v);
    }

    [[nodiscard]] double inner(const Vector<double>& x, const Vector<double>& z) const override
    {
        return m_norm(x, z);
    }

    // TODO: apply_model_metric?  (residual computations)
    void apply_metric(const Vector<double>& src, Vector<double>& dst) const override
    {
        A_coarse.vmult(dst, src);
    }

    MetricKind get_metric() const override { return metric_t; }

private:
    // TODO: the coarse model is const, but we require a non-const reference for updating the state of the coarse oracle
    // Note: if Base::update_model(x) is called, this will be reflected in MassCoarseOracle
    // TODO: wrap Base::update_model to simplify the calling interface?
    CoarseOracleBase& m_model;
    GrossPitaevskiiCoarseResidual<dim> m_coarse_res;
    SolverOptions options;

    GrossPitaevskiiOracle<dim>& gp_coarse;
    const OperatorType &M_coarse, &A_coarse;
    InverseOpType& A_inv_coarse;

    SpdNorm<OperatorType> m_norm;
};


// =========================================================================
// Frobenius Coarse Family
// =========================================================================

template <int dim>
class FrobeniusCoarseOracle : public OracleBase
{
public:
    const char* id() const override { return "FC"; }
    static constexpr auto model_t = MetricKind::FROBENIUS; // coarse model evaluated in F-metric
    static constexpr auto metric_t = MetricKind::FROBENIUS; // gradient evaluated in F-metric

    FrobeniusCoarseOracle(CoarseOracleBase& model, SolverOptions)
        : m_model(model)
          , m_coarse_res(model)
          // Assume CoarseOracleBase<> was constructed from GrossPitaevskiiOracle<>
          , gp_coarse(dynamic_cast<GrossPitaevskiiOracle<dim>&>(model.coarse()))
          , M_coarse(gp_coarse.get_M())
          , A_coarse(gp_coarse.get_A())
    {
        AssertThrow(model.coarse().get_metric() == model_t, dealii::ExcInternalError("Frobenius metric expected"));
    }

    void update(const Vector<double>& x) override
    {
        gp_coarse.update(x);
    }

    [[nodiscard]] double value(const Vector<double>& x) const override
    {
        const auto& coarse_step = m_model.get_state();

        return detail::coarse_frobenius_value(x, coarse_step.y, coarse_step.w, M_coarse, gp_coarse.value(x));
    }

    [[nodiscard]] double directional_derivative(const Vector<double>& x, const Vector<double>& z) const override
    {
        const auto& coarse_step = m_model.get_state();

        return detail::coarse_frobenius_dir_deriv(x, coarse_step.y, coarse_step.w, z, this->M_coarse, this->A_coarse);
    }

    [[nodiscard]] double residual(const Vector<double>& x) const override
    {
        return m_coarse_res.residual(x);
    }

    // Wrapper method for providing residual*TOL to matrix solver
    // Frobenius: no-op since no matrix inversion is involved
    GradInfo gradient(const Vector<double>& x, Vector<double>& output, const double) const override
    {
        return gradient(x, output); // No matrix inversion
    }

    GradInfo gradient(const Vector<double>& x, Vector<double>& output) const override
    {
        dealii::Timer timer;
        GradInfo info{};

        timer.start();
        const auto& coarse_step = this->m_model.get_state();

        detail::coarse_frobenius_grad(this->M_coarse, this->A_coarse, x, coarse_step.y, coarse_step.w, output);
        timer.stop();
        info.elapsed_time = timer.cpu_time();

        return info;
    }

    [[nodiscard]] unsigned n_dofs() const override
    {
        return m_model.coarse().n_dofs();
    }

    const auto& get_M() const { return M_coarse; }
    const auto& get_A() const { return A_coarse; }

    [[nodiscard]] double norm(const Vector<double>& v) const override
    {
        return std::sqrt(v * v);
    }

    [[nodiscard]] double inner(const Vector<double>& x, const Vector<double>& z) const override
    {
        return x * z;
    }

    void apply_metric(const Vector<double>& src, Vector<double>& dst) const override
    {
        dst = src;
    }

    MetricKind get_metric() const override { return metric_t; }

private:
    // TODO: the coarse model is const, but we require a non-const reference for updating the state of the coarse oracle
    // Note: if Base::update_model(x) is called, this will be reflected in MassCoarseOracle
    // TODO: wrap Base::update_model to simplify the calling interface?
    CoarseOracleBase& m_model;
    GrossPitaevskiiCoarseResidual<dim> m_coarse_res;

    GrossPitaevskiiOracle<dim>& gp_coarse;
    const OperatorType &M_coarse, &A_coarse;
};


template <int dim>
class FrobeniusCoarseOracleEnergyAdaptive : public OracleBase
{
public:
    const char* id() const override { return "FCA"; }
    static constexpr auto model_t = MetricKind::FROBENIUS; // coarse model evaluated in F-metric
    static constexpr auto metric_t = MetricKind::ENERGY_ADAPTIVE; // gradient evaluated in A-metric

    FrobeniusCoarseOracleEnergyAdaptive(CoarseOracleBase& model, SolverOptions options)
        : m_model(model)
          , m_coarse_res(model)
          , options(options)
          // Assume CoarseOracleBase<> was constructed from GrossPitaevskiiOracle<>
          , gp_coarse(dynamic_cast<GrossPitaevskiiOracle<dim>&>(model.coarse()))
          , M_coarse(gp_coarse.get_M())
          , A_coarse(gp_coarse.get_A())
          , A_inv_coarse(gp_coarse.get_A_inv())
          , m_norm(A_coarse)
    {
        AssertThrow(model.coarse().get_metric() == model_t, dealii::ExcInternalError("Frobenius metric expected"));
    }

    void update(const Vector<double>& x) override
    {
        gp_coarse.update(x);
    }

    [[nodiscard]] double value(const Vector<double>& x) const override
    {
        const auto& coarse_step = m_model.get_state();

        return detail::coarse_frobenius_value(x, coarse_step.y, coarse_step.w, M_coarse, gp_coarse.value(x));
    }

    [[nodiscard]] double directional_derivative(const Vector<double>& x, const Vector<double>& z) const override
    {
        const auto& coarse_step = m_model.get_state();

        return detail::coarse_frobenius_dir_deriv(x, coarse_step.y, coarse_step.w, z, M_coarse, A_coarse);
    }

    [[nodiscard]] double residual(const Vector<double>& x) const override
    {
        return m_coarse_res.residual(x);
    }

    // Wrapper method for providing residual*TOL to matrix solver
    GradInfo gradient(const Vector<double>& x, Vector<double>& output, const double residual) const override
    {
        dealii::Timer timer;
        GradInfo info{};

        if (residual > 0)
        {
            A_inv_coarse.set_tol(residual * options.tol_inner_res);
        }

        timer.start();
        const auto& coarse_step = this->m_model.get_state();

        detail::coarse_frobenius_grad_energy_adaptive(this->M_coarse, A_inv_coarse, this->A_coarse,
            x, coarse_step.y, coarse_step.w, output);
        timer.stop();

        info.num_iter = A_inv_coarse.control().last_step();
        info.tolerance = A_inv_coarse.control().tolerance();
        info.elapsed_time = timer.cpu_time();

        return info;
    }

    // TODO: use options.tol_inner_res by default with opt-in residual?
    GradInfo gradient(const Vector<double>& x, Vector<double>& output) const override
    {
        // TODO: include residual in CPU time evaluation
        const double coarse_residual = this->residual(x);
        Assert(coarse_residual >= 0, dealii::ExcInternalError("residual must be positive"));

        auto info = gradient(x, output, coarse_residual);
        info.residual = coarse_residual;

        return info;
    }

    [[nodiscard]] unsigned n_dofs() const override
    {
        return m_model.coarse().n_dofs();
    }

    const auto& get_M() const { return M_coarse; }
    const auto& get_A() const { return A_coarse; }

    [[nodiscard]] double norm(const Vector<double>& v) const override
    {
        return m_norm(v);
    }

    [[nodiscard]] double inner(const Vector<double>& x, const Vector<double>& z) const override
    {
        return m_norm(x, z);
    }

    // TODO: apply_model_metric?  (residual computations)
    void apply_metric(const Vector<double>& src, Vector<double>& dst) const override
    {
        A_coarse.vmult(dst, src);
    }

    //MetricKind get_model() const { return model_t; }
    MetricKind get_metric() const override { return metric_t; }

private:
    // TODO: the coarse model is const, but we require a non-const reference for updating the state of the coarse oracle
    // Note: if Base::update_model(x) is called, this will be reflected in MassCoarseOracle
    // TODO: wrap Base::update_model to simplify the calling interface?
    CoarseOracleBase& m_model;
    GrossPitaevskiiCoarseResidual<dim> m_coarse_res;
    SolverOptions options;

    GrossPitaevskiiOracle<dim>& gp_coarse;
    const OperatorType &M_coarse, &A_coarse;
    InverseOpType& A_inv_coarse;

    SpdNorm<OperatorType> m_norm;
};

} // namespace rmo::gpe

#endif //RMO_GPE_ORACLE_COARSE_H
