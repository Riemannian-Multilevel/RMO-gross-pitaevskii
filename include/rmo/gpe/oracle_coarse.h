//
// Created by Ferdinand Vanmaele on 12.05.26.
//

#ifndef RMO_GPE_ORACLE_COARSE_H
#define RMO_GPE_ORACLE_COARSE_H

#include <rmo/gpe/oracle.h>
#include <rmo/gpe/residual.h>
#include <rmo/gpe/manifold.h>
#include <rmo/gpe/metric.h>

#include <rmo/ropt/oracle_coarse.h>

#include <algorithm>
#include <cstdint>
#include <memory>

/**
 * @file
 * @brief Coarse oracles of the coarse models \f$ \Psi(\zeta) = E_c(\zeta) - \langle w, R_y^{-1}(\zeta) \rangle \f$
 * (see kernels.h), built on a CoarseOracleBase.
 *
 * The CoarseOracleBase computes the correction \f$ w = \grad E_c(y) - \mathcal{R} \grad E_f(x) \f$ from its coarse
 * and fine tilt oracles; their metric (M or F) defines the inner product of the model. The fine tilt oracle is a
 * level oracle, or a coarse oracle in a recursive cycle. The metric of the Riemannian gradient of \f$ \Psi \f$
 * may differ from that of the model:
 * - MassCoarseOracle: M-metric model, gradient in the M-metric;
 * - MassCoarseOracleEnergyAdaptive: M-metric model, gradient in the energy-adaptive metric;
 * - FrobeniusCoarseOracle: F-metric model, gradient in the F-metric;
 * - FrobeniusCoarseOracleEnergyAdaptive: F-metric model, gradient in the energy-adaptive metric;
 * - DensityCoarseOracle: model with the density lifting on the positive sphere (M-metric), gradient selected by
 *   DensityGradient.
 */
namespace rmo::gpe
{


// =========================================================================
// Mass Coarse Family
// =========================================================================
/** @brief M-metric coarse model, Riemannian gradient in the M-metric. */
template <typename System>
class MassCoarseOracle : public OracleBase
{
public:
    using GPOracle = GrossPitaevskiiOracle<System>;
    using Operator = typename GPOracle::Operator;
    using InverseM = typename GPOracle::InverseM;

    const char* id() const override { return "MC"; }
    static constexpr auto model_t = MetricKind::MASS; // coarse model evaluated in M-metric
    static constexpr auto metric_t = MetricKind::MASS; // gradient evaluated in M-metric

    MassCoarseOracle(CoarseOracleBase& model, SolverOptions options)
        : m_model(model)
          , options(options)
          // Assume CoarseOracleBase<> was constructed from GrossPitaevskiiOracle<>
          , gp_coarse(dynamic_cast<GPOracle&>(model.coarse()))
          , M_coarse(gp_coarse.get_M())
          , A_coarse(gp_coarse.get_A())
          , m_coarse_res(model, gp_coarse.get_functional())
          , M_inv_coarse(gp_coarse.get_M_inv())
          , m_metric(M_coarse, metric_t)
    {
        AssertThrow(model.coarse().metric().kind() == model_t, dealii::ExcInternalError("mass metric expected"));
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

        return kernels::coarse_mass_value(x, coarse_step.y, coarse_step.w, M_coarse, gp_coarse.value(x));
    }

    [[nodiscard]] double directional_derivative(const Vector<double>& x, const Vector<double>& z) const override
    {
        const auto& coarse_step = m_model.get_state();

        return kernels::coarse_mass_dir_deriv(x, coarse_step.y, coarse_step.w, z, M_coarse, A_coarse);
    }


    // Wrapper method for providing residual*TOL to matrix solver
    GradInfo gradient(const Vector<double>& x, Vector<double>& output, const double residual) const override
    {
        AssertResidualAt(m_coarse_res, x, residual);

        dealii::Timer timer;
        GradInfo info{};

        if (residual > 0)
        {
            M_inv_coarse.set_tol(residual * options.tol_inner_res);
        }

        timer.start();
        const auto& coarse_step = this->m_model.get_state();

        kernels::coarse_mass_grad(M_coarse, M_inv_coarse, A_coarse, x, coarse_step.y, coarse_step.w, output);
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
        const double coarse_residual = m_coarse_res.residual(x);
        Assert(coarse_residual >= 0, dealii::ExcInternalError("residual must be positive"));

        auto info = gradient(x, output, coarse_residual);
        info.residual = coarse_residual;

        return info;
    }

    [[nodiscard]] unsigned n_dofs() const override
    {
        return gp_coarse.n_dofs();
    }

    const Operator& get_M() const { return M_coarse; }
    const Operator& get_A() const { return A_coarse; }

    [[nodiscard]] const MetricBase& metric() const override { return m_metric; }

    const GrossPitaevskiiCoarseResidual<System>& get_residual() const { return m_coarse_res; }

private:
    // TODO: the coarse model is const, but we require a non-const reference for updating the state of the coarse oracle
    // Note: if Base::update_model(x) is called, this will be reflected in MassCoarseOracle
    // TODO: wrap Base::update_model to simplify the calling interface?
    CoarseOracleBase& m_model;
    SolverOptions options;

    // TODO: dynamic_cast to const? (M, A const methods)
    GPOracle& gp_coarse;
    const Operator &M_coarse, &A_coarse;
    GrossPitaevskiiCoarseResidual<System> m_coarse_res;
    InverseM& M_inv_coarse;

    OperatorMetric<Operator> m_metric;
};


/** @brief M-metric coarse model, Riemannian gradient in the energy-adaptive metric. */
template <typename System>
class MassCoarseOracleEnergyAdaptive : public OracleBase
{
public:
    using GPOracle = GrossPitaevskiiOracle<System>;
    using Operator = typename GPOracle::Operator;
    using InverseA = typename GPOracle::InverseA;

    const char* id() const override { return "MCA"; }
    static constexpr auto model_t = MetricKind::MASS; // coarse model evaluated in M-metric
    static constexpr auto metric_t = MetricKind::ENERGY_ADAPTIVE; // gradient evaluated in A-metric

    MassCoarseOracleEnergyAdaptive(CoarseOracleBase& model, SolverOptions options)
        : m_model(model)
          , options(options)
          // Assume CoarseOracleBase<> was constructed from GrossPitaevskiiOracle<>
          , gp_coarse(dynamic_cast<GPOracle&>(model.coarse()))
          , M_coarse(gp_coarse.get_M())
          , A_coarse(gp_coarse.get_A())
          , m_coarse_res(model, gp_coarse.get_functional())
          , A_inv_coarse(gp_coarse.get_A_inv())
          , m_metric(A_coarse, metric_t)
    {
        AssertThrow(model.coarse().metric().kind() == model_t, dealii::ExcInternalError("mass metric expected"));
    }

    void update(const Vector<double>& x) override
    {
        gp_coarse.update(x);
    }

    [[nodiscard]] double value(const Vector<double>& x) const override
    {
        const auto& coarse_step = m_model.get_state();

        return kernels::coarse_mass_value(x, coarse_step.y, coarse_step.w, M_coarse, gp_coarse.value(x));
    }

    [[nodiscard]] double directional_derivative(const Vector<double>& x, const Vector<double>& z) const override
    {
        const auto& coarse_step = m_model.get_state();

        return kernels::coarse_mass_dir_deriv(x, coarse_step.y, coarse_step.w, z, M_coarse, A_coarse);
    }


    // Wrapper method for providing residual*TOL to matrix solver
    GradInfo gradient(const Vector<double>& x, Vector<double>& output, const double residual) const override
    {
        AssertResidualAt(m_coarse_res, x, residual);

        dealii::Timer timer;
        GradInfo info{};

        if (residual > 0)
        {
            A_inv_coarse.set_tol(residual * options.tol_inner_res);
        }

        timer.start();
        const auto& coarse_step = this->m_model.get_state();

        kernels::coarse_mass_grad_energy_adaptive(M_coarse, A_inv_coarse, x, coarse_step.y, coarse_step.w, output);
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
        const double coarse_residual = m_coarse_res.residual(x);
        Assert(coarse_residual >= 0, dealii::ExcInternalError("residual must be positive"));

        auto info = gradient(x, output, coarse_residual);
        info.residual = coarse_residual;

        return info;
    }

    [[nodiscard]] unsigned n_dofs() const override
    {
        return gp_coarse.n_dofs();
    }

    const Operator& get_M() const { return M_coarse; }
    const Operator& get_A() const { return A_coarse; }

    [[nodiscard]] const MetricBase& metric() const override { return m_metric; }

    const GrossPitaevskiiCoarseResidual<System>& get_residual() const { return m_coarse_res; }

private:
    // TODO: the coarse model is const, but we require a non-const reference for updating the state of the coarse oracle
    // Note: if Base::update_model(x) is called, this will be reflected in MassCoarseOracle
    // TODO: wrap Base::update_model to simplify the calling interface?
    CoarseOracleBase& m_model;
    SolverOptions options;

    GPOracle& gp_coarse;
    const Operator &M_coarse, &A_coarse;
    GrossPitaevskiiCoarseResidual<System> m_coarse_res;
    InverseA& A_inv_coarse;

    OperatorMetric<Operator> m_metric;
};


// =========================================================================
// Frobenius Coarse Family
// =========================================================================

/** @brief F-metric coarse model, Riemannian gradient in the F-metric. */
template <typename System>
class FrobeniusCoarseOracle : public OracleBase
{
public:
    using GPOracle = GrossPitaevskiiOracle<System>;
    using Operator = typename GPOracle::Operator;

    const char* id() const override { return "FC"; }
    static constexpr auto model_t = MetricKind::FROBENIUS; // coarse model evaluated in F-metric
    static constexpr auto metric_t = MetricKind::FROBENIUS; // gradient evaluated in F-metric

    FrobeniusCoarseOracle(CoarseOracleBase& model, SolverOptions)
        : m_model(model)
          // Assume CoarseOracleBase<> was constructed from GrossPitaevskiiOracle<>
          , gp_coarse(dynamic_cast<GPOracle&>(model.coarse()))
          , M_coarse(gp_coarse.get_M())
          , A_coarse(gp_coarse.get_A())
          , m_coarse_res(model, gp_coarse.get_functional())
    {
        AssertThrow(model.coarse().metric().kind() == model_t, dealii::ExcInternalError("Frobenius metric expected"));
    }

    void update(const Vector<double>& x) override
    {
        gp_coarse.update(x);
    }

    [[nodiscard]] double value(const Vector<double>& x) const override
    {
        const auto& coarse_step = m_model.get_state();

        return kernels::coarse_frobenius_value(x, coarse_step.y, coarse_step.w, M_coarse, gp_coarse.value(x));
    }

    [[nodiscard]] double directional_derivative(const Vector<double>& x, const Vector<double>& z) const override
    {
        const auto& coarse_step = m_model.get_state();

        return kernels::coarse_frobenius_dir_deriv(x, coarse_step.y, coarse_step.w, z, this->M_coarse, this->A_coarse);
    }


    // Wrapper method for providing residual*TOL to matrix solver
    // Frobenius: no-op since no matrix inversion is involved
    GradInfo gradient(const Vector<double>& x, Vector<double>& output, const double residual) const override
    {
        AssertResidualAt(m_coarse_res, x, residual);
        return gradient(x, output); // No matrix inversion
    }

    GradInfo gradient(const Vector<double>& x, Vector<double>& output) const override
    {
        dealii::Timer timer;
        GradInfo info{};

        timer.start();
        const auto& coarse_step = this->m_model.get_state();

        kernels::coarse_frobenius_grad(this->M_coarse, this->A_coarse, x, coarse_step.y, coarse_step.w, output);
        timer.stop();
        info.elapsed_time = timer.cpu_time();

        return info;
    }

    [[nodiscard]] unsigned n_dofs() const override
    {
        return m_model.coarse().n_dofs();
    }

    const Operator& get_M() const { return M_coarse; }
    const Operator& get_A() const { return A_coarse; }

    [[nodiscard]] const MetricBase& metric() const override { return m_metric; }

    const GrossPitaevskiiCoarseResidual<System>& get_residual() const { return m_coarse_res; }

private:
    // TODO: the coarse model is const, but we require a non-const reference for updating the state of the coarse oracle
    // Note: if Base::update_model(x) is called, this will be reflected in MassCoarseOracle
    // TODO: wrap Base::update_model to simplify the calling interface?
    CoarseOracleBase& m_model;

    GPOracle& gp_coarse;
    const Operator &M_coarse, &A_coarse;
    GrossPitaevskiiCoarseResidual<System> m_coarse_res;  // no inner solver: only checks the residual passed to gradient()

    EuclideanMetric m_metric;
};


/** @brief F-metric coarse model, Riemannian gradient in the energy-adaptive metric. */
template <typename System>
class FrobeniusCoarseOracleEnergyAdaptive : public OracleBase
{
public:
    using GPOracle = GrossPitaevskiiOracle<System>;
    using Operator = typename GPOracle::Operator;
    using InverseA = typename GPOracle::InverseA;

    const char* id() const override { return "FCA"; }
    static constexpr auto model_t = MetricKind::FROBENIUS; // coarse model evaluated in F-metric
    static constexpr auto metric_t = MetricKind::ENERGY_ADAPTIVE; // gradient evaluated in A-metric

    FrobeniusCoarseOracleEnergyAdaptive(CoarseOracleBase& model, SolverOptions options)
        : m_model(model)
          , options(options)
          // Assume CoarseOracleBase<> was constructed from GrossPitaevskiiOracle<>
          , gp_coarse(dynamic_cast<GPOracle&>(model.coarse()))
          , M_coarse(gp_coarse.get_M())
          , A_coarse(gp_coarse.get_A())
          , m_coarse_res(model, gp_coarse.get_functional())
          , A_inv_coarse(gp_coarse.get_A_inv())
          , m_metric(A_coarse, metric_t)
    {
        AssertThrow(model.coarse().metric().kind() == model_t, dealii::ExcInternalError("Frobenius metric expected"));
    }

    void update(const Vector<double>& x) override
    {
        gp_coarse.update(x);
    }

    [[nodiscard]] double value(const Vector<double>& x) const override
    {
        const auto& coarse_step = m_model.get_state();

        return kernels::coarse_frobenius_value(x, coarse_step.y, coarse_step.w, M_coarse, gp_coarse.value(x));
    }

    [[nodiscard]] double directional_derivative(const Vector<double>& x, const Vector<double>& z) const override
    {
        const auto& coarse_step = m_model.get_state();

        return kernels::coarse_frobenius_dir_deriv(x, coarse_step.y, coarse_step.w, z, M_coarse, A_coarse);
    }


    // Wrapper method for providing residual*TOL to matrix solver
    GradInfo gradient(const Vector<double>& x, Vector<double>& output, const double residual) const override
    {
        AssertResidualAt(m_coarse_res, x, residual);

        dealii::Timer timer;
        GradInfo info{};

        if (residual > 0)
        {
            A_inv_coarse.set_tol(residual * options.tol_inner_res);
        }

        timer.start();
        const auto& coarse_step = this->m_model.get_state();

        kernels::coarse_frobenius_grad_energy_adaptive(this->M_coarse, A_inv_coarse, this->A_coarse,
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
        const double coarse_residual = m_coarse_res.residual(x);
        Assert(coarse_residual >= 0, dealii::ExcInternalError("residual must be positive"));

        auto info = gradient(x, output, coarse_residual);
        info.residual = coarse_residual;

        return info;
    }

    [[nodiscard]] unsigned n_dofs() const override
    {
        return m_model.coarse().n_dofs();
    }

    const Operator& get_M() const { return M_coarse; }
    const Operator& get_A() const { return A_coarse; }

    [[nodiscard]] const MetricBase& metric() const override { return m_metric; }

    const GrossPitaevskiiCoarseResidual<System>& get_residual() const { return m_coarse_res; }

private:
    // TODO: the coarse model is const, but we require a non-const reference for updating the state of the coarse oracle
    // Note: if Base::update_model(x) is called, this will be reflected in MassCoarseOracle
    // TODO: wrap Base::update_model to simplify the calling interface?
    CoarseOracleBase& m_model;
    SolverOptions options;

    GPOracle& gp_coarse;
    const Operator &M_coarse, &A_coarse;
    GrossPitaevskiiCoarseResidual<System> m_coarse_res;
    InverseA& A_inv_coarse;

    OperatorMetric<Operator> m_metric;
};

// =========================================================================
// Density-lifting coarse model
// =========================================================================

/**
 * @brief Riemannian gradient of DensityCoarseOracle: mass, Frobenius, or energy-adaptive with the metric
 * \f$ \hat A(z) \f$ of the model (MODEL), its shift \f$ \hat A_\sigma = \hat A + \sigma D \f$ (MODEL_SHIFT, the
 * safeguard where \f$ \hat A \f$ may not be positive definite), or \f$ A(z) \f$ of the coarse functional (FINE).
 */
enum class DensityGradient : std::uint8_t { MASS, FROBENIUS, ENERGY_MODEL, ENERGY_MODEL_SHIFT, ENERGY_FINE };

/**
 * @brief Coarse model with the density lifting, \f$ q(z) = f_H(z) - \langle w, \mathcal{L}^m_y(z) \rangle_D \f$
 * (tex/drafts/positive_multilevel.tex), on the positive sphere of a lumped system.
 *
 * With \f$ w \f$ tangent at \f$ y \f$ in the mass metric (\f$ y^\top D w = 0 \f$), the correction is a shift of the
 * potential: \f$ q(z) = f_H(z) + \frac12 z^\top W z \f$, \f$ W = -D \operatorname{diag}(w/y) \f$, so
 * \f$ \nabla q(z) = \hat A(z) z \f$ with \f$ \hat A = A_0 + \beta M_{\phi\phi}(z) + W \f$. The model is built in the
 * mass metric (the coarse tilt oracle must be an M-metric oracle); the Riemannian gradient is selected by @p G:
 * - MASS: \f$ D^{-1} (\hat A z - (z^\top \hat A z) D z) \f$;
 * - FROBENIUS: \f$ \hat A z - \frac{z^\top D \hat A z}{z^\top D^2 z} D z \f$;
 * - ENERGY_MODEL: metric \f$ \hat A(z) \f$, \f$ z - \hat A^{-1} D z / (z^\top D \hat A^{-1} D z) \f$. Needs
 *   \f$ \hat A \f$ positive definite; then (a symmetric Z-matrix) \f$ \hat A^{-1} \ge 0 \f$ and \f$ -\grad q > -z \f$.
 * - ENERGY_MODEL_SHIFT: the same with \f$ \hat A_\sigma = \hat A + \sigma D \f$,
 *   \f$ \sigma = \max(0, \max_j (w_j / y_j - (A_0 \onebb)_j / d_j)) \f$ (\f$ (A_0 \onebb)_j / d_j = V_j \f$ for a
 *   Neumann boundary): every row sum of \f$ \hat A_\sigma \f$ is at least \f$ \beta d_j z_j^2 > 0 \f$, so for
 *   \f$ A_0 \f$ with non-positive off-diagonals it is strictly diagonally dominant and positive definite. On the
 *   sphere the shift adds the constant \f$ \sigma / 2 \f$ to \f$ q \f$; only the metric changes.
 * - ENERGY_FINE: metric \f$ A(z) \f$ of \f$ f_H \f$ (as MassCoarseOracleEnergyAdaptive), the \f$ A \f$-orthogonal
 *   projection of \f$ A^{-1} \nabla q \f$; no positivity bound.
 *
 * update() recomputes \f$ W \f$ (and \f$ \sigma \f$) from the state of the CoarseOracleBase, and counts the points at
 * which a row sum of \f$ \hat A(z) \f$ is not positive (where ENERGY_MODEL is not guaranteed positive definite).
 */
template <typename System, DensityGradient G>
class DensityCoarseOracle : public OracleBase
{
    static_assert(is_diagonal_matrix_v<typename System::MassMatrix>, "the density lifting requires a lumped system");

public:
    using GPOracle   = GrossPitaevskiiOracle<System>;
    using Functional = typename GPOracle::Functional;
    using Operator   = typename GPOracle::Operator;
    using InverseA   = typename GPOracle::InverseA;
    using MassMatrix = typename System::MassMatrix;

    static constexpr auto model_t  = MetricKind::MASS;  // the correction term <w, L(z)> is in the M-metric
    static constexpr auto metric_t = G == DensityGradient::MASS      ? MetricKind::MASS
                                   : G == DensityGradient::FROBENIUS ? MetricKind::FROBENIUS
                                                                     : MetricKind::ENERGY_ADAPTIVE;
    static constexpr bool model_metric = G == DensityGradient::ENERGY_MODEL || G == DensityGradient::ENERGY_MODEL_SHIFT;

    const char* id() const override
    {
        switch (G) {
            case DensityGradient::MASS: return "DM";
            case DensityGradient::FROBENIUS: return "DF";
            case DensityGradient::ENERGY_MODEL: return "DA";
            case DensityGradient::ENERGY_MODEL_SHIFT: return "DAs";
            default: return "DAf";
        }
    }

    DensityCoarseOracle(CoarseOracleBase& model, SolverOptions options)
        : m_model(model)
        , options(options)
        , gp_coarse(dynamic_cast<GPOracle&>(model.coarse()))
        , func(gp_coarse.get_functional())
        , M_coarse(gp_coarse.get_M())
        , A_coarse(gp_coarse.get_A())
        , d(M_coarse.diagonal())
        , row_sum0(d.size())
        , W(Vector<double>(d.size()))
        , W_metric(Vector<double>(d.size()))
        , m_res(model, func)
        , A_metric_inv(A_metric, options)
    {
        AssertThrow(model.coarse().metric().kind() == model_t,
                    dealii::ExcMessage("DensityCoarseOracle: the coarse model must be built in the mass metric"));

        // A_hat = A_0 + beta M_pp(z) + W, and its metric version with W_metric = W + sigma D
        A_hat.add_component(1.0, func.get_A0());
        A_hat.add_component(func.get_beta(), func.get_Mpp());
        A_hat.add_component(1.0, W);
        A_hat.reinit(d);
        A_metric.add_component(1.0, func.get_A0());
        A_metric.add_component(func.get_beta(), func.get_Mpp());
        A_metric.add_component(1.0, W_metric);
        A_metric.reinit(d);
        A_metric_inv.update_static(func.get_A0());

        // (A_0 1)_j / d_j: the potential V_j for a Neumann boundary
        Vector<double> ones(d.size());
        ones = 1.0;
        func.get_A0().vmult(row_sum0, ones);
        for (unsigned j = 0; j < d.size(); ++j) {
            row_sum0[j] /= d[j];
        }

        if constexpr (G == DensityGradient::FROBENIUS) {
            m_metric = std::make_unique<EuclideanMetric>();
        } else if constexpr (G == DensityGradient::MASS) {
            m_metric = std::make_unique<OperatorMetric<Operator>>(M_coarse, metric_t);
        } else if constexpr (model_metric) {
            m_metric = std::make_unique<OperatorMetric<Operator>>(A_metric, metric_t);
        } else {
            m_metric = std::make_unique<OperatorMetric<Operator>>(A_coarse, metric_t);
        }
    }

    // The operators refer to the members W, W_metric
    DensityCoarseOracle(const DensityCoarseOracle&) = delete;
    DensityCoarseOracle& operator=(const DensityCoarseOracle&) = delete;

    void update(const Vector<double>& z) override
    {
        gp_coarse.update(z);  // M_pp(z)

        const auto& state = m_model.get_state();
        auto& W_vec = W.get_vector();
        double shift = 0.0;
        bool dominant = true;
        for (unsigned j = 0; j < z.size(); ++j) {
            const double rate = state.w[j] / state.y[j];
            W_vec[j] = -d[j] * rate;
            shift = std::max(shift, rate - row_sum0[j]);
            // Row sum of A_hat(z) / d_j: V_j + beta z_j^2 - w_j / y_j (for a Neumann boundary)
            dominant = dominant && row_sum0[j] + func.get_beta() * z[j] * z[j] - rate > 0;
        }
        n_updates++;
        n_not_dd += dominant ? 0 : 1;

        sigma = G == DensityGradient::ENERGY_MODEL_SHIFT ? shift : 0.0;
        sigma_max = std::max(sigma_max, sigma);
        auto& Wm_vec = W_metric.get_vector();
        for (unsigned j = 0; j < z.size(); ++j) {
            Wm_vec[j] = W_vec[j] + sigma * d[j];
        }
        if constexpr (model_metric) {
            A_metric_inv.update_dynamic(A_metric.diagonal());
        }
    }

    //! \f$ q(z) = f_H(z) + \frac12 z^\top W z \f$ (the oracle is at z)
    [[nodiscard]] double value(const Vector<double>& z) const override
    {
        double correction = 0.0;
        for (unsigned j = 0; j < z.size(); ++j) {
            correction += W.get_vector()[j] * z[j] * z[j];
        }
        return gp_coarse.value(z) + 0.5 * correction;
    }

    [[nodiscard]] double directional_derivative(const Vector<double>& z, const Vector<double>& v) const override
    {
        return euclidean_gradient(z) * v;
    }

    GradInfo gradient(const Vector<double>& z, Vector<double>& output, const double residual) const override
    {
        AssertResidualAt(m_res, z, residual);

        dealii::Timer timer;
        GradInfo info{};
        timer.start();
        const Vector<double> g = euclidean_gradient(z);

        if constexpr (G == DensityGradient::MASS) {
            Vector<double> Mz(z.size());
            M_coarse.vmult(Mz, z);
            Vector<double> r(g);
            r.add(-(z * g), Mz);
            gp_coarse.get_M_inv().vmult(output, r);
        }
        else if constexpr (G == DensityGradient::FROBENIUS) {
            Vector<double> Mz(z.size()), Mg(z.size()), MMz(z.size());
            M_coarse.vmult(Mz, z);
            M_coarse.vmult(Mg, g);
            M_coarse.vmult(MMz, Mz);
            output = g;
            output.add(-(z * Mg) / (z * MMz), Mz);
        }
        else if constexpr (model_metric) {
            if (residual > 0) {
                A_metric_inv.set_tol(residual * options.tol_inner_res);
            }
            kernels::grad_energy_adaptive(A_metric_inv, M_coarse, z, output);
            info.num_iter  = A_metric_inv.control().last_step();
            info.tolerance = A_metric_inv.control().tolerance();
        }
        else {
            InverseA& A_inv = gp_coarse.get_A_inv();
            if (residual > 0) {
                A_inv.set_tol(residual * options.tol_inner_res);
            }
            Vector<double> u(z.size());
            A_inv.vmult(u, g);
            metric::energy::project_onto_tangent_space(A_inv, z, M_coarse, u, output);
            info.num_iter  = A_inv.control().last_step();
            info.tolerance = A_inv.control().tolerance();
        }
        timer.stop();
        info.elapsed_time = timer.cpu_time();
        return info;
    }

    GradInfo gradient(const Vector<double>& z, Vector<double>& output) const override
    {
        const double coarse_residual = m_res.residual(z);
        auto info = gradient(z, output, coarse_residual);
        info.residual = coarse_residual;
        return info;
    }

    [[nodiscard]] unsigned n_dofs() const override { return gp_coarse.n_dofs(); }

    [[nodiscard]] const MetricBase& metric() const override { return *m_metric; }

    const DensityCoarseResidual<System>& get_residual() const { return m_res; }

    //! Shift of the last update() (0 unless ENERGY_MODEL_SHIFT), and its maximum over all updates
    [[nodiscard]] double get_sigma() const { return sigma; }
    [[nodiscard]] double get_sigma_max() const { return sigma_max; }
    //! Number of update() calls, and of those at which a row sum of \f$ \hat A(z) \f$ was not positive
    [[nodiscard]] unsigned get_n_updates() const { return n_updates; }
    [[nodiscard]] unsigned get_n_not_dominant() const { return n_not_dd; }

private:
    //! \f$ \nabla q(z) = \hat A(z) z = A(z) z + W z \f$
    [[nodiscard]] Vector<double> euclidean_gradient(const Vector<double>& z) const
    {
        Vector<double> g(z.size());
        A_coarse.vmult(g, z);
        W.vmult_add(g, z);
        return g;
    }

    CoarseOracleBase& m_model;
    SolverOptions options;

    GPOracle& gp_coarse;
    const Functional& func;
    const Operator &M_coarse, &A_coarse;
    Vector<double> d;         // lumped masses
    Vector<double> row_sum0;  // (A_0 1)_j / d_j

    MassMatrix W, W_metric;   // -D diag(w/y), and + sigma D
    Operator A_hat, A_metric;
    DensityCoarseResidual<System> m_res;
    mutable InverseA A_metric_inv;
    std::unique_ptr<MetricBase> m_metric;

    double sigma = 0.0, sigma_max = 0.0;
    unsigned n_updates = 0, n_not_dd = 0;
};

} // namespace rmo::gpe

#endif //RMO_GPE_ORACLE_COARSE_H
