//
// Created by Ferdinand Vanmaele on 12.05.26.
//

#ifndef RMO_GPE_ORACLE_COARSE_H
#define RMO_GPE_ORACLE_COARSE_H

#include <rmo/gpe/oracle.h>
#include <rmo/gpe/residual.h>
#include <rmo/gpe/manifold.h>
#include <rmo/gpe/metric.h>

#include <rmo/ropt/oracle_coarse_base.h>

namespace rmo::gpe
{

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
    GrossPitaevskiiCoarseResidual<System> m_coarse_res;  // not used for gradient(): no inner solver

    EuclideanMetric m_metric;
};


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

} // namespace rmo::gpe

#endif //RMO_GPE_ORACLE_COARSE_H
