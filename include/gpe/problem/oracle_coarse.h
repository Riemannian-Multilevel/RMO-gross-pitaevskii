//
// Created by Ferdinand Vanmaele on 12.05.26.
//

#ifndef GPE_ORACLE_COARSE_H
#define GPE_ORACLE_COARSE_H

#include <gpe/problem/oracle.h>
#include <gpe/ropt/oracle_coarse.h>

namespace gpe
{

template <int dim>
class GrossPitaevskiiCoarseResidual
{
public:
    // M, A: matrices for computing residual of (uncorrected) objective E_GP
    // M_tilt: matrix for computing residual of coarse correction term <w,L(z)>
    GrossPitaevskiiCoarseResidual(const CoarseOracleBase& model)
        : m_model(model)
    // Assume CoarseOracleBase<> was constructed from GrossPitaevskiiOracle<>
        , gp_coarse(dynamic_cast<const GrossPitaevskiiOracle<dim>&>(model.coarse()))
        , m_norm(gp_coarse.get_M())
    {}

    double residual(const Vector<double>& x) const
    {
        // This is fixed for different coarse models
        return m_norm(residual_vector(x));
    }

protected:
    Vector<double> residual_vector(const Vector<double>& x) const
    {
        const auto& state = m_model.get_state();
        const auto& M     = gp_coarse.get_M();
        const auto& A     = gp_coarse.get_A();

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
    const GrossPitaevskiiOracle<dim> &gp_coarse;

    SpdNorm<OperatorType> m_norm;  // M-norm
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
    static constexpr auto model_t  = MetricKind::MASS;  // coarse model evaluated in M-metric
    static constexpr auto metric_t = MetricKind::MASS;  // gradient evaluated in M-metric

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

    double value(const Vector<double>& x) const override
    {
        const auto& coarse_step= m_model.get_state();

        return coarse::mass::function_value(x, coarse_step.y, coarse_step.w, M_coarse, gp_coarse.value(x));
    }

    double directional_derivative(const Vector<double>& x, const Vector<double>& z) const override
    {
        const auto& coarse_step= m_model.get_state();

        return coarse::mass::directional_derivative(x, coarse_step.y, coarse_step.w, z, M_coarse, A_coarse);
    }

    double residual(const Vector<double>& x) const override
    {
        return m_coarse_res.residual(x);
    }

    // Wrapper method for providing residual*TOL to matrix solver
    GradInfo gradient(const Vector<double>& x, Vector<double>& output, const double residual) const override
    {
        dealii::Timer timer;
        GradInfo info{};

        if (residual > 0) {
            M_inv_coarse.set_tol(residual * options.tol_inner_res);
        }

        timer.start();
        const auto& coarse_step = this->m_model.get_state();

        coarse::mass::gradient(M_coarse, M_inv_coarse, A_coarse, x, coarse_step.y, coarse_step.w, output);
        timer.stop();

        info.num_iter     = M_inv_coarse.control().last_step();
        info.tolerance    = M_inv_coarse.control().tolerance();
        info.elapsed_time = timer.cpu_time();

        return info;
    }

    // TODO: use options.tol_inner_res by default with opt-in residual?
    GradInfo gradient(const Vector<double>& x, Vector<double>& output) const override
    {
        // TODO: include residual in CPU time evaluation
        const double coarse_residual = this->residual(x);
        Assert(coarse_residual >= 0, dealii::ExcInternalError("residual must be positive"));

        auto info     = gradient(x, output, coarse_residual);
        info.residual = coarse_residual;

        return info;
    }

    unsigned n_dofs() const override
    {
        return gp_coarse.n_dofs();
    }

    const auto& get_M()  const { return M_coarse; }
    const auto& get_A()  const { return A_coarse; }

    double norm(const Vector<double>& v) const override
    {
        return m_norm(v);
    }

    double metric(const Vector<double>& x, const Vector<double>& z) const override
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
    CoarseOracleBase &m_model;
    GrossPitaevskiiCoarseResidual<dim> m_coarse_res;
    SolverOptions options;

    // TODO: dynamic_cast to const? (M, A const methods)
    GrossPitaevskiiOracle<dim> &gp_coarse;
    const OperatorType &M_coarse, &A_coarse;
    InverseOpType &M_inv_coarse;

    SpdNorm<OperatorType> m_norm;
};


template <int dim>
class MassCoarseOracleEnergyAdaptive : public OracleBase
{
public:
    const char* id() const override { return "MCA"; }
    static constexpr auto model_t  = MetricKind::MASS;             // coarse model evaluated in M-metric
    static constexpr auto metric_t = MetricKind::ENERGY_ADAPTIVE;  // gradient evaluated in A-metric

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

    double value(const Vector<double>& x) const override
    {
        const auto& coarse_step= m_model.get_state();

        return coarse::mass::function_value(x, coarse_step.y, coarse_step.w, M_coarse, gp_coarse.value(x));
    }

    double directional_derivative(const Vector<double>& x, const Vector<double>& z) const override
    {
        const auto& coarse_step= m_model.get_state();

        return coarse::mass::directional_derivative(x, coarse_step.y, coarse_step.w, z, M_coarse, A_coarse);
    }

    double residual(const Vector<double>& x) const override
    {
        return m_coarse_res.residual(x);
    }

    // Wrapper method for providing residual*TOL to matrix solver
    GradInfo gradient(const Vector<double>& x, Vector<double>& output, const double residual) const override
    {
        dealii::Timer timer;
        GradInfo info{};

        if (residual > 0) {
            A_inv_coarse.set_tol(residual * options.tol_inner_res);
        }

        timer.start();
        const auto& coarse_step = this->m_model.get_state();

        coarse::mass::energy_adaptive_gradient(M_coarse, A_inv_coarse, x, coarse_step.y, coarse_step.w, output);
        timer.stop();

        info.num_iter     = A_inv_coarse.control().last_step();
        info.tolerance    = A_inv_coarse.control().tolerance();
        info.elapsed_time = timer.cpu_time();

        return info;
    }

    // TODO: use options.tol_inner_res by default with opt-in residual?
    GradInfo gradient(const Vector<double>& x, Vector<double>& output) const override
    {
        // TODO: include residual in CPU time evaluation
        const double coarse_residual = this->residual(x);
        Assert(coarse_residual >= 0, dealii::ExcInternalError("residual must be positive"));

        auto info     = gradient(x, output, coarse_residual);
        info.residual = coarse_residual;

        return info;
    }

    unsigned n_dofs() const override
    {
        return gp_coarse.n_dofs();
    }

    const auto& get_M()  const { return M_coarse; }
    const auto& get_A()  const { return A_coarse; }

    double norm(const Vector<double>& v) const override
    {
        return m_norm(v);
    }

    double metric(const Vector<double>& x, const Vector<double>& z) const override
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
    CoarseOracleBase &m_model;
    GrossPitaevskiiCoarseResidual<dim> m_coarse_res;
    SolverOptions options;

    GrossPitaevskiiOracle<dim> &gp_coarse;
    const OperatorType &M_coarse, &A_coarse;
    InverseOpType &A_inv_coarse;

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
    static constexpr auto model_t  = MetricKind::FROBENIUS;  // coarse model evaluated in F-metric
    static constexpr auto metric_t = MetricKind::FROBENIUS;  // gradient evaluated in F-metric

    FrobeniusCoarseOracle(CoarseOracleBase& model, SolverOptions options = {})
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

    double value(const Vector<double>& x) const override
    {
        const auto& coarse_step= m_model.get_state();

        return coarse::frobenius::function_value(x, coarse_step.y, coarse_step.w, M_coarse, gp_coarse.value(x));
    }

    double directional_derivative(const Vector<double>& x, const Vector<double>& z) const override
    {
        const auto& coarse_step= m_model.get_state();

        return coarse::frobenius::directional_derivative(x, coarse_step.y, coarse_step.w, z, this->M_coarse, this->A_coarse);
    }

    double residual(const Vector<double>& x) const override
    {
        return m_coarse_res.residual(x);
    }

    // Wrapper method for providing residual*TOL to matrix solver
    // Frobenius: no-op since no matrix inversion is involved
    GradInfo gradient(const Vector<double>& x, Vector<double>& output, const double) const override
    {
        return gradient(x, output);  // No matrix inversion
    }

    GradInfo gradient(const Vector<double>& x, Vector<double>& output) const override
    {
        dealii::Timer timer;
        GradInfo info{};

        timer.start();
        const auto& coarse_step = this->m_model.get_state();

        coarse::frobenius::gradient(this->M_coarse, this->A_coarse, x, coarse_step.y, coarse_step.w, output);
        timer.stop();
        info.elapsed_time = timer.cpu_time();

        return info;
    }

    unsigned n_dofs() const override
    {
        return m_model.coarse().n_dofs();
    }

    const auto& get_M()  const { return M_coarse; }
    const auto& get_A()  const { return A_coarse; }

    double norm(const Vector<double>& v) const override
    {
        return std::sqrt(v*v);
    }

    double metric(const Vector<double>& x, const Vector<double>& z) const override
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
    CoarseOracleBase &m_model;
    GrossPitaevskiiCoarseResidual<dim> m_coarse_res;

    GrossPitaevskiiOracle<dim> &gp_coarse;
    const OperatorType &M_coarse, &A_coarse;
};


template <int dim>
class FrobeniusCoarseOracleEnergyAdaptive : public OracleBase
{
public:
    const char* id() const override { return "FCA"; }
    static constexpr auto model_t  = MetricKind::FROBENIUS;        // coarse model evaluated in F-metric
    static constexpr auto metric_t = MetricKind::ENERGY_ADAPTIVE;  // gradient evaluated in A-metric

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

    double value(const Vector<double>& x) const override
    {
        const auto& coarse_step= m_model.get_state();

        return coarse::frobenius::function_value(x, coarse_step.y, coarse_step.w, M_coarse, gp_coarse.value(x));
    }

    double directional_derivative(const Vector<double>& x, const Vector<double>& z) const override
    {
        const auto& coarse_step= m_model.get_state();

        return coarse::frobenius::directional_derivative(x, coarse_step.y, coarse_step.w, z, M_coarse, A_coarse);
    }

    double residual(const Vector<double>& x) const override
    {
        return m_coarse_res.residual(x);
    }

    // Wrapper method for providing residual*TOL to matrix solver
    GradInfo gradient(const Vector<double>& x, Vector<double>& output, const double residual) const override
    {
        dealii::Timer timer;
        GradInfo info{};

        if (residual > 0) {
            A_inv_coarse.set_tol(residual * options.tol_inner_res);
        }

        timer.start();
        const auto& coarse_step = this->m_model.get_state();

        coarse::frobenius::energy_adaptive_gradient(this->M_coarse, A_inv_coarse, this->A_coarse,
            x, coarse_step.y, coarse_step.w, output);
        timer.stop();

        info.num_iter     = A_inv_coarse.control().last_step();
        info.tolerance    = A_inv_coarse.control().tolerance();
        info.elapsed_time = timer.cpu_time();

        return info;
    }

    // TODO: use options.tol_inner_res by default with opt-in residual?
    GradInfo gradient(const Vector<double>& x, Vector<double>& output) const override
    {
        // TODO: include residual in CPU time evaluation
        const double coarse_residual = this->residual(x);
        Assert(coarse_residual >= 0, dealii::ExcInternalError("residual must be positive"));

        auto info     = gradient(x, output, coarse_residual);
        info.residual = coarse_residual;

        return info;
    }

    unsigned n_dofs() const override
    {
        return m_model.coarse().n_dofs();
    }

    const auto& get_M()  const { return M_coarse; }
    const auto& get_A()  const { return A_coarse; }

    double norm(const Vector<double>& v) const override
    {
        return m_norm(v);
    }

    double metric(const Vector<double>& x, const Vector<double>& z) const override
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
    CoarseOracleBase &m_model;
    GrossPitaevskiiCoarseResidual<dim> m_coarse_res;
    SolverOptions options;

    GrossPitaevskiiOracle<dim> &gp_coarse;
    const OperatorType& M_coarse, &A_coarse;
    InverseOpType &A_inv_coarse;

    SpdNorm<OperatorType> m_norm;
};

} // namespace gpe

#endif //GPE_ORACLE_COARSE_H
