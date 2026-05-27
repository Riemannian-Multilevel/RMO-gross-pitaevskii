//
// Created by Ferdinand Vanmaele on 12.05.26.
//

#ifndef GPE_ORACLE_COARSE_H
#define GPE_ORACLE_COARSE_H

#include <gpe/problem/oracle.h>

#include "fmt/base.h"

namespace gpe
{

// TODO: keep track of fine vector for consistency
struct CoarseState
{
    Vector<double> x;             // fine point
    Vector<double> y;             // restricted point (base point for coarse model)
    Vector<double> y_grad;        // gradient of restricted point
    Vector<double> x_grad;        // fine gradient
    Vector<double> x_grad_restr;  // restricted gradient
    Vector<double> w;             // correction vector

    CoarseState(unsigned n_fine, unsigned n_coarse)
        : x(n_fine)
        , y(n_coarse)
        , y_grad(n_coarse)
        , x_grad(n_fine)
        , x_grad_restr(n_coarse)
        , w(n_coarse)
    {}
};


template <typename CoarseModelType>
class GrossPitaevskiiCoarseResidual
{
public:
    // M, A: matrices for computing residual of (uncorrected) objective E_GP
    // M_tilt: matrix for computing residual of coarse correction term <w,L(z)>
    GrossPitaevskiiCoarseResidual(const CoarseModelType& model)
        : m_model(model)
        , m_norm(model.coarse().get_M())
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
        const auto& M     = m_model.coarse().get_M();
        const auto& A     = m_model.coarse().get_A();

        Vector<double> Mx(x.size());
        M.vmult(Mx, x);

        const double mass = x * Mx;
        //AssertThrow(std::abs(mass - 1) < 1e-12, dealii::ExcInternalError("mass constraint not fulfilled"));

        // 1. Compute the pullback of the tilt (u)
        Vector<double> u(x.size());
        ellipsoid::retract_inv_diff_by_norm_adjoint(M, state.y, x, state.w, u);

        Vector<double> grad_tilt(x.size());
        // This varies for different coarse models
        m_model.apply_metric(grad_tilt, u);

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
    const CoarseModelType& m_model;

    EnergyNorm<OperatorType> m_norm;  // M-norm
};


// Class which implements all needed terms for the Nash coarse model. It assumes an oracle on a fine and coarse
// level of discretization (implementing Riemannain gradient descent for a certain metric),
// used to compute a correction vector between coarse and fine gradients.
// TODO: for a multilevel implementation, O_fine(, O_coarse) need to be set to previous levels
template <int dim, typename FineOracleType, typename CoarseOracleType>
class CoarseOracleBase
{
public:
    static constexpr int dimension = dim;

    CoarseOracleBase(FineOracleType   &O_fine,
                     CoarseOracleType &O_coarse,
                     const ManifoldBase &coarse_manifold,
                     const ManifoldTransferBase &point_transfer,
                     const VectorTransportBase  &vector_transport)
    // Problem evaluation
        : O_fine(O_fine), O_coarse(O_coarse)
        , coarse_manifold(coarse_manifold)
        , n_fine(O_fine.n_dofs())
        , n_coarse(O_coarse.n_dofs())

    // Grid transfer
        , point_transfer(point_transfer)
        , vector_transport(vector_transport)

    // Coarse parameter initialization
        , m_state(n_fine, n_coarse)
    {
        Assert(FineOracleType::metric_t == CoarseOracleType::metric_t,
            dealii::ExcInternalError("non-corresponding metrics for coarse and fine oracle types"));
    }

    // Compute parameters for coarse model
    // Note that unlike the coarse models, which take a coarse vector as argument, this takes a fine vector
    // to build a new model based on the difference w of coarse gradients and restricted fine gradients.
    void update_model(const Vector<double>& x_fine)
    {
        AssertDimension(x_fine.size(), n_fine);
        m_state.x = x_fine;
        // Underlying state of O_fine assumed to match x (OracleBase::update -> GrossPitaevskiiSystem::update)

        // Compute base point for coarse model
#ifdef CPU_TIME
        std::cerr << "[" << timer.cpu_time() << "] coarse: point transfer\n";
#endif
        point_transfer.restriction(m_state.x, m_state.y);
        AssertDimension(m_state.y.size(), n_coarse);

#ifdef CPU_TIME
        std::cerr << "[" << timer.cpu_time() << "] coarse: assemble matrix\n";
#endif
        O_coarse.update(m_state.y);

        // Compute coarse (F or M)-gradient
#ifdef CPU_TIME
        std::cerr << "[" << timer.cpu_time() << "] coarse: " << O_coarse.id << "-coarse gradient\n";
#endif
        // TODO: increase tolerance for gradients defining the coarse model
        O_coarse.gradient(m_state.y, m_state.y_grad);
        AssertDimension(m_state.y_grad.size(), n_coarse);

        // Compute fine (F or M)-gradient
#ifdef CPU_TIME
        std::cerr << "[" << timer.cpu_time() << "] coarse: " << O_fine.id << "-fine gradient\n";
#endif
        // TODO: increase tolerance for gradients defining the coarse model
        O_fine.gradient(m_state.x, m_state.x_grad);
        AssertDimension(m_state.x_grad.size(), n_fine);

        // Compute restricted gradient
#ifdef CPU_TIME
        std::cerr << "[" << timer.cpu_time() << "] coarse: M-vector restriction\n";
#endif
        vector_transport.vector_restriction(m_state.y, m_state.x, m_state.x_grad, m_state.x_grad_restr);
        AssertDimension(m_state.x_grad_restr.size(), n_coarse);

        // Compute correction term
        m_state.w = m_state.y_grad;
        m_state.w.add(-1.0, m_state.x_grad_restr);
    }

    double norm(const Vector<double> &x) const { return O_coarse.norm(x); }
    double metric(const Vector<double> &x, const Vector<double> &z) const { return O_coarse.metric(x, z); }

    void set_timer(const dealii::Timer& timer_new) const { timer = timer_new; }
    const CoarseState& get_state() const { return m_state; }

    const FineOracleType& fine() const { return O_fine; }  // fine tilt oracle
    FineOracleType& fine() { return O_fine; }

    const CoarseOracleType& coarse() const { return O_coarse; }  // coarse tilt oracle
    CoarseOracleType& coarse() { return O_coarse; }


protected:
    // Coarse and fine level evaluation for correction vector w
    FineOracleType      &O_fine;
    CoarseOracleType    &O_coarse;
    const ManifoldBase  &coarse_manifold;
    unsigned n_fine, n_coarse;

    // Operators for transferring solutions and gradients
    const ManifoldTransferBase &point_transfer;
    const VectorTransportBase  &vector_transport;

    // Coarse model parameters
    CoarseState m_state;

    // Benchmarking
    mutable dealii::Timer timer;
};


// O_coarse: oracle for evaluating \grad E_c(y) in correction term w = \grad E_c(y) - R \grad E_f(x)
//           assumed to be consistent with metric in oracle for evaluating <w, .>_y

// O_fine:   oracle for evaluating \grad E_f(x) in correction term w = \grad E_c(y) - R \grad E_f(x)
//           assumed to be consistent with metric in oracle for evaluating <w, .>_y
//           independent of oracle used for gradient descent on the fine level

// this:     oracle for evaluating coarse model q_k(y) = E_c(y) + <w, .>_y
//           oracle for evaluating gradient of coarse model \grad q_k(y)
//           metric for \grad q_k(y) can differ from gradient of w and <w, .>_y


// =========================================================================
// Mass Coarse Family
// =========================================================================
template <int dim, typename FineOracleType>
class MassCoarseOracle : public OracleBase
{
public:
    static constexpr const char* id = "MC";
    using Base = CoarseOracleBase<dim, FineOracleType, MassOracle<dim>>;

    MassCoarseOracle(Base& coarse_model, SolverOptions options)
        : m_model(coarse_model)
        , m_coarse_res(coarse_model)
        , options(options)
        , M_coarse(coarse_model.coarse().get_M())
        , A_coarse(coarse_model.coarse().get_A())
        , M_inv_coarse(M_coarse, options)
        , m_norm(M_coarse)
    {}

    // Update for _evaluation_ of the coarse model
    // Distinguish from Base::update_model(x), which updates the coarse parameters w_k
    void update(const Vector<double>& x) override
    {
        m_model.coarse().update(x);
    }

    double value(const Vector<double>& x) const override
    {
        const auto& O_coarse = m_model.coarse();
        const auto& coarse_step  = m_model.get_state();

        return coarse::mass::function_value(x, coarse_step.y, coarse_step.w, M_coarse, O_coarse.value(x));
    }

    double directional_derivative(const Vector<double>& x, const Vector<double>& z) const override
    {
        const auto& coarse_step = m_model.get_state();

        return coarse::mass::directional_derivative(x, coarse_step.y, coarse_step.w, z, M_coarse, A_coarse);
    }

    double residual(const Vector<double>& x) const override
    {
        return m_coarse_res.residual(x);
    }

    GradInfo gradient(const Vector<double>& x, Vector<double>& output, double inv_tol) const
    {
        dealii::Timer timer;
        GradInfo info{};

        if (inv_tol > 0) {
            M_inv_coarse.set_tol(inv_tol);
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

    GradInfo gradient(const Vector<double>& x, Vector<double>& output) const override
    {
        // TODO: include residual in CPU time evaluation
        const double coarse_residual = this->residual(x);
        Assert(coarse_residual >= 0, dealii::ExcInternalError("residual must be positive"));

        const double inv_tol = coarse_residual * options.tol_inner_res;
        auto info     = gradient(x, output, inv_tol);
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

    void apply_metric(const Vector<double>& src, Vector<double>& dst) const override
    {
        M_coarse.vmult(dst, src);
    }

private:
    // TODO: the coarse model is const, but we require a non-const reference for updating the state of the coarse oracle
    //       (non-linear assembly)
    // Note: if Base::update_model(x) is called, this will be reflected in MassCoarseOracle
    // TODO: wrap Base::update_model in MassCoarseOracle to simplify the calling interface?
    Base& m_model;
    GrossPitaevskiiCoarseResidual<Base> m_coarse_res;
    SolverOptions options;

    const OperatorType &M_coarse, &A_coarse;
    InverseOpType M_inv_coarse;
    EnergyNorm<OperatorType> m_norm;
};


template <int dim, typename FineOracleType>
class MassCoarseOracleEnergyAdaptive : public OracleBase
{
public:
    static constexpr const char* id = "MCA";
    using Base = CoarseOracleBase<dim, FineOracleType, MassOracle<dim>>;

    MassCoarseOracleEnergyAdaptive(Base& coarse_model, SolverOptions options)
        : m_model(coarse_model)
        , m_coarse_res(coarse_model)
        , options(options)
        , M_coarse(m_model.coarse().get_M())
        , A_coarse(m_model.coarse().get_A())
        , A_inv_coarse(A_coarse, options)
        , m_norm(M_coarse)
    {
        A_inv_coarse.update_static(m_model.coarse().get_A0());
    }

    void update(const Vector<double>& x) override
    {
        m_model.coarse().update(x);

        A_inv_coarse.update_dynamic(A_coarse.diagonal());
    }

    double value(const Vector<double>& x) const override
    {
        const auto& O_coarse = m_model.coarse();
        const auto& coarse_step  = m_model.get_state();

        return coarse::mass::function_value(x, coarse_step.y, coarse_step.w, M_coarse, O_coarse.value(x));
    }

    double directional_derivative(const Vector<double>& x, const Vector<double>& z) const override
    {
        const auto& coarse_step = m_model.get_state();

        return coarse::mass::directional_derivative(x, coarse_step.y, coarse_step.w, z, M_coarse, A_coarse);
    }

    double residual(const Vector<double>& x) const override
    {
        return m_coarse_res.residual(x);
    }

    GradInfo gradient(const Vector<double>& x, Vector<double>& output, double inv_tol) const
    {
        dealii::Timer timer;
        GradInfo info{};

        if (inv_tol > 0) {
            A_inv_coarse.set_tol(inv_tol);
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

    GradInfo gradient(const Vector<double>& x, Vector<double>& output) const override
    {
        // TODO: include residual in CPU time evaluation
        const double coarse_residual = this->residual(x);
        Assert(coarse_residual >= 0, dealii::ExcInternalError("residual must be positive"));

        const double inv_tol = coarse_residual * options.tol_inner_res;
        auto info     = gradient(x, output, inv_tol);
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

    void apply_metric(const Vector<double>& src, Vector<double>& dst) const override
    {
        M_coarse.vmult(dst, src);
    }

private:
    // TODO: the coarse model is const, but we require a non-const reference for updating the state of the coarse oracle
    Base& m_model;
    GrossPitaevskiiCoarseResidual<Base> m_coarse_res;
    SolverOptions options;

    const OperatorType &M_coarse, &A_coarse;
    InverseOpType A_inv_coarse;
    EnergyNorm<OperatorType> m_norm;
};


// =========================================================================
// Frobenius Coarse Family
// =========================================================================

template <int dim, typename FineOracleType>
class FrobeniusCoarseOracle : public OracleBase
{
public:
    static constexpr const char* id = "FC";
    using Base = CoarseOracleBase<dim, FineOracleType, FrobeniusOracle<dim>>;

    FrobeniusCoarseOracle(Base& coarse_model)
        : m_model(coarse_model)
        , m_coarse_res(coarse_model)
        , M_coarse(m_model.coarse().get_M())
        , A_coarse(m_model.coarse().get_A())
    {}

    void update(const Vector<double>& x) override
    {
        m_model.coarse().update(x);
    }

    double value(const Vector<double>& x) const override
    {
        const auto& O_coarse    = m_model.coarse();
        const auto& coarse_step = m_model.get_state();

        return coarse::frobenius::function_value(x, coarse_step.y, coarse_step.w, M_coarse, O_coarse.value(x));
    }

    double directional_derivative(const Vector<double>& x, const Vector<double>& z) const override
    {
        const auto& coarse_step = m_model.get_state();

        return coarse::frobenius::directional_derivative(x, coarse_step.y, coarse_step.w, z, this->M_coarse, this->A_coarse);
    }

    double residual(const Vector<double>& x) const override
    {
        return m_coarse_res.residual(x);
    }

    GradInfo gradient(const Vector<double>& x, Vector<double>& output, double inv_tol) const
    {
        gradient(x, output);  // No matrix inversion
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

private:
    // TODO: the coarse model is const, but we require a non-const reference for updating the state of the coarse oracle
    Base& m_model;
    GrossPitaevskiiCoarseResidual<Base> m_coarse_res;
    SolverOptions options;

    const OperatorType &M_coarse, &A_coarse;
};


template <int dim, typename FineOracleType>
class FrobeniusCoarseOracleEnergyAdaptive : public OracleBase
{
public:
    static constexpr const char* id = "FCA";
    using Base = CoarseOracleBase<dim, FineOracleType, FrobeniusOracle<dim>>;

    FrobeniusCoarseOracleEnergyAdaptive(Base& coarse_model, SolverOptions options)
        : m_model(coarse_model)
        , m_coarse_res(coarse_model)
        , options(options)
        , M_coarse(m_model.coarse().get_M())
        , A_coarse(m_model.coarse().get_A())
        , A_inv_coarse(A_coarse, options)
    {
        A_inv_coarse.update_static(m_model.coarse().get_A0());
    }

    void update(const Vector<double>& x) override
    {
        m_model.coarse().update(x);

        A_inv_coarse.update_dynamic(A_coarse.diagonal());
    }

    double value(const Vector<double>& x) const override
    {
        const auto& O_coarse = m_model.coarse();
        const auto& coarse_step = m_model.get_state();

        return coarse::frobenius::function_value(x, coarse_step.y, coarse_step.w, M_coarse, O_coarse.value(x));
    }

    double directional_derivative(const Vector<double>& x, const Vector<double>& z) const override
    {
        const auto& coarse_step = m_model.get_state();

        return coarse::frobenius::directional_derivative(x, coarse_step.y, coarse_step.w, z, M_coarse, A_coarse);
    }
    
    double residual(const Vector<double>& x) const override
    {
        return m_coarse_res.residual(x);
    }

    GradInfo gradient(const Vector<double>& x, Vector<double>& output, const double inv_tol) const
    {
        dealii::Timer timer;
        GradInfo info{};

        if (inv_tol > 0) {
            A_inv_coarse.set_tol(inv_tol);
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

    GradInfo gradient(const Vector<double>& x, Vector<double>& output) const override
    {
        // TODO: include residual in CPU time evaluation
        const double coarse_residual = this->residual(x);
        Assert(coarse_residual >= 0, dealii::ExcInternalError("residual must be positive"));

        const double inv_tol = coarse_residual * options.tol_inner_res;
        auto info     = gradient(x, output, inv_tol);
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

private:
    // TODO: the coarse model is const, but we require a non-const reference for updating the state of the coarse oracle
    Base& m_model;
    GrossPitaevskiiCoarseResidual<Base> m_coarse_res;
    SolverOptions options;

    const OperatorType& M_coarse, &A_coarse;
    InverseOpType A_inv_coarse;
};

} // namespace gpe

#endif //GPE_ORACLE_COARSE_H
