//
// Created by Ferdinand Vanmaele on 12.05.26.
//

#ifndef GPE_ORACLE_COARSE_H
#define GPE_ORACLE_COARSE_H

#include <gpe/problem/oracle.h>

namespace gpe
{

// Class which implements all needed terms for the Nash coarse model. It assumes an oracle on a fine and coarse
// level of discretization (implementing Riemannain gradient descent for a certain metric),
// used to compute a correction vector between coarse and fine gradients.
// TODO: for a multilevel implementation, O(, O_coarse) need to be set to previous levels
template <int dim, typename TiltOracleFine, typename TiltOracleCoarse>
class CoarseOracleBase
{
public:
    static constexpr int dimension = dim;

    struct CoarseState
    {
        Vector<double> y;             // restricted point (base point for coarse model)
        Vector<double> y_grad;        // gradient of restricted point
        Vector<double> x_grad;        // fine gradient
        Vector<double> x_grad_restr;  // restricted gradient
        Vector<double> w;             // correction vector

        CoarseState(unsigned n_fine, unsigned n_coarse)
            : y(n_coarse)
            , y_grad(n_coarse)
            , x_grad(n_fine)
            , x_grad_restr(n_coarse)
            , w(n_coarse)
        {}
    };

    CoarseOracleBase(TiltOracleFine& O,
                     TiltOracleCoarse& O_coarse,
                     const ManifoldTransferBase& point_transfer,
                     const VectorTransportBase& vector_transport)
        : O(O), O_coarse(O_coarse)

    // Problem evaluation
        , n(O.n_dofs())
        , n_coarse(O_coarse.n_dofs())

    // Grid transfer
        , point_transfer(point_transfer)
        , vector_transport(vector_transport)

    // Coarse parameters
        , m_state(n, n_coarse)
    {}

    // Compute parameters for coarse model
    void update(const Vector<double>& x)
    {
        AssertDimension(x.size(), n);

        // Update state for fine oracle
#ifdef CPU_TIME
        std::cerr << "[" << timer.cpu_time() << "] fine: assemble matrix\n";
#endif
        O.update(x);

        // Set base point for coarse model
#ifdef CPU_TIME
        std::cerr << "[" << timer.cpu_time() << "] coarse: point transfer\n";
#endif
        point_transfer.restriction(x, m_state.y);

#ifdef CPU_TIME
        std::cerr << "[" << timer.cpu_time() << "] coarse: assemble matrix\n";
#endif
        O_coarse.update(m_state.y);

        // Compute coarse M-gradient
#ifdef CPU_TIME
        std::cerr << "[" << timer.cpu_time() << "] coarse: " << O_coarse.id << "-coarse gradient\n";
#endif
        O_coarse.gradient(m_state.y, m_state.y_grad);

        // Compute fine M-gradient
#ifdef CPU_TIME
        std::cerr << "[" << timer.cpu_time() << "] coarse: " << O.id << "-fine gradient\n";
#endif
        O.gradient(x, m_state.x_grad);

        // Compute restricted gradient
#ifdef CPU_TIME
        std::cerr << "[" << timer.cpu_time() << "] coarse: M-vector restriction\n";
#endif
        vector_transport.vector_restriction(m_state.y, x, m_state.x_grad, m_state.x_grad_restr);

        // Compute correction term
        m_state.w = m_state.y_grad;
        m_state.w.add(-1.0, m_state.x_grad_restr);
    }

    void set_timer(const dealii::Timer& timer_new) const
    {
        timer = timer_new;
    }

    const CoarseState& get_state() const
    {
        return m_state;
    }

    const TiltOracleFine& objective_fine() const
    {
        return O;
    }

    const TiltOracleCoarse& objective_coarse() const
    {
        return O_coarse;
    }

protected:
    // Coarse and fine level evaluation for correction vector w
    TiltOracleFine &O;
    TiltOracleCoarse &O_coarse;
    unsigned n, n_coarse;

    // Operators for transferring solutions and gradients
    const ManifoldTransferBase& point_transfer;
    const VectorTransportBase& vector_transport;

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
template <int dim, typename FineOracleType>
class MassCoarseOracle : public OracleBase
{
public:
    static constexpr const char* id = "MC";
    static constexpr int dimension = dim;

    MassCoarseOracle(CoarseOracleBase<dim, FineOracleType, MassOracle<dim>>& coarse_model, SolverOptions options)
        : coarse_model(coarse_model)
        , options(options)
        , M_coarse(coarse_model.objective_coarse().get_M())
        , A_coarse(coarse_model.objective_coarse().get_A())
        , M_inv_coarse(M_coarse, options)
    {}

    // Assembly:
    //   O.update(x) (fine)
    //   O_coarse.update(x) (coarse)
    // Model parameters:
    //   CoarseOracleBase::CoarseState (restricted gradients)
    void update(const Vector<double>& x) override
    {
        coarse_model.update(x);
    }

    double value(const Vector<double>& x) const override
    {
        const auto& O_coarse = coarse_model.objective_coarse();
        const auto& coarse_step = coarse_model.get_state();

        return coarse::mass::function_value(x, coarse_step.y, coarse_step.w,
            M_coarse, O_coarse.value(x));
    }

    double directional_derivative(const Vector<double>& x, const Vector<double>& z) const override
    {
        const auto& O_coarse = coarse_model.objective_coarse();
        const auto& coarse_step = coarse_model.get_state();

        Vector<double> Mw(n_dofs());
        M_coarse.vmult(Mw, coarse_step.w);

        return O_coarse.directional_derivative(x, z) - (Mw * z);
    }

    /**
     * @brief Computes the coarse model gradient in the M-metric.
     */
    unsigned gradient(const Vector<double>& x, Vector<double>& output) const override
    {
        const auto& coarse_step = coarse_model.get_state();

        coarse::mass::gradient(M_coarse, M_inv_coarse, A_coarse,
            x, coarse_step.y, coarse_step.w, output);

        return M_inv_coarse.control().last_step();
    }

    iteration::State residual(const Vector<double>& x) const final
    {
        return {.energy=value(x)};
    }

    unsigned n_dofs() const override
    {
        return coarse_model.objective_coarse().n_dofs();
    }

    double norm(const Vector<double>& v) const override
    {
        Vector<double> Mv(n_dofs());
        M_coarse.vmult(Mv, v);

        return std::sqrt(v*Mv);
    }

    // For recursive calls (n-level algorithms where fine oracle = CoarseOracleBase)
    const auto& get_M()  const
    {
        return coarse_model.objective_coarse().get_M();
    }
    const auto& get_A()  const
    {
        return coarse_model.objective_coarse().get_A();
    }
    const auto& get_A0() const
    {
        return coarse_model.objective_coarse().get_A0();
    }

private:
    CoarseOracleBase<dim, FineOracleType, MassOracle<dim>>& coarse_model;
    SolverOptions options;

    const OperatorType &M_coarse, &A_coarse;  // operators owned by MassOracle <- GrossPitaevskiiFunctional
    InverseOpType M_inv_coarse;
};


template <int dim, typename FineOracleType>
class MassCoarseOracleEnergyAdaptive : public OracleBase
{
public:
    static constexpr const char* id = "MCA";
    static constexpr int dimension = dim;

    MassCoarseOracleEnergyAdaptive(CoarseOracleBase<dim, FineOracleType, MassOracle<dim>>& coarse_model,
                                   SolverOptions options)
        : coarse_model(coarse_model)
        , options(options)
        , M_coarse(coarse_model.objective_coarse().get_M())
        , A_coarse(coarse_model.objective_coarse().get_A())
        , A_inv_coarse(A_coarse, options)
    {
        A_inv_coarse.update_static(coarse_model.objective_coarse().get_A0());
    }

    void update(const Vector<double>& x) override
    {
        coarse_model.update(x);

        A_inv_coarse.update_dynamic(A_coarse.diagonal());
    }

    double value(const Vector<double>& x) const override
    {
        const auto& O_coarse = coarse_model.objective_coarse();
        const auto& coarse_step = coarse_model.get_state();

        return coarse::mass::function_value(x, coarse_step.y, coarse_step.w,
            M_coarse, O_coarse.value(x));
    }

    double directional_derivative(const Vector<double>& x, const Vector<double>& z) const override
    {
        const MassOracle<dim>& O_coarse = coarse_model.objective_coarse();
        const auto& coarse_step = coarse_model.get_state();

        Vector<double> Mw(n_dofs());
        M_coarse.vmult(Mw, coarse_step.w);

        return O_coarse.directional_derivative(x, z) - (Mw * z);
    }

    /**
     * @brief Computes the coarse model gradient in the A-metric.
     * $$ \nabla_A q_k(x) = \nabla_A E_H(x) - w $$
     */
    unsigned gradient(const Vector<double>& x, Vector<double>& output) const override
    {
        const auto& coarse_step = coarse_model.get_state();

        coarse::mass::energy_adaptive_gradient(M_coarse, A_inv_coarse,
            x, coarse_step.y, coarse_step.w, output);

        return A_inv_coarse.control().last_step();
    }

    iteration::State residual(const Vector<double>& x) const override
    {
        return {.energy=value(x)};
    }

    unsigned n_dofs() const override
    {
        return coarse_model.objective_coarse().n_dofs();
    }

    double norm(const Vector<double>& v) const override
    {
        Vector<double> Mv(n_dofs());
        M_coarse.vmult(Mv, v);

        return std::sqrt(v*Mv);
    }

    // For recursive calls (n-level algorithms where fine oracle = CoarseOracleBase)
    const auto& get_M()  const
    {
        return coarse_model.objective_coarse().get_M();
    }
    const auto& get_A()  const
    {
        return coarse_model.objective_coarse().get_A();
    }
    const auto& get_A0() const
    {
        return coarse_model.objective_coarse().get_A0();
    }


private:
    CoarseOracleBase<dim, FineOracleType, MassOracle<dim>>& coarse_model;
    SolverOptions options;

    const OperatorType &M_coarse, &A_coarse;  // operators owned by EnergyOracle <- GrossPitaevskiiFunctional
    InverseOpType A_inv_coarse;
};


template <int dim, typename FineOracleType>
class FrobeniusCoarseOracle : public OracleBase
{
public:
    static constexpr const char* id = "FC";
    static constexpr int dimension = dim;

    FrobeniusCoarseOracle(CoarseOracleBase<dim, FineOracleType, FrobeniusOracle<dim>>& coarse_model,
                          SolverOptions options)
        : coarse_model(coarse_model)
        , options(options)
        , M_coarse(coarse_model.objective_coarse().get_M())
        , A_coarse(coarse_model.objective_coarse().get_A())
    {}

    void update(const Vector<double>& x) override
    {
        coarse_model.update(x);
    }

    double value(const Vector<double>& x) const override
    {
        const auto& O_coarse = coarse_model.objective_coarse();
        const auto& coarse_step = coarse_model.get_state();

        return coarse::frobenius::function_value(x, coarse_step.y, coarse_step.w,
            M_coarse, O_coarse.value(x));
    }

    double directional_derivative(const Vector<double>& x, const Vector<double>& z) const override
    {
        const auto& O_coarse = coarse_model.objective_coarse();
        const auto& coarse_step = coarse_model.get_state();

        return O_coarse.directional_derivative(x, z) - coarse_step.w * z;
    }

    /**
     * @brief Computes the coarse model gradient in the F-metric.
     * $$ \nabla_F q_k(\zeta) = \Pi_{\zeta, F}\left(A_\zeta \zeta - \frac{1}{\phi^\top M\zeta}w\right) $$
     */
    unsigned gradient(const Vector<double>& x, Vector<double>& output) const override
    {
        const auto& coarse_step = coarse_model.get_state();

        // Compute the pure Frobenius gradient
        coarse::frobenius::gradient(M_coarse, A_coarse, x, coarse_step.y, coarse_step.w, output);

        return 0;  // 0 iterations, as no Krylov solver is used
    }

    iteration::State residual(const Vector<double>& x) const override
    {
        return {.energy=value(x)};
    }

    unsigned n_dofs() const override
    {
        return coarse_model.objective_coarse().n_dofs();
    }

    double norm(const Vector<double>& v) const override
    {
        return std::sqrt(v*v);
    }

    // For recursive calls (n-level algorithms where fine oracle = CoarseOracleBase)
    const auto& get_M()  const
    {
        return coarse_model.objective_coarse().get_M();
    }
    const auto& get_A()  const
    {
        return coarse_model.objective_coarse().get_A();
    }
    const auto& get_A0() const
    {
        return coarse_model.objective_coarse().get_A0();
    }


private:
    CoarseOracleBase<dim, FineOracleType, FrobeniusOracle<dim>>& coarse_model;
    SolverOptions options;

    const OperatorType &M_coarse, &A_coarse;  // operators owned by EnergyOracle <- GrossPitaevskiiFunctional
};


// TODO: Vector m_w is computed in the caller and assumed consistent (computed using same metric) as ::gradient
template <int dim, typename FineOracleType>
class FrobeniusCoarseOracleEnergyAdaptive : public OracleBase
{
public:
    static constexpr const char* id = "FCA";
    static constexpr int dimension = dim;

    FrobeniusCoarseOracleEnergyAdaptive(CoarseOracleBase<dim, FineOracleType, FrobeniusOracle<dim>>& coarse_model,
                                        SolverOptions options)
        : coarse_model(coarse_model)
        , options(options)
        , M_coarse(coarse_model.objective_coarse().get_M())
        , A_coarse(coarse_model.objective_coarse().get_A())
        , A_inv_coarse(A_coarse, options)
    {
        A_inv_coarse.update_static(coarse_model.objective_coarse().get_A0());
    }

    void update(const Vector<double>& x) override
    {
        coarse_model.update(x);

        A_inv_coarse.update_dynamic(A_coarse.diagonal());
    }

    double value(const Vector<double>& x) const override
    {
        const auto& O_coarse = coarse_model.objective_coarse();
        const auto& coarse_step = coarse_model.get_state();

        return coarse::frobenius::function_value(x, coarse_step.y, coarse_step.w,
            M_coarse, O_coarse.value(x));
    }

    double directional_derivative(const Vector<double>& x, const Vector<double>& z) const override
    {
        const auto& O_coarse = coarse_model.objective_coarse();
        const auto& coarse_step = coarse_model.get_state();

        return O_coarse.directional_derivative(x, z) - coarse_step.w * z;
    }

    /**
     * @brief Computes the coarse model gradient in the energy-adaptive metric.
     * $$ \nabla_A q_k(\zeta) = \tilde{A}_\zeta^{-1} \left( \nabla_F q_k(\zeta) \right) $$
     */
    unsigned gradient(const Vector<double>& x, Vector<double>& output) const final
    {
        const auto& coarse_step = coarse_model.get_state();

        // Computes the F-gradient and applies the A_inv preconditioner
        coarse::frobenius::energy_adaptive_gradient(M_coarse, A_inv_coarse,A_coarse,
            x, coarse_step.y, coarse_step.w, output);

        // Return the number of Krylov iterations used by A_inv
        return A_inv_coarse.control().last_step();
    }

    iteration::State residual(const Vector<double>& x) const final
    {
        return {.energy=value(x)};
    }

    unsigned n_dofs() const override
    {
        return coarse_model.objective_coarse().n_dofs();
    }

    double norm(const Vector<double>& v) const override
    {
        return std::sqrt(v*v);
    }

    // For recursive calls (n-level algorithms where fine oracle = CoarseOracleBase)
    const auto& get_M()  const
    {
        return coarse_model.objective_coarse().get_M();
    }
    const auto& get_A()  const
    {
        return coarse_model.objective_coarse().get_A();
    }
    const auto& get_A0() const
    {
        return coarse_model.objective_coarse().get_A0();
    }


private:
    CoarseOracleBase<dim, FineOracleType, FrobeniusOracle<dim>>& coarse_model;
    SolverOptions options;

    const OperatorType &M_coarse, &A_coarse;  // operators owned by EnergyOracle <- GrossPitaevskiiFunctional
    InverseOpType A_inv_coarse;
};

} // namespace gpe

#endif //GPE_ORACLE_COARSE_H
