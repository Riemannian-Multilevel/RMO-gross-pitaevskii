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
// TODO: for a multilevel implementation, O(_coarse) need to be set to previous levels
template <int dim, typename TiltOracle>
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

    CoarseOracleBase(const TiltOracle& O, const TiltOracle& O_coarse,
                     const ManifoldTransferBase& point_transfer,
                     const VectorTransportBase&  vector_transport)
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
        std::cerr << "[" << timer.cpu_time() << "] coarse: " << O_fine.id << "-fine gradient\n";
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


protected:
    // Coarse and fine level evaluation for correction vector w
    const TiltOracle& O, O_coarse;
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
template <int dim>
class MassCoarseOracle : public OracleBase
{
public:
    static constexpr const char* id = "MC";


    double value(const Vector<double>& x) const final
    {
        const double energy = this->problem.value(x, this->beta);

        return coarse::mass::function_value(x, this->m_phi, this->m_w, this->M_coarse, energy);
    }

    /**
     * @brief Computes the coarse model gradient in the M-metric.
     */
    unsigned gradient(const Vector<double>& x, Vector<double>& output) const final
    {
        coarse::mass::gradient(this->M_coarse, this->M_inv_coarse, this->A_coarse,
            x, this->m_state.y, this->m_state.w, output);

        return this->M_inv_coarse.control().last_step();
    }

    iteration::State residual(const Vector<double>& x) const final
    {
        return {.energy=value(x)};
    }
};


template <int dim>
class MassCoarseOracleEnergyAdaptive : public OracleBase
{
public:
    static constexpr const char* id = "MCA";

    MassCoarseOracleEnergyAdaptive(const GrossPitaevskiiSystem<dim>& problem,
                                   const GrossPitaevskiiSystem<dim>& problem_coarse,
                                   const ManifoldTransferBase& point_transfer,
                                   const VectorTransportBase& vector_transport,
                                   double beta, SolverOptions options, SolverOptions options_coarse)
        : CoarseOracleBase<dim, MassOracle<dim>>(problem, problem_coarse,
            point_transfer, vector_transport, beta, options, options_coarse)
    {
        this->A_inv_coarse.update_static(this->problem_coarse.get_A0());
    }

    void update(const Vector<double>& x) override
    {
        CoarseOracleBase<dim, MassOracle<dim>>::update(x); // Calls assembly in parent

        this->A_inv_coarse.update_dynamic(this->A_coarse.diagonal());
    }

    double value(const Vector<double>& x) const override
    {
        const double energy = this->problem.value(x, this->beta);

        return coarse::mass::function_value(x, this->m_phi, this->m_w, this->M_coarse, energy);
    }

    /**
     * @brief Computes the coarse model gradient in the A-metric.
     * $$ \nabla_A q_k(x) = \nabla_A E_H(x) - w $$
     */
    unsigned gradient(const Vector<double>& x, Vector<double>& output) const override
    {
        coarse::mass::energy_adaptive_gradient(this->M_coarse, this->A_inv_coarse,
            x, this->m_state.y, this->m_state.w, output);

        return this->A_inv_coarse.control().last_step();
    }

    iteration::State residual(const Vector<double>& x) const override
    {
        return {.energy=value(x)};
    }
};


template <int dim>
class FrobeniusCoarseOracle : public OracleBase
{
public:
    static constexpr const char* id = "FC";


    double value(const Vector<double>& x) const final
    {
        const double energy = this->problem.value(x, this->beta);

        return coarse::frobenius::function_value(x, this->m_phi, this->m_w, this->M_coarse, energy);
    }

    /**
     * @brief Computes the coarse model gradient in the F-metric.
     * $$ \nabla_F q_k(\zeta) = \Pi_{\zeta, F}\left(A_\zeta \zeta - \frac{1}{\phi^\top M\zeta}w\right) $$
     */
    unsigned gradient(const Vector<double>& x, Vector<double>& output) const final
    {
        // Compute the pure Frobenius gradient
        coarse::frobenius::gradient(this->M_coarse, this->A_coarse,
            x, this->m_phi, this->m_w, output);

        return 0; // 0 iterations, as no Krylov solver is used
    }

    iteration::State residual(const Vector<double>& x) const final
    {
        return {.energy=value(x)};
    }
};


// TODO: Vector m_w is computed in the caller and assumed consistent (computed using same metric) as ::gradient
template <int dim>
class FrobeniusCoarseOracleEnergyAdaptive : public CoarseOracleBase<dim, FrobeniusOracle<dim>>
{
public:
    static constexpr const char* id = "FCA";
    static constexpr auto metric = MetricKind::FROBENIUS;
    MetricKind get_metric() const override { return metric; }

    FrobeniusCoarseOracleEnergyAdaptive(const GrossPitaevskiiSystem<dim>& problem,
                                        const GrossPitaevskiiSystem<dim>& problem_coarse,
                                        const ManifoldTransferBase& point_transfer,
                                        const VectorTransportBase& vector_transport,
                                        double beta, SolverOptions options, SolverOptions options_coarse)
        // Computes correction vector w in update() method
        : CoarseOracleBase<dim, FrobeniusOracle<dim>>(problem, problem_coarse,
            point_transfer, vector_transport, beta, options, options_coarse)
    {
        this->A_inv_coarse.update_static(this->problem.get_A0());
    }

    void update(const Vector<double>& x) final
    {
        CoarseOracleBase<dim, FrobeniusOracle<dim>>::update(x); // Calls assembly in parent

        this->A_inv_coarse.update_dynamic(this->A_coarse.diagonal());
    }

    double value(const Vector<double>& x) const final
    {
        const double energy = this->problem.value(x, this->beta);

        return coarse::frobenius::function_value(x, this->m_phi, this->m_w, this->M_coarse, energy);
    }

    /**
     * @brief Computes the coarse model gradient in the energy-adaptive metric.
     * $$ \nabla_A q_k(\zeta) = \tilde{A}_\zeta^{-1} \left( \nabla_F q_k(\zeta) \right) $$
     */
    unsigned gradient(const Vector<double>& x, Vector<double>& output) const final
    {
        // Computes the F-gradient and applies the A_inv preconditioner
        coarse::frobenius::energy_adaptive_gradient(this->M_coarse, this->A_inv_coarse,this->A_coarse,
            x, this->m_phi, this->m_w, output);

        // Return the number of Krylov iterations used by A_inv
        return this->A_inv_coarse.control().last_step();
    }

    iteration::State residual(const Vector<double>& x) const final
    {
        return {.energy=value(x)};
    }
};

} // namespace gpe

#endif //GPE_ORACLE_COARSE_H
