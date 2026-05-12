//
// Created by Ferdinand Vanmaele on 12.05.26.
//

#ifndef GPE_ORACLE_COARSE_H
#define GPE_ORACLE_COARSE_H

#include <gpe/problem/oracle.h>

namespace gpe
{

// The gradient of the coarse model is used to find a descent direction dk.
// Therefore, CoarseOracleBase should include update(), gradient() and (for line search) value methods.
// A full approximation / multilevel scheme should build the descent direction based on this:
//    grad (coarse) -> retract_inv_by_norm -> vector_prolongation
// TODO: for a multilevel implementation, O(_coarse) need to be set to previous levels
//   -> define a seperate constructor taking O, O_coarse as arguments (instead of problem(_coarse) for initialization)
//      this assumes TiltOracle is a wrapper containing no heavy objects as state
template <int dim, typename TiltOracle>
class GrossPitaevskiiCoarseOracle : OracleBase
{
public:
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

    GrossPitaevskiiCoarseOracle(const GrossPitaevskiiSystem<dim>& problem,
                                const GrossPitaevskiiSystem<dim>& problem_coarse,
                                const ManifoldTransferBase& point_transfer,
                                const VectorTransportBase& vector_transport,
                                double beta, SolverOptions options, SolverOptions options_coarse)
        : problem(problem), problem_coarse(problem_coarse), beta(beta)
        , options(options), options_coarse(options_coarse)

    // Problem evaluation
        , n(problem.n_dofs())
        , n_coarse(problem_coarse.n_dofs())
        , O(problem, beta, options)
        , O_coarse(problem_coarse, beta, options_coarse)

    // Grid transfer
        , point_transfer(point_transfer)
        , vector_transport(vector_transport)

    // Coarse parameters
        , m_state(n, n_coarse)

    // Linear operators
        , M_coarse(problem_coarse.get_operator_M())
        , A_coarse(problem_coarse.get_operator_A(beta))  // or OracleBase::get_operator_A
        , M_inv_coarse(M_coarse, options_coarse)
        , A_inv_coarse(A_coarse, options_coarse)
        , M(problem.get_operator_M())
    {}

    // TODO: conditional define of problem?
    //   problem.assemble_nonlinear_term(x) -> O.update(x)
    //   problem_coarse.assemble_nonlinear_term(y) -> O_coarse.update(y)
    // CoarseOracleBase(const TiltOracle& O, const TiltOracle& O_coarse,
    //                  const ManifoldTransferBase& point_transfer,
    //                  const VectorTransportBase& vector_transport,
    //                  double beta, SolverOptions options, SolverOptions options_coarse)
    // {}

    void set_timer(const dealii::Timer& timer_new) const
    {
        timer = timer_new;
    }

    // Compute parameters for coarse model
    void update(const Vector<double>& x) override
    {
        AssertDimension(x.size(), n);

        // Update state for fine oracle
#ifdef CPU_TIME
        std::cerr << "[" << timer.cpu_time() << "] fine: assemble matrix\n";
#endif
        problem.assemble_nonlinear_term(x);

        // Set base point for coarse model
#ifdef CPU_TIME
        std::cerr << "[" << timer.cpu_time() << "] coarse: point transfer\n";
#endif
        point_transfer.restriction(x, m_state.y);

#ifdef CPU_TIME
        std::cerr << "[" << timer.cpu_time() << "] coarse: assemble matrix\n";
#endif
        problem_coarse.assemble_nonlinear_term(m_state.y);

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

    const CoarseState& get_state() const
    {
        return m_state;
    }

    double norm(const Vector<double>& v) const override
    {
        return TiltOracle::norm(v);
    }

    iteration::State residual(const Vector<double>& x) const final
    {
        return {.energy=value(x)};
    }

private:
    // Finite element discretization for fine and coarse mesh
    const GrossPitaevskiiSystem<dim>& problem, problem_coarse;

    // Problem parameters for fine and coarse objective
    double beta;
    SolverOptions options, options_coarse;
    unsigned n, n_coarse;

    // Coarse and fine level evaluation for correction vector w
    // Includes references to matrices/linear operators (A, M)
    TiltOracle O, O_coarse;

    // Operators for transferring solutions and gradients
    const ManifoldTransferBase& point_transfer;
    const VectorTransportBase& vector_transport;

    // Coarse model parameters
    CoarseState m_state;

    // Linear operators (coarse)
    OperatorType M_coarse, A_coarse;    // for evaluation value and gradient of coarse model
    InverseOpType M_inv_coarse, A_inv_coarse;

    // Linear operators (fine)
    OperatorType M;  // for evaluation of coarse condition

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
class MassCoarseOracle : public GrossPitaevskiiCoarseOracle<dim, MassOracle<dim>>
{
public:
    static constexpr const char* id = "MC";

    using GrossPitaevskiiCoarseOracle<dim, MassOracle<dim>>::GrossPitaevskiiCoarseOracle;

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
            x, this->m_phi, this->m_w, output);

        return this->M_inv_coarse.control().last_step();
    }
};


template <int dim>
class MassCoarseOracleEnergyAdaptive : public GrossPitaevskiiCoarseOracle<dim, MassOracle<dim>>
{
public:
    static constexpr const char* id = "MCA";

    MassCoarseOracleEnergyAdaptive(const GrossPitaevskiiSystem<dim>& problem,
                                   const GrossPitaevskiiSystem<dim>& problem_coarse,
                                   const ManifoldTransferBase& point_transfer,
                                   const VectorTransportBase& vector_transport,
                                   double beta, SolverOptions options, SolverOptions options_coarse)
        : GrossPitaevskiiCoarseOracle<dim, MassOracle<dim>>(problem, problem_coarse,
            point_transfer, vector_transport, beta, options, options_coarse)
    {
        this->A_inv_coarse.update_static(this->problem_coarse.get_A0());
    }

    void update(const Vector<double>& x) final
    {
        GrossPitaevskiiCoarseOracle<dim, MassOracle<dim>>::update(x); // Calls assembly in parent

        this->A_inv_coarse.update_dynamic(this->A_coarse.diagonal());
    }

    double value(const Vector<double>& x) const final
    {
        const double energy = this->problem.value(x, this->beta);

        return coarse::mass::function_value(x, this->m_phi, this->m_w, this->M_coarse, energy);
    }

    /**
     * @brief Computes the coarse model gradient in the A-metric.
     * $$ \nabla_A q_k(x) = \nabla_A E_H(x) - w $$
     */
    unsigned gradient(const Vector<double>& x, Vector<double>& output) const final
    {
        coarse::mass::energy_adaptive_gradient(this->M_coarse, this->A_inv_coarse,
            x, this->m_phi, this->m_w, output);

        return this->A_inv_coarse.control().last_step();
    }
};


template <int dim>
class FrobeniusCoarseOracle : public GrossPitaevskiiCoarseOracle<dim, FrobeniusOracle<dim>>
{
public:
    static constexpr const char* id = "FC";

    using GrossPitaevskiiCoarseOracle<dim, FrobeniusOracle<dim>>::GrossPitaevskiiCoarseOracle;

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
};


// TODO: Vector m_w is computed in the caller and assumed consistent (computed using same metric) as ::gradient
template <int dim>
class FrobeniusCoarseOracleEnergyAdaptive : public GrossPitaevskiiCoarseOracle<dim, FrobeniusOracle<dim>>
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
        : GrossPitaevskiiCoarseOracle<dim, FrobeniusOracle<dim>>(problem, problem_coarse,
            point_transfer, vector_transport, beta, options, options_coarse)
    {
        this->A_inv_coarse.update_static(this->problem.get_A0());
    }

    void update(const Vector<double>& x) final
    {
        GrossPitaevskiiCoarseOracle<dim, FrobeniusOracle<dim>>::update(x); // Calls assembly in parent

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
};

} // namespace gpe

#endif //GPE_ORACLE_COARSE_H
