#ifndef GPE_ORACLE_H
#define GPE_ORACLE_H

#include <gpe/lac.h>
#include <gpe/problem/gpe.h>
#include <gpe/ropt/manifold.h>
#include <gpe/ropt/transport.h>

namespace gpe
{
/**
 * @brief Functor computing the square of the Euclidean norm of a point.
 *
 * Computes \f$ f(p) = \sum_{d=0}^{dim-1} p_d^2 \f$.
 * Used primarily for initializing test cases or potentials.
 *
 * @tparam dim The spatial dimension of the point.
 */
template <int dim>
class Square
{
public:
    double operator()(const Point<dim>& p) const {
        typename Point<dim>::value_type out = 0.0;
        for (unsigned d = 0; d < dim; d++) {
            out += p[d]*p[d];
        }
        return out;
    }
};


// TODO: consolidate with manifold.h (namespace gpe::ellipsoid)
class ManifoldBase
{
public:
    virtual ~ManifoldBase() = default;

    virtual void retract(const Vector<double>& z, Vector<double>& x, double factor) const = 0;
    virtual void retract(const Vector<double>& z, Vector<double>& x, Vector<double>& output, double factor) = 0;

    virtual void retract_inv(Vector<double>& v, const Vector<double>& x) const = 0;

    virtual void retract_diff(const Vector<double>& x, const Vector<double>& v,
        const Vector<double>& w, Vector<double>& output) const = 0;

    virtual void retract_inv_diff(const Vector<double>& x, const Vector<double>& zeta,
        const Vector<double>& u, Vector<double>& output) const = 0;

    virtual void retract_inv_diff_adjoint(const Vector<double>& x, const Vector<double>& zeta,
        const Vector<double>& u, Vector<double>& output) const = 0;
};


// TODO: consolidate with manifold.h (namespace gpe::ellipsoid)
template <int dim>
class UnitMassSphere : ManifoldBase
{
public:
    UnitMassSphere(const OperatorType& M) : M(M) {}

    /**
     * @brief Retracts a tangent vector back to the unit-mass manifold.
     * $$ R_x(z) = \frac{x + z}{\|x + z\|_M} $$
     */
    void retract(const Vector<double>& z, Vector<double>& x, double factor = 1.0) const override
    {
        ellipsoid::retract_by_norm(M, z, x, factor);
    }

    void retract(const Vector<double>& z, const Vector<double>& x,
                 Vector<double>& output, double factor = 1.0) const override
    {
        output = x;
        retract(z, output, factor);
    }

    void retract_diff(const Vector<double>& x, const Vector<double>& v, const Vector<double>& w,
                      Vector<double>& output) const override
    {
        ellipsoid::retract_diff_by_norm(M, x, v, w, output);
    }

    void retract_inv(Vector<double>& v, const Vector<double>& x) const override
    {
        ellipsoid::retract_inv_by_norm(M, v, x);
    }

    void retract_inv_diff(const Vector<double>& x, const Vector<double>& zeta, const Vector<double>& u,
                          Vector<double>& output) const override
    {
        ellipsoid::retract_inv_diff_by_norm(M, x, zeta, u, output);
    }

    void retract_inv_diff_adjoint(const Vector<double>& x, const Vector<double>& zeta, const Vector<double>& u,
                                  Vector<double>& output) const override
    {
        ellipsoid::retract_inv_diff_by_norm_adjoint(M, x, zeta, u, output);
    }

    // Accessors
    const auto& get_M() const { return M; }

private:
    const OperatorType& M;
};


// Basic oracle interface
class OracleBase
{
public:
    virtual ~OracleBase() = default;
    // TODO: MetricBase
    virtual MetricKind get_metric() const { return MetricKind::NONE; }

    virtual void update(const Vector<double>& x) = 0;

    // TODO: leave `x` argument in update() exclusively, to avoid mismatches
    //       check marker `needs_assembly`
    virtual double value(const Vector<double>&) const = 0;

    // TODO: Compute directional derivative and Riemannian gradient successively
    virtual double directional_derivative(const Vector<double>&, const Vector<double>&) const = 0;

    // TODO: leave `x` argument in update() exclusively, to avoid mismatches
    //       check marker `needs_gradient
    virtual unsigned gradient(const Vector<double>&, Vector<double>&) const = 0;
    virtual iteration::State residual(const Vector<double>&) const = 0;
};


// TODO: pure abstract base class for OracleBase (store 1 level of discretization) and CoarseOracleBase (store multiple)
//       refactor into UnitMassSphere (<- Manifold) and GrossPitaevskiiOracle (<- A, M and inverses)
/**
 * @brief Base Oracle for the Gross-Pitaevskii energy functional.
 * This class translates physical concepts (matrices and assembly) into optimization concepts
 * (functional values and gradients) for use in Riemannian descent algorithms.
 *
 * @tparam dim The spatial dimension of the problem.
 */
template <int dim>
class GrossPitaevskiiOracle : OracleBase
{
public:
    static constexpr int dimension = dim;
    static constexpr const char* id = "";

    /**
     * @brief Constructs the Oracle by referencing an existing GPE problem.
     * @note The Oracle holds a reference; the Problem object must outlive this Oracle.
     */
    // The discretization GrossPitaevskiiProblem is used to define (references to) the matrices used, being
    //  A, M, A_inv, M_inv, as well as updating the underlying state for new incumbent solutions x -> A_x -> A_x_inv.
    // TODO: separate the problem state from the geometry (retract, ...) and value evaluation (value, gradient, ...)
    GrossPitaevskiiOracle(const GrossPitaevskiiProblem<dim>& problem_, double beta_, SolverOptions options_)
        : problem(problem_)
        , beta(beta_)
        , options(options_)
        , A(problem.get_operator_A(beta))
        , M(problem.get_operator_M())
    {}

    /**
     * @brief Updates the problem state and preconditioner for a new evaluation point.
     */
    void update(const Vector<double>& x) override
    {
        problem.assemble_nonlinear_term(x);  // mutable
        // No other state but OracleBase assumes non-const method
    }

    // Accessors
    const auto& get_M() const { return M; }
    const auto& get_A() const { return A; }

    SolverOptions get_options() const { return options; }
    double get_beta() const { return beta; }
    unsigned n_dofs() const { return problem.n_dofs(); }


protected:
    // Finite element discretization
    const GrossPitaevskiiProblem<dim>& problem;
    // Problem parameters
    double beta;
    SolverOptions options;
};


template <int dim>
class MassOracle : public GrossPitaevskiiOracle<dim>
{
public:
    static constexpr const char* id = "M";
    static constexpr auto metric = MetricKind::MASS;

    MassOracle(const GrossPitaevskiiProblem<dim>& problem, double beta, SolverOptions options)
        : GrossPitaevskiiOracle<dim>(problem, beta, options)
        , A(this->problem.get_operator_A(beta))
        , M(this->problem.get_operator_M())
        , M_inv(M, options) // Specialized member
    {}

    MetricKind get_metric() const override { return metric; }

    /**
      * @brief Computes the Gross-Pitaevskii energy functional value.
      * $$ E(\phi) = \langle \phi, A_0 \phi \rangle + \frac{\beta}{2} \langle \phi, M_{pp}(\phi) \phi \rangle $$
    */
    double value(const Vector<double>& x) const override
    {
        return this->problem.value(x, this->beta);  // metric-independent
    }

    double directional_derivative(const Vector<double>& x, const Vector<double>& z) const override
    {
        return this->problem.directional_derivative(x, z, this->beta);  // metric-independent
    }

    /**
     * @brief Computes the Riemannian gradient in the M-metric.
     */
    unsigned gradient(const Vector<double>& x, Vector<double>& output) const override
    {
        if (m_res.residual > 0) {
            M_inv.set_tol(m_res.residual*this->options.tol_inner_res);
        }
        ellipsoid::mass::gradient(M_inv, A, M, x, output);
        return M_inv.control().last_step();
    }

    iteration::State residual(const Vector<double>& x) const override
    {
        m_res = iteration::residual(A, M, x);
        return m_res;
    }

private:
    OperatorType A, M;
    InverseOpType M_inv;

    mutable iteration::State m_res;
};


template <int dim>
class EnergyOracle : public GrossPitaevskiiOracle<dim>
{
public:
    static constexpr const char* id = "A";
    static constexpr auto metric = MetricKind::ENERGY_ADAPTIVE;
    MetricKind get_metric() const override { return metric; }

    EnergyOracle(const GrossPitaevskiiProblem<dim>& problem, double beta, SolverOptions options)
        : GrossPitaevskiiOracle<dim>(problem, beta, options)
        , M(this->problem.get_operator_M())
        , A(this->problem.get_operator_A(beta))
        , A_inv(A, options) // Specialized member
    {
        A_inv.update_static(this->problem.get_A0());
    }

    // Additional method for updating (Jacobi) preconditioner for new x
    void update(const Vector<double>& x) override
    {
        GrossPitaevskiiOracle<dim>::update(x); // Calls assembly in parent class
        A_inv.update_dynamic(A.diagonal());
    }

    /**
     * @brief Computes the Gross-Pitaevskii energy functional value.
     * $$ E(\phi) = \langle \phi, A_0 \phi \rangle + \frac{\beta}{2} \langle \phi, M_{pp}(\phi) \phi \rangle $$
     */
    double value(const Vector<double>& x) const override
    {
        return this->problem.value(x, this->beta);
    }

    // Metric-free implementation
    double directional_derivative(const Vector<double>& x, const Vector<double>& z) const override
    {
        return this->problem.directional_derivative(x, z, this->beta);
    }

    /**
     * @brief Computes the Riemannian gradient in the A-metric.
     * Solves the inner linear system $ A^{-1} \nabla E $ using the PreconditionInverse wrapper.
     */
    unsigned gradient(const Vector<double>& x, Vector<double>& output) const override
    {
        if (m_res.residual > 0) {
            A_inv.set_tol(m_res.residual*this->options.tol_inner_res);
        }
        ellipsoid::energy::gradient(A_inv, M, x, output);
        return A_inv.control().last_step();
    }

    iteration::State residual(const Vector<double>& x) const override
    {
        m_res = iteration::residual(A, M, x);  // TODO: less general namespace?
        return m_res;
    }

private:
    OperatorType M, A;
    InverseOpType A_inv;
    mutable iteration::State m_res;
};


template <int dim>
class FrobeniusOracle : public GrossPitaevskiiOracle<dim>
{
public:
    static constexpr const char* id = "F";
    static constexpr auto metric = MetricKind::FROBENIUS;
    MetricKind get_metric() const override { return metric; }

    FrobeniusOracle(const GrossPitaevskiiProblem<dim>& problem, double beta, SolverOptions options)
        : GrossPitaevskiiOracle<dim>(problem, beta, options)
        , M(this->problem.get_operator_M())
        , A(this->problem.get_operator_A(beta))
    {}

    /**
     * @brief Computes the Gross-Pitaevskii energy functional value.
     */
    double value(const Vector<double>& x) const override
    {
        return this->problem.value(x, this->beta);  // metric-independent
    }

    double directional_derivative(const Vector<double>& x, const Vector<double>& z) const override
    {
        return this->problem.directional_derivative(x, z, this->beta);  // metric-independent
    }

    /**
     * @brief Computes the Riemannian gradient in the F-metric.
     * \grad_{\rm F} E^{\rm GP}(\phi) = A_{\phi}\phi - \frac{\phi^\top M A_{\phi}\phi}{\phi^\top M^2 \phi} M \phi
     */
    unsigned gradient(const Vector<double>& x, Vector<double>& output) const override
    {
        ellipsoid::frobenius::gradient(A, M, x, output);

        // F-gradient evaluation does not involve a linear solver.
        return 0;
    }

    /**
     * @brief Evaluates the current optimization state.
     */
    iteration::State residual(const Vector<double>& x) const override
    {
        m_res = iteration::residual(A, M, x, false);
        return m_res;
    }

private:
    OperatorType M, A;
    mutable iteration::State m_res;
};


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
    GrossPitaevskiiCoarseOracle(const GrossPitaevskiiProblem<dim>& problem,
                                const GrossPitaevskiiProblem<dim>& problem_coarse,
                                const ManifoldTransferBase& point_transfer,
                                const VectorTransportBase& vector_transport,
                                double beta, SolverOptions options, SolverOptions options_coarse)
        : problem(problem), problem_coarse(problem_coarse), beta(beta)
        , options(options), options_coarse(options_coarse)
        , n(problem.n_dofs())
        , n_coarse(problem_coarse.n_dofs())
        , O(problem, beta, options)
        , O_coarse(problem_coarse, beta, options_coarse)
        , point_transfer(point_transfer)
        , vector_transport(vector_transport)
        , m_y(n_coarse)
        , m_y_grad(n_coarse)
        , m_x_grad(n)
        , m_x_grad_restr(n_coarse)
        , m_w(n_coarse)
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
        point_transfer.restriction(x, m_y);

#ifdef CPU_TIME
        std::cerr << "[" << timer.cpu_time() << "] coarse: assemble matrix\n";
#endif
        problem_coarse.assemble_nonlinear_term(m_y);

        // Compute coarse M-gradient
#ifdef CPU_TIME
        std::cerr << "[" << timer.cpu_time() << "] coarse: " << O_coarse.id << "-coarse gradient\n";
#endif
        O_coarse.gradient(m_y, m_y_grad);

        // Compute fine M-gradient
#ifdef CPU_TIME
        std::cerr << "[" << timer.cpu_time() << "] coarse: " << O_fine.id << "-fine gradient\n";
#endif
        O.gradient(x, m_x_grad);

        // Compute restricted gradient
#ifdef CPU_TIME
        std::cerr << "[" << timer.cpu_time() << "] coarse: M-vector restriction\n";
#endif
        vector_transport.vector_restriction(m_y, x, m_x_grad, m_x_grad_restr);

        // Compute correction term
        m_w = m_y_grad;
        m_w.add(-1.0, m_x_grad_restr);
    }

    iteration::State residual(const Vector<double>& x) const final
    {
        return {.energy=value(x)};
    }

private:
    // Finite element discretization for fine and coarse mesh
    const GrossPitaevskiiProblem<dim>& problem, problem_coarse;

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
    Vector<double> m_y;             // restricted point (base point for coarse model)
    Vector<double> m_y_grad;        // gradient of restricted point
    Vector<double> m_x_grad;        // fine gradient
    Vector<double> m_x_grad_restr;  // restricted gradient
    Vector<double> m_w;             // correction vector

    // Benchmarking
    mutable dealii::Timer timer;
};


// TODO: Vector m_w is computed in the caller and assumed consistent (computed using same metric) as ::gradient
//       moving the computation of m_w here requires both a fine and coarse OracleBase
//       (-> GrossPitaevskiiProblem for coarse and fine discretizations)
template <int dim>
class MassCoarseOracle : public GrossPitaevskiiCoarseOracle<dim, MassOracle<dim>>
{
public:
    static constexpr const char* id = "MC";
    static constexpr auto metric = MetricKind::MASS;
    MetricKind get_metric() const override { return metric; }

    MassCoarseOracle(const GrossPitaevskiiProblem<dim>& problem,
                     const GrossPitaevskiiProblem<dim>& problem_coarse,
                     const ManifoldTransferBase& point_transfer,
                     const VectorTransportBase& vector_transport,
                     double beta, SolverOptions options, SolverOptions options_coarse)
        // Computes correction vector w in update() method
        : GrossPitaevskiiCoarseOracle<dim, MassOracle<dim>>(
            problem, problem_coarse, point_transfer, vector_transport, beta, options, options_coarse)

        // Evaluate gradient and value
        , M_coarse(problem_coarse.get_operator_M())
        , A_coarse(problem_coarse.get_operator_A(beta))  // or OracleBase::get_operator_A
        , M_inv_coarse(M_coarse, options_coarse)
    {}

    double value(const Vector<double>& x) const final
    {
        const double energy = this->problem.value(x, this->beta);

        return coarse::mass::function_value(x, this->m_phi, this->m_w, this->M, energy);
    }

    /**
     * @brief Computes the coarse model gradient in the M-metric.
     */
    unsigned gradient(const Vector<double>& x, Vector<double>& output) const final
    {
        coarse::mass::gradient(M_coarse, M_inv_coarse, A_coarse, x, this->m_phi, this->m_w, output);

        return M_inv_coarse.control().last_step();
    }

private:
    OperatorType M_coarse, A_coarse;
    InverseOpType M_inv_coarse;
};


template <int dim>
class MassCoarseOracleEnergyAdaptive : public GrossPitaevskiiCoarseOracle<dim, MassOracle<dim>>
{
public:
    static constexpr const char* id = "MCA";
    static constexpr auto metric = MetricKind::MASS;
    MetricKind get_metric() const override { return metric; }

    MassCoarseOracleEnergyAdaptive(const GrossPitaevskiiProblem<dim>& problem,
                                   const GrossPitaevskiiProblem<dim>& problem_coarse,
                                   const ManifoldTransferBase& point_transfer,
                                   const VectorTransportBase& vector_transport,
                                   double beta, SolverOptions options, SolverOptions options_coarse)
        // Computes correction vector w in update() method
        : GrossPitaevskiiCoarseOracle<dim, MassOracle<dim>>(
            problem, problem_coarse, point_transfer, vector_transport, beta, options, options_coarse)

        // Evaluate gradient and value
        , M_coarse(problem_coarse.get_operator_M())
        , A_coarse(problem_coarse.get_operator_A(beta))  // or OracleBase::get_operator_A
        , A_inv_coarse(A_coarse, options_coarse)
    {
        A_inv_coarse.update_static(this->problem_coarse.get_A0());
    }

    void update(const Vector<double>& x) final
    {
        GrossPitaevskiiCoarseOracle<dim, MassOracle<dim>>::update(x); // Calls assembly in parent

        A_inv_coarse.update_dynamic(A_coarse.diagonal());
    }

    double value(const Vector<double>& x) const final
    {
        const double energy = this->problem.value(x, this->beta);

        return coarse::mass::function_value(x, this->m_phi, this->m_w, M_coarse, energy);
    }

    /**
     * @brief Computes the coarse model gradient in the A-metric.
     * $$ \nabla_A q_k(x) = \nabla_A E_H(x) - w $$
     */
    unsigned gradient(const Vector<double>& x, Vector<double>& output) const final
    {
        coarse::mass::energy_adaptive_gradient(M_coarse, A_inv_coarse, x, this->m_phi, this->m_w, output);

        return A_inv_coarse.control().last_step();
    }

private:
    OperatorType M_coarse, A_coarse;
    InverseOpType A_inv_coarse;
};


template <int dim>
class FrobeniusCoarseOracle : public GrossPitaevskiiCoarseOracle<dim, FrobeniusOracle<dim>>
{
public:
    static constexpr const char* id = "FC";
    static constexpr auto metric = MetricKind::FROBENIUS;
    MetricKind get_metric() const override { return metric; }

    FrobeniusCoarseOracle(const GrossPitaevskiiProblem<dim>& problem,
                          const GrossPitaevskiiProblem<dim>& problem_coarse,
                          const ManifoldTransferBase& point_transfer,
                          const VectorTransportBase& vector_transport,
                          double beta, SolverOptions options, SolverOptions options_coarse)
        // Computes correction vector w in update() method
        : GrossPitaevskiiCoarseOracle<dim, FrobeniusOracle<dim>>(
            problem, problem_coarse, point_transfer, vector_transport, beta, options, options_coarse)

        // Evaluate gradient and value
        , M_coarse(problem_coarse.get_operator_M())
        , A_coarse(problem_coarse.get_operator_A(beta))
    {}

    double value(const Vector<double>& x) const final
    {
        const double energy = this->problem.value(x, this->beta);

        return coarse::frobenius::function_value(x, this->m_phi, this->m_w, M_coarse, energy);
    }

    /**
     * @brief Computes the coarse model gradient in the F-metric.
     * $$ \nabla_F q_k(\zeta) = \Pi_{\zeta, F}\left(A_\zeta \zeta - \frac{1}{\phi^\top M\zeta}w\right) $$
     */
    unsigned gradient(const Vector<double>& x, Vector<double>& output) const final
    {
        // Compute the pure Frobenius gradient
        coarse::frobenius::gradient(M_coarse, A_coarse, x, this->m_phi, this->m_w, output);

        return 0; // 0 iterations, as no Krylov solver is used
    }

private:
    OperatorType M_coarse, A_coarse;
};


// TODO: Vector m_w is computed in the caller and assumed consistent (computed using same metric) as ::gradient
template <int dim>
class FrobeniusCoarseOracleEnergyAdaptive : public GrossPitaevskiiCoarseOracle<dim, FrobeniusOracle<dim>>
{
public:
    static constexpr const char* id = "FCA";
    static constexpr auto metric = MetricKind::FROBENIUS;
    MetricKind get_metric() const override { return metric; }

    FrobeniusCoarseOracleEnergyAdaptive(const GrossPitaevskiiProblem<dim>& problem,
                                        const GrossPitaevskiiProblem<dim>& problem_coarse,
                                        const ManifoldTransferBase& point_transfer,
                                        const VectorTransportBase& vector_transport,
                                        double beta, SolverOptions options, SolverOptions options_coarse)
        // Computes correction vector w in update() method
        : GrossPitaevskiiCoarseOracle<dim, FrobeniusOracle<dim>>(
            problem, problem_coarse, point_transfer, vector_transport, beta, options, options_coarse)

        // Evaluate gradient and value
        , M_coarse(problem_coarse.get_operator_M())
        , A_coarse(problem_coarse.get_operator_A(beta))
        , A_inv_coarse(A_coarse, options_coarse)
    {
        A_inv_coarse.update_static(this->problem.get_A0());
    }

    void update(const Vector<double>& x) final
    {
        GrossPitaevskiiCoarseOracle<dim, FrobeniusOracle<dim>>::update(x); // Calls assembly in parent

        A_inv_coarse.update_dynamic(A_coarse.diagonal());
    }

    double value(const Vector<double>& x) const final
    {
        const double energy = this->problem.value(x, this->beta);

        return coarse::frobenius::function_value(x, this->m_phi, this->m_w, M_coarse, energy);
    }

    /**
     * @brief Computes the coarse model gradient in the energy-adaptive metric.
     * $$ \nabla_A q_k(\zeta) = \tilde{A}_\zeta^{-1} \left( \nabla_F q_k(\zeta) \right) $$
     */
    unsigned gradient(const Vector<double>& x, Vector<double>& output) const final
    {
        // Computes the F-gradient and applies the A_inv preconditioner
        coarse::frobenius::energy_adaptive_gradient(M_coarse, A_inv_coarse,A_coarse,
            x, this->m_phi, this->m_w, output);

        // Return the number of Krylov iterations used by A_inv
        return A_inv_coarse.control().last_step();
    }

private:
    OperatorType M_coarse, A_coarse;
    InverseOpType A_inv_coarse;
};

} // namespace gpe

#endif //GPE_ORACLE_H