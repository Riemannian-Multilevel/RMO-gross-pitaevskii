#ifndef GPE_ORACLE_H
#define GPE_ORACLE_H

#include <gpe/lac.h>
#include <gpe/problem/gpe.h>
#include <gpe/ropt/manifold.h>

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


// TODO: refactor into UnitMassSphere (<- Manifold) and GrossPitaevskiiOracle (<- A, M and inverses)
/**
 * @brief Base Oracle for the Gross-Pitaevskii energy functional.
 * This class translates physical concepts (matrices and assembly) into optimization concepts
 * (functional values and gradients) for use in Riemannian descent algorithms.
 *
 * @tparam dim The spatial dimension of the problem.
 */
template <int dim>
class OracleBase
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
    OracleBase(const GrossPitaevskiiProblem<dim>& problem_, double beta_, SolverOptions options_)
        : problem(problem_)
        , beta(beta_)
        , options(options_)
        , A(problem.get_operator_A(beta))
        , M(problem.get_operator_M())
    {}

    virtual ~OracleBase() = default;
    virtual MetricKind get_metric() const { return MetricKind::NONE; }

    /**
     * @brief Updates the problem state and preconditioner for a new evaluation point.
     */
    virtual void update(const Vector<double>& x) const
    {
        problem.assemble_nonlinear_term(x);
    }

    /**
     * @brief Retracts a tangent vector back to the unit-mass manifold.
     * $$ R_x(z) = \frac{x + z}{\|x + z\|_M} $$
     */
    void retract(const Vector<double>& z, Vector<double>& x, double factor = 1.0) const
    {
        ellipsoid::retract_by_norm(M, z, x, factor);
    }

    void retract(const Vector<double>& z, const Vector<double>& x,
                 Vector<double>& output, double factor = 1.0) const
    {
        output = x;
        retract(z, output, factor);
    }

    // Accessors
    const auto& get_M() const { return M; }
    const auto& get_A() const { return A; }

    SolverOptions get_options() const { return options; }
    double get_beta() const { return beta; }
    unsigned n_dofs() const { return problem.n_dofs(); }

    // Pure virtual interface
    // TODO: leave `x` argument in update() exclusively, to avoid mismatches
    //       check marker `needs_assembly`
    virtual double value(const Vector<double>&) const = 0;

    // TODO: Compute directional derivative and Riemannian gradient successively
    virtual double directional_derivative(const Vector<double>&, const Vector<double>&) const
    {
        throw dealii::ExcNotImplemented();
    }
    // TODO: leave `x` argument in update() exclusively, to avoid mismatches
    //       check marker `needs_gradient
    virtual unsigned gradient(const Vector<double>&, Vector<double>&) const = 0;
    virtual iteration::State residual(const Vector<double>&) const = 0;

protected:
    const GrossPitaevskiiProblem<dim>& problem;
    double beta;
    SolverOptions options;
    OperatorType A, M;
};


template <int dim>
class MassOracle : public OracleBase<dim>
{
public:
    static constexpr const char* id = "M";
    static constexpr auto metric = MetricKind::MASS;

    MassOracle(const GrossPitaevskiiProblem<dim>& problem_, double beta_, SolverOptions options_)
        : OracleBase<dim>(problem_, beta_, options_)
        , M_inv(this->M, options_) // Specialized member
    {}

    MetricKind get_metric() const override { return metric; }

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
     * @brief Computes the Riemannian gradient in the M-metric.
     */
    unsigned gradient(const Vector<double>& x, Vector<double>& output) const override
    {
        if (m_res.residual > 0) {
            M_inv.set_tol(m_res.residual*this->options.tol_inner_res);
        }
        ellipsoid::mass::gradient(M_inv, this->A, this->M, x, output);
        return M_inv.control().last_step();
    }

    iteration::State residual(const Vector<double>& x) const override
    {
        m_res = iteration::residual(x, this->A, this->M);
        return m_res;
    }

private:
    InverseOpType M_inv;
    mutable iteration::State m_res;
};


template <int dim>
class EnergyOracle : public OracleBase<dim>
{
public:
    static constexpr const char* id = "A";
    static constexpr auto metric = MetricKind::ENERGY_ADAPTIVE;

    EnergyOracle(const GrossPitaevskiiProblem<dim>& problem_, double beta_, SolverOptions options_)
        : OracleBase<dim>(problem_, beta_, options_)
        , A_inv(this->A, options_) // Specialized member
    {
        A_inv.update_static(this->problem.get_A0());
    }

    MetricKind get_metric() const override { return metric; }

    void update(const Vector<double>& x) const override
    {
        OracleBase<dim>::update(x); // Calls assembly
        A_inv.update_dynamic(this->A.diagonal());
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
            this->A_inv.set_tol(m_res.residual*this->options.tol_inner_res);
        }
        ellipsoid::energy::gradient(this->A_inv, this->M, x, output);
        return this->A_inv.control().last_step();
    }

    iteration::State residual(const Vector<double>& x) const override
    {
        m_res = iteration::residual(x, this->A, this->M);
        return m_res;
    }

private:
    mutable InverseOpType A_inv; // Moved from Base
    mutable iteration::State m_res;
};


template <int dim>
class FrobeniusOracle : public OracleBase<dim>
{
public:
    static constexpr const char* id = "F";
    using OracleBase<dim>::OracleBase;
    static constexpr auto metric = MetricKind::FROBENIUS;

    MetricKind get_metric() const override { return metric; }

    /**
     * @brief Computes the Gross-Pitaevskii energy functional value.
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
     * @brief Computes the Riemannian gradient in the F-metric.
     * \grad_{\rm F} E^{\rm GP}(\phi) = A_{\phi}\phi - \frac{\phi^\top M A_{\phi}\phi}{\phi^\top M^2 \phi} M \phi
     */
    unsigned gradient(const Vector<double>& x, Vector<double>& output) const override
    {
        ellipsoid::frobenius::gradient(this->A, this->M, x, output);

        // F-gradient evaluation does not involve a linear solver.
        return 0;
    }

    /**
     * @brief Evaluates the current optimization state.
     */
    iteration::State residual(const Vector<double>& x) const override
    {
        m_res = iteration::residual(x, this->A, this->M, false);
        return m_res;
    }

private:
    mutable iteration::State m_res;
};


// Coarse oracles provide some additional functionality, such as a shift vector (m_w), base coarse point (m_phi)
// and methods for evaluating a coarse condition
// TODO: instead of inheriting from OracleBase (which, in its current form, is fixed to a single level of discretization),
//       define a constructor which allows a fine and coarse level of discretization.
//       Then, the tilt vector w canm be computed directly inside an (implementation of) CoarseOracleBase.
template <int dim>
class CoarseOracleBase : public OracleBase<dim>
{
public:
    static constexpr const char* id = "C";
    // TODO: abuse of notation: Galerkin condition vs. metric used for the shift w, dot product <w,.> and coarse condition
    using OracleBase<dim>::OracleBase;

    MetricKind get_metric() const override { return MetricKind::NONE; }

    void update_parameters(const Vector<double>& w_new, const Vector<double>& phi_new)
    {
        m_w = w_new;
        m_phi = phi_new;
    }

    iteration::State residual(const Vector<double>& x) const final
    {
        return {.energy=this->value(x)};
    }

protected:
    Vector<double> m_w;
    Vector<double> m_phi;
};


// TODO: Vector m_w is computed in the caller and assumed consistent (computed using same metric) as ::gradient
//       moving the computation of m_w here requires both a fine and coarse OracleBase
//       (-> GrossPitaevskiiProblem for coarse and fine discretizations)
template <int dim>
class MassCoarseOracle : public CoarseOracleBase<dim>
{
public:
    static constexpr const char* id = "MC";
    static constexpr auto metric = MetricKind::MASS;

    MassCoarseOracle(const GrossPitaevskiiProblem<dim>& problem_, double beta_, SolverOptions options_)
        : CoarseOracleBase<dim>(problem_, beta_, options_)
        , M_inv(this->M, options_) // Specialized member
    {}

    MetricKind get_metric() const override { return metric; }

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
        coarse::mass::gradient(this->M, M_inv, this->A, x, this->m_phi, this->m_w, output);

        return M_inv.control().last_step();
    }

private:
    InverseOpType M_inv;
};


// TODO: Vector m_w is computed in the caller and assumed consistent (computed using same metric) as ::gradient
template <int dim>
class MassCoarseOracleEnergyAdaptive : public CoarseOracleBase<dim>
{
public:
    static constexpr const char* id = "MCA";
    static constexpr auto metric = MetricKind::MASS;

    MassCoarseOracleEnergyAdaptive(const GrossPitaevskiiProblem<dim>& problem_, double beta_, SolverOptions options_)
        : CoarseOracleBase<dim>(problem_, beta_, options_)
        , A_inv(this->A, options_) // Specialized member
    {
        A_inv.update_static(this->problem.get_A0());
    }

    MetricKind get_metric() const override { return metric; }

    void update(const Vector<double>& x) const override
    {
        CoarseOracleBase<dim>::update(x); // Calls assembly
        A_inv.update_dynamic(this->A.diagonal());
    }

    double value(const Vector<double>& x) const final
    {
        const double energy = this->problem.value(x, this->beta);

        return coarse::mass::function_value(x, this->m_phi, this->m_w, this->M, energy);
    }

    /**
     * @brief Computes the coarse model gradient in the A-metric.
     * $$ \nabla_A q_k(x) = \nabla_A E_H(x) - w $$
     */
    unsigned gradient(const Vector<double>& x, Vector<double>& output) const final
    {
        coarse::mass::energy_adaptive_gradient(this->M, A_inv, x, this->m_phi, this->m_w, output);

        return this->A_inv.control().last_step();
    }

private:
    mutable InverseOpType A_inv;
};


// TODO: Vector m_w is computed in the caller and assumed consistent (computed using same metric) as ::gradient
template <int dim>
class FrobeniusCoarseOracle : public CoarseOracleBase<dim>
{
public:
    static constexpr const char* id = "FC";
    static constexpr auto metric = MetricKind::FROBENIUS;
    using CoarseOracleBase<dim>::CoarseOracleBase;

    MetricKind get_metric() const override { return metric; }

    double value(const Vector<double>& x) const final
    {
        const double energy = this->problem.value(x, this->beta);

        return coarse::frobenius::function_value(x, this->m_phi, this->m_w, this->M, energy);
    }

    /**
     * @brief Computes the coarse model gradient in the F-metric.
     * $$ \nabla_F q_k(\zeta) = \Pi_{\zeta, F}\left(A_\zeta \zeta - \frac{1}{\phi^\top M\zeta}w\right) $$
     */
    unsigned gradient(const Vector<double>& x, Vector<double>& output) const final
    {
        // Compute the pure Frobenius gradient
        coarse::frobenius::gradient(this->M, this->A, x, this->m_phi, this->m_w, output);

        return 0; // 0 iterations, as no Krylov solver is used
    }
};


// TODO: Vector m_w is computed in the caller and assumed consistent (computed using same metric) as ::gradient
template <int dim>
class FrobeniusCoarseOracleEnergyAdaptive : public CoarseOracleBase<dim>
{
public:
    static constexpr const char* id = "FCA";
    static constexpr auto metric = MetricKind::FROBENIUS;

    FrobeniusCoarseOracleEnergyAdaptive(const GrossPitaevskiiProblem<dim>& problem_, double beta_, SolverOptions options_)
        : CoarseOracleBase<dim>(problem_, beta_, options_)
        , A_inv(this->A, options_) // Specialized member
    {
            A_inv.update_static(this->problem.get_A0());
    }

    MetricKind get_metric() const override { return metric; }

    void update(const Vector<double>& x) const override
    {
        CoarseOracleBase<dim>::update(x); // Calls assembly
        A_inv.update_dynamic(this->A.diagonal());
    }

    double value(const Vector<double>& x) const final
    {
        const double energy = this->problem.value(x, this->beta);

        return coarse::frobenius::function_value(x, this->m_phi, this->m_w, this->M, energy);
    }

    /**
     * @brief Computes the coarse model gradient in the energy-adaptive metric.
     * $$ \nabla_A q_k(\zeta) = \tilde{A}_\zeta^{-1} \left( \nabla_F q_k(\zeta) \right) $$
     */
    unsigned gradient(const Vector<double>& x, Vector<double>& output) const final
    {
        // Computes the F-gradient and applies the A_inv preconditioner
        coarse::frobenius::energy_adaptive_gradient(this->M, this->A_inv,this->A,
            x, this->m_phi, this->m_w, output);

        // Return the number of Krylov iterations used by A_inv
        return this->A_inv.control().last_step();
    }

private:
    mutable InverseOpType A_inv;
};

} // namespace gpe

#endif //GPE_ORACLE_H