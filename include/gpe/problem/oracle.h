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


// TODO: refactor into UnitMassSphere (<- Manifold) and Oracle
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
    using OperatorType  = LinearCombination<SparseMatrix<double>, Vector<double>>;
    using InverseOpType = PreconditionInverse<OperatorType, SparseMatrix<double>>;

    /**
     * @brief Constructs the Oracle by referencing an existing GPE problem.
     * @note The Oracle holds a reference; the Problem object must outlive this Oracle.
     */
    OracleBase(const GrossPitaevskiiProblem<dim>& problem_, double beta_, SolverOptions options_)
        : problem(problem_)
        , beta(beta_)
        , options(options_)
        , A(problem.get_operator_A(beta))
        , M(problem.get_operator_M())
        , A_inv(A, options_)
        , M_inv(M, options_)
    {
        A_inv.update_static(problem.get_A0());
    }

    virtual ~OracleBase() = default;

    /**
     * @brief Updates the problem state and preconditioner for a new evaluation point.
     * @note This method is NOT const, as it mutates the internal operator state.
     */
    void update(const Vector<double>& x)
    {
        problem.assemble_nonlinear_term(x);
        A_inv.update_dynamic(A.diagonal());
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
    virtual double directional_derivative(const Vector<double>&, const Vector<double>&, const Vector<double>&) const
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
    InverseOpType A_inv, M_inv;
};


template <int dim>
class MassOracle : public OracleBase<dim>
{
public:
    static constexpr const char* id = "M";
    using OracleBase<dim>::OracleBase;

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
            this->M_inv.set_tol(m_res.residual*this->options.tol_inner_res);
        }
        ellipsoid::mass::gradient(this->M_inv, this->A, this->M, x, output);
        return this->M_inv.control().last_step();
    }

    iteration::State residual(const Vector<double>& x) const override
    {
        m_res = iteration::residual(x, this->A, this->M);
        return m_res;
    }

private:
    mutable iteration::State m_res;
};


template <int dim>
class EnergyOracle : public OracleBase<dim>
{
public:
    static constexpr const char* id = "A";
    // Inherit constructors from OracleBase
    using OracleBase<dim>::OracleBase;

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
    mutable iteration::State m_res;
};


template <int dim>
class FrobeniusOracle : public OracleBase<dim>
{
public:
    static constexpr const char* id = "F";
    using OracleBase<dim>::OracleBase;

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


// TODO: use tag dispatch for gradient metric
template <int dim>
class MassCoarseOracle : public OracleBase<dim>
{
public:
    static constexpr const char* id = "MC";
    using OracleBase<dim>::OracleBase;

    void update_parameters(const Vector<double>& w_new, const Vector<double>& phi_new)
    {
        m_w = w_new;
        m_phi = phi_new;
    }

    double value(const Vector<double>& x) const override
    {
        const double energy = this->problem.value(x, this->beta);

        return coarse::mass::function_value(x, m_phi, m_w, this->M, energy);
    }

    /**
     * @brief Computes the coarse model gradient in the M-metric.
     */
    unsigned gradient(const Vector<double>& x, Vector<double>& output) const override
    {
        coarse::mass::gradient(this->M, this->M_inv, this->A, x, m_phi, m_w, output);

        return this->M_inv.control().last_step();
    }

    iteration::State residual(const Vector<double>& x) const override
    {
        return {.energy=value(x)};
    }

private:
    Vector<double> m_w;
    Vector<double> m_phi;
    mutable double last_grad_norm = 0.0;
};


// TODO: use tag dispatch for gradient metric
template <int dim>
class MassCoarseOracleEnergyAdaptive : public OracleBase<dim>
{
public:
    static constexpr const char* id = "MCA";
    using OracleBase<dim>::OracleBase;

    void update_parameters(const Vector<double>& w_new, const Vector<double>& phi_new)
    {
        m_w = w_new;
        m_phi = phi_new;
    }

    double value(const Vector<double>& x) const override
    {
        const double energy = this->problem.value(x, this->beta);

        return coarse::mass::function_value(x, m_phi, m_w, this->M, energy);
    }

    /**
     * @brief Computes the coarse model gradient in the A-metric.
     * $$ \nabla_A q_k(x) = \nabla_A E_H(x) - w $$
     */
    unsigned gradient(const Vector<double>& x, Vector<double>& output) const override
    {
        coarse::mass::energy_adaptive_gradient(this->M, this->A_inv, x, m_phi, m_w, output);

        return this->A_inv.control().last_step();
    }

    iteration::State residual(const Vector<double>& x) const override
    {
        return {.energy=value(x)};
    }

private:
    Vector<double> m_w;
    Vector<double> m_phi;
};


// TODO: use tag dispatch for gradient metric
template <int dim>
class FrobeniusCoarseOracle : public OracleBase<dim>
{
public:
    static constexpr const char* id = "FC";
    using OracleBase<dim>::OracleBase;

    void update_parameters(const Vector<double>& w_new, const Vector<double>& phi_new)
    {
        m_w = w_new;
        m_phi = phi_new;
    }

    double value(const Vector<double>& x) const override
    {
        const double energy = this->problem.value(x, this->beta);

        return coarse::frobenius::function_value(x, m_phi, m_w, this->M, energy);
    }

    /**
     * @brief Computes the coarse model gradient in the F-metric.
     * $$ \nabla_F q_k(\zeta) = \Pi_{\zeta, F}\left(A_\zeta \zeta - \frac{1}{\phi^\top M\zeta}w\right) $$
     */
    unsigned gradient(const Vector<double>& x, Vector<double>& output) const override
    {
        // Compute the pure Frobenius gradient
        coarse::frobenius::gradient(this->problem.get_M(), this->A, x, m_phi, m_w, output);

        // Energy-adaptive gradient
        // coarse_frobenius::energy_adaptive_gradient(this->problem.get_M(), this->A_inv, this->A, x, phi, w, output);.
        return 0; // 0 iterations, as no Krylov solver is used
    }

    iteration::State residual(const Vector<double>& x) const override
    {
        return {.energy = value(x)};
    }

private:
    Vector<double> m_w;
    Vector<double> m_phi;
};


// TODO: use tag dispatch for gradient metric
template <int dim>
class FrobeniusCoarseOracleEnergyAdaptive : public OracleBase<dim>
{
public:
    static constexpr const char* id = "FCA";
    using OracleBase<dim>::OracleBase;

    void update_parameters(const Vector<double>& w_new, const Vector<double>& phi_new)
    {
        m_w = w_new;
        m_phi = phi_new;
    }

    double value(const Vector<double>& x) const override
    {
        const double energy = this->problem.value(x, this->beta);

        return coarse::frobenius::function_value(x, m_phi, m_w, energy);
    }

    /**
     * @brief Computes the coarse model gradient in the energy-adaptive metric.
     * $$ \nabla_A q_k(\zeta) = \tilde{A}_\zeta^{-1} \left( \nabla_F q_k(\zeta) \right) $$
     */
    unsigned gradient(const Vector<double>& x, Vector<double>& output) const override
    {
        // Computes the F-gradient and applies the A_inv preconditioner
        coarse::frobenius::energy_adaptive_gradient(this->M, this->A_inv,this->A, x, m_phi, m_w, output);

        // Return the number of Krylov iterations used by A_inv
        return this->A_inv.control().last_step();
    }

    /**
     * @brief Evaluates the convergence of the coarse model.
     */
    iteration::State residual(const Vector<double>& x) const override
    {
        return {.energy = value(x)};
    }

private:
    Vector<double> m_w;
    Vector<double> m_phi;
};

} // namespace gpe

#endif //GPE_ORACLE_H