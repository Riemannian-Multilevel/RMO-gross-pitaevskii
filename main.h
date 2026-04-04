//
// Created by Ferdinand Vanmaele on 24.02.26.
//
#ifndef GPE_MAIN_H
#define GPE_MAIN_H

#include "manifold.h"
#include "function.h"
#include "lac.h"
#include "gpe.h"
#include "grid_operators.h"

namespace gpe
{

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

    // Compute directional derivative and Riemannian gradient successively
    virtual double directional_derivative(const Vector<double>& x, const Vector<double>& z) const = 0;

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
        return ellipsoid::function_value(x, this->problem.get_A0(), this->problem.get_Mpp(), this->beta);
    }

    double directional_derivative(const Vector<double>& x, const Vector<double>& z) const override
    {
        // AssertDimension(g.size(), z.size());
        // Vector<double> Mz(z.size());
        // this->M.vmult(Mz, z);
        // return g * Mz;
        return ellipsoid::directional_derivative(x, z, this->A);
    }

    /**
     * @brief Computes the Riemannian gradient in the M-metric.
     */
    unsigned gradient(const Vector<double>& x, Vector<double>& output) const override
    {
        ellipsoid::mass::gradient(this->M_inv, this->A, this->M, x, output);
        return this->M_inv.control().last_step();
    }

    iteration::State residual(const Vector<double>& x) const override
    {
        return iteration::residual(x, this->A, this->M);
    }
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
        return ellipsoid::function_value(x, this->problem.get_A0(), this->problem.get_Mpp(), this->beta);
    }

    double directional_derivative(const Vector<double>& x, const Vector<double>& z) const override
    {
        // AssertDimension(g.size(), g.size());
        // Vector<double> Az(z.size());
        // this->A.vmult(Az, z);
        // return g * Az;
        return ellipsoid::directional_derivative(x, z, this->A);
    }

    /**
     * @brief Computes the Riemannian gradient in the A-metric.
     * Solves the inner linear system $ A^{-1} \nabla E $ using the PreconditionInverse wrapper.
     */
    unsigned gradient(const Vector<double>& x, Vector<double>& output) const override
    {
        ellipsoid::energy::gradient(this->A_inv, this->M, x, output);
        return this->A_inv.control().last_step();
    }

    iteration::State residual(const Vector<double>& x) const override
    {
        return iteration::residual(x, this->A, this->M);
    }
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
        return ellipsoid::function_value(x, this->problem.get_A0(), this->problem.get_Mpp(), this->beta);
    }

    double directional_derivative(const Vector<double>& x, const Vector<double>& z) const override
    {
        // return g * z
        return ellipsoid::directional_derivative(x, z, this->A);
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
        return iteration::residual(x, this->A, this->M, false);
    }
};


// TODO: use tag dispatch for gradient metric
template <int dim>
class MassCoarseOracle : public OracleBase<dim>
{
public:
    static constexpr const char* id = "MC";
    // Explicit constructor to initialize the correction parameters
    MassCoarseOracle(const GrossPitaevskiiProblem<dim>& problem_, double beta_,
                 const Vector<double>& w_, const Vector<double>& phi_,
                 SolverOptions options_)
        : OracleBase<dim>(problem_, beta_, options_)
        , w(w_), phi(phi_)
    {}

    void update_parameters(const Vector<double>& w_new, const Vector<double>& phi_new)
    {
        w = w_new;
        phi = phi_new;
    }

    double value(const Vector<double>& x) const override
    {
        return coarse::mass::function_value(x, phi, w, this->problem.get_M(),
            this->problem.get_A0(), this->problem.get_Mpp(), this->beta);
    }

    double directional_derivative(const Vector<double>& x, const Vector<double>& z) const override
    {
        throw dealii::ExcNotImplemented();
    }

    /**
     * @brief Computes the coarse model gradient in the M-metric.
     */
    unsigned gradient(const Vector<double>& x, Vector<double>& output) const override
    {
        coarse::mass::gradient(this->M, x, phi, w, output);
        return 0; // No linear solver needed for pure M-gradient
    }

    iteration::State residual(const Vector<double>& x) const override
    {
        return {.energy=value(x)};
    }

private:
    Vector<double> w;
    Vector<double> phi;
};


// TODO: use tag dispatch for gradient metric
template <int dim>
class MassCoarseOracleEnergyAdaptive : public OracleBase<dim>
{
public:
    static constexpr const char* id = "MCA";

    // Explicit constructor to initialize the correction parameters
    MassCoarseOracleEnergyAdaptive(const GrossPitaevskiiProblem<dim>& problem_, double beta_,
                 const Vector<double>& w_, const Vector<double>& phi_,
                 SolverOptions options_)
        : OracleBase<dim>(problem_, beta_, options_)
        , w(w_), phi(phi_)
    {}

    void update_parameters(const Vector<double>& w_new, const Vector<double>& phi_new)
    {
        w = w_new;
        phi = phi_new;
    }

    double value(const Vector<double>& x) const override
    {
        return coarse::mass::function_value(x, phi, w, this->problem.get_M(),
            this->problem.get_A0(), this->problem.get_Mpp(), this->beta);
    }

    double directional_derivative(const Vector<double>& x, const Vector<double>& z) const override
    {
        throw dealii::ExcNotImplemented();
    }

    /**
     * @brief Computes the coarse model gradient in the A-metric.
     * $$ \nabla_A q_k(x) = \nabla_A E_H(x) - w $$
     */
    unsigned gradient(const Vector<double>& x, Vector<double>& output) const override
    {
        coarse::mass::energy_adaptive_gradient(this->M, this->A_inv, x, phi, w, output);
        return 0; // No linear solver needed for pure M-gradient
    }

    iteration::State residual(const Vector<double>& x) const override
    {
        return {.energy=value(x)};
    }

private:
    Vector<double> w;
    Vector<double> phi;
};


// TODO: use tag dispatch for gradient metric
template <int dim>
class FrobeniusCoarseOracle : public OracleBase<dim>
{
public:
    static constexpr const char* id = "FC";

    FrobeniusCoarseOracle(const GrossPitaevskiiProblem<dim>& problem_, double beta_,
                          const Vector<double>& w_, const Vector<double>& phi_,
                          SolverOptions options_)
        : OracleBase<dim>(problem_, beta_, options_)
        , w(w_)
        , phi(phi_)
    {}

    void update_parameters(const Vector<double>& w_new, const Vector<double>& phi_new)
    {
        w = w_new;
        phi = phi_new;
    }

    double value(const Vector<double>& x) const override
    {
        return coarse::frobenius::function_value(x, phi, w,
            this->problem.get_M(), this->problem.get_A0(), this->problem.get_Mpp(), this->beta);
    }

    double directional_derivative(const Vector<double>& x, const Vector<double>& z) const override
    {
        throw dealii::ExcNotImplemented();
    }

    /**
     * @brief Computes the coarse model gradient in the F-metric.
     * $$ \nabla_F q_k(\zeta) = \Pi_{\zeta, F}\left(A_\zeta \zeta - \frac{1}{\phi^\top M\zeta}w\right) $$
     */
    unsigned gradient(const Vector<double>& x, Vector<double>& output) const override
    {
        // Compute the pure Frobenius gradient
        coarse::frobenius::gradient(this->problem.get_M(), this->A, x, phi, w, output);

        // Energy-adaptive gradient
        // coarse_frobenius::energy_adaptive_gradient(this->problem.get_M(), this->A_inv, this->A, x, phi, w, output);.
        return 0; // 0 iterations, as no Krylov solver is used
    }

    iteration::State residual(const Vector<double>& x) const override
    {
        return {.energy = value(x)};
    }

private:
    Vector<double> w;
    Vector<double> phi;
};


// TODO: use tag dispatch for gradient metric
template <int dim>
class FrobeniusCoarseOracleEnergyAdaptive : public OracleBase<dim>
{
public:
    static constexpr const char* id = "FCA";

    FrobeniusCoarseOracleEnergyAdaptive(const GrossPitaevskiiProblem<dim>& problem_, double beta_,
                                        const dealii::Vector<double>& w_, const dealii::Vector<double>& phi_,
                                        SolverOptions options_)
        : OracleBase<dim>(problem_, beta_, options_)
        , w(w_)
        , phi(phi_)
    {}

    void update_parameters(const dealii::Vector<double>& w_new, const dealii::Vector<double>& phi_new)
    {
        w = w_new;
        phi = phi_new;
    }

    double value(const Vector<double>& x) const override
    {
        return coarse::frobenius::function_value(x, phi, w,
            this->problem.get_M(), this->problem.get_A0(),
            this->problem.get_Mpp(), this->beta);
    }

    double directional_derivative(const Vector<double>& x, const Vector<double>& z) const override
    {
        throw dealii::ExcNotImplemented();
    }

    /**
     * @brief Computes the coarse model gradient in the energy-adaptive metric.
     * $$ \nabla_A q_k(\zeta) = \tilde{A}_\zeta^{-1} \left( \nabla_F q_k(\zeta) \right) $$
     */
    unsigned gradient(const Vector<double>& x, Vector<double>& output) const override
    {
        // Computes the F-gradient and applies the A_inv preconditioner
        coarse::frobenius::energy_adaptive_gradient(this->M, this->A_inv,this->A, x, phi, w, output);

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
    Vector<double> w;
    Vector<double> phi;
};


/**
 * @brief Orchestrator for Gross-Pitaevskii simulations.
 * The @ref EnergySimulator manages the persistent @ref GrossPitaevskiiPackage
 * (discretization) and coordinates the execution of the energy minimization
 * using a given @ref Oracle.
 *
 * @tparam dim The spatial dimension.
 */
template <int dim, typename Oracle>
class GrossPitaevskiiSimulator
{
public:
    static_assert(Oracle::dimension == dim);

    /**
     * @brief Constructor.
     * @tparam Potential Functor or class representing the external potential \f$ V(x) \f$.
     * @param V The potential object.
     * @param options General options for GPE discretization.
     * @param n_levels Number of global mesh refinements.
     */
    template <typename Potential>
    GrossPitaevskiiSimulator(Potential&& V, const GPE_Options& options, unsigned int n_levels)
        : package(options, n_levels)
        , problem(package.problem(std::forward<Potential>(V)))
        , options(options)
    {}

    // Allow to change the potential without re-discretizing the domain.
    template <typename Potential>
    void reinit(Potential&& V)
    {
        problem = package.problem(std::forward<Potential>(V));
    }

    void distribute(Vector<double>& x) const
    {
        package.distribute(x);
    }

    /**
     * @brief Runs the energy minimization for a given potential.
     * @param x0
     * @param beta The interaction strength constant.
     * @param options_inner
     * @param options_gd Options for the gradient descent algorithm.
     */
    // TODO factor this out to caller, see get_oracle()
    Vector<double>
    run(const Vector<double>& x0, double beta,
        const SolverOptions&  options_inner,
        const DescentOptions& options_gd, std::ostream& os) const
    {
        Assert(x0.size() == package.n_dofs(), dealii::ExcDimensionMismatch(x0.size(), package.n_dofs()));
        // Create the oracle (light-weight object, references problem matrices)
        Oracle oracle(problem, beta, options_inner);

        // Termination criterium
        auto conv_check = [&options_gd](const iteration::State& current, const iteration::State& previous)
        {
            const double lmb_diff   = std::abs(current.lambda - previous.lambda);
            const double lmb_factor = 1.0 + std::abs(current.lambda);

            return (lmb_diff < options_gd.tol_lambda * lmb_factor && current.residual < options_gd.tol_residual);
        };

        // Riemannian gradient descent
        // Note: the update strategy can be arbitrary complex (e.g. for multilevel algorithms)
        return gradient_descent(oracle, x0, options_gd, os, conv_check);
    }

    /** @brief Access the discretization package. */
    const GrossPitaevskiiPackage<dim>& get_package() const { return package; }
    const GrossPitaevskiiProblem<dim>& get_problem() const { return problem; }

    unsigned int n_dofs() const { return package.n_dofs(); }

    const dealii::DoFHandler<dim>&
    get_dofs() const { return package.get_dofs(); }

    const dealii::AffineConstraints<double>&
    get_constraints() const { return package.get_constraints(); }

    Oracle get_oracle(double beta, SolverOptions options_gd) const
    {
        return Oracle(problem, beta, options_gd);
    }

    auto get_M() const { return problem.get_operator_M(); }

    auto get_M_inv(SolverOptions options_gd) const
    {
        using InverseOpType = InverseMatrix<decltype(this->get_M()), dealii::PreconditionIdentity>;

        return InverseOpType(get_M(), options_gd.solver, dealii::PreconditionIdentity{},
            options_gd.max_inner, options_gd.tol_inner);
    }

    auto get_A(double beta) const { return problem.get_operator_A(beta); }

    auto get_A_inv(double beta, SolverOptions options_gd) const
    {
        using InverseOpType = InverseMatrix<decltype(this->get_A(beta)), dealii::PreconditionIdentity>;

        return InverseOpType(get_A(beta), options_gd.solver, dealii::PreconditionIdentity{},
            options_gd.max_inner, options_gd.tol_inner);
    }

private:
    /** @brief Persistent discretization infrastructure. */
    GrossPitaevskiiPackage<dim> package;
    /** @brief Assembly and storage of matrices. */
    GrossPitaevskiiProblem<dim> problem;
    /** @brief Problem configuration options. */
    GPE_Options options;
};


template <typename MatrixType>
double M_norm(const MatrixType& M, const Vector<double>& x)
{
    Vector<double> Mx(x.size());
    M.vmult(Mx, x);
    return std::sqrt(x*Mx);
}

struct CycleInfo
{
    unsigned iter;
    unsigned lac_iter;
    double   step_size;
    double   elapsed;
    bool     coarse;
};

template <typename Oracle>
void cycle_eval(const Oracle& O, const Vector<double>& y,
                dealii::ConvergenceTable& convergence_table,
                const CycleInfo info)
{
    auto state   = O.residual(y);
    state.energy = O.value(y);

    convergence_table.add_value("iter", info.iter);
    convergence_table.add_value("coarse", info.coarse ? "*" : " ");
    convergence_table.add_value("lac_iter", info.lac_iter);
    convergence_table.add_value("mass", state.mass);
    convergence_table.add_value("lambda", state.lambda);
    convergence_table.add_value("residual", state.residual);
    convergence_table.add_value("energy", state.energy);
    convergence_table.add_value("step",info.step_size);
    convergence_table.add_value("elapsed",info.elapsed);
}

inline void cycle_finalize(dealii::ConvergenceTable& convergence_table, std::ostream& os,
                           dealii::TableHandler::TextOutputFormat format)
{
    convergence_table.set_precision("mass", 4);
    convergence_table.set_precision("lambda", 4);
    convergence_table.set_precision("residual", 4);
    convergence_table.set_precision("energy", 8);
    convergence_table.set_precision("step", 4);
    convergence_table.set_precision("elapsed", 4);

    convergence_table.set_scientific("lambda", true);
    convergence_table.set_scientific("residual", true);
    convergence_table.set_scientific("energy", true);
    convergence_table.set_scientific("step", true);
    convergence_table.set_scientific("elapsed", true);

    convergence_table.evaluate_convergence_rates("residual", reduction_rate);
    convergence_table.evaluate_convergence_rates("residual", reduction_rate_log2);
    convergence_table.write_text(os, format);
}


struct CoarseStep
{
    CoarseStep(const Vector<double>& y, const unsigned n_coarse, const unsigned n_fine)
        : y(y), x(n_coarse), x_grad(n_coarse), y_grad(n_fine), y_grad_restr(n_coarse)
    {
        AssertDimension(y.size(),n_fine);
    }

    const Vector<double>& y;     // fine point (unchanged in coarse cycle)
    Vector<double> x;            // coarse point
    Vector<double> x_grad;       // (M-)coarse gradient
    Vector<double> y_grad;       // (M-)fine gradient
    Vector<double> y_grad_restr; // restricted gradient
};


// TODO: CoarseModel decides which metric DEFINES the Nash model.
//       CoarseOracle decides which metric SOLVES the Nash model.
//       -> Do not hardcode the Oracle, but take a common interface.
template <int dim, typename Oracle, typename CoarseOracle, typename VectorTransport>
class CoarseModel
{
public:
    using OperatorType  = LinearCombination<SparseMatrix<double>,Vector<double>>;
    using MatrixType    = SparseMatrix<double>;
    using InverseOpType = PreconditionInverse<OperatorType, SparseMatrix<double>>;

    CoarseModel(const GrossPitaevskiiProblem<dim>& problem_coarse,
                const GrossPitaevskiiProblem<dim>& problem_fine,
                const LinearTransferBase& transfer, double beta,
                SolverOptions options, SolverOptions options_coarse)
        // Function evaluation
        : options(options), options_coarse(options_coarse)
        , O_coarse(problem_coarse, beta, options_coarse)
        , O_fine(problem_fine, beta, options)
        , qk(problem_coarse, beta, Vector<double>(), Vector<double>(), options)

        // Problem components
        , n_coarse(problem_coarse.n_dofs())
        , n_fine(problem_fine.n_dofs())
        , M_coarse(problem_coarse.get_operator_M())
        , M_fine(problem_fine.get_operator_M())
        , A_coarse(problem_coarse.get_operator_A(beta))
        , A_fine(problem_fine.get_operator_A(beta))  // only needed for A-gradient
        // TODO: separate preconditioners for gradient descent (qk), and inverse of M (coarse gradients)
        , M_coarse_inv(M_coarse, options_coarse)
        , M_fine_inv(M_fine, options)

        // Grid operators
        , transfer(transfer)
        , point_transfer(M_coarse, M_fine, transfer)
        , vector_transport(M_coarse, M_fine, transfer, point_transfer)
    {}

    void set_timer(const dealii::Timer& timer_new)
    {
        timer = timer_new;
    }

    // We separate the "setup" phase (fine/coarse vectors) from the "model" phase
    // so that a coarse criterion can be efficiently evaluated.
    CoarseStep
    setup(const Vector<double>& y)
    {
        AssertDimension(y.size(),n_fine);
        CoarseStep step(y, n_coarse, n_fine);

        // Starting coarse point
        std::cerr << "[" << timer.cpu_time() << "] coarse: point transfer\n";
        point_transfer.restriction(y, step.x);

        std::cerr << "[" << timer.cpu_time() << "] coarse: assemble matrix\n";
        O_coarse.update(step.x);

        // Compute coarse M-gradient
        std::cerr << "[" << timer.cpu_time() << "] coarse: " << O_coarse.id << "-coarse gradient\n";
        O_coarse.gradient(step.x, step.x_grad);

        // Compute fine M-gradient
        std::cerr << "[" << timer.cpu_time() << "] coarse: " << O_fine.id << "-fine gradient\n";
        O_fine.gradient(y, step.y_grad);

        // Compute coarse correction step
        std::cerr << "[" << timer.cpu_time() << "] coarse: M-vector restriction\n";
        vector_transport.vector_restriction(step.x, y, step.y_grad, step.y_grad_restr);

        return step;
    }

    // TODO: Use cycle_fine() for consistency instead of gradient_descent()
    void solve(const CoarseStep& step, DescentOptions options_gd, Vector<double>& dst)
    {
        AssertDimension(dst.size(),n_fine);

        Vector<double> w(n_coarse);
        w = step.x_grad;
        w.add(-1.0, step.y_grad_restr);

        // Set up coarse model
        qk.update_parameters(w, step.x);
        Vector<double> zk(n_coarse);

        // Find zk such that qk(zk) < qk(x)
        // TODO: variable method
        std::cerr << "[" << timer.cpu_time() << "] coarse: " << qk.id << "-gradient descent\n";
        zk = gradient_descent(qk, step.x, options_gd, std::cerr);

        // Compute the search direction, zk <- L_x(zk)
        // FIXME? use separate variable to hold ambient tangent vector L_x(zk)
        std::cerr << "[" << timer.cpu_time() << "] coarse: inverse retraction\n";
        ellipsoid::retract_inv_by_norm(M_coarse, zk, step.x);

        std::cerr << "[" << timer.cpu_time() << "] coarse: " << vector_transport.id << "-vector prolongation\n";
        vector_transport.vector_prolongation(step.y, step.x, zk, dst);
    }

    unsigned n_dofs() const { return n_coarse; }

private:
    dealii::Timer timer;
    SolverOptions options, options_coarse;

    // Function evaluation
    Oracle O_coarse, O_fine;
    CoarseOracle qk;
    unsigned n_coarse, n_fine;

    OperatorType M_coarse, M_fine;
    OperatorType A_coarse, A_fine;
    InverseOpType M_coarse_inv, M_fine_inv;

    // Grid operators
    const LinearTransferBase& transfer;
    ManifoldTransfer<dim, OperatorType> point_transfer;
    VectorTransport vector_transport;
};


// TODO: factor out the cycle_smooth() methods (consolidate with descent.h)
//       convergence check (cf. EnergySimulator::run)
template <int dim, typename SmoothOracle, typename CoarseModel>
class FullApproximationScheme
{
public:
    using OperatorType  = LinearCombination<SparseMatrix<double>,Vector<double>>;
    using MatrixType    = SparseMatrix<double>;
    using InverseOpType = PreconditionInverse<OperatorType, SparseMatrix<double>>;

    FullApproximationScheme(const GrossPitaevskiiProblem<dim>& problem_coarse,
                            const GrossPitaevskiiProblem<dim>& problem_fine,
                            const LinearTransferBase& transfer, double beta,
                            SolverOptions options, SolverOptions options_coarse)
        : O_fine(problem_fine, beta, options)
        // TODO: separate preconditioners for gradient descent, and inverse of M (coarse gradients)
        , O_coarse(problem_coarse, problem_fine, transfer, beta, options, options_coarse)
        , problem_coarse(problem_coarse)
        , problem_fine(problem_fine)
    {
        O_coarse.set_timer(timer);
    }

    void cycle_condition(const Vector<double>& y0, std::ostream& os,
                         DescentOptions options_gd, DescentOptions options_gd_coarse,
                         double kappa, double eps, unsigned coarse_every = 1)
    {
        timer.reset();

        // 0. Initialize oracle and coarse model
        Vector<double> y(y0);
        O_fine.update(y);
        Vector<double> y_grad(y.size());
        Vector<double> dk(y.size());

        const auto& M_fine = problem_fine.get_M();
        const auto& M_coarse = problem_coarse.get_M();
        bool check_coarse_cond = true;

        for (unsigned i = 0; i < options_gd.max_iter; i++) {
            // Compute coarse condition
            if (check_coarse_cond && (i == 0 || i % coarse_every == 0)) {
                // TODO: If the coarse model is evaluated in the A-gradient (or the fine objected solved
                //       in the M-metric), this step results in negligible additional effort.
                //       Metric-free formulation of the coarse condition?
                auto coarse_step = O_coarse.setup(y);
                // Norm of fine (M-)gradient
                double y_grad_norm = M_norm(M_fine, coarse_step.y_grad);
                // Norm of restricted (M-)gradient
                double y_grad_restr_norm = M_norm(M_coarse, coarse_step.y_grad_restr);
                convergence_table.add_value("grad_restr_norm", y_grad_restr_norm);
                convergence_table.add_value("grad_norm", y_grad_norm);

                if (y_grad_restr_norm <= eps) {
                    check_coarse_cond = false;  // stop coarse condition evaluation once threshold was reached
                }
                if (y_grad_restr_norm >= kappa*y_grad_norm && y_grad_restr_norm > eps) {
                    // Coarse step
                    O_coarse.solve(coarse_step, options_gd_coarse, dk);

                    // Evaluate directional derivative in M-norm
                    //auto dir_deriv = O_fine.metric(y_grad, dk);
                    auto dir_deriv = O_fine.directional_derivative(y, dk);
                    CycleInfo info = cycle_smooth(y, dir_deriv, dk, options_gd);
                    info.iter      = i;
                    info.coarse    = true;
                    info.lac_iter  = 0;

                    cycle_eval(O_fine, y, convergence_table, info);
                } else {
                    goto fine_step;
                }
            } else {
                convergence_table.add_value("grad_restr_norm", 0);
                convergence_table.add_value("grad_norm", 0);
fine_step:
                // Update gradient
                std::cerr << "[" << timer.cpu_time() << "] fine: A-gradient\n";
                auto lac_iter = O_fine.gradient(y, y_grad);
                dk  = y_grad;
                dk *= -1.0;

                // Evaluate directional derivative in A-norm
                //auto dir_deriv = O_fine.metric(y_grad, dk);
                auto dir_deriv = O_fine.directional_derivative(y, dk);
                CycleInfo info = cycle_smooth(y, dir_deriv, dk, options_gd);
                info.iter      = i;
                info.coarse    = false;
                info.lac_iter  = lac_iter;

                cycle_eval(O_fine, y, convergence_table, info);
            }
        }
        convergence_table.set_precision("grad_restr_norm", 4);
        convergence_table.set_precision("grad_norm", 4);
        convergence_table.set_scientific("grad_restr_norm", true);
        convergence_table.set_scientific("grad_norm", true);

        cycle_finalize(convergence_table, os, dealii::TableHandler::TextOutputFormat::org_mode_table);
    }

    // TODO: move loop to caller or separate method
    void cycle(const Vector<double>& y0, std::ostream& os,
               DescentOptions options_gd, DescentOptions options_gd_coarse,
               unsigned n_pre = 1, unsigned n_post = 1)
    {
        timer.reset();

        // 0. Initialize oracle and coarse model
        Vector<double> y(y0);
        O_fine.update(y);

        unsigned i = 1;
        while (true) {
            // 1. Pre-smoothing
            Vector<double> y_grad(O_fine.n_dofs());
            Vector<double> dk(O_fine.n_dofs());

            for (unsigned pre = 0; pre < n_pre; pre++) {
                std::cerr << "Pre-smoothing: " << pre << "\n";
                if (i > options_gd.max_iter) goto finalize;

                // Update gradient
                std::cerr << "[" << timer.cpu_time() << "] fine: A-gradient\n";
                auto lac_iter = O_fine.gradient(y, y_grad);
                dk  = y_grad;
                dk *= -1.0;

                // Apply step
                //auto dir_deriv = O_fine.metric(y_grad, dk);
                auto dir_deriv = O_fine.directional_derivative(y, dk);
                CycleInfo info = cycle_smooth(y, dir_deriv, dk, options_gd);
                info.iter      = i++;
                info.coarse    = false;
                info.lac_iter  = lac_iter;

                cycle_eval(O_fine, y, convergence_table, info);
            }

            // 2. Coarse step
            {
                if (i > options_gd.max_iter) goto finalize;

                // Update coarse direction
                auto coarse_step = O_coarse.setup(y);
                O_coarse.solve(coarse_step, options_gd_coarse, dk);

                // Apply step
                // Evaluate directional derivative in the M-norm
                // auto dir_deriv = coarse_step.metric(coarse_step.y_grad, dk);
                auto dir_deriv = O_fine.directional_derivative(y, dk);
                CycleInfo info = cycle_smooth(y, dir_deriv, dk, options_gd);
                info.iter      = i++;
                info.coarse    = true;
                info.lac_iter  = 0;

                cycle_eval(O_fine, y, convergence_table, info);
            }

            // 3. Post-smoothing
            for (unsigned post = 0; post < n_post; ++post) {
                std::cerr << "Post-smoothing: " << post << "\n";
                if (i > options_gd.max_iter) goto finalize;

                // Update gradient
                std::cerr << "[" << timer.cpu_time() << "] fine: A-gradient\n";
                auto lac_iter = O_fine.gradient(y, y_grad);
                dk  = y_grad;
                dk *= -1.0;

                // Apply step
                //auto dir_deriv = O_fine.metric(y_grad, dk);
                auto dir_deriv = O_fine.directional_derivative(y, dk);
                CycleInfo info = cycle_smooth(y, dir_deriv, dk, options_gd);
                info.iter      = i++;
                info.coarse    = false;
                info.lac_iter  = lac_iter;

                cycle_eval(O_fine, y, convergence_table, info);
            }
        }
finalize:
        cycle_finalize(convergence_table, os, dealii::TableHandler::TextOutputFormat::org_mode_table);
    }

private:
    dealii::ConvergenceTable convergence_table;
    dealii::Timer timer;
    SmoothOracle O_fine;
    CoarseModel O_coarse;  // encodes both the objective, and the method to solve it
    const GrossPitaevskiiProblem<dim>& problem_coarse;
    const GrossPitaevskiiProblem<dim>& problem_fine;

    // Implementation of smoothing steps
    CycleInfo cycle_smooth(Vector<double>& y, const double dir_deriv,
                           const Vector<double>& eta, DescentOptions options_gd)
    {
        timer.start();
        double step_size = options_gd.step_size;

        // Update point (fixed step or line search)
        if (options_gd.line_search) {
            std::cerr << "[" << timer.cpu_time() << "] " << "fine: line search" << std::endl;
            double Ex = O_fine.value(y);

            step_size = armijo_line_search(O_fine, y, eta, Ex, dir_deriv, options_gd);

            if (step_size <= options_gd.ls_min) {
                std::cerr << "  -> Coarse step rejected by line search." << std::endl;
            }
        }
        else {
            std::cerr << "[" << timer.cpu_time() << "] " << "fine: retraction" << std::endl;
            O_fine.retract(eta, y, options_gd.step_size);  // update y

            std::cerr << "[" << timer.cpu_time() << "] " << "fine: assembly" << std::endl;
            O_fine.update(y);
        }

        timer.stop();
        return {.step_size = step_size, .elapsed = timer.cpu_time()};
    }
};

} // namespace gpe

#endif //GPE_MAIN_H