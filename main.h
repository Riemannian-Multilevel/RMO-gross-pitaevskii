//
// Created by Ferdinand Vanmaele on 24.02.26.
//
#ifndef GPE_MAIN_H
#define GPE_MAIN_H

#include "manifold.h"
#include "lac.h"
#include "gpe.h"

namespace gpe
{

// TODO: define interface and move to `oracle.h`, merge functions from `manifold.h`
/**
 * @brief Mathematical Oracle for the Gross-Pitaevskii energy functional.
 * This class provides the interface required by the @ref gradient_descent algorithm.
 * It translates physical concepts (matrices and assembly) into optimization concepts
 * (functional values and gradients).
 *
 * @tparam dim The spatial dimension of the problem.
 */
template <int dim>
class EnergyOracle
{
public:
    /** @brief Alias for the combined linear operator type. */
    using OperatorType  = LinearCombination<SparseMatrix<double>, Vector<double>>;
    /** @brief Alias for the preconditioned inverse operator type. */
    using InverseOpType = InverseMatrix<OperatorType, dealii::SparseILU<double>>;

    /**
     * @brief Constructs the Oracle by referencing an existing GPE problem.
     * @param problem_ Reference to the assembled GPE problem.
     * @param beta_ Non-linear coupling constant (interaction strength).
     * @note The Oracle does not own the problem; it holds a reference. The Problem object
     * must outlive this Oracle.
     */
    EnergyOracle(const GrossPitaevskiiProblem<dim>& problem_, double beta_)
        : problem(problem_)
        , beta(beta_)
        , A(problem.get_operator_A(beta))
        , M(problem.get_operator_M())
    {
        // ILU preconditioning is usually based on the stationary linear part (A0)
        precond.initialize(problem.get_A0());
    }

    // TODO: store `x` argument, set markers `needs_assembly=true`, `need_gradient=true`
    void update(const Vector<double>& x) const
    {
        problem.assemble_nonlinear_term(x);
    }

    /**
     * @brief Computes the Gross-Pitaevskii energy functional value.
     * \f[ E(\phi) = \langle \phi, A_0 \phi \rangle + \frac{\beta}{2} \langle \phi, M_{pp}(\phi) \phi \rangle \f]
     * @param x The current state vector.
     * @return The energy value.
     */
    // TODO: leave `x` argument in update() exclusively, to avoid mismatches
    //       check marker `needs_assembly`
    double value(const Vector<double>& x) const
    {
        return ellipsoid::function_value(x, problem.get_A0(), problem.get_Mpp(), beta);
    }

    /**
     * @brief Computes the Riemannian gradient.
     * Solves the inner linear system \f$ A^{-1} \nabla E \f$ using the @ref InverseMatrix
     * wrapper and the provided solver options.
     * @param x The current state vector.
     * @param output Vector to store the computed gradient.
     * @param options Solver and tolerance options for the inner iteration.
     * @return The number of inner iterations performed by the linear solver.
     */
    // TODO: leave `x` argument in update() exclusively, to avoid mismatches
    //       check marker `needs_gradient`
    //       use LinearOptions instead of DescentOptions (separate inner solve and rgd options)
    unsigned gradient(const Vector<double>& x, Vector<double>& output, const DescentOptions& options) const
    {
        const InverseOpType A_inv(A, options.solver, precond, options.max_inner, options.tol_inner);

        ellipsoid::gradient(A_inv, M, x, output);

        return A_inv.control().last_step();
    }

    /**
     * @brief Retracts a tangent vector back to the unit-mass manifold.
     * \f[ R_x(z) = \frac{x + z}{\|x + z\|_M} \f]
     * @param z The update vector.
     * @param x The base vector (modified in place).
     * @param factor Step size scaling factor.
     */
    void retract(const Vector<double>& z, Vector<double>& x, double factor = 1.0) const
    {
        ellipsoid::retract_by_norm(problem.get_M(), z, x, factor);
    }

    void retract(const Vector<double>& z, const Vector<double>& x,
                 Vector<double>& output, double factor = 1.0) const
    {
        output = x;
        retract(z, output, factor);
    }

    /**
     * @brief Computes the iteration state (eigenvalue, residual, etc.) at point x.
     */
    iteration::State residual(Vector<double>& x) const
    {
        return iteration::residual(x, A, M);
    }

    double metric(const Vector<double>& y, const Vector<double>& z) const
    {
        AssertDimension(y.size(), z.size());

        Vector<double> Az(z.size());
        A.vmult(Az, y);
        return y*Az;
    }

    /**
     * @brief Checks if the optimization has converged.
     * Convergence is achieved if both the eigenvalue (\f$ \lambda \f$) and the
     * non-linear residual fall below specified tolerances.
     * @param current
     * @param previous
     * @param options Termination criteria.
     */
    static bool check_convergence(const iteration::State& current,
                                  const iteration::State& previous,
                                  const DescentOptions& options)
    {
        const double lmb_diff   = std::abs(current.lambda - previous.lambda);
        const double lmb_factor = 1.0 + std::abs(current.lambda);

        return (lmb_diff < options.tol_lambda * lmb_factor && current.residual < options.tol_residual);
    }

    auto get_M() { return problem.get_operator_M(); }
    auto get_A() const { return problem.get_operator_A(beta); }
    double get_beta() const { return beta; }

    unsigned n_dofs() const { return problem.n_dofs(); }

private:
    /** @brief Reference to the problem matrices. */
    const GrossPitaevskiiProblem<dim>& problem;
    /** @brief Interaction strength parameter. */
    double beta;
    /** @brief Incomplete LU preconditioner. */
    dealii::SparseILU<double> precond;

    /** @brief Total operator \f$ A_0 + \beta M_{pp} \f$. */
    OperatorType A;
    /** @brief Mass operator \f$ M \f$. */
    OperatorType M;
};

// TODO: separate geometry and objective
template <int dim>
class CoarseOracle
{
public:
    using OperatorType  = LinearCombination<SparseMatrix<double>, Vector<double>>;
    using InverseOpType = InverseMatrix<OperatorType, dealii::SparseILU<double>>;

    CoarseOracle(const GrossPitaevskiiProblem<dim>& problem, double beta,
                 const Vector<double>& w, const Vector<double>& phi)
        : problem(problem)
        , beta(beta)
        , M(problem.get_operator_M())
        , A(problem.get_operator_A(beta))
        , w(w)
        , phi(phi)
    {
        precond.initialize(problem.get_A0());
    }

    void update(const Vector<double>& x) const
    {
        problem.assemble_nonlinear_term(x);
    }

    double value(const Vector<double>& x) const
    {
        return coarse::function_value(x, phi, w,
            problem.get_M(), problem.get_A0(), problem.get_Mpp(), beta);
    }

    // Gradient in the energy metric
    unsigned gradient(const Vector<double>& x, Vector<double>& output,
        const DescentOptions& options) const
    {
        const InverseOpType A_inv(A, options.solver, precond, options.max_inner, options.tol_inner);

        coarse::gradient(M, A_inv, x, phi, w, output);

        return A_inv.control().last_step();
    }

    // XXX: geometry of the problem
    void retract(const Vector<double>& z, Vector<double>& x, double factor = 1.0) const
    {
        ellipsoid::retract_by_norm(problem.get_M(), z, x, factor);
    }

    void retract(const Vector<double>& z, const Vector<double>& x,
                 Vector<double>& output, double factor = 1.0) const
    {
        output = x;
        retract(z, output, factor);
    }

    double metric(const Vector<double>& y, const Vector<double>& z) const
    {
        AssertDimension(y.size(), z.size());

        Vector<double> Az(z.size());
        A.vmult(Az, y);
        return y*Az;
    }

    // XXX: methods for gradient_descent()
    iteration::State residual(Vector<double>& x) const
    {
        return {.energy=value(x)};
    }

    static bool check_convergence(const iteration::State&, const iteration::State&,
                                  const DescentOptions&) { return false; }

private:
    const GrossPitaevskiiProblem<dim>& problem;
    double beta;
    OperatorType M;
    OperatorType A;
    Vector<double> w;
    Vector<double> phi;
    dealii::SparseILU<double> precond;
};

/**
 * @brief Orchestrator for Gross-Pitaevskii simulations.
 * The @ref EnergySimulator manages the persistent @ref GrossPitaevskiiPackage
 * (discretization) and coordinates the execution of the energy minimization
 * using the @ref EnergyOracle.
 *
 * @tparam dim The spatial dimension.
 */
template <int dim>
class EnergySimulator
{
public:
    /**
     * @brief Constructor.
     * @tparam Potential Functor or class representing the external potential \f$ V(x) \f$.
     * @param V The potential object.
     * @param options General options for GPE discretization.
     * @param n_levels Number of global mesh refinements.
     */
    template <typename Potential>
    EnergySimulator(Potential&& V, const GPE_Options& options, unsigned int n_levels)
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
     * @param beta The interaction strength constant.
     * @param gd_options Options for the gradient descent algorithm.
     */
    Vector<double>
    run(const Vector<double>& x0, double beta, const DescentOptions& gd_options, std::ostream& os) const
    {
        Assert(x0.size() == package.n_dofs(), dealii::ExcDimensionMismatch(x0.size(), package.n_dofs()));
        // Create the oracle (light-weight object, references problem matrices)
        EnergyOracle<dim> oracle(problem, beta);

        // Riemannian gradient descent
        // Note: the update strategy can be arbitrary complex (e.g. for multilevel algorithms)
        return gradient_descent(oracle, x0, gd_options, os);
    }

    /** @brief Access the discretization package. */
    const GrossPitaevskiiPackage<dim>& get_package() const { return package; }
    const GrossPitaevskiiProblem<dim>& get_problem() const { return problem; }

    unsigned int n_dofs() const { return package.n_dofs(); }
    const dealii::DoFHandler<dim>& get_dofs() const { return package.get_dofs(); }
    const dealii::AffineConstraints<double>& get_constraints() const { return package.get_constraints(); }

private:
    /** @brief Persistent discretization infrastructure. */
    GrossPitaevskiiPackage<dim> package;
    /** @brief Assembly and storage of matrices. */
    GrossPitaevskiiProblem<dim> problem;
    /** @brief Problem configuration options. */
    GPE_Options options;
};

// TODO
double first_order_coherence()
{
    throw dealii::ExcNotImplemented(__PRETTY_FUNCTION__);
}

template <typename MatrixType>
double M_norm(const MatrixType& M, const Vector<double>& x)
{
    Vector<double> Ax(x.size());
    M.vmult(Ax, x);
    return std::sqrt(x*Ax);
}

// 2-level FAS
template <int dim, typename Oracle>
void full_approximation_scheme(Oracle&& O_coarse, Oracle&& O_fine,
                               const GrossPitaevskiiPackage<dim>& domain_coarse,
                               const GrossPitaevskiiPackage<dim>& domain_fine,
                               const GrossPitaevskiiProblem<dim>& problem_coarse,
                               const Vector<double>& y0, DescentOptions options,
                               DescentOptions options_coarse,
                               unsigned n_cycles, std::ostream& os)
{
    using OperatorType = LinearCombination<SparseMatrix<double>,Vector<double>>;
    using InverseOpType = InverseMatrix<OperatorType, dealii::PreconditionIdentity>;

    unsigned n_coarse = O_coarse.n_dofs();
    unsigned n_fine = O_fine.n_dofs();
    const auto& M_coarse = O_coarse.get_M();
    const auto& A_coarse = O_coarse.get_A();
    const auto& M_fine = O_fine.get_M();
    const auto& A_fine = O_fine.get_A();
    InverseOpType M_coarse_inv(M_coarse, options_coarse.solver, dealii::PreconditionIdentity{},
        options_coarse.max_inner, options_coarse.tol_inner);
    InverseOpType M_fine_inv(M_fine, options.solver, dealii::PreconditionIdentity{},
        options.max_inner, options.tol_inner);

    // TODO: ignore termination criteria? (avoid post-smoothing breaks down)
    options.max_iter = 1;  // number of pre-/post-smoothing steps
    Vector<double> y(y0);
    O_fine.update(y); // TODO

    LinearTransfer transfer(domain_coarse.get_dofs(), domain_fine.get_dofs(),
        domain_coarse.get_constraints(), domain_fine.get_constraints());
    ManifoldTransfer<dim, OperatorType> point_transfer(M_coarse, M_fine, transfer);
    ProjectionTransport vector_transport(M_coarse, M_fine, transfer, point_transfer);
    dealii::ConvergenceTable convergence_table;

    int i = 1;
    for (unsigned cycle = 0; cycle < n_cycles; cycle++) {
        //Vector<double> y_grad_m_restr(n_coarse);
        //vector_transport.vector_restriction(x, y_grad_m, y_grad_m_restr);

        // double restr_grad_norm = M_norm(M_coarse, y_grad_m_restr);
        // double grad_norm = M_norm(M_fine, y_grad_m);
        // double eta = 0.3;
        // std::cout << "Restricted gradient norm: " << restr_grad_norm << std::endl;
        // std::cout << "Gradient norm: " << grad_norm << std::endl;

        // 1. Pre-smoothing
        Vector<double> y_grad(n_fine);
        auto lac_iter = O_fine.gradient(y, y_grad, options);
        O_fine.retract(y_grad, y, -options.step_size);
        O_fine.update(y);

        // Print stats
        auto state = O_fine.residual(y);
        state.energy = O_fine.value(y);
        convergence_table.add_value("iter", i++);
        convergence_table.add_value("coarse", " ");
        convergence_table.add_value("lac_iter", lac_iter);
        convergence_table.add_value("mass", state.mass);
        convergence_table.add_value("lambda", state.lambda);
        convergence_table.add_value("residual", state.residual);
        convergence_table.add_value("energy", state.energy);
        convergence_table.add_value("step",options.step_size);

        // 2. Coarse step
        // Restrict point from fine to coarse manifold, x = r(y)
        Vector<double> x(n_coarse);
        point_transfer.restriction(y, x);
        O_coarse.update(x);
        // Compute fine A-gradient
        O_fine.gradient(y, y_grad, options);
        // Compute coarse M-gradient
        Vector<double> x_grad_m(n_coarse);
        ellipsoid::gradient(M_coarse_inv, A_coarse, M_coarse, x, x_grad_m);
        // Compute fine M-gradient
        Vector<double> y_grad_m(n_fine);
        ellipsoid::gradient(M_fine_inv, A_fine, M_fine, y, y_grad_m);
        // Compute coarse correction step
        Vector<double> w(n_coarse);
        coarse_correction(vector_transport, x_grad_m, y_grad_m, x, w);
        // Set up coarse model
        CoarseOracle<dim> qk(problem_coarse, O_coarse.get_beta(), w, x);
        Vector<double> zk(n_coarse);
        // Find zk such that qk(zk) < qk(x)
        zk = gradient_descent(qk, x, options_coarse, std::cerr);
        // Compute the search direction, zk <- L_x(zk)
        ellipsoid::retract_inv_by_norm(M_coarse, zk, x);
        Vector<double> dk(n_fine);
        vector_transport.vector_prolongation(y, zk, dk);
        // DIAGNOSTIC
        double dd = O_fine.metric(y_grad, dk); // <grad f(x), -grad f(x)>_x
        std::cerr << "COARSE DESCENT: " << dd << std::endl;
        // double coarse_step = 1.0*std::pow(0.5,cycle);
        // O_fine.retract(dk, y, coarse_step);
        // O_fine.update(y);
        // Armijo line search along dk
        double Ex = O_fine.value(y);
        Vector<double> y_new(y);
        // runs update(), retract()
        double coarse_step = armijo_line_search(O_fine, y, dk, Ex, dd, options, y_new);
        if (coarse_step > 0.0) {
            y = y_new; // Accept the step
            //O_fine.update(y);
        } else {
            std::cerr << "  -> Coarse step rejected by line search." << std::endl;
        }

        // Print stats
        state = O_fine.residual(y);
        state.energy = O_fine.value(y);
        convergence_table.add_value("iter",  i++);
        convergence_table.add_value("coarse", "*");
        convergence_table.add_value("lac_iter", lac_iter);
        convergence_table.add_value("mass", state.mass);
        convergence_table.add_value("lambda", state.lambda);
        convergence_table.add_value("residual", state.residual);
        convergence_table.add_value("energy", state.energy);
        convergence_table.add_value("step", coarse_step);

        // 3. Post-smoothing
        lac_iter = O_fine.gradient(y, y_grad, options);
        O_fine.retract(y_grad, y, -options.step_size);
        O_fine.update(y);

        // Print stats
        state = O_fine.residual(y);
        state.energy = O_fine.value(y);
        convergence_table.add_value("iter", i++);
        convergence_table.add_value("coarse", " ");
        convergence_table.add_value("lac_iter", lac_iter);
        convergence_table.add_value("mass", state.mass);
        convergence_table.add_value("lambda", state.lambda);
        convergence_table.add_value("residual", state.residual);
        convergence_table.add_value("energy", state.energy);
        convergence_table.add_value("step",options.step_size);
    }

    convergence_table.set_precision("mass", 4);
    convergence_table.set_precision("lambda", 4);
    convergence_table.set_precision("residual", 4);
    convergence_table.set_precision("energy", 8);
    convergence_table.set_precision("step", 4);

    convergence_table.set_scientific("lambda", true);
    convergence_table.set_scientific("residual", true);
    convergence_table.set_scientific("energy", true);
    convergence_table.set_scientific("step", true);

    convergence_table.evaluate_convergence_rates("residual", reduction_rate);
    convergence_table.evaluate_convergence_rates("residual", reduction_rate_log2);
    convergence_table.write_text(os, dealii::TableHandler::TextOutputFormat::table_with_headers);

}

} // namespace gpe

#endif //GPE_MAIN_H