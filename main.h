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
    using InverseOpType = InverseMatrix<OperatorType, Vector<double>, dealii::SparseILU<double>>;

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
        , Aop(problem.get_operator_A(beta))
        , Mop(problem.get_operator_M())
    {
        // ILU preconditioning is usually based on the stationary linear part (A0)
        precond.initialize(problem.get_A0());
    }

    /**
     * @brief Prepares the Oracle for a new iteration.
     * Triggers the re-assembly of the non-linear term \f$ M_{pp} \f$ based on the
     * current solution \f$ \phi \in x \f$.
     * @param x The current solution vector.
     */
    void initialize(Vector<double>& x)
    {
        problem.assemble_nonlinear_term(x);
    }

    /**
     * @brief Computes the Gross-Pitaevskii energy functional value.
     * \f[ E(\phi) = \langle \phi, A_0 \phi \rangle + \frac{\beta}{2} \langle \phi, M_{pp}(\phi) \phi \rangle \f]
     * @param x The current state vector.
     * @return The energy value.
     */
    double value(const Vector<double>& x) const
    {
        return energy::function_value(x, problem.get_A0(), problem.get_Mpp(), beta);
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
    unsigned gradient(const Vector<double>& x, Vector<double>& output, const GdOptions& options) const
    {
        const InverseOpType Aop_inv(Aop, options.solver, precond, options.max_inner, options.tol_inner);

        energy::gradient(Aop_inv, Mop, x, output);

        return Aop_inv.control().last_step();
    }

    /**
     * @brief Retracts a tangent vector back to the unit-mass manifold.
     * \f[ R_x(z) = \frac{x + z}{\|x + z\|_M} \f]
     * @param z The update vector.
     * @param x The base vector (modified in place).
     * @param factor Step size scaling factor.
     */
    void retract(const Vector<double>& z, Vector<double>& x, double factor) const
    {
        energy::retract_by_norm(problem.get_M(), z, x, factor);
    }

    /**
     * @brief Computes the iteration state (eigenvalue, residual, etc.) at point x.
     */    iteration::State residual(Vector<double>& x) const
    {
        return iteration::residual(x, problem.get_A0(), problem.get_Mpp(), problem.get_M(), beta);
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
                                  const GdOptions& options)
    {
        const double lmb_diff   = std::abs(current.lambda - previous.lambda);
        const double lmb_factor = 1.0 + std::abs(current.lambda);

        return (lmb_diff < options.tol_lambda * lmb_factor && current.residual < options.tol_residual);
    }

private:
    /** @brief Reference to the problem matrices. */
    const GrossPitaevskiiProblem<dim>& problem;
    /** @brief Interaction strength parameter. */
    double beta;
    /** @brief Incomplete LU preconditioner. */
    dealii::SparseILU<double> precond;

    /** @brief Total operator \f$ A_0 + \beta M_{pp} \f$. */
    OperatorType Aop;
    /** @brief Mass operator \f$ M \f$. */
    OperatorType Mop;
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
     * @param options General options for GPE discretization.
     * @param n_levels Number of global mesh refinements.
     */
    EnergySimulator(const GPE_Options& options, unsigned int n_levels)
        : package(options, n_levels)
        , options(options)
    {}

    /**
     * @brief Runs the energy minimization for a given potential.
     * @tparam Potential Functor or class representing the external potential \f$ V(x) \f$.
     * @param V The potential object.
     * @param gd_options Options for the gradient descent algorithm.
     * @param beta The interaction strength constant.
     */
    template <typename Potential>
    Vector<double>
    run(Potential&& V, const GdOptions& gd_options, double beta)
    {
        // Generate the problem object (owns matrices A0 and M)
        auto problem = package.problem(std::forward<Potential>(V));

        // Create the oracle (references problem matrices)
        EnergyOracle<dim> oracle(problem, beta);

        // Prepare initial guess (unit constant vector)
        Vector<double> x0(package.get_dofs().n_dofs());
        x0 = 1.0;
        package.get_constraints().distribute(x0);

        // Riemannian gradient descent
        return gradient_descent(oracle, x0, gd_options, std::cout);
    }

    /** @brief Access the discretization package. */
    const GrossPitaevskiiPackage<dim>& get_package() const { return package; }

private:
    /** @brief Persistent discretization infrastructure. */
    GrossPitaevskiiPackage<dim> package;
    /** @brief Problem configuration options. */
    GPE_Options options;
};

} // namespace gpe

#endif //GPE_MAIN_H