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
    /** @brief Templated alias for the preconditioned inverse operator type. */
    template <typename PrecondType>
    using InverseOpType = InverseMatrix<OperatorType, Vector<double>, PrecondType>;

    /**
     * @brief Constructs the Oracle by referencing an existing GPE problem.
     * @param problem_ Reference to the assembled GPE problem.
     * @param beta_ Non-linear coupling constant (interaction strength).
     * @note The Oracle does not own the problem; it holds a reference. The Problem object
     * must outlive this Oracle.
     */
    EnergyOracle(const GrossPitaevskiiProblem<dim>& problem_, Precondition precond_, double beta_)
        : problem(problem_)
        , beta(beta_)
        , precond_type(precond_)
        , Aop(problem.get_operator_A(beta))
        , Mop(problem.get_operator_M())
    {
        // TODO: extract to setup_preconditioner(const auto& matrix), apply to A every N steps
        if (precond_type == Precondition::SPARSE_ILU) {
            ilu_precond.initialize(problem.get_A0());
        }

        if (precond_type == Precondition::JACOBI || precond_type == Precondition::SSOR) {
            // reinit() uses the same SparsityPattern, so no new pattern is created
            A_assembled.reinit(problem.get_A0().get_sparsity_pattern());
        }
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

        update_dynamic_preconditioners();
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
    template <typename PrecondType>
    unsigned gradient(const Vector<double>& x, Vector<double>& output, const GdOptions& options,
                      const PrecondType& precond) const
    {
        const InverseOpType<PrecondType> Aop_inv(Aop, options.solver, precond, options.max_inner, options.tol_inner);

        energy::gradient(Aop_inv, Mop, x, output);

        return Aop_inv.control().last_step();
    }

    unsigned gradient(const Vector<double>& x, Vector<double>& output, const GdOptions& options) const
    {
        switch (precond_type) {
            case Precondition::SPARSE_ILU:
                return gradient(x, output, options, ilu_precond);
            case Precondition::IDENTITY:
                return gradient(x, output, options, dealii::PreconditionIdentity());
            case Precondition::JACOBI:
                return gradient(x, output, options, jacobi_precond);
            case Precondition::SSOR:
                return gradient(x, output, options, ssor_precond);
            case Precondition::AMG:
                throw std::logic_error("AMG not implemented yet");
            default:
                throw std::logic_error("unknown preconditioner");
        }
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
    Precondition precond_type;
    /** @brief Incomplete LU preconditioner. */
    dealii::SparseILU<double> ilu_precond{};
    dealii::PreconditionJacobi<SparseMatrix<double>> jacobi_precond{};
    dealii::PreconditionSSOR<SparseMatrix<double>> ssor_precond{};
    // Assembled matrix for dynamic preconditioners
    SparseMatrix<double> A_assembled;
    /** @brief Total operator \f$ A_0 + \beta M_{pp} \f$. */
    OperatorType Aop;
    /** @brief Mass operator \f$ M \f$. */
    OperatorType Mop;

    void update_dynamic_preconditioners()
    {
        if (precond_type != Precondition::JACOBI && precond_type != Precondition::SSOR) {
            return;
        }
        // Assemble A = A0 + beta * Mpp
        A_assembled.copy_from(problem.get_A0());
        A_assembled.add(beta, problem.get_Mpp());

        // Refresh the chosen preconditioner
        if (precond_type == Precondition::JACOBI) {
            dealii::PreconditionJacobi<SparseMatrix<double>>::AdditionalData data(0.6);
            jacobi_precond.initialize(A_assembled, data);
        }
        else if (precond_type == Precondition::SSOR) {
            dealii::PreconditionSSOR<SparseMatrix<double>>::AdditionalData data(1.2);
            ssor_precond.initialize(A_assembled, data);
        }
    }
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
        EnergyOracle<dim> oracle(problem, gd_options.precond, beta);

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