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

template <int dim>
class CoarseMassOracle
{
public:
    // TODO: refactor from EnergyOracle
private:

};

template <int dim>
class CoarseActiveOracle
{
public:
    // TODO: refactor from EnergyOracle
private:

};

/**
 * @brief Mathematical Oracle for the Gross-Pitaevskii energy functional.
 * This class provides the interface required by the Riemannian Gradient Descent algorithm.
 * It manages the mapping between physical assembly and mathematical optimization by:
 * 1. Computing the functional value (Energy).
 * 2. Computing the Riemannian gradient using an @ref InverseMatrix.
 * 3. Performing retractions back to the constraint manifold (Unit Mass).
 */
/**
 * @brief Mathematical Oracle for the Gross-Pitaevskii energy functional.
 */
template <int dim>
class EnergyOracle
{
public:
    using OperatorType  = LinearCombination<SparseMatrix<double>, Vector<double>>;
    using InverseOpType = InverseMatrix<OperatorType, Vector<double>, dealii::SparseILU<double>>;

    EnergyOracle(const GrossPitaevskiiProblem<dim>& problem_, double beta_)
        : problem(problem_)
        , beta(beta_)
        , Aop(problem.get_operator_A(beta))
        , Mop(problem.get_operator_M())
    {
        precond.initialize(problem.get_A0());
    }

    void initialize(Vector<double>& x)
    {
        // Clean call: No const_cast needed anymore
        problem.assemble_nonlinear_term(x);

        iter_prev = iter;
        iter = energy::residual(x, problem.get_A0(), problem.get_Mpp(), problem.get_M(), beta);
    }

    double value(const Vector<double>& x) const
    {
        return energy::function_value(x, problem.get_A0(), problem.get_Mpp());
    }

    unsigned gradient(const Vector<double>& x, Vector<double>& output, const GdOptions& options) const
    {
        const InverseOpType Aop_inv(Aop, options.solver, precond, options.max_iter, options.tol_inner);
        energy::gradient(Aop_inv, Mop, x, output);

        return Aop_inv.control().last_step();
    }

    void retract(const Vector<double>& z, Vector<double>& x, double factor) const
    {
        energy::retract_by_norm(problem.get_M(), z, x, factor);
    }

    energy::State residual() const { return iter; }

    bool check_convergence(const GdOptions& options) const
    {
        const double lmb_diff   = std::abs(iter.lambda - iter_prev.lambda);
        const double lmb_factor = 1.0 + std::abs(iter.lambda);

        return (lmb_diff < options.tol_lambda * lmb_factor && iter.residual < options.tol_residual);
    }

private:
    const GrossPitaevskiiProblem<dim>& problem;
    double beta;
    dealii::SparseILU<double> precond;

    OperatorType Aop, Mop;
    energy::State iter, iter_prev;
};


template <int dim>
class EnergySimulator
{
public:
    EnergySimulator(const GPE_Options& options, unsigned int n_levels)
        : package(options, n_levels)
        , options(options)
    {}

    template <typename Potential>
    void run(Potential&& V, const GdOptions& gd_options, double beta)
    {
        // 1. Set up the matrices (linear assembly)
        auto problem = package.problem(std::forward<Potential>(V));

        // 2. Initialize oracle (passes problem by reference)
        EnergyOracle<dim> oracle(problem, beta);

        // 3. Initial guess
        Vector<double> x(package.get_dofs().n_dofs());
        x = 1.0;
        package.get_constraints().distribute(x);

        // 4. Run Riemannian gradient descent
        gradient_descent(oracle, x, gd_options, std::cout);
    }

private:
    GrossPitaevskiiPackage<dim> package;
    GPE_Options                options;
};

} // namespace gpe

#endif //GPE_MAIN_H