//
// Created by Ferdinand Vanmaele on 24.02.26.
//
#ifndef GPE_MAIN_H
#define GPE_MAIN_H

#include "lac.h"
#include "oracle.h"

namespace gpe
{
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
    auto get_A(double beta) const { return problem.get_operator_A(beta); }

private:
    /** @brief Persistent discretization infrastructure. */
    GrossPitaevskiiPackage<dim> package;
    /** @brief Assembly and storage of matrices. */
    GrossPitaevskiiProblem<dim> problem;
    /** @brief Problem configuration options. */
    GPE_Options options;
};

} // namespace gpe

#endif //GPE_MAIN_H
