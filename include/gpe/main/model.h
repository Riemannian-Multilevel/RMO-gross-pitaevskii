//
// Created by Ferdinand Vanmaele on 12.05.26.
//

#ifndef GPE_MODEL_H
#define GPE_MODEL_H

#include <gpe/problem/gpe.h>
#include <gpe/option_types.h>


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
template <int dim>
class ModelBuilder
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
    ModelBuilder(Potential&& V, const GPE_Options& options, unsigned int n_levels)
    // discretization
        : package(options, n_levels)
    // linear system
        , system(package.system(std::forward<Potential>(V)))
    // problem parameters
        , options(options)
    {}

    // Allow to change the potential without re-discretizing the domain.
    template <typename Potential>
    void reinit(Potential&& V)
    {
        system = package.system(std::forward<Potential>(V));
    }

    void distribute(Vector<double>& x) const
    {
        package.distribute(x);
    }

    /** @brief Access the discretization package. */
    const GrossPitaevskiiPackage<dim>& get_package() const { return package; }
    const dealii::DoFHandler<dim>& get_dofs() const { return package.get_dofs(); }

    const GrossPitaevskiiSystem<dim>& get_system() const { return system; }
    GrossPitaevskiiSystem<dim>& get_system() { return system; }

    /** @brief Computation of value and derivatives in ambient space.
     * Non-const so calls to GrossPitaevskiiSystem::update() can propagate
     */
    auto get_eval(double beta, SolverOptions options_slv)
    {
        return GrossPitaevskiiFunctional<dim>(system, beta, options_slv);
    }

    auto get_eval(SolverOptions options_slv)
    {
        return GrossPitaevskiiFunctional<dim>(system, options.beta, options_slv);
    }

    unsigned int n_dofs() const { return package.n_dofs(); }

    // References to sparse matrix stored in GrossPitaevskiiSystem
    // (system.get_operator_* are factories for LinearCombination objects.)
    const auto& get_M() const { return system.get_M(); }
    const auto& get_A(double beta) const { return system.get_A(beta); }
    const auto& get_A0() const { return system.get_A0(); }

private:
    /** @brief Persistent discretization infrastructure. */
    GrossPitaevskiiPackage<dim> package;

    /** @brief Assembly and storage of matrices. */
    GrossPitaevskiiSystem<dim> system;

    /** @brief Problem configuration options. */
    GPE_Options options;
};

} // namespace gpe

#endif //GPE_MODEL_H
