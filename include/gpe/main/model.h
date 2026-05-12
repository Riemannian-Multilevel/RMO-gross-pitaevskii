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
    const GrossPitaevskiiSystem<dim>& get_problem() const { return system; }

    /** @brief Computation of value and derivatives in ambient space. */
    auto get_eval(double beta) const
    {
        return GrossPitaevskiiFunctional<dim>(system, beta);
    }

    unsigned int n_dofs() const { return package.n_dofs(); }


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
