//
// Created by Ferdinand Vanmaele on 12.05.26.
//

#ifndef RMO_GPE_MODEL_H
#define RMO_GPE_MODEL_H

#include <rmo/gpe/gpe.h>
#include <rmo/option_types.h>


namespace rmo::gpe
{

/**
 * @brief Orchestrator for Gross-Pitaevskii simulations.
 * The ModelBuilder owns the persistent @ref GrossPitaevskiiPackage (discretization) and the
 * assembled @ref GrossPitaevskiiSystem, and creates the @ref GrossPitaevskiiFunctional evaluators
 * (see get_eval()) used by the oracles, e.g. @ref GrossPitaevskiiOracle.
 *
 * @tparam System GrossPitaevskiiSystem or GrossPitaevskiiLumpedSystem.
 */
template <typename System>
class ModelBuilder
{
public:
    static constexpr int dim = System::dimension;
    using Functional = GrossPitaevskiiFunctional<System>;

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
        , system(package.template system<System>(std::forward<Potential>(V)))
    // problem parameters
        , options(options)
    {}

    void distribute(Vector<double>& x) const
    {
        package.distribute(x);
    }

    /** @brief Access the discretization package. */
    const GrossPitaevskiiPackage<dim>& get_package() const { return package; }
    const dealii::DoFHandler<dim>& get_dofs() const { return package.get_dofs(); }

    const System& get_system() const { return system; }
    System& get_system() { return system; }

    /** @brief Computation of value and derivatives in ambient space.
     * Non-const so calls to GrossPitaevskiiSystem::update() can propagate
     */
    Functional get_eval(double beta, SolverOptions options_slv)
    {
        return Functional(system, beta, options_slv);
    }

    Functional get_eval(SolverOptions options_slv)
    {
        return Functional(system, options.beta, options_slv);
    }

    unsigned int n_dofs() const { return package.n_dofs(); }

    // References to matrices stored in the system
    // (system.get_operator_* are factories for LinearCombination objects.)
    const typename System::MassMatrix& get_M() const { return system.get_M(); }
    const SparseMatrix<double>& get_A0() const { return system.get_A0(); }

private:
    /** @brief Persistent discretization infrastructure. */
    GrossPitaevskiiPackage<dim> package;

    /** @brief Assembly and storage of matrices. */
    System system;

    /** @brief Problem configuration options. */
    GPE_Options options;
};

} // namespace rmo::gpe

#endif //RMO_GPE_MODEL_H
