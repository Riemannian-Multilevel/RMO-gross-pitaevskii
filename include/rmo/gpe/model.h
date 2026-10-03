//
// Created by Ferdinand Vanmaele on 12.05.26.
//

#ifndef RMO_GPE_MODEL_H
#define RMO_GPE_MODEL_H

#include <rmo/gpe/gpe.h>
#include <rmo/option_types.h>
#include <rmo/util/random.h>

#include <deal.II/base/function.h>
#include <deal.II/numerics/vector_tools.h>

#include <cmath>
#include <numbers>


/**
 * @file
 * @brief ModelBuilder: owner of the package and system of one level, and factory of its functionals.
 */
namespace rmo::gpe
{

/**
 * @brief Cosine bump \f$ u_0(x) = \prod_d \cos(\pi x_d / (2r)) \f$ for \f$ |x_d| < r \f$, and 0 otherwise.
 */
template <int dim>
class CosineBump : public dealii::Function<dim>
{
public:
    explicit CosineBump(double r)
        : r(r)
    {
        AssertThrow(r > 0, dealii::ExcMessage("radius of the cosine bump must be positive"));
    }

    double value(const Point<dim>& p, unsigned = 0) const override
    {
        double u = 1.0;
        for (unsigned d = 0; d < dim; ++d) {
            u *= (std::abs(p[d]) < r) ? std::cos(0.5 * std::numbers::pi * p[d] / r) : 0.0;
        }
        return u;
    }

private:
    double r;
};


/**
 * @brief Owns the GrossPitaevskiiPackage and the assembled @p System (GrossPitaevskiiSystem or
 * GrossPitaevskiiLumpedSystem) of one level, and creates GrossPitaevskiiFunctional objects for it.
 */
template <typename System>
class ModelBuilder
{
public:
    static constexpr int dim = System::dimension;
    using Functional = GrossPitaevskiiFunctional<System>;

    /**
     * @brief Discretizes with @p options on @p n_levels mesh levels (`n_levels - 1` global refinements) and assembles
     * the system for the potential @p V (see namespace potential).
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

    /**
     * @brief Starting value of the iteration, from GPE_Options::initial and GPE_Options::initial_arg (without
     * constraints and normalization).
     */
    Vector<double> initial_value() const
    {
        Vector<double> x0(n_dofs());

        switch (options.initial) {
            case InitialValue::CONSTANT:
                x0 = options.initial_arg.value_or(1.0);
                break;
            case InitialValue::COSINE:
                dealii::VectorTools::interpolate(package.get_mapping(), package.get_dofs(),
                                                 CosineBump<dim>(options.initial_arg.value_or(options.radius)), x0);
                break;
            case InitialValue::RANDOM: {
                Engine engine(options.seed);
                UniformDistribution dist(engine, 0.5, 1.5);
                dist.fill(x0);
                break;
            }
        }
        return x0;
    }

    const GrossPitaevskiiPackage<dim>& get_package() const { return package; }
    const dealii::DoFHandler<dim>& get_dofs() const { return package.get_dofs(); }

    const System& get_system() const { return system; }
    System& get_system() { return system; }

    /**
     * @brief Functional of the system with @p beta (default: from the options). Non-const, since the functional
     * reassembles \f$ M_{\phi\phi} \f$ of the system.
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
    GrossPitaevskiiPackage<dim> package;
    System system;
    GPE_Options options;
};

} // namespace rmo::gpe

#endif //RMO_GPE_MODEL_H
