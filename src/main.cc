//
// Created by Ferdinand Vanmaele on 01.10.25.
//
#include <gpe/problem/gpe.h>
#include <gpe/problem/oracle.h>
#include <gpe/ropt/descent.h>
#include <gpe/option.h>
#include <gpe/util/util.h>

#include <iostream>
#include <fmt/format.h>

using namespace gpe;
using namespace dealii;


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
        , problem(package.problem(std::forward<Potential>(V)))
    // problem parameters
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

    /** @brief Access the discretization package. */
    const GrossPitaevskiiPackage<dim>& get_package() const { return package; }
    const GrossPitaevskiiSystem<dim>& get_problem() const { return problem; }

    /** @brief Computation of value and derivatives in ambient space. */
    auto get_eval(double beta) const
    {
        return GrossPitaevskiiFunctional<dim>(problem, beta);
    }

    unsigned int n_dofs() const { return package.n_dofs(); }


private:
    /** @brief Persistent discretization infrastructure. */
    GrossPitaevskiiPackage<dim> package;

    /** @brief Assembly and storage of matrices. */
    GrossPitaevskiiSystem<dim> problem;

    /** @brief Problem configuration options. */
    GPE_Options options;
};


int main(int argc, char* argv[])
{
    GPE_Options    options{};
    DescentOptions options_gd{};
    SolverOptions  options_slv{};
    MG_Options     options_mg{};

    // TODO: add configuration file (cf. boost tutorial)
    try {
        po::options_description all("Allowed options");
        all.add_options()("help", "produce help message");
        all.add(gpe_cli_options());
        all.add(descent_cli_options());
        all.add(mg_cli_options());
        all.add(inner_cli_options());

        po::variables_map vm;
        po::store(po::parse_command_line(argc, argv, all), vm);
        po::notify(vm);

        if (vm.contains("help")) {
            std::cout << all << "\n";
            return 0;
        }
        apply_gpe_options(vm, options);
        apply_descent_options(vm, options_gd);
        apply_mg_options(vm, options_mg);
        apply_inner_options(vm, options_slv);

        // TODO: use multiresolution if multilevel=true
        //       timer carried on across levels
        with_dimension(options.dimension, [&]<typename T0>(T0 D)
        {
            constexpr int dim = T0::value;
            unsigned int min_level = options_mg.multilevel ? options_mg.min_level : options_mg.max_level-1;
            unsigned int max_level = options_mg.max_level;

            for (unsigned int level = min_level; level < max_level; ++level) {
                // Set up the grid (Package) and finite element space
                ModelBuilder<dim> context(Square<dim>(), options, level + 1);

                // Set starting value, sufficiently far from an optimal solution
                Vector<double> x0(context.n_dofs());
                x0 = 1.0;
                context.distribute(x0);

                // Define objective in ambient space
                auto gp = context.get_eval(options.beta);
                // Define manifold
                auto manifold = UnitMassSphere<dim>(gp.get_M());
                // Define Riemannian metric
                EnergyOracle<dim> oracle(gp, options_slv);

                // Termination criterion for gradient descent
                auto conv_check = [&options_gd](const iteration::State& current, const iteration::State& previous)
                {
                    const double lmb_diff = std::abs(current.lambda - previous.lambda);
                    const double lmb_factor = 1.0 + std::abs(current.lambda);

                    return (lmb_diff < options_gd.tol_lambda * lmb_factor && current.residual < options_gd.tol_residual);
                };
                auto x = gradient_descent(oracle, manifold, x0, options_gd, std::cout, conv_check);

                // Plot solution
                std::string filename = fmt::format("solution_{}d_lvl{}.vtk", dim, level);
                output_results(x, context.get_package().get_dofs(), DataOutBase::OutputFormat::vtk, filename);
            }
        });
    }
    catch (std::exception& e) {
        std::cerr << "error: " << e.what() << "\n";
        return 1;
    }
    catch (...) {
        std::cerr << "Exception of unknown type!\n";
        return 1;
    }
}
