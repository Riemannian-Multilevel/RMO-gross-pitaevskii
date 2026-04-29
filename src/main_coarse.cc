#include <gpe/ropt/manifold.h>
#include <gpe/ropt/coarse.h>
#include <gpe/problem/oracle.h>
#include <gpe/problem/gpe.h>
#include <gpe/option.h>

using namespace dealii;
using namespace gpe;

// TODO: separate preconditioners for gradient descent, and inverse of M (coarse gradients)
using OperatorType   = LinearCombination<SparseMatrix<double>, Vector<double>>;
using InverseOpType  = PreconditionInverse<OperatorType, SparseMatrix<double>>;

//FAS.cycle(y0, std::cout, options_gd, options_gd_coarse);
//FAS.cycle_condition(x0, std::cout, options_gd, options_gd_coarse,
//    options_fas.kappa, options_fas.eps, options_fas.coarse_every);

int main(int argc, char* argv[])
{
    GPE_Options    options{};
    DescentOptions options_gd{};
    SolverOptions  options_slv{};
    MG_Options     options_mg{};
    FAS_Options    options_fas{};


    // TODO: add configuration file (cf. boost tutorial)
    try {
        po::options_description all("Allowed options");
        all.add_options()("help", "produce help message");
        all.add(gpe_cli_options());
        all.add(descent_cli_options());
        all.add(mg_cli_options());
        all.add(inner_cli_options());
        all.add(fas_cli_options());

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
        apply_fas_options(vm, options_fas);


        with_dimension(options.dimension, [&]<typename T0>(T0)
        {
            constexpr int dim = T0::value;
            SolverOptions options_slv_coarse = options_slv;
            // TODO: reduce tolerance for coarse level

            DescentOptions options_gd_coarse = options_gd;
            // TODO: heuristic: take steps UNTIL armijo line search fails OR max_iter encountered
            options_gd_coarse.max_iter    = 4;
            options_gd_coarse.step_size   = 1.0;
            options_gd_coarse.line_search = false;

            // Default potential
            Square<dim> V;

            // Default refinement for coarse and fine grid
            const int min_i = options_mg.min_level;
            const int min_i_def = 8;
            const int max_i = options_mg.max_level;
            const int max_i_def = 9;
            const int levels_i = options_mg.n_levels;
            const int n_coarse_levels = (min_i == 0 ? min_i_def : min_i);
            const int n_fine_levels = (max_i == levels_i ? max_i_def : max_i);

            // TODO: support more than 2 levels
            AssertThrow(n_fine_levels - n_coarse_levels == 1,
                dealii::ExcNotImplemented("only 2 levels are supported"));

            // Set up discretization for fine and coarse level
            GrossPitaevskiiSimulator<dim, EnergyOracle<dim>> GP_coarse(V, options, n_coarse_levels);
            const auto& problem_coarse = GP_coarse.get_problem();
            GrossPitaevskiiSimulator<dim, EnergyOracle<dim>> GP_fine(V, options, n_fine_levels);
            const auto& problem_fine = GP_fine.get_problem();

            // Set up experiments


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

